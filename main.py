import numpy as np
import time
import torch
from numba import njit, prange

from physics_engine import (
    gather_electric_field,
    compute_electrostatic_energy,
    compute_radiative_cooling_power,
    compute_dt_cross_section,
    compute_volumetric_fusion_power,
    compute_radiation_losses,
    compute_radiation_losses_grid,          # vectorized radiation-loss grid reduction
    evaluate_q_factors,
    compute_cic_charge_density,
    HPCPhysicsAccelerator,
    vectorized_boris_push_numba_fallback,
    compute_cic_charge_density_torch,       # large-N (>=100K) GPU-resident kernels
    vectorized_gather_and_B_torch,
    check_confinement_torch,
    apply_vectorized_collisions_torch,
    compute_alpha_heating_power_torch,
    _vectorized_boris_push_metal_impl,      # eager push; shapes churn every step, so the static-compiled path would recompile constantly
    _vectorized_boris_push_metal_dynamic,   # dynamic-shape compile, handles that churn; None if the compile failed at import
    check_confinement_flux,                 # psi-surface confinement, replaces the circular boundary
    interpolate_psi,                        # bilinear psi/B-grid lookup for the numba gather
    compute_poloidal_field_grids,           # psi-derived poloidal B (real grad-B / mirror force)
    boris_push_substeps_torch,              # alpha sub-stepping (GPU)
    vectorized_boris_push_numba_substeps    # alpha sub-stepping (CPU)
)
from config import SimulationConfiguration
import initialization
import diagnostics

# =======================================================
# HYBRID SOLVER NUMBA KERNELS (CPU -> GPU BRIDGE)
# =======================================================
@njit(parallel=True, fastmath=True)
def vectorized_gather_and_B(pos_arr, E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid,
                            R_min, R_max, Z_min, Z_max, nR, nZ, B0, R0, t, b_perturb, m_mode, n_mode, gamma):
    N = pos_arr.shape[0]
    E_arr = np.zeros((N, 3), dtype=np.float32)
    B_arr = np.zeros((N, 3), dtype=np.float32)

    for i in prange(N):
        E_arr[i] = gather_electric_field(pos_arr[i], E_R_grid, E_Z_grid, R_min, R_max, Z_min, Z_max, nR, nZ)

        r = np.sqrt(pos_arr[i, 0]**2 + pos_arr[i, 1]**2)
        phi = np.arctan2(pos_arr[i, 1], pos_arr[i, 0])
        z = pos_arr[i, 2]

        # 1. Toroidal field B_phi = B0 * R0 / R. R0 must match the value used by the
        # confinement centre, tearing mode and poloidal field.
        b_mag_tor = B0 * (R0 / r) if r > 0 else B0
        Bx = -b_mag_tor * np.sin(phi)
        By = b_mag_tor * np.cos(phi)
        Bz = 0.0

        # 2. Poloidal field, interpolated from the psi-derived B_R / B_Z grids
        # (B_R = -(1/R) dpsi/dZ, B_Z = (1/R) dpsi/dR). The old linear-in-position
        # stand-in had no gradient structure and so no magnetic mirror; this follows the
        # same flux surfaces the confinement check uses as the loss boundary.
        if r > 0:
            B_R_pol = interpolate_psi(r, z, B_R_pol_grid, R_min, R_max, Z_min, Z_max, nR, nZ)
            B_Z_pol = interpolate_psi(r, z, B_Z_pol_grid, R_min, R_max, Z_min, Z_max, nR, nZ)

            Bx += B_R_pol * np.cos(phi)
            By += B_R_pol * np.sin(phi)
            Bz += B_Z_pol

        # 3. Tearing mode perturbation
        if b_perturb > 0.0:
            amplitude = b_perturb * np.exp(gamma * t)
            theta = np.arctan2(z, r - R0) if r != R0 else 0.0
            dB_R = amplitude * np.sin(m_mode * theta - n_mode * phi)
            dB_Z = amplitude * np.cos(m_mode * theta - n_mode * phi)
            Bx += dB_R * np.cos(phi)
            By += dB_R * np.sin(phi)
            Bz += dB_Z

        B_arr[i, 0] = Bx
        B_arr[i, 1] = By
        B_arr[i, 2] = Bz

    return E_arr, B_arr

# NOTE: the circular check_confinement (r - 1.0)^2 + z^2 > 0.3^2 that used to live here
# has been removed. It was a second, disagreeing definition of the plasma boundary: the
# real last closed flux surface reaches |Z| ~ 0.354 and R ~ 1.386, so the circle cut
# particles the physics still considered confined. All boundary logic now goes through
# check_confinement_flux (CPU) / check_confinement_torch (GPU), which test psi against
# psi_edge and are the only authority on particle loss.

@njit(parallel=True, fastmath=True)
def apply_vectorized_collisions(vel_arr, type_arr, nu_c, dt):
    N = vel_arr.shape[0]
    for i in prange(N):
        if (type_arr[i] == 0 or type_arr[i] == 1) and np.random.rand() < nu_c * dt:
            speed = np.linalg.norm(vel_arr[i])
            costheta = 2.0 * np.random.rand() - 1.0
            sintheta = np.sqrt(1.0 - costheta**2)
            phi = 2.0 * np.pi * np.random.rand()
            vel_arr[i, 0] = speed * sintheta * np.cos(phi)
            vel_arr[i, 1] = speed * sintheta * np.sin(phi)
            vel_arr[i, 2] = speed * costheta
    return vel_arr

# =======================================================
# MULTI-CORE & GPU HPC BENCHMARK
# =======================================================
def run_hpc_benchmark(cfg):
    print("==================================================")
    print("   MULTI-CORE & GPU HPC BENCHMARK RUN    ")
    print("==================================================")
    
    cpu_times = []
    gpu_times = []
    
    q, m, dt = cfg.e_charge, cfg.m_deuterium, cfg.reactor_dt
    hpc_engine = HPCPhysicsAccelerator(cfg.HPC_DEVICE)
    
    for num_particles in cfg.BENCHMARK_PARTICLE_COUNTS:
        print(f"[{num_particles:,} Particles] Generating Tensors and Caches...")
        pos_arr = np.random.rand(num_particles, 3).astype(np.float32)
        vel_arr = np.random.rand(num_particles, 3).astype(np.float32)
        B_arr = np.ones((num_particles, 3), dtype=np.float32) * cfg.B0
        E_arr = np.random.rand(num_particles, 3).astype(np.float32)
        
        pos_cpu, vel_cpu = pos_arr.copy(), vel_arr.copy()
        vectorized_boris_push_numba_fallback(pos_cpu[:10], vel_cpu[:10], q, m, B_arr[:10], E_arr[:10], dt)
        
        start_cpu = time.time()
        for _ in range(cfg.BENCHMARK_STEPS):
            vectorized_boris_push_numba_fallback(pos_cpu, vel_cpu, q, m, B_arr, E_arr, dt)
        cpu_duration = time.time() - start_cpu
        cpu_times.append(cpu_duration)
        
        if cfg.HPC_DEVICE.type != "cpu":
            pos_tensor = torch.tensor(pos_arr, device=cfg.HPC_DEVICE)
            vel_tensor = torch.tensor(vel_arr, device=cfg.HPC_DEVICE)
            B_tensor = torch.tensor(B_arr, device=cfg.HPC_DEVICE)
            E_tensor = torch.tensor(E_arr, device=cfg.HPC_DEVICE)
            
            # Warm up at the FULL particle count: the compiled push specializes per exact
            # shape, so a 10-particle warmup would leave the real shape to compile inside
            # the timed loop below and inflate gpu_duration.
            hpc_engine.vectorized_boris_push_metal(pos_tensor, vel_tensor, q, m, B_tensor, E_tensor, dt)
            if cfg.HPC_DEVICE.type == "cuda": torch.cuda.synchronize()
            if cfg.HPC_DEVICE.type == "mps": torch.mps.synchronize()
            
            start_gpu = time.time()
            for _ in range(cfg.BENCHMARK_STEPS):
                pos_tensor, vel_tensor = hpc_engine.vectorized_boris_push_metal(pos_tensor, vel_tensor, q, m, B_tensor, E_tensor, dt)
            
            if cfg.HPC_DEVICE.type == "mps": torch.mps.synchronize() 
            elif cfg.HPC_DEVICE.type == "cuda": torch.cuda.synchronize()
                
            gpu_duration = time.time() - start_gpu
            gpu_times.append(gpu_duration)
        else:
            gpu_times.append(None)
            
        gpu_str = f"{gpu_duration:.4f}s" if gpu_times[-1] is not None else "N/A"
        print(f"  -> CPU Parallel Time: {cpu_duration:.4f}s | Apple Metal Time: {gpu_str}")
        
    print("[SYSTEM] HPC Benchmark Complete! Handing off to diagnostics...")
    diagnostics.plot_hpc_benchmark(cfg.BENCHMARK_PARTICLE_COUNTS, cpu_times, gpu_times)

GPU_PARTICLE_THRESHOLD = 1000000
# the threshold at which simulation run shifts from Numba JIT CPU run to PyTorch GPU run 
# PyTorch GPU run is Apple Metal Performance Shaders on Apple Silicon; CUDA on PC's

# The push is memory-bound at ~0.85 flop/byte on a shared bus, and going to the device
# adds a round-trip plus dispatch overhead, so no particle count repays it.
#
# None means "never dispatch the push to the GPU". Kept as a named knob so the decision
# stays visible and is easy to re-test.
PUSH_GPU_THRESHOLD = None


def _boris_push_adaptive(pos, vel, q, m, B, E, dt, device):
    # Mutates pos, vel in place either way (matching the numba kernel's convention), so
    # call sites don't need to branch.
    if PUSH_GPU_THRESHOLD is not None and len(pos) >= PUSH_GPU_THRESHOLD:
        pt = torch.tensor(pos, device=device, dtype=torch.float32)
        vt = torch.tensor(vel, device=device, dtype=torch.float32)
        Bt = torch.tensor(B, device=device, dtype=torch.float32)
        Et = torch.tensor(E, device=device, dtype=torch.float32)
        pn, vn = _vectorized_boris_push_metal_impl(pt, vt, q, m, Bt, Et, dt)
        pos[:] = pn.cpu().numpy()
        vel[:] = vn.cpu().numpy()
    else:
        vectorized_boris_push_numba_fallback(pos, vel, q, m, B, E, dt)


def _run_reactor_loop_cpu(cfg, engine, pos_tensor, vel_tensor, type_tensor, rho_grid, phi_grid, E_R_grid, E_Z_grid,
                          B_R_pol_grid, B_Z_pol_grid):
    # The whole per-step pipeline runs on Numpy/Numba. At this particle count (~10k-20k,
    # growing via injection) prange beats MPS outright: dispatching dozens of small
    # per-step kernels (gather, CIC, collisions, confinement) costs more in Metal
    # command-buffer overhead than the math itself. Pull the tensors to Numpy once here
    # and never touch torch inside the hot loop.
    pos_np = pos_tensor.cpu().numpy()
    vel_np = vel_tensor.cpu().numpy()
    type_np = type_tensor.cpu().numpy()

    # Stable per-particle IDs. Lost particles are physically removed from the pools, not
    # just flagged type=-1, which shifts every row index -- so trajectory bookkeeping
    # cannot be keyed on array position. pid_np stays sorted ascending (IDs are monotonic
    # and compaction preserves order), so pid -> row is a searchsorted away.
    n_init = len(pos_np)
    pid_np = np.arange(n_init, dtype=np.int64)
    next_pid = n_init

    n_track = min(1000, n_init)
    history_tracks = {int(pid_np[i]): [pos_np[i].copy()] for i in range(n_track)}
    tracked_type = {int(pid_np[i]): 0 for i in range(n_track)}
    tracked_lastpos = {int(pid_np[i]): pos_np[i].copy() for i in range(n_track)}
    tracked_lastvel = {int(pid_np[i]): vel_np[i].copy() for i in range(n_track)}
    tracked_nbis = 0
    tracked_alphas = 0
    # Tracked pids that struck the wall. The type_np == -1 flag is erased by the
    # compaction below in the same step, so the fate is recorded here instead;
    # tracked_type holds the species.
    tracked_lost = set()

    total_injected = cfg.initial_thermal_count
    total_lost = 0
    inventory_history, energy_history_keV, instability_amp_history = [], [], []
    time_history, temp_history, rad_power_history = [], [], []
    trigger_time, T_core_baseline, T_core = None, 0.0, 0.0

    alpha_heating_power_history_MW, external_heating_power_history_MW = [], []
    bremsstrahlung_power_history_MW, cyclotron_power_history_MW = [], []
    q_sci_history, q_eng_history, lawson_triple_product_history = [], [], []

    print("[SYSTEM] Igniting time-domain reactor loop on CPU/Numba (fastest path at this particle count)...")

    for step in range(cfg.reactor_num_steps):
        t = step * cfg.reactor_dt
        time_history.append(t)

        # --- BATCH NBI INJECTION (tapering source) ---
        # A fixed batch forever meant inventory only climbed and no steady state was
        # reachable. The source decays as exp(-step/tau), with fractional rates realised
        # stochastically so the beam thins out smoothly.
        if step % cfg.inject_every_n_steps == 0 and step > 0:
            nbi_rate = cfg.NBI_BATCH_SIZE * np.exp(-step / cfg.NBI_DECAY_TAU_STEPS)
            batch_size = int(nbi_rate)
            if np.random.random() < (nbi_rate - batch_size):
                batch_size += 1

            if batch_size > 0:
                psi_bounds = (engine.eq.psi_grid, engine.eq.psi_core, engine.eq.psi_R_min, engine.eq.psi_R_max,
                              engine.eq.psi_Z_min, engine.eq.psi_Z_max, engine.eq.psi_nR, engine.eq.psi_nZ)
                p_nbi, v_nbi = engine.inject_neutral_beam_cartesian(num_ions=batch_size, E_keV=cfg.nbi_energy_keV, psi_bounds=psi_bounds)
                pos_np = np.vstack((pos_np, p_nbi)).astype(np.float32)
                vel_np = np.vstack((vel_np, v_nbi)).astype(np.float32)
                type_np = np.append(type_np, np.full(batch_size, 1)).astype(np.int32)

                new_pids = np.arange(next_pid, next_pid + batch_size, dtype=np.int64)
                pid_np = np.append(pid_np, new_pids)
                next_pid += batch_size
                total_injected += batch_size

                for i in range(batch_size):
                    if tracked_nbis < 1000:
                        idx = len(type_np) - batch_size + i
                        pid = int(new_pids[i])
                        history_tracks[pid] = [pos_np[idx].copy()]
                        tracked_type[pid] = 1
                        tracked_lastpos[pid] = pos_np[idx].copy()
                        tracked_lastvel[pid] = vel_np[idx].copy()
                        tracked_nbis += 1

        # --- BATCH ALPHA INJECTION ---
        if step % (cfg.inject_every_n_steps * 2) == 0 and step > 0:
            alpha_batch = 1
            phi_pos = np.random.uniform(0, 2 * np.pi, alpha_batch)
            R_birth = np.random.normal(cfg.R0_major, 0.05, alpha_batch)
            Z_birth = np.random.normal(0.0, 0.05, alpha_batch)
            aR, aZ, avR, avphi, avZ = engine.spawn_alpha_particles(alpha_batch, R_birth[0], Z_birth[0], cfg.ALPHA_ENERGY_JOULES, cfg.MASS_ALPHA)

            p_alpha = np.zeros((alpha_batch, 3), dtype=np.float32)
            v_alpha = np.zeros((alpha_batch, 3), dtype=np.float32)
            p_alpha[:, 0] = aR * np.cos(phi_pos)
            p_alpha[:, 1] = aR * np.sin(phi_pos)
            p_alpha[:, 2] = aZ
            v_alpha[:, 0] = avR * np.cos(phi_pos) - avphi * np.sin(phi_pos)
            v_alpha[:, 1] = avR * np.sin(phi_pos) + avphi * np.cos(phi_pos)
            v_alpha[:, 2] = avZ

            pos_np = np.vstack((pos_np, p_alpha)).astype(np.float32)
            vel_np = np.vstack((vel_np, v_alpha)).astype(np.float32)
            type_np = np.append(type_np, np.full(alpha_batch, 2)).astype(np.int32)

            new_pids = np.arange(next_pid, next_pid + alpha_batch, dtype=np.int64)
            pid_np = np.append(pid_np, new_pids)
            next_pid += alpha_batch

            for i in range(alpha_batch):
                if tracked_alphas < 1000:
                    idx = len(type_np) - alpha_batch + i
                    pid = int(new_pids[i])
                    history_tracks[pid] = [pos_np[idx].copy()]
                    tracked_type[pid] = 2
                    tracked_lastpos[pid] = pos_np[idx].copy()
                    tracked_lastvel[pid] = vel_np[idx].copy()
                    tracked_alphas += 1

        mask_valid = (type_np == 0) | (type_np == 1)
        R_coords = np.sqrt(pos_np[mask_valid, 0]**2 + pos_np[mask_valid, 1]**2)
        Z_coords = pos_np[mask_valid, 2]
        charges = np.full(np.sum(mask_valid), cfg.e_charge)
        rho_grid = compute_cic_charge_density(R_coords, Z_coords, charges, cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ, False)

        if step > 0:
            # Warm-start the Poisson SOR solve from last step's converged phi: same
            # answer within the same tol, far fewer iterations to reach it.
            phi_grid, E_R_grid, E_Z_grid = engine.solve_fields(rho_grid, cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, phi_init=phi_grid)

        E_np, B_np = vectorized_gather_and_B(
            pos_np, E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid,
            cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ,
            cfg.B0, cfg.R0_major, t, cfg.b_perturb_initial, cfg.m_mode, cfg.n_mode, cfg.gamma_growth
        )

        # --- PARTICLE PUSH: Numba/CPU below PUSH_GPU_THRESHOLD, eager-GPU above it ---
        mask_d = (type_np == 0) | (type_np == 1)
        if np.any(mask_d):
            pos_d = pos_np[mask_d].copy()
            vel_d = vel_np[mask_d].copy()
            _boris_push_adaptive(pos_d, vel_d, cfg.e_charge, cfg.m_deuterium, B_np[mask_d], E_np[mask_d], cfg.reactor_dt, cfg.HPC_DEVICE)
            pos_np[mask_d] = pos_d
            vel_np[mask_d] = vel_d

        # Alphas are sub-stepped: at 3.5 MeV they cover ~1.3e-2 m per global 1 ns step
        # against a ~2.2e-2 m Larmor radius, under two samples per gyro-arc. Deuterons
        # stay on the single global step (Larmor radius ~5e-4 m, already resolved).
        mask_a = type_np == 2
        if np.any(mask_a):
            pos_a = np.ascontiguousarray(pos_np[mask_a], dtype=np.float32)
            vel_a = np.ascontiguousarray(vel_np[mask_a], dtype=np.float32)
            vectorized_boris_push_numba_substeps(
                pos_a, vel_a, cfg.CHARGE_ALPHA, cfg.MASS_ALPHA,
                np.ascontiguousarray(B_np[mask_a]), np.ascontiguousarray(E_np[mask_a]),
                cfg.reactor_dt, cfg.ALPHA_SUBSTEPS
            )
            pos_np[mask_a] = pos_a
            vel_np[mask_a] = vel_a

        vel_np = apply_vectorized_collisions(vel_np, type_np, cfg.nu_c, cfg.reactor_dt)
        # Confinement against the psi flux surface, not a circle
        newly_lost = check_confinement_flux(
            pos_np, type_np, engine.eq.psi_grid, engine.eq.psi_edge,
            engine.eq.psi_R_min, engine.eq.psi_R_max, engine.eq.psi_Z_min, engine.eq.psi_Z_max,
            engine.eq.psi_nR, engine.eq.psi_nZ
        )
        total_lost += newly_lost

        # --- WALL LOSS COMPACTION ---
        # check_confinement_flux only FLAGS wall strikes (type = -1). Leaving them in the
        # arrays meant the pools only ever grew, every kernel paid for dead particles, and
        # the confined count could never fall. Trajectory history is pid-keyed, so it
        # survives the row-index shift and the diagnostics payload is unchanged.
        if newly_lost > 0:
            alive = type_np != -1
            # Preserve the last known state of any tracked particle before it is dropped
            dead_pids = pid_np[~alive]
            if len(dead_pids) > 0:
                dead_rows = np.nonzero(~alive)[0]
                for k, dpid in zip(dead_rows, dead_pids):
                    dpid = int(dpid)
                    if dpid in history_tracks:
                        tracked_lastpos[dpid] = pos_np[k].copy()
                        tracked_lastvel[dpid] = vel_np[k].copy()
                        tracked_lost.add(dpid)
                        # Append the impact point so the red trace ends at the wall,
                        # not at the last 20-step sampling tick
                        history_tracks[dpid].append(pos_np[k].copy())

            pos_np = np.ascontiguousarray(pos_np[alive])
            vel_np = np.ascontiguousarray(vel_np[alive])
            type_np = np.ascontiguousarray(type_np[alive])
            pid_np = np.ascontiguousarray(pid_np[alive])

        # --- TRAJECTORY SAMPLING (pid-keyed, so removal cannot corrupt it) ---
        # Alphas are sampled far more often than thermals: their orbit is only ~2.2e-2 m
        # across, so a 20-step cadence (~0.26 m of travel) aliases it away.
        sample_thermal = (step % 20 == 0)
        sample_alpha = (step % cfg.ALPHA_HISTORY_EVERY == 0)
        if (sample_thermal or sample_alpha) and len(history_tracks) > 0 and len(pid_np) > 0:
            want = np.fromiter(history_tracks.keys(), dtype=np.int64, count=len(history_tracks))
            rows = np.searchsorted(pid_np, want)
            np.clip(rows, 0, len(pid_np) - 1, out=rows)
            live = pid_np[rows] == want
            for k in range(len(want)):
                if not live[k]:
                    continue
                pid = int(want[k])
                is_alpha = tracked_type.get(pid, 0) == 2
                if (is_alpha and sample_alpha) or ((not is_alpha) and sample_thermal):
                    r = rows[k]
                    history_tracks[pid].append(pos_np[r].copy())
                    tracked_lastpos[pid] = pos_np[r].copy()
                    tracked_lastvel[pid] = vel_np[r].copy()

        mask_alphas = type_np == 2
        alpha_deposited_kev = 0.0
        if np.any(mask_alphas):
            alpha_vels = vel_np[mask_alphas].astype(np.float64)
            v_mags = np.linalg.norm(alpha_vels, axis=1)
            alpha_energies_kev = (0.5 * cfg.MASS_ALPHA * (v_mags**2)) / 1.602e-16
            new_energies_kev, alpha_power_mw, alpha_deposited_kev = engine.compute_alpha_heating_power(
                alpha_energies_kev, cfg.reactor_dt, cfg)
            new_v_mags = np.sqrt(2.0 * (new_energies_kev * 1.602e-16) / cfg.MASS_ALPHA)
            scale_factors = new_v_mags / np.where(v_mags == 0, 1e-10, v_mags)
            alpha_vels *= scale_factors[:, np.newaxis]
            vel_np[mask_alphas] = alpha_vels.astype(np.float32)
        else:
            alpha_power_mw = 0.0

        alpha_heating_power_history_MW.append(alpha_power_mw)
        external_heating_power_history_MW.append(cfg.EXTERNAL_HEATING_MW)

        thermals_and_nbi_mask = (type_np == 0) | (type_np == 1)
        current_confined = int(np.sum(thermals_and_nbi_mask))
        inventory_history.append(current_confined)

        # --- ALPHA -> BULK ENERGY TRANSFER (energy conservation) ---
        # Energy drained from the alphas used to vanish -- removed from the fast
        # population and given to nothing, so the thermal plasma never felt the heating.
        # Deposit it by scaling bulk speeds, using the simulation-scale keV (no
        # macro_weight) so particles stay self-consistent; the MW figure above is
        # separately scaled for reactor-equivalent output.
        if alpha_deposited_kev > 0.0 and np.any(thermals_and_nbi_mask):
            bulk_vels = vel_np[thermals_and_nbi_mask].astype(np.float64)
            bulk_energy_kev = float(np.sum(0.5 * cfg.m_deuterium * np.sum(bulk_vels**2, axis=1))) / 1.602e-16
            if bulk_energy_kev > 0.0:
                boost = np.sqrt(1.0 + alpha_deposited_kev / bulk_energy_kev)
                vel_np[thermals_and_nbi_mask] = (bulk_vels * boost).astype(np.float32)

        if np.any(thermals_and_nbi_mask):
            current_energy_joules = np.sum(0.5 * cfg.m_deuterium * (np.linalg.norm(vel_np[thermals_and_nbi_mask].astype(np.float64), axis=1)**2))
        else:
            current_energy_joules = 0.0

        current_energy_keV = current_energy_joules / (cfg.e_charge * 1000.0)
        energy_history_keV.append(current_energy_keV)

        current_amp = cfg.b_perturb_initial * np.exp(cfg.gamma_growth * t)
        instability_amp_history.append(current_amp)

        # --- T_core IS A TEMPERATURE, NOT A TOTAL ---
        # Assigning the plasma's TOTAL stored energy here made the "core temperature"
        # read ~50,000 keV, and every consumer (brem's sqrt(T_e), the Lawson triple
        # product, the quench baseline) was fed that. For a 3D Maxwellian
        # <E> = (3/2)kT, so kT = (2/3) * <E>.
        n_bulk = max(current_confined, 1)
        T_core_kinetic = (2.0 / 3.0) * (current_energy_keV / n_bulk)

        if current_amp > cfg.MAX_ISLAND_WIDTH_THRESHOLD and not cfg.SPI_TRIGGERED:
            cfg.SPI_TRIGGERED = True
            trigger_time = t
            T_core_baseline = T_core_kinetic
            print(f"  [⚠️ EMERGENCY] Magnetic Island exceeded {cfg.MAX_ISLAND_WIDTH_THRESHOLD*100}% minor radius!")
            print(f"  [⚙️ SYSTEM] Firing Shattered Pellet Injection (SPI) at t={t*1000:.2f} ms...")
            print(f"  [⚙️ SYSTEM] Pre-disruption core temperature: {T_core_baseline:.3f} keV")

        # --- REPORTED TEMPERATURE IS MEASURED, NOT PRESCRIBED ---
        # Post-trigger T_core used to come from trigger_thermal_quench()'s analytic
        # exponential -- a curve drawn on top of the simulation. It ran on
        # TQ_DECAY_TIME = 2 ms against a 10 us run, so it advanced 0.25% and read as a
        # flatline, and it would have reported a quench even with the drain below off.
        # T_core is now the real kinetic temperature in both regimes, so the disruption
        # shows up because radiation genuinely removes the energy.
        T_core = T_core_kinetic

        if cfg.SPI_TRIGGERED:
            post_quench_keV = cfg.POST_QUENCH_TEMP / 1000.0
            # Feeding the MEASURED temperature back into P_rad closes the loop: as the
            # plasma cools, sqrt(T_e) falls and the impurity radiation weakens with it,
            # so the quench self-limits rather than following a prescribed trajectory.
            P_rad = compute_radiative_cooling_power(1.0e20, cfg.IMPURITY_DENSITY_NZ, T_core * 1000.0, cfg.RADIATIVE_COOLING_COEFF)

            # --- RADIATIVE ENERGY DRAIN (energy conservation) ---
            # P_rad used to be diagnostic-only: reported and plotted, but never taken out
            # of the particles, so the bulk kept its full kinetic energy through the
            # disruption -- quenched on the chart, pre-SPI in the velocity distribution.
            #
            # P_rad is a power DENSITY [W/m^3] at the reference density passed in above
            # (1e20 m^-3), so convert it to a per-step loss FRACTION against the thermal
            # energy density W = (3/2) * n_e * kT that same reference plasma stores. The
            # reference density cancels, which keeps this independent of macro_weight:
            #
            #   f = P_rad * dt / ((3/2) * n_ref * kT)
            #
            # kT is measured, so the drain stays proportional to the energy the particles
            # actually hold. Energy goes as v^2, so removing fraction f means scaling
            # every bulk velocity by sqrt(1 - f).
            T_kin_joules = T_core_kinetic * 1000.0 * cfg.e_charge
            W_thermal = 1.5 * 1.0e20 * T_kin_joules
            if W_thermal > 0.0 and np.any(thermals_and_nbi_mask):
                loss_fraction = float(P_rad) * cfg.reactor_dt / W_thermal
                # Radiation cools toward the post-quench floor, not through it: cap the
                # drain at the energy above POST_QUENCH_TEMP so a large P_rad can never
                # scale the velocities to or past zero.
                headroom = max(1.0 - post_quench_keV / max(T_core_kinetic, 1e-12), 0.0)
                loss_fraction = min(max(loss_fraction, 0.0), headroom)
                if loss_fraction > 0.0:
                    drain = np.sqrt(1.0 - loss_fraction)
                    vel_np[thermals_and_nbi_mask] = (
                        vel_np[thermals_and_nbi_mask].astype(np.float64) * drain
                    ).astype(np.float32)
        else:
            # No impurities before SPI fires, so nothing radiates
            P_rad = 0.0

        # energy_history_keV / T_core_kinetic above are recorded BEFORE the drain, which
        # is correct: the radiation covers [t, t+dt] and so lands in the next step's
        # ledger, not retroactively in this one.

        # One vectorized Numba call over the whole grid, replacing nR*nZ Python-level
        # calls into compute_radiation_losses.
        n_e_grid_raw = np.abs(rho_grid / cfg.e_charge)
        max_n_e = np.max(n_e_grid_raw)
        scale_factor = 1.0e20 / (max_n_e + 1e-10)
        norm_n_e = 1.0 / (max_n_e + 1e-10)
        dR, dZ = (cfg.R_max - cfg.R_min) / (cfg.nR - 1), (cfg.Z_max - cfg.Z_min) / (cfg.nZ - 1)
        total_brem_watts, total_cyc_watts = compute_radiation_losses_grid(
            n_e_grid_raw, T_core, cfg.R_min, dR, dZ, cfg.nR, cfg.nZ, cfg.B0, cfg.Z_eff, scale_factor, norm_n_e,
            cfg.CYCLOTRON_REABSORPTION, cfg.R0_major
        )
        bremsstrahlung_power_history_MW.append(total_brem_watts / 1e6)
        cyclotron_power_history_MW.append(total_cyc_watts / 1e6)

        total_fus_MW = alpha_power_mw * 5.0
        q_sci, q_eng, p_elec_out, p_elec_in = evaluate_q_factors(total_fus_MW, cfg.EXTERNAL_HEATING_MW, cfg.eta_thermal, cfg.eta_heating)
        q_sci_history.append(q_sci)
        q_eng_history.append(q_eng)

        loss_rate = total_lost / t if t > 0 else 1e-5
        tau_E = min(current_confined / loss_rate if loss_rate > 0 else 0.5, 3.0)
        # --- TRIPLE PRODUCT DENSITY MUST MATCH THE Q-FACTOR NORMALISATION ---
        # n_e was hardcoded to 1e20 m^-3 here, which had nothing to do with the
        # macro-particle weight that scales alpha heating into the reactor-scale MW
        # feeding Q_sci on the panel directly above this one.
        # Deriving n_e from the same MACRO_WEIGHT_REACTOR and the psi_edge volume makes
        # the panels describe one plasma. Units are unchanged and already correct:
        # m^-3 * keV * s, matching cfg.lawson_target and the axis label.
        n_e_reactor = (current_confined * cfg.MACRO_WEIGHT_REACTOR) / cfg.PLASMA_VOLUME_M3
        lawson_triple_product_history.append(n_e_reactor * T_core * tau_E)

        temp_history.append(T_core)
        rad_power_history.append(P_rad)

        if (step + 1) % 500 == 0:
            max_rho, max_phi = np.max(rho_grid), np.max(np.abs(phi_grid))
            print(f"  Step {step+1:04d}/{cfg.reactor_num_steps} | Confined: {current_confined:,} | Max Rho: {max_rho:.3e} | Max |Phi|: {max_phi:.2e} V")

    # Tracked-particle metadata is returned as pid-keyed dicts, which removal cannot
    # corrupt -- row indices no longer survive a step now that losses are compacted out.
    return (pos_np, vel_np, type_np, history_tracks, tracked_type, tracked_lastpos, tracked_lastvel,
            tracked_lost,
            total_injected, total_lost,
            inventory_history, energy_history_keV, instability_amp_history, time_history,
            temp_history, rad_power_history, trigger_time, T_core,
            alpha_heating_power_history_MW, external_heating_power_history_MW,
            bremsstrahlung_power_history_MW, cyclotron_power_history_MW,
            q_sci_history, q_eng_history, lawson_triple_product_history, rho_grid, phi_grid)


def _run_reactor_loop_gpu(cfg, engine, pos_tensor, vel_tensor, type_tensor, rho_grid, phi_grid, E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid):
    # GPU-resident pipeline for large particle counts (>= GPU_PARTICLE_THRESHOLD, e.g.
    # 1,000,000). The particle tensors never leave the device inside the loop, apart from
    # the small (nR x nZ) field grid the CPU Poisson solver needs and the tracked-particle
    # subset pulled every 20 steps for plotting.
    #
    # The push uses the DYNAMIC-shape compiled kernel where available, falling back to
    # eager if that compile failed at import. STATIC compile is wrong here: this loop's
    # masked subset shapes (mask_d / mask_a) change nearly every step as particles are
    # lost or injected, and static re-specializes per exact shape, so it would recompile
    # constantly -- it only pays off at fixed, reused shapes (see run_hpc_benchmark).
    # Dynamic handles the churn without recompiling and is worth ~4x on the push alone
    # (0.426 -> 0.098 ms at N=50,000). The loop as a whole is still slower than the CPU
    # path at every size tested, which is why GPU_PARTICLE_THRESHOLD is None.
    device = cfg.HPC_DEVICE

    pos_tensor = pos_tensor.to(device)
    vel_tensor = vel_tensor.to(device)
    type_tensor = type_tensor.to(device)

    # The psi grid is static for the whole run, so upload it once outside the loop
    psi_tensor = torch.tensor(engine.eq.psi_grid, device=device, dtype=torch.float32)
    psi_bounds = (engine.eq.psi_grid, engine.eq.psi_core, engine.eq.psi_R_min, engine.eq.psi_R_max,
                  engine.eq.psi_Z_min, engine.eq.psi_Z_max, engine.eq.psi_nR, engine.eq.psi_nZ)

    # Likewise the psi-derived poloidal B grids, reused by every gather. Without them the
    # gather sees a purely toroidal B with no |B| gradient structure -- no grad-B drift,
    # no mirror force, no trapped/passing separation in phase space.
    B_R_pol_t = torch.tensor(B_R_pol_grid, device=device, dtype=torch.float32)
    B_Z_pol_t = torch.tensor(B_Z_pol_grid, device=device, dtype=torch.float32)

    # One bulk transfer each: a per-index .cpu() in the dict comprehensions below would
    # be up to 1000 separate device syncs at startup.
    pos_np_init = pos_tensor.cpu().numpy()
    vel_np_init = vel_tensor.cpu().numpy()

    # Stable per-particle IDs, mirroring the CPU path. Lost particles are physically
    # removed from the tensors, which shifts every row index, so trajectory bookkeeping
    # cannot be keyed on row position. pid_tensor stays sorted ascending (IDs are
    # monotonic, mask compaction preserves order), so pid -> row is a searchsorted away.
    n_init = pos_tensor.shape[0]
    pid_tensor = torch.arange(n_init, device=device, dtype=torch.int64)
    next_pid = n_init

    n_track = min(1000, n_init)
    history_tracks = {i: [pos_np_init[i].copy()] for i in range(n_track)}
    tracked_type = {i: 0 for i in range(n_track)}
    tracked_lastpos = {i: pos_np_init[i].copy() for i in range(n_track)}
    tracked_lastvel = {i: vel_np_init[i].copy() for i in range(n_track)}
    tracked_nbis = 0
    tracked_alphas = 0
    max_tracked_pid = n_track - 1
    # See the CPU path: tracked_type holds the species, this holds the fate
    tracked_lost = set()

    total_injected = cfg.initial_thermal_count
    total_lost = 0
    inventory_history, energy_history_keV, instability_amp_history = [], [], []
    time_history, temp_history, rad_power_history = [], [], []
    trigger_time, T_core_baseline, T_core = None, 0.0, 0.0

    alpha_heating_power_history_MW, external_heating_power_history_MW = [], []
    bremsstrahlung_power_history_MW, cyclotron_power_history_MW = [], []
    q_sci_history, q_eng_history, lawson_triple_product_history = [], [], []

    print("[SYSTEM] Igniting time-domain reactor loop on GPU (fastest path at this particle count)...")
    if device.type == "mps":
        print("[SYSTEM] First step will pause briefly — torch.compile is fusing the field-gather")
        print("[SYSTEM] kernel into a single Metal dispatch (one-time cost, not a hang).")

    for step in range(cfg.reactor_num_steps):
        t = step * cfg.reactor_dt
        time_history.append(t)

        # --- BATCH NBI INJECTION (tapering source) ---
        # Same decaying source as the CPU path, driven by the same two cfg knobs:
        # exp(-step/tau), with fractional rates realised stochastically.
        if step % cfg.inject_every_n_steps == 0 and step > 0:
            nbi_rate = cfg.NBI_BATCH_SIZE * np.exp(-step / cfg.NBI_DECAY_TAU_STEPS)
            batch_size = int(nbi_rate)
            if np.random.random() < (nbi_rate - batch_size):
                batch_size += 1

            if batch_size > 0:
                p_nbi, v_nbi = engine.inject_neutral_beam_cartesian(num_ions=batch_size, E_keV=cfg.nbi_energy_keV, psi_bounds=psi_bounds)
                pos_tensor = torch.cat([pos_tensor, torch.tensor(p_nbi, device=device, dtype=torch.float32)], dim=0)
                vel_tensor = torch.cat([vel_tensor, torch.tensor(v_nbi, device=device, dtype=torch.float32)], dim=0)
                type_tensor = torch.cat([type_tensor, torch.full((batch_size,), 1, device=device, dtype=torch.int32)], dim=0)

                new_pids = torch.arange(next_pid, next_pid + batch_size, device=device, dtype=torch.int64)
                pid_tensor = torch.cat([pid_tensor, new_pids], dim=0)
                total_injected += batch_size

                for i in range(batch_size):
                    if tracked_nbis < 1000:
                        pid = next_pid + i
                        history_tracks[pid] = [p_nbi[i].copy()]
                        tracked_type[pid] = 1
                        tracked_lastpos[pid] = p_nbi[i].copy()
                        tracked_lastvel[pid] = v_nbi[i].copy()
                        max_tracked_pid = max(max_tracked_pid, pid)
                        tracked_nbis += 1

                next_pid += batch_size

        # --- BATCH ALPHA INJECTION ---
        if step % (cfg.inject_every_n_steps * 2) == 0 and step > 0:
            alpha_batch = 1
            phi_pos = np.random.uniform(0, 2 * np.pi, alpha_batch)
            R_birth = np.random.normal(cfg.R0_major, 0.05, alpha_batch)
            Z_birth = np.random.normal(0.0, 0.05, alpha_batch)
            aR, aZ, avR, avphi, avZ = engine.spawn_alpha_particles(alpha_batch, R_birth[0], Z_birth[0], cfg.ALPHA_ENERGY_JOULES, cfg.MASS_ALPHA)

            p_alpha = np.zeros((alpha_batch, 3), dtype=np.float32)
            v_alpha = np.zeros((alpha_batch, 3), dtype=np.float32)
            p_alpha[:, 0] = aR * np.cos(phi_pos)
            p_alpha[:, 1] = aR * np.sin(phi_pos)
            p_alpha[:, 2] = aZ
            v_alpha[:, 0] = avR * np.cos(phi_pos) - avphi * np.sin(phi_pos)
            v_alpha[:, 1] = avR * np.sin(phi_pos) + avphi * np.cos(phi_pos)
            v_alpha[:, 2] = avZ

            pos_tensor = torch.cat([pos_tensor, torch.tensor(p_alpha, device=device, dtype=torch.float32)], dim=0)
            vel_tensor = torch.cat([vel_tensor, torch.tensor(v_alpha, device=device, dtype=torch.float32)], dim=0)
            type_tensor = torch.cat([type_tensor, torch.full((alpha_batch,), 2, device=device, dtype=torch.int32)], dim=0)

            new_pids = torch.arange(next_pid, next_pid + alpha_batch, device=device, dtype=torch.int64)
            pid_tensor = torch.cat([pid_tensor, new_pids], dim=0)

            for i in range(alpha_batch):
                if tracked_alphas < 1000:
                    pid = next_pid + i
                    history_tracks[pid] = [p_alpha[i].copy()]
                    tracked_type[pid] = 2
                    tracked_lastpos[pid] = p_alpha[i].copy()
                    tracked_lastvel[pid] = v_alpha[i].copy()
                    max_tracked_pid = max(max_tracked_pid, pid)
                    tracked_alphas += 1

            next_pid += alpha_batch

        # --- CHARGE DENSITY MAPPING (GPU, eager) ---
        mask_valid = (type_tensor == 0) | (type_tensor == 1)
        R_coords = torch.sqrt(pos_tensor[mask_valid, 0]**2 + pos_tensor[mask_valid, 1]**2)
        Z_coords = pos_tensor[mask_valid, 2]
        rho_grid_t = compute_cic_charge_density_torch(
            R_coords, Z_coords, cfg.e_charge,
            cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ, device
        )
        # Only the small (nR x nZ) grid crosses to the CPU -- the SOR Poisson solve is
        # inherently serial and stays on Numba/CPU whatever the particle count.
        rho_grid = rho_grid_t.cpu().numpy().astype(np.float64)

        if step > 0:
            # Warm-start from last step's converged phi, as in the CPU path
            phi_grid, E_R_grid, E_Z_grid = engine.solve_fields(rho_grid, cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, phi_init=phi_grid)

        E_R_grid_t = torch.tensor(E_R_grid, device=device, dtype=torch.float32)
        E_Z_grid_t = torch.tensor(E_Z_grid, device=device, dtype=torch.float32)

        # --- E-FIELD GATHER + ANALYTIC B-FIELD (GPU, torch.compile'd, dynamic shapes) ---
        # Fed the psi-derived poloidal B grids and cfg.R0_major. That argument slot is R0
        # -- the toroidal 1/R falloff centre and tearing-mode axis -- and passing
        # cfg.B_poloidal (0.3) there put the field's centre outside the plasma entirely.
        E_tensor, B_tensor = vectorized_gather_and_B_torch(
            pos_tensor, E_R_grid_t, E_Z_grid_t, B_R_pol_t, B_Z_pol_t,
            cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ,
            cfg.B0, cfg.R0_major, t, cfg.b_perturb_initial, cfg.m_mode, cfg.n_mode, cfg.gamma_growth
        )

        # --- PARTICLE PUSH (GPU; see the note at the top of this function) ---
        mask_d = (type_tensor == 0) | (type_tensor == 1)
        if torch.any(mask_d):
            _push = _vectorized_boris_push_metal_dynamic or _vectorized_boris_push_metal_impl
            p_d, v_d = _push(pos_tensor[mask_d], vel_tensor[mask_d], cfg.e_charge, cfg.m_deuterium, B_tensor[mask_d], E_tensor[mask_d], cfg.reactor_dt)
            pos_tensor[mask_d] = p_d
            vel_tensor[mask_d] = v_d

        # Alphas are sub-stepped: at 3.5 MeV they cover ~1.3e-2 m per global 1 ns step
        # against a ~2.2e-2 m Larmor radius, under two samples per gyro-arc. Deuterons
        # stay on the single global step. E and B are held fixed across the sub-steps --
        # the point is to resolve gyration about the local B, not to re-gather the field.
        # Stays on-device; boris_push_substeps_torch re-enters the same MPS kernel.
        mask_a = type_tensor == 2
        if torch.any(mask_a):
            p_a, v_a = boris_push_substeps_torch(
                pos_tensor[mask_a], vel_tensor[mask_a], cfg.CHARGE_ALPHA, cfg.MASS_ALPHA,
                B_tensor[mask_a], E_tensor[mask_a], cfg.reactor_dt, cfg.ALPHA_SUBSTEPS
            )
            pos_tensor[mask_a] = p_a
            vel_tensor[mask_a] = v_a

        # --- COLLISIONS + CONFINEMENT CHECK (GPU, eager; psi-surface, not circular) ---
        vel_tensor = apply_vectorized_collisions_torch(vel_tensor, type_tensor, cfg.nu_c, cfg.reactor_dt)
        type_tensor, newly_lost = check_confinement_torch(
            pos_tensor, type_tensor, psi_tensor, engine.eq.psi_edge,
            engine.eq.psi_R_min, engine.eq.psi_R_max, engine.eq.psi_Z_min, engine.eq.psi_Z_max,
            engine.eq.psi_nR, engine.eq.psi_nZ
        )
        total_lost += newly_lost

        # --- WALL LOSS COMPACTION (GPU) ---
        # check_confinement_torch only FLAGS wall strikes (type = -1). Leaving them in the
        # tensors meant the pools only ever grew, every kernel paid for dead particles,
        # and the confined count could never fall. Boolean-mask indexing keeps the removal
        # a single on-device gather per tensor -- only the last-known state of tracked
        # particles that just died round-trips to the CPU.
        if newly_lost > 0:
            alive = type_tensor != -1

            # Preserve the last known state of any TRACKED particle before it is dropped.
            # Gated on pid <= max_tracked_pid so a large loss event doesn't drag the whole
            # dead population across the memory boundary to find the few plotted pids.
            dying_tracked = (~alive) & (pid_tensor <= max_tracked_pid)
            if bool(dying_tracked.any()):
                d_rows = torch.nonzero(dying_tracked, as_tuple=False).squeeze(1)
                d_pids = pid_tensor[d_rows].cpu().numpy()
                d_pos = pos_tensor[d_rows].cpu().numpy()
                d_vel = vel_tensor[d_rows].cpu().numpy()
                for k in range(len(d_pids)):
                    dpid = int(d_pids[k])
                    if dpid in history_tracks:
                        tracked_lastpos[dpid] = d_pos[k].copy()
                        tracked_lastvel[dpid] = d_vel[k].copy()
                        tracked_lost.add(dpid)
                        # End the red trace at the wall, not at the last sampling tick
                        history_tracks[dpid].append(d_pos[k].copy())

            pos_tensor = pos_tensor[alive]
            vel_tensor = vel_tensor[alive]
            type_tensor = type_tensor[alive]
            pid_tensor = pid_tensor[alive]

        # Pull only the tracked particles (<=2000) needed for trajectory plots, never the
        # full inventory. Keyed by stable pid via searchsorted, so the row shift from the
        # compaction above cannot silently re-point a track at a different particle.
        # Alphas are sampled far more often than thermals: their orbit is only ~2.2e-2 m
        # across, so a 20-step cadence (~0.26 m of travel) aliases it away.
        sample_thermal = (step % 20 == 0)
        sample_alpha = (step % cfg.ALPHA_HISTORY_EVERY == 0)
        if (sample_thermal or sample_alpha) and len(history_tracks) > 0 and pid_tensor.numel() > 0:
            want_list = list(history_tracks.keys())
            want_t = torch.tensor(want_list, device=device, dtype=torch.int64)
            rows = torch.searchsorted(pid_tensor, want_t).clamp(max=pid_tensor.numel() - 1)
            live = pid_tensor[rows] == want_t
            subset_pos = pos_tensor[rows].cpu().numpy()
            subset_vel = vel_tensor[rows].cpu().numpy()
            live_np = live.cpu().numpy()
            for k, pid in enumerate(want_list):
                if not live_np[k]:
                    continue
                is_alpha = tracked_type.get(pid, 0) == 2
                if (is_alpha and sample_alpha) or ((not is_alpha) and sample_thermal):
                    history_tracks[pid].append(subset_pos[k].copy())
                    tracked_lastpos[pid] = subset_pos[k].copy()
                    tracked_lastvel[pid] = subset_vel[k].copy()

        # --- ALPHA HEATING (GPU, eager) ---
        mask_alphas = type_tensor == 2
        alpha_deposited_kev = 0.0
        if torch.any(mask_alphas):
            alpha_vels = vel_tensor[mask_alphas]
            v_mags = torch.linalg.norm(alpha_vels, dim=1)
            alpha_energies_kev = (0.5 * cfg.MASS_ALPHA * v_mags**2) / 1.602e-16
            # Third return value is the raw simulation-scale keV drained, needed by the
            # energy transfer below.
            new_energies_kev, alpha_power_mw, alpha_deposited_kev = compute_alpha_heating_power_torch(
                alpha_energies_kev, cfg.reactor_dt, cfg)
            new_v_mags = torch.sqrt(2.0 * (new_energies_kev * 1.602e-16) / cfg.MASS_ALPHA)
            safe_v_mags = torch.where(v_mags == 0, torch.full_like(v_mags, 1e-10), v_mags)
            scale_factors = new_v_mags / safe_v_mags
            vel_tensor[mask_alphas] = alpha_vels * scale_factors.unsqueeze(1)
        else:
            alpha_power_mw = 0.0

        alpha_heating_power_history_MW.append(alpha_power_mw)
        external_heating_power_history_MW.append(cfg.EXTERNAL_HEATING_MW)

        thermals_and_nbi_mask = (type_tensor == 0) | (type_tensor == 1)
        current_confined = int(thermals_and_nbi_mask.sum().item())
        inventory_history.append(current_confined)

        # --- ALPHA -> BULK ENERGY TRANSFER (energy conservation) ---
        # Energy drained from the alphas used to vanish -- removed from the fast
        # population and given to nothing, so the thermal plasma never felt the heating.
        # Deposit it by scaling bulk speeds, using the simulation-scale keV (no
        # macro_weight) so particles stay self-consistent; the MW figure above is
        # separately scaled for reactor-equivalent output.
        if alpha_deposited_kev > 0.0 and torch.any(thermals_and_nbi_mask):
            bulk_vels = vel_tensor[thermals_and_nbi_mask]
            bulk_energy_kev = float((0.5 * cfg.m_deuterium * torch.sum(bulk_vels**2, dim=1)).sum().item()) / 1.602e-16
            if bulk_energy_kev > 0.0:
                boost = float(np.sqrt(1.0 + alpha_deposited_kev / bulk_energy_kev))
                vel_tensor[thermals_and_nbi_mask] = bulk_vels * boost

        if torch.any(thermals_and_nbi_mask):
            current_energy_joules = float((0.5 * cfg.m_deuterium * torch.sum(vel_tensor[thermals_and_nbi_mask]**2, dim=1)).sum().item())
        else:
            current_energy_joules = 0.0

        current_energy_keV = current_energy_joules / (cfg.e_charge * 1000.0)
        energy_history_keV.append(current_energy_keV)

        current_amp = cfg.b_perturb_initial * np.exp(cfg.gamma_growth * t)
        instability_amp_history.append(current_amp)

        # --- T_core IS A TEMPERATURE, NOT A TOTAL ---
        # Assigning the plasma's TOTAL stored energy here made the "core temperature"
        # read ~50,000 keV, and every consumer (brem's sqrt(T_e), the Lawson triple
        # product, the quench baseline) was fed that. For a 3D Maxwellian
        # <E> = (3/2)kT, so kT = (2/3) * <E>.
        n_bulk = max(current_confined, 1)
        T_core_kinetic = (2.0 / 3.0) * (current_energy_keV / n_bulk)

        if current_amp > cfg.MAX_ISLAND_WIDTH_THRESHOLD and not cfg.SPI_TRIGGERED:
            cfg.SPI_TRIGGERED = True
            trigger_time = t
            T_core_baseline = T_core_kinetic
            print(f"  [EMERGENCY] Magnetic Island exceeded {cfg.MAX_ISLAND_WIDTH_THRESHOLD*100}% minor radius!")
            print(f"  [SYSTEM] Firing Shattered Pellet Injection (SPI) at t={t*1000:.2f} ms...")
            print(f"  [SYSTEM] Pre-disruption core temperature: {T_core_baseline:.3f} keV")

        # --- REPORTED TEMPERATURE IS MEASURED, NOT PRESCRIBED ---
        # Post-trigger T_core used to come from trigger_thermal_quench()'s analytic
        # exponential -- a curve drawn on top of the simulation. It ran on
        # TQ_DECAY_TIME = 2 ms against a 10 us run, so it advanced 0.25% and read as a
        # flatline, and it would have reported a quench even with the drain below off.
        # T_core is now the real kinetic temperature in both regimes, so the disruption
        # shows up because radiation genuinely removes the energy.
        T_core = T_core_kinetic

        if cfg.SPI_TRIGGERED:
            post_quench_keV = cfg.POST_QUENCH_TEMP / 1000.0
            # Feeding the MEASURED temperature back into P_rad closes the loop: as the
            # plasma cools, sqrt(T_e) falls and the impurity radiation weakens with it,
            # so the quench self-limits rather than following a prescribed trajectory.
            P_rad = compute_radiative_cooling_power(1.0e20, cfg.IMPURITY_DENSITY_NZ, T_core * 1000.0, cfg.RADIATIVE_COOLING_COEFF)

            # --- RADIATIVE ENERGY DRAIN (energy conservation) ---
            # P_rad used to be diagnostic-only: reported and plotted, but never taken out
            # of the particles, so the bulk kept its full kinetic energy through the
            # disruption -- quenched on the chart, pre-SPI in the velocity distribution.
            #
            # P_rad is a power DENSITY [W/m^3] at the reference density passed in above
            # (1e20 m^-3), so convert it to a per-step loss FRACTION against the thermal
            # energy density W = (3/2) * n_e * kT that same reference plasma stores. The
            # reference density cancels, which keeps this independent of macro_weight:
            #
            #   f = P_rad * dt / ((3/2) * n_ref * kT)
            #
            # kT is measured, so the drain stays proportional to the energy the particles
            # actually hold. Energy goes as v^2, so removing fraction f means scaling
            # every bulk velocity by sqrt(1 - f).
            T_kin_joules = T_core_kinetic * 1000.0 * cfg.e_charge
            W_thermal = 1.5 * 1.0e20 * T_kin_joules
            if W_thermal > 0.0 and torch.any(thermals_and_nbi_mask):
                loss_fraction = float(P_rad) * cfg.reactor_dt / W_thermal
                # Radiation cools toward the post-quench floor, not through it: cap the
                # drain at the energy above POST_QUENCH_TEMP so a large P_rad can never
                # scale the velocities to or past zero.
                headroom = max(1.0 - post_quench_keV / max(T_core_kinetic, 1e-12), 0.0)
                loss_fraction = min(max(loss_fraction, 0.0), headroom)
                if loss_fraction > 0.0:
                    # float32 is fine here: f runs ~6e-5/step at the few-keV temperatures
                    # this reactor reaches, ~500x above float32 epsilon, so unlike the
                    # alpha-drag drain this multiply does not round back to the original.
                    drain = float(np.sqrt(1.0 - loss_fraction))
                    vel_tensor[thermals_and_nbi_mask] = vel_tensor[thermals_and_nbi_mask] * drain
        else:
            # No impurities before SPI fires, so nothing radiates
            P_rad = 0.0

        # energy_history_keV / T_core_kinetic above are recorded BEFORE the drain, which
        # is correct: the radiation covers [t, t+dt] and so lands in the next step's
        # ledger, not retroactively in this one.
        n_e_grid_raw = np.abs(rho_grid / cfg.e_charge)
        max_n_e = np.max(n_e_grid_raw)
        scale_factor = 1.0e20 / (max_n_e + 1e-10)
        norm_n_e = 1.0 / (max_n_e + 1e-10)
        dR, dZ = (cfg.R_max - cfg.R_min) / (cfg.nR - 1), (cfg.Z_max - cfg.Z_min) / (cfg.nZ - 1)
        # Pass the cyclotron re-absorption factor and the real major radius explicitly --
        # falling back on the defaults gave the GPU path a different radiation balance
        # from the CPU path.
        total_brem_watts, total_cyc_watts = compute_radiation_losses_grid(
            n_e_grid_raw, T_core, cfg.R_min, dR, dZ, cfg.nR, cfg.nZ, cfg.B0, cfg.Z_eff, scale_factor, norm_n_e,
            cfg.CYCLOTRON_REABSORPTION, cfg.R0_major
        )
        bremsstrahlung_power_history_MW.append(total_brem_watts / 1e6)
        cyclotron_power_history_MW.append(total_cyc_watts / 1e6)

        total_fus_MW = alpha_power_mw * 5.0
        q_sci, q_eng, p_elec_out, p_elec_in = evaluate_q_factors(total_fus_MW, cfg.EXTERNAL_HEATING_MW, cfg.eta_thermal, cfg.eta_heating)
        q_sci_history.append(q_sci)
        q_eng_history.append(q_eng)

        loss_rate = total_lost / t if t > 0 else 1e-5
        tau_E = min(current_confined / loss_rate if loss_rate > 0 else 0.5, 3.0)
        # --- TRIPLE PRODUCT DENSITY MUST MATCH THE Q-FACTOR NORMALISATION ---
        # n_e was hardcoded to 1e20 m^-3 here, which had nothing to do with the
        # macro-particle weight that scales alpha heating into the reactor-scale MW
        # feeding Q_sci on the panel directly above this one. 
        # Deriving n_e from the same MACRO_WEIGHT_REACTOR and the psi_edge volume makes
        # the panels describe one plasma. Units are unchanged and already correct:
        # m^-3 * keV * s, matching cfg.lawson_target and the axis label.
        n_e_reactor = (current_confined * cfg.MACRO_WEIGHT_REACTOR) / cfg.PLASMA_VOLUME_M3
        lawson_triple_product_history.append(n_e_reactor * T_core * tau_E)

        temp_history.append(T_core)
        rad_power_history.append(P_rad)

        if (step + 1) % 500 == 0:
            max_rho, max_phi = np.max(rho_grid), np.max(np.abs(phi_grid))
            print(f"  Step {step+1:04d}/{cfg.reactor_num_steps} | Confined: {current_confined:,} | Max Rho: {max_rho:.3e} | Max |Phi|: {max_phi:.2e} V")

    # --- Leave the GPU only here, at the end of the loop ---
    pos_np = pos_tensor.cpu().numpy()
    vel_np = vel_tensor.cpu().numpy()
    type_np = type_tensor.cpu().numpy()

    return (pos_np, vel_np, type_np, history_tracks, tracked_type, tracked_lastpos, tracked_lastvel,
            tracked_lost,
            total_injected, total_lost,
            inventory_history, energy_history_keV, instability_amp_history, time_history,
            temp_history, rad_power_history, trigger_time, T_core,
            alpha_heating_power_history_MW, external_heating_power_history_MW,
            bremsstrahlung_power_history_MW, cyclotron_power_history_MW,
            q_sci_history, q_eng_history, lawson_triple_product_history, rho_grid, phi_grid)


def run_reactor_steady_state():
    print("==================================================")
    print("      PIC FULL CYCLE (LORENTZ PUSH)      ")
    print("==================================================")

    cfg = SimulationConfiguration()
    eq, engine, pos_tensor, vel_tensor, type_tensor, rho_grid, phi_grid, E_R_grid, E_Z_grid = initialization.initialize_reactor(cfg)

    # The psi-derived poloidal B field, resampled onto the solver grid so the pusher's B
    # lookup reuses the indices/weights the E gather computes. Without it the pusher sees
    # a purely toroidal B whose only gradient is the smooth 1/R falloff -- no mirror ratio
    # along a field line, so nothing is trapped and the v_parallel-vs-R map collapses to
    # the injected Maxwellian.
    B_R_pol_grid, B_Z_pol_grid, pol_scale, B_pol_target = compute_poloidal_field_grids(
        eq, cfg.B0, R0=cfg.R0_major, a_minor=0.3, q_target=cfg.Q_SAFETY_TARGET,
        dst_R_min=cfg.R_min, dst_R_max=cfg.R_max, dst_Z_min=cfg.Z_min, dst_Z_max=cfg.Z_max,
        dst_nR=cfg.nR, dst_nZ=cfg.nZ
    )
    print(f"[SYSTEM] Poloidal field from Grad-Shafranov psi: B_pol ~ {B_pol_target:.3f} T "
          f"(q = {cfg.Q_SAFETY_TARGET:.1f}, psi rescale x{pol_scale:.3e})")

    # GPU_PARTICLE_THRESHOLD is None by default -- the CPU loop measured faster at every
    # count tested, up to 3,000,000. See its definition for the numbers.
    loop_fn = (_run_reactor_loop_gpu
               if GPU_PARTICLE_THRESHOLD is not None and cfg.initial_thermal_count >= GPU_PARTICLE_THRESHOLD
               else _run_reactor_loop_cpu)
    (pos_np, vel_np, type_np, history_tracks, tracked_type, tracked_lastpos, tracked_lastvel,
     tracked_lost,
     total_injected, total_lost,
     inventory_history, energy_history_keV, instability_amp_history, time_history,
     temp_history, rad_power_history, trigger_time, T_core,
     alpha_heating_power_history_MW, external_heating_power_history_MW,
     bremsstrahlung_power_history_MW, cyclotron_power_history_MW,
     q_sci_history, q_eng_history, lawson_triple_product_history, rho_grid, phi_grid
     ) = loop_fn(cfg, engine, pos_tensor, vel_tensor, type_tensor, rho_grid, phi_grid, E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid)

    dR, dZ = (cfg.R_max - cfg.R_min) / (cfg.nR - 1), (cfg.Z_max - cfg.Z_min) / (cfg.nZ - 1)

    print("[SYSTEM] Packaging particle arrays for Graphing Utilities...")

    # history_tracks is keyed by particle ID, not row index: lost particles are compacted
    # out of pos_np/vel_np, so indexing those by a track key would read an unrelated
    # particle (and IndexError once the pools shrink past it). Final per-track state comes
    # from the pid-keyed snapshots, so a particle that hit the wall keeps the state it had
    # then. The dict payload handed to diagnostics is unchanged.
    mock_active = []
    mock_alphas = []
    for pid, hist in history_tracks.items():
        ptype = tracked_type.get(pid, 0)
        p_dict = {
            "history": hist,
            "type": ptype,
            # diagnostics._scrub_particles reads "status"; without it written here every
            # particle defaults to "confined" and the wall-loss colouring never fires.
            "status": "lost" if pid in tracked_lost else "confined",
            "pos": tracked_lastpos[pid],
            "vel": tracked_lastvel[pid],
        }
        if ptype == 2:
            mock_alphas.append(p_dict)
            # A lost alpha still belongs on the wall-loss plots: the alpha-orbit plot keys
            # off species, the 2D/3D reactor plots off status.
            if p_dict["status"] == "lost":
                mock_active.append(p_dict)
        else:
            mock_active.append(p_dict)

    final_active = [{"status": "confined", "pos": p, "vel": v} for p, v, typ in zip(pos_np, vel_np, type_np) if typ != -1]

    R_centers, density_profile, pressure_profile = engine.compute_fluid_moments(
        final_active, cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.num_radial_bins
    )

    # --- TRUE v_parallel FOR THE PHASE-SPACE MAP ---
    # (x*vy - y*vx)/R is v_TOROIDAL, which only equals v_parallel if B is purely toroidal
    # -- and pitch angle against the ACTUAL field line is the whole quantity separating
    # trapped from passing. Projecting onto the real local b_hat (toroidal + psi-derived
    # poloidal) is what makes that boundary appear instead of a Gaussian blob.
    print("[SYSTEM] Projecting velocities onto local field lines for phase-space map...")
    alive_mask = type_np != -1
    R_phase, v_parallel_phase = [], []
    if np.any(alive_mask):
        pos_alive = np.ascontiguousarray(pos_np[alive_mask], dtype=np.float32)
        vel_alive = np.ascontiguousarray(vel_np[alive_mask], dtype=np.float32)
        zero_grid = np.zeros((cfg.nR, cfg.nZ), dtype=np.float64)
        # b_perturb = 0: the pitch angle should be defined by the equilibrium field line,
        # not the instantaneous tearing-mode ripple.
        _, B_final = vectorized_gather_and_B(
            pos_alive, zero_grid, zero_grid, B_R_pol_grid, B_Z_pol_grid,
            cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ,
            cfg.B0, cfg.R0_major, 0.0, 0.0, cfg.m_mode, cfg.n_mode, 0.0
        )
        B_mag = np.linalg.norm(B_final, axis=1)
        safe_B = np.where(B_mag > 0.0, B_mag, 1.0)
        b_hat = B_final / safe_B[:, np.newaxis]
        v_par_all = np.sum(vel_alive * b_hat, axis=1)
        R_all = np.sqrt(pos_alive[:, 0]**2 + pos_alive[:, 1]**2)
        keep = (R_all > 0) & (B_mag > 0.0)
        R_phase = R_all[keep].tolist()
        v_parallel_phase = v_par_all[keep].tolist()

    print("[SYSTEM] Mapping Volumetric Fusion Power ")
    n_e_grid_raw = np.abs(rho_grid / cfg.e_charge)
    scale_factor = 1.0e20 / (np.max(n_e_grid_raw) + 1e-10)
    n_e_grid_scaled = n_e_grid_raw * scale_factor
    n_D_grid = n_T_grid = 0.5 * n_e_grid_scaled
    T_i_grid_keV = T_core * (n_e_grid_raw / (np.max(n_e_grid_raw) + 1e-10))
    P_fusion_grid = compute_volumetric_fusion_power(n_D_grid, n_T_grid, T_i_grid_keV, cfg)

    total_fusion_power_watts = 0.0
    for i in range(cfg.nR):
        R_i = cfg.R_min + i * dR
        cell_volume = 2.0 * np.pi * R_i * dR * dZ if R_i > 0 else 1.0
        for j in range(cfg.nZ):
            total_fusion_power_watts += P_fusion_grid[i, j] * cell_volume
            
    print("==================================================")
    print(f" TOTAL INTEGRATED FUSION POWER: {total_fusion_power_watts / 1e6:.2f} MW ")
    print("==================================================")

    diagnostics.run_steady_state_diagnostics(
        cfg, eq, rho_grid, phi_grid, energy_history_keV, 
        mock_active, total_injected, total_lost, inventory_history,
        R_centers, density_profile, pressure_profile,
        R_phase, v_parallel_phase, instability_amp_history, P_fusion_grid,
        alpha_particles=mock_alphas, 
        alpha_power_history=alpha_heating_power_history_MW,
        ext_power_history=external_heating_power_history_MW,
        brem_power_history=bremsstrahlung_power_history_MW, 
        cyc_power_history=cyclotron_power_history_MW,        
        q_sci_history=q_sci_history, q_eng_history=q_eng_history, lawson_history=lawson_triple_product_history 
    )

    if cfg.SPI_TRIGGERED:
        diagnostics.plot_disruption_mitigation(time_history, temp_history, rad_power_history, trigger_time)


def run_plasma_oscillation_test():
    print("==================================================")
    print("  HYBRID TENSOR OSCILLATIONS & SHIELDING ")
    print("==================================================")
    
    cfg = SimulationConfiguration()
    eq, engine, pos_tensor, vel_tensor, type_tensor, rho_grid = initialization.initialize_oscillation_test(cfg)

    # A purely electrostatic unit test: it measures collective Langmuir ringing against
    # the toroidal field only, so the gather gets zero B_pol grids rather than the
    # Grad-Shafranov ones.
    B_R_pol_grid = np.zeros((cfg.nR, cfg.nZ), dtype=np.float64)
    B_Z_pol_grid = np.zeros((cfg.nR, cfg.nZ), dtype=np.float64)
    hpc_engine = HPCPhysicsAccelerator(cfg.HPC_DEVICE)
    
    w_es_history, time_arr = [], []
    pos_np = pos_tensor.cpu().numpy()
    vel_np = vel_tensor.cpu().numpy()
    type_np = type_tensor.cpu().numpy()
    
    print("[SYSTEM] Running Unit Test: Capturing collective electrostatic ringing...")
    
    for step in range(cfg.osc_num_steps):
        mask_valid = type_np != -1
        R_coords = np.sqrt(pos_np[mask_valid, 0]**2 + pos_np[mask_valid, 1]**2)
        Z_coords = pos_np[mask_valid, 2]
        charges = np.full(np.sum(mask_valid), -cfg.e_charge * cfg.macro_weight)
        
        rho_grid = compute_cic_charge_density(R_coords, Z_coords, charges, cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ, True)
        phi_grid, E_R_grid, E_Z_grid = engine.solve_fields(rho_grid, cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max)
        
        w_es = compute_electrostatic_energy(E_R_grid, E_Z_grid, cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max)
        w_es_history.append(w_es)
        time_arr.append(step * cfg.osc_dt)
        
        E_np, B_np = vectorized_gather_and_B(
            pos_np, E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid,
            cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ,
            cfg.B0, cfg.R0_major, 0.0, 0.0, 1, 1, 0.0
        )
        
        pos_tensor = torch.tensor(pos_np, device=cfg.HPC_DEVICE)
        vel_tensor = torch.tensor(vel_np, device=cfg.HPC_DEVICE)
        E_tensor = torch.tensor(E_np, device=cfg.HPC_DEVICE)
        B_tensor = torch.tensor(B_np, device=cfg.HPC_DEVICE)
        
        p_e, v_e = hpc_engine.vectorized_boris_push_metal(pos_tensor, vel_tensor, -cfg.e_charge * cfg.macro_weight, cfg.m_electron * cfg.macro_weight, B_tensor, E_tensor, cfg.osc_dt)
        
        pos_np, vel_np = p_e.cpu().numpy(), v_e.cpu().numpy()
        # Unified on the psi surface, same as the reactor loop
        check_confinement_flux(
            pos_np, type_np, eq.psi_grid, eq.psi_edge,
            eq.psi_R_min, eq.psi_R_max, eq.psi_Z_min, eq.psi_Z_max,
            eq.psi_nR, eq.psi_nZ
        )
        
        if (step + 1) % 250 == 0:
            print(f"  Step {step+1:04d}/{cfg.osc_num_steps} | W_ES: {w_es:.3e} Joules | E_Z Max: {np.max(np.abs(E_Z_grid)):.2e} V/m")

    print("==================================================")
    print("  TEST COMPLETE: GENERATING DIAGNOSTIC ARTIFACTS  ")
    print("==================================================")
    diagnostics.run_oscillation_diagnostics(cfg, time_arr, w_es_history)

def run_nuclear_reaction_dynamics():
    print("==================================================")
    print("  EVALUATING D-T QUANTUM CROSS-SECTION   ")
    print("==================================================")
    E_kev_arr = np.linspace(1.0, 200.0, 500)
    sigma_arr = np.array([compute_dt_cross_section(e) for e in E_kev_arr])
    diagnostics.plot_fusion_cross_section(E_kev_arr, sigma_arr)

if __name__ == "__main__":
    cfg = SimulationConfiguration()
    run_hpc_benchmark(cfg)
    run_reactor_steady_state()
    run_plasma_oscillation_test()
    run_nuclear_reaction_dynamics()