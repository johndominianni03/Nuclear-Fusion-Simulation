import numpy as np
import time
import numba
import sys
import torch
from numba import njit, prange

from physics_engine import (
    gather_electric_field,
    gather_electric_field_scalar,   # allocation-free scalar twin, used by the prange gather
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
from profiling import (
    StageProfiler,
    _profile_header,
    _log_device_memory,
    _report_peak_rss,
    _report_sor_histogram,
)
from kernels import (
    vectorized_gather_and_B_into,
    vectorized_gather_and_B,
    apply_vectorized_collisions,
)
from particle_pool import (
    _ParticlePool,
    _pool_capacity_bounds,
)
from track_store import (
    _MAX_TRACKED_SLOTS,
    _pid_capacity_bound,
    _require_pid_capacity,
    _TrackStore,
)
from reactor_cpu import _run_reactor_loop_cpu


def _report_gs_provenance(eq):
    """Say whether psi came from the disk cache or a fresh solve.

    A cold miss and a warm hit differ by seconds; without this line the
    initialize_reactor row in the table is easy to misread as loop-adjacent
    cost when it is really just the Grad-Shafranov solve.
    """
    source = getattr(eq, "last_solve_source", None)
    backend = getattr(eq, "last_solve_backend", None)
    if source is None:
        print("[PROFILE] Grad-Shafranov: provenance unavailable")
        return
    if source == "cache-hit":
        print(f"[PROFILE] Grad-Shafranov: CACHE HIT (backend '{backend}') -- "
              "no SOR iteration ran; initialize_reactor excludes the solve cost.")
    else:
        print(f"[PROFILE] Grad-Shafranov: COLD SOLVE on the '{backend}' backend -- "
              "initialize_reactor below INCLUDES the full solve.")


GPU_PARTICLE_THRESHOLD = None
# The particle count at which the reactor run shifts from the Numba JIT CPU loop
# (_run_reactor_loop_cpu) to the PyTorch GPU loop (_run_reactor_loop_gpu): Apple Metal
# Performance Shaders on Apple Silicon, CUDA on PC.
#
# None means "always take the CPU loop". It held 1,000,000 from the initial commit
# through step 6, which silently sent every 1M-particle run down the slower path:
# measured at 1,000,000 particles / 200 steps during step 6, the GPU loop took 69.6 s
# wall against ~32-36 s for the CPU loop. The CPU loop measured faster at every count
# tested, up to 3,000,000.
#
# NOT a deprecation of the GPU path -- step 12 optimizes that loop and step 13 reassesses
# this number. To force the GPU loop, set this to an int <= cfg.initial_thermal_count,
# either by editing this line (what README.md documents) or at runtime without editing:
#     import main; main.GPU_PARTICLE_THRESHOLD = 0; main.run_reactor_steady_state()
# The dispatch reads this global when run_reactor_steady_state is called, so the runtime
# form works; `cfg.PROFILE = True` prints which loop was chosen.


def _run_reactor_loop_gpu(cfg, engine, pos_tensor, vel_tensor, type_tensor, rho_grid, phi_grid, E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid):
    # GPU-resident pipeline, selected only when GPU_PARTICLE_THRESHOLD is set to an int
    # <= cfg.initial_thermal_count. It is None by default, so this loop is opt-in -- see
    # that constant's definition for the numbers and for how to force it. The particle
    # tensors never leave the device inside the loop, apart from the small (nR x nZ) field
    # grid the CPU Poisson solver needs and the tracked-particle subset pulled every 20
    # steps for plotting.
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
    next_pid = n_init

    # pid -> current row, dense and ON DEVICE. Same reasoning as the CPU path:
    # torch.searchsorted needs pid_tensor sorted ascending, an invariant step 9
    # retires. Keeping the map on the device means compaction can renumber it
    # with a scatter and the sampler can gather rows without a sync.
    pid_capacity = _pid_capacity_bound(cfg, n_init)
    pid_row_t = torch.full((pid_capacity,), -1, device=device, dtype=torch.int64)
    pid_row_t[:n_init] = torch.arange(n_init, device=device, dtype=torch.int64)

    # Device-resident capacity pool, replacing the torch.cat growth below. Same
    # hard-ceiling contract as the CPU path: add() raises rather than growing.
    bulk_capacity, alpha_capacity = _pool_capacity_bounds(cfg, n_init)
    bulk = _ParticlePool(bulk_capacity, device=device)
    alpha = _ParticlePool(alpha_capacity, device=device)
    bulk.add(pos_tensor, vel_tensor, 0,
             torch.arange(n_init, device=device, dtype=torch.int64))
    bulk.type[:n_init] = type_tensor
    # The input tensors have been copied into the pool; drop the names so nothing
    # downstream can read the stale pre-pool copies.
    del pos_tensor, vel_tensor, type_tensor
    b_pos, b_vel, b_type, b_pid = bulk.views()
    a_pos, a_vel, a_type, a_pid = alpha.views()
    # Liveness and slot assignment stay on the HOST. That is what lets the host
    # know how many rows a sample will write without asking the device, so the
    # vertex write pointer advances with no .item() sync. Tracked particles die
    # only at the wall, and the compaction below already pulls exactly that
    # (small, pid-gated) set across, so the host copy stays in step cheaply.
    pid_slot = np.full(pid_capacity, -1, dtype=np.int64)
    # pid -> pool, host-side and write-once at birth; see the CPU path for why
    # compaction never touches it (species cannot migrate).
    pid_pool = np.full(pid_capacity, -1, dtype=np.int8)
    pid_pool[:n_init] = 0
    slot_alive = np.zeros(_MAX_TRACKED_SLOTS, dtype=bool)

    n_track = min(1000, n_init)
    # host_dtype float64: this path records its injection vertex straight from
    # the float64 array inject_neutral_beam_cartesian returns, before the cast
    # onto the device, and that is the value it recorded before this rewrite.
    tracks = _TrackStore(_MAX_TRACKED_SLOTS, device=device, host_dtype=np.float64)
    init_pids = np.arange(n_track, dtype=np.int64)
    init_slots = tracks.add_slots(init_pids, 0,
                                  pos_np_init[:n_track], vel_np_init[:n_track])
    pid_slot[init_pids] = init_slots
    slot_alive[init_slots] = True
    tracked_nbis = 0
    tracked_alphas = 0
    max_tracked_pid = n_track - 1

    total_injected = cfg.initial_thermal_count
    total_lost = 0
    inventory_history, energy_history_keV, instability_amp_history = [], [], []
    time_history, temp_history, rad_power_history = [], [], []
    trigger_time, T_core_baseline, T_core = None, 0.0, 0.0

    alpha_heating_power_history_MW, external_heating_power_history_MW = [], []
    bremsstrahlung_power_history_MW, cyclotron_power_history_MW = [], []
    q_sci_history, q_eng_history, lawson_triple_product_history = [], [], []

    # --- Profiling state (inert unless cfg.PROFILE) ---
    # MPS/CUDA queue work asynchronously, so every clock read on this path is
    # preceded by a device synchronize -- otherwise the numbers measure how
    # fast the CPU enqueued the work, not how long the GPU took to do it.
    profiling = bool(getattr(cfg, "PROFILE", False))
    _sync = None
    if profiling:
        if device.type == "mps":
            _sync = torch.mps.synchronize
        elif device.type == "cuda":
            _sync = torch.cuda.synchronize
    prof = StageProfiler(profiling, sync=_sync, label="GPU path")
    sor_iter_counts = []
    if profiling:
        _profile_header(cfg, f"GPU path ({device.type})", bulk.n_live + alpha.n_live)
        if _sync is None:
            print("   [WARN] no synchronize for this device: GPU stage times "
                  "will measure enqueue, not execution.")
    loop_wall_start = time.perf_counter() if profiling else None

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
        _t = prof.mark()
        if step % cfg.inject_every_n_steps == 0 and step > 0:
            nbi_rate = cfg.NBI_BATCH_SIZE * np.exp(-step / cfg.NBI_DECAY_TAU_STEPS)
            batch_size = int(nbi_rate)
            if np.random.random() < (nbi_rate - batch_size):
                batch_size += 1

            if batch_size > 0:
                p_nbi, v_nbi = engine.inject_neutral_beam_cartesian(num_ions=batch_size, E_keV=cfg.nbi_energy_keV, psi_bounds=psi_bounds)
                _require_pid_capacity(next_pid, batch_size, pid_capacity, "GPU NBI injection")
                new_pids = torch.arange(next_pid, next_pid + batch_size, device=device, dtype=torch.int64)
                base_row = bulk.add(
                    torch.tensor(p_nbi, device=device, dtype=torch.float32),
                    torch.tensor(v_nbi, device=device, dtype=torch.float32),
                    1, new_pids)
                b_pos, b_vel, b_type, b_pid = bulk.views()
                total_injected += batch_size

                pid_row_t[new_pids] = torch.arange(base_row, base_row + batch_size,
                                                   device=device, dtype=torch.int64)
                pid_pool[next_pid:next_pid + batch_size] = 0

                n_new = min(batch_size, 1000 - tracked_nbis)
                if n_new > 0:
                    host_pids = np.arange(next_pid, next_pid + n_new, dtype=np.int64)
                    slots = tracks.add_slots(host_pids, 1, p_nbi[:n_new], v_nbi[:n_new])
                    pid_slot[host_pids] = slots
                    slot_alive[slots] = True
                    max_tracked_pid = max(max_tracked_pid, int(host_pids[-1]))
                    tracked_nbis += n_new

                next_pid += batch_size

        prof.add("NBI injection", _t)

        # --- BATCH ALPHA INJECTION ---
        _t = prof.mark()
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

            _require_pid_capacity(next_pid, alpha_batch, pid_capacity, "GPU alpha injection")
            new_pids = torch.arange(next_pid, next_pid + alpha_batch, device=device, dtype=torch.int64)
            base_row = alpha.add(
                torch.tensor(p_alpha, device=device, dtype=torch.float32),
                torch.tensor(v_alpha, device=device, dtype=torch.float32),
                2, new_pids)
            a_pos, a_vel, a_type, a_pid = alpha.views()

            pid_row_t[new_pids] = torch.arange(base_row, base_row + alpha_batch,
                                               device=device, dtype=torch.int64)
            pid_pool[next_pid:next_pid + alpha_batch] = 1

            n_new = min(alpha_batch, 1000 - tracked_alphas)
            if n_new > 0:
                host_pids = np.arange(next_pid, next_pid + n_new, dtype=np.int64)
                slots = tracks.add_slots(host_pids, 2, p_alpha[:n_new], v_alpha[:n_new])
                pid_slot[host_pids] = slots
                slot_alive[slots] = True
                max_tracked_pid = max(max_tracked_pid, int(host_pids[-1]))
                tracked_alphas += n_new

            next_pid += alpha_batch

        prof.add("alpha injection", _t)

        # --- CHARGE DENSITY MAPPING (GPU, eager) ---
        _t = prof.mark()
        # Bulk pool only, no mask -- see the CPU twin for why no dead rows can be
        # present at this point in the step.
        R_coords = torch.sqrt(b_pos[:, 0]**2 + b_pos[:, 1]**2)
        Z_coords = b_pos[:, 2]
        rho_grid_t = compute_cic_charge_density_torch(
            R_coords, Z_coords, cfg.e_charge,
            cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ, device
        )
        # Only the small (nR x nZ) grid crosses to the CPU -- the SOR Poisson solve is
        # inherently serial and stays on Numba/CPU whatever the particle count.
        rho_grid = rho_grid_t.cpu().numpy().astype(np.float64)
        prof.add("CIC deposition", _t)

        _t = prof.mark()
        if step > 0:
            # Warm-start from last step's converged phi, as in the CPU path
            phi_grid, E_R_grid, E_Z_grid, _sor_iters = engine.solve_fields(
                rho_grid, cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max,
                phi_init=phi_grid, return_iters=True)
            if profiling:
                sor_iter_counts.append(_sor_iters)
        prof.add("solve_fields (Poisson)", _t)

        _t = prof.mark()
        E_R_grid_t = torch.tensor(E_R_grid, device=device, dtype=torch.float32)
        E_Z_grid_t = torch.tensor(E_Z_grid, device=device, dtype=torch.float32)

        # --- E-FIELD GATHER + ANALYTIC B-FIELD (GPU, torch.compile'd, dynamic shapes) ---
        # Fed the psi-derived poloidal B grids and cfg.R0_major. That argument slot is R0
        # -- the toroidal 1/R falloff centre and tearing-mode axis -- and passing
        # cfg.B_poloidal (0.3) there put the field's centre outside the plasma entirely.
        _gather_args = (E_R_grid_t, E_Z_grid_t, B_R_pol_t, B_Z_pol_t,
                        cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ,
                        cfg.B0, cfg.R0_major, t, cfg.b_perturb_initial,
                        cfg.m_mode, cfg.n_mode, cfg.gamma_growth)
        E_b, B_b = vectorized_gather_and_B_torch(b_pos, *_gather_args)
        if alpha.n_live > 0:
            E_a, B_a = vectorized_gather_and_B_torch(a_pos, *_gather_args)
        prof.add("vectorized_gather_and_B", _t)

        # --- PARTICLE PUSH (GPU; see the note at the top of this function) ---
        _t = prof.mark()
        # mask_d is gone: the bulk pool IS the deuteron set. The kernel still
        # RETURNS new tensors, so the results are copied back into the pool views
        # rather than rebound -- rebinding would detach the loop from pool.pos
        # and strand every later write (the step 8 collisions bug).
        if bulk.n_live > 0:
            _push = _vectorized_boris_push_metal_dynamic or _vectorized_boris_push_metal_impl
            p_d, v_d = _push(b_pos, b_vel, cfg.e_charge, cfg.m_deuterium,
                             B_b, E_b, cfg.reactor_dt)
            b_pos.copy_(p_d)
            b_vel.copy_(v_d)
        prof.add("deuteron Boris push", _t)

        # Alphas are sub-stepped: at 3.5 MeV they cover ~1.3e-2 m per global 1 ns step
        # against a ~2.2e-2 m Larmor radius, under two samples per gyro-arc. Deuterons
        # stay on the single global step. E and B are held fixed across the sub-steps --
        # the point is to resolve gyration about the local B, not to re-gather the field.
        # Stays on-device; boris_push_substeps_torch re-enters the same MPS kernel.
        _t = prof.mark()
        # mask_a likewise. Same returns-new-tensors contract, same copy_ back.
        if alpha.n_live > 0:
            p_a, v_a = boris_push_substeps_torch(
                a_pos, a_vel, cfg.CHARGE_ALPHA, cfg.MASS_ALPHA,
                B_a, E_a, cfg.reactor_dt, cfg.ALPHA_SUBSTEPS
            )
            a_pos.copy_(p_a)
            a_vel.copy_(v_a)
        prof.add("alpha substep push", _t)

        # --- COLLISIONS + CONFINEMENT CHECK (GPU, eager; psi-surface, not circular) ---
        _t = prof.mark()
        # MUST write back into the pool view rather than rebind. Unlike the numba
        # twin, which mutates vel_arr in place and returns the same array,
        # apply_vectorized_collisions_torch returns torch.where(...) -- a NEW
        # tensor -- whenever any collision fires. Rebinding vel_tensor to it
        # detaches the loop from pool.vel, so the pool keeps stale velocities and
        # the next injection's pool.views() silently reverts every update made
        # since. At nu_c*dt = 5e-6 with ~2,500 particles a collision fires only
        # about every 80 steps, which is what made this look like late-onset
        # chaotic drift rather than a lost write.
        # Bulk pool only: the kernel's eligible mask is types 0/1, so the alpha
        # pool would be a guaranteed no-op. The write-back is still mandatory --
        # this function returns torch.where(...), a NEW tensor, whenever any
        # collision fires, and rebinding would detach the loop from pool.vel.
        vel_updated = apply_vectorized_collisions_torch(b_vel, b_type, cfg.nu_c, cfg.reactor_dt)
        if vel_updated is not b_vel:
            b_vel.copy_(vel_updated)
        prof.add("apply_vectorized_collisions", _t)

        _t = prof.mark()
        # Both pools; total_lost sums them. Mutates type in place and returns the
        # same tensor, so the rebinding is a no-op and the views stay attached.
        _psi_args = (psi_tensor, engine.eq.psi_edge,
                     engine.eq.psi_R_min, engine.eq.psi_R_max,
                     engine.eq.psi_Z_min, engine.eq.psi_Z_max,
                     engine.eq.psi_nR, engine.eq.psi_nZ)
        b_type, lost_b = check_confinement_torch(b_pos, b_type, *_psi_args)
        if alpha.n_live > 0:
            a_type, lost_a = check_confinement_torch(a_pos, a_type, *_psi_args)
        else:
            lost_a = 0
        newly_lost = lost_b + lost_a
        prof.add("check_confinement_flux", _t)
        total_lost += newly_lost

        # --- WALL LOSS COMPACTION (GPU) ---
        _t = prof.mark()
        # check_confinement_torch only FLAGS wall strikes (type = -1). Leaving them in the
        # tensors meant the pools only ever grew, every kernel paid for dead particles,
        # and the confined count could never fall. Boolean-mask indexing keeps the removal
        # a single on-device gather per tensor -- only the last-known state of tracked
        # particles that just died round-trips to the CPU.
        if newly_lost > 0:
            for _pool, _pos, _vel, _type, _pid, _lost in (
                    (bulk, b_pos, b_vel, b_type, b_pid, lost_b),
                    (alpha, a_pos, a_vel, a_type, a_pid, lost_a)):
                if _lost == 0:
                    continue
                alive = _type != -1

                # Preserve the last known state of any TRACKED particle before it is
                # dropped. Gated on pid <= max_tracked_pid so a large loss event
                # doesn't drag the whole dead population across the memory boundary
                # to find the few plotted pids.
                dying_tracked = (~alive) & (_pid <= max_tracked_pid)
                if bool(dying_tracked.any()):
                    d_rows = torch.nonzero(dying_tracked, as_tuple=False).squeeze(1)
                    d_pids = _pid[d_rows].cpu().numpy()
                    d_pos = _pos[d_rows].cpu().numpy()
                    d_vel = _vel[d_rows].cpu().numpy()
                    d_slots = pid_slot[d_pids]
                    is_tracked = d_slots >= 0
                    if np.any(is_tracked):
                        t_slots = d_slots[is_tracked]
                        # End the crimson trace at the wall, not at the last
                        # sampling tick. Always terminal: slot_alive goes False
                        # here, so the sampler never selects the slot again.
                        tracks.record_impact(t_slots, d_pos[is_tracked], d_vel[is_tracked])
                        slot_alive[t_slots] = False

                pid_row_t[_pid[~alive]] = -1
                _pool.compact(torch.nonzero(alive, as_tuple=False).squeeze(1))

            b_pos, b_vel, b_type, b_pid = bulk.views()
            a_pos, a_vel, a_type, a_pid = alpha.views()
            # Renumber the device-side pid -> row map for whichever pools moved.
            # pid_pool is untouched -- rows move, pools do not.
            if lost_b:
                pid_row_t[b_pid] = torch.arange(bulk.n_live, device=device, dtype=torch.int64)
            if lost_a:
                pid_row_t[a_pid] = torch.arange(alpha.n_live, device=device, dtype=torch.int64)
        prof.add("wall-loss compaction", _t)

        # Pull only the tracked particles (<=2000) needed for trajectory plots, never the
        # full inventory. Keyed by stable pid via searchsorted, so the row shift from the
        # compaction above cannot silently re-point a track at a different particle.
        # Alphas are sampled far more often than thermals: their orbit is only ~2.2e-2 m
        # across, so a 20-step cadence (~0.26 m of travel) aliases it away.
        _t = prof.mark()
        sample_thermal = (step % 20 == 0)
        sample_alpha = (step % cfg.ALPHA_HISTORY_EVERY == 0)
        if (sample_thermal or sample_alpha) and tracks.n_slots > 0:
            # The whole selection -- which slots are due, which are still alive,
            # and therefore HOW MANY rows this sample writes -- is decided on the
            # host, from slot_alive. That is what keeps the write pointer free of
            # any .item() sync. Only the gather runs on the device, and the
            # sampled coordinates stay there: the old form downloaded two
            # (n_tracked, 3) arrays every sampling step and walked them in Python.
            due = tracks.slots_due(sample_thermal, sample_alpha)
            if due.size > 0:
                sel_slots = due[slot_alive[due]]
                if sel_slots.size > 0:
                    # A row index only means something relative to a pool, so
                    # pid_pool splits the due set before the gather. The split is
                    # decided on the HOST, which is what keeps the vertex write
                    # pointer free of any .item() sync.
                    sel_pools = pid_pool[tracks.pids[sel_slots]]
                    for _which, _pos, _vel in ((0, b_pos, b_vel), (1, a_pos, a_vel)):
                        grp = sel_pools == _which
                        if not np.any(grp):
                            continue
                        g_slots = sel_slots[grp]
                        g_slots_t = torch.from_numpy(g_slots).to(device)
                        g_pids_t = torch.from_numpy(tracks.pids[g_slots]).to(device)
                        rows_t = pid_row_t[g_pids_t]
                        sampled_pos = _pos[rows_t]
                        tracks.append_samples(g_slots, sampled_pos)
                        tracks.set_last_device(g_slots, g_slots_t,
                                               sampled_pos, _vel[rows_t])

        prof.add("trajectory sampling", _t)

        # --- ALPHA HEATING (GPU, eager) ---
        _t = prof.mark()
        # Reads the alpha pool, writes the bulk pool below. Ordering preserved
        # exactly: alpha energies updated here, deposit after, bulk energy last.
        alpha_deposited_kev = 0.0
        if alpha.n_live > 0:
            alpha_vels = a_vel
            v_mags = torch.linalg.norm(alpha_vels, dim=1)
            alpha_energies_kev = (0.5 * cfg.MASS_ALPHA * v_mags**2) / 1.602e-16
            # Third return value is the raw simulation-scale keV drained, needed by the
            # energy transfer below.
            new_energies_kev, alpha_power_mw, alpha_deposited_kev = compute_alpha_heating_power_torch(
                alpha_energies_kev, cfg.reactor_dt, cfg)
            new_v_mags = torch.sqrt(2.0 * (new_energies_kev * 1.602e-16) / cfg.MASS_ALPHA)
            safe_v_mags = torch.where(v_mags == 0, torch.full_like(v_mags, 1e-10), v_mags)
            scale_factors = new_v_mags / safe_v_mags
            a_vel.copy_(alpha_vels * scale_factors.unsqueeze(1))
        else:
            alpha_power_mw = 0.0

        alpha_heating_power_history_MW.append(alpha_power_mw)
        external_heating_power_history_MW.append(cfg.EXTERNAL_HEATING_MW)

        # The bulk pool holds nothing but types 0 and 1, and the compaction above
        # removed every -1 this step, so its live count IS the confined count --
        # and reading it costs no device sync, unlike the old .sum().item().
        current_confined = int(bulk.n_live)
        inventory_history.append(current_confined)

        # --- ALPHA -> BULK ENERGY TRANSFER (energy conservation) ---
        # Energy drained from the alphas used to vanish -- removed from the fast
        # population and given to nothing, so the thermal plasma never felt the heating.
        # Deposit it by scaling bulk speeds, using the simulation-scale keV (no
        # macro_weight) so particles stay self-consistent; the MW figure above is
        # separately scaled for reactor-equivalent output.
        if alpha_deposited_kev > 0.0 and bulk.n_live > 0:
            bulk_energy_kev = float((0.5 * cfg.m_deuterium * torch.sum(b_vel**2, dim=1)).sum().item()) / 1.602e-16
            if bulk_energy_kev > 0.0:
                boost = float(np.sqrt(1.0 + alpha_deposited_kev / bulk_energy_kev))
                b_vel.copy_(b_vel * boost)

        if bulk.n_live > 0:
            current_energy_joules = float((0.5 * cfg.m_deuterium * torch.sum(b_vel**2, dim=1)).sum().item())
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
            if W_thermal > 0.0 and bulk.n_live > 0:
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
                    b_vel.copy_(b_vel * drain)
        else:
            # No impurities before SPI fires, so nothing radiates
            P_rad = 0.0

        prof.add("alpha heating + energy transfer", _t)

        # energy_history_keV / T_core_kinetic above are recorded BEFORE the drain, which
        # is correct: the radiation covers [t, t+dt] and so lands in the next step's
        # ledger, not retroactively in this one.
        _t = prof.mark()
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
        prof.add("radiation + Q-factor diagnostics", _t)

        if profiling and (step + 1) % 500 == 0:
            _log_device_memory(device, step + 1)

        if (step + 1) % 500 == 0:
            max_rho, max_phi = np.max(rho_grid), np.max(np.abs(phi_grid))
            print(f"  Step {step+1:04d}/{cfg.reactor_num_steps} | Confined: {current_confined:,} | Max Rho: {max_rho:.3e} | Max |Phi|: {max_phi:.2e} V")

    if profiling:
        if _sync is not None:
            _sync()
        loop_wall = time.perf_counter() - loop_wall_start
        prof.report(wall_total=loop_wall,
                    title=f"STAGE TIMING -- reactor loop (GPU / {device.type})")
        _report_sor_histogram(sor_iter_counts)
        _log_device_memory(device, cfg.reactor_num_steps, final=True)
        _report_peak_rss("[PROFILE][GPU]")

    # --- Leave the GPU only here, at the end of the loop ---
    # Rejoin the pools, then restore the legacy row order. See the CPU twin: a
    # bare concatenate puts every alpha after every bulk particle, which is a
    # different order from the single-pool version. Sorting by pid restores it,
    # and pids are unique across both pools from the one shared counter.
    pos_all = torch.cat((b_pos, a_pos), dim=0).cpu().numpy()
    vel_all = torch.cat((b_vel, a_vel), dim=0).cpu().numpy()
    type_all = torch.cat((b_type, a_type), dim=0).cpu().numpy()
    pid_all = torch.cat((b_pid, a_pid), dim=0).cpu().numpy()
    legacy_order = np.argsort(pid_all, kind="stable")
    pos_np = np.ascontiguousarray(pos_all[legacy_order])
    vel_np = np.ascontiguousarray(vel_all[legacy_order])
    type_np = np.ascontiguousarray(type_all[legacy_order])

    # Flat-array accumulation -> the dict payload, once. This is the only place
    # the sampled vertices and the device-side last_pos/last_vel cross to the host.
    (history_tracks, tracked_type, tracked_lastpos,
     tracked_lastvel, tracked_lost) = tracks.to_dicts()
    return (pos_np, vel_np, type_np, history_tracks, tracked_type, tracked_lastpos, tracked_lastvel,
            tracked_lost,
            total_injected, total_lost,
            inventory_history, energy_history_keV, instability_amp_history, time_history,
            temp_history, rad_power_history, trigger_time, T_core,
            alpha_heating_power_history_MW, external_heating_power_history_MW,
            bremsstrahlung_power_history_MW, cyclotron_power_history_MW,
            q_sci_history, q_eng_history, lawson_triple_product_history, rho_grid, phi_grid)


def package_reactor_results(cfg, eq, engine, B_R_pol_grid, B_Z_pol_grid,
                            pos_np, vel_np, type_np,
                            history_tracks, tracked_type, tracked_lastpos,
                            tracked_lastvel, tracked_lost,
                            rho_grid, T_core, prof=None):
    """Turns the reactor loop's raw output into the arrays diagnostics plots.

    Lifted verbatim out of run_reactor_steady_state so a regression test can
    reach these products without rendering anything: the loop is the physics,
    this is the packaging, and rendering is a third step. run_reactor_steady_state
    calls this and forwards the contents to diagnostics exactly as before.

    `eq` is not read here today -- the poloidal field it produced arrives
    already resampled as B_R_pol_grid/B_Z_pol_grid. It stays in the signature
    so callers pass the same equilibrium object they pass everywhere else, and
    so a future moment that needs psi does not change every call site.

    Returns a dict of exactly the products the caller forwards onward.

    `prof` is an optional StageProfiler. When None (the default, and what
    tests/test_regression.py passes) every hook below is a no-op.
    """
    if prof is None:
        prof = StageProfiler(False)
    dR, dZ = (cfg.R_max - cfg.R_min) / (cfg.nR - 1), (cfg.Z_max - cfg.Z_min) / (cfg.nZ - 1)

    print("[SYSTEM] Packaging particle arrays for Graphing Utilities...")
    _t = prof.mark()

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

    prof.add("pkg: history_tracks -> mock_active/mock_alphas", _t)

    # NOTE: this builds one Python dict per surviving particle, then
    # compute_fluid_moments walks that list in Python. There is no
    # array-based path here yet, so at 1M particles both lines below are a
    # real chunk of wall clock -- they are timed separately to make that
    # visible rather than to hide it inside one "packaging" number.
    _t = prof.mark()
    final_active = [{"status": "confined", "pos": p, "vel": v} for p, v, typ in zip(pos_np, vel_np, type_np) if typ != -1]
    prof.add("pkg: final_active (dict per particle)", _t)

    _t = prof.mark()
    R_centers, density_profile, pressure_profile = engine.compute_fluid_moments(
        final_active, cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.num_radial_bins
    )
    prof.add("pkg: compute_fluid_moments", _t)

    # --- TRUE v_parallel FOR THE PHASE-SPACE MAP ---
    # (x*vy - y*vx)/R is v_TOROIDAL, which only equals v_parallel if B is purely toroidal
    # -- and pitch angle against the ACTUAL field line is the whole quantity separating
    # trapped from passing. Projecting onto the real local b_hat (toroidal + psi-derived
    # poloidal) is what makes that boundary appear instead of a Gaussian blob.
    print("[SYSTEM] Projecting velocities onto local field lines for phase-space map...")
    _t = prof.mark()
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

    prof.add("pkg: v_parallel phase-space projection", _t)

    print("[SYSTEM] Mapping Volumetric Fusion Power ")
    _t = prof.mark()
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
            
    prof.add("pkg: fusion power grid + volume integral", _t)

    print("==================================================")
    print(f" TOTAL INTEGRATED FUSION POWER: {total_fusion_power_watts / 1e6:.2f} MW ")
    print("==================================================")

    return {
        "mock_active": mock_active,
        "mock_alphas": mock_alphas,
        "final_active": final_active,
        "R_centers": R_centers,
        "density_profile": density_profile,
        "pressure_profile": pressure_profile,
        "R_phase": R_phase,
        "v_parallel_phase": v_parallel_phase,
        "P_fusion_grid": P_fusion_grid,
        "total_fusion_power_watts": total_fusion_power_watts,
    }


def run_reactor_steady_state(save_plots=None):
    """Run the reactor loop end to end.

    save_plots gates only the PNG-writing diagnostic calls at the tail. None
    (the default) resolves to `not cfg.PROFILE`, so a profiling run never
    overwrites the committed plots in the repo root; pass True to force the
    plots out of a profiling run, False to suppress them in a normal one.

    The diagnostic *computation* is deliberately not gated:
    package_reactor_results (final_active, compute_fluid_moments, the
    phase-space projection) runs either way, so profile numbers stay
    comparable to the step 4/5/6 baselines.
    """
    print("==================================================")
    print("      PIC FULL CYCLE (LORENTZ PUSH)      ")
    print("==================================================")

    cfg = SimulationConfiguration()

    if save_plots is None:
        save_plots = not bool(getattr(cfg, "PROFILE", False))

    # Out-of-loop phases are timed on their own profiler, so the loop table
    # stays a table about the loop.
    outer = StageProfiler(bool(getattr(cfg, "PROFILE", False)), label="out-of-loop")

    _t = outer.mark()
    eq, engine, pos_tensor, vel_tensor, type_tensor, rho_grid, phi_grid, E_R_grid, E_Z_grid = initialization.initialize_reactor(cfg)
    outer.add("initialize_reactor", _t)

    # The psi-derived poloidal B field, resampled onto the solver grid so the pusher's B
    # lookup reuses the indices/weights the E gather computes. Without it the pusher sees
    # a purely toroidal B whose only gradient is the smooth 1/R falloff -- no mirror ratio
    # along a field line, so nothing is trapped and the v_parallel-vs-R map collapses to
    # the injected Maxwellian.
    _t = outer.mark()
    B_R_pol_grid, B_Z_pol_grid, pol_scale, B_pol_target = compute_poloidal_field_grids(
        eq, cfg.B0, R0=cfg.R0_major, a_minor=0.3, q_target=cfg.Q_SAFETY_TARGET,
        dst_R_min=cfg.R_min, dst_R_max=cfg.R_max, dst_Z_min=cfg.Z_min, dst_Z_max=cfg.Z_max,
        dst_nR=cfg.nR, dst_nZ=cfg.nZ
    )
    outer.add("compute_poloidal_field_grids", _t)
    print(f"[SYSTEM] Poloidal field from Grad-Shafranov psi: B_pol ~ {B_pol_target:.3f} T "
          f"(q = {cfg.Q_SAFETY_TARGET:.1f}, psi rescale x{pol_scale:.3e})")

    # GPU_PARTICLE_THRESHOLD is None, so this always selects the CPU loop: the CPU loop
    # measured faster at every count tested, up to 3,000,000 (step 6, at 1,000,000
    # particles / 200 steps: GPU loop 69.6 s wall vs CPU loop ~32-36 s). The `is not None`
    # guard short-circuits before the comparison, so None never reaches the `>=`.
    # See the constant's definition for how to force the GPU loop.
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

    _t = outer.mark()
    packaged = package_reactor_results(
        cfg, eq, engine, B_R_pol_grid, B_Z_pol_grid,
        pos_np, vel_np, type_np,
        history_tracks, tracked_type, tracked_lastpos,
        tracked_lastvel, tracked_lost,
        rho_grid, T_core, prof=outer,
    )
    outer.add("package_reactor_results (total)", _t)

    if outer.enabled:
        _report_gs_provenance(eq)
        # package_reactor_results' sub-stages are charged to `outer` as well,
        # so its own "(total)" row overlaps them -- read that row as the
        # roll-up, not as another slice to add in.
        outer.report(title="STAGE TIMING -- out-of-loop phases")
    mock_active = packaged["mock_active"]
    mock_alphas = packaged["mock_alphas"]
    R_centers = packaged["R_centers"]
    density_profile = packaged["density_profile"]
    pressure_profile = packaged["pressure_profile"]
    R_phase = packaged["R_phase"]
    v_parallel_phase = packaged["v_parallel_phase"]
    P_fusion_grid = packaged["P_fusion_grid"]

    # Every diagnostic below writes a PNG into the repo root under its fixed
    # name (plot_manifest.py's baseline depends on those names), so they are
    # the only thing save_plots gates.
    if not save_plots:
        print("[SYSTEM] save_plots is off -- skipping diagnostic plots "
              "(no PNGs written).")
    else:
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
    # The reactor run and nothing else. The benchmark sweep (now
    # benchmarks.run_hpc_benchmark), run_plasma_oscillation_test and
    # run_nuclear_reaction_dynamics used to run here too, which made bare
    # `python3 main.py` unsafe as a reactor entry point; call them explicitly
    # if you want them. tests/test_regression.py's compare-plots child script
    # still drives all four, because the plot manifest covers all of them.
    #
    # save_plots=True: this is the full-artifact entry point, it exists to
    # regenerate the plots, so it asks for them explicitly rather than
    # inheriting the cfg.PROFILE default.
    run_reactor_steady_state(save_plots=True)