"""GPU / PyTorch reactor loop, split out of main.py (step 14).

Imports the step-14 leaf modules (profiling, particle_pool, track_store) plus
physics_engine; none of them import back into this one. Opt-in -- main.py's
GPU_PARTICLE_THRESHOLD dispatch stays in main.py alongside the constant.
"""
import numpy as np
import time
import torch

from physics_engine import (
    compute_radiative_cooling_power,
    update_force_balance_field,             # quasi-neutral E_r from thermal-ion force balance (host)
    deposit_ion_charge_density,             # end-of-run weighted CIC density map (host)
    vectorized_gather_and_B_torch,          # large-N (>=100K) GPU-resident kernels
    check_confinement_torch,
    apply_vectorized_collisions_torch,
    compute_alpha_heating_power_torch,
    _vectorized_boris_push_metal_impl,      # eager push; shapes churn every step, so the static-compiled path would recompile constantly
    _vectorized_boris_push_metal_dynamic,   # dynamic-shape compile, handles that churn; None if the compile failed at import
    boris_push_substeps_regather_torch,     # alpha sub-stepping (GPU), field re-gathered per sub-step
    relative_weight_lut,                    # per-species weight / thermal-D weight
)
from profiling import (
    StageProfiler,
    _profile_header,
    _log_device_memory,
    _report_peak_rss,
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


def _push_alphas_torch(cfg, a_pos, a_vel, E_R_grid_t, E_Z_grid_t, B_R_pol_t, B_Z_pol_t, t):
    # Twin of reactor_cpu._push_alphas: sub-stepped Boris with E and B re-gathered
    # at every sub-step. The kernel returns new tensors, so the results are copied
    # back into the pool views rather than rebound (kernel mutation contract).
    p_a, v_a = boris_push_substeps_regather_torch(
        a_pos, a_vel, cfg.CHARGE_ALPHA, cfg.MASS_ALPHA, cfg.reactor_dt, cfg.ALPHA_SUBSTEPS,
        E_R_grid_t, E_Z_grid_t, B_R_pol_t, B_Z_pol_t,
        cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ,
        cfg.B0, cfg.R0_major, t, cfg.b_perturb_initial, cfg.m_mode, cfg.n_mode, cfg.gamma_growth)
    a_pos.copy_(p_a)
    a_vel.copy_(v_a)


def _run_reactor_loop_gpu(cfg, engine, pos_tensor, vel_tensor, type_tensor, rho_grid, phi_grid, E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid):
    # GPU-resident pipeline, selected only when GPU_PARTICLE_THRESHOLD is set to an int
    # <= cfg.initial_thermal_count. It is None by default, so this loop is opt-in -- see
    # that constant's definition for the numbers and for how to force it. The particle
    # tensors never leave the device inside the loop, apart from the bulk pool copied to
    # the host every FIELD_UPDATE_EVERY steps for the force-balance field (only the two
    # nR x nZ E grids go back) and the tracked-particle subset pulled every 20 steps for
    # plotting.
    #
    # The push uses the DYNAMIC-shape compiled kernel where available, falling back to
    # eager if that compile failed at import. STATIC compile is wrong here: the pool
    # shapes this loop pushes (bulk.n_live / alpha.n_live) change nearly every step as
    # particles are injected or lost, and static re-specializes per exact shape, so it
    # would recompile constantly -- it only pays off at fixed, reused shapes (see
    # benchmarks.run_hpc_benchmark). Dynamic handles the churn without recompiling and
    # is worth ~4x on the push alone (0.426 -> 0.098 ms at N=50,000). The loop as a
    # whole is still slower than the CPU path at every size tested, which is why
    # GPU_PARTICLE_THRESHOLD is None.
    #
    # (This comment said "mask_d / mask_a" until step 14 session H. Step 9 retired those
    # masks entirely in favour of the split bulk/alpha pools; the shape-churn argument
    # is unchanged, only the names it cites. run_hpc_benchmark moved to benchmarks.py
    # in step 14 session B.)
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

    # Per-species weight relative to thermal D, indexed by type code, on the
    # device. Twin of the CPU loop's rel_w: energies and counts below read in
    # thermal-D macroparticle units. Equal weights make every entry 1.0.
    rel_w_t = torch.tensor(relative_weight_lut(cfg), device=device, dtype=torch.float32)

    total_injected = cfg.initial_thermal_count
    total_lost = 0
    inventory_history, energy_history_keV = [], []
    temp_history, lawson_triple_product_history = [], []
    trigger_time, T_core_baseline, T_core = None, 0.0, 0.0
    # First step whose tau_E is measured (t > 0 and at least one particle lost).
    # Earlier entries use the fixed tau_E below, so the plot starts here. None if
    # nothing was ever lost. Bookkeeping only: reads counters, writes nothing else.
    lawson_first_measured_step = None

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

        # --- QUASI-NEUTRAL RADIAL ELECTRIC FIELD ---
        # Twin of the CPU loop's block: E from thermal-ion force balance, rebuilt at
        # step 0 and every FIELD_UPDATE_EVERY steps. The profile binning runs on the
        # host, so the bulk pool crosses to the CPU once per update and only the two
        # small (nR x nZ) E grids go back. No dead rows can be present here -- see
        # the CPU twin.
        _t = prof.mark()
        if step % cfg.FIELD_UPDATE_EVERY == 0:
            phi_grid, E_R_grid, E_Z_grid, _ = update_force_balance_field(
                cfg, engine.eq, b_pos.cpu().numpy(), b_vel.cpu().numpy(), b_type.cpu().numpy())
            E_R_grid_t = torch.tensor(E_R_grid, device=device, dtype=torch.float32)
            E_Z_grid_t = torch.tensor(E_Z_grid, device=device, dtype=torch.float32)
        prof.add("E_r force balance", _t)

        _t = prof.mark()
        # --- E-FIELD GATHER + ANALYTIC B-FIELD (GPU, torch.compile'd, dynamic shapes) ---
        # Fed the psi-derived poloidal B grids and cfg.R0_major. That argument slot is R0
        # -- the toroidal 1/R falloff centre and tearing-mode axis -- and passing
        # cfg.B_poloidal (0.3) there put the field's centre outside the plasma entirely.
        _gather_args = (E_R_grid_t, E_Z_grid_t, B_R_pol_t, B_Z_pol_t,
                        cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ,
                        cfg.B0, cfg.R0_major, t, cfg.b_perturb_initial,
                        cfg.m_mode, cfg.n_mode, cfg.gamma_growth)
        # Bulk only; the alphas gather inside their sub-step push below.
        E_b, B_b = vectorized_gather_and_B_torch(b_pos, *_gather_args)
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

        # Alphas are sub-stepped, with the field re-gathered every sub-step (see
        # _push_alphas_torch). Deuterons stay on the single global step.
        _t = prof.mark()
        if alpha.n_live > 0:
            _push_alphas_torch(cfg, a_pos, a_vel, E_R_grid_t, E_Z_grid_t, B_R_pol_t, B_Z_pol_t, t)
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
            # Second return value is the raw simulation-scale keV drained, needed by the
            # energy transfer below.
            # Weighted: sum(w_a dE) / w_thermal, the real energy given up.
            new_energies_kev, alpha_deposited_kev = compute_alpha_heating_power_torch(
                alpha_energies_kev, cfg.reactor_dt, cfg, rel_weights=rel_w_t[a_type.long()])
            new_v_mags = torch.sqrt(2.0 * (new_energies_kev * 1.602e-16) / cfg.MASS_ALPHA)
            safe_v_mags = torch.where(v_mags == 0, torch.full_like(v_mags, 1e-10), v_mags)
            scale_factors = new_v_mags / safe_v_mags
            a_vel.copy_(alpha_vels * scale_factors.unsqueeze(1))

        # The bulk pool holds nothing but types 0 and 1, and the compaction above
        # removed every -1 this step, so its live count IS the confined count --
        # and reading it costs no device sync, unlike the old .sum().item().
        current_confined = int(bulk.n_live)
        inventory_history.append(current_confined)

        # --- ALPHA -> BULK ENERGY TRANSFER (energy conservation) ---
        # Energy drained from the alphas used to vanish -- removed from the fast
        # population and given to nothing, so the thermal plasma never felt the heating.
        # Deposit it by scaling every bulk speed by one factor. Both sides are
        # weighted (twin of the CPU loop), so real energy is conserved even when
        # the species weights differ.
        r_b = rel_w_t[b_type.long()]
        if alpha_deposited_kev > 0.0 and bulk.n_live > 0:
            bulk_energy_kev = float((r_b * (0.5 * cfg.m_deuterium * torch.sum(b_vel**2, dim=1))).sum().item()) / 1.602e-16
            if bulk_energy_kev > 0.0:
                boost = float(np.sqrt(1.0 + alpha_deposited_kev / bulk_energy_kev))
                b_vel.copy_(b_vel * boost)

        # Weighted macroparticle count of the bulk (thermal-D units).
        r_sum = float(r_b.sum().item()) if bulk.n_live > 0 else 0.0
        if bulk.n_live > 0:
            current_energy_joules = float((r_b * (0.5 * cfg.m_deuterium * torch.sum(b_vel**2, dim=1))).sum().item())
        else:
            current_energy_joules = 0.0

        current_energy_keV = current_energy_joules / (cfg.e_charge * 1000.0)
        energy_history_keV.append(current_energy_keV)

        current_amp = cfg.b_perturb_initial * np.exp(cfg.gamma_growth * t)

        # --- T_core IS A TEMPERATURE, NOT A TOTAL ---
        # Assigning the plasma's TOTAL stored energy here made the "core temperature"
        # read ~50,000 keV, and every consumer (the Lawson triple product, the
        # quench baseline) was fed that. For a 3D Maxwellian
        # <E> = (3/2)kT, so kT = (2/3) * <E>, with <E> per real ion (weighted).
        n_bulk = max(r_sum, 1.0)
        T_core_kinetic = (2.0 / 3.0) * (current_energy_keV / n_bulk)

        if current_amp > cfg.MAX_ISLAND_WIDTH_THRESHOLD and not cfg.SPI_TRIGGERED:
            cfg.SPI_TRIGGERED = True
            trigger_time = t
            T_core_baseline = T_core_kinetic
            print(f"  [EMERGENCY] Magnetic Island exceeded {cfg.MAX_ISLAND_WIDTH_THRESHOLD*100}% minor radius!")
            print(f"  [SYSTEM] Firing Shattered Pellet Injection (SPI) at t={t*1000:.2f} ms...")
            print(f"  [SYSTEM] Pre-disruption core temperature: {T_core_baseline:.3f} keV")

        # --- REPORTED TEMPERATURE IS MEASURED, NOT PRESCRIBED ---
        # T_core is the kinetic temperature of the bulk in both regimes, before and
        # after the SPI trigger, so a quench shows up only because the drain below
        # removes energy from the particles.
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

        prof.add("alpha heating + energy transfer", _t)

        # energy_history_keV / T_core_kinetic above are recorded BEFORE the drain, which
        # is correct: the radiation covers [t, t+dt] and so lands in the next step's
        # ledger, not retroactively in this one.
        _t = prof.mark()
        loss_rate = total_lost / t if t > 0 else 1e-5
        tau_E = min(current_confined / loss_rate if loss_rate > 0 else 0.5, 3.0)
        if lawson_first_measured_step is None and t > 0 and total_lost > 0:
            lawson_first_measured_step = step
        # The triple product's density is the confined real ion count (weighted
        # macroparticle count times WEIGHT[0]) over the psi_edge volume. Units:
        # m^-3 * keV * s, matching cfg.lawson_target and the axis label.
        n_e_reactor = (r_sum * cfg.WEIGHT[0]) / cfg.PLASMA_VOLUME_M3
        lawson_triple_product_history.append(n_e_reactor * T_core * tau_E)

        temp_history.append(T_core)
        prof.add("Lawson diagnostics", _t)

        if profiling and (step + 1) % 500 == 0:
            _log_device_memory(device, step + 1)

        if (step + 1) % 500 == 0:
            max_E, max_phi = np.max(np.hypot(E_R_grid, E_Z_grid)), np.max(np.abs(phi_grid))
            print(f"  Step {step+1:04d}/{cfg.reactor_num_steps} | Confined: {current_confined:,} | Max |E|: {max_E:.3e} V/m | Max |Phi|: {max_phi:.2e} V")

    if profiling:
        if _sync is not None:
            _sync()
        loop_wall = time.perf_counter() - loop_wall_start
        prof.report(wall_total=loop_wall,
                    title=f"STAGE TIMING -- reactor loop (GPU / {device.type})")
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

    # End-of-run ion density map: weighted CIC of every live ion, computed once.
    rho_grid = deposit_ion_charge_density(cfg, pos_np, type_np)

    # Flat-array accumulation -> the dict payload, once. This is the only place
    # the sampled vertices and the device-side last_pos/last_vel cross to the host.
    (history_tracks, tracked_type, tracked_lastpos,
     tracked_lastvel, tracked_lost) = tracks.to_dicts()
    return (pos_np, vel_np, type_np, history_tracks, tracked_type, tracked_lastpos, tracked_lastvel,
            tracked_lost,
            total_injected, total_lost,
            inventory_history, energy_history_keV,
            temp_history, trigger_time, T_core,
            lawson_triple_product_history, lawson_first_measured_step, rho_grid, phi_grid)
