"""CPU / Numba reactor loop, split out of main.py (step 14).

Imports the step-14 leaf modules (profiling, kernels, particle_pool,
track_store) plus physics_engine; none of them import back into this one.
"""
import numpy as np
import time
import torch

from physics_engine import (
    compute_radiative_cooling_power,
    compute_radiation_losses_grid,          # vectorized radiation-loss grid reduction
    evaluate_q_factors,
    compute_cic_charge_density,
    vectorized_boris_push_numba_fallback,
    _vectorized_boris_push_metal_impl,      # eager push; shapes churn every step, so the static-compiled path would recompile constantly
    check_confinement_flux,                 # psi-surface confinement, replaces the circular boundary
    vectorized_boris_push_numba_substeps    # alpha sub-stepping (CPU)
)
from profiling import (
    StageProfiler,
    _profile_header,
    _report_peak_rss,
    _report_sor_histogram,
)
from kernels import (
    vectorized_gather_and_B_into,
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
    pos_init = pos_tensor.cpu().numpy()
    vel_init = vel_tensor.cpu().numpy()
    type_init = type_tensor.cpu().numpy()

    # Stable per-particle IDs. Lost particles are physically removed from the pools, not
    # just flagged type=-1, which shifts every row index -- so trajectory bookkeeping
    # cannot be keyed on array position.
    n_init = len(pos_init)
    next_pid = n_init

    # pid -> current row, dense. This replaces np.searchsorted(pid_np, want),
    # which was correct only because pid_np happened to be sorted ascending
    # (monotonic ids, and compaction preserves order). Step 9 splits the bulk and
    # alpha pools and retires that invariant permanently, at which point a
    # searchsorted would not fail loudly -- it would silently return the wrong
    # row and re-point a track at an unrelated particle. A dense map makes the
    # lookup one gather and assumes no ordering at all.
    # Sized from the largest pid the run can reach, NOT from n_init: pids are
    # never reused, so injection walks them past the initial count.
    pid_capacity = _pid_capacity_bound(cfg, n_init)
    pid_row = np.full(pid_capacity, -1, dtype=np.int64)
    pid_row[:n_init] = np.arange(n_init, dtype=np.int64)
    # pid -> track slot, -1 when the pid is not plotted. Replaces the
    # `dpid in history_tracks` dict membership test in the compaction below.
    pid_slot = np.full(pid_capacity, -1, dtype=np.int64)
    # pid -> which pool the row index above is relative to: 0 = bulk, 1 = alpha.
    # With the pools split, pid_row alone is ambiguous -- row 7 means a different
    # particle in each pool -- so the two arrays are only meaningful together.
    #
    # WRITE-ONCE, at birth, and deliberately NOT updated by compaction. That is
    # sound rather than an oversight: species never migrates. type is written only
    # at birth (0/1/2) and at death (-1) -- THERMALIZATION_ENERGY_KEV gates whether
    # an alpha keeps heating, it does not reclassify it -- so a particle can never
    # change pool. Only its row moves, and pid_row carries that.
    pid_pool = np.full(pid_capacity, -1, dtype=np.int8)
    pid_pool[:n_init] = 0

    # --- SPLIT POOLS (step 9) ---
    # bulk holds types 0 and 1, alpha holds type 2. The masks that used to select
    # them out of one mixed pool (mask_d, mask_a, mask_valid, mask_alphas) are
    # gone: each kernel now runs on a contiguous [:n_live] view with no gather,
    # no .copy() and no masked write-back.
    #
    # ORDER IS LOAD-BEARING. The single pool was always ascending by pid --
    # injection appends and compaction is an order-preserving forward gather --
    # so a boolean mask over it yielded each species in ascending-pid order. Each
    # split pool is independently ascending by pid, so both the CIC deposition
    # over the bulk and the alpha energy accumulation see the SAME sequence of
    # particles in the SAME order as before. Neither float reduction reassociates.
    bulk_capacity, alpha_capacity = _pool_capacity_bounds(cfg, n_init)
    bulk = _ParticlePool(bulk_capacity, device=None)
    alpha = _ParticlePool(alpha_capacity, device=None)
    bulk.add(pos_init, vel_init, 0, np.arange(n_init, dtype=np.int64))
    bulk.type[:n_init] = type_init   # preserve any non-zero species from init
    b_pos, b_vel, b_type, b_pid = bulk.views()
    a_pos, a_vel, a_type, a_pid = alpha.views()

    n_track = min(1000, n_init)
    # 1000 initial thermals + the 1000-each NBI and alpha caps below.
    tracks = _TrackStore(_MAX_TRACKED_SLOTS, device=None, host_dtype=np.float32)
    init_slots = tracks.add_slots(b_pid[:n_track], 0,
                                  b_pos[:n_track], b_vel[:n_track])
    pid_slot[b_pid[:n_track]] = init_slots
    tracked_nbis = 0
    tracked_alphas = 0

    # Per-step field buffers, allocated once and reused. vectorized_gather_and_B
    # used to return a fresh pair of (N, 3) float32 arrays every step -- 24 MB
    # allocated and freed per step at 1M particles, ~10,000 times over a full run.
    # The loop now owns the storage and passes [:n] views to the _into kernel,
    # which writes every row it is given, so stale rows past n are never read.
    # Capacity starts above the initial count (NBI and alpha injection grow the
    # pools) and doubles only if the live count outruns it.
    # One field-buffer pair per pool, each sized from that pool's capacity rather
    # than a second, independently-grown number. Two capacities that can disagree
    # is exactly the bug this avoids: vectorized_gather_and_B_into does not
    # bounds-check its E_out/B_out stores, so a field buffer shorter than n_live
    # is a silent out-of-range write, not an IndexError.
    E_buf_b = np.empty((bulk.capacity, 3), dtype=np.float32)
    B_buf_b = np.empty((bulk.capacity, 3), dtype=np.float32)
    E_buf_a = np.empty((alpha.capacity, 3), dtype=np.float32)
    B_buf_a = np.empty((alpha.capacity, 3), dtype=np.float32)

    total_injected = cfg.initial_thermal_count
    total_lost = 0
    inventory_history, energy_history_keV, instability_amp_history = [], [], []
    time_history, temp_history, rad_power_history = [], [], []
    trigger_time, T_core_baseline, T_core = None, 0.0, 0.0

    alpha_heating_power_history_MW, external_heating_power_history_MW = [], []
    bremsstrahlung_power_history_MW, cyclotron_power_history_MW = [], []
    q_sci_history, q_eng_history, lawson_triple_product_history = [], [], []

    print("[SYSTEM] Igniting time-domain reactor loop on CPU/Numba (fastest path at this particle count)...")

    # --- Profiling state (inert unless cfg.PROFILE) ---
    profiling = bool(getattr(cfg, "PROFILE", False))
    prof = StageProfiler(profiling, label="CPU path")
    # Sub-stages of the deuteron push, on their own profiler so they do not
    # double-count inside the main table. The push "stage" is not just the
    # push: it also pays for the mask, four fancy-index gathers (pos, vel,
    # B, E) and two masked scatters, all over the full particle array.
    push_prof = StageProfiler(profiling, label="deuteron push breakdown")
    sor_iter_counts = []
    if profiling:
        _profile_header(cfg, "CPU / Numba path", bulk.n_live + alpha.n_live)
    loop_wall_start = time.perf_counter() if profiling else None

    for step in range(cfg.reactor_num_steps):
        t = step * cfg.reactor_dt
        time_history.append(t)

        # --- BATCH NBI INJECTION (tapering source) ---
        # A fixed batch forever meant inventory only climbed and no steady state was
        # reachable. The source decays as exp(-step/tau), with fractional rates realised
        # stochastically so the beam thins out smoothly.
        _t = prof.mark()
        if step % cfg.inject_every_n_steps == 0 and step > 0:
            nbi_rate = cfg.NBI_BATCH_SIZE * np.exp(-step / cfg.NBI_DECAY_TAU_STEPS)
            batch_size = int(nbi_rate)
            if np.random.random() < (nbi_rate - batch_size):
                batch_size += 1

            if batch_size > 0:
                psi_bounds = (engine.eq.psi_grid, engine.eq.psi_core, engine.eq.psi_R_min, engine.eq.psi_R_max,
                              engine.eq.psi_Z_min, engine.eq.psi_Z_max, engine.eq.psi_nR, engine.eq.psi_nZ)
                p_nbi, v_nbi = engine.inject_neutral_beam_cartesian(num_ions=batch_size, E_keV=cfg.nbi_energy_keV, psi_bounds=psi_bounds)
                _require_pid_capacity(next_pid, batch_size, pid_capacity, "CPU NBI injection")
                new_pids = np.arange(next_pid, next_pid + batch_size, dtype=np.int64)
                # Writes into rows [n_live : n_live+batch] -- no reallocation.
                # Assigning float64 p_nbi into the float32 pool rounds exactly as
                # the old vstack(...).astype(np.float32) did; verified
                # bit-identical on real injector output before this change.
                base_row = bulk.add(p_nbi, v_nbi, 1, new_pids)
                b_pos, b_vel, b_type, b_pid = bulk.views()
                pid_row[new_pids] = np.arange(base_row, base_row + batch_size,
                                              dtype=np.int64)
                pid_pool[new_pids] = 0
                next_pid += batch_size
                total_injected += batch_size

                # The old loop tracked the first (1000 - tracked_nbis) of the
                # batch and skipped the rest; take that same prefix.
                n_new = min(batch_size, 1000 - tracked_nbis)
                if n_new > 0:
                    rows = np.arange(base_row, base_row + n_new, dtype=np.int64)
                    slots = tracks.add_slots(new_pids[:n_new], 1,
                                             b_pos[rows], b_vel[rows])
                    pid_slot[new_pids[:n_new]] = slots
                    tracked_nbis += n_new

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

            _require_pid_capacity(next_pid, alpha_batch, pid_capacity, "CPU alpha injection")
            new_pids = np.arange(next_pid, next_pid + alpha_batch, dtype=np.int64)
            base_row = alpha.add(p_alpha, v_alpha, 2, new_pids)
            a_pos, a_vel, a_type, a_pid = alpha.views()
            pid_row[new_pids] = np.arange(base_row, base_row + alpha_batch,
                                          dtype=np.int64)
            pid_pool[new_pids] = 1
            next_pid += alpha_batch

            n_new = min(alpha_batch, 1000 - tracked_alphas)
            if n_new > 0:
                rows = np.arange(base_row, base_row + n_new, dtype=np.int64)
                slots = tracks.add_slots(new_pids[:n_new], 2,
                                         a_pos[rows], a_vel[rows])
                pid_slot[new_pids[:n_new]] = slots
                tracked_alphas += n_new

        prof.add("alpha injection", _t)

        _t = prof.mark()
        # Bulk pool only, and no mask: the bulk pool holds nothing but types 0
        # and 1. Dead rows cannot be present either -- check_confinement_flux
        # flags them later in the step and the compaction directly after it
        # removes every one, so by the time the next step reaches this line the
        # pool is all-live. Same particles, same ascending-pid order as the old
        # mask_valid selection, so the float accumulation is unchanged.
        R_coords = np.sqrt(b_pos[:, 0]**2 + b_pos[:, 1]**2)
        Z_coords = b_pos[:, 2]
        charges = np.full(bulk.n_live, cfg.e_charge)
        rho_grid = compute_cic_charge_density(R_coords, Z_coords, charges, cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ, False)
        prof.add("CIC deposition", _t)

        _t = prof.mark()
        if step > 0:
            # Warm-start the Poisson SOR solve from last step's converged phi: same
            # answer within the same tol, far fewer iterations to reach it.
            phi_grid, E_R_grid, E_Z_grid, _sor_iters = engine.solve_fields(
                rho_grid, cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max,
                phi_init=phi_grid, return_iters=True)
            if profiling:
                sor_iter_counts.append(_sor_iters)
        prof.add("solve_fields (Poisson)", _t)

        # Gather runs per pool, each into its own buffer pair.
        _t = prof.mark()
        E_b = E_buf_b[:bulk.n_live]
        B_b = B_buf_b[:bulk.n_live]
        vectorized_gather_and_B_into(
            b_pos, E_b, B_b,
            E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid,
            cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ,
            cfg.B0, cfg.R0_major, t, cfg.b_perturb_initial, cfg.m_mode, cfg.n_mode, cfg.gamma_growth
        )
        E_a = E_buf_a[:alpha.n_live]
        B_a = B_buf_a[:alpha.n_live]
        if alpha.n_live > 0:
            vectorized_gather_and_B_into(
                a_pos, E_a, B_a,
                E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid,
                cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ,
                cfg.B0, cfg.R0_major, t, cfg.b_perturb_initial, cfg.m_mode, cfg.n_mode, cfg.gamma_growth
            )
        prof.add("vectorized_gather_and_B", _t)

        # --- PARTICLE PUSH: Numba/CPU below PUSH_GPU_THRESHOLD, eager-GPU above it ---
        # mask_d is gone. The bulk pool IS the deuteron set, so the kernel gets a
        # contiguous view and mutates it in place -- no gather, no .copy(), no
        # masked write-back. That trio was ~65% of this stage's cost.
        _t = prof.mark()
        _ts = push_prof.mark()
        push_prof.add("mask + pos/vel gather", _ts)
        if bulk.n_live > 0:
            _ts = push_prof.mark()
            _boris_push_adaptive(b_pos, b_vel, cfg.e_charge, cfg.m_deuterium,
                                 B_b, E_b, cfg.reactor_dt, cfg.HPC_DEVICE)
            push_prof.add("push call (+ B/E gather)", _ts)
            _ts = push_prof.mark()
            push_prof.add("masked write-back", _ts)
        prof.add("deuteron Boris push", _t)

        # Alphas are sub-stepped: at 3.5 MeV they cover ~1.3e-2 m per global 1 ns step
        # against a ~2.2e-2 m Larmor radius, under two samples per gyro-arc. Deuterons
        # stay on the single global step (Larmor radius ~5e-4 m, already resolved).
        _t = prof.mark()
        # mask_a is gone for the same reason: the alpha pool IS the alpha set.
        # The views are already contiguous float32, so the ascontiguousarray
        # copies the old form needed are gone too.
        if alpha.n_live > 0:
            vectorized_boris_push_numba_substeps(
                a_pos, a_vel, cfg.CHARGE_ALPHA, cfg.MASS_ALPHA,
                B_a, E_a, cfg.reactor_dt, cfg.ALPHA_SUBSTEPS
            )
        prof.add("alpha substep push", _t)

        _t = prof.mark()
        # The numba kernel mutates vel_arr in place and returns the same array, so
        # this rebinding is a no-op and vel_np stays the pool view. Do not replace
        # it with anything that allocates -- see the torch twin in the GPU loop.
        # Bulk pool only. The kernel's own guard is
        #     if (type == 0 or type == 1) and np.random.rand() < nu_c*dt
        # and `and` short-circuits, so a type-2 row never draws. Running it on the
        # alpha pool would therefore be a guaranteed no-op, and skipping it does
        # NOT perturb the RNG stream: the single-pool call drew for exactly the
        # bulk rows, in ascending-pid order, which is precisely what this call
        # now walks. Mutates in place and returns the same array, so the
        # rebinding is a no-op and b_vel stays the pool view.
        b_vel = apply_vectorized_collisions(b_vel, b_type, cfg.nu_c, cfg.reactor_dt)
        prof.add("apply_vectorized_collisions", _t)

        # Confinement against the psi flux surface, not a circle
        _t = prof.mark()
        # Both pools; total_lost sums them. The kernel flags in place (type = -1)
        # and returns a count, so each call writes straight into its own pool.
        # Bulk first, matching the order the single pool's rows were visited.
        psi_args = (engine.eq.psi_grid, engine.eq.psi_edge,
                    engine.eq.psi_R_min, engine.eq.psi_R_max,
                    engine.eq.psi_Z_min, engine.eq.psi_Z_max,
                    engine.eq.psi_nR, engine.eq.psi_nZ)
        lost_b = check_confinement_flux(b_pos, b_type, *psi_args)
        lost_a = check_confinement_flux(a_pos, a_type, *psi_args) if alpha.n_live > 0 else 0
        newly_lost = lost_b + lost_a
        prof.add("check_confinement_flux", _t)
        total_lost += newly_lost

        # --- WALL LOSS COMPACTION ---
        _t = prof.mark()
        # check_confinement_flux only FLAGS wall strikes (type = -1). Leaving them in the
        # arrays meant the pools only ever grew, every kernel paid for dead particles, and
        # the confined count could never fall. Trajectory history is pid-keyed, so it
        # survives the row-index shift and the diagnostics payload is unchanged.
        if newly_lost > 0:
            # Each pool compacts independently; rows are renumbered within a pool,
            # which is exactly what pid_row means now that pid_pool says which
            # pool the index belongs to.
            for _pool, _pos, _vel, _type, _pid, _lost in (
                    (bulk, b_pos, b_vel, b_type, b_pid, lost_b),
                    (alpha, a_pos, a_vel, a_type, a_pid, lost_a)):
                if _lost == 0:
                    continue
                alive = _type != -1
                dead_rows = np.nonzero(~alive)[0]
                if dead_rows.size > 0:
                    dead_pids = _pid[dead_rows]
                    dead_slots = pid_slot[dead_pids]
                    is_tracked = dead_slots >= 0
                    if np.any(is_tracked):
                        t_rows = dead_rows[is_tracked]
                        t_slots = dead_slots[is_tracked]
                        impact_pos = _pos[t_rows]
                        # The impact point, off the normal sampling cadence, so
                        # the crimson trace ends at the wall rather than at the
                        # last 20-step tick. Always a track's final vertex:
                        # pid_row goes to -1 below, so the sampler can never
                        # touch it again.
                        tracks.record_impact(t_slots, impact_pos, _vel[t_rows])
                    pid_row[dead_pids] = -1

                # In-place forward compaction. alive_rows is ascending, so every
                # destination has been read before it is written, and the
                # survivors keep their ascending-pid order.
                _pool.compact(np.nonzero(alive)[0])

            b_pos, b_vel, b_type, b_pid = bulk.views()
            a_pos, a_vel, a_type, a_pid = alpha.views()
            # Rebuild the row map for whichever pools moved. pid_pool is NOT
            # touched: a particle cannot change pool, only rows move.
            if lost_b:
                pid_row[b_pid] = np.arange(bulk.n_live, dtype=np.int64)
            if lost_a:
                pid_row[a_pid] = np.arange(alpha.n_live, dtype=np.int64)
        prof.add("wall-loss compaction", _t)

        # --- TRAJECTORY SAMPLING (pid-keyed, so removal cannot corrupt it) ---
        _t = prof.mark()
        # Alphas are sampled far more often than thermals: their orbit is only ~2.2e-2 m
        # across, so a 20-step cadence (~0.26 m of travel) aliases it away.
        sample_thermal = (step % 20 == 0)
        sample_alpha = (step % cfg.ALPHA_HISTORY_EVERY == 0)
        if (sample_thermal or sample_alpha) and tracks.n_slots > 0:
            # One mask for (due this step AND still alive), then a single fancy
            # index per pool. A row index only means something relative to a
            # pool, so pid_pool splits the due set before the gather -- this is
            # the reader that made pid_pool necessary.
            #
            # Each slot is written at most once per step, so splitting the append
            # into two blocks cannot disturb a track's internal ordering: to_dicts
            # regroups by a stable argsort on the slot column.
            due = tracks.slots_due(sample_thermal, sample_alpha)
            if due.size > 0:
                due_pids = tracks.pids[due]
                rows = pid_row[due_pids]
                keep = rows >= 0
                sel_slots = due[keep]
                sel_rows = rows[keep]
                sel_pools = pid_pool[due_pids[keep]]
                for _which, _pos, _vel in ((0, b_pos, b_vel), (1, a_pos, a_vel)):
                    grp = sel_pools == _which
                    if not np.any(grp):
                        continue
                    g_slots = sel_slots[grp]
                    g_rows = sel_rows[grp]
                    sampled_pos = _pos[g_rows]
                    tracks.append_samples(g_slots, sampled_pos)
                    tracks.set_last_host(g_slots, sampled_pos, _vel[g_rows])
        prof.add("trajectory sampling", _t)

        _t = prof.mark()
        # Reads the alpha pool, writes the bulk pool below. ORDER IS PRESERVED
        # EXACTLY: alpha energies are updated here, the deposit happens after
        # that and before the bulk energy is recorded.
        alpha_deposited_kev = 0.0
        if alpha.n_live > 0:
            alpha_vels = a_vel.astype(np.float64)
            v_mags = np.linalg.norm(alpha_vels, axis=1)
            alpha_energies_kev = (0.5 * cfg.MASS_ALPHA * (v_mags**2)) / 1.602e-16
            new_energies_kev, alpha_power_mw, alpha_deposited_kev = engine.compute_alpha_heating_power(
                alpha_energies_kev, cfg.reactor_dt, cfg)
            new_v_mags = np.sqrt(2.0 * (new_energies_kev * 1.602e-16) / cfg.MASS_ALPHA)
            scale_factors = new_v_mags / np.where(v_mags == 0, 1e-10, v_mags)
            alpha_vels *= scale_factors[:, np.newaxis]
            a_vel[:] = alpha_vels.astype(np.float32)
        else:
            alpha_power_mw = 0.0

        alpha_heating_power_history_MW.append(alpha_power_mw)
        external_heating_power_history_MW.append(cfg.EXTERNAL_HEATING_MW)

        # The bulk pool holds nothing but types 0 and 1, and the compaction above
        # removed every -1 this step, so its live count IS the confined count.
        current_confined = int(bulk.n_live)
        inventory_history.append(current_confined)

        # --- ALPHA -> BULK ENERGY TRANSFER (energy conservation) ---
        # Energy drained from the alphas used to vanish -- removed from the fast
        # population and given to nothing, so the thermal plasma never felt the heating.
        # Deposit it by scaling bulk speeds, using the simulation-scale keV (no
        # macro_weight) so particles stay self-consistent; the MW figure above is
        # separately scaled for reactor-equivalent output.
        if alpha_deposited_kev > 0.0 and bulk.n_live > 0:
            bulk_vels = b_vel.astype(np.float64)
            bulk_energy_kev = float(np.sum(0.5 * cfg.m_deuterium * np.sum(bulk_vels**2, axis=1))) / 1.602e-16
            if bulk_energy_kev > 0.0:
                boost = np.sqrt(1.0 + alpha_deposited_kev / bulk_energy_kev)
                b_vel[:] = (bulk_vels * boost).astype(np.float32)

        if bulk.n_live > 0:
            current_energy_joules = np.sum(0.5 * cfg.m_deuterium * (np.linalg.norm(b_vel.astype(np.float64), axis=1)**2))
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
            print(f" [EMERGENCY] Magnetic Island exceeded {cfg.MAX_ISLAND_WIDTH_THRESHOLD*100}% minor radius!")
            print(f" [SYSTEM] Firing Shattered Pellet Injection (SPI) at t={t*1000:.2f} ms...")
            print(f" [SYSTEM] Pre-disruption core temperature: {T_core_baseline:.3f} keV")

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
                    drain = np.sqrt(1.0 - loss_fraction)
                    b_vel[:] = (b_vel.astype(np.float64) * drain).astype(np.float32)
        else:
            # No impurities before SPI fires, so nothing radiates
            P_rad = 0.0

        # energy_history_keV / T_core_kinetic above are recorded BEFORE the drain, which
        # is correct: the radiation covers [t, t+dt] and so lands in the next step's
        # ledger, not retroactively in this one.

        prof.add("alpha heating + energy transfer", _t)

        # One vectorized Numba call over the whole grid, replacing nR*nZ Python-level
        # calls into compute_radiation_losses.
        _t = prof.mark()
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
        prof.add("radiation + Q-factor diagnostics", _t)

        if (step + 1) % 500 == 0:
            max_rho, max_phi = np.max(rho_grid), np.max(np.abs(phi_grid))
            print(f"  Step {step+1:04d}/{cfg.reactor_num_steps} | Confined: {current_confined:,} | Max Rho: {max_rho:.3e} | Max |Phi|: {max_phi:.2e} V")

    if profiling:
        loop_wall = time.perf_counter() - loop_wall_start
        prof.report(wall_total=loop_wall,
                    title="STAGE TIMING -- reactor loop (CPU / Numba)")
        push_prof.report(wall_total=prof.totals.get("deuteron Boris push"),
                         title="BREAKDOWN -- deuteron Boris push stage")
        _report_sor_histogram(sor_iter_counts)
        _report_peak_rss("[PROFILE][CPU]")

    # --- REJOIN THE POOLS (once, after the loop) ---
    # package_reactor_results still wants single pos_np/vel_np/type_np arrays.
    # A bare concatenate([bulk, alpha]) would put every alpha after every bulk
    # particle, which is a different row order from the single-pool version and
    # would fail a bit-identity check for a reason that has nothing to do with
    # correctness. The single pool was always ascending by pid -- injection
    # appends and compaction preserves relative order -- so sorting the
    # concatenation by pid restores exactly that order. pids are unique across
    # both pools (one shared counter), so the sort is total; kind="stable" is
    # belt-and-braces.
    pos_np = np.concatenate((b_pos, a_pos), axis=0)
    vel_np = np.concatenate((b_vel, a_vel), axis=0)
    type_np = np.concatenate((b_type, a_type), axis=0)
    pid_all = np.concatenate((b_pid, a_pid), axis=0)
    legacy_order = np.argsort(pid_all, kind="stable")
    pos_np = np.ascontiguousarray(pos_np[legacy_order])
    vel_np = np.ascontiguousarray(vel_np[legacy_order])
    type_np = np.ascontiguousarray(type_np[legacy_order])

    # Tracked-particle metadata is returned as pid-keyed dicts, which removal cannot
    # corrupt -- row indices no longer survive a step now that losses are compacted out.
    # The loop accumulates into flat arrays; the dicts are rebuilt once, here, so the
    # return signature and package_reactor_results are unchanged.
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
