"""CPU / Numba reactor loop, split out of main.py (step 14).

Imports the step-14 leaf modules (profiling, kernels, particle_pool,
track_store) plus physics_engine; none of them import back into this one.
"""
import numpy as np
import time
import torch

from physics_engine import (
    compute_radiative_cooling_power,
    update_force_balance_field,             # quasi-neutral E_r from thermal-ion force balance
    deposit_ion_charge_density,             # end-of-run weighted CIC density map
    vectorized_boris_push_numba_fallback,
    _vectorized_boris_push_metal_impl,      # eager push; shapes churn every step, so the static-compiled path would recompile constantly
    check_confinement_flux,                 # psi-surface confinement, replaces the circular boundary
)
from profiling import (
    StageProfiler,
    _profile_header,
    _report_peak_rss,
)
from kernels import (
    vectorized_gather_and_B_into,
    boris_push_substeps_regather,           # alpha sub-stepping, field re-gathered per sub-step
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


def _push_alphas(cfg, a_pos, a_vel, E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid, t):
    # Alphas are sub-stepped: at 3.5 MeV they cover ~1.3e-2 m per global 1 ns step
    # against a ~2.2e-2 m Larmor radius. E and B are re-gathered at every sub-step
    # inside the kernel; holding them at the start-of-step position collapsed every
    # alpha's pitch to xi = -1 (see kernels.boris_push_substeps_regather). Mutates
    # a_pos / a_vel in place. Twin: reactor_gpu._push_alphas_torch.
    boris_push_substeps_regather(
        a_pos, a_vel, cfg.CHARGE_ALPHA, cfg.MASS_ALPHA, cfg.reactor_dt, cfg.ALPHA_SUBSTEPS,
        E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid,
        cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ,
        cfg.B0, cfg.R0_major, t, cfg.b_perturb_initial, cfg.m_mode, cfg.n_mode, cfg.gamma_growth)


def _run_reactor_loop_cpu(cfg, engine, pos_tensor, vel_tensor, type_tensor, rho_grid, phi_grid, E_R_grid, E_Z_grid,
                          B_R_pol_grid, B_Z_pol_grid):
    # The whole per-step pipeline runs on Numpy/Numba. At this particle count (~10k-20k,
    # growing via injection) prange beats MPS outright: dispatching dozens of small
    # per-step kernels (gather, push, collisions, confinement) costs more in Metal
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
    # split pool is independently ascending by pid, so the profile binning over
    # the bulk and the alpha energy accumulation see each species in ascending-pid
    # order, and the end-of-run rejoin below restores the single-pool order.
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
    # Sized from the bulk pool's capacity rather than a second, independently-grown
    # number. Two capacities that can disagree is exactly the bug this avoids:
    # vectorized_gather_and_B_into does not bounds-check its E_out/B_out stores, so
    # a field buffer shorter than n_live is a silent out-of-range write, not an
    # IndexError. The alpha pool needs no buffer: _push_alphas gathers its field
    # inside the sub-step kernel.
    E_buf_b = np.empty((bulk.capacity, 3), dtype=np.float32)
    B_buf_b = np.empty((bulk.capacity, 3), dtype=np.float32)

    total_injected = cfg.initial_thermal_count
    total_lost = 0
    inventory_history, energy_history_keV = [], []
    temp_history, lawson_triple_product_history = [], []
    trigger_time, T_core_baseline, T_core = None, 0.0, 0.0
    # First step whose tau_E is measured (t > 0 and at least one particle lost).
    # Earlier entries use the fixed tau_E below, so the plot starts here. None if
    # nothing was ever lost. Bookkeeping only: reads counters, writes nothing else.
    lawson_first_measured_step = None

    print("[SYSTEM] Igniting time-domain reactor loop on CPU/Numba (fastest path at this particle count)...")

    # --- Profiling state (inert unless cfg.PROFILE) ---
    profiling = bool(getattr(cfg, "PROFILE", False))
    prof = StageProfiler(profiling, label="CPU path")
    # Sub-stages of the deuteron push, on their own profiler so they do not
    # double-count inside the main table. The push "stage" is not just the
    # push: it also pays for the mask, four fancy-index gathers (pos, vel,
    # B, E) and two masked scatters, all over the full particle array.
    push_prof = StageProfiler(profiling, label="deuteron push breakdown")
    if profiling:
        _profile_header(cfg, "CPU / Numba path", bulk.n_live + alpha.n_live)
    loop_wall_start = time.perf_counter() if profiling else None

    for step in range(cfg.reactor_num_steps):
        t = step * cfg.reactor_dt

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

        # --- QUASI-NEUTRAL RADIAL ELECTRIC FIELD ---
        # E from thermal-ion radial force balance (no rotation), rebuilt from the
        # current bulk pool at step 0 and every FIELD_UPDATE_EVERY steps. The bulk
        # pool is all-live here (the compaction below removes every -1 each step).
        # Twin: the same call in reactor_gpu on host copies of the pool.
        _t = prof.mark()
        if step % cfg.FIELD_UPDATE_EVERY == 0:
            phi_grid, E_R_grid, E_Z_grid, _ = update_force_balance_field(
                cfg, engine.eq, b_pos, b_vel, b_type)
        prof.add("E_r force balance", _t)

        # Bulk gather. The alphas gather inside their sub-step push below.
        _t = prof.mark()
        E_b = E_buf_b[:bulk.n_live]
        B_b = B_buf_b[:bulk.n_live]
        vectorized_gather_and_B_into(
            b_pos, E_b, B_b,
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

        # Alphas are sub-stepped, with the field re-gathered every sub-step (see
        # _push_alphas). Deuterons stay on the single global step (Larmor radius
        # ~5e-4 m, already resolved). The alpha pool IS the alpha set, so the kernel
        # gets the contiguous float32 views and mutates them in place.
        _t = prof.mark()
        if alpha.n_live > 0:
            _push_alphas(cfg, a_pos, a_vel, E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid, t)
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
            new_energies_kev, alpha_deposited_kev = engine.compute_alpha_heating_power(
                alpha_energies_kev, cfg.reactor_dt, cfg)
            new_v_mags = np.sqrt(2.0 * (new_energies_kev * 1.602e-16) / cfg.MASS_ALPHA)
            scale_factors = new_v_mags / np.where(v_mags == 0, 1e-10, v_mags)
            alpha_vels *= scale_factors[:, np.newaxis]
            a_vel[:] = alpha_vels.astype(np.float32)

        # The bulk pool holds nothing but types 0 and 1, and the compaction above
        # removed every -1 this step, so its live count IS the confined count.
        current_confined = int(bulk.n_live)
        inventory_history.append(current_confined)

        # --- ALPHA -> BULK ENERGY TRANSFER (energy conservation) ---
        # Energy drained from the alphas used to vanish -- removed from the fast
        # population and given to nothing, so the thermal plasma never felt the heating.
        # Deposit it by scaling bulk speeds, using the simulation-scale keV (no
        # macro_weight) so particles stay self-consistent.
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

        # --- T_core IS A TEMPERATURE, NOT A TOTAL ---
        # Assigning the plasma's TOTAL stored energy here made the "core temperature"
        # read ~50,000 keV, and every consumer (the Lawson triple product, the
        # quench baseline) was fed that. For a 3D Maxwellian
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
                    drain = np.sqrt(1.0 - loss_fraction)
                    b_vel[:] = (b_vel.astype(np.float64) * drain).astype(np.float32)

        # energy_history_keV / T_core_kinetic above are recorded BEFORE the drain, which
        # is correct: the radiation covers [t, t+dt] and so lands in the next step's
        # ledger, not retroactively in this one.

        prof.add("alpha heating + energy transfer", _t)

        _t = prof.mark()
        loss_rate = total_lost / t if t > 0 else 1e-5
        tau_E = min(current_confined / loss_rate if loss_rate > 0 else 0.5, 3.0)
        if lawson_first_measured_step is None and t > 0 and total_lost > 0:
            lawson_first_measured_step = step
        # The triple product's density is the confined macro-particle count times
        # MACRO_WEIGHT over the psi_edge volume. Units: m^-3 * keV * s, matching
        # cfg.lawson_target and the axis label.
        n_e_reactor = (current_confined * cfg.MACRO_WEIGHT) / cfg.PLASMA_VOLUME_M3
        lawson_triple_product_history.append(n_e_reactor * T_core * tau_E)

        temp_history.append(T_core)
        prof.add("Lawson diagnostics", _t)

        if (step + 1) % 500 == 0:
            max_E, max_phi = np.max(np.hypot(E_R_grid, E_Z_grid)), np.max(np.abs(phi_grid))
            print(f"  Step {step+1:04d}/{cfg.reactor_num_steps} | Confined: {current_confined:,} | Max |E|: {max_E:.3e} V/m | Max |Phi|: {max_phi:.2e} V")

    if profiling:
        loop_wall = time.perf_counter() - loop_wall_start
        prof.report(wall_total=loop_wall,
                    title="STAGE TIMING -- reactor loop (CPU / Numba)")
        push_prof.report(wall_total=prof.totals.get("deuteron Boris push"),
                         title="BREAKDOWN -- deuteron Boris push stage")
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

    # End-of-run ion density map: weighted CIC of every live ion, computed once.
    rho_grid = deposit_ion_charge_density(cfg, pos_np, type_np)

    # Tracked-particle metadata is returned as pid-keyed dicts, which removal cannot
    # corrupt -- row indices no longer survive a step now that losses are compacted out.
    # The loop accumulates into flat arrays; the dicts are rebuilt once, here, so the
    # return signature and package_reactor_results are unchanged.
    (history_tracks, tracked_type, tracked_lastpos,
     tracked_lastvel, tracked_lost) = tracks.to_dicts()
    return (pos_np, vel_np, type_np, history_tracks, tracked_type, tracked_lastpos, tracked_lastvel,
            tracked_lost,
            total_injected, total_lost,
            inventory_history, energy_history_keV,
            temp_history, trigger_time, T_core,
            lawson_triple_product_history, lawson_first_measured_step, rho_grid, phi_grid)
