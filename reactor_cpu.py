"""CPU / Numba reactor loop, split out of main.py (step 14).

Imports the step-14 leaf modules (profiling, kernels, particle_pool,
track_store) plus physics_engine; none of them import back into this one.
"""
import numpy as np
import time
import torch

from physics_engine import (
    update_force_balance_field,             # quasi-neutral E_r from thermal-ion force balance
    deposit_ion_charge_density,             # end-of-run weighted CIC density map
    vectorized_boris_push_numba_fallback,
    _vectorized_boris_push_metal_impl,      # eager push; shapes churn every step, so the static-compiled path would recompile constantly
    check_confinement_flux,                 # psi-surface confinement, replaces the circular boundary
    relative_weight_lut,                    # per-species weight / thermal-D weight
    compute_alpha_birth_rates,              # thermal D-T reactions/s per psi_N shell
    AlphaBirthSampler,                      # shell by rate, position by rejection on psi_N
    weight_lut,
    species_energy_and_count,               # real J and real ion count per species
    thermal_temperature_keV,                # T from thermal D + T only
    STORED_ENERGY_SPECIES,
    species_masses,
    inventory_line,                         # weighted real ions per species [macroparticles]
    loss_energy_by_species,                 # weighted energy carried out through the LCFS
    WindowedLossPower,
    energy_confinement_time,
    tearing_amplitude_over_B0,              # prescribed ripple / B0, reported at the end
    alpha_floor_lut_keV,                    # (3/2) T_i per psi_N shell, latest profiles
    alpha_thermalization_floor_keV,         # that floor at each alpha's flux surface
    drain_alpha_energy,
    weighted_kinetic_energy_kev,
    bulk_boost_factor,
)
from profiling import (
    StageProfiler,
    _profile_header,
    _report_peak_rss,
)
from kernels import (
    vectorized_gather_and_B_into,
    boris_push_substeps_regather,           # alpha and beam sub-stepping, field re-gathered per sub-step
    apply_vectorized_collisions,
)
from particle_pool import (
    _ParticlePool,
    _pool_capacity_bounds,
    _run_log,
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


def _push_beam(cfg, nb_pos, nb_vel, E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid, t):
    # Beam deuterons: at 1 MeV they cover ~1 cm per global step against a ~1.7 cm
    # Larmor radius, so they are sub-stepped exactly like the alphas -- the same
    # kernel, field re-gathered every sub-step, cfg.NBI_SUBSTEPS from the alphas'
    # gyro-phase criterion (config.py). q = e, m = m_D. Mutates in place.
    # Twin: reactor_gpu._push_beam_torch.
    boris_push_substeps_regather(
        nb_pos, nb_vel, cfg.e_charge, cfg.m_deuterium, cfg.reactor_dt, cfg.NBI_SUBSTEPS,
        E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid,
        cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ,
        cfg.B0, cfg.R0_major, t, cfg.b_perturb_initial, cfg.m_mode, cfg.n_mode, cfg.gamma_growth)


def _push_thermal_deuterium(cfg, b_pos, b_vel, B_b, E_b):
    # Thermal deuterons: one Boris step at the global dt with q = e, m = m_D
    # (Larmor radius ~3 mm at 25 keV against the 2 cm field grid). Mutates in
    # place. Twin: the deuteron push in reactor_gpu._run_reactor_loop_gpu.
    _boris_push_adaptive(b_pos, b_vel, cfg.e_charge, cfg.m_deuterium,
                         B_b, E_b, cfg.reactor_dt, cfg.HPC_DEVICE)


def _push_thermal_tritium(cfg, t_pos, t_vel, B_t, E_t):
    # Thermal tritons: one Boris step at the global dt with q = e, m = m_T. Mutates
    # t_pos / t_vel in place. Twin: reactor_gpu._push_thermal_tritium_torch.
    _boris_push_adaptive(t_pos, t_vel, cfg.e_charge, cfg.m_tritium,
                         B_t, E_t, cfg.reactor_dt, cfg.HPC_DEVICE)


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
    # pid -> track slot, -1 when the pid is not plotted. Replaces the
    # `dpid in history_tracks` dict membership test in the compaction below.
    pid_slot = np.full(pid_capacity, -1, dtype=np.int64)
    # pid -> which pool the row index above is relative to: 0 = thermal D (bulk),
    # 1 = alpha, 2 = tritium, 3 = beam D. With the pools split, pid_row alone is ambiguous --
    # row 7 means a different particle in each pool -- so the two arrays are only
    # meaningful together.
    #
    # WRITE-ONCE, at birth, and deliberately NOT updated by compaction. That is
    # sound rather than an oversight: species never migrates. type is written only
    # at birth (0/1/2/3) and at death (-1) -- the thermalization floor gates whether
    # an alpha keeps heating, it does not reclassify it -- so a particle can never
    # change pool. Only its row moves, and pid_row carries that.
    pid_pool = np.full(pid_capacity, -1, dtype=np.int8)
    # pid -> species type code, write-once at birth like pid_pool. The confinement
    # check overwrites a lost particle's type with -1, so this is what says which
    # species carried energy out through the last closed flux surface.
    pid_species = np.full(pid_capacity, -1, dtype=np.int8)
    pid_species[:n_init] = type_init

    # --- SPLIT POOLS ---
    # bulk holds thermal deuterium (type 0), nbi the beam deuterons (type 1), alpha
    # type 2, trit thermal tritium (type 3): one species per pool. Each kernel runs on
    # a contiguous [:n_live] view with no gather, no .copy() and no masked
    # write-back; the one boolean split below happens once, before the loop.
    #
    # ORDER IS LOAD-BEARING. Each pool is independently ascending by pid --
    # injection appends and compaction is an order-preserving forward gather --
    # so the profile binning and the energy sums see each species in ascending-pid
    # order, and the end-of-run rejoin below restores one ascending-pid order.
    rows_d = np.nonzero(type_init == 0)[0]
    rows_b = np.nonzero(type_init == 1)[0]          # beam ions present at load (tests only)
    rows_t = np.nonzero(type_init == 3)[0]
    if len(rows_d) + len(rows_b) + len(rows_t) != n_init:
        raise RuntimeError("initial particles must be type 0, 1 or 3")
    bulk_capacity, alpha_capacity, trit_capacity, nbi_capacity = _pool_capacity_bounds(
        cfg, len(rows_d), len(rows_t), len(rows_b))
    bulk = _ParticlePool(bulk_capacity, device=None)
    nbi = _ParticlePool(nbi_capacity, device=None)
    # The alpha pool carries a weight per alpha: each keeps the weight it was born with.
    alpha = _ParticlePool(alpha_capacity, device=None, with_weight=True)
    trit = _ParticlePool(trit_capacity, device=None)
    for _pool, _rows, _code, _which in ((bulk, rows_d, 0, 0), (nbi, rows_b, 1, 3), (trit, rows_t, 3, 2)):
        _pool.add(pos_init[_rows], vel_init[_rows], _code, _rows.astype(np.int64))
        pid_row[_rows] = np.arange(len(_rows), dtype=np.int64)
        pid_pool[_rows] = _which
    b_pos, b_vel, b_type, b_pid = bulk.views()
    nb_pos, nb_vel, nb_type, nb_pid = nbi.views()
    a_pos, a_vel, a_type, a_pid = alpha.views()
    a_wt = alpha.weight_view()
    t_pos, t_vel, t_type, t_pid = trit.views()

    # 500 thermal D + 500 thermal T + the 1000-each NBI and alpha caps below.
    tracks = _TrackStore(_MAX_TRACKED_SLOTS, device=None, host_dtype=np.float32)
    n_track_d = min(500, bulk.n_live)
    init_slots = tracks.add_slots(b_pid[:n_track_d], 0,
                                  b_pos[:n_track_d], b_vel[:n_track_d])
    pid_slot[b_pid[:n_track_d]] = init_slots
    n_track_t = min(500, trit.n_live)
    init_slots = tracks.add_slots(t_pid[:n_track_t], 3,
                                  t_pos[:n_track_t], t_vel[:n_track_t])
    pid_slot[t_pid[:n_track_t]] = init_slots
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
    # IndexError. The alpha and beam pools need no buffer: _push_alphas and
    # _push_beam gather their field inside the sub-step kernel.
    E_buf_b = np.empty((bulk.capacity, 3), dtype=np.float32)
    B_buf_b = np.empty((bulk.capacity, 3), dtype=np.float32)
    # Same contract for the tritium pool, sized from its own capacity.
    E_buf_t = np.empty((trit.capacity, 3), dtype=np.float32)
    B_buf_t = np.empty((trit.capacity, 3), dtype=np.float32)

    # Per-species weight relative to thermal D, indexed by type code. Energies and
    # counts below are weighted with it, so they read in thermal-D macroparticle
    # units (times cfg.WEIGHT[0] = physical). Equal weights make every entry 1.0.
    rel_w = relative_weight_lut(cfg)
    w_lut = weight_lut(cfg)
    stored_cols = np.array([code for code, _ in STORED_ENERGY_SPECIES], dtype=np.int64)

    total_injected = cfg.initial_thermal_count
    total_lost = 0
    # stored_energy_J: one row per step, real kinetic energy [J] per species in
    # STORED_ENERGY_SPECIES order (thermal D, thermal T, beam D, alpha).
    inventory_history, stored_energy_J = [], []
    # Source bookkeeping returned in source_log: beam macroparticles injected each
    # step and the weighted kinetic energy they carried in.
    nbi_per_step, nbi_energy_J = [], 0.0
    # Fusion-driven alpha source: shell rates and the birth weight are refreshed at
    # every field update; alpha_log records what was born and from which rates.
    alpha_sampler = AlphaBirthSampler(engine.eq)
    alpha_rates = np.zeros(alpha_sampler.n_bins)
    w_alpha = 0.0
    alpha_per_step, alpha_rate_per_step, alpha_birth_weight = [], [], 0.0
    alpha_birth_shells, alpha_birth_update, alpha_rate_updates = [], [], []
    temp_history, lawson_triple_product_history = [], []
    T_core = 0.0
    # --- LOSSES THROUGH THE LAST CLOSED FLUX SURFACE, PER SPECIES ---
    # Weighted kinetic energy [J], macroparticles and real ions lost, by type code.
    mass_lut = np.asarray(species_masses(cfg), dtype=np.float64)
    lost_energy_J = np.zeros(len(mass_lut))
    lost_macro = np.zeros(len(mass_lut))
    lost_real = np.zeros(len(mass_lut))
    # Ion energy confinement time from orbit losses: tau_E = W_thermal / P_loss, with
    # P_loss the thermal-ion loss power over a trailing TAU_E_WINDOW_STEPS window.
    loss_power = WindowedLossPower(cfg.TAU_E_WINDOW_STEPS, cfg.reactor_dt)
    tau_E_history, P_loss_history = [], []
    # First step at which a thermal ion has been lost: tau_E (and the triple product)
    # is NaN before it -- there is no loss power to divide by. None if no thermal
    # ion was ever lost.
    lawson_first_measured_step = None
    # Alpha thermalization floor, (3/2) T_i per psi_N shell from the latest
    # profiles; refreshed at every field update (step 0 first).
    alpha_floor_lut = None

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
        _profile_header(cfg, "CPU / Numba path", bulk.n_live + nbi.n_live + alpha.n_live + trit.n_live)
    loop_wall_start = time.perf_counter() if profiling else None

    for step in range(cfg.reactor_num_steps):
        t = step * cfg.reactor_dt

        # --- NBI INJECTION (constant power) ---
        # NBI_MACRO_PER_STEP beam macroparticles every step, fractional part realised
        # stochastically, each of weight WEIGHT[1] = P_NBI dt / (E_beam * rate), so
        # the weighted injected power is cfg.P_NBI_MW.
        _t = prof.mark()
        batch_size = int(cfg.NBI_MACRO_PER_STEP)
        nbi_frac = cfg.NBI_MACRO_PER_STEP - batch_size
        if nbi_frac > 0.0 and np.random.random() < nbi_frac:
            batch_size += 1
        nbi_per_step.append(batch_size)

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
            base_row = nbi.add(p_nbi, v_nbi, 1, new_pids)
            nb_pos, nb_vel, nb_type, nb_pid = nbi.views()
            # Injected power bookkeeping, from the stored (float32) rows.
            _vb = nb_vel[base_row:base_row + batch_size].astype(np.float64)
            nbi_energy_J += cfg.WEIGHT[1] * float(np.sum(0.5 * cfg.m_deuterium * np.sum(_vb * _vb, axis=1)))
            pid_row[new_pids] = np.arange(base_row, base_row + batch_size,
                                          dtype=np.int64)
            pid_pool[new_pids] = 3
            pid_species[new_pids] = 1
            next_pid += batch_size
            total_injected += batch_size

            # The old loop tracked the first (1000 - tracked_nbis) of the
            # batch and skipped the rest; take that same prefix.
            n_new = min(batch_size, 1000 - tracked_nbis)
            if n_new > 0:
                rows = np.arange(base_row, base_row + n_new, dtype=np.int64)
                slots = tracks.add_slots(new_pids[:n_new], 1,
                                         nb_pos[rows], nb_vel[rows])
                pid_slot[new_pids[:n_new]] = slots
                tracked_nbis += n_new

        prof.add("NBI injection", _t)


        # --- QUASI-NEUTRAL RADIAL ELECTRIC FIELD ---
        # E from thermal-ion (D + T) radial force balance (no rotation), rebuilt from
        # the current bulk-D and tritium pools at step 0 and every FIELD_UPDATE_EVERY
        # steps. The pools are all-live here (the compaction below removes every -1
        # each step). Twin: the same call in reactor_gpu on host copies of the pools.
        _t = prof.mark()
        if step % cfg.FIELD_UPDATE_EVERY == 0:
            phi_grid, E_R_grid, E_Z_grid, _pot = update_force_balance_field(
                cfg, engine.eq, np.concatenate((b_pos, t_pos)), np.concatenate((b_vel, t_vel)),
                np.concatenate((b_type, t_type)))
            # The same thermal profiles give the D-T reaction rate per shell, and the
            # birth weight that makes ALPHA_MACRO_PER_STEP macro-alphas per step carry
            # the computed total rate.
            alpha_rates = compute_alpha_birth_rates(_pot["profile"], cfg)
            # ALPHA_MACRO_PER_STEP = 0 switches the source off (no births, weight 0).
            w_alpha = (float(alpha_rates.sum()) * cfg.reactor_dt / cfg.ALPHA_MACRO_PER_STEP
                       if cfg.ALPHA_MACRO_PER_STEP > 0 else 0.0)
            cfg.WEIGHT[2] = w_alpha
            cfg.ALPHA_BIRTH_WEIGHT = w_alpha
            alpha_rate_updates.append(alpha_rates.copy())
            alpha_floor_lut = alpha_floor_lut_keV(_pot["profile"])
            if step == 0:
                print(f"[ALPHA] w_alpha = {w_alpha:.3e} alphas per macro-alpha "
                      f"(thermal D-T rate {alpha_rates.sum():.3e} /s, "
                      f"{cfg.ALPHA_MACRO_PER_STEP:g} macro-alphas/step)")
        prof.add("E_r force balance", _t)

        # --- FUSION-DRIVEN ALPHA BIRTHS ---
        # ALPHA_MACRO_PER_STEP per step (fractional part stochastic), each of weight
        # w_alpha, born in a shell drawn by rate_k at a rejection-sampled position,
        # isotropic at 3.5 MeV. No reactions (zero rate) -> no births.
        _t = prof.mark()
        alpha_batch = int(cfg.ALPHA_MACRO_PER_STEP)
        a_frac = cfg.ALPHA_MACRO_PER_STEP - alpha_batch
        if a_frac > 0.0 and np.random.random() < a_frac:
            alpha_batch += 1
        if w_alpha <= 0.0:
            alpha_batch = 0
        alpha_per_step.append(alpha_batch)
        alpha_rate_per_step.append(float(alpha_rates.sum()))
        if alpha_batch > 0:
            p_alpha, v_alpha, a_shells = alpha_sampler.sample(alpha_rates, alpha_batch, engine, cfg)
            alpha_birth_weight += alpha_batch * w_alpha
            alpha_birth_shells.extend(a_shells.tolist())
            alpha_birth_update.extend([len(alpha_rate_updates) - 1] * alpha_batch)

            _require_pid_capacity(next_pid, alpha_batch, pid_capacity, "CPU alpha injection")
            new_pids = np.arange(next_pid, next_pid + alpha_batch, dtype=np.int64)
            base_row = alpha.add(p_alpha, v_alpha, 2, new_pids, np.full(alpha_batch, w_alpha))
            a_pos, a_vel, a_type, a_pid = alpha.views()
            a_wt = alpha.weight_view()
            pid_row[new_pids] = np.arange(base_row, base_row + alpha_batch,
                                          dtype=np.int64)
            pid_pool[new_pids] = 1
            pid_species[new_pids] = 2
            next_pid += alpha_batch

            n_new = min(alpha_batch, 1000 - tracked_alphas)
            if n_new > 0:
                rows = np.arange(base_row, base_row + n_new, dtype=np.int64)
                slots = tracks.add_slots(new_pids[:n_new], 2,
                                         a_pos[rows], a_vel[rows])
                pid_slot[new_pids[:n_new]] = slots
                tracked_alphas += n_new
        prof.add("alpha births", _t)

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
        E_t = E_buf_t[:trit.n_live]
        B_t = B_buf_t[:trit.n_live]
        vectorized_gather_and_B_into(
            t_pos, E_t, B_t,
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
            _push_thermal_deuterium(cfg, b_pos, b_vel, B_b, E_b)
            push_prof.add("push call (+ B/E gather)", _ts)
            _ts = push_prof.mark()
            push_prof.add("masked write-back", _ts)
        prof.add("deuteron Boris push", _t)

        # Thermal tritium: same single global step, its own mass, its own pool.
        _t = prof.mark()
        if trit.n_live > 0:
            _push_thermal_tritium(cfg, t_pos, t_vel, B_t, E_t)
        prof.add("tritium Boris push", _t)

        # Beam deuterons and alphas are sub-stepped, with the field re-gathered every
        # sub-step (_push_beam, _push_alphas). Thermal ions stay on the single global
        # step. Each pool IS its species, so the kernel gets the contiguous float32
        # views and mutates them in place.
        _t = prof.mark()
        if nbi.n_live > 0:
            _push_beam(cfg, nb_pos, nb_vel, E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid, t)
        prof.add("beam substep push", _t)

        _t = prof.mark()
        if alpha.n_live > 0:
            _push_alphas(cfg, a_pos, a_vel, E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid, t)
        prof.add("alpha substep push", _t)

        _t = prof.mark()
        # The numba kernel mutates vel_arr in place and returns the same array, so
        # this rebinding is a no-op and vel_np stays the pool view. Do not replace
        # it with anything that allocates -- see the torch twin in the GPU loop.
        # Isotropic re-randomisation at a fixed rate per species, cfg.NU_C: the NRL
        # ion collision rate at the loaded central density and temperature (deuteron
        # rate for thermal and beam D, triton rate for thermal T). Alphas do not
        # collide (the kernel's guard admits types 0, 1, 3 only). Mutates in place
        # and returns the same array, so each rebinding is a no-op and the names
        # stay the pool views.
        b_vel = apply_vectorized_collisions(b_vel, b_type, cfg.NU_C[0], cfg.reactor_dt)
        if nbi.n_live > 0:
            nb_vel = apply_vectorized_collisions(nb_vel, nb_type, cfg.NU_C[1], cfg.reactor_dt)
        t_vel = apply_vectorized_collisions(t_vel, t_type, cfg.NU_C[3], cfg.reactor_dt)
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
        lost_nb = check_confinement_flux(nb_pos, nb_type, *psi_args) if nbi.n_live > 0 else 0
        lost_a = check_confinement_flux(a_pos, a_type, *psi_args) if alpha.n_live > 0 else 0
        lost_t = check_confinement_flux(t_pos, t_type, *psi_args) if trit.n_live > 0 else 0
        newly_lost = lost_b + lost_nb + lost_a + lost_t
        prof.add("check_confinement_flux", _t)
        total_lost += newly_lost

        # --- WALL LOSS COMPACTION ---
        _t = prof.mark()
        # check_confinement_flux only FLAGS wall strikes (type = -1). Leaving them in the
        # arrays meant the pools only ever grew, every kernel paid for dead particles, and
        # the confined count could never fall. Trajectory history is pid-keyed, so it
        # survives the row-index shift and the diagnostics payload is unchanged.
        step_lost_J = 0.0                   # thermal D + T energy out this step
        if newly_lost > 0:
            # Each pool compacts independently; rows are renumbered within a pool,
            # which is exactly what pid_row means now that pid_pool says which
            # pool the index belongs to.
            for _pool, _pos, _vel, _type, _pid, _lost, _wt in (
                    (bulk, b_pos, b_vel, b_type, b_pid, lost_b, None),
                    (nbi, nb_pos, nb_vel, nb_type, nb_pid, lost_nb, None),
                    (alpha, a_pos, a_vel, a_type, a_pid, lost_a, a_wt),
                    (trit, t_pos, t_vel, t_type, t_pid, lost_t, None)):
                if _lost == 0:
                    continue
                alive = _type != -1
                dead_rows = np.nonzero(~alive)[0]
                if dead_rows.size > 0:
                    dead_pids = _pid[dead_rows]
                    # Energy carried out, at the crossing velocity, by species.
                    sp = pid_species[dead_pids]
                    e_sp, n_sp, w_sp = loss_energy_by_species(
                        _vel[dead_rows], sp, mass_lut,
                        w_lut[sp] if _wt is None else _wt[dead_rows])
                    lost_energy_J += e_sp
                    lost_macro += n_sp
                    lost_real += w_sp
                    step_lost_J += float(e_sp[0] + e_sp[3])
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
            nb_pos, nb_vel, nb_type, nb_pid = nbi.views()
            a_pos, a_vel, a_type, a_pid = alpha.views()
            a_wt = alpha.weight_view()
            t_pos, t_vel, t_type, t_pid = trit.views()
            # Rebuild the row map for whichever pools moved. pid_pool is NOT
            # touched: a particle cannot change pool, only rows move.
            if lost_b:
                pid_row[b_pid] = np.arange(bulk.n_live, dtype=np.int64)
            if lost_nb:
                pid_row[nb_pid] = np.arange(nbi.n_live, dtype=np.int64)
            if lost_a:
                pid_row[a_pid] = np.arange(alpha.n_live, dtype=np.int64)
            if lost_t:
                pid_row[t_pid] = np.arange(trit.n_live, dtype=np.int64)
        prof.add("wall-loss compaction", _t)

        # --- TRAJECTORY SAMPLING (pid-keyed, so removal cannot corrupt it) ---
        _t = prof.mark()
        # Fast ions (beam D, alphas) are sampled far more often than thermals: their
        # orbit is only ~2 cm across, so a 20-step cadence (~0.2-0.26 m of travel)
        # aliases it away.
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
                for _which, _pos, _vel in ((0, b_pos, b_vel), (1, a_pos, a_vel), (2, t_pos, t_vel),
                                           (3, nb_pos, nb_vel)):
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
        # alpha_deposited_kev is the weighted (real) energy given up, sum(w_a dE) / w_thermal.
        alpha_deposited_kev = 0.0
        if alpha.n_live > 0:
            alpha_vels = a_vel.astype(np.float64)
            # Per-alpha weights, relative to the thermal-D weight like the bulk's.
            # Thermalization floor: (3/2) T_i on each alpha's own flux surface.
            a_floor = alpha_thermalization_floor_keV(a_pos, engine.eq, alpha_floor_lut)
            alpha_deposited_kev = drain_alpha_energy(engine, cfg, alpha_vels, a_wt / w_lut[0],
                                                     cfg.reactor_dt, a_floor)
            a_vel[:] = alpha_vels.astype(np.float32)

        # The thermal-D, beam and tritium pools hold nothing but types 0, 1 and 3, and
        # the compaction above removed every -1 this step, so their live counts ARE
        # the confined ion count (alphas are not counted, as before).
        current_confined = int(bulk.n_live) + int(nbi.n_live) + int(trit.n_live)
        inventory_history.append(current_confined)

        # --- ALPHA -> BULK ENERGY TRANSFER (energy conservation) ---
        # Energy drained from the alphas used to vanish -- removed from the fast
        # population and given to nothing, so the thermal plasma never felt the heating.
        # Deposit it by scaling every thermal-D and thermal-T speed by one common
        # factor. The beam ions are not part of the thermal background and get none
        # of it (before Tier 2.5 they shared the bulk pool and were scaled too).
        # Both sides are weighted -- given up sum(w_a dE_a), received sum(w_b dE_b)
        # over both pools -- so real energy is conserved even when the species
        # weights differ.
        r_b = rel_w[b_type]
        r_t = rel_w[t_type]
        if alpha_deposited_kev > 0.0 and (bulk.n_live + trit.n_live) > 0:
            bulk_vels = b_vel.astype(np.float64)
            trit_vels = t_vel.astype(np.float64)
            bulk_energy_kev = (weighted_kinetic_energy_kev(bulk_vels, r_b, cfg.m_deuterium)
                               + weighted_kinetic_energy_kev(trit_vels, r_t, cfg.m_tritium))
            if bulk_energy_kev > 0.0:
                boost = bulk_boost_factor(alpha_deposited_kev, bulk_energy_kev)
                b_vel[:] = (bulk_vels * boost).astype(np.float32)
                t_vel[:] = (trit_vels * boost).astype(np.float32)

        # --- STORED ENERGY AND REAL ION COUNT PER SPECIES ---
        # Real kinetic energy [J] and real ion count per type code, one weighted
        # bincount per pool (alphas included, after this step's drag).
        E_spec, N_spec = species_energy_and_count(b_vel, b_type, cfg.m_deuterium, w_lut)
        E_nb_, N_nb_ = species_energy_and_count(nb_vel, nb_type, cfg.m_deuterium, w_lut)
        E_t_, N_t_ = species_energy_and_count(t_vel, t_type, cfg.m_tritium, w_lut)
        E_a_, N_a_ = species_energy_and_count(a_vel, a_type, cfg.MASS_ALPHA, w_lut, w_row=a_wt)
        E_spec = E_spec + E_nb_ + E_t_ + E_a_
        N_spec = N_spec + N_nb_ + N_t_ + N_a_
        stored_energy_J.append(E_spec[stored_cols])

        # --- T_core IS THE THERMAL-ION TEMPERATURE ---
        # (2/3) <E> per real thermal ion, from thermal D and thermal T only: the
        # beam ions and the alphas are not part of the thermal plasma and used to
        # pull this several keV above it. Measured, never prescribed: nothing in the
        # loop removes energy from the particles except losses through the last
        # closed flux surface (the scripted pellet quench is gone, Tier 2.5).
        T_core = thermal_temperature_keV(E_spec, N_spec)

        prof.add("alpha heating + energy transfer", _t)

        _t = prof.mark()
        # tau_E = W_thermal / P_loss: weighted thermal D + T kinetic energy over the
        # thermal-ion loss power through the LCFS (trailing window). Ion orbit losses
        # only -- no electrons, no transport -- and NaN until a thermal ion is lost.
        P_loss = loss_power.add(step_lost_J)
        if lawson_first_measured_step is None and step_lost_J > 0.0:
            lawson_first_measured_step = step
        tau_E = (energy_confinement_time(E_spec[0] + E_spec[3], P_loss)
                 if lawson_first_measured_step is not None else float("nan"))
        tau_E_history.append(tau_E)
        P_loss_history.append(P_loss)
        # The triple product's density is the weighted thermal-ion density: real
        # thermal D + T count over the psi_edge volume. Units: m^-3 * keV * s,
        # matching cfg.lawson_target and the axis label.
        n_e_reactor = (N_spec[0] + N_spec[3]) / cfg.PLASMA_VOLUME_M3
        lawson_triple_product_history.append(n_e_reactor * T_core * tau_E)

        temp_history.append(T_core)
        prof.add("Lawson diagnostics", _t)

        if (step + 1) % 500 == 0:
            max_E, max_phi = np.max(np.hypot(E_R_grid, E_Z_grid)), np.max(np.abs(phi_grid))
            n_macro = np.array([bulk.n_live, nbi.n_live, alpha.n_live, trit.n_live], dtype=np.float64)
            print(f"  Step {step+1:04d}/{cfg.reactor_num_steps} | ions {inventory_line(N_spec, n_macro)} "
                  f"| Max |E|: {max_E:.3e} V/m | Max |Phi|: {max_phi:.2e} V | w_alpha: {w_alpha:.3e}")

    # The prescribed tearing-mode ripple (kernels._field_at) acts on every particle
    # through the field gather; report how large it got, relative to B0, at the
    # last time level the loop pushed.
    tearing_amp_over_B0 = tearing_amplitude_over_B0(cfg, (cfg.reactor_num_steps - 1) * cfg.reactor_dt)
    print(f"[SYSTEM] Tearing-mode amplitude at end of run: {tearing_amp_over_B0:.3e} of B0 "
          f"({tearing_amp_over_B0 * cfg.B0:.3e} T)")

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
    pos_np = np.concatenate((b_pos, nb_pos, a_pos, t_pos), axis=0)
    vel_np = np.concatenate((b_vel, nb_vel, a_vel, t_vel), axis=0)
    type_np = np.concatenate((b_type, nb_type, a_type, t_type), axis=0)
    pid_all = np.concatenate((b_pid, nb_pid, a_pid, t_pid), axis=0)
    # Real ions per row: the species weight for D, beam D and T, each alpha's own.
    weight_np = np.concatenate((w_lut[b_type], w_lut[nb_type], a_wt, w_lut[t_type]))
    legacy_order = np.argsort(pid_all, kind="stable")
    pos_np = np.ascontiguousarray(pos_np[legacy_order])
    vel_np = np.ascontiguousarray(vel_np[legacy_order])
    type_np = np.ascontiguousarray(type_np[legacy_order])
    weight_np = np.ascontiguousarray(weight_np[legacy_order])

    # End-of-run ion density map: weighted CIC of every live ion, computed once.
    rho_grid = deposit_ion_charge_density(cfg, pos_np, type_np, weights=weight_np)

    # Tracked-particle metadata is returned as pid-keyed dicts, which removal cannot
    # corrupt -- row indices no longer survive a step now that losses are compacted out.
    # The loop accumulates into flat arrays; the dicts are rebuilt once, here, so the
    # return signature and package_reactor_results are unchanged.
    (history_tracks, tracked_type, tracked_lastpos,
     tracked_lastvel, tracked_lost) = tracks.to_dicts()
    return (pos_np, vel_np, type_np, history_tracks, tracked_type, tracked_lastpos, tracked_lastvel,
            tracked_lost,
            total_injected, total_lost,
            inventory_history, np.asarray(stored_energy_J, dtype=np.float64).reshape(-1, len(stored_cols)),
            temp_history, T_core,
            lawson_triple_product_history, lawson_first_measured_step, rho_grid, phi_grid,
            _run_log(nbi_per_step, nbi_energy_J, alpha_per_step, alpha_rate_per_step,
                     alpha_birth_weight, alpha_birth_shells, alpha_birth_update,
                     alpha_rate_updates, weight_np, tearing_amp_over_B0,
                     dict(tau_E=tau_E_history, P_loss_thermal_W=P_loss_history,
                          lost_energy_J=lost_energy_J, lost_macro=lost_macro,
                          lost_real=lost_real)))
