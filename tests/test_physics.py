#!/usr/bin/env python3
"""Physics tests against known answers (Tier 1).

Unlike tests/test_regression.py, which pins numbers bit-for-bit, these check
that a quantity matches what the physics says it must be: an adiabatic
invariant stays invariant, a moment of a known distribution comes back as the
parameter it was sampled with, and so on.

    venv/bin/python -m pytest tests/test_physics.py -q
    venv/bin/python tests/test_physics.py          # same tests, no pytest needed
"""
import os
import sys

os.environ.setdefault("FUSION_HEADLESS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import contextlib                                  # noqa: E402
import functools                                   # noqa: E402
import io                                          # noqa: E402

import numpy as np                                 # noqa: E402


# =======================================================
# SHARED FIXTURES
# =======================================================
@functools.lru_cache(maxsize=None)
def _production_fields():
    """Production equilibrium and resampled poloidal-field grids (built once)."""
    from config import SimulationConfiguration
    import initialization
    from physics_engine import compute_poloidal_field_grids

    with contextlib.redirect_stdout(io.StringIO()):
        cfg = SimulationConfiguration()
        cfg.initial_thermal_count = 100      # init only; no bulk is pushed
        state = np.random.get_state()
        np.random.seed(1)
        eq, engine, *_ = initialization.initialize_reactor(cfg)
        np.random.set_state(state)
        B_R, B_Z, _, _ = compute_poloidal_field_grids(
            eq, cfg.B0, R0=cfg.R0_major, a_minor=0.3, q_target=cfg.Q_SAFETY_TARGET,
            dst_R_min=cfg.R_min, dst_R_max=cfg.R_max, dst_Z_min=cfg.Z_min,
            dst_Z_max=cfg.Z_max, dst_nR=cfg.nR, dst_nZ=cfg.nZ)
    return cfg, eq, engine, B_R, B_Z


def _equilibrium_B(cfg, pos, B_R, B_Z):
    """The equilibrium field (no E, no tearing mode) at pos, float64."""
    from kernels import vectorized_gather_and_B
    zero = np.zeros((cfg.nR, cfg.nZ))
    _, B = vectorized_gather_and_B(
        np.ascontiguousarray(pos, dtype=np.float32), zero, zero, B_R, B_Z,
        cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ,
        cfg.B0, cfg.R0_major, 0.0, 0.0, cfg.m_mode, cfg.n_mode, 0.0)
    return B.astype(np.float64)


def _pitch_and_mu(cfg, pos, vel, B_R, B_Z, mass):
    B = _equilibrium_B(cfg, pos, B_R, B_Z)
    Bm = np.linalg.norm(B, axis=1)
    v = vel.astype(np.float64)
    vm = np.linalg.norm(v, axis=1)
    vpar = np.sum(v * B, axis=1) / Bm
    vperp2 = np.maximum(vm**2 - vpar**2, 0.0)
    return vpar / vm, mass * vperp2 / (2.0 * Bm)


def _isotropic_alphas(cfg, n, rng):
    """n 3.5 MeV alphas, isotropic, inside the plasma around R = 1.1 m."""
    v0 = np.sqrt(2.0 * cfg.ALPHA_ENERGY_JOULES / cfg.MASS_ALPHA)
    R = rng.normal(1.10, 0.03, n)
    Z = rng.normal(0.0, 0.03, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    pos = np.stack([R * np.cos(phi), R * np.sin(phi), Z], axis=1).astype(np.float32)
    u = rng.normal(size=(n, 3))
    vel = (v0 * u / np.linalg.norm(u, axis=1)[:, None]).astype(np.float32)
    return np.ascontiguousarray(pos), np.ascontiguousarray(vel)


# =======================================================
# PHASE 1 -- alpha pusher
# =======================================================
ALPHA_TEST_STEPS = 1500     # 1.5 us; the frozen-field kernel collapsed mu by >10x here
ALPHA_TEST_COUNT = 64


def _alpha_invariants_after_push(push_fn, steps=ALPHA_TEST_STEPS):
    """Push an isotropic alpha population and return (xi0, xi1, mu0, mu1, inside)."""
    cfg, eq, engine, B_R, B_Z = _production_fields()
    from physics_engine import interpolate_psi_array
    rng = np.random.default_rng(7)
    pos, vel = _isotropic_alphas(cfg, ALPHA_TEST_COUNT, rng)
    xi0, mu0 = _pitch_and_mu(cfg, pos, vel, B_R, B_Z, cfg.MASS_ALPHA)
    E_zero = np.zeros((cfg.nR, cfg.nZ))
    for step in range(steps):
        push_fn(cfg, pos, vel, E_zero, E_zero, B_R, B_Z, step * cfg.reactor_dt)
    xi1, mu1 = _pitch_and_mu(cfg, pos, vel, B_R, B_Z, cfg.MASS_ALPHA)
    R = np.hypot(pos[:, 0], pos[:, 1]).astype(np.float64)
    psi = interpolate_psi_array(R, pos[:, 2].astype(np.float64), eq.psi_grid,
                                eq.psi_R_min, eq.psi_R_max, eq.psi_Z_min, eq.psi_Z_max,
                                eq.psi_nR, eq.psi_nZ)
    inside = psi > eq.psi_edge
    return xi0, xi1, mu0, mu1, inside


def test_alpha_push_conserves_magnetic_moment():
    """mu = m v_perp^2 / 2|B| is an adiabatic invariant: the 3.5 MeV alpha's
    Larmor radius (2.2 cm) is small against the field's scale length, so mu
    oscillates with gyro-phase (a few percent) but must not drift.

    Exercises reactor_cpu._push_alphas, the function the CPU loop calls.
    The frozen-field sub-step kernel it replaced drove the median mu ratio
    to ~0.01 and every pitch to xi = -1 within ~2 us.
    """
    import reactor_cpu
    xi0, xi1, mu0, mu1, inside = _alpha_invariants_after_push(reactor_cpu._push_alphas)
    assert inside.sum() >= 0.8 * len(inside), f"only {inside.sum()} alphas still confined"
    ratio = mu1[inside] / mu0[inside]
    med = float(np.median(ratio))
    assert 0.9 < med < 1.1, f"median mu(end)/mu(start) = {med:.4f}"
    frac_collapsed = float(np.mean(xi1[inside] < -0.95))
    born_collapsed = float(np.mean(xi0 < -0.95))
    assert frac_collapsed < born_collapsed + 0.1, (
        f"{frac_collapsed:.2f} of alphas at xi < -0.95 after the push (born: {born_collapsed:.2f})")


def test_alpha_push_conserves_magnetic_moment_gpu_twin():
    """Same invariant through reactor_gpu._push_alphas_torch, the GPU loop's
    alpha push, on cfg.HPC_DEVICE. Shorter run: each sub-step is a separate
    eager torch gather + push."""
    import torch
    import reactor_gpu
    cfg, eq, engine, B_R, B_Z = _production_fields()
    dev = cfg.HPC_DEVICE
    rng = np.random.default_rng(7)
    pos, vel = _isotropic_alphas(cfg, ALPHA_TEST_COUNT, rng)
    xi0, mu0 = _pitch_and_mu(cfg, pos, vel, B_R, B_Z, cfg.MASS_ALPHA)
    f32 = dict(device=dev, dtype=torch.float32)
    pos_t, vel_t = torch.tensor(pos, **f32), torch.tensor(vel, **f32)
    E0 = torch.zeros((cfg.nR, cfg.nZ), **f32)
    BR_t, BZ_t = torch.tensor(B_R, **f32), torch.tensor(B_Z, **f32)
    for step in range(600):
        reactor_gpu._push_alphas_torch(cfg, pos_t, vel_t, E0, E0, BR_t, BZ_t, step * cfg.reactor_dt)
    xi1, mu1 = _pitch_and_mu(cfg, pos_t.cpu().numpy(), vel_t.cpu().numpy(), B_R, B_Z, cfg.MASS_ALPHA)
    med = float(np.median(mu1 / mu0))
    assert 0.9 < med < 1.1, f"median mu(end)/mu(start) = {med:.4f}"


def test_alpha_energy_drain_preserves_direction():
    """compute_alpha_heating_power changes speed only: the loop rescales the
    float64 velocity by a scalar and stores float32, so the direction must
    survive to float32 precision."""
    cfg, eq, engine, B_R, B_Z = _production_fields()
    rng = np.random.default_rng(3)
    _, vel = _isotropic_alphas(cfg, 1000, rng)
    vel = vel * rng.uniform(0.05, 1.0, (1000, 1)).astype(np.float32)   # spread of energies
    v64 = vel.astype(np.float64)
    vm = np.linalg.norm(v64, axis=1)
    from constants import KEV_J
    ek = (0.5 * cfg.MASS_ALPHA * vm**2) / KEV_J
    # Same lines as the loop, with a large dt so the drain is visible (and the
    # 15 keV floor this test was written against).
    new_ek, _ = engine.compute_alpha_heating_power(ek, 1e-2, cfg, floor_keV=15.0)
    new_vm = np.sqrt(2.0 * (new_ek * KEV_J) / cfg.MASS_ALPHA)
    out = (v64 * (new_vm / vm)[:, None]).astype(np.float32).astype(np.float64)
    cosang = np.sum(out * v64, axis=1) / (np.linalg.norm(out, axis=1) * vm)
    assert np.all(new_ek <= ek)
    assert np.max(1.0 - cosang) < 1e-6, f"direction changed: max 1-cos = {np.max(1.0 - cosang):.2e}"


# =======================================================
# PHASE 2 -- flux-surface profiles
# =======================================================
def _species_masses(cfg):
    from physics_engine import species_masses
    return species_masses(cfg)          # (thermal D, beam D, alpha, thermal T)


def _positions_in_psiN(eq, n, rng, lo=0.0, hi=1.0, margin=0.0):
    """n Cartesian positions with lo <= psi_N < hi, sampled uniformly in (R, Z)."""
    from physics_engine import compute_psi_normalised
    R_out, Z_out = [], []
    got = 0
    while got < n:
        R = rng.uniform(eq.psi_R_min + margin, eq.psi_R_max - margin, 4 * n)
        Z = rng.uniform(eq.psi_Z_min + margin, eq.psi_Z_max - margin, 4 * n)
        pn = compute_psi_normalised(R, Z, eq)
        keep = (pn >= lo) & (pn < hi)
        R_out.append(R[keep]); Z_out.append(Z[keep]); got += int(keep.sum())
    R = np.concatenate(R_out)[:n]; Z = np.concatenate(Z_out)[:n]
    phi = rng.uniform(0, 2 * np.pi, n)
    return np.stack([R * np.cos(phi), R * np.sin(phi), Z], axis=1)


def test_flux_shell_volumes_sum_to_plasma_volume():
    """The psi_N shells tile the plasma, so their volumes must add up to the
    volume of the revolved psi_edge contour (computed independently, by Pappus
    on the ray-marched boundary polygon)."""
    from physics_engine import compute_flux_shell_volumes
    cfg, eq, *_ = _production_fields()
    V = compute_flux_shell_volumes(eq)
    assert np.all(V > 0.0), "an empty psi_N shell"
    rel = abs(V.sum() / eq.plasma_volume_m3 - 1.0)
    assert rel < 0.01, f"sum V_k = {V.sum():.4f} vs plasma volume {eq.plasma_volume_m3:.4f} ({rel:.2%})"


def test_flux_profile_density_integrates_to_particle_count():
    """sum_k n_k V_k must recover weight x (particles inside psi_N <= 1) per
    species, exactly up to float rounding; particles outside are excluded."""
    from physics_engine import compute_flux_profiles, compute_psi_normalised
    cfg, eq, *_ = _production_fields()
    rng = np.random.default_rng(11)
    pos = _positions_in_psiN(eq, 30000, rng, 0.0, 1.6)   # some outside the LCFS
    vel = rng.normal(0.0, 3e5, pos.shape)
    types = rng.choice([0, 1, 2, -1], size=len(pos), p=[0.6, 0.25, 0.1, 0.05])
    inside = compute_psi_normalised(np.hypot(pos[:, 0], pos[:, 1]), pos[:, 2], eq) <= 1.0
    assert 0 < inside.sum() < len(pos)
    for w in (1.0, 7.3e15):
        prof = compute_flux_profiles(pos, vel, types, eq, _species_masses(cfg), weight=w)
        for s in range(3):
            expect = w * np.sum(inside & (types == s))
            got = float(np.sum(prof["n"][s] * prof["shell_volume"]))
            assert abs(got - expect) <= 1e-12 * expect, f"species {s}, w={w}: {got} vs {expect}"


def test_flux_profile_temperature_of_maxwellian():
    """A 1 keV Maxwellian comes back as 1 keV within 2%, with and without a
    rigid toroidal rotation: the flow is a mean velocity, not heat."""
    from physics_engine import compute_flux_profiles
    cfg, eq, engine, *_ = _production_fields()
    rng = np.random.default_rng(5)
    n = 50000
    pos = _positions_in_psiN(eq, n, rng, 0.40, 0.44)       # one shell: bin 10
    state = np.random.get_state()
    np.random.seed(5)
    vel = engine.initialize_velocities(n, T_keV=1.0)
    np.random.set_state(state)
    masses = (engine.m, engine.m, cfg.MASS_ALPHA)
    types = np.zeros(n, dtype=np.int64)
    prof = compute_flux_profiles(pos, vel, types, eq, masses, weight=1.0)
    assert prof["count"][0, 10] == n
    T0 = prof["T_keV"][0, 10]
    assert abs(T0 - 1.0) < 0.02, f"T = {T0:.4f} keV"

    v_rot = 1.0e5                                          # rigid u_phi
    R = np.hypot(pos[:, 0], pos[:, 1])
    vel_rot = vel.copy()
    vel_rot[:, 0] += -v_rot * pos[:, 1] / R
    vel_rot[:, 1] += v_rot * pos[:, 0] / R
    prof_rot = compute_flux_profiles(pos, vel_rot, types, eq, masses, weight=1.0)
    assert abs(prof_rot["u"][0, 10, 1] - v_rot) < 0.01 * v_rot
    assert abs(prof_rot["T_keV"][0, 10] / T0 - 1.0) < 1e-6, "rotation leaked into T"


# =======================================================
# PHASE 3 -- physical weight, quasi-neutral field
# =======================================================
def test_weighted_cic_deposits_total_charge():
    """sum(rho * cell_volume) over the grid equals the deposited charge,
    N_in_domain * w * e (alphas at CHARGE_ALPHA), to 1e-10 relative."""
    from physics_engine import deposit_ion_charge_density
    cfg, eq, *_ = _production_fields()
    rng = np.random.default_rng(21)
    pos = _positions_in_psiN(eq, 20000, rng, 0.0, 1.0)
    types = rng.choice([0, 1, 2, -1], size=len(pos), p=[0.8, 0.1, 0.05, 0.05])
    w = 7.04e15
    saved = cfg.WEIGHT
    cfg.WEIGHT = {s: w for s in saved}
    try:
        rho = deposit_ion_charge_density(cfg, pos, types)
    finally:
        cfg.WEIGHT = saved
    dR = (cfg.R_max - cfg.R_min) / (cfg.nR - 1)
    dZ = (cfg.Z_max - cfg.Z_min) / (cfg.nZ - 1)
    R_nodes = cfg.R_min + np.arange(cfg.nR) * dR
    cell_volume = 2.0 * np.pi * R_nodes[:, None] * dR * dZ
    total = float(np.sum(rho * cell_volume))
    # Every LCFS position lies well inside the grid, so all live ions deposit.
    expect = w * (cfg.e_charge * np.sum((types == 0) | (types == 1))
                  + cfg.CHARGE_ALPHA * np.sum(types == 2))
    assert abs(total / expect - 1.0) < 1e-10, f"{total:.12e} vs {expect:.12e}"


def _synthetic_profile(n0, p_of_x, n_bins=25, count=10**6):
    """A compute_flux_profiles-shaped dict with exact n and p on the bin centres."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    x = 0.5 * (edges[:-1] + edges[1:])
    z = np.zeros((3, n_bins))
    prof = {"psiN_centers": x, "count": np.zeros((3, n_bins), dtype=np.int64),
            "n": z.copy(), "p": z.copy()}
    prof["count"][0] = count
    prof["n"][0] = n0
    prof["p"][0] = p_of_x(x)
    return prof


def test_force_balance_potential_exact_profile():
    """p = p0 (1 - psi_N), n = n0 gives dPhi/dpsi_N = p0 / (e n0), so
    Phi(x) = (p0 / (e n0)) (x - x_end) with Phi = 0 at the support edge."""
    from physics_engine import compute_force_balance_potential, e_charge
    n0, T0_J = 1.0e20, 1.0e3 * e_charge
    p0 = n0 * T0_J
    pot = compute_force_balance_potential(_synthetic_profile(n0, lambda x: p0 * (1.0 - x)))
    x, Phi = pot["x"], pot["Phi"]
    inside = x <= pot["x_end"]
    analytic = (p0 / (e_charge * n0)) * (x - pot["x_end"])
    err = np.max(np.abs(Phi[inside] - analytic[inside])) / abs(analytic[0])
    assert err < 0.02, f"max |Phi - analytic| = {err:.2%} of |Phi(axis)|"
    assert np.all(Phi[~inside] == 0.0)


def test_force_balance_potential_sampled_particles():
    """The same analytic Phi, measured end to end from sampled particles:
    uniform n, local Maxwellian at T = T0 (1 - psi_N), so p = n0 T0 (1 - psi_N)."""
    from physics_engine import (compute_flux_profiles, compute_force_balance_potential,
                                compute_psi_normalised, e_charge)
    cfg, eq, engine, *_ = _production_fields()
    rng = np.random.default_rng(13)
    # 600k ions: at 200k the max error over eight seeds reached 1.99%, too close
    # to the 2% bound for a statistical test.
    n = 600000
    # Uniform 3D density: accept a uniform (R, Z) sample with probability R / R_max.
    pos = _positions_in_psiN(eq, 3 * n, rng, 0.0, 1.0)
    R = np.hypot(pos[:, 0], pos[:, 1])
    pos = pos[rng.uniform(size=len(R)) < R / eq.psi_R_max][:n]
    assert len(pos) == n
    psiN = compute_psi_normalised(np.hypot(pos[:, 0], pos[:, 1]), pos[:, 2], eq)
    T0 = 1.0
    sigma = np.sqrt(T0 * (1.0 - psiN) * 1e3 * e_charge / cfg.m_deuterium)
    vel = rng.normal(size=(n, 3)) * sigma[:, None]
    prof = compute_flux_profiles(pos, vel, np.zeros(n, dtype=np.int64), eq,
                                 _species_masses(cfg), weight=7.0e15)
    pot = compute_force_balance_potential(prof)
    x, Phi = pot["x"], pot["Phi"]
    inside = x <= pot["x_end"]
    analytic = (T0 * 1e3) * (x - pot["x_end"])         # p0/(e n0) = T0/e, in volts
    err = np.max(np.abs(Phi[inside] - analytic[inside])) / abs(analytic[0])
    assert err < 0.02, f"max |Phi - analytic| = {err:.2%} of |Phi(axis)| = {analytic[0]:.0f} V"


def test_force_balance_field_zero_outside_lcfs_and_inward():
    """E vanishes at every grid node outside psi_N = 1, and for a pressure that
    falls outward it points down the pressure gradient's opposite: inward."""
    from physics_engine import compute_force_balance_field, compute_grad_psiN_grids, e_charge
    cfg, eq, *_ = _production_fields()
    n0 = 1.0e20
    prof = _synthetic_profile(n0, lambda x: n0 * 1e3 * e_charge * (1.0 - x) ** 2)
    phi, E_R, E_Z, pot = compute_force_balance_field(
        prof, eq, cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ)
    psiN, gR, gZ = compute_grad_psiN_grids(eq, cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ)
    outside = psiN > 1.0
    assert outside.sum() > 0 and (~outside).sum() > 0
    assert np.all(E_R[outside] == 0.0) and np.all(E_Z[outside] == 0.0)
    assert np.all(phi[outside] == 0.0)
    Emag = np.hypot(E_R, E_Z)
    on = Emag > 0.0
    assert on.sum() > 0.5 * (~outside).sum()
    assert np.all((E_R * gR + E_Z * gZ)[on] < 0.0), "E not inward where p falls outward"


def test_initial_loading_profiles_and_weight():
    """initialize_reactor loads n = n0 (1 - psi_N), T = T0 (1 - psi_N) for thermal D
    and thermal T each, with <n_D + n_T> = N_AVG_TARGET and
    WEIGHT[0] = WEIGHT[3] = N_AVG_TARGET * V / N."""
    from config import SimulationConfiguration
    import initialization
    from physics_engine import compute_flux_profiles
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = SimulationConfiguration()
        state = np.random.get_state()
        np.random.seed(2)
        eq, engine, p, v, t, *_ = initialization.initialize_reactor(cfg)
        np.random.set_state(state)
    N = cfg.initial_thermal_count
    assert cfg.WEIGHT[0] == cfg.N_AVG_TARGET * cfg.PLASMA_VOLUME_M3 / N
    assert cfg.WEIGHT[3] == cfg.WEIGHT[0]
    pos, vel, typ = p.cpu().numpy(), v.cpu().numpy(), t.cpu().numpy()
    prof = compute_flux_profiles(pos, vel, typ, eq, _species_masses(cfg),
                                 weight=[cfg.WEIGHT[s] for s in range(4)])
    V = prof["shell_volume"]
    n_avg = np.sum((prof["n"][0] + prof["n"][3]) * V) / V.sum()
    assert abs(n_avg / cfg.N_AVG_TARGET - 1.0) < 1e-3, f"<n> = {n_avg:.4e}"
    x = prof["psiN_centers"]
    core = x < 0.8                                       # >= ~500 ions per shell per species
    for s in (0, 3):
        # Fit n = n0 (1 - x) and T = T0 (1 - x) by least squares over the core shells.
        n0 = np.sum(prof["n"][s, core] * (1 - x[core])) / np.sum((1 - x[core]) ** 2)
        T0 = np.sum(prof["T_keV"][s, core] * (1 - x[core])) / np.sum((1 - x[core]) ** 2)
        assert abs(T0 / cfg.T_thermal_keV - 1.0) < 0.02, f"species {s}: fitted T0 = {T0:.4f} keV"
        resid_n = np.abs(prof["n"][s, core] / (n0 * (1 - x[core])) - 1.0)
        resid_T = np.abs(prof["T_keV"][s, core] / (T0 * (1 - x[core])) - 1.0)
        # Half the ions per species, so the per-shell counting noise is sqrt(2)
        # larger than when one species held them all (bound 0.1 -> 0.15).
        assert resid_n.max() < 0.15 and resid_T.max() < 0.15, (s, resid_n.max(), resid_T.max())


# =======================================================
# PHASE 4 -- thermal fusion power from the profiles
# =======================================================
def _bosch_hale(cfg, T_keV):
    from physics_engine import compute_bosch_hale_reactivity
    return compute_bosch_hale_reactivity(T_keV, cfg.BH_C1, cfg.BH_C2, cfg.BH_C3,
                                         cfg.BH_C4, cfg.BH_C5, cfg.BH_C6, cfg.BH_C7)


def test_bosch_hale_reactivity_at_10_keV():
    """D-T <sigma v>(10 keV) = 1.136e-22 m^3/s (Bosch & Hale 1992, Table VIII)."""
    cfg, *_ = _production_fields()
    sv = _bosch_hale(cfg, 10.0)
    assert abs(sv / 1.136e-22 - 1.0) < 0.01, f"<sigma v>(10 keV) = {sv:.4e}"


def _profile_for_power(eq, n_of_x, T_keV_of_x, nT_of_x=None, TT_keV_of_x=None):
    """Synthetic profile with thermal D in species 0 and thermal T in species 3.
    With only n_of_x / T_keV_of_x given, n_of_x is the TOTAL thermal density,
    split n_D = n_T = n/2 at one temperature (the old 50/50 layout); otherwise
    n_of_x, T_keV_of_x are D's and nT_of_x, TT_keV_of_x are T's."""
    from physics_engine import compute_flux_shell_volumes
    V = compute_flux_shell_volumes(eq)
    x = 0.5 * (np.linspace(0, 1, len(V) + 1)[:-1] + np.linspace(0, 1, len(V) + 1)[1:])
    z = np.zeros((4, len(V)))
    prof = {"psiN_centers": x, "shell_volume": V, "n": z.copy(), "T_keV": np.full((4, len(V)), np.nan)}
    if nT_of_x is None:
        prof["n"][0] = prof["n"][3] = 0.5 * n_of_x(x)
        prof["T_keV"][0] = prof["T_keV"][3] = T_keV_of_x(x)
    else:
        prof["n"][0], prof["T_keV"][0] = n_of_x(x), T_keV_of_x(x)
        prof["n"][3], prof["T_keV"][3] = nT_of_x(x), TT_keV_of_x(x)
    return prof


def test_thermal_fusion_power_is_the_shell_sum():
    """P_total = sum_k P_k V_k, and each P_k = (n_k/2)^2 <sigma v>(T_k) E_fus
    for a total thermal density n_k split evenly between D and T."""
    from physics_engine import compute_thermal_fusion_power
    cfg, eq, *_ = _production_fields()
    prof = _profile_for_power(eq, lambda x: 2e20 * (1 - x), lambda x: 12.0 * (1 - x))
    P, total = compute_thermal_fusion_power(prof, cfg)
    assert abs(total - np.sum(P * prof["shell_volume"])) <= 1e-12 * total
    for k in (0, 7, 15):
        n, T = prof["n"][0, k] + prof["n"][3, k], prof["T_keV"][0, k]
        expect = (n / 2) ** 2 * _bosch_hale(cfg, T) * cfg.E_FUSION_JOULES
        assert abs(P[k] / expect - 1.0) < 1e-12
    assert P[-1] < P[0]


def test_thermal_fusion_power_uniform_profile():
    """Uniform n and T: P_total = n^2/4 <sigma v>(T) E_fus V_plasma."""
    from physics_engine import compute_thermal_fusion_power
    cfg, eq, *_ = _production_fields()
    n, T = 1.0e20, 10.0
    prof = _profile_for_power(eq, lambda x: np.full_like(x, n), lambda x: np.full_like(x, T))
    _, total = compute_thermal_fusion_power(prof, cfg)
    expect = n**2 / 4 * 1.136e-22 * cfg.E_FUSION_JOULES * eq.plasma_volume_m3
    assert abs(total / expect - 1.0) < 0.01, f"{total:.4e} W vs {expect:.4e} W"


# =======================================================
# TIER 2 PHASE 1 -- per-species weights
# =======================================================
def test_flux_profile_density_with_unequal_species_weights():
    """With a different weight per species, sum_k n_k V_k must equal
    N_species * w_species for every species (particles inside psi_N <= 1),
    and the weighted CIC deposit must carry sum_s N_s w_s q_s."""
    from physics_engine import (compute_flux_profiles, compute_psi_normalised,
                                deposit_ion_charge_density)
    cfg, eq, *_ = _production_fields()
    rng = np.random.default_rng(31)
    pos = _positions_in_psiN(eq, 30000, rng, 0.0, 1.0)
    vel = rng.normal(0.0, 3e5, pos.shape)
    types = rng.choice([0, 1, 2, 3, -1], size=len(pos), p=[0.35, 0.2, 0.1, 0.3, 0.05])
    inside = compute_psi_normalised(np.hypot(pos[:, 0], pos[:, 1]), pos[:, 2], eq) <= 1.0
    w = (7.04e15, 2.1e12, 3.3e3, 5.5e15)
    prof = compute_flux_profiles(pos, vel, types, eq, _species_masses(cfg), weight=w)
    for s in range(4):
        expect = w[s] * np.sum(inside & (types == s))
        got = float(np.sum(prof["n"][s] * prof["shell_volume"]))
        assert abs(got - expect) <= 1e-12 * expect, f"species {s}: {got} vs {expect}"

    saved = cfg.WEIGHT
    cfg.WEIGHT = {s: w[s] for s in range(4)}
    try:
        rho = deposit_ion_charge_density(cfg, pos, types)
    finally:
        cfg.WEIGHT = saved
    dR = (cfg.R_max - cfg.R_min) / (cfg.nR - 1)
    dZ = (cfg.Z_max - cfg.Z_min) / (cfg.nZ - 1)
    R_nodes = cfg.R_min + np.arange(cfg.nR) * dR
    total = float(np.sum(rho * 2.0 * np.pi * R_nodes[:, None] * dR * dZ))
    expect = (cfg.e_charge * (w[0] * np.sum(types == 0) + w[1] * np.sum(types == 1)
                              + w[3] * np.sum(types == 3))
              + cfg.CHARGE_ALPHA * w[2] * np.sum(types == 2))
    assert abs(total / expect - 1.0) < 1e-10, f"{total:.12e} vs {expect:.12e}"


def _alpha_to_bulk_transfer(cfg, engine, a_vel, a_w, groups, dt):
    """Run the reactor loops' alpha -> bulk transfer (physics_engine helpers, in
    the loop's order) on float64 copies. groups: [(vel, w, mass)] bulk pools.
    Weights are absolute; the loop's relative weights are w / w_ref."""
    from physics_engine import drain_alpha_energy, weighted_kinetic_energy_kev, bulk_boost_factor
    w_ref = 7.04e15
    a = a_vel.astype(np.float64)
    dep = drain_alpha_energy(engine, cfg, a, a_w / w_ref, dt, 15.0)   # every alpha far above it
    out = [g[0].astype(np.float64) for g in groups]
    E_bulk = sum(weighted_kinetic_energy_kev(v, w / w_ref, m) for v, (_, w, m) in zip(out, groups))
    boost = bulk_boost_factor(dep, E_bulk)
    return a, [v * boost for v in out], dep * w_ref


def _real_energy(vel, w, m):
    return float(np.sum(w * 0.5 * m * np.sum(vel.astype(np.float64) ** 2, axis=1)))


def test_alpha_to_bulk_transfer_conserves_weighted_energy():
    """Real energy given up by the alphas, sum(w_a dE_a), equals the real energy
    received by the bulk, sum(w_b dE_b), to 1e-12 relative, with alpha, thermal
    and beam weights that all differ (and differ per alpha)."""
    cfg, eq, engine, *_ = _production_fields()
    rng = np.random.default_rng(41)
    _, a_vel = _isotropic_alphas(cfg, 500, rng)
    a_vel = a_vel * rng.uniform(0.3, 1.0, (500, 1))         # spread of energies
    # Per-alpha weights, large enough that the deposit is a resolvable fraction of
    # the bulk energy (at production alpha weights it is ~1e-12 of it, below the
    # float64 resolution of the bulk total, and the comparison would test rounding).
    a_w = rng.uniform(0.5e14, 2.0e14, 500)
    n_b = 4000
    b_vel = rng.normal(0.0, 3e5, (n_b, 3))
    b_type = rng.choice([0, 1], size=n_b, p=[0.8, 0.2])
    b_w = np.where(b_type == 0, 7.04e15, 2.1e12)
    dt = 1e-2                                                  # a drain well above rounding
    a1, (b1,), _ = _alpha_to_bulk_transfer(cfg, engine, a_vel, a_w,
                                           [(b_vel, b_w, cfg.m_deuterium)], dt)
    given = _real_energy(a_vel, a_w, cfg.MASS_ALPHA) - _real_energy(a1, a_w, cfg.MASS_ALPHA)
    received = _real_energy(b1, b_w, cfg.m_deuterium) - _real_energy(b_vel, b_w, cfg.m_deuterium)
    assert given > 0.0
    assert abs(received / given - 1.0) < 1e-12, f"given {given:.15e} J, received {received:.15e} J"


# =======================================================
# TIER 2 PHASE 2 -- thermal tritium
# =======================================================
def _gyro_phase_rate(push, cfg, mass, B_mag, dt, steps=400):
    """Angular rate of v_perp for one ion in a uniform B = B_mag z_hat, E = 0,
    pushed by `push(cfg, pos, vel, B, E)` with cfg.reactor_dt = dt."""
    pos = np.array([[1.1, 0.0, 0.0]], dtype=np.float32)
    vel = np.array([[2.0e5, 0.0, 1.0e5]], dtype=np.float32)
    B = np.array([[0.0, 0.0, B_mag]], dtype=np.float32)
    E = np.zeros((1, 3), dtype=np.float32)
    saved = cfg.reactor_dt
    cfg.reactor_dt = dt
    try:
        ang = [np.arctan2(vel[0, 1], vel[0, 0])]
        for _ in range(steps):
            push(cfg, pos, vel, B, E)
            ang.append(np.arctan2(vel[0, 1], vel[0, 0]))
    finally:
        cfg.reactor_dt = saved
    phase = np.unwrap(np.asarray(ang, dtype=np.float64))
    return abs(phase[-1] - phase[0]) / (steps * dt)


def test_tritium_gyrofrequency():
    """reactor_cpu._push_thermal_tritium (the loop's triton push) gyrates at
    Omega = e B / m_T: at the production dt it matches the Boris rotation
    2 atan(Omega dt / 2) / dt to float32 precision, and as dt -> 0 it converges
    on e B / m_T itself. m_T is the CODATA 2022 triton mass (constants.py)."""
    import reactor_cpu
    cfg, *_ = _production_fields()
    from constants import M_TRITON
    assert cfg.m_tritium == M_TRITON
    B_mag = 12.0
    omega = cfg.e_charge * B_mag / cfg.m_tritium
    push = reactor_cpu._push_thermal_tritium
    w_prod = _gyro_phase_rate(push, cfg, cfg.m_tritium, B_mag, cfg.reactor_dt)
    boris = 2.0 * np.arctan(0.5 * omega * cfg.reactor_dt) / cfg.reactor_dt
    assert abs(w_prod / boris - 1.0) < 1e-5, f"{w_prod:.6e} vs Boris {boris:.6e} rad/s"
    w_fine = _gyro_phase_rate(push, cfg, cfg.m_tritium, B_mag, 1e-11, steps=2000)
    assert abs(w_fine / omega - 1.0) < 1e-4, f"{w_fine:.6e} vs eB/m_T {omega:.6e} rad/s"


def test_tritium_gyrofrequency_gpu_twin():
    """Same Boris rotation rate through reactor_gpu._push_thermal_tritium_torch,
    the GPU loop's triton push, on cfg.HPC_DEVICE (float32 throughout)."""
    import torch
    import reactor_gpu
    cfg, *_ = _production_fields()
    f32 = dict(device=cfg.HPC_DEVICE, dtype=torch.float32)

    def push(cfg_, pos, vel, B, E):
        p_t, v_t = torch.tensor(pos, **f32), torch.tensor(vel, **f32)
        reactor_gpu._push_thermal_tritium_torch(cfg_, p_t, v_t, torch.tensor(B, **f32),
                                                torch.tensor(E, **f32))
        pos[:] = p_t.cpu().numpy()
        vel[:] = v_t.cpu().numpy()

    omega = cfg.e_charge * 12.0 / cfg.m_tritium
    w = _gyro_phase_rate(push, cfg, cfg.m_tritium, 12.0, cfg.reactor_dt, steps=200)
    boris = 2.0 * np.arctan(0.5 * omega * cfg.reactor_dt) / cfg.reactor_dt
    assert abs(w / boris - 1.0) < 1e-4, f"{w:.6e} vs Boris {boris:.6e} rad/s"


def test_tritium_push_conserves_magnetic_moment():
    """mu = m_T v_perp^2 / 2|B| is conserved by the loop's triton push in the
    production equilibrium (field gathered every step, as the loop does): 1 keV
    tritons have a ~0.5 mm Larmor radius, far below the field scale length."""
    import reactor_cpu
    from kernels import vectorized_gather_and_B_into
    cfg, eq, engine, B_R, B_Z = _production_fields()
    rng = np.random.default_rng(17)
    n = 256
    pos = _positions_in_psiN(eq, n, rng, 0.05, 0.6).astype(np.float32)
    vel = engine.initialize_velocities_profile(np.full(n, 1.0), mass=cfg.m_tritium).astype(np.float32)
    pos, vel = np.ascontiguousarray(pos), np.ascontiguousarray(vel)
    _, mu0 = _pitch_and_mu(cfg, pos, vel, B_R, B_Z, cfg.m_tritium)
    zero = np.zeros((cfg.nR, cfg.nZ))
    E = np.empty((n, 3), dtype=np.float32)
    B = np.empty((n, 3), dtype=np.float32)
    for step in range(1500):
        vectorized_gather_and_B_into(pos, E, B, zero, zero, B_R, B_Z,
                                     cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ,
                                     cfg.B0, cfg.R0_major, 0.0, 0.0, cfg.m_mode, cfg.n_mode, 0.0)
        reactor_cpu._push_thermal_tritium(cfg, pos, vel, B, E)
    _, mu1 = _pitch_and_mu(cfg, pos, vel, B_R, B_Z, cfg.m_tritium)
    ratio = mu1 / mu0
    ok = mu0 > 1e-3 * np.median(mu0)          # drop near-parallel ions (mu ~ 0)
    med = float(np.median(ratio[ok]))
    assert 0.99 < med < 1.01, f"median mu(end)/mu(start) = {med:.5f}"
    assert np.percentile(np.abs(ratio[ok] - 1.0), 90) < 0.05


def test_loaded_thermal_counts_are_half_d_half_t():
    """The production load is exactly 25,000 thermal D (type 0) and 25,000
    thermal T (type 3), nothing else."""
    from config import SimulationConfiguration
    import initialization
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = SimulationConfiguration()
        state = np.random.get_state()
        np.random.seed(4)
        eq, engine, p, v, t, *_ = initialization.initialize_reactor(cfg)
        np.random.set_state(state)
    typ = t.cpu().numpy()
    assert cfg.initial_thermal_count == 50000
    assert int(np.sum(typ == 0)) == 25000 and int(np.sum(typ == 3)) == 25000
    assert len(typ) == 50000


def test_fusion_power_with_equal_d_t_equals_old_formula():
    """With n_D = n_T = n/2 and T_D = T_T = T on every shell, the D-T power
    equals the old 50/50 formula (n/2)^2 <sigma v>(T) E_fus shell by shell;
    with n_D != n_T and T_D != T_T it is n_D n_T <sigma v>(T_i) E_fus at the
    density-weighted T_i = (n_D T_D + n_T T_T) / (n_D + n_T)."""
    from physics_engine import compute_thermal_fusion_power, compute_volumetric_fusion_power
    cfg, eq, *_ = _production_fields()
    n_tot = lambda x: 1.5e20 * (1 - x) ** 0.7
    T_of = lambda x: 9.0 * (1 - x) + 0.5
    prof = _profile_for_power(eq, n_tot, T_of)
    P, total = compute_thermal_fusion_power(prof, cfg)
    x = prof["psiN_centers"]
    half = 0.5 * n_tot(x)                                   # the old formula, verbatim
    P_old = compute_volumetric_fusion_power(half[None, :], half[None, :], T_of(x)[None, :], cfg)[0]
    assert np.all(np.abs(P - P_old) <= 1e-12 * np.abs(P_old))
    assert abs(total / float(np.sum(P_old * prof["shell_volume"])) - 1.0) < 1e-12

    nD = lambda x: 1.0e20 * (1 - x)
    nT = lambda x: 0.6e20 * (1 - x) ** 2 + 1e18
    TD = lambda x: 8.0 * (1 - x) + 0.5
    TT = lambda x: 4.0 * (1 - x) + 0.5
    prof = _profile_for_power(eq, nD, TD, nT, TT)
    P, _ = compute_thermal_fusion_power(prof, cfg)
    for k in (0, 6, 13, 20):
        Ti = (nD(x[k]) * TD(x[k]) + nT(x[k]) * TT(x[k])) / (nD(x[k]) + nT(x[k]))
        expect = nD(x[k]) * nT(x[k]) * _bosch_hale(cfg, Ti) * cfg.E_FUSION_JOULES
        assert abs(P[k] / expect - 1.0) < 1e-12, (k, P[k], expect)


# =======================================================
# TIER 2 PHASE 3 -- thermal diagnostics separate from beam ions
# =======================================================
@functools.lru_cache(maxsize=None)
def _one_step_run(n_beam):
    """A production-size load (50,000 thermals, seed 8) plus n_beam 50 keV beam
    ions appended as type 1, run through reactor_cpu._run_reactor_loop_cpu for
    one step. Returns (cfg, loop output, initial types)."""
    import torch
    from config import SimulationConfiguration
    import initialization
    import reactor_cpu
    from physics_engine import compute_poloidal_field_grids
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = SimulationConfiguration()
        cfg.reactor_num_steps = 1
        state = np.random.get_state()
        np.random.seed(8)
        eq, engine, p, v, t, rho, phi, ER, EZ = initialization.initialize_reactor(cfg)
        if n_beam:
            psi_b = (eq.psi_grid, eq.psi_core, eq.psi_R_min, eq.psi_R_max,
                     eq.psi_Z_min, eq.psi_Z_max, eq.psi_nR, eq.psi_nZ)
            pb, vb = engine.inject_neutral_beam_cartesian(n_beam, E_keV=cfg.nbi_energy_keV, psi_bounds=psi_b)
            p = torch.cat((p, torch.tensor(pb, dtype=torch.float32, device=p.device)))
            v = torch.cat((v, torch.tensor(vb, dtype=torch.float32, device=v.device)))
            t = torch.cat((t, torch.ones(n_beam, dtype=torch.int32, device=t.device)))
        np.random.set_state(state)
        B_R, B_Z, _, _ = compute_poloidal_field_grids(
            eq, cfg.B0, R0=cfg.R0_major, a_minor=0.3, q_target=cfg.Q_SAFETY_TARGET,
            dst_R_min=cfg.R_min, dst_R_max=cfg.R_max, dst_Z_min=cfg.Z_min,
            dst_Z_max=cfg.Z_max, dst_nR=cfg.nR, dst_nZ=cfg.nZ)
        types0 = t.cpu().numpy().copy()
        out = reactor_cpu._run_reactor_loop_cpu(cfg, engine, p, v, t, rho, phi, ER, EZ, B_R, B_Z)
    return cfg, eq, out, types0


def _loaded_density_weighted_T(eq, T0, n_mesh=800):
    """<T>_n = int n T dV / int n dV for the loaded n, T proportional to (1 - psi_N),
    on a fine (R, Z) mesh with dV = 2 pi R dR dZ."""
    from physics_engine import compute_psi_normalised
    Rc = eq.psi_R_min + (np.arange(n_mesh) + 0.5) * (eq.psi_R_max - eq.psi_R_min) / n_mesh
    Zc = eq.psi_Z_min + (np.arange(n_mesh) + 0.5) * (eq.psi_Z_max - eq.psi_Z_min) / n_mesh
    RR, ZZ = np.meshgrid(Rc, Zc, indexing="ij")
    x = compute_psi_normalised(RR, ZZ, eq)
    w = np.where(x <= 1.0, (1.0 - x) * RR, 0.0)
    return T0 * float(np.sum(w * (1.0 - np.minimum(x, 1.0))) / np.sum(w))


def test_T_core_of_fresh_plasma_is_loaded_density_weighted_temperature():
    """The loop's T_core at step 0 (temp_history[0]) equals the density-weighted
    temperature of the loaded thermal profiles, <T>_n = T0 int (1-x)^2 dV /
    int (1-x) dV, within 4 sigma of counting noise (sigma from the spread of
    (2/3) E_i over the 50,000 loaded thermals)."""
    from physics_engine import e_charge
    cfg, eq, out, types0 = _one_step_run(0)
    pos, vel, typ = out[0], out[1], out[2]
    T_core0 = out[12][0]
    expect = _loaded_density_weighted_T(eq, cfg.T_thermal_keV)
    th = (typ == 0) | (typ == 3)
    m = np.where(typ[th] == 3, cfg.m_tritium, cfg.m_deuterium)
    t_i = (2.0 / 3.0) * 0.5 * m * np.sum(vel[th].astype(np.float64) ** 2, axis=1) / (e_charge * 1e3)
    sigma = float(np.std(t_i)) / np.sqrt(th.sum())
    assert abs(T_core0 - expect) < 4.0 * sigma, \
        f"T_core {T_core0:.5f} keV vs loaded <T>_n {expect:.5f} keV (sigma {sigma:.5f})"


def test_beam_ions_do_not_change_T_core():
    """Adding 5,000 beam ions (cfg.nbi_energy_keV each, far above thermal) to the same
    load leaves T_core unchanged, and the per-species stored energy columns are
    the weighted kinetic energies of each species. Collisions run in the loop
    (prange thread chunking moves with the beam rows and re-randomised velocities
    are rebuilt in float32), so "unchanged" is 1e-9, not bit equality."""
    from physics_engine import STORED_ENERGY_SPECIES
    cfg0, _, out0, _ = _one_step_run(0)
    cfg1, _, out1, _ = _one_step_run(5000)
    T0, T1 = out0[12][0], out1[12][0]
    assert abs(T1 / T0 - 1.0) < 1e-9, f"T_core {T0:.12f} -> {T1:.12f} keV with beam ions"

    stored = out1[11]                                # (steps, 4), joules
    vel, typ = out1[1].astype(np.float64), out1[2]
    masses = {0: cfg1.m_deuterium, 1: cfg1.m_deuterium, 2: cfg1.MASS_ALPHA, 3: cfg1.m_tritium}
    for col, (code, label) in enumerate(STORED_ENERGY_SPECIES):
        sel = typ == code
        expect = cfg1.WEIGHT[code] * float(np.sum(0.5 * masses[code] * np.sum(vel[sel] ** 2, axis=1)))
        got = float(stored[0, col])
        if expect == 0.0:
            assert got == 0.0, (label, got)
        else:
            assert abs(got / expect - 1.0) < 1e-12, (label, got, expect)
    # The beam column is populated, with the energy the 5,000 beam ions were
    # injected with: n_beam * w_beam * E_beam to 1% (one 1 ns step; the electric
    # field does a negligible amount of work on them). (This replaced a ratio test,
    # stored[beam] > 1e-3 stored[thermal D], tuned to 50 keV beams in a 1 keV plasma.)
    E_beam_J = cfg1.nbi_energy_keV * 1e3 * cfg1.e_charge
    assert abs(stored[0, 2] / (5000 * cfg1.WEIGHT[1] * E_beam_J) - 1.0) < 1e-2, stored[0, 2]


# =======================================================
# TIER 2 PHASE 4 -- neutral beam calibrated by power
# =======================================================
@functools.lru_cache(maxsize=None)
def _source_run(n_steps, nbi_macro_per_step, n_thermal=400, seed=9, alpha_macro_per_step=1.0):
    """Small load run through reactor_cpu._run_reactor_loop_cpu for n_steps at the
    given NBI_MACRO_PER_STEP (set before initialize_reactor, which derives the
    beam weight from it) and ALPHA_MACRO_PER_STEP. Returns (cfg, loop output, the
    initial (pos, vel, type) arrays, eq)."""
    from config import SimulationConfiguration
    import initialization
    import reactor_cpu
    from physics_engine import compute_poloidal_field_grids
    with contextlib.redirect_stdout(io.StringIO()):
        cfg = SimulationConfiguration()
        cfg.initial_thermal_count = n_thermal
        cfg.reactor_num_steps = n_steps
        cfg.NBI_MACRO_PER_STEP = nbi_macro_per_step
        cfg.ALPHA_MACRO_PER_STEP = alpha_macro_per_step
        state = np.random.get_state()
        np.random.seed(seed)
        eq, engine, p, v, t, rho, phi, ER, EZ = initialization.initialize_reactor(cfg)
        init = (p.cpu().numpy().copy(), v.cpu().numpy().copy(), t.cpu().numpy().copy())
        B_R, B_Z, _, _ = compute_poloidal_field_grids(
            eq, cfg.B0, R0=cfg.R0_major, a_minor=0.3, q_target=cfg.Q_SAFETY_TARGET,
            dst_R_min=cfg.R_min, dst_R_max=cfg.R_max, dst_Z_min=cfg.Z_min,
            dst_Z_max=cfg.Z_max, dst_nR=cfg.nR, dst_nZ=cfg.nZ)
        out = reactor_cpu._run_reactor_loop_cpu(cfg, engine, p, v, t, rho, phi, ER, EZ, B_R, B_Z)
        np.random.set_state(state)
    return cfg, out, init, eq


def test_nbi_injected_power_equals_P_NBI():
    """The weighted kinetic energy of every injected beam macroparticle, summed in
    the loop from the stored rows, divided by the elapsed time, equals P_NBI
    within 1% over a 2,000-step run; w_beam = P_NBI dt / (E_beam * rate)."""
    cfg, out, _, _ = _source_run(2000, 2.0)
    log = out[-1]
    E_beam_J = cfg.nbi_energy_keV * 1e3 * cfg.e_charge
    w_expect = cfg.P_NBI_MW * 1e6 * cfg.reactor_dt / (E_beam_J * cfg.NBI_MACRO_PER_STEP)
    assert abs(cfg.WEIGHT[1] / w_expect - 1.0) < 1e-12
    elapsed = cfg.reactor_num_steps * cfg.reactor_dt
    P_inj = log["nbi_energy_J"] / elapsed
    assert abs(P_inj / (cfg.P_NBI_MW * 1e6) - 1.0) < 0.01, f"P_inj = {P_inj / 1e6:.4f} MW"
    assert int(log["nbi_per_step"].sum()) == 2 * cfg.reactor_num_steps


def test_nbi_mean_macroparticles_per_step_fractional_rate():
    """At a fractional rate (1.37 per step) the realised mean per step matches it
    within 4 sigma of Bernoulli counting noise, and the injected power matches
    P_NBI within the same relative noise."""
    rate = 1.37
    cfg, out, _, _ = _source_run(4000, rate)
    log = out[-1]
    k = log["nbi_per_step"]
    assert len(k) == cfg.reactor_num_steps and set(np.unique(k)) <= {1, 2}
    f = rate - int(rate)
    sigma = np.sqrt(f * (1 - f) / len(k))
    assert abs(k.mean() - rate) < 4 * sigma, f"mean {k.mean():.4f} vs {rate} (sigma {sigma:.4f})"
    P_inj = log["nbi_energy_J"] / (cfg.reactor_num_steps * cfg.reactor_dt)
    assert abs(P_inj / (cfg.P_NBI_MW * 1e6) - 1.0) < 4 * sigma / rate + 1e-3, f"P_inj = {P_inj / 1e6:.4f} MW"


# =======================================================
# TIER 2 PHASE 5 -- fusion-driven alpha source
# =======================================================
def test_alpha_birth_rate_equals_fusion_reaction_rate():
    """sum(w_alpha) over every macro-alpha born, per second, equals the thermal
    D-T reaction rate the loop computed (time average of the rate in force each
    step): exactly at ALPHA_MACRO_PER_STEP = 1, within 4 sigma of counting noise
    at a fractional 0.6. The step-0 rate is P_fusion / E_fusion of the loaded
    thermal profiles, and sum_k rate_k reproduces compute_thermal_fusion_power."""
    from physics_engine import (compute_flux_profiles, compute_thermal_fusion_power,
                                species_masses, weight_lut)
    for rate, n_steps in ((1.0, 1000), (0.6, 3000)):
        cfg, out, init, eq = _source_run(n_steps, 2.0, n_thermal=2000, seed=12,
                                         alpha_macro_per_step=rate)
        log = out[-1]
        elapsed = n_steps * cfg.reactor_dt
        mean_rate = float(np.mean(log["alpha_rate_per_step"]))
        assert mean_rate > 0.0
        born_rate = log["alpha_birth_weight"] / elapsed
        if rate == 1.0:
            assert abs(born_rate / mean_rate - 1.0) < 1e-12, (born_rate, mean_rate)
            assert np.all(log["alpha_per_step"] == 1)
            # Step-0 rate from the loaded thermal D and T, independently of the loop.
            pos0, vel0, typ0 = init
            th = (typ0 == 0) | (typ0 == 3)
            prof0 = compute_flux_profiles(pos0[th], vel0[th], typ0[th], eq, species_masses(cfg),
                                          weight=weight_lut(cfg))
            _, P0 = compute_thermal_fusion_power(prof0, cfg)
            r0 = log["alpha_rate_updates"][0].sum()
            assert abs(r0 / (P0 / cfg.E_FUSION_JOULES) - 1.0) < 1e-12, (r0, P0 / cfg.E_FUSION_JOULES)
        else:
            f = rate - int(rate)
            sigma = np.sqrt(f * (1 - f) / n_steps) / rate
            assert abs(born_rate / mean_rate - 1.0) < 4 * sigma, (born_rate, mean_rate, sigma)


def test_alpha_birth_radial_distribution_matches_rates():
    """Births land in psi_N shells in proportion to rate_k: chi-square p > 0.01
    for 20,000 births from AlphaBirthSampler on a peaked rate profile, and for
    the births of a loop run against the rates in force when each was born.
    Every birth lies inside its drawn shell, and within shells the births are
    uniform in volume (KS test of psi_N against the volume CDF, p > 0.01)."""
    from scipy import stats
    from physics_engine import (AlphaBirthSampler, compute_psi_normalised,
                                compute_flux_shell_volumes)
    cfg, eq, engine, *_ = _production_fields()
    sampler = AlphaBirthSampler(eq)
    nb = sampler.n_bins
    x_c = (np.arange(nb) + 0.5) / nb
    V = compute_flux_shell_volumes(eq)
    rates = V * (1 - x_c) ** 2 * np.exp(-((1 - x_c) * 2.0) ** -1)   # peaked, ~0 at the edge
    state = np.random.get_state()
    np.random.seed(23)
    n = 20000
    pos, vel, shells = sampler.sample(rates, n, engine, cfg)
    np.random.set_state(state)
    R = np.hypot(pos[:, 0], pos[:, 1]).astype(np.float64)
    x = compute_psi_normalised(R, pos[:, 2].astype(np.float64), eq)
    assert np.all((x >= shells / nb - 1e-6) & (x <= (shells + 1) / nb + 1e-6)), "birth outside its shell"
    vmag = np.linalg.norm(vel.astype(np.float64), axis=1)
    v0 = np.sqrt(2 * cfg.ALPHA_ENERGY_JOULES / cfg.MASS_ALPHA)
    assert np.max(np.abs(vmag / v0 - 1.0)) < 1e-6

    def chi2_p(counts, expected):
        keep = expected >= 5
        obs = np.append(counts[keep], counts[~keep].sum())
        exp = np.append(expected[keep], expected[~keep].sum())
        if exp[-1] < 5:
            obs, exp = obs[:-1], exp[:-1]
        exp = exp * obs.sum() / exp.sum()
        return stats.chisquare(obs, exp).pvalue

    p = chi2_p(np.bincount(shells, minlength=nb).astype(float), n * rates / rates.sum())
    assert p > 0.01, f"sampler chi-square p = {p:.4f}"

    # Within shells: volume CDF on 40 sub-bins per shell, from a finer mesh.
    sub = 40
    Vf = compute_flux_shell_volumes(eq, n_bins=nb * sub, n_mesh=1200)
    pf = np.repeat(rates / V, sub) * Vf
    cdf_nodes = np.concatenate([[0.0], np.cumsum(pf) / pf.sum()])
    grid = np.linspace(0.0, 1.0, nb * sub + 1)
    ks = stats.kstest(np.clip(x, 0, 1), lambda q: np.interp(q, grid, cdf_nodes))
    assert ks.pvalue > 0.01, f"within-shell KS p = {ks.pvalue:.4f} (D = {ks.statistic:.4f})"

    # The loop's own births against the rates in force when each was born.
    cfg2, out, _, _ = _source_run(3000, 2.0, n_thermal=2000, seed=12, alpha_macro_per_step=1.0)
    log = out[-1]
    upd = log["alpha_rate_updates"]
    probs = upd / upd.sum(axis=1, keepdims=True)
    expected = probs[log["alpha_birth_update"]].sum(axis=0)
    counts = np.bincount(log["alpha_birth_shells"], minlength=nb).astype(float)
    p_loop = chi2_p(counts, expected)
    assert p_loop > 0.01, f"loop births chi-square p = {p_loop:.4f}"


def test_per_alpha_weights_follow_compaction_and_conserve_energy():
    """Each macro-alpha keeps its birth weight: weights written by
    _ParticlePool.add stay attached to their particles through wall-loss
    compaction, and the alpha -> bulk transfer driven by those per-alpha weights
    conserves weighted energy to 1e-12."""
    from particle_pool import _ParticlePool
    cfg, eq, engine, *_ = _production_fields()
    rng = np.random.default_rng(29)
    pool = _ParticlePool(2000, device=None, with_weight=True)
    _, a_vel = _isotropic_alphas(cfg, 1200, rng)
    a_pos = np.zeros_like(a_vel)
    # Births at three epochs with three different birth weights.
    w_epochs = (0.8e14, 1.3e14, 2.1e14)
    for e, w in enumerate(w_epochs):
        sl = slice(400 * e, 400 * (e + 1))
        pool.add(a_pos[sl], a_vel[sl], 2, np.arange(400 * e, 400 * (e + 1)), np.full(400, w))
    alive = np.sort(rng.choice(1200, 700, replace=False))
    pool.compact(alive)
    _, vel_v, _, pid_v = pool.views()
    w_v = pool.weight_view()
    assert np.array_equal(pid_v, alive)
    assert np.array_equal(w_v, np.asarray(w_epochs)[alive // 400])
    assert np.array_equal(vel_v, a_vel[alive])

    n_b = 4000
    b_vel = rng.normal(0.0, 3e5, (n_b, 3))
    b_w = np.where(rng.random(n_b) < 0.8, 7.04e15, 2.06e12)
    a1, (b1,), _ = _alpha_to_bulk_transfer(cfg, engine, vel_v, w_v.copy(),
                                           [(b_vel, b_w, cfg.m_deuterium)], 1e-2)
    given = _real_energy(vel_v, w_v, cfg.MASS_ALPHA) - _real_energy(a1, w_v, cfg.MASS_ALPHA)
    received = _real_energy(b1, b_w, cfg.m_deuterium) - _real_energy(b_vel, b_w, cfg.m_deuterium)
    assert given > 0.0 and abs(received / given - 1.0) < 1e-12, (given, received)


# =======================================================
# TIER 2.5 PHASE 1 -- no hidden energy sink
# =======================================================
ENERGY_TEST_STEPS = 1000


def _closed_run(loop_fn, n_thermal=4000, steps=ENERGY_TEST_STEPS, seed=31, psiN_max=0.8,
                edge_scale=None):
    """A closed system run through a reactor loop: E = 0 (the force-balance update
    is wrapped to return zero grids), both sources off (NBI_MACRO_PER_STEP =
    ALPHA_MACRO_PER_STEP = 0), and only thermals with psi_N < psiN_max loaded, so
    none reaches the last closed flux surface in `steps` steps. Returns
    (cfg, loop output, initial real kinetic energy [J], per-particle real
    energies [J] at load)."""
    import torch
    from config import SimulationConfiguration
    import initialization
    import physics_engine
    import reactor_cpu
    import reactor_gpu
    from physics_engine import compute_poloidal_field_grids, compute_psi_normalised

    real_update = physics_engine.update_force_balance_field

    def zero_field(cfg_, eq_, pos, vel, types):
        phi, E_R, E_Z, pot = real_update(cfg_, eq_, pos, vel, types)
        return np.zeros_like(phi), np.zeros_like(E_R), np.zeros_like(E_Z), pot

    with contextlib.redirect_stdout(io.StringIO()):
        cfg = SimulationConfiguration()
        cfg.initial_thermal_count = n_thermal
        cfg.reactor_num_steps = steps
        cfg.NBI_MACRO_PER_STEP = 0.0
        cfg.ALPHA_MACRO_PER_STEP = 0.0
        state = np.random.get_state()
        np.random.seed(seed)
        eq, engine, p, v, t, rho, phi, ER, EZ = initialization.initialize_reactor(cfg)
        np.random.set_state(state)
        pn = p.cpu().numpy()
        keep = compute_psi_normalised(np.hypot(pn[:, 0], pn[:, 1]), pn[:, 2], eq) < psiN_max
        if edge_scale is not None:
            # Pull the loss boundary inward (CLAUDE.md's forced wall-loss recipe):
            # psi_edge -> edge_scale * psi_core, so the loop has losses to account for.
            eq.psi_edge = eq.psi_core * edge_scale
        idx = torch.from_numpy(np.nonzero(keep)[0]).to(p.device)
        p, v, t = p[idx].contiguous(), v[idx].contiguous(), t[idx].contiguous()
        cfg.initial_thermal_count = int(keep.sum())
        B_R, B_Z, _, _ = compute_poloidal_field_grids(
            eq, cfg.B0, R0=cfg.R0_major, a_minor=0.3, q_target=cfg.Q_SAFETY_TARGET,
            dst_R_min=cfg.R_min, dst_R_max=cfg.R_max, dst_Z_min=cfg.Z_min,
            dst_Z_max=cfg.Z_max, dst_nR=cfg.nR, dst_nZ=cfg.nZ)
        vel0 = v.cpu().numpy().astype(np.float64)
        typ0 = t.cpu().numpy()
        mass = np.where(typ0 == 3, cfg.m_tritium, cfg.m_deuterium)
        w = np.array([cfg.WEIGHT[int(s)] for s in typ0])
        E_p = w * 0.5 * mass * np.sum(vel0 ** 2, axis=1)
        saved = (reactor_cpu.update_force_balance_field, reactor_gpu.update_force_balance_field)
        reactor_cpu.update_force_balance_field = zero_field
        reactor_gpu.update_force_balance_field = zero_field
        try:
            out = loop_fn(cfg, engine, p, v, t, rho, phi, ER, EZ, B_R, B_Z)
        finally:
            reactor_cpu.update_force_balance_field, reactor_gpu.update_force_balance_field = saved
    return cfg, out, float(E_p.sum()), E_p


def test_closed_system_conserves_weighted_kinetic_energy():
    """With E = 0, both sources off and no wall losses, nothing may change the
    total weighted kinetic energy except float32 rounding: the Boris rotation is
    norm-preserving (computed in float64 in the numba push), the collisions
    re-randomise direction at fixed speed, and the alpha drag has no alphas to act
    on. A hidden sink (the scripted pellet drain removed ~6e-5 of the thermal
    energy per step) would show up here.

    The bound. Velocities are stored as float32 after every push (and after a
    collision), each component rounded to nearest, so per store and per component
    |dv_i| <= 2^-24 |v_i| and |d(v^2)| <= 2^-23 v^2 (to first order).
      * Hard (worst case, every rounding the same sign, two stores per particle
        per step to cover collisions): |dW| / W <= 2 * steps * 2^-23 = 2.4e-4 at
        1,000 steps.
      * Statistical: each rounding error is uniform on [-ulp/2, ulp/2], so
        var(dE_p) <= (4/3) 2^-48 E_p^2 per store, and
        sigma_W <= 2^-24 sqrt((4/3) * steps * sum_p E_p^2); require |dW| < 6 sigma_W
        (~2e-7 of W here). A sink of 1e-9 per step would fail it.
    """
    import reactor_cpu
    cfg, out, W0, E_p = _closed_run(reactor_cpu._run_reactor_loop_cpu)
    total_injected, total_lost = out[8], out[9]
    assert total_lost == 0, f"{total_lost} particles lost: not a closed system"
    assert total_injected == cfg.initial_thermal_count
    assert np.all(out[-1]["nbi_per_step"] == 0) and np.all(out[-1]["alpha_per_step"] == 0)
    W1 = float(out[11][-1].sum())                       # after the last of 1,000 steps
    rel = abs(W1 - W0) / W0
    hard = 2 * ENERGY_TEST_STEPS * 2.0 ** -23
    sigma = 2.0 ** -24 * np.sqrt(4.0 / 3.0 * ENERGY_TEST_STEPS * np.sum(E_p ** 2)) / W0
    assert rel < hard, f"|dW|/W = {rel:.3e} exceeds the worst-case float32 bound {hard:.3e}"
    assert rel < 6.0 * sigma, f"|dW|/W = {rel:.3e} exceeds 6 sigma = {6 * sigma:.3e} of float32 rounding"


def test_closed_system_conserves_weighted_kinetic_energy_gpu_twin():
    """The same closed system through reactor_gpu._run_reactor_loop_gpu. MPS has
    no float64, so the whole Boris push is float32 arithmetic, not just the
    store: the rotation v' = v- + v- x t, v+ = v- + v' x s rounds ~16 times per
    component chain (measured drift ~4e-7, ~9x the store-only sigma). Bound: 16
    roundings of v^2 per particle per step, worst case, |dW| / W <= 16 * steps *
    2^-23 = 1.9e-3 at 1,000 steps -- still 30x below what the removed pellet
    drain took out (6e-2 over 1,000 steps)."""
    import reactor_gpu
    cfg, out, W0, _ = _closed_run(reactor_gpu._run_reactor_loop_gpu)
    assert out[9] == 0, f"{out[9]} particles lost: not a closed system"
    W1 = float(out[11][-1].sum())
    rel = abs(W1 - W0) / W0
    hard = 16 * ENERGY_TEST_STEPS * 2.0 ** -23
    assert rel < hard, f"|dW|/W = {rel:.3e} exceeds the float32-arithmetic bound {hard:.3e}"


# =======================================================
# TIER 2.5 PHASE 2 -- constants, tau_E
# =======================================================
def test_no_mass_or_charge_literal_outside_constants():
    """Every mass and charge comes from constants.py (CODATA 2022). Scan every
    project .py file except constants.py for a numeric literal that is a particle
    mass (any *e-27 or *e-31 kg number, an atomic-mass-unit multiplier), the
    elementary charge or 1 keV in joules (1.6...e-19 / e-16), mu0 (4e-7 pi,
    1.2566...e-6) or epsilon0 (8.85...e-12). Comments and strings are scanned
    too: a stale value in a comment is how a literal comes back."""
    import re
    pattern = re.compile(
        r"\b\d+(\.\d+)?[eE]-(27|31)\b"                     # kg masses
        r"|\b1\.6\d*[eE]-1[0-9]\b"                          # e, keV, MeV in SI
        r"|\b4(\.0*)?[eE]-0?7\s*\*\s*(np\.|math\.)?pi\b"   # mu0 = 4 pi 1e-7
        r"|\b1\.2566\d*[eE]-0?6\b"                           # mu0
        r"|\b8\.85\d*[eE]-12\b"                              # epsilon0
        r"|\b(1\.6605|1\.6726|9\.109)\d*\b")                 # u, m_p, m_e mantissas
    skip_dirs = {"venv", ".venv", ".git", "__pycache__", ".cache", "docs"}
    hits = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for name in files:
            if not name.endswith(".py") or (root == REPO and name == "constants.py"):
                continue
            path = os.path.join(root, name)
            with open(path) as fh:
                for k, line in enumerate(fh, 1):
                    if pattern.search(line):
                        hits.append(f"{os.path.relpath(path, REPO)}:{k}: {line.strip()}")
    assert not hits, "mass/charge literals outside constants.py:\n" + "\n".join(hits)


def test_energy_confinement_time_synthetic_exact():
    """tau_E = W / P_loss with P_loss averaged over the trailing window.
    (a) A constant loss of P dt per step gives P_loss = P from the first step on
    (the average runs over the steps so far until the window fills), so tau_E = W/P
    to float64 rounding. (b) A single loss E at step s gives P_loss = E / (k dt)
    with k = min(step + 1, window) for the next `window` steps, then 0 and tau_E =
    NaN (nothing being lost)."""
    from physics_engine import WindowedLossPower, energy_confinement_time
    dt, window = 1.0e-9, 1000
    W, P = 3.7e5, 2.5e7                                   # J, W -> tau_E = 1.48e-2 s
    lp = WindowedLossPower(window, dt)
    for step in range(3 * window):
        P_loss = lp.add(P * dt)
        assert abs(P_loss / P - 1.0) < 1e-12, (step, P_loss)
        assert abs(energy_confinement_time(W, P_loss) / (W / P) - 1.0) < 1e-12
    E, s = 42.0, 1500
    lp = WindowedLossPower(window, dt)
    for step in range(s + window + 10):
        P_loss = lp.add(E if step == s else 0.0)
        if step < s or step >= s + window:
            assert P_loss == 0.0, (step, P_loss)
            assert np.isnan(energy_confinement_time(W, P_loss))
        else:
            assert P_loss == E / (window * dt), (step, P_loss)
            assert energy_confinement_time(W, P_loss) == W / (E / (window * dt))
    lp = WindowedLossPower(window, dt)                    # early in the run: k = step + 1
    for step in range(5):
        P_loss = lp.add(E if step == 0 else 0.0)
        assert P_loss == E / ((step + 1) * dt), (step, P_loss)


def test_lost_energy_closes_the_energy_balance():
    """The loss bookkeeping behind P_loss, end to end through the CPU loop: in a
    closed system (E = 0, sources off) whose loss boundary is pulled inward
    (psi_edge = 1.06 psi_core), the energy carried out through the LCFS plus the
    energy still stored equals the energy loaded, W0 = W_end + sum(lost_energy_J),
    to the float32 rounding bound of the closed-system test (2 * steps * 2^-23).
    The per-species loss counts add up to total_lost, and tau_E is NaN before the
    first thermal loss and W_thermal / P_loss (finite) from then on."""
    import reactor_cpu
    steps = 200
    cfg, out, W0, _ = _closed_run(reactor_cpu._run_reactor_loop_cpu, n_thermal=4000,
                                  steps=steps, psiN_max=1.0, edge_scale=1.06)
    log = out[-1]
    assert out[9] > 100, f"only {out[9]} losses: the forced-loss setup did not bite"
    assert int(log["lost_macro"].sum()) == out[9]
    assert log["lost_macro"][1] == 0 and log["lost_macro"][2] == 0     # no beam, no alphas
    W_end = float(out[11][-1].sum())
    E_out = float(log["lost_energy_J"].sum())
    rel = abs(W0 - W_end - E_out) / W0
    assert rel < 2 * steps * 2.0 ** -23, f"W0 - W_end - E_lost = {rel:.3e} of W0"
    tau = log["tau_E"]
    k0 = out[15]
    assert k0 is not None and np.all(np.isnan(tau[:k0]))
    E_th = out[11][:, 0] + out[11][:, 1]
    P = log["P_loss_thermal_W"]
    ok = P > 0
    assert np.all(np.isfinite(tau[k0:][ok[k0:]]))
    assert np.allclose(tau[ok], E_th[ok] / P[ok], rtol=1e-14, atol=0.0)


# =======================================================
# TIER 2.5 PHASE 3 -- ITER-referenced scenario
# =======================================================
def _thermal_mu_ratio(push, mass, T_keV, n=256, steps=1500, seed=17):
    """mu(end)/mu(start) for n ions loaded as a T_keV Maxwellian at 0.05 < psi_N
    < 0.6, pushed `steps` global steps by the loop's push function `push(cfg, pos,
    vel, B, E)` with the equilibrium field gathered every step, as the loop does
    (E = 0, no tearing mode: mu is the equilibrium invariant)."""
    from kernels import vectorized_gather_and_B_into
    cfg, eq, engine, B_R, B_Z = _production_fields()
    rng = np.random.default_rng(seed)
    pos = _positions_in_psiN(eq, n, rng, 0.05, 0.6).astype(np.float32)
    state = np.random.get_state()
    np.random.seed(seed)
    vel = engine.initialize_velocities_profile(np.full(n, T_keV), mass=mass).astype(np.float32)
    np.random.set_state(state)
    pos, vel = np.ascontiguousarray(pos), np.ascontiguousarray(vel)
    _, mu0 = _pitch_and_mu(cfg, pos, vel, B_R, B_Z, mass)
    zero = np.zeros((cfg.nR, cfg.nZ))
    E = np.empty((n, 3), dtype=np.float32)
    B = np.empty((n, 3), dtype=np.float32)
    for _ in range(steps):
        vectorized_gather_and_B_into(pos, E, B, zero, zero, B_R, B_Z,
                                     cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ,
                                     cfg.B0, cfg.R0_major, 0.0, 0.0, cfg.m_mode, cfg.n_mode, 0.0)
        push(cfg, pos, vel, B, E)
    _, mu1 = _pitch_and_mu(cfg, pos, vel, B_R, B_Z, mass)
    ok = mu0 > 1e-3 * np.median(mu0)          # drop near-parallel ions (mu ~ 0)
    return (mu1 / mu0)[ok]


def test_thermal_push_conserves_magnetic_moment_at_25_keV():
    """At the scenario's 25 keV on-axis temperature the thermal Larmor radius is
    ~3 mm (5x the 1 keV value), and the global 1 ns step (omega_c dt = 0.575 rad
    for D, 0.384 rad for T at B0) must still conserve mu: the same bounds as
    test_tritium_push_conserves_magnetic_moment (median within 1%, 90th percentile
    of |mu1/mu0 - 1| below 5%), for thermal D and thermal T, through the loop's
    own push functions."""
    import reactor_cpu
    cfg, *_ = _production_fields()
    for name, push, mass in (("D", reactor_cpu._push_thermal_deuterium, cfg.m_deuterium),
                             ("T", reactor_cpu._push_thermal_tritium, cfg.m_tritium)):
        ratio = _thermal_mu_ratio(push, mass, 25.0)
        med = float(np.median(ratio))
        assert 0.99 < med < 1.01, f"thermal {name}: median mu(end)/mu(start) = {med:.5f}"
        p90 = float(np.percentile(np.abs(ratio - 1.0), 90))
        assert p90 < 0.05, f"thermal {name}: 90th percentile |mu1/mu0 - 1| = {p90:.4f}"


def _isotropic_beam(cfg, n, rng, E_keV):
    """n isotropic deuterons at E_keV around R = 1.1 m (the alpha test's load)."""
    from constants import KEV_J
    v0 = np.sqrt(2.0 * E_keV * KEV_J / cfg.m_deuterium)
    R = rng.normal(1.10, 0.03, n)
    Z = rng.normal(0.0, 0.03, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    pos = np.stack([R * np.cos(phi), R * np.sin(phi), Z], axis=1).astype(np.float32)
    u = rng.normal(size=(n, 3))
    vel = (v0 * u / np.linalg.norm(u, axis=1)[:, None]).astype(np.float32)
    return np.ascontiguousarray(pos), np.ascontiguousarray(vel)


def test_beam_push_conserves_magnetic_moment_at_1_MeV():
    """1 MeV beam deuterons (Larmor radius ~1.7 cm, ~1 cm of travel per global
    step) pushed by reactor_cpu._push_beam -- cfg.NBI_SUBSTEPS sub-steps with the
    field re-gathered every sub-step -- conserve mu by the alpha test's bounds
    (median mu ratio within 10% after 1.5 us, >= 80% still confined, no pitch
    collapse to xi = -1). The sub-step count comes from the alphas' criterion."""
    import reactor_cpu
    from physics_engine import interpolate_psi_array
    cfg, eq, engine, B_R, B_Z = _production_fields()
    assert cfg.NBI_SUBSTEPS == 10
    rng = np.random.default_rng(7)
    pos, vel = _isotropic_beam(cfg, ALPHA_TEST_COUNT, rng, 1000.0)
    xi0, mu0 = _pitch_and_mu(cfg, pos, vel, B_R, B_Z, cfg.m_deuterium)
    E_zero = np.zeros((cfg.nR, cfg.nZ))
    for step in range(ALPHA_TEST_STEPS):
        reactor_cpu._push_beam(cfg, pos, vel, E_zero, E_zero, B_R, B_Z, step * cfg.reactor_dt)
    xi1, mu1 = _pitch_and_mu(cfg, pos, vel, B_R, B_Z, cfg.m_deuterium)
    R = np.hypot(pos[:, 0], pos[:, 1]).astype(np.float64)
    psi = interpolate_psi_array(R, pos[:, 2].astype(np.float64), eq.psi_grid,
                                eq.psi_R_min, eq.psi_R_max, eq.psi_Z_min, eq.psi_Z_max,
                                eq.psi_nR, eq.psi_nZ)
    inside = psi > eq.psi_edge
    assert inside.sum() >= 0.8 * len(inside), f"only {inside.sum()} beam ions still confined"
    med = float(np.median(mu1[inside] / mu0[inside]))
    assert 0.9 < med < 1.1, f"median mu(end)/mu(start) = {med:.4f}"
    assert float(np.mean(xi1[inside] < -0.95)) < float(np.mean(xi0 < -0.95)) + 0.1


def test_beam_push_conserves_magnetic_moment_at_1_MeV_gpu_twin():
    """Same invariant through reactor_gpu._push_beam_torch on cfg.HPC_DEVICE
    (shorter run, as for the alpha GPU twin)."""
    import torch
    import reactor_gpu
    cfg, eq, engine, B_R, B_Z = _production_fields()
    rng = np.random.default_rng(7)
    pos, vel = _isotropic_beam(cfg, ALPHA_TEST_COUNT, rng, 1000.0)
    _, mu0 = _pitch_and_mu(cfg, pos, vel, B_R, B_Z, cfg.m_deuterium)
    f32 = dict(device=cfg.HPC_DEVICE, dtype=torch.float32)
    pos_t, vel_t = torch.tensor(pos, **f32), torch.tensor(vel, **f32)
    E0 = torch.zeros((cfg.nR, cfg.nZ), **f32)
    BR_t, BZ_t = torch.tensor(B_R, **f32), torch.tensor(B_Z, **f32)
    for step in range(600):
        reactor_gpu._push_beam_torch(cfg, pos_t, vel_t, E0, E0, BR_t, BZ_t, step * cfg.reactor_dt)
    _, mu1 = _pitch_and_mu(cfg, pos_t.cpu().numpy(), vel_t.cpu().numpy(), B_R, B_Z, cfg.m_deuterium)
    med = float(np.median(mu1 / mu0))
    assert 0.9 < med < 1.1, f"median mu(end)/mu(start) = {med:.4f}"


def test_nrl_collision_frequency_hand_computed():
    """NRL Plasma Formulary ion collision rate, nu_i = 4.80e-8 Z^4 mu^-1/2 n_i
    [cm^-3] T_i[eV]^-3/2 ln(Lambda), against a value worked by hand: n_i = 1e20
    m^-3 = 1e14 cm^-3, T_i = 10 keV, mu = 2, Z = 1, ln(Lambda) = 20:
        4.80e-8 * 0.70711 * 1e14 * 1e-6 * 20 = 67.882 s^-1.
    The mixed ion-ion Coulomb log for D-T at n_D = n_T = 5e19 m^-3 and T = 10 keV
    (equal temperatures, so (mu+mu')/(mu T' + mu' T) = 1/T = 1e-4 eV^-1, and
    sqrt(n_D/T + n_T/T) = sqrt(1e14 / 1e4) = 1e5): 23 - ln(10) = 20.6974.
    The loop's rates (cfg.NU_C, set by initialize_reactor) are this formula at
    the loaded central density and temperature, and that density is the on-axis
    value of the (1 - psi_N) load: n0 = <n> V / sum_k (1 - x_k) V_k to 0.5%."""
    from physics_engine import (nrl_ion_collision_frequency, coulomb_log_ion_ion,
                                loaded_central_density, compute_flux_shell_volumes)
    from constants import M_PROTON
    nu = nrl_ion_collision_frequency(1.0e20, 10.0, 2.0, 20.0)
    assert abs(nu / 67.882 - 1.0) < 1e-4, nu
    lnL = coulomb_log_ion_ion(5.0e19, 5.0e19, 10.0, 10.0, 2.0136, 3.0155)
    assert abs(lnL - 20.6974) < 1e-4, lnL
    cfg, eq, *_ = _production_fields()
    n0 = loaded_central_density(eq, cfg.N_AVG_TARGET)
    V = compute_flux_shell_volumes(eq, n_bins=200, n_mesh=800)
    x = (np.arange(200) + 0.5) / 200
    assert abs(n0 / (cfg.N_AVG_TARGET * V.sum() / np.sum((1 - x) * V)) - 1.0) < 5e-3
    T0 = cfg.T_thermal_keV
    mu_D, mu_T = cfg.m_deuterium / M_PROTON, cfg.m_tritium / M_PROTON
    L = coulomb_log_ion_ion(n0 / 2, n0 / 2, T0, T0, mu_D, mu_T)
    assert cfg.NU_C[0] == nrl_ion_collision_frequency(n0, T0, mu_D, L)
    assert cfg.NU_C[1] == cfg.NU_C[0]
    assert cfg.NU_C[3] == nrl_ion_collision_frequency(n0, T0, mu_T, L)


def test_alpha_at_thermalization_floor_deposits_nothing():
    """The thermalization floor is (3/2) T_i on each alpha's own flux surface
    (alpha_floor_lut_keV of the latest profiles, looked up by
    alpha_thermalization_floor_keV). An alpha exactly at its floor deposits
    nothing and keeps its velocity; one a little above it (with a drag step far
    larger than the gap) deposits exactly the gap and lands on the floor; the
    torch lookup the GPU loop uses agrees with the numpy one."""
    import torch
    from physics_engine import (alpha_floor_lut_keV, alpha_thermalization_floor_keV,
                                alpha_thermalization_floor_keV_torch, drain_alpha_energy,
                                compute_flux_shell_volumes)
    from constants import KEV_J
    cfg, eq, engine, *_ = _production_fields()
    nb = len(compute_flux_shell_volumes(eq))
    x_c = (np.arange(nb) + 0.5) / nb
    prof = {"n": np.zeros((4, nb)), "T_keV": np.full((4, nb), np.nan)}
    prof["n"][0] = prof["n"][3] = 1e20 * (1 - x_c)
    prof["T_keV"][0] = 25.0 * (1 - x_c)
    prof["T_keV"][3] = 20.0 * (1 - x_c)
    lut = alpha_floor_lut_keV(prof)
    assert np.allclose(lut, 1.5 * 22.5 * (1 - x_c), rtol=1e-12)      # density-weighted T_i
    rng = np.random.default_rng(51)
    pos = np.concatenate([_positions_in_psiN(eq, 20, rng, (k + 0.2) / nb, (k + 0.8) / nb)
                          for k in range(0, nb - 1, 3)]).astype(np.float32)
    floor = alpha_thermalization_floor_keV(pos, eq, lut)
    assert np.all(floor > 0.0)
    u = rng.normal(size=pos.shape)
    u /= np.linalg.norm(u, axis=1)[:, None]

    def vel_at(E_keV):
        return u * np.sqrt(2.0 * E_keV * KEV_J / cfg.MASS_ALPHA)[:, None]

    rel_w = np.full(len(pos), 1e-6)
    # Just below the looked-up floor (1e-12 relative, far above float64 rounding):
    # nothing deposited, velocities untouched.
    v_at = vel_at(floor * (1.0 - 1e-12))
    v_drained = v_at.copy()
    dep = drain_alpha_energy(engine, cfg, v_drained, rel_w, 1.0, floor)
    assert dep == 0.0, dep
    assert np.array_equal(v_drained, v_at), "an alpha below its floor was slowed"
    # Exactly at the floor, with the energy computed from the velocity the way the
    # drain computes it (a round trip through |v| moves E by ~1e-16).
    E_exact = 0.5 * cfg.MASS_ALPHA * np.linalg.norm(v_at, axis=1) ** 2 / KEV_J
    v_drained = v_at.copy()
    assert drain_alpha_energy(engine, cfg, v_drained, rel_w, 1.0, E_exact) == 0.0
    assert np.array_equal(v_drained, v_at), "an alpha at its floor was slowed"

    gap = 1e-3 * floor
    v_above = vel_at(floor + gap)
    dep = drain_alpha_energy(engine, cfg, v_above, rel_w, 1.0, floor)   # drag fraction 0.99 >> gap
    E_after = 0.5 * cfg.MASS_ALPHA * np.sum(v_above ** 2, axis=1) / KEV_J
    assert np.allclose(E_after, floor, rtol=1e-9)
    assert abs(dep / float(np.sum(rel_w * gap)) - 1.0) < 1e-6

    psi_t = torch.tensor(eq.psi_grid, device=cfg.HPC_DEVICE, dtype=torch.float32)
    f_t = alpha_thermalization_floor_keV_torch(
        torch.tensor(pos, device=cfg.HPC_DEVICE), psi_t, eq,
        torch.tensor(lut, device=cfg.HPC_DEVICE, dtype=torch.float32)).cpu().numpy()
    assert np.allclose(f_t, floor, rtol=1e-6), "torch floor lookup disagrees with numpy"


# =======================================================
# PLAIN-PYTHON RUNNER
# =======================================================
if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
    sys.exit(1 if failed else 0)
