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
    ek = (0.5 * cfg.MASS_ALPHA * vm**2) / 1.602e-16
    # Same lines as the loop, with a large dt so the drain is visible.
    new_ek, _ = engine.compute_alpha_heating_power(ek, 1e-2, cfg)
    new_vm = np.sqrt(2.0 * (new_ek * 1.602e-16) / cfg.MASS_ALPHA)
    out = (v64 * (new_vm / vm)[:, None]).astype(np.float32).astype(np.float64)
    cosang = np.sum(out * v64, axis=1) / (np.linalg.norm(out, axis=1) * vm)
    assert np.all(new_ek <= ek)
    assert np.max(1.0 - cosang) < 1e-6, f"direction changed: max 1-cos = {np.max(1.0 - cosang):.2e}"


# =======================================================
# PHASE 2 -- flux-surface profiles
# =======================================================
def _species_masses(cfg):
    return (cfg.m_deuterium, cfg.m_deuterium, cfg.MASS_ALPHA)


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
    """initialize_reactor loads n = n0 (1 - psi_N), T = T0 (1 - psi_N) with
    <n> = N_AVG_TARGET, and WEIGHT[0] = N_AVG_TARGET * V / N."""
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
    pos, vel, typ = p.cpu().numpy(), v.cpu().numpy(), t.cpu().numpy()
    prof = compute_flux_profiles(pos, vel, typ, eq, _species_masses(cfg), weight=cfg.WEIGHT[0])
    V = prof["shell_volume"]
    n_avg = np.sum(prof["n"][0] * V) / V.sum()
    assert abs(n_avg / cfg.N_AVG_TARGET - 1.0) < 1e-3, f"<n> = {n_avg:.4e}"
    x = prof["psiN_centers"]
    core = x < 0.8                                       # >= ~1000 ions per shell
    # Fit n = n0 (1 - x) and T = T0 (1 - x) by least squares over the core shells.
    n0 = np.sum(prof["n"][0, core] * (1 - x[core])) / np.sum((1 - x[core]) ** 2)
    T0 = np.sum(prof["T_keV"][0, core] * (1 - x[core])) / np.sum((1 - x[core]) ** 2)
    assert abs(T0 / cfg.T_thermal_keV - 1.0) < 0.02, f"fitted T0 = {T0:.4f} keV"
    resid_n = np.abs(prof["n"][0, core] / (n0 * (1 - x[core])) - 1.0)
    resid_T = np.abs(prof["T_keV"][0, core] / (T0 * (1 - x[core])) - 1.0)
    assert resid_n.max() < 0.1 and resid_T.max() < 0.1, (resid_n.max(), resid_T.max())


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


def _profile_for_power(eq, n_of_x, T_keV_of_x):
    from physics_engine import compute_flux_shell_volumes
    V = compute_flux_shell_volumes(eq)
    x = 0.5 * (np.linspace(0, 1, len(V) + 1)[:-1] + np.linspace(0, 1, len(V) + 1)[1:])
    z = np.zeros((3, len(V)))
    prof = {"psiN_centers": x, "shell_volume": V, "n": z.copy(), "T_keV": np.full((3, len(V)), np.nan)}
    prof["n"][0] = n_of_x(x)
    prof["T_keV"][0] = T_keV_of_x(x)
    return prof


def test_thermal_fusion_power_is_the_shell_sum():
    """P_total = sum_k P_k V_k, and each P_k = (n_k/2)^2 <sigma v>(T_k) E_fus."""
    from physics_engine import compute_thermal_fusion_power
    cfg, eq, *_ = _production_fields()
    prof = _profile_for_power(eq, lambda x: 2e20 * (1 - x), lambda x: 12.0 * (1 - x))
    P, total = compute_thermal_fusion_power(prof, cfg)
    assert abs(total - np.sum(P * prof["shell_volume"])) <= 1e-12 * total
    for k in (0, 7, 15):
        n, T = prof["n"][0, k], prof["T_keV"][0, k]
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
    types = rng.choice([0, 1, 2, -1], size=len(pos), p=[0.6, 0.25, 0.1, 0.05])
    inside = compute_psi_normalised(np.hypot(pos[:, 0], pos[:, 1]), pos[:, 2], eq) <= 1.0
    w = (7.04e15, 2.1e12, 3.3e3)
    prof = compute_flux_profiles(pos, vel, types, eq, _species_masses(cfg), weight=w)
    for s in range(3):
        expect = w[s] * np.sum(inside & (types == s))
        got = float(np.sum(prof["n"][s] * prof["shell_volume"]))
        assert abs(got - expect) <= 1e-12 * expect, f"species {s}: {got} vs {expect}"

    saved = cfg.WEIGHT
    cfg.WEIGHT = {s: w[s] for s in range(3)}
    try:
        rho = deposit_ion_charge_density(cfg, pos, types)
    finally:
        cfg.WEIGHT = saved
    dR = (cfg.R_max - cfg.R_min) / (cfg.nR - 1)
    dZ = (cfg.Z_max - cfg.Z_min) / (cfg.nZ - 1)
    R_nodes = cfg.R_min + np.arange(cfg.nR) * dR
    total = float(np.sum(rho * 2.0 * np.pi * R_nodes[:, None] * dR * dZ))
    expect = (cfg.e_charge * (w[0] * np.sum(types == 0) + w[1] * np.sum(types == 1))
              + cfg.CHARGE_ALPHA * w[2] * np.sum(types == 2))
    assert abs(total / expect - 1.0) < 1e-10, f"{total:.12e} vs {expect:.12e}"


def _alpha_to_bulk_transfer(cfg, engine, a_vel, a_w, groups, dt):
    """Run the reactor loops' alpha -> bulk transfer (physics_engine helpers, in
    the loop's order) on float64 copies. groups: [(vel, w, mass)] bulk pools.
    Weights are absolute; the loop's relative weights are w / w_ref."""
    from physics_engine import drain_alpha_energy, weighted_kinetic_energy_kev, bulk_boost_factor
    w_ref = 7.04e15
    a = a_vel.astype(np.float64)
    dep = drain_alpha_energy(engine, cfg, a, a_w / w_ref, dt)
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
