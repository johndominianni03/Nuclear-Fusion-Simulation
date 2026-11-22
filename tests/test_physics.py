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
