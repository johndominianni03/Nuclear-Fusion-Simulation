#!/usr/bin/env python3
"""Alpha pitch-angle diagnostic (Tier 1, Phase 1).

Follows an alpha population through the same kernels the CPU reactor loop
uses -- reactor_cpu._push_alphas, compute_alpha_heating_power and
check_confinement_flux -- on the production equilibrium, and records every
alpha's pitch xi = v_par/|v|, R, Z, energy, magnetic moment and birth step at
a fixed cadence.

The bulk plasma is not simulated. The alphas see the bulk only through E
(the Poisson field, which is ~1e-3 V/m here; --E-from-loop replays the
loop's own end-of-run E grids instead of zero) and nothing else: collisions
skip the alpha pool and the alpha->bulk deposit is one-way. That makes the
alpha-only run a faithful, ~100x cheaper copy of the loop's alpha dynamics.

The alphas here are injected on this tool's own fixed schedule (one every
ALPHA_EVERY steps, Gaussian about R0_major), the Tier 1 schedule, not the
reactor's fusion-driven source: the tool tests the pusher, so it keeps the
population it was validated with.

Variants, one switch each, so a hypothesis is tested by changing one thing:

    --push loop        reactor_cpu._push_alphas, what the loop runs (default):
                       E and B re-gathered at every sub-step
    --push frozen      the pre-Tier-1 kernel (a copy is kept below): E and B
                       gathered once per global step and held across all
                       sub-steps. This is the bug that collapsed the pitch.
    --substeps N       override cfg.ALPHA_SUBSTEPS
    --no-heating       skip compute_alpha_heating_power
    --no-perturb       tearing-mode b_perturb = 0 in the pusher's field
    --E-from-loop      run the real CPU loop first (2,000 particles) and use
                       its final E_R/E_Z grids instead of zero

    venv/bin/python tools/alpha_pitch_check.py --steps 10000
"""
import argparse
import os
import sys

os.environ.setdefault("FUSION_HEADLESS", "1")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np                                     # noqa: E402

ALPHA_EVERY = 4      # this tool's fixed alpha schedule (the Tier 1 reactor schedule)
from numba import njit, prange                         # noqa: E402

from config import SimulationConfiguration             # noqa: E402
from constants import KEV_J                            # noqa: E402
import initialization                                  # noqa: E402
from kernels import vectorized_gather_and_B_into       # noqa: E402
from physics_engine import (                           # noqa: E402
    compute_poloidal_field_grids,
    check_confinement_flux,
)
from reactor_cpu import _push_alphas                   # noqa: E402


@njit(parallel=True, fastmath=True)
def frozen_field_substeps(pos_arr, vel_arr, q, m, B_arr, E_arr, dt, n_sub):
    """The removed physics_engine.vectorized_boris_push_numba_substeps, verbatim
    in its arithmetic: B_arr/E_arr were gathered once at the start-of-step
    position and held across all n_sub sub-steps. Kept only so --push frozen
    can reproduce the pitch collapse."""
    dt_sub = dt / n_sub
    qm = q / m
    half = dt_sub / 2.0
    for i in prange(pos_arr.shape[0]):
        ex = E_arr[i, 0]; ey = E_arr[i, 1]; ez = E_arr[i, 2]
        tx = qm * B_arr[i, 0] * half
        ty = qm * B_arr[i, 1] * half
        tz = qm * B_arr[i, 2] * half
        f = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz)
        sx = tx * f; sy = ty * f; sz = tz * f
        px = np.float64(pos_arr[i, 0]); py = np.float64(pos_arr[i, 1]); pz = np.float64(pos_arr[i, 2])
        vx = np.float64(vel_arr[i, 0]); vy = np.float64(vel_arr[i, 1]); vz = np.float64(vel_arr[i, 2])
        for _ in range(n_sub):
            vmx = vx + qm * ex * half
            vmy = vy + qm * ey * half
            vmz = vz + qm * ez * half
            ax = vmy * tz - vmz * ty
            ay = vmz * tx - vmx * tz
            az = vmx * ty - vmy * tx
            vpx = vmx + ax; vpy = vmy + ay; vpz = vmz + az
            cx = vpy * sz - vpz * sy
            cy = vpz * sx - vpx * sz
            cz = vpx * sy - vpy * sx
            vx = vmx + cx + qm * ex * half
            vy = vmy + cy + qm * ey * half
            vz = vmz + cz + qm * ez * half
            px += vx * dt_sub; py += vy * dt_sub; pz += vz * dt_sub
        vel_arr[i, 0] = vx; vel_arr[i, 1] = vy; vel_arr[i, 2] = vz
        pos_arr[i, 0] = px; pos_arr[i, 1] = py; pos_arr[i, 2] = pz


def build(seed):
    np.random.seed(seed)
    cfg = SimulationConfiguration()
    cfg.initial_thermal_count = 1000      # init only; the bulk is not pushed here
    eq, engine, *_ = initialization.initialize_reactor(cfg)
    B_R, B_Z, _, _ = compute_poloidal_field_grids(
        eq, cfg.B0, R0=cfg.R0_major, a_minor=0.3, q_target=cfg.Q_SAFETY_TARGET,
        dst_R_min=cfg.R_min, dst_R_max=cfg.R_max, dst_Z_min=cfg.Z_min,
        dst_Z_max=cfg.Z_max, dst_nR=cfg.nR, dst_nZ=cfg.nZ)
    return cfg, eq, engine, B_R, B_Z


def loop_E_grids(cfg_steps=200, n=2000, seed=1):
    """E_R/E_Z of the force-balance field built from the CPU loop's state at the
    end of a short run (the field the loop would apply after its next update)."""
    import main
    import reactor_cpu
    from physics_engine import update_force_balance_field
    np.random.seed(seed)
    cfg = SimulationConfiguration()
    cfg.initial_thermal_count = n
    cfg.reactor_num_steps = cfg_steps
    eq, engine, p, v, t, rho, phi, ER, EZ = initialization.initialize_reactor(cfg)
    B_R, B_Z, _, _ = compute_poloidal_field_grids(
        eq, cfg.B0, R0=cfg.R0_major, a_minor=0.3, q_target=cfg.Q_SAFETY_TARGET,
        dst_R_min=cfg.R_min, dst_R_max=cfg.R_max, dst_Z_min=cfg.Z_min,
        dst_Z_max=cfg.Z_max, dst_nR=cfg.nR, dst_nZ=cfg.nZ)
    out = reactor_cpu._run_reactor_loop_cpu(cfg, engine, p, v, t, rho, phi, ER, EZ, B_R, B_Z)
    _, E_R, E_Z, _ = update_force_balance_field(cfg, eq, out[0], out[1], out[2])
    return E_R, E_Z


def gather(cfg, pos, E_R, E_Z, B_R, B_Z, t, b_perturb):
    n = len(pos)
    E = np.empty((n, 3), dtype=np.float32)
    B = np.empty((n, 3), dtype=np.float32)
    vectorized_gather_and_B_into(pos, E, B, E_R, E_Z, B_R, B_Z,
                                 cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ,
                                 cfg.B0, cfg.R0_major, t, b_perturb, cfg.m_mode, cfg.n_mode,
                                 cfg.gamma_growth)
    return E, B


def equilibrium_b(cfg, pos, B_R, B_Z):
    """Equilibrium field (no E, no tearing mode) -- what package_reactor_results projects on."""
    z = np.zeros((cfg.nR, cfg.nZ))
    _, B = gather(cfg, np.ascontiguousarray(pos, dtype=np.float32), z, z, B_R, B_Z, 0.0, 0.0)
    return B.astype(np.float64)


def run(args):
    cfg, eq, engine, B_R, B_Z = build(args.seed)
    if args.substeps is not None:
        cfg.ALPHA_SUBSTEPS = args.substeps
    n_sub = cfg.ALPHA_SUBSTEPS
    if args.no_perturb:
        cfg.b_perturb_initial = 0.0
    if args.E_from_loop:
        E_R, E_Z = loop_E_grids()
        print(f"[alpha] replaying loop E grids: max|E_R| {np.abs(E_R).max():.3e} V/m, "
              f"max|E_Z| {np.abs(E_Z).max():.3e} V/m")
    else:
        E_R = np.zeros((cfg.nR, cfg.nZ))
        E_Z = np.zeros((cfg.nR, cfg.nZ))
    b_perturb = 0.0 if args.no_perturb else cfg.b_perturb_initial
    psi_args = (eq.psi_grid, eq.psi_edge, eq.psi_R_min, eq.psi_R_max,
                eq.psi_Z_min, eq.psi_Z_max, eq.psi_nR, eq.psi_nZ)

    pos = np.zeros((0, 3), dtype=np.float32)
    vel = np.zeros((0, 3), dtype=np.float32)
    typ = np.zeros(0, dtype=np.int32)
    birth = np.zeros(0, dtype=np.int64)
    pid = np.zeros(0, dtype=np.int64)
    next_pid = 0
    records = []   # (step, pid, birth, R, Z, xi, E_keV, mu)
    lost_log = []  # (step, pid, birth, xi_at_loss)

    for step in range(args.steps):
        t = step * cfg.reactor_dt
        if step % ALPHA_EVERY == 0 and step > 0:
            phi_pos = np.random.uniform(0, 2 * np.pi, 1)
            R_birth = np.random.normal(cfg.R0_major, 0.05, 1)
            Z_birth = np.random.normal(0.0, 0.05, 1)
            aR, aZ, avR, avphi, avZ = engine.spawn_alpha_particles(
                1, R_birth[0], Z_birth[0], cfg.ALPHA_ENERGY_JOULES, cfg.MASS_ALPHA)
            p = np.zeros((1, 3), dtype=np.float32)
            v = np.zeros((1, 3), dtype=np.float32)
            p[:, 0] = aR * np.cos(phi_pos); p[:, 1] = aR * np.sin(phi_pos); p[:, 2] = aZ
            v[:, 0] = avR * np.cos(phi_pos) - avphi * np.sin(phi_pos)
            v[:, 1] = avR * np.sin(phi_pos) + avphi * np.cos(phi_pos)
            v[:, 2] = avZ
            pos = np.ascontiguousarray(np.vstack([pos, p]))
            vel = np.ascontiguousarray(np.vstack([vel, v]))
            typ = np.append(typ, np.int32(2))
            birth = np.append(birth, step)
            pid = np.append(pid, next_pid)
            next_pid += 1

        if len(pos) > 0:
            if args.push == "frozen":
                E, B = gather(cfg, pos, E_R, E_Z, B_R, B_Z, t, b_perturb)
                frozen_field_substeps(pos, vel, cfg.CHARGE_ALPHA, cfg.MASS_ALPHA,
                                      B, E, cfg.reactor_dt, n_sub)
            else:
                _push_alphas(cfg, pos, vel, E_R, E_Z, B_R, B_Z, t)

            if not args.no_heating:
                av = vel.astype(np.float64)
                vm = np.linalg.norm(av, axis=1)
                ek = (0.5 * cfg.MASS_ALPHA * vm**2) / KEV_J
                new_ek, _ = engine.compute_alpha_heating_power(ek, cfg.reactor_dt, cfg)
                new_vm = np.sqrt(2.0 * (new_ek * KEV_J) / cfg.MASS_ALPHA)
                av *= (new_vm / np.where(vm == 0, 1e-10, vm))[:, None]
                vel[:] = av.astype(np.float32)

            lost = check_confinement_flux(pos, typ, *psi_args)
            if lost:
                dead = typ == -1
                Bd = equilibrium_b(cfg, pos[dead], B_R, B_Z)
                bd = Bd / np.linalg.norm(Bd, axis=1)[:, None]
                vd = vel[dead].astype(np.float64)
                xi_d = np.sum(vd * bd, axis=1) / np.linalg.norm(vd, axis=1)
                for a, b_, x in zip(pid[dead], birth[dead], xi_d):
                    lost_log.append((step, a, b_, x))
                keep = ~dead
                pos = np.ascontiguousarray(pos[keep]); vel = np.ascontiguousarray(vel[keep])
                typ = typ[keep]; birth = birth[keep]; pid = pid[keep]

        if step % args.record_every == 0 and len(pos) > 0:
            Beq = equilibrium_b(cfg, pos, B_R, B_Z)
            Bm = np.linalg.norm(Beq, axis=1)
            b = Beq / Bm[:, None]
            v64 = vel.astype(np.float64)
            vm = np.linalg.norm(v64, axis=1)
            vpar = np.sum(v64 * b, axis=1)
            vperp2 = np.maximum(vm**2 - vpar**2, 0.0)
            R = np.hypot(pos[:, 0], pos[:, 1]).astype(np.float64)
            for k in range(len(pos)):
                records.append((step, pid[k], birth[k], R[k], float(pos[k, 2]), vpar[k] / vm[k],
                                0.5 * cfg.MASS_ALPHA * vm[k]**2 / KEV_J,
                                cfg.MASS_ALPHA * vperp2[k] / (2.0 * Bm[k])))

    rec = np.array(records, dtype=[("step", "i8"), ("pid", "i8"), ("birth", "i8"), ("R", "f8"),
                                   ("Z", "f8"), ("xi", "f8"), ("E_keV", "f8"), ("mu", "f8")])
    lost = np.array(lost_log, dtype=[("step", "i8"), ("pid", "i8"), ("birth", "i8"), ("xi", "f8")])
    return rec, lost, next_pid


def report(rec, lost, n_born, args):
    edges = np.linspace(-1.0, 1.0, 11)
    print(f"\n=== alpha pitch report: push={args.push} substeps={args.substeps} "
          f"heating={not args.no_heating} perturb={not args.no_perturb} E_from_loop={args.E_from_loop} ===")
    print(f"born {n_born}, lost {len(lost)}")
    if len(lost):
        h, _ = np.histogram(lost["xi"], edges)
        print("pitch at loss  (bins -1..1 by 0.2):", " ".join(f"{c:5d}" for c in h))
    steps = np.unique(rec["step"])
    pick = steps[np.linspace(0, len(steps) - 1, min(8, len(steps))).astype(int)]
    print("\npitch histogram of live alphas (xi bins -1..1 by 0.2), and the fraction with xi < -0.95")
    for s in pick:
        sel = rec[rec["step"] == s]
        h, _ = np.histogram(sel["xi"], edges)
        print(f" step {s:6d}  n={len(sel):5d} | " + " ".join(f"{c:5d}" for c in h)
              + f" | xi<-0.95: {np.mean(sel['xi'] < -0.95):.2f}")

    print("\nper-alpha trajectories (xi, mu/mu0, E/E0, R) for the first few alphas born")
    for a in np.unique(rec["pid"])[:args.n_traj]:
        tr = rec[rec["pid"] == a]
        idx = np.linspace(0, len(tr) - 1, min(6, len(tr))).astype(int)
        mu0 = tr["mu"][0] if tr["mu"][0] > 0 else np.nan
        cells = [f"s{tr['step'][i]}: xi={tr['xi'][i]:+.3f} mu={tr['mu'][i]/mu0:.3f} "
                 f"E={tr['E_keV'][i]/tr['E_keV'][0]:.4f} R={tr['R'][i]:.3f}" for i in idx]
        print(f" pid {a} (born {tr['birth'][0]}):\n   " + "\n   ".join(cells))

    # Magnetic moment drift across all alphas that lived the whole recorded span
    print("\nmagnetic moment drift: median over alphas of mu(last)/mu(first), by age")
    for a_min in (500, 2000, 5000):
        ratios, dxi = [], []
        for a in np.unique(rec["pid"]):
            tr = rec[rec["pid"] == a]
            if tr["step"][-1] - tr["step"][0] >= a_min and tr["mu"][0] > 0:
                ratios.append(tr["mu"][-1] / tr["mu"][0])
                dxi.append(tr["xi"][-1] - tr["xi"][0])
        if ratios:
            print(f" age >= {a_min:5d} steps: n={len(ratios):4d}  median mu ratio {np.median(ratios):.3f}"
                  f"  (10%/90%: {np.percentile(ratios,10):.3f}/{np.percentile(ratios,90):.3f})"
                  f"  median d(xi) {np.median(dxi):+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--record-every", type=int, default=100)
    ap.add_argument("--push", choices=("loop", "frozen"), default="loop")
    ap.add_argument("--substeps", type=int, default=None)
    ap.add_argument("--no-heating", action="store_true")
    ap.add_argument("--no-perturb", action="store_true")
    ap.add_argument("--E-from-loop", action="store_true")
    ap.add_argument("--n-traj", type=int, default=4)
    ap.add_argument("--save", default=None, help="write the records to this .npz")
    args = ap.parse_args()
    rec, lost, n_born = run(args)
    report(rec, lost, n_born, args)
    if args.save:
        np.savez(args.save, rec=rec, lost=lost)


if __name__ == "__main__":
    main()
