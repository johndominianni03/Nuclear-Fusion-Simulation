#!/usr/bin/env python3
"""Guiding-centre vs full-orbit benchmark (Tier 3 Phase 2).

50 fusion alphas and 50 beam deuterons are followed for 1 ms from the same
initial conditions by
  * the reactor's full-orbit Boris pusher, kernels.boris_push_substeps_regather
    (field re-gathered at every 0.1 ns sub-step, the production alpha / beam
    sub-step), in the reactor's own bilinear fields: B_pol from
    compute_poloidal_field_grids, E from Tier 1 force balance on the same grid
    (here for the background profiles), tearing-mode ripple off; and
  * the guiding-centre RK4 pusher (gc_push) in the spline fields of fields_gc.

Initial conditions: alphas isotropic at 3.5 MeV, born from the thermal D-T rate
of the background profiles (physics_engine.compute_alpha_birth_rates +
AlphaBirthSampler); beam ions from the reactor's injector
(ParticlePusher.inject_neutral_beam_cartesian, 1 MeV). The GC starts at the
particle's first-order guiding centre with the full orbit's time-averaged
energy and P_phi (gc_push.gc_from_particle_invariants): the reactor's bilinear
B_pol is not the curl of a single psi, so the full orbit conserves them only on
average (P_phi fluctuates by 0.2-0.6% peak to peak).

The full orbit is reduced to its first-order guiding centre every 20 ns (one
kernel call of 200 sub-steps). Per orbit (orbit_metrics), for both codes:
  * trapped / passing / marginal from the smoothed sign of v_par, with a 1% of v
    dead band at the separatrix;
  * period: mean interval between crossings of the ray Z = Z_c, R > R_c through
    the orbit's own centre -- the bounce period of a trapped orbit, the poloidal
    transit time of a passing one -- averaged over every period in the run;
  * width: separation of the two clusters of midplane crossings (banana width
    of a trapped orbit, midplane diameter of a passing one);
  * lost: the guiding centre reached psi_N > 1 (for both codes); the reactor's
    particle-position criterion is counted separately (finite-Larmor-radius
    losses, which a GC model cannot see).
Orbits whose GC period moves by > 3% for P_phi +- 1e-3 |P_phi| or mu x (1 +- 1%)
-- the uncertainty of the matched invariants -- are flagged invariant-sensitive
(separatrix orbits, near-axis streaming/drift cancellation) and reported apart.

    venv/bin/python tools/gc_vs_full_orbit.py            # writes the PNG and npz
Outputs: docs/images/gc_vs_full_orbit.png, data/gc_vs_full_orbit.npz.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
os.environ.setdefault("FUSION_HEADLESS", "1")

import numpy as np                                  # noqa: E402

from constants import E_CHARGE, M_ALPHA, M_DEUTERON, MEV_J, KEV_J   # noqa: E402

REC_DT = 2.0e-8          # full-orbit reduction cadence [s]
DT_SUB = 1.0e-10         # full-orbit sub-step [s] = reactor_dt / ALPHA_SUBSTEPS
SPECIES = {"alpha": (2.0 * E_CHARGE, M_ALPHA), "beam": (E_CHARGE, M_DEUTERON)}


# =======================================================
# SETUP
# =======================================================
def setup():
    """GC field + background, plus the reactor's bilinear B_pol and E grids."""
    from fields_gc import build_gc_field
    from physics_engine import compute_poloidal_field_grids, compute_grad_psiN_grids
    field, bg = build_gc_field()
    cfg, eq = field.cfg, field.eq
    B_R, B_Z, _, _ = compute_poloidal_field_grids(
        eq, cfg.B0, R0=cfg.R0_major, a_minor=0.3, q_target=cfg.Q_SAFETY_TARGET,
        dst_R_min=cfg.R_min, dst_R_max=cfg.R_max, dst_Z_min=cfg.Z_min,
        dst_Z_max=cfg.Z_max, dst_nR=cfg.nR, dst_nZ=cfg.nZ)
    # Tier 1's field construction (compute_force_balance_field) with the
    # background's dPhi/dpsi_N: E = -(dPhi/dpsi_N) grad psi_N inside the LCFS.
    psiN, gR, gZ = compute_grad_psiN_grids(eq, cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max,
                                           cfg.nR, cfg.nZ)
    _, dPhi = bg.force_balance_potential(psiN)
    dPhi = np.where(psiN <= 1.0, dPhi, 0.0)
    E_R = np.ascontiguousarray(-dPhi * gR)
    E_Z = np.ascontiguousarray(-dPhi * gZ)
    return {"field": field, "bg": bg, "cfg": cfg, "eq": eq,
            "B_R": B_R, "B_Z": B_Z, "E_R": E_R, "E_Z": E_Z}


def initial_conditions(env, n_alpha, n_beam, seed):
    """Cartesian (pos, vel) float64 for n_alpha alphas and n_beam beam ions."""
    from physics_engine import (ParticlePusher, AlphaBirthSampler,
                                compute_alpha_birth_rates)
    cfg, eq, bg = env["cfg"], env["eq"], env["bg"]
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        engine = ParticlePusher(eq)
        rates = compute_alpha_birth_rates(bg.shell_profile(eq), cfg)
        pa, va, _ = AlphaBirthSampler(eq).sample(rates, n_alpha, engine, cfg)
        psi_bounds = (eq.psi_grid, eq.psi_core, eq.psi_R_min, eq.psi_R_max,
                      eq.psi_Z_min, eq.psi_Z_max, eq.psi_nR, eq.psi_nZ)
        pb, vb = engine.inject_neutral_beam_cartesian(n_beam, E_keV=cfg.nbi_energy_keV,
                                                      psi_bounds=psi_bounds)
    finally:
        np.random.set_state(state)
    # The reactor stores float32; start both codes from those rounded values.
    pa = pa.astype(np.float32).astype(np.float64)
    va = va.astype(np.float32).astype(np.float64)
    pb = pb.astype(np.float32).astype(np.float64)
    vb = vb.astype(np.float32).astype(np.float64)
    return {"alpha": (pa, va), "beam": (pb, vb)}


# =======================================================
# THE TWO PUSHERS
# =======================================================
def _reactor_B(env, pos32):
    from kernels import vectorized_gather_and_B
    cfg = env["cfg"]
    zero = np.zeros((cfg.nR, cfg.nZ))
    _, B = vectorized_gather_and_B(pos32, zero, zero, env["B_R"], env["B_Z"],
                                   cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ,
                                   cfg.B0, cfg.R0_major, 0.0, 0.0, cfg.m_mode, cfg.n_mode, 0.0)
    return B.astype(np.float64)


def run_full_orbit(env, pos, vel, q, m, t_end, rec_dt=REC_DT, dt_sub=DT_SUB):
    """Full-orbit run reduced to guiding centres every rec_dt.
    Returns t (n_rec,), gc (N, n_rec, 4) [R, phi, Z, v_par], lost_rec (N,)
    (first record index outside the LCFS, -1 if confined), wall time, and the
    particle's W and P_phi (N, n_rec) in the spline fields."""
    from kernels import boris_push_substeps_regather
    from gc_push import gc_from_particle
    cfg, field, eq = env["cfg"], env["field"], env["eq"]
    n_sub = int(round(rec_dt / dt_sub))
    n_rec = int(round(t_end / rec_dt)) + 1
    p32 = np.ascontiguousarray(pos, dtype=np.float32)
    v32 = np.ascontiguousarray(vel, dtype=np.float32)
    N = p32.shape[0]
    out = np.empty((N, n_rec, 4))
    lost = np.full(N, -1, dtype=np.int64)
    Wt = np.empty((N, n_rec))
    Pt = np.empty((N, n_rec))
    from physics_engine import compute_psi_normalised
    from gc_push import particle_invariants
    t0 = time.perf_counter()
    for k in range(n_rec):
        if k > 0:
            boris_push_substeps_regather(
                p32, v32, q, m, rec_dt, n_sub, env["E_R"], env["E_Z"], env["B_R"], env["B_Z"],
                cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ,
                cfg.B0, cfg.R0_major, 0.0, 0.0, cfg.m_mode, cfg.n_mode, 0.0)
        st, _ = gc_from_particle(field, p32, v32, q, m, B_at_particle=_reactor_B(env, p32))
        out[:, k] = st
        Wt[:, k], Pt[:, k] = particle_invariants(field, p32, v32, q, m)
        R = np.hypot(p32[:, 0].astype(np.float64), p32[:, 1].astype(np.float64))
        x = compute_psi_normalised(R, p32[:, 2].astype(np.float64), eq)
        newly = (lost < 0) & (x > 1.0)
        lost[newly] = k
    wall = time.perf_counter() - t0
    return np.arange(n_rec) * rec_dt, out, lost, wall, Wt, Pt


def run_gc(env, pos, vel, q, m, t_end, rec_target=REC_DT, fraction=None, init="invariants",
           mu_scale=1.0, W=None, P_phi=None):
    """GC run from the particles' first-order guiding centres. Returns
    t (n_rec,), gc (N, n_rec, 4), lost_rec (N,), wall, dt."""
    import gc_push as G
    field = env["field"]
    B_p = _reactor_B(env, np.ascontiguousarray(pos, dtype=np.float32))
    if init == "invariants":
        st, mu = G.gc_from_particle_invariants(field, pos, vel, q, m, B_at_particle=B_p,
                                               W=W, P_phi=P_phi)
    else:
        st, mu = G.gc_from_particle(field, pos, vel, q, m, B_at_particle=B_p)
    mu = mu * mu_scale
    v = np.sqrt(np.sum(np.asarray(vel) ** 2, axis=1)).max()
    dt = float(G.dt_transit_fraction(field, v, G.DEFAULT_TRANSIT_FRACTION if fraction is None else fraction))
    rec_every = max(1, int(round(rec_target / dt)))
    n_steps = int(np.ceil(t_end / dt / rec_every)) * rec_every
    t0 = time.perf_counter()
    out, alive, lost_step = G.push_record(field, st, mu, q, m, dt, n_steps, rec_every)
    wall = time.perf_counter() - t0
    lost = np.where(alive == 0, (lost_step + rec_every - 1) // rec_every, -1)
    t = np.arange(out.shape[1]) * dt * rec_every
    return t, out, lost, wall, dt


# =======================================================
# ORBIT METRICS
# =======================================================
def midplane_map(field, n=201):
    """psi_N -> outboard-midplane R of that flux surface (table)."""
    x = np.linspace(0.0, 1.0, n)
    Rm = np.array([field.R_axis] + [field.flux_surface_midplane(v)[1] for v in x[1:]])
    return x, Rm


MARGINAL_XI = 0.01       # |v_par| / v dead band around the trapped/passing boundary
SENSITIVE = 0.03         # period change under the invariant uncertainty that flags an orbit


def orbit_metrics(t, gc, lost_rec, field, mid, smooth_s=1.8e-7, v=None):
    """Per-orbit metrics from guiding-centre tracks (N, n_rec, 4).

    cls     : 0 passing, 1 trapped, 2 marginal, from s = v_par / v smoothed over
              smooth_s (180 ns -- far below every bounce period, enough to
              average the ~2% gyro-jitter of a full orbit's first-order v_par).
              No sign change: passing, or marginal if min |s| < MARGINAL_XI.
              Sign change: trapped, or marginal if one excursion (max s or
              -min s) stays below MARGINAL_XI while the other exceeds it -- an
              orbit grazing the trapped/passing separatrix, where the label is
              decided by |v_par| < 1% of v. (A deeply trapped orbit, both
              excursions small, stays trapped.)
    period  : mean interval between successive crossings of the horizontal ray
              Z = Z_c, R > R_c through the orbit's own centre (time-mean R, Z),
              in the direction that occurs most (one per period for both
              passing and trapped orbits).
    width   : separation of the two clusters of midplane (Z = Z_c) crossings in
              R -- the banana width of a trapped orbit (outer leg minus inner
              leg), the midplane diameter of a passing one -- each cluster
              averaged over every crossing in the run, which averages out the
              gyro-jitter of a full orbit's first-order guiding centre (~1 mm,
              too large for a max - min extent).
    lost    : the guiding centre reached psi_N > 1.
    """
    N = gc.shape[0]
    rec_dt = t[1] - t[0]
    w = max(1, int(round(smooth_s / rec_dt)))
    cls = np.full(N, -1, dtype=np.int64)
    period = np.full(N, np.nan)
    width = np.full(N, np.nan)
    nper = np.zeros(N, dtype=np.int64)
    lost = np.zeros(N, dtype=bool)
    for i in range(N):
        R = gc[i, :, 0]
        Z = gc[i, :, 2]
        x = field.psi_N(R, Z)
        out_k = np.nonzero(x > 1.0)[0]
        end = gc.shape[1]
        if lost_rec[i] >= 0:
            end = min(end, max(int(lost_rec[i]), 1))
        if out_k.size:
            lost[i] = True
            end = min(end, max(int(out_k[0]), 1))
        if end < 2 * w + 3:
            continue
        R, Z, x, tt = R[:end], Z[:end], x[:end], t[:end]
        vi = v[i] if v is not None else np.max(np.abs(gc[i, :end, 3]))
        sm = np.convolve(gc[i, :end, 3], np.ones(w) / w, mode="valid") / vi
        if np.any(np.sign(sm[1:]) != np.sign(sm[:-1])):
            lo, hi = min(sm.max(), -sm.min()), max(sm.max(), -sm.min())
            cls[i] = 2 if (lo < MARGINAL_XI <= hi) else 1
        else:
            cls[i] = 2 if np.min(np.abs(sm)) < MARGINAL_XI else 0
        Rc, Zc = R.mean(), Z.mean()
        dZ = Z - Zc
        right = R > Rc
        cross = right[:-1] & right[1:] & (np.sign(dZ[1:]) != np.sign(dZ[:-1])) & (dZ[:-1] != 0)
        idx = np.nonzero(cross)[0]
        allc = np.nonzero((np.sign(dZ[1:]) != np.sign(dZ[:-1])) & (dZ[:-1] != 0))[0]
        if allc.size >= 4:
            fa = dZ[allc] / (dZ[allc] - dZ[allc + 1])
            Ra = np.sort(R[allc] + fa * (R[allc + 1] - R[allc]))
            g = int(np.argmax(np.diff(Ra)))
            width[i] = Ra[g + 1:].mean() - Ra[:g + 1].mean()
        if idx.size:
            frac = dZ[idx] / (dZ[idx] - dZ[idx + 1])
            tc = tt[idx] + frac * (tt[idx + 1] - tt[idx])
            up = dZ[idx + 1] > 0
            use = tc[up] if up.sum() >= (~up).sum() else tc[~up]
            if use.size >= 2:
                period[i] = (use[-1] - use[0]) / (use.size - 1)
                nper[i] = use.size - 1
    return {"cls": cls, "trapped": cls == 1, "marginal": cls == 2, "period": period,
            "width": width, "n_periods": nper, "lost": lost}


def compare(env, n_alpha=50, n_beam=50, t_end=1.0e-3, seed=20261122, verbose=True, init="invariants"):
    """Run both pushers on both species; return per-species metrics and tracks."""
    import gc_push as G
    field = env["field"]
    ics = initial_conditions(env, n_alpha, n_beam, seed)
    mid = midplane_map(field)
    res = {}
    for name, (q, m) in SPECIES.items():
        pos, vel = ics[name]
        if pos.shape[0] == 0:
            continue
        t_fo, gc_fo, lost_fo, wall_fo, W_fo, P_fo = run_full_orbit(env, pos, vel, q, m, t_end)
        # The reactor's bilinear B_pol is not the curl of a single psi, so the full
        # orbit conserves W and P_phi only on average (P_phi oscillates by 0.2-0.6%
        # peak to peak, drifts ~1e-3 per ms). The GC is given the full orbit's
        # time-averaged W and P_phi -- its constants of motion -- at the particle's
        # first-order guiding centre (init="invariants"); "instant" uses t = 0 values.
        W_bar = P_bar = None
        if init == "invariants":
            end = np.where(lost_fo >= 0, np.maximum(lost_fo, 1), W_fo.shape[1])
            W_bar = np.array([W_fo[i, :end[i]].mean() for i in range(W_fo.shape[0])])
            P_bar = np.array([P_fo[i, :end[i]].mean() for i in range(P_fo.shape[0])])
        gc_init = "invariants" if init in ("invariants", "instant") else init
        t_gc, gc_gc, lost_gc, wall_gc, dt_gc = run_gc(env, pos, vel, q, m, t_end, init=gc_init,
                                                      W=W_bar, P_phi=P_bar)
        # Losses are compared on one footing, the guiding centre leaving the LCFS
        # (orbit_metrics); the reactor's particle-position criterion (lost_fo) is
        # kept to count finite-Larmor-radius losses the GC model cannot see.
        v = np.sqrt(np.sum(vel**2, axis=1))
        m_fo = orbit_metrics(t_fo, gc_fo, np.full(lost_fo.shape, -1), field, mid, v=v)
        m_gc = orbit_metrics(t_gc, gc_gc, lost_gc, field, mid, v=v)
        m_fo["lost_particle"] = lost_fo >= 0
        # Invariant sensitivity. The GC is matched to invariants known only to
        # ~1e-3 |P_phi| (the full orbit's own P_phi fluctuation in the reactor's
        # bilinear field) and ~1% in mu (first-order guiding-centre theory, O(rho/L)).
        # Each orbit is rerun at P_phi +- 1e-3 |P_phi| and mu x (1 +- 1%); an orbit
        # whose period moves by more than SENSITIVE (3%) under either is near a
        # separatrix or in a streaming/drift cancellation and cannot be expected to agree.
        W_ref, P_ref = (W_bar, P_bar) if W_bar is not None else G.particle_invariants(field, pos, vel, q, m)
        dT = np.zeros(pos.shape[0])
        for kw in ({"P_phi": P_ref + 1e-3 * np.abs(P_ref)}, {"P_phi": P_ref - 1e-3 * np.abs(P_ref)},
                   {"mu_scale": 1.01}, {"mu_scale": 0.99}):
            args = {"W": W_ref, "P_phi": P_ref}
            args.update(kw)
            ts, gs, ls, _, _ = run_gc(env, pos, vel, q, m, t_end, init="invariants", **args)
            dT = np.maximum(dT, np.abs(orbit_metrics(ts, gs, ls, field, mid, v=v)["period"] / m_gc["period"] - 1.0))
        m_gc["sensitivity"] = np.where(np.isfinite(dT), dT, np.inf)
        m_gc["sensitive"] = m_gc["sensitivity"] > SENSITIVE
        res[name] = {"fo": m_fo, "gc": m_gc, "t_fo": t_fo, "t_gc": t_gc, "W_fo": W_fo, "P_fo": P_fo,
                     "gc_fo": gc_fo, "gc_gc": gc_gc, "wall_fo": wall_fo, "wall_gc": wall_gc,
                     "dt_gc": dt_gc, "pos": pos, "vel": vel}
        if verbose:
            s = summarize(res[name])
            print(f"[{name}] full orbit {wall_fo:.1f} s, GC {wall_gc:.1f} s (dt_GC = {dt_gc:.3e} s); "
                  f"class agree {s['class_agree']}/{s['n_decided']} (+{s['n_marginal']} marginal), "
                  f"lost FO/GC {s['n_lost_fo']}/{s['n_lost_gc']} (particle criterion {s['n_lost_fo_particle']}), "
                  f"period |rel| median {s['period_med']:.2%} max {s['period_max']:.2%} "
                  f"[{s['n_sensitive']} invariant-sensitive; without them max {s['period_ins_max']:.2%}], "
                  f"banana |rel| median {s['banana_med']:.2%} max {s['banana_max']:.2%} (n={s['n_banana']}), "
                  f"orbit width |rel| median {s['width_med']:.2%} max {s['width_max']:.2%}")
    return res


def summarize(r):
    """Agreement statistics for one species' result."""
    fo, gc = r["fo"], r["gc"]
    n = fo["trapped"].size
    both_conf = ~fo["lost"] & ~gc["lost"]
    marginal = fo["marginal"] | gc["marginal"]
    lost_agree = fo["lost"] == gc["lost"]
    decided = both_conf & ~marginal
    class_agree = int(np.sum(decided & (fo["cls"] == gc["cls"])))
    same = (fo["cls"] == gc["cls"]) | marginal
    ok = both_conf & same & np.isfinite(fo["period"]) & np.isfinite(gc["period"])
    prel = gc["period"][ok] / fo["period"][ok] - 1.0
    wok = both_conf & same & np.isfinite(fo["width"]) & np.isfinite(gc["width"])
    wrel = gc["width"][wok] / fo["width"][wok] - 1.0
    bok = wok & fo["trapped"]
    brel = gc["width"][bok] / fo["width"][bok] - 1.0

    def stats(a):
        a = np.abs(a)
        if not a.size:
            return np.nan, np.nan, np.nan
        return float(np.median(a)), float(np.percentile(a, 90)), float(np.max(a))

    p_med, p_90, p_max = stats(prel)
    ins = ~gc["sensitive"][ok]
    q_med, q_90, q_max = stats(prel[ins])
    winsens = ~gc["sensitive"][wok]
    x_med, x_90, x_max = stats(wrel[winsens])
    w_med, w_90, w_max = stats(wrel)
    b_med, b_90, b_max = stats(brel)
    return {"n": n, "class_agree": class_agree, "n_decided": int(decided.sum()),
            "n_marginal": int((marginal & both_conf).sum()), "lost_agree": int(lost_agree.sum()), "n_trapped_fo": int(fo["trapped"].sum()),
            "n_trapped_gc": int(gc["trapped"].sum()), "n_lost_fo": int(fo["lost"].sum()),
            "n_lost_gc": int(gc["lost"].sum()), "n_lost_fo_particle": int(fo["lost_particle"].sum()),
            "n_period": int(ok.sum()), "n_width": int(wok.sum()), "n_banana": int(bok.sum()),
            "period_rel": prel, "width_rel": wrel, "banana_rel": brel,
            "period_med": p_med, "period_p90": p_90, "period_max": p_max,
            "width_med": w_med, "width_p90": w_90, "width_max": w_max,
            "banana_med": b_med, "banana_p90": b_90, "banana_max": b_max,
            "period_mean_signed": float(np.mean(prel)) if prel.size else np.nan,
            "n_sensitive": int(np.sum(gc["sensitive"] & ok)),
            "period_ins_med": q_med, "period_ins_p90": q_90, "period_ins_max": q_max,
            "width_ins_med": x_med, "width_ins_p90": x_90, "width_ins_max": x_max}


# =======================================================
# PLOT AND ENTRY POINT
# =======================================================
C_FO, C_GC = "#2a78d6", "#eb6834"          # categorical slots 1, 2
C_SPECIES = {"alpha": "#1baf7a", "beam": "#e87ba4"}   # slots 3, 5
MARK_SPECIES = {"alpha": "o", "beam": "s"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def _pick(r, want_trapped):
    fo, gc = r["fo"], r["gc"]
    ok = (~fo["lost"] & ~gc["lost"] & ~fo["marginal"] & ~gc["marginal"]
          & (fo["trapped"] == want_trapped) & (gc["trapped"] == want_trapped)
          & ~gc["sensitive"] & np.isfinite(fo["period"]))
    idx = np.nonzero(ok)[0]
    if not idx.size:
        return None
    # the widest orbit of its class shows the drift structure best
    return int(idx[np.argmax(fo["width"][idx])])


def plot(res, env, path, t_end):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    field = env["field"]
    Rw, Zw = field.flux_surface_contour(1.0, n_theta=361)
    fig = plt.figure(figsize=(17, 9.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.28)
    panels = [("alpha", True), ("alpha", False), ("beam", True), ("beam", False)]
    for k, (name, trapped) in enumerate(panels):
        ax = fig.add_subplot(gs[0, k])
        ax.plot(Rw, Zw, color=MUTED, lw=1.0)
        ax.plot([field.R_axis], [field.Z_axis], "+", color=MUTED, ms=8)
        r = res[name]
        i = _pick(r, trapped)
        label = f"{name}, {'trapped' if trapped else 'passing'}"
        if i is not None:
            # one full period of each, starting at the run's start
            for tr, t, col, ls, lw, lab in ((r["gc_fo"], r["t_fo"], C_FO, "-", 2.0, "full orbit (GC of particle)"),
                                            (r["gc_gc"], r["t_gc"], C_GC, "--", 2.0, "guiding centre")):
                n = np.searchsorted(t, 1.05 * r["fo"]["period"][i])
                ax.plot(tr[i, :n, 0], tr[i, :n, 2], color=col, ls=ls, lw=lw, label=lab)
            pf, pg = r["fo"]["period"][i], r["gc"]["period"][i]
            wf, wg = r["fo"]["width"][i], r["gc"]["width"][i]
            label += (f"\nperiod {pf*1e6:.3f} / {pg*1e6:.3f} us (FO / GC)"
                      f"\nwidth {wf*100:.2f} / {wg*100:.2f} cm")
        ax.set_title(label, fontsize=9.5, color=INK)
        ax.set_aspect("equal")
        ax.set_xlabel("R [m]", color=MUTED)
        if k == 0:
            ax.set_ylabel("Z [m]", color=MUTED)
            ax.legend(loc="upper left", fontsize=8.5, frameon=False)
        ax.grid(True, color=GRID, lw=0.6)
        for s in ax.spines.values():
            s.set_color(GRID)

    for col, (key, xlabel, ylabel) in enumerate((("period", "full-orbit period [us]", "GC / full orbit - 1  [%]"),
                                                 ("width", "full-orbit radial width [cm]", "GC / full orbit - 1  [%]"))):
        ax = fig.add_subplot(gs[1, col])
        for name, r in res.items():
            fo, gc = r["fo"], r["gc"]
            ok = (~fo["lost"] & ~gc["lost"] & ((fo["cls"] == gc["cls"]) | fo["marginal"] | gc["marginal"])
                  & np.isfinite(fo[key]) & np.isfinite(gc[key]))
            x = fo[key] * (1e6 if key == "period" else 100.0)
            y = 100.0 * (gc[key] / fo[key] - 1.0)
            sen = gc["sensitive"]
            ax.scatter(x[ok & ~sen], y[ok & ~sen], s=34, marker=MARK_SPECIES[name], color=C_SPECIES[name],
                       edgecolor="white", linewidth=1.0, label=f"{name} ({(ok & ~sen).sum()})", zorder=3)
            if np.any(ok & sen):
                ax.scatter(x[ok & sen], y[ok & sen], s=40, marker=MARK_SPECIES[name], facecolor="none",
                           edgecolor=C_SPECIES[name], linewidth=1.5,
                           label=f"{name}, invariant-sensitive ({(ok & sen).sum()})", zorder=3)
        ax.axhline(0.0, color=MUTED, lw=1.0)
        for yl in (-3, 3):
            ax.axhline(yl, color=GRID, lw=1.0, ls=":")
        ax.set_xlabel(xlabel, color=MUTED)
        ax.set_ylabel(ylabel, color=MUTED)
        ax.set_title("bounce / transit period" if key == "period" else "orbit width (banana width if trapped)",
                     fontsize=11, color=INK)
        ax.legend(fontsize=8.5, frameon=False)
        ax.grid(True, color=GRID, lw=0.6)
        for s in ax.spines.values():
            s.set_color(GRID)

    ax = fig.add_subplot(gs[1, 2:])
    ax.axis("off")
    rows = [f"{'':24s}{'alpha':>18s}{'beam D':>18s}"]

    def row(label, fmt):
        vals = [fmt(summarize(res[n])) if n in res else "-" for n in ("alpha", "beam")]
        rows.append(f"{label:24s}{vals[0]:>18s}{vals[1]:>18s}")

    row("orbits", lambda s: f"{s['n']}")
    row("trapped/passing agree", lambda s: f"{s['class_agree']}/{s['n_decided']}")
    row("marginal (|v_par|<1%v)", lambda s: f"{s['n_marginal']}")
    row("trapped FO / GC", lambda s: f"{s['n_trapped_fo']} / {s['n_trapped_gc']}")
    row("lost FO / GC (GC crit.)", lambda s: f"{s['n_lost_fo']} / {s['n_lost_gc']}")
    row("lost FO (particle crit.)", lambda s: f"{s['n_lost_fo_particle']}")
    row("period |rel| median", lambda s: f"{s['period_med']:.2%}")
    row("period |rel| 90th pct", lambda s: f"{s['period_p90']:.2%}")
    row("period |rel| max", lambda s: f"{s['period_max']:.2%}")
    row("period mean signed", lambda s: f"{s['period_mean_signed']:+.2%}")
    row("invariant-sensitive", lambda s: f"{s['n_sensitive']}")
    row("  period max without", lambda s: f"{s['period_ins_max']:.2%}")
    row("banana |rel| median", lambda s: f"{s['banana_med']:.2%} (n={s['n_banana']})")
    row("banana |rel| max", lambda s: f"{s['banana_max']:.2%}")
    row("width |rel| median", lambda s: f"{s['width_med']:.2%}")
    row("width |rel| max", lambda s: f"{s['width_max']:.2%}")
    row("wall FO / GC [s]", lambda s: "")
    rows[-1] = (f"{'wall FO / GC [s]':24s}"
                + "".join(f"{res[n]['wall_fo']:>9.1f} / {res[n]['wall_gc']:<6.1f}" for n in ("alpha", "beam")))
    ax.text(0.0, 1.02, "\n".join(rows), family="monospace", fontsize=9, color=INK, va="top")
    fig.text(0.515, 0.02, f"{t_end*1e3:g} ms per orbit. Full orbit: kernels.boris_push_substeps_regather, 0.1 ns sub-steps,\n"
             "reactor bilinear B and E, ripple off; reduced to its first-order guiding centre every 20 ns.\n"
             f"GC: RK4, dt = 1/200 of the toroidal transit at birth speed (alpha {res['alpha']['dt_gc']*1e9:.2f} ns, "
             f"beam {res['beam']['dt_gc']*1e9:.2f} ns),\nspline fields, started with the full orbit's time-averaged "
             "energy and P_phi.\nHollow markers / 'invariant-sensitive': the GC period moves > 3% for\n"
             "dP_phi = 1e-3 |P_phi| or dmu = 1% (separatrix orbits, near-axis drift cancellation).",
             fontsize=8.5, color=MUTED, va="bottom", ha="left")
    fig.suptitle("Guiding centre vs full orbit: 3.5 MeV alphas and 1 MeV beam deuterons", fontsize=13, color=INK)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_npz(res, path):
    out = {}
    for name, r in res.items():
        for side in ("fo", "gc"):
            for k, v in r[side].items():
                out[f"{name}_{side}_{k}"] = np.asarray(v)
        out[f"{name}_pos0"] = r["pos"]
        out[f"{name}_vel0"] = r["vel"]
        out[f"{name}_dt_gc"] = np.float64(r["dt_gc"])
        out[f"{name}_wall_fo"] = np.float64(r["wall_fo"])
        out[f"{name}_wall_gc"] = np.float64(r["wall_gc"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)


def main(n_alpha=50, n_beam=50, t_end=1.0e-3):
    t0 = time.perf_counter()
    env = setup()
    res = compare(env, n_alpha, n_beam, t_end)
    plot(res, env, os.path.join(REPO, "docs", "images", "gc_vs_full_orbit.png"), t_end)
    save_npz(res, os.path.join(REPO, "data", "gc_vs_full_orbit.npz"))
    print(f"wrote docs/images/gc_vs_full_orbit.png and data/gc_vs_full_orbit.npz "
          f"({time.perf_counter() - t0:.0f} s wall)")
    return res


if __name__ == "__main__":
    main()
