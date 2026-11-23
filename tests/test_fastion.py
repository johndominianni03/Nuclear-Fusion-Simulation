#!/usr/bin/env python3
"""Known-answer tests for the Tier 3 fast-ion module: smooth GC fields and the
background plasma (Phase 1), the guiding-centre pusher (Phase 2), the Monte
Carlo Fokker-Planck operator (Phase 3) and the slowing-down runs (Phase 4).

    venv/bin/python -m pytest tests/test_fastion.py -q
    venv/bin/python tests/test_fastion.py          # same tests, no pytest needed

Every tolerance below is stated next to the value it was measured at.
"""
import os
import sys

os.environ.setdefault("FUSION_HEADLESS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import functools                                   # noqa: E402

import numpy as np                                 # noqa: E402


# =======================================================
# SHARED FIXTURES
# =======================================================
@functools.lru_cache(maxsize=None)
def _gc():
    """(GCField, Background) of the production equilibrium and scenario."""
    from fields_gc import build_gc_field
    return build_gc_field()


@functools.lru_cache(maxsize=None)
def _reactor_B_grids():
    """The full-orbit run's resampled poloidal-field grids (50 x 50)."""
    from physics_engine import compute_poloidal_field_grids
    field, _ = _gc()
    cfg, eq = field.cfg, field.eq
    B_R, B_Z, scale, _ = compute_poloidal_field_grids(
        eq, cfg.B0, R0=cfg.R0_major, a_minor=0.3, q_target=cfg.Q_SAFETY_TARGET,
        dst_R_min=cfg.R_min, dst_R_max=cfg.R_max, dst_Z_min=cfg.Z_min,
        dst_Z_max=cfg.Z_max, dst_nR=cfg.nR, dst_nZ=cfg.nZ)
    return B_R, B_Z, scale


def _field_grid_nodes(cfg):
    R = np.linspace(cfg.R_min, cfg.R_max, cfg.nR)
    Z = np.linspace(cfg.Z_min, cfg.Z_max, cfg.nZ)
    return np.meshgrid(R, Z, indexing="ij")


def _interior_points(field, n, rng, psiN_max=0.95):
    """n random (R, Z) points inside the psi_N = psiN_max surface."""
    R_out, Z_out = [], []
    while sum(len(r) for r in R_out) < n:
        R = rng.uniform(0.65, 1.45, 4 * n)
        Z = rng.uniform(-0.45, 0.45, 4 * n)
        ok = field.psi_N(R, Z) < psiN_max
        R_out.append(R[ok])
        Z_out.append(Z[ok])
    return np.concatenate(R_out)[:n], np.concatenate(Z_out)[:n]


# =======================================================
# PHASE 1 -- SMOOTH FIELDS AND BACKGROUND PLASMA
# =======================================================
def test_gc_spline_reproduces_reactor_psi_at_nodes():
    """The bicubic spline tables interpolate the reactor's q-rescaled psi:
    at every 60 x 60 node psi_spline = scale * psi_GS to 1e-6 of psi_axis
    (measured 4e-14), and the q-rescale is the full-orbit run's own factor."""
    field, _ = _gc()
    _, _, scale = _reactor_B_grids()
    assert scale == field.scale
    RR, ZZ = np.meshgrid(field.R_nodes, field.Z_nodes, indexing="ij")
    err = np.max(np.abs(field.psi(RR, ZZ) - field.eq.psi_grid * scale)) / field.psi_axis
    assert err < 1e-6, f"node psi error {err:.2e} of psi_axis"


def test_gc_spline_B_matches_full_orbit_B_at_field_grid_nodes():
    """Spline B against the full-orbit run's bilinear B at the 50 x 50 field-grid
    nodes inside the last closed flux surface.

    B_phi = B0 R0 / R is analytic in both and agrees to rounding. The poloidal
    field differs by the full-orbit run's discretisation -- central differences of
    the 60 x 60 psi, then bilinear resampling -- not by the spline: the bound is
    max |B_spline - B_full| / |B| < 1e-3 (measured 4.2e-4), and the gap must
    fall as h^2 (measured rms-gap ratios 3.98x for 60 -> 119 nodes, 4.02x for
    119 -> 237). Taking the spline as the exact psi and pushing it through the
    full-orbit pipeline (np.gradient, 1/R, physics_engine.resample_grid_to) from
    a grid with half the spacing must shrink the gap by 4x (accepted 3x-5.5x).
    The 60 x 60 pipeline replica must reproduce the reactor's grids to 1e-12.
    """
    from physics_engine import resample_grid_to
    field, _ = _gc()
    cfg = field.cfg
    B_R_full, B_Z_full, _ = _reactor_B_grids()
    RR, ZZ = _field_grid_nodes(cfg)
    inside = field.psi_N(RR, ZZ) <= 1.0
    f = field.fields(RR, ZZ)
    B_phi_full = cfg.B0 * cfg.R0_major / RR
    assert np.max(np.abs(f["Bphi"] - B_phi_full) / B_phi_full) < 1e-14
    B_full = np.sqrt(B_R_full**2 + B_Z_full**2 + B_phi_full**2)
    gap = np.hypot(f["BR"] - B_R_full, f["BZ"] - B_Z_full)
    rel = np.max(gap[inside] / B_full[inside])
    assert rel < 1e-3, f"max |dB|/|B| = {rel:.2e}"

    def pipeline(n_src):
        """The full-orbit B_pol construction from n_src x n_src samples of the
        spline psi, resampled to the field grid."""
        R = np.linspace(field.R_nodes[0], field.R_nodes[-1], n_src)
        Z = np.linspace(field.Z_nodes[0], field.Z_nodes[-1], n_src)
        Rg, Zg = np.meshgrid(R, Z, indexing="ij")
        psi = field.psi(Rg, Zg)
        dpsi_dR, dpsi_dZ = np.gradient(psi, R[1] - R[0], Z[1] - Z[0])
        src = (R[0], R[-1], Z[0], Z[-1], n_src, n_src)
        dst = (cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ)
        BR = resample_grid_to(np.ascontiguousarray(-dpsi_dZ / Rg), *src, *dst)
        BZ = resample_grid_to(np.ascontiguousarray(dpsi_dR / Rg), *src, *dst)
        return BR, BZ

    BR60, BZ60 = pipeline(field.nR)
    assert np.max(np.abs(BR60 - B_R_full)) < 1e-12 * np.max(np.abs(B_R_full))
    assert np.max(np.abs(BZ60 - B_Z_full)) < 1e-12 * np.max(np.abs(B_Z_full))
    BR119, BZ119 = pipeline(2 * field.nR - 1)
    # Interior nodes: np.gradient is one-sided (first order) on the box edge.
    interior = inside.copy()
    gap60 = np.hypot(f["BR"] - BR60, f["BZ"] - BZ60)[interior]
    gap119 = np.hypot(f["BR"] - BR119, f["BZ"] - BZ119)[interior]
    ratio = np.sqrt(np.mean(gap60**2) / np.mean(gap119**2))
    assert 3.0 < ratio < 5.5, f"halving h shrank the rms gap by {ratio:.2f}x, not ~4x"


def test_gc_field_is_divergence_free():
    """div B = (1/R) d(R B_R)/dR + dB_Z/dZ of the spline field, by central
    differences (h = 1e-5 m) at 2,000 interior points: |div B| a / |B_pol|
    < 1e-6 with a = 0.3 m (measured 1.3e-7, the O(h^2) and knot-crossing error of
    the difference itself; the full-orbit run's bilinear grids give 6e-3)."""
    field, _ = _gc()
    rng = np.random.default_rng(11)
    R, Z = _interior_points(field, 2000, rng)
    h = 1e-5
    fRp, fRm = field.fields(R + h, Z), field.fields(R - h, Z)
    fZp, fZm = field.fields(R, Z + h), field.fields(R, Z - h)
    div = (((R + h) * fRp["BR"] - (R - h) * fRm["BR"]) / (2 * h) / R
           + (fZp["BZ"] - fZm["BZ"]) / (2 * h))
    f0 = field.fields(R, Z)
    Bpol = np.hypot(f0["BR"], f0["BZ"])
    Bpol_ref = np.median(Bpol)
    worst = np.max(np.abs(div)) * 0.3 / Bpol_ref
    assert worst < 1e-6, f"|div B| a / B_pol = {worst:.2e}"


def test_gc_curl_b_and_grad_B_match_finite_differences():
    """The analytic curl b and grad|B| from the spline second derivatives against
    central differences of b and |B| (h = 1e-6 m) at 2,000 interior points:
    max error < 1e-5 of |curl b| ~ 1/R and of max |grad B| (measured 1.7e-10 and
    1.4e-10)."""
    field, _ = _gc()
    rng = np.random.default_rng(12)
    R, Z = _interior_points(field, 2000, rng)
    h = 1e-6

    def b(R, Z):
        f = field.fields(R, Z)
        return f["BR"] / f["Bmag"], f["Bphi"] / f["Bmag"], f["BZ"] / f["Bmag"], f["Bmag"]

    bRp, bPp, bZp, BRp = b(R + h, Z)
    bRm, bPm, bZm, BRm = b(R - h, Z)
    bRu, bPu, bZu, BZu = b(R, Z + h)
    bRd, bPd, bZd, BZd = b(R, Z - h)
    curl_R = -(bPu - bPd) / (2 * h)
    curl_phi = (bRu - bRd) / (2 * h) - (bZp - bZm) / (2 * h)
    curl_Z = ((R + h) * bPp - (R - h) * bPm) / (2 * h) / R
    f = field.fields(R, Z)
    scale = np.median(np.sqrt(f["curlb_R"]**2 + f["curlb_phi"]**2 + f["curlb_Z"]**2))
    err = max(np.max(np.abs(curl_R - f["curlb_R"])), np.max(np.abs(curl_phi - f["curlb_phi"])),
              np.max(np.abs(curl_Z - f["curlb_Z"]))) / scale
    assert err < 1e-5, f"curl b error {err:.2e} of |curl b|"
    gR = (BRp - BRm) / (2 * h)
    gZ = (BZu - BZd) / (2 * h)
    gscale = np.max(np.hypot(f["gradB_R"], f["gradB_Z"]))
    gerr = max(np.max(np.abs(gR - f["gradB_R"])), np.max(np.abs(gZ - f["gradB_Z"]))) / gscale
    assert gerr < 1e-5, f"grad|B| error {gerr:.2e}"


def test_background_integrates_to_N_AVG_times_volume():
    """The background ion density n_D + n_T integrated over the spline's psi_N <= 1
    volume (1200^2 revolved mesh) equals N_AVG_TARGET * PLASMA_VOLUME_M3 within
    0.5% (measured +0.15%: n0 is the reactor's, computed with the bilinear psi_N,
    and the spline LCFS encloses 0.08% more volume than the Pappus contour).
    Quasi-neutrality n_e = n_D + n_T and the 50/50 D-T split hold pointwise; T is
    25 keV on axis for every species."""
    field, bg = _gc()
    cfg = field.cfg
    n = 1200
    dR = (field.R_nodes[-1] - field.R_nodes[0]) / n
    dZ = (field.Z_nodes[-1] - field.Z_nodes[0]) / n
    Rc = field.R_nodes[0] + (np.arange(n) + 0.5) * dR
    Zc = field.Z_nodes[0] + (np.arange(n) + 0.5) * dZ
    RR, ZZ = np.meshgrid(Rc, Zc, indexing="ij")
    x = field.psi_N(RR, ZZ)
    inside = x <= 1.0
    dV = 2.0 * np.pi * RR * dR * dZ
    n_i = bg.n("D", x) + bg.n("T", x)
    N = float(np.sum(n_i[inside] * dV[inside]))
    expect = cfg.N_AVG_TARGET * cfg.PLASMA_VOLUME_M3
    assert abs(N / expect - 1.0) < 5e-3, f"int n_i dV / (N_AVG V) = {N / expect:.5f}"
    assert np.allclose(bg.n("e", x), n_i, rtol=1e-15)
    assert np.allclose(bg.n("D", x), bg.n("T", x), rtol=1e-15)
    assert bg.T_keV(0.0) == cfg.T_thermal_keV


def test_background_tables_match_profiles():
    """The njit lookup (background.background_at) returns the analytic profiles
    to 1e-12 for every species at psi_N in [0, 1], and zero outside."""
    from background import background_at, SPECIES
    from constants import KEV_J
    _, bg = _gc()
    n_out = np.empty(3)
    T_out = np.empty(3)
    for x in np.linspace(0.0, 1.0, 37):
        background_at(x, bg.dens, bg.temp, bg.x0, bg.dx, n_out, T_out)
        for s, name in enumerate(SPECIES):
            assert abs(n_out[s] - bg.n(name, x)) <= 1e-12 * bg.n0
        assert np.all(np.abs(T_out - bg.T_keV(x) * KEV_J) <= 1e-12 * bg.T0_keV * KEV_J)
    background_at(1.01, bg.dens, bg.temp, bg.x0, bg.dx, n_out, T_out)
    assert np.all(n_out == 0.0) and np.all(T_out == 0.0)


def test_gc_potential_is_tier1_force_balance():
    """Phi(psi_N) of the GC field is Tier 1's e n_i E = grad p_i for the
    background profiles. Tier 1's own solver
    (physics_engine.compute_force_balance_potential) on the exact shell profile
    gives dPhi/dpsi_N = 2 T0 / e within 2% across its support (it smooths), and
    the GC field's Phi = -(2 T0 / e)(1 - psi_N) is -50 kV on axis, with
    E = -grad Phi pointing inward (towards the axis)."""
    from physics_engine import compute_force_balance_potential
    from constants import KEV_J, E_CHARGE
    field, bg = _gc()
    prof = bg.shell_profile(field.eq)
    edges = prof["psiN_edges"]
    x = prof["psiN_centers"]
    prof["count"] = np.zeros((4, len(x)), dtype=np.int64)
    prof["count"][[0, 3]] = 10**6
    prof["p"] = np.zeros((4, len(x)))
    prof["p"][0] = prof["n"][0] * prof["T_keV"][0] * KEV_J
    prof["p"][3] = prof["n"][3] * prof["T_keV"][3] * KEV_J
    pot = compute_force_balance_potential(prof, species=(0, 3))
    g = 2.0 * bg.T0_keV * KEV_J / E_CHARGE
    sel = (pot["x"] > 0.05) & (pot["x"] < pot["x_end"] - 0.05)
    assert np.max(np.abs(pot["dPhi_dx"][sel] / g - 1.0)) < 0.02
    assert len(edges) == len(x) + 1
    f = field.fields(np.array([field.R_axis]), np.array([field.Z_axis]))
    psiN_ax = f["psiN"][0]
    assert abs(f["Phi"][0] - (-g * (1.0 - psiN_ax))) < 1e-9 * g
    # Outboard midplane point: E_R < 0 (inward)
    R_in, R_out = field.flux_surface_midplane(0.5)
    fo = field.fields(np.array([R_out]), np.array([field.Z_axis]))
    assert fo["E_R"][0] < 0.0


# =======================================================
# PHASE 2 -- GUIDING-CENTRE PUSHER
# =======================================================
def test_gc_conserves_energy_and_canonical_momentum():
    """3.5 MeV alphas launched at the outboard midplane of psi_N = 0.2 and 0.5
    with pitches +0.1 ... -0.9 (trapped and co-/counter-passing), 1e5 RK4 steps
    at dt = 1/200 of the toroidal transit (2.79 ns, 0.28 ms in all). For every
    confined orbit
        |W - W0| / W0 < 1e-6            (measured max 2.4e-8)
        |P_phi - P0| / |P0| < 1e-6      (measured max 1.3e-7)
    with W = m v_par^2/2 + mu B + q Phi (Phi = -50 kV on axis) and
    P_phi = m v_par b_phi R + q psi. The other sign, m v_par b_phi R - q psi,
    is not an invariant in this code's convention: it varies by > 5%.
    The sample holds both trapped and passing orbits."""
    import gc_push as G
    from constants import M_ALPHA, E_CHARGE, MEV_J
    field, _ = _gc()
    q, m = 2.0 * E_CHARGE, M_ALPHA
    R, xi = [], []
    for x in (0.2, 0.5):
        R_out = field.flux_surface_midplane(x)[1]
        for p in (0.1, 0.3, 0.7, 0.95, -0.4, -0.9):
            R.append(R_out)
            xi.append(p)
    state, mu, v = G.gc_state_from_energy_pitch(field, np.array(R), field.Z_axis, 3.5 * MEV_J,
                                                np.array(xi), m)
    dt = G.dt_transit_fraction(field, v)
    out, alive, _ = G.push_record(field, state, mu, q, m, dt, 100000, 100)
    n = out.shape[1]
    W, P = G.invariants(field, out.reshape(-1, 4), np.repeat(mu, n), q, m)
    W, P = W.reshape(-1, n), P.reshape(-1, n)
    f = field.fields(out[:, :, 0].ravel(), out[:, :, 2].ravel())
    P_wrong = (m * out[:, :, 3].ravel() * f["Bphi"] / f["Bmag"] * out[:, :, 0].ravel()
               - q * f["psi"]).reshape(-1, n)
    trapped = np.any(np.sign(out[:, 1:, 3]) != np.sign(out[:, :-1, 3]), axis=1)
    ok = alive == 1
    assert ok.sum() >= 10
    assert np.any(trapped & ok) and np.any(~trapped & ok)
    dW = np.max(np.abs(W[ok] / W[ok, :1] - 1.0))
    dP = np.max(np.abs(P[ok] - P[ok, :1]) / np.abs(P[ok, :1]))
    assert dW < 1e-6, f"energy error {dW:.2e}"
    assert dP < 1e-6, f"P_phi error {dP:.2e}"
    assert np.min(np.ptp(P_wrong[ok], axis=1) / np.abs(P_wrong[ok, 0])) > 0.05


def _trapped_passing_boundary(field, psiN, E_J, q, m):
    """xi_c at the |B| minimum of the psi_N surface by bisection on GC orbits
    (64 pitches per round, 3 rounds: resolution 4e-6). Returns
    (xi_c, sqrt(1 - Bmin/Bmax), midplane-eps formula value)."""
    import gc_push as G
    Rc, Zc = field.flux_surface_contour(psiN, 1441)
    B = field.fields(Rc, Zc)["Bmag"]
    k = int(np.argmin(B))
    xi_exact = np.sqrt(1.0 - B.min() / B.max())
    R_in, R_out = field.flux_surface_midplane(psiN)
    eps = (R_out - R_in) / (R_out + R_in)
    v = np.sqrt(2.0 * E_J / m)
    dt = float(G.dt_transit_fraction(field, v))
    lo, hi = 0.0, 1.0
    for _ in range(3):
        xis = np.linspace(lo, hi, 64)
        st, mu, _ = G.gc_state_from_energy_pitch(field, np.full(64, Rc[k]), np.full(64, Zc[k]),
                                                 E_J, xis, m)
        out, alive, _ = G.push_record(field, st, mu, q, m, dt, 12 * 200, 1)   # 12 toroidal transits
        assert np.all(alive == 1)
        vp = out[:, :, 3]
        trapped = np.any(np.sign(vp[:, 1:]) != np.sign(vp[:, :-1]), axis=1)
        j = int(np.nonzero(~trapped)[0][0])
        assert np.all(trapped[:j]) and np.all(~trapped[j:]), "classification not monotone in xi"
        lo, hi = xis[j - 1], xis[j]
    return 0.5 * (lo + hi), xi_exact, np.sqrt(2.0 * eps / (1.0 + eps))


def test_trapped_passing_boundary_matches_mirror_ratio():
    """The GC trapped/passing boundary at three surfaces against the exact
    magnetic-mirror boundary xi_c = sqrt(1 - Bmin/Bmax), Bmin and Bmax the
    extremes of the spline |B| on the surface, for 1 keV deuterons launched at
    the Bmin point with E off (Phi = 0) and a ~3 mm banana width: within 5%
    (measured -0.60%, -0.36%, -0.28% at psi_N = 0.2, 0.5, 0.8 -- finite orbit
    width). The large-aspect-ratio sqrt(2 eps/(1+eps)) with
    eps = (R_out - R_in)/(R_out + R_in) from the midplane extent gives 0.4895,
    0.6058, 0.6775 against exact 0.4850, 0.5940, 0.6629 (+0.9% to +2.2%)."""
    from fields_gc import GCField
    from constants import M_DEUTERON, E_CHARGE, KEV_J
    f0, _ = _gc()
    field = GCField(f0.cfg, f0.eq, f0.scale)        # no potential
    for x in (0.2, 0.5, 0.8):
        xi_c, xi_exact, xi_eps = _trapped_passing_boundary(field, x, KEV_J, E_CHARGE, M_DEUTERON)
        rel = xi_c / xi_exact - 1.0
        assert abs(rel) < 0.05, f"psi_N={x}: xi_c {xi_c:.4f} vs {xi_exact:.4f} ({rel:+.2%})"
        assert abs(rel) < 0.01, f"psi_N={x}: {rel:+.2%} (finite-orbit-width bias expected < 1%)"


def test_gc_matches_full_orbit_reduced_benchmark():
    """Reduced form of tools/gc_vs_full_orbit.py (whose 1 ms, 50 + 50 run is
    the full benchmark): 10 alphas and 10 beam ions for 0.2 ms through the
    reactor's full-orbit Boris pusher and the GC pusher, the GC given the full
    orbit's time-averaged energy and P_phi.

    Every orbit outside the 1% separatrix dead band gets the same trapped/passing
    label and the same loss outcome. Orbits whose GC period moves by > 3% when
    P_phi shifts by 1e-3 |P_phi| (the full orbit's own P_phi fluctuation) or mu by
    1% (first-order theory) are invariant-sensitive -- separatrix orbits and
    near-axis streaming/drift cancellations -- and are counted, not compared; at
    most 30% of the sample (measured 1 alpha, 2 beam). Over the rest:
        period  median < 1.5%, max < 4%  (measured 0.56% / 1.31% alpha, 0.43% / 1.16% beam)
        width   median < 1.5%, max < 7%  (measured 0.46% / 4.18% alpha, 0.27% / 0.96% beam)
    The full 1 ms benchmark measures period median 0.49% / 0.81% and max 2.85% /
    4.26% (alpha / beam), widths max 5.8% (a 5 cm near-axis alpha orbit)."""
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import gc_vs_full_orbit as B
    env = B.setup()
    res = B.compare(env, 10, 10, 2.0e-4, verbose=False)
    for name in ("alpha", "beam"):
        s = B.summarize(res[name])
        assert s["class_agree"] == s["n_decided"], f"{name}: {s['class_agree']}/{s['n_decided']}"
        assert s["lost_agree"] == s["n"], f"{name}: loss outcomes differ"
        assert s["n_sensitive"] <= 0.3 * s["n"], f"{name}: {s['n_sensitive']} sensitive orbits"
        assert s["n_period"] - s["n_sensitive"] >= 6
        assert s["period_ins_med"] < 0.015, f"{name}: period median {s['period_ins_med']:.2%}"
        assert s["period_ins_max"] < 0.04, f"{name}: period max {s['period_ins_max']:.2%}"
        assert s["width_ins_med"] < 0.015, f"{name}: width median {s['width_ins_med']:.2%}"
        assert s["width_ins_max"] < 0.07, f"{name}: width max {s['width_ins_max']:.2%}"


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
