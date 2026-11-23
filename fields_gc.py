"""Smooth equilibrium fields for guiding-centre (GC) motion (Tier 3).

The full-orbit reactor run reads B from bilinear grids: the poloidal field is
built by central differences (np.gradient) of the 60 x 60 Grad-Shafranov psi,
divided by R, rescaled to q(a) = Q_SAFETY_TARGET and resampled onto the 50 x 50
field grid (physics_engine.compute_poloidal_field_grids). That field is only
C0, its divergence is not zero (~0.6% of B_pol / a by central differences) and
it has no flux function an exact invariant could be built from.

The GC pusher needs the second derivatives of psi (grad|B| and curl b), so here
the SAME psi -- the reactor's 60 x 60 solve, times the SAME q-rescaling factor
that compute_poloidal_field_grids applies to B_pol -- is represented by an
interpolating bicubic spline (scipy RectBivariateSpline, s = 0):

    B_R = -(1/R) dpsi/dZ,   B_Z = (1/R) dpsi/dR,   B_phi = B0 R0 / R

(the reactor's sign convention and toroidal field). B is then divergence-free
to rounding, and with R A_phi = +psi the canonical toroidal momentum
    P_phi = m R v_phi + q psi
is an exact invariant of axisymmetric motion. (Written m R v_phi - q psi in the
opposite convention, B_pol = grad phi x grad psi.)

Agreement with the full-orbit field: the spline reproduces psi at the 60 x 60
nodes to rounding; its B differs from the reactor's bilinear B at the 50 x 50
nodes by the reactor's own finite-difference and resampling error, ~4e-4 of
|B| at worst (tests/test_fastion.py measures it and shows it falls as h^2).

Numba tables. The spline is a bicubic polynomial on every grid cell (its knots
are a subset of the nodes), so each cell's 16 monomial coefficients are
recovered exactly by fitting the spline at a 4 x 4 lattice of points inside
the cell. gc_psi_derivs evaluates psi and all derivatives up to second order
from those tables, from inside njit code.

Electric field: the quasi-neutral potential of Tier 1, e n_i E = grad p_i
(physics_engine.compute_force_balance_potential), from the background profiles
(background.py), tabulated as Phi(psi_N) with dPhi/dpsi_N on a uniform psi_N
mesh and evaluated by cubic Hermite interpolation (exact for the linear Phi of
n, T proportional to 1 - psi_N).

Omitted: the prescribed tearing-mode ripple of the full-orbit run. It breaks
axisymmetry (and with it the P_phi invariant), and it is ~1e-5 of B0 at the
end of the 10 us reactor run while it grows as exp(gamma t) -- meaningless over
the ~1 s slowing-down time followed here.
"""
import contextlib
import io

import numpy as np
from numba import njit
from scipy.interpolate import RectBivariateSpline


# =======================================================
# EQUILIBRIUM -> SPLINE TABLES
# =======================================================
def build_reactor_equilibrium(cfg=None, quiet=True):
    """The reactor's equilibrium exactly as main.run_reactor_steady_state builds
    it: MHDEquilibrium() defaults, the cached GS solve, the flux-surface
    thresholds (psi_axis, psi_edge, plasma volume) and the poloidal-field
    rescale factor for q(a) = Q_SAFETY_TARGET. Returns (cfg, eq, scale)."""
    from config import SimulationConfiguration
    from mhd_equilibrium import MHDEquilibrium
    from initialization import _attach_flux_surfaces
    from physics_engine import compute_poloidal_field_grids

    ctx = contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext()
    with ctx:
        if cfg is None:
            cfg = SimulationConfiguration()
        eq = MHDEquilibrium()
        eq.solve_grad_shafranov()
        _attach_flux_surfaces(eq, cfg)
        _, _, scale, _ = compute_poloidal_field_grids(
            eq, cfg.B0, R0=cfg.R0_major, a_minor=0.3, q_target=cfg.Q_SAFETY_TARGET)
    return cfg, eq, float(scale)


def _cell_coefficients(spline, R_nodes, Z_nodes):
    """(nR-1, nZ-1, 4, 4) monomial coefficients c[i, j, a, b] with
    psi = sum_ab c_ab t^a u^b, t = (R - R_i)/dR, u = (Z - Z_j)/dZ in cell (i, j)."""
    # Four points strictly inside each cell (so the per-cell samples are strictly
    # increasing across cells, as a grid evaluation requires); a cubic is fixed
    # by any four distinct points.
    s = np.array([0.125, 0.375, 0.625, 0.875])
    V = np.vander(s, 4, increasing=True)            # V[k, a] = s_k^a
    Vinv = np.linalg.inv(V)
    dR = R_nodes[1] - R_nodes[0]
    dZ = Z_nodes[1] - Z_nodes[0]
    nR, nZ = len(R_nodes), len(Z_nodes)
    # Sample points: every cell's 4 x 4 lattice, evaluated in one call.
    Rs = (R_nodes[:-1, None] + s[None, :] * dR).ravel()          # (nR-1)*4
    Zs = (Z_nodes[:-1, None] + s[None, :] * dZ).ravel()          # (nZ-1)*4
    F = spline(Rs, Zs, grid=True).reshape(nR - 1, 4, nZ - 1, 4)  # [i, k, j, l]
    F = F.transpose(0, 2, 1, 3)                                  # [i, j, k, l]
    # F = V C V^T per cell  ->  C = Vinv F Vinv^T
    C = np.einsum("ak,ijkl,bl->ijab", Vinv, F, Vinv)
    return np.ascontiguousarray(C)


class GCField:
    """Spline equilibrium + potential, with the arrays njit kernels need.

    Attributes (all SI): coef, R_min, Z_min, dR, dZ, nR, nZ (spline cells),
    F = B0 R0 (R B_phi), psi_axis, psi_edge (scaled, the reactor's values times
    the q-rescale), phi_x0, phi_dx, phi_val, phi_der (Phi table over psi_N),
    scale, R_axis, Z_axis (spline magnetic axis).
    """

    def __init__(self, cfg, eq, scale, Phi_of_psiN=None, n_phi=401):
        self.cfg = cfg
        self.eq = eq
        self.scale = float(scale)
        self.R_nodes = np.linspace(eq.psi_R_min, eq.psi_R_max, eq.psi_nR)
        self.Z_nodes = np.linspace(eq.psi_Z_min, eq.psi_Z_max, eq.psi_nZ)
        self.psi_nodes = np.ascontiguousarray(eq.psi_grid * self.scale)   # (R, Z)
        self.spline = RectBivariateSpline(self.R_nodes, self.Z_nodes, self.psi_nodes,
                                          kx=3, ky=3, s=0)
        self.coef = _cell_coefficients(self.spline, self.R_nodes, self.Z_nodes)
        self.R_min = float(self.R_nodes[0])
        self.Z_min = float(self.Z_nodes[0])
        self.dR = float(self.R_nodes[1] - self.R_nodes[0])
        self.dZ = float(self.Z_nodes[1] - self.Z_nodes[0])
        self.nR = int(eq.psi_nR)
        self.nZ = int(eq.psi_nZ)
        self.F = float(cfg.B0 * cfg.R0_major)
        # The reactor's flux labels (grid maximum and the ring minimum), scaled,
        # so psi_N = 1 is the same last closed flux surface value.
        self.psi_axis = float(eq.psi_axis * self.scale)
        self.psi_edge = float(eq.psi_edge * self.scale)
        self.R_axis, self.Z_axis = self._find_axis()

        # Potential table over psi_N in [0, 1]; extended linearly a little below 0
        # (the spline maximum sits just above the grid-node maximum).
        x = np.linspace(-0.05, 1.0, n_phi)
        if Phi_of_psiN is None:
            val = np.zeros_like(x)
            der = np.zeros_like(x)
        else:
            val, der = Phi_of_psiN(x)
        self.phi_x0 = float(x[0])
        self.phi_dx = float(x[1] - x[0])
        self.phi_val = np.ascontiguousarray(val, dtype=np.float64)
        self.phi_der = np.ascontiguousarray(der, dtype=np.float64)

    # ---- packed argument tuple for the njit kernels -----------------------
    def args(self):
        return (self.coef, self.R_min, self.Z_min, self.dR, self.dZ, self.nR, self.nZ,
                self.F, self.psi_axis, self.psi_edge,
                self.phi_x0, self.phi_dx, self.phi_val, self.phi_der)

    def _find_axis(self):
        from scipy.optimize import minimize
        i, j = np.unravel_index(int(np.argmax(self.psi_nodes)), self.psi_nodes.shape)
        x0 = np.array([self.R_nodes[i], self.Z_nodes[j]])
        res = minimize(lambda p: -self.spline.ev(p[0], p[1]), x0, method="Nelder-Mead",
                       options={"xatol": 1e-10, "fatol": 1e-16, "maxiter": 2000})
        return float(res.x[0]), float(res.x[1])

    # ---- vectorised python-side evaluation (diagnostics / tests) -----------
    def psi(self, R, Z):
        return gc_psi_array(np.ascontiguousarray(R, dtype=np.float64).ravel(),
                            np.ascontiguousarray(Z, dtype=np.float64).ravel(),
                            *self.args())[0].reshape(np.shape(R))

    def psi_N(self, R, Z):
        return (self.psi_axis - self.psi(R, Z)) / (self.psi_axis - self.psi_edge)

    def fields(self, R, Z):
        """dict of B (R, phi, Z components), |B|, grad|B| (R, Z), b, curl b
        (R, phi, Z), psi, psi_N, Phi, E (R, Z) at arrays R, Z."""
        R = np.ascontiguousarray(R, dtype=np.float64)
        Z = np.ascontiguousarray(Z, dtype=np.float64)
        out = gc_fields_array(R.ravel(), Z.ravel(), *self.args())
        names = ["BR", "Bphi", "BZ", "Bmag", "gradB_R", "gradB_Z",
                 "curlb_R", "curlb_phi", "curlb_Z", "psi", "psiN", "Phi", "E_R", "E_Z"]
        return {k: out[:, n].reshape(R.shape) for n, k in enumerate(names)}

    def flux_surface_midplane(self, psiN, Z=None):
        """(R_in, R_out) where the psi_N surface crosses Z = Z_axis."""
        from scipy.optimize import brentq
        Zc = self.Z_axis if Z is None else Z
        f = lambda R: self.psi_N(np.array([R]), np.array([Zc]))[0] - psiN
        R_in = brentq(f, self.R_min + 1e-6, self.R_axis)
        R_out = brentq(f, self.R_axis, self.R_min + (self.nR - 1) * self.dR - 1e-6)
        return R_in, R_out

    def flux_surface_contour(self, psiN, n_theta=721):
        """(R, Z) of the psi_N surface, ray-traced from the spline axis."""
        from scipy.optimize import brentq
        th = np.linspace(0.0, 2.0 * np.pi, n_theta)
        r_max = 0.49
        R_out = np.empty(n_theta)
        Z_out = np.empty(n_theta)
        for k, t in enumerate(th):
            c, s = np.cos(t), np.sin(t)
            f = lambda r: self.psi_N(np.array([self.R_axis + r * c]),
                                     np.array([self.Z_axis + r * s]))[0] - psiN
            rr = np.linspace(1e-6, r_max, 200)
            vals = self.psi_N(self.R_axis + rr * c, self.Z_axis + rr * s) - psiN
            idx = np.nonzero(vals > 0)[0]
            hi = rr[idx[0]] if idx.size else r_max
            lo = rr[idx[0] - 1] if idx.size and idx[0] > 0 else 1e-6
            r = brentq(f, lo, hi) if idx.size else r_max
            R_out[k] = self.R_axis + r * c
            Z_out[k] = self.Z_axis + r * s
        return R_out, Z_out


# =======================================================
# NJIT FIELD EVALUATION
# =======================================================
@njit(cache=True)
def gc_psi_derivs(R, Z, coef, R_min, Z_min, dR, dZ, nR, nZ):
    """psi, psi_R, psi_Z, psi_RR, psi_RZ, psi_ZZ of the bicubic spline at (R, Z).
    Points outside the spline box are clamped into the edge cell (the GC loss
    test, psi_N > 1, fires long before a marker reaches the box)."""
    tx = (R - R_min) / dR
    ty = (Z - Z_min) / dZ
    i = int(np.floor(tx))
    j = int(np.floor(ty))
    if i < 0:
        i = 0
    elif i > nR - 2:
        i = nR - 2
    if j < 0:
        j = 0
    elif j > nZ - 2:
        j = nZ - 2
    t = tx - i
    u = ty - j
    # Powers and their derivatives in t and u
    t2 = t * t
    t3 = t2 * t
    u2 = u * u
    u3 = u2 * u
    p = 0.0; pt = 0.0; pu = 0.0; ptt = 0.0; ptu = 0.0; puu = 0.0
    for a in range(4):
        ta = 1.0 if a == 0 else (t if a == 1 else (t2 if a == 2 else t3))
        dta = 0.0 if a == 0 else (1.0 if a == 1 else (2.0 * t if a == 2 else 3.0 * t2))
        d2ta = 0.0 if a < 2 else (2.0 if a == 2 else 6.0 * t)
        # Inner polynomial in u for this a
        c0 = coef[i, j, a, 0]; c1 = coef[i, j, a, 1]; c2 = coef[i, j, a, 2]; c3 = coef[i, j, a, 3]
        q = c0 + c1 * u + c2 * u2 + c3 * u3
        qu = c1 + 2.0 * c2 * u + 3.0 * c3 * u2
        quu = 2.0 * c2 + 6.0 * c3 * u
        p += ta * q
        pt += dta * q
        pu += ta * qu
        ptt += d2ta * q
        ptu += dta * qu
        puu += ta * quu
    return (p, pt / dR, pu / dZ, ptt / (dR * dR), ptu / (dR * dZ), puu / (dZ * dZ))


@njit(cache=True)
def gc_phi(psiN, phi_x0, phi_dx, phi_val, phi_der):
    """Phi(psi_N) and dPhi/dpsi_N by cubic Hermite interpolation of the table;
    0 outside the last closed flux surface (psi_N > 1), as in Tier 1."""
    n = phi_val.shape[0]
    if psiN > 1.0:
        return 0.0, 0.0
    s = (psiN - phi_x0) / phi_dx
    k = int(np.floor(s))
    if k < 0:
        k = 0
    elif k > n - 2:
        k = n - 2
    h = s - k
    h2 = h * h
    h3 = h2 * h
    y0 = phi_val[k]; y1 = phi_val[k + 1]
    m0 = phi_der[k] * phi_dx; m1 = phi_der[k + 1] * phi_dx
    val = ((2 * h3 - 3 * h2 + 1) * y0 + (h3 - 2 * h2 + h) * m0
           + (-2 * h3 + 3 * h2) * y1 + (h3 - h2) * m1)
    der = ((6 * h2 - 6 * h) * y0 + (3 * h2 - 4 * h + 1) * m0
           + (-6 * h2 + 6 * h) * y1 + (3 * h2 - 2 * h) * m1) / phi_dx
    return val, der


@njit(cache=True)
def gc_fields(R, Z, coef, R_min, Z_min, dR, dZ, nR, nZ, F,
              psi_axis, psi_edge, phi_x0, phi_dx, phi_val, phi_der):
    """Every field quantity the GC equations need at (R, Z), axisymmetric,
    components in the orthonormal (R, phi, Z) basis:
        B_R, B_phi, B_Z, |B|, d|B|/dR, d|B|/dZ, (curl b)_R, (curl b)_phi,
        (curl b)_Z, psi, psi_N, Phi, E_R, E_Z.
    """
    p, pR, pZ, pRR, pRZ, pZZ = gc_psi_derivs(R, Z, coef, R_min, Z_min, dR, dZ, nR, nZ)
    iR = 1.0 / R
    BR = -pZ * iR
    BZ = pR * iR
    Bp = F * iR
    B = np.sqrt(BR * BR + BZ * BZ + Bp * Bp)
    iB = 1.0 / B
    # Derivatives of the components
    dBR_dR = pZ * iR * iR - pRZ * iR
    dBR_dZ = -pZZ * iR
    dBZ_dR = pRR * iR - pR * iR * iR
    dBZ_dZ = pRZ * iR
    dBp_dR = -F * iR * iR
    dB_dR = (BR * dBR_dR + BZ * dBZ_dR + Bp * dBp_dR) * iB
    dB_dZ = (BR * dBR_dZ + BZ * dBZ_dZ) * iB
    # Unit vector derivatives  d b_k = (d B_k - b_k d|B|) / |B|
    bR = BR * iB
    bZ = BZ * iB
    bp = Bp * iB
    dbR_dZ = (dBR_dZ - bR * dB_dZ) * iB
    dbZ_dR = (dBZ_dR - bZ * dB_dR) * iB
    dbp_dR = (dBp_dR - bp * dB_dR) * iB
    dbp_dZ = (-bp * dB_dZ) * iB
    # curl b in cylindrical coordinates, d/dphi = 0
    cR = -dbp_dZ
    cp = dbR_dZ - dbZ_dR
    cZ = bp * iR + dbp_dR
    # Potential and E = -dPhi/dpsi_N grad psi_N, grad psi_N = -grad psi / dpsi
    dpsi = psi_axis - psi_edge
    psiN = (psi_axis - p) / dpsi
    Phi, dPhi = gc_phi(psiN, phi_x0, phi_dx, phi_val, phi_der)
    ER = dPhi * pR / dpsi
    EZ = dPhi * pZ / dpsi
    return BR, Bp, BZ, B, dB_dR, dB_dZ, cR, cp, cZ, p, psiN, Phi, ER, EZ


@njit(cache=True)
def gc_psi_array(R, Z, coef, R_min, Z_min, dR, dZ, nR, nZ, F,
                 psi_axis, psi_edge, phi_x0, phi_dx, phi_val, phi_der):
    n = R.shape[0]
    out = np.empty((6, n))
    for k in range(n):
        d = gc_psi_derivs(R[k], Z[k], coef, R_min, Z_min, dR, dZ, nR, nZ)
        for m in range(6):
            out[m, k] = d[m]
    return out


@njit(cache=True)
def gc_fields_array(R, Z, coef, R_min, Z_min, dR, dZ, nR, nZ, F,
                    psi_axis, psi_edge, phi_x0, phi_dx, phi_val, phi_der):
    n = R.shape[0]
    out = np.empty((n, 14))
    for k in range(n):
        f = gc_fields(R[k], Z[k], coef, R_min, Z_min, dR, dZ, nR, nZ, F,
                      psi_axis, psi_edge, phi_x0, phi_dx, phi_val, phi_der)
        for m in range(14):
            out[k, m] = f[m]
    return out


def build_gc_field(cfg=None, background=None, quiet=True):
    """Reactor equilibrium + background plasma -> (GCField, Background). The
    potential is the background's quasi-neutral force-balance Phi(psi_N)."""
    from background import Background
    cfg, eq, scale = build_reactor_equilibrium(cfg, quiet=quiet)
    if background is None:
        background = Background.from_scenario(cfg, eq)
    field = GCField(cfg, eq, scale, Phi_of_psiN=background.force_balance_potential)
    return field, background
