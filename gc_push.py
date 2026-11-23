"""Guiding-centre pusher for fast ions (Tier 3).

Non-relativistic guiding-centre equations in the B* formulation (Littlejohn,
J. Plasma Phys. 29, 111 (1983); Cary & Brizard, Rev. Mod. Phys. 81, 693 (2009),
Sec. III), static axisymmetric fields:

    B*      = B + (m v_par / q) curl b
    B*_par  = b . B*
    E*      = E - (mu / q) grad|B|
    dX/dt   = (v_par B* + E* x b) / B*_par
    dv_par/dt = (q / m) (B* . E*) / B*_par
    mu      = m v_perp^2 / (2 |B|)                     (constant)

State per marker: (R, phi, Z, v_par), components in the orthonormal (R, phi, Z)
basis, dphi/dt = V_phi / R. Exact invariants of these equations in the fields
of fields_gc.GCField:
    W     = m v_par^2 / 2 + mu |B| + q Phi(psi)
    P_phi = m v_par b_phi R + q psi          (R A_phi = +psi in this code)

Integration: classical fixed-step RK4, one Numba prange iteration per marker.
Loss: the marker is lost when its guiding centre reaches psi_N > 1 (the last
closed flux surface, the same psi_edge as the full-orbit run's loss test).

Time step criterion (dt_transit_fraction): dt = f * 2 pi R_axis / v, a fixed
fraction f of the marker's toroidal transit time at its speed v. The poloidal
transit time is q times longer (q = 1.4 to 2), so f = 1/200 gives 280 to 400
steps per poloidal transit; trapped orbits get more per bounce. For a 3.5 MeV
alpha (v = 1.30e7 m/s, R_axis = 1.153 m) f = 1/200 is dt = 2.8 ns.
"""
import numpy as np
from numba import njit, prange

from fields_gc import gc_fields, gc_psi_value

DEFAULT_TRANSIT_FRACTION = 1.0 / 200.0


def dt_transit_fraction(field, v, fraction=DEFAULT_TRANSIT_FRACTION):
    """dt = fraction * (2 pi R_axis / v) [s] for speed(s) v [m/s]."""
    return fraction * 2.0 * np.pi * field.R_axis / np.asarray(v, dtype=np.float64)


# =======================================================
# EQUATIONS OF MOTION
# =======================================================
@njit(cache=True, fastmath=True, inline="always")
def gc_rhs(R, Z, vpar, mu, q, m,
           coef, R_min, Z_min, dR, dZ, nR, nZ, F,
           psi_axis, psi_edge, phi_x0, phi_dx, phi_val, phi_der):
    """(dR/dt, dphi/dt, dZ/dt, dv_par/dt, psi_N) at one guiding-centre state."""
    BR, Bp, BZ, B, gR, gZ, cR, cp, cZ, psi, psiN, Phi, ER, EZ = gc_fields(
        R, Z, coef, R_min, Z_min, dR, dZ, nR, nZ, F,
        psi_axis, psi_edge, phi_x0, phi_dx, phi_val, phi_der)
    iB = 1.0 / B
    bR = BR * iB
    bp = Bp * iB
    bZ = BZ * iB
    k = m * vpar / q
    BsR = BR + k * cR
    Bsp = Bp + k * cp
    BsZ = BZ + k * cZ
    Bspar = bR * BsR + bp * Bsp + bZ * BsZ
    muq = mu / q
    EsR = ER - muq * gR
    EsZ = EZ - muq * gZ            # E*_phi = 0 (axisymmetric)
    iBs = 1.0 / Bspar
    VR = (vpar * BsR - EsZ * bp) * iBs
    Vp = (vpar * Bsp + EsZ * bR - EsR * bZ) * iBs
    VZ = (vpar * BsZ + EsR * bp) * iBs
    dv = (q / m) * (BsR * EsR + BsZ * EsZ) * iBs
    return VR, Vp / R, VZ, dv, psiN


@njit(cache=True, fastmath=True, inline="always")
def gc_rk4_step(R, ph, Z, vp, mu, q, m, dt,
                coef, R_min, Z_min, dR, dZ, nR, nZ, F,
                psi_axis, psi_edge, phi_x0, phi_dx, phi_val, phi_der):
    """One classical RK4 step. Returns the new (R, phi, Z, v_par)."""
    a1, b1, c1, d1, _ = gc_rhs(R, Z, vp, mu, q, m, coef, R_min, Z_min, dR, dZ, nR, nZ, F,
                               psi_axis, psi_edge, phi_x0, phi_dx, phi_val, phi_der)
    h = 0.5 * dt
    a2, b2, c2, d2, _ = gc_rhs(R + h * a1, Z + h * c1, vp + h * d1, mu, q, m,
                               coef, R_min, Z_min, dR, dZ, nR, nZ, F,
                               psi_axis, psi_edge, phi_x0, phi_dx, phi_val, phi_der)
    a3, b3, c3, d3, _ = gc_rhs(R + h * a2, Z + h * c2, vp + h * d2, mu, q, m,
                               coef, R_min, Z_min, dR, dZ, nR, nZ, F,
                               psi_axis, psi_edge, phi_x0, phi_dx, phi_val, phi_der)
    a4, b4, c4, d4, _ = gc_rhs(R + dt * a3, Z + dt * c3, vp + dt * d3, mu, q, m,
                               coef, R_min, Z_min, dR, dZ, nR, nZ, F,
                               psi_axis, psi_edge, phi_x0, phi_dx, phi_val, phi_der)
    s = dt / 6.0
    return (R + s * (a1 + 2.0 * a2 + 2.0 * a3 + a4),
            ph + s * (b1 + 2.0 * b2 + 2.0 * b3 + b4),
            Z + s * (c1 + 2.0 * c2 + 2.0 * c3 + c4),
            vp + s * (d1 + 2.0 * d2 + 2.0 * d3 + d4))


# =======================================================
# KERNELS
# =======================================================
@njit(parallel=True, cache=True, fastmath=True)
def gc_push(state, mu, q, m, dt, n_steps, alive, lost_step,
            coef, R_min, Z_min, dR, dZ, nR, nZ, F,
            psi_axis, psi_edge, phi_x0, phi_dx, phi_val, phi_der):
    """Advance every alive marker n_steps RK4 steps of its own dt[i], in place.

    state (N, 4) float64: R, phi, Z, v_par.  mu (N,), dt (N,).  alive (N,) int8:
    a marker whose guiding centre reaches psi_N > 1 is frozen at that point,
    alive[i] = 0, and lost_step[i] = the step (1-based) on which it crossed.
    """
    N = state.shape[0]
    for i in prange(N):
        if alive[i] == 0:
            continue
        R = state[i, 0]; ph = state[i, 1]; Z = state[i, 2]; vp = state[i, 3]
        mui = mu[i]; dti = dt[i]
        for s in range(n_steps):
            R, ph, Z, vp = gc_rk4_step(R, ph, Z, vp, mui, q, m, dti,
                                       coef, R_min, Z_min, dR, dZ, nR, nZ, F,
                                       psi_axis, psi_edge, phi_x0, phi_dx, phi_val, phi_der)
            psiN = (psi_axis - gc_psi_value(R, Z, coef, R_min, Z_min, dR, dZ, nR, nZ)) / (psi_axis - psi_edge)
            if psiN > 1.0:
                alive[i] = 0
                lost_step[i] = s + 1
                break
        state[i, 0] = R; state[i, 1] = ph; state[i, 2] = Z; state[i, 3] = vp


@njit(parallel=True, cache=True, fastmath=True)
def gc_push_record(state, mu, q, m, dt, n_steps, rec_every, out, alive, lost_step,
                   coef, R_min, Z_min, dR, dZ, nR, nZ, F,
                   psi_axis, psi_edge, phi_x0, phi_dx, phi_val, phi_der):
    """gc_push that also writes the state every rec_every steps into
    out[i, k, :] (k = 0 is the initial state). Rows after a loss repeat the
    loss point."""
    N = state.shape[0]
    n_rec = out.shape[1]
    for i in prange(N):
        R = state[i, 0]; ph = state[i, 1]; Z = state[i, 2]; vp = state[i, 3]
        out[i, 0, 0] = R; out[i, 0, 1] = ph; out[i, 0, 2] = Z; out[i, 0, 3] = vp
        k = 1
        mui = mu[i]; dti = dt[i]
        for s in range(n_steps):
            if alive[i] != 0:
                R, ph, Z, vp = gc_rk4_step(R, ph, Z, vp, mui, q, m, dti,
                                           coef, R_min, Z_min, dR, dZ, nR, nZ, F,
                                           psi_axis, psi_edge, phi_x0, phi_dx, phi_val, phi_der)
                psiN = (psi_axis - gc_psi_value(R, Z, coef, R_min, Z_min, dR, dZ, nR, nZ)) / (psi_axis - psi_edge)
                if psiN > 1.0:
                    alive[i] = 0
                    lost_step[i] = s + 1
            if (s + 1) % rec_every == 0 and k < n_rec:
                out[i, k, 0] = R; out[i, k, 1] = ph; out[i, k, 2] = Z; out[i, k, 3] = vp
                k += 1
        state[i, 0] = R; state[i, 1] = ph; state[i, 2] = Z; state[i, 3] = vp


# =======================================================
# PYTHON-SIDE HELPERS
# =======================================================
def invariants(field, state, mu, q, m):
    """(W [J], P_phi [kg m^2/s]) of guiding-centre states (N, 4):
    W = m v_par^2/2 + mu |B| + q Phi,  P_phi = m v_par b_phi R + q psi."""
    R, Z, vp = state[:, 0], state[:, 2], state[:, 3]
    f = field.fields(R, Z)
    W = 0.5 * m * vp**2 + mu * f["Bmag"] + q * f["Phi"]
    P = m * vp * (f["Bphi"] / f["Bmag"]) * R + q * f["psi"]
    return W, P


def gc_state_from_energy_pitch(field, R, Z, E_J, xi, m, phi=None):
    """Guiding-centre states at (R, Z) with kinetic energy E_J and pitch
    xi = v_par / v. Returns (state (N, 4), mu (N,), v (N,))."""
    R = np.atleast_1d(np.asarray(R, dtype=np.float64))
    Z = np.broadcast_to(np.asarray(Z, dtype=np.float64), R.shape).copy()
    E = np.broadcast_to(np.asarray(E_J, dtype=np.float64), R.shape)
    xi = np.broadcast_to(np.asarray(xi, dtype=np.float64), R.shape)
    v = np.sqrt(2.0 * E / m)
    B = field.fields(R, Z)["Bmag"]
    state = np.zeros((R.shape[0], 4))
    state[:, 0] = R
    state[:, 1] = 0.0 if phi is None else phi
    state[:, 2] = Z
    state[:, 3] = xi * v
    mu = 0.5 * m * (v * v - state[:, 3] ** 2) / B
    return state, mu, v


def gc_from_particle(field, pos, vel, q, m, B_at_particle=None):
    """First-order guiding centre of full-orbit particles.

    pos, vel (N, 3) Cartesian. X = x - (b x v) / Omega, Omega = q |B| / m, with b
    and |B| at the particle (from B_at_particle (N, 3) Cartesian if given, e.g.
    the full-orbit run's bilinear field, else the spline field). v_par = v . b,
    mu = m v_perp^2 / (2 |B|), both at the particle.
    Returns (state (N, 4), mu (N,)).
    """
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    if B_at_particle is None:
        R = np.hypot(pos[:, 0], pos[:, 1])
        c, s = pos[:, 0] / R, pos[:, 1] / R
        f = field.fields(R, pos[:, 2])
        Bvec = np.stack([f["BR"] * c - f["Bphi"] * s, f["BR"] * s + f["Bphi"] * c, f["BZ"]], axis=1)
    else:
        Bvec = np.asarray(B_at_particle, dtype=np.float64)
    Bm = np.linalg.norm(Bvec, axis=1)
    b = Bvec / Bm[:, None]
    vpar = np.sum(vel * b, axis=1)
    vperp2 = np.maximum(np.sum(vel * vel, axis=1) - vpar**2, 0.0)
    Omega = q * Bm / m
    X = pos - np.cross(b, vel) / Omega[:, None]
    state = np.empty((pos.shape[0], 4))
    state[:, 0] = np.hypot(X[:, 0], X[:, 1])
    state[:, 1] = np.arctan2(X[:, 1], X[:, 0])
    state[:, 2] = X[:, 2]
    state[:, 3] = vpar
    mu = 0.5 * m * vperp2 / Bm
    return state, mu


def push(field, state, mu, q, m, dt, n_steps, alive=None, lost_step=None):
    """gc_push on a GCField (dt scalar or per-marker). Returns (alive, lost_step)."""
    N = state.shape[0]
    alive = np.ones(N, dtype=np.int8) if alive is None else alive
    lost_step = np.zeros(N, dtype=np.int64) if lost_step is None else lost_step
    dt = np.broadcast_to(np.asarray(dt, dtype=np.float64), (N,)).copy()
    gc_push(state, mu, q, m, dt, int(n_steps), alive, lost_step, *field.args())
    return alive, lost_step


def push_record(field, state, mu, q, m, dt, n_steps, rec_every):
    """gc_push_record on a GCField. Returns (track (N, n_rec, 4), alive, lost_step)."""
    N = state.shape[0]
    n_rec = int(n_steps) // int(rec_every) + 1
    out = np.zeros((N, n_rec, 4))
    alive = np.ones(N, dtype=np.int8)
    lost_step = np.zeros(N, dtype=np.int64)
    dt = np.broadcast_to(np.asarray(dt, dtype=np.float64), (N,)).copy()
    gc_push_record(state, mu, q, m, dt, int(n_steps), int(rec_every), out, alive, lost_step,
                   *field.args())
    return out, alive, lost_step


def particle_invariants(field, pos, vel, q, m):
    """(W, P_phi) of full-orbit particles in the spline fields:
    W = m v^2/2 + q Phi(x), P_phi = m R v_phi + q psi(x)."""
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    Rp = np.hypot(pos[:, 0], pos[:, 1])
    fp = field.fields(Rp, pos[:, 2])
    v_phi = (-vel[:, 0] * pos[:, 1] + vel[:, 1] * pos[:, 0]) / Rp
    return 0.5 * m * np.sum(vel * vel, axis=1) + q * fp["Phi"], m * Rp * v_phi + q * fp["psi"]


def gc_from_particle_invariants(field, pos, vel, q, m, B_at_particle=None, W=None, P_phi=None):
    """Guiding centre of full-orbit particles that matches their exact
    invariants. X is the first-order guiding centre (gc_from_particle); then
    v_par and mu are chosen so the GC has the particle's energy
    W = m v^2/2 + q Phi(x) and canonical momentum P_phi = m R v_phi + q psi(x):
        v_par = (P_phi - q psi(X)) / (m b_phi(X) R_X)
        mu    = (W - q Phi(X) - m v_par^2 / 2) / |B(X)|
    Both are exact constants of the full and the guiding-centre motion in an
    axisymmetric static field, so the two orbits belong to the same (W, P_phi)
    family; v_par and mu taken at the particle carry O(rho/L) gyrophase errors
    (~2% for a 3.5 MeV alpha). W and P_phi may be supplied instead (e.g. their
    time averages along a full orbit whose fields conserve them only on
    average). Returns (state (N, 4), mu (N,))."""
    state, _ = gc_from_particle(field, pos, vel, q, m, B_at_particle)
    W0, P0 = particle_invariants(field, pos, vel, q, m)
    W = W0 if W is None else np.asarray(W, dtype=np.float64)
    P = P0 if P_phi is None else np.asarray(P_phi, dtype=np.float64)
    fX = field.fields(state[:, 0], state[:, 2])
    vpar = (P - q * fX["psi"]) / (m * (fX["Bphi"] / fX["Bmag"]) * state[:, 0])
    mu = (W - q * fX["Phi"] - 0.5 * m * vpar**2) / fX["Bmag"]
    state[:, 3] = vpar
    return state, np.maximum(mu, 0.0)
