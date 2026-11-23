"""Monte Carlo Fokker-Planck collision operator for fast ions (Tier 3).

Test particles (charge Z_a e, mass m_a, speed v, pitch xi = v_par / v) collide
with fixed Maxwellian backgrounds b = e, D, T (background.py). With
x_b = v / v_tb, v_tb = sqrt(2 T_b / m_b), the Chandrasekhar function
G(x) = [erf(x) - x erf'(x)] / (2 x^2) and
    Gamma_b = (Z_a Z_b e^2)^2 ln(Lambda_ab) / (4 pi eps0^2 m_a^2),
the Fokker-Planck coefficients of species b are (NRL Plasma Formulary,
"Relaxation rates": nu_s, nu_perp, nu_par with psi(x_NRL) = 2 x^2 G(x)):
    <dv_par>/dt     = -A_b,   A_b = (1 + m_a/m_b) 2 Gamma_b n_b G(x_b) / v_tb^2
    <dv_perp^2>/dt  = 2 nu_D,b v^2,   nu_D,b = Gamma_b n_b [erf(x_b) - G(x_b)] / v^3
    <dv_par^2>/dt   = D_b = 2 Gamma_b n_b G(x_b) / v

Monte Carlo step (Boozer & Kuo-Petravic, Phys. Fluids 24, 851 (1981)): with
random signs s = +-1,
    pitch:  xi' = xi (1 - nu_D dt) + s0 sqrt((1 - xi^2) nu_D dt),  nu_D = sum_b nu_D,b
    speed:  v'  = v + sum_b dv_b,  dv_b = (-A_b + nu_D,b v) dt + s_b sqrt(D_b dt)
(the speed drift is <dv> = <dv_par> + <dv_perp^2>/(2v), Ito). A step with
v' < 0 is reflected through the origin of velocity space (v' -> -v', xi' -> -xi').
The operator is sub-cycled so that every rate times the sub-step stays below
0.02 (only ever active close to thermal speeds).

Energy accounting, exact per step: with dv = sum_b dv_b,
    dE = (m/2)((v + dv)^2 - v^2) = sum_b [m v dv_b + (m/2) dv_b dv]
and species b receives -[m v dv_b + (m/2) dv_b dv]; the mean of that is b's
energy-exchange rate m v (-A_b + nu_D,b v) + m D_b / 2. Pitch scattering
conserves energy. Electron and ion (D + T) totals are kept separately.

Coulomb logarithm, one per species pair, from the local background and not
from the test particle's speed (a speed-dependent ln Lambda breaks detailed
balance: the Maxwellian would no longer be the operator's equilibrium):
  * ion-electron: NRL electron-ion form for T_e > 10 Z^2 eV,
        ln Lambda = 24 - ln(n_e^(1/2) / T_e)          (n_e in cm^-3, T_e in eV);
  * ion-ion (b = D, T): NRL counterstreaming ions in warm electrons,
        ln Lambda = 43 - ln[Z_a Z_b (mu_a + mu_b) / (mu_a mu_b beta_D^2) (n_e / T_e)^(1/2)]
    evaluated at the test species' critical speed, beta_D = v_c / c with
    E_c = (1/2) m_a v_c^2 = 14.8 A_a T_e [sum_j n_j Z_j^2 / (n_e A_j)]^(2/3)
    (Stix, Plasma Phys. 14, 367 (1972)) -- the speed at which ion drag takes
    over from electron drag, i.e. where the ion channel matters. (Over the
    alpha slowing-down this form would run from 25.3 at birth to ~21.7 near
    thermal; at v_c it is 23.8 on axis.)

Random numbers: a per-marker splitmix64 counter (rng_state[i]), so a seeded
run is reproducible whatever the thread count.
"""
import math

import numpy as np
from numba import njit, prange

from constants import E_CHARGE, EPSILON_0, M_PROTON, C_LIGHT, KEV_J
from background import SPECIES_MASS, SPECIES_CHARGE, background_at

SQRT_PI = math.sqrt(math.pi)
SUBCYCLE_LIMIT = 0.02
MAX_SUBCYCLES = 10000


# =======================================================
# RANDOM NUMBERS (splitmix64, one counter per marker)
# =======================================================
@njit(cache=True, inline="always")
def _splitmix64(state):
    z = (state + np.uint64(0x9E3779B97F4A7C15))
    s = z
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return s, z


@njit(cache=True)
def rng_uniform(rng_state, i):
    """Uniform double in [0, 1) from marker i's counter."""
    s, z = _splitmix64(rng_state[i])
    rng_state[i] = s
    return (z >> np.uint64(11)) * (1.0 / 9007199254740992.0)


def seed_rng(n, seed):
    """n independent splitmix64 counters from one seed."""
    ss = np.random.SeedSequence(seed)
    return ss.generate_state(n, dtype=np.uint64)


# =======================================================
# COEFFICIENTS
# =======================================================
@njit(cache=True, inline="always")
def chandrasekhar_G(x):
    """G(x) = [erf(x) - x erf'(x)] / (2 x^2); series below x = 1e-2."""
    if x < 1e-2:
        x2 = x * x
        return (2.0 * x / (3.0 * SQRT_PI)) * (1.0 - 0.6 * x2 + 3.0 * x2 * x2 / 14.0)
    return (math.erf(x) - x * (2.0 / SQRT_PI) * math.exp(-x * x)) / (2.0 * x * x)


def gamma_prefactor(Za, ma):
    """(Z_a e^2)^2 / (4 pi eps0^2 m_a^2) per unit Z_b^2 ln Lambda, SI."""
    return (Za * E_CHARGE**2) ** 2 / (4.0 * np.pi * EPSILON_0**2 * ma**2)


@njit(cache=True, inline="always")
def fp_coefficients(v, ma, gam, n, T, lnL, mass, charge, A, nuD, D):
    """Fill A[b], nuD[b], D[b] (b = e, D, T) at speed v. gam = gamma_prefactor."""
    for b in range(3):
        if n[b] <= 0.0 or T[b] <= 0.0:
            A[b] = 0.0
            nuD[b] = 0.0
            D[b] = 0.0
            continue
        vt2 = 2.0 * T[b] / mass[b]
        x = v / math.sqrt(vt2)
        G = chandrasekhar_G(x)
        Gb = gam * charge[b] * charge[b] * lnL[b] * n[b]
        A[b] = (1.0 + ma / mass[b]) * 2.0 * Gb * G / vt2
        nuD[b] = Gb * (math.erf(x) - G) / (v * v * v)
        D[b] = 2.0 * Gb * G / v


@njit(cache=True)
def fp_step(v, xi, dt, ma, gam, n, T, lnL, mass, charge, rng_state, i, dE, A, nuD, D):
    """One collision step of length dt (sub-cycled as needed). Returns (v, xi);
    adds the energy each species received to dE[b] (J, per real particle)."""
    t_left = dt
    while t_left > 0.0:
        fp_coefficients(v, ma, gam, n, T, lnL, mass, charge, A, nuD, D)
        nuDt = nuD[0] + nuD[1] + nuD[2]
        rate = nuDt
        for b in range(3):
            r1 = A[b] / v
            r2 = D[b] / (v * v)
            if r1 > rate:
                rate = r1
            if r2 > rate:
                rate = r2
        h = t_left
        if rate * h > SUBCYCLE_LIMIT:
            # at most MAX_SUBCYCLES per step: the rates grow as 1/v^3 as v -> 0, a
            # region of measure ~(v/v_t)^3 that must not stall the step
            h = max(SUBCYCLE_LIMIT / rate, dt / MAX_SUBCYCLES)
        t_left -= h
        # one draw supplies all four signs
        u = rng_uniform(rng_state, i)
        bits = int(u * 16.0)
        s0 = 1.0 if (bits & 1) else -1.0
        # pitch (Lorentz operator, BKP)
        xi_new = xi * (1.0 - nuDt * h) + s0 * math.sqrt(max(1.0 - xi * xi, 0.0) * nuDt * h)
        # speed, per species
        dv0 = (-A[0] + nuD[0] * v) * h + (1.0 if (bits & 2) else -1.0) * math.sqrt(D[0] * h)
        dv1 = (-A[1] + nuD[1] * v) * h + (1.0 if (bits & 4) else -1.0) * math.sqrt(D[1] * h)
        dv2 = (-A[2] + nuD[2] * v) * h + (1.0 if (bits & 8) else -1.0) * math.sqrt(D[2] * h)
        dv = dv0 + dv1 + dv2
        dE[0] -= ma * v * dv0 + 0.5 * ma * dv0 * dv
        dE[1] -= ma * v * dv1 + 0.5 * ma * dv1 * dv
        dE[2] -= ma * v * dv2 + 0.5 * ma * dv2 * dv
        v_new = v + dv
        if v_new < 0.0:
            v_new = -v_new
            xi_new = -xi_new
        if xi_new > 1.0:
            xi_new = 2.0 - xi_new
        elif xi_new < -1.0:
            xi_new = -2.0 - xi_new
        v = v_new
        xi = xi_new
        if v <= 0.0:
            v = 1e-30
    return v, xi


# =======================================================
# COULOMB LOGARITHMS
# =======================================================
def critical_energy_J(ma, n_e, T_e_J, n_ions, mass_ions, Z_ions=None):
    """Stix critical energy E_c = 14.8 A_a T_e [sum_j n_j Z_j^2 / (n_e A_j)]^(2/3)."""
    Z_ions = np.ones(len(n_ions)) if Z_ions is None else np.asarray(Z_ions, dtype=np.float64)
    s = sum(nj * Zj**2 / (n_e * (mj / M_PROTON)) for nj, Zj, mj in zip(n_ions, Z_ions, mass_ions))
    return 14.8 * (ma / M_PROTON) * T_e_J * s ** (2.0 / 3.0)


def coulomb_logs(Za, ma, n, T):
    """(ln Lambda_ae, ln Lambda_aD, ln Lambda_aT) for test species (Za, ma) in
    the local background n[3] (m^-3), T[3] (J), species order e, D, T. See the
    module docstring for the forms."""
    n_e, T_e = n[0], T[0]
    if n_e <= 0.0 or T_e <= 0.0:
        return np.zeros(3)
    ne_cm = n_e * 1e-6
    Te_eV = T_e / E_CHARGE
    lnL_e = 24.0 - math.log(math.sqrt(ne_cm) / Te_eV)
    Ec = critical_energy_J(ma, n_e, T_e, n[1:], SPECIES_MASS[1:])
    beta2 = 2.0 * Ec / ma / C_LIGHT**2
    mu_a = ma / M_PROTON
    out = [lnL_e]
    for b in (1, 2):
        mu_b = SPECIES_MASS[b] / M_PROTON
        Zb = SPECIES_CHARGE[b]
        arg = Za * Zb * (mu_a + mu_b) / (mu_a * mu_b * beta2) * math.sqrt(ne_cm / Te_eV)
        out.append(43.0 - math.log(arg))
    return np.array(out)


def coulomb_log_table(bg, Za, ma):
    """ln Lambda (3, n_x) on the background's psi_N mesh for one test species."""
    nx = bg.dens.shape[1]
    tab = np.zeros((3, nx))
    for k in range(nx):
        tab[:, k] = coulomb_logs(Za, ma, bg.dens[:, k], bg.temp[:, k])
    return np.ascontiguousarray(tab)


@njit(cache=True, inline="always")
def lnL_at(psiN, tab, x0, dx, out):
    nx = tab.shape[1]
    s = (psiN - x0) / dx
    if s < 0.0:
        s = 0.0
    k = int(math.floor(s))
    if k > nx - 2:
        k = nx - 2
    w = s - k
    if w > 1.0:
        w = 1.0
    for b in range(3):
        out[b] = tab[b, k] * (1.0 - w) + tab[b, k + 1] * w


# =======================================================
# UNIFORM-PLASMA KERNEL (tests, no orbits)
# =======================================================
@njit(parallel=True, cache=True)
def fp_uniform(v, xi, alive, dt, n_steps, ma, gam, n, T, lnL, mass, charge, rng_state,
               E_stop, dE_out, t_stop, energy_only, pitch_only):
    """Collide markers in a uniform background, marker i for n_steps[i] steps of dt.

    v, xi (N,) in place. alive (N,) int8: a marker whose energy drops below
    E_stop[i] stops (alive = 0, t_stop[i] = time of the step). dE_out (N, 3)
    accumulates the energy given to e, D, T. energy_only / pitch_only switch the
    pitch or the speed update off (for the single-process tests).
    """
    N = v.shape[0]
    for i in prange(N):
        if alive[i] == 0:
            continue
        A = np.empty(3)
        nuD = np.empty(3)
        D = np.empty(3)
        dE = np.zeros(3)
        vi = v[i]
        xii = xi[i]
        for s in range(n_steps[i]):
            v_new, xi_new = fp_step(vi, xii, dt, ma, gam, n, T, lnL, mass, charge,
                                    rng_state, i, dE, A, nuD, D)
            if pitch_only:
                # discard the speed change (and its energy) -- fixed energy
                for b in range(3):
                    dE[b] = 0.0
                v_new = vi
            if energy_only:
                xi_new = xii
            vi = v_new
            xii = xi_new
            if 0.5 * ma * vi * vi < E_stop[i]:
                alive[i] = 0
                t_stop[i] += (s + 1) * dt
                break
        v[i] = vi
        xi[i] = xii
        for b in range(3):
            dE_out[i, b] += dE[b]


class UniformPlasma:
    """A uniform Maxwellian background (n[3] m^-3, T[3] J, species e, D, T)
    and the arrays the kernels need for one test species."""

    def __init__(self, n, T, Za, ma, lnL=None):
        self.n = np.ascontiguousarray(n, dtype=np.float64)
        self.T = np.ascontiguousarray(T, dtype=np.float64)
        self.Za = float(Za)
        self.ma = float(ma)
        self.gam = gamma_prefactor(Za, ma)
        self.lnL = np.ascontiguousarray(coulomb_logs(Za, ma, self.n, self.T) if lnL is None else lnL,
                                        dtype=np.float64)
        self.mass = SPECIES_MASS.copy()
        self.charge = SPECIES_CHARGE.copy()

    def coefficients(self, v):
        """(A, nuD, D) arrays (3,) at speed v."""
        A = np.empty(3)
        nuD = np.empty(3)
        D = np.empty(3)
        fp_coefficients(float(v), self.ma, self.gam, self.n, self.T, self.lnL, self.mass,
                        self.charge, A, nuD, D)
        return A, nuD, D

    def spitzer_tau_s(self):
        """Spitzer electron slowing-down time
        tau_s = 3 (2 pi)^(3/2) eps0^2 m_a T_e^(3/2) / (Z_a^2 e^4 n_e m_e^(1/2) ln Lambda_e)."""
        from constants import M_ELECTRON
        return (3.0 * (2.0 * np.pi) ** 1.5 * EPSILON_0**2 * self.ma * self.T[0] ** 1.5
                / (self.Za**2 * E_CHARGE**4 * self.n[0] * math.sqrt(M_ELECTRON) * self.lnL[0]))

    def critical_speed(self):
        """v_c with v_c^3 = tau_s sum_ions (m_a/m_b) Gamma_b n_b: the speed at which
        the operator's fast-limit energy loss to the ions, m_a Gamma_b n_b
        (m_a/m_b) / v (the "1" of the (1 + m_a/m_b) momentum drag is returned by
        perpendicular diffusion), equals its slow-limit loss to the electrons,
        m_a v^2 / tau_s. E_c = m_a v_c^2 / 2 is then the critical energy of these
        coefficients -- Stix's 14.8 formula when ln Lambda_i = ln Lambda_e."""
        s = 0.0
        for b in (1, 2):
            s += (self.ma / self.mass[b]) * self.gam * self.charge[b] ** 2 * self.lnL[b] * self.n[b]
        return (self.spitzer_tau_s() * s) ** (1.0 / 3.0)

    def run(self, v, xi, dt, n_steps, rng_state, E_stop=None, energy_only=False, pitch_only=False,
            alive=None, dE=None, t_stop=None):
        N = v.shape[0]
        alive = np.ones(N, dtype=np.int8) if alive is None else alive
        dE = np.zeros((N, 3)) if dE is None else dE
        t_stop = np.zeros(N) if t_stop is None else t_stop
        E_stop = np.zeros(N) if E_stop is None else np.broadcast_to(E_stop, (N,)).astype(np.float64)
        steps = np.broadcast_to(np.asarray(n_steps, dtype=np.int64), (N,)).copy()
        fp_uniform(v, xi, alive, float(dt), steps, self.ma, self.gam, self.n, self.T,
                   self.lnL, self.mass, self.charge, rng_state, E_stop, dE, t_stop,
                   bool(energy_only), bool(pitch_only))
        return alive, dE, t_stop
