"""Maxwellian background plasma for the fast-ion module (Tier 3).

Three fixed Maxwellian species -- electrons, deuterons and tritons -- with
flux-surface profiles from the reactor scenario (config.py, initialization.py):

    n_i(psi_N) = n_D + n_T = n0 (1 - psi_N),    n_D = n_T = n_i / 2   (50/50 D-T)
    n_e(psi_N) = n_D + n_T                       (quasi-neutrality, Z = 1 ions)
    T_s(psi_N) = T0 (1 - psi_N)                  for s = e, D, T
    T0 = cfg.T_thermal_keV = 25 keV on axis

n0 is the reactor's loaded on-axis ion density, physics_engine.
loaded_central_density(eq, N_AVG_TARGET) = 2.03e20 m^-3: the value whose
(1 - psi_N) profile has volume average N_AVG_TARGET = 1e20 m^-3, and the value
initialize_reactor evaluates the NRL collision rates at.

T_e = T_i is an assumption (the reactor run has no electrons). ITER Q = 10
baseline modelling has T_e(0) ~ 27 keV against T_i(0) ~ 25 keV
(arXiv:2306.08442, METIS scenario), so T_e = T_i is within ~8% on axis; the
electron drag on fast ions scales as T_e^(-3/2), so T_e = 27 keV would lengthen
the electron slowing-down time by ~12%.

Potential: Tier 1's quasi-neutral radial force balance of the thermal ions with
no rotation, e n_i E = grad p_i (physics_engine.compute_force_balance_potential),
here for the analytic profiles: p_i = n_i T = n0 T0 (1 - psi_N)^2, so
    dPhi/dpsi_N = -(1/(e n_i)) dp_i/dpsi_N = 2 T0 / e,   Phi(1) = 0,
    Phi(psi_N) = -(2 T0 / e) (1 - psi_N)      (-50 kV on axis at 25 keV).

The backgrounds are fixed: fast-ion heating does not feed back on them.
"""
import numpy as np
from numba import njit

from constants import E_CHARGE, M_ELECTRON, M_DEUTERON, M_TRITON, KEV_J

# Background species order, shared with collisions_fp.py
SPECIES = ("e", "D", "T")
SPECIES_MASS = np.array([M_ELECTRON, M_DEUTERON, M_TRITON])
SPECIES_CHARGE = np.array([-1.0, 1.0, 1.0])            # in units of e
SPECIES_DENSITY_FRACTION = np.array([1.0, 0.5, 0.5])   # of n_i = n_D + n_T


class Background:
    """Fixed Maxwellian e, D, T profiles in psi_N with njit lookup tables.

    dens[s, k] [m^-3] and temp[s, k] [J] on the uniform psi_N mesh
    x_k = x0 + k dx, x0 = -0.05 (the spline axis sits marginally above the
    grid-node maximum that defines psi_N = 0), up to psi_N = 1. Linear
    interpolation -- exact for the linear scenario profiles.
    """

    def __init__(self, n0, T0_keV, n_table=1051):
        self.n0 = float(n0)
        self.T0_keV = float(T0_keV)
        self.x0 = -0.05
        x = np.linspace(self.x0, 1.0, n_table)
        self.dx = float(x[1] - x[0])
        shape = np.clip(1.0 - x, 0.0, None)
        self.dens = np.ascontiguousarray(SPECIES_DENSITY_FRACTION[:, None] * self.n0 * shape[None, :])
        self.temp = np.ascontiguousarray(np.repeat((self.T0_keV * KEV_J * shape)[None, :], 3, axis=0))
        self.mass = SPECIES_MASS.copy()
        self.charge = SPECIES_CHARGE.copy()

    @classmethod
    def from_scenario(cls, cfg, eq):
        from physics_engine import loaded_central_density
        n0 = loaded_central_density(eq, cfg.N_AVG_TARGET)
        return cls(n0, cfg.T_thermal_keV)

    # ---- profiles (python side) -------------------------------------------
    def n(self, species, psiN):
        s = SPECIES.index(species)
        return SPECIES_DENSITY_FRACTION[s] * self.n0 * np.clip(1.0 - np.asarray(psiN), 0.0, None)

    def T_keV(self, psiN):
        return self.T0_keV * np.clip(1.0 - np.asarray(psiN), 0.0, None)

    def force_balance_potential(self, psiN):
        """(Phi [V], dPhi/dpsi_N [V]) of e n_i E = grad p_i for these profiles."""
        x = np.asarray(psiN, dtype=np.float64)
        g = 2.0 * self.T0_keV * KEV_J / E_CHARGE
        return -g * (1.0 - x), np.full_like(x, g)

    def args(self):
        return (self.dens, self.temp, self.x0, self.dx, self.mass, self.charge)

    def shell_profile(self, eq, n_bins=None):
        """A compute_flux_profiles-shaped dict of the background on the reactor's
        psi_N shells (shell-centre values), for the Tier 2 alpha-source machinery
        (physics_engine.compute_alpha_birth_rates). Species rows follow the
        reactor's type codes: 0 thermal D, 1 beam D (empty), 2 alpha (empty),
        3 thermal T."""
        from physics_engine import FLUX_PROFILE_BINS, compute_flux_shell_volumes
        n_bins = FLUX_PROFILE_BINS if n_bins is None else n_bins
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        c = 0.5 * (edges[:-1] + edges[1:])
        n = np.zeros((4, n_bins))
        T = np.full((4, n_bins), np.nan)
        n[0] = self.n("D", c)
        n[3] = self.n("T", c)
        T[0] = self.T_keV(c)
        T[3] = self.T_keV(c)
        return {"psiN_edges": edges, "psiN_centers": c,
                "shell_volume": compute_flux_shell_volumes(eq, n_bins),
                "n": n, "T_keV": T}


@njit(cache=True)
def background_at(psiN, dens, temp, x0, dx, out_n, out_T):
    """Fill out_n[s], out_T[s] (m^-3, J) for the 3 species at psi_N (0 outside
    psi_N > 1)."""
    nx = dens.shape[1]
    s = (psiN - x0) / dx
    if psiN > 1.0 or s >= nx - 1:
        for k in range(3):
            out_n[k] = 0.0
            out_T[k] = 0.0
        return
    if s < 0.0:
        s = 0.0
    k = int(np.floor(s))
    if k > nx - 2:
        k = nx - 2
    w = s - k
    for m in range(3):
        out_n[m] = dens[m, k] * (1.0 - w) + dens[m, k + 1] * w
        out_T[m] = temp[m, k] * (1.0 - w) + temp[m, k + 1] * w
