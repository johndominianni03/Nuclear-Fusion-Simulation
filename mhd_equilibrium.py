import hashlib
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# =======================================================
# GRAD-SHAFRANOV DISK CACHE
# =======================================================
# The solved psi is a pure function of the grid geometry and the SOR controls --
# there is no run-to-run variation -- but the pure-Python SOR loop costs several
# seconds, and a single run solves twice (initialize_reactor and
# initialize_oscillation_test both build an MHDEquilibrium). So the result is
# memoized to disk, keyed by a hash of every parameter it depends on.
#
# Set FUSION_NO_GS_CACHE=1 to force a fresh solve; the cache file is then
# rewritten from that solve's result.
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")

# Bumped whenever the stored format or the discretization changes, so stale
# entries from an older solver can never be loaded as if they were current.
_CACHE_FORMAT = "gs-v1"

_CACHE_OFF = {"", "0", "false", "no", "off"}
GS_CACHE_DISABLED = (
    os.environ.get("FUSION_NO_GS_CACHE", "").strip().lower() not in _CACHE_OFF
)


# =======================================================
# SOR BACKEND SELECTION
# =======================================================
# "python" is the reference solver and the default. "numba" JIT-compiles the
# same sweep with fastmath=True, which reassociates floating-point
# arithmetic -- so it is FAST BUT NOT BIT-IDENTICAL to the reference. It is
# opt-in only:
#
#     FUSION_GS_BACKEND=numba python main.py
#     eq.solve_grad_shafranov(backend="numba")
#
# Compare the two before trusting it:  python tools/gs_backend_compare.py
#
# The backend is part of the cache key, so a numba-solved psi is never
# served to a run that asked for the Python solver, or vice versa.
DEFAULT_GS_BACKEND = "python"
_VALID_BACKENDS = ("python", "numba")

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:                                # numba is optional
    NUMBA_AVAILABLE = False


def _resolve_backend(backend):
    """Explicit argument wins, then FUSION_GS_BACKEND, then the default."""
    if backend is None:
        backend = os.environ.get("FUSION_GS_BACKEND", "").strip().lower()
    if not backend:
        backend = DEFAULT_GS_BACKEND
    if backend not in _VALID_BACKENDS:
        print(f"[GS SOLVER] Unknown backend {backend!r}; "
              f"falling back to {DEFAULT_GS_BACKEND!r}. "
              f"Valid: {', '.join(_VALID_BACKENDS)}.")
        return DEFAULT_GS_BACKEND
    if backend == "numba" and not NUMBA_AVAILABLE:
        print("[GS SOLVER] numba backend requested but numba is not "
              "installed; falling back to the Python solver.")
        return DEFAULT_GS_BACKEND
    return backend


if NUMBA_AVAILABLE:
    @njit(fastmath=True, cache=True)
    def _sor_sweep_numba(psi, RR, dR, dZ, omega):
        """One in-place SOR sweep; returns max |change| over the sweep.

        Deliberately serial and in row-major order: SOR reads the values
        its own sweep has already updated, so parallelising it (prange)
        would change the answer, not just its rounding.

        fastmath=True lets LLVM reassociate these expressions, so results
        differ from the Python sweep in the last bits.
        """
        nz, nx = psi.shape
        inv_dR2 = 1.0 / (dR * dR)
        inv_dZ2 = 1.0 / (dZ * dZ)
        denom = 2.0 * (inv_dR2 + inv_dZ2)
        max_diff = 0.0

        for i in range(1, nz - 1):
            for j in range(1, nx - 1):
                R_val = RR[i, j]

                source_term = -(R_val ** 2)

                factor_R_minus = inv_dR2 - 1.0 / (2.0 * R_val * dR)
                factor_R_plus = inv_dR2 + 1.0 / (2.0 * R_val * dR)

                psi_new = (
                    factor_R_plus * psi[i, j + 1]
                    + factor_R_minus * psi[i, j - 1]
                    + inv_dZ2 * (psi[i + 1, j] + psi[i - 1, j])
                    - source_term
                ) / denom

                psi_prev = psi[i, j]
                psi[i, j] = (1.0 - omega) * psi_prev + omega * psi_new

                diff = abs(psi[i, j] - psi_prev)
                if diff > max_diff:
                    max_diff = diff

        return max_diff


class MHDEquilibrium:
    """
    Solves the 2D Grad-Shafranov equation and provides 3D field evaluation plus
    spatial derivatives (grad B, curvature) for kinetic particle pushing.
    """
    def __init__(self, nx=60, nz=60, R_min=0.5, R_max=1.5, Z_min=-0.5, Z_max=0.5, B0=1.0, R0=1.0):
        self.nx = nx
        self.nz = nz
        self.R_min = R_min
        self.R_max = R_max
        self.Z_min = Z_min
        self.Z_max = Z_max
        
        # Reference toroidal field & major radius
        self.B0 = B0
        self.R0 = R0

        # 1D and 2D spatial grids
        self.R_1d = np.linspace(R_min, R_max, nx)
        self.Z_1d = np.linspace(Z_min, Z_max, nz)
        self.RR, self.ZZ = np.meshgrid(self.R_1d, self.Z_1d)
        
        self.dR = self.R_1d[1] - self.R_1d[0]
        self.dZ = self.Z_1d[1] - self.Z_1d[0]

        # How psi was last obtained ("cache-hit" / "solved") and by which
        # backend. Read by the profiler only.
        self.last_solve_source = None
        self.last_solve_backend = None

        # Field arrays, populated by solve_grad_shafranov()
        self.psi = None
        self.BR_grid = None
        self.BZ_grid = None
        self.Bphi_grid = None
        self.B_mag_grid = None

        # Interpolators for fast spatial queries
        self.interp_BR = None
        self.interp_BZ = None
        self.interp_Bphi = None
        self.interp_Bmag = None

    # -------------------------------------------------------------------
    # Disk cache
    # -------------------------------------------------------------------

    def _cache_key(self, max_iters, omega, tolerance, backend):
        """Hash of every input the solved psi depends on.

        B0 and R0 do not enter the SOR loop itself (they first appear in
        _compute_magnetic_fields), but they are part of the equilibrium's
        identity, so they are keyed on as well -- a cache entry can then
        never be shared between two differently-configured reactors.
        `backend` distinguishes solvers that are not bit-identical to each
        other, so results from one can never be served to the other.
        """
        payload = "|".join(repr(v) for v in (
            _CACHE_FORMAT,
            backend,
            self.nx, self.nz,
            self.R_min, self.R_max,
            self.Z_min, self.Z_max,
            self.B0, self.R0,
            max_iters, omega, tolerance,
        ))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _cache_path(self, key):
        return os.path.join(CACHE_DIR, f"gs_{key}.npy")

    def _load_cached_psi(self, path):
        """Return a valid cached psi, or None if unusable (caller re-solves)."""
        try:
            psi = np.load(path)
        except Exception as exc:                       # truncated / corrupt file
            print(f"[GS CACHE] UNREADABLE ({exc}); re-solving.")
            return None

        expected = (self.nz, self.nx)
        if psi.shape != expected:
            print(f"[GS CACHE] STALE: shape {psi.shape} != expected {expected}; "
                  "re-solving.")
            return None
        if psi.dtype != np.float64:
            print(f"[GS CACHE] STALE: dtype {psi.dtype} != float64; re-solving.")
            return None
        if not np.all(np.isfinite(psi)):
            print("[GS CACHE] STALE: cached psi contains NaN/Inf; re-solving.")
            return None
        return psi

    def _store_cached_psi(self, path, psi):
        """Write psi to the cache atomically, so an interrupted run cannot
        leave a half-written file that a later run would load."""
        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            tmp = f"{path}.tmp{os.getpid()}"
            np.save(tmp, psi, allow_pickle=False)
            # np.save appends .npy to a name that lacks it.
            if not tmp.endswith(".npy"):
                tmp += ".npy"
            os.replace(tmp, path)
            print(f"[GS CACHE] WROTE {os.path.relpath(path, CACHE_DIR)} "
                  f"({psi.nbytes / 1024:.1f} KiB)")
        except Exception as exc:                       # read-only dir, full disk
            print(f"[GS CACHE] WRITE FAILED ({exc}); continuing without cache.")

    # -------------------------------------------------------------------
    # Solver
    # -------------------------------------------------------------------

    def solve_grad_shafranov(self, max_iters=2000, omega=1.3, tolerance=1e-6,
                              backend=None):
        """Solves Grad-Shafranov on the (R, Z) grid, via cache or SOR iteration.

        The psi returned by a cache hit is bit-identical to the one the SOR
        loop produced when the entry was written. _compute_magnetic_fields and
        _build_interpolators run either way, so a hit and a miss leave the
        object in exactly the same state.

        backend: None (default) resolves to FUSION_GS_BACKEND, else "python".
        "numba" is an opt-in fast path for cold caches and is NOT
        bit-identical to "python" -- see the note at the top of this module.
        """
        backend = _resolve_backend(backend)
        key = self._cache_key(max_iters, omega, tolerance, backend=backend)
        path = self._cache_path(key)

        if GS_CACHE_DISABLED:
            print("[GS CACHE] DISABLED by FUSION_NO_GS_CACHE; forcing a solve.")
        elif os.path.isfile(path):
            psi = self._load_cached_psi(path)
            if psi is not None:
                print(f"[GS CACHE] HIT  gs_{key}.npy -- skipping SOR iteration.")
                # Provenance for the profiler: a cold miss and a warm hit
                # differ by seconds, and the report should say which it was
                # rather than leave it to be misread as loop-adjacent cost.
                self.last_solve_source = "cache-hit"
                self.last_solve_backend = backend
                self.psi = psi
                self._compute_magnetic_fields()
                self._build_interpolators()
                return self.psi
        else:
            print(f"[GS CACHE] MISS gs_{key}.npy -- solving from scratch.")

        if backend == "numba":
            psi = self._sor_solve_numba(max_iters, omega, tolerance)
        else:
            psi = self._sor_solve_python(max_iters, omega, tolerance)

        self.last_solve_source = "solved"
        self.last_solve_backend = backend

        self._store_cached_psi(path, psi)

        self.psi = psi
        self._compute_magnetic_fields()
        self._build_interpolators()
        return self.psi

    def _sor_solve_python(self, max_iters, omega, tolerance):
        """Pure-Python SOR sweep. The reference implementation: whatever else
        is added, this is what 'bit-identical' is measured against."""
        psi = np.zeros((self.nz, self.nx))

        print("Solving Grad-Shafranov equilibrium via SOR iteration...")
        for iteration in range(max_iters):
            psi_old = psi.copy()
            max_diff = 0.0

            for i in range(1, self.nz - 1):
                for j in range(1, self.nx - 1):
                    R_val = self.RR[i, j]
                    
                    # Toroidal current source (J_phi ~ R)
                    source_term = - (R_val ** 2)

                    # 5-point finite difference operator
                    factor_R_minus = 1.0 / (self.dR**2) - 1.0 / (2.0 * R_val * self.dR)
                    factor_R_plus = 1.0 / (self.dR**2) + 1.0 / (2.0 * R_val * self.dR)
                    factor_Z = 1.0 / (self.dZ**2)
                    denom = 2.0 * (1.0 / (self.dR**2) + 1.0 / (self.dZ**2))

                    psi_new = (
                        factor_R_plus * psi[i, j + 1]
                        + factor_R_minus * psi[i, j - 1]
                        + factor_Z * (psi[i + 1, j] + psi[i - 1, j])
                        - source_term
                    ) / denom

                    # SOR update
                    psi[i, j] = (1.0 - omega) * psi[i, j] + omega * psi_new

                    diff = abs(psi[i, j] - psi_old[i, j])
                    if diff > max_diff:
                        max_diff = diff

            if iteration % 200 == 0:
                print(f"  Iteration {iteration:4d} | Max Delta: {max_diff:.6e}")

            if max_diff < tolerance:
                print(f"  Converged successfully at iteration {iteration}!")
                break

        return psi

    def _sor_solve_numba(self, max_iters, omega, tolerance):
        """JIT fast path. Same sweep order and convergence test as
        _sor_solve_python, but fastmath reordering means the result agrees
        only to floating-point tolerance, not bit-for-bit."""
        psi = np.zeros((self.nz, self.nx))
        RR = np.ascontiguousarray(self.RR, dtype=np.float64)

        print("Solving Grad-Shafranov equilibrium via SOR iteration "
              "[numba fastmath -- not bit-identical to the Python solver]...")
        for iteration in range(max_iters):
            max_diff = _sor_sweep_numba(psi, RR, self.dR, self.dZ, omega)

            if iteration % 200 == 0:
                print(f"  Iteration {iteration:4d} | Max Delta: {max_diff:.6e}")

            if max_diff < tolerance:
                print(f"  Converged successfully at iteration {iteration}!")
                break

        return psi

    def _compute_magnetic_fields(self):
        """
        Poloidal and toroidal field from psi:
        B_R = -(1/R) dpsi/dZ,  B_Z = (1/R) dpsi/dR,  B_phi = B0 * R0 / R
        """
        # Central differences on interior grid points
        dpsi_dZ, dpsi_dR = np.gradient(self.psi, self.dZ, self.dR)

        self.BR_grid = - (1.0 / self.RR) * dpsi_dZ
        self.BZ_grid = (1.0 / self.RR) * dpsi_dR
        self.Bphi_grid = (self.B0 * self.R0) / self.RR

        self.B_mag_grid = np.sqrt(self.BR_grid**2 + self.BZ_grid**2 + self.Bphi_grid**2)

    def _build_interpolators(self):
        """2D interpolators over the (Z, R) grid."""
        bounds_error = False
        fill_value = None

        self.interp_BR = RegularGridInterpolator(
            (self.Z_1d, self.R_1d), self.BR_grid, bounds_error=bounds_error, fill_value=fill_value
        )
        self.interp_BZ = RegularGridInterpolator(
            (self.Z_1d, self.R_1d), self.BZ_grid, bounds_error=bounds_error, fill_value=fill_value
        )
        self.interp_Bphi = RegularGridInterpolator(
            (self.Z_1d, self.R_1d), self.Bphi_grid, bounds_error=bounds_error, fill_value=fill_value
        )
        self.interp_Bmag = RegularGridInterpolator(
            (self.Z_1d, self.R_1d), self.B_mag_grid, bounds_error=bounds_error, fill_value=fill_value
        )

    # -------------------------------------------------------------------
    # API (queried by physics_engine.py every time step)
    # -------------------------------------------------------------------

    def get_B_field(self, pos):
        """B = (Bx, By, Bz) at position (x, y, z)."""
        x, y, z = pos[0], pos[1], pos[2]
        R = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

        # Cylindrical components (B_R, B_Z, B_phi)
        pt = np.array([[z, R]])
        BR = self.interp_BR(pt)[0]
        BZ = self.interp_BZ(pt)[0]
        Bphi = self.interp_Bphi(pt)[0]

        # Cylindrical -> Cartesian
        Bx = BR * np.cos(phi) - Bphi * np.sin(phi)
        By = BR * np.sin(phi) + Bphi * np.cos(phi)
        Bz = BZ

        return np.array([Bx, By, Bz])

    def get_B_mag(self, pos):
        """Scalar |B| at position (x, y, z)."""
        x, y, z = pos[0], pos[1], pos[2]
        R = np.sqrt(x**2 + y**2)
        return float(self.interp_Bmag(np.array([[z, R]]))[0])

    def get_b_unit(self, pos):
        """Unit vector b_hat = B / |B| at position (x, y, z)."""
        B_vec = self.get_B_field(pos)
        B_mag = np.linalg.norm(B_vec)
        return B_vec / B_mag if B_mag > 0 else np.zeros(3)

    def get_grad_B(self, pos, h=1e-4):
        """nabla(|B|) by central finite differences."""
        grad = np.zeros(3)
        for i in range(3):
            pos_plus = pos.copy()
            pos_minus = pos.copy()
            pos_plus[i] += h
            pos_minus[i] -= h
            grad[i] = (self.get_B_mag(pos_plus) - self.get_B_mag(pos_minus)) / (2.0 * h)
        return grad

    def get_curvature(self, pos, h=1e-4):
        """Field-line curvature kappa = (b_hat . nabla) b_hat, for curvature drift."""
        b0 = self.get_b_unit(pos)

        # db_hat/dx, db_hat/dy, db_hat/dz
        db_dx = (self.get_b_unit(pos + np.array([h, 0, 0])) - self.get_b_unit(pos - np.array([h, 0, 0]))) / (2.0 * h)
        db_dy = (self.get_b_unit(pos + np.array([0, h, 0])) - self.get_b_unit(pos - np.array([0, h, 0]))) / (2.0 * h)
        db_dz = (self.get_b_unit(pos + np.array([0, 0, h])) - self.get_b_unit(pos - np.array([0, 0, h]))) / (2.0 * h)

        # kappa = b_x * db/dx + b_y * db/dy + b_z * db/dz
        kappa = b0[0] * db_dx + b0[1] * db_dy + b0[2] * db_dz
        return kappa

    def plot_equilibrium(self, ax=None):
        """
        Plots poloidal flux surfaces psi(R, Z). Renders standalone, or draws onto a
        supplied Axes for overlays.
        """
        show_plot = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
            show_plot = True

        # Poloidal flux contours
        cs = ax.contourf(self.RR, self.ZZ, self.psi, levels=25, cmap='plasma', alpha=0.85)
        ax.contour(self.RR, self.ZZ, self.psi, levels=12, colors='k', linewidths=0.5, alpha=0.6)

        ax.set_xlabel('Major Radius R [m]')
        ax.set_ylabel('Height Z [m]')
        ax.set_title('Tokamak MHD Equilibrium & Poloidal Flux Surfaces')
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.4)

        if show_plot:
            plt.colorbar(cs, ax=ax, label=r'Poloidal Flux $\psi(R, Z)$')
            plt.tight_layout()
            plt.show()

        return ax


if __name__ == "__main__":
    # Test run
    eq = MHDEquilibrium()
    eq.solve_grad_shafranov()
    
    # Test point R=1.2m, phi=0, Z=0.1m
    test_pos = np.array([1.2, 0.0, 0.1])
    print("\n--- Magnetic Field API Evaluation Test ---")
    print(f"Position (X, Y, Z)     : {test_pos}")
    print(f"B Field Vector (Bx,By,Bz): {eq.get_B_field(test_pos)}")
    print(f"B Magnitude |B|        : {eq.get_B_mag(test_pos):.4f} T")
    print(f"Grad |B|               : {eq.get_grad_B(test_pos)}")
    print(f"Curvature Vector kappa : {eq.get_curvature(test_pos)}")

    eq.plot_equilibrium()