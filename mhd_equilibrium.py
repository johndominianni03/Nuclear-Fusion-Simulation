import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

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

    def solve_grad_shafranov(self, max_iters=2000, omega=1.3, tolerance=1e-6):
        """Solves Grad-Shafranov on the (R, Z) grid via Successive Over-Relaxation."""
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

        self.psi = psi
        self._compute_magnetic_fields()
        self._build_interpolators()
        return self.psi

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