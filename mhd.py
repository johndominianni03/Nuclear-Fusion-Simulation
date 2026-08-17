import numpy as np
import matplotlib.pyplot as plt

class GradShafranovSolver:
    def __init__(self, R0=1.0, r0=0.3, nx=100, nz=100):
        self.R0 = R0  # Major radius (m)
        self.r0 = r0  # Minor radius (m)
        self.nx = nx  # Grid points along R
        self.nz = nz  # Grid points along Z

        # 1. 2D cross-section boundaries
        self.R_min = R0 - r0
        self.R_max = R0 + r0
        self.Z_min = -r0
        self.Z_max = r0

        # 2. 1D coordinate arrays
        self.R_grid_1d = np.linspace(self.R_min, self.R_max, self.nx)
        self.Z_grid_1d = np.linspace(self.Z_min, self.Z_max, self.nz)

        # 3. 2D mesh
        self.R, self.Z = np.meshgrid(self.R_grid_1d, self.Z_grid_1d)

        # 4. Magnetic flux psi
        self.psi = np.zeros((nz, nx))

    def display_grid_info(self):
        print(f"[MHD ENGINE] 2D Grid Initialized: {self.nx}x{self.nz} resolution.")
        print(f"[MHD ENGINE] R-range: [{self.R_min:.2f}m, {self.R_max:.2f}m]")
        print(f"[MHD ENGINE] Z-range: [{self.Z_min:.2f}m, {self.Z_max:.2f}m]")

# Quick test execution
if __name__ == "__main__":
    gs_solver = GradShafranovSolver(R0=1.0, r0=0.3, nx=100, nz=100)
    gs_solver.display_grid_info()

    # Visual verification
    plt.figure(figsize=(8, 6))

    # Grid points of the 2D cross-section
    plt.scatter(gs_solver.R, gs_solver.Z, s=1, color='blue', alpha=0.5, label='Grid Nodes')

    # Reference circle for the vacuum vessel wall
    theta = np.linspace(0, 2 * np.pi, 200)
    wall_R = gs_solver.R0 + gs_solver.r0 * np.cos(theta)
    wall_Z = gs_solver.r0 * np.sin(theta)
    plt.plot(wall_R, wall_Z, color='red', linewidth=2, label='Vessel Wall (r0)')
    
    plt.title("Tokamak 2D Cross-Sectional Grid (R-Z Plane)", fontsize=12)
    plt.xlabel("Major Radius R (m)", fontsize=10)
    plt.ylabel("Vertical Height Z (m)", fontsize=10)
    plt.axvline(gs_solver.R0, color='gray', linestyle='--', alpha=0.7, label='Magnetic Axis (R0)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    print("[VISUALIZATION] Rendering grid verification window...")
    plt.show()