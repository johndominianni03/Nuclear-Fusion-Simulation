import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class PlasmaVisualizer:
    def __init__(self, df_list: list):
        # Accept a list of pandas DataFrames (one per particle)
        self.df_list = df_list if isinstance(df_list, list) else [df_list]

    def plot_3d_trajectory(
        self,
        output_file: str = "multi_particle_torus.png",
        R0: float = 1.0,
        r0: float = 0.3,
        classifications: list = None,  # Optional: 0 = Passing, 1 = Trapped, -1 = Lost
    ):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # 1. GENERATE THE TOKAMAK TORUS MESH (Vacuum Vessel Wall)
        theta = np.linspace(0, 2 * np.pi, 50)
        phi = np.linspace(0, 2 * np.pi, 50)
        theta, phi = np.meshgrid(theta, phi)

        X_torus = (R0 + r0 * np.cos(theta)) * np.cos(phi)
        Y_torus = (R0 + r0 * np.cos(theta)) * np.sin(phi)
        Z_torus = r0 * np.sin(theta)

        ax.plot_surface(
            X_torus, Y_torus, Z_torus,
            color="gray", alpha=0.12, edgecolor="none"
        )

        # 2. PLOT ALL PARTICLE TRAJECTORIES
        for i, df in enumerate(self.df_list):
            # Default styling (Your original glowing red cloud)
            line_color = "red"
            line_alpha = 0.08
            line_width = 1.0

            # Override styling if classifications are provided
            if classifications is not None and i < len(classifications):
                c_type = classifications[i]
                if c_type == 1:
                    line_color = "gold"        # Trapped Bananas
                    line_alpha = 0.6
                    line_width = 1.2
                elif c_type == 0:
                    line_color = "dodgerblue"  # Passing Particles
                    line_alpha = 0.25
                    line_width = 0.8
                elif c_type == -1:
                    line_color = "crimson"     # Lost to Wall
                    line_alpha = 0.4
                    line_width = 0.8

            ax.plot(
                df["x"], df["y"], df["z"],
                color=line_color,
                alpha=line_alpha,
                linewidth=line_width,
            )
            
            # Start point dots (only show if using default red or if specifically desired)
            if classifications is None:
                ax.scatter(df["x"].iloc[0], df["y"].iloc[0], df["z"].iloc[0], s=10, color="white", alpha=0.3, zorder=5)

        # 3. LABELS & AXIS SCALING
        ax.set_title("Multi-Particle Plasma Beam Confined in Tokamak Vessel", fontsize=14, pad=15)
        ax.set_xlabel("X (m)", fontsize=10)
        ax.set_ylabel("Y (m)", fontsize=10)
        ax.set_zlabel("Z (m)", fontsize=10)

        max_range = R0 + r0 + 0.1
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)

        plt.tight_layout()

        print("[VISUALIZATION] Launching interactive tokamak viewer...")
        plt.show()

        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"[VISUALIZATION] Saved plot to {output_file}")