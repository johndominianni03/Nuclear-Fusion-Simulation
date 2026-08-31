import os

import matplotlib

# --- Headless mode ---------------------------------------------------------
# Set FUSION_HEADLESS=1 to run without an interactive display (CI, batch runs,
# automated tests). matplotlib.use() must be called before pyplot is imported,
# so this block stays above the pyplot import below. Every savefig() still runs
# in headless mode; only the blocking show() is suppressed.
_HEADLESS_OFF = {"", "0", "false", "no", "off"}
FUSION_HEADLESS = os.environ.get("FUSION_HEADLESS", "").strip().lower() not in _HEADLESS_OFF

if FUSION_HEADLESS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _show():
    """plt.show(), or a no-op under FUSION_HEADLESS so runs never block."""
    if FUSION_HEADLESS:
        return
    plt.show()


class PlasmaVisualizer:
    def __init__(self, df_list: list):
        # One DataFrame per particle
        self.df_list = df_list if isinstance(df_list, list) else [df_list]

    def plot_3d_trajectory(
        self,
        output_file: str = "multi_particle_torus.png",
        R0: float = 1.0,
        r0: float = 0.3,
        classifications: list = None,  # Optional: 0 = Passing, 1 = Trapped, -1 = Lost
        wall_contour=None,             # Optional (R, Z) polygon of the psi_edge surface
    ):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # 1. Vacuum vessel wall.
        #
        # When the caller supplies the psi = psi_edge contour, the wall is that surface
        # revolved about the Z axis -- the same boundary check_confinement_flux uses to
        # declare a particle lost. The circular torus below is only a fallback: the real
        # flux surface is not a circle (it reaches |Z| ~ 0.354 and R ~ 1.386 against the
        # circle's 0.3), so drawing the circle put the crimson impact crosses visibly
        # outside the wall they had supposedly struck.
        phi = np.linspace(0, 2 * np.pi, 60)
        if wall_contour is not None:
            R_wall = np.asarray(wall_contour[0], dtype=float)
            Z_wall = np.asarray(wall_contour[1], dtype=float)
            X_torus = R_wall[:, None] * np.cos(phi)[None, :]
            Y_torus = R_wall[:, None] * np.sin(phi)[None, :]
            Z_torus = np.repeat(Z_wall[:, None], phi.size, axis=1)
        else:
            theta = np.linspace(0, 2 * np.pi, 50)
            theta, phi_m = np.meshgrid(theta, np.linspace(0, 2 * np.pi, 50))
            X_torus = (R0 + r0 * np.cos(theta)) * np.cos(phi_m)
            Y_torus = (R0 + r0 * np.cos(theta)) * np.sin(phi_m)
            Z_torus = r0 * np.sin(theta)

        ax.plot_surface(
            X_torus, Y_torus, Z_torus,
            color="gray", alpha=0.12, edgecolor="none"
        )

        # 2. Particle trajectories
        for i, df in enumerate(self.df_list):
            # Default styling (glowing red cloud)
            line_color = "red"
            line_alpha = 0.08
            line_width = 1.0


            # Override styling if classifications are provided
            if classifications is not None and i < len(classifications):
                c_type = classifications[i]
                if c_type == 1:
                    line_color = "gold"        # Neutral beam injections
                    line_alpha = 0.3
                    line_width = 1.2
                elif c_type == 0:
                    line_color = "dodgerblue"  # Passing particles
                    line_alpha = 0.25
                    line_width = 0.8
                elif c_type == -1:
                    line_color = "crimson"     # Wall collisions
                    line_alpha = 0.4
                    line_width = 0.8
                elif c_type == 2:
                    line_color = "magenta"     # 3.5 MeV alphas (trapped banana orbits)
                    # Alpha tracks carry ~10x the vertices of a thermal, so heavier
                    # styling blended over and buried the other three classes. Matched
                    # to the gold/blue/crimson weighting instead.
                    line_alpha = 0.3
                    line_width = 1.0


            ax.plot(
                df["x"], df["y"], df["z"],
                color=line_color,
                alpha=line_alpha,
                linewidth=line_width,
            )
            
            # Start-point dots, default styling only
            if classifications is None:
                ax.scatter(df["x"].iloc[0], df["y"].iloc[0], df["z"].iloc[0], s=10, color="white", alpha=0.3, zorder=5)

        # 3. Labels & axis scaling
        ax.set_title("Multi-Particle Plasma Beam Confined in Tokamak Vessel", fontsize=14, pad=15)
        ax.set_xlabel("X (m)", fontsize=10)
        ax.set_ylabel("Y (m)", fontsize=10)
        ax.set_zlabel("Z (m)", fontsize=10)

        # Frame to the wall that was actually drawn, so the psi_edge surface is not clipped
        if wall_contour is not None:
            max_range = float(np.max(np.abs(R_wall))) + 0.1
        else:
            max_range = R0 + r0 + 0.1
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)


        # --- Legend ---
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='gold', lw=2, label='50 keV NBI Fast Ions'),
            Line2D([0], [0], color='dodgerblue', lw=2, label='1 keV Thermal Core Plasma'),
            Line2D([0], [0], color='magenta', lw=2, label='3.5 MeV Alphas (Banana Orbits)'),
            Line2D([0], [0], color='crimson', lw=2, label='Lost to Divertor Wall'),
            Line2D([0], [0], color='gray', lw=6, alpha=0.35,
                   label='Last Closed Flux Surface ($\\psi_{edge}$)' if wall_contour is not None
                         else 'Vacuum Vessel Wall')
        ]
        ax.legend(handles=legend_elements, loc='upper left', framealpha=0.8)
        

        plt.tight_layout()

        # Save while the figure is still active, then show
        plt.savefig(output_file, dpi=300)
        print(f"[VISUALIZATION] Saved plot to {output_file}")

        if not FUSION_HEADLESS:
            print("[VISUALIZATION] Launching interactive tokamak viewer...")
        _show()
        plt.close()