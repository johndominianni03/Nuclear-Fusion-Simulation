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
from matplotlib.lines import Line2D
from visualizer import PlasmaVisualizer
from physics_engine import interpolate_psi_array


def _show():
    """plt.show(), or a no-op under FUSION_HEADLESS so runs never block."""
    if FUSION_HEADLESS:
        return
    plt.show()


# =======================================================
# TENSOR SCRUBBERS & NAN HANDLERS (Torch -> Numpy)
# =======================================================
def _scrub_1d(lst):
    if lst is None: return None
    arr = np.array([float(x.item() if hasattr(x, 'item') else x) for x in lst])
    return np.nan_to_num(arr, nan=0.0)

def _scrub_grid(grid):
    if grid is None: return None
    if hasattr(grid, 'detach'): arr = grid.detach().cpu().numpy().astype(float)
    else: arr = np.array(grid, dtype=float)
    return np.nan_to_num(arr, nan=0.0)

def _scrub_particles(particles):
    if particles is None: return []
    clean = []
    for p in particles:
        clean.append({
            "history": np.nan_to_num(np.array(p["history"], dtype=float), nan=0.0),
            "type": int(p.get("type", 0)),
            "status": str(p.get("status", "confined"))
        })
    return clean

# =======================================================
# SIMULATION RUN BENCHMARK GRAPH
# =======================================================
def plot_hpc_benchmark(particle_counts, cpu_times, gpu_times):
    print("==================================================")
    print("      SIMULATION RUN BENCHMARK SCALING GRAPH        ")
    print("==================================================")
    plt.figure(figsize=(9, 6))
    
    counts = np.array(particle_counts)
    cpu_t = np.array(cpu_times)
    
    plt.plot(counts, cpu_t, color='crimson', linewidth=2.5, marker='o', markersize=8, label='CPU Baseline (Numba Prange)')
    
    valid_gpu = [t for t in gpu_times if t is not None]
    if len(valid_gpu) == len(counts):
        gpu_t = np.array(valid_gpu)
        plt.plot(counts, gpu_t, color='dodgerblue', linewidth=2.5, marker='s', markersize=8, label='GPU Accelerated (Apple Metal/PyTorch)')
        plt.fill_between(counts, gpu_t, cpu_t, color='lightgray', alpha=0.3, label='Hardware Acceleration Gap')
        
    plt.title("Simulation Performance: Numba JIT (CPU) vs PyTorch (GPU)")
    plt.xlabel("Number of Particles (Log Scale)")
    plt.ylabel("Execution Time for 100 Steps (Seconds - Log Scale)")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", linestyle=':', alpha=0.6)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("benchmark_scaling.png", dpi=300)
    _show()
    plt.close()

# =======================================================
# STEADY STATE DIAGNOSTICS (LIVE MATH RENDERING)
# =======================================================
def run_steady_state_diagnostics(cfg, eq, rho_grid, phi_grid, energy_history_keV, active_particles, total_injected, total_lost, inventory_history, R_phase=None, v_parallel_phase=None, P_fusion_grid=None, alpha_particles=None, lawson_history=None, lawson_first_measured_step=None,
                                 n_ion_grid=None, flux_profiles=None, total_fusion_power_watts=None, species_phase=None):
    
    rho_grid = _scrub_grid(rho_grid)
    phi_grid = _scrub_grid(phi_grid)
    n_ion_grid = _scrub_grid(n_ion_grid)
    P_fusion_grid = _scrub_grid(P_fusion_grid)
    energy_history_keV = _scrub_1d(energy_history_keV)
    inventory_history = _scrub_1d(inventory_history)
    active_particles = _scrub_particles(active_particles)
    alpha_particles = _scrub_particles(alpha_particles)
    
    avg_inventory = np.mean(inventory_history) if len(inventory_history) > 0 else 0
    loss_rate = total_lost / (cfg.reactor_num_steps * cfg.reactor_dt) if (cfg.reactor_num_steps * cfg.reactor_dt) > 0 else 0
    tau_p = (avg_inventory / loss_rate) if loss_rate > 0 else float('inf')

    print("==================================================")
    print("         REACTOR PERFORMANCE DIAGNOSTIC     ")
    print("==================================================")
    print(f"Total Particles Sourced       : {total_injected}")
    print(f"Total Particles Lost (Sink)   : {total_lost}")
    print(f"Estimated Confinement Time (τ): {tau_p:.2e} seconds")
    print(f"Peak Ion Charge Density       : {np.max(rho_grid):.3e} C/m^3")
    if n_ion_grid is not None:
        print(f"Peak Ion Density              : {np.max(n_ion_grid):.3e} m^-3")
    print(f"Force-Balance Potential Range : {np.min(phi_grid):.3e} to {np.max(phi_grid):.3e} V")
    if total_fusion_power_watts is not None:
        print(f"Thermal D-T Fusion Power      : {total_fusion_power_watts:.3e} W")
    print("==================================================")

    R_axis = np.linspace(cfg.R_min, cfg.R_max, cfg.nR)
    Z_axis = np.linspace(cfg.Z_min, cfg.Z_max, cfg.nZ)
    R_mesh, Z_mesh = np.meshgrid(R_axis, Z_axis)

    # The simulation domain is a rectangular solver grid, not the plasma's shape. The real
    # boundary is the last closed flux surface psi = psi_edge that check_confinement_flux
    # tests against -- NOT the circle (R-1.0)^2 + Z^2 = 0.3^2 that used to be assumed
    # here. That circle disagreed with the physics everywhere except the inboard midplane
    # (the true surface reaches |Z| ~ 0.354 and R ~ 1.386), which is why wall strikes were
    # drawn outside the wall they had struck.
    #
    # eq carries the contour and thresholds from initialization._attach_flux_surfaces.
    # Falls back to the legacy circle if a caller passes an eq without them.
    TORUS_R0, TORUS_A_MINOR = 1.0, 0.3
    WALL_R = getattr(eq, "wall_R", None)
    WALL_Z = getattr(eq, "wall_Z", None)
    _view_margin = 0.05

    if WALL_R is not None and WALL_Z is not None:
        _psi_mesh = interpolate_psi_array(
            np.ascontiguousarray(R_mesh.ravel().astype(np.float64)),
            np.ascontiguousarray(Z_mesh.ravel().astype(np.float64)),
            eq.psi_grid, eq.psi_R_min, eq.psi_R_max, eq.psi_Z_min, eq.psi_Z_max,
            eq.psi_nR, eq.psi_nZ
        ).reshape(R_mesh.shape)
        # Outside the last closed flux surface == outside the plasma
        _confinement_mask = _psi_mesh <= eq.psi_edge
        VIEW_XLIM = (float(WALL_R.min()) - _view_margin, float(WALL_R.max()) + _view_margin)
        VIEW_YLIM = (float(WALL_Z.min()) - _view_margin, float(WALL_Z.max()) + _view_margin)
    else:
        _confinement_mask = (R_mesh - TORUS_R0)**2 + Z_mesh**2 > TORUS_A_MINOR**2
        VIEW_XLIM = (TORUS_R0 - TORUS_A_MINOR - _view_margin, TORUS_R0 + TORUS_A_MINOR + _view_margin)
        VIEW_YLIM = (-TORUS_A_MINOR - _view_margin, TORUS_A_MINOR + _view_margin)

    def _draw_wall(axis, label=None):
        # One shared helper so every poloidal plot marks the same boundary.
        if WALL_R is None or WALL_Z is None:
            return
        axis.plot(WALL_R, WALL_Z, color='white', linewidth=2.6, alpha=0.9, zorder=6)
        axis.plot(WALL_R, WALL_Z, color='black', linewidth=1.2, alpha=0.85, zorder=7,
                  linestyle='--', label=label)

    # --- 2D ION DENSITY MAP ---
    # Weighted cloud-in-cell deposition of every live ion (thermal, beam, alpha), each
    # macroparticle standing for cfg.MACRO_WEIGHT ions, computed once from the final state.
    if n_ion_grid is not None:
        plt.figure(figsize=(7, 6))
        n_clean = np.where(_confinement_mask, np.nan, n_ion_grid.T)
        n_max = np.nanmax(n_clean)
        if not np.isfinite(n_max) or n_max <= 0: n_max = 1.0
        contour_n = plt.contourf(R_mesh, Z_mesh, n_clean, levels=np.linspace(0.0, n_max, 40), cmap='plasma')
        plt.colorbar(contour_n, label="Ion density $n_i$ (m$^{-3}$)")
        plt.title("Ion Density, End of Run\n(weighted CIC, all ion species)")
        plt.xlabel("Major Radius R (m)")
        plt.ylabel("Height Z (m)")
        _draw_wall(plt.gca())
        plt.xlim(*VIEW_XLIM)
        plt.ylim(*VIEW_YLIM)
        plt.gca().set_aspect('equal')
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig("ion_density_map.png", dpi=300)
        _show()
        plt.close()

    # --- ELECTROSTATIC POTENTIAL MAP ---
    # Phi from thermal-ion radial force balance, E_r = grad(p_i) / (e n_i), quasi-neutral,
    # no rotation; Phi = 0 at the outer edge of the ion profile. This is the field the
    # loop applied over its last FIELD_UPDATE_EVERY steps.
    plt.figure(figsize=(7, 6))
    phi_clean = np.where(_confinement_mask, np.nan, phi_grid.T)
    p_min, p_max = np.nanmin(phi_clean), np.nanmax(phi_clean)
    if not np.isfinite(p_min): p_min = 0.0
    if not np.isfinite(p_max) or p_max <= p_min: p_max = p_min + 1e-9

    contour_phi = plt.contourf(R_mesh, Z_mesh, phi_clean, levels=np.linspace(p_min, p_max, 40), cmap='viridis')
    plt.colorbar(contour_phi, label="Electrostatic Potential $\\Phi$ (V)")
    plt.title("Quasi-Neutral Potential from Ion Pressure Balance\n"
              "$E_r = \\nabla p_i / (e\\,n_i)$, no rotation")
    plt.xlabel("Major Radius R (m)")
    plt.ylabel("Height Z (m)")
    _draw_wall(plt.gca())
    plt.xlim(*VIEW_XLIM)
    plt.ylim(*VIEW_YLIM)
    plt.gca().set_aspect('equal')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig("potential_field_map.png", dpi=300)
    _show()
    plt.close()

    # --- STORED ENERGY WAVEFORM ---
    # energy_history_keV sums the kinetic energy of the bulk-pool macroparticles
    # (thermal + beam; alphas are not included); times MACRO_WEIGHT it is the
    # stored ion kinetic energy of the plasma they represent.
    plt.figure(figsize=(8, 4))
    _w = getattr(cfg, "MACRO_WEIGHT", None)
    if _w:
        plt.plot(np.arange(len(energy_history_keV)) * cfg.reactor_dt * 1e6,
                 energy_history_keV * _w * 1.0e3 * cfg.e_charge / 1.0e3, color='firebrick', linewidth=1.5)
        plt.ylabel("Stored Ion Kinetic Energy (kJ)")
    else:
        plt.plot(np.arange(len(energy_history_keV)) * cfg.reactor_dt * 1e6, energy_history_keV, color='firebrick', linewidth=1.5)
        plt.ylabel("Macroparticle Kinetic Energy (keV)")
    plt.title("Stored Kinetic Energy of Thermal + Beam Ions vs. Time")
    plt.xlabel("Time ($\\mu s$)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plasma_stored_energy_time.png", dpi=300)
    _show()
    plt.close()

    # --- TOKAMAK TRAJECTORIES (2D, axes locked) ---
    cartesian_dfs = []
    classifications = []
    
    fig, ax = plt.subplots(figsize=(8, 8))
    eq.plot_equilibrium(ax=ax)
    
    # --- DRAW ORDER: LOST TRACES FIRST, SO THEY RENDER BEHIND THE CORE ---
    # A lost fast ion is sampled every ALPHA_HISTORY_EVERY steps and traces a wide
    # gyro-orbit on its way out, so each carries ~10x the vertices of a thermal track.
    # Drawn last they composite over everything, and a couple of hundred of them were
    # enough to bury thousands of confined blue/gold traces -- the plasma read as wiped
    # out when >99% of it was in fact confined. Sorting lost first puts that ink at the
    # back of the z-order in both this plot and the 3D view (cartesian_dfs is built in
    # this same loop, so the two stay in lockstep).
    #
    # This reorders the list only. Every particle still contributes exactly one entry,
    # and classifications[i] still pairs with cartesian_dfs[i].
    _ordered_particles = sorted(active_particles, key=lambda p: 0 if p["status"] == "lost" else 1)

    for p in _ordered_particles:
        traj = p["history"]
        if len(traj) < 2: continue

        # Colour by FATE first, species second. Keying off p["type"] alone (which only
        # holds the species) left the crimson arm unreachable, so no wall strike was ever
        # drawn in red.
        lost = (p["status"] == "lost")
        cartesian_dfs.append(pd.DataFrame(traj, columns=["x", "y", "z"]))
        # -1 is the visualizer's "lost to wall" class; species only matters if it survived
        classifications.append(-1 if lost else p["type"])

        R_path = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
        Z_path = traj[:, 2]

        if lost:
            ax.plot(R_path, Z_path, color='crimson', alpha=0.30, linewidth=0.7, zorder=1)
            ax.plot(R_path[-1], Z_path[-1], marker='x', color='crimson', markersize=5, alpha=0.8, zorder=2)
        elif p["type"] == 1:
            ax.plot(R_path, Z_path, color='gold', alpha=0.7, linewidth=1.2, zorder=4)
        else:
            ax.plot(R_path, Z_path, color='dodgerblue', alpha=0.2, linewidth=0.8, zorder=3)

    # Framed to the curved confinement boundary, not the full solver grid
    _draw_wall(ax)
    ax.set_xlim(*VIEW_XLIM)
    ax.set_ylim(*VIEW_YLIM)
    ax.set_aspect('equal')

    ax.set_title("Steady-State Reactor Particle Trajectories")
    ax.set_xlabel("Major Radius R (m)")
    ax.set_ylabel("Height Z (m)")

    custom_lines = [
        Line2D([0], [0], color='gold', lw=2, label='50 keV NBI Fast Ions (Injected)'),
        Line2D([0], [0], color='dodgerblue', lw=2, label='1 keV Thermal Core Plasma'),
        Line2D([0], [0], color='crimson', lw=2, label='Lost past last closed flux surface'),
        Line2D([0], [0], color='black', lw=1.4, linestyle='--',
               label='Last Closed Flux Surface ($\\psi_{edge}$)')
    ]
    # zorder above the wall (6/7) and the traces, or the flux-surface line is drawn
    # straight through the legend text now that the view is framed to the taller surface
    _leg = ax.legend(handles=custom_lines, loc='upper right', framealpha=0.95)
    _leg.set_zorder(20)
    plt.grid(True, alpha=0.3)
    plt.savefig("tokamak_reactor_2d.png", dpi=300)
    _show()
    plt.close()

    # --- 3D TOKAMAK RENDERING MOVED ---
    # Runs after section 10 so the alpha trajectories are in cartesian_dfs first. Alphas
    # are the clearest banana orbits in the simulation but were previously drawn only on
    # the 2D alpha plot, never in the 3D torus view.

    # --- FLUX-SURFACE PROFILES ---
    # n, T, p per species on psi_N shells (compute_flux_profiles), weighted, against
    # rho = sqrt(psi_N). Log axes: the alpha temperature is ~1000x the thermal one.
    if flux_profiles is not None:
        rho_c = flux_profiles["rho_centers"]
        species = ((0, "Thermal D", "tab:blue"), (1, "Beam D (50 keV NBI)", "tab:orange"),
                   (2, "Alpha (3.5 MeV birth)", "tab:pink"))
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        for s_idx, s_label, s_color in species:
            n_s = flux_profiles["n"][s_idx]
            T_s = flux_profiles["T_keV"][s_idx]
            p_s = flux_profiles["p"][s_idx]
            has = n_s > 0
            if not np.any(has):
                continue
            axes[0].plot(rho_c[has], n_s[has], marker='o', ms=3, color=s_color, label=s_label)
            okT = np.isfinite(T_s) & (T_s > 0)
            axes[1].plot(rho_c[okT], T_s[okT], marker='o', ms=3, color=s_color, label=s_label)
            okp = p_s > 0
            axes[2].plot(rho_c[okp], p_s[okp], marker='o', ms=3, color=s_color, label=s_label)
        axes[0].set_ylabel("Density $n$ (m$^{-3}$)")
        axes[1].set_ylabel("Temperature $T$ (keV)")
        axes[2].set_ylabel("Pressure $p = nT$ (Pa)")
        for a in axes:
            a.set_yscale('log')
            a.grid(True, which='both', alpha=0.3)
        axes[0].legend(loc='best')
        axes[2].set_xlabel("$\\rho = \\sqrt{\\psi_N}$  (0 = magnetic axis, 1 = last closed flux surface)")
        axes[2].set_xlim(0.0, 1.0)
        beta_txt = ""
        if "beta_avg" in flux_profiles:
            mu0 = 4.0e-7 * np.pi
            V = flux_profiles["shell_volume"]
            beta_th = 2.0 * mu0 * float(np.sum(flux_profiles["p"][0] * V) / np.sum(V)) / cfg.B0**2
            beta_txt = (f"\n$\\langle\\beta\\rangle$ = {flux_profiles['beta_avg']:.2e} all species, "
                        f"{beta_th:.2e} thermal  ($B_0$ = {cfg.B0:g} T)")
        axes[0].set_title("Flux-Surface Profiles at End of Run (weighted)" + beta_txt)
        fig.tight_layout()
        plt.savefig("radial_profiles.png", dpi=300)
        _show()
        plt.close()

    # --- PHASE-SPACE DISTRIBUTION ---
    # Deuterons and alphas on separate panels, each scaled to its own largest |v_par|:
    # alpha speeds (~1.3e7 m/s) are ~40x the thermal ones, so one shared axis either
    # crushes the deuterons onto v_par = 0 or clips the fastest alphas.
    if R_phase is not None and v_parallel_phase is not None and len(R_phase) > 0:
        R_ph = _scrub_1d(R_phase)
        v_ph = _scrub_1d(v_parallel_phase)
        sp = np.asarray(species_phase, dtype=int) if species_phase is not None and len(species_phase) == len(R_ph) \
            else np.zeros(len(R_ph), dtype=int)
        groups = [("Thermal and beam deuterons", ((0, 'darkviolet', 'Thermal D'), (1, 'darkorange', 'Beam D'))),
                  ("Alphas", ((2, '#FF66CC', 'Alpha'),))]
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        for ax_ph, (g_title, members) in zip(axes, groups):
            g_mask = np.isin(sp, [m[0] for m in members])
            for code, color, label in members:
                m = sp == code
                if np.any(m):
                    ax_ph.scatter(R_ph[m], v_ph[m], s=2, color=color, alpha=0.3, label=f"{label} ({int(m.sum()):,})")
            ax_ph.axhline(0, color='black', linewidth=1, linestyle='--')
            ax_ph.set_xlim(*VIEW_XLIM)
            v_max = float(np.max(np.abs(v_ph[g_mask]))) if np.any(g_mask) else 0.0
            if v_max > 0:
                ax_ph.set_ylim(-1.05 * v_max, 1.05 * v_max)
                ax_ph.legend(loc='upper right', markerscale=4)
            ax_ph.set_title(g_title)
            ax_ph.set_xlabel("Major Radius R (m)")
            ax_ph.set_ylabel("$v_\\parallel$ Along the Local Field (m/s)")
            ax_ph.grid(True, linestyle=':', alpha=0.6)
        fig.suptitle("Phase-Space Map (Trapped vs. Passing)")
        fig.tight_layout()
        plt.savefig("phase_space_map.png", dpi=300)
        _show()
        plt.close()

    # --- THERMAL D-T FUSION POWER DENSITY ---
    # P(psi_N) = (n_th/2)^2 <sigma v>(T_th) E_fus from the weighted thermal-ion profiles,
    # mapped through psi_N. 50/50 D-T is assumed (no tritium species); beam ions and
    # alphas are excluded.
    if P_fusion_grid is not None:
        plt.figure(figsize=(7, 6))
        P_clean = np.where(_confinement_mask, np.nan, P_fusion_grid.T)
        p_max = np.nanmax(P_clean)
        if not np.isfinite(p_max) or p_max <= 0: p_max = 1e-9
        contour_pf = plt.contourf(R_mesh, Z_mesh, P_clean, levels=np.linspace(0, p_max, 40), cmap='inferno')
        plt.colorbar(contour_pf, label="Fusion Power Density (W/m$^3$)")
        total_txt = f", total {total_fusion_power_watts:.3g} W" if total_fusion_power_watts is not None else ""
        plt.title("Thermal D-T Fusion Power Density" + total_txt +
                  "\n(50/50 D-T assumed; beam ions and alphas excluded)")
        plt.xlabel("Major Radius R (m)")
        plt.ylabel("Height Z (m)")
        _draw_wall(plt.gca())
        plt.xlim(*VIEW_XLIM)
        plt.ylim(*VIEW_YLIM)
        plt.gca().set_aspect('equal')
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig("fusion_power_density.png", dpi=300)
        _show()
        plt.close()

    # --- ALPHA PARTICLE TRAJECTORIES ---
    if alpha_particles is not None and len(alpha_particles) > 0:
        print("--- RENDERING ALPHA ORBITS ---")
        fig, ax = plt.subplots(figsize=(8, 8))
        eq.plot_equilibrium(ax=ax)

        thermal_count = 0
        nbi_count = 0
        
        # 1. Background particles first (zorder 1 and 2) so they don't cover the alphas
        for p in active_particles:
            traj = p["history"]
            if len(traj) > 1:
                R_path = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
                if p.get("type") == 0 and thermal_count < 50:
                    ax.plot(R_path, traj[:, 2], color='cyan', alpha=0.4, linewidth=1.0, zorder=1)
                    thermal_count += 1
                elif p.get("type") == 1 and nbi_count < 50:
                    ax.plot(R_path, traj[:, 2], color='orange', alpha=0.4, linewidth=1.0, zorder=2)
                    nbi_count += 1
                
        # 2. Alphas on top (zorder 3).
        # The 3D torus view gets only a SAMPLE. Each alpha track carries ~40x the ink of a
        # thermal (~5000 vertices, ~130 m of path over a 10 us run), so handing all ~800
        # to the torus render composites to solid magenta at any opacity and buries the
        # other classes. Strided, not truncated, so the sample stays representative across
        # the whole run instead of only the earliest births.
        ALPHA_3D_SAMPLE = 15
        # An alpha covers ~2.6 toroidal laps per bounce (tau_b ~ 1.25 us), so a full-run
        # track wraps the vessel ~20 times and reads as a shell. Two bounces shows the
        # banana shape.
        ALPHA_3D_BOUNCES = 2
        # This plot samples too. The pink region is the ENSEMBLE ENVELOPE, not one
        # particle's excursion: a single banana is ~14 cm wide against a 30 cm minor
        # radius, so ~1000 at random poloidal phase tile the cross-section into a solid
        # disc. Clipping in TIME doesn't shrink it (median excursion is 10.6 cm after a
        # quarter bounce, 13.9 cm by eight), so the lever is fewer orbits, not shorter
        # ones. A dozen leaves visible gaps between bananas. To narrow the ORBITS instead
        # of the envelope, the physics knobs are cfg.Q_SAFETY_TARGET and cfg.B0 (width
        # scales as q/B0).
        ALPHA_2D_SAMPLE = 12
        _tau_b_steps = int(1.25e-6 / cfg.reactor_dt)
        _alpha_max_pts = max(2, (ALPHA_3D_BOUNCES * _tau_b_steps) // cfg.ALPHA_HISTORY_EVERY)
        _alpha_stride = max(1, len(alpha_particles) // ALPHA_3D_SAMPLE)
        _alpha_stride_2d = max(1, len(alpha_particles) // ALPHA_2D_SAMPLE)
        for a_idx, p in enumerate(alpha_particles):
            traj = p["history"]
            if len(traj) > 1:
                if a_idx % _alpha_stride_2d == 0:
                    R_path = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
                    ax.plot(R_path, traj[:, 2], color='#FF66CC', alpha=0.6, linewidth=1.5, zorder=3)
                # Lost alphas are already in active_particles as class -1; skip them here
                # so the same trajectory isn't drawn twice.
                if p["status"] != "lost" and a_idx % _alpha_stride == 0:
                    clipped = traj[:_alpha_max_pts]
                    cartesian_dfs.append(pd.DataFrame(clipped, columns=["x", "y", "z"]))
                    classifications.append(2)

        # Framed to the actual curved confinement boundary, not the full rectangular solver grid.
        _draw_wall(ax)
        ax.set_xlim(*VIEW_XLIM)
        ax.set_ylim(*VIEW_YLIM)
        ax.set_aspect('equal')
                
        ax.set_title("3.5 MeV Alpha Orbits vs Thermal Background")
        ax.set_xlabel("Major Radius R (m)")
        ax.set_ylabel("Height Z (m)")
        
        custom_lines = [
            Line2D([0], [0], color='#FF66CC', lw=2, label='3.5 MeV Helium-4 (Alpha)'),
            Line2D([0], [0], color='cyan', lw=2, label='Thermal Background Ion'),
            Line2D([0], [0], color='orange', lw=2, label='50 keV NBI Fast Ion')
        ]
        _leg_a = ax.legend(handles=custom_lines, loc='upper right', framealpha=0.95)
        _leg_a.set_zorder(20)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("alpha_orbits.png", dpi=300)
        _show()
        plt.close()

    # --- 3D TOKAMAK RENDERING ---
    # Runs here so cartesian_dfs already carries the alpha orbits.
    print("--- RENDERING 3D STEADY-STATE TOKAMAK ---")
    try:
        viz = PlasmaVisualizer(cartesian_dfs)
        # Hand the 3D view the same psi_edge surface the 2D plots and the loss test use,
        # so the vessel it draws is the boundary particles are actually flagged against.
        _wall = (WALL_R, WALL_Z) if (WALL_R is not None and WALL_Z is not None) else None
        viz.plot_3d_trajectory(output_file="tokamak_reactor_3d.png",
                               classifications=classifications, wall_contour=_wall)
    except Exception as e:
        print(f"[WARNING] Skipping 3D visualization due to missing dependency or window issue: {e}")

    # =======================================================
    # LAWSON TRIPLE PRODUCT
    # =======================================================
    if lawson_history is not None and len(lawson_history) > 0:
        fig, ax = plt.subplots(figsize=(9, 5))

        lawson_arr = _scrub_1d(lawson_history)
        time_array = np.arange(len(lawson_arr)) * cfg.reactor_dt * 1e6

        # Before the first particle loss tau_E is a fixed placeholder, not a
        # measurement, so that segment is not drawn.
        if lawson_first_measured_step is not None:
            k0 = int(lawson_first_measured_step)
            ax.plot(time_array[k0:], lawson_arr[k0:], color='firebrick', linewidth=2.5,
                    label='Simulated Triple Product ($n_e T_i \\tau_E$)')
        else:
            ax.text(0.5, 0.5, "No particle lost: confinement time not measured",
                    transform=ax.transAxes, ha='center', va='center', color='dimgray')
        ax.set_xlim(0.0, len(lawson_arr) * cfg.reactor_dt * 1e6)
        ax.axhline(cfg.lawson_target, color='mediumseagreen', linestyle='--', linewidth=2, label=f'Reference value ({cfg.lawson_target:.1e})')

        ax.set_title("Lawson Triple Product Evolution")
        ax.set_xlabel("Simulation Time ($\\mu s$)")
        ax.set_ylabel("Triple Product ($keV \\cdot s \\cdot m^{-3}$)")
        ax.set_yscale('log')
        ax.grid(True, which="both", linestyle=':', alpha=0.6)
        # 'best': the trace ends low-right in a production run and high-right in
        # a short one, so no fixed corner stays clear of it.
        ax.legend(loc='best')

        plt.tight_layout()
        plt.savefig("lawson_triple_product.png", dpi=300)
        _show()
        plt.close()
