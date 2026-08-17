import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from visualizer import PlasmaVisualizer

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
# HPC BENCHMARK GRAPH
# =======================================================
def plot_hpc_benchmark(particle_counts, cpu_times, gpu_times):
    print("==================================================")
    print("      WEEK 23: HPC BENCHMARK SCALING GRAPH        ")
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
        
    plt.title("Week 23: HPC Drag Race (Particle Push Scaling)")
    plt.xlabel("Number of Particles (Log Scale)")
    plt.ylabel("Execution Time for 100 Steps (Seconds - Log Scale)")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", linestyle=':', alpha=0.6)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig("benchmark_scaling.png", dpi=300)
    plt.show()

# =======================================================
# STEADY STATE DIAGNOSTICS (LIVE MATH RENDERING)
# =======================================================
def run_steady_state_diagnostics(cfg, eq, rho_grid, phi_grid, energy_history_keV, active_particles, total_injected, total_lost, inventory_history, R_centers=None, density_profile=None, pressure_profile=None, R_phase=None, v_parallel_phase=None, instability_amp_history=None, P_fusion_grid=None, alpha_particles=None, alpha_power_history=None, ext_power_history=None, brem_power_history=None, cyc_power_history=None, q_sci_history=None, q_eng_history=None, lawson_history=None):
    
    rho_grid = _scrub_grid(rho_grid)
    phi_grid = _scrub_grid(phi_grid)
    P_fusion_grid = _scrub_grid(P_fusion_grid)
    energy_history_keV = _scrub_1d(energy_history_keV)
    inventory_history = _scrub_1d(inventory_history)
    active_particles = _scrub_particles(active_particles)
    alpha_particles = _scrub_particles(alpha_particles)
    
    avg_inventory = np.mean(inventory_history) if len(inventory_history) > 0 else 0
    loss_rate = total_lost / (cfg.reactor_num_steps * cfg.reactor_dt) if (cfg.reactor_num_steps * cfg.reactor_dt) > 0 else 0
    tau_p = (avg_inventory / loss_rate) if loss_rate > 0 else float('inf')

    print("==================================================")
    print("      WEEK 11: REACTOR PERFORMANCE DIAGNOSTIC     ")
    print("==================================================")
    print(f"Total Particles Sourced       : {total_injected}")
    print(f"Total Particles Lost (Sink)   : {total_lost}")
    print(f"Estimated Confinement Time (τ): {tau_p:.2e} seconds")
    print(f"Peak Charge Density           : {np.max(rho_grid):.3e} C/m^3")
    print(f"Peak Electrostatic Potential  : {np.max(phi_grid):.3e} V")
    print("==================================================")

    R_axis = np.linspace(cfg.R_min, cfg.R_max, cfg.nR)
    Z_axis = np.linspace(cfg.Z_min, cfg.Z_max, cfg.nZ)
    R_mesh, Z_mesh = np.meshgrid(R_axis, Z_axis)

    # The simulation domain is a rectangular solver grid, not the plasma's shape -- the
    # real boundary is the curved flux surface (R-1.0)^2 + Z^2 = 0.3^2 that
    # check_confinement uses. These frame the plots below around that geometry instead of
    # drawing a box around a much smaller region.
    TORUS_R0, TORUS_A_MINOR = 1.0, 0.3
    _confinement_mask = (R_mesh - TORUS_R0)**2 + Z_mesh**2 > TORUS_A_MINOR**2
    _view_margin = 0.05
    VIEW_XLIM = (TORUS_R0 - TORUS_A_MINOR - _view_margin, TORUS_R0 + TORUS_A_MINOR + _view_margin)
    VIEW_YLIM = (-TORUS_A_MINOR - _view_margin, TORUS_A_MINOR + _view_margin)

    # --- 1. 2D CHARGE DENSITY MAP (percentile clipped) ---
    plt.figure(figsize=(7, 6))
    rho_clean = np.where(_confinement_mask, np.nan, rho_grid.T)
    p_min, p_max = np.nanpercentile(rho_clean, 2), np.nanpercentile(rho_clean, 98)
    if not np.isfinite(p_min): p_min = 0.0
    if not np.isfinite(p_max) or p_max <= p_min: p_max = p_min + 1e-9

    contour_rho = plt.contourf(R_mesh, Z_mesh, rho_clean, levels=np.linspace(p_min, p_max, 40), cmap='plasma', extend='both')
    plt.colorbar(contour_rho, label="Charge Density $\\rho$ (C/m$^3$)")
    plt.title("Week 11: Poloidal Charge Density Distribution (CIC)")
    plt.xlabel("Major Radius R (m)")
    plt.ylabel("Height Z (m)")
    plt.xlim(*VIEW_XLIM)
    plt.ylim(*VIEW_YLIM)
    plt.gca().set_aspect('equal')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig("charge_density_map.png", dpi=300)
    plt.show()

    # --- 2. ELECTROSTATIC POTENTIAL MAP (percentile clipped) ---
    plt.figure(figsize=(7, 6))
    phi_clean = np.where(_confinement_mask, np.nan, phi_grid.T)
    p_min, p_max = np.nanpercentile(phi_clean, 2), np.nanpercentile(phi_clean, 98)
    if not np.isfinite(p_min): p_min = 0.0
    if not np.isfinite(p_max) or p_max <= p_min: p_max = p_min + 1e-9

    contour_phi = plt.contourf(R_mesh, Z_mesh, phi_clean, levels=np.linspace(p_min, p_max, 40), cmap='viridis', extend='both')
    plt.colorbar(contour_phi, label="Electrostatic Potential $\\phi$ (Volts)")
    plt.title("Week 11: Self-Consistent Potential Field (Poisson Solver)")
    plt.xlabel("Major Radius R (m)")
    plt.ylabel("Height Z (m)")
    plt.xlim(*VIEW_XLIM)
    plt.ylim(*VIEW_YLIM)
    plt.gca().set_aspect('equal')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig("potential_field_map.png", dpi=300)
    plt.show()

    # --- 3. STORED ENERGY WAVEFORM ---
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(energy_history_keV)) * cfg.reactor_dt * 1e6, energy_history_keV, color='firebrick', linewidth=1.5)
    plt.title("Stored Plasma Energy vs. Time (Full Lorentz Push)")
    plt.xlabel("Time ($\\mu s$)")
    plt.ylabel("Stored Energy (keV)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plasma_stored_energy_time.png", dpi=300)
    plt.show()

    # --- 4. TOKAMAK TRAJECTORIES (2D, axes locked) ---
    cartesian_dfs = []
    classifications = []
    
    fig, ax = plt.subplots(figsize=(8, 8))
    eq.plot_equilibrium(ax=ax)
    
    for p in active_particles:
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
            ax.plot(R_path, Z_path, color='crimson', alpha=0.45, linewidth=0.9)
            ax.plot(R_path[-1], Z_path[-1], marker='x', color='crimson', markersize=5, alpha=0.8)
        elif p["type"] == 1:
            ax.plot(R_path, Z_path, color='gold', alpha=0.7, linewidth=1.2)
        else:
            ax.plot(R_path, Z_path, color='dodgerblue', alpha=0.2, linewidth=0.8)

    # Framed to the curved confinement boundary, not the full solver grid
    ax.set_xlim(*VIEW_XLIM)
    ax.set_ylim(*VIEW_YLIM)
    ax.set_aspect('equal')
    
    ax.set_title("Week 11: Steady-State Reactor with Self-Consistent PIC")
    ax.set_xlabel("Major Radius R (m)")
    ax.set_ylabel("Height Z (m)")
    
    custom_lines = [
        Line2D([0], [0], color='gold', lw=2, label='50 keV NBI Fast Ions (Injected)'),
        Line2D([0], [0], color='dodgerblue', lw=2, label='1 keV Thermal Core Plasma'),
        Line2D([0], [0], color='crimson', lw=2, label='Particles Lost to Divertor Wall')
    ]
    ax.legend(handles=custom_lines, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.savefig("tokamak_reactor_2d.png", dpi=300)
    plt.show()

    # --- 5. (3D TOKAMAK RENDERING MOVED) ---
    # Runs after section 10 so the alpha trajectories are in cartesian_dfs first. Alphas
    # are the clearest banana orbits in the simulation but were previously drawn only on
    # the 2D alpha plot, never in the 3D torus view.

    # --- 6. 1D RADIAL FLUID PROFILES ---
    if R_centers is not None and density_profile is not None and pressure_profile is not None:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        
        color_dens = 'tab:blue'
        ax1.set_xlabel("Major Radius R (m)")
        ax1.set_ylabel("Electron Density $n_e$ ($m^{-3}$)", color=color_dens)
        ax1.plot(_scrub_1d(R_centers), _scrub_1d(density_profile), color=color_dens, linewidth=2, label="Density Profile")
        ax1.tick_params(axis='y', labelcolor=color_dens)
        ax1.grid(True, alpha=0.3)
        
        ax2 = ax1.twinx()
        color_pres = 'tab:red'
        ax2.set_ylabel("Fluid Pressure P (Pa)", color=color_pres)
        ax2.plot(_scrub_1d(R_centers), _scrub_1d(pressure_profile), color=color_pres, linewidth=2, linestyle='--', label="Pressure Profile")
        ax2.tick_params(axis='y', labelcolor=color_pres)
        
        plt.title("Week 13: Macroscopic Fluid Approximations (Density & Pressure)")
        fig.tight_layout()
        plt.savefig("radial_profiles.png", dpi=300)
        plt.show()

    # --- 7. PHASE-SPACE DISTRIBUTION ---
    if R_phase is not None and v_parallel_phase is not None and len(R_phase) > 0:
        plt.figure(figsize=(8, 6))
        v_par_scrubbed = _scrub_1d(v_parallel_phase)
        
        plt.scatter(_scrub_1d(R_phase), v_par_scrubbed, s=2, color='darkviolet', alpha=0.3)
        plt.title("Week 14: Phase-Space Map (Banana Orbits vs. Passing Particles)")
        plt.xlabel("Major Radius R (m)")
        plt.ylabel("Parallel (Toroidal) Velocity $v_\\parallel$ (m/s)")
        plt.axhline(0, color='black', linewidth=1, linestyle='--')
        
        # Framed to the actual curved confinement boundary, not the full rectangular solver grid.
        plt.xlim(*VIEW_XLIM)
        if len(v_par_scrubbed) > 0:
            v_max = np.percentile(np.abs(v_par_scrubbed), 98)
            plt.ylim(-v_max * 1.2 if v_max > 0 else -1000, v_max * 1.2 if v_max > 0 else 1000)
            
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig("phase_space_map.png", dpi=300)
        plt.show()

    # --- 8. MHD INSTABILITY EXPONENTIAL GROWTH ---
    if instability_amp_history is not None and len(instability_amp_history) > 0:
        plt.figure(figsize=(8, 5))
        amp_history = _scrub_1d(instability_amp_history)
        time_array = np.arange(len(amp_history)) * cfg.reactor_dt * 1e6 
        plt.plot(time_array, amp_history, color='darkorange', linewidth=2.5)
        
        plt.title(f"Week 15: Tearing Mode Instability Growth (m={cfg.m_mode}, n={cfg.n_mode})")
        plt.xlabel("Simulation Time ($\\mu s$)")
        plt.ylabel("Perturbation Amplitude $\\delta B$ (Tesla)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig("instability_growth.png", dpi=300)
        plt.show()

    # --- 9. VOLUMETRIC FUSION POWER HEATMAP (percentile clipped) ---
    if P_fusion_grid is not None:
        plt.figure(figsize=(7, 6))
        P_mw_grid = np.where(_confinement_mask, np.nan, P_fusion_grid.T / 1e6)

        # Clip the top 1% of outliers to keep the gradient readable
        p_max = np.nanpercentile(P_mw_grid, 99)
        if not np.isfinite(p_max) or p_max <= 0: p_max = 1e-9

        contour_pf = plt.contourf(R_mesh, Z_mesh, P_mw_grid, levels=np.linspace(0, p_max, 40), cmap='inferno', extend='max')
        plt.colorbar(contour_pf, label="Volumetric Fusion Power ($MW/m^3$)")
        plt.title("Week 18: D-T Core Fusion Power Density")
        plt.xlabel("Major Radius R (m)")
        plt.ylabel("Height Z (m)")
        plt.xlim(*VIEW_XLIM)
        plt.ylim(*VIEW_YLIM)
        plt.gca().set_aspect('equal')
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig("fusion_power_density.png", dpi=300)
        plt.show()

    # --- 10. ALPHA PARTICLE TRAJECTORIES ---
    if alpha_particles is not None and len(alpha_particles) > 0:
        print("--- RENDERING WEEK 19 ALPHA ORBITS ---")
        fig, ax = plt.subplots(figsize=(8, 8))
        eq.plot_equilibrium(ax=ax)

        """
        
        thermal_count = 0
        for p in active_particles:
            if p.get("type") == 0 and thermal_count < 3:
                traj = p["history"]
                if len(traj) > 1:
                    R_path = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
                    ax.plot(R_path, traj[:, 2], color='cyan', alpha=0.8, linewidth=1.2)
                    thermal_count += 1
                
        for p in alpha_particles:
            traj = p["history"]
            if len(traj) > 1:
                R_path = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
                ax.plot(R_path, traj[:, 2], color='magenta', alpha=0.9, linewidth=1.8)
        """
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
        ax.set_xlim(*VIEW_XLIM)
        ax.set_ylim(*VIEW_YLIM)
        ax.set_aspect('equal')
                
        ax.set_title("Week 19: 3.5 MeV Alpha Orbits vs Thermal Background")
        ax.set_xlabel("Major Radius R (m)")
        ax.set_ylabel("Height Z (m)")
        
        custom_lines = [
            Line2D([0], [0], color='#FF66CC', lw=2, label='3.5 MeV Helium-4 (Alpha)'),
            Line2D([0], [0], color='cyan', lw=2, label='Thermal Background Ion'),
            Line2D([0], [0], color='orange', lw=2, label='50 keV NBI Fast Ion')
        ]
        ax.legend(handles=custom_lines, loc='upper right')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("alpha_orbits.png", dpi=300)
        plt.show()

    # --- 5 (deferred). 3D TOKAMAK RENDERING ---
    # Runs here so cartesian_dfs already carries the alpha orbits.
    print("--- RENDERING 3D STEADY-STATE TOKAMAK ---")
    try:
        viz = PlasmaVisualizer(cartesian_dfs)
        viz.plot_3d_trajectory(output_file="tokamak_reactor_3d.png", classifications=classifications)
    except Exception as e:
        print(f"[WARNING] Skipping 3D visualization due to missing dependency or window issue: {e}")

    # =======================================================
    # 11. ALPHA HEATING & IGNITION BALANCE GRAPH
    # =======================================================
    if alpha_power_history is not None and ext_power_history is not None and len(alpha_power_history) > 0:
        plt.figure(figsize=(8, 5))
        
        alpha_pwr_arr = _scrub_1d(alpha_power_history)
        ext_pwr_arr = _scrub_1d(ext_power_history)
        time_array = np.arange(len(alpha_pwr_arr)) * cfg.reactor_dt * 1e6

        plt.plot(time_array, ext_pwr_arr, color='dodgerblue', linewidth=2, linestyle='--', label='External Auxiliary Heating (40 MW)')
        plt.plot(time_array, alpha_pwr_arr, color='magenta', linewidth=2.5, label='Internal Alpha Heating')
        plt.fill_between(time_array, alpha_pwr_arr, ext_pwr_arr, where=(alpha_pwr_arr > ext_pwr_arr), interpolate=True, color='magenta', alpha=0.2, label='Ignition Achieved (Self-Heating)')

        plt.title("Week 20: Reactor Ignition Criteria (Alpha Heating Balance)")
        plt.xlabel("Simulation Time ($\\mu s$)")
        plt.ylabel("Heating Power (MW)")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper left')
        
        plt.tight_layout()
        plt.savefig("alpha_heating_balance.png", dpi=300)
        plt.show()
        
    # =======================================================
    # 12. RADIATION LOSS PROFILE
    # =======================================================
    if brem_power_history is not None and cyc_power_history is not None and alpha_power_history is not None and len(brem_power_history) > 0:
        plt.figure(figsize=(8, 6))
        
        alpha_pwr_arr = _scrub_1d(alpha_power_history)
        brem_pwr_arr = _scrub_1d(brem_power_history)
        cyc_pwr_arr = _scrub_1d(cyc_power_history)
        time_array = np.arange(len(brem_pwr_arr)) * cfg.reactor_dt * 1e6
        
        total_rad_loss = brem_pwr_arr + cyc_pwr_arr
        
        plt.plot(time_array, alpha_pwr_arr, color='magenta', linewidth=2.5, label='Alpha Heating Power (Input)')
        plt.plot(time_array, total_rad_loss, color='black', linewidth=2.5, linestyle='-', label='Total Radiation Loss')
        plt.plot(time_array, brem_pwr_arr, color='dodgerblue', linewidth=1.5, linestyle='--', label='Bremsstrahlung (X-Ray)')
        plt.plot(time_array, cyc_pwr_arr, color='darkorange', linewidth=1.5, linestyle=':', label='Cyclotron (Microwave)')
        
        plt.fill_between(time_array, alpha_pwr_arr, total_rad_loss, where=(alpha_pwr_arr >= total_rad_loss), interpolate=True, color='lightgreen', alpha=0.3, label='Net Heating (Ignition Maintained)')
        plt.fill_between(time_array, alpha_pwr_arr, total_rad_loss, where=(alpha_pwr_arr < total_rad_loss), interpolate=True, color='salmon', alpha=0.3, label='Net Cooling (Reactor Dying)')
        
        plt.title("Week 21: Thermodynamic Balance (Alpha Heating vs Radiation Loss)")
        plt.xlabel("Simulation Time ($\\mu s$)")
        plt.ylabel("Power (MW)")
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='upper left')
        
        plt.tight_layout()
        plt.savefig("radiation_loss_profile.png", dpi=300)
        plt.show()

    # =======================================================
    # 13. Q-FACTORS & LAWSON CRITERION
    # =======================================================
    if q_sci_history is not None and q_eng_history is not None and lawson_history is not None and len(q_sci_history) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        
        q_sci_history = _scrub_1d(q_sci_history)
        q_eng_history = _scrub_1d(q_eng_history)
        lawson_arr = _scrub_1d(lawson_history)
        time_array = np.arange(len(q_sci_history)) * cfg.reactor_dt * 1e6
        
        ax1.plot(time_array, q_sci_history, color='magenta', linewidth=2.5, label='Scientific Gain ($Q_{sci}$)')
        ax1.plot(time_array, q_eng_history, color='dodgerblue', linewidth=2.5, linestyle='--', label='Engineering Gain ($Q_{eng}$)')
        ax1.axhline(1.0, color='gold', linestyle='-', linewidth=2, label='Break-Even (Q=1)')
        ax1.fill_between(time_array, q_sci_history, 1.0, where=(q_sci_history >= 1.0), interpolate=True, color='lightgreen', alpha=0.2)
                         
        ax1.set_title("Week 22: Reactor Performance & Gain (Q-Factors)")
        ax1.set_ylabel("Gain Factor (Q)")
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='upper left')
        
        ax2.plot(time_array, lawson_arr, color='firebrick', linewidth=2.5, label='Simulated Triple Product ($n_e T_i \\tau_E$)')
        ax2.axhline(cfg.lawson_target, color='mediumseagreen', linestyle='--', linewidth=2, label=f'Ignition Target ({cfg.lawson_target:.1e})')
        ax2.fill_between(time_array, lawson_arr, cfg.lawson_target, where=(lawson_arr >= cfg.lawson_target), interpolate=True, color='mediumseagreen', alpha=0.2)
        
        ax2.set_title("Lawson Triple Product Evolution")
        ax2.set_xlabel("Simulation Time ($\\mu s$)")
        ax2.set_ylabel("Triple Product ($keV \\cdot s \\cdot m^{-3}$)")
        ax2.set_yscale('log')
        ax2.grid(True, which="both", linestyle=':', alpha=0.6)
        ax2.legend(loc='lower right')
        
        plt.tight_layout()
        plt.savefig("lawson_q_factor.png", dpi=300)
        plt.show()

def run_oscillation_diagnostics(cfg, time_arr, w_es_history):
    time_arr = _scrub_1d(time_arr) * 1e9
    w_es_history = _scrub_1d(w_es_history)
    
    N = len(w_es_history)
    w_es_ac = w_es_history - np.mean(w_es_history) 
    
    fft_vals = np.fft.fft(w_es_ac)
    fft_freqs = np.fft.fftfreq(N, d=cfg.osc_dt)
    
    pos_mask = fft_freqs > 0
    freqs = fft_freqs[pos_mask]
    power = np.abs(fft_vals[pos_mask])**2
    
    dominant_freq = freqs[np.argmax(power)]
    plasma_freq = dominant_freq / 2.0  

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
    
    ax1.plot(time_arr, w_es_history, color='dodgerblue', linewidth=1.5)
    ax1.set_title("Debye/Langmuir Validation: Electrostatic Energy ($W_{ES}$) Ringing")
    ax1.set_xlabel("Time (ns)")
    ax1.set_ylabel("Stored E-Field Energy (Joules)")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(freqs, power, color='crimson', linewidth=1.5)
    ax2.set_xlim(0, dominant_freq * 3.0)
    ax2.axvline(dominant_freq, color='gold', linestyle='--', label=f'Spectral Peak: {dominant_freq:.2e} Hz ($2f_{{pe}}$)')
    ax2.set_title("Frequency Spectrum (FFT) of Field Energy")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Power")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plasma_oscillation_frequency.png", dpi=300)
    plt.show()

def plot_disruption_mitigation(time_history, temp_history, rad_power_history, trigger_time=None):
    time_history = _scrub_1d(time_history)
    temp_history = _scrub_1d(temp_history)
    rad_power_history = _scrub_1d(rad_power_history)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    time_ms = time_history * 1000
    
    ax1.plot(time_ms, temp_history, color='crimson', linewidth=2.5, label='Core Electron Temp ($T_e$)')
    if trigger_time is not None:
        ax1.axvline(x=trigger_time * 1000, color='black', linestyle='--', label='SPI Triggered')
        
    ax1.set_title("Week 16: Active Disruption Mitigation (Shattered Pellet Injection)")
    ax1.set_ylabel("Core Temperature (keV)")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper right')
    
    rad_power_mw = rad_power_history / 1e6
    ax2.plot(time_ms, rad_power_mw, color='dodgerblue', linewidth=2.5, label='Radiated Power ($P_{rad}$)')
    if trigger_time is not None:
        ax2.axvline(x=trigger_time * 1000, color='black', linestyle='--')
        
    ax2.set_xlabel("Simulation Time (ms)")
    ax2.set_ylabel("Radiative Cooling Loss (MW/$m^3$)")
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig("disruption_mitigation.png", dpi=300)
    plt.show()

def plot_fusion_cross_section(E_kev_arr, sigma_arr):
    plt.figure(figsize=(8, 5))
    plt.semilogy(E_kev_arr, sigma_arr, color='darkviolet', linewidth=2.5, label='D-T Cross-Section $\\sigma$')
    
    peak_idx = np.argmax(sigma_arr)
    peak_energy = E_kev_arr[peak_idx]
    peak_sigma = sigma_arr[peak_idx]
    
    plt.axvline(x=peak_energy, color='gold', linestyle='--', label=f'Resonance Peak (~{peak_energy:.1f} keV)')
    plt.plot(peak_energy, peak_sigma, 'ro')
    
    plt.title("Week 17: D-T Fusion Cross-Section vs. Collision Energy")
    plt.xlabel("Center-of-Mass Collision Energy (keV)")
    plt.ylabel("Cross-Section $\\sigma$ (barns)")
    plt.xlim(0, 200)
    plt.ylim(1e-5, np.max(sigma_arr) * 5)
    plt.grid(True, which="both", linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig("fusion_cross_section.png", dpi=300)
    plt.show()