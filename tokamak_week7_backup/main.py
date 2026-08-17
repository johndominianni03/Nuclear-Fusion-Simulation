
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mhd_equilibrium import MHDEquilibrium
from physics_engine import ParticlePusher, apply_guiding_center_collisions, classify_particle_orbit
from visualizer import PlasmaVisualizer

def run_reactor_steady_state():
    print("==================================================")
    print("      WEEK 7: TOKAMAK REACTOR STEADY-STATE NBI    ")
    print("==================================================")
    
    # Initialize MHD and Engine
    eq = MHDEquilibrium()
    eq.solve_grad_shafranov()
    engine = ParticlePusher(equilibrium=eq)

    # --- REACTOR TIME DOMAIN & PARAMETERS ---
    num_steps = 2000
    dt = 1.0e-7
    nu_c = 5000.0
    
    # Initial Thermal Background Plasma
    initial_thermal_count = 50
    T_thermal_keV = 1.0
    
    # Neutral Beam Injection (NBI) Parameters
    nbi_energy_keV = 50.0
    inject_every_n_steps = 20  # Fire NBI beam every 20 steps
    
    print(f"[SYSTEM] Initializing thermal background ({initial_thermal_count} particles at {T_thermal_keV} keV)...")
    thermal_velocities = engine.initialize_velocities(initial_thermal_count, T_keV=T_thermal_keV)
    
    active_particles = []
    
    # 1. Load Initial Thermal Plasma
    for i in range(initial_thermal_count):
        R = np.random.uniform(1.0, 1.20)
        Z = np.random.uniform(-0.1, 0.1)
        phi = np.random.uniform(0, 2 * np.pi)
        pos = np.array([R * np.cos(phi), R * np.sin(phi), Z])
        
        vel = thermal_velocities[i]
        B_vec = eq.get_B_field(pos)
        B_mag = np.linalg.norm(B_vec)
        b_unit = B_vec / B_mag if B_mag > 0 else np.array([0, 0, 1])
        
        v_para = np.dot(vel, b_unit)
        v_perp_vec = vel - (v_para * b_unit)
        v_perp = np.linalg.norm(v_perp_vec)
        mu = (engine.m * v_perp**2) / (2.0 * B_mag)
        
        active_particles.append({
            "pos": pos,
            "v_para": v_para,
            "mu": mu,
            "history": [pos.copy()],
            "status": "confined",
            "type": 0  # 0 = Thermal Background
        })

    # --- METRIC TRACKERS ---
    total_injected = initial_thermal_count
    total_lost = 0
    inventory_history = []
    energy_history_keV = []  # Tracks total stored energy W_plasma in keV

    print("[SYSTEM] Igniting time-domain reactor loop. Tracking Inventory & Stored Energy...")


    # ==========================================
    #      THE MASTER TIME LOOP (STEADY-STATE)
    # ==========================================
    for step in range(num_steps):
        
        # 1. THE SOURCE (S): Inject Fast NBI Ions periodically
        if step % inject_every_n_steps == 0 and step > 0:
            pos_nbi, v_para_nbi, mu_nbi = engine.inject_neutral_beam(num_ions=1, E_keV=nbi_energy_keV)
            active_particles.append({
                "pos": pos_nbi[0],
                "v_para": v_para_nbi[0],
                "mu": mu_nbi[0],
                "history": [pos_nbi[0].copy()],
                "status": "confined",
                "type": 1  # 1 = NBI Fast Ion
            })
            total_injected += 1

        current_confined = 0
        current_energy_joules = 0.0

        # 2. PUSH, COLLIDE & DIAGNOSE ENERGY FOR ALL ACTIVE PARTICLES
        for p in active_particles:
            if p["status"] == "confined":
                # Push guiding center
                new_pos, new_v_para, is_confined = engine.guiding_center_push(
                    p["pos"], p["v_para"], p["mu"], dt
                )
                
                # Check for Divertor Wall Sink (L)
                if not is_confined:
                    p["status"] = "lost"
                    p["type"] = -1
                    total_lost += 1
                    continue
                
                # Apply collisions
                B_vec_curr = eq.get_B_field(new_pos)
                B_mag_curr = np.linalg.norm(B_vec_curr)
                if B_mag_curr > 0:
                    new_v_para, new_mu = apply_guiding_center_collisions(
                        new_v_para, p["mu"], B_mag_curr, engine.m, dt, nu_c
                    )
                
                # Update State
                p["pos"] = new_pos
                p["v_para"] = new_v_para
                p["mu"] = new_mu
                p["history"].append(new_pos.copy())
                current_confined += 1

                # ENERGY DIAGNOSTIC: E_kin = 0.5 * m * v_para^2 + mu * B
                E_para = 0.5 * engine.m * (new_v_para**2)
                E_perp = new_mu * B_mag_curr
                current_energy_joules += (E_para + E_perp)

        inventory_history.append(current_confined)
        
        # Convert total stored energy from Joules to keV (1 eV = 1.602e-19 J)
        current_energy_keV = current_energy_joules / (engine.q * 1000.0)
        energy_history_keV.append(current_energy_keV)
        
        if (step + 1) % 500 == 0:
            print(f"  Step {step+1}/{num_steps} | Confined: {current_confined} | Lost: {total_lost} | Stored Energy: {current_energy_keV:.2f} keV")

    # ==========================================
    #      POST-PROCESSING & DIAGNOSTICS
    # ==========================================
    avg_inventory = np.mean(inventory_history)
    avg_energy_keV = np.mean(energy_history_keV[-200:])  # Average over final steady state steps
    sim_time_seconds = num_steps * dt
    loss_rate = total_lost / sim_time_seconds if sim_time_seconds > 0 else 0
    tau_p = (avg_inventory / loss_rate) if loss_rate > 0 else float('inf')

    print("\n==================================================")
    print("      WEEK 7: REACTOR PERFORMANCE DIAGNOSTIC     ")
    print("==================================================")
    print(f"Total Simulation Time         : {sim_time_seconds:.2e} s")
    print(f"Total Particles Sourced       : {total_injected} (Thermal + NBI)")
    print(f"Total Particles Lost (Sink)   : {total_lost}")
    print(f"Final Steady-State Inventory  : {inventory_history[-1]} particles")
    print(f"Final Stored Energy (W_plasma): {energy_history_keV[-1]:.2f} keV")
    print(f"Average Loss Rate (L)         : {loss_rate:.2e} particles/sec")
    print(f"Estimated Confinement Time (τ): {tau_p:.2e} seconds")
    print("==================================================")

    # --- PLOTTING STORED ENERGY WAVEFORM ---
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(num_steps) * dt * 1e6, energy_history_keV, color='firebrick', linewidth=1.5)
    plt.title("Stored Plasma Energy ($W_{plasma}$) vs. Time (NBI Ramp-Up)")
    plt.xlabel("Time ($mu s$)")
    plt.ylabel("Stored Energy (keV)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plasma_stored_energy_time.png", dpi=300)
    plt.show()

    # --- PLOTTING TOKAMAK TRAJECTORIES ---
    cartesian_dfs = []
    classifications = []
    
    fig, ax = plt.subplots(figsize=(8, 8))
    eq.plot_equilibrium(ax=ax)
    
    for p in active_particles:
        traj = np.array(p["history"])
        cartesian_dfs.append(pd.DataFrame(traj, columns=["x", "y", "z"]))
        classifications.append(p["type"])
        
        R_path = np.sqrt(traj[:, 0]**2 + traj[:, 1]**2)
        Z_path = traj[:, 2]
        
        if p["type"] == 1:
            ax.plot(R_path, Z_path, color='gold', alpha=0.7, linewidth=1.2)
        elif p["type"] == 0:
            ax.plot(R_path, Z_path, color='dodgerblue', alpha=0.2, linewidth=0.8)
        else:
            ax.plot(R_path, Z_path, color='crimson', alpha=0.3, linewidth=0.5)

    ax.set_title("Week 7: Steady-State Reactor (Thermal vs. NBI Injection)")
    ax.set_xlabel("Major Radius R (m)")
    ax.set_ylabel("Height Z (m)")
    
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='gold', lw=2),
        Line2D([0], [0], color='dodgerblue', lw=2),
        Line2D([0], [0], color='crimson', lw=2)
    ]
    ax.legend(custom_lines, ['50 keV NBI Fast Ions', '1 keV Thermal Core', 'Lost to Wall'], loc='upper right')
    
    plt.grid(True, alpha=0.3)
    plt.savefig("tokamak_reactor_2d.png", dpi=300)
    plt.show()

    print("--- RENDERING 3D STEADY-STATE TOKAMAK ---")
    viz = PlasmaVisualizer(cartesian_dfs)
    viz.plot_3d_trajectory(output_file="tokamak_reactor_3d.png", classifications=classifications)

if __name__ == "__main__":
    run_reactor_steady_state()

