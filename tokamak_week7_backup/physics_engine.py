import numpy as np
import matplotlib.pyplot as plt
from numba import njit

# --- CONSTANTS ---
e_charge = 1.602e-19       # Coulombs
m_deuterium = 3.3435e-27   # kg

class ParticlePusher:
    def __init__(self, equilibrium):
        self.eq = equilibrium
        self.m = m_deuterium
        self.q = e_charge

    def initialize_velocities(self, num_particles, T_keV=1.0):
        
        # Initializes a 3D Maxwell-Boltzmann velocity distribution for the thermal background plasma.
        
        T_Joules = T_keV * 1e3 * self.q
        v_thermal = np.sqrt(2.0 * T_Joules / self.m)
        velocities = np.random.normal(0, v_thermal / np.sqrt(2), (num_particles, 3))
        return velocities

    def plot_velocity_diagnostic(self, velocities, T_keV, filename="velocity_diagnostic.png"):
        
        # Plots the Maxwellian distribution to verify the initialization.
        
        speeds = np.linalg.norm(velocities, axis=1)
        plt.figure(figsize=(8, 6))
        plt.hist(speeds, bins=50, density=True, color='crimson', alpha=0.7)
        plt.title(f"Initial Maxwell-Boltzmann Particle Speeds ({T_keV} keV)")
        plt.xlabel("Speed (m/s)")
        plt.ylabel("Probability Density")
        plt.grid(True, alpha=0.3)
        plt.savefig(filename, dpi=300)
        plt.close()
        
    def inject_neutral_beam(self, num_ions, E_keV=50.0, R_target=1.05, Z_target=0.0):
        
        # WEEK 7: Neutral Beam Injection (NBI).
        # Injects fast ions into the core with high tangential (parallel) velocity.
        # Returns: positions (N,3), v_parallels (N,), mus (N,)
        
        E_Joules = E_keV * 1e3 * self.q
        v_beam = np.sqrt(2.0 * E_Joules / self.m)
        
        positions = np.zeros((num_ions, 3))
        v_parallels = np.zeros(num_ions)
        mus = np.zeros(num_ions)
        
        for i in range(num_ions):
            # 1. Spatial Spread: Inject tightly around the core target
            R = np.random.normal(R_target, 0.05)
            Z = np.random.normal(Z_target, 0.05)
            phi = np.random.uniform(0, 2 * np.pi)
            pos = np.array([R * np.cos(phi), R * np.sin(phi), Z])
            positions[i] = pos
            
            # 2. Velocity Profile: Tangential injection (mostly parallel, small pitch angle)
            pitch_angle = np.random.normal(0.1, 0.05) # Radians
            v_para = v_beam * np.cos(pitch_angle)
            v_perp = v_beam * np.abs(np.sin(pitch_angle))
            v_parallels[i] = v_para
            
            # 3. Calculate magnetic moment (mu) based on local field
            B_vec = self.eq.get_B_field(pos)
            B_mag = np.linalg.norm(B_vec)
            if B_mag == 0:
                B_mag = 1e-5 # Prevent division by zero just in case
                
            mu = (self.m * v_perp**2) / (2.0 * B_mag)
            mus[i] = mu
            
        return positions, v_parallels, mus

    def guiding_center_push(self, pos, v_parallel, mu, dt):
        
        # Wrapper to call the Numba-compiled JIT pusher.
        
        B_vec = self.eq.get_B_field(pos)
        grad_B = self.eq.get_grad_B(pos)
        return _jit_guiding_center_push(pos, v_parallel, mu, B_vec, grad_B, self.m, self.q, dt)


# --- CLASSLESS NUMBA KERNELS ---

@njit(fastmath=True)
def _jit_guiding_center_push(pos, v_parallel, mu, B_vec, grad_B, m, q, dt):
    
    #Week 4: The core collisionless guiding-center drifts (Grad-B & Curvature).
    
    B_mag = np.linalg.norm(B_vec)
    if B_mag == 0:
        return pos, v_parallel, False

    b_unit = B_vec / B_mag

    # Grad-B Drift
    v_grad_B = (mu / (q * B_mag)) * np.cross(b_unit, grad_B)

    # Curvature Drift (simplified proxy)
    R_vec = np.array([pos[0], pos[1], 0.0])
    R_mag = np.linalg.norm(R_vec)
    if R_mag > 0:
        R_unit = R_vec / R_mag
        v_curv = (m * v_parallel**2 / (q * B_mag * R_mag)) * np.cross(b_unit, R_unit)
    else:
        v_curv = np.zeros(3)

    # Total Velocity
    v_drift = v_grad_B + v_curv
    v_total = v_parallel * b_unit + v_drift

    # Update Position
    pos_new = pos + v_total * dt

    # Boundary Check (Divertor Walls)
    R_new = np.sqrt(pos_new[0]**2 + pos_new[1]**2)
    Z_new = abs(pos_new[2])

    is_confined = True
    if R_new < 0.7 or R_new > 1.3 or Z_new > 0.3:
        is_confined = False

    return pos_new, v_parallel, is_confined


@njit(fastmath=True)
def apply_guiding_center_collisions(v_parallel, mu, B_mag, m, dt, nu_c):
    
    # Week 5: Monte Carlo Coulomb Collisions (Pitch-Angle Scattering) with Thermostatting.
    
    if B_mag <= 0:
        return v_parallel, mu

    # 1. Back-calculate current v_perp
    v_perp = np.sqrt(2.0 * mu * B_mag / m)

    # 2. Thermostat: lock total velocity magnitude
    v_mag = np.sqrt(v_parallel**2 + v_perp**2)
    if v_mag == 0:
        return v_parallel, mu

    # 3. Current pitch angle
    pitch_angle = np.arccos(v_parallel / v_mag)

    # 4. Apply stochastic kick
    scattering_variance = nu_c * dt
    delta_pitch = np.random.normal(0.0, np.sqrt(scattering_variance))
    new_pitch = pitch_angle + delta_pitch

    # 5. Reconstruct velocities using original v_mag (Energy Conservation)
    v_parallel_new = v_mag * np.cos(new_pitch)
    v_perp_new = v_mag * np.abs(np.sin(new_pitch))
    mu_new = (m * v_perp_new**2) / (2.0 * B_mag)

    return v_parallel_new, mu_new


@njit(fastmath=True)
def classify_particle_orbit(v_parallel_history):
    
    # Week 6: Classifies Trapped (Banana) vs Passing Orbits.
    
    sign_changes = 0
    for i in range(1, len(v_parallel_history)):
        if v_parallel_history[i] * v_parallel_history[i-1] < 0.0:
            sign_changes += 1
            break
            
    return 1 if sign_changes > 0 else 0
