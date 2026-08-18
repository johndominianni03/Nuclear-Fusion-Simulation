import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import torch

# --- CONSTANTS ---
e_charge = 1.602e-19       # Coulombs
m_deuterium = 3.3435e-27   # kg
m_electron = 9.109e-31     # kg
eps_0 = 8.854e-12          # Vacuum permittivity (F/m)

class ParticlePusher:
    def __init__(self, equilibrium, species="deuterium"):
        self.eq = equilibrium
        # Species is swappable so electron Langmuir waves can be simulated
        self.species = species
        if self.species == "electron":
            self.m = m_electron
            self.q = -e_charge
        else:
            self.m = m_deuterium
            self.q = e_charge

    def initialize_velocities(self, num_particles, T_keV=1.0):
        T_Joules = T_keV * 1e3 * abs(self.q)
        v_thermal = np.sqrt(2.0 * T_Joules / self.m)
        velocities = np.random.normal(0, v_thermal / np.sqrt(2), (num_particles, 3))
        return velocities

    def plot_velocity_diagnostic(self, velocities, T_keV, filename="velocity_diagnostic.png"):
        speeds = np.linalg.norm(velocities, axis=1)
        plt.figure(figsize=(8, 6))
        plt.hist(speeds, bins=50, density=True, color='crimson', alpha=0.7)
        plt.title(f"Initial Maxwell-Boltzmann Particle Speeds ({T_keV} keV)")
        plt.xlabel("Speed (m/s)")
        plt.ylabel("Probability Density")
        plt.grid(True, alpha=0.3)
        plt.savefig(filename, dpi=300)
        plt.close()

    def map_charge_density(self, active_particles, R_min=0.5, R_max=1.5, Z_min=-0.5, Z_max=0.5, nR=50, nZ=50, apply_neutral_background=False):
        valid_particles = [p for p in active_particles if p["type"] != -1]
        
        if len(valid_particles) == 0:
            return np.zeros((nR, nZ))
            
        R_coords = np.array([p["pos"][0] for p in valid_particles], dtype=np.float64)
        Z_coords = np.array([p["pos"][1] for p in valid_particles], dtype=np.float64)
        charges = np.full(len(R_coords), self.q, dtype=np.float64)
        
        rho_grid = compute_cic_charge_density(
            R_coords, Z_coords, charges, 
            R_min, R_max, Z_min, Z_max, nR, nZ, apply_neutral_background
        )
        return rho_grid
        
    def solve_fields(self, rho_grid, R_min, R_max, Z_min, Z_max, phi_init=None):
        nR, nZ = rho_grid.shape
        dR = (R_max - R_min) / (nR - 1)
        dZ = (Z_max - Z_min) / (nZ - 1)

        # Warm-start SOR from the previous step's converged phi: the field barely
        # shifts step-to-step, so it reaches the same tolerance in far fewer iterations.
        if phi_init is None:
            phi_init = np.zeros((nR, nZ), dtype=np.float64)

        phi = solve_poisson_sor_cylindrical(rho_grid, R_min, R_max, Z_min, Z_max, phi_init)
        E_R, E_Z = compute_electric_field(phi, dR, dZ)
        return phi, E_R, E_Z

    def inject_neutral_beam(self, num_ions, E_keV=50.0, R_target=1.05, Z_target=0.0):
        E_Joules = E_keV * 1e3 * abs(self.q)
        v_beam = np.sqrt(2.0 * E_Joules / self.m)
        positions = np.zeros((num_ions, 3))
        v_parallels = np.zeros(num_ions)
        mus = np.zeros(num_ions)
        
        for i in range(num_ions):
            R = np.random.normal(R_target, 0.05)
            Z = np.random.normal(Z_target, 0.05)
            phi = np.random.uniform(0, 2 * np.pi)
            pos = np.array([R * np.cos(phi), R * np.sin(phi), Z])
            positions[i] = pos
            
            pitch_angle = np.random.normal(0.1, 0.05)
            v_para = v_beam * np.cos(pitch_angle)
            v_perp = v_beam * np.abs(np.sin(pitch_angle))
            v_parallels[i] = v_para
            
            B_vec = self.eq.get_B_field(pos)
            B_mag = np.linalg.norm(B_vec)
            if B_mag == 0: B_mag = 1e-5
            mu = (self.m * v_perp**2) / (2.0 * B_mag)
            mus[i] = mu
            
        return positions, v_parallels, mus

    def inject_neutral_beam_cartesian(self, num_ions, E_keV=50.0, R_target=1.05, Z_target=0.0, psi_bounds=None):
        E_Joules = E_keV * 1e3 * abs(self.q)
        v_beam = np.sqrt(2.0 * E_Joules / self.m)
        positions = np.zeros((num_ions, 3))
        velocities = np.zeros((num_ions, 3))

        for i in range(num_ions):
            # Rejection-sample the birth position against the core flux contour, so
            # injected ions land inside the magnetic surface rather than wherever the
            # Cartesian Gaussian happened to put them.
            if psi_bounds is not None:
                psi_grid, psi_core, R_min, R_max, Z_min, Z_max, nR, nZ = psi_bounds
                for _attempt in range(50):
                    R = np.random.normal(R_target, 0.05)
                    Z = np.random.normal(Z_target, 0.05)
                    if interpolate_psi(R, Z, psi_grid, R_min, R_max, Z_min, Z_max, nR, nZ) > psi_core:
                        break
            else:
                R = np.random.normal(R_target, 0.05)
                Z = np.random.normal(Z_target, 0.05)

            phi_ang = np.random.uniform(0, 2 * np.pi)
            pos = np.array([R * np.cos(phi_ang), R * np.sin(phi_ang), Z])
            positions[i] = pos

            # Seed the parallel fraction xi = v_par/v directly. The beam was previously
            # aimed radially inward, giving every ion v_parallel = 0 -- it could never
            # stream along a field line or trace a banana orbit. Real NBI is tangential.
            # Mean 0.2 sits just below the trapped/passing boundary sqrt(2*eps/(1+eps))
            # = 0.309 at the R_target=1.05 birth surface, so the beam is predominantly
            # trapped while the 0.25 spread still carries a passing tail.
            xi = np.clip(np.random.normal(0.2, 0.25), -0.99, 0.99)
            v_par = v_beam * xi
            v_perp = v_beam * np.sqrt(1.0 - xi * xi)
            gyro = np.random.uniform(0, 2 * np.pi)
            # phi_hat = (-sin, cos, 0);  R_hat = (cos, sin, 0);  Z_hat = (0, 0, 1)
            v_x = -v_par * np.sin(phi_ang) + v_perp * np.cos(gyro) * np.cos(phi_ang)
            v_y = v_par * np.cos(phi_ang) + v_perp * np.cos(gyro) * np.sin(phi_ang)
            v_z = v_perp * np.sin(gyro)
            velocities[i] = np.array([v_x, v_y, v_z])

        return positions, velocities

    def inject_langmuir_perturbation(self, num_particles, R_min, R_max, Z_min, Z_max, k_Z, perturbation_amplitude=1e4):
        # Uniform positions plus a sinusoidal velocity kick to excite plasma oscillations
        positions = np.zeros((num_particles, 3))
        velocities = np.zeros((num_particles, 3))
        
        for i in range(num_particles):
            R = np.random.uniform(R_min, R_max)
            Z = np.random.uniform(Z_min, Z_max)
            phi_ang = np.random.uniform(0, 2 * np.pi)
            positions[i] = np.array([R * np.cos(phi_ang), R * np.sin(phi_ang), Z])
            
            # Velocity wave perturbation
            v_z = perturbation_amplitude * np.sin(k_Z * Z)
            velocities[i] = np.array([0.0, 0.0, v_z])
            
        return positions, velocities

    # Both push wrappers return motion only -- (pos, v_parallel) and (pos, vel). They no
    # longer report a confinement flag: the single authority on whether a particle is
    # still inside the plasma is check_confinement_flux / check_confinement_torch, which
    # test psi against psi_edge.
    def guiding_center_push(self, pos, v_parallel, mu, dt, E_vec, t=0.0, b_perturb=0.0, m_mode=2, n_mode=1, gamma=0.0):
        B_vec = self.eq.get_B_field(pos)
        # Tearing mode perturbation, if active
        if b_perturb > 0.0:
            B_vec += _jit_apply_magnetic_perturbation(pos, t, b_perturb, m_mode, n_mode, gamma)
            
        grad_B = self.eq.get_grad_B(pos)
        return _jit_guiding_center_push(pos, v_parallel, mu, B_vec, grad_B, E_vec, self.m, self.q, dt)

    def lorentz_push(self, pos, vel, dt, E_vec, t=0.0, b_perturb=0.0, m_mode=2, n_mode=1, gamma=0.0):
        B_vec = self.eq.get_B_field(pos)
        # Tearing mode perturbation, if active
        if b_perturb > 0.0:
            B_vec += _jit_apply_magnetic_perturbation(pos, t, b_perturb, m_mode, n_mode, gamma)
            
        return _jit_lorentz_push(pos, vel, B_vec, E_vec, self.m, self.q, dt)

    def compute_fluid_moments(self, active_particles, R_min, R_max, Z_min, Z_max, num_bins):
        # 0th (density) and 2nd (pressure) fluid moments, by binning particles radially.
        R_edges = np.linspace(R_min, R_max, num_bins + 1)
        R_centers = 0.5 * (R_edges[:-1] + R_edges[1:])
        
        density_profile = np.zeros(num_bins)
        pressure_profile = np.zeros(num_bins)
        
        # Positions and velocities of confined particles
        R_pos = []
        vels = []
        for p in active_particles:
            if p["status"] == "confined":
                r = np.sqrt(p["pos"][0]**2 + p["pos"][1]**2)
                R_pos.append(r)
                vels.append(p["vel"])
                
        R_pos = np.array(R_pos)
        vels = np.array(vels)
        
        if len(R_pos) == 0:
            return R_centers, density_profile, pressure_profile
            
        # Bin radially into fluid rings
        bin_indices = np.digitize(R_pos, R_edges) - 1
        
        for i in range(num_bins):
            mask = (bin_indices == i)
            v_in_bin = vels[mask]
            count = len(v_in_bin)
            
            # Cylindrical shell volume
            V_shell = np.pi * (R_edges[i+1]**2 - R_edges[i]**2) * (Z_max - Z_min)

            if count > 0:
                # 0th moment: density n(r)
                n_e = count / V_shell
                density_profile[i] = n_e

                # 1st moment: bulk velocity u(r)
                u_bulk = np.mean(v_in_bin, axis=0)

                # 2nd moment: pressure P(r) = (m / 3V) * sum((v - u)^2)
                v_random_sq = np.sum((v_in_bin - u_bulk)**2, axis=1)
                P = (self.m / (3.0 * V_shell)) * np.sum(v_random_sq)
                
                pressure_profile[i] = P
                
        return R_centers, density_profile, pressure_profile

    # ====================================================
    # ALPHA PARTICLE SPAWNING & KINEMATICS
    # ====================================================
    def spawn_alpha_particles(self, num_alphas, R_birth, Z_birth, alpha_energy_joules, mass_alpha):
        # Isotropically spawns 3.5 MeV D-T alphas; returns positions and velocities.

        # Speed from the 3.5 MeV birth energy
        v_alpha_mag = np.sqrt(2.0 * alpha_energy_joules / mass_alpha)

        alpha_R = np.full(num_alphas, R_birth)
        alpha_Z = np.full(num_alphas, Z_birth)

        # Isotropic scattering in spherical coordinates
        phi = np.random.uniform(0, 2 * np.pi, num_alphas)
        costheta = np.random.uniform(-1, 1, num_alphas)
        theta = np.arccos(costheta)

        # Back to cylindrical components
        alpha_vR = v_alpha_mag * np.sin(theta) * np.cos(phi)
        alpha_vphi = v_alpha_mag * np.sin(theta) * np.sin(phi)
        alpha_vZ = v_alpha_mag * np.cos(theta)
        
        return alpha_R, alpha_Z, alpha_vR, alpha_vphi, alpha_vZ

    def compute_alpha_heating_power(self, alpha_energies_kev, dt, cfg):
        # Drains energy from fast alphas via Coulomb drag, updates their kinetic
        # energies, and returns the deposited heating power in MW.

        if len(alpha_energies_kev) == 0:
            return alpha_energies_kev, 0.0, 0.0

        # Precision: the per-step drain fraction 2*NU_S*dt = 4e-9 is ~30x below float32
        # epsilon, so the old `E - E*fraction` on float32 rounded back to E and the
        # heating flatlined. Promote to float64 and compute the deposit directly rather
        # than as a difference of nearly-equal numbers.
        E = np.asarray(alpha_energies_kev, dtype=np.float64)

        # 1. Loss fraction from the slowing-down time: dE = -2 * nu_s * E * dt
        energy_loss_fraction = min(2.0 * cfg.NU_S * dt, 0.99)

        # 2. Energy given up this step. Capped by the headroom above the thermalization
        #    floor -- a cooled alpha is Helium ash and deposits nothing further.
        headroom = np.maximum(E - cfg.THERMALIZATION_ENERGY_KEV, 0.0)
        deposited_kev = np.minimum(E * energy_loss_fraction, headroom)

        # 3. Drain it from the alphas
        new_alpha_energies = E - deposited_kev

        # 4. Total transferred to the bulk this step
        total_energy_deposited_kev = float(np.sum(deposited_kev))

        # 5. Convert to reactor-scale MW (1 keV = 1.602e-16 J), scaled by a macro-particle
        # weight representing the full plasma volume.
        joules_per_kev = 1.602e-16
        # Shared with the Lawson triple product so both describe the same plasma
        macro_weight = cfg.MACRO_WEIGHT_REACTOR

        total_joules = total_energy_deposited_kev * joules_per_kev * macro_weight

        power_watts = total_joules / dt
        power_mw = power_watts / 1e6

        # Third return value is raw simulation-scale keV (no macro_weight), so the caller
        # can conserve energy by depositing it into the thermal population.
        return new_alpha_energies, power_mw, total_energy_deposited_kev

# --- CLASSLESS NUMBA KERNELS ---

@njit(fastmath=True)
def _jit_lorentz_push(pos, vel, B_vec, E_vec, m, q, dt):
    B_mag = np.linalg.norm(B_vec)
    
    if B_mag == 0:
        vel_new = vel + (q / m) * E_vec * dt
        pos_new = pos + vel_new * dt
    else:
        q_prime = q * dt / (2.0 * m)
        t_vec = q_prime * B_vec
        t_mag2 = t_vec[0]**2 + t_vec[1]**2 + t_vec[2]**2
        s_vec = 2.0 * t_vec / (1.0 + t_mag2)
        
        v_minus = vel + q_prime * E_vec
        v_prime = v_minus + np.cross(v_minus, t_vec)
        v_plus = v_minus + np.cross(v_prime, s_vec)
        
        vel_new = v_plus + q_prime * E_vec
        pos_new = pos + vel_new * dt

    # No confinement test here. This kernel integrates motion and nothing else; whether a
    # particle is still inside the plasma is decided exclusively by check_confinement_flux
    # against the psi = psi_edge surface. The hard-coded box that used to live here
    # (R < 0.7, R > 1.3, |Z| > 0.3) was a stale second boundary that disagreed with that
    # surface -- the real one bulges to |Z| ~ 0.35 and R ~ 1.37 -- so it could only ever
    # report a different answer than the loop it fed.
    return pos_new, vel_new


@njit(fastmath=True)
def _jit_guiding_center_push(pos, v_parallel, mu, B_vec, grad_B, E_vec, m, q, dt):
    B_mag = np.linalg.norm(B_vec)
    if B_mag == 0:
        return pos, v_parallel

    b_unit = B_vec / B_mag
    v_grad_B = (mu / (q * B_mag)) * np.cross(b_unit, grad_B)

    R_vec = np.array([pos[0], pos[1], 0.0])
    R_mag = np.linalg.norm(R_vec)
    if R_mag > 0:
        R_unit = R_vec / R_mag
        v_curv = (m * v_parallel**2 / (q * B_mag * R_mag)) * np.cross(b_unit, R_unit)
    else:
        v_curv = np.zeros(3)

    v_ExB = np.cross(E_vec, B_vec) / (B_mag**2)
    v_drift = v_grad_B + v_curv + v_ExB
    v_total = v_parallel * b_unit + v_drift

    pos_new = pos + v_total * dt

    # As in _jit_lorentz_push: motion only. Loss is check_confinement_flux's decision.
    return pos_new, v_parallel


@njit(fastmath=True)
def apply_cartesian_collisions(vel, nu_c, dt):
    if np.random.rand() < nu_c * dt:
        speed = np.linalg.norm(vel)
        costheta = 2.0 * np.random.rand() - 1.0
        sintheta = np.sqrt(1.0 - costheta**2)
        phi = 2.0 * np.pi * np.random.rand()
        
        vel_new = np.array([
            speed * sintheta * np.cos(phi),
            speed * sintheta * np.sin(phi),
            speed * costheta
        ], dtype=np.float64)
        return vel_new
    return vel


@njit(fastmath=True)
def apply_guiding_center_collisions(v_parallel, mu, B_mag, m, dt, nu_c):
    if B_mag <= 0: return v_parallel, mu
    v_perp = np.sqrt(2.0 * mu * B_mag / m)
    v_mag = np.sqrt(v_parallel**2 + v_perp**2)
    if v_mag == 0: return v_parallel, mu

    pitch_angle = np.arccos(v_parallel / v_mag)
    scattering_variance = nu_c * dt
    delta_pitch = np.random.normal(0.0, np.sqrt(scattering_variance))
    new_pitch = pitch_angle + delta_pitch

    v_parallel_new = v_mag * np.cos(new_pitch)
    v_perp_new = v_mag * np.abs(np.sin(new_pitch))
    mu_new = (m * v_perp_new**2) / (2.0 * B_mag)

    return v_parallel_new, mu_new


@njit(fastmath=True)
def classify_particle_orbit(v_parallel_history):
    sign_changes = 0
    for i in range(1, len(v_parallel_history)):
        if v_parallel_history[i] * v_parallel_history[i-1] < 0.0:
            sign_changes += 1
            break
    return 1 if sign_changes > 0 else 0


@njit(fastmath=True)
def compute_cic_charge_density(R_coords, Z_coords, charges, R_min, R_max, Z_min, Z_max, nR, nZ, apply_neutral_background=False):
    rho = np.zeros((nR, nZ), dtype=np.float64)
    dR = (R_max - R_min) / (nR - 1)
    dZ = (Z_max - Z_min) / (nZ - 1)

    for p in range(len(R_coords)):
        r_p = R_coords[p]
        z_p = Z_coords[p]
        q_p = charges[p]

        if r_p < R_min or r_p > R_max or z_p < Z_min or z_p > Z_max:
            continue

        r_idx = (r_p - R_min) / dR
        z_idx = (z_p - Z_min) / dZ
        i = int(np.floor(r_idx))
        j = int(np.floor(z_idx))
        wR = r_idx - i
        wZ = z_idx - j

        if i < nR - 1 and j < nZ - 1:
            rho[i, j]         += q_p * (1.0 - wR) * (1.0 - wZ)
            rho[i + 1, j]     += q_p * wR * (1.0 - wZ)
            rho[i, j + 1]     += q_p * (1.0 - wR) * wZ
            rho[i + 1, j + 1] += q_p * wR * wZ

    for i in range(nR):
        R_i = R_min + i * dR
        cell_volume = 2.0 * np.pi * R_i * dR * dZ if R_i > 0 else 1.0
        for j in range(nZ):
            rho[i, j] /= cell_volume

    # Jellium model: subtract the mean to stand in for a uniform neutralizing ion background.
    if apply_neutral_background:
        avg_rho = np.mean(rho)
        rho -= avg_rho

    return rho


@njit(fastmath=True)
def solve_poisson_sor_cylindrical(rho, R_min, R_max, Z_min, Z_max, phi_init, max_iter=500, tol=1e-5):
    nR, nZ = rho.shape
    dR = (R_max - R_min) / (nR - 1)
    dZ = (Z_max - Z_min) / (nZ - 1)
    phi = phi_init.copy()
    omega = 1.8
    
    dR2 = dR**2
    dZ2 = dZ**2
    inv_denom = 1.0 / (2.0/dR2 + 2.0/dZ2)

    for it in range(max_iter):
        max_diff = 0.0
        for i in range(1, nR - 1):
            R_i = R_min + i * dR
            for j in range(1, nZ - 1):
                phi_old = phi[i, j]
                
                term_R = (phi[i+1, j] + phi[i-1, j]) / dR2
                term_R_grad = (phi[i+1, j] - phi[i-1, j]) / (2.0 * R_i * dR)
                term_Z = (phi[i, j+1] + phi[i, j-1]) / dZ2
                
                phi_new = inv_denom * (term_R + term_R_grad + term_Z + rho[i, j] / 8.854e-12)
                phi[i, j] = (1.0 - omega) * phi_old + omega * phi_new
                
                diff = abs(phi[i, j] - phi_old)
                if diff > max_diff: max_diff = diff
                    
        if max_diff < tol: break
    return phi


@njit(fastmath=True)
def compute_electric_field(phi, dR, dZ):
    nR, nZ = phi.shape
    E_R = np.zeros((nR, nZ), dtype=np.float64)
    E_Z = np.zeros((nR, nZ), dtype=np.float64)
    
    for i in range(1, nR - 1):
        for j in range(1, nZ - 1):
            E_R[i, j] = -(phi[i+1, j] - phi[i-1, j]) / (2.0 * dR)
            E_Z[i, j] = -(phi[i, j+1] - phi[i, j-1]) / (2.0 * dZ)
            
    return E_R, E_Z


@njit(fastmath=True)
def gather_electric_field(pos, E_R_grid, E_Z_grid, R_min, R_max, Z_min, Z_max, nR, nZ):
    r_p = np.sqrt(pos[0]**2 + pos[1]**2)
    z_p = pos[2]
    
    if r_p < R_min or r_p > R_max or z_p < Z_min or z_p > Z_max:
        return np.zeros(3, dtype=np.float64)
        
    dR = (R_max - R_min) / (nR - 1)
    dZ = (Z_max - Z_min) / (nZ - 1)
    
    r_idx = (r_p - R_min) / dR
    z_idx = (z_p - Z_min) / dZ
    
    i = int(np.floor(r_idx))
    j = int(np.floor(z_idx))
    
    if i >= nR - 1 or j >= nZ - 1 or i < 0 or j < 0:
         return np.zeros(3, dtype=np.float64)
         
    wR = r_idx - i
    wZ = z_idx - j
    
    er = (E_R_grid[i, j] * (1-wR)*(1-wZ) + 
          E_R_grid[i+1, j] * wR*(1-wZ) +
          E_R_grid[i, j+1] * (1-wR)*wZ +
          E_R_grid[i+1, j+1] * wR*wZ)
          
    ez = (E_Z_grid[i, j] * (1-wR)*(1-wZ) + 
          E_Z_grid[i+1, j] * wR*(1-wZ) +
          E_Z_grid[i, j+1] * (1-wR)*wZ +
          E_Z_grid[i+1, j+1] * wR*wZ)
          
    phi_p = np.arctan2(pos[1], pos[0])
    ex = er * np.cos(phi_p)
    ey = er * np.sin(phi_p)
    
    return np.array([ex, ey, ez], dtype=np.float64)


@njit(fastmath=True)
def compute_electrostatic_energy(E_R, E_Z, R_min, R_max, Z_min, Z_max):
    # Integrates 1/2 eps_0 E^2 over the grid volume to track wave energy
    nR, nZ = E_R.shape
    dR = (R_max - R_min) / (nR - 1)
    dZ = (Z_max - Z_min) / (nZ - 1)
    
    w_es = 0.0
    for i in range(nR):
        R_i = R_min + i * dR
        cell_volume = 2.0 * np.pi * R_i * dR * dZ if R_i > 0 else 1.0
        for j in range(nZ):
            e_mag_sq = E_R[i, j]**2 + E_Z[i, j]**2
            w_es += 0.5 * 8.854e-12 * e_mag_sq * cell_volume
            
    return w_es

@njit(fastmath=True)
def _jit_apply_magnetic_perturbation(pos, t, b_amp, m, n, gamma):
    # Localized delta_B from an exponentially growing m/n tearing mode (island).

    # To cylindrical coordinates
    R = np.sqrt(pos[0]**2 + pos[1]**2)
    Z = pos[2]
    phi = np.arctan2(pos[1], pos[0])

    # Minor radius and poloidal angle about the magnetic axis (R_0 = 1.0)
    R_0 = 1.0
    r = np.sqrt((R - R_0)**2 + Z**2)

    if r == 0:
        theta = 0.0
    else:
        theta = np.arctan2(Z, R - R_0)

    # Instability growth: amplitude = b_amp * e^(gamma * t)
    amplitude = b_amp * np.exp(gamma * t)

    # Helical island perturbation (divergence-free approximation)
    dB_R = amplitude * np.sin(m * theta - n * phi)
    dB_Z = amplitude * np.cos(m * theta - n * phi)

    # dB_R back to Cartesian for the pusher
    dB_x = dB_R * np.cos(phi)
    dB_y = dB_R * np.sin(phi)
    
    return np.array([dB_x, dB_y, dB_Z], dtype=np.float64)

# =======================================================
# DISRUPTION MITIGATION FLUID FUNCTIONS
# =======================================================

def compute_radiative_cooling_power(n_e, n_z, T_e, cooling_coeff):
    # Radiative power loss P_rad = n_e * n_z * L_z(T_e) [W/m^3] after a Shattered
    # Pellet Injection introduces heavy impurities (Neon/Argon).

    # Keep the sqrt proxy well-defined at low/negative T
    safe_Te = np.clip(T_e, 0.1, None)

    # Simplified proxy for line radiation/bremsstrahlung
    p_rad = n_e * n_z * cooling_coeff * np.sqrt(safe_Te)

    return p_rad

def trigger_thermal_quench(current_temp_profile, time_since_trigger, tq_timescale, post_quench_temp):
    # Rapid Thermal Quench: exponentially decays the core temperature over a few ms.

    decay_factor = np.exp(-time_since_trigger / tq_timescale)

    # Decays toward the cold baseline, not absolute zero
    new_temp_profile = (current_temp_profile - post_quench_temp) * decay_factor + post_quench_temp
    
    return np.maximum(new_temp_profile, post_quench_temp)

# =======================================================
# NUCLEAR REACTION DYNAMICS (QUANTUM TUNNELING)
# =======================================================

@njit(fastmath=True)
def compute_dt_cross_section(E_keV):
    # D-T fusion cross-section in barns, from the Gamow tunneling factor:
    #   sigma(E) = (S(E) / E) * exp(-sqrt(E_G / E))

    # Guard against division by zero for near-stationary particles
    if E_keV <= 0.05:
        return 0.0

    E_G = 34.38  # Gamow energy for D-T (keV)

    # Breit-Wigner resonance on the S-factor, modelling the spike at 64 keV
    resonance_width = 150.0
    S_E = 1000.0 + (80000.0) / ((E_keV - 64.0)**2 + resonance_width)

    gamow_factor = np.exp(-np.sqrt(E_G / E_keV))

    sigma_barns = (S_E / E_keV) * gamow_factor

    return sigma_barns

# =======================================================
# REACTIVITY MATRICES & VOLUMETRIC FUSION RATES
# =======================================================

@njit(fastmath=True)
def compute_bosch_hale_reactivity(T_i_keV, C1, C2, C3, C4, C5, C6, C7):
    # Maxwellian-averaged D-T reactivity <sigma v> in m^3/s, via the analytical
    # Bosch-Hale parameterization.

    if T_i_keV <= 0.1:
        return 0.0

    # Dimensionless parameter theta
    numerator = T_i_keV * (C2 + T_i_keV * (C4 + T_i_keV * C6))
    denominator = 1.0 + T_i_keV * (C3 + T_i_keV * (C5 + T_i_keV * C7))
    theta = T_i_keV / (1.0 - numerator / denominator)

    # Dimensionless parameter xi (B_G = 34.3827)
    B_G_sq = 1182.17005929
    xi = (B_G_sq / (4.0 * theta))**(1.0/3.0)

    # Bosch-Hale fit, cm^3/s
    sigmav_cm3_s = C1 * theta * np.sqrt(xi / (T_i_keV**3)) * np.exp(-3.0 * xi)

    return sigmav_cm3_s * 1e-6  # -> m^3/s


def compute_volumetric_fusion_power(n_D_grid, n_T_grid, T_i_grid_keV, cfg):
    # Volumetric fusion power (W/m^3) across the 2D grid:
    #   P_fusion = n_D * n_T * <sigma v> * E_fusion

    nR, nZ = n_D_grid.shape
    P_fusion_grid = np.zeros((nR, nZ), dtype=np.float64)
    
    for i in range(nR):
        for j in range(nZ):
            n_D = n_D_grid[i, j]
            n_T = n_T_grid[i, j]
            T_i = T_i_grid_keV[i, j]
            
            # Skip empty vacuum regions
            if n_D <= 0 or n_T <= 0 or T_i <= 0.1:
                continue

            reactivity = compute_bosch_hale_reactivity(
                T_i, cfg.BH_C1, cfg.BH_C2, cfg.BH_C3,
                cfg.BH_C4, cfg.BH_C5, cfg.BH_C6, cfg.BH_C7
            )

            P_fusion_grid[i, j] = n_D * n_T * reactivity * cfg.E_FUSION_JOULES
            
    return P_fusion_grid

@njit(fastmath=True)
def compute_radiation_losses(n_e, T_e_keV, B_mag, Z_eff, V_cell, cyc_reabsorption=5.0e-4):

    # Bremsstrahlung and cyclotron radiation power lost from one grid cell (Watts).
    # n_e [m^-3], T_e_keV, B_mag [T], Z_eff, V_cell [m^3], and cyc_reabsorption =
    # the net fraction of raw cyclotron emission that actually escapes.

    P_brem_density = 0.0
    P_cyc_density = 0.0

    # Only cells with active plasma
    if T_e_keV > 0.0 and n_e > 0.0:
        # Bremsstrahlung (X-rays). Optically thin, so essentially all of it escapes --
        # the dominant radiative channel in a D-T reactor.
        P_brem_density = 4.8e-37 * Z_eff * (n_e**2) * np.sqrt(T_e_keV)

        # Cyclotron (microwave). The coefficient is RAW emission; a reactor is optically
        # thick at the fundamental and the wall reflects most of the rest, so net loss is
        # a small fraction of brem (the old 0.05 factor put it ~2000x above).
        # cyc_reabsorption is a lumped, tuned stand-in -- see cfg.CYCLOTRON_REABSORPTION.
        if B_mag > 0.0:
            P_cyc_density = 6.21e-17 * n_e * T_e_keV * (B_mag**2) * cyc_reabsorption

    # Power density (W/m^3) * cell volume (m^3) -> Watts
    P_brem_watts = P_brem_density * V_cell
    P_cyc_watts = P_cyc_density * V_cell

    return P_brem_watts, P_cyc_watts


@njit(fastmath=True)
def compute_radiation_losses_grid(n_e_grid_raw, T_core, R_min, dR, dZ, nR, nZ, B0, Z_eff, scale_factor, norm_n_e,
                                  cyc_reabsorption=5.0e-4, R0=1.0):
    # The whole (nR x nZ) radiation-loss reduction as one compiled Numba loop, instead
    # of nR*nZ Python-level calls into compute_radiation_losses.
    total_brem_watts = 0.0
    total_cyc_watts = 0.0
    for i in range(nR):
        R_i = R_min + i * dR
        V_cell = 2.0 * np.pi * R_i * dR * dZ if R_i > 0 else 1.0
        # Toroidal field falls as R0/R about the major radius
        B_mag_cell = B0 * (R0 / R_i) if R_i > 0 else B0
        for j in range(nZ):
            n_e_cell = n_e_grid_raw[i, j] * scale_factor
            T_e_cell = T_core * (n_e_grid_raw[i, j] * norm_n_e)
            p_b, p_c = compute_radiation_losses(n_e_cell, T_e_cell, B_mag_cell, Z_eff, V_cell, cyc_reabsorption)
            total_brem_watts += p_b
            total_cyc_watts += p_c
    return total_brem_watts, total_cyc_watts

@njit(fastmath=True)
def evaluate_q_factors(P_fus_MW, P_ext_MW, eta_th, eta_heat):
    # Scientific gain Q_sci (thermal out / thermal in) and engineering gain Q_eng
    # (electrical out / electrical in).

    Q_sci = P_fus_MW / P_ext_MW if P_ext_MW > 0.0 else 0.0

    # Assumes the blanket captures 100% of fusion + external heat
    P_elec_out = (P_fus_MW + P_ext_MW) * eta_th

    # Grid power needed to run the heating systems
    P_elec_in = P_ext_MW / eta_heat
    
    Q_eng = P_elec_out / P_elec_in if P_elec_in > 0.0 else 0.0
    
    return Q_sci, Q_eng, P_elec_out, P_elec_in


# =====================================================================
# HPC, MULTI-CORE PROCESSING & METAL GPU ACCELERATION
# =====================================================================

def _vectorized_boris_push_metal_impl(pos_tensor, vel_tensor, q, m, B_tensor, E_tensor, dt):

    # Parallel Boris push on the GPU. Inputs are (N, 3) tensors on the device.

    # 1. Half-step electric acceleration
    q_m = q / m
    v_minus = vel_tensor + (q_m * E_tensor * (dt / 2.0))

    # 2. Magnetic rotation vectors t and s
    t = q_m * B_tensor * (dt / 2.0)
    t_mag2 = torch.sum(t**2, dim=1, keepdim=True)
    s = 2.0 * t / (1.0 + t_mag2)

    # 3. Cross products across the whole matrix
    v_prime = v_minus + torch.cross(v_minus, t, dim=1)
    v_plus = v_minus + torch.cross(v_prime, s, dim=1)

    # 4. Final half-step acceleration and position update
    vel_next = v_plus + (q_m * E_tensor * (dt / 2.0))
    pos_next = pos_tensor + (vel_next * dt)

    return pos_next, vel_next


# Static-shape compile. Eager issues ~16-20 Metal dispatches at ~18 us each, a ~0.4 ms
# floor at ANY particle count; fusing collapses that to one or two. The benchmark and
# oscillation-test call sites use fixed shapes, so static compile applies and warms up
# cheaply (~15ms per N). Measured warm at N=50,000: 0.426 ms eager -> 0.069 ms.
# Falls back to eager on compile failure.
try:
    _vectorized_boris_push_metal_compiled = torch.compile(_vectorized_boris_push_metal_impl)
except Exception:
    _vectorized_boris_push_metal_compiled = None


# Dynamic-shape compile, for call sites where N changes call to call (the reactor loop:
# injection and wall-loss compaction move N every few steps, and a static compile would
# re-specialize ~15ms per new shape). Earlier notes claimed dynamic=True broke on any
# kernel with a reduction; re-verified on torch 2.8.0, it does not. Measured warm at
# N=50,000: 0.426 ms eager -> 0.098 ms, holding at 0.100 ms across shifting shapes.
#
# Kept for the GPU-resident loop only -- at every count this project runs, the CPU/Numba
# push beats any GPU path (see vectorized_boris_push_numba_fallback).
try:
    _vectorized_boris_push_metal_dynamic = torch.compile(_vectorized_boris_push_metal_impl, dynamic=True)
except Exception:
    _vectorized_boris_push_metal_dynamic = None


class HPCPhysicsAccelerator:

    # Hardware-accelerated physics: parallel arrays via Apple Metal (MPS) or Numba prange.

    def __init__(self, device):
        self.device = device

    def vectorized_boris_push_metal(self, pos_tensor, vel_tensor, q, m, B_tensor, E_tensor, dt):
        if _vectorized_boris_push_metal_compiled is not None and pos_tensor.device.type in ("mps", "cuda"):
            try:
                return _vectorized_boris_push_metal_compiled(pos_tensor, vel_tensor, q, m, B_tensor, E_tensor, dt)
            except Exception:
                pass
        return _vectorized_boris_push_metal_impl(pos_tensor, vel_tensor, q, m, B_tensor, E_tensor, dt)

# CPU multi-threaded push (Numba prange).
#
# Written in scalar components rather than 3-element array slices. Slices allocate ~a
# dozen heap temporaries per particle per step, and that allocation traffic -- not the
# arithmetic -- was the entire cost: ~0.9 GB/s against a ~40 GB/s streaming ceiling.
# Scalars stay in registers and let Numba vectorize the prange body. Measured:
#
#        N        before        after
#   10,000       1.693 ms     0.105 ms
#   50,000      11.446 ms     0.308 ms
#  200,000      39.728 ms     0.397 ms
# 1,000,000    195.125 ms     1.964 ms   (~36.7 GB/s -- at the bandwidth roof)
# 4,000,000    804.531 ms     6.998 ms
#
# Bit-identical to the old kernel but for one float32 ULP on position (5.96e-08 relative;
# velocities exactly equal), from not round-tripping intermediates through float64.
#
# This is what makes CPU the fastest path at every count this project runs; see
# PUSH_GPU_THRESHOLD in main.py.
@njit(parallel=True, fastmath=True, cache=True)
def vectorized_boris_push_numba_fallback(pos_arr, vel_arr, q, m, B_arr, E_arr, dt):

    qm = q / m
    half = dt / 2.0
    # prange splits the loop into chunks processed simultaneously
    for i in prange(pos_arr.shape[0]):
        ex = E_arr[i, 0]; ey = E_arr[i, 1]; ez = E_arr[i, 2]

        # 1. Half-step electric acceleration
        vmx = vel_arr[i, 0] + qm * ex * half
        vmy = vel_arr[i, 1] + qm * ey * half
        vmz = vel_arr[i, 2] + qm * ez * half

        # 2. Magnetic rotation vectors t and s
        tx = qm * B_arr[i, 0] * half
        ty = qm * B_arr[i, 1] * half
        tz = qm * B_arr[i, 2] * half
        f = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz)
        sx = tx * f; sy = ty * f; sz = tz * f

        # 3. v' = v- + (v- x t), then v+ = v- + (v' x s), cross products inlined
        px = vmy * tz - vmz * ty
        py = vmz * tx - vmx * tz
        pz = vmx * ty - vmy * tx
        vpx = vmx + px; vpy = vmy + py; vpz = vmz + pz

        cx = vpy * sz - vpz * sy
        cy = vpz * sx - vpx * sz
        cz = vpx * sy - vpy * sx

        # 4. Final half-step electric acceleration, then position update
        nvx = vmx + cx + qm * ex * half
        nvy = vmy + cy + qm * ey * half
        nvz = vmz + cz + qm * ez * half

        vel_arr[i, 0] = nvx; vel_arr[i, 1] = nvy; vel_arr[i, 2] = nvz
        pos_arr[i, 0] += nvx * dt
        pos_arr[i, 1] += nvy * dt
        pos_arr[i, 2] += nvz * dt


# =====================================================================
# ALPHA SUB-STEPPING
#
# A 3.5 MeV alpha covers ~1.3e-2 m per global 1 ns step against a ~2.2e-2 m Larmor
# radius -- under two samples per gyro-arc, hence jagged orbits. Deuterons are fine at
# the global step (Larmor radius ~5e-4 m, ~3e-4 m of travel per step).
#
# Sub-stepping only the alphas fixes the sampling without paying for it on the far more
# numerous thermals. E and B are held fixed across the sub-steps -- evaluated once at
# the start-of-step position, since the point is to resolve gyration about that local B,
# not to re-gather the field 10x per step.
# =====================================================================

def boris_push_substeps_torch(pos_tensor, vel_tensor, q, m, B_tensor, E_tensor, dt, n_sub):
    if n_sub is None or n_sub <= 1:
        return _vectorized_boris_push_metal_impl(pos_tensor, vel_tensor, q, m, B_tensor, E_tensor, dt)

    dt_sub = dt / float(n_sub)
    pos, vel = pos_tensor, vel_tensor
    for _ in range(int(n_sub)):
        pos, vel = _vectorized_boris_push_metal_impl(pos, vel, q, m, B_tensor, E_tensor, dt_sub)
    return pos, vel


@njit(parallel=True, fastmath=True, cache=True)
def vectorized_boris_push_numba_substeps(pos_arr, vel_arr, q, m, B_arr, E_arr, dt, n_sub):
    # Numba/CPU counterpart. Mutates pos_arr/vel_arr in place, matching
    # vectorized_boris_push_numba_fallback's calling convention.
    #
    # Scalar components, for the same reason as the single-step kernel above -- and the
    # slice form paid that allocation cost once PER SUB-STEP. Two extra wins: t and s
    # depend only on fixed quantities, so they hoist out of the inner loop; and the
    # float64 accumulators become plain scalars, no .astype() round-trip.
    # Verified bit-identical to the previous kernel (max abs difference 0.0 in position
    # and velocity over 20,000 randomised particles at n_sub = 10).
    dt_sub = dt / n_sub
    qm = q / m
    half = dt_sub / 2.0
    for i in prange(pos_arr.shape[0]):
        ex = E_arr[i, 0]; ey = E_arr[i, 1]; ez = E_arr[i, 2]

        # Rotation vectors are constant across the sub-steps (B and E are frozen at the
        # start-of-step position), so hoist them out of the inner loop.
        tx = qm * B_arr[i, 0] * half
        ty = qm * B_arr[i, 1] * half
        tz = qm * B_arr[i, 2] * half
        f = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz)
        sx = tx * f; sy = ty * f; sz = tz * f

        # Accumulate in float64: summing ALPHA_SUBSTEPS small position increments in the
        # arrays' native float32 would reintroduce the roundoff sub-stepping cures.
        px = np.float64(pos_arr[i, 0]); py = np.float64(pos_arr[i, 1]); pz = np.float64(pos_arr[i, 2])
        vx = np.float64(vel_arr[i, 0]); vy = np.float64(vel_arr[i, 1]); vz = np.float64(vel_arr[i, 2])

        for _ in range(n_sub):
            vmx = vx + qm * ex * half
            vmy = vy + qm * ey * half
            vmz = vz + qm * ez * half

            ax = vmy * tz - vmz * ty
            ay = vmz * tx - vmx * tz
            az = vmx * ty - vmy * tx
            vpx = vmx + ax; vpy = vmy + ay; vpz = vmz + az

            cx = vpy * sz - vpz * sy
            cy = vpz * sx - vpx * sz
            cz = vpx * sy - vpy * sx

            vx = vmx + cx + qm * ex * half
            vy = vmy + cy + qm * ey * half
            vz = vmz + cz + qm * ez * half

            px += vx * dt_sub; py += vy * dt_sub; pz += vz * dt_sub

        # Back down to the storage dtype on write-out
        vel_arr[i, 0] = vx; vel_arr[i, 1] = vy; vel_arr[i, 2] = vz
        pos_arr[i, 0] = px; pos_arr[i, 1] = py; pos_arr[i, 2] = pz


# =====================================================================
# GPU-RESIDENT KERNELS FOR LARGE PARTICLE COUNTS (>= ~100K)
#
# The reactor loop is small-N (~10K-20K) by default, where CPU/Numba wins outright --
# MPS dispatch overhead dominates the compute at that scale. At large N (e.g. 1M,
# run_hpc_benchmark's top tier) the compute per kernel is big enough that dispatch
# overhead stops mattering. These kernels back that path.
#
# Compile notes:
#   - vectorized_gather_and_B: no reductions, so dynamic=True works and is worth ~9x.
#     The `t` argument is wrapped as a tensor first -- a raw Python float that changes
#     every call makes Dynamo recompile constantly.
#   - check_confinement / apply_vectorized_collisions / compute_alpha_heating_power:
#     contain torch.sum/torch.any, which hit an Inductor bug on this MPS backend under
#     dynamic=True ("cannot determine truth value of Relational"). Left eager -- still a
#     large win over CPU/Numba at N=1M.
#   - compute_cic_charge_density: index_add_ falls back to Inductor's C++ codegen under
#     compile, whose build command breaks on the space in this project's folder name.
#     Left eager, where no C++ step is involved.
# =====================================================================

def compute_cic_charge_density_torch(R_coords, Z_coords, charge, R_min, R_max, Z_min, Z_max, nR, nZ, device, apply_neutral_background=False):
    rho = torch.zeros((nR, nZ), dtype=torch.float32, device=device)
    if R_coords.numel() == 0:
        return rho

    dR = (R_max - R_min) / (nR - 1)
    dZ = (Z_max - Z_min) / (nZ - 1)

    r_idx = (R_coords - R_min) / dR
    z_idx = (Z_coords - Z_min) / dZ
    i = torch.floor(r_idx).long()
    j = torch.floor(z_idx).long()

    valid = (R_coords >= R_min) & (R_coords <= R_max) & (Z_coords >= Z_min) & (Z_coords <= Z_max) & \
            (i >= 0) & (i < nR - 1) & (j >= 0) & (j < nZ - 1)

    i = i[valid]
    j = j[valid]
    wR = (r_idx[valid] - i.to(torch.float32))
    wZ = (z_idx[valid] - j.to(torch.float32))
    q = torch.full((i.numel(),), float(charge), dtype=torch.float32, device=device)

    rho_flat = rho.view(-1)
    rho_flat.index_add_(0, i * nZ + j, q * (1.0 - wR) * (1.0 - wZ))
    rho_flat.index_add_(0, (i + 1) * nZ + j, q * wR * (1.0 - wZ))
    rho_flat.index_add_(0, i * nZ + (j + 1), q * (1.0 - wR) * wZ)
    rho_flat.index_add_(0, (i + 1) * nZ + (j + 1), q * wR * wZ)

    R_i = R_min + torch.arange(nR, device=device, dtype=torch.float32) * dR
    cell_volume = torch.where(R_i > 0, 2.0 * float(np.pi) * R_i * dR * dZ, torch.ones_like(R_i))
    rho = rho / cell_volume.unsqueeze(1)

    if apply_neutral_background:
        rho = rho - rho.mean()

    return rho


def _vectorized_gather_and_B_torch_impl(pos, E_R_grid_t, E_Z_grid_t, B_R_pol_t, B_Z_pol_t,
                                        R_min, R_max, Z_min, Z_max, nR, nZ, B0, R0, t, b_perturb, m_mode, n_mode, gamma):
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    r = torch.sqrt(x**2 + y**2)
    phi = torch.atan2(y, x)

    dR = (R_max - R_min) / (nR - 1)
    dZ = (Z_max - Z_min) / (nZ - 1)

    r_idx = (r - R_min) / dR
    z_idx = (z - Z_min) / dZ
    i = torch.floor(r_idx).long()
    j = torch.floor(z_idx).long()

    valid = (r >= R_min) & (r <= R_max) & (z >= Z_min) & (z <= Z_max) & \
            (i >= 0) & (i < nR - 1) & (j >= 0) & (j < nZ - 1)

    i_c = torch.clamp(i, 0, nR - 2)
    j_c = torch.clamp(j, 0, nZ - 2)
    wR = r_idx - i_c.to(torch.float32)
    wZ = z_idx - j_c.to(torch.float32)

    def bilinear(grid):
        g00 = grid[i_c, j_c]
        g10 = grid[i_c + 1, j_c]
        g01 = grid[i_c, j_c + 1]
        g11 = grid[i_c + 1, j_c + 1]
        return g00 * (1 - wR) * (1 - wZ) + g10 * wR * (1 - wZ) + g01 * (1 - wR) * wZ + g11 * wR * wZ

    zeros = torch.zeros_like(r)
    er = torch.where(valid, bilinear(E_R_grid_t), zeros)
    ez = torch.where(valid, bilinear(E_Z_grid_t), zeros)

    ex = er * torch.cos(phi)
    ey = er * torch.sin(phi)
    E_out = torch.stack([ex, ey, ez], dim=1)

    # Toroidal field: B_phi = B0 * R0 / R. R0 must match the value used by the
    # confinement centre, tearing mode and poloidal field, or |B| and every Larmor
    # radius come out scaled wrong.
    r_safe = torch.clamp(r, min=1e-12)
    b_mag_tor = torch.where(r > 0, B0 * (R0 / r_safe), torch.full_like(r, B0))
    Bx = -b_mag_tor * torch.sin(phi)
    By = b_mag_tor * torch.cos(phi)
    Bz = torch.zeros_like(r)

    # Poloidal field: interpolated from the psi-derived B_R / B_Z grids, reusing the
    # bilinear indices/weights already computed for the E gather. Replaces a
    # linear-in-position stand-in with no real gradient structure, which is what makes
    # grad-B drift and the mirror force physical.
    B_R_pol = torch.where(valid, bilinear(B_R_pol_t), zeros)
    B_Z_pol = torch.where(valid, bilinear(B_Z_pol_t), zeros)
    Bx = Bx + B_R_pol * torch.cos(phi)
    By = By + B_R_pol * torch.sin(phi)
    Bz = Bz + B_Z_pol

    if b_perturb > 0.0:
        amplitude = b_perturb * torch.exp(gamma * t)
        theta = torch.where(r != R0, torch.atan2(z, r - R0), torch.zeros_like(r))
        dB_R = amplitude * torch.sin(m_mode * theta - n_mode * phi)
        dB_Z = amplitude * torch.cos(m_mode * theta - n_mode * phi)
        Bx = Bx + dB_R * torch.cos(phi)
        By = By + dB_R * torch.sin(phi)
        Bz = Bz + dB_Z

    B_out = torch.stack([Bx, By, Bz], dim=1)
    return E_out, B_out


try:
    _vectorized_gather_and_B_torch_compiled = torch.compile(_vectorized_gather_and_B_torch_impl, dynamic=True)
except Exception:
    _vectorized_gather_and_B_torch_compiled = None


def vectorized_gather_and_B_torch(pos, E_R_grid_t, E_Z_grid_t, B_R_pol_t, B_Z_pol_t,
                                  R_min, R_max, Z_min, Z_max, nR, nZ, B0, R0, t, b_perturb, m_mode, n_mode, gamma):
    t_tensor = t if torch.is_tensor(t) else torch.as_tensor(t, device=pos.device, dtype=pos.dtype)

    if _vectorized_gather_and_B_torch_compiled is not None and pos.device.type == "mps":
        try:
            return _vectorized_gather_and_B_torch_compiled(
                pos, E_R_grid_t, E_Z_grid_t, B_R_pol_t, B_Z_pol_t, R_min, R_max, Z_min, Z_max, nR, nZ,
                B0, R0, t_tensor, b_perturb, m_mode, n_mode, gamma
            )
        except Exception:
            pass
    return _vectorized_gather_and_B_torch_impl(
        pos, E_R_grid_t, E_Z_grid_t, B_R_pol_t, B_Z_pol_t, R_min, R_max, Z_min, Z_max, nR, nZ,
        B0, R0, t_tensor, b_perturb, m_mode, n_mode, gamma
    )


# NOTE: the old Cartesian/circular check_confinement_torch was deleted from here -- it
# was shadowed by, and is superseded by, the psi-surface version further down this file.


def apply_vectorized_collisions_torch(vel_tensor, type_tensor, nu_c, dt):
    # Eager only -- the torch.any reduction hits the dynamic-shape Inductor bug.
    device = vel_tensor.device
    N = vel_tensor.shape[0]

    eligible = (type_tensor == 0) | (type_tensor == 1)
    rolls = torch.rand((N, 3), device=device)
    collide_mask = eligible & (rolls[:, 0] < (nu_c * dt))

    if not torch.any(collide_mask):
        return vel_tensor

    speed = torch.linalg.norm(vel_tensor, dim=1)
    costheta = 2.0 * rolls[:, 1] - 1.0
    sintheta = torch.sqrt(torch.clamp(1.0 - costheta**2, min=0.0))
    phi = 2.0 * float(np.pi) * rolls[:, 2]

    new_vel = torch.stack([
        speed * sintheta * torch.cos(phi),
        speed * sintheta * torch.sin(phi),
        speed * costheta
    ], dim=1)

    return torch.where(collide_mask.unsqueeze(1), new_vel, vel_tensor)


def compute_alpha_heating_power_torch(alpha_energies_kev, dt, cfg):
    # Eager only -- the torch.sum reduction hits the dynamic-shape Inductor bug.
    if alpha_energies_kev.numel() == 0:
        return alpha_energies_kev, 0.0, 0.0

    # Same float32 cancellation as the numpy version, but MPS has no float64 so the
    # promote-to-double remedy isn't available. Compute the deposit directly instead:
    # E * 4e-9 is representable in float32 (only the `E - E*4e-9` subtraction collapses),
    # so summing the drain term keeps full precision on-device.
    E = alpha_energies_kev
    energy_loss_fraction = min(2.0 * cfg.NU_S * dt, 0.99)

    headroom = torch.clamp(E - cfg.THERMALIZATION_ENERGY_KEV, min=0.0)
    deposited_kev = torch.minimum(E * energy_loss_fraction, headroom)

    new_alpha_energies = E - deposited_kev

    # The reduction is over the precise per-particle drain, so no cancellation here
    total_energy_deposited_kev = float(torch.sum(deposited_kev).item())

    joules_per_kev = 1.602e-16
    macro_weight = cfg.MACRO_WEIGHT_REACTOR   # see config: shared with the Lawson math
    total_joules = total_energy_deposited_kev * joules_per_kev * macro_weight
    power_watts = total_joules / dt
    power_mw = power_watts / 1e6

    return new_alpha_energies, power_mw, total_energy_deposited_kev


# =====================================================================
# MAGNETIC FLUX (psi) SURFACE BOUNDARIES
#
# Confinement follows the poloidal flux contours of the solved Grad-Shafranov
# equilibrium rather than a hard-coded circle. This solver's psi is a dome peaking at
# the magnetic axis and pinned to 0 at the rectangular domain edges (Dirichlet), so:
#   psi(R, Z) > psi_edge  ->  inside the last closed flux surface (confined)
#   psi(R, Z) <= psi_edge ->  lost
# psi_edge is calibrated as the minimum psi sampled around the legacy circular boundary
# (R0=1.0, a_minor=0.3), so everything the old circle confined is still confined and the
# new boundary can only extend outward along the real topology. psi_core (used for
# particle birth, not loss) is the midpoint between psi_edge and the axis.
# =====================================================================

@njit(fastmath=True)
def interpolate_psi(r_p, z_p, psi_grid, R_min, R_max, Z_min, Z_max, nR, nZ):
    # Bilinear lookup on the (R, Z)-ordered psi grid, as in gather_electric_field.
    if r_p < R_min or r_p > R_max or z_p < Z_min or z_p > Z_max:
        return 0.0

    dR = (R_max - R_min) / (nR - 1)
    dZ = (Z_max - Z_min) / (nZ - 1)

    r_idx = (r_p - R_min) / dR
    z_idx = (z_p - Z_min) / dZ

    i = int(np.floor(r_idx))
    j = int(np.floor(z_idx))

    if i >= nR - 1 or j >= nZ - 1 or i < 0 or j < 0:
        return 0.0

    wR = r_idx - i
    wZ = z_idx - j

    return (psi_grid[i, j] * (1 - wR) * (1 - wZ) +
            psi_grid[i + 1, j] * wR * (1 - wZ) +
            psi_grid[i, j + 1] * (1 - wR) * wZ +
            psi_grid[i + 1, j + 1] * wR * wZ)


@njit(parallel=True, fastmath=True)
def interpolate_psi_array(R_coords, Z_coords, psi_grid, R_min, R_max, Z_min, Z_max, nR, nZ):
    N = len(R_coords)
    out = np.zeros(N, dtype=np.float64)
    for k in prange(N):
        out[k] = interpolate_psi(R_coords[k], Z_coords[k], psi_grid, R_min, R_max, Z_min, Z_max, nR, nZ)
    return out


def compute_flux_thresholds(eq, R0=1.0, a_minor=0.3, n_samples=128):
    # Called once after eq.solve_grad_shafranov(). Not numba -- touches the eq object
    # directly and runs only a handful of times per simulation.
    psi_grid = np.ascontiguousarray(eq.psi.T)  # (Z, R) in mhd_equilibrium -> (R, Z) here
    R_min, R_max = float(eq.R_1d[0]), float(eq.R_1d[-1])
    Z_min, Z_max = float(eq.Z_1d[0]), float(eq.Z_1d[-1])
    nR_psi, nZ_psi = psi_grid.shape

    psi_axis = float(np.max(psi_grid))

    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    R_ring = R0 + a_minor * np.cos(theta)
    Z_ring = a_minor * np.sin(theta)
    psi_ring = interpolate_psi_array(R_ring, Z_ring, psi_grid, R_min, R_max, Z_min, Z_max, nR_psi, nZ_psi)

    psi_edge = float(np.min(psi_ring))
    if psi_edge >= psi_axis:  # degenerate fallback; shouldn't trigger for this geometry
        psi_edge = 0.1 * psi_axis
    psi_core = psi_edge + 0.5 * (psi_axis - psi_edge)

    return psi_axis, psi_edge, psi_core, psi_grid, R_min, R_max, Z_min, Z_max, nR_psi, nZ_psi


def compute_flux_boundary_contour(psi_grid, psi_edge, R_min, R_max, Z_min, Z_max, nR, nZ,
                                  n_theta=241, n_radial=2000):
    # The psi = psi_edge contour as a closed (R, Z) polygon -- the SAME surface
    # check_confinement_flux uses to declare a particle lost, so the plots can draw the
    # wall where the physics actually puts it instead of at the legacy r = 0.3 circle.
    #
    # Traced by ray-marching outward from the magnetic axis rather than by contouring,
    # which keeps the result ordered by poloidal angle and closed -- exactly the form the
    # 3D renderer needs to revolve it into a surface of revolution. The psi dome is
    # star-shaped about its own maximum, so one crossing per ray is well defined.
    i_ax, j_ax = np.unravel_index(int(np.argmax(psi_grid)), psi_grid.shape)
    dR = (R_max - R_min) / (nR - 1)
    dZ = (Z_max - Z_min) / (nZ - 1)
    R_axis = R_min + i_ax * dR
    Z_axis = Z_min + j_ax * dZ

    # endpoint=True so the polygon closes on itself for plotting
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    r_span = float(max(R_max - R_min, Z_max - Z_min))
    r_samples = np.linspace(0.0, r_span, n_radial)

    # Evaluate psi on every ray at once: (n_theta, n_radial)
    RR = R_axis + r_samples[None, :] * np.cos(theta)[:, None]
    ZZ = Z_axis + r_samples[None, :] * np.sin(theta)[:, None]
    psi_rays = interpolate_psi_array(
        np.ascontiguousarray(RR.ravel()), np.ascontiguousarray(ZZ.ravel()),
        psi_grid, R_min, R_max, Z_min, Z_max, nR, nZ
    ).reshape(n_theta, n_radial)

    # First sample at or below psi_edge along each ray is the crossing. interpolate_psi
    # returns 0.0 outside the solver box, which is below psi_edge, so a ray that never
    # crosses inside the plasma still terminates at the domain edge.
    below = psi_rays <= psi_edge
    below[:, -1] = True
    k_cross = np.argmax(below, axis=1)
    k_cross = np.maximum(k_cross, 1)

    # Linear interpolation between the bracketing samples for sub-sample accuracy
    k_prev = k_cross - 1
    rows = np.arange(n_theta)
    psi_hi = psi_rays[rows, k_prev]
    psi_lo = psi_rays[rows, k_cross]
    r_hi = r_samples[k_prev]
    r_lo = r_samples[k_cross]
    denom = psi_hi - psi_lo
    frac = np.where(np.abs(denom) > 1e-30, (psi_hi - psi_edge) / np.where(denom == 0, 1.0, denom), 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    r_edge = r_hi + frac * (r_lo - r_hi)

    R_wall = R_axis + r_edge * np.cos(theta)
    Z_wall = Z_axis + r_edge * np.sin(theta)
    return R_wall, Z_wall


def compute_plasma_volume(R_wall, Z_wall):
    # Volume enclosed by revolving the psi_edge contour about the Z axis (Pappus):
    #     V = 2*pi * R_centroid * A_cross_section
    # Using the real contour rather than 2*pi^2*R0*a^2 matters -- the flux surface is not
    # the a = 0.3 circle, and the true volume is ~2.79 m^3 against that formula's 1.78.
    R = np.asarray(R_wall, dtype=np.float64)
    Z = np.asarray(Z_wall, dtype=np.float64)
    if R[0] == R[-1] and Z[0] == Z[-1]:   # drop the duplicated closing vertex
        R, Z = R[:-1], Z[:-1]

    cross = R * np.roll(Z, -1) - np.roll(R, -1) * Z
    A_signed = 0.5 * np.sum(cross)
    if abs(A_signed) < 1e-30:
        return 0.0
    # Shoelace area centroid in R
    R_centroid = np.sum((R + np.roll(R, -1)) * cross) / (6.0 * A_signed)
    return float(2.0 * np.pi * R_centroid * abs(A_signed))


def resample_grid_to(src_grid, src_R_min, src_R_max, src_Z_min, src_Z_max, src_nR, src_nZ,
                     dst_R_min, dst_R_max, dst_Z_min, dst_Z_max, dst_nR, dst_nZ):
    # Bilinearly resample a scalar field from the equilibrium's psi grid (60x60) onto the
    # solver grid (nR x nZ), so the B lookup can reuse the E gather's indices/weights.
    Rd = np.linspace(dst_R_min, dst_R_max, dst_nR)
    Zd = np.linspace(dst_Z_min, dst_Z_max, dst_nZ)
    RR, ZZ = np.meshgrid(Rd, Zd, indexing='ij')
    flat = interpolate_psi_array(
        np.ascontiguousarray(RR.ravel()), np.ascontiguousarray(ZZ.ravel()),
        src_grid, src_R_min, src_R_max, src_Z_min, src_Z_max, src_nR, src_nZ
    )
    return np.ascontiguousarray(flat.reshape(dst_nR, dst_nZ))


def compute_poloidal_field_grids(eq, B0, R0=1.0, a_minor=0.3, q_target=2.0,
                                 dst_R_min=None, dst_R_max=None, dst_Z_min=None, dst_Z_max=None,
                                 dst_nR=None, dst_nZ=None):
    # Poloidal field from the solved Grad-Shafranov flux function,
    #     B_R = -(1/R) dpsi/dZ ,  B_Z = (1/R) dpsi/dR
    # replacing a linear-in-position stand-in whose near-constant gradient gave no mirror
    # structure to trap particles. The real field's gradient follows the same nested flux
    # surfaces check_confinement_flux uses as the loss boundary, which is what lets
    # trapped ("banana") orbits bounce on the surfaces confining them.
    #
    # Raw psi is in arbitrary units (normalised source term), so the field is rescaled to
    # safety factor q = (a/R0) * (B_tor/B_pol). q_target ~ 2 is the standard stable
    # operating point; q < 1 would be kink-unstable.
    psi_grid = eq.psi_grid
    src_nR, src_nZ = psi_grid.shape
    src_R_min, src_R_max = eq.psi_R_min, eq.psi_R_max
    src_Z_min, src_Z_max = eq.psi_Z_min, eq.psi_Z_max

    R_1d = np.linspace(src_R_min, src_R_max, src_nR)
    dR = R_1d[1] - R_1d[0]
    dZ = (src_Z_max - src_Z_min) / (src_nZ - 1)

    dpsi_dR, dpsi_dZ = np.gradient(psi_grid, dR, dZ)
    RR = R_1d[:, None]
    B_R_raw = -dpsi_dZ / RR
    B_Z_raw = dpsi_dR / RR

    # Rescale so that q at the PLASMA edge equals q_target.
    #
    # The reference magnitude must be sampled ON the r = a_minor flux surface. The
    # previous 90th-percentile-over-the-whole-array reference did not: psi is pinned to 0
    # by Dirichlet conditions on the rectangular solver box (R in [0.5, 1.5], Z in
    # [-0.5, 0.5]), so |grad psi| -- and hence raw |B_pol| -- is steepest in the vacuum
    # corridor between the plasma edge and that box, well outside r = a_minor. The 90th
    # percentile therefore anchored on the SOLVER WALL, measured 1.86x the true edge
    # value, and scaled the whole poloidal field down by that factor: B_pol(a) came out
    # 0.97 T instead of 1.80 T, giving q(a) = 3.71 against the configured
    # Q_SAFETY_TARGET = 2.0.
    #
    # Weak B_pol is directly a fast-ion confinement loss: the poloidal Larmor radius
    # rho_pol = m*v/(q*B_pol) sets the banana-orbit width, so a 1.86x-too-weak B_pol
    # doubles it. For 3.5 MeV alphas that moved the banana width from 0.150 m to 0.278 m
    # against a 0.3 m minor radius -- i.e. from marginally confined to barely confined,
    # and ~19% of alphas were lost to the wall as a result.
    #
    # Sampling the ring reproduces the documented q profile: 1.38 on axis rising
    # monotonically to 2.00 at r = a, with q0 > 1 (kink-stable), as the config intends.
    B_pol_mag = np.ascontiguousarray(np.sqrt(B_R_raw**2 + B_Z_raw**2))
    theta_edge = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    R_edge = np.ascontiguousarray(R0 + a_minor * np.cos(theta_edge))
    Z_edge = np.ascontiguousarray(a_minor * np.sin(theta_edge))
    B_pol_edge = interpolate_psi_array(
        R_edge, Z_edge, B_pol_mag, src_R_min, src_R_max, src_Z_min, src_Z_max, src_nR, src_nZ
    )
    B_pol_ref = float(np.mean(B_pol_edge))
    B_pol_target = (a_minor / R0) * B0 / q_target
    scale = B_pol_target / max(B_pol_ref, 1e-12)

    B_R_scaled = B_R_raw * scale
    B_Z_scaled = B_Z_raw * scale

    # Resample onto the solver grid if one was supplied
    if dst_nR is not None:
        B_R_scaled = resample_grid_to(np.ascontiguousarray(B_R_scaled), src_R_min, src_R_max, src_Z_min, src_Z_max,
                                      src_nR, src_nZ, dst_R_min, dst_R_max, dst_Z_min, dst_Z_max, dst_nR, dst_nZ)
        B_Z_scaled = resample_grid_to(np.ascontiguousarray(B_Z_scaled), src_R_min, src_R_max, src_Z_min, src_Z_max,
                                      src_nR, src_nZ, dst_R_min, dst_R_max, dst_Z_min, dst_Z_max, dst_nR, dst_nZ)

    return np.ascontiguousarray(B_R_scaled), np.ascontiguousarray(B_Z_scaled), scale, B_pol_target


@njit(parallel=True, fastmath=True)
def check_confinement_flux(pos_arr, type_arr, psi_grid, psi_edge, R_min, R_max, Z_min, Z_max, nR, nZ):
    # A particle is lost once it crosses outside the last closed flux surface
    # (psi <= psi_edge), rather than a hard-coded box or circle.
    N = pos_arr.shape[0]
    lost = 0
    for i in prange(N):
        if type_arr[i] != -1:
            r = np.sqrt(pos_arr[i, 0]**2 + pos_arr[i, 1]**2)
            z = pos_arr[i, 2]
            psi_val = interpolate_psi(r, z, psi_grid, R_min, R_max, Z_min, Z_max, nR, nZ)
            if psi_val <= psi_edge:
                type_arr[i] = -1
                lost += 1
    return lost


def check_confinement_torch(pos_tensor, type_tensor, psi_tensor, psi_edge, R_min, R_max, Z_min, Z_max, nR, nZ):
    # GPU counterpart of check_confinement_flux. Eager only -- the torch.sum reduction
    # hits the dynamic-shape Inductor bug.
    x, y, z = pos_tensor[:, 0], pos_tensor[:, 1], pos_tensor[:, 2]
    r = torch.sqrt(x**2 + y**2)

    dR = (R_max - R_min) / (nR - 1)
    dZ = (Z_max - Z_min) / (nZ - 1)
    r_idx = (r - R_min) / dR
    z_idx = (z - Z_min) / dZ
    i = torch.floor(r_idx).long()
    j = torch.floor(z_idx).long()

    valid = (r >= R_min) & (r <= R_max) & (z >= Z_min) & (z <= Z_max) & \
            (i >= 0) & (i < nR - 1) & (j >= 0) & (j < nZ - 1)

    i_c = torch.clamp(i, 0, nR - 2)
    j_c = torch.clamp(j, 0, nZ - 2)
    wR = r_idx - i_c.to(torch.float32)
    wZ = z_idx - j_c.to(torch.float32)

    g00 = psi_tensor[i_c, j_c]
    g10 = psi_tensor[i_c + 1, j_c]
    g01 = psi_tensor[i_c, j_c + 1]
    g11 = psi_tensor[i_c + 1, j_c + 1]
    psi_val = g00 * (1 - wR) * (1 - wZ) + g10 * wR * (1 - wZ) + g01 * (1 - wR) * wZ + g11 * wR * wZ
    psi_val = torch.where(valid, psi_val, torch.zeros_like(psi_val))

    active_mask = type_tensor != -1
    lost_mask = active_mask & (psi_val <= psi_edge)

    lost_count = int(lost_mask.sum().item())
    if lost_count > 0:
        type_tensor[lost_mask] = -1

    return type_tensor, lost_count