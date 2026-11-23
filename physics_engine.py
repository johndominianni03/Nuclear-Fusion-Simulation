import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import torch

# --- CONSTANTS ---
e_charge = 1.602e-19       # Coulombs
m_deuterium = 3.3435e-27   # kg

class ParticlePusher:
    def __init__(self, equilibrium):
        self.eq = equilibrium
        self.m = m_deuterium
        self.q = e_charge

    def initialize_velocities(self, num_particles, T_keV=1.0):
        T_Joules = T_keV * 1e3 * abs(self.q)
        v_thermal = np.sqrt(2.0 * T_Joules / self.m)
        velocities = np.random.normal(0, v_thermal / np.sqrt(2), (num_particles, 3))
        return velocities

    def initialize_velocities_profile(self, T_keV, mass=None):
        # Maxwellian with a per-particle temperature: each Cartesian component is
        # normal with variance T/m. mass: one value or one per particle (default:
        # the deuteron mass).
        T_Joules = np.asarray(T_keV, dtype=np.float64) * 1e3 * abs(self.q)
        m = self.m if mass is None else np.asarray(mass, dtype=np.float64)
        sigma = np.sqrt(T_Joules / m)
        return np.random.normal(0.0, 1.0, (T_Joules.shape[0], 3)) * sigma[:, None]

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

    def compute_alpha_heating_power(self, alpha_energies_kev, dt, cfg, rel_weights=None):
        # Drains energy from fast alphas via Coulomb drag and returns
        # (new alpha energies, total keV given up this step). The total is
        # sum(rel_w * dE) when rel_weights (per-alpha weight / thermal-D weight)
        # is given, i.e. real energy expressed in thermal-D macroparticle units.

        if len(alpha_energies_kev) == 0:
            return alpha_energies_kev, 0.0

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

        # 4. Total transferred to the bulk this step, weighted: sum(rel_w * dE) is the
        # real energy given up, in units of one thermal-D macroparticle's weight, so the
        # caller conserves real energy by depositing it with the bulk weights.
        if rel_weights is None:
            total_energy_deposited_kev = float(np.sum(deposited_kev))
        else:
            total_energy_deposited_kev = float(np.sum(rel_weights * deposited_kev))

        return new_alpha_energies, total_energy_deposited_kev

# --- CLASSLESS NUMBA KERNELS ---

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
def gather_electric_field_scalar(px, py, pz, E_R_grid, E_Z_grid,
                                 R_min, R_max, Z_min, Z_max, nR, nZ,
                                 cos_phi, sin_phi):
    """Scalar, allocation-free bilinear gather of the electric field.

    Bilinear gather plus the cylindrical -> Cartesian rotation, returned as a
    3-tuple of floats rather than a freshly allocated 3-element array. Called
    once per particle per step from the prange loop in
    kernels.vectorized_gather_and_B_into, where an array return was the
    dominant cost of the stage and kept numba from vectorizing the loop.

    cos_phi / sin_phi are passed in rather than derived from an arctan2 here:
    the caller already needs phi to build the toroidal and poloidal B, so the
    sqrt and arctan2 are computed once per particle and shared.

    Arithmetic is float32 throughout, matching the caller's E_arr/B_arr, so
    storing a result costs no per-element downcast.
    """
    r_p = np.float32(np.sqrt(px * px + py * py))
    z_p = np.float32(pz)

    zero = np.float32(0.0)
    if r_p < R_min or r_p > R_max or z_p < Z_min or z_p > Z_max:
        return zero, zero, zero

    dR = np.float32((R_max - R_min) / (nR - 1))
    dZ = np.float32((Z_max - Z_min) / (nZ - 1))

    r_idx = (r_p - np.float32(R_min)) / dR
    z_idx = (z_p - np.float32(Z_min)) / dZ

    i = int(np.floor(r_idx))
    j = int(np.floor(z_idx))

    if i >= nR - 1 or j >= nZ - 1 or i < 0 or j < 0:
        return zero, zero, zero

    wR = r_idx - np.float32(i)
    wZ = z_idx - np.float32(j)
    one = np.float32(1.0)

    w00 = (one - wR) * (one - wZ)
    w10 = wR * (one - wZ)
    w01 = (one - wR) * wZ
    w11 = wR * wZ

    er = (np.float32(E_R_grid[i, j]) * w00 +
          np.float32(E_R_grid[i + 1, j]) * w10 +
          np.float32(E_R_grid[i, j + 1]) * w01 +
          np.float32(E_R_grid[i + 1, j + 1]) * w11)

    ez = (np.float32(E_Z_grid[i, j]) * w00 +
          np.float32(E_Z_grid[i + 1, j]) * w10 +
          np.float32(E_Z_grid[i, j + 1]) * w01 +
          np.float32(E_Z_grid[i + 1, j + 1]) * w11)

    return er * cos_phi, er * sin_phi, ez


# =======================================================
# PRESCRIBED TEARING MODE
# =======================================================

def tearing_amplitude_over_B0(cfg, t):
    """Amplitude of the prescribed m/n tearing-mode ripple at time t, relative to
    B0: b_perturb_initial * exp(gamma_growth * t) / B0. This is the amplitude
    kernels._field_at adds to every particle's B."""
    return float(cfg.b_perturb_initial * np.exp(cfg.gamma_growth * t) / cfg.B0)

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

    # Dimensionless parameter xi (Gamow constant B_G = 34.3827 keV^1/2)
    B_G_sq = 1182.17005929
    xi = (B_G_sq / (4.0 * theta))**(1.0/3.0)

    # Bosch-Hale fit, cm^3/s:  C1 theta sqrt(xi / (m_r c^2 T^3)) exp(-3 xi), with the
    # D-T reduced-mass energy m_r c^2 = 1,124,656 keV (Bosch & Hale 1992, Table VII).
    # It was missing, which put every <sigma v> sqrt(1124656) = 1060x too high.
    MRC2_KEV = 1124656.0
    sigmav_cm3_s = C1 * theta * np.sqrt(xi / (MRC2_KEV * T_i_keV**3)) * np.exp(-3.0 * xi)

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
# floor at ANY particle count; fusing collapses that to one or two. The benchmark
# call site uses fixed shapes, so static compile applies and warms up
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
# radius. Deuterons are fine at the global step (Larmor radius ~5e-4 m, ~3e-4 m of
# travel per step), so only the alphas are sub-stepped, and the field is re-gathered
# at every sub-step. The CPU kernel is kernels.boris_push_substeps_regather (it needs
# the scalar field helper that lives there); the torch twin is below, after the torch
# gather it calls.
#
# The previous kernels held E and B at the start-of-step position across all
# sub-steps. That is first-order and not volume-preserving, and in the sheared
# Grad-Shafranov field it drove every alpha's pitch to xi = -1 within a few
# microseconds. See tools/alpha_pitch_check.py.
# =====================================================================


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
# =====================================================================

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


def boris_push_substeps_regather_torch(pos, vel, q, m, dt, n_sub,
                                       E_R_grid_t, E_Z_grid_t, B_R_pol_t, B_Z_pol_t,
                                       R_min, R_max, Z_min, Z_max, nR, nZ, B0, R0, t,
                                       b_perturb, m_mode, n_mode, gamma):
    """Torch twin of kernels.boris_push_substeps_regather.

    n_sub Boris sub-steps of dt/n_sub, each with E and B gathered at that
    sub-step's start position and time level. Returns NEW (pos, vel) tensors;
    the caller copies them back into the pool views. Positions are stored in
    float32 between sub-steps (the GPU pools are float32), unlike the CPU
    kernel's float64 accumulators.
    """
    n_sub = max(int(n_sub), 1)
    dt_sub = dt / float(n_sub)
    for k in range(n_sub):
        E, B = vectorized_gather_and_B_torch(
            pos, E_R_grid_t, E_Z_grid_t, B_R_pol_t, B_Z_pol_t,
            R_min, R_max, Z_min, Z_max, nR, nZ, B0, R0, t + k * dt_sub,
            b_perturb, m_mode, n_mode, gamma)
        pos, vel = _vectorized_boris_push_metal_impl(pos, vel, q, m, B, E, dt_sub)
    return pos, vel


# NOTE: the old Cartesian/circular check_confinement_torch was deleted from here -- it
# was shadowed by, and is superseded by, the psi-surface version further down this file.


def apply_vectorized_collisions_torch(vel_tensor, type_tensor, nu_c, dt):
    # Eager only -- the torch.any reduction hits the dynamic-shape Inductor bug.
    device = vel_tensor.device
    N = vel_tensor.shape[0]

    eligible = (type_tensor == 0) | (type_tensor == 1) | (type_tensor == 3)
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


def compute_alpha_heating_power_torch(alpha_energies_kev, dt, cfg, rel_weights=None):
    # Eager only -- the torch.sum reduction hits the dynamic-shape Inductor bug.
    if alpha_energies_kev.numel() == 0:
        return alpha_energies_kev, 0.0

    # Same float32 cancellation as the numpy version, but MPS has no float64 so the
    # promote-to-double remedy isn't available. Compute the deposit directly instead:
    # E * 4e-9 is representable in float32 (only the `E - E*4e-9` subtraction collapses),
    # so summing the drain term keeps full precision on-device.
    E = alpha_energies_kev
    energy_loss_fraction = min(2.0 * cfg.NU_S * dt, 0.99)

    headroom = torch.clamp(E - cfg.THERMALIZATION_ENERGY_KEV, min=0.0)
    deposited_kev = torch.minimum(E * energy_loss_fraction, headroom)

    new_alpha_energies = E - deposited_kev

    # The reduction is over the precise per-particle drain, so no cancellation here.
    # Weighted exactly as the numpy twin: sum(rel_w * dE).
    if rel_weights is None:
        total_energy_deposited_kev = float(torch.sum(deposited_kev).item())
    else:
        total_energy_deposited_kev = float(torch.sum(rel_weights * deposited_kev).item())

    return new_alpha_energies, total_energy_deposited_kev


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
    # Bilinear lookup on the (R, Z)-ordered psi grid, as in gather_electric_field_scalar.
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
    # the a = 0.3 circle, and the true volume is ~3.519 m^3 against that formula's 1.78.
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


# =====================================================================
# FLUX-SURFACE PROFILES
#
# Moments binned by normalised poloidal flux
#     psi_N = (psi_axis - psi) / (psi_axis - psi_edge)
# 0 on the magnetic axis, 1 on the last closed flux surface (the same psi_edge
# check_confinement_flux uses as the loss boundary). Shell volumes come from the
# psi grid itself, so n = weight * count / V_shell is a true number density.
# =====================================================================

FLUX_PROFILE_BINS = 25
FLUX_VOLUME_MESH = 400      # fine R-Z mesh per side for the shell volumes


def compute_psi_normalised(R, Z, eq):
    """psi_N at (R, Z), by the same bilinear lookup the confinement test uses."""
    R = np.ascontiguousarray(R, dtype=np.float64)
    Z = np.ascontiguousarray(Z, dtype=np.float64)
    psi = interpolate_psi_array(R.ravel(), Z.ravel(), eq.psi_grid,
                                eq.psi_R_min, eq.psi_R_max, eq.psi_Z_min, eq.psi_Z_max,
                                eq.psi_nR, eq.psi_nZ).reshape(R.shape)
    return (eq.psi_axis - psi) / (eq.psi_axis - eq.psi_edge)


def compute_flux_shell_volumes(eq, n_bins=FLUX_PROFILE_BINS, n_mesh=FLUX_VOLUME_MESH):
    """Volume of each psi_N shell [k/n_bins, (k+1)/n_bins), k = 0..n_bins-1.

    psi is sampled at the cell centres of an n_mesh x n_mesh mesh over the
    equilibrium box and each cell contributes its revolved volume 2*pi*R*dR*dZ to
    the shell its psi_N falls in. Cells with psi_N > 1 lie outside the plasma and
    are dropped. Cached on `eq` per (n_bins, n_mesh): psi never changes in a run.
    """
    cache = getattr(eq, "_flux_shell_volume_cache", None)
    if cache is None:
        cache = {}
        eq._flux_shell_volume_cache = cache
    key = (int(n_bins), int(n_mesh))
    if key in cache:
        return cache[key].copy()

    dR = (eq.psi_R_max - eq.psi_R_min) / n_mesh
    dZ = (eq.psi_Z_max - eq.psi_Z_min) / n_mesh
    Rc = eq.psi_R_min + (np.arange(n_mesh) + 0.5) * dR
    Zc = eq.psi_Z_min + (np.arange(n_mesh) + 0.5) * dZ
    RR, ZZ = np.meshgrid(Rc, Zc, indexing="ij")
    psi_N = compute_psi_normalised(RR, ZZ, eq)
    inside = psi_N <= 1.0
    idx = np.minimum((psi_N[inside] * n_bins).astype(np.int64), n_bins - 1)
    dV = 2.0 * np.pi * RR[inside] * dR * dZ
    volumes = np.bincount(idx, weights=dV, minlength=n_bins)
    cache[key] = volumes
    return volumes.copy()


def compute_flux_profiles(pos, vel, types, eq, masses, weight=1.0,
                          n_bins=FLUX_PROFILE_BINS, n_mesh=FLUX_VOLUME_MESH, B0=None,
                          particle_weight=None):
    """Per-species moments on psi_N shells.

    pos, vel : (N, 3) Cartesian positions [m] and velocities [m/s]
    types    : (N,) species, 0 thermal D, 1 beam D, 2 alpha, 3 thermal T (-1 lost; ignored)
    masses   : per-species masses in kg, indexed by type code (species_masses)
    weight   : real ions per macroparticle, either one number for every species
               or a per-species sequence indexed by type code (weight_lut)
    B0       : field for beta [T]; beta is omitted when None
    particle_weight : optional (N,) real ions per macroparticle, one per row; it
               replaces `weight` for every species, and u and T become
               weight-weighted means (needed for the alphas, whose macroparticles
               each keep their birth weight)

    Velocities are moments in the local cylindrical basis (R, phi, Z): the mean
    flow u of a shell is (u_R, u_phi, u_Z), so a toroidal rotation is a flow and
    not heat. A Cartesian mean over a whole toroidal shell would average a
    rotation to zero and count it as temperature.

    Per species s and shell k:
        n  = weight_s * count / V_k                     [m^-3]
        u  = <v_cyl>                                    [m/s]
        T  = m_s <|v_cyl - u|^2> / 3                    [J]   (also T_keV)
        p  = n T                                        [Pa]
    Shells with fewer than 2 particles of a species have T = NaN and p = 0.
    beta_profile = 2 mu0 sum_s p_s / B0^2 per shell, and beta_avg is the same with
    the volume-averaged pressure sum_k p_k V_k / sum_k V_k.
    """
    pos = np.asarray(pos, dtype=np.float64)
    vel = np.asarray(vel, dtype=np.float64)
    types = np.asarray(types)
    V = compute_flux_shell_volumes(eq, n_bins, n_mesh)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    R = np.hypot(pos[:, 0], pos[:, 1])
    psi_N = compute_psi_normalised(R, pos[:, 2], eq)
    inside = (types >= 0) & (psi_N <= 1.0)
    idx = np.minimum((np.clip(psi_N, 0.0, None) * n_bins).astype(np.int64), n_bins - 1)

    safe_R = np.where(R > 0.0, R, 1.0)
    cphi, sphi = pos[:, 0] / safe_R, pos[:, 1] / safe_R
    v_cyl = np.stack([vel[:, 0] * cphi + vel[:, 1] * sphi,
                      -vel[:, 0] * sphi + vel[:, 1] * cphi,
                      vel[:, 2]], axis=1)

    n_species = len(masses)
    w_s = np.broadcast_to(np.asarray(weight, dtype=np.float64), (n_species,))
    count = np.zeros((n_species, n_bins), dtype=np.int64)
    n = np.zeros((n_species, n_bins))
    u = np.zeros((n_species, n_bins, 3))
    T = np.full((n_species, n_bins), np.nan)
    p = np.zeros((n_species, n_bins))
    safe_V = np.where(V > 0.0, V, np.inf)
    pw = None if particle_weight is None else np.asarray(particle_weight, dtype=np.float64)
    for s in range(n_species):
        sel = inside & (types == s)
        k = idx[sel]
        vs = v_cyl[sel]
        c = np.bincount(k, minlength=n_bins)
        count[s] = c
        if pw is None:
            n[s] = w_s[s] * c / safe_V
            norm = np.where(c > 0, c, 1)
            ws = None
        else:
            ws = pw[sel]
            wsum = np.bincount(k, weights=ws, minlength=n_bins)
            n[s] = wsum / safe_V
            norm = np.where(wsum > 0, wsum, 1.0)
        for d in range(3):
            u[s, :, d] = np.bincount(k, weights=vs[:, d] if ws is None else ws * vs[:, d],
                                     minlength=n_bins) / norm
        dv2 = np.sum((vs - u[s, k]) ** 2, axis=1)
        mean_dv2 = np.bincount(k, weights=dv2 if ws is None else ws * dv2, minlength=n_bins) / norm
        ok = c >= 2
        T[s, ok] = masses[s] * mean_dv2[ok] / 3.0
        p[s, ok] = n[s, ok] * T[s, ok]

    out = {
        "psiN_edges": edges,
        "psiN_centers": centers,
        "rho_centers": np.sqrt(centers),
        "shell_volume": V,
        "count": count,
        "n": n,
        "u": u,
        "T": T,
        "T_keV": T / (e_charge * 1.0e3),
        "p": p,
        "weight": tuple(float(x) for x in w_s) if pw is None else "per-particle",
    }
    if B0 is not None:
        mu0 = 4.0e-7 * np.pi
        p_tot = p.sum(axis=0)
        out["beta_profile"] = 2.0 * mu0 * p_tot / B0**2
        out["beta_avg"] = 2.0 * mu0 * float(np.sum(p_tot * V) / np.sum(V)) / B0**2
    return out


# =====================================================================
# QUASI-NEUTRAL RADIAL ELECTRIC FIELD FROM ION FORCE BALANCE
#
# Radial force balance for the thermal ions with no rotation,
#     e n_i E = grad p_i,
# is a flux function relation. With psi_N as the flux label,
#     dPhi/dpsi_N = -(1 / (e n_i)) dp_i/dpsi_N,     Phi(edge of support) = 0,
#     E = -grad Phi = -(dPhi/dpsi_N) grad psi_N.
# n_i and p_i come from compute_flux_profiles (the weight cancels in p/n).
# =====================================================================

FIELD_MIN_COUNT = 100       # bins with fewer thermals (>10% Poisson error) carry no field


def compute_grad_psiN_grids(eq, dst_R_min, dst_R_max, dst_Z_min, dst_Z_max, dst_nR, dst_nZ):
    """(psi_N, d psi_N/dR, d psi_N/dZ) on the destination grid nodes.

    The gradient is taken by central differences on the equilibrium's own psi
    grid and resampled bilinearly, exactly as compute_poloidal_field_grids builds
    B_pol, so E and B_pol share one discretisation of grad psi.
    """
    nR_s, nZ_s = eq.psi_grid.shape
    dR = (eq.psi_R_max - eq.psi_R_min) / (nR_s - 1)
    dZ = (eq.psi_Z_max - eq.psi_Z_min) / (nZ_s - 1)
    dpsi_dR, dpsi_dZ = np.gradient(eq.psi_grid, dR, dZ)
    scale = -1.0 / (eq.psi_axis - eq.psi_edge)
    src = (eq.psi_R_min, eq.psi_R_max, eq.psi_Z_min, eq.psi_Z_max, nR_s, nZ_s)
    dst = (dst_R_min, dst_R_max, dst_Z_min, dst_Z_max, dst_nR, dst_nZ)
    gR = scale * resample_grid_to(np.ascontiguousarray(dpsi_dR), *src, *dst)
    gZ = scale * resample_grid_to(np.ascontiguousarray(dpsi_dZ), *src, *dst)
    Rd = np.linspace(dst_R_min, dst_R_max, dst_nR)
    Zd = np.linspace(dst_Z_min, dst_Z_max, dst_nZ)
    RR, ZZ = np.meshgrid(Rd, Zd, indexing="ij")
    psiN = compute_psi_normalised(RR, ZZ, eq)
    return psiN, gR, gZ


def smooth_profile(x, y, sigma):
    """Cubic smoothing spline through (x, y) with 1-sigma errors `sigma`.

    The smoothing factor is the number of points, the standard choice that makes
    the weighted residual chi^2 equal to its expectation: the curve follows the
    data down to its counting noise and no further. Returns the spline.
    """
    from scipy.interpolate import UnivariateSpline
    return UnivariateSpline(x, y, w=1.0 / sigma, k=3, s=float(len(x)))


def compute_force_balance_potential(prof, species=0, min_count=FIELD_MIN_COUNT, n_fine=2001):
    """Phi(psi_N) and dPhi/dpsi_N from ion force balance, on a fine psi_N mesh.

    The support is the contiguous run of shells from the axis outward holding at
    least `min_count` ions of `species`; outside it there is no plasma to balance
    and dPhi/dpsi_N = 0. n and p are smoothed separately (smooth_profile, errors
    from counting statistics: sigma_n/n = 1/sqrt(c), sigma_p/p = sqrt(5/(3c)) for a
    Maxwellian) and dp/dpsi_N is the spline's analytic derivative. The fine mesh
    runs from 0 to the last supported shell centre; Phi = 0 there and beyond.

    `species` is one type code or several; several are summed (n, p and counts),
    which is the force balance of a multi-species ion fluid of charge e:
    e (n_D + n_T) E = grad (p_D + p_T).

    Returns dict(x, Phi, dPhi_dx, x_end) with Phi in volts.
    """
    sp = np.atleast_1d(species)
    c = prof["count"][sp].sum(axis=0)
    centers = prof["psiN_centers"]
    K = 0
    while K < len(c) and c[K] >= min_count:
        K += 1
    x = np.linspace(0.0, 1.0, n_fine)
    if K < 4:                                      # a cubic needs 4 points
        return {"x": x, "Phi": np.zeros_like(x), "dPhi_dx": np.zeros_like(x), "x_end": 0.0}
    xs = centers[:K]
    n = prof["n"][sp, :K].sum(axis=0)
    p = prof["p"][sp, :K].sum(axis=0)
    ck = c[:K].astype(np.float64)
    n_spl = smooth_profile(xs, n, n / np.sqrt(ck))
    p_spl = smooth_profile(xs, p, p * np.sqrt(5.0 / (3.0 * ck)))
    x_end = float(xs[-1])
    inside = x <= x_end
    g = np.zeros_like(x)
    n_fit = n_spl(x[inside])
    g[inside] = -p_spl.derivative()(x[inside]) / (e_charge * n_fit)
    # Phi(x) = -int_x^x_end g dx', by trapezoid on the fine mesh
    seg = 0.5 * (g[1:] + g[:-1]) * np.diff(x)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    Phi = np.where(inside, cum - cum[inside][-1], 0.0)
    return {"x": x, "Phi": Phi, "dPhi_dx": g, "x_end": x_end}


def compute_force_balance_field(prof, eq, R_min, R_max, Z_min, Z_max, nR, nZ,
                                species=0, min_count=FIELD_MIN_COUNT):
    """phi, E_R, E_Z on the (nR, nZ) field grid from ion force balance.

    E = -(dPhi/dpsi_N) grad psi_N at each node, with dPhi/dpsi_N interpolated in
    psi_N from compute_force_balance_potential. Zero outside the last closed flux
    surface (and outside the ion support, where dPhi/dpsi_N = 0 already).
    Returns (phi_grid, E_R_grid, E_Z_grid, potential_dict).
    """
    pot = compute_force_balance_potential(prof, species, min_count)
    psiN, gR, gZ = compute_grad_psiN_grids(eq, R_min, R_max, Z_min, Z_max, nR, nZ)
    inside = psiN <= 1.0
    dPhi = np.where(inside, np.interp(psiN, pot["x"], pot["dPhi_dx"]), 0.0)
    phi = np.where(inside, np.interp(psiN, pot["x"], pot["Phi"]), 0.0)
    E_R = np.ascontiguousarray(-dPhi * gR)
    E_Z = np.ascontiguousarray(-dPhi * gZ)
    return np.ascontiguousarray(phi), E_R, E_Z, pot


def map_flux_profile_to_grid(values, centers, eq, R_min, R_max, Z_min, Z_max, nR, nZ):
    """A shell profile f(psi_N) on the (nR, nZ) grid nodes: linear in psi_N between
    shell centres, held at the end values inside the first/last half shell, and 0
    outside the last closed flux surface."""
    Rd = np.linspace(R_min, R_max, nR)
    Zd = np.linspace(Z_min, Z_max, nZ)
    RR, ZZ = np.meshgrid(Rd, Zd, indexing="ij")
    psiN = compute_psi_normalised(RR, ZZ, eq)
    return np.where(psiN <= 1.0, np.interp(psiN, centers, values), 0.0)


def thermal_ion_temperature_keV(prof):
    """Thermal-ion temperature per shell: the density-weighted mean of the thermal
    D and thermal T temperatures,
        T_i = (n_D T_D + n_T T_T) / (n_D + n_T) = (p_D + p_T) / (n_D + n_T),
    i.e. the temperature of the combined thermal-ion pressure. A species whose
    shell temperature is undefined (fewer than 2 particles) is left out of that
    shell's mean; 0 where neither is defined."""
    n_D, n_T = prof["n"][0], prof["n"][3]
    T_D, T_T = prof["T_keV"][0], prof["T_keV"][3]
    okD, okT = np.isfinite(T_D), np.isfinite(T_T)
    num = np.where(okD, n_D * np.nan_to_num(T_D), 0.0) + np.where(okT, n_T * np.nan_to_num(T_T), 0.0)
    den = np.where(okD, n_D, 0.0) + np.where(okT, n_T, 0.0)
    return np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), 0.0)


def compute_thermal_fusion_power(prof, cfg):
    """Thermal D-T fusion power density per psi_N shell and its volume integral.

    Per shell k, from the weighted thermal-D and thermal-T profiles,
        P_k = n_D,k n_T,k <sigma v>(T_i,k) E_fusion     [W/m^3]
        P_total = sum_k P_k V_k                         [W]
    with the Bosch-Hale reactivity at the thermal-ion temperature T_i (the
    density-weighted mean of T_D and T_T, thermal_ion_temperature_keV). Beam ions
    and alphas are excluded: this is the thermonuclear power of the thermal
    population only. Shells without a temperature contribute nothing.
    Returns (P_k, P_total).
    """
    n_D, n_T = prof["n"][0], prof["n"][3]
    T = thermal_ion_temperature_keV(prof)
    P = compute_volumetric_fusion_power(n_D[None, :], n_T[None, :], T[None, :], cfg)[0]
    return P, float(np.sum(P * prof["shell_volume"]))


def compute_alpha_birth_rates(prof, cfg):
    """Thermal D-T reactions per second in each psi_N shell,
        rate_k = n_D,k n_T,k <sigma v>(T_i,k) V_k = P_k V_k / E_fusion,
    the same shell sum compute_thermal_fusion_power integrates (one alpha per
    reaction). Returns an array over the shells."""
    P, _ = compute_thermal_fusion_power(prof, cfg)
    return P * prof["shell_volume"] / cfg.E_FUSION_JOULES


class AlphaBirthSampler:
    """Birth positions for fusion alphas: a psi_N shell is drawn with probability
    proportional to its reaction rate, then a position inside that shell by
    rejection sampling -- uniform (R, Z) proposals in the shell's bounding box,
    accepted when psi_N falls in the shell and with probability R / R_box_max
    (the revolved volume element is 2 pi R dR dZ) -- and a uniform toroidal angle.
    The reaction density is taken as uniform within a shell, the resolution at
    which the rates are known.

    Bounding boxes come from the same n_mesh mesh as the shell volumes, padded by
    two cells, so every point of a shell lies inside its box.
    """

    def __init__(self, eq, n_bins=FLUX_PROFILE_BINS, n_mesh=FLUX_VOLUME_MESH):
        self.eq = eq
        self.n_bins = n_bins
        dR = (eq.psi_R_max - eq.psi_R_min) / n_mesh
        dZ = (eq.psi_Z_max - eq.psi_Z_min) / n_mesh
        Rc = eq.psi_R_min + (np.arange(n_mesh) + 0.5) * dR
        Zc = eq.psi_Z_min + (np.arange(n_mesh) + 0.5) * dZ
        RR, ZZ = np.meshgrid(Rc, Zc, indexing="ij")
        x = compute_psi_normalised(RR, ZZ, eq)
        k = np.where(x <= 1.0, np.minimum((np.clip(x, 0.0, None) * n_bins).astype(np.int64), n_bins - 1), -1)
        self.box = np.zeros((n_bins, 4))
        for s in range(n_bins):
            m = k == s
            self.box[s] = (max(RR[m].min() - 2 * dR, eq.psi_R_min), min(RR[m].max() + 2 * dR, eq.psi_R_max),
                           max(ZZ[m].min() - 2 * dZ, eq.psi_Z_min), min(ZZ[m].max() + 2 * dZ, eq.psi_Z_max))

    def draw_shells(self, rates, n):
        """n shell indices with probability rate_k / sum(rate)."""
        p = np.asarray(rates, dtype=np.float64)
        cdf = np.cumsum(p / p.sum())
        return np.minimum(np.searchsorted(cdf, np.random.random(n), side="right"), self.n_bins - 1)

    def positions_in_shells(self, shells):
        """(R, Z) for each requested shell index, by rejection sampling."""
        n = len(shells)
        R = np.empty(n)
        Z = np.empty(n)
        for i, s in enumerate(shells):
            R0, R1, Z0, Z1 = self.box[s]
            lo, hi = s / self.n_bins, (s + 1) / self.n_bins
            while True:
                Rt = np.random.uniform(R0, R1, 64)
                Zt = np.random.uniform(Z0, Z1, 64)
                x = compute_psi_normalised(Rt, Zt, self.eq)
                ok = (x >= lo) & (x < hi) if s < self.n_bins - 1 else (x >= lo) & (x <= 1.0)
                ok &= np.random.uniform(0.0, 1.0, 64) < Rt / R1
                hit = np.nonzero(ok)[0]
                if hit.size:
                    R[i], Z[i] = Rt[hit[0]], Zt[hit[0]]
                    break
        return R, Z

    def sample(self, rates, n, engine, cfg):
        """n isotropic 3.5 MeV alphas born in proportion to the shell rates.
        Returns float32 Cartesian (pos, vel) and the shell indices drawn."""
        shells = self.draw_shells(rates, n)
        R, Z = self.positions_in_shells(shells)
        phi = np.random.uniform(0.0, 2.0 * np.pi, n)
        _, _, vR, vphi, vZ = engine.spawn_alpha_particles(n, R, Z, cfg.ALPHA_ENERGY_JOULES, cfg.MASS_ALPHA)
        pos = np.empty((n, 3), dtype=np.float32)
        vel = np.empty((n, 3), dtype=np.float32)
        pos[:, 0] = R * np.cos(phi)
        pos[:, 1] = R * np.sin(phi)
        pos[:, 2] = Z
        vel[:, 0] = vR * np.cos(phi) - vphi * np.sin(phi)
        vel[:, 1] = vR * np.sin(phi) + vphi * np.cos(phi)
        vel[:, 2] = vZ
        return pos, vel, shells


# Species type codes: 0 thermal D, 1 beam D, 2 alpha, 3 thermal T (-1 lost).
THERMAL_SPECIES = (0, 3)


def species_masses(cfg):
    """Per-species masses indexed by type code: (thermal D, beam D, alpha, thermal T)."""
    return (cfg.m_deuterium, cfg.m_deuterium, cfg.MASS_ALPHA, cfg.m_tritium)


def weight_lut(cfg):
    """cfg.WEIGHT (real ions per macroparticle, keyed by type code) as a float64
    array indexed by type code, for vectorised per-particle lookup. Index it with
    live types only (>= 0)."""
    return np.array([cfg.WEIGHT[s] for s in range(len(species_masses(cfg)))], dtype=np.float64)


def relative_weight_lut(cfg):
    """weight_lut / the thermal-D weight. The reactor loops weight energies and
    counts with these ratios, so a weighted sum reads in units of one thermal-D
    macroparticle; with equal weights every ratio is exactly 1.0 and the weighted
    sums are bit-identical to the unweighted ones."""
    lut = weight_lut(cfg)
    return lut / lut[0]


# Column order of the reactor loops' per-species stored-energy history
# (stored_energy_J): (type code, label).
STORED_ENERGY_SPECIES = ((0, "Thermal D"), (3, "Thermal T"), (1, "Beam D"), (2, "Alpha"))


def species_energy_and_count(vel, types, mass, w_lut, w_row=None):
    """Real kinetic energy [J] and real ion count per species for one pool.

    vel (N, 3) and types (N,) are a pool's live rows (types >= 0). Each row
    carries w_lut[type] real ions, or w_row[i] when per-particle weights are
    given. Returns two float64 arrays indexed by type code (length len(w_lut)):
    sum_i w_i (1/2) m |v_i|^2 and sum_i w_i. One bincount per quantity, no masks."""
    n_species = len(w_lut)
    v = np.asarray(vel, dtype=np.float64)
    e = 0.5 * mass * np.sum(v * v, axis=1)
    t = np.asarray(types, dtype=np.int64)
    if w_row is None:
        E = np.bincount(t, weights=e, minlength=n_species) * w_lut
        N = np.bincount(t, minlength=n_species) * w_lut
    else:
        E = np.bincount(t, weights=w_row * e, minlength=n_species)
        N = np.bincount(t, weights=w_row, minlength=n_species).astype(np.float64)
    return E, N


def thermal_temperature_keV(E_J, N):
    """Kinetic temperature of the thermal ions (D + T only, no beam ions or
    alphas): T = (2/3) <E> per real thermal ion, from species_energy_and_count
    totals indexed by type code."""
    n_th = N[0] + N[3]
    if n_th <= 0.0:
        return 0.0
    return (2.0 / 3.0) * (E_J[0] + E_J[3]) / n_th / (e_charge * 1.0e3)


def weighted_kinetic_energy_kev(vels64, rel_w, mass):
    """sum_i rel_w_i * (1/2) m |v_i|^2, in keV: the real kinetic energy of a set of
    macroparticles in thermal-D macroparticle units."""
    return float(np.sum(rel_w * (0.5 * mass * np.sum(vels64**2, axis=1)))) / 1.602e-16


def drain_alpha_energy(engine, cfg, alpha_vels, rel_w, dt):
    """Slow the alphas by the fixed drag (compute_alpha_heating_power) and return
    the real energy given up, sum(rel_w * dE) in keV (thermal-D macroparticle units).
    alpha_vels is float64 and is rescaled in place (speed only)."""
    v_mags = np.linalg.norm(alpha_vels, axis=1)
    alpha_energies_kev = (0.5 * cfg.MASS_ALPHA * (v_mags**2)) / 1.602e-16
    new_energies_kev, deposited_kev = engine.compute_alpha_heating_power(
        alpha_energies_kev, dt, cfg, rel_weights=rel_w)
    new_v_mags = np.sqrt(2.0 * (new_energies_kev * 1.602e-16) / cfg.MASS_ALPHA)
    scale_factors = new_v_mags / np.where(v_mags == 0, 1e-10, v_mags)
    alpha_vels *= scale_factors[:, np.newaxis]
    return deposited_kev


def bulk_boost_factor(deposited_kev, bulk_energy_kev):
    """Common speed factor that adds deposited_kev to a population holding
    bulk_energy_kev (both weighted, same units): energy goes as v^2, so every
    velocity scales by sqrt(1 + dE/E) and the real energy received,
    (boost^2 - 1) * E, equals the real energy given up whatever the weights."""
    return np.sqrt(1.0 + deposited_kev / bulk_energy_kev)


def update_force_balance_field(cfg, eq, pos, vel, types):
    """The reactor loops' field update: thermal-ion profiles of the given
    particles -> (phi_grid, E_R_grid, E_Z_grid, potential_dict). Host arrays."""
    prof = compute_flux_profiles(pos, vel, types, eq, species_masses(cfg), weight=weight_lut(cfg))
    # Both thermal species: e (n_D + n_T) E = grad (p_D + p_T).
    phi, E_R, E_Z, pot = compute_force_balance_field(prof, eq, cfg.R_min, cfg.R_max, cfg.Z_min,
                                                     cfg.Z_max, cfg.nR, cfg.nZ, species=THERMAL_SPECIES)
    pot["profile"] = prof        # the thermal profiles, reused for the alpha birth rates
    return phi, E_R, E_Z, pot


def _deposit_weighted(cfg, pos, types, per_particle, weights=None):
    pos = np.asarray(pos)
    types = np.asarray(types)
    live = types >= 0
    R = np.ascontiguousarray(np.hypot(pos[live, 0], pos[live, 1]), dtype=np.float64)
    Z = np.ascontiguousarray(pos[live, 2], dtype=np.float64)
    w = (weight_lut(cfg)[types[live]] if weights is None
         else np.asarray(weights, dtype=np.float64)[live])
    q = np.ascontiguousarray(per_particle(types[live]) * w, dtype=np.float64)
    return compute_cic_charge_density(R, Z, q, cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max,
                                      cfg.nR, cfg.nZ, False)


def deposit_ion_charge_density(cfg, pos, types, weights=None):
    """Weighted CIC ion charge density (C/m^3) of every live particle.

    Each macroparticle carries cfg.WEIGHT[type] ions -- or weights[i] when a
    per-row weight array is given (alphas keep their birth weight) -- of its
    species' charge: e for deuterons and tritons, CHARGE_ALPHA for alphas.
    """
    return _deposit_weighted(cfg, pos, types,
                             lambda t: np.where(t == 2, cfg.CHARGE_ALPHA, cfg.e_charge), weights)


def deposit_ion_number_density(cfg, pos, types, weights=None):
    """Weighted CIC ion number density (m^-3) of every live particle, all species."""
    return _deposit_weighted(cfg, pos, types, lambda t: np.ones(t.shape[0]), weights)


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