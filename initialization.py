import numpy as np
import torch
from mhd_equilibrium import MHDEquilibrium
from physics_engine import (
    ParticlePusher,
    compute_flux_thresholds,
    interpolate_psi_array,
    compute_flux_boundary_contour,   # psi_edge wall polygon, shared by physics and plots
    compute_plasma_volume,           # revolved volume of that surface
)

def _attach_flux_surfaces(eq, cfg=None, R0=1.0, a_minor=0.3):
    # Flux-surface (psi) boundaries, calibrated against the legacy circular region
    # (R0=1.0, a_minor=0.3) so the confined volume is comparably sized but follows the
    # real magnetic topology. Stashed on `eq` so downstream code reads them via eq.psi_*
    # without changing any caller's positionally-unpacked return signature.
    #
    # Shared by the reactor and the oscillation test: both now decide particle loss with
    # check_confinement_flux against this one surface, so both need it attached.
    psi_axis, psi_edge, psi_core, psi_grid, psi_R_min, psi_R_max, psi_Z_min, psi_Z_max, psi_nR, psi_nZ = \
        compute_flux_thresholds(eq, R0=R0, a_minor=a_minor)
    eq.psi_axis, eq.psi_edge, eq.psi_core = psi_axis, psi_edge, psi_core
    eq.psi_grid = psi_grid
    eq.psi_R_min, eq.psi_R_max = psi_R_min, psi_R_max
    eq.psi_Z_min, eq.psi_Z_max = psi_Z_min, psi_Z_max
    eq.psi_nR, eq.psi_nZ = psi_nR, psi_nZ

    # The wall the diagnostics draw and the volume the Lawson density divides by are both
    # derived from this same contour, so plot, physics and reactor metrics cannot drift
    # apart the way the r = 0.3 circle did.
    eq.wall_R, eq.wall_Z = compute_flux_boundary_contour(
        psi_grid, psi_edge, psi_R_min, psi_R_max, psi_Z_min, psi_Z_max, psi_nR, psi_nZ
    )
    eq.plasma_volume_m3 = compute_plasma_volume(eq.wall_R, eq.wall_Z)
    if cfg is not None and eq.plasma_volume_m3 > 0.0:
        cfg.PLASMA_VOLUME_M3 = eq.plasma_volume_m3
    return eq


def initialize_reactor(cfg):
    # Sets up the steady-state NBI reactor environment using PyTorch Tensors.
    eq = MHDEquilibrium()
    eq.solve_grad_shafranov()
    engine = ParticlePusher(equilibrium=eq)

    _attach_flux_surfaces(eq, cfg)
    psi_grid = eq.psi_grid
    psi_core = eq.psi_core
    psi_R_min, psi_R_max = eq.psi_R_min, eq.psi_R_max
    psi_Z_min, psi_Z_max = eq.psi_Z_min, eq.psi_Z_max
    psi_nR, psi_nZ = eq.psi_nR, eq.psi_nZ
    print(f"[SYSTEM] Confined plasma volume from psi_edge surface: {eq.plasma_volume_m3:.3f} m^3")

    print(f"[SYSTEM] Initializing thermal background ({cfg.initial_thermal_count:,} particles at {cfg.T_thermal_keV} keV)...")

    # 1. Velocities (Numpy)
    thermal_velocities = engine.initialize_velocities(cfg.initial_thermal_count, T_keV=cfg.T_thermal_keV)

    # 2. Positions by rejection sampling against the core flux contour (psi > psi_core),
    # so births land inside the magnetic surface rather than anywhere in the rectangle.
    R_accum = np.empty(0, dtype=np.float64)
    Z_accum = np.empty(0, dtype=np.float64)
    while R_accum.shape[0] < cfg.initial_thermal_count:
        batch = max(2 * (cfg.initial_thermal_count - R_accum.shape[0]), 1000)
        R_try = np.random.uniform(psi_R_min, psi_R_max, batch)
        Z_try = np.random.uniform(psi_Z_min, psi_Z_max, batch)
        psi_try = interpolate_psi_array(R_try, Z_try, psi_grid, psi_R_min, psi_R_max, psi_Z_min, psi_Z_max, psi_nR, psi_nZ)
        keep = psi_try > psi_core
        R_accum = np.concatenate([R_accum, R_try[keep]])
        Z_accum = np.concatenate([Z_accum, Z_try[keep]])

    R = R_accum[:cfg.initial_thermal_count]
    Z = Z_accum[:cfg.initial_thermal_count]
    phi = np.random.uniform(0, 2 * np.pi, cfg.initial_thermal_count)

    pos_arr = np.zeros((cfg.initial_thermal_count, 3), dtype=np.float32)
    pos_arr[:, 0] = R * np.cos(phi)
    pos_arr[:, 1] = R * np.sin(phi)
    pos_arr[:, 2] = Z

    # 3. Particle types (0 = Thermal, 1 = NBI, 2 = Alpha, -1 = Lost)
    type_arr = np.zeros(cfg.initial_thermal_count, dtype=np.int32)

    # 4. Load into Apple Metal unified memory (PyTorch tensors)
    pos_tensor = torch.tensor(pos_arr, device=cfg.HPC_DEVICE, dtype=torch.float32)
    vel_tensor = torch.tensor(thermal_velocities, device=cfg.HPC_DEVICE, dtype=torch.float32)
    type_tensor = torch.tensor(type_arr, device=cfg.HPC_DEVICE, dtype=torch.int32)

    # 5. Grid arrays stay Numpy -- CIC interpolation runs on Numba/CPU. This is the
    # Poisson solver's Cartesian (R, Z) mesh; only particle spawning/confinement changed.
    rho_grid = np.zeros((cfg.nR, cfg.nZ), dtype=np.float64)
    phi_grid = np.zeros((cfg.nR, cfg.nZ), dtype=np.float64)
    E_R_grid = np.zeros((cfg.nR, cfg.nZ), dtype=np.float64)
    E_Z_grid = np.zeros((cfg.nR, cfg.nZ), dtype=np.float64)

    return eq, engine, pos_tensor, vel_tensor, type_tensor, rho_grid, phi_grid, E_R_grid, E_Z_grid


def initialize_oscillation_test(cfg):
    # Sets up the high-frequency electron plasma oscillation unit test with Tensors.
    eq = MHDEquilibrium()
    eq.solve_grad_shafranov()
    
    # Same psi surface the reactor uses, so check_confinement_flux has a boundary to test
    # against here too (this test previously used the removed circular check_confinement).
    _attach_flux_surfaces(eq)

    engine = ParticlePusher(equilibrium=eq, species="electron")
    engine.m *= cfg.macro_weight
    engine.q *= cfg.macro_weight
    
    print(f"[SYSTEM] Injecting {cfg.num_electrons:,} electrons with a Langmuir velocity perturbation...")
    k_Z = 2.0 * np.pi / (cfg.Z_max - cfg.Z_min)
    
    pos_arr, vel_arr = engine.inject_langmuir_perturbation(
        cfg.num_electrons, cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, k_Z, perturbation_amplitude=cfg.perturbation_amplitude
    )
    type_arr = np.zeros(cfg.num_electrons, dtype=np.int32)  # 0 = thermal electron

    # Send to GPU
    pos_tensor = torch.tensor(pos_arr, device=cfg.HPC_DEVICE, dtype=torch.float32)
    vel_tensor = torch.tensor(vel_arr, device=cfg.HPC_DEVICE, dtype=torch.float32)
    type_tensor = torch.tensor(type_arr, device=cfg.HPC_DEVICE, dtype=torch.int32)
    
    rho_grid = np.zeros((cfg.nR, cfg.nZ), dtype=np.float64)
    
    return eq, engine, pos_tensor, vel_tensor, type_tensor, rho_grid