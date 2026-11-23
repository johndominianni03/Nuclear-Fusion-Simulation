import numpy as np
import torch
from mhd_equilibrium import MHDEquilibrium
from physics_engine import (
    ParticlePusher,
    compute_flux_thresholds,
    compute_flux_boundary_contour,   # psi_edge wall polygon, shared by physics and plots
    compute_plasma_volume,           # revolved volume of that surface
    compute_psi_normalised,
    thermal_collision_frequencies,   # NRL ion-ion rate at the loaded central n, T
)

def _attach_flux_surfaces(eq, cfg=None, R0=1.0, a_minor=0.3):
    # Flux-surface (psi) boundaries, calibrated against the legacy circular region
    # (R0=1.0, a_minor=0.3) so the confined volume is comparably sized but follows the
    # real magnetic topology. Stashed on `eq` so downstream code reads them via eq.psi_*
    # without changing any caller's positionally-unpacked return signature.
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
    psi_R_min, psi_R_max = eq.psi_R_min, eq.psi_R_max
    psi_Z_min, psi_Z_max = eq.psi_Z_min, eq.psi_Z_max
    print(f"[SYSTEM] Confined plasma volume from psi_edge surface: {eq.plasma_volume_m3:.3f} m^3")

    # Real ions per macroparticle, per species (keyed by type code), so that the
    # thermal population (D + T together) represents a volume-averaged ion density
    # of N_AVG_TARGET.
    w_thermal = cfg.N_AVG_TARGET * cfg.PLASMA_VOLUME_M3 / cfg.initial_thermal_count
    # Beam weight from the injected power: NBI_MACRO_PER_STEP macroparticles per
    # step at E_beam each carry P_NBI * dt of energy between them.
    E_beam_J = cfg.nbi_energy_keV * 1.0e3 * cfg.e_charge
    # NBI_MACRO_PER_STEP = 0 switches the beam off (no injections, weight 0).
    w_beam = (cfg.P_NBI_MW * 1.0e6 * cfg.reactor_dt / (E_beam_J * cfg.NBI_MACRO_PER_STEP)
              if cfg.NBI_MACRO_PER_STEP > 0 else 0.0)
    cfg.WEIGHT = {0: w_thermal,     # thermal D
                  1: w_beam,        # beam D
                  2: 0.0,           # alpha: each macro-alpha carries its own birth weight
                                    # in the alpha pool; the loop keeps the latest here
                  3: w_thermal}     # thermal T
    print(f"[SYSTEM] Thermal macroparticle weight: {w_thermal:.3e} ions per macroparticle "
          f"(<n> = {cfg.N_AVG_TARGET:.2e} m^-3 over {cfg.PLASMA_VOLUME_M3:.3f} m^3)")
    print(f"[SYSTEM] w_beam = {w_beam:.3e} ions per beam macroparticle "
          f"(P_NBI = {cfg.P_NBI_MW:.1f} MW at {cfg.nbi_energy_keV:.0f} keV, "
          f"{cfg.NBI_MACRO_PER_STEP:g} macroparticles/step"
          + (f"; {w_thermal / w_beam:,.0f}x lighter than a thermal macroparticle)" if w_beam > 0 else ")"))

    # Collision rates: the NRL ion collision rate at the loaded central density and
    # temperature (isotropic re-randomisation at that fixed rate in the loops).
    cfg.NU_C, _nu_info = thermal_collision_frequencies(cfg, eq)
    print(f"[SYSTEM] Collisions (NRL ion-ion rate at n0 = {_nu_info['n0']:.3e} m^-3, "
          f"T0 = {_nu_info['T0_keV']:g} keV, ln Lambda = {_nu_info['coulomb_log']:.2f}): "
          f"nu_D = {cfg.NU_C[0]:.2f} /s, nu_T = {cfg.NU_C[3]:.2f} /s")

    # Thermal D and thermal T, half each (D gets the odd one out of an odd count).
    N = cfg.initial_thermal_count
    N_D = N - N // 2
    N_T = N // 2
    print(f"[SYSTEM] Initializing thermal background ({N_D:,} D + {N_T:,} T macroparticles, "
          f"n and T proportional to 1 - psi_N, T = {cfg.T_thermal_keV} keV on axis)...")

    # 1. Positions: peaked density n = n0 (1 - psi_N) over the whole last closed flux
    # surface. A sample uniform in (R, Z) represents a 3D density proportional to 1/R
    # (a cell's revolved volume is 2 pi R dR dZ), so each is accepted with probability
    # R (1 - psi_N) / R_max, which leaves the accepted set with n proportional to (1 - psi_N).
    # The samples are independent, so the first N_D become deuterons and the rest
    # tritons: each species gets the same density profile.
    R_accum = np.empty(0, dtype=np.float64)
    Z_accum = np.empty(0, dtype=np.float64)
    while R_accum.shape[0] < N:
        batch = max(8 * (N - R_accum.shape[0]), 1000)
        R_try = np.random.uniform(psi_R_min, psi_R_max, batch)
        Z_try = np.random.uniform(psi_Z_min, psi_Z_max, batch)
        psiN_try = compute_psi_normalised(R_try, Z_try, eq)
        accept = np.where(psiN_try < 1.0, R_try * (1.0 - psiN_try) / psi_R_max, 0.0)
        keep = np.random.uniform(0.0, 1.0, batch) < accept
        R_accum = np.concatenate([R_accum, R_try[keep]])
        Z_accum = np.concatenate([Z_accum, Z_try[keep]])

    R = R_accum[:N]
    Z = Z_accum[:N]

    # 2. Velocities: a local Maxwellian at T = T0 (1 - psi_N), each species at its
    # own mass (the deuteron rows keep engine.m, the mass they were loaded with
    # before tritium existed).
    T_local_keV = cfg.T_thermal_keV * (1.0 - compute_psi_normalised(R, Z, eq))
    mass_row = np.concatenate([np.full(N_D, engine.m), np.full(N_T, cfg.m_tritium)])
    thermal_velocities = engine.initialize_velocities_profile(T_local_keV, mass=mass_row)

    phi = np.random.uniform(0, 2 * np.pi, cfg.initial_thermal_count)

    pos_arr = np.zeros((cfg.initial_thermal_count, 3), dtype=np.float32)
    pos_arr[:, 0] = R * np.cos(phi)
    pos_arr[:, 1] = R * np.sin(phi)
    pos_arr[:, 2] = Z

    # 3. Particle types (0 = thermal D, 1 = beam D, 2 = alpha, 3 = thermal T, -1 = lost)
    type_arr = np.zeros(cfg.initial_thermal_count, dtype=np.int32)
    type_arr[N_D:] = 3

    # 4. Load into Apple Metal unified memory (PyTorch tensors)
    pos_tensor = torch.tensor(pos_arr, device=cfg.HPC_DEVICE, dtype=torch.float32)
    vel_tensor = torch.tensor(thermal_velocities, device=cfg.HPC_DEVICE, dtype=torch.float32)
    type_tensor = torch.tensor(type_arr, device=cfg.HPC_DEVICE, dtype=torch.int32)

    # 5. Field grids on the Cartesian (R, Z) mesh, Numpy. The reactor loop fills E
    # and phi from ion force balance at step 0; rho is the end-of-run density map.
    rho_grid = np.zeros((cfg.nR, cfg.nZ), dtype=np.float64)
    phi_grid = np.zeros((cfg.nR, cfg.nZ), dtype=np.float64)
    E_R_grid = np.zeros((cfg.nR, cfg.nZ), dtype=np.float64)
    E_Z_grid = np.zeros((cfg.nR, cfg.nZ), dtype=np.float64)

    return eq, engine, pos_tensor, vel_tensor, type_tensor, rho_grid, phi_grid, E_R_grid, E_Z_grid
