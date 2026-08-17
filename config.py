import numpy as np
import torch
import platform

class SimulationConfiguration:
    def __init__(self):
        # --- PHYSICS CONSTANTS ---
        self.e_charge = 1.602e-19       # Elementary charge (C)
        self.m_deuterium = 3.344e-27    # Deuterium mass (kg)
        self.m_electron = 9.109e-31     # Electron mass (kg)
        self.eps_0 = 8.854e-12          # Vacuum permittivity (F/m)

        # Legacy aliases
        self.q = self.e_charge 
        self.m = self.m_deuterium 
        self.t_start = 0.0
        self.t_end = 4.0e-5
        self.max_step = 1e-10

        # --- MAGNETIC FIELD ---
        self.B0 = 12.0                  # Toroidal field (T)
        self.B_poloidal = 0.3           # Poloidal field (T)
        self.R0_major = 1.0             # Major radius (magnetic axis), shared by field/confinement/tearing mode

        # Target safety factor q = (a/R0) * (B_tor/B_pol), used to rescale the
        # arbitrary-unit Grad-Shafranov poloidal field. q ~ 2 is stable; q < 1 is
        # kink-unstable. Lower q -> stronger B_pol -> faster banana-orbit bounce.
        self.Q_SAFETY_TARGET = 2.0

        # --- SPATIAL GRID SETTINGS (CIC & Poisson) ---
        self.R_min = 0.5
        self.R_max = 1.5
        self.Z_min = -0.5
        self.Z_max = 0.5
        self.nR = 50
        self.nZ = 50

        # --- STEADY-STATE REACTOR SETTINGS ---
        self.reactor_num_steps = 10000
        self.reactor_dt = 1.0e-9        # Short dt to resolve the full gyro-orbit
        self.nu_c = 5000.0              # Collision frequency
        
        self.initial_thermal_count = 1000001
        self.T_thermal_keV = 1.0
        self.nbi_energy_keV = 50.0
        self.inject_every_n_steps = 2

        # Decaying beam source (a fixed batch forever meant inventory only grew, so no
        # steady state): rate(step) = NBI_BATCH_SIZE * exp(-step / NBI_DECAY_TAU_STEPS).
        # Fractional rates are realised stochastically so the beam thins out smoothly.
        self.NBI_BATCH_SIZE = 1
        self.NBI_DECAY_TAU_STEPS = 2500.0

        # 3.5 MeV alphas cover ~1.3e-2 m per global step against a ~2.2e-2 m Larmor
        # radius -- under two samples per gyro-arc, hence jagged orbits. Sub-step only
        # the alphas; deuterons/NBI are already resolved (Larmor radius ~5e-4 m).
        self.ALPHA_SUBSTEPS = 10
        # Alphas also need a finer history cadence than the 20-step thermal one, or the
        # plot stays aliased however finely the pusher integrates.
        self.ALPHA_HISTORY_EVERY = 2

        # --- PLASMA OSCILLATION TEST SETTINGS ---
        self.osc_num_steps = 1500
        self.osc_dt = 2.0e-11           # 20 ps, to capture ultra-fast electron motion
        self.num_electrons = 4000
        self.macro_weight = 5e14        # Scales charge/mass up to realistic plasma densities
        self.perturbation_amplitude = 2e5

        # --- Radial Fluid Bins ---
        self.num_radial_bins = 50

        # --- MHD INSTABILITIES & TEARING MODES ---
        self.b_perturb_initial = 1e-4   # Seed magnetic island amplitude (T)
        self.m_mode = 2                 # Poloidal mode number
        self.n_mode = 1                 # Toroidal mode number
        self.gamma_growth = 35000.0     # Exponential growth rate (gamma)

        # =======================================================
        # DISRUPTION MITIGATION & SAFETY CONTROLS (SPI)
        # =======================================================
        # Trigger threshold. Must sit above b_perturb_initial: at exactly 1e-4 the island
        # cleared it on step 1 and the run spent its whole life post-quench, flatlining
        # the thermodynamics (brem scales as sqrt(T_e)). The island grows to ~1.42e-4 over
        # 10,000 steps, so 1.2e-4 fires around the midpoint -- hot first, then quenched.
        self.MAX_ISLAND_WIDTH_THRESHOLD = 1.2e-4
        self.SPI_TRIGGERED = False              # Emergency-system flag

        # Shattered Pellet Injection (SPI) constants
        self.IMPURITY_DENSITY_NZ = 5.0e19       # Injected Neon/Argon density (m^-3)
        self.RADIATIVE_COOLING_COEFF = 1.5e-32  # Simplified L_z(T_e) [W * m^3]

        # Thermal Quench (TQ) timescales
        self.TQ_DECAY_TIME = 2.0e-3             # TQ duration (2 ms)
        self.POST_QUENCH_TEMP = 10.0            # Cold baseline temperature after TQ (eV)

      # =======================================================
        # NUCLEAR REACTION DYNAMICS (D-T FUSION)
        # =======================================================
        # Tunneling & cross-section constants
        self.E_G_keV = 34.38          # Gamow energy for D-T (keV)
        self.S_factor_base = 1000.0   # Astrophysical S-factor baseline (keV * barn)
        self.E_resonance = 64.0       # D-T resonance peak (keV)

        # =======================================================
        # REACTIVITY MATRICES & VOLUMETRIC FUSION RATES
        # =======================================================
        # Bosch-Hale D-T fit constants (1992 standard)
        self.BH_C1 = 1.17302e-9
        self.BH_C2 = 1.51361e-2
        self.BH_C3 = 7.51886e-2
        self.BH_C4 = 4.60643e-3
        self.BH_C5 = 1.35000e-2
        self.BH_C6 = -1.06750e-4
        self.BH_C7 = 1.36600e-5
        
        # Energy yield per D-T reaction
        self.E_FUSION_MEV = 17.6        # Energy released per D-T fusion (MeV)
        self.E_FUSION_JOULES = self.E_FUSION_MEV * 1e6 * self.e_charge  # ~2.8198e-12 J

        # ====================================================
        # ALPHA PARTICLE (He-4) CONSTANTS
        # ====================================================
        self.ALPHA_ENERGY_MEV = 3.5              # Birth energy
        self.ALPHA_ENERGY_JOULES = self.ALPHA_ENERGY_MEV * 1e6 * 1.6022e-19
        self.MASS_ALPHA = 4.0015 * 1.660539e-27  # He-4 nucleus mass (kg)
        self.CHARGE_ALPHA = 2.0 * 1.6022e-19     # +2e (C)

        # =====================================================================
        # ALPHA HEATING & IGNITION METRICS (PHASE V FINALE)
        # =====================================================================

        # Baseline NBI/RF heating power (MW), compared against alpha heating for ignition
        self.EXTERNAL_HEATING_MW = 40.0

        # Slowing-down time tau_s: how long a 3.5 MeV alpha takes to drag its energy
        # into the background plasma (scaled for simulation)
        self.ALPHA_SLOWING_TIME = 0.5

        # nu_s = energy drag, nu_pitch = directional scattering
        self.NU_S = 1.0 / self.ALPHA_SLOWING_TIME
        self.NU_PITCH = self.NU_S * 0.1  # Pitch scattering is much weaker for fast alphas

        # Below this energy (keV) an alpha counts as thermalized (Helium ash) and
        # stops heating
        self.THERMALIZATION_ENERGY_KEV = 15.0

        # ==========================================
        # RADIATION LOSS MODELS
        # ==========================================
        # Effective ion charge. Pure D-T is 1.0; 1.2-1.5 accounts for helium ash and
        # wall impurities.
        self.Z_eff = 1.2

        # P_brem [W/m^3] = 4.8e-37 * Z_eff * n_e^2 * sqrt(T_e [keV])
        self.bremsstrahlung_coeff = 4.8e-37

        # P_cyc [W/m^3] = 6.21e-17 * n_e * T_e [keV] * B^2
        self.cyclotron_coeff = 6.21e-17

        # Net escaping fraction of the RAW cyclotron emission above. A reactor is
        # optically thick at the fundamental and the wall reflects most of the rest, so
        # net loss sits well below bremsstrahlung (the old 0.05 put it ~2000x above).
        # Lumped, tuned stand-in for optical depth + wall reflectivity.
        self.CYCLOTRON_REABSORPTION = 5.0e-4

        # ==========================================
        # LAWSON CRITERION & Q-FACTORS
        # ==========================================
        self.eta_thermal = 0.40   # Thermal-to-electrical conversion (steam turbine)
        self.eta_heating = 0.50   # Wall-plug efficiency of NBI/RF heating
        self.lawson_target = 3.0e21  # D-T ignition triple product (keV * s * m^-3)

        # =====================================================================
        # HPC, MULTI-CORE PROCESSING & METAL GPU ACCELERATION
        # =====================================================================

        self.USE_GPU_ACCELERATION = True

        # Auto-detect Apple Silicon Metal Performance Shaders (MPS)
        if self.USE_GPU_ACCELERATION and torch.backends.mps.is_available():
            self.HPC_DEVICE = torch.device("mps")
            print("🔥 APPLE SILICON METAL ACCELERATION (MPS) ACTIVATED 🔥")
        elif self.USE_GPU_ACCELERATION and torch.cuda.is_available():
            self.HPC_DEVICE = torch.device("cuda")
            print("🔥 CUDA GPU ACTIVATED 🔥")
        else:
            self.HPC_DEVICE = torch.device("cpu")
            print("⚠️ GPU NOT FOUND. FALLING BACK TO CPU THREADS ⚠️")

        # Particle scales for the timing sweep
        self.BENCHMARK_PARTICLE_COUNTS = [1000, 10000, 100000, 1000000]
        self.BENCHMARK_STEPS = 100  # Steps per time trial