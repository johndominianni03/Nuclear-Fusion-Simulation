import numpy as np
import torch

class SimulationConfiguration:
    def __init__(self):
        # --- PHYSICS CONSTANTS ---
        self.e_charge = 1.602e-19       # Elementary charge (C)
        self.m_deuterium = 3.344e-27    # Deuterium mass (kg)
        self.m_tritium = 3.016 * 1.660539e-27   # Tritium mass (kg), 3.016 u

        # --- MAGNETIC FIELD ---
        self.B0 = 12.0                  # Toroidal field (T)
        self.R0_major = 1.0             # Major radius (magnetic axis), shared by field/confinement/tearing mode

        # Target safety factor q = (a/R0) * (B_tor/B_pol), used to rescale the
        # arbitrary-unit Grad-Shafranov poloidal field. q ~ 2 is stable; q < 1 is
        # kink-unstable. Lower q -> stronger B_pol -> faster banana-orbit bounce.
        self.Q_SAFETY_TARGET = 2.0

        # --- FIELD GRID (E_r from force balance, end-of-run density map) ---
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

        """
        initial thermal count which sets initial plasma particle count. 
        in main.py, GPU_THRESHOLD determines the initial thermal particle count 
        at which below the threshold, the simulation runs on Numba JIT (on the CPU), 
        and at or above the threshold, the simulation runs on PyTorch (on the GPU)
        """
        
        self.initial_thermal_count = 50000
        # Thermal ions are loaded over the whole last closed flux surface with peaked
        # profiles n = n0 (1 - psi_N), T = T0 (1 - psi_N); T_thermal_keV is T0, the
        # on-axis temperature. n0 follows from N_AVG_TARGET below. The count is split
        # evenly between thermal deuterium (type 0) and thermal tritium (type 3), each
        # with those same profiles and the same weight; tritium has its own pool.
        self.T_thermal_keV = 1.0
        self.nbi_energy_keV = 50.0

        # Neutral beam source, calibrated by injected power. P_NBI_MW is an INPUT, not
        # a result: 33 MW is ITER's design neutral-beam heating power. The beam is
        # injected at a constant NBI_MACRO_PER_STEP beam macroparticles per step
        # (fractional rates realised stochastically), each carrying
        #     WEIGHT[1] = P_NBI * dt / (E_beam * NBI_MACRO_PER_STEP)
        # real ions (~2e12, about 3,500x lighter than a thermal macroparticle), so
        # the weighted injected power is P_NBI while the beam keeps good statistics.
        self.P_NBI_MW = 33.0
        self.NBI_MACRO_PER_STEP = 2.0

        # Fusion-driven alpha source. Alphas are born from the computed thermal D-T
        # reaction rate, per flux shell rate_k = n_D n_T <sigma v>(T_i) V_k, recomputed
        # at every field update. ALPHA_MACRO_PER_STEP macro-alphas are born per step
        # (fractional rates realised stochastically), each of weight
        #     w_alpha = (sum_k rate_k) * dt / ALPHA_MACRO_PER_STEP,
        # which it keeps for life (the alpha pool stores one weight per alpha).
        self.ALPHA_MACRO_PER_STEP = 1.0
        self.ALPHA_BIRTH_WEIGHT = None          # latest w_alpha, set by the loop

        # 3.5 MeV alphas cover ~1.3e-2 m per global step against a ~2.2e-2 m Larmor
        # radius -- under two samples per gyro-arc, hence jagged orbits. Sub-step only
        # the alphas; deuterons/NBI are already resolved (Larmor radius ~5e-4 m).
        self.ALPHA_SUBSTEPS = 10
        # Alphas also need a finer history cadence than the 20-step thermal one, or the
        # plot stays aliased however finely the pusher integrates.
        self.ALPHA_HISTORY_EVERY = 2

        # --- MHD INSTABILITIES & TEARING MODES ---
        self.b_perturb_initial = 1e-4   # Seed magnetic island amplitude (T)
        self.m_mode = 2                 # Poloidal mode number
        self.n_mode = 1                 # Toroidal mode number
        self.gamma_growth = 35000.0     # Exponential growth rate (gamma)
        # The ripple is prescribed: b_perturb_initial * exp(gamma_growth * t), added to
        # the B every particle sees. It drives nothing else (the scripted pellet quench
        # it used to trigger was removed in Tier 2.5); the loops report its final
        # amplitude relative to B0.

        # Per-stage wall-clock instrumentation in the reactor loops. Purely
        # additive: it reads clocks and accumulates floats, and touches no
        # array, no RNG and no control flow, so a profiled run and an
        # unprofiled one produce identical numbers.
        self.PROFILE = False

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
        # ALPHA SLOWING-DOWN
        # =====================================================================

        # Slowing-down time tau_s: how long a 3.5 MeV alpha takes to drag its energy
        # into the background plasma (scaled for simulation)
        self.ALPHA_SLOWING_TIME = 0.5

        # nu_s = energy drag
        self.NU_S = 1.0 / self.ALPHA_SLOWING_TIME

        # Below this energy (keV) an alpha counts as thermalized (Helium ash) and
        # stops heating
        self.THERMALIZATION_ENERGY_KEV = 15.0

        # Volume-averaged ion density the thermal macroparticles represent (m^-3).
        # After the plasma volume is known, initialization sets the per-species
        # weights (real ions per macroparticle), keyed by type code:
        #     WEIGHT[0] (thermal D) = N_AVG_TARGET * PLASMA_VOLUME_M3 / initial_thermal_count
        # (~7e15 at production size). Every quantity that turns macroparticles into
        # physics -- the deposited density, the flux-surface profiles, stored energy,
        # the Lawson density, beta, the alpha -> bulk energy transfer -- multiplies
        # each macroparticle by its own species' weight.
        self.N_AVG_TARGET = 1.0e20
        self.WEIGHT = None                      # {type code: weight}, set by initialize_reactor

        # The radial electric field is taken from thermal-ion force balance (quasi-
        # neutral, no rotation) and rebuilt from the current particles at step 0 and
        # every FIELD_UPDATE_EVERY steps. The flux-surface profiles change only at the
        # counting-noise level over 100 steps (measured over a full production run).
        self.FIELD_UPDATE_EVERY = 100

        # Confined plasma volume (m^3). Filled in at runtime by revolving the actual
        # psi = psi_edge flux surface (physics_engine.compute_plasma_volume); the value
        # here is the circular-torus estimate 2*pi^2*R0*a^2 used only as a fallback.
        self.PLASMA_VOLUME_M3 = 2.0 * np.pi**2 * self.R0_major * (0.3 ** 2)

        # ==========================================
        # LAWSON CRITERION
        # ==========================================
        self.lawson_target = 3.0e21  # D-T ignition triple product (keV * s * m^-3)

        # =====================================================================
        # HPC, MULTI-CORE PROCESSING & METAL GPU ACCELERATION
        # =====================================================================

        self.USE_GPU_ACCELERATION = True

        # Auto-detect Apple Silicon Metal Performance Shaders (MPS)
        if self.USE_GPU_ACCELERATION and torch.backends.mps.is_available():
            self.HPC_DEVICE = torch.device("mps")
            print("APPLE SILICON METAL PERFORMANCE SHADERS SIMULATION RUN")
        elif self.USE_GPU_ACCELERATION and torch.cuda.is_available():
            self.HPC_DEVICE = torch.device("cuda")
            print("CUDA GPU SIMULATION RUN")
        else:
            self.HPC_DEVICE = torch.device("cpu")
            print("GPU NOT FOUND. FALLING BACK TO CPU THREADS")

        # Particle scales for the timing sweep
        self.BENCHMARK_PARTICLE_COUNTS = [1000, 10000, 100000, 1000000]
        self.BENCHMARK_STEPS = 100  # Steps per time trial