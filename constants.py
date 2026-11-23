"""Physical constants, CODATA 2022 recommended values (SI).

The single source of every mass and charge in the code: config, the injector,
physics_engine's module constants and the tests all import from here, and
tests/test_physics.py fails if a mass or charge literal appears anywhere else.

Source: P. J. Mohr, D. B. Newell, B. N. Taylor and E. Tiesinga, "CODATA
recommended values of the fundamental physical constants: 2022", Rev. Mod. Phys.
97, 025002 (2025); https://physics.nist.gov/constants. The elementary charge is
exact by definition in the 2019 SI. The particle masses are the bare-nucleus
masses (deuteron, triton, alpha particle), not the atomic masses.
"""

E_CHARGE = 1.602176634e-19          # elementary charge e [C] (exact)
M_PROTON = 1.67262192595e-27        # proton mass [kg]
M_DEUTERON = 3.3435837768e-27       # deuteron mass [kg]
M_TRITON = 5.0073567512e-27         # triton mass [kg]
M_ALPHA = 6.6446573450e-27          # alpha-particle (He-4 nucleus) mass [kg]
M_ELECTRON = 9.1093837139e-31       # electron mass [kg]
EPSILON_0 = 8.8541878188e-12        # vacuum electric permittivity [F/m]
MU_0 = 1.25663706127e-6             # vacuum magnetic permeability [N/A^2]
C_LIGHT = 299792458.0               # speed of light in vacuum [m/s] (exact)

KEV_J = 1.0e3 * E_CHARGE            # 1 keV in joules
MEV_J = 1.0e6 * E_CHARGE            # 1 MeV in joules
