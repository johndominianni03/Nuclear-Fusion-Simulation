import numpy as np

class SimulationConfiguration:

    def __init__(self):
            # Properties of each Particle
            self.q = 1.602e-19 # Charge in Coulombs
            self.m = 3.344e-27 # Mass of Deuterium Atom (in kg)

            # Magnetic Field Calibrations
            self.B0 = 1.5 # Baseline toroidal magnetic field intensity in Tesla
            self.B_poloidal = 0.3 # Baseline poloidal magnetic field intensity in Tesla

            # Time Setting
            self.t_start = 0.0
            self.t_end = 4.0e-5
            self.max_step = 1e-10