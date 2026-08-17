# Hardware-Adaptive Tokamak Plasma PIC (Particle-In-Cell) Simulation

> A 26-week High-Performance Computing (HPC) physics engine simulating multi-particle plasma dynamics, magnetic confinement, and D-T fusion ignition criteria.

## Overview
This project is a custom-built Particle-In-Cell (PIC) Tokamak reactor simulation. It models the complex kinetic behaviors of multi-species plasmas—including 1 keV Thermal Ions, 50 keV Neutral Beam Injection (NBI) Fast Ions, and 3.5 MeV Alpha particles—within a self-consistent Grad-Shafranov magnetic equilibrium. 

Crucially, the simulation features a **hardware-adaptive hybrid backend**. It dynamically routes compute workloads between a highly multi-threaded CPU path (pure NumPy/Numba) and a massive GPU path (PyTorch via Apple Metal MPS or NVIDIA CUDA) based on real-time particle counts, thermal limitations, and memory bandwidth profiling.

---

## Development Timeline & Physics Explanations

### Phase I: Single Particle Kinematics & Boundaries
* **Part 1:** Grad-Shafranov Grid Initialization & Field Interpolation. *(The Grad-Shafranov equation maps the steady-state magnetic flux surfaces, establishing the foundational magnetic equilibrium for the tokamak).*
* **Part 2:** 3D Guiding Center / Boris Particle Pusher. *(The Boris algorithm is a symplectic numerical integrator used to solve the Lorentz force equations, implemented to accurately conserve energy and phase-space volume over long simulations, i.e. keeping the simulation as accurate as possible).*
* **Part 3:** Numba JIT Acceleration & NumPy Vectorization. *(performs the calculations of the main loop using Numba JIT (Just-In-Time) by converting the Python code into C code, allowing the simulation to run faster. Numba is used for running the simulation on the CPU -- better for smaller sample sizes of particles.
* **Part 4:** Maxwellian Thermal Tails & Divertor Wall-Loss Metrics. *(Initializes the thermal core plasma using a Maxwell-Boltzmann velocity distribution, accounting for statistical high-energy "tails").*

### Phase II: Plasma Collisionality & External Heating
* **Part 5:** Monte Carlo Coulomb Collisions (Pitch-Angle Scattering).
* **Part 6:** Magnetic Trapping & Banana Orbit Diagnostics. *(The 1/R decay of the toroidal magnetic field creates a magnetic mirror effect, trapping specific particles on the outboard side of the torus and creating clearly defined "banana-shaped" drift trajectories (mirroring is the vertical bouncing of the particles)).*
* **Part 7:** External Heating Models (Neutral Beam Injection - NBI). *(Simulates the injection of high-energy neutral atoms that penetrate the magnetic field before ionizing and transferring their kinetic energy to the bulk plasma). This is what actually causes fusion to take place, as the NBI Particles heat up the plasma to fusion temperatures.*
* **Part 8:** Refactoring.

### Phase III: Self-Consistent Electromagnetic Fields (PIC)
* **Part 9:** Charge Density Mapping (Particle-to-Grid Weighting). *(Maps the actual densities/clustering of plasma particles/charges)*
* **Part 10:** Poisson’s Equation Solver (Electric Field Generation). *(Maps the electric field potential of the plasma)*
* **Part 11:** Particle-in-Cell (PIC) Integration. *(Particles are weighted to a spatial grid to map localized charge density rho, allowing the Poisson solver to dynamically calculate the self-consistent electrostatic potential $\phi$ at each timestep).*
* **Part 12:** Debye Shielding & Plasma Oscillations. *(Demonstrates the plasma's natural tendency to shield electric charges, validated by electrostatic energy ringing perfectly at $2f_{pe}$, or twice the plasma frequency).*

### Phase IV: Magnetohydrodynamics (MHD) & Instabilities
* **Part 13:** Fluid Approximations (Density & Pressure Profiles).
* **Part 14:** The Vlasov Equation & Kinetic-Fluid Bridging.
* **Part 15:** Simulating Plasma Instabilities (e.g., Sawtooth / Tearing Modes). *(Models magnetohydrodynamic instabilities, such as the $m=2, n=1$ tearing mode, which creates magnetic islands and disrupts confinement).*
* **Part 16:** Disruption Mitigation Diagnostics. *(Models Shattered Pellet Injection (SPI), where rapid injection of material causes a controlled thermal quench to radiate away energy and protect the reactor walls).*

### Phase V: Nuclear Reaction Dynamics
* **Part 17:** D-T Fusion Cross-Section Algorithms. *(Models the quantum tunneling probability of Deuterium-Tritium fusion, successfully capturing the resonance peak at ~8.6 keV center-of-mass collision energy).*
* **Part 18:** Reactivity Matrices & Volumetric Fusion Rates.
* **Part 19:** Alpha Particle Generation & Birth Trajectories (3.5 MeV).
* **Part 20:** Alpha Heating (Self-Sustaining Ignition Metrics). *(Tracks the transition into a "Burning Plasma" regime, strictly defined as the threshold where internal 3.5 MeV alpha self-heating officially exceeds external auxiliary heating).*

### Phase VI: Advanced Reactor Engineering & HPC Polish
* **Part 21:** Bremsstrahlung & Cyclotron Radiation Loss Models.
* **Part 22:** The Lawson Criterion & Q-Factor Calculation. *(Accurately calculates both Scientific Gain $Q_{sci}$ and Engineering Gain $Q_{eng}$. The simulation mathematically respects thermodynamics by forcing $Q_{eng}$ to be lower than $Q_{sci}$ due to heating system inefficiencies and thermal-to-electric conversion losses).*
* **Part 23:** Multi-Core Processing & Universal GPU Acceleration (Numba/MPS/CUDA).

---

## HPC Architecture & Hardware Scaling
The physics engine is designed around the fundamental HPC principle of **Arithmetic Intensity**. Because the Boris particle push relies on heavy data movement ($\sim 72$ bytes per particle) but minimal floating-point operations ($\sim 40$ FLOPs, for a ratio of 0.55 FLOPs/byte), the simulation is strictly **memory-bandwidth bound**, not compute-bound.

To maximize throughput across completely different silicon architectures, the engine utilizes dynamic dispatch thresholds:
* **CPU Path (Numba `prange`):** Used for standard runs. Utilizes 100% of available CPU cores with zero Python interpreter overhead to bypass GPU dispatch tolls entirely.
* **GPU Path (PyTorch MPS/CUDA):** Used for massive array processing. Bypasses the CPU to load multi-gigabyte tensors directly into unified memory or discrete VRAM to saturate massive memory pipelines.


---

## Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/johndominianni03/Nuclear-Fusion-Simulation.git](https://github.com/johndominianni03/Nuclear-Fusion-Simulation.git)
cd Nuclear-Fusion-Simulation
