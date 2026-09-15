# Tokamak Fusion Simulation

A multi-species particle-in-cell simulation of deuterium–tritium fusion in a magnetically
confined tokamak plasma. Models Grad–Shafranov equilibrium, Monte Carlo Coulomb collisions,
neutral beam injection, alpha particle self-heating, radiation losses, and MHD instabilities,
with 17 diagnostic outputs.

Written in Python with Numba JIT and PyTorch backends.

---

## Performance

Production case: 50,000 particles, 10,000 steps, CPU/Numba path.

| | loop wall |
|---|---|
| baseline | 88.4 s |
| optimized | 33.0 s |

The dominant cost in the original loop was not computation. Profiling showed roughly 65% of
the Boris push stage was masked gather/scatter — building boolean masks over the full particle
array, copying the selected rows out, pushing them, and writing them back — to operate on a
species that made up a fraction of a percent of the population. Restructuring particles into
contiguous per-species memory pools removed that pattern entirely; the two sub-timers that
measured it now total 0.006 s of a 20.4 s stage.

Every optimization was verified numerically inert against a byte-level regression harness:
43 arrays, bit-identical, strict `rtol=1e-6`. The 2,437-line main module was subsequently
split into eight modules across eight commits, each one proven bit-identical.

---

## Scope and limitations

The reactor path models magnetic confinement with collisional transport. The electrostatic
field is **not** self-consistent: the plasma is quasi-neutral by construction with no separate
electron population, and the 50×50 grid under-resolves the Debye length by ~870×, so a
resolved electrostatic solve isn't meaningful at this scale. Fusion power, radiation losses,
and Lawson diagnostics are computed independently and are unaffected.

The charge deposition, Poisson solve, and field gather are implemented and run every step, but
should be read as structural scaffolding for a PIC loop rather than as a converged
electrostatic solution. The solver's convergence test is absolute (`1e-5` V) against a solution
of the same order, so it exits after a single sweep; the accumulated field is a running sum of
un-converged sweeps. This is documented rather than hidden because the physics the project
does claim to model — confinement, collisional transport, fusion power, alpha heating, the
Lawson criterion — does not depend on it.

---

## Installation

Running in a virtual environment is strongly recommended.

```bash
git clone https://github.com/johndominianni03/Nuclear-Fusion-Simulation.git
cd Nuclear-Fusion-Simulation
```

**macOS / Linux** (macOS uses Apple Metal Performance Shaders for PyTorch)

```bash
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows** (uses NVIDIA CUDA for PyTorch)

```bash
python -m venv venv
venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install numba numpy pandas matplotlib scipy
```

---

## Running it

The main reactor run, which writes the 14 reactor diagnostic plots:

```bash
python main.py
```

Or, to call it directly and control plotting:

```bash
python -c "import main; main.run_reactor_steady_state()"
```

Three diagnostics live behind their own entry points and are not part of the reactor run:

```bash
python -c "import benchmarks; benchmarks.run_hpc_benchmark()"          # benchmark_scaling.png
python -c "import main; main.run_plasma_oscillation_test()"            # plasma_oscillation_frequency.png
python -c "import main; main.run_nuclear_reaction_dynamics()"          # fusion_cross_section.png
```

**Profiling.** Set `cfg.PROFILE = True` in `config.py` to print per-stage wall-clock tables:
the loop table, the deuteron-push breakdown, the SOR iteration histogram, and an out-of-loop
table. Plot writing is gated behind it — a profiling run writes no PNGs unless you pass
`save_plots=True` explicitly.

**Regression tests.**

```bash
python tests/test_regression.py compare           # strict, rtol 1e-6
python tests/test_regression.py compare --loose   # rtol 1e-3
python tests/test_regression.py compare-plots     # renders to a temp dir, diffs the manifest
```

### Choosing the CPU or GPU path

The simulation dispatches between a Numba JIT loop and a PyTorch loop at runtime, based on
`GPU_PARTICLE_THRESHOLD` in `main.py` and `initial_thermal_particle_count` in `config.py`.

`GPU_PARTICLE_THRESHOLD` **defaults to `None`, which always selects the CPU path.** That is
deliberate: on the development machine (Apple Silicon / MPS) the Numba loop measured faster at
every particle count tested, up to 3,000,000 — 32–36 s against 69.6 s for the GPU loop at
1M particles / 200 steps. The GPU path is fully functional and maintained, just not the default.

To force the GPU path, set the threshold to any integer at or below your particle count, either
by editing `main.py` or at runtime:

```python
import main
main.GPU_PARTICLE_THRESHOLD = 0
main.run_reactor_steady_state()
```

Performance scales with hardware on both paths. The GPU path is expected to do better on a
discrete NVIDIA card than on Apple Silicon: a unified-memory architecture cannot fully offload
the workload to the GPU, whereas a discrete card can. On Apple Silicon, memory bandwidth
(GB/s) is the dominant factor.

---

## Architecture

| module | lines | holds |
|---|---|---|
| `reactor_gpu.py` | 658 | the PyTorch reactor loop |
| `reactor_cpu.py` | 649 | the Numba reactor loop, adaptive Boris push |
| `main.py` | 401 | entry points, loop dispatch, result packaging |
| `physics_engine.py` | 1518 | physics kernels, CPU and torch |
| `track_store.py` | 280 | trajectory sampling, pid→row maps |
| `profiling.py` | 200 | per-stage wall-clock instrumentation |
| `particle_pool.py` | 166 | capacity-backed particle pools, compaction |
| `kernels.py` | 152 | field gather, collisions |
| `benchmarks.py` | 75 | hardware scaling sweep |

Plus `config.py`, `mhd_equilibrium.py` (Grad–Shafranov solve with disk cache),
`initialization.py`, `diagnostics.py`, `visualizer.py`, `tests/test_regression.py`, and
`tools/`.

---

## Development timeline and physics

### Phase I — Single particle kinematics and boundaries

**1. Grad–Shafranov grid initialization and field interpolation.** The Grad–Shafranov equation
maps the steady-state magnetic flux surfaces of the reactor. This establishes the foundational
magnetic equilibrium, shaping the plasma into a realistic D-shaped torus rather than a simple,
unphysical cylinder.

**2. 3D guiding center / Boris particle pusher.** Standard integrators gradually add artificial
energy to a simulation, causing virtual particles to speed up over time and ruin the data. The
Boris algorithm is used because it is volume-preserving and time-reversible: its energy error
stays bounded and oscillatory rather than accumulating, so orbits remain stable over millions
of steps. (It is not symplectic in the canonical sense — shown by Qin et al., 2013 — but it has
the conservation property that matters here.)

**3. Numba JIT acceleration and NumPy vectorization.** Pure Python is far too slow for
multi-particle physics. Numba compiles the hot loops through LLVM to native machine code at
first call, unlocking multi-core data parallelism on the CPU.

**4. Maxwellian thermal tails and divertor wall-loss metrics.** Plasmas do not sit at one
uniform temperature. The core plasma is initialized from a Maxwell–Boltzmann distribution to
capture high-energy tails — the rare, ultra-fast outliers most likely to overcome the Coulomb
barrier and fuse.

### Phase II — Collisionality and external heating

**5. Monte Carlo Coulomb collisions (pitch-angle scattering).** Charged particles constantly
deflect off one another's fields. Randomized scattering models realistic confinement
degradation as particles are knocked off ideal orbits and toward the walls.

**6. Magnetic trapping and banana orbit diagnostics.** A tokamak's toroidal field falls off as
1/R, creating a magnetic mirror. Particles on the outboard side bounce back and forth in the
tightening field, tracing the characteristic banana-shaped orbits. Modeling this matters for
understanding which particles stay trapped rather than circulating freely.

**7. External heating — neutral beam injection.** Magnetic fields cannot push more heat into a
magnetic cage. NBI fires high-energy neutral atoms, which ignore the field, straight into the
core; they ionize on arrival, become trapped, and collide with the bulk plasma, raising core
temperature toward fusion-relevant levels.

**8. Codebase refactoring and energy conservation audits.** A structural milestone: clean the
architecture, remove redundant code, and verify that no energy is artificially created or
destroyed during the heating phases.

### Phase III — Charge deposition and field solve

See **Scope and limitations** above: this phase implements the machinery of a PIC loop, but the
electrostatic field it produces is not self-consistent at this grid resolution.

**9. Charge density mapping (particle-to-grid weighting).** Computing pairwise forces between
millions of particles is intractable. Cloud-in-cell deposition maps discrete particles onto a
continuous spatial grid, locating where charge is pooling.

**10. Poisson solver (electric field generation).** A successive over-relaxation solve in
cylindrical coordinates translates the charge density grid into a macroscopic electric field.
As documented above, the convergence criterion is inert at this scale — the solve exits after
one sweep and the field is a running sum of un-converged sweeps.

**11. Particle-in-cell integration.** Ties the loop together: particles move and deposit
charge, the charge produces a field, the field pushes back. The structural feedback loop is
present; the electrostatic component of it is not physically resolved here.

**12. Debye shielding and plasma oscillations.** A separate standalone test demonstrating the
plasma's tendency to rearrange charge to screen out rogue electric fields. Run independently of
the reactor loop via `run_plasma_oscillation_test`, since the reactor path is quasi-neutral by
construction and has no separate electron population to do the screening.

### Phase IV — Magnetohydrodynamics and instabilities

**13. Fluid approximations (density and pressure profiles).** Individual particle tracking
captures micro-physics, but reactors are governed by macro-physics. The bulk plasma is also
modeled as a continuous compressible fluid to analyze global pressure gradients.

**14. Vlasov equation and kinetic–fluid bridging.** The translation layer between micro-scale
particle tracking and macro-scale fluid dynamics, keeping both descriptions consistent as the
simulation evolves.

**15. Plasma instabilities (sawtooth / tearing modes).** Plasmas actively fight confinement.
MHD instabilities such as magnetic islands act as potholes in the field, forcing the simulation
to contend with realistic energy bleed-out and structural disruption.

**16. Disruption mitigation diagnostics.** A plasma losing control can melt the reactor wall.
This is the emergency brake: shattered pellet injection rapidly introduces heavy material to
force a controlled thermal quench, radiating energy away before it lands on the wall.

### Phase V — Nuclear reaction dynamics

**17. D–T fusion cross-section.** Nuclei repel each other and classically shouldn't fuse. This
computes the quantum tunneling probability for deuterium and tritium, letting sufficiently fast
particles penetrate the Coulomb barrier.

**18. Reactivity matrices and volumetric fusion rates.** Scales individual fusion probabilities
to a macroscopic rate, giving megawatts of fusion power per cubic meter in real time.

**19. Alpha particle generation and birth trajectories.** D–T fusion leaves a helium nucleus.
Alphas are spawned dynamically at 3.5 MeV and their wide-looping birth orbits are tracked.

**20. Alpha heating and ignition metrics.** The threshold where heat from newly born alphas
overtakes external NBI heating — the transition to a self-sustaining burning plasma.

### Phase VI — Reactor engineering and HPC

**21. Bremsstrahlung and cyclotron radiation losses.** Plasmas radiate heavily in X-rays and
microwaves. Continuously subtracting radiated energy prevents artificial overheating and
enforces realistic thermodynamic limits.

**22. Lawson criterion and Q-factor.** The scorecard: heat generated against energy lost,
yielding both scientific and engineering gain, with realistic system inefficiencies such as
thermal-to-electric conversion losses included.

**23. Multi-core and GPU backends.** Two complete implementations of the reactor loop — Numba
JIT across CPU cores, and PyTorch targeting CUDA or Apple MPS — selected at runtime. Which one
wins depends on hardware: see **Choosing the CPU or GPU path** above for the measured numbers
on this machine, where the CPU path is currently faster and is the default.

---

## Optimization and verification

Four changes to the hot loop:

- **Allocation-free field gather.** The gather kernel allocated and freed a fresh pair of
  `(N, 3)` float32 arrays every step — 24 MB per step at 1M particles, ~10,000 times per run.
  Replaced with an out-parameter form writing into buffers allocated once.
- **Vectorized trajectory history.** Four pid-keyed Python dicts replaced with slot-indexed
  arrays and a single flat vertex buffer, regrouped once at end of run by a stable argsort.
- **Capacity-backed particle pools.** Particle arrays were grown by full reallocation on every
  injection. Replaced with fixed-capacity pools and an `n_live` pointer, with a hard ceiling
  that raises rather than growing silently.
- **Split bulk and alpha pools.** The change that mattered. Five boolean masks eliminated;
  every kernel now runs on a contiguous `[:n_live]` view.

Verification throughout was bit-identity rather than tolerance: a 43-array dump compared
byte-for-byte against a reference tree. Where a stage gain was real but the whole-loop
difference sat inside run-to-run noise, no total speedup is claimed.