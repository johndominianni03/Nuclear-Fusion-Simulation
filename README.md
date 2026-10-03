# Particle-In-Cell Tokamak Fusion Simulation

A multi-species particle-in-cell simulation of deuterium–tritium fusion in a magnetically
confined tokamak plasma. Models Grad–Shafranov equilibrium, Monte Carlo Coulomb collisions,
neutral beam injection, alpha particle self-heating, radiation losses, and MHD instabilities,
with 17 diagnostic outputs.

Written in Python with Numba JIT and PyTorch backends.

---

## How it works

Each timestep of the reactor run follows the same loop:

1. **Field lookup.** Each particle's position is mapped onto the Grad–Shafranov equilibrium to get the local magnetic field.
2. **Push.** The Boris integrator advances every particle's velocity and position under the Lorentz force.
3. **Collisions.** Monte Carlo Coulomb collisions scatter particles in pitch angle, which degrades confinement.
4. **Heating and sources.** Neutral beam injection adds fast ions. D–T fusion events spawn 3.5 MeV alpha particles.
5. **Losses.** Bremsstrahlung and cyclotron radiation are subtracted. Particles that reach the wall or divertor are counted and removed.
6. **Instabilities.** Sawtooth and tearing-mode perturbations and disruption-mitigation events are applied.
7. **Diagnostics.** Fusion power, alpha heating, Lawson quantities and profiles are accumulated and then plotted at the end of the run.

## Physics and features

**Magnetic equilibrium.** `mhd_equilibrium.py` solves the Grad–Shafranov equation for a D-shaped tokamak cross-section. The result is cached to disk, so repeated runs skip the solve. Fields are interpolated to each particle's position.

**Particle dynamics.** A 3D Boris pusher advances all particles. Its energy error stays bounded and oscillatory rather than drifting, so orbits remain stable over long runs. The bulk plasma is initialized from a Maxwell–Boltzmann distribution so the high-energy tail that dominates fusion reactivity is present from the start. Trapped particles trace banana orbits in the 1/R field.

**Collisional transport.** Monte Carlo Coulomb collisions (pitch-angle scattering) knock particles off ideal orbits and drive transport toward the walls.

**External heating.** Neutral beam injection adds high-energy ions in the core, raising the temperature toward fusion-relevant values.

**Fusion and self-heating.** D–T cross-sections and reactivity give a volumetric fusion rate and power. Alphas are spawned at 3.5 MeV, tracked along their wide birth orbits, and deposit heat in the plasma. The run reports when alpha heating overtakes NBI, which marks the move toward a burning plasma.

**Radiation and gain.** Bremsstrahlung and cyclotron losses are subtracted every step. The Lawson criterion and the Q-factor (scientific and engineering gain, including thermal-to-electric conversion losses) are computed independently of the electrostatic scaffolding.

**MHD and disruptions.** The bulk plasma is also described as a compressible fluid (density and pressure profiles), bridged to the kinetic particles via the Vlasov description. Sawtooth and tearing-mode instabilities cause realistic energy bleed-out. Shattered pellet injection is modeled as a disruption-mitigation response that forces a controlled thermal quench.

**Charge deposition and field solve.** Cloud-in-cell deposition and an SOR Poisson solve run every step as structural scaffolding for a PIC loop. As described in *Scope and limitations*, the resulting electrostatic field is not self-consistent at this grid resolution, and the confinement, transport, fusion-power and Lawson results do not depend on it.

**Dual backends.** The reactor loop exists as a Numba JIT CPU implementation and a PyTorch GPU implementation (CUDA or Apple MPS), chosen at runtime (see *Choosing the CPU or GPU path*).

## Graphical outputs

A full run writes 14 reactor plots to the repo root. Three more diagnostics have their own entry points.

### Reactor run (`python main.py`)

**Geometry and structure**

| Output | What it shows |
|---|---|
| `tokamak_reactor_2d.png` | Poloidal cross-section of the D-shaped plasma: flux surfaces and particle distribution. |
| `tokamak_reactor_3d.png` | 3D view of the toroidal plasma and particle positions. |
| `radial_profiles.png` | Radial profiles of density, temperature and pressure from the core to the edge. |
| `phase_space_map.png` | Particle distribution in phase space (position versus velocity), showing the thermal bulk, the Maxwellian tail and injected beam ions. |

**Fusion, heating and gain**

| Output | What it shows |
|---|---|
| `fusion_power_density.png` | Volumetric D–T fusion power density. |
| `alpha_heating_balance.png` | Alpha self-heating against external NBI heating (the ignition metric). |
| `alpha_orbits.png` | Birth trajectories of fusion alphas, showing their wide-looping orbits. |
| `lawson_q_factor.png` | Lawson criterion and Q-factor (scientific and engineering gain). |
| `radiation_loss_profile.png` | Bremsstrahlung and cyclotron radiation losses. |
| `plasma_stored_energy_time.png` | Plasma stored energy over time. |

**Instabilities and disruptions**

| Output | What it shows |
|---|---|
| `instability_growth.png` | Growth of the sawtooth and tearing-mode (magnetic island) perturbations. |
| `disruption_mitigation.png` | Thermal quench and radiated-energy response during shattered pellet injection. |

**Electrostatic scaffolding**

| Output | What it shows |
|---|---|
| `charge_density_map.png` | Cloud-in-cell charge deposition on the grid. |
| `potential_field_map.png` | Electrostatic potential from the SOR Poisson solve. |

The two electrostatic plots show the PIC machinery working, but they are not a converged self-consistent field (see *Scope and limitations*).

`alpha_orbits.png` and `disruption_mitigation.png` appear only when the run is long and large enough to produce those events, so they are absent at small test sizes. The regression manifest accounts for this.

### Standalone diagnostics

| Entry point | Output | What it shows |
|---|---|---|
| `benchmarks.run_hpc_benchmark()` | `benchmark_scaling.png` | Runtime versus particle count on your hardware, for both backends. |
| `main.run_plasma_oscillation_test()` | `plasma_oscillation_frequency.png` | Measured plasma oscillation frequency against theory, a Debye-shielding check kept separate from the quasi-neutral reactor path. |
| `main.run_nuclear_reaction_dynamics()` | `fusion_cross_section.png` | D–T fusion cross-section versus energy, showing the tunneling-enabled reaction rate. |

### Reading the results

The plots most directly tied to the physics this project claims are the radial profiles, orbit plots, fusion power density, alpha heating balance, radiation losses, stored energy and the Lawson/Q panel. Treat the charge density and potential maps as diagnostics of the scaffolding, per *Scope and limitations*.
---

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
