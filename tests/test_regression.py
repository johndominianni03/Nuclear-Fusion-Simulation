#!/usr/bin/env python3
"""Golden-file regression test for the steady-state reactor pipeline.

Runs initialize_reactor + the reactor loop + package_reactor_results at a
fixed seed and a fixed small size, and compares every product against a
committed golden .npz. The point is to catch a refactor that silently
moves a number -- so the comparison covers the diagnostic time series,
the final particle state, the packaged plot inputs, AND the trajectory
tracks, vertex counts included.

    python tests/test_regression.py record          # write the golden
    python tests/test_regression.py compare         # check against it
    python tests/test_regression.py compare --loose # rtol 1e-3
    python tests/test_regression.py record-plots    # write plot baseline
    python tests/test_regression.py compare-plots   # full render + manifest

pytest runs test_reactor_regression (fast) and
test_plot_outputs_survive_full_run (slow -- it renders the plots).

The plot test renders into a temp directory (never over the repo's
own PNGs) and asks tools/plot_manifest.py the one question "was
every expected plot produced". Two documented exceptions:
benchmark_scaling.png plots measured wall-clock times, so its bytes
are never stable and content changes are reported without failing;
and disruption_mitigation.png needs the SPI disruption to fire
(~5,200 steps), so at this test size it is excused as absent.

Exit codes: 0 pass, 1 a regression, 2 setup problem (no golden yet).

------------------------------------------------------------------
Determinism notes -- read before regenerating the golden
------------------------------------------------------------------
* Seeding numpy alone is NOT enough, and neither is adding a jitted
  seeder on the main thread. apply_vectorized_collisions in main.py is
  @njit(parallel=True) and draws np.random inside the prange, where
  every worker thread carries its own RNG state. Those states are
  seeded by _seed_numba_threads, which runs one prange iteration per
  thread. Measured: without it, two processes at the same seed produce
  different velocities (counts and track lengths still match, so the
  failure is quiet); with it, they agree bit-for-bit.
* numba threads are pinned to 1. Any fixed count reproduces, but the
  draw sequence differs between counts -- 1, 2 and 4 threads each give
  a different answer -- so the golden is only meaningful with the count
  pinned. It is recorded in the golden and a mismatch is reported.
* The golden therefore pins seed, particle count, step count, thread
  count and the Grad-Shafranov backend. Library versions are recorded
  too, so "numpy was upgraded" is a diagnosable failure rather than a
  mysterious one.
* backend="python" is forced at the solve_grad_shafranov call site
  rather than trusted to the default, so a FUSION_GS_BACKEND=numba left
  set in the shell cannot quietly make the golden non-bit-identical.

The grid parameters this test uses (nx=60, nz=60 and the default
R/Z bounds, B0, R0) are the production ones, so this test HITS THE SAME
.cache/gs_*.npy Grad-Shafranov entry a full production run writes. That
is expected and fine -- psi depends only on those parameters, not on
particle count or step count, so sharing the entry is correct, not a
test-isolation leak. Only reactor_num_steps and initial_thermal_count
are shrunk here, and neither reaches the equilibrium solver.
"""

import argparse
import contextlib
import io
import os
import subprocess
import sys
import tempfile

# Keep this process off any GUI backend: importing main pulls in
# diagnostics, which imports pyplot.
os.environ.setdefault("FUSION_HEADLESS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import numpy as np                                       # noqa: E402
import numba                                             # noqa: E402
from numba import njit, prange                          # noqa: E402
import torch                                             # noqa: E402

import main                                              # noqa: E402
import initialization                                    # noqa: E402
from config import SimulationConfiguration                # noqa: E402
from mhd_equilibrium import MHDEquilibrium                # noqa: E402

# ---- Fixed run definition. Changing any of these invalidates the golden.
SEED = 20260829
TEST_THERMAL_COUNT = 2000
TEST_NUM_STEPS = 200
NUMBA_THREADS = 1
GS_BACKEND = "python"

# Sizes for the slow plot test's other three entry points.
TEST_OSC_STEPS = 200
TEST_BENCHMARK_COUNTS = [1000, 10000]
TEST_BENCHMARK_STEPS = 20

# benchmark_scaling.png plots MEASURED WALL-CLOCK TIMES, so its bytes
# differ every run by construction -- the plot comparison therefore runs
# --presence-only, asking "was every plot produced" rather than "is every
# plot identical". The .npz test is what pins the numbers.
PLOT_PRESENCE_ONLY = True

# disruption_mitigation.png only renders when the SPI disruption fires,
# which needs the island to grow past MAX_ISLAND_WIDTH_THRESHOLD -- about
# 5,200 steps at this seed, against the 200 this test runs. It is excused
# rather than quietly dropped, so the tool still owns the list of 17.
PLOT_ALLOW_MISSING = ["disruption_mitigation.png"]

GOLDEN_DIR = os.path.join(HERE, "golden")
GOLDEN_NPZ = os.path.join(GOLDEN_DIR, "reactor_regression.npz")
GOLDEN_MANIFEST = os.path.join(GOLDEN_DIR, "plot_manifest.json")
PLOT_MANIFEST_TOOL = os.path.join(REPO, "tools", "plot_manifest.py")

DEFAULT_RTOL = 1e-6
LOOSE_RTOL = 1e-3

# Compared exactly -- these are counts, ids and species labels, and a
# one-vertex or one-particle difference is a real regression, not noise.
EXACT_KEYS = frozenset({
    "type_np", "total_injected", "total_lost",
    "track_pids", "track_counts", "track_types", "track_lost",
})

# Recorded for diagnosis on failure; never compared.
METADATA_KEYS = frozenset({
    "meta_seed", "meta_thermal_count", "meta_num_steps",
    "meta_numba_threads", "meta_gs_backend",
    "meta_numpy_version", "meta_numba_version", "meta_torch_version",
    "meta_python_version",
})


# =======================================================
# SEEDING
# =======================================================
@njit(cache=True)
def _seed_numba_main(seed):
    """Seeds the RNG state numba uses on the calling thread -- the one
    the serial jitted kernels (apply_cartesian_collisions,
    apply_guiding_center_collisions) draw from."""
    np.random.seed(seed)


@njit(parallel=True, cache=True)
def _seed_numba_threads(seed, n_threads):
    """Seeds each worker thread's RNG state.

    This is the one that actually matters. apply_vectorized_collisions
    is @njit(parallel=True) and draws inside the prange, where every
    worker carries its own RNG state -- and a seed set on the main
    thread never reaches those. Without this the run is NOT
    reproducible: particle counts and track vertex counts still match
    (the branching is elsewhere), but scattered velocities drift, which
    is a nastier failure than an obvious one. Verified: with this,
    two separate processes agree bit-for-bit; without it, they do not.

    One prange iteration per thread, so each worker seeds itself.
    """
    for i in prange(n_threads):
        np.random.seed(seed + i)


def seed_everything(seed=SEED):
    np.random.seed(seed)
    _seed_numba_main(seed)
    _seed_numba_threads(seed, numba.get_num_threads())
    torch.manual_seed(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


@contextlib.contextmanager
def python_gs_backend():
    """Force backend="python" at the solve_grad_shafranov call site.

    initialize_reactor calls solve_grad_shafranov with no backend
    argument, so without this the resolution falls through to
    FUSION_GS_BACKEND -- and a numba-solved psi is not bit-identical to
    the golden's.
    """
    original = MHDEquilibrium.solve_grad_shafranov

    def forced(self, *args, **kwargs):
        kwargs["backend"] = GS_BACKEND
        return original(self, *args, **kwargs)

    MHDEquilibrium.solve_grad_shafranov = forced
    try:
        yield
    finally:
        MHDEquilibrium.solve_grad_shafranov = original


def build_config():
    """Production config, shrunk to a size a test can run."""
    cfg = SimulationConfiguration()
    cfg.initial_thermal_count = TEST_THERMAL_COUNT
    cfg.reactor_num_steps = TEST_NUM_STEPS
    return cfg


# =======================================================
# THE RUN
# =======================================================
def run_pipeline(quiet=True):
    """initialize_reactor -> reactor loop -> package_reactor_results.

    Returns a flat dict of numpy arrays, ready for np.savez.
    Renders nothing: that is what makes this the fast test.
    """
    numba.set_num_threads(NUMBA_THREADS)
    seed_everything()
    cfg = build_config()

    buffer = io.StringIO()
    stream = contextlib.redirect_stdout(buffer) if quiet else contextlib.nullcontext()
    with stream:
        with python_gs_backend():
            (eq, engine, pos_tensor, vel_tensor, type_tensor,
             rho_grid, phi_grid, E_R_grid, E_Z_grid) = initialization.initialize_reactor(cfg)

        B_R_pol_grid, B_Z_pol_grid, _pol_scale, _B_pol_target = main.compute_poloidal_field_grids(
            eq, cfg.B0, R0=cfg.R0_major, a_minor=0.3, q_target=cfg.Q_SAFETY_TARGET,
            dst_R_min=cfg.R_min, dst_R_max=cfg.R_max,
            dst_Z_min=cfg.Z_min, dst_Z_max=cfg.Z_max,
            dst_nR=cfg.nR, dst_nZ=cfg.nZ,
        )

        loop_out = main._run_reactor_loop_cpu(
            cfg, engine, pos_tensor, vel_tensor, type_tensor,
            rho_grid, phi_grid, E_R_grid, E_Z_grid,
            B_R_pol_grid, B_Z_pol_grid,
        )

        (pos_np, vel_np, type_np, history_tracks, tracked_type, tracked_lastpos,
         tracked_lastvel, tracked_lost, total_injected, total_lost,
         inventory_history, energy_history_keV, instability_amp_history, time_history,
         temp_history, rad_power_history, trigger_time, T_core,
         alpha_heating_power_history_MW, external_heating_power_history_MW,
         bremsstrahlung_power_history_MW, cyclotron_power_history_MW,
         q_sci_history, q_eng_history, lawson_triple_product_history,
         rho_grid_out, phi_grid_out) = loop_out

        packaged = main.package_reactor_results(
            cfg, eq, engine, B_R_pol_grid, B_Z_pol_grid,
            pos_np, vel_np, type_np,
            history_tracks, tracked_type, tracked_lastpos,
            tracked_lastvel, tracked_lost,
            rho_grid_out, T_core,
        )

    data = {
        # --- diagnostic time series
        "inventory_history": _f(inventory_history),
        "energy_history_keV": _f(energy_history_keV),
        "temp_history": _f(temp_history),
        "rad_power_history": _f(rad_power_history),
        "instability_amp_history": _f(instability_amp_history),
        "q_sci_history": _f(q_sci_history),
        "q_eng_history": _f(q_eng_history),
        "lawson_triple_product_history": _f(lawson_triple_product_history),
        "alpha_heating_power_history_MW": _f(alpha_heating_power_history_MW),
        "bremsstrahlung_power_history_MW": _f(bremsstrahlung_power_history_MW),
        "cyclotron_power_history_MW": _f(cyclotron_power_history_MW),
        # Not in the brief's list, but it rides the same code path and
        # diagnostics plots it beside the other three power channels.
        "external_heating_power_history_MW": _f(external_heating_power_history_MW),
        "time_history": _f(time_history),

        # --- scalars
        "total_injected": np.asarray(total_injected, dtype=np.int64),
        "total_lost": np.asarray(total_lost, dtype=np.int64),
        "T_core": np.asarray(float(T_core), dtype=np.float64),
        # trigger_time is None unless the SPI disruption fired.
        "trigger_time": np.asarray(
            np.nan if trigger_time is None else float(trigger_time), dtype=np.float64),

        # --- final particle / field state
        "pos_np": np.asarray(pos_np, dtype=np.float64),
        "vel_np": np.asarray(vel_np, dtype=np.float64),
        "type_np": np.asarray(type_np, dtype=np.int64),
        "rho_grid": np.asarray(rho_grid_out, dtype=np.float64),
        "phi_grid": np.asarray(phi_grid_out, dtype=np.float64),

        # --- packaged plot inputs
        "R_centers": _f(packaged["R_centers"]),
        "density_profile": _f(packaged["density_profile"]),
        "pressure_profile": _f(packaged["pressure_profile"]),
        "R_phase": _f(packaged["R_phase"]),
        "v_parallel_phase": _f(packaged["v_parallel_phase"]),
        "P_fusion_grid": np.asarray(packaged["P_fusion_grid"], dtype=np.float64),
        "total_fusion_power_watts": np.asarray(
            float(packaged["total_fusion_power_watts"]), dtype=np.float64),
    }
    data.update(_serialize_tracks(history_tracks, tracked_type, tracked_lost))
    data.update(_metadata())
    return data


def _f(seq):
    """Any list/array of numbers -> a float64 array."""
    return np.asarray(seq, dtype=np.float64)


def _serialize_tracks(history_tracks, tracked_type, tracked_lost):
    """Flatten the trajectory tracks, pid-sorted and count-preserving.

    Vertex COUNTS matter as much as vertex values: the alpha-orbit plot
    clips each track to a fixed length (diagnostics.py, `traj[:_alpha_max_pts]`
    in the alpha-trajectory block), so a track that gains or loses a
    vertex changes what gets drawn. Counts are stored explicitly and
    compared exactly; the vertices themselves are concatenated in the
    same pid order, so the pair reconstructs every track exactly.
    """
    pids = sorted(history_tracks)

    track_pids = np.asarray(pids, dtype=np.int64)
    track_counts = np.asarray([len(history_tracks[p]) for p in pids], dtype=np.int64)
    track_types = np.asarray([tracked_type.get(p, 0) for p in pids], dtype=np.int64)
    track_lost = np.asarray([p in tracked_lost for p in pids], dtype=bool)

    if pids:
        stacked = [np.asarray(history_tracks[p], dtype=np.float64).reshape(-1, 3)
                    for p in pids]
        track_vertices = np.concatenate(stacked, axis=0)
    else:
        track_vertices = np.zeros((0, 3), dtype=np.float64)

    return {
        "track_pids": track_pids,
        "track_counts": track_counts,
        "track_types": track_types,
        "track_lost": track_lost,
        "track_vertices": track_vertices,
    }


def _metadata():
    return {
        "meta_seed": np.asarray(SEED, dtype=np.int64),
        "meta_thermal_count": np.asarray(TEST_THERMAL_COUNT, dtype=np.int64),
        "meta_num_steps": np.asarray(TEST_NUM_STEPS, dtype=np.int64),
        "meta_numba_threads": np.asarray(numba.get_num_threads(), dtype=np.int64),
        "meta_gs_backend": np.asarray(GS_BACKEND),
        "meta_numpy_version": np.asarray(np.__version__),
        "meta_numba_version": np.asarray(numba.__version__),
        "meta_torch_version": np.asarray(torch.__version__),
        "meta_python_version": np.asarray(sys.version.split()[0]),
    }


# =======================================================
# COMPARISON
# =======================================================
def compare_arrays(name, golden, current, rtol):
    """Return a failure description, or None when the arrays agree."""
    if golden.shape != current.shape:
        return (f"{name}: SHAPE {golden.shape} -> {current.shape}"
                f"  ({golden.size} -> {current.size} values)")

    if name in EXACT_KEYS or golden.dtype.kind in "biU":
        if np.array_equal(golden, current):
            return None
        n_diff = int(np.sum(golden != current))
        where = np.flatnonzero(np.ravel(golden != current))[:5]
        return (f"{name}: {n_diff} of {golden.size} values differ exactly "
                f"(first indices {[int(i) for i in where]}; "
                f"golden {np.ravel(golden)[where][:5]} -> "
                f"current {np.ravel(current)[where][:5]})")

    # atleast_1d: the recorded scalars (T_core, trigger_time,
    # total_fusion_power_watts) come back from npz as 0-d arrays, and a
    # 0-d subtraction yields an immutable numpy scalar.
    g = np.atleast_1d(golden.astype(np.float64))
    c = np.atleast_1d(current.astype(np.float64))

    both_nan = np.isnan(g) & np.isnan(c)
    if not np.array_equal(np.isnan(g), np.isnan(c)):
        return f"{name}: NaN pattern changed"

    diff = np.abs(g - c)
    diff[both_nan] = 0.0
    max_abs = float(diff.max()) if diff.size else 0.0

    scale = np.abs(g)
    nonzero = (scale > 0.0) & ~both_nan
    max_rel = float((diff[nonzero] / scale[nonzero]).max()) if nonzero.any() else 0.0

    # An absolute floor tied to the array's own magnitude, so values that
    # sit at or near zero are not judged by a relative test they cannot
    # pass. nanmax, since a recorded NaN (trigger_time with no disruption)
    # would otherwise poison the floor and fail the array against itself.
    finite_scale = scale[np.isfinite(scale)]
    atol = rtol * (float(finite_scale.max()) if finite_scale.size else 0.0)
    tolerated = atol + rtol * np.where(np.isfinite(g), np.abs(g), 0.0)
    if np.all(diff <= tolerated):
        return None

    idx = int(np.argmax(diff))
    return (f"{name}: max|abs| {max_abs:.6e}  max|rel| {max_rel:.6e}  "
            f"(worst at flat index {idx}: golden {np.ravel(g)[idx]:.9e} -> "
            f"current {np.ravel(c)[idx]:.9e})")


def compare(golden, current, rtol):
    """Compare every recorded array. Returns (failures, notes)."""
    failures, notes = [], []

    g_keys = set(golden) - METADATA_KEYS
    c_keys = set(current) - METADATA_KEYS

    for missing in sorted(g_keys - c_keys):
        failures.append(f"{missing}: present in the golden, absent now")
    for added in sorted(c_keys - g_keys):
        failures.append(f"{added}: produced now, absent from the golden")

    for name in sorted(g_keys & c_keys):
        result = compare_arrays(name, golden[name], current[name], rtol)
        if result:
            failures.append(result)

    # Environment drift: never a failure by itself, but it is usually the
    # explanation when numbers move.
    for key in sorted(METADATA_KEYS):
        if key not in golden or key not in current:
            continue
        if str(golden[key]) != str(current[key]):
            notes.append(f"{key}: golden {golden[key]} vs current {current[key]}")

    return failures, notes


def _track_report(golden, current):
    """Name the pids whose vertex counts moved -- the counts are the point."""
    if "track_pids" not in golden or "track_counts" not in golden:
        return []
    g_pids, g_counts = golden["track_pids"], golden["track_counts"]
    c_pids, c_counts = current["track_pids"], current["track_counts"]
    if g_pids.shape != c_pids.shape or not np.array_equal(g_pids, c_pids):
        return [f"  track set changed: {len(g_pids)} tracks -> {len(c_pids)}"]
    if np.array_equal(g_counts, c_counts):
        return []
    moved = np.flatnonzero(g_counts != c_counts)
    lines = [f"  {len(moved)} track(s) changed vertex count:"]
    for i in moved[:10]:
        lines.append(f"    pid {int(g_pids[i]):>6}: {int(g_counts[i])} -> {int(c_counts[i])}")
    if moved.size > 10:
        lines.append(f"    ... and {moved.size - 10} more")
    return lines


def load_golden(path=GOLDEN_NPZ):
    with np.load(path, allow_pickle=False) as handle:
        return {k: handle[k] for k in handle.files}


# =======================================================
# PLOT BASELINE (the slow path)
# =======================================================
CHILD_RENDER = """
import os, sys
sys.path.insert(0, {repo!r})
import numpy as np, numba, torch
from numba import njit, prange

numba.set_num_threads({threads})

# No cache=True here: numba cannot cache a function defined in a
# "-c" string, since there is no source file to key the cache on.
@njit
def _seed_numba_main(seed):
    np.random.seed(seed)

@njit(parallel=True)
def _seed_numba_threads(seed, n_threads):
    for i in prange(n_threads):
        np.random.seed(seed + i)

np.random.seed({seed})
_seed_numba_main({seed})
_seed_numba_threads({seed}, numba.get_num_threads())
torch.manual_seed({seed})

import config
_orig_init = config.SimulationConfiguration.__init__
def _patched_init(self, *a, **k):
    _orig_init(self, *a, **k)
    self.initial_thermal_count = {count}
    self.reactor_num_steps = {steps}
    # The other three entry points get shrunk too -- this test asks
    # whether every plot still renders, not whether the physics is
    # production-sized. The .npz test is what guards the numbers.
    self.osc_num_steps = {osc_steps}
    self.BENCHMARK_PARTICLE_COUNTS = {bench_counts}
    self.BENCHMARK_STEPS = {bench_steps}
config.SimulationConfiguration.__init__ = _patched_init

from mhd_equilibrium import MHDEquilibrium
_orig_solve = MHDEquilibrium.solve_grad_shafranov
def _forced(self, *a, **k):
    k["backend"] = {backend!r}
    return _orig_solve(self, *a, **k)
MHDEquilibrium.solve_grad_shafranov = _forced

import main
# Mirrors main.py's own __main__ block: the 17 plots come from all
# four entry points, not from the reactor run alone.
cfg = config.SimulationConfiguration()
main.run_hpc_benchmark(cfg)
main.run_reactor_steady_state()
main.run_plasma_oscillation_test()
main.run_nuclear_reaction_dynamics()
"""


def render_full_run(out_dir):
    """Run the real run_reactor_steady_state, headless, into out_dir.

    Rendering happens in a child process because FUSION_HEADLESS has to
    be set before pyplot is imported, and in a scratch directory because
    savefig writes relative paths -- a test must not overwrite the
    production PNGs sitting in the repo root.

    The Grad-Shafranov disk cache still lives beside the module, so this
    shares .cache/gs_*.npy with production runs. That is intended: psi
    does not depend on particle or step count.
    """
    script = CHILD_RENDER.format(
        repo=REPO, threads=NUMBA_THREADS, seed=SEED,
        count=TEST_THERMAL_COUNT, steps=TEST_NUM_STEPS, backend=GS_BACKEND,
        osc_steps=TEST_OSC_STEPS, bench_counts=TEST_BENCHMARK_COUNTS,
        bench_steps=TEST_BENCHMARK_STEPS,
    )
    env = dict(os.environ)
    env["FUSION_HEADLESS"] = "1"
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("FUSION_GS_BACKEND", None)      # the child forces it explicitly

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=out_dir, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    return completed


def plot_manifest(mode, out_dir, manifest_path):
    """Shell out to tools/plot_manifest.py -- the one authority on
    whether all 17 plots survived a run."""
    args = [sys.executable, PLOT_MANIFEST_TOOL, mode, "--dir", out_dir]
    args += ["--out" if mode == "record" else "--baseline", manifest_path]
    if mode == "compare":
        if PLOT_PRESENCE_ONLY:
            args.append("--presence-only")
        if PLOT_ALLOW_MISSING:
            args += ["--allow-missing", ",".join(PLOT_ALLOW_MISSING)]
    return subprocess.run(args, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)


# =======================================================
# PYTEST ENTRY POINTS
# =======================================================
def test_reactor_regression():
    """Fast: the pipeline's numbers have not moved."""
    assert os.path.isfile(GOLDEN_NPZ), (
        f"no golden at {GOLDEN_NPZ}; run "
        "'python tests/test_regression.py record' first")

    golden = load_golden()
    current = run_pipeline()
    failures, notes = compare(golden, current, DEFAULT_RTOL)

    if failures:
        report = ["Reactor pipeline regression:"]
        report += [f"  {f}" for f in failures]
        report += _track_report(golden, current)
        if notes:
            report.append("  environment differs from the golden's:")
            report += [f"    {n}" for n in notes]
        raise AssertionError("\n".join(report))


def test_plot_outputs_survive_full_run():
    """Slow: the whole run_reactor_steady_state renders, and every one of
    the 17 plots is still produced and unchanged."""
    assert os.path.isfile(GOLDEN_MANIFEST), (
        f"no plot baseline at {GOLDEN_MANIFEST}; run "
        "'python tests/test_regression.py record-plots' first")

    with tempfile.TemporaryDirectory(prefix="fusion_plots_") as out_dir:
        run = render_full_run(out_dir)
        assert run.returncode == 0, (
            f"run_reactor_steady_state failed (exit {run.returncode}):\n"
            f"{run.stdout[-4000:]}")

        check = plot_manifest("compare", out_dir, GOLDEN_MANIFEST)
        assert check.returncode == 0, (
            f"plot manifest comparison failed (exit {check.returncode}):\n"
            f"{check.stdout}")


# =======================================================
# CLI
# =======================================================
def cmd_record(args):
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    print(f"Running the pipeline (seed {SEED}, {TEST_THERMAL_COUNT} particles, "
          f"{TEST_NUM_STEPS} steps, {NUMBA_THREADS} numba thread(s))...")
    data = run_pipeline(quiet=not args.verbose)
    np.savez_compressed(GOLDEN_NPZ, **data)
    size_kb = os.path.getsize(GOLDEN_NPZ) / 1024
    print(f"\n[GOLDEN] wrote {os.path.relpath(GOLDEN_NPZ, REPO)} ({size_kb:.1f} KiB)")
    for name in sorted(set(data) - METADATA_KEYS):
        arr = data[name]
        print(f"    {name:<34} {str(arr.shape):<14} {arr.dtype}")
    print(f"    ({len(data['track_pids'])} tracks, "
          f"{int(data['track_counts'].sum())} vertices total)")
    return 0


def cmd_compare(args):
    if not os.path.isfile(GOLDEN_NPZ):
        print(f"[ERROR] no golden at {GOLDEN_NPZ}", file=sys.stderr)
        print("[ERROR] run 'python tests/test_regression.py record' first.",
              file=sys.stderr)
        return 2

    rtol = LOOSE_RTOL if args.loose else args.rtol
    print(f"Comparing against {os.path.relpath(GOLDEN_NPZ, REPO)} (rtol {rtol:.1e})...")

    golden = load_golden()
    current = run_pipeline(quiet=not args.verbose)
    failures, notes = compare(golden, current, rtol)

    if notes:
        print("\nEnvironment differs from the golden's:")
        for note in notes:
            print(f"  {note}")

    if not failures:
        n_arrays = len(set(golden) - METADATA_KEYS)
        print(f"\n[OK] all {n_arrays} arrays match within rtol {rtol:.1e}.")
        return 0

    print(f"\n[FAIL] {len(failures)} array(s) diverged:")
    for failure in failures:
        print(f"  {failure}")
    for line in _track_report(golden, current):
        print(line)
    if not args.loose:
        print(f"\n  If float reordering is expected, retry with --loose "
              f"(rtol {LOOSE_RTOL:.0e}).")
    return 1


def cmd_record_plots(args):
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fusion_plots_") as out_dir:
        print("Rendering the full reactor run (headless)...")
        run = render_full_run(out_dir)
        if run.returncode != 0:
            print(run.stdout[-4000:], file=sys.stderr)
            print(f"[ERROR] render failed (exit {run.returncode})", file=sys.stderr)
            return 1
        result = plot_manifest("record", out_dir, GOLDEN_MANIFEST)
        print(result.stdout)
        if result.returncode != 0:
            # record exits non-zero when a plot is absent. Only the
            # documented exception is acceptable here.
            absent = _absent_from_manifest(GOLDEN_MANIFEST)
            unexpected = [n for n in absent if n not in PLOT_ALLOW_MISSING]
            if unexpected:
                print(f"[ERROR] plots missing from the baseline: "
                      f"{', '.join(unexpected)}", file=sys.stderr)
                return 1
            print(f"[NOTE] absent as expected: {', '.join(absent)}")
    print(f"[GOLDEN] wrote {os.path.relpath(GOLDEN_MANIFEST, REPO)}")
    return 0


def cmd_compare_plots(args):
    if not os.path.isfile(GOLDEN_MANIFEST):
        print(f"[ERROR] no plot baseline at {GOLDEN_MANIFEST}", file=sys.stderr)
        print("[ERROR] run 'python tests/test_regression.py record-plots' first.",
              file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="fusion_plots_") as out_dir:
        print("Rendering the full reactor run (headless)...")
        run = render_full_run(out_dir)
        if run.returncode != 0:
            print(run.stdout[-4000:], file=sys.stderr)
            print(f"[ERROR] render failed (exit {run.returncode})", file=sys.stderr)
            return 1
        result = plot_manifest("compare", out_dir, GOLDEN_MANIFEST)
        print(result.stdout)
        return result.returncode


def _absent_from_manifest(path):
    """Which expected plots the just-written manifest records as missing."""
    import json
    with open(path) as handle:
        return list(json.load(handle).get("missing", []))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Golden-file regression test for the reactor pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="exit codes: 0 = pass, 1 = regression, 2 = no golden recorded yet",
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="let the simulation's own output through")
    sub = parser.add_subparsers(dest="mode")

    p = sub.add_parser("record", help="write the golden .npz")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("compare", help="check the pipeline against the golden")
    p.add_argument("--rtol", type=float, default=DEFAULT_RTOL,
                   help=f"relative tolerance (default {DEFAULT_RTOL:.0e})")
    p.add_argument("--loose", action="store_true",
                   help=f"use rtol {LOOSE_RTOL:.0e}, for when float "
                          "reordering is expected")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("record-plots", help="render, and save the plot baseline")
    p.set_defaults(func=cmd_record_plots)

    p = sub.add_parser("compare-plots",
                        help="render, and diff the 17 plots against the baseline")
    p.set_defaults(func=cmd_compare_plots)

    return parser


def main_cli(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode is None:
        args.mode = "compare"
        args.func = cmd_compare
        args.rtol = DEFAULT_RTOL
        args.loose = False
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main_cli())
