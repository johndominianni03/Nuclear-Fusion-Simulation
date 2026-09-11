import numpy as np
import time
import numba
import sys
import torch
from numba import njit, prange

from physics_engine import (
    gather_electric_field,
    gather_electric_field_scalar,   # allocation-free scalar twin, used by the prange gather
    compute_electrostatic_energy,
    compute_radiative_cooling_power,
    compute_dt_cross_section,
    compute_volumetric_fusion_power,
    compute_radiation_losses,
    compute_radiation_losses_grid,          # vectorized radiation-loss grid reduction
    evaluate_q_factors,
    compute_cic_charge_density,
    HPCPhysicsAccelerator,
    vectorized_boris_push_numba_fallback,
    compute_cic_charge_density_torch,       # large-N (>=100K) GPU-resident kernels
    vectorized_gather_and_B_torch,
    check_confinement_torch,
    apply_vectorized_collisions_torch,
    compute_alpha_heating_power_torch,
    _vectorized_boris_push_metal_impl,      # eager push; shapes churn every step, so the static-compiled path would recompile constantly
    _vectorized_boris_push_metal_dynamic,   # dynamic-shape compile, handles that churn; None if the compile failed at import
    check_confinement_flux,                 # psi-surface confinement, replaces the circular boundary
    interpolate_psi,                        # bilinear psi/B-grid lookup for the numba gather
    compute_poloidal_field_grids,           # psi-derived poloidal B (real grad-B / mirror force)
    boris_push_substeps_torch,              # alpha sub-stepping (GPU)
    vectorized_boris_push_numba_substeps    # alpha sub-stepping (CPU)
)
from config import SimulationConfiguration
import initialization
import diagnostics


# =======================================================
# PER-STAGE WALL-CLOCK INSTRUMENTATION (cfg.PROFILE)
# =======================================================
# Purely additive. Every hook is a clock read plus a float add into a dict:
# no array is touched, no RNG is drawn, no branch that the physics can see
# is taken. With cfg.PROFILE False, mark() returns None and add() returns
# immediately, so a profiled run and an unprofiled one produce bit-identical
# results -- which tests/test_regression.py checks in both modes.
#
# On the GPU path the profiler is constructed with a sync callable. MPS and
# CUDA both queue work asynchronously, so a bare perf_counter() around a
# GPU op measures how fast the CPU enqueued it, not how long it ran. The
# sync is called immediately before every clock read, on both ends of the
# interval, so the elapsed time brackets work that has actually finished.


class StageProfiler:
    """Accumulates wall time per named stage."""

    def __init__(self, enabled, sync=None, label=""):
        self.enabled = bool(enabled)
        self.label = label
        # sync is only invoked when profiling: it must never slow, or
        # otherwise perturb, an unprofiled run.
        self._sync = sync if self.enabled else None
        self.totals = {}
        self.counts = {}
        self._order = []

    def mark(self):
        """Timestamp for the start of a stage (None when disabled)."""
        if not self.enabled:
            return None
        if self._sync is not None:
            self._sync()
        return time.perf_counter()

    def add(self, name, t0):
        """Charge the elapsed time since t0 to `name`."""
        if not self.enabled or t0 is None:
            return
        if self._sync is not None:
            self._sync()
        elapsed = time.perf_counter() - t0
        if name not in self.totals:
            self.totals[name] = 0.0
            self.counts[name] = 0
            self._order.append(name)
        self.totals[name] += elapsed
        self.counts[name] += 1

    def total(self):
        return sum(self.totals.values())

    def report(self, wall_total=None, title=None, min_percent=0.0):
        """Print the stage table, sorted by share of wall time."""
        if not self.enabled or not self.totals:
            return
        denom = wall_total if wall_total is not None else self.total()
        if denom <= 0:
            denom = self.total() or 1.0

        heading = title or f"STAGE TIMING{(' -- ' + self.label) if self.label else ''}"
        print("")
        print("=" * 78)
        print(f" {heading}")
        print("=" * 78)
        print(f" {'stage':<34}{'seconds':>11}{'% wall':>9}{'calls':>9}{'ms/call':>12}")
        print(" " + "-" * 76)

        for name in sorted(self.totals, key=lambda k: self.totals[k], reverse=True):
            seconds = self.totals[name]
            percent = 100.0 * seconds / denom
            if percent < min_percent:
                continue
            calls = self.counts[name]
            per_call_ms = 1000.0 * seconds / calls if calls else 0.0
            # Truncate rather than let a long stage name shove the numeric
            # columns out of alignment.
            label = name if len(name) <= 33 else name[:32] + "\u2026"
            print(f" {label:<34}{seconds:>11.3f}{percent:>8.1f}%{calls:>9,}{per_call_ms:>12.4f}")

        print(" " + "-" * 76)
        measured = self.total()
        print(f" {'measured':<34}{measured:>11.3f}{100.0 * measured / denom:>8.1f}%")
        if wall_total is not None:
            unmeasured = wall_total - measured
            print(f" {'unmeasured (loop overhead, etc.)':<34}"
                  f"{unmeasured:>11.3f}{100.0 * unmeasured / denom:>8.1f}%")
            print(f" {'WALL TOTAL':<34}{wall_total:>11.3f}{100.0:>8.1f}%")
        print("=" * 78)


def _profile_header(cfg, path_label, n_particles):
    """One-line provenance for a profiled run.

    numba's thread count is printed and NOT pinned: apply_vectorized_collisions
    draws RNG inside a prange, so the thread count changes the answer -- but
    forcing single-threaded here would misrepresent production performance.
    The report says which count produced it instead.
    """
    print("")
    print("=" * 78)
    print(f" PROFILING RUN -- {path_label}")
    print("=" * 78)
    print(f"   particles (initial) : {n_particles:,}")
    print(f"   steps               : {cfg.reactor_num_steps:,}")
    print(f"   numba threads       : {numba.get_num_threads()} "
          f"(of {numba.config.NUMBA_NUM_THREADS} available)")
    print(f"   device              : {cfg.HPC_DEVICE}")
    print("=" * 78)


def _peak_rss_bytes():
    """Peak resident set size of this process, or None if unavailable."""
    try:
        import resource
    except ImportError:                      # not POSIX
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports kilobytes.
    return peak if sys.platform == "darwin" else peak * 1024


def _report_gs_provenance(eq):
    """Say whether psi came from the disk cache or a fresh solve.

    A cold miss and a warm hit differ by seconds; without this line the
    initialize_reactor row in the table is easy to misread as loop-adjacent
    cost when it is really just the Grad-Shafranov solve.
    """
    source = getattr(eq, "last_solve_source", None)
    backend = getattr(eq, "last_solve_backend", None)
    if source is None:
        print("[PROFILE] Grad-Shafranov: provenance unavailable")
        return
    if source == "cache-hit":
        print(f"[PROFILE] Grad-Shafranov: CACHE HIT (backend '{backend}') -- "
              "no SOR iteration ran; initialize_reactor excludes the solve cost.")
    else:
        print(f"[PROFILE] Grad-Shafranov: COLD SOLVE on the '{backend}' backend -- "
              "initialize_reactor below INCLUDES the full solve.")


def _log_device_memory(device, step, final=False):
    """Current GPU allocation, sampled every 500 steps on the GPU path."""
    tag = "final" if final else f"step {step:,}"
    try:
        if device.type == "mps":
            allocated = torch.mps.current_allocated_memory()
            driver = getattr(torch.mps, "driver_allocated_memory", None)
            extra = f", driver {driver() / (1024 ** 3):.3f} GiB" if driver else ""
            print(f"[PROFILE][MPS] {tag}: allocated "
                  f"{allocated / (1024 ** 3):.3f} GiB{extra}")
        elif device.type == "cuda":
            print(f"[PROFILE][CUDA] {tag}: allocated "
                  f"{torch.cuda.memory_allocated() / (1024 ** 3):.3f} GiB, "
                  f"peak {torch.cuda.max_memory_allocated() / (1024 ** 3):.3f} GiB")
        else:
            print(f"[PROFILE] {tag}: device {device.type} exposes no allocator query")
    except Exception as exc:                 # never let instrumentation break a run
        print(f"[PROFILE] {tag}: device memory query failed ({exc})")


def _report_peak_rss(prefix="[PROFILE]"):
    peak = _peak_rss_bytes()
    if peak is None:
        print(f"{prefix} peak RSS: unavailable on this platform")
        return
    print(f"{prefix} peak RSS: {peak / (1024 ** 3):.3f} GiB")


def _report_sor_histogram(iter_counts, max_iter=500):
    """Histogram of Poisson SOR iterations per step.

    The question this answers: does the warm start let the solve exit in a
    handful of sweeps, or is it pinned against the iteration cap?
    """
    if not iter_counts:
        return
    counts = np.asarray(iter_counts, dtype=np.int64)
    print("")
    print("=" * 78)
    print(" POISSON SOR ITERATIONS PER STEP")
    print("=" * 78)
    print(f"   solves      : {counts.size:,}")
    print(f"   min / median / max : {counts.min()} / "
          f"{int(np.median(counts))} / {counts.max()}")
    print(f"   mean        : {counts.mean():.2f}")
    at_cap = int(np.sum(counts >= max_iter))
    print(f"   at the {max_iter}-iteration cap : {at_cap:,} "
          f"({100.0 * at_cap / counts.size:.1f}%)")
    print("")

    edges = [1, 2, 3, 4, 5, 6, 8, 11, 16, 26, 51, 101, 201, max_iter, max_iter + 1]
    widest = 0
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        n = int(np.sum((counts >= lo) & (counts < hi)))
        if n == 0:
            continue
        label = f"{lo}" if hi == lo + 1 else f"{lo}-{hi - 1}"
        rows.append((label, n))
        widest = max(widest, n)
    for label, n in rows:
        bar = "#" * max(1, int(round(46.0 * n / widest)))
        print(f"   {label:>9} iters | {n:>7,}  {bar}")
    print("=" * 78)


# =======================================================
# HYBRID SOLVER NUMBA KERNELS (CPU -> GPU BRIDGE)
# =======================================================
@njit(parallel=True, fastmath=True)
def vectorized_gather_and_B_into(pos_arr, E_out, B_out,
                                 E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid,
                                 R_min, R_max, Z_min, Z_max, nR, nZ, B0, R0, t, b_perturb, m_mode, n_mode, gamma):
    # Out-parameter form: E_out and B_out are (>=N, 3) float32 arrays supplied by
    # the caller, and every row this touches is written in full, so they need not
    # be zeroed first. Nothing is returned. Allocating the pair here cost 24 MB of
    # allocate-and-free per step at 1M particles, ~10,000 times over a full run;
    # _run_reactor_loop_cpu now owns one pair of buffers and passes [:n] slices.
    # vectorized_gather_and_B (just below) is the allocating wrapper for the
    # callers that run once per run rather than once per step.
    #
    # NOTE: N comes from pos_arr, and numba does not bounds-check these stores --
    # E_out/B_out must have at least pos_arr.shape[0] rows.
    #
    # r, phi and its cos/sin are computed ONCE per particle and shared between the
    # E gather and the B construction. The previous version called
    # gather_electric_field, which redid the sqrt and an arctan2 of its own and
    # returned a freshly allocated 3-element float64 array per particle -- one
    # heap allocation per particle per step, and an array return that stopped the
    # prange body from vectorizing. gather_electric_field_scalar returns a tuple
    # instead and this writes straight into E_out[i, 0..2].
    #
    # All arithmetic is float32, matching E_out/B_out, so no per-element downcast
    # happens on store. B0/R0/t and the tearing-mode parameters are cast once here
    # rather than promoting the whole B expression to float64 per particle.
    #
    # Call paths: the reactor loops pass live E grids and b_perturb > 0;
    # package_reactor_results passes zero E grids and b_perturb = 0.0 for the
    # phase-space projection. Both are handled by the same branches as before --
    # the b_perturb > 0.0 guard is unchanged, and zero E grids simply gather zero.
    N = pos_arr.shape[0]

    B0_f = np.float32(B0)
    R0_f = np.float32(R0)
    zero_f = np.float32(0.0)
    amplitude = np.float32(b_perturb * np.exp(gamma * t))
    m_f = np.float32(m_mode)
    n_f = np.float32(n_mode)

    for i in prange(N):
        px = pos_arr[i, 0]
        py = pos_arr[i, 1]
        pz = pos_arr[i, 2]

        r = np.float32(np.sqrt(px * px + py * py))
        phi = np.float32(np.arctan2(py, px))
        cos_phi = np.float32(np.cos(phi))
        sin_phi = np.float32(np.sin(phi))
        z = np.float32(pz)

        ex, ey, ez = gather_electric_field_scalar(
            px, py, pz, E_R_grid, E_Z_grid,
            R_min, R_max, Z_min, Z_max, nR, nZ,
            cos_phi, sin_phi
        )
        E_out[i, 0] = ex
        E_out[i, 1] = ey
        E_out[i, 2] = ez

        # 1. Toroidal field B_phi = B0 * R0 / R. R0 must match the value used by the
        # confinement centre, tearing mode and poloidal field.
        b_mag_tor = B0_f * (R0_f / r) if r > zero_f else B0_f
        Bx = -b_mag_tor * sin_phi
        By = b_mag_tor * cos_phi
        Bz = zero_f

        # 2. Poloidal field, interpolated from the psi-derived B_R / B_Z grids
        # (B_R = -(1/R) dpsi/dZ, B_Z = (1/R) dpsi/dR). The old linear-in-position
        # stand-in had no gradient structure and so no magnetic mirror; this follows the
        # same flux surfaces the confinement check uses as the loss boundary.
        if r > zero_f:
            B_R_pol = np.float32(interpolate_psi(r, z, B_R_pol_grid, R_min, R_max, Z_min, Z_max, nR, nZ))
            B_Z_pol = np.float32(interpolate_psi(r, z, B_Z_pol_grid, R_min, R_max, Z_min, Z_max, nR, nZ))

            Bx += B_R_pol * cos_phi
            By += B_R_pol * sin_phi
            Bz += B_Z_pol

        # 3. Tearing mode perturbation
        if b_perturb > 0.0:
            theta = np.float32(np.arctan2(z, r - R0_f)) if r != R0_f else zero_f
            angle = m_f * theta - n_f * phi
            dB_R = amplitude * np.float32(np.sin(angle))
            dB_Z = amplitude * np.float32(np.cos(angle))
            Bx += dB_R * cos_phi
            By += dB_R * sin_phi
            Bz += dB_Z

        B_out[i, 0] = Bx
        B_out[i, 1] = By
        B_out[i, 2] = Bz


def vectorized_gather_and_B(pos_arr, E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid,
                            R_min, R_max, Z_min, Z_max, nR, nZ, B0, R0, t, b_perturb, m_mode, n_mode, gamma):
    """Allocating wrapper around vectorized_gather_and_B_into.

    Same signature and same (E_arr, B_arr) return as before the out-parameter
    split, for the callers that run once per run rather than once per step --
    package_reactor_results' phase-space projection and the plasma-oscillation
    test -- where one pair of (N,3) arrays is not worth a reused buffer. The
    reactor loop calls the _into form directly against buffers it owns.
    """
    N = pos_arr.shape[0]
    E_arr = np.empty((N, 3), dtype=np.float32)
    B_arr = np.empty((N, 3), dtype=np.float32)
    vectorized_gather_and_B_into(
        pos_arr, E_arr, B_arr,
        E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid,
        R_min, R_max, Z_min, Z_max, nR, nZ, B0, R0, t, b_perturb, m_mode, n_mode, gamma
    )
    return E_arr, B_arr

# NOTE: the circular check_confinement (r - 1.0)^2 + z^2 > 0.3^2 that used to live here
# has been removed. It was a second, disagreeing definition of the plasma boundary: the
# real last closed flux surface reaches |Z| ~ 0.354 and R ~ 1.386, so the circle cut
# particles the physics still considered confined. All boundary logic now goes through
# check_confinement_flux (CPU) / check_confinement_torch (GPU), which test psi against
# psi_edge and are the only authority on particle loss.

@njit(parallel=True, fastmath=True)
def apply_vectorized_collisions(vel_arr, type_arr, nu_c, dt):
    N = vel_arr.shape[0]
    for i in prange(N):
        if (type_arr[i] == 0 or type_arr[i] == 1) and np.random.rand() < nu_c * dt:
            speed = np.linalg.norm(vel_arr[i])
            costheta = 2.0 * np.random.rand() - 1.0
            sintheta = np.sqrt(1.0 - costheta**2)
            phi = 2.0 * np.pi * np.random.rand()
            vel_arr[i, 0] = speed * sintheta * np.cos(phi)
            vel_arr[i, 1] = speed * sintheta * np.sin(phi)
            vel_arr[i, 2] = speed * costheta
    return vel_arr

# =======================================================
# MULTI-CORE & GPU HPC BENCHMARK
# =======================================================
def run_hpc_benchmark(cfg):
    print("==================================================")
    print("   MULTI-CORE & GPU HPC BENCHMARK RUN    ")
    print("==================================================")
    
    cpu_times = []
    gpu_times = []
    
    q, m, dt = cfg.e_charge, cfg.m_deuterium, cfg.reactor_dt
    hpc_engine = HPCPhysicsAccelerator(cfg.HPC_DEVICE)
    
    for num_particles in cfg.BENCHMARK_PARTICLE_COUNTS:
        print(f"[{num_particles:,} Particles] Generating Tensors and Caches...")
        pos_arr = np.random.rand(num_particles, 3).astype(np.float32)
        vel_arr = np.random.rand(num_particles, 3).astype(np.float32)
        B_arr = np.ones((num_particles, 3), dtype=np.float32) * cfg.B0
        E_arr = np.random.rand(num_particles, 3).astype(np.float32)
        
        pos_cpu, vel_cpu = pos_arr.copy(), vel_arr.copy()
        vectorized_boris_push_numba_fallback(pos_cpu[:10], vel_cpu[:10], q, m, B_arr[:10], E_arr[:10], dt)
        
        start_cpu = time.time()
        for _ in range(cfg.BENCHMARK_STEPS):
            vectorized_boris_push_numba_fallback(pos_cpu, vel_cpu, q, m, B_arr, E_arr, dt)
        cpu_duration = time.time() - start_cpu
        cpu_times.append(cpu_duration)
        
        if cfg.HPC_DEVICE.type != "cpu":
            pos_tensor = torch.tensor(pos_arr, device=cfg.HPC_DEVICE)
            vel_tensor = torch.tensor(vel_arr, device=cfg.HPC_DEVICE)
            B_tensor = torch.tensor(B_arr, device=cfg.HPC_DEVICE)
            E_tensor = torch.tensor(E_arr, device=cfg.HPC_DEVICE)
            
            # Warm up at the FULL particle count: the compiled push specializes per exact
            # shape, so a 10-particle warmup would leave the real shape to compile inside
            # the timed loop below and inflate gpu_duration.
            hpc_engine.vectorized_boris_push_metal(pos_tensor, vel_tensor, q, m, B_tensor, E_tensor, dt)
            if cfg.HPC_DEVICE.type == "cuda": torch.cuda.synchronize()
            if cfg.HPC_DEVICE.type == "mps": torch.mps.synchronize()
            
            start_gpu = time.time()
            for _ in range(cfg.BENCHMARK_STEPS):
                pos_tensor, vel_tensor = hpc_engine.vectorized_boris_push_metal(pos_tensor, vel_tensor, q, m, B_tensor, E_tensor, dt)
            
            if cfg.HPC_DEVICE.type == "mps": torch.mps.synchronize() 
            elif cfg.HPC_DEVICE.type == "cuda": torch.cuda.synchronize()
                
            gpu_duration = time.time() - start_gpu
            gpu_times.append(gpu_duration)
        else:
            gpu_times.append(None)
            
        gpu_str = f"{gpu_duration:.4f}s" if gpu_times[-1] is not None else "N/A"
        print(f"  -> CPU Parallel Time: {cpu_duration:.4f}s | Apple Metal Time: {gpu_str}")
        
    print("[SYSTEM] HPC Benchmark Complete! Handing off to diagnostics...")
    diagnostics.plot_hpc_benchmark(cfg.BENCHMARK_PARTICLE_COUNTS, cpu_times, gpu_times)

GPU_PARTICLE_THRESHOLD = None
# The particle count at which the reactor run shifts from the Numba JIT CPU loop
# (_run_reactor_loop_cpu) to the PyTorch GPU loop (_run_reactor_loop_gpu): Apple Metal
# Performance Shaders on Apple Silicon, CUDA on PC.
#
# None means "always take the CPU loop". It held 1,000,000 from the initial commit
# through step 6, which silently sent every 1M-particle run down the slower path:
# measured at 1,000,000 particles / 200 steps during step 6, the GPU loop took 69.6 s
# wall against ~32-36 s for the CPU loop. The CPU loop measured faster at every count
# tested, up to 3,000,000.
#
# NOT a deprecation of the GPU path -- step 12 optimizes that loop and step 13 reassesses
# this number. To force the GPU loop, set this to an int <= cfg.initial_thermal_count,
# either by editing this line (what README.md documents) or at runtime without editing:
#     import main; main.GPU_PARTICLE_THRESHOLD = 0; main.run_reactor_steady_state()
# The dispatch reads this global when run_reactor_steady_state is called, so the runtime
# form works; `cfg.PROFILE = True` prints which loop was chosen.

# The push is memory-bound at ~0.85 flop/byte on a shared bus, and going to the device
# adds a round-trip plus dispatch overhead, so no particle count repays it.
#
# None means "never dispatch the push to the GPU". Kept as a named knob so the decision
# stays visible and is easy to re-test.
PUSH_GPU_THRESHOLD = None


def _boris_push_adaptive(pos, vel, q, m, B, E, dt, device):
    # Mutates pos, vel in place either way (matching the numba kernel's convention), so
    # call sites don't need to branch.
    if PUSH_GPU_THRESHOLD is not None and len(pos) >= PUSH_GPU_THRESHOLD:
        pt = torch.tensor(pos, device=device, dtype=torch.float32)
        vt = torch.tensor(vel, device=device, dtype=torch.float32)
        Bt = torch.tensor(B, device=device, dtype=torch.float32)
        Et = torch.tensor(E, device=device, dtype=torch.float32)
        pn, vn = _vectorized_boris_push_metal_impl(pt, vt, q, m, Bt, Et, dt)
        pos[:] = pn.cpu().numpy()
        vel[:] = vn.cpu().numpy()
    else:
        vectorized_boris_push_numba_fallback(pos, vel, q, m, B, E, dt)


@njit(cache=True)
def _compact_pool_numba(pos, vel, typ, pid, idx, m):
    """In-place stream compaction of the four particle arrays.

    idx holds the surviving row indices in ASCENDING order, so idx[k] >= k for
    every k: the destination has always already been read by the time it is
    written. That is what makes this safe to do in place with no temporary,
    where `pool.pos[:m] = pool.pos[idx]` would materialise an (m, 3) copy first.

    Sequential on purpose -- do NOT prange this. The read/write overlap that
    makes the forward pass safe depends on the strict k ordering, and under
    parallel execution one thread can write row k_A while another still needs it
    as the source idx[k_B] = k_A for some k_B < k_A.
    """
    for k in range(m):
        src = idx[k]
        pos[k, 0] = pos[src, 0]
        pos[k, 1] = pos[src, 1]
        pos[k, 2] = pos[src, 2]
        vel[k, 0] = vel[src, 0]
        vel[k, 1] = vel[src, 1]
        vel[k, 2] = vel[src, 2]
        typ[k] = typ[src]
        pid[k] = pid[src]


class _ParticlePool:
    """Capacity-backed particle storage: pos/vel/type/pid allocated once, with an
    n_live write pointer.

    Both loops used to grow by full reallocation -- np.vstack/np.append on the
    CPU path, torch.cat on the GPU path -- once every cfg.inject_every_n_steps.
    At 1,000,000 particles that copied ~36 MB per event (12 MB pos + 12 MB vel +
    4 MB type + 8 MB pid), about 5,000 times over a production run, and churned
    the MPS allocator badly on the GPU side.

    The capacity is a HARD CEILING, not a guess: _pid_capacity_bound counts every
    particle the injection schedule can source, and losses only ever compact
    downward, so n_live can never legitimately exceed it. add() therefore RAISES
    on overflow instead of reallocating. A silent grow would paper over a broken
    bound, and the bound is also what sizes pid_row/pid_slot/pid_pool -- so if it
    were wrong, the quiet failure being hidden is an out-of-range write in a numba
    kernel that does not bounds-check.

    device=None keeps the arrays in numpy (CPU loop); pass a torch device and the
    same structure holds device tensors instead.
    """

    def __init__(self, capacity, device=None):
        self.capacity = int(capacity)
        self.device = device
        self.n_live = 0
        if device is None:
            self.pos = np.empty((self.capacity, 3), dtype=np.float32)
            self.vel = np.empty((self.capacity, 3), dtype=np.float32)
            self.type = np.empty(self.capacity, dtype=np.int32)
            self.pid = np.empty(self.capacity, dtype=np.int64)
        else:
            self.pos = torch.empty((self.capacity, 3), dtype=torch.float32, device=device)
            self.vel = torch.empty((self.capacity, 3), dtype=torch.float32, device=device)
            self.type = torch.empty(self.capacity, dtype=torch.int32, device=device)
            self.pid = torch.empty(self.capacity, dtype=torch.int64, device=device)

    def views(self):
        """The live prefix of each array.

        A leading-axis slice of a C-contiguous array is itself C-contiguous, so
        every numba signature and every torch kernel sees exactly what it saw
        before the pools existed -- no recompilation, no hidden copy. Kernels
        that mutate in place (the Boris push, check_confinement_flux,
        apply_vectorized_collisions) write straight through into the pool.
        """
        n = self.n_live
        return self.pos[:n], self.vel[:n], self.type[:n], self.pid[:n]

    def _require(self, extra):
        if self.n_live + extra > self.capacity:
            raise AssertionError(
                f"_ParticlePool overflow: n_live={self.n_live} + {extra} exceeds "
                f"capacity={self.capacity}. The capacity comes from "
                f"_pid_capacity_bound, which is meant to be a hard ceiling on the "
                f"injection schedule -- reaching it means that bound is wrong. "
                f"Fix the bound; do not grow the pool here.")

    def add(self, pos_block, vel_block, type_value, pid_block):
        """Append a batch. Returns the first row written."""
        batch = len(pid_block)
        if batch == 0:
            return self.n_live
        self._require(batch)
        start, end = self.n_live, self.n_live + batch
        self.pos[start:end] = pos_block
        self.vel[start:end] = vel_block
        self.type[start:end] = type_value
        self.pid[start:end] = pid_block
        self.n_live = end
        return start

    def compact(self, alive_idx):
        """Keep only alive_idx (ascending), in place. Returns the new n_live."""
        m = len(alive_idx)
        if self.device is None:
            _compact_pool_numba(self.pos, self.vel, self.type, self.pid,
                                alive_idx, m)
        else:
            # index_select allocates one temporary per array. Left as is on
            # purpose: wall-loss compaction measures 0.0% of loop wall, so the
            # contortion to avoid it would buy nothing.
            self.pos[:m] = self.pos.index_select(0, alive_idx)
            self.vel[:m] = self.vel.index_select(0, alive_idx)
            self.type[:m] = self.type.index_select(0, alive_idx)
            self.pid[:m] = self.pid.index_select(0, alive_idx)
        self.n_live = m
        return m


# Hard ceiling on plotted trajectories: 1000 initial thermals plus the
# tracked_nbis and tracked_alphas caps of 1000 each, all three enforced at the
# injection sites. _TrackStore raises rather than silently overrunning.
_MAX_TRACKED_SLOTS = 3000


def _pid_capacity_bound(cfg, n_init):
    """Upper bound on the largest pid a run can reach, plus headroom.

    Sizing the pid -> row map from the initial particle count is WRONG: pids are
    handed out monotonically and never reused, so injection walks them past
    n_init and a map sized that way overruns partway through a long run (a
    5,200-step disruption run reaches ~n_init + 34,000). The bound below counts
    the injections the loop can actually perform:

      NBI    -- one batch every cfg.inject_every_n_steps steps, and a batch is
                int(rate) plus at most one more from the stochastic remainder,
                with rate <= cfg.NBI_BATCH_SIZE because the exponential taper
                only ever shrinks it.
      alphas -- exactly one every cfg.inject_every_n_steps * 2 steps.

    Step 8 makes this bound load-bearing in a second way: it also sizes the
    _ParticlePool row capacity, because max rows and max pid are the same
    quantity (n_init + everything injection can source, since losses only
    compact downward). The pool RAISES on overflow rather than growing, so a
    wrong bound surfaces immediately instead of being absorbed.
    """
    steps = int(cfg.reactor_num_steps)
    nbi_events = steps // max(1, int(cfg.inject_every_n_steps)) + 1
    alpha_events = steps // max(1, int(cfg.inject_every_n_steps) * 2) + 1
    nbi_max = int(cfg.NBI_BATCH_SIZE) + 1
    return int(n_init + nbi_events * nbi_max + alpha_events) + 1024


def _require_pid_capacity(next_pid, batch, capacity, where):
    """Tripwire for the pid maps, which share the pool's hard ceiling.

    Replaces the pair of _ensure_pid_capacity helpers that used to DOUBLE the
    maps on overflow. Growing was the wrong response: pid_capacity comes from
    _pid_capacity_bound, which counts every particle the injection schedule can
    source, so overflowing it means that bound is wrong -- and quietly enlarging
    the map hides the bug while _ParticlePool.add, sized from the same number,
    would raise a step later anyway. Fail here, naming the site.
    """
    if next_pid + batch > capacity:
        raise AssertionError(
            f"pid capacity exceeded at {where}: next_pid={next_pid} + {batch} > "
            f"capacity={capacity}. _pid_capacity_bound is meant to be a hard "
            f"ceiling on the injection schedule; fix the bound rather than "
            f"growing the map here.")


class _TrackStore:
    """Array-backed storage for the plotted particle trajectories.

    Replaces four pid-keyed dicts (history_tracks / tracked_type /
    tracked_lastpos / tracked_lastvel) and a set (tracked_lost), plus the Python
    loop that appended one (3,) copy per tracked pid per sampling tick -- up to
    3,000 dict lookups and copies every second step, on the order of 10^7 Python
    iterations over a full run.

    SLOTS, not pids, index every array here: a pid gets the next free slot when
    it starts being tracked. Sampled vertices go into ONE flat buffer as
    (slot, xyz) pairs appended in step order, and are regrouped per track once,
    at the end of the run, by a stable argsort on the slot column. Stable is
    load-bearing -- it is the only thing keeping each track's vertices in
    chronological order.

    Two vertices are held apart from that buffer because they are not on the
    sampling cadence:
      * the injection vertex, always a track's first, and
      * the wall-impact vertex, always a track's last -- the particle is dead
        from that step on, so it can never be sampled again. This is what makes
        the crimson wall-strike traces terminate at the wall in
        tokamak_reactor_2d.png / tokamak_reactor_3d.png.
    Both are stored per slot in host arrays, so to_dicts can splice them onto
    the ends without disturbing the sort.

    device=None keeps sampled coordinates in a numpy buffer (CPU loop). Pass a
    torch device and they accumulate in a device tensor instead, so the GPU loop
    never routes a sample through the host: one transfer at end of run replaces
    two per sampling step. The slot column stays on the host either way, because
    slot ids are host-side bookkeeping in both loops -- which is also what lets
    the host advance the write pointer without a device sync.

    host_dtype exists to preserve each loop's existing vertex values exactly.
    The CPU loop reads its injection vertex back out of the float32 pos_np; the
    GPU loop reads it straight from the float64 array
    inject_neutral_beam_cartesian returns, before it is cast down onto the
    device. Those are different values, and both are what their path recorded
    before this rewrite.
    """

    def __init__(self, slot_capacity, device=None, host_dtype=np.float32,
                 vertex_capacity=4096):
        self.device = device
        self.slot_capacity = int(slot_capacity)
        self.n_slots = 0

        self.pids = np.full(self.slot_capacity, -1, dtype=np.int64)
        self.species = np.zeros(self.slot_capacity, dtype=np.int8)
        self.lost = np.zeros(self.slot_capacity, dtype=bool)

        self.init_xyz = np.zeros((self.slot_capacity, 3), dtype=host_dtype)
        self.final_xyz = np.zeros((self.slot_capacity, 3), dtype=host_dtype)
        self.has_final = np.zeros(self.slot_capacity, dtype=bool)

        # last_pos / last_vel have two sources. Injection and wall impact are
        # host-side, sampling is device-side on the GPU path; last_is_host says
        # which one currently holds the live value for a slot.
        self.last_pos_host = np.zeros((self.slot_capacity, 3), dtype=host_dtype)
        self.last_vel_host = np.zeros((self.slot_capacity, 3), dtype=host_dtype)
        self.last_is_host = np.ones(self.slot_capacity, dtype=bool)

        self.v_slot = np.empty(int(vertex_capacity), dtype=np.int32)
        self.n_verts = 0
        if device is None:
            self.v_xyz = np.empty((int(vertex_capacity), 3), dtype=np.float32)
            self.last_pos_dev = None
            self.last_vel_dev = None
        else:
            self.v_xyz = torch.empty((int(vertex_capacity), 3),
                                     dtype=torch.float32, device=device)
            self.last_pos_dev = torch.zeros((self.slot_capacity, 3),
                                            dtype=torch.float32, device=device)
            self.last_vel_dev = torch.zeros((self.slot_capacity, 3),
                                            dtype=torch.float32, device=device)

    # -- slot bookkeeping -------------------------------------------------
    def add_slots(self, pids, species, init_pos, init_vel):
        """Start tracking pids. Returns the slot indices assigned, in order."""
        n = len(pids)
        if n == 0:
            return np.empty(0, dtype=np.int64)
        if self.n_slots + n > self.slot_capacity:
            raise AssertionError(
                f"_TrackStore slot overflow: {self.n_slots} + {n} > "
                f"{self.slot_capacity}; the 1000-each tracked_nbis/tracked_alphas "
                f"caps should have prevented this")
        slots = np.arange(self.n_slots, self.n_slots + n, dtype=np.int64)
        self.pids[slots] = pids
        self.species[slots] = species
        self.init_xyz[slots] = init_pos
        self.last_pos_host[slots] = init_pos
        self.last_vel_host[slots] = init_vel
        self.last_is_host[slots] = True
        self.n_slots += n
        return slots

    # -- vertex accumulation ----------------------------------------------
    def _reserve(self, m):
        need = self.n_verts + m
        if need <= self.v_slot.shape[0]:
            return
        cap = self.v_slot.shape[0]
        while cap < need:
            cap *= 2
        self.v_slot = np.resize(self.v_slot, cap)
        if self.device is None:
            grown = np.empty((cap, 3), dtype=np.float32)
            grown[:self.n_verts] = self.v_xyz[:self.n_verts]
        else:
            grown = torch.empty((cap, 3), dtype=torch.float32, device=self.device)
            grown[:self.n_verts] = self.v_xyz[:self.n_verts]
        self.v_xyz = grown

    def append_samples(self, slots_np, xyz):
        """Append one sampled vertex per slot. xyz is numpy (CPU) or a device
        tensor (GPU) with one row per entry of slots_np, already in slot order."""
        m = len(slots_np)
        if m == 0:
            return
        self._reserve(m)
        w = self.n_verts
        self.v_slot[w:w + m] = slots_np
        self.v_xyz[w:w + m] = xyz
        self.n_verts = w + m

    def set_last_host(self, slots_np, pos, vel):
        self.last_pos_host[slots_np] = pos
        self.last_vel_host[slots_np] = vel
        self.last_is_host[slots_np] = True

    def set_last_device(self, slots_np, slots_t, pos_t, vel_t):
        self.last_pos_dev[slots_t] = pos_t
        self.last_vel_dev[slots_t] = vel_t
        self.last_is_host[slots_np] = False

    def record_impact(self, slots_np, pos, vel):
        """Wall strike: the terminal vertex, off the sampling cadence."""
        if len(slots_np) == 0:
            return
        self.final_xyz[slots_np] = pos
        self.has_final[slots_np] = True
        self.lost[slots_np] = True
        self.set_last_host(slots_np, pos, vel)

    # -- slot selection ----------------------------------------------------
    def slots_due(self, sample_thermal, sample_alpha):
        """Slot indices due to be sampled this step, by species cadence."""
        n = self.n_slots
        if n == 0:
            return np.empty(0, dtype=np.int64)
        if sample_thermal and sample_alpha:
            return np.arange(n, dtype=np.int64)
        is_alpha = self.species[:n] == 2
        return np.nonzero(is_alpha if sample_alpha else ~is_alpha)[0]

    # -- teardown ----------------------------------------------------------
    def to_dicts(self):
        """Rebuild the dict-of-lists payload the return signature still uses.

        Runs once, after the loop. On the GPU path this is the ONLY place the
        sampled vertices and the device-side last_pos/last_vel cross back to the
        host.
        """
        n = self.n_verts
        v_xyz = self.v_xyz[:n]
        last_pos_dev = last_vel_dev = None
        if self.device is not None:
            v_xyz = v_xyz.cpu().numpy()
            last_pos_dev = self.last_pos_dev[:self.n_slots].cpu().numpy()
            last_vel_dev = self.last_vel_dev[:self.n_slots].cpu().numpy()

        slots = self.v_slot[:n]
        order = np.argsort(slots, kind="stable")
        counts = (np.bincount(slots, minlength=self.n_slots) if n
                  else np.zeros(self.n_slots, dtype=np.int64))

        history_tracks, tracked_type = {}, {}
        tracked_lastpos, tracked_lastvel = {}, {}
        tracked_lost = set()

        offset = 0
        for s in range(self.n_slots):
            pid = int(self.pids[s])
            c = int(counts[s])
            verts = [self.init_xyz[s].copy()]
            if c:
                verts.extend(v_xyz[order[offset:offset + c]])
                offset += c
            if self.has_final[s]:
                verts.append(self.final_xyz[s].copy())

            history_tracks[pid] = verts
            tracked_type[pid] = int(self.species[s])
            if self.last_is_host[s] or last_pos_dev is None:
                tracked_lastpos[pid] = self.last_pos_host[s].copy()
                tracked_lastvel[pid] = self.last_vel_host[s].copy()
            else:
                tracked_lastpos[pid] = last_pos_dev[s].copy()
                tracked_lastvel[pid] = last_vel_dev[s].copy()
            if self.lost[s]:
                tracked_lost.add(pid)

        return (history_tracks, tracked_type, tracked_lastpos,
                tracked_lastvel, tracked_lost)


def _run_reactor_loop_cpu(cfg, engine, pos_tensor, vel_tensor, type_tensor, rho_grid, phi_grid, E_R_grid, E_Z_grid,
                          B_R_pol_grid, B_Z_pol_grid):
    # The whole per-step pipeline runs on Numpy/Numba. At this particle count (~10k-20k,
    # growing via injection) prange beats MPS outright: dispatching dozens of small
    # per-step kernels (gather, CIC, collisions, confinement) costs more in Metal
    # command-buffer overhead than the math itself. Pull the tensors to Numpy once here
    # and never touch torch inside the hot loop.
    pos_init = pos_tensor.cpu().numpy()
    vel_init = vel_tensor.cpu().numpy()
    type_init = type_tensor.cpu().numpy()

    # Stable per-particle IDs. Lost particles are physically removed from the pools, not
    # just flagged type=-1, which shifts every row index -- so trajectory bookkeeping
    # cannot be keyed on array position.
    n_init = len(pos_init)
    next_pid = n_init

    # pid -> current row, dense. This replaces np.searchsorted(pid_np, want),
    # which was correct only because pid_np happened to be sorted ascending
    # (monotonic ids, and compaction preserves order). Step 9 splits the bulk and
    # alpha pools and retires that invariant permanently, at which point a
    # searchsorted would not fail loudly -- it would silently return the wrong
    # row and re-point a track at an unrelated particle. A dense map makes the
    # lookup one gather and assumes no ordering at all.
    # Sized from the largest pid the run can reach, NOT from n_init: pids are
    # never reused, so injection walks them past the initial count.
    pid_capacity = _pid_capacity_bound(cfg, n_init)
    pid_row = np.full(pid_capacity, -1, dtype=np.int64)
    pid_row[:n_init] = np.arange(n_init, dtype=np.int64)
    # pid -> track slot, -1 when the pid is not plotted. Replaces the
    # `dpid in history_tracks` dict membership test in the compaction below.
    pid_slot = np.full(pid_capacity, -1, dtype=np.int64)
    # pid -> which pool the particle lives in. Only one pool exists today, so
    # every live pid maps to 0; step 9 splits bulk (0) from alpha (1) and this
    # becomes the thing that says which pool a row index is relative to.
    # Added now rather than in step 9 so the birth sites are touched once.
    #
    # WRITE-ONCE, at birth. It is deliberately NOT updated by compaction, and
    # that is sound rather than an oversight: species never migrates. type is
    # written only at birth (0/1/2) and at death (-1) -- THERMALIZATION_ENERGY_KEV
    # gates whether an alpha keeps heating, it does not reclassify it -- so a
    # particle can never change pool. pid_row already carries -1 for the dead.
    pid_pool = np.full(pid_capacity, -1, dtype=np.int8)
    pid_pool[:n_init] = 0

    # Capacity-backed storage, replacing the vstack/append growth below.
    pool = _ParticlePool(pid_capacity, device=None)
    pool.add(pos_init, vel_init, 0, np.arange(n_init, dtype=np.int64))
    pos_np, vel_np, type_np, pid_np = pool.views()
    type_np[:] = type_init          # preserve any non-zero species from init

    n_track = min(1000, n_init)
    # 1000 initial thermals + the 1000-each NBI and alpha caps below.
    tracks = _TrackStore(_MAX_TRACKED_SLOTS, device=None, host_dtype=np.float32)
    init_slots = tracks.add_slots(pid_np[:n_track], 0,
                                  pos_np[:n_track], vel_np[:n_track])
    pid_slot[pid_np[:n_track]] = init_slots
    tracked_nbis = 0
    tracked_alphas = 0

    # Per-step field buffers, allocated once and reused. vectorized_gather_and_B
    # used to return a fresh pair of (N, 3) float32 arrays every step -- 24 MB
    # allocated and freed per step at 1M particles, ~10,000 times over a full run.
    # The loop now owns the storage and passes [:n] views to the _into kernel,
    # which writes every row it is given, so stale rows past n are never read.
    # Capacity starts above the initial count (NBI and alpha injection grow the
    # pools) and doubles only if the live count outruns it.
    # Sized from the pool's capacity rather than a second, independently-grown
    # number. Two capacities that can disagree is exactly the bug this avoids:
    # vectorized_gather_and_B_into does not bounds-check its E_out/B_out stores,
    # so a field buffer shorter than n_live is a silent out-of-range write, not
    # an IndexError.
    E_buf = np.empty((pool.capacity, 3), dtype=np.float32)
    B_buf = np.empty((pool.capacity, 3), dtype=np.float32)

    total_injected = cfg.initial_thermal_count
    total_lost = 0
    inventory_history, energy_history_keV, instability_amp_history = [], [], []
    time_history, temp_history, rad_power_history = [], [], []
    trigger_time, T_core_baseline, T_core = None, 0.0, 0.0

    alpha_heating_power_history_MW, external_heating_power_history_MW = [], []
    bremsstrahlung_power_history_MW, cyclotron_power_history_MW = [], []
    q_sci_history, q_eng_history, lawson_triple_product_history = [], [], []

    print("[SYSTEM] Igniting time-domain reactor loop on CPU/Numba (fastest path at this particle count)...")

    # --- Profiling state (inert unless cfg.PROFILE) ---
    profiling = bool(getattr(cfg, "PROFILE", False))
    prof = StageProfiler(profiling, label="CPU path")
    # Sub-stages of the deuteron push, on their own profiler so they do not
    # double-count inside the main table. The push "stage" is not just the
    # push: it also pays for the mask, four fancy-index gathers (pos, vel,
    # B, E) and two masked scatters, all over the full particle array.
    push_prof = StageProfiler(profiling, label="deuteron push breakdown")
    sor_iter_counts = []
    if profiling:
        _profile_header(cfg, "CPU / Numba path", len(pos_np))
    loop_wall_start = time.perf_counter() if profiling else None

    for step in range(cfg.reactor_num_steps):
        t = step * cfg.reactor_dt
        time_history.append(t)

        # --- BATCH NBI INJECTION (tapering source) ---
        # A fixed batch forever meant inventory only climbed and no steady state was
        # reachable. The source decays as exp(-step/tau), with fractional rates realised
        # stochastically so the beam thins out smoothly.
        _t = prof.mark()
        if step % cfg.inject_every_n_steps == 0 and step > 0:
            nbi_rate = cfg.NBI_BATCH_SIZE * np.exp(-step / cfg.NBI_DECAY_TAU_STEPS)
            batch_size = int(nbi_rate)
            if np.random.random() < (nbi_rate - batch_size):
                batch_size += 1

            if batch_size > 0:
                psi_bounds = (engine.eq.psi_grid, engine.eq.psi_core, engine.eq.psi_R_min, engine.eq.psi_R_max,
                              engine.eq.psi_Z_min, engine.eq.psi_Z_max, engine.eq.psi_nR, engine.eq.psi_nZ)
                p_nbi, v_nbi = engine.inject_neutral_beam_cartesian(num_ions=batch_size, E_keV=cfg.nbi_energy_keV, psi_bounds=psi_bounds)
                _require_pid_capacity(next_pid, batch_size, pid_capacity, "CPU NBI injection")
                new_pids = np.arange(next_pid, next_pid + batch_size, dtype=np.int64)
                # Writes into rows [n_live : n_live+batch] -- no reallocation.
                # Assigning float64 p_nbi into the float32 pool rounds exactly as
                # the old vstack(...).astype(np.float32) did; verified
                # bit-identical on real injector output before this change.
                base_row = pool.add(p_nbi, v_nbi, 1, new_pids)
                pos_np, vel_np, type_np, pid_np = pool.views()
                pid_row[new_pids] = np.arange(base_row, base_row + batch_size,
                                              dtype=np.int64)
                pid_pool[new_pids] = 0
                next_pid += batch_size
                total_injected += batch_size

                # The old loop tracked the first (1000 - tracked_nbis) of the
                # batch and skipped the rest; take that same prefix.
                n_new = min(batch_size, 1000 - tracked_nbis)
                if n_new > 0:
                    rows = np.arange(base_row, base_row + n_new, dtype=np.int64)
                    slots = tracks.add_slots(new_pids[:n_new], 1,
                                             pos_np[rows], vel_np[rows])
                    pid_slot[new_pids[:n_new]] = slots
                    tracked_nbis += n_new

        prof.add("NBI injection", _t)

        # --- BATCH ALPHA INJECTION ---
        _t = prof.mark()
        if step % (cfg.inject_every_n_steps * 2) == 0 and step > 0:
            alpha_batch = 1
            phi_pos = np.random.uniform(0, 2 * np.pi, alpha_batch)
            R_birth = np.random.normal(cfg.R0_major, 0.05, alpha_batch)
            Z_birth = np.random.normal(0.0, 0.05, alpha_batch)
            aR, aZ, avR, avphi, avZ = engine.spawn_alpha_particles(alpha_batch, R_birth[0], Z_birth[0], cfg.ALPHA_ENERGY_JOULES, cfg.MASS_ALPHA)

            p_alpha = np.zeros((alpha_batch, 3), dtype=np.float32)
            v_alpha = np.zeros((alpha_batch, 3), dtype=np.float32)
            p_alpha[:, 0] = aR * np.cos(phi_pos)
            p_alpha[:, 1] = aR * np.sin(phi_pos)
            p_alpha[:, 2] = aZ
            v_alpha[:, 0] = avR * np.cos(phi_pos) - avphi * np.sin(phi_pos)
            v_alpha[:, 1] = avR * np.sin(phi_pos) + avphi * np.cos(phi_pos)
            v_alpha[:, 2] = avZ

            _require_pid_capacity(next_pid, alpha_batch, pid_capacity, "CPU alpha injection")
            new_pids = np.arange(next_pid, next_pid + alpha_batch, dtype=np.int64)
            base_row = pool.add(p_alpha, v_alpha, 2, new_pids)
            pos_np, vel_np, type_np, pid_np = pool.views()
            pid_row[new_pids] = np.arange(base_row, base_row + alpha_batch,
                                          dtype=np.int64)
            # Still pool 0: step 9 is what makes alphas their own pool.
            pid_pool[new_pids] = 0
            next_pid += alpha_batch

            n_new = min(alpha_batch, 1000 - tracked_alphas)
            if n_new > 0:
                rows = np.arange(base_row, base_row + n_new, dtype=np.int64)
                slots = tracks.add_slots(new_pids[:n_new], 2,
                                         pos_np[rows], vel_np[rows])
                pid_slot[new_pids[:n_new]] = slots
                tracked_alphas += n_new

        prof.add("alpha injection", _t)

        _t = prof.mark()
        mask_valid = (type_np == 0) | (type_np == 1)
        R_coords = np.sqrt(pos_np[mask_valid, 0]**2 + pos_np[mask_valid, 1]**2)
        Z_coords = pos_np[mask_valid, 2]
        charges = np.full(np.sum(mask_valid), cfg.e_charge)
        rho_grid = compute_cic_charge_density(R_coords, Z_coords, charges, cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ, False)
        prof.add("CIC deposition", _t)

        _t = prof.mark()
        if step > 0:
            # Warm-start the Poisson SOR solve from last step's converged phi: same
            # answer within the same tol, far fewer iterations to reach it.
            phi_grid, E_R_grid, E_Z_grid, _sor_iters = engine.solve_fields(
                rho_grid, cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max,
                phi_init=phi_grid, return_iters=True)
            if profiling:
                sor_iter_counts.append(_sor_iters)
        prof.add("solve_fields (Poisson)", _t)

        _t = prof.mark()
        n_live = pool.n_live
        # No growth check: the buffers are pool.capacity rows, and pool.add
        # raises before n_live could ever exceed that.
        # Views, not copies: the masked gathers below index these exactly as they
        # did the freshly allocated arrays.
        E_np = E_buf[:n_live]
        B_np = B_buf[:n_live]
        vectorized_gather_and_B_into(
            pos_np, E_np, B_np,
            E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid,
            cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ,
            cfg.B0, cfg.R0_major, t, cfg.b_perturb_initial, cfg.m_mode, cfg.n_mode, cfg.gamma_growth
        )
        prof.add("vectorized_gather_and_B", _t)

        # --- PARTICLE PUSH: Numba/CPU below PUSH_GPU_THRESHOLD, eager-GPU above it ---
        _t = prof.mark()
        _ts = push_prof.mark()
        mask_d = (type_np == 0) | (type_np == 1)
        if np.any(mask_d):
            pos_d = pos_np[mask_d].copy()
            vel_d = vel_np[mask_d].copy()
            push_prof.add("mask + pos/vel gather", _ts)

            _ts = push_prof.mark()
            _boris_push_adaptive(pos_d, vel_d, cfg.e_charge, cfg.m_deuterium, B_np[mask_d], E_np[mask_d], cfg.reactor_dt, cfg.HPC_DEVICE)
            # Includes evaluating B_np[mask_d] and E_np[mask_d] as arguments:
            # two more full-array fancy-index gathers, not part of the kernel.
            push_prof.add("push call (+ B/E gather)", _ts)

            _ts = push_prof.mark()
            pos_np[mask_d] = pos_d
            vel_np[mask_d] = vel_d
            push_prof.add("masked write-back", _ts)
        else:
            push_prof.add("mask + pos/vel gather", _ts)
        prof.add("deuteron Boris push", _t)

        # Alphas are sub-stepped: at 3.5 MeV they cover ~1.3e-2 m per global 1 ns step
        # against a ~2.2e-2 m Larmor radius, under two samples per gyro-arc. Deuterons
        # stay on the single global step (Larmor radius ~5e-4 m, already resolved).
        _t = prof.mark()
        mask_a = type_np == 2
        if np.any(mask_a):
            pos_a = np.ascontiguousarray(pos_np[mask_a], dtype=np.float32)
            vel_a = np.ascontiguousarray(vel_np[mask_a], dtype=np.float32)
            vectorized_boris_push_numba_substeps(
                pos_a, vel_a, cfg.CHARGE_ALPHA, cfg.MASS_ALPHA,
                np.ascontiguousarray(B_np[mask_a]), np.ascontiguousarray(E_np[mask_a]),
                cfg.reactor_dt, cfg.ALPHA_SUBSTEPS
            )
            pos_np[mask_a] = pos_a
            vel_np[mask_a] = vel_a
        prof.add("alpha substep push", _t)

        _t = prof.mark()
        # The numba kernel mutates vel_arr in place and returns the same array, so
        # this rebinding is a no-op and vel_np stays the pool view. Do not replace
        # it with anything that allocates -- see the torch twin in the GPU loop.
        vel_np = apply_vectorized_collisions(vel_np, type_np, cfg.nu_c, cfg.reactor_dt)
        prof.add("apply_vectorized_collisions", _t)

        # Confinement against the psi flux surface, not a circle
        _t = prof.mark()
        newly_lost = check_confinement_flux(
            pos_np, type_np, engine.eq.psi_grid, engine.eq.psi_edge,
            engine.eq.psi_R_min, engine.eq.psi_R_max, engine.eq.psi_Z_min, engine.eq.psi_Z_max,
            engine.eq.psi_nR, engine.eq.psi_nZ
        )
        prof.add("check_confinement_flux", _t)
        total_lost += newly_lost

        # --- WALL LOSS COMPACTION ---
        _t = prof.mark()
        # check_confinement_flux only FLAGS wall strikes (type = -1). Leaving them in the
        # arrays meant the pools only ever grew, every kernel paid for dead particles, and
        # the confined count could never fall. Trajectory history is pid-keyed, so it
        # survives the row-index shift and the diagnostics payload is unchanged.
        if newly_lost > 0:
            alive = type_np != -1
            # Preserve the last known state of any tracked particle before it is dropped
            dead_rows = np.nonzero(~alive)[0]
            if dead_rows.size > 0:
                dead_pids = pid_np[dead_rows]
                dead_slots = pid_slot[dead_pids]
                is_tracked = dead_slots >= 0
                if np.any(is_tracked):
                    t_rows = dead_rows[is_tracked]
                    t_slots = dead_slots[is_tracked]
                    impact_pos = pos_np[t_rows]
                    # The impact point, off the normal sampling cadence, so the
                    # crimson trace ends at the wall rather than at the last
                    # 20-step tick. Always a track's final vertex: pid_row goes
                    # to -1 below, so the sampler can never touch it again.
                    tracks.record_impact(t_slots, impact_pos, vel_np[t_rows])
                pid_row[dead_pids] = -1

            # In-place forward compaction into the same buffers, replacing four
            # full-array reallocations. alive_rows is ascending, so every
            # destination has been read before it is written.
            alive_rows = np.nonzero(alive)[0]
            pool.compact(alive_rows)
            pos_np, vel_np, type_np, pid_np = pool.views()
            # Compaction renumbers every surviving row, so the map is rebuilt
            # here. One scatter over the live pids replaces the per-track
            # searchsorted the sampler used to do every other step. pid_pool is
            # NOT touched: a particle cannot change pool, only rows move.
            pid_row[pid_np] = np.arange(len(pid_np), dtype=np.int64)
        prof.add("wall-loss compaction", _t)

        # --- TRAJECTORY SAMPLING (pid-keyed, so removal cannot corrupt it) ---
        _t = prof.mark()
        # Alphas are sampled far more often than thermals: their orbit is only ~2.2e-2 m
        # across, so a 20-step cadence (~0.26 m of travel) aliases it away.
        sample_thermal = (step % 20 == 0)
        sample_alpha = (step % cfg.ALPHA_HISTORY_EVERY == 0)
        if (sample_thermal or sample_alpha) and tracks.n_slots > 0 and len(pid_np) > 0:
            # One mask for (due this step AND still alive), then a single fancy
            # index for every sampled position. The old form looped in Python
            # over up to 3,000 tracked pids with a dict lookup and two .copy()
            # calls each, every second step.
            due = tracks.slots_due(sample_thermal, sample_alpha)
            if due.size > 0:
                rows = pid_row[tracks.pids[due]]
                keep = rows >= 0
                sel_slots = due[keep]
                sel_rows = rows[keep]
                if sel_rows.size > 0:
                    sampled_pos = pos_np[sel_rows]
                    tracks.append_samples(sel_slots, sampled_pos)
                    tracks.set_last_host(sel_slots, sampled_pos, vel_np[sel_rows])
        prof.add("trajectory sampling", _t)

        _t = prof.mark()
        mask_alphas = type_np == 2
        alpha_deposited_kev = 0.0
        if np.any(mask_alphas):
            alpha_vels = vel_np[mask_alphas].astype(np.float64)
            v_mags = np.linalg.norm(alpha_vels, axis=1)
            alpha_energies_kev = (0.5 * cfg.MASS_ALPHA * (v_mags**2)) / 1.602e-16
            new_energies_kev, alpha_power_mw, alpha_deposited_kev = engine.compute_alpha_heating_power(
                alpha_energies_kev, cfg.reactor_dt, cfg)
            new_v_mags = np.sqrt(2.0 * (new_energies_kev * 1.602e-16) / cfg.MASS_ALPHA)
            scale_factors = new_v_mags / np.where(v_mags == 0, 1e-10, v_mags)
            alpha_vels *= scale_factors[:, np.newaxis]
            vel_np[mask_alphas] = alpha_vels.astype(np.float32)
        else:
            alpha_power_mw = 0.0

        alpha_heating_power_history_MW.append(alpha_power_mw)
        external_heating_power_history_MW.append(cfg.EXTERNAL_HEATING_MW)

        thermals_and_nbi_mask = (type_np == 0) | (type_np == 1)
        current_confined = int(np.sum(thermals_and_nbi_mask))
        inventory_history.append(current_confined)

        # --- ALPHA -> BULK ENERGY TRANSFER (energy conservation) ---
        # Energy drained from the alphas used to vanish -- removed from the fast
        # population and given to nothing, so the thermal plasma never felt the heating.
        # Deposit it by scaling bulk speeds, using the simulation-scale keV (no
        # macro_weight) so particles stay self-consistent; the MW figure above is
        # separately scaled for reactor-equivalent output.
        if alpha_deposited_kev > 0.0 and np.any(thermals_and_nbi_mask):
            bulk_vels = vel_np[thermals_and_nbi_mask].astype(np.float64)
            bulk_energy_kev = float(np.sum(0.5 * cfg.m_deuterium * np.sum(bulk_vels**2, axis=1))) / 1.602e-16
            if bulk_energy_kev > 0.0:
                boost = np.sqrt(1.0 + alpha_deposited_kev / bulk_energy_kev)
                vel_np[thermals_and_nbi_mask] = (bulk_vels * boost).astype(np.float32)

        if np.any(thermals_and_nbi_mask):
            current_energy_joules = np.sum(0.5 * cfg.m_deuterium * (np.linalg.norm(vel_np[thermals_and_nbi_mask].astype(np.float64), axis=1)**2))
        else:
            current_energy_joules = 0.0

        current_energy_keV = current_energy_joules / (cfg.e_charge * 1000.0)
        energy_history_keV.append(current_energy_keV)

        current_amp = cfg.b_perturb_initial * np.exp(cfg.gamma_growth * t)
        instability_amp_history.append(current_amp)

        # --- T_core IS A TEMPERATURE, NOT A TOTAL ---
        # Assigning the plasma's TOTAL stored energy here made the "core temperature"
        # read ~50,000 keV, and every consumer (brem's sqrt(T_e), the Lawson triple
        # product, the quench baseline) was fed that. For a 3D Maxwellian
        # <E> = (3/2)kT, so kT = (2/3) * <E>.
        n_bulk = max(current_confined, 1)
        T_core_kinetic = (2.0 / 3.0) * (current_energy_keV / n_bulk)

        if current_amp > cfg.MAX_ISLAND_WIDTH_THRESHOLD and not cfg.SPI_TRIGGERED:
            cfg.SPI_TRIGGERED = True
            trigger_time = t
            T_core_baseline = T_core_kinetic
            print(f" [EMERGENCY] Magnetic Island exceeded {cfg.MAX_ISLAND_WIDTH_THRESHOLD*100}% minor radius!")
            print(f" [SYSTEM] Firing Shattered Pellet Injection (SPI) at t={t*1000:.2f} ms...")
            print(f" [SYSTEM] Pre-disruption core temperature: {T_core_baseline:.3f} keV")

        # --- REPORTED TEMPERATURE IS MEASURED, NOT PRESCRIBED ---
        # Post-trigger T_core used to come from trigger_thermal_quench()'s analytic
        # exponential -- a curve drawn on top of the simulation. It ran on
        # TQ_DECAY_TIME = 2 ms against a 10 us run, so it advanced 0.25% and read as a
        # flatline, and it would have reported a quench even with the drain below off.
        # T_core is now the real kinetic temperature in both regimes, so the disruption
        # shows up because radiation genuinely removes the energy.
        T_core = T_core_kinetic

        if cfg.SPI_TRIGGERED:
            post_quench_keV = cfg.POST_QUENCH_TEMP / 1000.0
            # Feeding the MEASURED temperature back into P_rad closes the loop: as the
            # plasma cools, sqrt(T_e) falls and the impurity radiation weakens with it,
            # so the quench self-limits rather than following a prescribed trajectory.
            P_rad = compute_radiative_cooling_power(1.0e20, cfg.IMPURITY_DENSITY_NZ, T_core * 1000.0, cfg.RADIATIVE_COOLING_COEFF)

            # --- RADIATIVE ENERGY DRAIN (energy conservation) ---
            # P_rad used to be diagnostic-only: reported and plotted, but never taken out
            # of the particles, so the bulk kept its full kinetic energy through the
            # disruption -- quenched on the chart, pre-SPI in the velocity distribution.
            #
            # P_rad is a power DENSITY [W/m^3] at the reference density passed in above
            # (1e20 m^-3), so convert it to a per-step loss FRACTION against the thermal
            # energy density W = (3/2) * n_e * kT that same reference plasma stores. The
            # reference density cancels, which keeps this independent of macro_weight:
            #
            #   f = P_rad * dt / ((3/2) * n_ref * kT)
            #
            # kT is measured, so the drain stays proportional to the energy the particles
            # actually hold. Energy goes as v^2, so removing fraction f means scaling
            # every bulk velocity by sqrt(1 - f).
            T_kin_joules = T_core_kinetic * 1000.0 * cfg.e_charge
            W_thermal = 1.5 * 1.0e20 * T_kin_joules
            if W_thermal > 0.0 and np.any(thermals_and_nbi_mask):
                loss_fraction = float(P_rad) * cfg.reactor_dt / W_thermal
                # Radiation cools toward the post-quench floor, not through it: cap the
                # drain at the energy above POST_QUENCH_TEMP so a large P_rad can never
                # scale the velocities to or past zero.
                headroom = max(1.0 - post_quench_keV / max(T_core_kinetic, 1e-12), 0.0)
                loss_fraction = min(max(loss_fraction, 0.0), headroom)
                if loss_fraction > 0.0:
                    drain = np.sqrt(1.0 - loss_fraction)
                    vel_np[thermals_and_nbi_mask] = (
                        vel_np[thermals_and_nbi_mask].astype(np.float64) * drain
                    ).astype(np.float32)
        else:
            # No impurities before SPI fires, so nothing radiates
            P_rad = 0.0

        # energy_history_keV / T_core_kinetic above are recorded BEFORE the drain, which
        # is correct: the radiation covers [t, t+dt] and so lands in the next step's
        # ledger, not retroactively in this one.

        prof.add("alpha heating + energy transfer", _t)

        # One vectorized Numba call over the whole grid, replacing nR*nZ Python-level
        # calls into compute_radiation_losses.
        _t = prof.mark()
        n_e_grid_raw = np.abs(rho_grid / cfg.e_charge)
        max_n_e = np.max(n_e_grid_raw)
        scale_factor = 1.0e20 / (max_n_e + 1e-10)
        norm_n_e = 1.0 / (max_n_e + 1e-10)
        dR, dZ = (cfg.R_max - cfg.R_min) / (cfg.nR - 1), (cfg.Z_max - cfg.Z_min) / (cfg.nZ - 1)
        total_brem_watts, total_cyc_watts = compute_radiation_losses_grid(
            n_e_grid_raw, T_core, cfg.R_min, dR, dZ, cfg.nR, cfg.nZ, cfg.B0, cfg.Z_eff, scale_factor, norm_n_e,
            cfg.CYCLOTRON_REABSORPTION, cfg.R0_major
        )
        bremsstrahlung_power_history_MW.append(total_brem_watts / 1e6)
        cyclotron_power_history_MW.append(total_cyc_watts / 1e6)

        total_fus_MW = alpha_power_mw * 5.0
        q_sci, q_eng, p_elec_out, p_elec_in = evaluate_q_factors(total_fus_MW, cfg.EXTERNAL_HEATING_MW, cfg.eta_thermal, cfg.eta_heating)
        q_sci_history.append(q_sci)
        q_eng_history.append(q_eng)

        loss_rate = total_lost / t if t > 0 else 1e-5
        tau_E = min(current_confined / loss_rate if loss_rate > 0 else 0.5, 3.0)
        # --- TRIPLE PRODUCT DENSITY MUST MATCH THE Q-FACTOR NORMALISATION ---
        # n_e was hardcoded to 1e20 m^-3 here, which had nothing to do with the
        # macro-particle weight that scales alpha heating into the reactor-scale MW
        # feeding Q_sci on the panel directly above this one.
        # Deriving n_e from the same MACRO_WEIGHT_REACTOR and the psi_edge volume makes
        # the panels describe one plasma. Units are unchanged and already correct:
        # m^-3 * keV * s, matching cfg.lawson_target and the axis label.
        n_e_reactor = (current_confined * cfg.MACRO_WEIGHT_REACTOR) / cfg.PLASMA_VOLUME_M3
        lawson_triple_product_history.append(n_e_reactor * T_core * tau_E)

        temp_history.append(T_core)
        rad_power_history.append(P_rad)
        prof.add("radiation + Q-factor diagnostics", _t)

        if (step + 1) % 500 == 0:
            max_rho, max_phi = np.max(rho_grid), np.max(np.abs(phi_grid))
            print(f"  Step {step+1:04d}/{cfg.reactor_num_steps} | Confined: {current_confined:,} | Max Rho: {max_rho:.3e} | Max |Phi|: {max_phi:.2e} V")

    if profiling:
        loop_wall = time.perf_counter() - loop_wall_start
        prof.report(wall_total=loop_wall,
                    title="STAGE TIMING -- reactor loop (CPU / Numba)")
        push_prof.report(wall_total=prof.totals.get("deuteron Boris push"),
                         title="BREAKDOWN -- deuteron Boris push stage")
        _report_sor_histogram(sor_iter_counts)
        _report_peak_rss("[PROFILE][CPU]")

    # Tracked-particle metadata is returned as pid-keyed dicts, which removal cannot
    # corrupt -- row indices no longer survive a step now that losses are compacted out.
    # The loop accumulates into flat arrays; the dicts are rebuilt once, here, so the
    # return signature and package_reactor_results are unchanged.
    (history_tracks, tracked_type, tracked_lastpos,
     tracked_lastvel, tracked_lost) = tracks.to_dicts()
    return (pos_np, vel_np, type_np, history_tracks, tracked_type, tracked_lastpos, tracked_lastvel,
            tracked_lost,
            total_injected, total_lost,
            inventory_history, energy_history_keV, instability_amp_history, time_history,
            temp_history, rad_power_history, trigger_time, T_core,
            alpha_heating_power_history_MW, external_heating_power_history_MW,
            bremsstrahlung_power_history_MW, cyclotron_power_history_MW,
            q_sci_history, q_eng_history, lawson_triple_product_history, rho_grid, phi_grid)


def _run_reactor_loop_gpu(cfg, engine, pos_tensor, vel_tensor, type_tensor, rho_grid, phi_grid, E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid):
    # GPU-resident pipeline, selected only when GPU_PARTICLE_THRESHOLD is set to an int
    # <= cfg.initial_thermal_count. It is None by default, so this loop is opt-in -- see
    # that constant's definition for the numbers and for how to force it. The particle
    # tensors never leave the device inside the loop, apart from the small (nR x nZ) field
    # grid the CPU Poisson solver needs and the tracked-particle subset pulled every 20
    # steps for plotting.
    #
    # The push uses the DYNAMIC-shape compiled kernel where available, falling back to
    # eager if that compile failed at import. STATIC compile is wrong here: this loop's
    # masked subset shapes (mask_d / mask_a) change nearly every step as particles are
    # lost or injected, and static re-specializes per exact shape, so it would recompile
    # constantly -- it only pays off at fixed, reused shapes (see run_hpc_benchmark).
    # Dynamic handles the churn without recompiling and is worth ~4x on the push alone
    # (0.426 -> 0.098 ms at N=50,000). The loop as a whole is still slower than the CPU
    # path at every size tested, which is why GPU_PARTICLE_THRESHOLD is None.
    device = cfg.HPC_DEVICE

    pos_tensor = pos_tensor.to(device)
    vel_tensor = vel_tensor.to(device)
    type_tensor = type_tensor.to(device)

    # The psi grid is static for the whole run, so upload it once outside the loop
    psi_tensor = torch.tensor(engine.eq.psi_grid, device=device, dtype=torch.float32)
    psi_bounds = (engine.eq.psi_grid, engine.eq.psi_core, engine.eq.psi_R_min, engine.eq.psi_R_max,
                  engine.eq.psi_Z_min, engine.eq.psi_Z_max, engine.eq.psi_nR, engine.eq.psi_nZ)

    # Likewise the psi-derived poloidal B grids, reused by every gather. Without them the
    # gather sees a purely toroidal B with no |B| gradient structure -- no grad-B drift,
    # no mirror force, no trapped/passing separation in phase space.
    B_R_pol_t = torch.tensor(B_R_pol_grid, device=device, dtype=torch.float32)
    B_Z_pol_t = torch.tensor(B_Z_pol_grid, device=device, dtype=torch.float32)

    # One bulk transfer each: a per-index .cpu() in the dict comprehensions below would
    # be up to 1000 separate device syncs at startup.
    pos_np_init = pos_tensor.cpu().numpy()
    vel_np_init = vel_tensor.cpu().numpy()

    # Stable per-particle IDs, mirroring the CPU path. Lost particles are physically
    # removed from the tensors, which shifts every row index, so trajectory bookkeeping
    # cannot be keyed on row position. pid_tensor stays sorted ascending (IDs are
    # monotonic, mask compaction preserves order), so pid -> row is a searchsorted away.
    n_init = pos_tensor.shape[0]
    next_pid = n_init

    # pid -> current row, dense and ON DEVICE. Same reasoning as the CPU path:
    # torch.searchsorted needs pid_tensor sorted ascending, an invariant step 9
    # retires. Keeping the map on the device means compaction can renumber it
    # with a scatter and the sampler can gather rows without a sync.
    pid_capacity = _pid_capacity_bound(cfg, n_init)
    pid_row_t = torch.full((pid_capacity,), -1, device=device, dtype=torch.int64)
    pid_row_t[:n_init] = torch.arange(n_init, device=device, dtype=torch.int64)

    # Device-resident capacity pool, replacing the torch.cat growth below. Same
    # hard-ceiling contract as the CPU path: add() raises rather than growing.
    pool = _ParticlePool(pid_capacity, device=device)
    pool.add(pos_tensor, vel_tensor, 0,
             torch.arange(n_init, device=device, dtype=torch.int64))
    pool.type[:n_init] = type_tensor
    pos_tensor, vel_tensor, type_tensor, pid_tensor = pool.views()
    # Liveness and slot assignment stay on the HOST. That is what lets the host
    # know how many rows a sample will write without asking the device, so the
    # vertex write pointer advances with no .item() sync. Tracked particles die
    # only at the wall, and the compaction below already pulls exactly that
    # (small, pid-gated) set across, so the host copy stays in step cheaply.
    pid_slot = np.full(pid_capacity, -1, dtype=np.int64)
    # pid -> pool, host-side and write-once at birth; see the CPU path for why
    # compaction never touches it (species cannot migrate).
    pid_pool = np.full(pid_capacity, -1, dtype=np.int8)
    pid_pool[:n_init] = 0
    slot_alive = np.zeros(_MAX_TRACKED_SLOTS, dtype=bool)

    n_track = min(1000, n_init)
    # host_dtype float64: this path records its injection vertex straight from
    # the float64 array inject_neutral_beam_cartesian returns, before the cast
    # onto the device, and that is the value it recorded before this rewrite.
    tracks = _TrackStore(_MAX_TRACKED_SLOTS, device=device, host_dtype=np.float64)
    init_pids = np.arange(n_track, dtype=np.int64)
    init_slots = tracks.add_slots(init_pids, 0,
                                  pos_np_init[:n_track], vel_np_init[:n_track])
    pid_slot[init_pids] = init_slots
    slot_alive[init_slots] = True
    tracked_nbis = 0
    tracked_alphas = 0
    max_tracked_pid = n_track - 1

    total_injected = cfg.initial_thermal_count
    total_lost = 0
    inventory_history, energy_history_keV, instability_amp_history = [], [], []
    time_history, temp_history, rad_power_history = [], [], []
    trigger_time, T_core_baseline, T_core = None, 0.0, 0.0

    alpha_heating_power_history_MW, external_heating_power_history_MW = [], []
    bremsstrahlung_power_history_MW, cyclotron_power_history_MW = [], []
    q_sci_history, q_eng_history, lawson_triple_product_history = [], [], []

    # --- Profiling state (inert unless cfg.PROFILE) ---
    # MPS/CUDA queue work asynchronously, so every clock read on this path is
    # preceded by a device synchronize -- otherwise the numbers measure how
    # fast the CPU enqueued the work, not how long the GPU took to do it.
    profiling = bool(getattr(cfg, "PROFILE", False))
    _sync = None
    if profiling:
        if device.type == "mps":
            _sync = torch.mps.synchronize
        elif device.type == "cuda":
            _sync = torch.cuda.synchronize
    prof = StageProfiler(profiling, sync=_sync, label="GPU path")
    sor_iter_counts = []
    if profiling:
        _profile_header(cfg, f"GPU path ({device.type})", pos_tensor.shape[0])
        if _sync is None:
            print("   [WARN] no synchronize for this device: GPU stage times "
                  "will measure enqueue, not execution.")
    loop_wall_start = time.perf_counter() if profiling else None

    print("[SYSTEM] Igniting time-domain reactor loop on GPU (fastest path at this particle count)...")
    if device.type == "mps":
        print("[SYSTEM] First step will pause briefly — torch.compile is fusing the field-gather")
        print("[SYSTEM] kernel into a single Metal dispatch (one-time cost, not a hang).")

    for step in range(cfg.reactor_num_steps):
        t = step * cfg.reactor_dt
        time_history.append(t)

        # --- BATCH NBI INJECTION (tapering source) ---
        # Same decaying source as the CPU path, driven by the same two cfg knobs:
        # exp(-step/tau), with fractional rates realised stochastically.
        _t = prof.mark()
        if step % cfg.inject_every_n_steps == 0 and step > 0:
            nbi_rate = cfg.NBI_BATCH_SIZE * np.exp(-step / cfg.NBI_DECAY_TAU_STEPS)
            batch_size = int(nbi_rate)
            if np.random.random() < (nbi_rate - batch_size):
                batch_size += 1

            if batch_size > 0:
                p_nbi, v_nbi = engine.inject_neutral_beam_cartesian(num_ions=batch_size, E_keV=cfg.nbi_energy_keV, psi_bounds=psi_bounds)
                _require_pid_capacity(next_pid, batch_size, pid_capacity, "GPU NBI injection")
                new_pids = torch.arange(next_pid, next_pid + batch_size, device=device, dtype=torch.int64)
                base_row = pool.add(
                    torch.tensor(p_nbi, device=device, dtype=torch.float32),
                    torch.tensor(v_nbi, device=device, dtype=torch.float32),
                    1, new_pids)
                pos_tensor, vel_tensor, type_tensor, pid_tensor = pool.views()
                total_injected += batch_size

                pid_row_t[new_pids] = torch.arange(base_row, base_row + batch_size,
                                                   device=device, dtype=torch.int64)
                pid_pool[next_pid:next_pid + batch_size] = 0

                n_new = min(batch_size, 1000 - tracked_nbis)
                if n_new > 0:
                    host_pids = np.arange(next_pid, next_pid + n_new, dtype=np.int64)
                    slots = tracks.add_slots(host_pids, 1, p_nbi[:n_new], v_nbi[:n_new])
                    pid_slot[host_pids] = slots
                    slot_alive[slots] = True
                    max_tracked_pid = max(max_tracked_pid, int(host_pids[-1]))
                    tracked_nbis += n_new

                next_pid += batch_size

        prof.add("NBI injection", _t)

        # --- BATCH ALPHA INJECTION ---
        _t = prof.mark()
        if step % (cfg.inject_every_n_steps * 2) == 0 and step > 0:
            alpha_batch = 1
            phi_pos = np.random.uniform(0, 2 * np.pi, alpha_batch)
            R_birth = np.random.normal(cfg.R0_major, 0.05, alpha_batch)
            Z_birth = np.random.normal(0.0, 0.05, alpha_batch)
            aR, aZ, avR, avphi, avZ = engine.spawn_alpha_particles(alpha_batch, R_birth[0], Z_birth[0], cfg.ALPHA_ENERGY_JOULES, cfg.MASS_ALPHA)

            p_alpha = np.zeros((alpha_batch, 3), dtype=np.float32)
            v_alpha = np.zeros((alpha_batch, 3), dtype=np.float32)
            p_alpha[:, 0] = aR * np.cos(phi_pos)
            p_alpha[:, 1] = aR * np.sin(phi_pos)
            p_alpha[:, 2] = aZ
            v_alpha[:, 0] = avR * np.cos(phi_pos) - avphi * np.sin(phi_pos)
            v_alpha[:, 1] = avR * np.sin(phi_pos) + avphi * np.cos(phi_pos)
            v_alpha[:, 2] = avZ

            _require_pid_capacity(next_pid, alpha_batch, pid_capacity, "GPU alpha injection")
            new_pids = torch.arange(next_pid, next_pid + alpha_batch, device=device, dtype=torch.int64)
            base_row = pool.add(
                torch.tensor(p_alpha, device=device, dtype=torch.float32),
                torch.tensor(v_alpha, device=device, dtype=torch.float32),
                2, new_pids)
            pos_tensor, vel_tensor, type_tensor, pid_tensor = pool.views()

            pid_row_t[new_pids] = torch.arange(base_row, base_row + alpha_batch,
                                               device=device, dtype=torch.int64)
            pid_pool[next_pid:next_pid + alpha_batch] = 0

            n_new = min(alpha_batch, 1000 - tracked_alphas)
            if n_new > 0:
                host_pids = np.arange(next_pid, next_pid + n_new, dtype=np.int64)
                slots = tracks.add_slots(host_pids, 2, p_alpha[:n_new], v_alpha[:n_new])
                pid_slot[host_pids] = slots
                slot_alive[slots] = True
                max_tracked_pid = max(max_tracked_pid, int(host_pids[-1]))
                tracked_alphas += n_new

            next_pid += alpha_batch

        prof.add("alpha injection", _t)

        # --- CHARGE DENSITY MAPPING (GPU, eager) ---
        _t = prof.mark()
        mask_valid = (type_tensor == 0) | (type_tensor == 1)
        R_coords = torch.sqrt(pos_tensor[mask_valid, 0]**2 + pos_tensor[mask_valid, 1]**2)
        Z_coords = pos_tensor[mask_valid, 2]
        rho_grid_t = compute_cic_charge_density_torch(
            R_coords, Z_coords, cfg.e_charge,
            cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ, device
        )
        # Only the small (nR x nZ) grid crosses to the CPU -- the SOR Poisson solve is
        # inherently serial and stays on Numba/CPU whatever the particle count.
        rho_grid = rho_grid_t.cpu().numpy().astype(np.float64)
        prof.add("CIC deposition", _t)

        _t = prof.mark()
        if step > 0:
            # Warm-start from last step's converged phi, as in the CPU path
            phi_grid, E_R_grid, E_Z_grid, _sor_iters = engine.solve_fields(
                rho_grid, cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max,
                phi_init=phi_grid, return_iters=True)
            if profiling:
                sor_iter_counts.append(_sor_iters)
        prof.add("solve_fields (Poisson)", _t)

        _t = prof.mark()
        E_R_grid_t = torch.tensor(E_R_grid, device=device, dtype=torch.float32)
        E_Z_grid_t = torch.tensor(E_Z_grid, device=device, dtype=torch.float32)

        # --- E-FIELD GATHER + ANALYTIC B-FIELD (GPU, torch.compile'd, dynamic shapes) ---
        # Fed the psi-derived poloidal B grids and cfg.R0_major. That argument slot is R0
        # -- the toroidal 1/R falloff centre and tearing-mode axis -- and passing
        # cfg.B_poloidal (0.3) there put the field's centre outside the plasma entirely.
        E_tensor, B_tensor = vectorized_gather_and_B_torch(
            pos_tensor, E_R_grid_t, E_Z_grid_t, B_R_pol_t, B_Z_pol_t,
            cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ,
            cfg.B0, cfg.R0_major, t, cfg.b_perturb_initial, cfg.m_mode, cfg.n_mode, cfg.gamma_growth
        )
        prof.add("vectorized_gather_and_B", _t)

        # --- PARTICLE PUSH (GPU; see the note at the top of this function) ---
        _t = prof.mark()
        mask_d = (type_tensor == 0) | (type_tensor == 1)
        if torch.any(mask_d):
            _push = _vectorized_boris_push_metal_dynamic or _vectorized_boris_push_metal_impl
            p_d, v_d = _push(pos_tensor[mask_d], vel_tensor[mask_d], cfg.e_charge, cfg.m_deuterium, B_tensor[mask_d], E_tensor[mask_d], cfg.reactor_dt)
            pos_tensor[mask_d] = p_d
            vel_tensor[mask_d] = v_d
        prof.add("deuteron Boris push", _t)

        # Alphas are sub-stepped: at 3.5 MeV they cover ~1.3e-2 m per global 1 ns step
        # against a ~2.2e-2 m Larmor radius, under two samples per gyro-arc. Deuterons
        # stay on the single global step. E and B are held fixed across the sub-steps --
        # the point is to resolve gyration about the local B, not to re-gather the field.
        # Stays on-device; boris_push_substeps_torch re-enters the same MPS kernel.
        _t = prof.mark()
        mask_a = type_tensor == 2
        if torch.any(mask_a):
            p_a, v_a = boris_push_substeps_torch(
                pos_tensor[mask_a], vel_tensor[mask_a], cfg.CHARGE_ALPHA, cfg.MASS_ALPHA,
                B_tensor[mask_a], E_tensor[mask_a], cfg.reactor_dt, cfg.ALPHA_SUBSTEPS
            )
            pos_tensor[mask_a] = p_a
            vel_tensor[mask_a] = v_a
        prof.add("alpha substep push", _t)

        # --- COLLISIONS + CONFINEMENT CHECK (GPU, eager; psi-surface, not circular) ---
        _t = prof.mark()
        # MUST write back into the pool view rather than rebind. Unlike the numba
        # twin, which mutates vel_arr in place and returns the same array,
        # apply_vectorized_collisions_torch returns torch.where(...) -- a NEW
        # tensor -- whenever any collision fires. Rebinding vel_tensor to it
        # detaches the loop from pool.vel, so the pool keeps stale velocities and
        # the next injection's pool.views() silently reverts every update made
        # since. At nu_c*dt = 5e-6 with ~2,500 particles a collision fires only
        # about every 80 steps, which is what made this look like late-onset
        # chaotic drift rather than a lost write.
        vel_updated = apply_vectorized_collisions_torch(vel_tensor, type_tensor, cfg.nu_c, cfg.reactor_dt)
        if vel_updated is not vel_tensor:
            vel_tensor.copy_(vel_updated)
        prof.add("apply_vectorized_collisions", _t)

        _t = prof.mark()
        type_tensor, newly_lost = check_confinement_torch(
            pos_tensor, type_tensor, psi_tensor, engine.eq.psi_edge,
            engine.eq.psi_R_min, engine.eq.psi_R_max, engine.eq.psi_Z_min, engine.eq.psi_Z_max,
            engine.eq.psi_nR, engine.eq.psi_nZ
        )
        prof.add("check_confinement_flux", _t)
        total_lost += newly_lost

        # --- WALL LOSS COMPACTION (GPU) ---
        _t = prof.mark()
        # check_confinement_torch only FLAGS wall strikes (type = -1). Leaving them in the
        # tensors meant the pools only ever grew, every kernel paid for dead particles,
        # and the confined count could never fall. Boolean-mask indexing keeps the removal
        # a single on-device gather per tensor -- only the last-known state of tracked
        # particles that just died round-trips to the CPU.
        if newly_lost > 0:
            alive = type_tensor != -1

            # Preserve the last known state of any TRACKED particle before it is dropped.
            # Gated on pid <= max_tracked_pid so a large loss event doesn't drag the whole
            # dead population across the memory boundary to find the few plotted pids.
            dying_tracked = (~alive) & (pid_tensor <= max_tracked_pid)
            if bool(dying_tracked.any()):
                d_rows = torch.nonzero(dying_tracked, as_tuple=False).squeeze(1)
                d_pids = pid_tensor[d_rows].cpu().numpy()
                d_pos = pos_tensor[d_rows].cpu().numpy()
                d_vel = vel_tensor[d_rows].cpu().numpy()
                d_slots = pid_slot[d_pids]
                is_tracked = d_slots >= 0
                if np.any(is_tracked):
                    t_slots = d_slots[is_tracked]
                    # End the crimson trace at the wall, not at the last sampling
                    # tick. Always terminal: slot_alive goes False here, so the
                    # sampler never selects the slot again.
                    tracks.record_impact(t_slots, d_pos[is_tracked], d_vel[is_tracked])
                    slot_alive[t_slots] = False

            dead_pids_t = pid_tensor[~alive]
            # In-place compaction into the same device buffers, replacing four
            # boolean-mask reallocations per loss event.
            alive_rows = torch.nonzero(alive, as_tuple=False).squeeze(1)
            pool.compact(alive_rows)
            pos_tensor, vel_tensor, type_tensor, pid_tensor = pool.views()
            # Renumber the device-side pid -> row map: dead pids to -1, then one
            # scatter giving every survivor its new row. Replaces the per-step
            # torch.searchsorted the sampler used to run. pid_pool is untouched --
            # rows move, pools do not.
            pid_row_t[dead_pids_t] = -1
            pid_row_t[pid_tensor] = torch.arange(pid_tensor.shape[0],
                                                 device=device, dtype=torch.int64)
        prof.add("wall-loss compaction", _t)

        # Pull only the tracked particles (<=2000) needed for trajectory plots, never the
        # full inventory. Keyed by stable pid via searchsorted, so the row shift from the
        # compaction above cannot silently re-point a track at a different particle.
        # Alphas are sampled far more often than thermals: their orbit is only ~2.2e-2 m
        # across, so a 20-step cadence (~0.26 m of travel) aliases it away.
        _t = prof.mark()
        sample_thermal = (step % 20 == 0)
        sample_alpha = (step % cfg.ALPHA_HISTORY_EVERY == 0)
        if (sample_thermal or sample_alpha) and tracks.n_slots > 0 and pid_tensor.numel() > 0:
            # The whole selection -- which slots are due, which are still alive,
            # and therefore HOW MANY rows this sample writes -- is decided on the
            # host, from slot_alive. That is what keeps the write pointer free of
            # any .item() sync. Only the gather runs on the device, and the
            # sampled coordinates stay there: the old form downloaded two
            # (n_tracked, 3) arrays every sampling step and walked them in Python.
            due = tracks.slots_due(sample_thermal, sample_alpha)
            if due.size > 0:
                sel_slots = due[slot_alive[due]]
                if sel_slots.size > 0:
                    sel_slots_t = torch.from_numpy(sel_slots).to(device)
                    track_pids_t = torch.from_numpy(tracks.pids[sel_slots]).to(device)
                    rows_t = pid_row_t[track_pids_t]
                    sampled_pos = pos_tensor[rows_t]
                    tracks.append_samples(sel_slots, sampled_pos)
                    tracks.set_last_device(sel_slots, sel_slots_t,
                                           sampled_pos, vel_tensor[rows_t])

        prof.add("trajectory sampling", _t)

        # --- ALPHA HEATING (GPU, eager) ---
        _t = prof.mark()
        mask_alphas = type_tensor == 2
        alpha_deposited_kev = 0.0
        if torch.any(mask_alphas):
            alpha_vels = vel_tensor[mask_alphas]
            v_mags = torch.linalg.norm(alpha_vels, dim=1)
            alpha_energies_kev = (0.5 * cfg.MASS_ALPHA * v_mags**2) / 1.602e-16
            # Third return value is the raw simulation-scale keV drained, needed by the
            # energy transfer below.
            new_energies_kev, alpha_power_mw, alpha_deposited_kev = compute_alpha_heating_power_torch(
                alpha_energies_kev, cfg.reactor_dt, cfg)
            new_v_mags = torch.sqrt(2.0 * (new_energies_kev * 1.602e-16) / cfg.MASS_ALPHA)
            safe_v_mags = torch.where(v_mags == 0, torch.full_like(v_mags, 1e-10), v_mags)
            scale_factors = new_v_mags / safe_v_mags
            vel_tensor[mask_alphas] = alpha_vels * scale_factors.unsqueeze(1)
        else:
            alpha_power_mw = 0.0

        alpha_heating_power_history_MW.append(alpha_power_mw)
        external_heating_power_history_MW.append(cfg.EXTERNAL_HEATING_MW)

        thermals_and_nbi_mask = (type_tensor == 0) | (type_tensor == 1)
        current_confined = int(thermals_and_nbi_mask.sum().item())
        inventory_history.append(current_confined)

        # --- ALPHA -> BULK ENERGY TRANSFER (energy conservation) ---
        # Energy drained from the alphas used to vanish -- removed from the fast
        # population and given to nothing, so the thermal plasma never felt the heating.
        # Deposit it by scaling bulk speeds, using the simulation-scale keV (no
        # macro_weight) so particles stay self-consistent; the MW figure above is
        # separately scaled for reactor-equivalent output.
        if alpha_deposited_kev > 0.0 and torch.any(thermals_and_nbi_mask):
            bulk_vels = vel_tensor[thermals_and_nbi_mask]
            bulk_energy_kev = float((0.5 * cfg.m_deuterium * torch.sum(bulk_vels**2, dim=1)).sum().item()) / 1.602e-16
            if bulk_energy_kev > 0.0:
                boost = float(np.sqrt(1.0 + alpha_deposited_kev / bulk_energy_kev))
                vel_tensor[thermals_and_nbi_mask] = bulk_vels * boost

        if torch.any(thermals_and_nbi_mask):
            current_energy_joules = float((0.5 * cfg.m_deuterium * torch.sum(vel_tensor[thermals_and_nbi_mask]**2, dim=1)).sum().item())
        else:
            current_energy_joules = 0.0

        current_energy_keV = current_energy_joules / (cfg.e_charge * 1000.0)
        energy_history_keV.append(current_energy_keV)

        current_amp = cfg.b_perturb_initial * np.exp(cfg.gamma_growth * t)
        instability_amp_history.append(current_amp)

        # --- T_core IS A TEMPERATURE, NOT A TOTAL ---
        # Assigning the plasma's TOTAL stored energy here made the "core temperature"
        # read ~50,000 keV, and every consumer (brem's sqrt(T_e), the Lawson triple
        # product, the quench baseline) was fed that. For a 3D Maxwellian
        # <E> = (3/2)kT, so kT = (2/3) * <E>.
        n_bulk = max(current_confined, 1)
        T_core_kinetic = (2.0 / 3.0) * (current_energy_keV / n_bulk)

        if current_amp > cfg.MAX_ISLAND_WIDTH_THRESHOLD and not cfg.SPI_TRIGGERED:
            cfg.SPI_TRIGGERED = True
            trigger_time = t
            T_core_baseline = T_core_kinetic
            print(f"  [EMERGENCY] Magnetic Island exceeded {cfg.MAX_ISLAND_WIDTH_THRESHOLD*100}% minor radius!")
            print(f"  [SYSTEM] Firing Shattered Pellet Injection (SPI) at t={t*1000:.2f} ms...")
            print(f"  [SYSTEM] Pre-disruption core temperature: {T_core_baseline:.3f} keV")

        # --- REPORTED TEMPERATURE IS MEASURED, NOT PRESCRIBED ---
        # Post-trigger T_core used to come from trigger_thermal_quench()'s analytic
        # exponential -- a curve drawn on top of the simulation. It ran on
        # TQ_DECAY_TIME = 2 ms against a 10 us run, so it advanced 0.25% and read as a
        # flatline, and it would have reported a quench even with the drain below off.
        # T_core is now the real kinetic temperature in both regimes, so the disruption
        # shows up because radiation genuinely removes the energy.
        T_core = T_core_kinetic

        if cfg.SPI_TRIGGERED:
            post_quench_keV = cfg.POST_QUENCH_TEMP / 1000.0
            # Feeding the MEASURED temperature back into P_rad closes the loop: as the
            # plasma cools, sqrt(T_e) falls and the impurity radiation weakens with it,
            # so the quench self-limits rather than following a prescribed trajectory.
            P_rad = compute_radiative_cooling_power(1.0e20, cfg.IMPURITY_DENSITY_NZ, T_core * 1000.0, cfg.RADIATIVE_COOLING_COEFF)

            # --- RADIATIVE ENERGY DRAIN (energy conservation) ---
            # P_rad used to be diagnostic-only: reported and plotted, but never taken out
            # of the particles, so the bulk kept its full kinetic energy through the
            # disruption -- quenched on the chart, pre-SPI in the velocity distribution.
            #
            # P_rad is a power DENSITY [W/m^3] at the reference density passed in above
            # (1e20 m^-3), so convert it to a per-step loss FRACTION against the thermal
            # energy density W = (3/2) * n_e * kT that same reference plasma stores. The
            # reference density cancels, which keeps this independent of macro_weight:
            #
            #   f = P_rad * dt / ((3/2) * n_ref * kT)
            #
            # kT is measured, so the drain stays proportional to the energy the particles
            # actually hold. Energy goes as v^2, so removing fraction f means scaling
            # every bulk velocity by sqrt(1 - f).
            T_kin_joules = T_core_kinetic * 1000.0 * cfg.e_charge
            W_thermal = 1.5 * 1.0e20 * T_kin_joules
            if W_thermal > 0.0 and torch.any(thermals_and_nbi_mask):
                loss_fraction = float(P_rad) * cfg.reactor_dt / W_thermal
                # Radiation cools toward the post-quench floor, not through it: cap the
                # drain at the energy above POST_QUENCH_TEMP so a large P_rad can never
                # scale the velocities to or past zero.
                headroom = max(1.0 - post_quench_keV / max(T_core_kinetic, 1e-12), 0.0)
                loss_fraction = min(max(loss_fraction, 0.0), headroom)
                if loss_fraction > 0.0:
                    # float32 is fine here: f runs ~6e-5/step at the few-keV temperatures
                    # this reactor reaches, ~500x above float32 epsilon, so unlike the
                    # alpha-drag drain this multiply does not round back to the original.
                    drain = float(np.sqrt(1.0 - loss_fraction))
                    vel_tensor[thermals_and_nbi_mask] = vel_tensor[thermals_and_nbi_mask] * drain
        else:
            # No impurities before SPI fires, so nothing radiates
            P_rad = 0.0

        prof.add("alpha heating + energy transfer", _t)

        # energy_history_keV / T_core_kinetic above are recorded BEFORE the drain, which
        # is correct: the radiation covers [t, t+dt] and so lands in the next step's
        # ledger, not retroactively in this one.
        _t = prof.mark()
        n_e_grid_raw = np.abs(rho_grid / cfg.e_charge)
        max_n_e = np.max(n_e_grid_raw)
        scale_factor = 1.0e20 / (max_n_e + 1e-10)
        norm_n_e = 1.0 / (max_n_e + 1e-10)
        dR, dZ = (cfg.R_max - cfg.R_min) / (cfg.nR - 1), (cfg.Z_max - cfg.Z_min) / (cfg.nZ - 1)
        # Pass the cyclotron re-absorption factor and the real major radius explicitly --
        # falling back on the defaults gave the GPU path a different radiation balance
        # from the CPU path.
        total_brem_watts, total_cyc_watts = compute_radiation_losses_grid(
            n_e_grid_raw, T_core, cfg.R_min, dR, dZ, cfg.nR, cfg.nZ, cfg.B0, cfg.Z_eff, scale_factor, norm_n_e,
            cfg.CYCLOTRON_REABSORPTION, cfg.R0_major
        )
        bremsstrahlung_power_history_MW.append(total_brem_watts / 1e6)
        cyclotron_power_history_MW.append(total_cyc_watts / 1e6)

        total_fus_MW = alpha_power_mw * 5.0
        q_sci, q_eng, p_elec_out, p_elec_in = evaluate_q_factors(total_fus_MW, cfg.EXTERNAL_HEATING_MW, cfg.eta_thermal, cfg.eta_heating)
        q_sci_history.append(q_sci)
        q_eng_history.append(q_eng)

        loss_rate = total_lost / t if t > 0 else 1e-5
        tau_E = min(current_confined / loss_rate if loss_rate > 0 else 0.5, 3.0)
        # --- TRIPLE PRODUCT DENSITY MUST MATCH THE Q-FACTOR NORMALISATION ---
        # n_e was hardcoded to 1e20 m^-3 here, which had nothing to do with the
        # macro-particle weight that scales alpha heating into the reactor-scale MW
        # feeding Q_sci on the panel directly above this one. 
        # Deriving n_e from the same MACRO_WEIGHT_REACTOR and the psi_edge volume makes
        # the panels describe one plasma. Units are unchanged and already correct:
        # m^-3 * keV * s, matching cfg.lawson_target and the axis label.
        n_e_reactor = (current_confined * cfg.MACRO_WEIGHT_REACTOR) / cfg.PLASMA_VOLUME_M3
        lawson_triple_product_history.append(n_e_reactor * T_core * tau_E)

        temp_history.append(T_core)
        rad_power_history.append(P_rad)
        prof.add("radiation + Q-factor diagnostics", _t)

        if profiling and (step + 1) % 500 == 0:
            _log_device_memory(device, step + 1)

        if (step + 1) % 500 == 0:
            max_rho, max_phi = np.max(rho_grid), np.max(np.abs(phi_grid))
            print(f"  Step {step+1:04d}/{cfg.reactor_num_steps} | Confined: {current_confined:,} | Max Rho: {max_rho:.3e} | Max |Phi|: {max_phi:.2e} V")

    if profiling:
        if _sync is not None:
            _sync()
        loop_wall = time.perf_counter() - loop_wall_start
        prof.report(wall_total=loop_wall,
                    title=f"STAGE TIMING -- reactor loop (GPU / {device.type})")
        _report_sor_histogram(sor_iter_counts)
        _log_device_memory(device, cfg.reactor_num_steps, final=True)
        _report_peak_rss("[PROFILE][GPU]")

    # --- Leave the GPU only here, at the end of the loop ---
    pos_np = pos_tensor.cpu().numpy()
    vel_np = vel_tensor.cpu().numpy()
    type_np = type_tensor.cpu().numpy()

    # Flat-array accumulation -> the dict payload, once. This is the only place
    # the sampled vertices and the device-side last_pos/last_vel cross to the host.
    (history_tracks, tracked_type, tracked_lastpos,
     tracked_lastvel, tracked_lost) = tracks.to_dicts()
    return (pos_np, vel_np, type_np, history_tracks, tracked_type, tracked_lastpos, tracked_lastvel,
            tracked_lost,
            total_injected, total_lost,
            inventory_history, energy_history_keV, instability_amp_history, time_history,
            temp_history, rad_power_history, trigger_time, T_core,
            alpha_heating_power_history_MW, external_heating_power_history_MW,
            bremsstrahlung_power_history_MW, cyclotron_power_history_MW,
            q_sci_history, q_eng_history, lawson_triple_product_history, rho_grid, phi_grid)


def package_reactor_results(cfg, eq, engine, B_R_pol_grid, B_Z_pol_grid,
                            pos_np, vel_np, type_np,
                            history_tracks, tracked_type, tracked_lastpos,
                            tracked_lastvel, tracked_lost,
                            rho_grid, T_core, prof=None):
    """Turns the reactor loop's raw output into the arrays diagnostics plots.

    Lifted verbatim out of run_reactor_steady_state so a regression test can
    reach these products without rendering anything: the loop is the physics,
    this is the packaging, and rendering is a third step. run_reactor_steady_state
    calls this and forwards the contents to diagnostics exactly as before.

    `eq` is not read here today -- the poloidal field it produced arrives
    already resampled as B_R_pol_grid/B_Z_pol_grid. It stays in the signature
    so callers pass the same equilibrium object they pass everywhere else, and
    so a future moment that needs psi does not change every call site.

    Returns a dict of exactly the products the caller forwards onward.

    `prof` is an optional StageProfiler. When None (the default, and what
    tests/test_regression.py passes) every hook below is a no-op.
    """
    if prof is None:
        prof = StageProfiler(False)
    dR, dZ = (cfg.R_max - cfg.R_min) / (cfg.nR - 1), (cfg.Z_max - cfg.Z_min) / (cfg.nZ - 1)

    print("[SYSTEM] Packaging particle arrays for Graphing Utilities...")
    _t = prof.mark()

    # history_tracks is keyed by particle ID, not row index: lost particles are compacted
    # out of pos_np/vel_np, so indexing those by a track key would read an unrelated
    # particle (and IndexError once the pools shrink past it). Final per-track state comes
    # from the pid-keyed snapshots, so a particle that hit the wall keeps the state it had
    # then. The dict payload handed to diagnostics is unchanged.
    mock_active = []
    mock_alphas = []
    for pid, hist in history_tracks.items():
        ptype = tracked_type.get(pid, 0)
        p_dict = {
            "history": hist,
            "type": ptype,
            # diagnostics._scrub_particles reads "status"; without it written here every
            # particle defaults to "confined" and the wall-loss colouring never fires.
            "status": "lost" if pid in tracked_lost else "confined",
            "pos": tracked_lastpos[pid],
            "vel": tracked_lastvel[pid],
        }
        if ptype == 2:
            mock_alphas.append(p_dict)
            # A lost alpha still belongs on the wall-loss plots: the alpha-orbit plot keys
            # off species, the 2D/3D reactor plots off status.
            if p_dict["status"] == "lost":
                mock_active.append(p_dict)
        else:
            mock_active.append(p_dict)

    prof.add("pkg: history_tracks -> mock_active/mock_alphas", _t)

    # NOTE: this builds one Python dict per surviving particle, then
    # compute_fluid_moments walks that list in Python. There is no
    # array-based path here yet, so at 1M particles both lines below are a
    # real chunk of wall clock -- they are timed separately to make that
    # visible rather than to hide it inside one "packaging" number.
    _t = prof.mark()
    final_active = [{"status": "confined", "pos": p, "vel": v} for p, v, typ in zip(pos_np, vel_np, type_np) if typ != -1]
    prof.add("pkg: final_active (dict per particle)", _t)

    _t = prof.mark()
    R_centers, density_profile, pressure_profile = engine.compute_fluid_moments(
        final_active, cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.num_radial_bins
    )
    prof.add("pkg: compute_fluid_moments", _t)

    # --- TRUE v_parallel FOR THE PHASE-SPACE MAP ---
    # (x*vy - y*vx)/R is v_TOROIDAL, which only equals v_parallel if B is purely toroidal
    # -- and pitch angle against the ACTUAL field line is the whole quantity separating
    # trapped from passing. Projecting onto the real local b_hat (toroidal + psi-derived
    # poloidal) is what makes that boundary appear instead of a Gaussian blob.
    print("[SYSTEM] Projecting velocities onto local field lines for phase-space map...")
    _t = prof.mark()
    alive_mask = type_np != -1
    R_phase, v_parallel_phase = [], []
    if np.any(alive_mask):
        pos_alive = np.ascontiguousarray(pos_np[alive_mask], dtype=np.float32)
        vel_alive = np.ascontiguousarray(vel_np[alive_mask], dtype=np.float32)
        zero_grid = np.zeros((cfg.nR, cfg.nZ), dtype=np.float64)
        # b_perturb = 0: the pitch angle should be defined by the equilibrium field line,
        # not the instantaneous tearing-mode ripple.
        _, B_final = vectorized_gather_and_B(
            pos_alive, zero_grid, zero_grid, B_R_pol_grid, B_Z_pol_grid,
            cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ,
            cfg.B0, cfg.R0_major, 0.0, 0.0, cfg.m_mode, cfg.n_mode, 0.0
        )
        B_mag = np.linalg.norm(B_final, axis=1)
        safe_B = np.where(B_mag > 0.0, B_mag, 1.0)
        b_hat = B_final / safe_B[:, np.newaxis]
        v_par_all = np.sum(vel_alive * b_hat, axis=1)
        R_all = np.sqrt(pos_alive[:, 0]**2 + pos_alive[:, 1]**2)
        keep = (R_all > 0) & (B_mag > 0.0)
        R_phase = R_all[keep].tolist()
        v_parallel_phase = v_par_all[keep].tolist()

    prof.add("pkg: v_parallel phase-space projection", _t)

    print("[SYSTEM] Mapping Volumetric Fusion Power ")
    _t = prof.mark()
    n_e_grid_raw = np.abs(rho_grid / cfg.e_charge)
    scale_factor = 1.0e20 / (np.max(n_e_grid_raw) + 1e-10)
    n_e_grid_scaled = n_e_grid_raw * scale_factor
    n_D_grid = n_T_grid = 0.5 * n_e_grid_scaled
    T_i_grid_keV = T_core * (n_e_grid_raw / (np.max(n_e_grid_raw) + 1e-10))
    P_fusion_grid = compute_volumetric_fusion_power(n_D_grid, n_T_grid, T_i_grid_keV, cfg)

    total_fusion_power_watts = 0.0
    for i in range(cfg.nR):
        R_i = cfg.R_min + i * dR
        cell_volume = 2.0 * np.pi * R_i * dR * dZ if R_i > 0 else 1.0
        for j in range(cfg.nZ):
            total_fusion_power_watts += P_fusion_grid[i, j] * cell_volume
            
    prof.add("pkg: fusion power grid + volume integral", _t)

    print("==================================================")
    print(f" TOTAL INTEGRATED FUSION POWER: {total_fusion_power_watts / 1e6:.2f} MW ")
    print("==================================================")

    return {
        "mock_active": mock_active,
        "mock_alphas": mock_alphas,
        "final_active": final_active,
        "R_centers": R_centers,
        "density_profile": density_profile,
        "pressure_profile": pressure_profile,
        "R_phase": R_phase,
        "v_parallel_phase": v_parallel_phase,
        "P_fusion_grid": P_fusion_grid,
        "total_fusion_power_watts": total_fusion_power_watts,
    }


def run_reactor_steady_state(save_plots=None):
    """Run the reactor loop end to end.

    save_plots gates only the PNG-writing diagnostic calls at the tail. None
    (the default) resolves to `not cfg.PROFILE`, so a profiling run never
    overwrites the committed plots in the repo root; pass True to force the
    plots out of a profiling run, False to suppress them in a normal one.

    The diagnostic *computation* is deliberately not gated:
    package_reactor_results (final_active, compute_fluid_moments, the
    phase-space projection) runs either way, so profile numbers stay
    comparable to the step 4/5/6 baselines.
    """
    print("==================================================")
    print("      PIC FULL CYCLE (LORENTZ PUSH)      ")
    print("==================================================")

    cfg = SimulationConfiguration()

    if save_plots is None:
        save_plots = not bool(getattr(cfg, "PROFILE", False))

    # Out-of-loop phases are timed on their own profiler, so the loop table
    # stays a table about the loop.
    outer = StageProfiler(bool(getattr(cfg, "PROFILE", False)), label="out-of-loop")

    _t = outer.mark()
    eq, engine, pos_tensor, vel_tensor, type_tensor, rho_grid, phi_grid, E_R_grid, E_Z_grid = initialization.initialize_reactor(cfg)
    outer.add("initialize_reactor", _t)

    # The psi-derived poloidal B field, resampled onto the solver grid so the pusher's B
    # lookup reuses the indices/weights the E gather computes. Without it the pusher sees
    # a purely toroidal B whose only gradient is the smooth 1/R falloff -- no mirror ratio
    # along a field line, so nothing is trapped and the v_parallel-vs-R map collapses to
    # the injected Maxwellian.
    _t = outer.mark()
    B_R_pol_grid, B_Z_pol_grid, pol_scale, B_pol_target = compute_poloidal_field_grids(
        eq, cfg.B0, R0=cfg.R0_major, a_minor=0.3, q_target=cfg.Q_SAFETY_TARGET,
        dst_R_min=cfg.R_min, dst_R_max=cfg.R_max, dst_Z_min=cfg.Z_min, dst_Z_max=cfg.Z_max,
        dst_nR=cfg.nR, dst_nZ=cfg.nZ
    )
    outer.add("compute_poloidal_field_grids", _t)
    print(f"[SYSTEM] Poloidal field from Grad-Shafranov psi: B_pol ~ {B_pol_target:.3f} T "
          f"(q = {cfg.Q_SAFETY_TARGET:.1f}, psi rescale x{pol_scale:.3e})")

    # GPU_PARTICLE_THRESHOLD is None, so this always selects the CPU loop: the CPU loop
    # measured faster at every count tested, up to 3,000,000 (step 6, at 1,000,000
    # particles / 200 steps: GPU loop 69.6 s wall vs CPU loop ~32-36 s). The `is not None`
    # guard short-circuits before the comparison, so None never reaches the `>=`.
    # See the constant's definition for how to force the GPU loop.
    loop_fn = (_run_reactor_loop_gpu
               if GPU_PARTICLE_THRESHOLD is not None and cfg.initial_thermal_count >= GPU_PARTICLE_THRESHOLD
               else _run_reactor_loop_cpu)
    (pos_np, vel_np, type_np, history_tracks, tracked_type, tracked_lastpos, tracked_lastvel,
     tracked_lost,
     total_injected, total_lost,
     inventory_history, energy_history_keV, instability_amp_history, time_history,
     temp_history, rad_power_history, trigger_time, T_core,
     alpha_heating_power_history_MW, external_heating_power_history_MW,
     bremsstrahlung_power_history_MW, cyclotron_power_history_MW,
     q_sci_history, q_eng_history, lawson_triple_product_history, rho_grid, phi_grid
     ) = loop_fn(cfg, engine, pos_tensor, vel_tensor, type_tensor, rho_grid, phi_grid, E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid)

    _t = outer.mark()
    packaged = package_reactor_results(
        cfg, eq, engine, B_R_pol_grid, B_Z_pol_grid,
        pos_np, vel_np, type_np,
        history_tracks, tracked_type, tracked_lastpos,
        tracked_lastvel, tracked_lost,
        rho_grid, T_core, prof=outer,
    )
    outer.add("package_reactor_results (total)", _t)

    if outer.enabled:
        _report_gs_provenance(eq)
        # package_reactor_results' sub-stages are charged to `outer` as well,
        # so its own "(total)" row overlaps them -- read that row as the
        # roll-up, not as another slice to add in.
        outer.report(title="STAGE TIMING -- out-of-loop phases")
    mock_active = packaged["mock_active"]
    mock_alphas = packaged["mock_alphas"]
    R_centers = packaged["R_centers"]
    density_profile = packaged["density_profile"]
    pressure_profile = packaged["pressure_profile"]
    R_phase = packaged["R_phase"]
    v_parallel_phase = packaged["v_parallel_phase"]
    P_fusion_grid = packaged["P_fusion_grid"]

    # Every diagnostic below writes a PNG into the repo root under its fixed
    # name (plot_manifest.py's baseline depends on those names), so they are
    # the only thing save_plots gates.
    if not save_plots:
        print("[SYSTEM] save_plots is off -- skipping diagnostic plots "
              "(no PNGs written).")
    else:
        diagnostics.run_steady_state_diagnostics(
            cfg, eq, rho_grid, phi_grid, energy_history_keV, 
            mock_active, total_injected, total_lost, inventory_history,
            R_centers, density_profile, pressure_profile,
            R_phase, v_parallel_phase, instability_amp_history, P_fusion_grid,
            alpha_particles=mock_alphas, 
            alpha_power_history=alpha_heating_power_history_MW,
            ext_power_history=external_heating_power_history_MW,
            brem_power_history=bremsstrahlung_power_history_MW, 
            cyc_power_history=cyclotron_power_history_MW,        
            q_sci_history=q_sci_history, q_eng_history=q_eng_history, lawson_history=lawson_triple_product_history 
        )

        if cfg.SPI_TRIGGERED:
            diagnostics.plot_disruption_mitigation(time_history, temp_history, rad_power_history, trigger_time)


def run_plasma_oscillation_test():
    print("==================================================")
    print("  HYBRID TENSOR OSCILLATIONS & SHIELDING ")
    print("==================================================")
    
    cfg = SimulationConfiguration()
    eq, engine, pos_tensor, vel_tensor, type_tensor, rho_grid = initialization.initialize_oscillation_test(cfg)

    # A purely electrostatic unit test: it measures collective Langmuir ringing against
    # the toroidal field only, so the gather gets zero B_pol grids rather than the
    # Grad-Shafranov ones.
    B_R_pol_grid = np.zeros((cfg.nR, cfg.nZ), dtype=np.float64)
    B_Z_pol_grid = np.zeros((cfg.nR, cfg.nZ), dtype=np.float64)
    hpc_engine = HPCPhysicsAccelerator(cfg.HPC_DEVICE)
    
    w_es_history, time_arr = [], []
    pos_np = pos_tensor.cpu().numpy()
    vel_np = vel_tensor.cpu().numpy()
    type_np = type_tensor.cpu().numpy()
    
    print("[SYSTEM] Running Unit Test: Capturing collective electrostatic ringing...")
    
    for step in range(cfg.osc_num_steps):
        mask_valid = type_np != -1
        R_coords = np.sqrt(pos_np[mask_valid, 0]**2 + pos_np[mask_valid, 1]**2)
        Z_coords = pos_np[mask_valid, 2]
        charges = np.full(np.sum(mask_valid), -cfg.e_charge * cfg.macro_weight)
        
        rho_grid = compute_cic_charge_density(R_coords, Z_coords, charges, cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ, True)
        phi_grid, E_R_grid, E_Z_grid = engine.solve_fields(rho_grid, cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max)
        
        w_es = compute_electrostatic_energy(E_R_grid, E_Z_grid, cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max)
        w_es_history.append(w_es)
        time_arr.append(step * cfg.osc_dt)
        
        E_np, B_np = vectorized_gather_and_B(
            pos_np, E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid,
            cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ,
            cfg.B0, cfg.R0_major, 0.0, 0.0, 1, 1, 0.0
        )
        
        pos_tensor = torch.tensor(pos_np, device=cfg.HPC_DEVICE)
        vel_tensor = torch.tensor(vel_np, device=cfg.HPC_DEVICE)
        E_tensor = torch.tensor(E_np, device=cfg.HPC_DEVICE)
        B_tensor = torch.tensor(B_np, device=cfg.HPC_DEVICE)
        
        p_e, v_e = hpc_engine.vectorized_boris_push_metal(pos_tensor, vel_tensor, -cfg.e_charge * cfg.macro_weight, cfg.m_electron * cfg.macro_weight, B_tensor, E_tensor, cfg.osc_dt)
        
        pos_np, vel_np = p_e.cpu().numpy(), v_e.cpu().numpy()
        # Unified on the psi surface, same as the reactor loop
        check_confinement_flux(
            pos_np, type_np, eq.psi_grid, eq.psi_edge,
            eq.psi_R_min, eq.psi_R_max, eq.psi_Z_min, eq.psi_Z_max,
            eq.psi_nR, eq.psi_nZ
        )
        
        if (step + 1) % 250 == 0:
            print(f"  Step {step+1:04d}/{cfg.osc_num_steps} | W_ES: {w_es:.3e} Joules | E_Z Max: {np.max(np.abs(E_Z_grid)):.2e} V/m")

    print("==================================================")
    print("  TEST COMPLETE: GENERATING DIAGNOSTIC ARTIFACTS  ")
    print("==================================================")
    diagnostics.run_oscillation_diagnostics(cfg, time_arr, w_es_history)

def run_nuclear_reaction_dynamics():
    print("==================================================")
    print("  EVALUATING D-T QUANTUM CROSS-SECTION   ")
    print("==================================================")
    E_kev_arr = np.linspace(1.0, 200.0, 500)
    sigma_arr = np.array([compute_dt_cross_section(e) for e in E_kev_arr])
    diagnostics.plot_fusion_cross_section(E_kev_arr, sigma_arr)

if __name__ == "__main__":
    cfg = SimulationConfiguration()
    run_hpc_benchmark(cfg)
    # The full-artifact entry point: it exists to regenerate the plots, so it
    # asks for them explicitly rather than inheriting the cfg.PROFILE default.
    run_reactor_steady_state(save_plots=True)
    run_plasma_oscillation_test()
    run_nuclear_reaction_dynamics()