"""Per-stage wall-clock instrumentation, split out of main.py (step 14).

Leaf module: imports nothing from this project.
"""

import numpy as np
import time
import numba
import sys
import torch


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
