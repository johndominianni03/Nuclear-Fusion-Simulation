#!/usr/bin/env python3
"""Compare the numba Grad-Shafranov solver against the Python reference.

The numba SOR path is compiled with fastmath=True, which lets the compiler
reassociate floating-point arithmetic -- so it is NOT bit-identical to the
pure-Python solver, and it stays opt-in until its output has been checked.
This script is that check: it solves the same equilibrium both ways, with
the cache bypassed, and reports how far apart the two answers are.

    python tools/gs_backend_compare.py
    python tools/gs_backend_compare.py --tolerance 1e-12

Both psi grids and every field derived from them (B_R, B_Z, B_phi, |B|) are
compared, since psi is only an intermediate -- what the particle pusher
actually reads is the interpolated field.

Exit codes:
    0  every comparison within --tolerance (numba path looks trustworthy)
    1  a difference exceeds --tolerance (do not switch the default)
    2  numba unavailable, so there is nothing to compare
"""

import argparse
import os
import sys
import time

# Import the module the same way the simulation does, from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                        # noqa: E402
import mhd_equilibrium                                     # noqa: E402
from mhd_equilibrium import MHDEquilibrium                 # noqa: E402


def solve_with(backend, quiet=True):
    """Solve from scratch on `backend`, bypassing the disk cache."""
    saved = mhd_equilibrium.GS_CACHE_DISABLED
    mhd_equilibrium.GS_CACHE_DISABLED = True    # never read or trust a cache
    saved_store = MHDEquilibrium._store_cached_psi
    MHDEquilibrium._store_cached_psi = lambda self, path, psi: None

    stdout = sys.stdout
    if quiet:
        sys.stdout = open(os.devnull, "w")
    try:
        eq = MHDEquilibrium()
        t0 = time.perf_counter()
        eq.solve_grad_shafranov(backend=backend)
        elapsed = time.perf_counter() - t0
    finally:
        if quiet:
            sys.stdout.close()
        sys.stdout = stdout
        mhd_equilibrium.GS_CACHE_DISABLED = saved
        MHDEquilibrium._store_cached_psi = saved_store

    return eq, elapsed


def report(name, a, b, tolerance):
    """Print an absolute/relative comparison of two arrays; True if within.

    The pass/fail test is np.allclose-style: a point is acceptable if it
    is within `tolerance` relatively OR within `tolerance` of the field's
    own scale absolutely. The second clause matters because B_R and B_Z
    cross zero -- pointwise relative error explodes there even when the
    two solvers agree to 1e-14 in absolute terms, so a bare relative
    test reports noise as disagreement.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    identical = np.array_equal(a, b)
    abs_diff = np.abs(a - b)
    max_abs = float(abs_diff.max())

    # Field scale: the dynamic range the difference should be judged against.
    scale_ref = float(np.abs(a).max())
    if scale_ref == 0.0:
        scale_ref = 1.0
    rel_to_scale = max_abs / scale_ref

    atol = tolerance * scale_ref
    ok = bool(np.all(abs_diff <= atol + tolerance * np.abs(b)))

    # Pointwise relative, reported for information only.
    scale = np.maximum(np.abs(a), np.abs(b))
    nonzero = scale > 0.0
    max_rel = float((abs_diff[nonzero] / scale[nonzero]).max()) if nonzero.any() else 0.0

    status = "IDENTICAL" if identical else ("ok" if ok else "OVER TOLERANCE")
    print(f"  {name:<12} max|abs| {max_abs:.3e}   rel-to-scale {rel_to_scale:.3e}"
          f"   pointwise-rel {max_rel:.3e}   {status}")
    return ok


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare the numba GS solver against the Python reference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="exit codes: 0 = within tolerance, 1 = over, 2 = numba missing",
    )
    parser.add_argument("--tolerance", type=float, default=1e-10,
                        help="max acceptable relative difference (default: 1e-10)")
    args = parser.parse_args(argv)

    if not mhd_equilibrium.NUMBA_AVAILABLE:
        print("[ERROR] numba is not installed; nothing to compare.", file=sys.stderr)
        return 2

    print("Solving with the Python reference solver...")
    eq_py, t_py = solve_with("python")
    print(f"  {t_py:.3f} s")

    print("Solving with the numba fastmath solver (includes JIT compile)...")
    eq_nb, t_nb = solve_with("numba")
    print(f"  {t_nb:.3f} s")

    print("\nSolving again with numba (warm JIT)...")
    _, t_nb_warm = solve_with("numba")
    print(f"  {t_nb_warm:.3f} s")

    speedup = t_py / t_nb_warm if t_nb_warm > 0 else float("inf")
    print(f"\nSpeedup (warm JIT): {speedup:.1f}x")

    print(f"\nDifferences (tolerance {args.tolerance:.1e}; rel-to-scale is the"
          " honest metric --")
    print("  pointwise-rel is inflated wherever a field crosses zero):")
    results = [
        report("psi", eq_py.psi, eq_nb.psi, args.tolerance),
        report("B_R", eq_py.BR_grid, eq_nb.BR_grid, args.tolerance),
        report("B_Z", eq_py.BZ_grid, eq_nb.BZ_grid, args.tolerance),
        report("B_phi", eq_py.Bphi_grid, eq_nb.Bphi_grid, args.tolerance),
        report("|B|", eq_py.B_mag_grid, eq_nb.B_mag_grid, args.tolerance),
    ]

    # A field query at a representative point, as the pusher would make it.
    pos = np.array([1.2, 0.0, 0.1])
    print("\nField query at (x, y, z) = (1.2, 0.0, 0.1):")
    print(f"  python: B = {eq_py.get_B_field(pos)}  |B| = {eq_py.get_B_mag(pos):.9f}")
    print(f"  numba:  B = {eq_nb.get_B_field(pos)}  |B| = {eq_nb.get_B_mag(pos):.9f}")

    if all(results):
        print(f"\n[OK] Every field agrees to within {args.tolerance:.1e} relative.")
        print("     The numba path looks trustworthy; it stays opt-in until you")
        print("     change DEFAULT_GS_BACKEND in mhd_equilibrium.py.")
        return 0

    print(f"\n[FAIL] A field differs by more than {args.tolerance:.1e} relative.")
    print("       Keep the Python solver as the default.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
