#!/usr/bin/env python3
"""Check which Grad-Shafranov operator the solved psi actually satisfies.

Solves the equilibrium fresh (cache bypassed, Python backend by default) and
evaluates, on interior grid points, the residual of

    (i)  the CORRECT operator  d2psi/dR2 - (1/R) dpsi/dR + d2psi/dZ2
    (ii) the WRONG operator    d2psi/dR2 + (1/R) dpsi/dR + d2psi/dZ2

against the source -R^2, each reported as max|residual| / max|source|.

After the sign fix, (i) should be small (limited by the loose SOR stopping
criterion) and (ii) large (order 0.5). Before the fix the two were reversed.

    python tools/gs_operator_check.py
    python tools/gs_operator_check.py --backend numba

Also prints the magnetic axis (R, Z of max psi) and psi max.

Exit codes:
    0  correct-operator residual is the smaller one, by a wide margin
    1  it is not (the stencil has the wrong sign)
"""

import argparse
import os
import sys

# Import the module the same way the simulation does, from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                        # noqa: E402
import mhd_equilibrium                                     # noqa: E402
from mhd_equilibrium import MHDEquilibrium                 # noqa: E402


def solve_fresh(backend):
    """Solve from scratch, never reading or writing the disk cache."""
    saved = mhd_equilibrium.GS_CACHE_DISABLED
    saved_store = MHDEquilibrium._store_cached_psi
    mhd_equilibrium.GS_CACHE_DISABLED = True
    MHDEquilibrium._store_cached_psi = lambda self, path, psi: None
    try:
        eq = MHDEquilibrium()
        eq.solve_grad_shafranov(backend=backend)
    finally:
        mhd_equilibrium.GS_CACHE_DISABLED = saved
        MHDEquilibrium._store_cached_psi = saved_store
    return eq


def residuals(eq):
    """Return (correct, old) = max|residual| / max|source| on interior points."""
    psi, RR = eq.psi, eq.RR
    dR, dZ = eq.dR, eq.dZ

    # psi is indexed [i (Z), j (R)].
    d2R = (psi[1:-1, 2:] - 2.0 * psi[1:-1, 1:-1] + psi[1:-1, :-2]) / dR**2
    d2Z = (psi[2:, 1:-1] - 2.0 * psi[1:-1, 1:-1] + psi[:-2, 1:-1]) / dZ**2
    dRp = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2.0 * dR)
    R = RR[1:-1, 1:-1]
    source = -(R ** 2)

    scale = np.abs(source).max()
    correct = np.abs(d2R - dRp / R + d2Z - source).max() / scale
    old = np.abs(d2R + dRp / R + d2Z - source).max() / scale
    return float(correct), float(old)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Residual of the solved psi against the correct and the "
                    "old (wrong-sign) Grad-Shafranov operator.")
    parser.add_argument("--backend", default="python", choices=("python", "numba"))
    args = parser.parse_args(argv)

    stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")           # silence the solver chatter
    try:
        eq = solve_fresh(args.backend)
    finally:
        sys.stdout.close()
        sys.stdout = stdout

    correct, old = residuals(eq)
    k = np.unravel_index(np.argmax(eq.psi), eq.psi.shape)

    print(f"backend: {args.backend}   grid: {eq.nz} x {eq.nx} (Z x R)")
    print(f"  (i)  correct  d2R - (1/R)dR + d2Z : max|res|/max|src| = {correct:.4e}")
    print(f"  (ii) old      d2R + (1/R)dR + d2Z : max|res|/max|src| = {old:.4e}")
    print(f"  magnetic axis: R = {eq.RR[k]:.4f} m, Z = {eq.ZZ[k]:.4f} m")
    print(f"  psi max      : {eq.psi[k]:.6f}")

    if correct < old / 10.0:
        print("[OK] psi satisfies the correct Grad-Shafranov operator.")
        return 0
    print("[FAIL] psi does not satisfy the correct operator better than the old one.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
