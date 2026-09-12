"""Hybrid-solver Numba kernels, split out of main.py (step 14).

Imports gather_electric_field_scalar and interpolate_psi from physics_engine
(both @njit, both called from inside the prange); nothing from the reactor
modules.
"""

import numpy as np
from numba import njit, prange

from physics_engine import (
    gather_electric_field_scalar,
    interpolate_psi,
)

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
