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
@njit(fastmath=True, inline="always")
def _field_at(px, py, pz, E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid,
              R_min, R_max, Z_min, Z_max, nR, nZ, B0, R0, b_perturb, amplitude, m_mode, n_mode):
    """E and B (Cartesian, float32) at one point -- the body of the field gather.

    Shared by vectorized_gather_and_B_into (one evaluation per particle per
    step) and boris_push_substeps_regather (one per alpha per sub-step), so
    the two cannot drift apart. `amplitude` is the tearing-mode amplitude
    b_perturb * exp(gamma * t), precomputed by the caller once per time level;
    `b_perturb` only gates whether the perturbation is applied.
    """
    B0_f = np.float32(B0)
    R0_f = np.float32(R0)
    zero_f = np.float32(0.0)
    m_f = np.float32(m_mode)
    n_f = np.float32(n_mode)

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

    return ex, ey, ez, Bx, By, Bz


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
    # E gather and the B construction. The previous version called a per-particle
    # array-returning gather, which redid the sqrt and an arctan2 of its own and
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
    amplitude = np.float32(b_perturb * np.exp(gamma * t))

    for i in prange(N):
        ex, ey, ez, bx, by, bz = _field_at(
            pos_arr[i, 0], pos_arr[i, 1], pos_arr[i, 2],
            E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid,
            R_min, R_max, Z_min, Z_max, nR, nZ, B0, R0,
            b_perturb, amplitude, m_mode, n_mode)
        E_out[i, 0] = ex
        E_out[i, 1] = ey
        E_out[i, 2] = ez
        B_out[i, 0] = bx
        B_out[i, 1] = by
        B_out[i, 2] = bz


@njit(parallel=True, fastmath=True)
def boris_push_substeps_regather(pos_arr, vel_arr, q, m, dt, n_sub,
                                 E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid,
                                 R_min, R_max, Z_min, Z_max, nR, nZ, B0, R0, t,
                                 b_perturb, m_mode, n_mode, gamma):
    """Alpha sub-stepped Boris push with E and B re-gathered at every sub-step.

    Mutates pos_arr / vel_arr in place (float32 storage, float64 accumulators).

    A 3.5 MeV alpha covers ~1.3e-2 m per 1 ns global step against a ~2.2e-2 m
    Larmor radius, so it is pushed in n_sub sub-steps of dt/n_sub. Each sub-step
    is a standard Boris step: the field is evaluated at the sub-step's start
    position, which is the time-centre of its leapfrogged velocity update.

    The kernel this replaces gathered the field once per GLOBAL step and held it
    across all n_sub sub-steps. Then the field used for the whole 1.3 cm advance
    sits at one end of it, not its centre: the scheme is first-order and not
    phase-space-volume-preserving, and in the sheared Grad-Shafranov field the
    error was secular, not oscillatory. It drove mu to ~0.1% of its birth value
    and every alpha's pitch to xi = -1 (to +1 with the sign of B_pol flipped)
    within a few microseconds. See tools/alpha_pitch_check.py.
    """
    N = pos_arr.shape[0]
    dt_sub = dt / n_sub
    qm = q / m
    half = dt_sub / 2.0
    # Tearing-mode amplitude at each sub-step's time level, as the per-step
    # gather computes it at t.
    amps = np.empty(n_sub, dtype=np.float32)
    for k in range(n_sub):
        amps[k] = np.float32(b_perturb * np.exp(gamma * (t + k * dt_sub)))

    for i in prange(N):
        px = np.float64(pos_arr[i, 0]); py = np.float64(pos_arr[i, 1]); pz = np.float64(pos_arr[i, 2])
        vx = np.float64(vel_arr[i, 0]); vy = np.float64(vel_arr[i, 1]); vz = np.float64(vel_arr[i, 2])

        for k in range(n_sub):
            # The field sees float32 coordinates, exactly as the per-step gather
            # does reading the float32 pool.
            ex, ey, ez, bx, by, bz = _field_at(
                np.float32(px), np.float32(py), np.float32(pz),
                E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid,
                R_min, R_max, Z_min, Z_max, nR, nZ, B0, R0,
                b_perturb, amps[k], m_mode, n_mode)

            tx = qm * bx * half
            ty = qm * by * half
            tz = qm * bz * half
            f = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz)
            sx = tx * f; sy = ty * f; sz = tz * f

            vmx = vx + qm * ex * half
            vmy = vy + qm * ey * half
            vmz = vz + qm * ez * half

            ax = vmy * tz - vmz * ty
            ay = vmz * tx - vmx * tz
            az = vmx * ty - vmy * tx
            vpx = vmx + ax; vpy = vmy + ay; vpz = vmz + az

            cx = vpy * sz - vpz * sy
            cy = vpz * sx - vpx * sz
            cz = vpx * sy - vpy * sx

            vx = vmx + cx + qm * ex * half
            vy = vmy + cy + qm * ey * half
            vz = vmz + cz + qm * ez * half

            px += vx * dt_sub; py += vy * dt_sub; pz += vz * dt_sub

        vel_arr[i, 0] = vx; vel_arr[i, 1] = vy; vel_arr[i, 2] = vz
        pos_arr[i, 0] = px; pos_arr[i, 1] = py; pos_arr[i, 2] = pz


def vectorized_gather_and_B(pos_arr, E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid,
                            R_min, R_max, Z_min, Z_max, nR, nZ, B0, R0, t, b_perturb, m_mode, n_mode, gamma):
    """Allocating wrapper around vectorized_gather_and_B_into.

    Same signature and same (E_arr, B_arr) return as before the out-parameter
    split, for the caller that runs once per run rather than once per step --
    package_reactor_results' phase-space projection -- where one pair of
    (N,3) arrays is not worth a reused buffer. The
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
