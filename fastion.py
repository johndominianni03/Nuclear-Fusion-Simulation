"""Fast-ion slowing-down module (Tier 3): guiding centre + Monte Carlo
Fokker-Planck, the way fast-ion codes such as ASCOT and LOCUST follow alphas
and beam ions for their whole slowing-down (~1 s).

Entry points:
    run_alpha_slowing_down(cfg)    3.5 MeV fusion alphas
    run_beam_slowing_down(cfg)     1 MeV neutral-beam deuterons
    main()                         production: both species, plots, sample npz

Markers
  * Alphas: born from the thermal D-T reaction rate of the background profiles
    (Tier 2 machinery: physics_engine.compute_alpha_birth_rates on the
    background's shell profile, positions by AlphaBirthSampler), isotropic at
    3.5 MeV. Each marker carries weight w = S_alpha / N [reactions/s], S_alpha
    the total birth rate.
  * Beam ions: the reactor's injector distribution
    (ParticlePusher.inject_neutral_beam_cartesian, 1 MeV D), weights summing to
    P_NBI / E_beam [ions/s].
  Each birth particle is reduced to its guiding centre with the particle's
  energy and canonical momentum (gc_push.gc_from_particle_invariants).

Following, per marker and per step (fastion_kernel)
  1. fields at the guiding centre (fields_gc.gc_fields): B, psi_N, Phi, ...;
  2. thermalization test: E < THERMAL_FACTOR x (3/2) T_i(psi_N) = 3 T_i local
     ends the marker (status THERMALIZED);
  3. dt = TRANSIT_FRACTION x 2 pi R_axis / v (the Phase 2 criterion, 1/200 of
     the toroidal transit at the marker's current speed);
  4. one collision step of dt (collisions_fp.fp_step) against the local e, D, T
     Maxwellians, energy to electrons and to ions deposited in the marker's
     psi_N bin; dwell time, p_par and p_perp moments accumulated;
  5. one RK4 guiding-centre step of dt (k1 from the fields of 1.);
  6. loss test: psi_N > 1 ends the marker (status LOST) at the crossing point;
  7. safety cap: t > T_MAX (status TIMEOUT; none expected).

Energy accounting per marker, exact up to the GC integration error:
    E_birth = (to electrons) + (to ions) + E_end + q (Phi_end - Phi_birth)
with E_end the kinetic energy carried out by a lost marker, kept by a
thermalized one (the thermalized sink), or still held by a running one.

Steady state (constant source): with N markers each of weight w_i
    n_fast,k  = sum_i w_i (time marker i spent in shell k) / V_k
    p_par,k   = sum_i w_i int m v_par^2 dt / V_k,  p_perp,k = sum_i w_i int mu B dt / V_k
    p_k       = (p_par + 2 p_perp) / 3,   beta = 2 mu0 <p>_V / B0^2
    P_e,k     = sum_i w_i (energy marker i gave the electrons in shell k) / V_k
psi_N shells use the spline psi_N with volumes from a revolved fine mesh.
"""
import math
import os
import time

import numpy as np
from numba import njit, prange

from constants import E_CHARGE, M_ALPHA, M_DEUTERON, MEV_J, KEV_J, MU_0
from fields_gc import gc_fields, gc_psi_value
from gc_push import gc_rhs, gc_rhs_from_fields, gc_from_particle_invariants
from background import background_at
from collisions_fp import (fp_step, gamma_prefactor, coulomb_log_table, lnL_at, seed_rng,
                           UniformPlasma)

REPO = os.path.dirname(os.path.abspath(__file__))

ACTIVE, THERMALIZED, LOST, TIMEOUT = 0, 1, 2, 3
TRANSIT_FRACTION = 1.0 / 200.0      # gc_push.DEFAULT_TRANSIT_FRACTION
THERMAL_FACTOR = 2.0                # thermalized below 2 x (3/2) T_i local
T_MAX = 5.0                         # s, safety cap (slowing-down times are < 1.5 s)
PROMPT_LOSS_TIME = 1.0e-4           # s: lost within 100 us = first orbit (bounce <= 13 us)
N_PSI_BINS = 50
N_E_BINS = 72
N_THETA_BINS = 36
CHUNK_STEPS = 200000


# =======================================================
# THE KERNEL
# =======================================================
@njit(parallel=True, cache=True)
def fastion_kernel(idx, state, mu, t, status, steps, q, m, rng, frac, R_axis, t_max, n_steps,
                   thermal_factor, E_max_bin,
                   dE_e, dE_i, dep_e, dep_i, dwell, ppar, pperp, dwellE,
                   track_slot, rec, rec_n, rec_every,
                   coef, R_min, Z_min, dR, dZ, nR, nZ, F, psi_axis, psi_edge,
                   phi_x0, phi_dx, phi_val, phi_der,
                   dens, temp, bx0, bdx, mass, charge, lnL_tab, gam):
    """Advance markers idx by up to n_steps steps each (see the module docstring).
    All per-marker arrays are updated in place; accumulators are per marker, so
    the result does not depend on thread scheduling or on the chunking."""
    nb = dep_e.shape[1]
    nE = dwellE.shape[1]
    dpsi = psi_axis - psi_edge
    two_pi_R = 2.0 * math.pi * R_axis
    for jj in prange(idx.shape[0]):
        i = idx[jj]
        if status[i] != 0:
            continue
        A = np.empty(3)
        nuD = np.empty(3)
        D = np.empty(3)
        dE = np.zeros(3)
        nloc = np.empty(3)
        Tloc = np.empty(3)
        lnL = np.empty(3)
        R = state[i, 0]
        ph = state[i, 1]
        Z = state[i, 2]
        vp = state[i, 3]
        mui = mu[i]
        ti = t[i]
        si = steps[i]
        slot = track_slot[i]
        for s in range(n_steps):
            BR, Bp, BZ, B, gR, gZ, cR, cp, cZ, psi, psiN, Phi, ER, EZ = gc_fields(
                R, Z, coef, R_min, Z_min, dR, dZ, nR, nZ, F,
                psi_axis, psi_edge, phi_x0, phi_dx, phi_val, phi_der)
            background_at(psiN, dens, temp, bx0, bdx, nloc, Tloc)
            v2 = vp * vp + 2.0 * mui * B / m
            v = math.sqrt(v2)
            E = 0.5 * m * v2
            if E < thermal_factor * 1.5 * Tloc[1]:
                status[i] = 1
                break
            if slot >= 0 and si % rec_every == 0 and rec_n[slot] < rec.shape[1]:
                k = rec_n[slot]
                rec[slot, k, 0] = R
                rec[slot, k, 1] = ph
                rec[slot, k, 2] = Z
                rec[slot, k, 3] = E
                rec[slot, k, 4] = ti
                rec_n[slot] = k + 1
            dt = frac * two_pi_R / v
            # --- collisions ---
            lnL_at(psiN, lnL_tab, bx0, bdx, lnL)
            dE[0] = 0.0
            dE[1] = 0.0
            dE[2] = 0.0
            xi = vp / v
            v, xi = fp_step(v, xi, dt, m, gam, nloc, Tloc, lnL, mass, charge, rng, i, dE, A, nuD, D)
            vp = xi * v
            mui = 0.5 * m * v * v * (1.0 - xi * xi) / B
            kb = int(psiN * nb)
            if kb < 0:
                kb = 0
            elif kb > nb - 1:
                kb = nb - 1
            de_i = dE[1] + dE[2]
            dE_e[i] += dE[0]
            dE_i[i] += de_i
            dep_e[i, kb] += dE[0]
            dep_i[i, kb] += de_i
            dwell[i, kb] += dt
            ppar[i, kb] += m * vp * vp * dt
            pperp[i, kb] += mui * B * dt
            kE = int(E / E_max_bin * nE)
            if kE > nE - 1:
                kE = nE - 1
            dwellE[i, kE] += dt
            # --- guiding-centre RK4, k1 from the fields above ---
            a1, b1, c1, d1 = gc_rhs_from_fields(BR, Bp, BZ, B, gR, gZ, cR, cp, cZ, ER, EZ, R, vp, mui, q, m)
            h = 0.5 * dt
            a2, b2, c2, d2, _ = gc_rhs(R + h * a1, Z + h * c1, vp + h * d1, mui, q, m,
                                       coef, R_min, Z_min, dR, dZ, nR, nZ, F,
                                       psi_axis, psi_edge, phi_x0, phi_dx, phi_val, phi_der)
            a3, b3, c3, d3, _ = gc_rhs(R + h * a2, Z + h * c2, vp + h * d2, mui, q, m,
                                       coef, R_min, Z_min, dR, dZ, nR, nZ, F,
                                       psi_axis, psi_edge, phi_x0, phi_dx, phi_val, phi_der)
            a4, b4, c4, d4, _ = gc_rhs(R + dt * a3, Z + dt * c3, vp + dt * d3, mui, q, m,
                                       coef, R_min, Z_min, dR, dZ, nR, nZ, F,
                                       psi_axis, psi_edge, phi_x0, phi_dx, phi_val, phi_der)
            w6 = dt / 6.0
            R += w6 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
            ph += w6 * (b1 + 2.0 * b2 + 2.0 * b3 + b4)
            Z += w6 * (c1 + 2.0 * c2 + 2.0 * c3 + c4)
            vp += w6 * (d1 + 2.0 * d2 + 2.0 * d3 + d4)
            ti += dt
            si += 1
            if (psi_axis - gc_psi_value(R, Z, coef, R_min, Z_min, dR, dZ, nR, nZ)) / dpsi > 1.0:
                status[i] = 2
                break
            if ti >= t_max:
                status[i] = 3
                break
        state[i, 0] = R
        state[i, 1] = ph
        state[i, 2] = Z
        state[i, 3] = vp
        mu[i] = mui
        t[i] = ti
        steps[i] = si


# =======================================================
# SIMULATION OBJECT
# =======================================================
SPECIES = {"alpha": (2.0 * E_CHARGE, M_ALPHA), "beam": (E_CHARGE, M_DEUTERON)}


class FastIonSim:
    """Equilibrium, background and per-species tables shared by the runs."""

    def __init__(self, cfg=None, quiet=True):
        from fields_gc import build_gc_field
        self.field, self.bg = build_gc_field(cfg, quiet=quiet)
        self.cfg = self.field.cfg
        self.eq = self.field.eq
        self.lnL = {k: coulomb_log_table(self.bg, q / E_CHARGE, m) for k, (q, m) in SPECIES.items()}
        self.gam = {k: gamma_prefactor(q / E_CHARGE, m) for k, (q, m) in SPECIES.items()}
        self._shell_volumes()
        self.wall_R, self.wall_Z = self.field.flux_surface_contour(1.0, n_theta=721)

    def _shell_volumes(self, n_mesh=1200):
        f = self.field
        dR = (f.R_nodes[-1] - f.R_nodes[0]) / n_mesh
        dZ = (f.Z_nodes[-1] - f.Z_nodes[0]) / n_mesh
        Rc = f.R_nodes[0] + (np.arange(n_mesh) + 0.5) * dR
        Zc = f.Z_nodes[0] + (np.arange(n_mesh) + 0.5) * dZ
        RR, ZZ = np.meshgrid(Rc, Zc, indexing="ij")
        x = f.psi_N(RR, ZZ)
        inside = x <= 1.0
        k = np.clip((np.clip(x[inside], 0.0, None) * N_PSI_BINS).astype(np.int64), 0, N_PSI_BINS - 1)
        self.shell_volume = np.bincount(k, weights=2.0 * np.pi * RR[inside] * dR * dZ, minlength=N_PSI_BINS)
        self.plasma_volume = float(self.shell_volume.sum())
        self.psi_edges = np.linspace(0.0, 1.0, N_PSI_BINS + 1)
        self.psi_centers = 0.5 * (self.psi_edges[:-1] + self.psi_edges[1:])

    # ---- births ------------------------------------------------------------
    def birth_alphas(self, n, seed):
        """(pos, vel, weight, total rate) for n alpha markers."""
        from physics_engine import ParticlePusher, AlphaBirthSampler, compute_alpha_birth_rates
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            rates = compute_alpha_birth_rates(self.bg.shell_profile(self.eq), self.cfg)
            pos, vel, _ = AlphaBirthSampler(self.eq).sample(rates, n, ParticlePusher(self.eq), self.cfg)
        finally:
            np.random.set_state(state)
        S = float(rates.sum())
        return pos.astype(np.float64), vel.astype(np.float64), np.full(n, S / n), S

    def birth_beams(self, n, seed):
        """(pos, vel, weight, total rate) for n beam markers (P_NBI / E_beam)."""
        from physics_engine import ParticlePusher
        cfg, eq = self.cfg, self.eq
        state = np.random.get_state()
        np.random.seed(seed)
        try:
            psi_bounds = (eq.psi_grid, eq.psi_core, eq.psi_R_min, eq.psi_R_max,
                          eq.psi_Z_min, eq.psi_Z_max, eq.psi_nR, eq.psi_nZ)
            pos, vel = ParticlePusher(eq).inject_neutral_beam_cartesian(
                n, E_keV=cfg.nbi_energy_keV, psi_bounds=psi_bounds)
        finally:
            np.random.set_state(state)
        S = cfg.P_NBI_MW * 1.0e6 / (cfg.nbi_energy_keV * KEV_J)
        return pos.astype(np.float64), vel.astype(np.float64), np.full(n, S / n), S

    # ---- following ---------------------------------------------------------
    def follow(self, species, pos, vel, weight, seed, t_max=None, n_track=0, rec_every=50000,
               n_rec_max=4000, chunk=CHUNK_STEPS, progress=True, label=None):
        """Follow birth particles to thermalization, loss or t_max. Returns a
        results dict (per-marker arrays and accumulators)."""
        t_max = T_MAX if t_max is None else t_max
        q, m = SPECIES[species]
        f = self.field
        N = pos.shape[0]
        state, mu = gc_from_particle_invariants(f, pos, vel, q, m)
        Rp = np.hypot(pos[:, 0], pos[:, 1])
        fp = f.fields(Rp, pos[:, 2])
        E_birth = 0.5 * m * np.sum(vel * vel, axis=1)
        Phi_birth = fp["Phi"]                     # W = E_birth + q Phi_birth (particle and GC)
        fb = f.fields(state[:, 0], state[:, 2])
        out = {
            "species": species, "q": q, "m": m, "N": N, "weight": weight.copy(),
            "pos0": pos.copy(), "vel0": vel.copy(), "state0": state.copy(), "mu0": mu.copy(),
            "E_birth": E_birth, "Phi_birth": Phi_birth, "psiN_birth": fb["psiN"],
            "xi_birth": state[:, 3] / np.sqrt(state[:, 3] ** 2 + 2.0 * mu * fb["Bmag"] / m),
        }
        t = np.zeros(N)
        status = np.zeros(N, dtype=np.int8)
        steps = np.zeros(N, dtype=np.int64)
        rng = seed_rng(N, seed)
        dE_e = np.zeros(N)
        dE_i = np.zeros(N)
        dep_e = np.zeros((N, N_PSI_BINS))
        dep_i = np.zeros((N, N_PSI_BINS))
        dwell = np.zeros((N, N_PSI_BINS))
        ppar = np.zeros((N, N_PSI_BINS))
        pperp = np.zeros((N, N_PSI_BINS))
        dwellE = np.zeros((N, N_E_BINS))
        E_max_bin = 1.2 * float(E_birth.max())
        track_slot = np.full(N, -1, dtype=np.int64)
        n_track = min(n_track, N)
        track_slot[:n_track] = np.arange(n_track)
        rec = np.zeros((max(n_track, 1), n_rec_max, 5))
        rec_n = np.zeros(max(n_track, 1), dtype=np.int64)
        bg = self.bg
        t0 = time.perf_counter()
        last = t0
        while True:
            idx = np.nonzero(status == ACTIVE)[0]
            if idx.size == 0:
                break
            fastion_kernel(idx, state, mu, t, status, steps, q, m, rng, TRANSIT_FRACTION, f.R_axis,
                           float(t_max), int(chunk), THERMAL_FACTOR, E_max_bin,
                           dE_e, dE_i, dep_e, dep_i, dwell, ppar, pperp, dwellE,
                           track_slot, rec, rec_n, int(rec_every),
                           *f.args(), bg.dens, bg.temp, bg.x0, bg.dx, bg.mass, bg.charge,
                           self.lnL[species], self.gam[species])
            now = time.perf_counter()
            if progress and now - last > 60.0:
                last = now
                n_act = int(np.sum(status == ACTIVE))
                print(f"  [{label or species}] {now - t0:7.0f} s: {N - n_act}/{N} done "
                      f"(thermalized {int(np.sum(status == THERMALIZED))}, lost {int(np.sum(status == LOST))}), "
                      f"{steps.sum() / 1e9:.2f}e9 steps", flush=True)
        wall = time.perf_counter() - t0
        fe = f.fields(state[:, 0], state[:, 2])
        E_end = 0.5 * m * state[:, 3] ** 2 + mu * fe["Bmag"]
        out.update({
            "state": state, "mu": mu, "t": t, "status": status, "steps": steps,
            "E_end": E_end, "Phi_end": fe["Phi"], "psiN_end": fe["psiN"],
            "theta_end": np.arctan2(state[:, 2] - f.Z_axis, state[:, 0] - f.R_axis),
            "dE_e": dE_e, "dE_i": dE_i, "dep_e": dep_e, "dep_i": dep_i, "dwell": dwell,
            "ppar": ppar, "pperp": pperp, "dwellE": dwellE, "E_max_bin": E_max_bin,
            "rec": rec[:n_track], "rec_n": rec_n[:n_track], "wall_s": wall,
        })
        return out


# =======================================================
# STEADY-STATE RESULTS
# =======================================================
def energy_balance(r):
    """Weighted energy balance [W]: birth = electrons + ions + lost + thermalized
    + running + q dPhi + residual (the GC integration error)."""
    w = r["weight"]
    st = r["status"]
    qdPhi = r["q"] * (r["Phi_end"] - r["Phi_birth"])
    P = {
        "birth": float(np.sum(w * r["E_birth"])),
        "electrons": float(np.sum(w * r["dE_e"])),
        "ions": float(np.sum(w * r["dE_i"])),
        "lost": float(np.sum((w * r["E_end"])[st == LOST])),
        "thermalized": float(np.sum((w * r["E_end"])[st == THERMALIZED])),
        "running": float(np.sum((w * r["E_end"])[(st == ACTIVE) | (st == TIMEOUT)])),
        "q_dPhi": float(np.sum(w * qdPhi)),
    }
    P["residual"] = P["birth"] - (P["electrons"] + P["ions"] + P["lost"] + P["thermalized"]
                                  + P["running"] + P["q_dPhi"])
    return P


def _rel_err_of_sum(x):
    """Relative 1-sigma statistical error of sum(x) over equal-weight markers."""
    x = np.asarray(x, dtype=np.float64)
    s = x.sum()
    return float(np.std(x, ddof=1) * math.sqrt(x.size) / abs(s)) if s != 0 and x.size > 1 else float("nan")


def steady_state(sim, r):
    """Profiles and totals from one species' marker histories."""
    w = r["weight"][:, None]
    V = np.where(sim.shell_volume > 0, sim.shell_volume, np.inf)
    n = np.sum(w * r["dwell"], axis=0) / V
    p_par = np.sum(w * r["ppar"], axis=0) / V
    p_perp = np.sum(w * r["pperp"], axis=0) / V
    p = (p_par + 2.0 * p_perp) / 3.0
    P_e = np.sum(w * r["dep_e"], axis=0) / V
    P_i = np.sum(w * r["dep_i"], axis=0) / V
    B0 = sim.cfg.B0
    p_vol = float(np.sum(p * sim.shell_volume) / sim.plasma_volume)
    st = r["status"]
    wv = r["weight"]
    lost = st == LOST
    prompt = lost & (r["t"] < PROMPT_LOSS_TIME)
    therm = st == THERMALIZED
    bal = energy_balance(r)
    # per-marker contributions -> statistical errors of the totals
    pe_i = r["weight"] * r["dE_e"]
    pi_i = r["weight"] * r["dE_i"]
    pl_i = np.where(lost, r["weight"] * r["E_end"], 0.0)
    pv_i = r["weight"] * np.sum((r["ppar"] + 2.0 * r["pperp"]) / 3.0, axis=1)
    N_i = r["weight"] * np.sum(r["dwell"], axis=1)
    N = r["N"]
    res = {
        "n": n, "p": p, "p_par": p_par, "p_perp": p_perp, "P_e": P_e, "P_i": P_i,
        "beta": 2.0 * MU_0 * p_vol / B0**2, "p_vol_avg": p_vol,
        "N_fast": float(np.sum(n * sim.shell_volume)),
        "P_birth": bal["birth"], "P_e_total": bal["electrons"], "P_i_total": bal["ions"],
        "P_lost": bal["lost"], "P_thermalized": bal["thermalized"], "P_running": bal["running"],
        "P_qdPhi": bal["q_dPhi"], "balance_residual": bal["residual"] / bal["birth"],
        "frac_e": bal["electrons"] / bal["birth"], "frac_i": bal["ions"] / bal["birth"],
        "frac_lost_power": bal["lost"] / bal["birth"],
        "frac_lost": float(np.sum(wv[lost]) / np.sum(wv)),
        "frac_prompt": float(np.sum(wv[prompt]) / np.sum(wv)),
        "n_lost": int(lost.sum()), "n_prompt": int(prompt.sum()),
        "n_thermalized": int(therm.sum()), "n_timeout": int((st == TIMEOUT).sum()),
        "t_sd_mean": float(np.mean(r["t"][therm])) if therm.any() else float("nan"),
        "t_sd_median": float(np.median(r["t"][therm])) if therm.any() else float("nan"),
        "t_sd_p10": float(np.percentile(r["t"][therm], 10)) if therm.any() else float("nan"),
        "t_sd_p90": float(np.percentile(r["t"][therm], 90)) if therm.any() else float("nan"),
        "err": {
            "P_e": _rel_err_of_sum(pe_i), "P_i": _rel_err_of_sum(pi_i),
            "P_e+P_i": _rel_err_of_sum(pe_i + pi_i),
            "frac_lost_abs": float(math.sqrt(max(np.mean(lost) * (1 - np.mean(lost)), 0.0) / N)),
            "frac_lost_power_abs": float(np.std(pl_i / np.mean(r["weight"] * r["E_birth"]), ddof=1)
                                         / math.sqrt(N)) if N > 1 else float("nan"),
            "beta": _rel_err_of_sum(pv_i), "N_fast": _rel_err_of_sum(N_i),
        },
        "wall_s": r["wall_s"], "steps_total": int(r["steps"].sum()), "N": N,
    }
    # wall load: lost power per unit LCFS area, by poloidal angle about the axis
    th_edges = np.linspace(-np.pi, np.pi, N_THETA_BINS + 1)
    Pw = np.bincount(np.clip(np.digitize(r["theta_end"][lost], th_edges) - 1, 0, N_THETA_BINS - 1),
                     weights=(r["weight"] * r["E_end"])[lost], minlength=N_THETA_BINS)
    f = sim.field
    Rm = 0.5 * (sim.wall_R[1:] + sim.wall_R[:-1])
    dl = np.hypot(np.diff(sim.wall_R), np.diff(sim.wall_Z))
    thm = np.arctan2(0.5 * (sim.wall_Z[1:] + sim.wall_Z[:-1]) - f.Z_axis, Rm - f.R_axis)
    area = np.bincount(np.clip(np.digitize(thm, th_edges) - 1, 0, N_THETA_BINS - 1),
                       weights=2.0 * np.pi * Rm * dl, minlength=N_THETA_BINS)
    res["theta_edges"] = th_edges
    res["wall_power"] = Pw
    res["wall_area"] = area
    res["wall_flux"] = np.where(area > 0, Pw / area, 0.0)
    return res


def analytic_slowing_down_spectrum(sim, r, E_edges):
    """Steady-state energy spectrum dN/dE [1/J] if every marker slowed down where
    it was born, with the classical rate dE/dt = -(2E/tau_s)(1 + (E_c/E)^1.5) of
    the operator's local tau_s and E_c (collisions_fp.UniformPlasma at the birth
    psi_N), from its birth energy to the thermalization threshold, weighted by
    birth rate: n(E) = sum_i w_i / |dE/dt|_i for E_th,i < E < E_birth,i."""
    q, m = r["q"], r["m"]
    Za = q / E_CHARGE
    Ec_ = 0.5 * (E_edges[1:] + E_edges[:-1])
    out = np.zeros_like(Ec_)
    for i in range(r["N"]):
        x = float(np.clip(r["psiN_birth"][i], 0.0, 0.999))
        nloc = np.array([sim.bg.n(s, x) for s in ("e", "D", "T")])
        T = sim.bg.T_keV(x) * KEV_J
        up = UniformPlasma(nloc, np.full(3, T), Za, m)
        tau = up.spitzer_tau_s()
        Ec = 0.5 * m * up.critical_speed() ** 2
        Eth = THERMAL_FACTOR * 1.5 * T
        sel = (Ec_ > Eth) & (Ec_ < r["E_birth"][i])
        out[sel] += r["weight"][i] / ((2.0 * Ec_[sel] / tau) * (1.0 + (Ec / Ec_[sel]) ** 1.5))
    return out


# =======================================================
# ENTRY POINTS
# =======================================================
def run_alpha_slowing_down(cfg=None, n_markers=400, seed=20261123, sim=None, **kw):
    sim = FastIonSim(cfg) if sim is None else sim
    pos, vel, w, S = sim.birth_alphas(n_markers, seed)
    r = sim.follow("alpha", pos, vel, w, seed + 1, label="alpha", **kw)
    r["source_rate"] = S
    return sim, r, steady_state(sim, r)


def run_beam_slowing_down(cfg=None, n_markers=400, seed=20261124, sim=None, **kw):
    sim = FastIonSim(cfg) if sim is None else sim
    pos, vel, w, S = sim.birth_beams(n_markers, seed)
    r = sim.follow("beam", pos, vel, w, seed + 1, label="beam", **kw)
    r["source_rate"] = S
    return sim, r, steady_state(sim, r)


def prompt_loss_run(sim, species, n_markers, seed, t_end=1.0e-3):
    """First-orbit losses of many markers, collisions included, for t_end (1 ms,
    ~100 bounce times and < 1% of a slowing-down time). Returns (fraction of
    markers lost within PROMPT_LOSS_TIME, its 1-sigma error, fraction lost
    within t_end, wall time, lost-marker poloidal angles and energies)."""
    birth = sim.birth_alphas if species == "alpha" else sim.birth_beams
    pos, vel, w, _ = birth(n_markers, seed)
    r = sim.follow(species, pos, vel, w, seed + 1, t_max=t_end, progress=False)
    lost = r["status"] == LOST
    prompt = lost & (r["t"] < PROMPT_LOSS_TIME)
    p = float(np.mean(prompt))
    pw = r["weight"] * r["E_birth"]
    return {"n": n_markers, "frac_prompt": p, "err": math.sqrt(p * (1.0 - p) / n_markers),
            "frac_lost_1ms": float(np.mean(lost)), "wall_s": r["wall_s"],
            "frac_prompt_power": float(np.sum((r["weight"] * r["E_end"])[prompt]) / np.sum(pw)),
            "prompt": prompt, "theta": r["theta_end"], "E_end": r["E_end"], "weight": r["weight"],
            "R_end": r["state"][:, 0], "Z_end": r["state"][:, 2]}


def combine_losses(sim, r, ss, pr):
    """Wall load and loss fractions from two disjoint samples: losses before
    PROMPT_LOSS_TIME from the large first-orbit run pr, later losses from the
    full run r. Adds *_comb entries to ss."""
    th_edges = ss["theta_edges"]
    nb = len(th_edges) - 1

    def binned(theta, P):
        return np.bincount(np.clip(np.digitize(theta, th_edges) - 1, 0, nb - 1), weights=P, minlength=nb)

    late = (r["status"] == LOST) & (r["t"] >= PROMPT_LOSS_TIME)
    P_late = binned(r["theta_end"][late], (r["weight"] * r["E_end"])[late])
    P_prompt = binned(pr["theta"][pr["prompt"]], (pr["weight"] * pr["E_end"])[pr["prompt"]])
    ss["wall_power_comb"] = P_late + P_prompt
    ss["wall_flux_comb"] = np.where(ss["wall_area"] > 0, ss["wall_power_comb"] / ss["wall_area"], 0.0)
    ss["P_lost_prompt"] = float(P_prompt.sum())
    ss["P_lost_late"] = float(P_late.sum())
    ss["frac_lost_power_comb"] = (ss["P_lost_prompt"] + ss["P_lost_late"]) / ss["P_birth"]
    n_late = int(late.sum())
    ss["frac_lost_late"] = float(np.sum(r["weight"][late]) / np.sum(r["weight"]))
    ss["frac_lost_late_err"] = float(math.sqrt(max(n_late, 1)) / r["N"]) if n_late else float(1.0 / r["N"])
    ss["frac_prompt_comb"] = pr["frac_prompt"]
    ss["frac_prompt_comb_err"] = pr["err"]
    ss["frac_lost_comb"] = pr["frac_prompt"] + ss["frac_lost_late"]
    ss["n_lost_late"] = n_late
    return ss


# =======================================================
# PLOTS
# =======================================================
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"      # categorical slots 1-3, 5
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED)


def plot_all(sim, res, outdir, prompt_pts=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    paths = []
    x = sim.psi_centers

    # 1. alpha slowing-down distribution, simulated vs analytic
    r, ss = res["alpha"]
    edges = np.linspace(0.0, r["E_max_bin"], N_E_BINS + 1)
    Ec = 0.5 * (edges[1:] + edges[:-1])
    sim_spec = np.sum(r["weight"][:, None] * r["dwellE"], axis=0) / np.diff(edges)     # 1/J
    ana = analytic_slowing_down_spectrum(sim, r, edges)
    MeV = MEV_J
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    a1.step(Ec / MeV, sim_spec * MeV, where="mid", color=C1, lw=2, label="GC + Fokker-Planck (steady state)")
    a1.plot(Ec / MeV, ana * MeV, color=C2, lw=2, ls="--",
            label="classical, local $\\tau_s$, $E_c$ at birth (no transport)")
    a1.set_yscale("log")
    a1.set_xlabel("alpha energy [MeV]", color=MUTED)
    a1.set_ylabel("dN/dE  [alphas / MeV]", color=MUTED)
    a1.set_title("Steady-state alpha energy spectrum", color=INK, fontsize=11)
    a1.legend(frameon=False, fontsize=9)
    _style(a1)
    ok = (ana > 0) & (Ec < 0.97 * 3.5 * MeV) & (Ec > 0.1 * 3.5 * MeV)
    a2.plot(Ec[ok] / MeV, sim_spec[ok] / ana[ok], color=C1, lw=2, marker="o", ms=3)
    a2.axhline(1.0, color=MUTED, lw=1)
    a2.set_xlabel("alpha energy [MeV]", color=MUTED)
    a2.set_ylabel("simulated / classical", color=MUTED)
    a2.set_title("Ratio, 0.1 to 0.97 of the birth energy", color=INK, fontsize=11)
    _style(a2)
    fig.suptitle(f"Alpha slowing-down distribution ({r['N']} markers, {ss['n_thermalized']} thermalized, "
                 f"{ss['n_lost']} lost)", color=INK)
    fig.tight_layout()
    p = os.path.join(outdir, "alpha_slowing_down_distribution.png")
    fig.savefig(p, dpi=130, facecolor="white")
    plt.close(fig)
    paths.append(p)

    # 2. heating profiles
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    for ax, sp in zip(axs, ("alpha", "beam")):
        r, ss = res[sp]
        ax.step(x, ss["P_e"] / 1e6, where="mid", color=C1, lw=2,
                label=f"to electrons ({ss['P_e_total'] / 1e6:.2f} MW, {ss['frac_e']:.1%})")
        ax.step(x, ss["P_i"] / 1e6, where="mid", color=C2, lw=2, ls="--",
                label=f"to ions ({ss['P_i_total'] / 1e6:.2f} MW, {ss['frac_i']:.1%})")
        ax.set_xlabel("$\\psi_N$", color=MUTED)
        ax.set_ylabel("heating power density [MW/m$^3$]", color=MUTED)
        name = "Fusion alphas, 3.5 MeV" if sp == "alpha" else f"Beam deuterons, {sim.cfg.nbi_energy_keV / 1e3:g} MeV"
        ax.set_title(f"{name}: birth power {ss['P_birth'] / 1e6:.2f} MW", color=INK, fontsize=11)
        ax.legend(frameon=False, fontsize=9)
        ax.set_xlim(0, 1)
        _style(ax)
    fig.suptitle("Fast-ion heating of the background plasma (steady state)", color=INK)
    fig.tight_layout()
    p = os.path.join(outdir, "fast_ion_heating_profiles.png")
    fig.savefig(p, dpi=130, facecolor="white")
    plt.close(fig)
    paths.append(p)

    # 3. pressure profiles vs thermal
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bg = sim.bg
    p_th = (bg.n("e", x) + bg.n("D", x) + bg.n("T", x)) * bg.T_keV(x) * KEV_J
    beta_th = 2.0 * MU_0 * float(np.sum(p_th * sim.shell_volume) / sim.plasma_volume) / sim.cfg.B0**2
    ax.plot(x, p_th / 1e3, color=MUTED, lw=2, label=f"thermal e + D + T ($\\beta$ = {beta_th:.2%})")
    for sp, c, ls in (("alpha", C1, "-"), ("beam", C2, "--")):
        r, ss = res[sp]
        ax.step(x, ss["p"] / 1e3, where="mid", color=c, lw=2, ls=ls,
                label=f"{'alphas' if sp == 'alpha' else 'beam ions'} ($\\beta$ = {ss['beta']:.3%})")
    ax.set_yscale("log")
    ax.set_xlabel("$\\psi_N$", color=MUTED)
    ax.set_ylabel("pressure [kPa]", color=MUTED)
    ax.set_title("Fast-ion pressure vs thermal pressure (steady state, $B_0$ = "
                 f"{sim.cfg.B0:g} T)", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(0, 1)
    _style(ax)
    fig.tight_layout()
    p = os.path.join(outdir, "fast_ion_pressure_profiles.png")
    fig.savefig(p, dpi=130, facecolor="white")
    plt.close(fig)
    paths.append(p)

    # 4. wall load
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.6, 1]})
    th = np.degrees(0.5 * (res["alpha"][1]["theta_edges"][1:] + res["alpha"][1]["theta_edges"][:-1]))
    for sp, c, ls, mk in (("alpha", C1, "-", "o"), ("beam", C2, "--", "s")):
        r, ss = res[sp]
        comb = "wall_flux_comb" in ss
        flux = ss["wall_flux_comb"] if comb else ss["wall_flux"]
        Pl = (ss["P_lost_prompt"] + ss["P_lost_late"]) if comb else ss["P_lost"]
        fl = ss["frac_lost_power_comb"] if comb else ss["frac_lost_power"]
        a1.step(th, flux / 1e3, where="mid", color=c, lw=2, ls=ls,
                label=f"{'alphas' if sp == 'alpha' else 'beam ions'}: {Pl / 1e6:.3f} MW lost "
                      f"({fl:.2%} of birth power)")
        lost = r["status"] == LOST
        late = lost & (r["t"] >= PROMPT_LOSS_TIME)
        Rl, Zl = list(r["state"][late, 0]), list(r["state"][late, 2])
        if comb and prompt_pts is not None and sp in prompt_pts:
            pr = prompt_pts[sp]
            Rl += list(pr["R_end"][pr["prompt"]])
            Zl += list(pr["Z_end"][pr["prompt"]])
        a2.scatter(Rl, Zl, s=22, marker=mk, color=c, edgecolor="white", linewidth=0.8, zorder=3,
                   label=f"{'alpha' if sp == 'alpha' else 'beam'} loss points ({len(Rl)})")
    a1.set_xlabel("poloidal angle about the magnetic axis [deg]  (0 = outboard midplane, +90 = top)",
                  color=MUTED)
    a1.set_ylabel("heat flux on the last closed flux surface [kW/m$^2$]", color=MUTED)
    a1.set_title("Fast-ion wall load (steady state; first-orbit losses from the large prompt-loss run)",
                 color=INK, fontsize=11)
    a1.set_xlim(-180, 180)
    a1.legend(frameon=False, fontsize=9)
    _style(a1)
    a2.plot(sim.wall_R, sim.wall_Z, color=MUTED, lw=1.2)
    a2.plot([sim.field.R_axis], [sim.field.Z_axis], "+", color=MUTED, ms=9)
    a2.set_aspect("equal")
    a2.set_xlabel("R [m]", color=MUTED)
    a2.set_ylabel("Z [m]", color=MUTED)
    a2.set_title("Where markers cross the LCFS", color=INK, fontsize=11)
    a2.legend(frameon=False, fontsize=8.5, loc="upper left")
    _style(a2)
    fig.tight_layout()
    p = os.path.join(outdir, "fast_ion_wall_load.png")
    fig.savefig(p, dpi=130, facecolor="white")
    plt.close(fig)
    paths.append(p)
    return paths


# =======================================================
# OUTPUT FILES
# =======================================================
def save_sample(sim, res, path, max_points=2000):
    """Downsampled trajectories of the tracked markers plus every marker's birth
    and end events, for later visualisation."""
    import gc_push as G
    out = {"R_axis": sim.field.R_axis, "Z_axis": sim.field.Z_axis,
           "wall_R": sim.wall_R, "wall_Z": sim.wall_Z,
           "status_codes": np.array(["active", "thermalized", "lost", "timeout"])}
    for sp, (r, _) in res.items():
        n = r["rec"].shape[0]
        traj = np.full((n, max_points, 5), np.nan)
        for k in range(n):
            m = int(r["rec_n"][k])
            take = np.unique(np.linspace(0, m - 1, min(m, max_points)).astype(int)) if m else []
            traj[k, :len(take)] = r["rec"][k, take]
        out[f"{sp}_track"] = traj                          # (marker, point, [R, phi, Z, E_J, t_s])
        out[f"{sp}_track_marker"] = np.arange(n)
        # one collisionless first orbit (20 us) of each tracked marker, finely sampled
        q, m = r["q"], r["m"]
        st = r["state0"][:n].copy()
        v = np.sqrt(np.sum(r["vel0"][:n] ** 2, axis=1))
        dt = G.dt_transit_fraction(sim.field, v)
        n_steps = int(np.ceil(2.0e-5 / dt.min()))
        rec_every = max(1, n_steps // 400)
        fo, _, _ = G.push_record(sim.field, st, r["mu0"][:n].copy(), q, m, dt, n_steps, rec_every)
        out[f"{sp}_first_orbit"] = fo                      # (marker, point, [R, phi, Z, v_par])
        for key in ("weight", "E_birth", "psiN_birth", "xi_birth", "status", "t", "E_end", "psiN_end",
                    "theta_end", "dE_e", "dE_i"):
            out[f"{sp}_{key}"] = r[key]
        out[f"{sp}_birth_pos"] = r["pos0"]
        out[f"{sp}_end_RphiZ"] = r["state"][:, :3]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)


def save_results(sim, res, extra, path):
    out = {"psiN_centers": sim.psi_centers, "shell_volume": sim.shell_volume}
    for sp, (r, ss) in res.items():
        for k, v in ss.items():
            if isinstance(v, (str, dict)) and k != "err":
                continue
            if k == "err":
                for ek, ev in v.items():
                    out[f"{sp}_err_{ek}"] = ev
            else:
                out[f"{sp}_{k}"] = np.asarray(v)
        out[f"{sp}_source_rate"] = r["source_rate"]
    for k, v in extra.items():
        out[k] = np.asarray(v)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **out)


def summary_lines(sim, res, prompt):
    L = []
    for sp in ("alpha", "beam"):
        r, s = res[sp]
        e = s["err"]
        L.append(f"[{sp}] {r['N']} markers, wall {s['wall_s']:.0f} s, {s['steps_total'] / 1e9:.2f}e9 GC steps")
        L.append(f"  birth power {s['P_birth'] / 1e6:.3f} MW (source {r['source_rate']:.3e} /s)")
        L.append(f"  to electrons {s['P_e_total'] / 1e6:.3f} MW ({s['frac_e']:.2%} +- {e['P_e']:.1%} rel)")
        L.append(f"  to ions      {s['P_i_total'] / 1e6:.3f} MW ({s['frac_i']:.2%} +- {e['P_i']:.1%} rel)")
        L.append(f"  lost         {s['P_lost'] / 1e6:.4f} MW ({s['frac_lost_power']:.2%} +- "
                 f"{e['frac_lost_power_abs']:.2%} abs of birth power); markers lost "
                 f"{s['n_lost']} ({s['frac_lost']:.2%} +- {e['frac_lost_abs']:.2%} abs), prompt {s['n_prompt']}")
        L.append(f"  thermalized sink {s['P_thermalized'] / 1e6:.4f} MW, q dPhi {s['P_qdPhi'] / 1e6:+.4f} MW, "
                 f"balance residual {s['balance_residual']:.2e}")
        L.append(f"  beta {s['beta']:.3e} (+- {e['beta']:.1%} rel), N_fast {s['N_fast']:.3e} (+- {e['N_fast']:.1%})")
        L.append(f"  slowing-down time mean {s['t_sd_mean']:.3f} s, median {s['t_sd_median']:.3f} s "
                 f"(10-90%: {s['t_sd_p10']:.3f}-{s['t_sd_p90']:.3f} s); timeouts {s['n_timeout']}")
        if sp in prompt:
            p = prompt[sp]
            L.append(f"  prompt-loss run: {p['n']} markers x 1 ms, wall {p['wall_s']:.0f} s: "
                     f"prompt (< 100 us) {p['frac_prompt']:.3%} +- {p['err']:.3%} of markers, "
                     f"{p['frac_prompt_power']:.3%} of birth power; lost within 1 ms {p['frac_lost_1ms']:.3%}")
            L.append(f"  combined: lost {s['frac_lost_comb']:.3%} of markers (prompt {s['frac_prompt_comb']:.3%} "
                     f"+- {s['frac_prompt_comb_err']:.3%}, later {s['frac_lost_late']:.3%} +- "
                     f"{s['frac_lost_late_err']:.3%}), {s['frac_lost_power_comb']:.3%} of birth power")
    return L


def main(n_alpha=150, n_beam=150, n_prompt=4000, n_track=16, outdir=None):
    t0 = time.perf_counter()
    sim = FastIonSim()
    outdir = os.path.join(REPO, "docs", "images") if outdir is None else outdir
    res = {}
    for sp, n, fn in (("alpha", n_alpha, run_alpha_slowing_down), ("beam", n_beam, run_beam_slowing_down)):
        print(f"[fastion] following {n} {sp} markers to thermalization or loss ...", flush=True)
        _, r, ss = fn(sim=sim, n_markers=n, n_track=n_track)
        res[sp] = (r, ss)
        print(f"[fastion] {sp}: {ss['wall_s']:.0f} s", flush=True)
    prompt = {}
    for sp, seed in (("alpha", 31), ("beam", 41)):
        if n_prompt:
            print(f"[fastion] first-orbit loss run: {n_prompt} {sp} markers x 1 ms ...", flush=True)
            prompt[sp] = prompt_loss_run(sim, sp, n_prompt, seed)
            combine_losses(sim, res[sp][0], res[sp][1], prompt[sp])
    paths = plot_all(sim, res, outdir, prompt_pts=prompt)
    save_sample(sim, res, os.path.join(REPO, "data", "fastion_sample.npz"))
    extra = {f"prompt_{sp}_{k}": v for sp, d in prompt.items() for k, v in d.items()
             if k in ("n", "frac_prompt", "err", "frac_lost_1ms", "wall_s", "frac_prompt_power")}
    extra["total_wall_s"] = time.perf_counter() - t0
    save_results(sim, res, extra, os.path.join(REPO, "data", "fastion_results.npz"))
    for line in summary_lines(sim, res, prompt):
        print(line)
    print(f"[fastion] plots: {', '.join(os.path.relpath(p, REPO) for p in paths)}")
    print(f"[fastion] total wall {time.perf_counter() - t0:.0f} s")
    return sim, res, prompt


if __name__ == "__main__":
    main()
