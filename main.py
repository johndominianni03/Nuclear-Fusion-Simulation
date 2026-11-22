import numpy as np

from physics_engine import (
    compute_poloidal_field_grids,           # psi-derived poloidal B (real grad-B / mirror force)
    compute_flux_profiles,                  # weighted per-species moments on psi_N shells
    compute_thermal_fusion_power,           # thermal D-T power from those profiles
    map_flux_profile_to_grid,
    species_masses,
)
from config import SimulationConfiguration
import initialization
import diagnostics
from profiling import StageProfiler
from kernels import vectorized_gather_and_B
from reactor_cpu import _run_reactor_loop_cpu
from reactor_gpu import _run_reactor_loop_gpu


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
# NOT a deprecation of the GPU path: it stays fully functional and forceable, it is
# simply not being optimized further. (This line used to say "step 12 optimizes that
# loop and step 13 reassesses this number". Both are retired: step 12 is a SKIP -- a
# decision, not a deferral -- and step 13 is OPTIONAL and moot, since step 9 made the
# CPU loop ~2.7x faster and moved the crossover further from the GPU, not closer. Only
# a hardware change, a discrete NVIDIA card in particular, would justify revisiting.)
# To force the GPU loop, set this to an int <= cfg.initial_thermal_count,
# either by editing this line (what README.md documents) or at runtime without editing:
#     import main; main.GPU_PARTICLE_THRESHOLD = 0; main.run_reactor_steady_state()
# The dispatch reads this global when run_reactor_steady_state is called, so the runtime
# form works; `cfg.PROFILE = True` prints which loop was chosen.


def package_reactor_results(cfg, eq, engine, B_R_pol_grid, B_Z_pol_grid,
                            pos_np, vel_np, type_np,
                            history_tracks, tracked_type, tracked_lastpos,
                            tracked_lastvel, tracked_lost, prof=None):
    """Turns the reactor loop's raw output into the arrays diagnostics plots.

    Lifted verbatim out of run_reactor_steady_state so a regression test can
    reach these products without rendering anything: the loop is the physics,
    this is the packaging, and rendering is a third step. run_reactor_steady_state
    calls this and forwards the contents to diagnostics exactly as before.

    `eq` supplies psi for the flux-surface profiles and the fusion power map.

    Returns a dict of exactly the products the caller forwards onward.

    `prof` is an optional StageProfiler. When None (the default, and what
    tests/test_regression.py passes) every hook below is a no-op.
    """
    if prof is None:
        prof = StageProfiler(False)

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

    print("[SYSTEM] Thermal D-T fusion power from flux-surface profiles")
    _t = prof.mark()
    # Weighted thermal-ion n(psi_N), T(psi_N); n_D = n_T = n_thermal / 2 is an
    # assumption (there is no tritium species). Beam ions and alphas are excluded.
    flux_profiles = compute_flux_profiles(pos_np, vel_np, type_np, eq, species_masses(cfg),
                                          weight=cfg.MACRO_WEIGHT, B0=cfg.B0)
    P_fusion_profile, total_fusion_power_watts = compute_thermal_fusion_power(flux_profiles, cfg)
    P_fusion_grid = map_flux_profile_to_grid(
        P_fusion_profile, flux_profiles["psiN_centers"], eq,
        cfg.R_min, cfg.R_max, cfg.Z_min, cfg.Z_max, cfg.nR, cfg.nZ)
    prof.add("pkg: flux profiles + thermal fusion power", _t)

    print("==================================================")
    print(f" THERMAL D-T FUSION POWER (50/50 D-T assumed, beams and alphas excluded): "
          f"{total_fusion_power_watts:.3e} W ")
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
        "P_fusion_profile": P_fusion_profile,
        "total_fusion_power_watts": total_fusion_power_watts,
        "flux_profiles": flux_profiles,
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
     inventory_history, energy_history_keV,
     temp_history, trigger_time, T_core,
     lawson_triple_product_history, lawson_first_measured_step, rho_grid, phi_grid
     ) = loop_fn(cfg, engine, pos_tensor, vel_tensor, type_tensor, rho_grid, phi_grid, E_R_grid, E_Z_grid, B_R_pol_grid, B_Z_pol_grid)

    _t = outer.mark()
    packaged = package_reactor_results(
        cfg, eq, engine, B_R_pol_grid, B_Z_pol_grid,
        pos_np, vel_np, type_np,
        history_tracks, tracked_type, tracked_lastpos,
        tracked_lastvel, tracked_lost, prof=outer,
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
            R_phase, v_parallel_phase, P_fusion_grid,
            alpha_particles=mock_alphas,
            lawson_history=lawson_triple_product_history,
            lawson_first_measured_step=lawson_first_measured_step
        )


if __name__ == "__main__":
    # The reactor run and nothing else. The benchmark sweep lives in
    # benchmarks.run_hpc_benchmark; call it explicitly if you want it.
    # tests/test_regression.py's compare-plots child script drives both,
    # because the plot manifest covers both.
    #
    # save_plots=True: this is the full-artifact entry point, it exists to
    # regenerate the plots, so it asks for them explicitly rather than
    # inheriting the cfg.PROFILE default.
    run_reactor_steady_state(save_plots=True)