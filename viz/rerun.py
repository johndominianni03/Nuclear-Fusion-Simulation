"""Re-simulate a scene's markers with physics overrides ("rerun" source).

    venv/bin/python -m viz.rerun --config scene.json [--set key=value ...]

Runs fastion.run_alpha_slowing_down and run_beam_slowing_down (unchanged: the
same seeds, 20261123 / 20261124 for the births and seed + 1 for the Monte
Carlo follow) for n_alpha / n_beam markers, all of them tracked, on the
scenario of config.py plus the scene's overrides, and writes the result with
fastion.save_sample into viz_cache/rerun_<physics hash>/sample.npz plus a
meta.json (inputs, wall times, summary numbers). The file is written to a
temporary name and renamed, so an interrupted run never leaves a cache entry;
rerunning simply starts again (16 + 16 markers take ~10 min on 6 cores).
The fast-ion kernel is bit-identical at any thread count, so the same config
always gives the same file.
"""
import contextlib
import io
import json
import os
import sys
import time

import numpy as np

from viz.scene import SceneConfig, source_path, SEEDS


def summary_numbers(path):
    """Slowing-down time, loss fraction and first-orbit width per species from
    a stored sample file."""
    out = {}
    with np.load(path, allow_pickle=False) as d:
        Rax, Zax = float(d["R_axis"]), float(d["Z_axis"])
        for sp in ("alpha", "beam"):
            st, t = d[f"{sp}_status"], d[f"{sp}_t"]
            th = st == 1
            fo = d[f"{sp}_first_orbit"]
            r = np.hypot(fo[:, :, 0] - Rax, fo[:, :, 2] - Zax)      # minor radius of the GC
            width = r.max(axis=1) - r.min(axis=1)
            out[sp] = {
                "n_markers": int(st.size),
                "n_thermalized": int(th.sum()), "n_lost": int((st == 2).sum()),
                "n_timeout": int((st == 3).sum()),
                "t_sd_median_s": float(np.median(t[th])) if th.any() else float("nan"),
                "t_sd_mean_s": float(np.mean(t[th])) if th.any() else float("nan"),
                "loss_fraction": float((st == 2).mean()),
                "first_orbit_width_median_m": float(np.median(width)),
                "frac_e": float(d[f"{sp}_dE_e"].sum() / d[f"{sp}_E_birth"].sum()),
                "frac_i": float(d[f"{sp}_dE_i"].sum() / d[f"{sp}_E_birth"].sum()),
            }
    return out


def run(cfg: SceneConfig, quiet=False):
    import fastion
    if cfg.source != "rerun":
        raise ValueError("viz.rerun needs a scene with source='rerun'")
    path = source_path(cfg)
    if os.path.exists(path):
        print(f"[viz.rerun] cached: {path}")
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        sim_cfg = cfg.make_cfg()
    sim = fastion.FastIonSim(sim_cfg)
    res, walls = {}, {}
    for sp, n, fn in (("alpha", cfg.n_alpha, fastion.run_alpha_slowing_down),
                      ("beam", cfg.n_beam, fastion.run_beam_slowing_down)):
        print(f"[viz.rerun] following {n} {sp} markers (overrides {dict(cfg.overrides)}) ...", flush=True)
        _, r, ss = fn(sim=sim, n_markers=n, seed=SEEDS[sp], n_track=n)
        res[sp] = (r, ss)
        walls[sp] = float(r["wall_s"])
        print(f"[viz.rerun] {sp}: {walls[sp]:.0f} s", flush=True)
    tmp = path[:-4] + ".partial.npz"
    fastion.save_sample(sim, res, tmp)
    os.replace(tmp, path)
    meta = {
        "physics_key": cfg.physics_key(), "physics_hash": cfg.physics_hash(),
        "inputs": {k: float(getattr(sim_cfg, k)) for k in
                   ("B0", "Q_SAFETY_TARGET", "T_thermal_keV", "N_AVG_TARGET", "nbi_energy_keV", "P_NBI_MW")},
        "follow_wall_s": walls, "total_wall_s": time.perf_counter() - t0,
        "summary": summary_numbers(path),
    }
    with open(os.path.join(os.path.dirname(path), "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    print(f"[viz.rerun] wrote {path} ({meta['total_wall_s']:.0f} s)", flush=True)
    return path


def main(argv=None):
    from viz.make import parse_config
    cfg = parse_config(argv)
    run(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
