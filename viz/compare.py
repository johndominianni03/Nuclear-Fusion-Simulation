"""Compare the simulated numbers behind two scenes.

    venv/bin/python -m viz.compare viz/scenes/rerun_baseline.json viz/scenes/rerun_B0_8T.json

For each species: slowing-down time of the thermalized markers (median and
mean), markers lost, the median first-orbit width (range of the guiding
centre's minor radius over the stored 20 us collisionless first orbit), and
the share of the birth energy given to electrons / ions. With 16 markers per
species these are an illustration, not statistics.
"""
import os
import sys

from viz.rerun import summary_numbers
from viz.scene import SceneConfig, source_path

ROWS = (("t_sd_median_s", "slowing-down time, median [s]", "{:.3f}"),
        ("t_sd_mean_s", "slowing-down time, mean [s]", "{:.3f}"),
        ("n_lost", "markers lost", "{:d}"),
        ("first_orbit_width_median_m", "first-orbit width, median [m]", "{:.3f}"),
        ("frac_e", "birth energy to electrons", "{:.1%}"),
        ("frac_i", "birth energy to ions", "{:.1%}"))


def table(paths_or_cfgs):
    cols = []
    for x in paths_or_cfgs:
        cfg = SceneConfig.from_json(x) if isinstance(x, str) else x
        path = source_path(cfg)
        if not os.path.exists(path):
            raise SystemExit(f"{path} missing: run `venv/bin/python -m viz.rerun --config {x}` first")
        label = ", ".join(f"{k} = {v:g}" for k, v in cfg.overrides) or (
            "stored sample" if cfg.source == "sample" else "baseline")
        cols.append((label, summary_numbers(path)))
    lines = []
    for sp in ("alpha", "beam"):
        n = [c[1][sp]["n_markers"] for c in cols]
        lines.append(f"{sp} ({' / '.join(map(str, n))} markers)".ljust(34)
                     + "".join(c[0].rjust(16) for c in cols))
        for key, name, fmt in ROWS:
            lines.append(f"  {name}".ljust(34) + "".join(fmt.format(c[1][sp][key]).rjust(16) for c in cols))
    return "\n".join(lines)


if __name__ == "__main__":
    print(table(sys.argv[1:]))
