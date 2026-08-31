#!/usr/bin/env python3
"""Record and compare a fingerprint manifest of the simulation's PNG output.

The run writes 17 diagnostic plots into the working directory. This tool walks
that directory for them, records each one's SHA-256 and pixel dimensions to a
JSON manifest, and can later diff a fresh run against a saved manifest to say
which plots changed -- a cheap regression check for refactors that are meant to
leave the physics alone.

Usage:
    # After a run, save the current plots as the baseline:
    python tools/plot_manifest.py record --out plot_manifest.json

    # After a later run, diff against that baseline:
    python tools/plot_manifest.py compare --baseline plot_manifest.json

Both modes warn loudly (and exit non-zero) if any of the 17 expected plots is
missing from the working directory.

Exit codes:
    0  all 17 present; for compare, every plot byte-identical to the baseline
    1  a plot is missing, or (compare) a plot changed / is new
    2  usage error -- e.g. the baseline manifest does not exist

Pure standard library: PNG dimensions are read from the IHDR chunk, so the tool
runs anywhere the simulation does, with or without Pillow installed.
"""

import argparse
import datetime
import hashlib
import json
import os
import sys

# The 17 PNGs a full run is expected to produce: 16 from diagnostics.py plus
# tokamak_reactor_3d.png, which diagnostics.py delegates to visualizer.py.
EXPECTED_PLOTS = (
    "benchmark_scaling.png",
    "charge_density_map.png",
    "potential_field_map.png",
    "plasma_stored_energy_time.png",
    "tokamak_reactor_2d.png",
    "radial_profiles.png",
    "phase_space_map.png",
    "instability_growth.png",
    "fusion_power_density.png",
    "alpha_orbits.png",
    "tokamak_reactor_3d.png",
    "alpha_heating_balance.png",
    "radiation_loss_profile.png",
    "lawson_q_factor.png",
    "plasma_oscillation_frequency.png",
    "disruption_mitigation.png",
    "fusion_cross_section.png",
)

DEFAULT_MANIFEST = "plot_manifest.json"
MANIFEST_VERSION = 1

# Directories that never hold run output; skipped when walking for plots.
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "env", "node_modules",
             ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints"}

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


# =======================================================
# PNG INSPECTION
# =======================================================
def png_dimensions(path):
    """Return (width, height) from a PNG's IHDR chunk.

    Raises ValueError if the file is not a PNG -- a truncated or half-written
    plot should be reported as a problem, not silently fingerprinted.
    """
    with open(path, "rb") as fh:
        header = fh.read(24)
    if len(header) < 24 or not header.startswith(PNG_SIGNATURE):
        raise ValueError(f"not a PNG file: {path}")
    if header[12:16] != b"IHDR":
        raise ValueError(f"PNG missing IHDR chunk: {path}")
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    return width, height


def sha256_of(path, chunk_size=1 << 20):
    """SHA-256 of a file, read in chunks (the 3D renders run to several MB)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path, root):
    """Record for one plot: hash, pixel dimensions, size, and relative path."""
    width, height = png_dimensions(path)
    return {
        "sha256": sha256_of(path),
        "width": width,
        "height": height,
        "bytes": os.path.getsize(path),
        "path": os.path.relpath(path, root).replace(os.sep, "/"),
    }


# =======================================================
# LOCATING THE PLOTS
# =======================================================
def find_plots(root):
    """Locate the expected PNGs under root.

    Files in root itself win; otherwise the shallowest match found by walking
    the tree is used, so the tool still works when a run is configured to drop
    its output in a subdirectory. Returns (found, missing) where found maps
    filename -> absolute path.
    """
    wanted = set(EXPECTED_PLOTS)
    found = {}

    for name in wanted:
        candidate = os.path.join(root, name)
        if os.path.isfile(candidate):
            found[name] = candidate

    remaining = wanted - set(found)
    if remaining:
        # os.walk is top-down, so the first hit for a name is the shallowest.
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames
                                 if d not in SKIP_DIRS and not d.startswith("."))
            for name in sorted(remaining & set(filenames)):
                found[name] = os.path.join(dirpath, name)
            remaining -= set(found)
            if not remaining:
                break

    missing = [name for name in EXPECTED_PLOTS if name not in found]
    return found, missing


def warn_missing(missing, root):
    """Loud, unmissable banner naming every expected plot that is absent."""
    if not missing:
        return
    sys.stdout.flush()   # keep the banner below the report when piped
    bar = "!" * 72
    print("", file=sys.stderr)
    print(bar, file=sys.stderr)
    print(f"!! WARNING: {len(missing)} of {len(EXPECTED_PLOTS)} EXPECTED PLOTS "
          f"ARE MISSING", file=sys.stderr)
    print(f"!! searched: {root}", file=sys.stderr)
    print(bar, file=sys.stderr)
    for name in missing:
        print(f"!!   MISSING: {name}", file=sys.stderr)
    print(bar, file=sys.stderr)
    print("!! The run did not produce a complete plot set; only the plots that",
          file=sys.stderr)
    print("!! do exist were fingerprinted.", file=sys.stderr)
    print(bar, file=sys.stderr)
    print("", file=sys.stderr)


def collect(root):
    """Fingerprint every expected plot present under root."""
    found, missing = find_plots(root)
    plots = {}
    unreadable = []
    for name in EXPECTED_PLOTS:
        if name not in found:
            continue
        try:
            plots[name] = fingerprint(found[name], root)
        except (OSError, ValueError) as exc:
            unreadable.append(name)
            print(f"[ERROR] could not fingerprint {name}: {exc}", file=sys.stderr)
    if unreadable:
        missing = missing + unreadable
        missing = [n for n in EXPECTED_PLOTS if n in set(missing)]
    return plots, missing


# =======================================================
# MODES
# =======================================================
def cmd_record(args):
    root = os.path.abspath(args.dir)
    plots, missing = collect(root)

    manifest = {
        "version": MANIFEST_VERSION,
        "generated": datetime.datetime.now(datetime.timezone.utc)
                             .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "root": root,
        "expected_count": len(EXPECTED_PLOTS),
        "missing": missing,
        "plots": plots,
    }

    with open(args.out, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"[MANIFEST] wrote {args.out}")
    print(f"[MANIFEST] recorded {len(plots)}/{len(EXPECTED_PLOTS)} plots "
          f"from {root}")
    for name in EXPECTED_PLOTS:
        if name in plots:
            entry = plots[name]
            print(f"    {name:<34} {entry['width']:>5}x{entry['height']:<5} "
                  f"{entry['sha256'][:12]}")

    warn_missing(missing, root)
    return 1 if missing else 0


def cmd_compare(args):
    root = os.path.abspath(args.dir)

    if not os.path.isfile(args.baseline):
        print(f"[ERROR] baseline manifest not found: {args.baseline}",
              file=sys.stderr)
        print("[ERROR] run 'plot_manifest.py record' first.", file=sys.stderr)
        return 2

    with open(args.baseline) as fh:
        try:
            baseline = json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"[ERROR] baseline manifest is not valid JSON: {exc}",
                  file=sys.stderr)
            return 2

    old_plots = baseline.get("plots", {})
    new_plots, missing = collect(root)

    allowed_missing = set(args.allow_missing or [])
    unknown = allowed_missing - set(EXPECTED_PLOTS)
    if unknown:
        print(f"[ERROR] --allow-missing names plots that are not expected: "
              f"{', '.join(sorted(unknown))}", file=sys.stderr)
        return 2
    excused = [n for n in missing if n in allowed_missing]
    missing = [n for n in missing if n not in allowed_missing]

    changed, unchanged, added = [], [], []
    for name in EXPECTED_PLOTS:
        old, new = old_plots.get(name), new_plots.get(name)
        if new is None:
            continue          # absent now -- reported by warn_missing below
        if old is None:
            added.append(name)
        elif old.get("sha256") != new.get("sha256"):
            changed.append(name)
        else:
            unchanged.append(name)

    dropped = [n for n in EXPECTED_PLOTS if n in old_plots
               and n not in new_plots and n not in allowed_missing]

    print(f"[COMPARE] baseline: {args.baseline} "
          f"(generated {baseline.get('generated', 'unknown')})")
    print(f"[COMPARE] current:  {root}")
    print(f"[COMPARE] {len(unchanged)} unchanged, {len(changed)} changed, "
          f"{len(added)} new, {len(dropped)} gone, {len(missing)} missing")
    if excused:
        print(f"[COMPARE] excused as absent by --allow-missing: "
              f"{', '.join(excused)}")
    if args.presence_only:
        print("[COMPARE] presence-only: content changes are reported but "
              "do not fail.")
    print("")

    if changed:
        print("CHANGED PLOTS")
        print("-" * 72)
        for name in changed:
            old, new = old_plots[name], new_plots[name]
            print(f"  {name}")
            print(f"      sha256 {old['sha256'][:16]} -> {new['sha256'][:16]}")
            if (old.get("width"), old.get("height")) != (new["width"], new["height"]):
                print(f"      pixels {old.get('width')}x{old.get('height')} -> "
                      f"{new['width']}x{new['height']}   <-- DIMENSIONS CHANGED")
            else:
                print(f"      pixels {new['width']}x{new['height']} (unchanged)")
            print(f"      bytes  {old.get('bytes')} -> {new['bytes']}")
        print("")

    if added:
        print("NEW PLOTS (not in baseline)")
        print("-" * 72)
        for name in added:
            entry = new_plots[name]
            print(f"  {name:<34} {entry['width']}x{entry['height']} "
                  f"{entry['sha256'][:12]}")
        print("")

    if dropped:
        print("GONE (in baseline, absent now)")
        print("-" * 72)
        for name in dropped:
            print(f"  {name}")
        print("")

    if args.verbose and unchanged:
        print("UNCHANGED")
        print("-" * 72)
        for name in unchanged:
            print(f"  {name}")
        print("")

    warn_missing(missing, root)

    if not changed and not added and not dropped and not missing:
        print(f"[COMPARE] all {len(EXPECTED_PLOTS) - len(excused)} compared "
              "plots byte-identical to the baseline.")
        return 0

    # --presence-only asks a narrower question -- "did every plot still get
    # produced" -- so a plot whose CONTENT moved is reported and forgiven.
    # A plot that is absent still fails: that is the whole question.
    if args.presence_only and not missing and not dropped:
        print("[COMPARE] every expected plot was produced "
              "(content differences ignored).")
        return 0
    return 1


# =======================================================
# ENTRY POINT
# =======================================================
def build_parser():
    parser = argparse.ArgumentParser(
        description="Record or compare a SHA-256/dimension manifest of the "
                    "simulation's 17 output plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="exit codes: 0 = clean, 1 = missing/changed plots, 2 = usage error",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_record = sub.add_parser("record", help="fingerprint the current plots")
    p_record.add_argument("--dir", default=".",
                          help="directory to search (default: working directory)")
    p_record.add_argument("--out", default=DEFAULT_MANIFEST,
                          help=f"manifest to write (default: {DEFAULT_MANIFEST})")
    p_record.set_defaults(func=cmd_record, presence_only=False, allow_missing=[])

    p_compare = sub.add_parser("compare", help="diff current plots vs a manifest")
    p_compare.add_argument("--dir", default=".",
                           help="directory to search (default: working directory)")
    p_compare.add_argument("--baseline", default=DEFAULT_MANIFEST,
                           help=f"manifest to diff against (default: {DEFAULT_MANIFEST})")
    p_compare.add_argument("-v", "--verbose", action="store_true",
                           help="also list the unchanged plots")
    p_compare.add_argument("--presence-only", action="store_true",
                           help="fail only on missing plots; report content "
                                "changes without failing. For runs whose "
                                "plots are legitimately not byte-stable "
                                "(e.g. benchmark_scaling.png plots measured "
                                "wall-clock times)")
    p_compare.add_argument("--allow-missing", metavar="NAME", default=[],
                           type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                           help="comma-separated plots that are expected to "
                                "be absent in this configuration; reported "
                                "as excused rather than failing")
    p_compare.set_defaults(func=cmd_compare)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
