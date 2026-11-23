"""CLI for the fusion-cycle animation.

    venv/bin/python -m viz.make --config scene.json [--set key=value ...]

--set takes any SceneConfig field; values are parsed as JSON when they can be
(numbers, true/false, lists), else kept as strings. Physics overrides go in
as  --set overrides.B0=8  (source must be "rerun").
"""
import argparse
import json
import sys

from viz.scene import SceneConfig


def _value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def build_parser():
    p = argparse.ArgumentParser(prog="python -m viz.make", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="scene JSON (SceneConfig fields); defaults if omitted")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override one SceneConfig field (repeatable); overrides.<name>=v for physics")
    return p


def config_from_args(args):
    d = SceneConfig().to_dict()
    if args.config:
        with open(args.config) as f:
            d.update(json.load(f))
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set needs KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        if k.startswith("overrides."):
            ov = dict(d.get("overrides") or {})
            ov[k.split(".", 1)[1]] = float(v)
            d["overrides"] = ov
        else:
            d[k] = _value(v)
    return SceneConfig.from_dict(d)


def parse_config(argv=None):
    return config_from_args(build_parser().parse_args(argv))


def main(argv=None):
    cfg = parse_config(argv)
    print(cfg.to_json(indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
