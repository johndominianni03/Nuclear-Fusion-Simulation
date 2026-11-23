"""CLI for the fusion-cycle animation: frames (cached, resumable) -> GIF + MP4.

    venv/bin/python -m viz.make --config scene.json [--set key=value ...]
                                [--out-dir DIR] [--jobs N] [--background]
                                [--frames-only] [--print-config]

--set takes any SceneConfig field; values are parsed as JSON when they can be
(numbers, true/false, lists), else kept as strings. Physics overrides go in as
--set overrides.B0=8 (needs source "rerun").

Frames go to viz_cache/<scene hash>/frame_00123.png. A frame that already
exists is not drawn again, and every frame is written to a temporary name and
renamed, so an interrupted render resumes at the first missing frame. A rerun
scene whose simulation is not cached yet is simulated first (viz.rerun,
~10 min for 16 + 16 markers). --background re-launches the same command
detached (nohup-style) with its log in viz_cache/<scene hash>/make.log.
Outputs: <out-dir>/<output>.gif (Pillow, one global palette of gif_colors, no dither,
gif_width wide, frames dropped until <= gif_max_mb) and <output>.mp4 (ffmpeg
from imageio-ffmpeg, H.264, -preset veryfast, CRF 20, yuv420p), plus
viz_cache/<scene hash>/<output>_render.json with frame counts, timings and sizes.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

from viz.scene import SceneConfig, CACHE_DIR, REPO, source_path


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
    p.add_argument("--out-dir", default="viz_out", help="where the GIF / MP4 go (default viz_out/)")
    p.add_argument("--jobs", type=int, default=4, help="frame-rendering processes (default 4)")
    p.add_argument("--background", action="store_true", help="detach and log to viz_cache/<hash>/make.log")
    p.add_argument("--frames-only", action="store_true", help="render the frames, skip GIF/MP4")
    p.add_argument("--print-config", action="store_true", help="print the resolved scene and its hashes")
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


def frame_dir(cfg):
    return os.path.join(CACHE_DIR, cfg.scene_hash())


def frame_path(cfg, i):
    return os.path.join(frame_dir(cfg), f"frame_{i:05d}.png")


# =======================================================
# FRAMES
# =======================================================
_WORKER = {}


def _worker_init(cfg_json):
    from viz.scene import load_scene
    from viz.render import Prepared
    cfg = SceneConfig.from_json(cfg_json)
    _WORKER["cfg"] = cfg
    _WORKER["prep"] = Prepared(load_scene(cfg))


def _worker_frame(i):
    from viz.render import save_frame
    save_frame(_WORKER["prep"], i, frame_path(_WORKER["cfg"], i))
    return i


def render_frames(cfg, jobs=4, log=print):
    """Draw every missing frame; returns (n_drawn, n_reused, wall_s)."""
    os.makedirs(frame_dir(cfg), exist_ok=True)
    with open(os.path.join(frame_dir(cfg), "scene.json"), "w") as f:
        f.write(cfg.to_json(indent=2))
    n = cfg.total_frames()
    todo = [i for i in range(n) if not os.path.exists(frame_path(cfg, i))]
    t0 = time.perf_counter()
    log(f"[viz.make] {n} frames, {n - len(todo)} cached, {len(todo)} to draw -> {frame_dir(cfg)}")
    if todo:
        if jobs <= 1:
            _worker_init(cfg.to_json())
            done_iter = map(_worker_frame, todo)
            pool = None
        else:
            import multiprocessing as mp
            pool = mp.get_context("spawn").Pool(jobs, initializer=_worker_init, initargs=(cfg.to_json(),))
            done_iter = pool.imap_unordered(_worker_frame, todo, chunksize=2)
        last = time.perf_counter()
        for k, _ in enumerate(done_iter, 1):
            now = time.perf_counter()
            if now - last > 20.0 or k == len(todo):
                last = now
                log(f"[viz.make] {k}/{len(todo)} frames drawn, {now - t0:.0f} s")
        if pool is not None:
            pool.close()
            pool.join()
    return len(todo), n - len(todo), time.perf_counter() - t0


# =======================================================
# ENCODING
# =======================================================
def encode_gif(cfg, path, log=print):
    """Downscale to gif_width; one global palette of gif_colors colours built
    from 12 sampled frames plus a swatch of the theme colours (so the energy
    ramp and the rare 'lost' red always get palette entries); no dither; drop
    every other frame (halving fps, same clip length) only if the file is
    still > gif_max_mb. Returns (bytes, n_frames, fps, step)."""
    from PIL import Image
    from viz.render import RAMP, CRITICAL, TEXT, TEXT2, MUTED, SURFACE
    files = sorted(glob.glob(os.path.join(frame_dir(cfg), "frame_*.png")))
    W = cfg.gif_width
    H = int(round(cfg.height * W / cfg.width))

    def load(f):
        return Image.open(f).convert("RGB").resize((W, H), Image.LANCZOS)

    sample = [load(f) for f in files[:: max(1, len(files) // 12)]]
    keys = RAMP + [CRITICAL, TEXT, TEXT2, MUTED, SURFACE]
    swatch = Image.new("RGB", (W, H))
    bw = W // len(keys)
    for k, c in enumerate(keys):
        swatch.paste(Image.new("RGB", (bw, H), c), (k * bw, 0))
    sample.append(swatch)
    sheet = Image.new("RGB", (W, H * len(sample)))
    for k, im in enumerate(sample):
        sheet.paste(im, (0, k * H))
    pal = sheet.quantize(colors=cfg.gif_colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    frames = [load(f).quantize(palette=pal, dither=Image.Dither.NONE) for f in files]
    step = 1
    while True:
        sel = frames[::step]
        fps = cfg.fps / step
        tmp = path + ".tmp.gif"
        sel[0].save(tmp, save_all=True, append_images=sel[1:], duration=int(round(1000.0 / fps)),
                    loop=0, disposal=1)
        size = os.path.getsize(tmp)
        log(f"[viz.make] GIF {W}x{H}, {cfg.gif_colors} colours, {len(sel)} frames at {fps:g} fps: "
            f"{size / 1e6:.1f} MB")
        if size <= cfg.gif_max_mb * 1e6 or step >= 8:
            os.replace(tmp, path)
            return size, len(sel), fps, step
        step *= 2


def encode_mp4(cfg, path, log=print):
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = path + ".tmp.mp4"
    cmd = [exe, "-y", "-loglevel", "error", "-framerate", str(cfg.fps),
           "-i", os.path.join(frame_dir(cfg), "frame_%05d.png"),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", tmp]
    subprocess.run(cmd, check=True)
    os.replace(tmp, path)
    size = os.path.getsize(path)
    log(f"[viz.make] MP4 {size / 1e6:.2f} MB")
    return size


def make(cfg, out_dir="viz_out", jobs=4, frames_only=False, log=print):
    t0 = time.perf_counter()
    report = {"scene": cfg.to_dict(), "scene_hash": cfg.scene_hash(), "physics_hash": cfg.physics_hash()}
    if cfg.source == "rerun" and not os.path.exists(source_path(cfg)):
        from viz import rerun
        t1 = time.perf_counter()
        rerun.run(cfg)
        report["rerun_wall_s"] = time.perf_counter() - t1
    drawn, reused, wall = render_frames(cfg, jobs=jobs, log=log)
    report.update({"frames_total": cfg.total_frames(), "frames_drawn": drawn, "frames_reused": reused,
                   "frames_wall_s": wall, "frame_dir": os.path.relpath(frame_dir(cfg), REPO)})
    if not frames_only:
        out_dir = out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir)
        os.makedirs(out_dir, exist_ok=True)
        gif = os.path.join(out_dir, cfg.output + ".gif")
        mp4 = os.path.join(out_dir, cfg.output + ".mp4")
        t1 = time.perf_counter()
        size, nf, fps, step = encode_gif(cfg, gif, log=log)
        report.update({"gif": os.path.relpath(gif, REPO), "gif_bytes": size, "gif_frames": nf,
                       "gif_fps": fps, "gif_frame_step": step, "gif_width": cfg.gif_width, "gif_colors": cfg.gif_colors,
                       "gif_wall_s": time.perf_counter() - t1})
        t1 = time.perf_counter()
        report.update({"mp4": os.path.relpath(mp4, REPO), "mp4_bytes": encode_mp4(cfg, mp4, log=log),
                       "mp4_wall_s": time.perf_counter() - t1})
        with open(os.path.join(frame_dir(cfg), cfg.output + "_render.json"), "w") as f:
            json.dump(report | {"total_wall_s": time.perf_counter() - t0}, f, indent=2, sort_keys=True)
    report["total_wall_s"] = time.perf_counter() - t0
    log(f"[viz.make] done in {report['total_wall_s']:.0f} s")
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    if args.print_config:
        print(cfg.to_json(indent=2))
        print(f"scene_hash {cfg.scene_hash()}  physics_hash {cfg.physics_hash()}  frames {cfg.total_frames()}")
        return
    if args.background:
        os.makedirs(frame_dir(cfg), exist_ok=True)
        log_path = os.path.join(frame_dir(cfg), "make.log")
        rest = [a for a in argv if a != "--background"]
        with open(log_path, "a") as logf:
            proc = subprocess.Popen([sys.executable, "-u", "-m", "viz.make"] + rest, cwd=REPO,
                                    stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
        print(f"[viz.make] running in the background, pid {proc.pid}, log {os.path.relpath(log_path, REPO)}")
        return
    make(cfg, out_dir=args.out_dir, jobs=args.jobs, frames_only=args.frames_only,
         log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
