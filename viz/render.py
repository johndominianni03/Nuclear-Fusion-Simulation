"""One frame of the fusion-cycle animation at physical time t.

Drawing rule: at time t every marker is drawn at its most recent stored sample
(time <= t); nothing is interpolated or extrapolated. A marker's stored stream
is its collisionless first orbit (0-20 us, ~50 ns apart), then its track
samples (every 50,000 GC steps, 140-250 us apart -- far longer than a bounce,
so they are drawn as dots, never joined), then its end event (end GC position,
E_end at t_end). Colour is the energy of the latest stored energy sample (the
track's E column, or E_end once the marker has ended).

frame_state() decides what is drawn and records, for every drawn point, the
stored array element it came from; draw_frame() draws exactly that state.
Not simulated, and labelled so on screen: the neutron (an arrow opposite the
alpha's birth velocity, D + T at rest in this model) and the time mapping.
"""
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm     # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection            # noqa: E402
import numpy as np                                                 # noqa: E402

from constants import MEV_J                                        # noqa: E402

# ---- theme (dataviz reference palette, dark surface) ----------------------
SURFACE = "#1a1a19"
TEXT = "#f0efec"
TEXT2 = "#c3c2b7"
MUTED = "#8f8e88"
GRID = "#383835"
CRITICAL = "#d03b3b"          # status: critical = lost, always with the x icon and a label
# Sequential blue ramp 650 -> 100: low energy is dark (recedes into the
# surface), high energy is light. One hue, monotonic lightness.
RAMP = ["#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5",
        "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"]
ENERGY_CMAP = LinearSegmentedColormap.from_list("energy_blue", RAMP)
E_MIN_MEV, E_MAX_MEV = 0.04, 4.0
MARKERS = {"alpha": "o", "beam": "D"}
LABEL = {"alpha": "alpha", "beam": "beam D"}
NEUTRON_ARROW_M = 0.45        # drawn length of the (untracked) neutron arrow
FIRST_ORBIT_FADE_DECADES = 1.0
FLASH_FRAMES = 8


def energy_norm():
    return LogNorm(E_MIN_MEV, E_MAX_MEV, clip=True)


def fmt_time(t):
    if t <= 0.0:
        return "0 s (birth)"
    for scale, unit in ((1.0, "s"), (1e-3, "ms"), (1e-6, "µs"), (1e-9, "ns")):
        if t >= scale:
            v = t / scale
            return f"{v:.3g} {unit}" if v < 100 else f"{v:.0f} {unit}"
    return f"{t:.2e} s"


# =======================================================
# PREPARED SCENE (Cartesian coordinates of the stored samples)
# =======================================================
class Prepared:
    def __init__(self, scene):
        self.scene = scene
        self.cfg = scene.config
        self.times = self.cfg.frame_times()
        self.sp = {}
        for name in self.cfg.species:
            sd = scene.species[name]
            n = min(self.cfg.n_show, sd.n_tracked)
            tr = sd.track[:n]
            fo = sd.first_orbit[:n]
            e = sd.end_RphiZ
            self.sp[name] = {
                "sd": sd, "n": n,
                "tr_xyz": np.stack([tr[:, :, 0] * np.cos(tr[:, :, 1]), tr[:, :, 0] * np.sin(tr[:, :, 1]),
                                    tr[:, :, 2]], axis=-1),
                "tr_t": tr[:, :, 4], "tr_E": tr[:, :, 3],
                "fo_xyz": np.stack([fo[:, :, 0] * np.cos(fo[:, :, 1]), fo[:, :, 0] * np.sin(fo[:, :, 1]),
                                    fo[:, :, 2]], axis=-1),
                "fo_t": sd.first_orbit_t[:n],
                "end_xyz": np.stack([e[:, 0] * np.cos(e[:, 1]), e[:, 0] * np.sin(e[:, 1]), e[:, 2]], axis=-1),
                # frame at which each marker's end event first shows
                "end_frame": np.searchsorted(self.times, sd.t_end, side="left"),
            }
        self.t_fo_end = min(float(p["fo_t"][:, -1].min()) for p in self.sp.values())

    def frame_time(self, i):
        return float(self.times[i])

    def azim(self, i):
        n = max(len(self.times) - 1, 1)
        return self.cfg.azim + self.cfg.rotate_deg * i / n


# =======================================================
# WHAT IS DRAWN (pure, with provenance)
# =======================================================
def _tail_alpha(age_rank, n):
    """1 for the newest of n tail samples, falling to 0.15 for the oldest."""
    if n <= 1:
        return np.ones(n)
    return 0.15 + 0.85 * (1.0 - age_rank / (n - 1))


def frame_state(prep, i):
    """Everything drawn in frame i. Point groups carry `src`: a list of
    (array name, index tuple) giving the stored element of each point."""
    cfg = prep.cfg
    t = prep.frame_time(i)
    tail_lo = prep.frame_time(max(i - int(round(cfg.tail_frames)), 0))
    st = {"t": t, "frame": i, "azim": prep.azim(i), "heads": [], "tails": [], "fo_lines": [],
          "births": [], "ends": [], "lost": [], "neutrons": [], "counts": {}, "budget": {},
          "etrace": []}
    in_birth = i < cfg.birth_frames
    for name, p in prep.sp.items():
        sd = p["sd"]
        # ---- tracked markers -------------------------------------------
        for j in range(p["n"]):
            fo_t = p["fo_t"][j]
            t_fo_end = float(fo_t[-1])
            tr_t = p["tr_t"][j]
            valid = ~np.isnan(tr_t)
            ended = t >= sd.t_end[j]
            n_fo = int(np.searchsorted(fo_t, t, side="right"))           # fo samples with time <= t
            ktr = np.nonzero(valid & (tr_t <= t) & (tr_t > t_fo_end))[0]  # track samples after the first orbit
            kE = np.nonzero(valid & (tr_t <= t))[0]                       # energy samples (incl. k = 0)
            if ended:
                xyz, src = p["end_xyz"][j], (f"{name}_end_RphiZ", (j,))
                E, Esrc = sd.E_end[j], (f"{name}_E_end", (j,))
            elif ktr.size:
                k = int(ktr[-1])
                xyz, src = p["tr_xyz"][j, k], (f"{name}_track", (j, k))
                E, Esrc = p["tr_E"][j, int(kE[-1])], (f"{name}_track", (j, int(kE[-1]), 3))
            else:
                k = max(n_fo - 1, 0)
                xyz, src = p["fo_xyz"][j, k], (f"{name}_first_orbit", (j, k))
                E, Esrc = p["tr_E"][j, int(kE[-1])], (f"{name}_track", (j, int(kE[-1]), 3))
            status = int(sd.status[j]) if ended else 0
            st["heads"].append({"species": name, "marker": j, "xyz": np.array(xyz), "src": src,
                                "E_MeV": float(E) / MEV_J, "E_src": Esrc, "status": status,
                                "lost_flash": ended and status == 2})
            # first orbit as a line (finely sampled): the part already flown,
            # bright inside the tail window, faint before it; fades after 20 us
            if n_fo >= 2:
                if t <= t_fo_end:
                    ghost = 0.30
                else:
                    ghost = 0.30 * max(0.0, 1.0 - math.log10(t / t_fo_end) / FIRST_ORBIT_FADE_DECADES)
                k_tail = int(np.searchsorted(fo_t, tail_lo, side="left"))
                st["fo_lines"].append({"species": name, "marker": j, "xyz": p["fo_xyz"][j, :n_fo],
                                       "src": (f"{name}_first_orbit", (j, slice(0, n_fo))),
                                       "ghost_alpha": ghost, "tail_from": min(k_tail, n_fo - 1),
                                       "E_MeV": float(p["tr_E"][j, 0]) / MEV_J})
            # track tail: stored samples inside the tail window, newest tail_max
            kt = ktr[tr_t[ktr] > tail_lo][-cfg.tail_max:] if ktr.size else ktr
            if kt.size:
                age = np.arange(kt.size)[::-1]
                st["tails"].append({"species": name, "marker": j, "xyz": p["tr_xyz"][j, kt],
                                    "E_MeV": p["tr_E"][j, kt] / MEV_J, "alpha": _tail_alpha(age, kt.size),
                                    "src": [(f"{name}_track", (j, int(k))) for k in kt]})
            # energy trace (inset): stored E samples up to t
            if kE.size:
                st["etrace"].append({"species": name, "marker": j, "t": tr_t[kE], "E_MeV": p["tr_E"][j, kE] / MEV_J,
                                     "end": (float(sd.t_end[j]), float(sd.E_end[j]) / MEV_J) if ended else None})
        # ---- every marker: births, ends, losses --------------------------
        N = sd.n_markers
        if True:
            if t <= prep.t_fo_end:
                a = 0.9 if in_birth else max(0.15, 0.9 - 0.75 * math.log10(max(t, cfg.t_min) / cfg.t_min)
                                             / math.log10(prep.t_fo_end / cfg.t_min))
                idx = np.arange(N) if cfg.show_untracked else np.arange(p["n"])
                st["births"].append({"species": name, "xyz": sd.birth_pos[idx], "alpha": a,
                                     "E_MeV": sd.E_birth[idx] / MEV_J,
                                     "src": [(f"{name}_birth_pos", (int(m),)) for m in idx]})
        done = sd.t_end <= t
        therm = np.nonzero(done & (sd.status == 1))[0]
        lost = np.nonzero(done & (sd.status == 2))[0]
        if cfg.show_untracked:
            ut = therm[therm >= p["n"]]
            if ut.size:
                st["ends"].append({"species": name, "xyz": p["end_xyz"][ut], "E_MeV": sd.E_end[ut] / MEV_J,
                                   "src": [(f"{name}_end_RphiZ", (int(m),)) for m in ut]})
        show_lost = lost if cfg.show_untracked else lost[lost < p["n"]]
        for m in show_lost:
            age = i - int(p["end_frame"][m])
            st["lost"].append({"species": name, "marker": int(m), "xyz": p["end_xyz"][m],
                               "src": (f"{name}_end_RphiZ", (int(m),)), "flash_age": age,
                               "t_end": float(sd.t_end[m])})
        st["counts"][name] = {"N": N, "fast": int(N - done.sum()), "thermalized": int(therm.size),
                              "lost": int(lost.size), "timeout": int(np.sum(done & (sd.status == 3)))}
        fin = np.nonzero(done)[0]
        Eb = float(sd.E_birth[fin].sum())
        st["budget"][name] = {"n": int(fin.size),
                              "e": float(sd.dE_e[fin].sum()) / Eb if Eb > 0 else float("nan"),
                              "i": float(sd.dE_i[fin].sum()) / Eb if Eb > 0 else float("nan")}
        # ---- neutrons (not simulated): only at t = 0 ----------------------
        if name == "alpha" and cfg.show_neutrons and in_birth:
            for j in range(p["n"]):
                v = sd.birth_vel[j]
                d = -v / np.linalg.norm(v)            # p_n = -p_alpha (reactants at rest)
                st["neutrons"].append({"marker": j, "from": sd.birth_pos[j].astype(np.float64),
                                       "dir": d, "src": ("alpha_birth_pos", (j,))})
    st["act"] = act_of(prep, i)
    return st


def act_of(prep, i):
    cfg = prep.cfg
    t = prep.frame_time(i)
    if i < cfg.birth_frames or t <= prep.t_fo_end:
        return 1
    if i >= cfg.birth_frames + cfg.n_frames:
        return 3
    return 2


ACT_TITLES = {1: "Act 1 · birth and first orbits",
              2: "Act 2 · slowing down on the background plasma",
              3: "Act 3 · final state"}


# =======================================================
# DRAWING
# =======================================================
def _torus(ax, scene, cut=(0.0, 1.5 * np.pi)):
    """Translucent psi_N = 1 surface over the toroidal range `cut`, nested flux
    surfaces in the two cut planes, and toroidal rings of the boundary."""
    ph = np.linspace(cut[0], cut[1], 64)
    R, Z = scene.wall_R[::6], scene.wall_Z[::6]
    X = R[:, None] * np.cos(ph)[None, :]
    Y = R[:, None] * np.sin(ph)[None, :]
    ZZ = np.repeat(Z[:, None], ph.size, axis=1)
    ax.plot_surface(X, Y, ZZ, color=TEXT2, alpha=0.045, linewidth=0, antialiased=False, shade=False,
                    gid="geometry:lcfs_surface")
    for c in cut:
        ax.plot(scene.wall_R * np.cos(c), scene.wall_R * np.sin(c), scene.wall_Z,
                color=TEXT2, lw=1.0, alpha=0.75, gid="geometry:lcfs_section")
        for x, Rs, Zs in scene.surfaces:
            ax.plot(Rs * np.cos(c), Rs * np.sin(c), Zs, color=MUTED, lw=0.6, alpha=0.55,
                    gid="geometry:flux_surface_section")
    ring = np.linspace(cut[0], cut[1], 160)
    for k in np.linspace(0, scene.wall_R.size - 1, 9)[:-1].astype(int):
        ax.plot(scene.wall_R[k] * np.cos(ring), scene.wall_R[k] * np.sin(ring), np.full(ring.size, scene.wall_Z[k]),
                color=TEXT2, lw=0.5, alpha=0.22, gid="geometry:lcfs_ring")
    ax.plot(scene.R_axis * np.cos(ring), scene.R_axis * np.sin(ring), np.full(ring.size, scene.Z_axis),
            color=MUTED, lw=0.6, alpha=0.4, ls=(0, (2, 3)), gid="geometry:magnetic_axis")


def _colors(E_MeV, alpha):
    c = ENERGY_CMAP(energy_norm()(np.atleast_1d(E_MeV)))
    c[:, 3] = alpha
    return c


def draw_frame(prep, i, state=None):
    """Figure for frame i (caller saves and closes it)."""
    cfg, scene = prep.cfg, prep.scene
    st = frame_state(prep, i) if state is None else state
    W, H, dpi = cfg.width, cfg.height, 100
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor=SURFACE)
    fig.viz_ax3d = None
    s = min(W / 960.0, H / 720.0)                 # font / marker scale
    fs = lambda x: x * s                          # noqa: E731

    # ---- 3D view ------------------------------------------------------------
    ax = fig.add_axes([-0.05, 0.2, 0.76, 0.68], projection="3d", facecolor=SURFACE)
    ax.set_axis_off()
    fig.viz_ax3d = ax
    lim = 1.48 / cfg.zoom
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-0.5 * lim, 0.5 * lim)
    ax.set_box_aspect((1.0, 1.0, 0.5), zoom=1.38)
    ax.view_init(elev=cfg.elev, azim=st["azim"])
    _torus(ax, scene)

    for b in st["births"]:
        ax.scatter(*b["xyz"].T, s=fs(5), c=_colors(b["E_MeV"], b["alpha"] * 0.8), marker=MARKERS[b["species"]],
                   linewidths=0, depthshade=False, gid=f"births:{b['species']}")
    for e in st["ends"]:
        ax.scatter(*e["xyz"].T, s=fs(5), c=_colors(e["E_MeV"], 0.45), marker=MARKERS[e["species"]],
                   linewidths=0, depthshade=False, gid=f"ends:{e['species']}")
    for f in st["fo_lines"]:
        xyz = f["xyz"]
        seg = np.stack([xyz[:-1], xyz[1:]], axis=1)
        a = np.full(len(seg), f["ghost_alpha"])
        k0 = f["tail_from"]
        if st["t"] <= prep.t_fo_end and k0 < len(seg):
            n_t = len(seg) - k0
            a[k0:] = 0.25 + 0.75 * np.linspace(0.0, 1.0, n_t)
        keep = a > 0.01
        if keep.any():
            col = _colors(np.full(keep.sum(), f["E_MeV"]), a[keep])
            ax.add_collection3d(Line3DCollection(seg[keep], colors=col, linewidths=fs(1.1),
                                                 gid=f"fo:{f['species']}:{f['marker']}"))
    for tl in st["tails"]:
        ax.scatter(*tl["xyz"].T, s=fs(7), c=_colors(tl["E_MeV"], tl["alpha"]), marker=MARKERS[tl["species"]],
                   linewidths=0, depthshade=False, gid=f"tail:{tl['species']}:{tl['marker']}")
    for h in st["heads"]:
        x, y, z = h["xyz"]
        if h["status"] == 1:                       # thermalized: dims into the plasma
            ax.scatter([x], [y], [z], s=fs(14), c=_colors(h["E_MeV"], 0.55), marker=MARKERS[h["species"]],
                       linewidths=0, depthshade=False, gid=f"head:{h['species']}:{h['marker']}")
        elif h["status"] == 0:
            ax.scatter([x], [y], [z], s=fs(46), c=_colors(h["E_MeV"], 1.0), marker=MARKERS[h["species"]],
                       edgecolors=SURFACE, linewidths=fs(0.8), depthshade=False,
                       gid=f"head:{h['species']}:{h['marker']}")
    for l in st["lost"]:                           # noqa: E741
        x, y, z = l["xyz"]
        age = l["flash_age"]
        if 0 <= age < FLASH_FRAMES:
            k = 1.0 - age / FLASH_FRAMES
            ax.scatter([x], [y], [z], s=fs(60 + 900 * k), c=[CRITICAL], alpha=0.25 + 0.35 * k, marker="o",
                       linewidths=0, depthshade=False, gid=f"flash:{l['species']}:{l['marker']}")
        ax.scatter([x], [y], [z], s=fs(70), c=[CRITICAL], marker="X", edgecolors=SURFACE, linewidths=fs(0.6),
                   depthshade=False, gid=f"lost:{l['species']}:{l['marker']}")
    for n in st["neutrons"]:
        a, d = n["from"], n["dir"]
        b = a + NEUTRON_ARROW_M * d
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=TEXT, lw=fs(1.0), alpha=0.8, ls=(0, (3, 2)),
                gid=f"neutron:{n['marker']}")
        ax.scatter([b[0]], [b[1]], [b[2]], s=fs(16), c=[TEXT], marker=">", alpha=0.8, linewidths=0,
                   depthshade=False, gid=f"neutron_head:{n['marker']}")

    # ---- header -------------------------------------------------------------
    title = cfg.title or default_title(scene)
    fig.text(0.03, 0.962, title, color=TEXT, fontsize=fs(15), weight="bold", va="center")
    fig.text(0.03, 0.925, subtitle(scene), color=TEXT2, fontsize=fs(10), va="center")
    fig.text(0.03, 0.885, ACT_TITLES[st["act"]], color=TEXT, fontsize=fs(11.5), va="center")

    # ---- readout (right column) ---------------------------------------------
    x0 = 0.685
    fig.text(x0, 0.86, "physical time", color=MUTED, fontsize=fs(9))
    fig.text(x0, 0.82, f"t = {fmt_time(st['t'])}", color=TEXT, fontsize=fs(21), weight="bold", va="center")
    fig.text(x0, 0.782, clock_text(cfg), color=MUTED, fontsize=fs(8.3))
    y = 0.725
    cx = (x0 + 0.165, x0 + 0.235, x0 + 0.282)
    Ns = sorted({st["counts"][n]["N"] for n in cfg.species})
    fig.text(x0, y, "markers", color=MUTED, fontsize=fs(9))
    fig.text(cx[0], y, "fast", color=MUTED, fontsize=fs(9), ha="right")
    fig.text(cx[1], y, "thermal.", color=MUTED, fontsize=fs(9), ha="right")
    fig.text(cx[2], y, "lost", color=MUTED, fontsize=fs(9), ha="right")
    for name in cfg.species:
        y -= 0.04
        c = st["counts"][name]
        sym = "●" if name == "alpha" else "◆"
        fig.text(x0, y, f"{sym} {LABEL[name]}" + ("" if len(Ns) == 1 else f" (of {c['N']})"), color=TEXT,
                 fontsize=fs(10.5))
        fig.text(cx[0], y, f"{c['fast']}", color=TEXT, fontsize=fs(10.5), ha="right")
        fig.text(cx[1], y, f"{c['thermalized']}", color=TEXT, fontsize=fs(10.5), ha="right")
        fig.text(cx[2], y, f"{c['lost']}", color=CRITICAL if c["lost"] else TEXT, fontsize=fs(10.5),
                 ha="right")
    y -= 0.055
    fig.text(x0, y, "birth energy of ended markers given to", color=MUTED, fontsize=fs(9))
    y -= 0.03
    fig.text(cx[1], y, "electrons", color=MUTED, fontsize=fs(9), ha="right")
    fig.text(cx[2], y, "ions", color=MUTED, fontsize=fs(9), ha="right")
    for name in cfg.species:
        y -= 0.034
        b = st["budget"][name]
        fig.text(x0, y, f"{LABEL[name]}  ({b['n']} ended)", color=TEXT2, fontsize=fs(9.5))
        fig.text(cx[1], y, _pct(b["e"]) if b["n"] else "–", color=TEXT, fontsize=fs(9.5), ha="right")
        fig.text(cx[2], y, _pct(b["i"]) if b["n"] else "–", color=TEXT, fontsize=fs(9.5), ha="right")
    _energy_inset(fig, prep, st, [x0 + 0.035, 0.245, 0.245, 0.19], fs)

    # ---- legend + colour bar --------------------------------------------------
    cax = fig.add_axes([0.05, 0.198, 0.29, 0.016])
    sm = plt.cm.ScalarMappable(norm=energy_norm(), cmap=ENERGY_CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.5])
    cb.set_ticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "3.5"])
    cb.outline.set_visible(False)
    cax.tick_params(colors=TEXT2, labelsize=fs(8.5), length=2)
    cax.set_title("marker energy (MeV, log scale)", color=TEXT2, fontsize=fs(9), loc="left", pad=4)
    lx, ly = 0.385, 0.236
    items = [("●", TEXT, "alpha, 3.5 MeV at birth"), ("◆", TEXT, "beam D, 1 MeV at birth"),
             ("✖", CRITICAL, "lost: guiding centre crossed ψ_N = 1")]
    if cfg.show_neutrons:
        items.append(("⇢", TEXT, f"neutron, {scene.neutron_energy_MeV:.1f} MeV (not tracked, t = 0 only)"))
    for k, (sym, col, txt) in enumerate(items):
        yy = ly - 0.024 * k
        fig.text(lx, yy, sym, color=col, fontsize=fs(10), va="center")
        fig.text(lx + 0.022, yy, txt, color=TEXT2, fontsize=fs(8.8), va="center")

    _timeline(fig, prep, st, [0.115, 0.083, 0.845, 0.045], fs)
    fig.text(0.03, 0.012, footer_text(scene), color=MUTED, fontsize=fs(7.8), linespacing=1.5)
    return fig, st


def _pct(x):
    v = 100.0 * x
    return "0%" if abs(v) < 0.5 else f"{v:.0f}%"


def clock_text(cfg):
    if cfg.time_mapping == "log":
        per = cfg.n_frames / math.log10(cfg.t_max / cfg.t_min)
        return f"log clock: {per:.0f} frames per decade, {fmt_time(cfg.t_min)} → {fmt_time(cfg.t_max)}"
    return f"linear clock: {fmt_time((cfg.t_max - cfg.t_min) / max(cfg.n_frames - 1, 1))} per frame"


def subtitle(scene):
    c = scene.config
    if c.source == "sample":
        n = {k: v.n_markers for k, v in scene.species.items()}
        return (f"Stored production sample: {' + '.join(str(n[s]) for s in c.species)} markers, "
                f"{c.n_show} trajectories per species drawn")
    ov = ", ".join(f"{k} = {v:g}" for k, v in c.overrides) or "baseline scenario inputs"
    return f"Re-simulated ({ov}) · illustrative sample: {c.n_alpha} + {c.n_beam} markers"


def default_title(scene):
    names = {"alpha": "fusion alphas", "beam": "beam ions"}
    who = " and ".join(names[s] for s in scene.config.species)
    return f"{who[0].upper() + who[1:]}: birth to thermalization"


def footer_text(scene):
    return ("Simulated: guiding-centre orbits + Monte Carlo Fokker-Planck slowing-down on a fixed Maxwellian "
            "e/D/T background (fastion.py).\nEach marker is drawn at its latest stored sample (first orbit "
            "~50 ns apart, then ~0.2 ms apart); nothing is interpolated. Neutrons are not simulated.")


def _energy_inset(fig, prep, st, rect, fs):
    ax = fig.add_axes(rect, facecolor=SURFACE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    cfg = prep.cfg
    ax.set_xlim(cfg.t_min, cfg.t_max)
    ax.set_ylim(E_MIN_MEV, E_MAX_MEV)
    style = {"alpha": dict(color=TEXT2, lw=fs(0.8), ls="-"), "beam": dict(color=MUTED, lw=fs(0.8), ls=(0, (3, 1.5)))}
    for tr in st["etrace"]:
        t, E = tr["t"], tr["E_MeV"]
        t = np.maximum(t, cfg.t_min)              # the t = 0 sample sits at the left edge
        if tr["end"] is not None:
            t = np.append(t, tr["end"][0])
            E = np.append(E, tr["end"][1])
        else:
            t = np.append(t, max(st["t"], cfg.t_min))   # hold the latest stored value up to now
            E = np.append(E, E[-1])
        ax.plot(t, E, drawstyle="steps-post", alpha=0.85, **style[tr["species"]])
    if st["t"] > 0:
        ax.axvline(max(st["t"], cfg.t_min), color=TEXT, lw=fs(0.8), alpha=0.6)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=fs(7.5), which="both", length=2)
    ax.set_xticks([1e-6, 1e-4, 1e-2, 1.0])
    ax.set_xticklabels(["1 µs", "100 µs", "10 ms", "1 s"])
    ax.set_yticks([0.05, 0.1, 0.3, 1.0, 3.5])
    ax.set_yticklabels(["0.05", "0.1", "0.3", "1", "3.5"])
    ax.minorticks_off()
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_title("tracked markers: stored energy (MeV)", color=TEXT2, fontsize=fs(8.5), loc="left", pad=3)
    ax.text(1.5e-6, 2.3, "alpha", color=TEXT2, fontsize=fs(7.5))
    ax.text(1.5e-6, 0.62, "beam D", color=MUTED, fontsize=fs(7.5))


def _timeline(fig, prep, st, rect, fs):
    cfg = prep.cfg
    ax = fig.add_axes(rect, facecolor=SURFACE)
    log = cfg.time_mapping == "log"
    if log:
        ax.set_xscale("log")
    ax.set_xlim(cfg.t_min, cfg.t_max)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=fs(8), which="both", length=2)
    if log:
        ticks = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["1 µs", "10 µs", "100 µs", "1 ms", "10 ms", "100 ms", "1 s"])
        ax.minorticks_off()
    ax.axvspan(cfg.t_min, prep.t_fo_end, color=GRID, alpha=0.6, lw=0)
    ax.text(math.sqrt(cfg.t_min * prep.t_fo_end) if log else prep.t_fo_end / 2, 0.86, "first orbits",
            color=MUTED, fontsize=fs(7.5), ha="center", va="center")
    # rug: stored end times of every marker of the drawn species
    for name, p in prep.sp.items():
        sd = p["sd"]
        y0, y1 = (0.04, 0.36) if name == "alpha" else (0.36, 0.66)
        for m in range(sd.n_markers):
            te = float(sd.t_end[m])
            if te <= 0 or te > cfg.t_max or te < cfg.t_min:
                continue
            lost = sd.status[m] == 2
            on = te <= st["t"]
            ax.plot([te, te], [y0, y1], color=CRITICAL if lost else TEXT2, lw=fs(0.6 if not lost else 1.2),
                    alpha=(0.9 if on else 0.25))
    ax.text(-0.008, 0.2, "alpha ends", color=TEXT2, fontsize=fs(7.8), va="center", ha="right", transform=ax.transAxes)
    if "beam" in prep.sp:
        ax.text(-0.008, 0.5, "beam ends", color=TEXT2, fontsize=fs(7.8), va="center", ha="right",
                transform=ax.transAxes)
    ax.text(-0.008, 0.86, "clock", color=MUTED, fontsize=fs(7.8), va="center", ha="right", transform=ax.transAxes)
    if st["t"] > 0:
        ax.axvline(max(st["t"], cfg.t_min), color=TEXT, lw=fs(1.4))


def save_frame(prep, i, path):
    fig, st = draw_frame(prep, i)
    tmp = path + ".tmp.png"
    fig.savefig(tmp, dpi=100, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)
    os.replace(tmp, path)
    return st
