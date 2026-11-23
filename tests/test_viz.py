#!/usr/bin/env python3
"""Tests for the Tier 3.5 fusion-cycle animation (viz/).

    venv/bin/python -m pytest tests/test_viz.py -q
    venv/bin/python tests/test_viz.py          # same tests, no pytest needed

Phase 1: the scene loader returns the stored arrays unchanged, hashes are
stable, and the stored tracks stay inside the loss boundary except at the
recorded loss events. Phase 4: rendering is deterministic, every drawn point
is a stored array element, and the frame cache resumes.
"""
import os
import sys

os.environ.setdefault("FUSION_HEADLESS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import functools                                   # noqa: E402

import numpy as np                                 # noqa: E402


@functools.lru_cache(maxsize=None)
def _scene():
    from viz.scene import SceneConfig, load_scene
    return load_scene(SceneConfig())


@functools.lru_cache(maxsize=None)
def _npz():
    from viz.scene import SAMPLE_PATH
    with np.load(SAMPLE_PATH, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


# =======================================================
# PHASE 1: SCENE DATA
# =======================================================
def test_sample_loader_returns_the_stored_arrays():
    s, d = _scene(), _npz()
    assert np.array_equal(s.wall_R, d["wall_R"]) and np.array_equal(s.wall_Z, d["wall_Z"])
    assert s.R_axis == float(d["R_axis"]) and s.Z_axis == float(d["Z_axis"])
    names = {"track": "track", "first_orbit": "first_orbit", "birth_pos": "birth_pos",
             "E_birth": "E_birth", "status": "status", "t_end": "t", "end_RphiZ": "end_RphiZ",
             "E_end": "E_end", "dE_e": "dE_e", "dE_i": "dE_i", "weight": "weight"}
    for sp in ("alpha", "beam"):
        sd = s.species[sp]
        for attr, key in names.items():
            a, b = getattr(sd, attr), d[f"{sp}_{key}"]
            assert a.shape == b.shape and a.dtype == b.dtype, (sp, attr)
            assert np.array_equal(a, b, equal_nan=True), (sp, attr)


def test_rederived_births_reproduce_the_stored_first_orbits():
    """Births re-derived with the production seeds, reduced to guiding centres
    the way FastIonSim.follow does and pushed with save_sample's schedule,
    reproduce the stored first orbits bit for bit -- which also pins the
    reconstructed first-orbit time base (dt_i, rec_every)."""
    import contextlib
    import io
    import fastion
    import gc_push as G
    s, d = _scene(), _npz()
    with contextlib.redirect_stdout(io.StringIO()):
        sim = fastion.FastIonSim()
    for sp in ("alpha", "beam"):
        sd = s.species[sp]
        n = sd.n_tracked
        pos, vel = sd.birth_pos[:n].astype(np.float64), sd.birth_vel[:n].astype(np.float64)
        state, mu = G.gc_from_particle_invariants(sim.field, pos, vel, sd.charge, sd.mass)
        v = np.sqrt(np.sum(vel ** 2, axis=1))
        dt = G.dt_transit_fraction(sim.field, v)
        n_steps = int(np.ceil(2.0e-5 / dt.min()))
        rec_every = max(1, n_steps // 400)
        fo, _, _ = G.push_record(sim.field, state, mu, sd.charge, sd.mass, dt, n_steps, rec_every)
        assert np.array_equal(fo, d[f"{sp}_first_orbit"]), sp
        t = np.arange(fo.shape[1])[None, :] * (rec_every * dt)[:, None]
        assert np.allclose(t, sd.first_orbit_t, rtol=1e-14, atol=0.0), sp
        assert np.array_equal(sd.track[:, 0, :3], fo[:, 0, :3]), sp       # both start at the birth GC


def test_first_orbit_clock_matches_the_toroidal_advance():
    """Independent of the schedule: int v_par/R dt over the 20 us agrees with the
    stored toroidal advance to 6.6% (alpha) / 1.4% (beam); the rest is the
    drift's share of dphi/dt. A clock off by 2x would miss by 50-100%."""
    s = _scene()
    for sp, tol in (("alpha", 0.10), ("beam", 0.03)):
        sd = s.species[sp]
        fo, t = sd.first_orbit, sd.first_orbit_t
        integ = np.trapezoid(fo[:, :, 3] / fo[:, :, 0], t, axis=1)
        dphi = fo[:, -1, 1] - fo[:, 0, 1]
        assert np.max(np.abs(integ / dphi - 1.0)) < tol, sp
        assert abs(t[:, -1].max() - 2.0e-5) < 1.0e-7


def test_same_config_same_hash():
    from viz.scene import SceneConfig
    a = SceneConfig()
    b = SceneConfig.from_json(a.to_json())
    assert a == b and a.scene_hash() == b.scene_hash() and a.physics_hash() == b.physics_hash()
    c = a.replace(azim=0.0)
    assert c.scene_hash() != a.scene_hash() and c.physics_hash() == a.physics_hash()
    r1 = SceneConfig(source="rerun", overrides={"B0": 8.0})
    r2 = SceneConfig(source="rerun", overrides=(("B0", 8),), azim=10.0)
    assert r1.physics_hash() == r2.physics_hash()
    assert r1.physics_hash() != SceneConfig(source="rerun").physics_hash()
    for bad in ({"R0_major": 2.0}, {"ALPHA_ENERGY_MEV": 3.0}):
        try:
            SceneConfig(source="rerun", overrides=bad)
        except ValueError:
            continue
        raise AssertionError(f"override {bad} accepted")
    try:
        SceneConfig(overrides={"B0": 8.0})          # the sample cannot take physics overrides
    except ValueError:
        pass
    else:
        raise AssertionError("physics override accepted on the stored sample")


def test_tracks_stay_inside_the_wall_except_at_loss_events():
    from matplotlib.path import Path
    s = _scene()
    wall = Path(np.c_[s.wall_R, s.wall_Z])
    import contextlib
    import io
    import fastion
    with contextlib.redirect_stdout(io.StringIO()):
        field = fastion.FastIonSim().field
    for sp, sd in s.species.items():
        ok = ~np.isnan(sd.track[:, :, 0])
        R, Z = sd.track[:, :, 0][ok], sd.track[:, :, 2][ok]
        assert wall.contains_points(np.c_[R, Z]).all(), sp
        assert field.psi_N(R, Z).max() <= 1.0, sp
        fo = sd.first_orbit
        lost_tracked = sd.status[:sd.n_tracked] == 2
        inside = field.psi_N(fo[~lost_tracked, :, 0], fo[~lost_tracked, :, 2])
        assert inside.max() <= 1.0, sp
        # recorded losses: the end GC is just past psi_N = 1, on the wall contour
        lost = sd.status == 2
        if lost.any():
            e = sd.end_RphiZ[lost]
            x = field.psi_N(e[:, 0], e[:, 2])
            assert np.all((x > 1.0) & (x < 1.01)), sp
            dist = np.min(np.hypot(e[:, 0, None] - s.wall_R[None], e[:, 2, None] - s.wall_Z[None]), axis=1)
            assert dist.max() < 0.01, sp
        # confined markers end inside
        conf = sd.status == 1
        assert field.psi_N(sd.end_RphiZ[conf, 0], sd.end_RphiZ[conf, 2]).max() <= 1.0, sp


# =======================================================
# PHASE 4: RENDERING
# =======================================================
SMALL = dict(n_frames=16, birth_frames=1, hold_frames=1, width=320, height=240, n_show=4)


def _prepared(**kw):
    """Prepared scene for a config differing from the default only in drawing
    inputs (the loaded data is shared)."""
    import dataclasses
    from viz.scene import SceneConfig
    from viz.render import Prepared
    return Prepared(dataclasses.replace(_scene(), config=SceneConfig(**kw)))


def _pixel_hash(path):
    import hashlib
    from PIL import Image
    with Image.open(path) as im:
        return hashlib.sha256(im.convert("RGBA").tobytes()).hexdigest()


def test_a_fixed_scene_renders_identical_pixels_twice():
    import tempfile
    from viz.render import save_frame
    prep = _prepared(**SMALL)
    with tempfile.TemporaryDirectory() as tmp:
        for i in (0, 6, 17):
            a, b = os.path.join(tmp, f"a{i}.png"), os.path.join(tmp, f"b{i}.png")
            save_frame(prep, i, a)
            save_frame(prep, i, b)
            assert _pixel_hash(a) == _pixel_hash(b), i


def _stored_xyz(d, name, idx, n_show):
    """Cartesian position of the stored element `idx` of array `name`, computed
    on the same array slice the renderer converts (identical arithmetic)."""
    sp, kind = name.split("_", 1)
    if kind in ("track", "first_orbit"):
        a = d[name][:n_show]
        X = np.stack([a[..., 0] * np.cos(a[..., 1]), a[..., 0] * np.sin(a[..., 1]), a[..., 2]], axis=-1)
        return X[idx[0], idx[1]]
    if kind == "end_RphiZ":
        e = d[name]
        X = np.stack([e[:, 0] * np.cos(e[:, 1]), e[:, 0] * np.sin(e[:, 1]), e[:, 2]], axis=-1)
        return X[idx[0]]
    if kind == "birth_pos":
        return d[name][idx[0]]
    raise KeyError(name)


def test_every_drawn_point_maps_to_a_stored_array_element():
    """On frames from all three acts: every artist in the 3D view is either
    equilibrium geometry or a data mark; each data mark's drawn coordinates
    equal the frame state's, each state point equals the stored array element
    it names, and that element's time is <= the frame's time."""
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Path3DCollection
    from constants import MEV_J
    from viz.render import draw_frame
    from viz.scene import SceneConfig
    d = _npz()
    cfg = SceneConfig()
    prep = _prepared()
    n_show = cfg.n_show
    B = cfg.birth_frames
    t_end = {sp: d[f"{sp}_t"] for sp in ("alpha", "beam")}
    fo_t = {sp: prep.sp[sp]["fo_t"] for sp in ("alpha", "beam")}
    frames = [0, B + 10, B + 40, B + 120, B + 230, cfg.total_frames() - 1]
    # the frame at which the first prompt loss flashes
    frames.append(int(prep.sp["alpha"]["end_frame"][np.argmin(np.where(d["alpha_status"] == 2, d["alpha_t"], 9))]))
    n_checked = 0
    for i in frames:
        fig, st = draw_frame(prep, i)
        t = st["t"]
        ax = fig.viz_ax3d
        heads = {(h["species"], h["marker"]): h for h in st["heads"]}
        tails = {(x["species"], x["marker"]): x for x in st["tails"]}
        fos = {(x["species"], x["marker"]): x for x in st["fo_lines"]}

        def check_src(xyz, src):
            name, idx = src
            sp = name.split("_", 1)[0]
            assert np.array_equal(np.asarray(xyz, dtype=np.float64), _stored_xyz(d, name, idx, n_show)), (i, src)
            if name.endswith("_track"):
                assert d[name][idx[0], idx[1], 4] <= t
            elif name.endswith("_first_orbit"):
                assert fo_t[sp][idx[0], idx[1]] <= t
            elif name.endswith("_end_RphiZ"):
                assert t_end[sp][idx[0]] <= t
            # birth positions are t = 0 samples

        for art in list(ax.collections) + list(ax.lines):
            gid = art.get_gid()
            assert gid, f"frame {i}: unlabelled artist {art}"
            kind, *rest = gid.split(":")
            if kind == "geometry":              # equilibrium surfaces, not particle data
                assert not isinstance(art, Path3DCollection), gid
                continue
            if isinstance(art, Path3DCollection):
                pts = np.column_stack([np.asarray(c, dtype=np.float64) for c in art._offsets3d])
            if kind == "head":
                h = heads[(rest[0], int(rest[1]))]
                assert np.array_equal(pts[0], h["xyz"])
                check_src(h["xyz"], h["src"])
                E_name, E_idx = h["E_src"]
                stored_E = d[E_name][E_idx] / MEV_J
                assert h["E_MeV"] == stored_E
            elif kind == "tail":
                tl = tails[(rest[0], int(rest[1]))]
                assert np.array_equal(pts, tl["xyz"])
                for p, s in zip(tl["xyz"], tl["src"]):
                    check_src(p, s)
            elif kind in ("births", "ends"):
                group = [g for g in st[kind] if g["species"] == rest[0]][0]
                assert np.array_equal(pts, group["xyz"])
                for p, s in zip(group["xyz"], group["src"]):
                    check_src(p, s)
            elif kind in ("lost", "flash"):
                sp, m = rest[0], int(rest[1])
                assert d[f"{sp}_status"][m] == 2
                check_src(pts[0], (f"{sp}_end_RphiZ", (m,)))
            elif kind == "fo":
                f = fos[(rest[0], int(rest[1]))]
                assert isinstance(art, Line3DCollection)
                rows = {tuple(r) for r in f["xyz"]}
                for seg in art._segments3d:
                    for p in np.asarray(seg):
                        assert tuple(p) in rows
                n_fo = f["src"][1][1].stop
                for k in range(n_fo):
                    check_src(f["xyz"][k], (f"{rest[0]}_first_orbit", (int(rest[1]), k)))
            elif kind in ("neutron", "neutron_head"):
                # not simulated: drawn only at t = 0, from the stored alpha birth point
                assert t == 0.0
                j = int(rest[0])
                if kind == "neutron":
                    x, y, z = art.get_data_3d()
                    assert np.array_equal([x[0], y[0], z[0]], d["alpha_birth_pos"][j])
            else:
                raise AssertionError(f"frame {i}: unknown artist kind {gid}")
            n_checked += 1
        # the counters are the stored statuses and end times
        for sp in ("alpha", "beam"):
            done = d[f"{sp}_t"] <= t
            c = st["counts"][sp]
            assert c["thermalized"] == int(np.sum(done & (d[f"{sp}_status"] == 1)))
            assert c["lost"] == int(np.sum(done & (d[f"{sp}_status"] == 2)))
            assert c["fast"] == int(np.sum(~done))
        plt.close(fig)
    assert n_checked > 200


def test_frame_cache_resumes_at_the_missing_frames():
    import tempfile
    import viz.make as M
    from viz.scene import SceneConfig
    cfg = SceneConfig(**SMALL)
    keep = M.CACHE_DIR
    with tempfile.TemporaryDirectory() as tmp:
        M.CACHE_DIR = tmp
        try:
            drawn, reused, _ = M.render_frames(cfg, jobs=1, log=lambda m: None)
            n = cfg.total_frames()
            assert (drawn, reused) == (n, 0)
            before = {i: _pixel_hash(M.frame_path(cfg, i)) for i in range(n)}
            for i in range(n - 10, n):
                os.remove(M.frame_path(cfg, i))
            drawn, reused, _ = M.render_frames(cfg, jobs=1, log=lambda m: None)
            assert (drawn, reused) == (10, n - 10)
            assert all(_pixel_hash(M.frame_path(cfg, i)) == before[i] for i in range(n))
            drawn, reused, _ = M.render_frames(cfg, jobs=1, log=lambda m: None)
            assert (drawn, reused) == (0, n)
        finally:
            M.CACHE_DIR = keep


def test_cli_sets_fields_and_rejects_physics_on_the_sample():
    from viz.make import parse_config
    c = parse_config(["--set", "species=alpha", "--set", "azim=10", "--set", "rotate_deg=0"])
    assert c.species == ("alpha",) and c.azim == 10.0 and c.rotate_deg == 0.0
    r = parse_config(["--set", "source=rerun", "--set", "overrides.B0=8"])
    assert r.overrides == (("B0", 8.0),)
    try:
        parse_config(["--set", "overrides.B0=8"])
    except ValueError:
        pass
    else:
        raise AssertionError("physics override accepted on the stored sample")


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:                  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
