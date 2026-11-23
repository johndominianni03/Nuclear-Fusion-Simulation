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
