"""Scene inputs and the stored-data loader for the fusion-cycle animation.

SceneConfig holds every code input of a render: the data source, which species
and how many tracked markers are shown, the clock (time window and time-scale
mapping), the camera, fps, resolution and output name. It is a frozen,
JSON-serialisable dataclass; `scene_hash` keys the frame cache and
`physics_hash` keys the re-simulation cache (only the inputs that change the
simulation).

Sources
  "sample"  replay data/fastion_sample.npz, the production run's stored sample
            (150 + 150 markers, 16 + 16 tracked).
  "rerun"   re-simulate n_alpha / n_beam markers (all of them tracked) with
            fastion.run_alpha_slowing_down / run_beam_slowing_down -- the same
            entry points, the same seeds (20261123 / 20261124, follow seed + 1)
            -- from the current scenario plus `overrides`, written in the same
            format by fastion.save_sample (viz.rerun).

What the loader returns is stored data, reshaped but never altered:
  * track (n, P, 5) = [R, phi, Z, E_J, t_s]: the guiding centre recorded by
    fastion_kernel every 50,000 GC steps (140-250 us apart), downsampled to at
    most 2,000 points; NaN rows pad markers with fewer points.
  * first_orbit (n, F, 4) = [R, phi, Z, v_par]: a separate collisionless 20 us
    GC push from the same birth state (fastion.save_sample). Its time base is
    not stored; it is fixed by save_sample's recording schedule,
        dt_i = (1/200) 2 pi R_axis / v_i,  v_i = sqrt(2 E_birth,i / m),
        n_steps = ceil(20 us / min_i dt_i),  rec_every = n_steps // 400,
        t_k = k rec_every dt_i,
    and the loader checks the predicted row count against the stored one.
  * per-marker events for every marker: birth position (particle, Cartesian),
    birth velocity, E_birth, status, end time, end GC (R, phi, Z), E_end and
    the energy given to electrons / ions (totals, not time-resolved).
    Birth velocities are not in the npz; they are re-derived by calling the
    deterministic birth samplers again with the run's seeds, and the loader
    requires the re-derived positions, weights and energies to equal the
    stored ones exactly (RuntimeError otherwise).
"""
import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO, "viz_cache")
SAMPLE_PATH = os.path.join(REPO, "data", "fastion_sample.npz")

# Seeds of fastion.run_alpha_slowing_down / run_beam_slowing_down (births; the
# Monte Carlo follow uses seed + 1).
SEEDS = {"alpha": 20261123, "beam": 20261124}
SPECIES = ("alpha", "beam")
STATUS = {0: "active", 1: "thermalized", 2: "lost", 3: "timeout"}

# Physics inputs a "rerun" may override: the SimulationConfiguration fields the
# fast-ion chain actually reads (fields_gc.build_reactor_equilibrium,
# background.Background.from_scenario, fastion.FastIonSim.birth_beams).
# R0_major is read too but is excluded: the Grad-Shafranov geometry is fixed
# (axis at R = 1.15 m), so changing it would only rescale F = B0 R0 out of
# step with the equilibrium. P_NBI_MW only sets the beam weights, so it changes
# no trajectory and no drawn point.
ALLOWED_OVERRIDES = {
    "B0": "toroidal field at R0 [T] (fields_gc: F = B0 R0, q-rescale of psi)",
    "Q_SAFETY_TARGET": "q(a) the poloidal field is rescaled to",
    "T_thermal_keV": "on-axis background temperature, e/D/T [keV]",
    "N_AVG_TARGET": "volume-averaged ion density [m^-3] (n0 = 2.03x this ratio on axis)",
    "nbi_energy_keV": "neutral-beam injection energy [keV]",
    "P_NBI_MW": "neutral-beam power [MW] (beam weights only)",
}

# Bump when the drawing code changes, so stale cached frames are not reused.
RENDER_VERSION = 1


@dataclass(frozen=True)
class SceneConfig:
    source: str = "sample"                 # "sample" | "rerun"
    sample_path: str = "data/fastion_sample.npz"
    species: tuple = ("alpha", "beam")     # which species are drawn
    n_show: int = 16                       # tracked markers drawn per species
    n_alpha: int = 16                      # rerun: alpha markers (all tracked)
    n_beam: int = 16                       # rerun: beam markers (all tracked)
    overrides: tuple = ()                  # rerun: ((key, value), ...) from ALLOWED_OVERRIDES
    # clock: birth_frames at t = 0, n_frames from t_min to t_max, hold_frames at t_max
    t_min: float = 1.0e-6                  # s
    t_max: float = 1.2                     # s (the last beam marker ends at 1.13 s)
    time_mapping: str = "log"              # "log" | "linear"
    n_frames: int = 252
    birth_frames: int = 14
    hold_frames: int = 34
    # drawing
    tail_frames: float = 3.0               # stored samples younger than this many frames
    tail_max: int = 48                     # ... capped at this many per marker
    show_untracked: bool = True            # birth / loss events of untracked markers
    show_neutrons: bool = True
    # camera (degrees); azimuth advances by rotate_deg over the whole clip
    elev: float = 24.0
    azim: float = -62.0
    rotate_deg: float = 100.0
    zoom: float = 1.0
    fps: int = 20
    width: int = 960                       # px
    height: int = 720                      # px
    output: str = "fusion_cycle"
    title: Optional[str] = None

    def __post_init__(self):
        sp = self.species
        sp = tuple(s.strip() for s in sp.split(",")) if isinstance(sp, str) else tuple(sp)
        object.__setattr__(self, "species", sp)
        ov = self.overrides.items() if isinstance(self.overrides, dict) else self.overrides
        ov = tuple(sorted((str(k), float(v)) for k, v in ov))
        object.__setattr__(self, "overrides", ov)
        if self.source not in ("sample", "rerun"):
            raise ValueError(f"source must be 'sample' or 'rerun', not {self.source!r}")
        if not sp or any(s not in SPECIES for s in sp):
            raise ValueError(f"species must be a non-empty subset of {SPECIES}, not {sp}")
        bad = [k for k, _ in ov if k not in ALLOWED_OVERRIDES]
        if bad:
            raise ValueError(f"unknown physics override(s) {bad}; allowed: {sorted(ALLOWED_OVERRIDES)}")
        if ov and self.source != "rerun":
            raise ValueError("physics overrides need source='rerun' (the stored sample is fixed)")
        if self.time_mapping not in ("log", "linear"):
            raise ValueError("time_mapping must be 'log' or 'linear'")
        if not (0.0 < self.t_min < self.t_max):
            raise ValueError("need 0 < t_min < t_max")
        if min(self.n_frames, self.fps, self.width, self.height, self.n_show) < 1:
            raise ValueError("n_frames, fps, width, height and n_show must be >= 1")

    # ---- serialisation ----------------------------------------------------
    def to_dict(self):
        d = dataclasses.asdict(self)
        d["species"] = list(self.species)
        d["overrides"] = {k: v for k, v in self.overrides}
        return d

    def to_json(self, **kw):
        return json.dumps(self.to_dict(), sort_keys=True, **kw)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown SceneConfig key(s): {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def from_json(cls, path_or_text):
        text = path_or_text
        if os.path.exists(path_or_text):
            with open(path_or_text) as f:
                text = f.read()
        return cls.from_dict(json.loads(text))

    def replace(self, **kw):
        d = self.to_dict()
        d.update(kw)
        return SceneConfig.from_dict(d)

    # ---- hashes -----------------------------------------------------------
    def physics_key(self):
        """The inputs that determine the simulated data."""
        if self.source == "sample":
            return {"source": "sample", "sample_path": self.sample_path}
        return {"source": "rerun", "n_alpha": self.n_alpha, "n_beam": self.n_beam,
                "overrides": {k: v for k, v in self.overrides}, "seeds": SEEDS}

    def physics_hash(self):
        return _hash(self.physics_key())

    def scene_hash(self):
        """Frame-cache key: every input plus the render-code version."""
        return _hash({"scene": self.to_dict(), "physics": self.physics_key(),
                      "render_version": RENDER_VERSION})

    # ---- clock ------------------------------------------------------------
    def total_frames(self):
        return self.birth_frames + self.n_frames + self.hold_frames

    def frame_times(self):
        """Physical time [s] of every frame: birth_frames at t = 0, then n_frames
        from t_min to t_max (log: t_k = t_min (t_max/t_min)^(k/(n-1)); linear:
        evenly spaced), then hold_frames at t_max."""
        n = self.n_frames
        k = np.arange(n) / max(n - 1, 1)
        if self.time_mapping == "log":
            main = self.t_min * (self.t_max / self.t_min) ** k
        else:
            main = self.t_min + (self.t_max - self.t_min) * k
        return np.concatenate([np.zeros(self.birth_frames), main, np.full(self.hold_frames, self.t_max)])

    def mapping_text(self):
        if self.time_mapping == "log":
            dec = np.log10(self.t_max / self.t_min)
            return (f"clock: log time, {self.n_frames / dec:.0f} frames per decade "
                    f"({self.t_min:.0e} s to {self.t_max:g} s)")
        return f"clock: linear time, {(self.t_max - self.t_min) / max(self.n_frames - 1, 1):.3g} s per frame"

    def make_cfg(self):
        """SimulationConfiguration with the physics overrides applied."""
        import contextlib
        import io
        from config import SimulationConfiguration
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = SimulationConfiguration()
        for k, v in self.overrides:
            setattr(cfg, k, float(v))
        return cfg


def _hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


# =======================================================
# THE SCENE
# =======================================================
@dataclass
class SpeciesData:
    name: str
    mass: float
    charge: float
    track: np.ndarray            # (n_tracked, P, 5) [R, phi, Z, E_J, t_s], NaN padded
    first_orbit: np.ndarray      # (n_tracked, F, 4) [R, phi, Z, v_par]
    first_orbit_t: np.ndarray    # (n_tracked, F) s, reconstructed recording schedule
    birth_pos: np.ndarray        # (N, 3) particle position at birth, Cartesian [m]
    birth_vel: np.ndarray        # (N, 3) re-derived birth velocity [m/s]
    E_birth: np.ndarray          # (N,) J
    status: np.ndarray           # (N,) int8: 0 active 1 thermalized 2 lost 3 timeout
    t_end: np.ndarray            # (N,) s
    end_RphiZ: np.ndarray        # (N, 3) end guiding centre (phi unwrapped)
    E_end: np.ndarray            # (N,) J
    dE_e: np.ndarray             # (N,) J given to electrons over the marker's life
    dE_i: np.ndarray             # (N,) J given to ions
    weight: np.ndarray           # (N,) real particles per second per marker

    @property
    def n_markers(self):
        return int(self.status.size)

    @property
    def n_tracked(self):
        return int(self.track.shape[0])

    def n_valid(self):
        return np.sum(~np.isnan(self.track[:, :, 0]), axis=1)


@dataclass
class Scene:
    config: SceneConfig
    R_axis: float
    Z_axis: float
    wall_R: np.ndarray           # psi_N = 1 contour = loss boundary (no first wall is modelled)
    wall_Z: np.ndarray
    surfaces: list               # [(psi_N, R, Z), ...] interior flux surfaces
    species: dict                # name -> SpeciesData
    source_path: str
    meta: dict = field(default_factory=dict)
    neutron_energy_MeV: float = 0.0

    def frame_times(self):
        return self.config.frame_times()


def _xyz(R, phi, Z):
    return R * np.cos(phi), R * np.sin(phi), Z


def first_orbit_times(sim_field, E_birth, m, n_rec, horizon=2.0e-5, n_pts=400):
    """Recording times of fastion.save_sample's first-orbit push (see module doc)."""
    import gc_push as G
    v = np.sqrt(2.0 * np.asarray(E_birth) / m)
    dt = G.dt_transit_fraction(sim_field, v)
    n_steps = int(np.ceil(horizon / dt.min()))
    rec_every = max(1, n_steps // n_pts)
    predicted = n_steps // rec_every + 1
    if predicted != n_rec:
        raise RuntimeError(f"first-orbit schedule predicts {predicted} rows, stored {n_rec}")
    return np.arange(n_rec)[None, :] * (rec_every * dt)[:, None]


def source_path(cfg: SceneConfig):
    if cfg.source == "sample":
        p = cfg.sample_path
        return p if os.path.isabs(p) else os.path.join(REPO, p)
    return os.path.join(CACHE_DIR, f"rerun_{cfg.physics_hash()}", "sample.npz")


def load_scene(cfg: SceneConfig, surfaces=(0.2, 0.4, 0.6, 0.8), n_theta=241):
    """Scene from the stored arrays of cfg's source (rerun data must exist:
    run viz.rerun first)."""
    import contextlib
    import io
    from constants import M_ALPHA, M_DEUTERON, E_CHARGE
    import fastion

    path = source_path(cfg)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found" + (
            " -- run `venv/bin/python -m viz.rerun --config <scene.json>` first (background, ~10 min)"
            if cfg.source == "rerun" else ""))
    meta = {}
    meta_path = os.path.join(os.path.dirname(path), "meta.json")
    if cfg.source == "rerun" and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    with contextlib.redirect_stdout(io.StringIO()):
        sim = fastion.FastIonSim(cfg.make_cfg())
    with np.load(path, allow_pickle=False) as d:
        raw = {k: d[k] for k in d.files}
    if not np.array_equal(raw["wall_R"], sim.wall_R) or not np.array_equal(raw["wall_Z"], sim.wall_Z):
        raise RuntimeError("stored wall contour differs from the equilibrium's psi_N = 1 contour")

    mass = {"alpha": M_ALPHA, "beam": M_DEUTERON}
    charge = {"alpha": 2.0 * E_CHARGE, "beam": E_CHARGE}
    birth = {"alpha": sim.birth_alphas, "beam": sim.birth_beams}
    species = {}
    for sp in SPECIES:
        if f"{sp}_track" not in raw:
            continue
        N = raw[f"{sp}_status"].size
        pos, vel, w, _ = birth[sp](N, SEEDS[sp])
        E = 0.5 * mass[sp] * np.sum(vel * vel, axis=1)
        if not (np.array_equal(pos, raw[f"{sp}_birth_pos"]) and np.array_equal(w, raw[f"{sp}_weight"])
                and np.array_equal(E, raw[f"{sp}_E_birth"])):
            raise RuntimeError(f"{sp}: re-derived births do not reproduce the stored birth_pos/weight/E_birth")
        fo = raw[f"{sp}_first_orbit"]
        n_tr = raw[f"{sp}_track"].shape[0]
        fo_t = first_orbit_times(sim.field, raw[f"{sp}_E_birth"][:n_tr], mass[sp], fo.shape[1])
        species[sp] = SpeciesData(
            name=sp, mass=mass[sp], charge=charge[sp],
            track=raw[f"{sp}_track"], first_orbit=fo, first_orbit_t=fo_t,
            birth_pos=raw[f"{sp}_birth_pos"], birth_vel=vel, E_birth=raw[f"{sp}_E_birth"],
            status=raw[f"{sp}_status"], t_end=raw[f"{sp}_t"], end_RphiZ=raw[f"{sp}_end_RphiZ"],
            E_end=raw[f"{sp}_E_end"], dE_e=raw[f"{sp}_dE_e"], dE_i=raw[f"{sp}_dE_i"],
            weight=raw[f"{sp}_weight"])

    surf = [(float(x),) + tuple(sim.field.flux_surface_contour(float(x), n_theta=n_theta)) for x in surfaces]
    c = sim.cfg
    return Scene(config=cfg, R_axis=float(raw["R_axis"]), Z_axis=float(raw["Z_axis"]),
                 wall_R=raw["wall_R"], wall_Z=raw["wall_Z"], surfaces=surf, species=species,
                 source_path=path, meta=meta,
                 # D + T -> alpha + n shares the reaction energy; the neutron takes the rest
                 neutron_energy_MeV=float(c.E_FUSION_MEV - c.ALPHA_ENERGY_MEV))
