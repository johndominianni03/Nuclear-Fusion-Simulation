"""Array-backed trajectory history storage, split out of main.py (step 14).

True leaf module: imports nothing from this project.

The module name undersells what is in here. _TrackStore is about plotted
trajectories, but _pid_capacity_bound and _require_pid_capacity are about PIDS:
they size pid_row / pid_slot / pid_pool and guard the shared pid counter at the
injection sites, which has nothing to do with trajectories. They live here
because the pid maps exist to serve the track store. Do not confuse
_pid_capacity_bound (singular, here, sizes the pid maps) with
particle_pool._pool_capacity_bounds (plural, there, sizes pool rows) -- the two
have different consumers and were deliberately put in different modules so that
every call site reads which one it means.
"""

import numpy as np
import torch


# Hard ceiling on plotted trajectories: 1000 initial thermals plus the
# tracked_nbis and tracked_alphas caps of 1000 each, all three enforced at the
# injection sites. _TrackStore raises rather than silently overrunning.
_MAX_TRACKED_SLOTS = 3000


def _pid_capacity_bound(cfg, n_init):
    """Upper bound on the largest pid a run can reach, plus headroom.

    Sizing the pid -> row map from the initial particle count is WRONG: pids are
    handed out monotonically and never reused, so injection walks them past
    n_init and a map sized that way overruns partway through a long run (a
    5,200-step disruption run reaches ~n_init + 34,000). The bound below counts
    the injections the loop can actually perform:

      NBI    -- one batch every cfg.inject_every_n_steps steps, and a batch is
                int(rate) plus at most one more from the stochastic remainder,
                with rate <= cfg.NBI_BATCH_SIZE because the exponential taper
                only ever shrinks it.
      alphas -- exactly one every cfg.inject_every_n_steps * 2 steps.

    Step 8 makes this bound load-bearing in a second way: it also sizes the
    _ParticlePool row capacity, because max rows and max pid are the same
    quantity (n_init + everything injection can source, since losses only
    compact downward). The pool RAISES on overflow rather than growing, so a
    wrong bound surfaces immediately instead of being absorbed.
    """
    steps = int(cfg.reactor_num_steps)
    nbi_events = steps // max(1, int(cfg.inject_every_n_steps)) + 1
    alpha_events = steps // max(1, int(cfg.inject_every_n_steps) * 2) + 1
    nbi_max = int(cfg.NBI_BATCH_SIZE) + 1
    return int(n_init + nbi_events * nbi_max + alpha_events) + 1024


def _require_pid_capacity(next_pid, batch, capacity, where):
    """Tripwire for the pid maps, which share the pool's hard ceiling.

    Replaces the pair of _ensure_pid_capacity helpers that used to DOUBLE the
    maps on overflow. Growing was the wrong response: pid_capacity comes from
    _pid_capacity_bound, which counts every particle the injection schedule can
    source, so overflowing it means that bound is wrong -- and quietly enlarging
    the map hides the bug while _ParticlePool.add, sized from the same number,
    would raise a step later anyway. Fail here, naming the site.
    """
    if next_pid + batch > capacity:
        # RuntimeError rather than assert: see _ParticlePool._require. Stripped
        # under `python -O`, this guard would hand an out-of-range index straight
        # to the pid maps.
        raise RuntimeError(
            f"pid capacity exceeded at {where}: next_pid={next_pid} + {batch} > "
            f"capacity={capacity}. _pid_capacity_bound is meant to be a hard "
            f"ceiling on the injection schedule; fix the bound rather than "
            f"growing the map here.")


class _TrackStore:
    """Array-backed storage for the plotted particle trajectories.

    Replaces four pid-keyed dicts (history_tracks / tracked_type /
    tracked_lastpos / tracked_lastvel) and a set (tracked_lost), plus the Python
    loop that appended one (3,) copy per tracked pid per sampling tick -- up to
    3,000 dict lookups and copies every second step, on the order of 10^7 Python
    iterations over a full run.

    SLOTS, not pids, index every array here: a pid gets the next free slot when
    it starts being tracked. Sampled vertices go into ONE flat buffer as
    (slot, xyz) pairs appended in step order, and are regrouped per track once,
    at the end of the run, by a stable argsort on the slot column. Stable is
    load-bearing -- it is the only thing keeping each track's vertices in
    chronological order.

    Two vertices are held apart from that buffer because they are not on the
    sampling cadence:
      * the injection vertex, always a track's first, and
      * the wall-impact vertex, always a track's last -- the particle is dead
        from that step on, so it can never be sampled again. This is what makes
        the crimson wall-strike traces terminate at the wall in
        tokamak_reactor_2d.png / tokamak_reactor_3d.png.
    Both are stored per slot in host arrays, so to_dicts can splice them onto
    the ends without disturbing the sort.

    device=None keeps sampled coordinates in a numpy buffer (CPU loop). Pass a
    torch device and they accumulate in a device tensor instead, so the GPU loop
    never routes a sample through the host: one transfer at end of run replaces
    two per sampling step. The slot column stays on the host either way, because
    slot ids are host-side bookkeeping in both loops -- which is also what lets
    the host advance the write pointer without a device sync.

    host_dtype exists to preserve each loop's existing vertex values exactly.
    The CPU loop reads its injection vertex back out of the float32 pos_np; the
    GPU loop reads it straight from the float64 array
    inject_neutral_beam_cartesian returns, before it is cast down onto the
    device. Those are different values, and both are what their path recorded
    before this rewrite.
    """

    def __init__(self, slot_capacity, device=None, host_dtype=np.float32,
                 vertex_capacity=4096):
        self.device = device
        self.slot_capacity = int(slot_capacity)
        self.n_slots = 0

        self.pids = np.full(self.slot_capacity, -1, dtype=np.int64)
        self.species = np.zeros(self.slot_capacity, dtype=np.int8)
        self.lost = np.zeros(self.slot_capacity, dtype=bool)

        self.init_xyz = np.zeros((self.slot_capacity, 3), dtype=host_dtype)
        self.final_xyz = np.zeros((self.slot_capacity, 3), dtype=host_dtype)
        self.has_final = np.zeros(self.slot_capacity, dtype=bool)

        # last_pos / last_vel have two sources. Injection and wall impact are
        # host-side, sampling is device-side on the GPU path; last_is_host says
        # which one currently holds the live value for a slot.
        self.last_pos_host = np.zeros((self.slot_capacity, 3), dtype=host_dtype)
        self.last_vel_host = np.zeros((self.slot_capacity, 3), dtype=host_dtype)
        self.last_is_host = np.ones(self.slot_capacity, dtype=bool)

        self.v_slot = np.empty(int(vertex_capacity), dtype=np.int32)
        self.n_verts = 0
        if device is None:
            self.v_xyz = np.empty((int(vertex_capacity), 3), dtype=np.float32)
            self.last_pos_dev = None
            self.last_vel_dev = None
        else:
            self.v_xyz = torch.empty((int(vertex_capacity), 3),
                                     dtype=torch.float32, device=device)
            self.last_pos_dev = torch.zeros((self.slot_capacity, 3),
                                            dtype=torch.float32, device=device)
            self.last_vel_dev = torch.zeros((self.slot_capacity, 3),
                                            dtype=torch.float32, device=device)

    # -- slot bookkeeping -------------------------------------------------
    def add_slots(self, pids, species, init_pos, init_vel):
        """Start tracking pids. Returns the slot indices assigned, in order."""
        n = len(pids)
        if n == 0:
            return np.empty(0, dtype=np.int64)
        if self.n_slots + n > self.slot_capacity:
            # RuntimeError rather than assert: see _ParticlePool._require.
            raise RuntimeError(
                f"_TrackStore slot overflow: {self.n_slots} + {n} > "
                f"{self.slot_capacity}; the 1000-each tracked_nbis/tracked_alphas "
                f"caps should have prevented this")
        slots = np.arange(self.n_slots, self.n_slots + n, dtype=np.int64)
        self.pids[slots] = pids
        self.species[slots] = species
        self.init_xyz[slots] = init_pos
        self.last_pos_host[slots] = init_pos
        self.last_vel_host[slots] = init_vel
        self.last_is_host[slots] = True
        self.n_slots += n
        return slots

    # -- vertex accumulation ----------------------------------------------
    def _reserve(self, m):
        need = self.n_verts + m
        if need <= self.v_slot.shape[0]:
            return
        cap = self.v_slot.shape[0]
        while cap < need:
            cap *= 2
        self.v_slot = np.resize(self.v_slot, cap)
        if self.device is None:
            grown = np.empty((cap, 3), dtype=np.float32)
            grown[:self.n_verts] = self.v_xyz[:self.n_verts]
        else:
            grown = torch.empty((cap, 3), dtype=torch.float32, device=self.device)
            grown[:self.n_verts] = self.v_xyz[:self.n_verts]
        self.v_xyz = grown

    def append_samples(self, slots_np, xyz):
        """Append one sampled vertex per slot. xyz is numpy (CPU) or a device
        tensor (GPU) with one row per entry of slots_np, already in slot order."""
        m = len(slots_np)
        if m == 0:
            return
        self._reserve(m)
        w = self.n_verts
        self.v_slot[w:w + m] = slots_np
        self.v_xyz[w:w + m] = xyz
        self.n_verts = w + m

    def set_last_host(self, slots_np, pos, vel):
        self.last_pos_host[slots_np] = pos
        self.last_vel_host[slots_np] = vel
        self.last_is_host[slots_np] = True

    def set_last_device(self, slots_np, slots_t, pos_t, vel_t):
        self.last_pos_dev[slots_t] = pos_t
        self.last_vel_dev[slots_t] = vel_t
        self.last_is_host[slots_np] = False

    def record_impact(self, slots_np, pos, vel):
        """Wall strike: the terminal vertex, off the sampling cadence."""
        if len(slots_np) == 0:
            return
        self.final_xyz[slots_np] = pos
        self.has_final[slots_np] = True
        self.lost[slots_np] = True
        self.set_last_host(slots_np, pos, vel)

    # -- slot selection ----------------------------------------------------
    def slots_due(self, sample_thermal, sample_alpha):
        """Slot indices due to be sampled this step, by species cadence."""
        n = self.n_slots
        if n == 0:
            return np.empty(0, dtype=np.int64)
        if sample_thermal and sample_alpha:
            return np.arange(n, dtype=np.int64)
        is_alpha = self.species[:n] == 2
        return np.nonzero(is_alpha if sample_alpha else ~is_alpha)[0]

    # -- teardown ----------------------------------------------------------
    def to_dicts(self):
        """Rebuild the dict-of-lists payload the return signature still uses.

        Runs once, after the loop. On the GPU path this is the ONLY place the
        sampled vertices and the device-side last_pos/last_vel cross back to the
        host.
        """
        n = self.n_verts
        v_xyz = self.v_xyz[:n]
        last_pos_dev = last_vel_dev = None
        if self.device is not None:
            v_xyz = v_xyz.cpu().numpy()
            last_pos_dev = self.last_pos_dev[:self.n_slots].cpu().numpy()
            last_vel_dev = self.last_vel_dev[:self.n_slots].cpu().numpy()

        slots = self.v_slot[:n]
        order = np.argsort(slots, kind="stable")
        counts = (np.bincount(slots, minlength=self.n_slots) if n
                  else np.zeros(self.n_slots, dtype=np.int64))

        history_tracks, tracked_type = {}, {}
        tracked_lastpos, tracked_lastvel = {}, {}
        tracked_lost = set()

        offset = 0
        for s in range(self.n_slots):
            pid = int(self.pids[s])
            c = int(counts[s])
            verts = [self.init_xyz[s].copy()]
            if c:
                verts.extend(v_xyz[order[offset:offset + c]])
                offset += c
            if self.has_final[s]:
                verts.append(self.final_xyz[s].copy())

            history_tracks[pid] = verts
            tracked_type[pid] = int(self.species[s])
            if self.last_is_host[s] or last_pos_dev is None:
                tracked_lastpos[pid] = self.last_pos_host[s].copy()
                tracked_lastvel[pid] = self.last_vel_host[s].copy()
            else:
                tracked_lastpos[pid] = last_pos_dev[s].copy()
                tracked_lastvel[pid] = last_vel_dev[s].copy()
            if self.lost[s]:
                tracked_lost.add(pid)

        return (history_tracks, tracked_type, tracked_lastpos,
                tracked_lastvel, tracked_lost)
