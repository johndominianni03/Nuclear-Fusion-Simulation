"""Capacity-backed particle pools, split out of main.py (step 14).

True leaf module: imports nothing from this project.
"""

import numpy as np
import torch
from numba import njit


@njit(cache=True)
def _compact_pool_numba(pos, vel, typ, pid, idx, m):
    """In-place stream compaction of the four particle arrays.

    idx holds the surviving row indices in ASCENDING order, so idx[k] >= k for
    every k: the destination has always already been read by the time it is
    written. That is what makes this safe to do in place with no temporary,
    where `pool.pos[:m] = pool.pos[idx]` would materialise an (m, 3) copy first.

    Sequential on purpose -- do NOT prange this. The read/write overlap that
    makes the forward pass safe depends on the strict k ordering, and under
    parallel execution one thread can write row k_A while another still needs it
    as the source idx[k_B] = k_A for some k_B < k_A.
    """
    for k in range(m):
        src = idx[k]
        pos[k, 0] = pos[src, 0]
        pos[k, 1] = pos[src, 1]
        pos[k, 2] = pos[src, 2]
        vel[k, 0] = vel[src, 0]
        vel[k, 1] = vel[src, 1]
        vel[k, 2] = vel[src, 2]
        typ[k] = typ[src]
        pid[k] = pid[src]


@njit(cache=True)
def _compact_1d_numba(arr, idx, m):
    """In-place forward compaction of one per-particle column (see
    _compact_pool_numba for why ascending idx makes this safe; sequential)."""
    for k in range(m):
        arr[k] = arr[idx[k]]


class _ParticlePool:
    """Capacity-backed particle storage: pos/vel/type/pid allocated once, with an
    n_live write pointer.

    Both loops used to grow by full reallocation -- np.vstack/np.append on the
    CPU path, torch.cat on the GPU path -- at every injection event.
    At 1,000,000 particles that copied ~36 MB per event (12 MB pos + 12 MB vel +
    4 MB type + 8 MB pid), about 5,000 times over a production run, and churned
    the MPS allocator badly on the GPU side.

    The capacity is a HARD CEILING, not a guess: _pid_capacity_bound counts every
    particle the injection schedule can source, and losses only ever compact
    downward, so n_live can never legitimately exceed it. add() therefore RAISES
    on overflow instead of reallocating. A silent grow would paper over a broken
    bound, and the bound is also what sizes pid_row/pid_slot/pid_pool -- so if it
    were wrong, the quiet failure being hidden is an out-of-range write in a numba
    kernel that does not bounds-check.

    device=None keeps the arrays in numpy (CPU loop); pass a torch device and the
    same structure holds device tensors instead.

    with_weight=True adds a per-particle weight column (real particles per
    macroparticle, float64 on the host, float32 on a device) for a species whose
    macroparticles do not share one weight -- the alphas, each of which keeps the
    weight it was born with. add() then needs the weights, and compact() carries
    them with the other columns.
    """

    def __init__(self, capacity, device=None, with_weight=False):
        self.capacity = int(capacity)
        self.device = device
        self.n_live = 0
        self.weight = None
        if with_weight:
            if device is None:
                self.weight = np.empty(self.capacity, dtype=np.float64)
            else:
                self.weight = torch.empty(self.capacity, dtype=torch.float32, device=device)
        if device is None:
            self.pos = np.empty((self.capacity, 3), dtype=np.float32)
            self.vel = np.empty((self.capacity, 3), dtype=np.float32)
            self.type = np.empty(self.capacity, dtype=np.int32)
            self.pid = np.empty(self.capacity, dtype=np.int64)
        else:
            self.pos = torch.empty((self.capacity, 3), dtype=torch.float32, device=device)
            self.vel = torch.empty((self.capacity, 3), dtype=torch.float32, device=device)
            self.type = torch.empty(self.capacity, dtype=torch.int32, device=device)
            self.pid = torch.empty(self.capacity, dtype=torch.int64, device=device)

    def views(self):
        """The live prefix of each array.

        A leading-axis slice of a C-contiguous array is itself C-contiguous, so
        every numba signature and every torch kernel sees exactly what it saw
        before the pools existed -- no recompilation, no hidden copy. Kernels
        that mutate in place (the Boris push, check_confinement_flux,
        apply_vectorized_collisions) write straight through into the pool.
        """
        n = self.n_live
        return self.pos[:n], self.vel[:n], self.type[:n], self.pid[:n]

    def weight_view(self):
        """The live prefix of the per-particle weight column (with_weight pools)."""
        return self.weight[:self.n_live]

    def _require(self, extra):
        # RuntimeError, not assert: `python -O` strips assert statements, and
        # AssertionError is what an assert raises, so either form would make this
        # guard optional at runtime. It must not be. The thing it stands in front
        # of is an out-of-range write in a numba kernel that does not bounds-check
        # -- silent memory corruption, not an IndexError.
        if self.n_live + extra > self.capacity:
            raise RuntimeError(
                f"_ParticlePool overflow: n_live={self.n_live} + {extra} exceeds "
                f"capacity={self.capacity}. The capacity comes from "
                f"particle_pool._pool_capacity_bounds, which is meant to be a hard "
                f"ceiling on the injection schedule -- reaching it means that bound "
                f"is wrong. Fix the bound; do not grow the pool here. (Not "
                f"track_store._pid_capacity_bound: that sizes the pid maps, and has "
                f"been a separate bound since step 9 split the pools.)")

    def add(self, pos_block, vel_block, type_value, pid_block, weight_block=None):
        """Append a batch. Returns the first row written."""
        batch = len(pid_block)
        if batch == 0:
            return self.n_live
        if (self.weight is None) != (weight_block is None):
            raise RuntimeError("_ParticlePool.add: weights must be given exactly when the "
                               "pool has a weight column")
        self._require(batch)
        start, end = self.n_live, self.n_live + batch
        self.pos[start:end] = pos_block
        self.vel[start:end] = vel_block
        self.type[start:end] = type_value
        self.pid[start:end] = pid_block
        if self.weight is not None:
            self.weight[start:end] = weight_block
        self.n_live = end
        return start

    def compact(self, alive_idx):
        """Keep only alive_idx (ascending), in place. Returns the new n_live."""
        m = len(alive_idx)
        if self.device is None:
            _compact_pool_numba(self.pos, self.vel, self.type, self.pid,
                                alive_idx, m)
            if self.weight is not None:
                _compact_1d_numba(self.weight, alive_idx, m)
        else:
            # index_select allocates one temporary per array. Left as is on
            # purpose: wall-loss compaction measures 0.0% of loop wall, so the
            # contortion to avoid it would buy nothing.
            self.pos[:m] = self.pos.index_select(0, alive_idx)
            self.vel[:m] = self.vel.index_select(0, alive_idx)
            self.type[:m] = self.type.index_select(0, alive_idx)
            self.pid[:m] = self.pid.index_select(0, alive_idx)
            if self.weight is not None:
                self.weight[:m] = self.weight.index_select(0, alive_idx)
        self.n_live = m
        return m


def _pool_capacity_bounds(cfg, n_bulk_init, n_trit_init):
    """Per-pool row ceilings for the split bulk-D / alpha / tritium pools.

    Step 8 sized ONE pool from _pid_capacity_bound, so the alpha rows and the
    bulk rows shared a single allocation and either could borrow the other's
    headroom. Split pools cannot: each ceiling has to stand on its own, and the
    alpha one is the tighter case because it is no longer padded by the bulk
    allocation's slack.

    Both are derived from the same injection schedule _pid_capacity_bound counts,
    just attributed to the pool that actually receives the particles:

      bulk  -- n_bulk_init thermal deuterons, plus the NBI source: one batch per
               step of int(rate) plus at most one more from the stochastic
               remainder, rate = cfg.NBI_MACRO_PER_STEP (constant).
      alpha -- the fusion-driven alpha source: one batch per step of int(rate)
               plus at most one more from the stochastic remainder,
               rate = cfg.ALPHA_MACRO_PER_STEP (constant; the weight carries
               the fusion rate).
      trit  -- n_trit_init thermal tritons; nothing is ever injected into it.

    Neither can be exceeded by a run the single-pool version survived: species
    never migrates, so a particle is counted against exactly one pool for its
    whole life, and the two bounds sum to the single-pool bound plus an extra
    margin. Losses only compact downward, so n_live never exceeds the number
    injected.
    """
    steps = int(cfg.reactor_num_steps)
    nbi_max = int(cfg.NBI_MACRO_PER_STEP) + 1
    alpha_max = int(cfg.ALPHA_MACRO_PER_STEP) + 1
    bulk_cap = int(n_bulk_init + steps * nbi_max) + 1024
    alpha_cap = int(steps * alpha_max) + 1024
    trit_cap = int(n_trit_init) + 1024
    return bulk_cap, alpha_cap, trit_cap


def _run_log(nbi_per_step, nbi_energy_J, alpha_per_step, alpha_rate_per_step,
             alpha_birth_weight, alpha_birth_shells, alpha_birth_update,
             alpha_rate_updates, weight_np):
    """The loops' last return value: source bookkeeping and per-row weights.

      nbi_per_step        beam macroparticles injected each step
      nbi_energy_J        weighted kinetic energy those carried in [J]
      alpha_per_step      macro-alphas born each step
      alpha_rate_per_step total thermal D-T rate in force at each step [1/s]
      alpha_birth_weight  sum of the birth weights of every macro-alpha (real alphas)
      alpha_birth_shells  psi_N shell index of each birth
      alpha_birth_update  index into alpha_rate_updates of the rates each birth used
      alpha_rate_updates  (n_updates, n_shells) rate_k at each field update [1/s]
      weight_np           real ions per row of the returned pos/vel/type arrays
    Shared by both reactor loops.
    """
    return {
        "nbi_per_step": np.asarray(nbi_per_step, dtype=np.int64),
        "nbi_energy_J": float(nbi_energy_J),
        "alpha_per_step": np.asarray(alpha_per_step, dtype=np.int64),
        "alpha_rate_per_step": np.asarray(alpha_rate_per_step, dtype=np.float64),
        "alpha_birth_weight": float(alpha_birth_weight),
        "alpha_birth_shells": np.asarray(alpha_birth_shells, dtype=np.int64),
        "alpha_birth_update": np.asarray(alpha_birth_update, dtype=np.int64),
        "alpha_rate_updates": np.asarray(alpha_rate_updates, dtype=np.float64),
        "weight_np": np.asarray(weight_np, dtype=np.float64),
    }
