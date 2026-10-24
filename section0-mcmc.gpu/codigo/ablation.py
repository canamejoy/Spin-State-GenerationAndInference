"""The four `DP x BP` cells of the test plan, as one Hamiltonian and four loops.

The plan's Part 1 crosses two accelerations:

- **DP**, matrix decomposition: the whole lattice's trial energy as array
  operations, instead of arithmetic on one site at a time.
- **BP**, the checkerboard: colouring sites by `(x+y+z) mod 2` so that no two
  sites updated together are neighbours.

They are not two independent speed knobs, and the ablation is worth running
precisely because it shows why:

| cell | loop | status |
|---|---|---|
| `dp_bp` | one array update per colour | the production kernel |
| `nodp_bp` | scalar, colour by colour | same chain, no vectorisation |
| `nodp_nobp` | scalar, lattice order | exact sequential Metropolis |
| `dp_nobp` | array expressions, one site at a time | correct, and *slower* |

`dp_nobp` is the literal reading of "keep the decomposition, drop the
colouring": a sequential chain cannot accept more than one site at a time, so
the vectorised energy is built and then thrown away but for one element. Taking
it seriously is what shows that DP is not a speedup on its own -- it is a
speedup *because* BP makes a simultaneous update legal.

Dropping BP while still updating every site at once is the other reading, and it
is simply wrong: two neighbours would each decide against the other's stale
spin. That variant is available as `masks=(disco,)` and is measured here for its
bias, not offered as a cell.

Every variant evaluates the same energy as `sim_core` -- same interfacial DMI,
same two cubic-anisotropy constants, same surface constant -- which
`tests/test_ablation_equivalence.py` pins against the reference total energy.
Only the loop differs.
"""

from __future__ import annotations

import time

import numpy as np

# ---------------------------------------------------------------------------
# Site orders
# ---------------------------------------------------------------------------


def active_sites(geo) -> list[tuple[int, int, int]]:
    """Every in-disk site as `(z, y, x)`, in C order."""
    return [tuple(int(v) for v in idx) for idx in zip(*np.nonzero(geo["disco"]))]


def lattice_order(geo):
    """Lattice order: what a textbook sequential sweep does."""
    return active_sites(geo)


def colour_order(geo):
    """Every even site, then every odd one: the checkerboard's own order."""
    ev = [tuple(int(v) for v in i) for i in zip(*np.nonzero(geo["even"]))]
    od = [tuple(int(v) for v in i) for i in zip(*np.nonzero(geo["odd"]))]
    return ev + od


def _neighbour_table(geo):
    """Flat `(n_sites, 6)` neighbour indices: xp, xm, yp, ym, zp, zm.

    `x` and `y` wrap and `z` does not, matching the engine's rolls. An absent
    `z` neighbour points at a sentinel slot appended to the spin vectors and
    holding zero, so the exchange sum needs no branch. Sites outside the disk
    are already zero, so they drop out of the sum on their own.
    """
    L, N = geo["L"], geo["N"]
    sent = L * N * N
    z, y, x = np.indices((L, N, N))
    flat = lambda zz, yy, xx: (zz * N + yy) * N + xx  # noqa: E731
    nb = np.empty((L, N, N, 6), dtype=np.int64)
    nb[..., 0] = flat(z, y, (x + 1) % N)
    nb[..., 1] = flat(z, y, (x - 1) % N)
    nb[..., 2] = flat(z, (y + 1) % N, x)
    nb[..., 3] = flat(z, (y - 1) % N, x)
    nb[..., 4] = np.where(z + 1 < L, flat(np.minimum(z + 1, L - 1), y, x), sent)
    nb[..., 5] = np.where(z - 1 >= 0, flat(np.maximum(z - 1, 0), y, x), sent)
    return nb.reshape(-1, 6), sent


# ---------------------------------------------------------------------------
# The energy difference, once
# ---------------------------------------------------------------------------


def _dE(sx, sy, sz, nb, i, k1, bx, by, bz, Jex, Kan2, KDM, hx, hz):
    """Energy change from setting site `i` to `(bx, by, bz)`. Plain floats.

    Written against Python lists rather than arrays on purpose: this is the
    *un*-decomposed arm, and giving it numpy's per-element overhead would charge
    the baseline for a cost the comparison is meant to isolate.
    """
    ax, ay, az = sx[i], sy[i], sz[i]
    dx, dy, dz = bx - ax, by - ay, bz - az

    xp, xm, yp, ym, zp, zm = nb[i]
    nx = sx[xp] + sx[xm] + sx[yp] + sx[ym] + sx[zp] + sx[zm]
    ny = sy[xp] + sy[xm] + sy[yp] + sy[ym] + sy[zp] + sy[zm]
    nz = sz[xp] + sz[xm] + sz[yp] + sz[ym] + sz[zp] + sz[zm]
    e = -Jex * (dx * nx + dy * ny + dz * nz)

    # Cubic anisotropy, K1 invariant plus the sixth-order K2 one.
    e += (k1 * ((bx * by) ** 2 + (bx * bz) ** 2 + (by * bz) ** 2)
          + Kan2 * (bx * by * bz) ** 2
          - k1 * ((ax * ay) ** 2 + (ax * az) ** 2 + (ay * az) ** 2)
          - Kan2 * (ax * ay * az) ** 2)

    # Interfacial DMI: in-plane bonds only, minus sign.
    e -= KDM * (dy * (sz[xp] - sz[xm]) - dz * (sy[xp] - sy[xm])
                + dz * (sx[yp] - sx[ym]) - dx * (sz[yp] - sz[ym]))

    e -= hx * dx + hz * dz
    return e


def _unpack(p, K1_eff, geo):
    Jex, Kan2, KDM = p["Jex"], p["Kan2"], p["KDM"]
    hx, hz = p["Hex"] * np.cos(p["g"]), p["Hex"] * np.sin(p["g"])
    k1 = np.asarray(K1_eff, dtype=np.float64).ravel().tolist()
    return Jex, Kan2, KDM, hx, hz, k1


def _flatten(spins, geo):
    """`(L, N, N, 3)` -> three Python lists with a trailing zero sentinel."""
    a = np.asarray(spins, dtype=np.float64).reshape(-1, 3)
    return ([*a[:, 0].tolist(), 0.0], [*a[:, 1].tolist(), 0.0],
            [*a[:, 2].tolist(), 0.0])


def local_dE(spins, idx, trial, p, K1_eff, geo):
    """Single-site energy difference, for the tests to check against `sim_core`."""
    L, N = geo["L"], geo["N"]
    nb, _ = _neighbour_table(geo)
    sx, sy, sz = _flatten(spins, geo)
    Jex, Kan2, KDM, hx, hz, k1 = _unpack(p, K1_eff, geo)
    z, y, x = idx
    i = (z * N + y) * N + x
    return _dE(sx, sy, sz, nb.tolist(), i, k1[i],
               float(trial[0]), float(trial[1]), float(trial[2]),
               Jex, Kan2, KDM, hx, hz)


# ---------------------------------------------------------------------------
# The loops
# ---------------------------------------------------------------------------


def prepare_scalar(geo, p, K1_eff):
    """Everything a scalar sweep needs that does not change between sweeps.

    Built once. Rebuilding the neighbour table inside the timed loop charged a
    fixed ~7,600-row setup to whatever fraction of the lattice a scaled run
    visited, which made the two scalar cells differ by 43% for no physical
    reason -- they do the same work in a different order.
    """
    nb, _ = _neighbour_table(geo)
    Jex, Kan2, KDM, hx, hz, k1 = _unpack(p, K1_eff, geo)
    return dict(nb=nb.tolist(), k1=k1, Jex=Jex, Kan2=Kan2, KDM=KDM,
                hx=hx, hz=hz, N=geo["N"], L=geo["L"])


def _sweep_flat(sx, sy, sz, ctx, idx, px, py, pz, uu, beta):
    """Sequential Metropolis over `idx`, mutating the flat lists in place.

    `idx` is a list of flat site indices and the four randomness lists are
    indexed by *position in that list*, not by site, so a timed run can draw
    exactly as many proposals as it uses while the tests can still hand both
    loops the same per-site values.
    """
    nb, k1 = ctx["nb"], ctx["k1"]
    Jex, Kan2, KDM, hx, hz = (ctx["Jex"], ctx["Kan2"], ctx["KDM"],
                              ctx["hx"], ctx["hz"])
    exp = np.exp
    for j, i in enumerate(idx):
        bx, by, bz = px[j], py[j], pz[j]
        e = _dE(sx, sy, sz, nb, i, k1[i], bx, by, bz,
                Jex, Kan2, KDM, hx, hz)
        if e < 0.0 or uu[j] < exp(-e * beta):
            sx[i], sy[i], sz[i] = bx, by, bz


def _unflatten(sx, sy, sz, geo):
    L, N = geo["L"], geo["N"]
    out = np.empty((L * N * N, 3))
    out[:, 0] = sx[:-1]
    out[:, 1] = sy[:-1]
    out[:, 2] = sz[:-1]
    return out.reshape(L, N, N, 3)


def _flat_indices(order, N):
    return [(z * N + y) * N + x for z, y, x in order]


def sweep_scalar(spins, prop, u, beta, p, K1_eff, geo, order, max_sites=None,
                 ctx=None):
    """Sequential single-site Metropolis in the given site order.

    `prop` and `u` are full-lattice fields so a scalar sweep and a vectorised
    one can be driven by *identical* randomness, which is what makes the
    equivalence test an equality rather than a comparison of distributions.

    `max_sites` stops after that many sites. Per-site cost does not depend on
    how many sites have already been visited, so a timed fraction of a sweep
    scales exactly -- which is the only way the un-vectorised arm is affordable.
    """
    ctx = ctx or prepare_scalar(geo, p, K1_eff)
    sx, sy, sz = _flatten(spins, geo)
    sites = order if max_sites is None else order[:max_sites]
    idx = _flat_indices(sites, geo["N"])
    pr = np.asarray(prop, dtype=np.float64).reshape(-1, 3)
    uu = np.asarray(u, dtype=np.float64).ravel()
    _sweep_flat(sx, sy, sz, ctx, idx,
                pr[idx, 0].tolist(), pr[idx, 1].tolist(), pr[idx, 2].tolist(),
                uu[idx].tolist(), beta)
    return _unflatten(sx, sy, sz, geo)


def sweep_vectorised(spins, prop, u, beta, p, K1_eff, geo, masks):
    """One array update per mask. `masks=(even, odd)` is the production kernel.

    `masks=(disco,)` is the un-coloured update: fast, and not the Boltzmann
    chain. It is kept here so its bias can be measured rather than assumed.
    """
    s = np.asarray(spins, dtype=np.float64).copy()
    pr = np.asarray(prop, dtype=np.float64)
    uu = np.asarray(u, dtype=np.float64)
    k1 = np.asarray(K1_eff, dtype=np.float64)
    Jex, Kan2, KDM = p["Jex"], p["Kan2"], p["KDM"]
    hx, hz = p["Hex"] * np.cos(p["g"]), p["Hex"] * np.sin(p["g"])

    def an(v):
        vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
        return (k1 * ((vx * vy) ** 2 + (vx * vz) ** 2 + (vy * vz) ** 2)
                + Kan2 * (vx * vy * vz) ** 2)

    for mask in masks:
        m = np.asarray(mask, dtype=bool)
        # z does not wrap; the rolled-in layer is zeroed, as in the engine.
        xp, xm = np.roll(s, -1, 2), np.roll(s, 1, 2)
        yp, ym = np.roll(s, -1, 1), np.roll(s, 1, 1)
        zp, zm = np.roll(s, -1, 0), np.roll(s, 1, 0)
        zp[-1] = 0.0
        zm[0] = 0.0

        trial = np.where(m[..., None], pr, s)
        d = trial - s
        nbsum = xp + xm + yp + ym + zp + zm
        dE = -Jex * np.sum(d * nbsum, axis=-1)
        dE += an(trial) - an(s)
        dE -= KDM * (d[..., 1] * (xp[..., 2] - xm[..., 2])
                     - d[..., 2] * (xp[..., 1] - xm[..., 1])
                     + d[..., 2] * (yp[..., 0] - ym[..., 0])
                     - d[..., 0] * (yp[..., 2] - ym[..., 2]))
        dE -= hx * d[..., 0] + hz * d[..., 2]

        with np.errstate(over="ignore"):
            acc = ((dE < 0.0) | (uu < np.exp(-dE * beta))) & m
        s = np.where(acc[..., None], trial, s)
    return s


def sweep_sitewise_batched(spins, prop, u, beta, p, K1_eff, geo, order,
                           max_sites=None):
    """Site-by-site over the lattice, vectorised over replicas.

    This is what the reference notebook's `run_mode_cpu` actually does, and it
    matters for reading the ablation: that cell is labelled "CPU clásico
    (site-by-site)" but its `get_neighbors` indexes `sp[:, z, y, (x+1) % Nx]`,
    so every numpy operation is spread across the whole replica batch. The
    decomposition is already there -- on the replica axis rather than the
    lattice axis.

    The two axes are not equivalent. Replicas are independent chains, so
    batching over them needs no colouring and raises no detailed-balance
    question at all; the lattice axis is the one the checkerboard exists for.
    Treating a scalar single-replica loop as "the CPU baseline" would therefore
    compare the production kernel against something nobody wrote.

    `spins`, `prop` and `u` carry a leading replica axis.
    """
    s = np.asarray(spins, dtype=np.float64).copy()
    pr = np.asarray(prop, dtype=np.float64)
    uu = np.asarray(u, dtype=np.float64)
    k1 = np.asarray(K1_eff, dtype=np.float64)
    Jex, Kan2, KDM = p["Jex"], p["Kan2"], p["KDM"]
    hx, hz = p["Hex"] * np.cos(p["g"]), p["Hex"] * np.sin(p["g"])
    B, L, N = s.shape[0], geo["L"], geo["N"]
    zero = np.zeros((B, 3))

    def an(v, kk):
        return (kk * ((v[:, 0] * v[:, 1]) ** 2 + (v[:, 0] * v[:, 2]) ** 2
                      + (v[:, 1] * v[:, 2]) ** 2)
                + Kan2 * (v[:, 0] * v[:, 1] * v[:, 2]) ** 2)

    sites = order if max_sites is None else order[:max_sites]
    for z, y, x in sites:
        xp, xm = s[:, z, y, (x + 1) % N], s[:, z, y, (x - 1) % N]
        yp, ym = s[:, z, (y + 1) % N, x], s[:, z, (y - 1) % N, x]
        zp = s[:, z + 1, y, x] if z + 1 < L else zero
        zm = s[:, z - 1, y, x] if z - 1 >= 0 else zero

        a, b = s[:, z, y, x], pr[:, z, y, x]
        d = b - a
        kk = k1[z, y, x]
        dE = -Jex * np.sum(d * (xp + xm + yp + ym + zp + zm), axis=-1)
        dE += an(b, kk) - an(a, kk)
        dE -= KDM * (d[:, 1] * (xp[:, 2] - xm[:, 2])
                     - d[:, 2] * (xp[:, 1] - xm[:, 1])
                     + d[:, 2] * (yp[:, 0] - ym[:, 0])
                     - d[:, 0] * (yp[:, 2] - ym[:, 2]))
        dE -= hx * d[:, 0] + hz * d[:, 2]
        with np.errstate(over="ignore"):
            acc = (dE < 0.0) | (uu[:, z, y, x] < np.exp(-dE * beta))
        s[:, z, y, x] = np.where(acc[:, None], b, a)
    return s


def sweep_vec_sequential(spins, prop, u, beta, p, K1_eff, geo, order,
                         max_sites=None):
    """`CON DP + SIN BP`: array expressions, one accepted site at a time.

    The full rolled neighbour field is built, one element of it is used, and the
    rest is discarded -- which is all a sequential chain can do with a
    vectorised energy, and the reason this cell is the slowest of the four.
    """
    s = np.asarray(spins, dtype=np.float64).copy()
    pr = np.asarray(prop, dtype=np.float64)
    uu = np.asarray(u, dtype=np.float64)
    k1 = np.asarray(K1_eff, dtype=np.float64)
    Jex, Kan2, KDM = p["Jex"], p["Kan2"], p["KDM"]
    hx, hz = p["Hex"] * np.cos(p["g"]), p["Hex"] * np.sin(p["g"])

    sites = order if max_sites is None else order[:max_sites]
    for z, y, x in sites:
        xp, xm = np.roll(s, -1, 2), np.roll(s, 1, 2)
        yp, ym = np.roll(s, -1, 1), np.roll(s, 1, 1)
        zp, zm = np.roll(s, -1, 0), np.roll(s, 1, 0)
        zp[-1] = 0.0
        zm[0] = 0.0
        nbsum = (xp + xm + yp + ym + zp + zm)[z, y, x]

        a, b = s[z, y, x], pr[z, y, x]
        d = b - a
        kk = k1[z, y, x]
        anv = lambda v: (kk * ((v[0] * v[1]) ** 2 + (v[0] * v[2]) ** 2  # noqa: E731
                              + (v[1] * v[2]) ** 2)
                         + Kan2 * (v[0] * v[1] * v[2]) ** 2)
        dE = -Jex * float(d @ nbsum) + anv(b) - anv(a)
        dE -= KDM * (d[1] * (xp[z, y, x, 2] - xm[z, y, x, 2])
                     - d[2] * (xp[z, y, x, 1] - xm[z, y, x, 1])
                     + d[2] * (yp[z, y, x, 0] - ym[z, y, x, 0])
                     - d[0] * (yp[z, y, x, 2] - ym[z, y, x, 2]))
        dE -= hx * d[0] + hz * d[2]
        if dE < 0.0 or uu[z, y, x] < np.exp(-min(dE * beta, 700.0)):
            s[z, y, x] = b
    return s


# ---------------------------------------------------------------------------
# The four cells, with their own randomness, for timing
# ---------------------------------------------------------------------------


def _randomness(rng, geo):
    pr = rng.normal(size=(geo["L"], geo["N"], geo["N"], 3))
    pr /= np.linalg.norm(pr, axis=-1, keepdims=True)
    pr *= geo["disco"][..., None]
    return pr, rng.random((geo["L"], geo["N"], geo["N"]))


def _cell(kind):
    def run(spins, rng, beta, p, K1_eff, geo, max_sites=None):
        pr, u = _randomness(rng, geo)
        if kind == "dp_bp":
            return sweep_vectorised(spins, pr, u, beta, p, K1_eff, geo,
                                    masks=(geo["even"], geo["odd"]))
        if kind == "nodp_bp":
            return sweep_scalar(spins, pr, u, beta, p, K1_eff, geo,
                                colour_order(geo), max_sites)
        if kind == "nodp_nobp":
            return sweep_scalar(spins, pr, u, beta, p, K1_eff, geo,
                                lattice_order(geo), max_sites)
        if kind == "dp_nobp":
            return sweep_vec_sequential(spins, pr, u, beta, p, K1_eff, geo,
                                        lattice_order(geo), max_sites)
        raise ValueError(kind)
    run.__name__ = kind
    return run


VARIANTS = {k: _cell(k) for k in ("dp_bp", "nodp_bp", "nodp_nobp", "dp_nobp")}

# Not a cell: the fast-and-wrong reading of "drop the checkerboard".
def sweep_uncoloured(spins, rng, beta, p, K1_eff, geo):
    pr, u = _randomness(rng, geo)
    return sweep_vectorised(spins, pr, u, beta, p, K1_eff, geo,
                            masks=(geo["disco"],))


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def time_cell(name, geo, p, K1_eff, beta=3.0, seed=0, sweeps=3,
              max_sites=None, warmup=True):
    """Wall clock for one cell, reported as ms per full lattice sweep.

    Everything that a real implementation would build once -- the neighbour
    table, the per-site anisotropy constant, the flattened state -- is built
    once, outside the timed loop. Only the sweep is timed, and the randomness it
    consumes is drawn to the size of the sites actually visited, so a scaled run
    is not charged for proposals it never uses.

    A cell timed with `max_sites` is charged pro rata: per-site cost does not
    depend on position in the sweep, so `n_active / max_sites` is an exact
    scaling and not an extrapolation of a trend.
    """
    rng = np.random.default_rng(seed)
    s = rng.normal(size=(geo["L"], geo["N"], geo["N"], 3))
    s /= np.linalg.norm(s, axis=-1, keepdims=True)
    s *= geo["disco"][..., None]

    order = colour_order(geo) if name == "nodp_bp" else lattice_order(geo)
    sites = order if max_sites is None else order[:max_sites]
    n_visit = len(sites)

    def draw():
        pr = rng.normal(size=(n_visit, 3))
        pr /= np.linalg.norm(pr, axis=-1, keepdims=True)
        return (pr[:, 0].tolist(), pr[:, 1].tolist(), pr[:, 2].tolist(),
                rng.random(n_visit).tolist())

    if name == "dp_bp":
        assert max_sites is None, "the vectorised cell sweeps the whole lattice"
        reps = sweeps + (1 if warmup else 0)
        pr_u = [_randomness(rng, geo) for _ in range(reps)]
        if warmup:
            s = sweep_vectorised(s, *pr_u[0], beta, p, K1_eff, geo,
                                 masks=(geo["even"], geo["odd"]))
        t0 = time.perf_counter()
        for pr, u in pr_u[1:] if warmup else pr_u:
            s = sweep_vectorised(s, pr, u, beta, p, K1_eff, geo,
                                 masks=(geo["even"], geo["odd"]))
        dt = time.perf_counter() - t0

    elif name in ("nodp_bp", "nodp_nobp"):
        ctx = prepare_scalar(geo, p, K1_eff)
        idx = _flat_indices(sites, geo["N"])
        sx, sy, sz = _flatten(s, geo)
        if warmup:
            _sweep_flat(sx, sy, sz, ctx, idx[:8], *[d[:8] for d in draw()], beta)
        pool = [draw() for _ in range(sweeps)]
        t0 = time.perf_counter()
        for px, py, pz, uu in pool:
            _sweep_flat(sx, sy, sz, ctx, idx, px, py, pz, uu, beta)
        dt = time.perf_counter() - t0
        s = _unflatten(sx, sy, sz, geo)

    elif name == "dp_nobp":
        full = lambda: _randomness(rng, geo)  # noqa: E731
        if warmup:
            pr, u = full()
            sweep_vec_sequential(s, pr, u, beta, p, K1_eff, geo, sites[:4])
        pool = [full() for _ in range(sweeps)]
        t0 = time.perf_counter()
        for pr, u in pool:
            s = sweep_vec_sequential(s, pr, u, beta, p, K1_eff, geo, sites)
        dt = time.perf_counter() - t0

    else:
        raise ValueError(name)

    done = n_visit * sweeps
    per_site_us = 1e6 * dt / done
    return dict(cell=name, wall_s=dt, sweeps=sweeps,
                sites_visited=done, n_active=geo["n_active"],
                us_per_site=per_site_us,
                ms_per_sweep=per_site_us * geo["n_active"] / 1000.0,
                scaled=max_sites is not None)


def time_sitewise_batched(geo, p, K1_eff, batch, beta=3.0, seed=0, sweeps=2,
                          max_sites=None):
    """Cost of the reference notebook's CPU mode, per sweep and per replica.

    Reported both ways on purpose. Per sweep it looks slow; per replica-sweep it
    is the number that belongs beside the GPU rows, because the batch axis is
    doing the same job there as the lattice axis does in the production kernel.
    """
    rng = np.random.default_rng(seed)
    L, N = geo["L"], geo["N"]
    s = rng.normal(size=(batch, L, N, N, 3))
    s /= np.linalg.norm(s, axis=-1, keepdims=True)
    s *= geo["disco"][None, ..., None]
    order = lattice_order(geo)
    sites = order if max_sites is None else order[:max_sites]

    def draw():
        pr = rng.normal(size=(batch, L, N, N, 3))
        pr /= np.linalg.norm(pr, axis=-1, keepdims=True)
        pr *= geo["disco"][None, ..., None]
        return pr, rng.random((batch, L, N, N))

    pr, u = draw()
    sweep_sitewise_batched(s, pr, u, beta, p, K1_eff, geo, sites[:4])
    pool = [draw() for _ in range(sweeps)]
    t0 = time.perf_counter()
    for pr, u in pool:
        s = sweep_sitewise_batched(s, pr, u, beta, p, K1_eff, geo, sites,
                                   max_sites)
    dt = time.perf_counter() - t0

    n_visit = len(sites)
    per_site_us = 1e6 * dt / (n_visit * sweeps)
    ms_sweep = per_site_us * geo["n_active"] / 1000.0
    return dict(cell="sitewise_batched", batch=int(batch), wall_s=dt,
                sweeps=sweeps, us_per_site=per_site_us,
                ms_per_sweep=ms_sweep,
                ms_per_replica_sweep=ms_sweep / batch,
                scaled=max_sites is not None)
