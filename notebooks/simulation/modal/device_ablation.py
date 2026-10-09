"""The full `DP x BP` x device grid: four cells on CPU, 1, 2 and 4 H100.

The CPU ablation answered what each acceleration costs on one processor. This
answers the test plan's actual Part 1 shape -- sixteen cells -- by measuring the
twelve that were previously filled in by argument rather than by measurement.

**Every cell runs the production configuration: 100 replicas.** The earlier
ablation used 1 replica, which is fine for isolating `DP` but makes a
cross-device chart meaningless, because the GPU arms exist to exploit the batch.
Reporting `ms/sweep` at a fixed 100 keeps one number comparable down a column and
across a row.

**Each device gets the best implementation available to it, not a shared one.**
On CPU the un-decomposed cells are a numpy loop batched over replicas -- which is
what the reference notebook actually does; on GPU they are eager JAX operations.
Forcing eager JAX onto the CPU so that both run "the same code" would compare two
bad implementations and make the headline claim -- that a GPU without the
decomposition is worse than a CPU -- an artefact of the harness. The point of the
comparison is what a practitioner would really write on each device.

Slow cells are timed over `max_sites` and charged pro rata. Per-site cost does not
depend on position in the sweep, so `n_active / max_sites` is an exact scaling,
not an extrapolated trend -- the same argument the CPU ablation rests on, and the
only reason the sequential GPU cells are affordable at all.
"""

from __future__ import annotations

import time

import numpy as np

CELLS = ("nodp_nobp", "nodp_bp", "dp_nobp", "dp_bp")

# "CON DP + CON BP" last: it is the production kernel and the conclusion of the
# sequence, not a middle entry.
CELL_LABEL = {
    "nodp_nobp": "SIN DP\nSIN BP",
    "nodp_bp": "SIN DP\nCON BP",
    "dp_nobp": "CON DP\nSIN BP",
    "dp_bp": "CON DP\nCON BP",
}


def _site_order(cell, geo):
    import ablation
    return (ablation.colour_order(geo) if cell.endswith("_bp") and
            cell.startswith("nodp") else ablation.lattice_order(geo))


# ---------------------------------------------------------------------------
# GPU / JAX
# ---------------------------------------------------------------------------


def measure_jax(cell, geo, params, replicas, max_sites, sweeps, seed=0):
    """One cell on whatever JAX devices are present, at `replicas` replicas."""
    import jax
    import jax.numpy as jnp
    import sim_core as S

    ndev = jax.local_device_count()
    assert replicas % ndev == 0, f"{replicas} replicas over {ndev} devices"
    per = replicas // ndev
    L, N = geo["L"], geo["N"]
    k1 = jnp.asarray(np.where(geo["surface"], params["KanS"], params["Kan1"]))
    beta = 1.0 / (0.086173404 * 3.0)
    eng = S.build_engine(geo)
    hx = params["Hex"] * np.cos(params["g"])
    hz = params["Hex"] * np.sin(params["g"])

    def shard(key):
        return S.init_spins(key, geo, per)

    keys = jax.random.split(jax.random.PRNGKey(seed), ndev)
    spins = jax.pmap(shard)(keys) if ndev > 1 else shard(keys[0])

    if cell == "dp_bp":
        ev = jnp.asarray(geo["even"][None, ..., None])
        od = jnp.asarray(geo["odd"][None, ..., None])

        def body(s, key, n):
            def one(_, v):
                s, k = v
                k, k1_, k2 = jax.random.split(k, 3)
                s = eng["substep"](s, k1_, beta, ev, params, k1)
                s = eng["substep"](s, k2, beta, od, params, k1)
                return s, k
            return jax.lax.fori_loop(0, n, one, (s, key))

        run = (jax.pmap(body, static_broadcasted_argnums=2) if ndev > 1
               else jax.jit(body, static_argnums=2))
        kk = (jax.random.split(jax.random.PRNGKey(seed + 1), ndev) if ndev > 1
              else jax.random.PRNGKey(seed + 1))
        # Warm up with the SAME sweep count. `n` is a static argument, so
        # compiling at 20 and timing at 100 recompiles inside the timed region
        # and reports the compiler: it read 120 ms/sweep against the production
        # run's measured 0.303.
        jax.block_until_ready(run(spins, kk, sweeps))
        t0 = time.perf_counter()
        jax.block_until_ready(run(spins, kk, sweeps))
        dt = time.perf_counter() - t0
        return dict(cell=cell, replicas=replicas, devices=ndev,
                    ms_per_sweep=1000 * dt / sweeps, scaled=False,
                    sites_timed=geo["n_active"], wall_s=dt)

    # The sequential cells. No `jit` over the site loop: fusing the sweep into
    # one kernel IS the decomposition, so jitting it would measure the cell this
    # one is defined as not being.
    order = _site_order(cell, geo)[:max_sites]
    zero = jnp.zeros((per, 3))

    def an(v, kk_):
        return (kk_ * ((v[:, 0] * v[:, 1]) ** 2 + (v[:, 0] * v[:, 2]) ** 2
                       + (v[:, 1] * v[:, 2]) ** 2)
                + params["Kan2"] * (v[:, 0] * v[:, 1] * v[:, 2]) ** 2)

    def site_step(s, key, z, y, x):
        key, kt, ka = jax.random.split(key, 3)
        b = jax.random.normal(kt, (per, 3))
        b = b / jnp.linalg.norm(b, axis=-1, keepdims=True)
        if cell == "dp_nobp":
            # CON DP: build the whole rolled neighbour field, then use one site.
            xp_, xm_ = jnp.roll(s, -1, 3), jnp.roll(s, 1, 3)
            yp_, ym_ = jnp.roll(s, -1, 2), jnp.roll(s, 1, 2)
            zp_ = jnp.roll(s, -1, 1).at[:, -1].set(0.0)
            zm_ = jnp.roll(s, 1, 1).at[:, 0].set(0.0)
            xp, xm = xp_[:, z, y, x], xm_[:, z, y, x]
            yp, ym = yp_[:, z, y, x], ym_[:, z, y, x]
            zp, zm = zp_[:, z, y, x], zm_[:, z, y, x]
        else:
            xp, xm = s[:, z, y, (x + 1) % N], s[:, z, y, (x - 1) % N]
            yp, ym = s[:, z, (y + 1) % N, x], s[:, z, (y - 1) % N, x]
            zp = s[:, z + 1, y, x] if z + 1 < L else zero
            zm = s[:, z - 1, y, x] if z - 1 >= 0 else zero
        a = s[:, z, y, x]
        d = b - a
        kk_ = k1[z, y, x]
        dE = -params["Jex"] * jnp.sum(d * (xp + xm + yp + ym + zp + zm), axis=-1)
        dE = dE + an(b, kk_) - an(a, kk_)
        dE = dE - params["KDM"] * (d[:, 1] * (xp[:, 2] - xm[:, 2])
                                   - d[:, 2] * (xp[:, 1] - xm[:, 1])
                                   + d[:, 2] * (yp[:, 0] - ym[:, 0])
                                   - d[:, 0] * (yp[:, 2] - ym[:, 2]))
        dE = dE - (hx * d[:, 0] + hz * d[:, 2])
        acc = (dE < 0) | (jax.random.uniform(ka, dE.shape) < jnp.exp(-dE * beta))
        return s.at[:, z, y, x].set(jnp.where(acc[:, None], b, a)), key

    def loop(s, key):
        for z, y, x in order:
            s, key = site_step(s, key, z, y, x)
        return s, key

    run = jax.pmap(loop) if ndev > 1 else loop
    kk = (jax.random.split(jax.random.PRNGKey(seed + 1), ndev) if ndev > 1
          else jax.random.PRNGKey(seed + 1))
    s, kk = run(spins, kk)
    jax.block_until_ready(s)
    t0 = time.perf_counter()
    for _ in range(sweeps):
        s, kk = run(s, kk)
    jax.block_until_ready(s)
    dt = time.perf_counter() - t0

    us_site = 1e6 * dt / (len(order) * sweeps)
    return dict(cell=cell, replicas=replicas, devices=ndev,
                ms_per_sweep=us_site * geo["n_active"] / 1000.0,
                us_per_site=us_site, scaled=True, sites_timed=len(order),
                wall_s=dt)


# ---------------------------------------------------------------------------
# CPU / numpy
# ---------------------------------------------------------------------------


def measure_cpu(cell, geo, params, replicas, max_sites, sweeps, seed=0):
    """One cell on CPU, with the best numpy implementation for it.

    The un-decomposed cells are batched over replicas, as the reference
    notebook's CPU mode is -- that is the honest CPU baseline, and a scalar
    single-replica loop multiplied by 100 would not be.
    """
    import ablation

    K1 = np.where(geo["surface"], params["KanS"], params["Kan1"])
    rng = np.random.default_rng(seed)
    L, N = geo["L"], geo["N"]

    def states(b):
        s = rng.normal(size=(b, L, N, N, 3))
        s /= np.linalg.norm(s, axis=-1, keepdims=True)
        return s * geo["disco"][None, ..., None]

    def draw(b):
        pr = rng.normal(size=(b, L, N, N, 3))
        pr /= np.linalg.norm(pr, axis=-1, keepdims=True)
        pr *= geo["disco"][None, ..., None]
        return pr, rng.random((b, L, N, N))

    # `dp_bp` is not handled here. The best CPU implementation of that cell is
    # the shipped JAX kernel, which `measure_jax` runs on a CPU container -- a
    # numpy loop calling the single-replica sweep 100 times would be a worse
    # implementation chosen only to keep one file tidy, and would understate the
    # cell the whole project depends on.
    assert cell != "dp_bp", "measure dp_bp on CPU with measure_jax"

    order = _site_order(cell, geo)[:max_sites]
    s = states(replicas)
    fn = (ablation.sweep_sitewise_batched if cell.startswith("nodp")
          else _vec_sequential_batched)
    pr, u = draw(replicas)
    fn(s, pr, u, 1.0, params, K1, geo, order[:2])
    pool = [draw(replicas) for _ in range(sweeps)]
    t0 = time.perf_counter()
    for pr, u in pool:
        s = fn(s, pr, u, 1.0, params, K1, geo, order)
    dt = time.perf_counter() - t0

    us_site = 1e6 * dt / (len(order) * sweeps)
    return dict(cell=cell, replicas=replicas, devices=0, wall_s=dt,
                ms_per_sweep=us_site * geo["n_active"] / 1000.0,
                us_per_site=us_site, scaled=True, sites_timed=len(order))


def _vec_sequential_batched(s, prop, u, beta, p, K1_eff, geo, order,
                            max_sites=None):
    """`CON DP + SIN BP` on CPU, batched over replicas.

    The whole rolled neighbour field is built for every accepted site and all but
    one element discarded -- which is all a sequential chain can do with a
    vectorised energy, and why this cell is the slowest on every device.
    """
    s = np.asarray(s, dtype=np.float64).copy()
    pr = np.asarray(prop, dtype=np.float64)
    uu = np.asarray(u, dtype=np.float64)
    k1 = np.asarray(K1_eff, dtype=np.float64)
    Jex, Kan2, KDM = p["Jex"], p["Kan2"], p["KDM"]
    hx, hz = p["Hex"] * np.cos(p["g"]), p["Hex"] * np.sin(p["g"])
    sites = order if max_sites is None else order[:max_sites]

    for z, y, x in sites:
        xp_, xm_ = np.roll(s, -1, 3), np.roll(s, 1, 3)
        yp_, ym_ = np.roll(s, -1, 2), np.roll(s, 1, 2)
        zp_, zm_ = np.roll(s, -1, 1), np.roll(s, 1, 1)
        zp_[:, -1] = 0.0
        zm_[:, 0] = 0.0
        xp, xm = xp_[:, z, y, x], xm_[:, z, y, x]
        yp, ym = yp_[:, z, y, x], ym_[:, z, y, x]
        zp, zm = zp_[:, z, y, x], zm_[:, z, y, x]

        a, b = s[:, z, y, x], pr[:, z, y, x]
        d = b - a
        kk = k1[z, y, x]

        def an(v):
            return (kk * ((v[:, 0] * v[:, 1]) ** 2 + (v[:, 0] * v[:, 2]) ** 2
                          + (v[:, 1] * v[:, 2]) ** 2)
                    + Kan2 * (v[:, 0] * v[:, 1] * v[:, 2]) ** 2)

        dE = -Jex * np.sum(d * (xp + xm + yp + ym + zp + zm), axis=-1)
        dE += an(b) - an(a)
        dE -= KDM * (d[:, 1] * (xp[:, 2] - xm[:, 2])
                     - d[:, 2] * (xp[:, 1] - xm[:, 1])
                     + d[:, 2] * (yp[:, 0] - ym[:, 0])
                     - d[:, 0] * (yp[:, 2] - ym[:, 2]))
        dE -= hx * d[:, 0] + hz * d[:, 2]
        with np.errstate(over="ignore"):
            acc = (dE < 0.0) | (uu[:, z, y, x] < np.exp(-dE * beta))
        s[:, z, y, x] = np.where(acc[:, None], b, a)
    return s


def sweep_vectorised_batched(s, prop, u, beta, p, K1_eff, geo, masks):
    """`MD + BP` in plain numpy, batched over replicas.

    The point of this one is to separate two things that are easy to conflate.
    **Matrix decomposition is an algorithm**: express the trial energy of many
    sites as array operations. **JAX is an execution technology**: it fuses those
    array operations into one compiled kernel and can place it on a GPU. JAX
    cannot create parallelism the algorithm does not express -- which is exactly
    why the `no MD` cells are catastrophic on a GPU.

    Running the identical algorithm, on the identical device, at the identical
    batch, in numpy and in JAX is the only way to attribute the speedup to one or
    the other instead of quoting a single combined number.
    """
    s = np.asarray(s, dtype=np.float64).copy()
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
        m = np.asarray(mask, dtype=bool)[None]
        xp, xm = np.roll(s, -1, 3), np.roll(s, 1, 3)
        yp, ym = np.roll(s, -1, 2), np.roll(s, 1, 2)
        zp, zm = np.roll(s, -1, 1), np.roll(s, 1, 1)
        zp[:, -1] = 0.0
        zm[:, 0] = 0.0

        trial = np.where(m[..., None], pr, s)
        d = trial - s
        dE = -Jex * np.sum(d * (xp + xm + yp + ym + zp + zm), axis=-1)
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


def measure_numpy_md(geo, params, replicas, sweeps, seed=0):
    """`MD + BP` in numpy at `replicas` replicas -- the JAX-free comparator."""
    K1 = np.where(geo["surface"], params["KanS"], params["Kan1"])
    rng = np.random.default_rng(seed)
    L, N = geo["L"], geo["N"]
    masks = (geo["even"], geo["odd"])

    def draw():
        pr = rng.normal(size=(replicas, L, N, N, 3))
        pr /= np.linalg.norm(pr, axis=-1, keepdims=True)
        pr *= geo["disco"][None, ..., None]
        return pr, rng.random((replicas, L, N, N))

    s = states = rng.normal(size=(replicas, L, N, N, 3))
    s /= np.linalg.norm(s, axis=-1, keepdims=True)
    s *= geo["disco"][None, ..., None]

    pr, u = draw()
    s = sweep_vectorised_batched(s, pr, u, 1.0, params, K1, geo, masks)
    pool = [draw() for _ in range(sweeps)]
    t0 = time.perf_counter()
    for pr, u in pool:
        s = sweep_vectorised_batched(s, pr, u, 1.0, params, K1, geo, masks)
    dt = time.perf_counter() - t0
    return dict(cell="dp_bp", engine="numpy", replicas=replicas, devices=0,
                wall_s=dt, ms_per_sweep=1000 * dt / sweeps, scaled=False,
                sites_timed=geo["n_active"], sweeps=sweeps)
