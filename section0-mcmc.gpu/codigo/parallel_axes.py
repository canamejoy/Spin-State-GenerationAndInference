"""Where should the parallelism go: over replicas, over experiments, or both?

The bipartite update is mandatory, so it is on in every configuration here --
this study is no longer about whether the algorithm is correct, but about how to
spend a fixed amount of parallel hardware.

The three configurations move **the same total work**: 4 parameter sets x 24
replicas = 96 chains, each doing the same number of sweeps. Only the grouping
changes, which is what makes the wall clocks comparable.

    bp_only    400 launches of   1 chain   (bipartite update the only parallelism)
    replicas     4 launches of 100 chains  (one theta at a time, replicas batched)
    states     100 launches of   4 chains  (all four thetas at once, one replica)
    both         1 launch  of 400 chains  (4 thetas x 100 replicas together)

`experiments` is the configuration that matters for building a dataset, because
the dataset varies theta and not the seed; `replicas` is what you want for error
bars on one state. The question the timings answer is what each grouping costs on
hardware that is only fully used by a wide batch.
"""

from __future__ import annotations

import json
import time

import numpy as np

N_REPLICAS = 100         # the production replica count
N_THETA = 4
TOTAL = N_THETA * N_REPLICAS      # 400 chains in every configuration


def load_thetas(path="theta4.json"):
    """Four medians from the released dataset's phase clusters.

    All four carry `Jex2 = Jex3 = Jex4 = 0`, so the bipartite colouring is exact
    for every one of them -- shells 2 and 4 would otherwise connect same-parity
    sites and the simultaneous update would stop being an exact reordering.
    """
    raw = json.load(open(path))
    out = []
    for name, p in raw.items():
        assert p["Jex2"] == p["Jex3"] == p["Jex4"] == 0.0, (
            f"{name} has a non-zero second/fourth shell; the colouring is not "
            "exact for it and this study would be measuring a different chain")
        out.append(dict(name=name, Jex=1.0, Kan1=p["Kan1"], Kan2=0.0,
                        KanS=p["KanS"], Hex=p["Hex"], KDM=p["KDM"],
                        g=float(np.deg2rad(90.0))))
    return out


def pack(thetas, repeats):
    """Per-chain parameter columns shaped (batch, 1, 1, 1).

    `sim_core` broadcasts scalars, so handing it a column per chain is what lets
    one batch carry several parameter sets instead of several replicas of one.
    """
    import jax.numpy as jnp
    keys = ("Jex", "Kan1", "Kan2", "KanS", "Hex", "KDM", "g")
    cols = {k: jnp.asarray(
        np.repeat([t[k] for t in thetas], repeats).astype(np.float32)
        .reshape(-1, 1, 1, 1)) for k in keys}
    return cols


def time_config(config, geo, thetas, sweeps, beta, seed=0):
    """Wall clock for one grouping of the same 96 chains."""
    import jax
    import jax.numpy as jnp
    import sim_core as S

    eng = S.build_engine(geo)
    ndev = jax.local_device_count()
    surf = jnp.asarray(geo["surface"])
    ev = jnp.asarray(geo["even"][None, ..., None])
    od = jnp.asarray(geo["odd"][None, ..., None])

    if config == "bp_only":
        # No batch axis at all: one chain at a time, the bipartite update being
        # the ONLY parallelism. A single chain cannot be sharded, so extra
        # devices sit idle -- which is the result, not a limitation of the test.
        launches = [([t], 1) for t in thetas for _ in range(N_REPLICAS)]
    elif config == "replicas":      # one theta per launch, replicas batched
        launches = [([t], N_REPLICAS) for t in thetas]
    elif config == "states":        # all four thetas, one replica each
        launches = [(thetas, 1)] * N_REPLICAS
    elif config == "both":          # everything in one batch
        launches = [(thetas, N_REPLICAS)]
    else:
        raise ValueError(config)

    # `body` is defined ONCE. An earlier version built this closure inside the
    # per-launch helper, so `jax.jit` saw a brand-new function every time and
    # `bp_only` -- which has 400 launches -- compiled 400 identical kernels
    # before it could time anything. Compiled functions are now cached by the
    # only thing that changes their shape: how many devices the batch spans.
    def body(s, key, k1_, p, n):
        def one(_, v):
            s, kk = v
            kk, ka, kb = jax.random.split(kk, 3)
            s = eng["substep"](s, ka, beta, ev, p, k1_)
            s = eng["substep"](s, kb, beta, od, p, k1_)
            return s, kk
        return jax.lax.fori_loop(0, n, one, (s, key))

    cache = {}

    def build(ths, rep):
        batch = len(ths) * rep
        # A batch narrower than the device count cannot be sharded. Rather than
        # pad it with idle work, run it on one device and let the number say so.
        use = ndev if batch % ndev == 0 and batch >= ndev else 1
        if use not in cache:
            cache[use] = (jax.jit(body, static_argnums=4) if use == 1
                          else jax.pmap(body, static_broadcasted_argnums=4))
        fn, per = cache[use], batch // use
        cols = pack(ths, rep)
        k1 = jnp.where(surf[None], cols["KanS"], cols["Kan1"])
        spins = S.init_spins(jax.random.PRNGKey(seed), geo, batch)
        if use == 1:
            return fn, spins, jax.random.PRNGKey(seed + 1), k1, cols
        shard = lambda a: a.reshape(use, per, *a.shape[1:])
        return (fn, shard(spins),
                jax.random.split(jax.random.PRNGKey(seed + 1), use),
                shard(k1), {k: shard(v) for k, v in cols.items()})

    built = [build(t, r) for t, r in launches]
    for fn, sp, kk, k1, p in built:                       # compile at this n
        jax.block_until_ready(fn(sp, kk, k1, p, sweeps))
    t0 = time.perf_counter()
    for fn, sp, kk, k1, p in built:
        jax.block_until_ready(fn(sp, kk, k1, p, sweeps))
    dt = time.perf_counter() - t0

    per_launch = len(launches)
    return dict(config=config, devices=ndev, launches=per_launch,
                chains_per_launch=TOTAL // per_launch, total_chains=TOTAL,
                sweeps=sweeps, wall_s=dt,
                ms_per_sweep=1000 * dt / sweeps,
                us_per_chain_sweep=1e6 * dt / (sweeps * TOTAL))
