"""B = 64 against B = 65: the smallest experiment that shows the rule.

One replica apart, and the whole effect:

    elements = B * L * N^2 * 3,  N odd, so  v2(elements) = v2(B) + v2(L)
    XLA gives each thread  min(4, 2^v2(elements))  elements

    L = 5, B = 64:  v2(5) + v2(64) = 0 + 6 = 6  ->  width 4  ->  penalty
    L = 5, B = 65:  v2(5) + v2(65) = 0 + 0 = 0  ->  width 1  ->  free

Everything else is held: the same lattice, the same schedule, the same seeds,
the same container, the same physical GPU. The only difference is one replica.

Two design points that matter for believing the result.

First, the two arms are **interleaved** -- 64, 65, 64, 65, ... -- not run as two
blocks. A GPU that warms up or gets throttled during the run would bias whole
blocks; interleaving spreads any drift across both arms instead of loading it
onto one.

Second, compilation is **outside** the timer. `.lower().compile()` runs before
the clock starts, so the number is execution only. That matters here because the
two shapes compile to different programs, and including compile time would
measure the compiler rather than the kernel.

Runs unchanged on Modal and on Kaggle: it imports nothing but jax, numpy and
`sim_core`, and takes the device as given.
"""

from __future__ import annotations

import json
import time


def run_pair(batches=(64, 65), repeats=3, Rd=18.30, L=5, n_temps=40,
             n_therm=1000, n_meas=500, seed=11, out=None, label=""):
    """Time each batch `repeats` times, interleaved, and report the contrast."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    import sim_core as S

    def v2(n):
        k = 0
        while n % 2 == 0:
            n //= 2
            k += 1
        return k

    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    eng = S.build_engine(geo)
    N = int(geo["N"])
    n_active = int(np.asarray(geo["disco"]).sum())
    kB = 0.086173404
    betas = jnp.asarray(1.0 / (kB * np.linspace(20.0, 0.1, int(n_temps))))
    sweeps = int(n_temps) * (int(n_therm) + int(n_meas))

    # One theta broadcast to every replica. `sim_core` wants per-chain columns,
    # so each parameter is a (B, 1, 1, 1) array rather than a scalar.
    theta = dict(Jex=1.0, Kan1=0.0012, Kan2=0.0, KanS=0.0012,
                 Hex=0.0, KDM=0.0, g=float(np.deg2rad(90.0)))

    dev = jax.devices()[0]
    print(f"device: {dev.device_kind}  ({dev.platform})", flush=True)
    print(f"lattice: Rd={Rd} L={L} N={N} n_active={n_active}  "
          f"stride={12 * L * N * N:,} B/replica", flush=True)
    print(f"schedule: {n_temps} temperatures x {n_therm + n_meas} sweeps "
          f"= {sweeps:,} sweeps per cell, compilation excluded", flush=True)
    for B in batches:
        ve = v2(B) + v2(int(L))
        print(f"  B={B:4d}  v2({B})={v2(B)} + v2(L={L})={v2(int(L))} -> "
              f"v2(elements)={ve}  width={min(4, 2 ** ve)}"
              f"{'   <-- predicted penalty' if ve >= 2 else ''}", flush=True)

    def body(sp, kk, p):
        return eng["anneal"](sp, kk, betas, p, int(n_therm), int(n_meas))

    built = {}
    for B in batches:
        cols = {k: jnp.asarray(np.full((B, 1, 1, 1), v, dtype=np.float32))
                for k, v in theta.items()}
        args = (S.init_spins(jax.random.PRNGKey(seed), geo, B),
                jax.random.PRNGKey(seed + 1), cols)
        built[B] = (jax.jit(body).lower(*args).compile(), args)
        print(f"  compiled B={B}", flush=True)

    # Interleaved: one repeat of every batch, then the next repeat.
    samples = {B: [] for B in batches}
    for r in range(int(repeats)):
        for B in batches:
            fn, args = built[B]
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*args))
            dt = time.perf_counter() - t0
            samples[B].append(dt)
            print(f"  round {r + 1}/{repeats}  B={B:4d}  {dt:7.2f} s  "
                  f"{dt * 1e6 / (sweeps * B):7.3f} us/replica-sweep", flush=True)

    rows = []
    for B in batches:
        a = np.array(samples[B])
        us = a * 1e6 / (sweeps * B)
        rows.append(dict(B=int(B), v2_elements=v2(B) + v2(int(L)),
                         width=min(4, 2 ** (v2(B) + v2(int(L)))),
                         wall_s=[float(x) for x in a],
                         us_per_replica_sweep=[float(x) for x in us],
                         us_mean=float(us.mean()), us_sd=float(us.std()),
                         us_min=float(us.min()), us_max=float(us.max())))

    wide = [r for r in rows if r["width"] == 4]
    narrow = [r for r in rows if r["width"] < 4]
    pen = overlap = None
    if wide and narrow:
        w = float(np.mean([r["us_mean"] for r in wide]))
        n = float(np.mean([r["us_mean"] for r in narrow]))
        pen = w / n - 1
        overlap = min(min(r["us_per_replica_sweep"]) for r in wide) <= \
            max(max(r["us_per_replica_sweep"]) for r in narrow)

    print("\n=== result ===", flush=True)
    for r in rows:
        print(f"  B={r['B']:4d} width {r['width']}  "
              f"{r['us_mean']:7.3f} +/- {r['us_sd']:.3f} us  "
              f"(min {r['us_min']:.3f}, max {r['us_max']:.3f})", flush=True)
    if pen is not None:
        print(f"  => width-4 penalty {pen:+.1%}   "
              f"overlap between the two samples: "
              f"{'YES' if overlap else 'NONE'}", flush=True)

    res = dict(label=label, rows=rows, penalty=pen, overlap=overlap,
               device=dev.device_kind, platform=dev.platform,
               Rd=float(Rd), L=int(L), N=N, n_active=n_active,
               stride_bytes=12 * int(L) * N * N, sweeps_per_cell=sweeps,
               n_temps=int(n_temps), n_therm=int(n_therm), n_meas=int(n_meas),
               repeats=int(repeats), seed=int(seed), interleaved=True,
               aot_compiled=True, jax_version=jax.__version__)
    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"\nwrote {out}", flush=True)
    return res
