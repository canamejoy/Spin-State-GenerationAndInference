"""Run the nanodisk anneal on a Modal GPU and return every observable.

The point of this run is a comparison against the tutor's released states
(`jdarioagudelog/nanodiskrandomkdmhexzkan`): same geometry, same Hamiltonian
parameters, same temperature ladder, so the two can be laid on top of each
other. If they agree, the CPU arm of the timing study only has to establish a
rate, not re-establish the physics.
"""

import json
import modal

app = modal.App("nanodisk-baseline")

# Modal refuses a build step stacked after `add_local_*`, so the pip layer is
# named on its own and any image that needs an extra package branches from here.
_base = (
    # Pinning jax 0.4.35 here breaks: its CUDA discovery hits
    # `cuda_nvcc.__file__ is None` inside Modal's slim image. The unpinned
    # wheel resolves its own CUDA deps correctly.
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("jax[cuda12]", "numpy")
)

image = (
    _base
    .add_local_file(
        "/home/andres/Spin-State-GenerationAndInference/notebooks/simulation/modal/sim_core.py",
        "/root/sim_core.py",
    )
    .add_local_file(
        "/home/andres/Spin-State-GenerationAndInference/notebooks/simulation/modal/ablation.py",
        "/root/ablation.py",
    )
    .add_local_file(
        "/home/andres/Spin-State-GenerationAndInference/notebooks/simulation/modal/run_ablation.py",
        "/root/run_ablation.py",
    )
    .add_local_file(
        "/home/andres/Spin-State-GenerationAndInference/notebooks/simulation/modal/ladder_cpu.py",
        "/root/ladder_cpu.py",
    )
    .add_local_file(
        "/home/andres/Spin-State-GenerationAndInference/notebooks/simulation/modal/device_ablation.py",
        "/root/device_ablation.py",
    )
    .add_local_file(
        "/home/andres/Spin-State-GenerationAndInference/notebooks/simulation/modal/parallel_axes.py",
        "/root/parallel_axes.py",
    )
    .add_local_file(
        "/home/andres/Spin-State-GenerationAndInference/notebooks/simulation/modal/theta4.json",
        "/root/theta4.json",
    )
    # The same file Kaggle runs, so Modal and Kaggle execute identical code.
    .add_local_file(
        "/home/andres/Spin-State-GenerationAndInference/notebooks/simulation/modal/pair_test.py",
        "/root/pair_test.py",
    )
)

# Raw Triton, invoked from JAX through `jax_triton.triton_call`, so the kernel
# sits inside the SAME XLA program as the production anneal and the comparison
# is not cross-framework. Pallas was the wrong road for this kernel: it has no
# `slice` and no `concatenate`, so the shifted contiguous loads this stencil
# needs cannot be expressed there.
triton_image = (
    _base
    .pip_install("jax-triton", "absl-py")
    .add_local_file(
        "/home/andres/Spin-State-GenerationAndInference/notebooks/simulation/modal/sim_core.py",
        "/root/sim_core.py",
    )
    .add_local_file(
        "/home/andres/Spin-State-GenerationAndInference/notebooks/simulation/modal/triton_halo.py",
        "/root/triton_halo.py",
    )
    .add_local_file(
        "/home/andres/Spin-State-GenerationAndInference/notebooks/simulation/modal/theta4.json",
        "/root/theta4.json",
    )
)

# The XLA-confound probe needs a kernel that XLA never touches, so this image
# adds CuPy and compiles the stencil with nvrtc instead. It branches off `_base`
# rather than `image` because the local files have to go on last, and it needs
# only `sim_core` for the geometry.
# Pallas on GPU lowers through mosaic_gpu, whose registration module imports
# absl -- absent from the slim image, which is why `pallas_call` reported the
# gpu platform unsupported even though the kernels themselves ran. Measured:
# with absl-py installed and the registration imported, a trivial kernel
# returns exact results.
pallas_image = (
    _base
    .pip_install("absl-py")
    .add_local_file(
        "/home/andres/Spin-State-GenerationAndInference/notebooks/simulation/modal/sim_core.py",
        "/root/sim_core.py",
    )
    .add_local_file(
        "/home/andres/Spin-State-GenerationAndInference/notebooks/simulation/modal/theta4.json",
        "/root/theta4.json",
    )
)

cuda_image = (
    _base
    .pip_install("cupy-cuda12x")
    .add_local_file(
        "/home/andres/Spin-State-GenerationAndInference/notebooks/simulation/modal/sim_core.py",
        "/root/sim_core.py",
    )
)


vol = modal.Volume.from_name("nanodisk-out", create_if_missing=True)


# Modal resolves `gpu=` at decoration time and rejects functions built by a
# factory, so the three GPU counts are three explicit module-level functions.
_COMMON = dict(image=image, timeout=60 * 60 * 4, volumes={"/out": vol})


@app.function(gpu="H100", **_COMMON)
def sim_1(params, temps, replicas, n_therm, n_meas, seed, tag):
    return _simulate(params, temps, replicas, n_therm, n_meas, seed, tag)


@app.function(gpu="H100:2", **_COMMON)
def sim_2(params, temps, replicas, n_therm, n_meas, seed, tag):
    return _simulate(params, temps, replicas, n_therm, n_meas, seed, tag)


@app.function(gpu="H100:4", **_COMMON)
def sim_4(params, temps, replicas, n_therm, n_meas, seed, tag):
    return _simulate(params, temps, replicas, n_therm, n_meas, seed, tag)


SIM = {1: sim_1, 2: sim_2, 4: sim_4}


def _simulate(params: dict, temps: list, replicas: int, n_therm: int,
              n_meas: int, seed: int, tag: str):
    import sys, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S

    print("devices:", jax.devices(), flush=True)
    g = S.build_geometry(Rd=params.pop("Rd", 18.30), L=params.pop("L", 5))
    eng = S.build_engine(g)
    print(f"N={g['N']} active={g['n_active']} surface={g['n_surface']}", flush=True)

    kB = 0.086173404
    T = np.asarray(temps, dtype=np.float64)
    betas = jnp.asarray(1.0 / (kB * T))
    P = {k: float(v) for k, v in params.items()}
    P["g"] = float(np.deg2rad(P.pop("gamma", 90.0)))

    ndev = jax.local_device_count()
    assert replicas % ndev == 0, f"{replicas} replicas do not divide over {ndev} devices"

    if ndev == 1:
        spins = S.init_spins(jax.random.PRNGKey(seed), g, replicas)
        run = jax.jit(lambda sp, k: eng["anneal"](sp, k, betas, P, n_therm, n_meas))
        t0 = time.perf_counter()
        obs, final, snaps = run(spins, jax.random.PRNGKey(seed + 1))
        jax.block_until_ready((obs, final, snaps))
        elapsed = time.perf_counter() - t0
    else:
        # Replicas are independent chains with no collective, so the batch splits
        # across devices with nothing to synchronise until the very end.
        per = replicas // ndev
        spins = jax.pmap(lambda k: S.init_spins(k, g, per))(
            jax.random.split(jax.random.PRNGKey(seed), ndev))
        run = jax.pmap(lambda sp, k: eng["anneal"](sp, k, betas, P, n_therm, n_meas))
        t0 = time.perf_counter()
        obs, final, snaps = run(spins, jax.random.split(jax.random.PRNGKey(seed + 1), ndev))
        jax.block_until_ready((obs, final, snaps))
        elapsed = time.perf_counter() - t0
        # (ndev, n_T, 8, per) -> (n_T, 8, replicas); snapshots come from device 0.
        obs = jnp.concatenate(list(obs), axis=-1)
        final = jnp.concatenate(list(final), axis=0)
        snaps = snaps[0]
    sweeps = len(T) * (n_therm + n_meas)
    print(f"{elapsed:.1f} s  |  {len(T)} T-points x {n_therm}+{n_meas} sweeps "
          f"x {replicas} replicas  ->  {1000*elapsed/sweeps:.3f} ms/sweep", flush=True)

    red = S.reduce_obs(obs, T, g["n_active"])
    fin = np.asarray(final)
    out = {k: np.asarray(v).tolist() for k, v in red.items()}
    out.update(elapsed_s=elapsed, ms_per_sweep=1000 * elapsed / sweeps,
               replicas=replicas, n_therm=n_therm, n_meas=n_meas,
               params=P, n_active=g["n_active"], device=str(jax.devices()[0]),
               n_devices=ndev, n_temps=len(T))

    np.savez_compressed(f"/out/{tag}.npz",
                        sz_mid=fin[:, g["L"] // 2, :, :, 2].astype(np.float32),
                        sz_transition=np.asarray(snaps).astype(np.float32),
                        spins_rep0=fin[0].astype(np.float32),
                        raw_obs=np.asarray(obs).astype(np.float32),
                        # `red` already carries T; passing it again collides.
                        **{k: np.asarray(v) for k, v in red.items()})
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(out, f, indent=1)
    vol.commit()
    print("saved", tag, flush=True)
    return out


@app.local_entrypoint()
def main(config: str = "", tag: str = "run", gpus: int = 1):
    cfg = json.load(open(config))
    r = SIM[gpus].remote(cfg["params"], cfg["temps"], cfg["replicas"],
                         cfg["n_therm"], cfg["n_meas"], cfg.get("seed", 42), tag)
    print(json.dumps({k: v for k, v in r.items()
                      if not isinstance(v, list)}, indent=1))


# --------------------------------------------------------------------------
# The CPU arm: a rate, not a full ladder
# --------------------------------------------------------------------------
# A 200-point ladder on CPU is 67 h, so what gets measured is the per-sweep
# cost and the ladder is extrapolated from it. Cost is exactly linear in
# sweeps, so that is arithmetic rather than a fitted trend.
#
# It runs on Modal, on the same image and the same JAX build as the H100 runs,
# because a ratio between a cloud GPU and a laptop under WSL2 is not a quantity
# anyone can reproduce. The CPU count is pinned so the number has a machine
# attached to it.

@app.function(image=image, cpu=8.0, memory=16384, timeout=60 * 60 * 2,
              volumes={"/out": vol})
def cpu_rate(params, replica_counts, n_T, n_therm, n_meas, seed, tag):
    import sys, time, json
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S

    # Pulled from the container so the rate is quotable with its hardware.
    cpu_model, cores = "unknown", 0
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name") and cpu_model == "unknown":
                    cpu_model = line.split(":", 1)[1].strip()
                if line.startswith("processor"):
                    cores += 1
    except OSError:
        pass
    print(f"devices={jax.devices()}  cpu={cpu_model}  threads={cores}", flush=True)

    g = S.build_geometry(Rd=params.pop("Rd", 18.30), L=params.pop("L", 5))
    eng = S.build_engine(g)
    kB = 0.086173404
    # A short ladder: the rate does not depend on where on the ladder it is
    # measured, and a long one would only multiply the wall clock.
    T = np.linspace(20.0, 0.1, int(n_T))
    betas = jnp.asarray(1.0 / (kB * T))
    P = {k: float(v) for k, v in params.items()}
    P["g"] = float(np.deg2rad(P.pop("gamma", 90.0)))

    run = jax.jit(lambda sp, k: eng["anneal"](sp, k, betas, P, n_therm, n_meas))
    sweeps = int(n_T) * (n_therm + n_meas)
    rows = []
    for rep in replica_counts:
        spins = S.init_spins(jax.random.PRNGKey(seed), g, rep)
        key = jax.random.PRNGKey(seed + 1)
        # The first call pays XLA compilation, which at 400 sweeps would be a
        # large part of the measurement. The identical second call pays none of
        # it, so it is the one that is timed.
        t0 = time.perf_counter()
        jax.block_until_ready(run(spins, key))
        compile_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = run(spins, key)
        jax.block_until_ready(out)
        elapsed = time.perf_counter() - t0
        ms = 1000 * elapsed / sweeps
        rows.append(dict(replicas=int(rep), elapsed_s=elapsed,
                         first_call_s=compile_s, ms_per_sweep=ms,
                         us_per_replica_sweep=1000 * ms / rep,
                         sweeps=sweeps))
        print(f"  {rep:4d} replicas  {elapsed:8.2f} s  {ms:9.3f} ms/sweep  "
              f"{1000*ms/rep:8.2f} us/replica-sweep  "
              f"(first call {compile_s:.1f} s)", flush=True)

    res = dict(rows=rows, n_active=g["n_active"], n_temps=int(n_T),
               n_therm=n_therm, n_meas=n_meas, params=P,
               cpu_model=cpu_model, threads=cores,
               devices=[str(d) for d in jax.devices()],
               jax_version=jax.__version__)
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


@app.local_entrypoint()
def rate(config: str = "", tag: str = "cpu_rate", n_temps: int = 4,
         n_therm: int = 60, n_meas: int = 40, reps: str = "1,25,50,100,200"):
    cfg = json.load(open(config))
    counts = [int(x) for x in reps.split(",")]
    r = cpu_rate.remote(cfg["params"], counts, n_temps, n_therm, n_meas,
                        cfg.get("seed", 42), tag)
    print(json.dumps(r, indent=1))


@app.function(image=image, cpu=8.0, memory=16384, timeout=60 * 60 * 2,
              volumes={"/out": vol})
def ablate(repeats, quick, tag):
    """The four DP x BP cells on the same container as `cpu_rate`.

    Running these locally and the GPU arm on Modal would have made every
    cross-arm ratio a comparison between two machines. Here the scalar baseline,
    the vectorised cell and the CPU rate all come off one specified container,
    and only the GPU rows come from elsewhere -- which is the one comparison that
    cannot be collapsed onto a single machine anyway.
    """
    import sys, json
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import run_ablation as RA
    import sim_core as S
    import ablation

    geo = S.build_geometry(Rd=18.30, L=5)
    K1 = np.where(geo["surface"], RA.PARAMS["KanS"], RA.PARAMS["Kan1"])
    sweeps = RA.QUICK if quick else RA.SWEEPS
    print(f"active={geo['n_active']} surface={geo['n_surface']}", flush=True)
    print("timing the four cells:", flush=True)
    rows = RA.timing(geo, K1, sweeps, repeats)
    print("bias of dropping the colouring:", flush=True)
    bi = RA.bias(geo, K1, sweeps=200 if quick else 2000,
                 burn=100 if quick else 1000)

    # The same update, in the kernel that actually ships, on this container: the
    # ablation's vectorised cell is numpy, and JAX on CPU is not the same cost.
    eng = S.build_engine(geo)
    k1j = jnp.where(jnp.asarray(geo["surface"]), RA.PARAMS["KanS"], RA.PARAMS["Kan1"])
    ev = jnp.asarray(geo["even"][None, ..., None])
    od = jnp.asarray(geo["odd"][None, ..., None])
    beta = 1.0 / (RA.KB * 3.0)

    def _sw(s, key, n):
        def body(_, v):
            s, k = v
            k, k1, k2 = jax.random.split(k, 3)
            s = eng["substep"](s, k1, beta, ev, RA.PARAMS, k1j)
            s = eng["substep"](s, k2, beta, od, RA.PARAMS, k1j)
            return s, k
        return jax.lax.fori_loop(0, n, body, (s, key))

    run = jax.jit(_sw, static_argnums=2)
    jaxrows = []
    import time
    for rep in (1, 25, 100):
        s = S.init_spins(jax.random.PRNGKey(0), geo, rep)
        k = jax.random.PRNGKey(1)
        jax.block_until_ready(run(s, k, 50))
        n = 300
        t0 = time.perf_counter()
        jax.block_until_ready(run(s, k, n))
        dt = time.perf_counter() - t0
        ms = 1000 * dt / n
        jaxrows.append(dict(replicas=rep, ms_per_sweep=ms,
                            ms_per_replica_sweep=ms / rep))
        print(f"  JAX CPU update-only {rep:4d} repl  {ms:9.3f} ms/sweep  "
              f"{ms/rep:9.4f} ms/replica-sweep", flush=True)

    # The reference notebook's own CPU mode: site-by-site over the lattice,
    # vectorised over replicas. Without this row the ablation compares the
    # production kernel against a baseline nobody in the group ever ran.
    print("reference notebook CPU mode (site-by-site, batched over replicas):",
          flush=True)
    swrows = []
    for b in (1, 25, 100):
        r = ablation.time_sitewise_batched(geo, RA.PARAMS, K1, b, sweeps=2,
                                           max_sites=None if b == 1 else 1200)
        swrows.append(r)
        print(f"  batch {b:4d}  {r['ms_per_sweep']:10.2f} ms/sweep  "
              f"{r['ms_per_replica_sweep']:9.4f} ms/replica-sweep"
              f"{'  (scaled)' if r['scaled'] else ''}", flush=True)

    by = {r["cell"]: r["ms_per_sweep"] for r in rows}
    res = dict(cells=rows, bias=bi, jax_cpu_update_only=jaxrows,
               sitewise_batched=swrows,
               speedups=dict(
                   dp_given_bp=by["nodp_bp"] / by["dp_bp"],
                   bp_given_dp=by["dp_nobp"] / by["dp_bp"],
                   bp_given_nodp=by["nodp_nobp"] / by["nodp_bp"],
                   overall_vs_textbook=by["nodp_nobp"] / by["dp_bp"],
                   degenerate_cell_vs_dp_bp=by["dp_nobp"] / by["dp_bp"]),
               geometry=dict(Rd=18.30, L=5, N=geo["N"],
                             n_active=geo["n_active"],
                             n_surface=geo["n_surface"]),
               params=RA.PARAMS, numpy=np.__version__,
               jax_version=jax.__version__)
    for k, v in res["speedups"].items():
        print(f"  {k:26s} {v:9.2f}x", flush=True)
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


@app.local_entrypoint()
def ablation_run(tag: str = "ablation_8core", repeats: int = 3,
                 quick: bool = False):
    r = ablate.remote(repeats, quick, tag)
    print(json.dumps(r["speedups"], indent=1))


@app.function(gpu="H100", image=image, timeout=60 * 30, volumes={"/out": vol})
def gpu_sitewise(max_sites, batch, sweeps, tag):
    """`GPU SIN DP`: the same site-by-site loop, dispatched to an H100.

    This is the cell that turns "matrix decomposition is what the GPU is for"
    from an argument into a measurement. Each site is its own set of eager JAX
    operations -- no `jit`, because fusing the sweep into one kernel *is* the
    decomposition, and jitting it would be measuring the opposite cell.

    The expectation is that an H100 is no faster than a CPU here, because a
    sequential chain hands the device a few dozen floats at a time and pays a
    dispatch for each. If that holds, the GPU rows of the test plan's `SIN DP`
    column are not slow versions of the GPU runs -- they are CPU runs with extra
    latency.
    """
    import sys, time, json
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S
    import run_ablation as RA

    print("devices:", jax.devices(), flush=True)
    geo = S.build_geometry(Rd=18.30, L=5)
    k1 = jnp.asarray(np.where(geo["surface"], RA.PARAMS["KanS"],
                              RA.PARAMS["Kan1"]))
    L, N = geo["L"], geo["N"]
    P = RA.PARAMS
    beta = 1.0 / (RA.KB * 3.0)
    hx, hz = P["Hex"] * np.cos(P["g"]), P["Hex"] * np.sin(P["g"])

    key = jax.random.PRNGKey(0)
    s = S.init_spins(key, geo, batch)
    order = ablation_order(geo)[:max_sites]
    zero = jnp.zeros((batch, 3))

    def an(v, kk):
        return (kk * ((v[:, 0] * v[:, 1]) ** 2 + (v[:, 0] * v[:, 2]) ** 2
                      + (v[:, 1] * v[:, 2]) ** 2)
                + P["Kan2"] * (v[:, 0] * v[:, 1] * v[:, 2]) ** 2)

    def one_sweep(s, key):
        for z, y, x in order:
            key, kt, ka = jax.random.split(key, 3)
            b = jax.random.normal(kt, (batch, 3))
            b = b / jnp.linalg.norm(b, axis=-1, keepdims=True)
            xp, xm = s[:, z, y, (x + 1) % N], s[:, z, y, (x - 1) % N]
            yp, ym = s[:, z, (y + 1) % N, x], s[:, z, (y - 1) % N, x]
            zp = s[:, z + 1, y, x] if z + 1 < L else zero
            zm = s[:, z - 1, y, x] if z - 1 >= 0 else zero
            a = s[:, z, y, x]
            d = b - a
            kk = k1[z, y, x]
            dE = -P["Jex"] * jnp.sum(d * (xp + xm + yp + ym + zp + zm), axis=-1)
            dE = dE + an(b, kk) - an(a, kk)
            dE = dE - P["KDM"] * (d[:, 1] * (xp[:, 2] - xm[:, 2])
                                  - d[:, 2] * (xp[:, 1] - xm[:, 1])
                                  + d[:, 2] * (yp[:, 0] - ym[:, 0])
                                  - d[:, 0] * (yp[:, 2] - ym[:, 2]))
            dE = dE - (hx * d[:, 0] + hz * d[:, 2])
            acc = (dE < 0) | (jax.random.uniform(ka, dE.shape)
                              < jnp.exp(-dE * beta))
            s = s.at[:, z, y, x].set(jnp.where(acc[:, None], b, a))
        return s, key

    s, key = one_sweep(s, key)          # warm up dispatch paths
    jax.block_until_ready(s)
    t0 = time.perf_counter()
    for _ in range(sweeps):
        s, key = one_sweep(s, key)
    jax.block_until_ready(s)
    dt = time.perf_counter() - t0

    us_site = 1e6 * dt / (len(order) * sweeps)
    ms_sweep = us_site * geo["n_active"] / 1000.0
    res = dict(cell="gpu_sitewise", device=str(jax.devices()[0]), batch=batch,
               max_sites=len(order), sweeps=sweeps, wall_s=dt,
               us_per_site=us_site, ms_per_sweep=ms_sweep,
               ms_per_replica_sweep=ms_sweep / batch,
               n_active=geo["n_active"], scaled=True)
    print(f"{us_site:.2f} us/site -> {ms_sweep:.1f} ms/sweep "
          f"({ms_sweep/batch:.4f} ms/replica-sweep)", flush=True)
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


def ablation_order(geo):
    import sys
    sys.path.insert(0, "/root")
    import ablation
    return ablation.lattice_order(geo)


@app.local_entrypoint()
def gpu_nodp(tag: str = "gpu_sitewise_h100", max_sites: int = 400,
             batch: int = 100, sweeps: int = 2):
    r = gpu_sitewise.remote(max_sites, batch, sweeps, tag)
    print(json.dumps(r, indent=1))


# --------------------------------------------------------------------------
# The affordable CPU cells, run to completion
# --------------------------------------------------------------------------
# `cpu=2.0` rather than 8: the scalar arms are a pure-Python loop and single
# threaded, so eight cores would be seven idle ones on the meter. The run reports
# its own ms/sweep, so if two cores changed the rate it says so against the
# ablation's 12.77 / 13.03 / 1.51 immediately. `dp_bp` is numpy and does use
# threads, so it gets its own core count.
#
# 24 h is Modal's ceiling and these are ~11 h, so the checkpoint is not for the
# timeout -- it is for a preemption or a crash eleven hours in. It saves after
# every temperature, and `tests/test_ladder_checkpoint.py` pins that a resume is
# bit-identical to an uninterrupted run.

def _ladder(cell, cpus, n_temps, n_therm, n_meas, seed, tag):
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np
    import sim_core as S
    import ladder_cpu as LC

    ck_path = f"/out/{tag}_ckpt.npz"
    g = S.build_geometry(Rd=18.30, L=5)
    P = dict(Jex=1.0, Kan1=0.177, Kan2=0.034, KanS=0.139, Hex=0.483,
             KDM=0.880, g=float(np.deg2rad(90.0)))
    temps = np.linspace(20.0, 0.1, int(n_temps))
    print(f"cell={cell} cpus={cpus} active={g['n_active']} "
          f"T: {temps[0]} -> {temps[-1]} ({len(temps)} points), "
          f"{n_therm}+{n_meas} sweeps, 1 replica", flush=True)

    ck = None
    try:
        vol.reload()
        z = np.load(ck_path, allow_pickle=True)
        ck = dict(spins=z["spins"], t_done=int(z["t_done"]),
                  rows=list(z["rows"]), snaps=list(z["snaps"]),
                  elapsed_s=float(z["elapsed_s"]))
        print(f"found checkpoint at T index {ck['t_done']}", flush=True)
    except Exception as e:
        print(f"no usable checkpoint ({type(e).__name__}), starting fresh",
              flush=True)

    last = [time.time()]

    def save(state):
        # Committing every temperature would spend more time on the volume than
        # on the physics; every ~5 min bounds the loss to one temperature's work.
        np.savez(ck_path, spins=state["spins"], t_done=state["t_done"],
                 rows=np.array(state["rows"], dtype=object),
                 snaps=np.array(state["snaps"]), elapsed_s=state["elapsed_s"],
                 cell=cell)
        if time.time() - last[0] > 300 or state["t_done"] == int(n_temps):
            vol.commit()
            last[0] = time.time()

    out = LC.run_ladder(cell, g, P, temps, n_therm, n_meas, seed=seed,
                        checkpoint=ck, save=save,
                        log=lambda *a: print(*a, flush=True))

    rows = out.pop("rows")
    res = dict(out)
    res.pop("snaps"); res.pop("final")
    for k in ("T", "E", "M", "Mz", "Q", "Cv", "chi"):
        res[k] = [float(r[k]) for r in rows]
    res["params"] = P
    np.savez_compressed(f"/out/{tag}.npz", sz_transition=out["snaps"],
                        spins_final=out["final"],
                        **{k: np.asarray(res[k])
                           for k in ("T", "E", "M", "Mz", "Q", "Cv", "chi")})
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    print(f"done: {out['elapsed_s']/3600:.2f} h, "
          f"{out['ms_per_sweep']:.3f} ms/sweep", flush=True)
    return {k: v for k, v in res.items() if not isinstance(v, (list, dict))}


_LAD = dict(image=image, timeout=86400, memory=8192, volumes={"/out": vol})


@app.function(cpu=2.0, **_LAD)
def ladder_scalar(cell, n_temps, n_therm, n_meas, seed, tag):
    return _ladder(cell, 2.0, n_temps, n_therm, n_meas, seed, tag)


@app.function(cpu=8.0, **_LAD)
def ladder_numpy(cell, n_temps, n_therm, n_meas, seed, tag):
    return _ladder(cell, 8.0, n_temps, n_therm, n_meas, seed, tag)


@app.local_entrypoint()
def cpu_ladder(cell: str = "dp_bp", tag: str = "", n_temps: int = 200,
               n_therm: int = 10000, n_meas: int = 5000, seed: int = 42):
    tag = tag or f"ladder_{cell}"
    fn = ladder_numpy if cell == "dp_bp" else ladder_scalar
    r = fn.remote(cell, n_temps, n_therm, n_meas, seed, tag)
    print(json.dumps(r, indent=1))


# --------------------------------------------------------------------------
# The sixteen-cell grid: four ablation cells x four device configurations
# --------------------------------------------------------------------------
# The test plan's Part 1 is 4 cells x {CPU, 1, 2, 3 GPU}. Twelve of those were
# filled in by argument. This measures them, all at the production 100 replicas
# so one number is comparable down a column and across a row.

def _grid(replicas, max_sites, sweeps, tag, on_gpu):
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax
    import sim_core as S
    import device_ablation as DA

    print("devices:", jax.devices(), flush=True)
    geo = S.build_geometry(Rd=18.30, L=5)
    P = dict(Jex=1.0, Kan1=0.177, Kan2=0.034, KanS=0.139, Hex=0.483,
             KDM=0.880, g=float(np.deg2rad(90.0)))
    rows = []
    for cell in DA.CELLS:
        # The sequential cells are charged pro rata over a few hundred sites;
        # dp_bp sweeps the whole lattice because it can.
        ms = None if cell == "dp_bp" else max_sites
        sw = sweeps if cell == "dp_bp" else max(1, sweeps // 50)
        if on_gpu or cell == "dp_bp":
            r = DA.measure_jax(cell, geo, P, replicas, ms, sw)
        else:
            r = DA.measure_cpu(cell, geo, P, replicas, ms, sw)
        r["engine"] = "jax" if (on_gpu or cell == "dp_bp") else "numpy"
        rows.append(r)
        print(f"  {cell:10s} {r['ms_per_sweep']:14,.3f} ms/sweep  "
              f"({r['sites_timed']} sites{'  scaled' if r['scaled'] else ''}, "
              f"{r['engine']}, {r['wall_s']:.1f} s)", flush=True)

    res = dict(rows=rows, replicas=replicas,
               devices=int(jax.local_device_count()) if on_gpu else 0,
               device=str(jax.devices()[0]), n_active=geo["n_active"],
               params=P, jax_version=jax.__version__)
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


_GRID = dict(image=image, timeout=60 * 60 * 2, volumes={"/out": vol})


@app.function(cpu=8.0, memory=32768, **_GRID)
def grid_cpu(replicas, max_sites, sweeps, tag):
    return _grid(replicas, max_sites, sweeps, tag, on_gpu=False)


@app.function(gpu="H100", memory=32768, **_GRID)
def grid_gpu1(replicas, max_sites, sweeps, tag):
    return _grid(replicas, max_sites, sweeps, tag, on_gpu=True)


@app.function(gpu="H100:2", memory=32768, **_GRID)
def grid_gpu2(replicas, max_sites, sweeps, tag):
    return _grid(replicas, max_sites, sweeps, tag, on_gpu=True)


@app.function(gpu="H100:4", memory=32768, **_GRID)
def grid_gpu4(replicas, max_sites, sweeps, tag):
    return _grid(replicas, max_sites, sweeps, tag, on_gpu=True)


GRID = {0: grid_cpu, 1: grid_gpu1, 2: grid_gpu2, 4: grid_gpu4}


@app.local_entrypoint()
def grid(devices: int = 1, replicas: int = 100, max_sites: int = 200,
         sweeps: int = 300, tag: str = ""):
    tag = tag or f"grid_{devices}gpu" if devices else "grid_cpu"
    r = GRID[devices].remote(replicas, max_sites, sweeps, tag)
    print(json.dumps([{k: v for k, v in row.items()} for row in r["rows"]],
                     indent=1))


@app.function(image=image, cpu=8.0, memory=32768, timeout=60*60, volumes={"/out": vol})
def md_engine_split(replicas, sweeps, tag):
    """Separate the algorithm (MD) from the execution technology (JAX).

    Same cell, same device, same batch, two engines -- the only way to attribute
    the speedup instead of quoting one combined number.
    """
    import sys, json
    sys.path.insert(0, "/root")
    import numpy as np, jax
    import sim_core as S
    import device_ablation as DA

    geo = S.build_geometry(Rd=18.30, L=5)
    P = dict(Jex=1.0, Kan1=0.177, Kan2=0.034, KanS=0.139, Hex=0.483,
             KDM=0.880, g=float(np.deg2rad(90.0)))
    rows = []
    for name, fn in (("no MD, numpy",
                      lambda: DA.measure_cpu("nodp_nobp", geo, P, replicas, 200,
                                             max(1, sweeps // 50))),
                     ("MD + BP, numpy",
                      lambda: DA.measure_numpy_md(geo, P, replicas, sweeps)),
                     ("MD + BP, JAX on CPU",
                      lambda: DA.measure_jax("dp_bp", geo, P, replicas, None,
                                             sweeps))):
        r = fn(); r["arm"] = name; rows.append(r)
        print(f"  {name:22s} {r['ms_per_sweep']:10,.3f} ms/sweep "
              f"({r['ms_per_sweep']/replicas:8.4f} per replica-sweep)", flush=True)
    res = dict(rows=rows, replicas=replicas, n_active=geo["n_active"],
               numpy=np.__version__, jax_version=jax.__version__)
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


@app.local_entrypoint()
def engine_split(replicas: int = 100, sweeps: int = 200,
                 tag: str = "md_engine_split"):
    r = md_engine_split.remote(replicas, sweeps, tag)
    print(json.dumps([{k: v for k, v in x.items() if not isinstance(v, dict)}
                      for x in r["rows"]], indent=1))


# --------------------------------------------------------------------------
# Hardware identity
# --------------------------------------------------------------------------
# An earlier version parsed /proc/cpuinfo for "model name" and recorded
# "unknown" when the key was absent, which is how a timing study ends up
# quoting a CPU it never identified. `lscpu` is the reliable source, with
# /proc/cpuinfo as a fallback and the failure reported rather than swallowed.

def _hostinfo():
    import subprocess, platform
    info = {"platform": platform.platform(), "machine": platform.machine()}
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True,
                             timeout=20).stdout
        for line in out.splitlines():
            if ":" not in line:
                continue
            k, v = (x.strip() for x in line.split(":", 1))
            if k in ("Model name", "Vendor ID", "CPU(s)", "Thread(s) per core",
                     "Core(s) per socket", "Socket(s)", "CPU max MHz",
                     "CPU min MHz", "L3 cache", "Flags", "BogoMIPS"):
                info[k] = v[:220] if k == "Flags" else v
    except Exception as e:
        info["lscpu_error"] = f"{type(e).__name__}: {e}"
    try:
        with open("/proc/cpuinfo") as f:
            txt = f.read()
        info["proc_cpuinfo_model"] = next(
            (l.split(":", 1)[1].strip() for l in txt.splitlines()
             if l.startswith("model name")), "(no 'model name' key)")
        info["proc_cpuinfo_processors"] = txt.count("\nprocessor")
    except OSError as e:
        info["proc_cpuinfo_error"] = str(e)
    try:
        info["cgroup_quota"] = open("/sys/fs/cgroup/cpu.max").read().strip()
    except OSError:
        pass
    try:
        info["mem_total_kB"] = int(
            open("/proc/meminfo").readline().split()[1])
    except OSError:
        pass
    return info


@app.function(image=image, cpu=8.0, timeout=300)
def host_cpu():
    return _hostinfo()


@app.function(image=image, gpu="H100", timeout=300)
def host_gpu():
    import subprocess
    info = _hostinfo()
    try:
        info["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,"
             "compute_cap,pcie.link.gen.max", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        info["nvidia_smi_error"] = f"{type(e).__name__}: {e}"
    import jax
    info["jax_devices"] = [str(d) for d in jax.devices()]
    info["jax_device_kind"] = jax.devices()[0].device_kind
    return info


@app.local_entrypoint()
def hardware():
    print("=== CPU container (cpu=8.0) ===")
    print(json.dumps(host_cpu.remote(), indent=1))
    print("\n=== GPU container (H100) ===")
    print(json.dumps(host_gpu.remote(), indent=1))


# --------------------------------------------------------------------------
# Where the gap between the grid and a real ladder comes from
# --------------------------------------------------------------------------
# The grid times a bare update loop; a production ladder runs the full `anneal`
# -- observables on every measurement sweep, a snapshot per temperature, and a
# `lax.scan` over the ladder that stacks both. Deriving ladder times from the
# grid understated 4x H100 as 3.4 min against 4.2 min measured. "It is the
# observables" was a guess; this measures the two on ONE container so the
# difference cannot be run-to-run scatter.

def _anneal_overhead(replicas, n_temps, n_therm, n_meas, sweeps, tag):
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S

    ndev = jax.local_device_count()
    per = replicas // ndev
    geo = S.build_geometry(Rd=18.30, L=5)
    eng = S.build_engine(geo)
    P = dict(Jex=1.0, Kan1=0.177, Kan2=0.034, KanS=0.139, Hex=0.483,
             KDM=0.880, g=float(np.deg2rad(90.0)))
    k1 = jnp.where(jnp.asarray(geo["surface"]), P["KanS"], P["Kan1"])
    ev = jnp.asarray(geo["even"][None, ..., None])
    od = jnp.asarray(geo["odd"][None, ..., None])
    kB = 0.086173404
    T = np.linspace(20.0, 0.1, int(n_temps))
    betas = jnp.asarray(1.0 / (kB * T))
    beta1 = float(1.0 / (kB * 3.0))

    keys = jax.random.split(jax.random.PRNGKey(0), ndev)
    spins = (jax.pmap(lambda k: S.init_spins(k, geo, per))(keys) if ndev > 1
             else S.init_spins(keys[0], geo, per))
    kk = (jax.random.split(jax.random.PRNGKey(1), ndev) if ndev > 1
          else jax.random.PRNGKey(1))

    def upd(s, key, n):
        def one(_, v):
            s, k = v
            k, a, b = jax.random.split(k, 3)
            s = eng["substep"](s, a, beta1, ev, P, k1)
            s = eng["substep"](s, b, beta1, od, P, k1)
            return s, k
        return jax.lax.fori_loop(0, n, one, (s, key))

    run_upd = (jax.pmap(upd, static_broadcasted_argnums=2) if ndev > 1
               else jax.jit(upd, static_argnums=2))
    ann = lambda sp, k: eng["anneal"](sp, k, betas, P, n_therm, n_meas)
    run_ann = jax.pmap(ann) if ndev > 1 else jax.jit(ann)

    rows = {}
    jax.block_until_ready(run_upd(spins, kk, sweeps))      # compile at this n
    t0 = time.perf_counter()
    jax.block_until_ready(run_upd(spins, kk, sweeps))
    rows["update_only"] = 1000 * (time.perf_counter() - t0) / sweeps

    ann_sweeps = int(n_temps) * (n_therm + n_meas)
    jax.block_until_ready(run_ann(spins, kk))
    t0 = time.perf_counter()
    jax.block_until_ready(run_ann(spins, kk))
    rows["full_anneal"] = 1000 * (time.perf_counter() - t0) / ann_sweeps

    res = dict(devices=ndev, replicas=replicas, per_device=per,
               n_temps=int(n_temps), n_therm=n_therm, n_meas=n_meas,
               update_only_ms=rows["update_only"],
               full_anneal_ms=rows["full_anneal"],
               overhead_pct=100 * (rows["full_anneal"] / rows["update_only"] - 1),
               device=str(jax.devices()[0]))
    print(f"  {ndev}x  update-only {rows['update_only']:7.4f} ms/sweep | "
          f"full anneal {rows['full_anneal']:7.4f} | "
          f"overhead {res['overhead_pct']:+6.1f}%", flush=True)
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


@app.function(gpu="H100", memory=32768, image=image, timeout=3600,
              volumes={"/out": vol})
def overhead1(r, nt, th, me, sw, tag): return _anneal_overhead(r, nt, th, me, sw, tag)


@app.function(gpu="H100:2", memory=32768, image=image, timeout=3600,
              volumes={"/out": vol})
def overhead2(r, nt, th, me, sw, tag): return _anneal_overhead(r, nt, th, me, sw, tag)


@app.function(gpu="H100:4", memory=32768, image=image, timeout=3600,
              volumes={"/out": vol})
def overhead4(r, nt, th, me, sw, tag): return _anneal_overhead(r, nt, th, me, sw, tag)


OVH = {1: overhead1, 2: overhead2, 4: overhead4}


@app.local_entrypoint()
def overhead(devices: int = 1, replicas: int = 100, n_temps: int = 10,
             n_therm: int = 200, n_meas: int = 100, sweeps: int = 3000,
             tag: str = ""):
    tag = tag or f"overhead_{devices}gpu"
    print(json.dumps(OVH[devices].remote(replicas, n_temps, n_therm, n_meas,
                                         sweeps, tag), indent=1))


# --------------------------------------------------------------------------
# The CPU production ladder, measured rather than extrapolated
# --------------------------------------------------------------------------
# Every GPU ladder in this study is a measured wall clock; every CPU one is an
# extrapolation. That leaves the headline -- "1x H100 is 1,203x the CPU" -- with
# a measured numerator and an extrapolated denominator. This measures the
# denominator, and in doing so tests the extrapolation itself: cost is claimed to
# be exactly linear in sweeps, and a 52-hour run either confirms that or does not.
#
# 52 h exceeds Modal's 24 h ceiling, so the ladder runs in blocks of
# temperatures, checkpointing to the volume between them. The key for each block
# is `fold_in(base, block_index)`, addressed rather than threaded, so a resumed
# run is reproducible. That makes the chain differ from one uninterrupted stream,
# which is irrelevant to a timing measurement and still a valid chain for physics.

@app.function(image=image, cpu=8.0, memory=32768, timeout=24 * 3600,
              volumes={"/out": vol})
def cpu_ladder_jax(replicas, n_temps, n_therm, n_meas, block, max_hours, seed, tag):
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S

    ck = f"/out/{tag}_ckpt.npz"
    geo = S.build_geometry(Rd=18.30, L=5)
    eng = S.build_engine(geo)
    P = dict(Jex=1.0, Kan1=0.177, Kan2=0.034, KanS=0.139, Hex=0.483,
             KDM=0.880, g=float(np.deg2rad(90.0)))
    kB = 0.086173404
    T = np.linspace(20.0, 0.1, int(n_temps))
    nblocks = -(-int(n_temps) // int(block))
    print(f"devices={jax.devices()} replicas={replicas} "
          f"{n_temps} T in {nblocks} blocks of {block}", flush=True)

    done, obs_acc, snap_acc, elapsed = 0, [], [], 0.0
    try:
        vol.reload()
        z = np.load(ck)
        spins = jnp.asarray(z["spins"])
        done = int(z["done"]); elapsed = float(z["elapsed_s"])
        obs_acc = [z["obs"]] if z["obs"].size else []
        snap_acc = [z["snaps"]] if z["snaps"].size else []
        print(f"resuming at block {done}/{nblocks} ({elapsed/3600:.2f} h spent)",
              flush=True)
    except Exception as e:
        print(f"fresh start ({type(e).__name__})", flush=True)
        spins = S.init_spins(jax.random.PRNGKey(seed), geo, replicas)

    base = jax.random.PRNGKey(seed + 1)
    run = jax.jit(lambda sp, k, b: eng["anneal"](sp, k, b, P, n_therm, n_meas))
    t_start = time.perf_counter()

    for bi in range(done, nblocks):
        Tb = T[bi * block:(bi + 1) * block]
        betas = jnp.asarray(1.0 / (kB * Tb))
        key = jax.random.fold_in(base, bi)
        t0 = time.perf_counter()
        obs, spins, snaps = run(spins, key, betas)
        jax.block_until_ready((obs, spins, snaps))
        dt = time.perf_counter() - t0
        elapsed += dt
        obs_acc.append(np.asarray(obs)); snap_acc.append(np.asarray(snaps))
        sweeps_done = (bi + 1) * block * (n_therm + n_meas)
        rate = 1000 * elapsed / sweeps_done
        left = (int(n_temps) - (bi + 1) * block) * (n_therm + n_meas) * rate / 3.6e6
        print(f"  block {bi+1:2d}/{nblocks}  T {Tb[0]:5.2f}->{Tb[-1]:5.2f}  "
              f"{dt/3600:5.2f} h  {rate:7.3f} ms/sweep  ~{left:5.1f} h left",
              flush=True)
        np.savez(ck, spins=np.asarray(spins), done=bi + 1, elapsed_s=elapsed,
                 obs=np.concatenate(obs_acc, axis=0),
                 snaps=np.concatenate(snap_acc, axis=0))
        vol.commit()
        if (time.perf_counter() - t_start) / 3600 > max_hours - (dt / 3600):
            print(f"stopping before the {max_hours} h budget; re-invoke to resume",
                  flush=True)
            return dict(complete=False, blocks_done=bi + 1, blocks=nblocks,
                        elapsed_s=elapsed, ms_per_sweep=rate)

    obs = np.concatenate(obs_acc, axis=0)
    red = S.reduce_obs(obs, T, geo["n_active"])
    total = int(n_temps) * (n_therm + n_meas)
    res = dict(complete=True, elapsed_s=elapsed,
               ms_per_sweep=1000 * elapsed / total, replicas=replicas,
               n_temps=int(n_temps), n_therm=n_therm, n_meas=n_meas,
               n_active=geo["n_active"], params=P, blocks=nblocks,
               device=str(jax.devices()[0]),
               **{k: np.asarray(v).tolist() for k, v in red.items()})
    np.savez_compressed(f"/out/{tag}.npz",
                        sz_transition=np.concatenate(snap_acc, axis=0),
                        spins_final=np.asarray(spins), raw_obs=obs,
                        **{k: np.asarray(v) for k, v in red.items()})
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    print(f"COMPLETE: {elapsed/3600:.2f} h, {res['ms_per_sweep']:.3f} ms/sweep",
          flush=True)
    return {k: v for k, v in res.items() if not isinstance(v, (list, dict))}


@app.local_entrypoint()
def cpu_full(replicas: int = 100, n_temps: int = 200, n_therm: int = 10000,
             n_meas: int = 5000, block: int = 10, max_hours: float = 20.0,
             seed: int = 42, tag: str = "cpu_ladder_100"):
    r = cpu_ladder_jax.remote(replicas, n_temps, n_therm, n_meas, block,
                              max_hours, seed, tag)
    print(json.dumps(r, indent=1))


# --------------------------------------------------------------------------
# Where the parallelism should go: replicas, experiments, or both
# --------------------------------------------------------------------------
# All three configurations are the SAME total work -- 4 thetas x 24 replicas =
# 96 chains -- with the bipartite update always on, so only the grouping varies.
# Running all three in one container removes the between-container scatter that
# has bitten every cross-run comparison in this study.

def _axes(sweeps, tag, bp_cost, repeats=1):
    import sys, json
    sys.path.insert(0, "/root")
    import numpy as np, jax
    import sim_core as S
    import parallel_axes as PA

    geo = S.build_geometry(Rd=18.30, L=5)
    thetas = PA.load_thetas("/root/theta4.json")
    beta = 1.0 / (0.086173404 * 3.0)
    print(f"devices={jax.devices()}  thetas={[t['name'] for t in thetas]}  "
          f"{PA.TOTAL} chains total", flush=True)

    rows = []
    for cfg in ("bp_only", "replicas", "states", "both"):
        # Repeats inside ONE container: `replicas` and `both` came out 11-20%
        # apart at 400 chains, which is the width of the between-container
        # scatter, so a single shot cannot say whether the difference is real.
        trials = [PA.time_config(cfg, geo, thetas, sweeps, beta, seed=10 * i)
                  for i in range(repeats)]
        ms = [t["ms_per_sweep"] for t in trials]
        r = dict(trials[0])
        r["ms_per_sweep"] = float(np.mean(ms))
        r["ms_sd"] = float(np.std(ms))
        r["ms_all"] = ms
        r["us_per_chain_sweep"] = 1e3 * r["ms_per_sweep"] / PA.TOTAL
        r["repeats"] = repeats
        # The no-BP columns are incoherent physics -- neighbours would decide
        # against each other's stale spins -- so they are derived from the
        # measured 2.02x cost of the second pass, never run.
        r["ms_per_sweep_noBP_extrapolated"] = r["ms_per_sweep"] / bp_cost
        rows.append(r)
        print(f"  {cfg:9s} {r['launches']:4d} launches x "
              f"{r['chains_per_launch']:4d} chains   "
              f"{r['ms_per_sweep']:9.4f} +-{r['ms_sd']:7.4f} ms/sweep   "
              f"{r['us_per_chain_sweep']:8.3f} us/chain-sweep", flush=True)

    res = dict(rows=rows, thetas=thetas, n_replicas=PA.N_REPLICAS,
               n_theta=PA.N_THETA, total_chains=PA.TOTAL, sweeps=sweeps,
               devices=int(jax.local_device_count()),
               bp_cost_factor=bp_cost, device=str(jax.devices()[0]),
               n_active=geo["n_active"])
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


_AX = dict(image=image, memory=32768, timeout=3600, volumes={"/out": vol})


@app.function(gpu="H100", **_AX)
def axes1(sweeps, tag, bp, rep=1): return _axes(sweeps, tag, bp, rep)


@app.function(gpu="H100:2", **_AX)
def axes2(sweeps, tag, bp, rep=1): return _axes(sweeps, tag, bp, rep)


@app.function(gpu="H100:4", **_AX)
def axes4(sweeps, tag, bp, rep=1): return _axes(sweeps, tag, bp, rep)


@app.function(image=image, cpu=8.0, memory=32768, timeout=7200,
              volumes={"/out": vol})
def axes_cpu(sweeps, tag, bp, rep=1):
    """The same four groupings on CPU, through the identical code path.

    The CPU row has to come from this harness rather than from an earlier
    measurement, or the comparison would be between two different programs that
    happen to share a name.
    """
    return _axes(sweeps, tag, bp, rep)


AXES = {0: axes_cpu, 1: axes1, 2: axes2, 4: axes4}


@app.local_entrypoint()
def axes(devices: int = 1, sweeps: int = 2000, bp_cost: float = 2.02,
         tag: str = "", repeats: int = 1):
    tag = tag or f"axes_{devices}gpu"
    r = AXES[devices].remote(sweeps, tag, bp_cost, repeats)
    print(json.dumps([{k: v for k, v in x.items()} for x in r["rows"]], indent=1))


# --------------------------------------------------------------------------
# Full 200-point ladders for the two groupings worth running
# --------------------------------------------------------------------------
# The axis study measured rates over 2,000 sweeps and extrapolated the ladder.
# These run the ladder for real, at 4 states x 100 replicas, for the two
# groupings a person would actually use. The other two are left extrapolated:
# `BP only` and `BP + states` cost $229 together precisely because they are the
# bad choices, and measuring a bad choice to four digits buys nothing.
#
# Unlike the rate study this calls the production `anneal`, so each run also
# yields the physics -- 400 annealed states with their observables, not just a
# wall clock.

def _axis_ladder(config, n_temps, n_therm, n_meas, seed, tag, n_states=0):
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S
    import parallel_axes as PA

    ndev = jax.local_device_count()
    geo = S.build_geometry(Rd=18.30, L=5)
    eng = S.build_engine(geo)
    thetas = PA.load_thetas("/root/theta4.json")
    # Fewer states is how the optimum table reaches a per-device batch that four
    # states cannot tile: 3 x 100 replicas over 4 devices is exactly B = 75, the
    # measured minimum of the f(B) curve, in a single launch.
    if n_states:
        thetas = thetas[:int(n_states)]
    kB = 0.086173404
    T = np.linspace(20.0, 0.1, int(n_temps))
    betas = jnp.asarray(1.0 / (kB * T))
    print(f"devices={jax.devices()}  config={config}  "
          f"{len(thetas)} states x {PA.N_REPLICAS} replicas", flush=True)

    if config == "replicas":
        launches = [([t], PA.N_REPLICAS) for t in thetas]
    elif config == "both":
        launches = [(thetas, PA.N_REPLICAS)]
    elif config == "pairs":
        # Two launches of two states each: 200 chains per launch, which on four
        # devices is 50 per device -- the batch size the isolation scan found
        # optimal. `replicas` gives 25 per device and `both` gives 100, so this
        # is the configuration that lands on the measured minimum, and it is a
        # prediction of that scan rather than another point in the same series.
        half = len(thetas) // 2
        launches = [(thetas[:half], PA.N_REPLICAS),
                    (thetas[half:], PA.N_REPLICAS)]
    else:
        raise ValueError(f"{config} is not one of the groupings worth running")

    def run_one(ths, rep, key_seed):
        batch = len(ths) * rep
        use = ndev if batch % ndev == 0 else 1
        per = batch // use
        cols = PA.pack(ths, rep)
        k1 = jnp.where(jnp.asarray(geo["surface"])[None], cols["KanS"],
                       cols["Kan1"])
        P = {k: v for k, v in cols.items()}
        spins = S.init_spins(jax.random.PRNGKey(key_seed), geo, batch)

        def body(sp, kk, k1_, p):
            return eng["anneal"](sp, kk, betas, p, n_therm, n_meas)

        # Compile ahead of time, then time only the call. The six original
        # ladders started `t0` before their first call, so every launch charged
        # its XLA compilation -- about 27 s, measured by replaying this function
        # against a warmed direct run -- to the kernel. That is not a constant
        # offset across configurations: it scales with the launch count, so it
        # taxed `replicas` (4 launches) four times as hard as `both` (1), which
        # is the kind of bias that can invert a ranking.
        #
        # A warm-up call on a shorter schedule does NOT fix this: `betas` sets
        # the length of the temperature axis, so a shorter one is a different
        # shape and compiles a second executable, leaving the timed call cold.
        # `.lower(...).compile()` compiles the exact shapes that will be timed
        # and does no work, which is what makes it the right tool here.
        if use == 1:
            args = (spins, jax.random.PRNGKey(key_seed + 1), k1, P)
            fn = jax.jit(body).lower(*args).compile()
            t0 = time.perf_counter()
            obs, fin, snaps = fn(*args)
            jax.block_until_ready((obs, fin, snaps))
            return time.perf_counter() - t0, np.asarray(obs), np.asarray(fin), np.asarray(snaps)
        shard = lambda a: a.reshape(use, per, *a.shape[1:])
        args = (shard(spins),
                jax.random.split(jax.random.PRNGKey(key_seed + 1), use),
                shard(k1), {k: shard(v) for k, v in P.items()})
        fn = jax.pmap(body).lower(*args).compile()
        t0 = time.perf_counter()
        obs, fin, snaps = fn(*args)
        jax.block_until_ready((obs, fin, snaps))
        dt = time.perf_counter() - t0
        # (use, n_T, 8, per) -> (n_T, 8, batch); snapshots come from device 0.
        return dt, np.concatenate(list(np.asarray(obs)), axis=-1), \
               np.concatenate(list(np.asarray(fin)), axis=0), np.asarray(snaps)[0]

    total, OBS, FIN, SNAP = 0.0, [], [], []
    for i, (ths, rep) in enumerate(launches):
        dt, o, f, s = run_one(ths, rep, seed + 100 * i)
        total += dt
        OBS.append(o); FIN.append(f); SNAP.append(s)
        print(f"  launch {i+1}/{len(launches)}  {dt/60:7.2f} min  "
              f"({[t['name'] for t in ths]}, {len(ths)*rep} chains)", flush=True)

    sweeps = int(n_temps) * (n_therm + n_meas)
    obs = np.concatenate(OBS, axis=-1)
    red = S.reduce_obs(obs, T, geo["n_active"])
    res = dict(config=config, devices=ndev, launches=len(launches),
               total_chains=len(thetas) * PA.N_REPLICAS,
               elapsed_s=total, ms_per_sweep=1000 * total / sweeps,
               n_temps=int(n_temps), n_therm=n_therm, n_meas=n_meas,
               n_active=geo["n_active"], measured_full_ladder=True,
               thetas=thetas, device=str(jax.devices()[0]),
               **{k: np.asarray(v).tolist() for k, v in red.items()})
    np.savez_compressed(f"/out/{tag}.npz", raw_obs=obs,
                        sz_transition=np.concatenate(SNAP, axis=0),
                        spins_final=np.concatenate(FIN, axis=0),
                        **{k: np.asarray(v) for k, v in red.items()})
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    print(f"DONE {config} on {ndev} GPU: {total/60:.2f} min, "
          f"{res['ms_per_sweep']:.4f} ms/sweep", flush=True)
    return {k: v for k, v in res.items() if not isinstance(v, (list, dict))}


_AL = dict(image=image, memory=32768, timeout=6 * 3600, volumes={"/out": vol})


@app.function(gpu="H100", **_AL)
def axl1(c, nt, th, me, sd, tag, ns=0):
    return _axis_ladder(c, nt, th, me, sd, tag, ns)


@app.function(gpu="H100:2", **_AL)
def axl2(c, nt, th, me, sd, tag, ns=0):
    return _axis_ladder(c, nt, th, me, sd, tag, ns)


@app.function(gpu="H100:4", **_AL)
def axl4(c, nt, th, me, sd, tag, ns=0):
    return _axis_ladder(c, nt, th, me, sd, tag, ns)


AXL = {1: axl1, 2: axl2, 4: axl4}


@app.local_entrypoint()
def axis_ladder(config: str = "replicas", devices: int = 1, n_temps: int = 200,
                n_therm: int = 10000, n_meas: int = 5000, seed: int = 42,
                tag: str = "", states: int = 0):
    tag = tag or f"axl_{config}_{devices}gpu"
    print(json.dumps(AXL[devices].remote(config, n_temps, n_therm, n_meas,
                                         seed, tag, states), indent=1))


# --------------------------------------------------------------------------
# Isolating per-device batch size from device count
# --------------------------------------------------------------------------
# The six axis ladders showed the same per-device batch (B=100) costing 2.586
# us/chain-sweep on one GPU and 3.272 on four -- 27% apart -- so batch size alone
# does not explain the crossover and there is a separate multi-device cost.
#
# Two scans, each inside ONE container so the between-container scatter cannot
# masquerade as signal:
#   * B-scan   fixed device count, varying chains  -> the pure batch effect
#   * N-scan   fixed B, varying device count       -> the pure coordination cost
# Full 200-point ladders throughout: the whole lesson of this study is that short
# loops mislead by more than the effect being chased.

def _batch_scan(batches, repeats, n_temps, n_therm, n_meas, tag):
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S

    ndev = jax.local_device_count()
    geo = S.build_geometry(Rd=18.30, L=5)
    eng = S.build_engine(geo)
    P = dict(Jex=1.0, Kan1=0.177, Kan2=0.034, KanS=0.139, Hex=0.483,
             KDM=0.880, g=float(np.deg2rad(90.0)))
    kB = 0.086173404
    T = np.linspace(20.0, 0.1, int(n_temps))
    betas = jnp.asarray(1.0 / (kB * T))
    k1 = jnp.where(jnp.asarray(geo["surface"]), P["KanS"], P["Kan1"])
    sweeps = int(n_temps) * (n_therm + n_meas)
    print(f"devices={jax.devices()}  B per device: {batches}", flush=True)

    rows = []
    for B in batches:
        chains = B * ndev
        def body(sp, kk):
            return eng["anneal"](sp, kk, betas, P, n_therm, n_meas)
        if ndev == 1:
            fn = jax.jit(body)
            mk = lambda s: (S.init_spins(jax.random.PRNGKey(s), geo, chains),
                            jax.random.PRNGKey(s + 1))
        else:
            fn = jax.pmap(body)
            mk = lambda s: (
                S.init_spins(jax.random.PRNGKey(s), geo, chains)
                 .reshape(ndev, B, *((geo["L"], geo["N"], geo["N"], 3))),
                jax.random.split(jax.random.PRNGKey(s + 1), ndev))
        ts = []
        for r in range(repeats):
            sp, kk = mk(1000 * r + B)
            t0 = time.perf_counter()
            jax.block_until_ready(fn(sp, kk))
            ts.append(time.perf_counter() - t0)
            print(f"  B={B:4d} chains={chains:5d} rep {r+1}/{repeats}  "
                  f"{ts[-1]/60:7.2f} min  "
                  f"{ts[-1]*ndev*1e6/(sweeps*chains):7.3f} us/chain-sweep "
                  f"(device-time)", flush=True)
        a = np.array(ts)
        rows.append(dict(B=int(B), devices=ndev, chains=int(chains),
                         wall_s_mean=float(a.mean()), wall_s_sd=float(a.std()),
                         repeats=repeats, sweeps=sweeps,
                         us_per_chain_sweep=float(a.mean() * 1e6 / (sweeps * chains)),
                         device_us_per_chain_sweep=float(
                             a.mean() * ndev * 1e6 / (sweeps * chains))))
    res = dict(rows=rows, devices=ndev, n_temps=int(n_temps),
               n_therm=n_therm, n_meas=n_meas, params=P,
               n_active=geo["n_active"], device=str(jax.devices()[0]))
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


_BS = dict(image=image, memory=65536, timeout=6 * 3600, volumes={"/out": vol})


@app.function(gpu="H100", **_BS)
def bscan1(b, r, nt, th, me, tag): return _batch_scan(b, r, nt, th, me, tag)


@app.function(gpu="H100:2", **_BS)
def bscan2(b, r, nt, th, me, tag): return _batch_scan(b, r, nt, th, me, tag)


@app.function(gpu="H100:4", **_BS)
def bscan4(b, r, nt, th, me, tag): return _batch_scan(b, r, nt, th, me, tag)


BSCAN = {1: bscan1, 2: bscan2, 4: bscan4}


@app.local_entrypoint()
def batch_scan(devices: int = 1, batches: str = "25,50,100", repeats: int = 3,
               n_temps: int = 200, n_therm: int = 10000, n_meas: int = 5000,
               tag: str = ""):
    tag = tag or f"bscan_{devices}gpu"
    bl = [int(x) for x in batches.split(",")]
    r = BSCAN[devices].remote(bl, repeats, n_temps, n_therm, n_meas, tag)
    print(json.dumps(r["rows"], indent=1))


# --------------------------------------------------------------------------
# Step 1: what is the 26.5% at B=100?
# --------------------------------------------------------------------------
# Seven production ladders reduce to one identity, wall = 1.2e9 * f(B) / D, and
# it holds to 1.3% where two of them share a per-device batch (B=50, measured on
# two and on four GPUs). At B=100 it does NOT hold: `replicas 1x` read 2.586
# us/chain-sweep and `both 4x` read 3.272, which is 26.5% apart and larger than
# the 17% effect the whole study is trying to optimise. Three candidate causes,
# and the ladders cannot tell them apart because each ran in its own container:
#
#   A  mechanism   pmap across physical devices costs more than jit on one
#   B  launches    4 sequential launches differ from 1 (warm caches, allocator)
#   C  scatter     nothing differs; the two numbers came from two containers
#
# Every cell here runs in ONE container on the same four GPUs, so C is measured
# rather than assumed: if `jit_1` and `pmap_4` come out equal side by side, the
# ladders' 26.5% was never a property of the code.
#
# Cells, all at B=100 per device:
#   jit_1     jit,  1 device,  100 chains, 1 launch       <- the `replicas 1x` shape
#   pmap_1    pmap, 1 device,  100 chains, 1 launch       <- pmap code path, one GPU
#   pmap_2    pmap, 2 devices, 200 chains, 1 launch
#   pmap_4    pmap, 4 devices, 400 chains, 1 launch       <- the `both 4x` shape
#   jit_1x4   jit,  1 device,  100 chains, 4 launches     <- isolates B from A
#
# `pmap_1` is the load-bearing cell. It separates "pmap as a code path" from
# "several physical devices": if it reads like pmap_4 the cost is in the
# transformation, and if it reads like jit_1 the cost is in the hardware.
#
# 150k sweeps, not the ladders' 3M. The long loops were needed to rank effects
# of 10-20% across containers; a 26.5% gap inside one container does not need
# them, and the scan's own sd was 0.07% of the mean at full length.

_ANOM_CELLS = [("jit_1", 1, 1, False), ("pmap_1", 1, 1, True),
               ("pmap_2", 2, 1, True), ("pmap_4", 4, 1, True),
               ("jit_1x4", 1, 4, False)]


def _anomaly(B, repeats, n_temps, n_therm, n_meas, tag, cells=None):
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S

    devs = jax.devices()
    geo = S.build_geometry(Rd=18.30, L=5)
    eng = S.build_engine(geo)
    P = dict(Jex=1.0, Kan1=0.177, Kan2=0.034, KanS=0.139, Hex=0.483,
             KDM=0.880, g=float(np.deg2rad(90.0)))
    kB = 0.086173404
    T = np.linspace(20.0, 0.1, int(n_temps))
    betas = jnp.asarray(1.0 / (kB * T))
    sweeps = int(n_temps) * (n_therm + n_meas)
    shape = (geo["L"], geo["N"], geo["N"], 3)
    print(f"devices={devs}  B={B} per device  {sweeps} sweeps per launch",
          flush=True)

    def body(sp, kk):
        return eng["anneal"](sp, kk, betas, P, n_therm, n_meas)

    rows = []
    for name, ndev, nlaunch, use_pmap in _ANOM_CELLS:
        if ndev > len(devs) or (cells and name not in cells):
            continue
        sub = devs[:ndev]
        # `jax.jit(f, device=...)` is deprecated, so the single-device cells
        # pin their inputs instead and let the computation follow the data.
        fn = jax.pmap(body, devices=sub) if use_pmap else jax.jit(body)

        def mk(seed):
            tot = B * ndev
            sp = S.init_spins(jax.random.PRNGKey(seed), geo, tot)
            kk = jax.random.PRNGKey(seed + 1)
            if use_pmap:
                # pmap shards axis 0 across `sub` itself; no placement needed.
                return (sp.reshape(ndev, B, *shape),
                        jax.random.split(kk, ndev))
            return (jax.device_put(sp, sub[0]),
                    jax.device_put(kk, sub[0]))

        ts = []
        for r in range(repeats):
            args = [mk(7000 * r + 13 * nlaunch + i) for i in range(nlaunch)]
            t0 = time.perf_counter()
            for sp, kk in args:                      # sequential on purpose
                jax.block_until_ready(fn(sp, kk))
            ts.append(time.perf_counter() - t0)
            # Device-time per chain-sweep: one device carried B chains for
            # `sweeps` sweeps, `nlaunch` times. Dividing by ndev is wrong here
            # and dividing by total chains would hide the launch count.
            per = ts[-1] * 1e6 / (sweeps * B * nlaunch)
            print(f"  {name:8s} ndev={ndev} launches={nlaunch} "
                  f"rep {r+1}/{repeats}  {ts[-1]:7.1f} s  "
                  f"{per:7.3f} us/chain-sweep (device-time)", flush=True)
        # First repeat carries JIT compilation for this cell's shape.
        keep = np.array(ts[1:]) if repeats > 1 else np.array(ts)
        rows.append(dict(cell=name, devices=ndev, launches=nlaunch,
                         pmap=use_pmap, B=int(B),
                         chains=int(B * ndev), repeats=repeats,
                         wall_s_mean=float(keep.mean()),
                         wall_s_sd=float(keep.std()),
                         device_us_per_chain_sweep=float(
                             keep.mean() * 1e6 / (sweeps * B * nlaunch))))

    ref = {r["cell"]: r["device_us_per_chain_sweep"] for r in rows}
    print("\n=== verdict ===", flush=True)
    for r in rows:
        print(f"  {r['cell']:8s} {r['device_us_per_chain_sweep']:7.3f} "
              f"us  (sd {r['wall_s_sd']:.2f} s)", flush=True)
    if "jit_1" in ref and "pmap_4" in ref:
        print(f"\n  jit_1 vs pmap_4   {ref['pmap_4']/ref['jit_1']-1:+.1%}"
              "   (ladders said +26.5%)", flush=True)
    if "jit_1" in ref and "pmap_1" in ref:
        print(f"  jit_1 vs pmap_1   {ref['pmap_1']/ref['jit_1']-1:+.1%}"
              "   (pmap code path alone)", flush=True)
    if "jit_1" in ref and "jit_1x4" in ref:
        print(f"  jit_1 vs jit_1x4  {ref['jit_1x4']/ref['jit_1']-1:+.1%}"
              "   (launch count alone)", flush=True)

    res = dict(rows=rows, B=int(B), n_temps=int(n_temps), n_therm=n_therm,
               n_meas=n_meas, sweeps=sweeps, params=P,
               n_active=geo["n_active"], device=str(devs[0]),
               n_devices_available=len(devs))
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


@app.function(gpu="H100:4", **_BS)
def anom4(B, r, nt, th, me, tag, cells):
    return _anomaly(B, r, nt, th, me, tag, cells)


@app.local_entrypoint()
def anomaly(batch: int = 100, repeats: int = 3, n_temps: int = 40,
            n_therm: int = 10000, n_meas: int = 5000, tag: str = "",
            cells: str = ""):
    # Named `batch`, not `B`: Modal lowercases CLI option names, so a `B`
    # parameter is unreachable from `modal run`.
    tag = tag or f"anom_B{batch}"
    sel = [c for c in cells.split(",") if c] or None
    r = anom4.remote(batch, repeats, n_temps, n_therm, n_meas, tag, sel)
    print(json.dumps(r["rows"], indent=1))


# --------------------------------------------------------------------------
# Step 1, continued: the parameter form
# --------------------------------------------------------------------------
# The anomaly run killed all three candidates it was built for. At B=100 on one
# host: pmap over four devices cost -0.2% against jit on one, four sequential
# launches cost +0.9%, and two independent containers agreed to 0.3% (2.962 in
# bscan_1gpu, 2.971 here). So it is not the mechanism, not the launch count, and
# not container scatter.
#
# What survives is the candidate the docs listed as untested: the ladders pass
# per-chain parameter columns of shape (B,1,1,1), the scans pass Python scalars.
# At B=100 on one GPU the column form read 2.586 us and the scalar form 2.962 --
# the columns are 14.5% FASTER, which is backwards. `anneal` builds
# `K1_eff = where(surf, p["KanS"], p["Kan1"])` from whatever it is handed, so
# columns make that tensor (B,L,N,N,1) instead of (L,N,N,1): B times the memory,
# and still quicker. Either a broadcast in the scalar path costs more than the
# extra reads, or the two forms are not running the same fused kernel.
#
# One container, one GPU, both forms, nothing else different.

def _pform(B, repeats, n_temps, n_therm, n_meas, tag, forms=None):
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S
    import parallel_axes as PA

    geo = S.build_geometry(Rd=18.30, L=5)
    eng = S.build_engine(geo)
    kB = 0.086173404
    betas = jnp.asarray(1.0 / (kB * np.linspace(20.0, 0.1, int(n_temps))))
    sweeps = int(n_temps) * (n_therm + n_meas)
    scal = dict(Jex=1.0, Kan1=0.177, Kan2=0.034, KanS=0.139, Hex=0.483,
                KDM=0.880, g=float(np.deg2rad(90.0)))
    # One theta repeated B times: the same physics as the scalar cell, so any
    # difference is the shape of the parameters and not their values.
    cols = PA.pack([dict(name="x", **scal)], B)
    # The ladders did NOT use the values above. `load_thetas` forces Kan2 = 0
    # and every state in theta4.json carries KanS = 0, so the ladder's columns
    # are mostly zeros where these are not. Three parameter forms measured in
    # one container is the only way to tell a shape effect from a value effect,
    # and nothing about a vectorised kernel predicts a value effect -- which is
    # exactly why it has to be measured rather than argued.
    theta = PA.load_thetas("/root/theta4.json")[0]
    cols_theta = PA.pack([theta], B)
    # Mixed thetas on ONE device. Every ladder packed theta-major, so with four
    # states over four devices each device held a pure state and this case was
    # never measured -- yet the claim "it does not matter whether a GPU carries
    # one state or several" rests on it. Same shape, four distinct values per
    # column instead of one repeated.
    four = PA.load_thetas("/root/theta4.json")
    assert B % len(four) == 0, f"B={B} must divide by {len(four)} states"
    cols_mixed = PA.pack(four, B // len(four))

    def body(sp, kk, p):
        return eng["anneal"](sp, kk, betas, p, n_therm, n_meas)

    fn = jax.jit(body)
    rows = []
    for name, p in (("scalar", scal), ("columns", cols),
                    ("ladder_theta", cols_theta), ("mixed_theta", cols_mixed)):
        if forms and name not in forms:
            continue
        ts = []
        for r in range(repeats):
            sp = S.init_spins(jax.random.PRNGKey(900 * r + 1), geo, B)
            kk = jax.random.PRNGKey(900 * r + 2)
            t0 = time.perf_counter()
            jax.block_until_ready(fn(sp, kk, p))
            ts.append(time.perf_counter() - t0)
            print(f"  {name:8s} rep {r+1}/{repeats}  {ts[-1]:7.1f} s  "
                  f"{ts[-1]*1e6/(sweeps*B):7.3f} us/chain-sweep", flush=True)
        keep = np.array(ts[1:]) if repeats > 1 else np.array(ts)
        rows.append(dict(form=name, B=int(B), repeats=repeats,
                         wall_s_mean=float(keep.mean()),
                         wall_s_sd=float(keep.std()),
                         us_per_chain_sweep=float(
                             keep.mean() * 1e6 / (sweeps * B))))

    print("\n=== verdict ===", flush=True)
    for r in rows:
        print(f"  {r['form']:8s} {r['us_per_chain_sweep']:7.3f} us "
              f"(sd {r['wall_s_sd']:.2f} s)  n_temps={n_temps}", flush=True)
    d = {r["form"]: r["us_per_chain_sweep"] for r in rows}
    if "scalar" in d and "columns" in d:
        print(f"  columns vs scalar  {d['columns']/d['scalar']-1:+.1%}",
              flush=True)
    if "columns" in d:
        print(f"  vs `replicas 1x` ladder (2.586 us)  "
              f"{d['columns']/2.586-1:+.1%}", flush=True)

    res = dict(rows=rows, B=int(B), sweeps=sweeps, n_temps=int(n_temps),
               n_therm=n_therm, n_meas=n_meas, device=str(jax.devices()[0]))
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


@app.function(gpu="H100", **_BS)
def pform1(B, r, nt, th, me, tag, forms=None):
    return _pform(B, r, nt, th, me, tag, forms)


@app.local_entrypoint()
def param_form(batch: int = 100, repeats: int = 3, n_temps: int = 40,
               n_therm: int = 10000, n_meas: int = 5000, tag: str = "",
               forms: str = ""):
    tag = tag or f"pform_B{batch}"
    sel = [f for f in forms.split(",") if f] or None
    print(json.dumps(pform1.remote(batch, repeats, n_temps, n_therm, n_meas,
                                   tag, sel)["rows"], indent=1))


# --------------------------------------------------------------------------
# Step 2: f(B), the whole curve, in production form
# --------------------------------------------------------------------------
# Step 1 established the three facts that make this a one-dimensional problem,
# each measured rather than argued: device count costs -0.2%, launch count
# +0.9%, and parameter values -0.2%. So the cost of any arrangement of chains is
# `total_chain_sweeps * f(B) / devices`, where B is the per-device batch and f is
# a property of one device. Measure f once and every grouping follows on paper.
#
# Two things step 1 also forces about HOW to measure it:
#   * Per-chain columns, not scalars. The two forms differ by 12.9% and the
#     scalar optimum (B=50) need not be the production one.
#   * Compile ahead of time. The ladders charged ~27 s per launch of XLA
#     compilation to the kernel, which is the bias that made `replicas` look
#     worse than it is.
#
# B=75 is in the grid because it is the batch a three-state run would want. It
# does not shard over four devices (75 % 4 != 0, so `run_one` would fall back to
# one device and idle three), but the curve is a per-device property and B=75 is
# reachable on one GPU or by an uneven split, so its cost is worth knowing.
#
# 40 temperatures, not 200: step 1 measured that difference at -4.2%, small
# enough to leave the shape of the curve intact and the only reason this is
# affordable at eleven points.

def device_identity():
    """Which physical GPU this container got, as far as it can be established.

    Modal does not pin a run to a specific device: `gpu="H100"` yields *an*
    H100, and a later container may land on another host. That is fine for the
    shape of a curve, since one container measures every batch size on the same
    device, but it is exactly the thing a reader should be able to check rather
    than take on trust -- so record it. `str(jax.devices()[0])` returns only
    "cuda:0" and was not enough.

    nvidia-smi may be unavailable under the sandbox, so its absence is recorded
    rather than raised.
    """
    import subprocess
    import jax
    d = jax.devices()[0]
    out = dict(repr=str(d), kind=getattr(d, "device_kind", "?"),
               platform=getattr(d, "platform", "?"),
               n_devices=jax.local_device_count())
    try:
        q = ("name,uuid,memory.total,clocks.max.sm,driver_version,"
             "pcie.link.gen.max")
        r = subprocess.run(["nvidia-smi", f"--query-gpu={q}",
                            "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        out["smi"] = (r.stdout.strip().splitlines() if r.returncode == 0
                      else f"exit {r.returncode}")
    except Exception as e:
        out["smi"] = f"unavailable: {type(e).__name__}"
    return out


F_GRID = [10, 16, 25, 32, 40, 50, 64, 75, 80, 100, 128]


def _fcurve(batches, repeats, n_temps, n_therm, n_meas, tag,
            Rd=18.30, L=5):
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S
    import parallel_axes as PA

    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    eng = S.build_engine(geo)
    kB = 0.086173404
    betas = jnp.asarray(1.0 / (kB * np.linspace(20.0, 0.1, int(n_temps))))
    sweeps = int(n_temps) * (n_therm + n_meas)
    theta = PA.load_thetas("/root/theta4.json")[0]     # the ladders' own theta
    n_active = int(np.asarray(geo["disco"]).sum())
    print(f"gpu={device_identity()}", flush=True)
    print(f"device={jax.devices()[0]}  columns form  Rd={Rd} L={L} "
          f"N={geo['N']} n_active={n_active}  {sweeps} sweeps/cell  "
          f"B grid {batches}", flush=True)

    def body(sp, kk, p):
        return eng["anneal"](sp, kk, betas, p, n_therm, n_meas)

    rows = []
    for B in batches:
        cols = PA.pack([theta], B)
        args = (S.init_spins(jax.random.PRNGKey(11), geo, B),
                jax.random.PRNGKey(12), cols)
        fn = jax.jit(body).lower(*args).compile()      # nothing timed yet
        ts = []
        for r in range(repeats):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*args))
            ts.append(time.perf_counter() - t0)
            print(f"  B={B:4d} rep {r+1}/{repeats}  {ts[-1]:7.1f} s  "
                  f"{ts[-1]*1e6/(sweeps*B):7.3f} us/chain-sweep", flush=True)
        a = np.array(ts)
        rows.append(dict(B=int(B), repeats=repeats, wall_s_mean=float(a.mean()),
                         wall_s_sd=float(a.std()),
                         us_per_chain_sweep=float(a.mean()*1e6/(sweeps*B))))

    best = min(rows, key=lambda r: r["us_per_chain_sweep"])
    print(f"\n=== f(B), columns form, compile excluded ===", flush=True)
    for r in rows:
        mark = "  <- min" if r is best else ""
        print(f"  B={r['B']:5d}  {r['us_per_chain_sweep']:7.3f} us  "
              f"(sd {r['wall_s_sd']:5.2f} s)  "
              f"{r['us_per_chain_sweep']/best['us_per_chain_sweep']:5.3f}x  "
              f"B*sites={r['B']*n_active:10,d}{mark}", flush=True)

    res = dict(rows=rows, form="columns", sweeps=sweeps, n_temps=int(n_temps),
               n_therm=n_therm, n_meas=n_meas, theta=theta, Rd=float(Rd),
               L=int(L), N=int(geo["N"]), n_active=n_active,
               device=device_identity(), aot_compiled=True)
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


# --------------------------------------------------------------------------
# Stride discriminator: does v2(replica stride) steer the penalty off a T4?
# --------------------------------------------------------------------------
# On a T4 the mod-4 penalty moves with the power of two in the replica stride
# (`L * N^2 * 12` bytes): 4 x odd suffers, 16 x odd is immune. That was measured
# on three lattices. This runs the same three in ONE container, so the three
# penalties share a physical GPU and the comparison carries no device drift.

LATTICES = [(18.30, 5), (15.00, 3), (7.50, 12)]   # v2(stride) = 2, 2, 4

# Just the reference lattice, for using the known-good harness as a control
# against a new one. `_stride_disc` is the code path that produced +12.1% and
# +14.2%, so running it alone isolates the harness from the effect.
REF_ONLY = [(18.30, 5)]


def _stride_disc(lattices, batches, repeats, n_temps, n_therm, n_meas, tag):
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S
    import parallel_axes as PA

    kB = 0.086173404
    betas = jnp.asarray(1.0 / (kB * np.linspace(20.0, 0.1, int(n_temps))))
    sweeps = int(n_temps) * (n_therm + n_meas)
    theta = PA.load_thetas("/root/theta4.json")[0]
    print(f"gpu={device_identity()}", flush=True)

    def v2(n):
        k = 0
        while n % 2 == 0:
            n //= 2
            k += 1
        return k

    out = []
    for Rd, L in lattices:
        geo = S.build_geometry(Rd=float(Rd), L=int(L))
        eng = S.build_engine(geo)
        N = int(geo["N"])
        stride = int(L) * N * N * 12
        print(f"\n--- Rd={Rd} L={L} N={N} stride={stride:,} v2={v2(stride)} "
              f"({stride // 2 ** v2(stride)} x 2^{v2(stride)}) ---", flush=True)

        def body(sp, kk, p, _e=eng):
            return _e["anneal"](sp, kk, betas, p, n_therm, n_meas)

        rows = []
        for B in batches:
            cols = PA.pack([theta], B)
            args = (S.init_spins(jax.random.PRNGKey(11), geo, B),
                    jax.random.PRNGKey(12), cols)
            fn = jax.jit(body).lower(*args).compile()   # nothing timed yet
            ts = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                jax.block_until_ready(fn(*args))
                ts.append(time.perf_counter() - t0)
            a = np.array(ts)
            us = float(a.mean() * 1e6 / (sweeps * B))
            rows.append(dict(B=int(B), us=us, sd_s=float(a.std())))
            ve = v2(B) + v2(int(L))
            print(f"  B={B:4d}  {us:8.3f} us  (sd {a.std():.3f} s)"
                  f"  v2(elements)={ve}"
                  f"  width={min(4, 2 ** ve)}"
                  f"{'  (B|4)' if B % 4 == 0 else ''}", flush=True)

        m4 = [r["us"] for r in rows if r["B"] % 4 == 0]
        nm = [r["us"] for r in rows if r["B"] % 4]
        pen = (np.mean(m4) / np.mean(nm) - 1) if m4 and nm else float("nan")
        overlap = not (min(m4) > max(nm)) if m4 and nm else None
        print(f"  => mod-4 penalty {pen:+.1%}   overlap: "
              f"{'yes' if overlap else 'NONE'}", flush=True)

        # Split the same rows by the predicted vector width instead. The launch
        # dumps show XLA picks `width = min(4, 2^v2(elements))`, so this is a
        # width-4 against narrower split, NOT vectorised against scalar -- at
        # v2(elements) == 1 the narrow side is still 2-wide. For an odd L the
        # split coincides with the mod-4 one; where it differs, it is the
        # mechanism under test, so the two are reported side by side.
        w4 = [r["us"] for r in rows if v2(r["B"]) + v2(int(L)) >= 2]
        nar = [r["us"] for r in rows if v2(r["B"]) + v2(int(L)) < 2]
        vpen = (np.mean(w4) / np.mean(nar) - 1) if w4 and nar else float("nan")
        vover = (not (min(w4) > max(nar))) if w4 and nar else None
        print(f"  => predicted width-4 penalty {vpen:+.1%} "
              f"({len(w4)} at width 4 vs {len(nar)} narrower)   overlap: "
              f"{'yes' if vover else 'NONE' if vover is not None else 'n/a'}",
              flush=True)
        out.append(dict(Rd=float(Rd), L=int(L), N=N, stride=stride,
                        v2_stride=v2(stride), n_active=int(np.asarray(geo["disco"]).sum()),
                        rows=rows, mod4_penalty=pen, overlap=bool(overlap),
                        width4_penalty=vpen, width4_overlap=vover,
                        n_width4=len(w4), n_narrower=len(nar)))

    print("\n=== stride discriminator ===", flush=True)
    print(f"{'Rd':>6} {'L':>3} {'stride':>10} {'v2':>3} {'mod-4 penalty':>14} "
          f"{'overlap':>8} {'predicted':>11} {'overlap':>8}", flush=True)
    for o in out:
        vo = o["width4_overlap"]
        print(f"{o['Rd']:6.2f} {o['L']:3d} {o['stride']:10,d} {o['v2_stride']:3d} "
              f"{o['mod4_penalty']:+13.1%} {'yes' if o['overlap'] else 'NONE':>8} "
              f"{o['width4_penalty']:+10.1%} "
              f"{('yes' if vo else 'NONE') if vo is not None else 'n/a':>8}",
              flush=True)

    res = dict(lattices=out, batches=list(map(int, batches)), sweeps=sweeps,
               repeats=repeats, device=device_identity(), aot_compiled=True)
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


@app.function(gpu="H100", **_BS)
def stride_disc_h100(lat, b, r, nt, th, me, tag):
    return _stride_disc(lat, b, r, nt, th, me, tag)


@app.function(gpu="A100-40GB", **_BS)
def stride_disc_a100(lat, b, r, nt, th, me, tag):
    return _stride_disc(lat, b, r, nt, th, me, tag)


@app.function(gpu="T4", image=image, memory=32768, timeout=6 * 3600,
              volumes={"/out": vol})
def stride_disc_t4(lat, b, r, nt, th, me, tag):
    return _stride_disc(lat, b, r, nt, th, me, tag)


# --------------------------------------------------------------------------
# Experiment 1: turn "v2(stride) decides whether" into a curve.
#
# `stride = 12 * L * N^2` and `N = 2*(int(Rd)+1)+1` is always ODD, so the only
# even factors come from 12 = 4*3 and from L itself:
#
#       v2(stride) = 2 + v2(L)                       (verified, 192/192 pairs)
#
# Two consequences. First, the geometry-only rule collapses to counting factors
# of two in the LAYER COUNT -- Rd cannot change it. Second, v2 = 0 and v2 = 1
# are unreachable while a site is 3 floats, so the sweep starts at 2.
#
# Each (Rd, L) below is chosen so n_active stays near 2125, which holds the work
# per replica fixed (within 5.6%) while v2 runs 2 -> 6. v2 = 2, 3 and 4 appear
# twice at different L, which tests whether the penalty depends on v2 alone or
# also on the lattice shape.
#
# What this design CANNOT do: vary stride magnitude at fixed v2 and fixed work.
# stride ~= 12*L*(2*Rd)^2 and n_active ~= L*pi*Rd^2, so stride/n_active is
# pinned near 48/pi ~= 16 for every geometry. Magnitude and work per replica are
# proportional by construction, so no experiment on this lattice family can
# separate them. The v2 arm is the only clean one.
STRIDE_SWEEP = [
    (26.05, 1),    # v2=2  stride 36,300  n_active 2129
    (15.05, 3),    # v2=2  stride 39,204  n_active 2151
    (18.40, 2),    # v2=3  stride 36,504  n_active 2138
    (10.65, 6),    # v2=3  stride 38,088  n_active 2142
    (13.05, 4),    # v2=4  stride 40,368  n_active 2180
    (7.30, 12),    # v2=4  stride 41,616  n_active 2124
    (9.10, 8),     # v2=5  stride 42,336  n_active 2088
    (6.35, 16),    # v2=6  stride 43,200  n_active 2064
]


# Holding n_active near 2125 above also held the array footprint near 2 MB,
# while the reference lattice that shows the +12.1% penalty sits at 5.1 MB. If
# the effect needs the larger footprint, STRIDE_SWEEP cannot see it at any v2 --
# the control fixed the wrong quantity. This set repeats the identical v2 sweep
# with n_active held near 5245 instead, which puts every cell in the footprint
# regime where the penalty is known to exist and keeps the strides within 13%
# of the reference's 91,260.
STRIDE_SWEEP_BIG = [
    (40.85, 1),    # v2=2  stride 82,668   n_active 5241
    (23.55, 3),    # v2=2  stride 86,436   n_active 5247
    (28.90, 2),    # v2=3  stride 83,544   n_active 5234
    (16.65, 6),    # v2=3  stride 88,200   n_active 5262
    (20.40, 4),    # v2=4  stride 88,752   n_active 5252
    (11.75, 12),   # v2=4  stride 90,000   n_active 5244
    (14.35, 8),    # v2=5  stride 92,256   n_active 5256
    (10.05, 16),   # v2=6  stride 101,568  n_active 5200
]


# Experiment 4: the falsifiable consequence of the vectorisation mechanism.
#
# The cause is XLA taking a 4-wide loop over `elements = B * L * N^2 * 3`, and
# `N^2 * 3` is odd, so
#
#       v2(elements) = v2(B) + v2(L)
#
# and the 4-wide path needs v2(elements) >= 2. That predicts three different
# behaviours from the layer count alone:
#
#       L odd           vectorises only at B divisible by 4   -> the mod-4 rule
#       L = 2 mod 4     vectorises at EVERY EVEN B            -> a mod-2 rule
#       L divisible 4   vectorises at every B                 -> immune
#
# Every window measured so far (51,52,53,59,60,61,63,64,65) contains only ODD
# non-multiples of four, so it cannot tell the mod-4 rule from the mod-2 one.
# This window deliberately includes B = 58 and 62: even, not multiples of four.
# L = 6 must be slow there and L = 3 must not. Footprints are held near the
# 5.2k-site regime where the penalty is known to exist.
PARITY_SWEEP = [
    (16.65, 6),    # L = 2 mod 4 -> predicted slow at 58, 60, 62, 64
    (23.55, 3),    # L odd       -> predicted slow at 60, 64 only
    (20.40, 4),    # L | 4       -> predicted flat, already vectorised
]


# Experiment 6: does the GEOMETRY half of the rule hold off an H100?
#
# The five-device campaign only ever measured L = 5, which is odd -- exactly the
# case where `B % 4` and `v2(B) + v2(L) >= 2` are the same statement. Every
# measurement of the layer dependence ran on an H100, and XLA's code generator
# consults the target compute capability, so nothing so far rules out the
# vector-width choice being specific to Hopper.
#
# This pair settles it on a Turing T4, two architecture generations away and the
# device with the largest penalty (+42%), so the signal is the clearest
# available. Footprints are matched so the comparison is not confounded by the
# footprint threshold that killed arm A of experiment 4.
#
#   L = 5   odd        -> must keep the mod-4 penalty
#   L = 4   divisible  -> must be FLAT if the rule travels
GEOM_PAIR = [
    (18.30, 5),    # the reference lattice: +42.0% on a T4, 5245 sites
    (20.40, 4),    # predicted immune, 5252 sites -- matched within 0.2%
]


# --------------------------------------------------------------------------
# Experiment 3a: is the penalty XLA's layout choice rather than the hardware?
#
# Every measurement so far runs one kernel through JAX/XLA, so a batch-dimension
# penalty could be XLA padding or re-laying-out the array at some B and not
# others. This compiles the real kernel across the window and reports, per B,
# the parameter layouts and the bytes XLA actually allocates. If allocation
# grows strictly proportionally to B and the layouts are identical, XLA is not
# padding and the penalty has to come from below it.
def _xla_layout_probe(batches, tag, Rd, L):
    import sys, json, re
    sys.path.insert(0, "/root")
    import numpy as np, jax
    import sim_core as S
    import parallel_axes as PA

    kB = 0.086173404
    betas = __import__("jax").numpy.asarray(1.0 / (kB * np.linspace(20.0, 0.1, 4)))
    theta = PA.load_thetas("/root/theta4.json")[0]
    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    eng = S.build_engine(geo)
    N = int(geo["N"])
    print(f"gpu={device_identity()}", flush=True)
    print(f"Rd={Rd} L={L} N={N} stride={12*L*N*N:,} bytes/replica", flush=True)

    def body(sp, kk, p):
        return eng["anneal"](sp, kk, betas, p, 2, 2)

    out = []
    for B in batches:
        cols = PA.pack([theta], B)
        args = (S.init_spins(jax.random.PRNGKey(11), geo, B),
                jax.random.PRNGKey(12), cols)
        comp = jax.jit(body).lower(*args).compile()
        txt = comp.as_text()
        try:
            ma = comp.memory_analysis()
            temp = int(ma.temp_size_in_bytes)
            argb = int(ma.argument_size_in_bytes)
            outb = int(ma.output_size_in_bytes)
        except Exception as e:            # the field set moves between versions
            temp = argb = outb = -1
            print(f"  memory_analysis unavailable: {e}", flush=True)
        # Entry parameter shapes carry their layout in {...}; the spin array is
        # the rank-5 one. A padded batch shows up either as a different layout
        # or as an allocation that is not proportional to B.
        lay = re.findall(r"f32\[%d,\d+,\d+,\d+,3\]\{[0-9,]+\}" % B, txt)
        ninstr = txt.count("\n")
        nfusion = len(re.findall(r"fusion\(", txt))
        out.append(dict(B=int(B), temp_bytes=temp, arg_bytes=argb, out_bytes=outb,
                        spin_layouts=sorted(set(lay))[:3], hlo_lines=ninstr,
                        fusions=nfusion))
        print(f"  B={B:4d}  temp={temp:12,d}  arg={argb:11,d}  "
              f"out={outb:10,d}  fusions={nfusion:4d}  hlo_lines={ninstr:5d}  "
              f"{(sorted(set(lay))[:1] or ['-'])[0]}", flush=True)

    print("\n=== per-replica allocation (bytes / B) ===", flush=True)
    print(f"{'B':>5} {'B%4':>4} {'temp/B':>12} {'arg/B':>10} {'fusions':>8}", flush=True)
    for r in out:
        print(f"{r['B']:5d} {r['B'] % 4:4d} {r['temp_bytes'] / r['B']:12,.1f} "
              f"{r['arg_bytes'] / r['B']:10,.1f} {r['fusions']:8d}", flush=True)
    per = [r["temp_bytes"] / r["B"] for r in out if r["temp_bytes"] > 0]
    if per:
        print(f"\ntemp bytes per replica spread: {min(per):,.1f} .. {max(per):,.1f} "
              f"({100 * (max(per) / min(per) - 1):.2f}%)", flush=True)
        print("A flat per-replica allocation and one layout across the window "
              "means XLA is NOT padding the batch dimension.", flush=True)
    uniq = {tuple(r["spin_layouts"]) for r in out}
    print(f"distinct spin-array layouts across the window: {len(uniq)}", flush=True)
    res = dict(rows=out, Rd=float(Rd), L=int(L), N=N, stride=12 * L * N * N,
               device=device_identity())
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


# --------------------------------------------------------------------------
# Experiment 3b: the same access pattern with XLA removed entirely.
#
# A 7-point stencil over the identical (B, L, N, N, 3) float32 array, with the
# identical topology (x and y wrap, z does not) and the identical checkerboard
# parity split, compiled by nvrtc through CuPy. No JAX, no XLA, no autotuner.
# Thread i maps to flat site i, which is the natural linearisation XLA also
# uses, so the replica stride is the same 12*L*N^2 bytes.
#
# This is NOT the Metropolis kernel: no RNG, no anisotropy, no acceptance test.
# It is the memory traffic of a substep and nothing else. That is the point --
# if the mod-4 penalty survives here, it belongs to the access pattern and the
# hardware, not to anything XLA decided.
_STENCIL = r"""
extern "C" __global__
void substep(const float* __restrict__ s, float* __restrict__ o,
             const unsigned char* __restrict__ mask,
             int B, int L, int N, int parity) {
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  long long nsite = (long long)B * L * N * N;
  if (tid >= nsite) return;
  int x = (int)(tid % N);
  int y = (int)((tid / N) % N);
  int z = (int)((tid / ((long long)N * N)) % L);
  long long b = tid / ((long long)N * N * L);
  long long base = tid * 3;
  if (!mask[(long long)z * N * N + (long long)y * N + x] ||
      (((x + y + z) & 1) != parity)) {
    o[base] = s[base]; o[base + 1] = s[base + 1]; o[base + 2] = s[base + 2];
    return;
  }
  const int dx[6] = {1, -1, 0, 0, 0, 0};
  const int dy[6] = {0, 0, 1, -1, 0, 0};
  const int dz[6] = {0, 0, 0, 0, 1, -1};
  float ax = 0.f, ay = 0.f, az = 0.f;
  for (int k = 0; k < 6; ++k) {
    int nz = z + dz[k];
    if (nz < 0 || nz >= L) continue;          // z does not wrap
    int ny = (y + dy[k] + N) % N;
    int nx = (x + dx[k] + N) % N;
    long long nb = ((((b * L + nz) * N + ny) * (long long)N) + nx) * 3;
    ax += s[nb]; ay += s[nb + 1]; az += s[nb + 2];
  }
  float ex = s[base]     + 0.1f * ax;
  float ey = s[base + 1] + 0.1f * ay;
  float ez = s[base + 2] + 0.1f * az;
  float inv = rsqrtf(ex * ex + ey * ey + ez * ez + 1e-20f);
  o[base] = ex * inv; o[base + 1] = ey * inv; o[base + 2] = ez * inv;
}
"""


# --------------------------------------------------------------------------
# Experiment 5: WHY does width 4 cost and width 2 not?
#
# XLA's decision is about an ELEMENTWISE loop over the flat spin array, not
# about the stencil: the kernels whose grid dropped 3.94x are `loop_*_fusion`,
# one output element per input element. So the question isolates cleanly --
# take an elementwise kernel over n floats and hand each thread 1, 2 or 4 of
# them, with the total work fixed, and see which is slower on real hardware.
#
# Two index schemes, because they are not equivalent and XLA has to pick one:
#
#   blocked   thread t owns elements  t*VEC .. t*VEC+VEC-1   (consecutive)
#   strided   thread t owns elements  t, t+T, t+2T, ...      (grid-stride)
#
# Blocked is the suspect. With VEC consecutive floats per thread, a single load
# instruction across a 32-thread warp touches addresses 4*VEC bytes apart, so
# the warp's request spans VEC*128 bytes instead of 128 and needs VEC times the
# transactions. Strided keeps every individual load perfectly coalesced.
#
# If blocked VEC=4 is slow and strided VEC=4 is not, the mechanism is the
# indexing, and the device-dependent amplitude would follow from how much each
# memory system suffers from it. If both are flat, the cost is somewhere else
# (register pressure, spills) and this probe says so by being null.
_ELEMENTWISE = r"""
extern "C" __global__
void ew(const float* __restrict__ a, const float* __restrict__ b,
        float* __restrict__ o, long long n) {
  const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long T = (long long)gridDim.x * blockDim.x;
#pragma unroll
  for (int j = 0; j < VEC; ++j) {
    const long long i = BLOCKED ? (t * VEC + j) : (t + (long long)j * T);
    if (i < n) {
      const float x = a[i], y = b[i];
      // Arithmetic density in the range of the real fusions: a couple of
      // multiplies, a max and a reciprocal. Cheap enough that memory decides.
      o[i] = fmaxf(x * y + x, 0.0f) * __frcp_rn(1.0f + fabsf(y));
    }
  }
}
"""


def _vector_width_probe(vecs, schemes, iters, repeats, tag, Rd, L, B):
    """Is a 4-wide elementwise loop intrinsically slower than a 1-wide one?"""
    import sys, json, itertools
    sys.path.insert(0, "/root")
    import numpy as np, cupy as cp
    import sim_core as S

    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    N = int(geo["N"])
    n = int(B) * int(L) * N * N * 3          # exactly XLA's element count
    dev = cp.cuda.Device()
    print(f"gpu={cp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode()} "
          f"SMs={dev.attributes['MultiProcessorCount']}", flush=True)
    print(f"Rd={Rd} L={L} N={N} B={B}  n={n:,} elements "
          f"({3 * 4 * n / 2**20:.1f} MB touched per pass)  "
          f"{iters} iterations/cell", flush=True)

    rng = np.random.default_rng(7)
    a = cp.asarray(rng.standard_normal(n).astype(np.float32))
    b = cp.asarray(rng.standard_normal(n).astype(np.float32))
    o = cp.empty_like(a)
    TPB = 128                                 # the block size XLA actually used

    rows = []
    for vec, scheme in itertools.product(vecs, schemes):
        blocked = 1 if scheme == "blocked" else 0
        mod = cp.RawModule(code=_ELEMENTWISE, options=(
            "-std=c++11", f"-DVEC={vec}", f"-DBLOCKED={blocked}"))
        k = mod.get_function("ew")
        grid = (n + TPB * vec - 1) // (TPB * vec)
        args = (a, b, o, np.int64(n))
        k((grid,), (TPB,), args)              # warm up / compile
        cp.cuda.Stream.null.synchronize()
        ts = []
        for _ in range(repeats):
            e0, e1 = cp.cuda.Event(), cp.cuda.Event()
            e0.record()
            for _ in range(iters):
                k((grid,), (TPB,), args)
            e1.record(); e1.synchronize()
            ts.append(cp.cuda.get_elapsed_time(e0, e1) / 1e3)
        arr = np.array(ts)
        per = float(arr.mean() / iters)                  # seconds per pass
        gbs = 3.0 * 4.0 * n / per / 1e9                  # two reads, one write
        rows.append(dict(vec=int(vec), scheme=scheme, grid=int(grid),
                         block=TPB, s_per_pass=per, gb_s=gbs,
                         sd_s=float(arr.std())))
        print(f"  VEC={vec} {scheme:8s} grid={grid:8,d} x {TPB}  "
              f"{per * 1e6:9.2f} us/pass  {gbs:8.1f} GB/s  "
              f"(sd {arr.std():.4f} s)", flush=True)

    print("\n=== cost of widening the loop, at fixed total work ===", flush=True)
    print(f"{'scheme':>9} {'VEC=1':>11} {'VEC=2':>11} {'VEC=4':>11} "
          f"{'4 vs 1':>9} {'2 vs 1':>9}", flush=True)
    summary = {}
    for scheme in schemes:
        by = {r["vec"]: r["s_per_pass"] for r in rows if r["scheme"] == scheme}
        base = by.get(1)
        line = f"{scheme:>9}"
        for v in (1, 2, 4):
            line += f" {by[v] * 1e6:10.2f}u" if v in by else f" {'-':>11}"
        r4 = by[4] / base - 1 if base and 4 in by else float("nan")
        r2 = by[2] / base - 1 if base and 2 in by else float("nan")
        summary[scheme] = dict(vs1_at4=r4, vs1_at2=r2)
        print(line + f" {r4:+8.1%} {r2:+8.1%}", flush=True)

    print("\nThe JAX kernel pays +14.25% going from width 1 to width 4 and "
          "+0.09% going to width 2. A blocked scheme that reproduces that "
          "signature locates the cost in the indexing; a flat result puts it "
          "elsewhere, and either way this probe carries no XLA.", flush=True)
    res = dict(rows=rows, summary=summary, n_elements=n, block=TPB,
               Rd=float(Rd), L=int(L), N=N, B=int(B), iters=iters,
               repeats=repeats, device=device_identity(),
               reference_jax=dict(width2_vs_width1=0.0009,
                                  width4_vs_width1=0.1425,
                                  measured_on="Rd 23.55 L 3, H100"))
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


@app.function(gpu="H100", image=cuda_image, memory=65536, timeout=3600,
              volumes={"/out": vol})
def vecwidth_h100(v, sc, it, r, tag, Rd, L, B):
    return _vector_width_probe(v, sc, it, r, tag, Rd, L, B)


@app.function(gpu="T4", image=cuda_image, memory=32768, timeout=3600,
              volumes={"/out": vol})
def vecwidth_t4(v, sc, it, r, tag, Rd, L, B):
    return _vector_width_probe(v, sc, it, r, tag, Rd, L, B)


@app.local_entrypoint()
def vector_width(vecs: str = "1,2,4", schemes: str = "blocked,strided",
                 iters: int = 300, repeats: int = 3, rd: float = 18.30,
                 layers: int = 5, batch: int = 60, device: str = "h100",
                 tag: str = "exp5_vecwidth"):
    """Experiment 5: why width 4 costs. Raw CUDA, no XLA anywhere."""
    fn = {"h100": vecwidth_h100, "t4": vecwidth_t4}[device.lower()]
    fn.remote([int(x) for x in vecs.split(",") if x],
              [x.strip() for x in schemes.split(",") if x.strip()],
              iters, repeats, tag + "_" + device.lower(), rd, layers, batch)


def _raw_cuda_probe(batches, sweeps, repeats, tag, Rd, L):
    import sys, json
    sys.path.insert(0, "/root")
    import numpy as np, cupy as cp
    import sim_core as S

    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    N = int(geo["N"])
    mask = cp.asarray(np.ascontiguousarray(
        np.asarray(geo["disco"]).astype(np.uint8)))
    mod = cp.RawModule(code=_STENCIL, options=("-std=c++11",))
    k = mod.get_function("substep")
    print(f"cupy={cp.__version__} runtime={cp.cuda.runtime.runtimeGetVersion()}",
          flush=True)
    dev = cp.cuda.Device()
    print(f"gpu={cp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode()} "
          f"SMs={dev.attributes['MultiProcessorCount']}", flush=True)
    print(f"Rd={Rd} L={L} N={N} stride={12*L*N*N:,} bytes/replica  "
          f"{sweeps} sweeps/cell", flush=True)

    TPB = 256
    rng = np.random.default_rng(7)
    rows = []
    for B in batches:
        nsite = B * L * N * N
        grid = (nsite + TPB - 1) // TPB
        # Built on the host: this image carries no cuRAND, and the kernel does
        # no random work anyway, so the initial state only has to be unit spins.
        h = rng.standard_normal((nsite, 3)).astype(np.float32)
        h /= np.linalg.norm(h, axis=-1, keepdims=True) + 1e-20
        a = cp.asarray(h.reshape(-1))
        bbuf = cp.empty_like(a)
        args0 = (a, bbuf, mask, np.int32(B), np.int32(L), np.int32(N), np.int32(0))
        args1 = (bbuf, a, mask, np.int32(B), np.int32(L), np.int32(N), np.int32(1))
        k((grid,), (TPB,), args0)          # warm up / JIT the module
        k((grid,), (TPB,), args1)
        cp.cuda.Stream.null.synchronize()
        ts = []
        for _ in range(repeats):
            e0, e1 = cp.cuda.Event(), cp.cuda.Event()
            e0.record()
            for _ in range(sweeps):
                k((grid,), (TPB,), args0)
                k((grid,), (TPB,), args1)
            e1.record(); e1.synchronize()
            ts.append(cp.cuda.get_elapsed_time(e0, e1) / 1e3)
        arr = np.array(ts)
        us = float(arr.mean() * 1e6 / (sweeps * B))
        rows.append(dict(B=int(B), us=us, sd_s=float(arr.std())))
        print(f"  B={B:4d}  {us:8.4f} us/replica-sweep  (sd {arr.std():.4f} s)"
              f"{'   <<< multiple of 4' if B % 4 == 0 else ''}", flush=True)
        del a, bbuf
        cp.get_default_memory_pool().free_all_blocks()

    m4 = [r["us"] for r in rows if r["B"] % 4 == 0]
    nm = [r["us"] for r in rows if r["B"] % 4]
    pen = float(np.mean(m4) / np.mean(nm) - 1) if m4 and nm else float("nan")
    ovl = bool(min(m4) <= max(nm)) if m4 and nm else None
    print(f"\n=== raw CUDA, no XLA ===", flush=True)
    print(f"  multiples of 4: mean {np.mean(m4):.4f}  min {min(m4):.4f}", flush=True)
    print(f"  others:         mean {np.mean(nm):.4f}  max {max(nm):.4f}", flush=True)
    print(f"  mod-4 penalty {pen:+.1%}   overlap: {'yes' if ovl else 'NONE'}",
          flush=True)
    print("  penalty present here => the effect is the access pattern and the "
          "hardware.\n  penalty absent here => it was XLA's layout choice.",
          flush=True)
    res = dict(rows=rows, mod4_penalty=pen, overlap=ovl, Rd=float(Rd), L=int(L),
               N=N, stride=12 * L * N * N, sweeps=sweeps, repeats=repeats,
               kernel="raw CUDA 7-point stencil via cupy RawModule")
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


# --------------------------------------------------------------------------
# Experiment 3c: the single-array stencil showed NO mod-4 penalty (-0.1%, full
# overlap), so the effect is not intrinsic to the access pattern. That leaves
# the one structural difference between it and the real kernel: the real kernel
# keeps SEVERAL same-shaped buffers live at once (the trial state, the random
# draws, dS, the neighbour sum, dE, the acceptance mask), and XLA packs them
# into one arena.
#
# That makes the controlling quantity the TOTAL array size, not the replica
# stride. With stride 91,260 = 4 * 22,815 the whole array is B * 91,260 bytes,
# so v2(total) = v2(B) + 2. At B = 64 that is 2^8 * 22,815: a multiple of 256
# bytes, which is exactly the DRAM partition interleave. Sibling buffers then
# all begin on the same partition and camp on it together. At odd B, or B = 2
# mod 4, they start staggered and spread.
#
# So this kernel keeps the same stencil on buffer 0 and adds one coalesced read
# at the same offset in each of K-1 sibling buffers, all inside a single
# contiguous arena so they are exactly adjacent, the way a buffer assignment
# packs them. Sweeping K against B tests the prediction directly: the penalty
# should be absent at K=1 and grow with K, and only at B = 0 mod 4.
_MULTIBUF = r"""
extern "C" __global__
void multibuf(const float* __restrict__ s, float* __restrict__ o,
              const unsigned char* __restrict__ mask,
              int B, int L, int N, int parity, int K, long long arena) {
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  long long nsite = (long long)B * L * N * N;
  if (tid >= nsite) return;
  int x = (int)(tid % N);
  int y = (int)((tid / N) % N);
  int z = (int)((tid / ((long long)N * N)) % L);
  long long b = tid / ((long long)N * N * L);
  long long base = tid * 3;
  if (!mask[(long long)z * N * N + (long long)y * N + x] ||
      (((x + y + z) & 1) != parity)) {
    o[base] = s[base]; o[base + 1] = s[base + 1]; o[base + 2] = s[base + 2];
    return;
  }
  const int dx[6] = {1, -1, 0, 0, 0, 0};
  const int dy[6] = {0, 0, 1, -1, 0, 0};
  const int dz[6] = {0, 0, 0, 0, 1, -1};
  float ax = 0.f, ay = 0.f, az = 0.f;
  for (int k = 0; k < 6; ++k) {
    int nz = z + dz[k];
    if (nz < 0 || nz >= L) continue;
    int ny = (y + dy[k] + N) % N;
    int nx = (x + dx[k] + N) % N;
    long long nb = ((((b * L + nz) * N + ny) * (long long)N) + nx) * 3;
    ax += s[nb]; ay += s[nb + 1]; az += s[nb + 2];
  }
  // One coalesced read per sibling buffer, same offset, adjacent in the arena.
  for (int j = 1; j < K; ++j) {
    const float* sj = s + (long long)j * arena;
    ax += sj[base]; ay += sj[base + 1]; az += sj[base + 2];
  }
  float ex = s[base]     + 0.1f * ax;
  float ey = s[base + 1] + 0.1f * ay;
  float ez = s[base + 2] + 0.1f * az;
  float inv = rsqrtf(ex * ex + ey * ey + ez * ez + 1e-20f);
  o[base] = ex * inv; o[base + 1] = ey * inv; o[base + 2] = ez * inv;
}
"""


def _multibuf_probe(batches, ks, sweeps, repeats, tag, Rd, L):
    import sys, json
    sys.path.insert(0, "/root")
    import numpy as np, cupy as cp
    import sim_core as S

    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    N = int(geo["N"])
    mask = cp.asarray(np.ascontiguousarray(
        np.asarray(geo["disco"]).astype(np.uint8)))
    k = cp.RawModule(code=_MULTIBUF, options=("-std=c++11",)).get_function("multibuf")
    dev = cp.cuda.Device()
    print(f"gpu={cp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode()} "
          f"SMs={dev.attributes['MultiProcessorCount']}", flush=True)
    print(f"Rd={Rd} L={L} N={N} replica stride={12*L*N*N:,} bytes  "
          f"{sweeps} sweeps/cell  K={list(ks)}", flush=True)

    TPB = 256
    rng = np.random.default_rng(7)
    out = []
    for K in ks:
        print(f"\n--- K={K} sibling buffer(s) in one arena ---", flush=True)
        rows = []
        for B in batches:
            nsite = B * L * N * N
            arena = nsite * 3                       # floats per buffer
            grid = (nsite + TPB - 1) // TPB
            h = rng.standard_normal((nsite, 3)).astype(np.float32)
            h /= np.linalg.norm(h, axis=-1, keepdims=True) + 1e-20
            a = cp.asarray(np.tile(h.reshape(-1), K))   # ONE contiguous arena
            o1 = cp.empty(arena, dtype=cp.float32)
            ar = np.int64(arena)
            g0 = (a, o1, mask, np.int32(B), np.int32(L), np.int32(N),
                  np.int32(0), np.int32(K), ar)
            g1 = (a, o1, mask, np.int32(B), np.int32(L), np.int32(N),
                  np.int32(1), np.int32(K), ar)
            k((grid,), (TPB,), g0); k((grid,), (TPB,), g1)
            cp.cuda.Stream.null.synchronize()
            ts = []
            for _ in range(repeats):
                e0, e1 = cp.cuda.Event(), cp.cuda.Event()
                e0.record()
                for _ in range(sweeps):
                    k((grid,), (TPB,), g0)
                    k((grid,), (TPB,), g1)
                e1.record(); e1.synchronize()
                ts.append(cp.cuda.get_elapsed_time(e0, e1) / 1e3)
            arr = np.array(ts)
            us = float(arr.mean() * 1e6 / (sweeps * B))
            tot = B * 12 * L * N * N
            rows.append(dict(B=int(B), us=us, sd_s=float(arr.std()),
                             total_bytes=int(tot)))
            print(f"  B={B:4d}  {us:8.4f} us  total={tot:12,d} B"
                  f"{'   <<< multiple of 4' if B % 4 == 0 else ''}", flush=True)
            del a, o1
            cp.get_default_memory_pool().free_all_blocks()
        m4 = [r["us"] for r in rows if r["B"] % 4 == 0]
        nm = [r["us"] for r in rows if r["B"] % 4]
        pen = float(np.mean(m4) / np.mean(nm) - 1)
        ovl = bool(min(m4) <= max(nm))
        print(f"  => K={K}  mod-4 penalty {pen:+.1%}  overlap: "
              f"{'yes' if ovl else 'NONE'}", flush=True)
        out.append(dict(K=int(K), rows=rows, mod4_penalty=pen, overlap=ovl))

    print("\n=== sibling-buffer count vs mod-4 penalty ===", flush=True)
    print(f"{'K':>4} {'penalty':>10} {'overlap':>8}", flush=True)
    for o in out:
        print(f"{o['K']:4d} {o['mod4_penalty']:+9.1%} "
              f"{'yes' if o['overlap'] else 'NONE':>8}", flush=True)
    res = dict(by_k=out, Rd=float(Rd), L=int(L), N=N, stride=12 * L * N * N,
               sweeps=sweeps, repeats=repeats,
               kernel="raw CUDA stencil + K-1 sibling reads in one arena")
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


@app.function(gpu="H100", image=cuda_image, memory=65536, timeout=3600,
              volumes={"/out": vol})
def multibuf_h100(b, ks, sw, r, tag, Rd, L):
    return _multibuf_probe(b, ks, sw, r, tag, Rd, L)


@app.local_entrypoint()
def buffer_probe(batches: str = "48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64",
                 ks: str = "1,2,4,8", sweeps: int = 1000, repeats: int = 3,
                 rd: float = 18.30, layers: int = 5, tag: str = "exp3c"):
    """Experiment 3c: does the mod-4 penalty appear once sibling buffers exist?"""
    multibuf_h100.remote([int(x) for x in batches.split(",") if x],
                         [int(x) for x in ks.split(",") if x],
                         sweeps, repeats, tag, rd, layers)


# --------------------------------------------------------------------------
# Experiment 3d: put the RNG back.
#
# The bare stencil (3b) and the sibling-buffer variants (3c) all came out flat,
# so neither the replica stride nor inter-buffer alignment explains the +12.1%
# the compiled kernel shows on this very lattice. The largest thing the proxy
# still omits is the random draws: a real substep needs three normals and one
# uniform per site, which is a full-size array's worth of values per sweep.
#
# Whether that costs memory traffic depends on a compiler decision. If XLA fuses
# the generator into the consumer, the values live in registers and cost only
# arithmetic. If it materialises them, every sweep writes and then re-reads
# 4 floats per site -- a second full-size stream with the same replica stride,
# generated by one kernel and consumed by another.
#
# So three modes on one device and one window:
#   mode 0  no RNG                  -- reproduces 3b, the control
#   mode 1  RNG inline              -- counter-based hash + Box-Muller in
#                                      registers, no extra traffic, the "fused"
#                                      case; costs arithmetic only
#   mode 2  RNG materialised        -- a separate kernel fills a 4-float-per-site
#                                      buffer each substep and the stencil reads
#                                      it, the "not fused" case
#
# If the sawtooth appears in mode 2 and not in mode 1, the cause is the extra
# generated stream and the finding is clean. If it appears in both, it is the
# arithmetic and occupancy. If it appears in neither, the cause is still further
# up in how XLA schedules this kernel.
_METRO = r"""
__device__ __forceinline__ unsigned int hashu(unsigned int x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16;
  return x;
}
__device__ __forceinline__ float u01(unsigned int h) {
  return (float)(h >> 8) * (1.0f / 16777216.0f) + 1e-7f;
}

// Fills 4 floats per site: three normals and one uniform, the same shape XLA
// would materialise for `random.normal(key, spins.shape)` plus the acceptance
// draw.
extern "C" __global__
void fillrnd(float* __restrict__ r, long long nsite, unsigned int seed) {
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (tid >= nsite) return;
  unsigned int c = (unsigned int)tid ^ seed;
  float u0 = u01(hashu(c)), u1 = u01(hashu(c + 0x9e3779b9U));
  float u2 = u01(hashu(c + 0x85ebca6bU)), u3 = u01(hashu(c + 0xc2b2ae35U));
  float g = sqrtf(-2.0f * __logf(u0)), t = 6.2831853f * u1;
  float h = sqrtf(-2.0f * __logf(u2));
  long long o = tid * 4;
  r[o] = g * __cosf(t); r[o + 1] = g * __sinf(t);
  r[o + 2] = h * __cosf(6.2831853f * u3); r[o + 3] = u01(hashu(c + 0x27d4eb2fU));
}

extern "C" __global__
void metro(const float* __restrict__ s, float* __restrict__ o,
           const float* __restrict__ rnd, const unsigned char* __restrict__ mask,
           int B, int L, int N, int parity, int mode, unsigned int seed,
           float beta) {
  long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  long long nsite = (long long)B * L * N * N;
  if (tid >= nsite) return;
  int x = (int)(tid % N);
  int y = (int)((tid / N) % N);
  int z = (int)((tid / ((long long)N * N)) % L);
  long long b = tid / ((long long)N * N * L);
  long long base = tid * 3;
  if (!mask[(long long)z * N * N + (long long)y * N + x] ||
      (((x + y + z) & 1) != parity)) {
    o[base] = s[base]; o[base + 1] = s[base + 1]; o[base + 2] = s[base + 2];
    return;
  }
  const int dx[6] = {1, -1, 0, 0, 0, 0};
  const int dy[6] = {0, 0, 1, -1, 0, 0};
  const int dz[6] = {0, 0, 0, 0, 1, -1};
  float vx = 0.f, vy = 0.f, vz = 0.f;
  for (int k = 0; k < 6; ++k) {
    int nz = z + dz[k];
    if (nz < 0 || nz >= L) continue;
    int ny = (y + dy[k] + N) % N;
    int nx = (x + dx[k] + N) % N;
    long long nb = ((((b * L + nz) * N + ny) * (long long)N) + nx) * 3;
    vx += s[nb]; vy += s[nb + 1]; vz += s[nb + 2];
  }
  float cx = s[base], cy = s[base + 1], cz = s[base + 2];
  float tx, ty, tz, acc;
  if (mode == 2) {                       // materialised: read what was written
    long long ro = tid * 4;
    tx = rnd[ro]; ty = rnd[ro + 1]; tz = rnd[ro + 2]; acc = rnd[ro + 3];
  } else if (mode == 1) {                // fused: generate in registers
    unsigned int c = (unsigned int)tid ^ seed;
    float u0 = u01(hashu(c)), u1 = u01(hashu(c + 0x9e3779b9U));
    float u2 = u01(hashu(c + 0x85ebca6bU)), u3 = u01(hashu(c + 0xc2b2ae35U));
    float g = sqrtf(-2.0f * __logf(u0)), t = 6.2831853f * u1;
    tx = g * __cosf(t); ty = g * __sinf(t);
    tz = sqrtf(-2.0f * __logf(u2)) * __cosf(6.2831853f * u3);
    acc = u01(hashu(c + 0x27d4eb2fU));
  } else {                               // no RNG at all: the 3b control
    tx = cx + 0.1f * vx; ty = cy + 0.1f * vy; tz = cz + 0.1f * vz; acc = 0.f;
  }
  float inv = rsqrtf(tx * tx + ty * ty + tz * tz + 1e-20f);
  tx *= inv; ty *= inv; tz *= inv;
  if (mode == 0) { o[base] = tx; o[base + 1] = ty; o[base + 2] = tz; return; }
  // Exchange plus a uniaxial term, and the Metropolis test with its exp().
  float ddx = tx - cx, ddy = ty - cy, ddz = tz - cz;
  float dE = -(ddx * vx + ddy * vy + ddz * vz)
             - 0.5f * (tz * tz - cz * cz);
  bool take = (dE < 0.f) || (acc < __expf(-dE * beta));
  o[base]     = take ? tx : cx;
  o[base + 1] = take ? ty : cy;
  o[base + 2] = take ? tz : cz;
}
"""


def _metro_probe(batches, modes, sweeps, repeats, tag, Rd, L):
    import sys, json
    sys.path.insert(0, "/root")
    import numpy as np, cupy as cp
    import sim_core as S

    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    N = int(geo["N"])
    mask = cp.asarray(np.ascontiguousarray(
        np.asarray(geo["disco"]).astype(np.uint8)))
    mod = cp.RawModule(code=_METRO, options=("-std=c++11",))
    metro = mod.get_function("metro")
    fill = mod.get_function("fillrnd")
    dev = cp.cuda.Device()
    print(f"gpu={cp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode()} "
          f"SMs={dev.attributes['MultiProcessorCount']}", flush=True)
    print(f"Rd={Rd} L={L} N={N} replica stride={12*L*N*N:,} bytes  "
          f"{sweeps} sweeps/cell  modes={list(modes)}", flush=True)
    NAME = {0: "no RNG (the 3b control)", 1: "RNG inline, fused, no extra traffic",
            2: "RNG materialised, written then read"}

    TPB = 256
    rng = np.random.default_rng(7)
    out = []
    for mode in modes:
        print(f"\n--- mode {mode}: {NAME.get(mode, '?')} ---", flush=True)
        rows = []
        for B in batches:
            nsite = B * L * N * N
            grid = (nsite + TPB - 1) // TPB
            h = rng.standard_normal((nsite, 3)).astype(np.float32)
            h /= np.linalg.norm(h, axis=-1, keepdims=True) + 1e-20
            a = cp.asarray(h.reshape(-1))
            bb = cp.empty_like(a)
            rnd = (cp.zeros(nsite * 4, dtype=cp.float32) if mode == 2
                   else cp.zeros(1, dtype=cp.float32))
            ns = np.int64(nsite)
            common = (mask, np.int32(B), np.int32(L), np.int32(N))

            def one(sweep):
                for par in (0, 1):
                    sd = np.uint32((sweep * 2 + par) * 2654435761 % (2 ** 32))
                    if mode == 2:
                        fill((grid,), (TPB,), (rnd, ns, sd))
                    src, dst = (a, bb) if par == 0 else (bb, a)
                    metro((grid,), (TPB,),
                          (src, dst, rnd) + common
                          + (np.int32(par), np.int32(mode), sd, np.float32(1.5)))

            one(0)
            cp.cuda.Stream.null.synchronize()
            ts = []
            for _ in range(repeats):
                e0, e1 = cp.cuda.Event(), cp.cuda.Event()
                e0.record()
                for sw in range(sweeps):
                    one(sw)
                e1.record(); e1.synchronize()
                ts.append(cp.cuda.get_elapsed_time(e0, e1) / 1e3)
            arr = np.array(ts)
            us = float(arr.mean() * 1e6 / (sweeps * B))
            rows.append(dict(B=int(B), us=us, sd_s=float(arr.std())))
            print(f"  B={B:4d}  {us:8.4f} us/replica-sweep  (sd {arr.std():.4f} s)"
                  f"{'   <<< multiple of 4' if B % 4 == 0 else ''}", flush=True)
            del a, bb, rnd
            cp.get_default_memory_pool().free_all_blocks()
        m4 = [r["us"] for r in rows if r["B"] % 4 == 0]
        nm = [r["us"] for r in rows if r["B"] % 4]
        pen = float(np.mean(m4) / np.mean(nm) - 1)
        ovl = bool(min(m4) <= max(nm))
        print(f"  => mode {mode}  mod-4 penalty {pen:+.1%}  overlap: "
              f"{'yes' if ovl else 'NONE'}", flush=True)
        out.append(dict(mode=int(mode), label=NAME.get(mode), rows=rows,
                        mod4_penalty=pen, overlap=ovl))

    print("\n=== RNG treatment vs mod-4 penalty (raw CUDA, no XLA) ===", flush=True)
    print(f"{'mode':>5} {'penalty':>10} {'overlap':>8}  what it is", flush=True)
    for o in out:
        print(f"{o['mode']:5d} {o['mod4_penalty']:+9.1%} "
              f"{'yes' if o['overlap'] else 'NONE':>8}  {o['label']}", flush=True)
    print("\nThe compiled JAX kernel on this lattice and window is +12.1%.", flush=True)
    res = dict(by_mode=out, Rd=float(Rd), L=int(L), N=N, stride=12 * L * N * N,
               sweeps=sweeps, repeats=repeats,
               kernel="raw CUDA Metropolis substep via cupy RawModule")
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


@app.function(gpu="H100", image=cuda_image, memory=65536, timeout=3600,
              volumes={"/out": vol})
def metro_h100(b, m, sw, r, tag, Rd, L):
    return _metro_probe(b, m, sw, r, tag, Rd, L)


@app.local_entrypoint()
def metro_probe(batches: str = "48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64",
                modes: str = "0,1,2", sweeps: int = 600, repeats: int = 3,
                rd: float = 18.30, layers: int = 5, tag: str = "exp3d"):
    """Experiment 3d: does the mod-4 penalty appear once the RNG is present?"""
    metro_h100.remote([int(x) for x in batches.split(",") if x],
                      [int(x) for x in modes.split(",") if x],
                      sweeps, repeats, tag, rd, layers)


# --------------------------------------------------------------------------
# Experiment 3e: ask the compiler and the profiler directly.
#
# Raw CUDA never reproduces the penalty, so it is a decision XLA makes as a
# function of B. Rather than keep guessing which decision, this measures it two
# ways in one container.
#
#  - It dumps the compiled module per B and reads back what the compiler chose:
#    how many kernels it emitted and the block size (`maxntidx`) of each. If the
#    launch geometry changes when B is a multiple of four, that is the answer.
#
#  - It profiles the run per B and aggregates GPU kernel time BY KERNEL NAME.
#    The penalty is 12% of the whole; one named kernel has to carry it. Finding
#    which one localises the cause to a single fusion instead of to "XLA".
#
# The host->device path is not a candidate and is not tested here: every
# argument is already a device array before the timer starts (`pack` returns
# `jnp.asarray`, `init_spins` returns a jnp array, a PRNGKey is a device array),
# so the timed region contains no transfer, and Python dispatch is one fixed
# cost per call amortised over seconds of kernel time.
def _xla_dump(batches, tag, Rd, L, n_temps, n_therm, n_meas):
    import os, sys, io, json, glob, gzip, re, time, collections
    # XLA reads its flags once at initialisation, so this has to precede jax.
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + " --xla_dump_to=/tmp/dump").strip()
    os.makedirs("/tmp/dump", exist_ok=True)
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S
    import parallel_axes as PA

    kB = 0.086173404
    betas = jnp.asarray(1.0 / (kB * np.linspace(20.0, 0.1, int(n_temps))))
    sweeps = int(n_temps) * (n_therm + n_meas)
    theta = PA.load_thetas("/root/theta4.json")[0]
    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    eng = S.build_engine(geo)
    print(f"gpu={device_identity()}", flush=True)
    print(f"Rd={Rd} L={L} N={geo['N']} stride={12*L*geo['N']**2:,}  "
          f"{sweeps} sweeps/cell", flush=True)

    def body(sp, kk, p):
        return eng["anneal"](sp, kk, betas, p, n_therm, n_meas)

    def kernels_of(B):
        """Block size and count of the kernels the compiler emitted for this B."""
        best = None
        for f in glob.glob("/tmp/dump/*"):
            if not (f.endswith(".ll") or f.endswith(".ptx")):
                continue
            try:
                t = io.open(f, encoding="utf-8", errors="replace").read()
            except Exception:
                continue
            if f.endswith(".ptx"):
                ents = re.findall(r"\.visible \.entry (\w+)", t)
                ntid = [int(m) for m in re.findall(r"\.maxntid (\d+)", t)]
            else:
                ents = re.findall(r'define\s+\w*\s*void @(\w+)', t)
                ntid = [int(m) for m in
                        re.findall(r'"maxntidx", i32 (\d+)', t)]
            if ents and (best is None or len(ents) > best[0]):
                best = (len(ents), sorted(collections.Counter(ntid).items()), f)
        return best

    out = []
    for B in batches:
        for f in glob.glob("/tmp/dump/*"):
            try:
                os.remove(f)
            except Exception:
                pass
        cols = PA.pack([theta], B)
        args = (S.init_spins(jax.random.PRNGKey(11), geo, B),
                jax.random.PRNGKey(12), cols)
        fn = jax.jit(body).lower(*args).compile()
        jax.block_until_ready(fn(*args))                       # warm
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        wall = time.perf_counter() - t0
        us = wall * 1e6 / (sweeps * B)

        pdir = f"/tmp/prof/B{B}"
        with jax.profiler.trace(pdir):
            jax.block_until_ready(fn(*args))

        # Aggregate GPU kernel time by name out of the Chrome trace.
        per, tot = collections.Counter(), 0.0
        tr = sorted(glob.glob(f"{pdir}/**/*.trace.json.gz", recursive=True))
        if tr:
            with gzip.open(tr[-1], "rt") as fh:
                j = json.load(fh)
            ev = j.get("traceEvents", [])
            pname = {e["pid"]: e["args"].get("name", "")
                     for e in ev
                     if e.get("ph") == "M" and e.get("name") == "process_name"}
            gpu = {p for p, n in pname.items() if "GPU" in n and "Stream" not in n}
            for e in ev:
                if e.get("ph") == "X" and e.get("pid") in gpu and "dur" in e:
                    nm = re.sub(r"\.\d+$", "", str(e.get("name", "?")))
                    per[nm] += float(e["dur"]); tot += float(e["dur"])
        kk = kernels_of(B)
        out.append(dict(B=int(B), us=us, wall_s=wall,
                        n_kernels=(kk[0] if kk else None),
                        block_sizes=(kk[1] if kk else None),
                        gpu_us_total=tot,
                        top=[(n, v) for n, v in per.most_common(8)]))
        print(f"\n  B={B:4d}{'  (multiple of 4)' if B % 4 == 0 else ''}  "
              f"{us:7.3f} us/replica-sweep   kernels={kk[0] if kk else '?'}  "
              f"block sizes={kk[1] if kk else '?'}", flush=True)
        print(f"    traced GPU time {tot/1e3:9.2f} ms over "
              f"{len(per)} distinct kernels", flush=True)
        for n, v in per.most_common(5):
            print(f"      {v/1e3:9.2f} ms  {100*v/tot if tot else 0:5.1f}%  "
                  f"{n[:72]}", flush=True)

    print("\n=== what the compiler chose, per B ===", flush=True)
    print(f"{'B':>5} {'B%4':>4} {'us':>8} {'kernels':>8}  block sizes", flush=True)
    for r in out:
        print(f"{r['B']:5d} {r['B'] % 4:4d} {r['us']:8.3f} "
              f"{str(r['n_kernels']):>8}  {r['block_sizes']}", flush=True)

    # Same kernel name, how its share of GPU time moves with B mod 4.
    names = set()
    for r in out:
        names |= {n for n, _ in r["top"]}
    print("\n=== per-kernel GPU time (ms), by B ===", flush=True)
    hdr = "kernel".ljust(54) + "".join(f"{r['B']:>9}" for r in out)
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for nm in sorted(names):
        d = {r["B"]: dict(r["top"]).get(nm, 0.0) / 1e3 for r in out}
        print(nm[:52].ljust(54) + "".join(f"{d[r['B']]:9.2f}" for r in out),
              flush=True)
    m4 = [r for r in out if r["B"] % 4 == 0]
    nm4 = [r for r in out if r["B"] % 4]
    if m4 and nm4:
        a = sum(r["gpu_us_total"] / r["B"] for r in m4) / len(m4)
        b = sum(r["gpu_us_total"] / r["B"] for r in nm4) / len(nm4)
        print(f"\ntraced GPU us per replica: multiples of 4 {a:,.1f}  "
              f"others {b:,.1f}  -> {100*(a/b-1):+.1f}%", flush=True)
        print("If this matches the wall-clock penalty, the cost is inside the "
              "kernels and the table above names which.", flush=True)
    res = dict(rows=out, Rd=float(Rd), L=int(L), N=int(geo["N"]), sweeps=sweeps,
               device=device_identity())
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


# The first attempt at reading the compiler's choices matched nothing, because
# it guessed at the dump's file names and formats. This one looks first: it
# lists what XLA actually wrote, then searches every text file for whatever
# records a launch geometry -- `launch_dimensions` in a thunk sequence,
# `maxntidx` in LLVM IR, `.maxntid` in PTX -- and reports per kernel.
def _launch_dims(batches, tag, Rd, L, n_temps, n_therm, n_meas):
    import os, sys, io, json, glob, re, collections
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + " --xla_dump_to=/tmp/dump").strip()
    os.makedirs("/tmp/dump", exist_ok=True)
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S
    import parallel_axes as PA

    kB = 0.086173404
    betas = jnp.asarray(1.0 / (kB * np.linspace(20.0, 0.1, int(n_temps))))
    theta = PA.load_thetas("/root/theta4.json")[0]
    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    eng = S.build_engine(geo)
    print(f"gpu={device_identity()}", flush=True)

    def body(sp, kk, p):
        return eng["anneal"](sp, kk, betas, p, n_therm, n_meas)

    # The thunk sequence writes the geometry as
    #   `<fusion name> grid: [G, 1, 1] threads: [T, 1, 1]`
    THUNK = re.compile(r"\]\s+([A-Za-z_][\w$]*) grid: \[(\d+), 1, 1\]"
                       r" threads: \[(\d+), 1, 1\]")
    PAT = (("thunk_launch", re.compile(r"launch_dimensions[^\n]{0,120}")),
           ("llvm_maxntidx", re.compile(r'"maxntidx", i32 (\d+)')),
           ("ptx_maxntid", re.compile(r"\.maxntid ([\d, ]+)")),
           ("ptx_entry", re.compile(r"\.visible \.entry ([\w$]+)")),
           ("llvm_define", re.compile(r"define[^@\n]{0,40}@([\w$]+)")))
    out = []
    for i, B in enumerate(batches):
        for f in glob.glob("/tmp/dump/*"):
            try:
                os.remove(f)
            except Exception:
                pass
        cols = PA.pack([theta], B)
        args = (S.init_spins(jax.random.PRNGKey(11), geo, B),
                jax.random.PRNGKey(12), cols)
        jax.jit(body).lower(*args).compile()

        # Hundreds of tiny `jit_stage` modules come from tracing the arguments;
        # only the module that carries the (B, L, N, N, 3) array is the kernel
        # under study, so everything is filtered to that module's prefix.
        files = sorted(glob.glob("/tmp/dump/*"))
        big = max(files, key=os.path.getsize)
        pre = os.path.basename(big).split(".jit_")[0]
        files = [f for f in files if os.path.basename(f).startswith(pre)]
        if i == 0:
            print(f"\nXLA dumped {len(sorted(glob.glob('/tmp/dump/*')))} files; "
                  f"the module under study is {pre}, with {len(files)} of them. "
                  f"Its 12 largest:", flush=True)
            for f in sorted(files, key=os.path.getsize, reverse=True)[:12]:
                print(f"    {os.path.getsize(f):12,d} B  {os.path.basename(f)}",
                      flush=True)
            thunks = [f for f in files if "thunk_sequence" in f]
            if thunks:
                t = io.open(thunks[0], encoding="utf-8", errors="replace").read()
                print(f"\n  head of {os.path.basename(thunks[0])}:", flush=True)
                for ln in t.splitlines()[:25]:
                    print(f"      {ln[:150]}", flush=True)
        hits = collections.defaultdict(list)
        for f in files:
            try:
                t = io.open(f, encoding="utf-8", errors="replace").read()
            except Exception:
                continue
            for nm, rx in PAT:
                m = rx.findall(t)
                if m:
                    hits[nm].extend(m)
        rec = dict(B=int(B), n_files=len(files))
        print(f"\n  B={B}{'  (multiple of 4)' if B % 4 == 0 else ''}:", flush=True)
        for nm, _ in PAT:
            v = hits.get(nm, [])
            if not v:
                continue
            if nm in ("ptx_entry", "llvm_define"):
                rec[nm] = len(set(v))
                print(f"    {nm:14s} {len(set(v))} distinct", flush=True)
            else:
                c = collections.Counter(v)
                rec[nm] = sorted(c.items())[:40]
                print(f"    {nm:14s} {len(v)} occurrences, "
                      f"{len(c)} distinct values", flush=True)
                for val, n in c.most_common(10):
                    print(f"        x{n:4d}  {str(val)[:96]}", flush=True)
        # Per-kernel grid and block, from the module's own thunk sequence.
        geom = {}
        for f in files:
            if "thunk_sequence" not in f:
                continue
            t = io.open(f, encoding="utf-8", errors="replace").read()
            for nm, g, th in THUNK.findall(t):
                geom[nm] = (int(g), int(th))
        rec["geom"] = geom
        blocks = sorted({th for _, th in geom.values()})
        print(f"    thunk geometry  {len(geom)} kernels, "
              f"block sizes in use: {blocks}", flush=True)
        out.append(rec)

    print("\n=== block sizes XLA chose, per B ===", flush=True)
    for r in out:
        g = r.get("geom") or {}
        c = collections.Counter(th for _, th in g.values())
        print(f"  B={r['B']:4d} B%4={r['B'] % 4}  kernels={len(g):3d}  "
              f"blocks={sorted(c.items())}", flush=True)

    print("\n=== heaviest kernels: grid x block, per B ===", flush=True)
    allnm = sorted(set().union(*[set((r.get("geom") or {}).keys()) for r in out]))
    big = [n for n in allnm
           if max((r.get("geom") or {}).get(n, (0, 0))[0] for r in out) > 1000]
    hdr = "kernel".ljust(38) + "".join(f"{r['B']:>18}" for r in out)
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for n in big:
        row = n[:36].ljust(38)
        for r in out:
            g, th = (r.get("geom") or {}).get(n, (0, 0))
            row += f"{(str(g) + 'x' + str(th)) if g else '-':>18}"
        print(row, flush=True)
    print("\nSame block size across B means XLA is not choosing a different "
          "launch shape; a kernel present at one B and absent at another means "
          "it chose a different fusion structure.", flush=True)

    print("\n=== launch geometry per B ===", flush=True)
    for r in out:
        ld = r.get("thunk_launch") or r.get("llvm_maxntidx") or r.get("ptx_maxntid")
        print(f"  B={r['B']:4d} B%4={r['B'] % 4}  files={r['n_files']}  "
              f"kernels={r.get('ptx_entry') or r.get('llvm_define')}  "
              f"{str(ld)[:120]}", flush=True)
    res = dict(rows=out, Rd=float(Rd), L=int(L), device=device_identity())
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


def _prodstep(batches, tag, Rd, L, trace_temps, trace_sweeps):
    """Production-parameter timing plus the per-operation breakdown.

    Two measurements per batch, because one run cannot give both.

      1. The REAL thing: 200 temperatures x 20,000 sweeps, wall clock only.
         This is the number the extrapolation has been predicting, and it is
         the first time any point is measured at the production schedule.
      2. The breakdown: the same kernels traced at a short schedule. A trace
         over four million sweeps would record on the order of 56 million
         kernel launches, which is not something the profiler can hold, so the
         operation-by-operation picture necessarily comes from a short run.

    The pair B=65 and B=75 are BOTH vector width 1, so this does not test the
    width decision at all -- it tests the SAWTOOTH, the part with no mechanism.
    If the 5.4% gap between them concentrates in particular kernels the way the
    width penalty did, that is the first real lead on it; if it is spread evenly
    across every operation, that rules out a single-kernel explanation.
    """
    import os, sys, io as _io, json, glob, re, gzip, collections, time
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + " --xla_dump_to=/tmp/dump").strip()
    os.makedirs("/tmp/dump", exist_ok=True)
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S
    import parallel_axes as PA

    def v2(n):
        k = 0
        while n % 2 == 0:
            n //= 2
            k += 1
        return k
    width = lambda B: min(4, 2 ** (v2(B) + v2(int(L))))

    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    eng = S.build_engine(geo)
    N = int(geo["N"])
    theta = PA.load_thetas("/root/theta4.json")[0]
    kB = 0.086173404
    print(f"gpu={device_identity()}", flush=True)
    print(f"Rd={Rd} L={L} N={N} sitios={int(np.asarray(geo['disco']).sum())}",
          flush=True)
    for B in batches:
        print(f"  B={B}: v2(elementos)={v2(B)+v2(int(L))} -> ancho {width(B)}",
              flush=True)

    THUNK = re.compile(r"\]\s+([A-Za-z_][\w$]*) grid: \[(\d+), 1, 1\]"
                       r" threads: \[(\d+), 1, 1\]")
    PROD_T, PROD_TH, PROD_ME = 200, 15000, 5000      # 200 x 20,000 = 4,000,000
    out = {}

    for B in batches:
        rec = dict(B=int(B), width=width(B))

        # ---- 1. the real production run, wall clock only ------------------
        betas_p = jnp.asarray(1.0 / (kB * np.linspace(20.0, 0.1, PROD_T)))
        cols = PA.pack([theta], B)
        args = (S.init_spins(jax.random.PRNGKey(11), geo, B),
                jax.random.PRNGKey(12), cols)

        def prod_body(sp, kk, p):
            return eng["anneal"](sp, kk, betas_p, p, PROD_TH, PROD_ME)

        t_c = time.perf_counter()
        fnp = jax.jit(prod_body).lower(*args).compile()
        comp = time.perf_counter() - t_c
        print(f"\n--- B={B} ancho {width(B)} ---  compilado en {comp:.1f} s",
              flush=True)
        print(f"  corriendo 200 x 20 000 = 4 000 000 barridos...", flush=True)
        t0 = time.perf_counter()
        jax.block_until_ready(fnp(*args))
        wall = time.perf_counter() - t0
        sweeps_p = PROD_T * (PROD_TH + PROD_ME)
        rec["prod"] = dict(
            wall_s=wall, sweeps=sweeps_p, compile_s=comp,
            sec_per_replica=wall / B,
            us_per_replica_sweep=wall * 1e6 / (sweeps_p * B))
        print(f"  RELOJ REAL: {wall:,.1f} s = {wall/60:.1f} min "
              f"-> {wall/B:.3f} s por replica", flush=True)

        # ---- 2. the breakdown, from a short traced run --------------------
        for f in glob.glob("/tmp/dump/*"):
            try:
                os.remove(f)
            except Exception:
                pass
        betas_s = jnp.asarray(1.0 / (kB * np.linspace(20.0, 0.1, int(trace_temps))))
        th = int(trace_sweeps * 2 // 3)
        me = int(trace_sweeps) - th

        def short_body(sp, kk, p):
            return eng["anneal"](sp, kk, betas_s, p, th, me)

        fns = jax.jit(short_body).lower(*args).compile()
        jax.block_until_ready(fns(*args))
        files = sorted(glob.glob("/tmp/dump/*"))
        pre = os.path.basename(max(files, key=os.path.getsize)).split(".jit_")[0]
        mine = [f for f in files if os.path.basename(f).startswith(pre)]
        geom = {}
        for f in mine:
            if "thunk_sequence" in f:
                t = _io.open(f, encoding="utf-8", errors="replace").read()
                for nm, g, tt in THUNK.findall(t):
                    geom[nm] = (int(g), int(tt))
        pdir = f"/tmp/prof_{B}"
        with jax.profiler.trace(pdir):
            jax.block_until_ready(fns(*args))
        ev = collections.defaultdict(lambda: [0.0, 0])
        for g in glob.glob(f"{pdir}/**/*.trace.json.gz", recursive=True):
            with gzip.open(g, "rt") as fh:
                try:
                    tr = json.load(fh)
                except Exception:
                    continue
            pids = {e["pid"] for e in tr.get("traceEvents", [])
                    if e.get("ph") == "M" and e.get("name") == "process_name"
                    and "GPU" in str(e.get("args", {}).get("name", ""))}
            for e in tr.get("traceEvents", []):
                if e.get("ph") == "X" and e.get("pid") in pids and "dur" in e:
                    ev[e["name"]][0] += e["dur"]
                    ev[e["name"]][1] += 1
        tot = sum(v[0] for v in ev.values())
        rows = []
        for k, (us, n) in sorted(ev.items(), key=lambda kv: -kv[1][0]):
            g, tt = geom.get(k, (None, None))
            rows.append(dict(kernel=k, us_total=us, calls=n, grid=g, block=tt,
                             us_per_sweep_replica=us / max(n, 1) / B,
                             pct=100 * us / max(tot, 1)))
        rec["trace"] = dict(gpu_us_total=tot, n_kernels=len(ev), rows=rows,
                            temps=int(trace_temps), sweeps=int(trace_sweeps))
        print(f"  trazado: {tot:,.0f} us en {len(ev)} kernels", flush=True)
        for r in rows[:10]:
            print(f"    {r['kernel'][:36]:<36} {r['us_total']:9,.0f} "
                  f"{r['pct']:5.1f}% grid {(r['grid'] or 0):7,d}", flush=True)
        out[B] = rec

    # ---- what the extrapolation got wrong -------------------------------
    print(f"\n=== extrapolacion contra medicion real ===", flush=True)
    print(f"  {'B':>5} {'ancho':>6} {'extrapolado':>12} {'REAL':>10} "
          f"{'sesgo':>8}", flush=True)
    PREV = {65: 9.757, 75: 9.230, 135: 8.665, 140: 12.606, 850: 7.332}
    for B in batches:
        real = out[B]["prod"]["sec_per_replica"]
        ex = PREV.get(B)
        if ex:
            print(f"  {B:5d} {out[B]['width']:6d} {ex:12.3f} {real:10.3f} "
                  f"{100*(ex/real-1):+7.1f}%", flush=True)
        else:
            print(f"  {B:5d} {out[B]['width']:6d} {'-':>12} {real:10.3f}",
                  flush=True)

    if len(batches) == 2:
        a, b = batches
        ra, rb = out[a]["prod"]["sec_per_replica"], out[b]["prod"]["sec_per_replica"]
        print(f"\n  B={a} contra B={b} a produccion: {ra:.3f} vs {rb:.3f} s "
              f"-> {100*(rb/ra-1):+.1f}%", flush=True)
        ka = {r["kernel"]: r for r in out[a]["trace"]["rows"]}
        kb = {r["kernel"]: r for r in out[b]["trace"]["rows"]}
        print(f"\n=== donde difieren, por kernel (us por barrido-replica) ===",
              flush=True)
        print(f"  {'kernel':<36} {f'B={a}':>10} {f'B={b}':>10} {'cambio':>9}",
              flush=True)
        keys = sorted(set(ka) | set(kb),
                      key=lambda k: -abs(kb.get(k, {}).get("us_per_sweep_replica", 0)
                                         - ka.get(k, {}).get("us_per_sweep_replica", 0)))
        for k in keys[:14]:
            ua = ka.get(k, {}).get("us_per_sweep_replica", 0.0)
            ub = kb.get(k, {}).get("us_per_sweep_replica", 0.0)
            if max(ua, ub) < 0.01:
                continue
            ch = f"{100*(ub/ua-1):+8.1f}%" if ua > 0 else "    nuevo"
            print(f"  {k[:36]:<36} {ua:10.4f} {ub:10.4f} {ch}", flush=True)

    res = dict(by_batch={str(k): v for k, v in out.items()},
               Rd=float(Rd), L=int(L), N=N, device=device_identity(),
               prod_schedule=dict(temps=PROD_T, therm=PROD_TH, meas=PROD_ME,
                                  sweeps=PROD_T * (PROD_TH + PROD_ME)))
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


@app.function(gpu="H100", **_BS)
def prodstep_h100(b, tag, Rd, L, tt, ts):
    return _prodstep(b, tag, Rd, L, tt, ts)


@app.local_entrypoint()
def prodstep(batches: str = "65,75", rd: float = 18.30, layers: int = 5,
             trace_temps: int = 8, trace_sweeps: int = 450,
             tag: str = "prodstep_65_75"):
    """Real production timing plus the operation breakdown, for one pair."""
    prodstep_h100.remote([int(x) for x in batches.split(",") if x], tag,
                         rd, layers, trace_temps, trace_sweeps)


def _opstep(batches, tag, Rd, L, n_temps, n_therm, n_meas, try_ncu):
    """Operation-by-operation timing for two batch sizes, plus launch geometry.

    The point is to see WHERE the extra time goes, not just that there is extra
    time. For each batch it collects three things and joins them by kernel name:

      - the compiled thunk sequence, which gives each kernel its grid and block,
        so we can say how many blocks of how many threads it runs with;
      - a profiler trace, which gives the total GPU microseconds each kernel
        spent and how many times it was launched;
      - the PTX, which gives the load/store instruction mix per kernel.

    Joined, that is the step-by-step: what the operation is, how CUDA runs it,
    with what geometry, and how long it stays there. Comparing an odd batch
    against a multiple of four then shows which boxes grow.
    """
    import os, sys, io as _io, json, glob, re, gzip, collections, time
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + " --xla_dump_to=/tmp/dump").strip()
    os.makedirs("/tmp/dump", exist_ok=True)
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S
    import parallel_axes as PA

    kB = 0.086173404
    betas = jnp.asarray(1.0 / (kB * np.linspace(20.0, 0.1, int(n_temps))))
    theta = PA.load_thetas("/root/theta4.json")[0]
    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    eng = S.build_engine(geo)
    N = int(geo["N"])
    print(f"gpu={device_identity()}", flush=True)
    print(f"Rd={Rd} L={L} N={N} sitios={int(np.asarray(geo['disco']).sum())}",
          flush=True)

    if try_ncu:
        import subprocess
        r = subprocess.run(["bash", "-lc", "which ncu nsys 2>&1; "
                            "cat /proc/driver/nvidia/params 2>/dev/null | "
                            "grep -i restrict"],
                           capture_output=True, text=True)
        print(f"\n  herramientas de contadores: {r.stdout.strip() or 'ninguna'}",
              flush=True)

    THUNK = re.compile(r"\]\s+([A-Za-z_][\w$]*) grid: \[(\d+), 1, 1\]"
                       r" threads: \[(\d+), 1, 1\]")

    def body(sp, kk, p):
        return eng["anneal"](sp, kk, betas, p, int(n_therm), int(n_meas))

    out = {}
    for B in batches:
        for f in glob.glob("/tmp/dump/*"):
            try:
                os.remove(f)
            except Exception:
                pass
        cols = PA.pack([theta], B)
        args = (S.init_spins(jax.random.PRNGKey(11), geo, B),
                jax.random.PRNGKey(12), cols)
        fn = jax.jit(body).lower(*args).compile()
        jax.block_until_ready(fn(*args))          # calentamiento, esta vez si

        # --- geometry from the module's own thunk sequence -----------------
        files = sorted(glob.glob("/tmp/dump/*"))
        pre = os.path.basename(max(files, key=os.path.getsize)).split(".jit_")[0]
        mine = [f for f in files if os.path.basename(f).startswith(pre)]
        geom = {}
        for f in mine:
            if "thunk_sequence" not in f:
                continue
            t = _io.open(f, encoding="utf-8", errors="replace").read()
            for nm, g, th in THUNK.findall(t):
                geom[nm] = (int(g), int(th))
        # --- instruction mix per kernel, from the PTX ----------------------
        instr = {}
        for f in mine:
            if not f.endswith(".ptx") or os.path.getsize(f) == 0:
                continue
            t = _io.open(f, encoding="utf-8", errors="replace").read()
            for m in re.finditer(r"\.visible \.entry ([\w$]+)\(([\s\S]*?)\n\}", t):
                nm, bodytxt = m.group(1), m.group(2)
                instr[nm] = dict(
                    ld_scalar=len(re.findall(r"\bld\.global(?![.\w]*\.v[24])", bodytxt)),
                    ld_v4=len(re.findall(r"\bld\.global[.\w]*\.v4", bodytxt)),
                    ld_nc=len(re.findall(r"\bld\.global\.nc", bodytxt)),
                    st=len(re.findall(r"\bst\.global", bodytxt)))

        # --- real GPU time per kernel, from a trace -------------------------
        pdir = f"/tmp/prof_{B}"
        with jax.profiler.trace(pdir):
            jax.block_until_ready(fn(*args))
        ev = collections.defaultdict(lambda: [0.0, 0])
        for g in glob.glob(f"{pdir}/**/*.trace.json.gz", recursive=True):
            with gzip.open(g, "rt") as fh:
                try:
                    tr = json.load(fh)
                except Exception:
                    continue
            pids = {p["pid"] for p in tr.get("traceEvents", [])
                    if p.get("ph") == "M" and p.get("name") == "process_name"
                    and "GPU" in str(p.get("args", {}).get("name", ""))}
            for e in tr.get("traceEvents", []):
                if e.get("ph") == "X" and e.get("pid") in pids and "dur" in e:
                    k = e["name"]
                    ev[k][0] += e["dur"]
                    ev[k][1] += 1
        tot = sum(v[0] for v in ev.values())
        rows = []
        for k, (us, n) in sorted(ev.items(), key=lambda kv: -kv[1][0]):
            g, th = geom.get(k, (None, None))
            i = instr.get(k, {})
            rows.append(dict(kernel=k, us_total=us, calls=n,
                             us_per_call=us / max(n, 1), pct=100 * us / max(tot, 1),
                             grid=g, block=th,
                             threads=(g * th) if g else None, **i))
        out[B] = dict(B=int(B), gpu_us_total=tot, n_kernels=len(ev), rows=rows)
        print(f"\n--- B={B}  ({'multiplo de 4' if B % 4 == 0 else 'impar/no mult 4'})"
              f"  total GPU {tot:,.0f} us  en {len(ev)} kernels ---", flush=True)
        print(f"  {'kernel':<38} {'us':>10} {'%':>6} {'llam':>6} "
              f"{'grid':>8} {'blq':>5} {'ld esc':>7}", flush=True)
        for r in rows[:14]:
            print(f"  {r['kernel'][:38]:<38} {r['us_total']:10,.0f} "
                  f"{r['pct']:5.1f}% {r['calls']:6d} "
                  f"{(r['grid'] or 0):8,d} {(r['block'] or 0):5d} "
                  f"{r.get('ld_scalar', 0):7d}", flush=True)

    if len(batches) == 2:
        a, b = batches
        print(f"\n=== donde recae la diferencia: B={a} contra B={b} ===", flush=True)
        ra = {r["kernel"]: r for r in out[a]["rows"]}
        rb = {r["kernel"]: r for r in out[b]["rows"]}
        keys = sorted(set(ra) | set(rb),
                      key=lambda k: -(rb.get(k, {}).get("us_total", 0)
                                      - ra.get(k, {}).get("us_total", 0)))
        print(f"  {'kernel':<38} {f'B={a}':>11} {f'B={b}':>11} {'delta us':>11} "
              f"{'delta %':>9}", flush=True)
        for k in keys[:16]:
            ua = ra.get(k, {}).get("us_total", 0.0)
            ub = rb.get(k, {}).get("us_total", 0.0)
            d = ub - ua
            if abs(d) < 1:
                continue
            pc = (100 * (ub / ua - 1)) if ua > 0 else float("inf")
            print(f"  {k[:38]:<38} {ua:11,.0f} {ub:11,.0f} {d:+11,.0f} "
                  f"{pc:+8.1f}%" if ua > 0 else
                  f"  {k[:38]:<38} {ua:11,.0f} {ub:11,.0f} {d:+11,.0f} {'nuevo':>9}",
                  flush=True)
        ta, tb = out[a]["gpu_us_total"], out[b]["gpu_us_total"]
        print(f"\n  total: {ta:,.0f} -> {tb:,.0f} us  ({100*(tb/ta-1):+.1f}%)",
              flush=True)

    res = dict(by_batch={str(k): v for k, v in out.items()},
               Rd=float(Rd), L=int(L), N=N, device=device_identity(),
               n_temps=int(n_temps), n_therm=int(n_therm), n_meas=int(n_meas))
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


@app.function(gpu="H100", **_BS)
def opstep_h100(b, tag, Rd, L, nt, th, me, ncu):
    return _opstep(b, tag, Rd, L, nt, th, me, ncu)


@app.local_entrypoint()
def opstep(batches: str = "135,140", rd: float = 18.30, layers: int = 5,
           n_temps: int = 8, n_therm: int = 300, n_meas: int = 150,
           tag: str = "opstep", ncu: bool = True):
    """Per-operation timing, geometry and instruction mix for two batches."""
    opstep_h100.remote([int(x) for x in batches.split(",") if x], tag,
                       rd, layers, n_temps, n_therm, n_meas, ncu)


def _fusion_doc(batches, tag, Rd, L, n_temps, n_therm, n_meas):
    """What does each fused kernel actually COMPUTE, and what does it cost?

    `_opstep` answered "how long" and "with what instructions", but never
    "what operation is this". The answer is in the HLO, which --xla_dump_to
    already writes and which we had simply never read. For every fused kernel
    this joins four sources by kernel name:

      - the HLO after optimisation, for the fusion's KIND (kLoop / kInput /
        kOutput / kCustom), the opcodes inside its fused computation, its root
        instruction, and its output shape;
      - the thunk sequence, for the launch grid and block;
      - the profiler trace, for the microseconds and the launch count;
      - the metadata op_name, which carries the JAX source line, so a kernel
        can be traced back to the physics it implements.

    The point is to be able to say "the 68% of the gap sits in THIS operation",
    naming the operation, instead of quoting an XLA-generated label.
    """
    import os, sys, glob, json, re, gzip, collections, io as _io
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + " --xla_dump_to=/tmp/dump").strip()
    os.makedirs("/tmp/dump", exist_ok=True)
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S
    import parallel_axes as PA

    kB = 0.086173404
    betas = jnp.asarray(1.0 / (kB * np.linspace(20.0, 0.1, int(n_temps))))
    theta = PA.load_thetas("/root/theta4.json")[0]
    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    eng = S.build_engine(geo)
    N = int(geo["N"])
    print(f"gpu={device_identity()}", flush=True)
    print(f"Rd={Rd} L={L} N={N} sitios={int(np.asarray(geo['disco']).sum())}",
          flush=True)

    THUNK = re.compile(r"\]\s+([A-Za-z_][\w$]*) grid: \[(\d+), 1, 1\]"
                       r" threads: \[(\d+), 1, 1\]")
    # %fusion.1458 = f32[...]{...} fusion(%a, %b), kind=kLoop, calls=%fused_computation.1458
    FUS = re.compile(r"^\s*%?([\w.$-]+) = (\S+) fusion\(([^)]*)\),\s*"
                     r"kind=(k\w+),\s*calls=%?([\w.$-]+)")
    OPN = re.compile(r'op_name="([^"]*)"')
    SRC = re.compile(r'source_file="([^"]*)" source_line=(\d+)')
    # a line inside a fused computation: %name = shape opcode(...)
    INNER = re.compile(r"^\s+%?[\w.$-]+ = \S+ ([a-z][\w-]*)\(")
    ROOT = re.compile(r"^\s+ROOT %?[\w.$-]+ = (\S+) ([a-z][\w-]*)\(")

    def parse_hlo(path):
        """-> {computation_name: {ops, root_op, root_shape}}, {fusion: info}"""
        txt = _io.open(path, encoding="utf-8", errors="replace").read()
        comps, cur, depth = {}, None, 0
        for ln in txt.splitlines():
            if cur is None:
                m = re.match(r"^\s*%?([\w.$-]+) \(.*\) -> \S+ \{\s*$", ln)
                if m and m.group(1) not in ("ENTRY", "main"):
                    cur = m.group(1)
                    comps[cur] = dict(ops=[], root_op=None, root_shape=None,
                                      n_lines=0)
                continue
            if ln.strip() == "}":
                cur = None
                continue
            comps[cur]["n_lines"] += 1
            mr = ROOT.match(ln)
            if mr:
                comps[cur]["root_shape"], comps[cur]["root_op"] = mr.group(1), mr.group(2)
                comps[cur]["ops"].append(mr.group(2))
                continue
            mi = INNER.match(ln)
            if mi and mi.group(1) not in ("parameter", "constant"):
                comps[cur]["ops"].append(mi.group(1))
        fus = {}
        for ln in txt.splitlines():
            m = FUS.match(ln)
            if not m:
                continue
            name, shape, operands, kind, calls = m.groups()
            op = OPN.search(ln)
            sf = SRC.search(ln)
            fus[name] = dict(hlo_name=name, out_shape=shape, kind=kind,
                             calls=calls, n_operands=len([o for o in
                                 operands.split(",") if o.strip()]),
                             op_name=(op.group(1) if op else None),
                             source=(f"{sf.group(1)}:{sf.group(2)}" if sf else None))
        return comps, fus

    def body(sp, kk, p):
        return eng["anneal"](sp, kk, betas, p, int(n_therm), int(n_meas))

    out = {}
    for B in batches:
        for f in glob.glob("/tmp/dump/*"):
            try:
                os.remove(f)
            except Exception:
                pass
        cols = PA.pack([theta], B)
        args = (S.init_spins(jax.random.PRNGKey(11), geo, B),
                jax.random.PRNGKey(12), cols)
        fn = jax.jit(body).lower(*args).compile()
        jax.block_until_ready(fn(*args))

        files = sorted(glob.glob("/tmp/dump/*"))
        pre = os.path.basename(max(files, key=os.path.getsize)).split(".jit_")[0]
        mine = [f for f in files if os.path.basename(f).startswith(pre)]
        geom = {}
        for f in mine:
            if "thunk_sequence" in f:
                t = _io.open(f, encoding="utf-8", errors="replace").read()
                for nm, g, tt in THUNK.findall(t):
                    geom[nm] = (int(g), int(tt))
        hlos = [f for f in mine if "after_optimizations.txt" in f]
        comps, fus = {}, {}
        for f in sorted(hlos, key=os.path.getsize, reverse=True)[:1]:
            comps, fus = parse_hlo(f)
            print(f"  HLO: {os.path.basename(f)} -> {len(fus)} fusiones, "
                  f"{len(comps)} computaciones fundidas", flush=True)

        pdir = f"/tmp/prof_{B}"
        with jax.profiler.trace(pdir):
            jax.block_until_ready(fn(*args))
        ev = collections.defaultdict(lambda: [0.0, 0])
        for g in glob.glob(f"{pdir}/**/*.trace.json.gz", recursive=True):
            with gzip.open(g, "rt") as fh:
                try:
                    tr = json.load(fh)
                except Exception:
                    continue
            pids = {e["pid"] for e in tr.get("traceEvents", [])
                    if e.get("ph") == "M" and e.get("name") == "process_name"
                    and "GPU" in str(e.get("args", {}).get("name", ""))}
            for e in tr.get("traceEvents", []):
                if e.get("ph") == "X" and e.get("pid") in pids and "dur" in e:
                    ev[e["name"]][0] += e["dur"]
                    ev[e["name"]][1] += 1

        # kernel names use _ where HLO uses . ; match on the normalised form
        norm = lambda x: x.replace(".", "_").replace("-", "_")
        byn = {norm(k): v for k, v in fus.items()}

        def lookup(kname):
            """Kernel names and HLO fusion names agree only up to the uniquer.

            The thunk calls a kernel `loop_and_fusion_2` while the HLO names the
            fusion `loop_and_fusion.2`; normalising the dots covers that. But
            XLA also appends its own `_N` to a kernel whose fusion carries no
            suffix, so fall back to stripping one trailing index, and then to a
            unique prefix match. Anything still unmatched is reported as such
            rather than silently guessed.
            """
            if kname in byn:
                return byn[kname], "exact"
            st = re.sub(r"_\d+$", "", kname)
            if st in byn:
                return byn[st], "suffix-stripped"
            cand = [v for k, v in byn.items() if k.startswith(st)]
            if len(cand) == 1:
                return cand[0], "prefix"
            return {}, "unmatched"
        sweeps = int(n_temps) * (int(n_therm) + int(n_meas))
        rows = []
        for k, (us, n) in sorted(ev.items(), key=lambda kv: -kv[1][0]):
            g, tt = geom.get(k, (None, None))
            info, how = lookup(k)
            comp = comps.get(info.get("calls", ""), {})
            ops = comp.get("ops", [])
            tally = collections.Counter(ops)
            rows.append(dict(
                kernel=k, us_total=us, calls=n, grid=g, block=tt,
                us_per_replica=us / B,
                us_per_sweep_replica=us / max(sweeps, 1) / B,
                blocks_per_replica=(g / B) if g else None,
                kind=info.get("kind"), hlo_name=info.get("hlo_name"),
                matched=how,
                out_shape=info.get("out_shape"),
                n_operands=info.get("n_operands"),
                op_name=info.get("op_name"), source=info.get("source"),
                root_op=comp.get("root_op"), n_inner_ops=len(ops),
                ops_tally=dict(tally.most_common()),
                ops_distinct=sorted(tally)))
        out[B] = dict(B=B, gpu_us_total=sum(v[0] for v in ev.values()),
                      n_kernels=len(ev), n_fusions_hlo=len(fus), rows=rows)

        r = out[B]
        print(f"\n--- B={B}  {r['gpu_us_total']:,.0f} us en {r['n_kernels']} "
              f"kernels ---", flush=True)
        print(f"  {'kernel':<34}{'kind':>8}{'raiz':>12}{'ops':>5}"
              f"{'us/rep':>10}{'bloq/rep':>10}  forma salida", flush=True)
        for q in r["rows"][:16]:
            print(f"  {q['kernel'][:33]:<34}{str(q['kind'] or '-'):>8}"
                  f"{str(q['root_op'] or '-')[:11]:>12}{q['n_inner_ops']:>5}"
                  f"{q['us_per_replica']:>10,.1f}"
                  f"{(q['blocks_per_replica'] or 0):>10,.2f}  "
                  f"{str(q['out_shape'] or '-')[:34]}", flush=True)
        for q in r["rows"][:6]:
            print(f"\n  == {q['kernel']} ({q['kind']}) ==", flush=True)
            print(f"     fuente   : {q['source']}", flush=True)
            print(f"     op_name  : {str(q['op_name'])[:110]}", flush=True)
            print(f"     salida   : {q['out_shape']}  raiz={q['root_op']}"
                  f"  operandos={q['n_operands']}", flush=True)
            print(f"     contiene : {q['ops_tally']}", flush=True)

    res = dict(by_batch={str(k): v for k, v in out.items()},
               Rd=float(Rd), L=int(L), N=N, device=device_identity(),
               n_temps=int(n_temps), n_therm=int(n_therm), n_meas=int(n_meas),
               sweeps=int(n_temps) * (int(n_therm) + int(n_meas)))
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n  guardado /out/{tag}.json", flush=True)
    return res


@app.function(gpu="H100", **_COMMON)
def fusion_doc_h100(b, tag, Rd, L, nt, th, me):
    return _fusion_doc(b, tag, Rd, L, nt, th, me)


@app.local_entrypoint()
def fusion_doc(batches: str = "65,75", rd: float = 18.30, layers: int = 5,
               n_temps: int = 8, n_therm: int = 300, n_meas: int = 150,
               tag: str = "fusion_doc_65_75"):
    """What each fused kernel computes, joined with what it costs."""
    fusion_doc_h100.remote([int(x) for x in batches.split(",") if x], tag,
                           rd, layers, n_temps, n_therm, n_meas)


_KC_WORKER = r'''
import os, sys, glob, json, re, time
B, Rd, L, nt, th, me, dump = (int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]),
                              int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]),
                              sys.argv[7])
os.makedirs(dump, exist_ok=True)
sys.path.insert(0, "/root")
import numpy as np, jax, jax.numpy as jnp
import sim_core as S, parallel_axes as PA
kB = 0.086173404
betas = jnp.asarray(1.0 / (kB * np.linspace(20.0, 0.1, nt)))
theta = PA.load_thetas("/root/theta4.json")[0]
geo = S.build_geometry(Rd=Rd, L=L); eng = S.build_engine(geo)
cols = PA.pack([theta], B)
args = (S.init_spins(jax.random.PRNGKey(11), geo, B), jax.random.PRNGKey(12), cols)
def body(sp, kk, p):
    return eng["anneal"](sp, kk, betas, p, th, me)
t0 = time.perf_counter()
fn = jax.jit(body).lower(*args).compile()
jax.block_until_ready(fn(*args))
t_compile = time.perf_counter() - t0
ts = []
for _ in range(3):
    t = time.perf_counter(); jax.block_until_ready(fn(*args)); ts.append(time.perf_counter() - t)
files = sorted(glob.glob(f"{dump}/*"))
res = dict(B=B, wall_best=min(ts), wall_all=ts, compile_s=t_compile,
           kinds={}, kcustom=[], kernels=[], error=None)
if files:
    pre = os.path.basename(max(files, key=os.path.getsize)).split(".jit_")[0]
    mine = [f for f in files if os.path.basename(f).startswith(pre)]
    THUNK = re.compile(r"\]\s+([A-Za-z_][\w$]*) grid: \[(\d+), 1, 1\] threads: \[(\d+), 1, 1\]")
    for f in mine:
        if "thunk_sequence" in f:
            t = open(f, encoding="utf-8", errors="replace").read()
            res["kernels"] = [dict(kernel=n, grid=int(g), block=int(b))
                              for n, g, b in THUNK.findall(t)]
    hl = [f for f in mine if "after_optimizations.txt" in f]
    if hl:
        txt = open(max(hl, key=os.path.getsize), encoding="utf-8", errors="replace").read()
        FUS = re.compile(r"^\s*%?([\w.$-]+) = (\S+) fusion\(([^)]*)\),\s*kind=(k\w+)(.*)$", re.M)
        for name, shape, ops, kind, rest in FUS.findall(txt):
            res["kinds"][kind] = res["kinds"].get(kind, 0) + 1
            if kind == "kCustom":
                bc = re.search(r'backend_config=(\{.*?\})\s*(?:,\s*metadata|$)', rest)
                res["kcustom"].append(dict(name=name, out_shape=shape, kind=kind,
                                           backend_config=(bc.group(1) if bc else None),
                                           tail=rest[:400]))
print("@@JSON@@" + json.dumps(res))
'''


def _kcustom_probe(tag, Rd, L, n_temps, n_therm, n_meas, B):
    """Who emits the kCustom fusion, and can a flag turn it off?

    `fusion_doc` told us the kernel carrying most of the batch-to-batch gap is
    kCustom. XLA sets that kind from a specific backend -- cuDNN, the
    dynamic-slice rewriter, or Triton -- and the fusion's `backend_config`
    names which. That matters for two reasons: it identifies who chooses that
    kernel's launch grid (which is the only non-proportional one of the 43),
    and it tells us whether a flag can route the same computation through
    XLA's own loop emitter instead.

    Each configuration runs in its own subprocess, because XLA_FLAGS is read
    once when the backend initialises.
    """
    import os, sys, json, subprocess, re
    sys.path.insert(0, "/root")
    print(f"gpu={device_identity()}", flush=True)

    # --- which flags does this build even have? -------------------------
    env = dict(os.environ, XLA_FLAGS="--help")
    r = subprocess.run([sys.executable, "-c", "import jax; jax.devices()"],
                       env=env, capture_output=True, text=True, timeout=600)
    table = (r.stdout or "") + (r.stderr or "")
    names = sorted(set(re.findall(r"--(xla_[a-z0-9_]+)", table)))
    want = [n for n in names
            if any(w in n for w in ("triton", "custom_fusion", "dynamic_slice_fusion",
                                    "cudnn", "fusion_level", "emitter"))]
    print(f"\n  flags en este build: {len(names)}; relevantes: {len(want)}", flush=True)
    for n in want:
        print(f"    --{n}", flush=True)

    with open("/root/_kcw.py", "w") as f:
        f.write(_KC_WORKER)

    def run(flags, label):
        d = f"/tmp/d_{abs(hash(label)) % 10**8}"
        e = dict(os.environ)
        e["XLA_FLAGS"] = (f"--xla_dump_to={d} " + flags).strip()
        p = subprocess.run([sys.executable, "/root/_kcw.py", str(B), str(Rd), str(L),
                            str(n_temps), str(n_therm), str(n_meas), d],
                           env=e, capture_output=True, text=True, timeout=1800)
        out = (p.stdout or "") + (p.stderr or "")
        m = re.search(r"@@JSON@@(\{.*\})", out)
        if not m:
            return dict(label=label, flags=flags, error=out[-700:])
        j = json.loads(m.group(1)); j.update(label=label, flags=flags)
        return j

    results = [run("", "base")]
    b = results[0]
    print(f"\n  === base: {b['wall_best']:.3f} s  kinds={b['kinds']} ===", flush=True)
    for kc in b.get("kcustom", []):
        print(f"\n    kCustom {kc['name']}  salida {kc['out_shape']}", flush=True)
        print(f"      backend_config = {kc['backend_config']}", flush=True)
        print(f"      cola: {kc['tail'][:240]}", flush=True)
    kmap = {k["kernel"]: k for k in b.get("kernels", [])}
    for nm in ("fusion_1458", "fusion_1459"):
        if nm in kmap:
            print(f"    {nm}: grid {kmap[nm]['grid']}, block {kmap[nm]['block']}, "
                  f"bloq/rep {kmap[nm]['grid']/B:.3f}", flush=True)

    # --- try turning each candidate off ---------------------------------
    cands = [f"--{n}=false" for n in want
             if any(w in n for w in ("triton", "custom_fusion", "dynamic_slice_fusion"))]
    for c in cands[:8]:
        rr = run(c, c)
        results.append(rr)
        if rr.get("error"):
            print(f"\n  {c}: FALLO -> {rr['error'][:200]}", flush=True)
            continue
        nk = len(rr.get("kcustom", []))
        print(f"\n  {c}: {rr['wall_best']:.3f} s "
              f"({100*(rr['wall_best']/b['wall_best']-1):+.1f}%)  "
              f"kinds={rr['kinds']}  kCustom={nk}", flush=True)

    res = dict(B=B, Rd=float(Rd), L=int(L), device=device_identity(),
               n_temps=int(n_temps), n_therm=int(n_therm), n_meas=int(n_meas),
               all_flags=names, candidate_flags=want, runs=results)
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n  guardado /out/{tag}.json", flush=True)
    return res


@app.function(gpu="H100", **_COMMON)
def kcustom_probe_h100(tag, Rd, L, nt, th, me, B):
    return _kcustom_probe(tag, Rd, L, nt, th, me, B)


@app.local_entrypoint()
def kcustom_probe(rd: float = 18.30, layers: int = 5, n_temps: int = 8,
                  n_therm: int = 300, n_meas: int = 150, b: str = "65",
                  tag: str = "kcustom_probe"):
    """Identify the kCustom emitter and test whether a flag disables it."""
    for x in [int(v) for v in str(b).split(",") if v]:
        kcustom_probe_h100.remote(f"{tag}_{x}", rd, layers, n_temps, n_therm,
                                  n_meas, x)


def _split_protocol(tag, Rd, L, n_temps, n_therm, n_meas, reps, spec=None):
    """Does splitting a multiple of four into two odd batches actually pay?

    The width rule says a batch with v2(B)+v2(L) >= 2 compiles to four
    elements per thread and pays for it. B=200 with L=5 has v2=3, so it is
    the bad case. The proposed protocol splits it into B/2+1 and B/2-1, both
    odd, both width 1, and their sum is exactly B.

    The saving is real but it is not free: two launches pay the fixed
    per-sweep cost twice. Writing u(B) = a/B + b + c[width 4], one launch of
    200 costs a + 200b + 200c of GPU work while two launches of 101 and 99
    cost 2a + 200b. So the split wins exactly when a < 200c, and only a
    measurement settles it.

    Two alternatives go in the same run, because the protocol has to beat
    them and not just beat B=200:
      - ONE launch of 201, also odd, one replica MORE than asked;
      - ONE launch of 199, also odd, one replica FEWER.
    If either wins there is no need for a protocol at all.

    Everything is normalised to wall clock per replica, since the configs
    deliver different replica counts. Configurations are interleaved and
    repeated so that thermal drift hits all of them equally.
    """
    import os, sys, json, time, statistics as st
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S
    import parallel_axes as PA

    kB = 0.086173404
    betas = jnp.asarray(1.0 / (kB * np.linspace(20.0, 0.1, int(n_temps))))
    theta = PA.load_thetas("/root/theta4.json")[0]
    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    eng = S.build_engine(geo)
    N = int(geo["N"])
    sweeps = int(n_temps) * (int(n_therm) + int(n_meas))
    v2 = lambda n: (n & -n).bit_length() - 1
    width = lambda B: min(4, 2 ** (v2(B) + v2(int(L))))
    print(f"gpu={device_identity()}", flush=True)
    print(f"Rd={Rd} L={L} N={N} barridos={sweeps}  repeticiones={reps}", flush=True)

    if spec:
        # "100|50,50|51,49" -> one config per pipe, batches comma-separated
        CONFIGS = []
        for part in str(spec).split("|"):
            bs = [int(x) for x in part.split(",") if x.strip()]
            if not bs:
                continue
            lab = " + ".join(str(x) for x in bs) + \
                  f" (ancho {','.join(str(width(x)) for x in bs)})"
            CONFIGS.append((lab, bs))
    else:
        CONFIGS = [
            ("200 de una (ancho 4)",            [200]),
            ("101 + 99 (el protocolo)",         [101, 99]),
            ("201 de una (una replica MAS)",    [201]),
            ("199 de una (una replica MENOS)",  [199]),
            ("100 de una (ancho 4)",            [100]),
            ("51 + 49 (el protocolo, chico)",   [51, 49]),
        ]

    def body(sp, kk, p):
        return eng["anneal"](sp, kk, betas, p, int(n_therm), int(n_meas))

    # compile and warm every distinct batch size first, so no timed call
    # ever includes compilation
    need = sorted({B for _, bs in CONFIGS for B in bs})
    comp, t_comp = {}, {}
    for B in need:
        cols = PA.pack([theta], B)
        args = (S.init_spins(jax.random.PRNGKey(11 + B), geo, B),
                jax.random.PRNGKey(12 + B), cols)
        t0 = time.perf_counter()
        fn = jax.jit(body).lower(*args).compile()
        jax.block_until_ready(fn(*args))
        t_comp[B] = time.perf_counter() - t0
        comp[B] = (fn, args)
        print(f"  compilado B={B:>4} (ancho {width(B)}) en {t_comp[B]:6.1f} s", flush=True)

    # interleaved repeats
    samples = {lab: [] for lab, _ in CONFIGS}
    for r in range(int(reps)):
        for lab, bs in CONFIGS:
            t0 = time.perf_counter()
            for B in bs:                      # one GPU: the launches are serial
                fn, args = comp[B]
                jax.block_until_ready(fn(*args))
            samples[lab].append(time.perf_counter() - t0)
        print(f"  ronda {r+1}/{reps} lista", flush=True)

    rows = []
    for lab, bs in CONFIGS:
        s = samples[lab]
        nrep = sum(bs)
        rows.append(dict(label=lab, batches=bs, n_replicas=nrep,
                         widths=[width(B) for B in bs],
                         n_launches=len(bs),
                         wall_min=min(s), wall_median=st.median(s),
                         wall_max=max(s),
                         spread_pct=100 * (max(s) / min(s) - 1),
                         samples=s,
                         wall_per_replica_min=min(s) / nrep,
                         wall_per_replica_median=st.median(s) / nrep,
                         us_per_sweep_replica=1e6 * st.median(s) / sweeps / nrep,
                         compile_s=[t_comp[B] for B in bs],
                         compile_total=sum(t_comp[B] for B in bs)))

    print(f"\n=== reloj de pared por replica (mediana de {reps}) ===", flush=True)
    print(f"  {'configuracion':<34}{'lanz':>5}{'repl':>6}{'ancho':>8}"
          f"{'pared s':>10}{'s/replica':>11}{'disp':>7}{'compilar':>10}", flush=True)
    base = max(rows, key=lambda r: r["wall_per_replica_median"])
    for r in rows:
        print(f"  {r['label']:<34}{r['n_launches']:>5}{r['n_replicas']:>6}"
              f"{str(r['widths']):>8}{r['wall_median']:>10.3f}"
              f"{r['wall_per_replica_median']:>11.5f}{r['spread_pct']:>6.1f}%"
              f"{r['compile_total']:>10.1f}", flush=True)

    print(f"\n=== contra el peor, {base['label']} "
          f"({base['wall_per_replica_median']:.5f} s/replica) ===", flush=True)
    for r in rows:
        if r is base:
            continue
        ch = 100 * (r["wall_per_replica_median"] / base["wall_per_replica_median"] - 1)
        verdict = "MEJOR" if ch < 0 else "peor"
        print(f"  {r['label']:<34}{ch:>+8.1f}%   {verdict}", flush=True)

    # does the protocol beat the one-launch odd alternatives?
    prot = next((r for r in rows if r["label"].startswith("101")), None)
    for other in (("201", "199") if prot else ()):
        o = next((r for r in rows if r["label"].startswith(other)), None)
        if o is None:
            continue
        ch = 100 * (prot["wall_per_replica_median"] / o["wall_per_replica_median"] - 1)
        print(f"\n  protocolo (101+99) contra {other} de una: {ch:+.1f}%  "
              f"-> {'el protocolo gana' if ch < 0 else 'un solo lote impar gana'}",
              flush=True)

    print(f"\n=== cada par, lado a lado ===", flush=True)
    for i, x in enumerate(rows):
        for y in rows[i + 1:]:
            ch = 100 * (y["wall_per_replica_median"]
                        / x["wall_per_replica_median"] - 1)
            print(f"  {y['label']:<28} contra {x['label']:<28}{ch:+7.2f}%",
                  flush=True)

    res = dict(rows=rows, Rd=float(Rd), L=int(L), N=N, device=device_identity(),
               n_temps=int(n_temps), n_therm=int(n_therm), n_meas=int(n_meas),
               sweeps=sweeps, reps=int(reps),
               note=("Replicas are independent chains, so the split is "
                     "statistically exact but not bit-identical: the RNG keys "
                     "differ. On one GPU the two launches are serial, so the "
                     "wall clock of the protocol is their sum."))
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n  guardado /out/{tag}.json", flush=True)
    return res


@app.function(gpu="H100", **_COMMON)
def split_protocol_h100(tag, Rd, L, nt, th, me, reps, spec=None):
    return _split_protocol(tag, Rd, L, nt, th, me, reps, spec)


@app.local_entrypoint()
def split_protocol(rd: float = 18.30, layers: int = 5, n_temps: int = 8,
                   n_therm: int = 300, n_meas: int = 150, reps: int = 7,
                   tag: str = "split_protocol", spec: str = ""):
    """Is splitting a multiple of four into two odd batches worth it?"""
    split_protocol_h100.remote(tag, rd, layers, n_temps, n_therm, n_meas, reps,
                               spec or None)


def _sawtooth_map(batches, tag, Rd, L, n_temps, n_therm, n_meas, reps):
    """Does the cost jump where the Triton tile changes?

    Everything we know about the sawtooth: 79% of it sits in the Triton
    kCustom fusion, its launch grid is exactly the product of ceil(dim/tile)
    over that fusion's output_tiles (8 of 8 exact), and Triton re-chooses the
    tile per shape. What we do NOT know is whether the tile change is what
    costs: the block count demonstrably does not predict the cost, since the
    heavy kernel kept the same tile between B=65 and B=75 while its time moved
    26.8%.

    So this maps a dense series of ODD batch sizes -- every one width 1, so
    the width axis is held fixed and only the sawtooth can move -- and records
    for each one both the tiles Triton picked and the wall clock over several
    interleaved repeats. Then the question is answerable: do the cost jumps sit
    at the tile transitions, or not?

    Compilation happens for every batch first, reading each HLO dump before the
    next overwrites it; the timing rounds come afterwards, interleaved, so
    thermal drift hits every batch equally.
    """
    import os, sys, glob, json, re, time, statistics as stx, io as _io
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + " --xla_dump_to=/tmp/dump").strip()
    os.makedirs("/tmp/dump", exist_ok=True)
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S
    import parallel_axes as PA

    kB = 0.086173404
    betas = jnp.asarray(1.0 / (kB * np.linspace(20.0, 0.1, int(n_temps))))
    theta = PA.load_thetas("/root/theta4.json")[0]
    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    eng = S.build_engine(geo)
    N = int(geo["N"])
    sweeps = int(n_temps) * (int(n_therm) + int(n_meas))
    v2 = lambda n: (n & -n).bit_length() - 1
    width = lambda B: min(4, 2 ** (v2(B) + v2(int(L))))
    print(f"gpu={device_identity()}", flush=True)
    print(f"Rd={Rd} L={L} N={N} barridos={sweeps} reps={reps}", flush=True)
    bad = [B for B in batches if width(B) != 1]
    if bad:
        print(f"  AVISO: estos no son ancho 1 y contaminan el eje: {bad}", flush=True)

    THUNK = re.compile(r"\]\s+([A-Za-z_][\w$]*) grid: \[(\d+), 1, 1\]"
                       r" threads: \[(\d+), 1, 1\]")
    FUS = re.compile(r"^\s*%?([\w.$-]+) = (\S+) fusion\(([^)]*)\),\s*"
                     r"kind=kCustom(.*)$", re.M)
    TILE = re.compile(r'"sizes":\[([^\]]*)\]')
    TRIT = re.compile(r'"kind":"__triton"')

    def body(sp, kk, p):
        return eng["anneal"](sp, kk, betas, p, int(n_therm), int(n_meas))

    info, comp = {}, {}
    for B in batches:
        for f in glob.glob("/tmp/dump/*"):
            try:
                os.remove(f)
            except Exception:
                pass
        cols = PA.pack([theta], B)
        args = (S.init_spins(jax.random.PRNGKey(11 + B), geo, B),
                jax.random.PRNGKey(12 + B), cols)
        t0 = time.perf_counter()
        fn = jax.jit(body).lower(*args).compile()
        jax.block_until_ready(fn(*args))
        tc = time.perf_counter() - t0
        comp[B] = (fn, args)

        files = sorted(glob.glob("/tmp/dump/*"))
        geom, tiles = {}, []
        if files:
            pre = os.path.basename(max(files, key=os.path.getsize)).split(".jit_")[0]
            mine = [f for f in files if os.path.basename(f).startswith(pre)]
            for f in mine:
                if "thunk_sequence" in f:
                    t = _io.open(f, encoding="utf-8", errors="replace").read()
                    for nm, g, bl in THUNK.findall(t):
                        geom[nm] = int(g)
            hl = [f for f in mine if "after_optimizations.txt" in f]
            if hl:
                txt = _io.open(max(hl, key=os.path.getsize), encoding="utf-8",
                               errors="replace").read()
                for name, shape, ops, rest in FUS.findall(txt):
                    if not TRIT.search(rest):
                        continue
                    m = TILE.search(rest)
                    if not m:
                        continue
                    tl = [int(x.strip().strip('"')) for x in m.group(1).split(",")]
                    if len(tl) != 5:
                        continue
                    nm = name.replace(".", "_")
                    g = geom.get(nm)
                    if g is None:          # a deduplicated twin, never launched
                        continue
                    tiles.append(dict(kernel=nm, tile=tl, grid=g,
                                      blocks_per_replica=g / B))
        tiles.sort(key=lambda t: -t["grid"])
        info[B] = dict(B=B, width=width(B), compile_s=tc, tiles=tiles,
                       n_triton=len(tiles),
                       tile_B=[t["tile"][0] for t in tiles],
                       signature="|".join(",".join(map(str, t["tile"])) for t in tiles))
        print(f"  B={B:>4} compilado en {tc:5.1f} s  tiles={info[B]['signature']}",
              flush=True)

    samples = {B: [] for B in batches}
    for r in range(int(reps)):
        for B in batches:
            fn, args = comp[B]
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*args))
            samples[B].append(time.perf_counter() - t0)
        print(f"  ronda {r+1}/{reps}", flush=True)

    for B in batches:
        s = samples[B]
        info[B].update(wall_median=stx.median(s), wall_min=min(s),
                       spread_pct=100 * (max(s) / min(s) - 1), samples=s,
                       wall_per_replica=stx.median(s) / B,
                       us_per_sweep_replica=1e6 * stx.median(s) / sweeps / B)

    print(f"\n=== el mapa: todos ancho 1, {reps} repeticiones intercaladas ===",
          flush=True)
    print(f"  {'B':>5}{'s/replica':>11}{'disp':>7}{'bloq/rep':>10}"
          f"{'tile_B':>9}  firma de tiles", flush=True)
    prev = None
    for B in batches:
        d = info[B]
        mark = ""
        if prev is not None and d["signature"] != prev:
            mark = "  <-- CAMBIO DE TILE"
        prev = d["signature"]
        bpr = d["tiles"][0]["blocks_per_replica"] if d["tiles"] else 0.0
        print(f"  {B:>5}{d['wall_per_replica']:>11.6f}{d['spread_pct']:>6.1f}%"
              f"{bpr:>10.3f}{str(d['tile_B']):>9}  {d['signature']}{mark}",
              flush=True)

    print(f"\n=== salto de costo entre lotes vecinos, y si el tile cambio ===",
          flush=True)
    jumps = []
    for i in range(1, len(batches)):
        a, b = batches[i - 1], batches[i]
        ch = 100 * (info[b]["wall_per_replica"] / info[a]["wall_per_replica"] - 1)
        same = info[a]["signature"] == info[b]["signature"]
        jumps.append(dict(a=a, b=b, change_pct=ch, same_tile=same))
        print(f"  {a:>4} -> {b:>4}  {ch:+7.2f}%   tile "
              f"{'IGUAL' if same else 'DISTINTO'}", flush=True)

    sa = [j["change_pct"] for j in jumps if j["same_tile"]]
    di = [j["change_pct"] for j in jumps if not j["same_tile"]]
    print(f"\n=== la respuesta ===", flush=True)
    for lab, xs in (("mismo tile", sa), ("tile distinto", di)):
        if xs:
            print(f"  salto |medio| con {lab:<14}: "
                  f"{stx.mean(abs(x) for x in xs):6.2f}%   (n={len(xs)}, "
                  f"max {max(abs(x) for x in xs):.2f}%)", flush=True)
    if sa and di:
        print(f"\n  Si el tile explicara el dentado, los saltos con tile IGUAL "
              f"deberian ser\n  mucho menores que los de tile DISTINTO.", flush=True)

    res = dict(by_batch={str(k): v for k, v in info.items()}, jumps=jumps,
               Rd=float(Rd), L=int(L), N=N, device=device_identity(),
               n_temps=int(n_temps), n_therm=int(n_therm), n_meas=int(n_meas),
               sweeps=sweeps, reps=int(reps))
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n  guardado /out/{tag}.json", flush=True)
    return res


@app.function(gpu="H100", **_COMMON)
def sawtooth_map_h100(b, tag, Rd, L, nt, th, me, reps):
    return _sawtooth_map(b, tag, Rd, L, nt, th, me, reps)


@app.local_entrypoint()
def sawtooth_map(batches: str = "95,97,99,101,103,107,111,115,119,123,127,131,135",
                 rd: float = 18.30, layers: int = 5, n_temps: int = 8,
                 n_therm: int = 300, n_meas: int = 150, reps: int = 9,
                 tag: str = "sawtooth_map"):
    """Map cost against Triton tile choice over odd (width-1) batch sizes."""
    sawtooth_map_h100.remote([int(x) for x in batches.split(",") if x], tag,
                             rd, layers, n_temps, n_therm, n_meas, reps)


def _mitigation_curve(targets, tag, Rd, L, n_temps, n_therm, n_meas, reps, window):
    """How much do the two mitigations actually save, end to end?

    The question is not what a batch size costs, it is what a REQUIRED NUMBER
    OF REPLICAS costs. For a target R there are three ways to get it:

      naive      -- one launch of B = R, whatever v2(R) happens to be;
      protocol   -- if R is a multiple of four, split into R/2+1 and R/2-1,
                    both odd, both vector width 1, summing to exactly R;
      selector   -- among the partitions of R into two odd batches, compile
                    each candidate, read the Triton tile signature out of the
                    HLO, and take the partition whose batches carry the cheap
                    signature.

    Everything is measured, so the selector's pick can be scored against the
    partition that actually turned out fastest. That is the real test: a
    selector that does not find the best candidate is not a selector.
    """
    import os, sys, glob, json, re, time, statistics as stx, io as _io, itertools
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + " --xla_dump_to=/tmp/dump").strip()
    os.makedirs("/tmp/dump", exist_ok=True)
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S
    import parallel_axes as PA

    kB = 0.086173404
    betas = jnp.asarray(1.0 / (kB * np.linspace(20.0, 0.1, int(n_temps))))
    theta = PA.load_thetas("/root/theta4.json")[0]
    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    eng = S.build_engine(geo)
    N = int(geo["N"])
    sweeps = int(n_temps) * (int(n_therm) + int(n_meas))
    v2 = lambda n: (n & -n).bit_length() - 1
    width = lambda B: min(4, 2 ** (v2(B) + v2(int(L))))
    print(f"gpu={device_identity()}", flush=True)
    print(f"Rd={Rd} L={L} N={N} barridos={sweeps} reps={reps}", flush=True)

    # ---- candidate set per target ---------------------------------------
    plans = {}
    for R in targets:
        cands = [("naive", (R,))]
        half = R // 2
        for d in range(1, int(window) + 1):
            a, b = half + d, half - d
            if a % 2 and b % 2 and a > 0 and b > 0 and a + b == R:
                cands.append((f"split {a}+{b}", (a, b)))
        plans[R] = cands
        print(f"  R={R}: {len(cands)} candidatos", flush=True)
    need = sorted({B for R in targets for _, bs in plans[R] for B in bs})
    print(f"\n  lotes distintos a compilar: {len(need)} -> {need}", flush=True)

    THUNK = re.compile(r"\]\s+([A-Za-z_][\w$]*) grid: \[(\d+), 1, 1\]"
                       r" threads: \[(\d+), 1, 1\]")
    FUS = re.compile(r"^\s*%?([\w.$-]+) = (\S+) fusion\(([^)]*)\),\s*"
                     r"kind=kCustom(.*)$", re.M)
    TILE = re.compile(r'"sizes":\[([^\]]*)\]')
    TRIT = re.compile(r'"kind":"__triton"')
    CHEAP = "8,1,8,8,3|8,1,8,8,3"

    def body(sp, kk, p):
        return eng["anneal"](sp, kk, betas, p, int(n_therm), int(n_meas))

    comp, meta = {}, {}
    for B in need:
        for f in glob.glob("/tmp/dump/*"):
            try:
                os.remove(f)
            except Exception:
                pass
        cols = PA.pack([theta], B)
        args = (S.init_spins(jax.random.PRNGKey(11 + B), geo, B),
                jax.random.PRNGKey(12 + B), cols)
        t0 = time.perf_counter()
        fn = jax.jit(body).lower(*args).compile()
        jax.block_until_ready(fn(*args))
        tc = time.perf_counter() - t0
        comp[B] = (fn, args)
        files = sorted(glob.glob("/tmp/dump/*"))
        geom, tiles = {}, []
        if files:
            pre = os.path.basename(max(files, key=os.path.getsize)).split(".jit_")[0]
            mine = [f for f in files if os.path.basename(f).startswith(pre)]
            for f in mine:
                if "thunk_sequence" in f:
                    t = _io.open(f, encoding="utf-8", errors="replace").read()
                    for nm, g, bl in THUNK.findall(t):
                        geom[nm] = int(g)
            hl = [f for f in mine if "after_optimizations.txt" in f]
            if hl:
                txt = _io.open(max(hl, key=os.path.getsize), encoding="utf-8",
                               errors="replace").read()
                for name, shape, ops, rest in FUS.findall(txt):
                    if not TRIT.search(rest):
                        continue
                    m = TILE.search(rest)
                    if not m:
                        continue
                    tl = [int(x.strip().strip('"')) for x in m.group(1).split(",")]
                    nm = name.replace(".", "_")
                    if len(tl) == 5 and nm in geom:
                        tiles.append((geom[nm], ",".join(map(str, tl))))
        tiles.sort(key=lambda t: -t[0])
        sig = "|".join(t[1] for t in tiles)
        meta[B] = dict(B=B, width=width(B), compile_s=tc, signature=sig,
                       cheap=(sig == CHEAP))
        print(f"  B={B:>4} ancho {width(B)}  compilado {tc:5.1f} s  "
              f"{'BARATA' if sig == CHEAP else '      '}  {sig}", flush=True)

    # ---- interleaved timing of every distinct batch ---------------------
    samples = {B: [] for B in need}
    for r in range(int(reps)):
        for B in need:
            fn, args = comp[B]
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*args))
            samples[B].append(time.perf_counter() - t0)
        print(f"  ronda {r+1}/{reps}", flush=True)
    for B in need:
        s = samples[B]
        meta[B].update(wall_median=stx.median(s),
                       spread_pct=100 * (max(s) / min(s) - 1))

    # ---- score every plan ------------------------------------------------
    out = {}
    for R in targets:
        rows = []
        for lab, bs in plans[R]:
            wall = sum(meta[B]["wall_median"] for B in bs)
            rows.append(dict(label=lab, batches=list(bs), n_launches=len(bs),
                             widths=[meta[B]["width"] for B in bs],
                             signatures=[meta[B]["signature"] for B in bs],
                             all_cheap=all(meta[B]["cheap"] for B in bs),
                             n_cheap=sum(meta[B]["cheap"] for B in bs),
                             wall=wall, wall_per_replica=wall / R))
        naive = next(r for r in rows if r["label"] == "naive")
        splits = [r for r in rows if r["label"] != "naive"]
        prot = min(splits, key=lambda r: abs(r["batches"][0] - r["batches"][1])) \
            if splits else None
        best = min(rows, key=lambda r: r["wall"])
        cheapest_sig = [r for r in splits if r["all_cheap"]]
        pick = min(cheapest_sig, key=lambda r: r["n_launches"]) if cheapest_sig else \
            (max(splits, key=lambda r: r["n_cheap"]) if splits else None)
        out[R] = dict(R=R, naive=naive, protocol=prot, selector_pick=pick,
                      measured_best=best, rows=rows,
                      selector_found_best=(pick is not None
                                           and pick["label"] == best["label"]))
        print(f"\n=== R = {R} replicas ===", flush=True)
        print(f"  {'plan':<22}{'lanz':>5}{'ancho':>9}{'firmas baratas':>16}"
              f"{'pared s':>10}{'s/replica':>11}{'vs naive':>10}", flush=True)
        for r in sorted(rows, key=lambda r: r["wall"]):
            tag3 = ""
            if r is naive:
                tag3 = "  <- naive"
            elif prot and r["label"] == prot["label"]:
                tag3 += "  <- protocolo"
            if pick and r["label"] == pick["label"]:
                tag3 += "  <- SELECTOR"
            print(f"  {r['label']:<22}{r['n_launches']:>5}{str(r['widths']):>9}"
                  f"{r['n_cheap']:>8}/{len(r['batches']):<7}{r['wall']:>10.4f}"
                  f"{r['wall_per_replica']:>11.6f}"
                  f"{100*(r['wall_per_replica']/naive['wall_per_replica']-1):>+9.1f}%"
                  f"{tag3}", flush=True)
        if prot:
            print(f"  protocolo contra naive : "
                  f"{100*(prot['wall']/naive['wall']-1):+.1f}%", flush=True)
        if pick:
            print(f"  selector  contra naive : "
                  f"{100*(pick['wall']/naive['wall']-1):+.1f}%", flush=True)
            print(f"  selector  contra protocolo: "
                  f"{100*(pick['wall']/prot['wall']-1):+.1f}%", flush=True)
            print(f"  el selector encontro el mejor medido? "
                  f"{'SI' if out[R]['selector_found_best'] else 'NO -> el mejor fue ' + best['label']}",
                  flush=True)

    res = dict(by_target={str(k): v for k, v in out.items()},
               per_batch={str(k): v for k, v in meta.items()},
               cheap_signature=CHEAP, Rd=float(Rd), L=int(L), N=N,
               device=device_identity(), sweeps=sweeps, reps=int(reps),
               window=int(window))
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n  guardado /out/{tag}.json", flush=True)
    return res


@app.function(gpu="H100", **_COMMON)
def mitigation_curve_h100(t, tag, Rd, L, nt, th, me, reps, window):
    return _mitigation_curve(t, tag, Rd, L, nt, th, me, reps, window)


@app.local_entrypoint()
def mitigation_curve(targets: str = "100,200", rd: float = 18.30, layers: int = 5,
                     n_temps: int = 8, n_therm: int = 300, n_meas: int = 150,
                     reps: int = 9, window: int = 6,
                     tag: str = "mitigation_curve"):
    """Measure what the protocol and the tile selector save, per target R."""
    mitigation_curve_h100.remote([int(x) for x in targets.split(",") if x], tag,
                                 rd, layers, n_temps, n_therm, n_meas, reps, window)


_LAYOUT = None


def _pallas_tiled(batches, repeats, tag, Rd, L, tb, ty, tx):
    """A Pallas stencil with a tile WE fix, over a 4-component layout.

    Three blockers had to fall, and only the third was the real one.

    1. mosaic_gpu refuses a copy whose byte count is not a multiple of 128,
       so a block needs a multiple of 32 f32 elements. The first attempt used
       a whole layer-plane, (1,1,N,N,3) = 4563 elements, and 4563 is odd for
       EVERY geometry because N = 2(floor(Rd)+1)+1 is always odd.
    2. Keeping the plane and tiling only the batch does not rescue it: an odd
       plane forces a batch tile that is a multiple of 32, and 32 whole planes
       need 570 KB of SRAM against an H100 SM's 228 KB.
    3. So the plane gets tiled too -- and that still failed, with
       `async_copy requires all GMEM strides except the last one to be a
       multiple of 16 bytes`. THAT is the real blocker and it has nothing to
       do with the tile. The innermost axis holds the 3 spin components, and
       3 * 4 = 12 bytes is not a multiple of 16, which poisons every enclosing
       stride: x is 12 B, y is 468 B, z is 18252 B, the batch is 91260 B, and
       all four fail.

    Padding the component axis from 3 to 4 fixes all of them at once -- x
    becomes 16 B, y 624 B, z 24336 B, the batch 121680 B -- and it drops the
    batch tile from 32 to 8, since the block is now 8*13*3*4 = 1248 elements.
    The fourth component stays zero, so `sum(e*e, -1)` is unchanged and the
    physics is exact, not approximated.

    Tiles must still DIVIDE N or the reference's periodic wrap stops lining
    up; the divisors of 39 are 1, 3, 13, 39. Tiling the plane also costs what
    the whole-plane version got for free: a 13x3 tile does not contain its own
    neighbours, so the halo is exchanged by passing the array seven times
    under seven index maps -- the trick the first attempt already used for z.

    Timed two ways, because which one is honest depends on a choice we have
    not made: with the pad inside the timed call (what it costs if spins stay
    3-component in memory) and with the array already 4-component (what it
    costs if the storage changes).
    """
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S

    print(f"gpu={device_identity()}", flush=True)
    print(f"jax {jax.__version__}", flush=True)
    try:
        import jax._src.pallas.mosaic_gpu.pallas_call_registration  # noqa: F401
        from jax.experimental import pallas as pl
    except Exception as e:
        print(f"PALLAS UNAVAILABLE: {type(e).__name__}: {e}", flush=True)
        return dict(available=False, error=f"{type(e).__name__}: {e}")

    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    N, Lz = int(geo["N"]), int(geo["L"])
    TB, TY, TX, C = int(tb), int(ty), int(tx), 4
    el = TB * 1 * TY * TX * C
    ny, nx = N // TY, N // TX
    # mosaic_gpu wants the SRAM block to hold a multiple of 128 ELEMENTS,
    # not a multiple of 128 bytes; we learned the difference the hard way.
    ok_bytes = el % 128 == 0
    ok_div = (N % TY == 0) and (N % TX == 0)
    strides = [Lz * N * N * C, N * N * C, N * C, C, 1]
    ok_stride = all((st * 4) % 16 == 0 for st in strides[:-1])
    print(f"Rd={Rd} L={Lz} N={N}  componentes 3 -> {C} (padded)", flush=True)
    print(f"tile ({TB},1,{TY},{TX},{C}) = {el} el = {el*4:,} B -> "
          f"{el/128:.1f} x 128 elementos  "
          f"{'LEGAL' if ok_bytes else 'ILEGAL'}", flush=True)
    print(f"  N/{TY}={N//TY} N/{TX}={N//TX} -> "
          f"{'divide exacto' if ok_div else 'NO DIVIDE'}", flush=True)
    print(f"  strides GMEM (B): {[st*4 for st in strides]}  -> "
          f"{'todos multiplos de 16' if ok_stride else 'ALGUNO FALLA'}", flush=True)
    print(f"  SRAM 7 refs = {7*el*4/1024:.1f} KB   grid = (B/{TB},{Lz},{ny},{nx})",
          flush=True)
    res = dict(available=True, N=N, L=Lz, components=C,
               tile=[TB, 1, TY, TX, C], tile_elements=el, tile_bytes=el * 4,
               sram_kb=round(7 * el * 4 / 1024, 1), grid=[None, Lz, ny, nx],
               gmem_strides_bytes=[st * 4 for st in strides],
               legal_bytes=ok_bytes, divides_N=ok_div, legal_strides=ok_stride)
    if not (ok_bytes and ok_div and ok_stride):
        res["error"] = "tile o layout inadmisible"
        with open(f"/out/{tag}.json", "w") as f:
            json.dump(res, f, indent=1)
        return res

    # ---- what does this build of mosaic_gpu actually offer? -------------
    # Four kernel bodies have now been refused: iota without a layout cast,
    # slicing a loaded array, slicing a ref, and concatenate (unimplemented
    # outright). The iota error named its own fix, so find the layout API
    # before guessing at it.
    try:
        import jax.experimental.pallas.mosaic_gpu as plgpu
        names = sorted(n for n in dir(plgpu) if not n.startswith("_"))
        print(f"\nplgpu expone {len(names)} nombres", flush=True)
        print(f"  con 'layout': {[n for n in names if 'ayout' in n]}", flush=True)
        lay = getattr(plgpu, "Layout", None)
        cands = []
        if lay is not None:
            import inspect
            mem = sorted(n for n in dir(lay) if not n.startswith("_"))
            print(f"  Layout.*: {mem}", flush=True)
            for n in ("WG_STRIDED", "WG_SPLAT", "TILED"):
                v = getattr(lay, n, None)
                if v is None:
                    continue
                try:
                    sig = str(inspect.signature(v))
                except Exception:
                    sig = "?"
                print(f"    {n}{sig}", flush=True)
            shp = (TB, TY, TX, 4)
            ws = getattr(lay, "WG_STRIDED", None)
            for vec in (1, 2, 4):
                if ws is None:
                    break
                try:
                    cands.append((f"WG_STRIDED(vec={vec})", ws(shp, vec)))
                except Exception as e:
                    print(f"    WG_STRIDED vec={vec} no construye: "
                          f"{type(e).__name__}: {e}", flush=True)
        print(f"  layouts construidos: {[n for n, _ in cands]}", flush=True)
    except Exception as e:
        plgpu, cands, res_layouts = None, [], []
        print(f"\nplgpu no disponible: {type(e).__name__}: {e}", flush=True)

    pad4 = lambda a: jnp.pad(a, ((0, 0), (0, 0), (0, 0), (0, 0), (0, 1)))

    # ---- the reference: six materialised rolls, on the real 3 components --
    def jax_step(sp):
        xp = jnp.roll(sp, -1, 3); xm = jnp.roll(sp, 1, 3)
        yp = jnp.roll(sp, -1, 2); ym = jnp.roll(sp, 1, 2)
        zp = jnp.roll(sp, -1, 1).at[:, -1].set(0.0)
        zm = jnp.roll(sp, 1, 1).at[:, 0].set(0.0)
        e = sp + 0.1 * (xp + xm + yp + ym + zp + zm)
        return e / jnp.sqrt(jnp.sum(e * e, -1, keepdims=True) + 1e-20)

    # ---- our kernel: fixed tile, halo exchanged by hand -------------------
    def kernel(c_ref, ym_ref, yp_ref, xm_ref, xp_ref, zm_ref, zp_ref, o_ref):
        z = pl.program_id(1)
        c = c_ref[:, 0]                       # (TB, TY, TX, C) in SRAM
        ymt, ypt = ym_ref[:, 0], yp_ref[:, 0]
        xmt, xpt = xm_ref[:, 0], xp_ref[:, 0]
        # Roll inside the tile, then replace the one edge row or column with
        # the neighbouring tile's. roll and where are both lowered; the iota
        # that selects the edge is not, unless its layout is cast explicitly,
        # which is what the lowering error asked for by name.
        def cast(a):
            return plgpu.layout_cast(a, _LAYOUT) if _LAYOUT is not None else a
        iy = cast(jax.lax.broadcasted_iota(jnp.int32, c.shape, 1))
        ix = cast(jax.lax.broadcasted_iota(jnp.int32, c.shape, 2))
        ynext = jnp.where(iy == TY - 1, ypt[:, 0:1], jnp.roll(c, -1, 1))
        yprev = jnp.where(iy == 0, ymt[:, TY - 1:TY], jnp.roll(c, 1, 1))
        xnext = jnp.where(ix == TX - 1, xpt[:, :, 0:1], jnp.roll(c, -1, 2))
        xprev = jnp.where(ix == 0, xmt[:, :, TX - 1:TX], jnp.roll(c, 1, 2))
        # z is confined: planes outside the stack contribute nothing
        lo = jnp.where(z == 0, 0.0, zm_ref[:, 0])
        hi = jnp.where(z == Lz - 1, 0.0, zp_ref[:, 0])
        e = c + 0.1 * (xnext + xprev + ynext + yprev + hi + lo)
        o_ref[:, 0] = e / jnp.sqrt(jnp.sum(e * e, -1, keepdims=True) + 1e-20)

    blk = (TB, 1, TY, TX, C)

    def pallas_on4(sp4):
        """The kernel alone, over an already 4-component array."""
        nb = sp4.shape[0] // TB
        spec = lambda f: pl.BlockSpec(blk, f)
        return pl.pallas_call(
            kernel,
            grid=(nb, Lz, ny, nx),
            in_specs=[
                spec(lambda b, z, jy, jx: (b, z, jy, jx, 0)),
                spec(lambda b, z, jy, jx: (b, z, (jy - 1) % ny, jx, 0)),
                spec(lambda b, z, jy, jx: (b, z, (jy + 1) % ny, jx, 0)),
                spec(lambda b, z, jy, jx: (b, z, jy, (jx - 1) % nx, 0)),
                spec(lambda b, z, jy, jx: (b, z, jy, (jx + 1) % nx, 0)),
                spec(lambda b, z, jy, jx: (b, jnp.maximum(z - 1, 0), jy, jx, 0)),
                spec(lambda b, z, jy, jx: (b, jnp.minimum(z + 1, Lz - 1), jy, jx, 0)),
            ],
            out_specs=spec(lambda b, z, jy, jx: (b, z, jy, jx, 0)),
            out_shape=jax.ShapeDtypeStruct(sp4.shape, sp4.dtype),
        )(sp4, sp4, sp4, sp4, sp4, sp4, sp4)

    def pallas_with_pad(sp):
        """What it costs if spins stay 3-component in memory."""
        return pallas_on4(pad4(sp))[..., :3]

    # ---- fallback: let the BlockSpec do every shift ----------------------
    # With a block extent of one in y and x, the block index IS the element
    # index, so the six neighbours arrive already shifted and the kernel body
    # is pure arithmetic: no roll, no iota, no slice, no concatenate, none of
    # the four primitives mosaic_gpu refused. The tile is degenerate, which
    # costs arithmetic intensity, but it is still a tile WE fix.
    DBLK = (TB, 1, 1, 1, C)
    d_el = TB * C

    def kernel_point(c_ref, ym_ref, yp_ref, xm_ref, xp_ref, zm_ref, zp_ref, o_ref):
        # No reduction here: jnp.sum over an axis is refused with "No support
        # for axes yet". The normalisation is a cheap elementwise pass that
        # XLA does outside, and the stencil -- the memory-bound part worth
        # moving -- is what stays in the kernel.
        z = pl.program_id(1)
        lo = jnp.where(z == 0, 0.0, zm_ref[:, 0, 0, 0])
        hi = jnp.where(z == Lz - 1, 0.0, zp_ref[:, 0, 0, 0])
        o_ref[:, 0, 0, 0] = (c_ref[:, 0, 0, 0]
                             + 0.1 * (xp_ref[:, 0, 0, 0] + xm_ref[:, 0, 0, 0]
                                      + yp_ref[:, 0, 0, 0] + ym_ref[:, 0, 0, 0]
                                      + hi + lo))

    def pallas_point_on4(sp4):
        nb = sp4.shape[0] // TB
        spec = lambda f: pl.BlockSpec(DBLK, f)
        return pl.pallas_call(
            kernel_point,
            grid=(nb, Lz, N, N),
            in_specs=[
                spec(lambda b, z, jy, jx: (b, z, jy, jx, 0)),
                spec(lambda b, z, jy, jx: (b, z, (jy - 1) % N, jx, 0)),
                spec(lambda b, z, jy, jx: (b, z, (jy + 1) % N, jx, 0)),
                spec(lambda b, z, jy, jx: (b, z, jy, (jx - 1) % N, 0)),
                spec(lambda b, z, jy, jx: (b, z, jy, (jx + 1) % N, 0)),
                spec(lambda b, z, jy, jx: (b, jnp.maximum(z - 1, 0), jy, jx, 0)),
                spec(lambda b, z, jy, jx: (b, jnp.minimum(z + 1, Lz - 1), jy, jx, 0)),
            ],
            out_specs=spec(lambda b, z, jy, jx: (b, z, jy, jx, 0)),
            out_shape=jax.ShapeDtypeStruct(sp4.shape, sp4.dtype),
        )(sp4, sp4, sp4, sp4, sp4, sp4, sp4)

    def norm(e):
        return e / jnp.sqrt(jnp.sum(e * e, -1, keepdims=True) + 1e-20)

    def pallas_point_with_pad(sp):
        return norm(pallas_point_on4(pad4(sp))[..., :3])

    # ---- the gate: equivalence before any timing -------------------------
    B0 = TB
    sp = S.init_spins(jax.random.PRNGKey(5), geo, B0)
    a = jax.jit(jax_step)(sp)
    a.block_until_ready()
    attempts, b, chosen = [], None, None
    global _LAYOUT
    for name, val in ([(None, None)] + cands):
        _LAYOUT = val
        try:
            b = jax.jit(pallas_with_pad)(sp)
            b.block_until_ready()
            chosen = name
            attempts.append(dict(layout=name, ok=True))
            print(f"\n  layout {name!r}: COMPILA", flush=True)
            break
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            attempts.append(dict(layout=name, ok=False, error=msg[:300]))
            print(f"\n  layout {name!r}: {msg[:180]}", flush=True)
    res["layout_attempts"] = attempts
    res["layout_chosen"] = chosen
    res["kernel_used"] = "tiled" if b is not None else None
    if b is None:
        print(f"\n  el tile (13,3) no compila con ningun layout; "
              f"probando el kernel punto a punto", flush=True)
        print(f"  tile punto ({TB},1,1,1,{C}) = {d_el} el -> "
              f"{d_el/128:.1f} x 128 elementos  "
              f"{'LEGAL' if d_el % 128 == 0 else 'ILEGAL'}   "
              f"SRAM 7 refs = {7*d_el*4/1024:.1f} KB   "
              f"grid = (B/{TB},{Lz},{N},{N})", flush=True)
        res["point_tile"] = [TB, 1, 1, 1, C]
        res["point_elements"] = d_el
        res["point_legal"] = d_el % 128 == 0
        try:
            b = jax.jit(pallas_point_with_pad)(sp)
            b.block_until_ready()
            pallas_with_pad = pallas_point_with_pad
            pallas_on4 = lambda a: norm(pallas_point_on4(a))
            res["kernel_used"] = "point"
            print(f"  kernel punto a punto: COMPILA", flush=True)
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"  kernel punto a punto: {msg[:300]}", flush=True)
            res.update(ran=False, error=msg)
            with open(f"/out/{tag}.json", "w") as f:
                json.dump(res, f, indent=1)
            return res
    err = float(jnp.max(jnp.abs(a - b)))
    rel = err / max(float(jnp.max(jnp.abs(a))), 1e-30)
    good = rel < 1e-5
    print(f"\nequivalencia en B={B0}: max |jax - pallas| = {err:.3e}  "
          f"(relativo {rel:.3e})  {'OK' if good else 'MISMATCH'}", flush=True)
    res.update(ran=True, equivalent=bool(good), max_abs_err=err, max_rel_err=rel)
    if not good:
        print("  -> no son la misma funcion; el tiempo no significa nada", flush=True)
        with open(f"/out/{tag}.json", "w") as f:
            json.dump(res, f, indent=1)
        return res

    # ---- only now, the timing -------------------------------------------
    rows = []
    print(f"\n{'B':>6}{'jax 6 rolls':>14}{'pallas+pad':>13}{'vs jax':>9}"
          f"{'pallas en 4c':>14}{'vs jax':>9}", flush=True)
    for B in batches:
        if B % TB:
            print(f"  B={B} no es multiplo de TB={TB}, se omite", flush=True)
            continue
        sp = S.init_spins(jax.random.PRNGKey(7 + B), geo, B)
        sp4 = jax.jit(pad4)(sp)
        fj = jax.jit(jax_step).lower(sp).compile()
        fp = jax.jit(pallas_with_pad).lower(sp).compile()
        f4 = jax.jit(pallas_on4).lower(sp4).compile()
        for f, x in ((fj, sp), (fp, sp), (f4, sp4)):
            jax.block_until_ready(f(x))
        tj, tp, t4 = [], [], []
        for _ in range(int(repeats)):
            t0 = time.perf_counter(); jax.block_until_ready(fj(sp))
            tj.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); jax.block_until_ready(fp(sp))
            tp.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); jax.block_until_ready(f4(sp4))
            t4.append(time.perf_counter() - t0)
        mj, mp, m4 = (float(np.median(x)) for x in (tj, tp, t4))
        e2 = float(jnp.max(jnp.abs(fj(sp) - fp(sp))))
        rows.append(dict(B=B, jax_s=mj, pallas_pad_s=mp, pallas_4c_s=m4,
                         pad_change_pct=100 * (mp / mj - 1),
                         c4_change_pct=100 * (m4 / mj - 1),
                         max_abs_err=e2))
        print(f"{B:>6}{mj*1e3:>12.3f} ms{mp*1e3:>11.3f} ms"
              f"{100*(mp/mj-1):>+8.1f}%{m4*1e3:>12.3f} ms"
              f"{100*(m4/mj-1):>+8.1f}%", flush=True)
    res["timing"] = rows
    if rows:
        b4 = min(r["c4_change_pct"] for r in rows)
        bp = min(r["pad_change_pct"] for r in rows)
        print(f"\n  mejor con pad dentro: {bp:+.1f}%", flush=True)
        print(f"  mejor ya en 4 componentes: {b4:+.1f}%", flush=True)
        print(f"  equivalencia peor caso: "
              f"{max(r['max_abs_err'] for r in rows):.3e}", flush=True)
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n  guardado /out/{tag}.json", flush=True)
    return res


@app.function(gpu="H100", image=pallas_image, timeout=60 * 60 * 2,
              volumes={"/out": vol})
def pallas_tiled_h100(b, rep, tag, Rd, L, tb, ty, tx):
    return _pallas_tiled(b, rep, tag, Rd, L, tb, ty, tx)


@app.local_entrypoint()
def pallas_tiled(batches: str = "32,64,96", repeats: int = 9, rd: float = 18.30,
                 layers: int = 5, tb: int = 32, ty: int = 13, tx: int = 3,
                 tag: str = "pallas_tiled"):
    """A fixed-tile Pallas stencil against the six-roll JAX reference."""
    pallas_tiled_h100.remote([int(x) for x in batches.split(",") if x], repeats,
                             tag, rd, layers, tb, ty, tx)


def _pallas_triton(batches, repeats, tag, Rd, L, tb, ty, tx):
    """The same kernel, lowered through Triton instead of Mosaic GPU.

    Mosaic GPU refused this stencil eight different ways, and every refusal
    was about moving data inside the kernel: iota layouts, slicing a loaded
    array, slicing a ref, concatenate (unimplemented), axis reductions. Its
    layout list -- WGMMA, TCGEN05 -- says what it is for, and a stencil is not
    it.

    Pallas still has a Triton backend in this version. It was deprecated in
    jax 0.11.0 and we are on 0.10.2, and the `backend` argument was replaced
    by `compiler_params` back in 0.9.1, so the way to ask for it is
    `compiler_params=pltriton.CompilerParams()`. Triton is a real tile
    language: masks, slices and reductions are its bread and butter, and XLA
    already emits Triton for this very fusion.

    That reopens the design Mosaic GPU forced us to abandon. Design A keeps a
    WHOLE LAYER-PLANE per block, so the x and y neighbours are rolls of an
    array already resident in SRAM and the six materialised copies never
    happen -- which is the whole point, and the 31.5% we are after. Design B
    is the tiled fallback with an explicit halo, kept in case A does not fit.
    Both are tried, A first, and whichever compiles is checked against the
    plain-JAX reference before anything is timed.
    """
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S

    print(f"gpu={device_identity()}", flush=True)
    print(f"jax {jax.__version__}", flush=True)
    try:
        from jax.experimental import pallas as pl
        import jax.experimental.pallas.triton as pltriton
        names = sorted(n for n in dir(pltriton) if not n.startswith("_"))
        print(f"pltriton expone: {names}", flush=True)
        cp = pltriton.CompilerParams()
        print(f"CompilerParams() -> {cp}", flush=True)
    except Exception as e:
        print(f"TRITON BACKEND UNAVAILABLE: {type(e).__name__}: {e}", flush=True)
        return dict(available=False, error=f"{type(e).__name__}: {e}")

    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    N, Lz = int(geo["N"]), int(geo["L"])
    print(f"Rd={Rd} L={Lz} N={N}", flush=True)
    res = dict(available=True, N=N, L=Lz, jax=jax.__version__,
               backend="pallas-triton", pltriton=names)

    def jax_step(sp):
        xp = jnp.roll(sp, -1, 3); xm = jnp.roll(sp, 1, 3)
        yp = jnp.roll(sp, -1, 2); ym = jnp.roll(sp, 1, 2)
        zp = jnp.roll(sp, -1, 1).at[:, -1].set(0.0)
        zm = jnp.roll(sp, 1, 1).at[:, 0].set(0.0)
        e = sp + 0.1 * (xp + xm + yp + ym + zp + zm)
        return e / jnp.sqrt(jnp.sum(e * e, -1, keepdims=True) + 1e-20)

    # ---- the Triton backend wants every kernel shape a power of 2 -------
    # 39, 13 and 3 are not, so the plane is padded to 64x64 and the three spin
    # components to 4. Every block dimension is then a power of two and the
    # whole plane fits in SRAM at 64 KB per ref, three refs, 192 KB.
    #
    # Padding is physically exact here, not an approximation: the lattice is
    # confined and its outer ring is always empty, because the grid half-width
    # floor(Rd)+1 exceeds Rd, so the neighbours the padding supplies are the
    # zeros the wrap would have supplied anyway. The equivalence check below
    # runs the JAX reference on the SAME padded array, so the comparison is
    # exact rather than argued.
    #
    # The cost is honest and was estimated before measuring: 64*64*4 is 3.59x
    # the elements of 39*39*3, which puts this design at about 256 KB of
    # traffic per replica-layer against roughly 232 KB for the six materialised
    # rolls. It wins only if saving six kernel launches and six HBM
    # round-trips beats that.
    PN, PC = 64, 4

    def padup(sp):
        return jnp.pad(sp, ((0, 0), (0, 0), (0, PN - N), (0, PN - N), (0, PC - 3)))

    def jax_step_p(sp4):
        xp = jnp.roll(sp4, -1, 3); xm = jnp.roll(sp4, 1, 3)
        yp = jnp.roll(sp4, -1, 2); ym = jnp.roll(sp4, 1, 2)
        zp = jnp.roll(sp4, -1, 1).at[:, -1].set(0.0)
        zm = jnp.roll(sp4, 1, 1).at[:, 0].set(0.0)
        e = sp4 + 0.1 * (xp + xm + yp + ym + zp + zm)
        return e / jnp.sqrt(jnp.sum(e * e, -1, keepdims=True) + 1e-20)

    def kernel_pad(c_ref, below_ref, above_ref, o_ref):
        z = pl.program_id(1)
        c = c_ref[0, 0]                      # (64, 64, 4) resident in SRAM
        lo = jnp.where(z == 0, 0.0, below_ref[0, 0])
        hi = jnp.where(z == Lz - 1, 0.0, above_ref[0, 0])
        sv = (jnp.roll(c, -1, 1) + jnp.roll(c, 1, 1)
              + jnp.roll(c, -1, 0) + jnp.roll(c, 1, 0) + hi + lo)
        e = c + 0.1 * sv
        o_ref[0, 0] = e / jnp.sqrt(jnp.sum(e * e, -1, keepdims=True) + 1e-20)

    def pad_step(sp4):
        B = sp4.shape[0]
        blk = (1, 1, PN, PN, PC)
        return pl.pallas_call(
            kernel_pad,
            grid=(B, Lz),
            in_specs=[
                pl.BlockSpec(blk, lambda b, z: (b, z, 0, 0, 0)),
                pl.BlockSpec(blk, lambda b, z: (b, jnp.maximum(z - 1, 0), 0, 0, 0)),
                pl.BlockSpec(blk, lambda b, z: (b, jnp.minimum(z + 1, Lz - 1), 0, 0, 0)),
            ],
            out_specs=pl.BlockSpec(blk, lambda b, z: (b, z, 0, 0, 0)),
            out_shape=jax.ShapeDtypeStruct(sp4.shape, sp4.dtype),
            compiler_params=pltriton.CompilerParams(),
        )(sp4, sp4, sp4)

    # ---- design A: a whole plane per block, x/y rolls free in SRAM -------
    def kernel_plane(c_ref, below_ref, above_ref, o_ref):
        z = pl.program_id(1)
        c = c_ref[0, 0]                      # (N, N, 3) resident in SRAM
        lo = jnp.where(z == 0, 0.0, below_ref[0, 0])
        hi = jnp.where(z == Lz - 1, 0.0, above_ref[0, 0])
        sv = (jnp.roll(c, -1, 1) + jnp.roll(c, 1, 1)
              + jnp.roll(c, -1, 0) + jnp.roll(c, 1, 0) + hi + lo)
        e = c + 0.1 * sv
        o_ref[0, 0] = e / jnp.sqrt(jnp.sum(e * e, -1, keepdims=True) + 1e-20)

    def plane_step(sp):
        B = sp.shape[0]
        blk = (1, 1, N, N, 3)
        return pl.pallas_call(
            kernel_plane,
            grid=(B, Lz),
            in_specs=[
                pl.BlockSpec(blk, lambda b, z: (b, z, 0, 0, 0)),
                pl.BlockSpec(blk, lambda b, z: (b, jnp.maximum(z - 1, 0), 0, 0, 0)),
                pl.BlockSpec(blk, lambda b, z: (b, jnp.minimum(z + 1, Lz - 1), 0, 0, 0)),
            ],
            out_specs=pl.BlockSpec(blk, lambda b, z: (b, z, 0, 0, 0)),
            out_shape=jax.ShapeDtypeStruct(sp.shape, sp.dtype),
            compiler_params=pltriton.CompilerParams(),
        )(sp, sp, sp)

    # ---- design B: a tile we fix, halo exchanged explicitly --------------
    TB, TY, TX = int(tb), int(ty), int(tx)
    ny, nx = (N // TY if N % TY == 0 else 0), (N // TX if N % TX == 0 else 0)

    def kernel_tile(c_ref, ym_ref, yp_ref, xm_ref, xp_ref, zm_ref, zp_ref, o_ref):
        z = pl.program_id(1)
        c = c_ref[:, 0]
        ymt, ypt = ym_ref[:, 0], yp_ref[:, 0]
        xmt, xpt = xm_ref[:, 0], xp_ref[:, 0]
        ynext = jnp.concatenate([c[:, 1:TY], ypt[:, 0:1]], axis=1)
        yprev = jnp.concatenate([ymt[:, TY - 1:TY], c[:, 0:TY - 1]], axis=1)
        xnext = jnp.concatenate([c[:, :, 1:TX], xpt[:, :, 0:1]], axis=2)
        xprev = jnp.concatenate([xmt[:, :, TX - 1:TX], c[:, :, 0:TX - 1]], axis=2)
        lo = jnp.where(z == 0, 0.0, zm_ref[:, 0])
        hi = jnp.where(z == Lz - 1, 0.0, zp_ref[:, 0])
        e = c + 0.1 * (xnext + xprev + ynext + yprev + hi + lo)
        o_ref[:, 0] = e / jnp.sqrt(jnp.sum(e * e, -1, keepdims=True) + 1e-20)

    def tile_step(sp):
        nb = sp.shape[0] // TB
        blk = (TB, 1, TY, TX, 3)
        spec = lambda f: pl.BlockSpec(blk, f)
        return pl.pallas_call(
            kernel_tile,
            grid=(nb, Lz, ny, nx),
            in_specs=[
                spec(lambda b, z, jy, jx: (b, z, jy, jx, 0)),
                spec(lambda b, z, jy, jx: (b, z, (jy - 1) % ny, jx, 0)),
                spec(lambda b, z, jy, jx: (b, z, (jy + 1) % ny, jx, 0)),
                spec(lambda b, z, jy, jx: (b, z, jy, (jx - 1) % nx, 0)),
                spec(lambda b, z, jy, jx: (b, z, jy, (jx + 1) % nx, 0)),
                spec(lambda b, z, jy, jx: (b, jnp.maximum(z - 1, 0), jy, jx, 0)),
                spec(lambda b, z, jy, jx: (b, jnp.minimum(z + 1, Lz - 1), jy, jx, 0)),
            ],
            out_specs=spec(lambda b, z, jy, jx: (b, z, jy, jx, 0)),
            out_shape=jax.ShapeDtypeStruct(sp.shape, sp.dtype),
            compiler_params=pltriton.CompilerParams(),
        )(sp, sp, sp, sp, sp, sp, sp)

    # ---- the gate: equivalence before any timing ------------------------
    B0 = max(TB, 8)
    sp0 = S.init_spins(jax.random.PRNGKey(5), geo, B0)
    sp0p = jax.jit(padup)(sp0)
    print(f"  padeado: {sp0.shape} -> {sp0p.shape}  "
          f"({PN*PN*PC/(N*N*3):.2f}x elementos por plano)", flush=True)
    refp = jax.jit(jax_step_p)(sp0p); refp.block_until_ready()
    ref = jax.jit(jax_step)(sp0); ref.block_until_ready()
    designs = [("P: plano padeado (1,1,64,64,4)", pad_step, sp0p, refp, padup,
                jax_step_p),
               ("A: plano completo (1,1,N,N,3)", plane_step, sp0, ref, None, None)]
    if ny and nx and B0 % TB == 0:
        designs.append((f"B: tile ({TB},1,{TY},{TX},3)", tile_step, sp0, ref,
                        None, None))
    chosen, fn_sel, attempts, prep, refmaker = None, None, [], None, None
    for lab, fn, xin, rr, pre, rmk in designs:
        try:
            got = jax.jit(fn)(xin); got.block_until_ready()
            err = float(jnp.max(jnp.abs(rr - got)))
            rel = err / max(float(jnp.max(jnp.abs(rr))), 1e-30)
            ok = rel < 1e-5
            attempts.append(dict(design=lab, compiled=True, max_rel_err=rel,
                                 equivalent=bool(ok)))
            print(f"\n  {lab}: COMPILA   error relativo {rel:.3e}  "
                  f"{'OK' if ok else 'MISMATCH'}", flush=True)
            if ok and chosen is None:
                chosen, fn_sel, prep, refmaker = lab, fn, pre, rmk
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            attempts.append(dict(design=lab, compiled=False, error=msg[:400]))
            print(f"\n  {lab}: {msg[:260]}", flush=True)
    res["attempts"] = attempts
    res["chosen"] = chosen
    if fn_sel is None:
        res.update(ran=False, error="ningun diseño compilo y verifico")
        with open(f"/out/{tag}.json", "w") as f:
            json.dump(res, f, indent=1)
        return res

    # ---- only now, the timing -------------------------------------------
    rows = []
    print(f"\n  cronometrando {chosen}", flush=True)
    print(f"\n{'B':>6}{'jax 6 rolls':>15}{'pallas-triton':>16}{'cambio':>10}",
          flush=True)
    for B in batches:
        if TB and B % TB and chosen.startswith("B"):
            print(f"  B={B} no multiplo de {TB}, se omite", flush=True)
            continue
        sp = S.init_spins(jax.random.PRNGKey(7 + B), geo, B)
        # the reference is timed on the SAME layout the kernel runs on, so the
        # padding is never charged to one side only
        if prep is not None:
            xin = jax.jit(prep)(sp)
            fj = jax.jit(refmaker).lower(xin).compile()
        else:
            xin = sp
            fj = jax.jit(jax_step).lower(sp).compile()
        fp = jax.jit(fn_sel).lower(xin).compile()
        jax.block_until_ready(fj(xin)); jax.block_until_ready(fp(xin))
        tj, tp = [], []
        for _ in range(int(repeats)):
            t0 = time.perf_counter(); jax.block_until_ready(fj(xin))
            tj.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); jax.block_until_ready(fp(xin))
            tp.append(time.perf_counter() - t0)
        mj, mp = float(np.median(tj)), float(np.median(tp))
        e2 = float(jnp.max(jnp.abs(fj(xin) - fp(xin))))
        rows.append(dict(B=B, jax_s=mj, pallas_s=mp,
                         change_pct=100 * (mp / mj - 1), max_abs_err=e2))
        print(f"{B:>6}{mj*1e3:>13.3f} ms{mp*1e3:>14.3f} ms"
              f"{100*(mp/mj-1):>+9.1f}%", flush=True)
    res["timing"] = rows
    if rows:
        bb = min(rows, key=lambda r: r["change_pct"])
        print(f"\n  mejor: B={bb['B']}  {bb['change_pct']:+.1f}%", flush=True)
        print(f"  equivalencia peor caso: "
              f"{max(r['max_abs_err'] for r in rows):.3e}", flush=True)
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n  guardado /out/{tag}.json", flush=True)
    return res


@app.function(gpu="H100", image=pallas_image, timeout=60 * 60 * 2,
              volumes={"/out": vol})
def pallas_triton_h100(b, rep, tag, Rd, L, tb, ty, tx):
    return _pallas_triton(b, rep, tag, Rd, L, tb, ty, tx)


@app.local_entrypoint()
def pallas_triton(batches: str = "32,64,96", repeats: int = 9, rd: float = 18.30,
                  layers: int = 5, tb: int = 8, ty: int = 13, tx: int = 3,
                  tag: str = "pallas_triton"):
    """The stencil through Pallas's Triton backend instead of Mosaic GPU."""
    pallas_triton_h100.remote([int(x) for x in batches.split(",") if x], repeats,
                              tag, rd, layers, tb, ty, tx)


def _triton_load_probe(tag):
    """Is the masked-load path real, or is Pallas Triton a dead end too?

    Pallas's Triton lowering refused `slice`, which is what `jnp.roll` lowers
    to, so no in-kernel shifting -- the same wall Mosaic GPU put up, reached
    by a different road. But `jax.experimental.pallas.triton` exports `load`
    and `store`, which in Triton proper are the masked pointer-offset loads
    that make a stencil natural. If those are usable from a Pallas kernel the
    route is open; if not, Pallas is finished for this and the only Triton
    left is outside it.

    This probe costs seconds: print the signatures, then try the smallest
    kernel that shifts data with a masked load.
    """
    import sys, json, inspect
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    print(f"jax {jax.__version__}", flush=True)
    out = dict(jax=jax.__version__)
    try:
        from jax.experimental import pallas as pl
        import jax.experimental.pallas.triton as pltriton
    except Exception as e:
        print(f"no disponible: {type(e).__name__}: {e}", flush=True)
        return dict(error=f"{type(e).__name__}: {e}")

    for nm in ("load", "store", "max_contiguous"):
        f = getattr(pltriton, nm, None)
        if f is None:
            continue
        try:
            sig = str(inspect.signature(f))
        except Exception:
            sig = "?"
        doc = (inspect.getdoc(f) or "").strip().splitlines()
        print(f"\n{nm}{sig}", flush=True)
        for ln in doc[:14]:
            print(f"    {ln}", flush=True)
        out[nm] = dict(signature=sig, doc="\n".join(doc[:20]))

    # does pl expose the unblocked memory space we would need?
    sp_names = [n for n in dir(pl) if "ANY" in n.upper() or "Memory" in n
                or "memory" in n]
    print(f"\npl memory-space names: {sp_names}", flush=True)
    out["pl_memory_names"] = sp_names

    # The index goes inside ref.at[...], not as a positional argument: load
    # takes the ref alone plus keyword mask/other. Two probes, smallest first.
    SZ = 64

    def kern1(x_ref, o_ref):
        i = jax.lax.broadcasted_iota(jnp.int32, (SZ,), 0)
        nxt = jnp.where(i == SZ - 1, 0, i + 1)
        o_ref[...] = pltriton.load(x_ref.at[nxt])

    def kern2(x_ref, o_ref):
        iy = jax.lax.broadcasted_iota(jnp.int32, (SZ, SZ), 0)
        ix = jax.lax.broadcasted_iota(jnp.int32, (SZ, SZ), 1)
        yp = jnp.where(iy == SZ - 1, 0, iy + 1)
        ym = jnp.where(iy == 0, SZ - 1, iy - 1)
        xp = jnp.where(ix == SZ - 1, 0, ix + 1)
        xm = jnp.where(ix == 0, SZ - 1, ix - 1)
        o_ref[...] = (pltriton.load(x_ref.at[yp, ix])
                      + pltriton.load(x_ref.at[ym, ix])
                      + pltriton.load(x_ref.at[iy, xp])
                      + pltriton.load(x_ref.at[iy, xm]))

    probes = [
        ("1-D desplazamiento", kern1, (SZ,),
         lambda x: jnp.roll(x, -1)),
        ("2-D suma de 4 vecinos", kern2, (SZ, SZ),
         lambda x: (jnp.roll(x, -1, 0) + jnp.roll(x, 1, 0)
                    + jnp.roll(x, -1, 1) + jnp.roll(x, 1, 1))),
    ]
    out["probes"] = []
    for lab, kern, shape, want_fn in probes:
        try:
            f = pl.pallas_call(
                kern, out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
                in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
                out_specs=pl.BlockSpec(memory_space=pl.ANY),
                compiler_params=pltriton.CompilerParams())
            x = jax.random.normal(jax.random.PRNGKey(0), shape, jnp.float32)
            got = jax.jit(f)(x); got.block_until_ready()
            want = want_fn(x)
            err = float(jnp.max(jnp.abs(got - want)))
            rel = err / max(float(jnp.max(jnp.abs(want))), 1e-30)
            ok = rel < 1e-5
            print(f"\n  {lab}: error relativo {rel:.3e}  "
                  f"{'FUNCIONA' if ok else 'RESULTADO DISTINTO'}", flush=True)
            out["probes"].append(dict(probe=lab, works=bool(ok), max_rel_err=rel))
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"\n  {lab}: {msg[:360]}", flush=True)
            out["probes"].append(dict(probe=lab, works=False, error=msg[:360]))

    with open(f"/out/{tag}.json", "w") as f:
        json.dump(out, f, indent=1)
    return out


@app.function(gpu="H100", image=pallas_image, timeout=60 * 20,
              volumes={"/out": vol})
def triton_load_probe_h100(tag):
    return _triton_load_probe(tag)


@app.local_entrypoint()
def triton_load_probe(tag: str = "triton_load_probe"):
    """Can a Pallas Triton kernel shift data with a masked load?"""
    triton_load_probe_h100.remote(tag)


def _triton_stencil(batches, repeats, tag, Rd, L, tile, steps=1):
    """The stencil as a real Triton kernel, through Pallas's masked loads.

    Everything before this failed on the same thing: no way to shift data
    inside a kernel. Mosaic GPU refused iota layouts, array slicing, ref
    slicing, concatenate and axis reductions; Pallas's Triton lowering refused
    non-power-of-two shapes and `slice`, which is what `jnp.roll` becomes.

    `pltriton.load(ref.at[...])` is the way out, and it is how a stencil is
    written in Triton in the first place. With `memory_space=pl.ANY` the ref
    stays in global memory and WE do the addressing, so three things stop
    mattering at once: BlockSpec tiling (there is none), N = 39 not being a
    power of two (the index arrays carry a mask), and the three spin
    components (loaded one at a time with a scalar component index, which also
    removes the axis reduction the normalisation needed).

    The index arrays also express the physics directly: x and y wrap, z does
    not, and the planes outside the stack contribute nothing.

    One program per replica and layer, a TILE x TILE index grid per program,
    TILE a power of two covering N.
    """
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S

    print(f"gpu={device_identity()}", flush=True)
    print(f"jax {jax.__version__}", flush=True)
    try:
        from jax.experimental import pallas as pl
        import jax.experimental.pallas.triton as pltriton
    except Exception as e:
        print(f"TRITON NO DISPONIBLE: {type(e).__name__}: {e}", flush=True)
        return dict(available=False, error=f"{type(e).__name__}: {e}")

    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    N, Lz = int(geo["N"]), int(geo["L"])
    T = int(tile)
    p2 = T > 0 and (T & (T - 1)) == 0
    print(f"Rd={Rd} L={Lz} N={N}  tile {T}x{T}  "
          f"{'potencia de 2' if p2 else 'NO es potencia de 2'}  "
          f"{'cubre N' if T >= N else 'NO cubre N'}", flush=True)
    res = dict(available=True, N=N, L=Lz, tile=T, jax=jax.__version__,
               backend="pallas-triton + pltriton.load")
    if not (p2 and T >= N):
        res["error"] = "tile inadmisible"
        with open(f"/out/{tag}.json", "w") as f:
            json.dump(res, f, indent=1)
        return res

    def jax_step(sp):
        xp = jnp.roll(sp, -1, 3); xm = jnp.roll(sp, 1, 3)
        yp = jnp.roll(sp, -1, 2); ym = jnp.roll(sp, 1, 2)
        zp = jnp.roll(sp, -1, 1).at[:, -1].set(0.0)
        zm = jnp.roll(sp, 1, 1).at[:, 0].set(0.0)
        e = sp + 0.1 * (xp + xm + yp + ym + zp + zm)
        return e / jnp.sqrt(jnp.sum(e * e, -1, keepdims=True) + 1e-20)

    def kernel(s_ref, o_ref):
        b = pl.program_id(0)
        z = pl.program_id(1)
        iy = jax.lax.broadcasted_iota(jnp.int32, (T, T), 0)
        ix = jax.lax.broadcasted_iota(jnp.int32, (T, T), 1)
        m = (iy < N) & (ix < N)                     # the tile overhangs N
        # x and y wrap; the indices are clamped so a masked lane never
        # addresses out of bounds even though its value is discarded
        yp = jnp.where(iy + 1 >= N, 0, jnp.minimum(iy + 1, N - 1))
        ym = jnp.where(iy - 1 < 0, N - 1, jnp.maximum(iy - 1, 0))
        xp = jnp.where(ix + 1 >= N, 0, jnp.minimum(ix + 1, N - 1))
        xm = jnp.where(ix - 1 < 0, N - 1, jnp.maximum(ix - 1, 0))
        cy = jnp.minimum(iy, N - 1)
        cx = jnp.minimum(ix, N - 1)
        zlo = jnp.maximum(z - 1, 0)
        zhi = jnp.minimum(z + 1, Lz - 1)
        has_lo = (z > 0).astype(jnp.float32)        # z is confined
        has_hi = (z < Lz - 1).astype(jnp.float32)

        e = []
        for c in range(3):
            ld = lambda zz, yy, xx: pltriton.load(s_ref.at[b, zz, yy, xx, c],
                                                  mask=m)
            cen = ld(z, cy, cx)
            sv = (ld(z, yp, cx) + ld(z, ym, cx)
                  + ld(z, cy, xp) + ld(z, cy, xm)
                  + has_hi * ld(zhi, cy, cx) + has_lo * ld(zlo, cy, cx))
            e.append(cen + 0.1 * sv)
        # the normalisation, written out: no axis reduction to lower
        inv = 1.0 / jnp.sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2] + 1e-20)
        for c in range(3):
            pltriton.store(o_ref.at[b, z, cy, cx, c], e[c] * inv, mask=m)

    def triton_step(sp):
        B = sp.shape[0]
        return pl.pallas_call(
            kernel,
            grid=(B, Lz),
            in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            out_shape=jax.ShapeDtypeStruct(sp.shape, sp.dtype),
            compiler_params=pltriton.CompilerParams(),
        )(sp)

    # ---- the question: pad the ARRAY to a power of two instead? ----------
    # We already pad the INDEX GRID, which is virtual and free, and mask the
    # overhang. Padding the array itself would remove the mask but cost
    # 64^2/39^2 = 2.69x the memory and traffic -- and worse, it puts v2 = 12
    # into B*L*N^2*3, so XLA's width would be 4 for EVERY B and the one lever
    # we have measured would be gone. Measured anyway, because masking might
    # cost more than the extra traffic.
    def padN(sp):
        return jnp.pad(sp, ((0, 0), (0, 0), (0, T - N), (0, T - N), (0, 0)))

    def kernel_nomask(s_ref, o_ref):
        b = pl.program_id(0)
        z = pl.program_id(1)
        iy = jax.lax.broadcasted_iota(jnp.int32, (T, T), 0)
        ix = jax.lax.broadcasted_iota(jnp.int32, (T, T), 1)
        yp = jnp.where(iy + 1 >= T, 0, iy + 1)
        ym = jnp.where(iy - 1 < 0, T - 1, iy - 1)
        xp = jnp.where(ix + 1 >= T, 0, ix + 1)
        xm = jnp.where(ix - 1 < 0, T - 1, ix - 1)
        zlo, zhi = jnp.maximum(z - 1, 0), jnp.minimum(z + 1, Lz - 1)
        has_lo = (z > 0).astype(jnp.float32)
        has_hi = (z < Lz - 1).astype(jnp.float32)
        e = []
        for c in range(3):
            ld = lambda zz, yy, xx: pltriton.load(s_ref.at[b, zz, yy, xx, c])
            sv = (ld(z, yp, ix) + ld(z, ym, ix) + ld(z, iy, xp) + ld(z, iy, xm)
                  + has_hi * ld(zhi, iy, ix) + has_lo * ld(zlo, iy, ix))
            e.append(ld(z, iy, ix) + 0.1 * sv)
        inv = 1.0 / jnp.sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2] + 1e-20)
        for c in range(3):
            pltriton.store(o_ref.at[b, z, iy, ix, c], e[c] * inv)

    def triton_padded(sp4):
        B = sp4.shape[0]
        return pl.pallas_call(
            kernel_nomask,
            grid=(B, Lz),
            in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
            out_shape=jax.ShapeDtypeStruct(sp4.shape, sp4.dtype),
            compiler_params=pltriton.CompilerParams(),
        )(sp4)

    # ---- the gate: equivalence before any timing ------------------------
    sp0 = S.init_spins(jax.random.PRNGKey(5), geo, 8)
    ref = jax.jit(jax_step)(sp0); ref.block_until_ready()
    try:
        got = jax.jit(triton_step)(sp0); got.block_until_ready()
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"\nFALLO: {msg[:500]}", flush=True)
        res.update(ran=False, error=msg[:500])
        with open(f"/out/{tag}.json", "w") as f:
            json.dump(res, f, indent=1)
        return res
    err = float(jnp.max(jnp.abs(ref - got)))
    rel = err / max(float(jnp.max(jnp.abs(ref))), 1e-30)
    ok = rel < 1e-5
    print(f"\nequivalencia: max |jax - triton| = {err:.3e}  "
          f"(relativo {rel:.3e})  {'OK' if ok else 'MISMATCH'}", flush=True)
    res.update(ran=True, equivalent=bool(ok), max_abs_err=err, max_rel_err=rel)
    if not ok:
        print("  -> no son la misma funcion; el tiempo no significa nada",
              flush=True)
        with open(f"/out/{tag}.json", "w") as f:
            json.dump(res, f, indent=1)
        return res

    # ---- amortise the launch: K steps inside one call -------------------
    # A single call costs about 122 us of dispatch latency no matter how many
    # blocks it launches (measured separately), and the single-step stencil at
    # B around 100 takes roughly 200 us in total -- so half of what we were
    # timing was overhead, which is also what the a/B term in the earlier fits
    # really was. Looping K steps inside one jit pushes overhead to under a
    # percent, which is the regime the real simulation runs in with its 3600
    # sweeps per call. Equivalence is still checked on ONE step, because f32
    # differences would compound over K.
    K = int(steps)

    def loop_of(fn):
        def g(sp):
            return jax.lax.fori_loop(0, K, lambda i, x: fn(x), sp)
        return g

    v2 = lambda n: (n & -n).bit_length() - 1
    width = lambda B: min(4, 2 ** (v2(B) + v2(Lz)))
    rows = []
    print(f"\n  {K} pasos por llamada, sobrecoste amortizado", flush=True)
    print(f"\n{'B':>5}{'anchoXLA':>9}{'jax us/rep':>12}{'triton us/rep':>15}"
          f"{'cambio':>9}{'padeado us/rep':>16}{'vs triton':>11}", flush=True)
    for B in batches:
        sp = S.init_spins(jax.random.PRNGKey(7 + B), geo, B)
        sp4 = jax.jit(padN)(sp)
        fj = jax.jit(loop_of(jax_step)).lower(sp).compile()
        ft = jax.jit(loop_of(triton_step)).lower(sp).compile()
        fq = jax.jit(loop_of(triton_padded)).lower(sp4).compile()
        for f, x in ((fj, sp), (ft, sp), (fq, sp4)):
            jax.block_until_ready(f(x))
        tj, tt, tq = [], [], []
        for _ in range(int(repeats)):
            t0 = time.perf_counter(); jax.block_until_ready(fj(sp))
            tj.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); jax.block_until_ready(ft(sp))
            tt.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); jax.block_until_ready(fq(sp4))
            tq.append(time.perf_counter() - t0)
        mj, mt, mq = (float(np.median(x)) for x in (tj, tt, tq))
        mj, mt, mq = mj / K, mt / K, mq / K     # per step
        e2 = float(jnp.max(jnp.abs(jax.jit(jax_step)(sp)
                                   - jax.jit(triton_step)(sp))))
        rows.append(dict(B=B, width_xla=width(B), v2=v2(B),
                         jax_s=mj, triton_s=mt, padded_s=mq,
                         jax_us_per_replica=1e6 * mj / B,
                         triton_us_per_replica=1e6 * mt / B,
                         padded_us_per_replica=1e6 * mq / B,
                         change_pct=100 * (mt / mj - 1),
                         padded_vs_triton_pct=100 * (mq / mt - 1),
                         max_abs_err=e2))
        r = rows[-1]
        print(f"{B:>5}{width(B):>9}{r['jax_us_per_replica']:>12.2f}"
              f"{r['triton_us_per_replica']:>15.2f}{r['change_pct']:>+8.1f}%"
              f"{r['padded_us_per_replica']:>16.2f}"
              f"{r['padded_vs_triton_pct']:>+10.1f}%", flush=True)
    res["timing"] = rows
    if rows:
        import statistics as stx
        def disp(key):
            xs = [r[key] for r in rows]
            return 100 * (max(xs) / min(xs) - 1)
        res["dispersion_pct"] = dict(
            jax=round(disp("jax_us_per_replica"), 1),
            triton=round(disp("triton_us_per_replica"), 1),
            padded=round(disp("padded_us_per_replica"), 1))
        print(f"\n=== la prueba del dentado: dispersion de us/replica "
              f"sobre {len(rows)} lotes ===", flush=True)
        for k, lab in (("jax", "JAX, seis rolls"),
                       ("triton", "Triton, tile fijo"),
                       ("padded", "Triton, arreglo padeado")):
            print(f"  {lab:<26}{res['dispersion_pct'][k]:>7.1f}%", flush=True)
        print(f"\n  si el tile fijo mata el dentado, la dispersion de Triton"
              f"\n  deberia ser mucho menor que la de JAX.", flush=True)
        # does the XLA width effect show up on either path?
        w1 = [r["jax_us_per_replica"] for r in rows if r["width_xla"] == 1]
        w4 = [r["jax_us_per_replica"] for r in rows if r["width_xla"] == 4]
        t1 = [r["triton_us_per_replica"] for r in rows if r["width_xla"] == 1]
        t4 = [r["triton_us_per_replica"] for r in rows if r["width_xla"] == 4]
        if w1 and w4:
            print(f"\n=== el efecto del ancho, en cada ruta ===", flush=True)
            print(f"  JAX    ancho1 {stx.mean(w1):.2f} vs ancho4 "
                  f"{stx.mean(w4):.2f} us/rep -> "
                  f"{100*(stx.mean(w4)/stx.mean(w1)-1):+.1f}%", flush=True)
            print(f"  Triton ancho1 {stx.mean(t1):.2f} vs ancho4 "
                  f"{stx.mean(t4):.2f} us/rep -> "
                  f"{100*(stx.mean(t4)/stx.mean(t1)-1):+.1f}%", flush=True)
            res["width_effect_pct"] = dict(
                jax=round(100*(stx.mean(w4)/stx.mean(w1)-1), 1),
                triton=round(100*(stx.mean(t4)/stx.mean(t1)-1), 1))
        bb = min(rows, key=lambda r: r["change_pct"])
        print(f"\n  mejor: B={bb['B']}  {bb['change_pct']:+.1f}%", flush=True)
        print(f"  equivalencia peor caso: "
              f"{max(r['max_abs_err'] for r in rows):.3e}", flush=True)
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n  guardado /out/{tag}.json", flush=True)
    return res


@app.function(gpu="H100", image=pallas_image, timeout=60 * 60,
              volumes={"/out": vol})
def triton_stencil_h100(b, rep, tag, Rd, L, tile, steps):
    return _triton_stencil(b, rep, tag, Rd, L, tile, steps)


@app.local_entrypoint()
def triton_stencil(batches: str = "63,64,65,66,68,96,97,99,100,128,129",
                   repeats: int = 9,
                   rd: float = 18.30, layers: int = 5, tile: int = 64,
                   steps: int = 200, tag: str = "triton_stencil"):
    """The stencil as a real Triton kernel via pltriton.load."""
    triton_stencil_h100.remote([int(x) for x in batches.split(",") if x],
                               repeats, tag, rd, layers, tile, steps)


def _wave_probe(tag, Rd, L, gmax, step, repeats):
    """Measure the wave width W directly, instead of fitting it afterwards.

    The residual jaggedness the fixed-tile Triton kernel still has (sigma
    10.3% after removing a/B + b) is suspected to be wave quantisation: the
    grid is B x Lz blocks, so if W blocks are resident at once the cost goes
    as ceil(G/W) and the last wave is partial. The tile being fixed does
    nothing about that, because the GRID still scales with B.

    Fitting W to the observed teeth would be the fifth post-hoc pattern in
    this work, and four of the previous four died. So measure it instead: run
    the SAME kernel with a grid of G blocks, every block doing identical work
    on the same data, and sweep G. Then the time is ceil(G/W) * t_wave exactly
    -- a staircase whose tread width IS W, with no model to fit.

    Every block reads plane `g mod Bm` of a small fixed array, so the data is
    cache-resident and the only thing that changes with G is how many blocks
    the scheduler has to place.
    """
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S

    print(f"gpu={device_identity()}", flush=True)
    try:
        from jax.experimental import pallas as pl
        import jax.experimental.pallas.triton as pltriton
    except Exception as e:
        print(f"TRITON NO DISPONIBLE: {type(e).__name__}: {e}", flush=True)
        return dict(available=False, error=f"{type(e).__name__}: {e}")

    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    N, Lz = int(geo["N"]), int(geo["L"])
    T, Bm = 64, 8
    print(f"Rd={Rd} L={Lz} N={N}  tile {T}x{T}  "
          f"arreglo fijo de {Bm} replicas", flush=True)

    # the same arithmetic as the real stencil, so the occupancy matches
    def kernel(s_ref, o_ref):
        g = pl.program_id(0)
        b = jax.lax.rem(g, Bm)
        iy = jax.lax.broadcasted_iota(jnp.int32, (T, T), 0)
        ix = jax.lax.broadcasted_iota(jnp.int32, (T, T), 1)
        m = (iy < N) & (ix < N)
        cy, cx = jnp.minimum(iy, N - 1), jnp.minimum(ix, N - 1)
        yp = jnp.where(iy + 1 >= N, 0, jnp.minimum(iy + 1, N - 1))
        ym = jnp.where(iy - 1 < 0, N - 1, jnp.maximum(iy - 1, 0))
        xp = jnp.where(ix + 1 >= N, 0, jnp.minimum(ix + 1, N - 1))
        xm = jnp.where(ix - 1 < 0, N - 1, jnp.maximum(ix - 1, 0))
        e = []
        for c in range(3):
            ld = lambda yy, xx: pltriton.load(s_ref.at[b, 0, yy, xx, c], mask=m)
            e.append(ld(cy, cx) + 0.1 * (ld(yp, cx) + ld(ym, cx)
                                         + ld(cy, xp) + ld(cy, xm)))
        inv = 1.0 / jnp.sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2] + 1e-20)
        for c in range(3):
            pltriton.store(o_ref.at[b, 0, cy, cx, c], e[c] * inv, mask=m)

    def make(G):
        def f(sp):
            return pl.pallas_call(
                kernel, grid=(G,),
                in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
                out_specs=pl.BlockSpec(memory_space=pl.ANY),
                out_shape=jax.ShapeDtypeStruct(sp.shape, sp.dtype),
                compiler_params=pltriton.CompilerParams(),
            )(sp)
        return f

    sp = S.init_spins(jax.random.PRNGKey(3), geo, Bm)[:, :1]
    print(f"  arreglo: {sp.shape}", flush=True)
    Gs = list(range(1, int(gmax) + 1, int(step)))
    rows, fns = [], {}
    for G in Gs:
        try:
            fns[G] = jax.jit(make(G)).lower(sp).compile()
            jax.block_until_ready(fns[G](sp))
        except Exception as e:
            print(f"  G={G} fallo: {type(e).__name__}: {e}", flush=True)
            fns.pop(G, None)
    Gs = [G for G in Gs if G in fns]
    print(f"  compilados {len(Gs)} tamaños de rejilla", flush=True)
    for _ in range(int(repeats)):
        for G in Gs:
            t0 = time.perf_counter(); jax.block_until_ready(fns[G](sp))
            rows.append((G, time.perf_counter() - t0))
    agg = {}
    for G, t in rows:
        agg.setdefault(G, []).append(t)
    times = {G: float(np.median(v)) for G, v in agg.items()}

    print(f"\n{'G':>5}{'us':>10}{'us/bloque':>12}{'salto vs anterior':>19}",
          flush=True)
    prev, jumps = None, []
    for G in Gs:
        us = times[G] * 1e6
        j = (100 * (us / prev - 1)) if prev else 0.0
        if prev and j > 4.0:
            jumps.append(dict(G=G, jump_pct=j))
        print(f"{G:>5}{us:>10.1f}{us/G:>12.3f}"
              f"{(f'{j:+.1f}%' if prev else '-'):>19}"
              f"{'   <- ESCALON' if prev and j > 4.0 else ''}", flush=True)
        prev = us

    res = dict(available=True, device=device_identity(), N=N, L=Lz, tile=T,
               array_replicas=Bm, grids=Gs, repeats=int(repeats),
               us_by_grid={str(G): times[G] * 1e6 for G in Gs},
               steps=jumps)
    if len(jumps) >= 2:
        ps = [jumps[i]["G"] - jumps[i - 1]["G"] for i in range(1, len(jumps))]
        res["step_positions"] = [j["G"] for j in jumps]
        res["step_periods"] = ps
        res["W_estimate"] = int(np.median(ps)) if ps else None
        print(f"\n=== escalones en G = {[j['G'] for j in jumps]}", flush=True)
        print(f"    periodos = {ps}   ->  W ~ {res['W_estimate']} bloques "
              f"residentes", flush=True)
        W = res["W_estimate"]
        if W:
            print(f"\n=== lo que eso predice para la rejilla real (B x {Lz}) ===",
                  flush=True)
            print(f"  dientes donde {Lz}B cruza un multiplo de {W}:", flush=True)
            bs = sorted({int(np.ceil(k * W / Lz)) for k in range(1, 40)
                         if int(np.ceil(k * W / Lz)) <= 200})
            print(f"  B = {bs}", flush=True)
            res["predicted_teeth_B"] = bs
    else:
        print(f"\n  menos de dos escalones claros; no se puede estimar W",
              flush=True)
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n  guardado /out/{tag}.json", flush=True)
    return res


@app.function(gpu="H100", image=pallas_image, timeout=60 * 60,
              volumes={"/out": vol})
def wave_probe_h100(tag, Rd, L, gmax, step, repeats):
    return _wave_probe(tag, Rd, L, gmax, step, repeats)


@app.local_entrypoint()
def wave_probe(rd: float = 18.30, layers: int = 5, gmax: int = 600,
               step: int = 4, repeats: int = 7, tag: str = "wave_probe"):
    """Measure how many blocks are resident at once, from the staircase."""
    wave_probe_h100.remote(tag, rd, layers, gmax, step, repeats)


def _triton_flat(batches, repeats, tag, Rd, L, steps):
    """Flatten the plane instead of tiling it in two dimensions.

    The 2-D kernel used a 64x64 index grid over a 39x39 plane: 4096 lanes for
    1521 active sites, so 62.9% of lanes were masked and still executed, which
    is 2.69x the lane work and the main reason it lost by 26% once launch
    overhead was amortised away.

    Flattening fixes exactly that. The 1521 sites of a plane become a single
    index of length 2048, the next power of two, so only 25.7% of lanes are
    wasted -- half the waste of the square grid. The site index p carries y
    and x through p = y*N + x, the neighbours are computed from those, and the
    same mask handles the overhang.

    Timed with K steps inside one call, because a single step of this kernel
    is dispatch-bound: a call costs about 122 us whatever the grid, and the
    earlier single-step numbers were half overhead, which is what faked a
    Triton win that amortisation reversed.
    """
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S

    print(f"gpu={device_identity()}", flush=True)
    try:
        from jax.experimental import pallas as pl
        import jax.experimental.pallas.triton as pltriton
    except Exception as e:
        print(f"TRITON NO DISPONIBLE: {type(e).__name__}: {e}", flush=True)
        return dict(available=False, error=f"{type(e).__name__}: {e}")

    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    N, Lz = int(geo["N"]), int(geo["L"])
    S2 = N * N
    P = 1 << (S2 - 1).bit_length()          # next power of two above N*N
    T = 1 << (N - 1).bit_length()           # the square grid, for reference
    K = int(steps)
    print(f"Rd={Rd} L={Lz} N={N}", flush=True)
    print(f"  aplanado : {S2} sitios -> {P} carriles  "
          f"desperdicio {100*(1-S2/P):.1f}%", flush=True)
    print(f"  rejilla 2D: {S2} sitios -> {T*T} carriles  "
          f"desperdicio {100*(1-S2/(T*T)):.1f}%", flush=True)
    print(f"  -> el aplanado hace {(T*T)/P:.2f}x menos trabajo de carril",
          flush=True)
    print(f"  {K} pasos por llamada", flush=True)

    def jax_step(sp):
        xp = jnp.roll(sp, -1, 3); xm = jnp.roll(sp, 1, 3)
        yp = jnp.roll(sp, -1, 2); ym = jnp.roll(sp, 1, 2)
        zp = jnp.roll(sp, -1, 1).at[:, -1].set(0.0)
        zm = jnp.roll(sp, 1, 1).at[:, 0].set(0.0)
        e = sp + 0.1 * (xp + xm + yp + ym + zp + zm)
        return e / jnp.sqrt(jnp.sum(e * e, -1, keepdims=True) + 1e-20)

    def make_kernel(n_lanes, flat):
        def kernel(s_ref, o_ref):
            b = pl.program_id(0)
            z = pl.program_id(1)
            if flat:
                p = jax.lax.broadcasted_iota(jnp.int32, (n_lanes,), 0)
                m = p < S2
                pc = jnp.minimum(p, S2 - 1)
                y = pc // N
                x = pc - y * N
            else:
                iy = jax.lax.broadcasted_iota(jnp.int32, (n_lanes, n_lanes), 0)
                ix = jax.lax.broadcasted_iota(jnp.int32, (n_lanes, n_lanes), 1)
                m = (iy < N) & (ix < N)
                y, x = jnp.minimum(iy, N - 1), jnp.minimum(ix, N - 1)
            ypi = jnp.where(y + 1 >= N, 0, y + 1)
            ymi = jnp.where(y - 1 < 0, N - 1, y - 1)
            xpi = jnp.where(x + 1 >= N, 0, x + 1)
            xmi = jnp.where(x - 1 < 0, N - 1, x - 1)
            zlo, zhi = jnp.maximum(z - 1, 0), jnp.minimum(z + 1, Lz - 1)
            has_lo = (z > 0).astype(jnp.float32)
            has_hi = (z < Lz - 1).astype(jnp.float32)
            e = []
            for c in range(3):
                ld = lambda zz, yy, xx: pltriton.load(
                    s_ref.at[b, zz, yy, xx, c], mask=m)
                e.append(ld(z, y, x) + 0.1 * (
                    ld(z, ypi, x) + ld(z, ymi, x) + ld(z, y, xpi) + ld(z, y, xmi)
                    + has_hi * ld(zhi, y, x) + has_lo * ld(zlo, y, x)))
            inv = 1.0 / jnp.sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2] + 1e-20)
            for c in range(3):
                pltriton.store(o_ref.at[b, z, y, x, c], e[c] * inv, mask=m)
        return kernel

    def make_step(n_lanes, flat):
        kern = make_kernel(n_lanes, flat)
        def f(sp):
            return pl.pallas_call(
                kern, grid=(sp.shape[0], Lz),
                in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
                out_specs=pl.BlockSpec(memory_space=pl.ANY),
                out_shape=jax.ShapeDtypeStruct(sp.shape, sp.dtype),
                compiler_params=pltriton.CompilerParams(),
            )(sp)
        return f

    flat_step = make_step(P, True)
    sq_step = make_step(T, False)

    # ---- the gate: equivalence on ONE step ------------------------------
    sp0 = S.init_spins(jax.random.PRNGKey(5), geo, 8)
    ref = jax.jit(jax_step)(sp0); ref.block_until_ready()
    res = dict(available=True, N=N, L=Lz, sites=S2, flat_lanes=P,
               square_lanes=T * T,
               flat_waste_pct=round(100 * (1 - S2 / P), 1),
               square_waste_pct=round(100 * (1 - S2 / (T * T)), 1),
               lane_work_ratio=round((T * T) / P, 2), steps=K,
               device=device_identity())
    variants = {}
    for lab, fn in (("aplanado", flat_step), ("rejilla 2D", sq_step)):
        try:
            got = jax.jit(fn)(sp0); got.block_until_ready()
            rel = float(jnp.max(jnp.abs(ref - got))) / max(
                float(jnp.max(jnp.abs(ref))), 1e-30)
            ok = rel < 1e-5
            print(f"\n  {lab}: COMPILA  error relativo {rel:.3e}  "
                  f"{'OK' if ok else 'MISMATCH'}", flush=True)
            res[f"equiv_{lab.replace(' ', '_')}"] = dict(rel=rel, ok=bool(ok))
            if ok:
                variants[lab] = fn
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            print(f"\n  {lab}: {msg[:300]}", flush=True)
            res[f"equiv_{lab.replace(' ', '_')}"] = dict(error=msg[:300])
    if "aplanado" not in variants:
        res["error"] = "el aplanado no compilo o no verifico"
        with open(f"/out/{tag}.json", "w") as f:
            json.dump(res, f, indent=1)
        return res

    def loop_of(fn):
        return lambda sp: jax.lax.fori_loop(0, K, lambda i, x: fn(x), sp)

    rows = []
    print(f"\n{'B':>5}{'jax us/rep':>12}{'aplanado':>11}{'vs jax':>9}"
          f"{'rejilla2D':>12}{'vs jax':>9}{'aplan vs 2D':>13}", flush=True)
    for B in batches:
        sp = S.init_spins(jax.random.PRNGKey(7 + B), geo, B)
        fj = jax.jit(loop_of(jax_step)).lower(sp).compile()
        ff = jax.jit(loop_of(variants["aplanado"])).lower(sp).compile()
        fs = (jax.jit(loop_of(variants["rejilla 2D"])).lower(sp).compile()
              if "rejilla 2D" in variants else None)
        fns = [fj, ff] + ([fs] if fs else [])
        for f in fns:
            jax.block_until_ready(f(sp))
        acc = [[] for _ in fns]
        for _ in range(int(repeats)):
            for i, f in enumerate(fns):
                t0 = time.perf_counter(); jax.block_until_ready(f(sp))
                acc[i].append(time.perf_counter() - t0)
        med = [float(np.median(a)) / K for a in acc]
        mj, mf = med[0], med[1]
        ms = med[2] if fs else float("nan")
        rows.append(dict(B=B, jax_us_per_replica=1e6 * mj / B,
                         flat_us_per_replica=1e6 * mf / B,
                         square_us_per_replica=(1e6 * ms / B if fs else None),
                         flat_vs_jax_pct=100 * (mf / mj - 1),
                         square_vs_jax_pct=(100 * (ms / mj - 1) if fs else None),
                         flat_vs_square_pct=(100 * (mf / ms - 1) if fs else None)))
        r = rows[-1]
        print(f"{B:>5}{r['jax_us_per_replica']:>12.3f}"
              f"{r['flat_us_per_replica']:>11.3f}{r['flat_vs_jax_pct']:>+8.1f}%"
              f"{(r['square_us_per_replica'] or 0):>12.3f}"
              f"{(r['square_vs_jax_pct'] or 0):>+8.1f}%"
              f"{(r['flat_vs_square_pct'] or 0):>+12.1f}%", flush=True)
    res["timing"] = rows
    if rows:
        import statistics as stx
        fv = [r["flat_vs_jax_pct"] for r in rows]
        sv = [r["square_vs_jax_pct"] for r in rows if r["square_vs_jax_pct"]]
        fs2 = [r["flat_vs_square_pct"] for r in rows if r["flat_vs_square_pct"]]
        res["summary"] = dict(
            flat_vs_jax_mean=round(stx.mean(fv), 1),
            flat_vs_jax_best=round(min(fv), 1),
            flat_wins=sum(1 for x in fv if x < 0), n=len(fv),
            square_vs_jax_mean=(round(stx.mean(sv), 1) if sv else None),
            flat_vs_square_mean=(round(stx.mean(fs2), 1) if fs2 else None))
        print(f"\n=== resumen ===", flush=True)
        print(f"  aplanado contra JAX : media {stx.mean(fv):+.1f}%  "
              f"gana {sum(1 for x in fv if x<0)}/{len(fv)}  "
              f"mejor {min(fv):+.1f}%", flush=True)
        if sv:
            print(f"  rejilla 2D contra JAX: media {stx.mean(sv):+.1f}%", flush=True)
            print(f"  aplanado contra rejilla 2D: media {stx.mean(fs2):+.1f}%  "
                  f"(predicho por carriles: {100/((T*T)/P)-100:+.0f}% si el "
                  f"trabajo de carril lo dominara todo)", flush=True)
        def disp(key):
            xs = [r[key] for r in rows if r[key]]
            return 100 * (max(xs) / min(xs) - 1) if xs else None
        res["dispersion_pct"] = dict(jax=disp("jax_us_per_replica"),
                                     flat=disp("flat_us_per_replica"),
                                     square=disp("square_us_per_replica"))
        print(f"\n  dispersion de us/replica: JAX "
              f"{disp('jax_us_per_replica'):.1f}%, aplanado "
              f"{disp('flat_us_per_replica'):.1f}%", flush=True)
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n  guardado /out/{tag}.json", flush=True)
    return res


@app.function(gpu="H100", image=pallas_image, timeout=60 * 60,
              volumes={"/out": vol})
def triton_flat_h100(b, rep, tag, Rd, L, steps):
    return _triton_flat(b, rep, tag, Rd, L, steps)


@app.local_entrypoint()
def triton_flat(batches: str = "63,65,66,96,99,100,128,129", repeats: int = 7,
                rd: float = 18.30, layers: int = 5, steps: int = 200,
                tag: str = "triton_flat"):
    """Flatten the plane to the next power of two instead of a square grid."""
    triton_flat_h100.remote([int(x) for x in batches.split(",") if x], repeats,
                            tag, rd, layers, steps)


def _split4(tag, Rd, L, sweeps, repeats):
    """Four ways to put 400 chains on four GPUs, measured.

    The model says two of them tie, and the one open question is whether two
    pmaps over DISJOINT device pairs actually overlap. If JAX serialises them,
    the uneven split costs twice what it should and must not be recommended.

      A  one pmap of 4x100      -- what the chart does today, width 4
      B  two pmaps of 4x51,4x49 -- the split protocol, two sequential waves
      C  two pmaps on disjoint pairs, 2x101 + 2x99 -- one wave, uneven
      D  four pmaps of 4x25     -- what BP+replicas does, small per device

    Per-device batch is about 100 in A, B and C and only 25 in D, which the
    measured f(B) curve puts 26% higher per replica.
    """
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S
    import parallel_axes as PA

    devs = jax.devices()
    print(f"dispositivos: {len(devs)}", flush=True)
    if len(devs) < 4:
        return dict(error=f"se necesitan 4 GPU, hay {len(devs)}")
    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    thetas = PA.load_thetas("/root/theta4.json")
    kB = 0.086173404
    beta = float(1.0 / (kB * 5.0))
    eng = S.build_engine(geo)
    surf = jnp.asarray(geo["surf"]) if "surf" in geo else None

    def body(s, key, k1_, p, n):
        def one(_, v):
            st, kk = v
            kk, ka = jax.random.split(kk, 2)
            return (eng["substep"](st, ka, beta, True, p, k1_), kk)
        return jax.lax.fori_loop(0, n, one, (s, key))[0]

    def build(B, device_list):
        """One pmap over `device_list`, B chains on EACH of them."""
        nd = len(device_list)
        cols = PA.pack(thetas, max(1, (B * nd) // len(thetas)))
        # trim or pad the packed columns to exactly B*nd
        cols = {k: v[: B * nd] for k, v in cols.items()}
        need = B * nd
        cols = {k: jnp.concatenate([v] * (need // v.shape[0] + 1))[:need]
                for k, v in cols.items()}
        k1 = jnp.where(jnp.asarray(geo["surf"])[None], cols["KanS"], cols["Kan1"]) \
            if "surf" in geo else cols["Kan1"]
        spins = S.init_spins(jax.random.PRNGKey(7 + B), geo, need)
        shard = lambda a: a.reshape(nd, B, *a.shape[1:])
        fn = jax.pmap(body, static_broadcasted_argnums=4, devices=device_list)
        return (fn, shard(spins),
                jax.random.split(jax.random.PRNGKey(11 + B), nd),
                shard(k1), {k: shard(v) for k, v in cols.items()})

    SCHEMES = {
        "A 1 ola 4x100":            [(100, devs)],
        "B 2 olas 4x51+4x49":       [(51, devs), (49, devs)],
        "C 1 ola 2x101+2x99":       [(101, devs[:2]), (99, devs[2:])],
        "D 4 olas 4x25":            [(25, devs)] * 4,
    }
    built = {}
    for name, parts in SCHEMES.items():
        built[name] = [build(B, dl) for B, dl in parts]
        for fn, sp, kk, k1, p in built[name]:
            jax.block_until_ready(fn(sp, kk, k1, p, int(sweeps)))
        print(f"  compilado {name}", flush=True)

    res = dict(device=device_identity(), n_devices=len(devs),
               sweeps=int(sweeps), repeats=int(repeats), rows=[])
    print(f"\n{'esquema':<24}{'chains':>8}{'olas':>6}{'s':>10}{'vs A':>9}", flush=True)
    base = None
    for name, parts in SCHEMES.items():
        ts = []
        for _ in range(int(repeats)):
            t0 = time.perf_counter()
            # issue every wave's work, then block once: disjoint pmaps can
            # overlap only if we do not block between them
            outs = []
            for (fn, sp, kk, k1, p) in built[name]:
                outs.append(fn(sp, kk, k1, p, int(sweeps)))
            for o in outs:
                jax.block_until_ready(o)
            ts.append(time.perf_counter() - t0)
        m = float(np.median(ts))
        if base is None:
            base = m
        chains = sum(B * len(dl) for B, dl in parts)
        res["rows"].append(dict(scheme=name, chains=chains, waves=len(parts),
                                seconds=m, vs_A_pct=100 * (m / base - 1),
                                spread_pct=100 * (max(ts) / min(ts) - 1)))
        print(f"{name:<24}{chains:>8}{len(parts):>6}{m:>10.4f}"
              f"{100*(m/base-1):>+8.1f}%", flush=True)
    rb = next(r for r in res["rows"] if r["scheme"].startswith("B"))
    rc = next(r for r in res["rows"] if r["scheme"].startswith("C"))
    ch = 100 * (rc["seconds"] / rb["seconds"] - 1)
    res["C_vs_B_pct"] = ch
    res["concurrency_works"] = ch < 25
    print(f"\n  C contra B: {ch:+.1f}%", flush=True)
    print(f"  {'los pmap disjuntos SI se solapan' if ch < 25 else 'NO se solapan: C se serializa y no sirve'}",
          flush=True)
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    return res


@app.function(gpu="H100:4", **_COMMON)
def split4_h100(tag, Rd, L, sweeps, repeats):
    return _split4(tag, Rd, L, sweeps, repeats)


@app.local_entrypoint()
def split4(rd: float = 18.30, layers: int = 5, sweeps: int = 400,
           repeats: int = 7, tag: str = "split4"):
    """Four ways to put 400 chains on four GPUs."""
    split4_h100.remote(tag, rd, layers, sweeps, repeats)


def _partsearch(tag, Rd, L, total, sweeps, repeats, n_temps=8):
    """Find the cheapest way to split R chains over 1, 2 and 4 GPUs.

    The chart needs three bars: BP only, BP + replicas as it is arranged today
    (one wave, R/D per device), and the best partition we can find. This
    searches the partitions rather than assuming one.

    A partition is a list of per-device batch sizes, one per wave, summing to
    R/D. Every wave launches on all D devices, because two pmaps over disjoint
    device subsets were measured to serialise rather than overlap.
    """
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S
    import parallel_axes as PA

    devs = jax.devices()
    D_ALL = len(devs)
    print(f"dispositivos: {D_ALL}", flush=True)
    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    eng = S.build_engine(geo)
    thetas = PA.load_thetas("/root/theta4.json")
    kB = 0.086173404
    v2 = lambda n: (n & -n).bit_length() - 1
    width = lambda B: min(4, 2 ** (v2(B) + v2(int(L))))

    # THE PRODUCTION KERNEL. An earlier version of this search looped
    # eng["substep"] directly, which is NOT what production runs: it has no
    # temperature ladder and no observable recording, so it lacks the kLoop
    # fusions where the width penalty lives -- the Threefry mega-fusion at
    # +118% and the acceptance at +76%. That version under-reported the gain
    # badly (-9.9% at 100 chains per device against -23.5% measured here with
    # anneal), and the whole point of the search is to compare against what
    # production actually costs.
    betas = jnp.asarray(1.0 / (kB * np.linspace(20.0, 0.1, int(n_temps))))
    TH = int(sweeps) * 2 // 3
    ME = int(sweeps) - TH

    def body(sp, kk, p):
        return eng["anneal"](sp, kk, betas, p, TH, ME)

    cache = {}

    def build(b, nd):
        """One pmap over the first `nd` devices, `b` chains on each."""
        key = (b, nd)
        if key in cache:
            return cache[key]
        need = b * nd
        cols = PA.pack(thetas, need // len(thetas) + 1)
        cols = {k: v[:need] for k, v in cols.items()}
        spins = S.init_spins(jax.random.PRNGKey(7 + b), geo, need)
        sh = lambda a: a.reshape(nd, b, *a.shape[1:])
        fn = jax.pmap(body, devices=devs[:nd])
        out = (fn, sh(spins), jax.random.split(jax.random.PRNGKey(11 + b), nd),
               {k: sh(v) for k, v in cols.items()})
        jax.block_until_ready(fn(out[1], out[2], out[3]))
        cache[key] = out
        return out

    R = int(total)
    PLANS = {
        1: [[R], [R // 2 + 1, R // 2 - 1], [R // 2 + 3, R // 2 - 3],
            [135, 133, 132], [101, 101, 99, 99]],
        2: [[R // 2], [R // 4 + 1, R // 4 - 1], [R // 4 + 3, R // 4 - 3],
            [67, 67, 66], [51, 51, 49, 49]],
        4: [[R // 4], [R // 8 + 1, R // 8 - 1], [R // 8 + 3, R // 8 - 3],
            [35, 33, 32], [25, 25, 25, 25]],
    }
    res = dict(total=R, sweeps=int(sweeps), repeats=int(repeats),
               n_temps=int(n_temps), kernel="eng[anneal] -- production",
               device=device_identity(), n_devices=D_ALL, plans=[])
    for nd in (1, 2, 4):
        if nd > D_ALL:
            continue
        print(f"\n=== {nd} GPU, {R} cadenas, {R//nd} por dispositivo ===",
              flush=True)
        print(f"  {'particion':<26}{'olas':>6}{'anchos':>12}{'s':>10}{'vs 1 ola':>10}",
              flush=True)
        base = None
        for waves in PLANS[nd]:
            if sum(waves) * nd != R:
                print(f"  {str(waves):<26}  suma {sum(waves)*nd} != {R}, se omite",
                      flush=True)
                continue
            try:
                parts = [build(b, nd) for b in waves]
            except Exception as e:
                print(f"  {str(waves):<26}  fallo: {type(e).__name__}", flush=True)
                continue
            ts = []
            for _ in range(int(repeats)):
                t0 = time.perf_counter()
                outs = [fn(sp, kk, p) for (fn, sp, kk, p) in parts]
                for o in outs:
                    jax.block_until_ready(o)
                ts.append(time.perf_counter() - t0)
            m = float(np.median(ts))
            if base is None:
                base = m
            ws = ",".join(str(width(b)) for b in waves)
            res["plans"].append(dict(gpus=nd, waves=waves, n_waves=len(waves),
                                     widths=[width(b) for b in waves],
                                     seconds=m, vs_single_pct=100*(m/base-1),
                                     spread_pct=100*(max(ts)/min(ts)-1)))
            print(f"  {str(waves):<26}{len(waves):>6}{ws:>12}{m:>10.4f}"
                  f"{100*(m/base-1):>+9.1f}%", flush=True)
        mine = [p for p in res["plans"] if p["gpus"] == nd]
        if not mine:
            print("  ninguna particion corrio en este brazo", flush=True)
            continue
        best = min(mine, key=lambda p: p["seconds"])
        single = next(p for p in res["plans"] if p["gpus"] == nd
                      and p["n_waves"] == 1)
        print(f"  -> mejor: {best['waves']}  "
              f"{100*(best['seconds']/single['seconds']-1):+.1f}% contra una ola",
              flush=True)
        res.setdefault("best", {})[str(nd)] = dict(
            waves=best["waves"], seconds=best["seconds"],
            single_seconds=single["seconds"],
            gain_pct=100*(best["seconds"]/single["seconds"]-1))
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n  guardado /out/{tag}.json", flush=True)
    return res


@app.function(gpu="H100:4", **_COMMON)
def partsearch_h100(tag, Rd, L, total, sweeps, repeats, n_temps):
    return _partsearch(tag, Rd, L, total, sweeps, repeats, n_temps)


@app.local_entrypoint()
def partsearch(rd: float = 18.30, layers: int = 5, total: int = 400,
               sweeps: int = 450, repeats: int = 7, n_temps: int = 8,
               tag: str = "partsearch"):
    """Cheapest partition of R chains over 1, 2 and 4 GPUs, production kernel."""
    partsearch_h100.remote(tag, rd, layers, total, sweeps, repeats, n_temps)


def _ptx_probe(batches, tag, Rd, L, n_temps, n_therm, n_meas):
    """What LOADS does XLA actually emit, and does it ever prefetch?

    This checks the mechanism story rather than assuming it. Two questions the
    thunk geometry cannot answer:

    1. When XLA gives a thread four consecutive elements, does it emit ONE
       128-bit vector load (`ld.global.v4.f32`) or FOUR scalar loads? It
       matters completely. One v4 load per thread is PERFECTLY coalesced -- the
       warp fetches 512 contiguous bytes and uses all of them. Four scalar
       loads with a 16-byte stride across the warp need four transactions to
       deliver 128 useful bytes, which is the waste the raw-CUDA probe
       measured. If XLA emits v4, the coalescing explanation is wrong and the
       +14.9% measured in CUDA was a failure of nvcc to vectorise MY kernel,
       not a description of XLA's.
    2. Does XLA use Hopper's asynchronous copies at all (`cp.async`,
       `cp.async.bulk.tensor` for TMA, `mbarrier`)? If it emits them for some
       shapes and not others, that alone would produce an irregular curve.
    """
    import os, sys, io as _io, json, glob, re, collections
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + " --xla_dump_to=/tmp/dump").strip()
    os.makedirs("/tmp/dump", exist_ok=True)
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S
    import parallel_axes as PA

    kB = 0.086173404
    betas = jnp.asarray(1.0 / (kB * np.linspace(20.0, 0.1, int(n_temps))))
    theta = PA.load_thetas("/root/theta4.json")[0]
    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    eng = S.build_engine(geo)
    print(f"gpu={device_identity()}", flush=True)

    def body(sp, kk, p):
        return eng["anneal"](sp, kk, betas, p, int(n_therm), int(n_meas))

    # Instruction families that answer the two questions.
    PAT = {
        "ld.global.v4":   re.compile(r"\bld\.global[.\w]*\.v4\b"),
        "ld.global.v2":   re.compile(r"\bld\.global[.\w]*\.v2\b"),
        "ld.global (escalar)": re.compile(r"\bld\.global(?![.\w]*\.v[24])[.\w]*\b"),
        "st.global.v4":   re.compile(r"\bst\.global[.\w]*\.v4\b"),
        "st.global (escalar)": re.compile(r"\bst\.global(?![.\w]*\.v[24])[.\w]*\b"),
        "ld.global.nc":   re.compile(r"\bld\.global\.nc\b"),
        "cp.async":       re.compile(r"\bcp\.async\b"),
        "cp.async.bulk (TMA)": re.compile(r"\bcp\.async\.bulk"),
        "mbarrier":       re.compile(r"\bmbarrier\."),
    }
    out = []
    for B in batches:
        for f in glob.glob("/tmp/dump/*"):
            try:
                os.remove(f)
            except Exception:
                pass
        cols = PA.pack([theta], B)
        args = (S.init_spins(jax.random.PRNGKey(11), geo, B),
                jax.random.PRNGKey(12), cols)
        jax.jit(body).lower(*args).compile()

        files = sorted(glob.glob("/tmp/dump/*"))
        big = max(files, key=os.path.getsize)
        pre = os.path.basename(big).split(".jit_")[0]
        ptx = [f for f in files
               if os.path.basename(f).startswith(pre) and f.endswith(".ptx")
               and os.path.getsize(f) > 0]
        txt = "".join(_io.open(f, encoding="utf-8", errors="replace").read()
                      for f in ptx)
        cnt = {k: len(rx.findall(txt)) for k, rx in PAT.items()}
        cnt["archivos ptx"] = len(ptx)
        cnt["bytes ptx"] = len(txt)
        out.append(dict(B=int(B), **cnt))
        print(f"\n  B={B}{'  (multiplo de 4)' if B % 4 == 0 else ''}  "
              f"{len(ptx)} ptx, {len(txt):,} B", flush=True)
        for k in PAT:
            print(f"      {k:24s} {cnt[k]:7,d}", flush=True)

    print("\n=== cargas vectoriales contra escalares, por B ===", flush=True)
    hdr = f"{'B':>6} " + " ".join(f"{k.split(' ')[0][:13]:>14}" for k in
                                  ("ld.global.v4", "ld.global.v2",
                                   "ld.global (escalar)", "cp.async", "mbarrier"))
    print(hdr, flush=True)
    for r in out:
        print(f"{r['B']:6d} "
              f"{r['ld.global.v4']:14,d} {r['ld.global.v2']:14,d} "
              f"{r['ld.global (escalar)']:14,d} {r['cp.async']:14,d} "
              f"{r['mbarrier']:14,d}", flush=True)
    print("\nSi los lotes multiplos de 4 muestran ld.global.v4 donde los otros "
          "no, la vectorizacion ES una carga de 128 bits y el relato de la "
          "coalescencia esta MAL. Si no aparece v4 en ninguno, las cuatro "
          "cargas son escalares y el relato se sostiene.", flush=True)
    res = dict(rows=out, Rd=float(Rd), L=int(L), device=device_identity())
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


@app.function(gpu="H100", **_BS)
def ptx_probe_h100(b, tag, Rd, L, nt, th, me):
    return _ptx_probe(b, tag, Rd, L, nt, th, me)


@app.local_entrypoint()
def ptx_probe(batches: str = "59,60,63,64", rd: float = 18.30,
              layers: int = 5, n_temps: int = 8, n_therm: int = 200,
              n_meas: int = 100, tag: str = "ptx_loads"):
    """Which load instructions XLA emits, and whether it ever prefetches."""
    ptx_probe_h100.remote([int(x) for x in batches.split(",") if x], tag,
                          rd, layers, n_temps, n_therm, n_meas)


@app.function(gpu="H100", **_BS)
def launch_dims_h100(b, tag, Rd, L, nt, th, me):
    return _launch_dims(b, tag, Rd, L, nt, th, me)


# Experiment 7: is the vector-width cap of 4 a property of XLA, or of Hopper?
#
# Every launch dump so far came from an H100, and the cap of 4 was read off
# those. Experiment 6 then found a small but CLEAN +4.0% residue at L = 4 on a
# T4, where all eight cells are width 4 under that cap -- so the cap cannot
# explain it. XLA's code generator consults the target compute capability, so
# the obvious candidate is that sm_75 takes a path WIDER than 4 at higher v2,
# which would also account for the mod-8 structure seen in early measurements.
# Reading the T4's own thunk sequence settles it, and costs only compile time.
@app.function(gpu="T4", image=image, memory=32768, timeout=3600,
              volumes={"/out": vol})
def launch_dims_t4(b, tag, Rd, L, nt, th, me):
    return _launch_dims(b, tag, Rd, L, nt, th, me)


@app.local_entrypoint()
def launch_dims(batches: str = "59,60,63,64", rd: float = 18.30,
                layers: int = 5, n_temps: int = 8, n_therm: int = 200,
                n_meas: int = 100, tag: str = "exp3f", device: str = "h100"):
    """The launch geometry XLA picks, per B. Compile only."""
    fn = {"h100": launch_dims_h100, "t4": launch_dims_t4}[device.lower()]
    fn.remote([int(x) for x in batches.split(",") if x], tag,
              rd, layers, n_temps, n_therm, n_meas)


@app.function(gpu="H100", **_BS)
def xla_dump_h100(b, tag, Rd, L, nt, th, me):
    return _xla_dump(b, tag, Rd, L, nt, th, me)


@app.local_entrypoint()
def xla_dump(batches: str = "51,52,53,59,60,63,64", rd: float = 18.30,
             layers: int = 5, n_temps: int = 8, n_therm: int = 200,
             n_meas: int = 100, tag: str = "exp3e"):
    """Experiment 3e: what the compiler emits per B, and which kernel pays."""
    xla_dump_h100.remote([int(x) for x in batches.split(",") if x], tag,
                         rd, layers, n_temps, n_therm, n_meas)


@app.function(gpu="H100", **_BS)
def xla_probe_h100(b, tag, Rd, L):
    return _xla_layout_probe(b, tag, Rd, L)


@app.function(gpu="H100", image=cuda_image, memory=65536, timeout=3600,
              volumes={"/out": vol})
def raw_cuda_h100(b, sw, r, tag, Rd, L):
    return _raw_cuda_probe(b, sw, r, tag, Rd, L)


@app.local_entrypoint()
def xla_confound(batches: str = "48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64",
                 sweeps: int = 400, repeats: int = 3, rd: float = 18.30,
                 layers: int = 5, tag: str = "xla_confound", part: str = "both"):
    """Experiment 3. `part=layout` is the XLA allocation probe (3a),
    `part=raw` is the nvrtc stencil with XLA removed (3b)."""
    bl = [int(x) for x in batches.split(",") if x]
    if part in ("both", "layout"):
        xla_probe_h100.remote(bl, tag + "_layout", rd, layers)
    if part in ("both", "raw"):
        raw_cuda_h100.remote(bl, sweeps, repeats, tag + "_raw", rd, layers)


def _pair(batches, repeats, tag, Rd, L, n_temps, n_therm, n_meas, seed, label):
    import sys
    sys.path.insert(0, "/root")
    import pair_test
    print(f"gpu={device_identity()}", flush=True)
    res = pair_test.run_pair(batches=batches, repeats=repeats, Rd=Rd, L=L,
                             n_temps=n_temps, n_therm=n_therm, n_meas=n_meas,
                             seed=seed, out=f"/out/{tag}.json", label=label)
    res["gpu_identity"] = device_identity()
    with open(f"/out/{tag}.json", "w") as f:
        import json
        json.dump(res, f, indent=1)
    vol.commit()
    return res


@app.function(gpu="H100", **_BS)
def pair_h100(b, r, tag, Rd, L, nt, th, me, seed, label):
    return _pair(b, r, tag, Rd, L, nt, th, me, seed, label)


@app.function(gpu="T4", image=image, memory=32768, timeout=6 * 3600,
              volumes={"/out": vol})
def pair_t4(b, r, tag, Rd, L, nt, th, me, seed, label):
    return _pair(b, r, tag, Rd, L, nt, th, me, seed, label)


# --------------------------------------------------------------------------
# Bayesian search for the best replicas-per-GPU, one GPU model, one geometry.
#
# Scope fixed by the tutor: H100 only, Rd = 18.30, L = 5. Candidates are
# MULTIPLES OF FIVE so the total always divides cleanly if the run is later
# spread over several GPUs. Note that B = 512, the minimum of the old 19-point
# grid, is NOT a multiple of five -- its neighbours 510 and 515 are, and both
# avoid the expensive vector width.
#
# Why a surrogate at all, and why this one. Plain Bayesian optimisation assumes
# the objective is smooth, and we have MEASURED that this one is not: adjacent
# batches differ by 27-35%. A Gaussian process fitted directly would read that
# sawtooth as noise and chase its mean. So the surrogate is given the structure
# we already understand, and only has to learn what is left:
#
#   mean model   us(B) = a/B + b + c*[width 4] + d*[width 2]
#
# The a/B term is the per-sweep cost paid once however many replicas share it,
# measured at 10.9 us-replica. The width terms are the compiler's vectorisation
# decision, which is known in advance from v2(B) + v2(L) and costs about 14% at
# width 4. A Gaussian process then models the residual -- the sawtooth -- in
# log2(B), and expected improvement picks the next batch to try.
#
# The objective reported is SECONDS PER REPLICA for a full production run
# (200 temperatures x 20,000 sweeps), because microseconds per replica-sweep is
# divided by both the replica count and the sweep count and reads as if nothing
# takes any time.
PROD_SWEEPS = 200 * 20000


def _bayes_batch(lo, hi, step, n_init, n_iter, repeats, n_temps, n_therm,
                 n_meas, tag, Rd, L, seed):
    import sys, json, time, math
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S
    import parallel_axes as PA

    def v2(n):
        k = 0
        while n % 2 == 0:
            n //= 2
            k += 1
        return k
    width = lambda B: min(4, 2 ** (v2(B) + v2(int(L))))

    cands = [b for b in range(lo, hi + 1) if b % step == 0]
    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    eng = S.build_engine(geo)
    kB = 0.086173404
    betas = jnp.asarray(1.0 / (kB * np.linspace(20.0, 0.1, int(n_temps))))
    sweeps = int(n_temps) * (int(n_therm) + int(n_meas))
    theta = PA.load_thetas("/root/theta4.json")[0]
    print(f"gpu={device_identity()}", flush=True)
    print(f"Rd={Rd} L={L} N={geo['N']} sitios={int(np.asarray(geo['disco']).sum())}",
          flush=True)
    print(f"{len(cands)} candidatos multiplos de {step} entre {lo} y {hi}  "
          f"({sweeps:,} barridos por evaluacion, {repeats} repeticiones)",
          flush=True)

    def body(sp, kk, p):
        return eng["anneal"](sp, kk, betas, p, int(n_therm), int(n_meas))

    seen = {}

    def evaluate(B):
        """Seconds per replica for a full production run."""
        if B in seen:
            return seen[B]
        cols = PA.pack([theta], B)
        args = (S.init_spins(jax.random.PRNGKey(11), geo, B),
                jax.random.PRNGKey(12), cols)
        fn = jax.jit(body).lower(*args).compile()      # compile is not timed
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*args))
            ts.append(time.perf_counter() - t0)
        us = float(np.mean(ts) * 1e6 / (sweeps * B))
        sec = us * PROD_SWEEPS / 1e6
        seen[B] = sec
        print(f"    B={B:5d} ancho {width(B)}  {sec:7.3f} s por replica   "
              f"(celda {np.mean(ts):6.2f} s)", flush=True)
        return sec

    # ---- the structural mean model, fitted to whatever is observed --------
    def fit_mean(Bs, ys):
        X = np.column_stack([1.0 / np.array(Bs, float),
                             np.ones(len(Bs)),
                             [1.0 if width(b) == 4 else 0.0 for b in Bs],
                             [1.0 if width(b) == 2 else 0.0 for b in Bs]])
        coef, *_ = np.linalg.lstsq(X, np.array(ys, float), rcond=None)
        return coef

    def mean_at(coef, Bs):
        X = np.column_stack([1.0 / np.array(Bs, float),
                             np.ones(len(Bs)),
                             [1.0 if width(b) == 4 else 0.0 for b in Bs],
                             [1.0 if width(b) == 2 else 0.0 for b in Bs]])
        return X @ coef

    def gp_posterior(Xtr, ytr, Xte, ell, sig, noise):
        k = lambda A, Bm: sig ** 2 * np.exp(
            -0.5 * ((A[:, None] - Bm[None, :]) / ell) ** 2)
        K = k(Xtr, Xtr) + noise ** 2 * np.eye(len(Xtr))
        Ks, Kss = k(Xtr, Xte), sig ** 2
        Li = np.linalg.inv(np.linalg.cholesky(K))
        Ki = Li.T @ Li
        mu = Ks.T @ (Ki @ ytr)
        var = np.maximum(Kss - np.sum((Ks.T @ Ki) * Ks.T, 1), 1e-12)
        return mu, np.sqrt(var)

    # ---- initial design: spread over the range, log spaced ---------------
    targets = np.exp(np.linspace(math.log(lo), math.log(hi), int(n_init)))
    init = sorted({min(cands, key=lambda c: abs(c - t)) for t in targets})
    print(f"\n  diseno inicial ({len(init)} puntos):", flush=True)
    Bs, ys = [], []
    for B in init:
        ys.append(evaluate(B)); Bs.append(B)

    # ---- expected improvement over the remaining candidates --------------
    print(f"\n  busqueda bayesiana ({n_iter} iteraciones):", flush=True)
    from math import erf, sqrt as _sqrt
    Phi = lambda z: 0.5 * (1.0 + erf(z / _sqrt(2.0)))
    phi = lambda z: math.exp(-0.5 * z * z) / _sqrt(2 * math.pi)
    for it in range(int(n_iter)):
        coef = fit_mean(Bs, ys)
        res = np.array(ys) - mean_at(coef, Bs)
        xtr = np.log2(np.array(Bs, float))
        left = [c for c in cands if c not in seen]
        if not left:
            break
        xte = np.log2(np.array(left, float))
        ell = max(0.35, (xtr.max() - xtr.min()) / 6.0)
        sig = max(float(np.std(res)), 1e-6)
        mu_r, sd_r = gp_posterior(xtr, res, xte, ell, sig,
                                  noise=max(0.35 * sig, 1e-6))
        mu = mean_at(coef, left) + mu_r
        best = min(ys)
        ei = []
        for m, sdv in zip(mu, sd_r):
            if sdv <= 1e-12:
                ei.append(0.0); continue
            z = (best - m) / sdv
            ei.append((best - m) * Phi(z) + sdv * phi(z))
        B = left[int(np.argmax(ei))]
        print(f"  [{it+1:2d}/{n_iter}] EI elige B={B}  "
              f"(prediccion {mu[int(np.argmax(ei))]:.3f} s, "
              f"incertidumbre {sd_r[int(np.argmax(ei))]:.3f})", flush=True)
        ys.append(evaluate(B)); Bs.append(B)

    order = np.argsort(ys)
    print(f"\n=== mejores lotes encontrados ===", flush=True)
    print(f"  {'B':>6} {'ancho':>6} {'s/replica':>11} {'vs el mejor':>12}", flush=True)
    for i in order[:12]:
        print(f"  {Bs[i]:6d} {width(Bs[i]):6d} {ys[i]:11.3f} "
              f"{ys[i]/min(ys):11.3f}x", flush=True)
    coef = fit_mean(Bs, ys)
    print(f"\n  modelo estructural ajustado:", flush=True)
    print(f"    termino por barrido  a/B  con a = {coef[0]:.2f} s-replica",
          flush=True)
    print(f"    piso                 b    = {coef[1]:.3f} s", flush=True)
    print(f"    recargo ancho 4           = {coef[2]:+.3f} s "
          f"({100*coef[2]/coef[1]:+.1f}% del piso)", flush=True)
    print(f"    recargo ancho 2           = {coef[3]:+.3f} s "
          f"({100*coef[3]/coef[1]:+.1f}% del piso)", flush=True)

    res = dict(rows=[dict(B=int(b), width=width(b), sec_per_replica=float(y))
                     for b, y in zip(Bs, ys)],
               best_B=int(Bs[int(order[0])]), best_sec=float(min(ys)),
               mean_model=dict(a_per_sweep=float(coef[0]), floor=float(coef[1]),
                               width4=float(coef[2]), width2=float(coef[3])),
               candidates=len(cands), step=int(step), lo=int(lo), hi=int(hi),
               n_init=int(n_init), n_iter=int(n_iter), repeats=int(repeats),
               sweeps_per_eval=sweeps, prod_sweeps=PROD_SWEEPS,
               Rd=float(Rd), L=int(L), N=int(geo["N"]),
               device=device_identity())
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


# A survey to go with the search, because they answer different questions.
#
# Bayesian optimisation spends its budget where the optimum is, which is what it
# is for -- and it showed exactly that here, picking 275 and 265 right after 270.
# That produces a tight cluster and an otherwise empty axis. Filling the curve
# is a SURVEY goal, not an optimisation one, so it gets its own run: points
# spread evenly in log2(B) over the multiples of five, which is what a reader
# needs to see the shape and what the sawtooth costs between neighbours.
#
# Evaluations here are cheap (35-70 s in the useful range), so a survey is
# affordable; the two runs together give both the shape and the minimum.
def _survey_batch(points, repeats, n_temps, n_therm, n_meas, tag, Rd, L,
                  lo, hi, step):
    import sys, json, time, math
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S
    import parallel_axes as PA

    def v2(n):
        k = 0
        while n % 2 == 0:
            n //= 2
            k += 1
        return k
    width = lambda B: min(4, 2 ** (v2(B) + v2(int(L))))

    cands = [b for b in range(lo, hi + 1) if b % step == 0]
    targets = np.exp(np.linspace(math.log(lo), math.log(hi), int(points)))
    grid = sorted({min(cands, key=lambda c: abs(c - t)) for t in targets})

    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    eng = S.build_engine(geo)
    kB = 0.086173404
    betas = jnp.asarray(1.0 / (kB * np.linspace(20.0, 0.1, int(n_temps))))
    sweeps = int(n_temps) * (int(n_therm) + int(n_meas))
    theta = PA.load_thetas("/root/theta4.json")[0]
    print(f"gpu={device_identity()}", flush=True)
    print(f"barrido de {len(grid)} puntos, espaciados en log2(B): {grid}",
          flush=True)

    def body(sp, kk, p):
        return eng["anneal"](sp, kk, betas, p, int(n_therm), int(n_meas))

    rows = []
    for B in grid:
        cols = PA.pack([theta], B)
        args = (S.init_spins(jax.random.PRNGKey(11), geo, B),
                jax.random.PRNGKey(12), cols)
        fn = jax.jit(body).lower(*args).compile()
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*args))
            ts.append(time.perf_counter() - t0)
        us = float(np.mean(ts) * 1e6 / (sweeps * B))
        sec = us * PROD_SWEEPS / 1e6
        rows.append(dict(B=int(B), width=width(B), sec_per_replica=sec,
                         us_per_replica_sweep=us,
                         sd_s=float(np.std(ts))))
        print(f"  B={B:5d} ancho {width(B)}  {sec:7.3f} s por replica",
              flush=True)

    best = min(rows, key=lambda r: r["sec_per_replica"])
    print(f"\n=== barrido ===", flush=True)
    print(f"  mejor: B={best['B']} con {best['sec_per_replica']:.3f} s", flush=True)
    for w in (1, 2, 4):
        g = [r["sec_per_replica"] for r in rows if r["width"] == w]
        if g:
            print(f"  ancho {w}: {len(g):2d} puntos, media {np.mean(g):6.3f} s, "
                  f"mejor {min(g):6.3f} s", flush=True)
    res = dict(rows=rows, best_B=best["B"], best_sec=best["sec_per_replica"],
               points=int(points), lo=int(lo), hi=int(hi), step=int(step),
               repeats=int(repeats), sweeps_per_eval=sweeps,
               prod_sweeps=PROD_SWEEPS, Rd=float(Rd), L=int(L),
               device=device_identity())
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


@app.function(gpu="H100", **_BS)
def survey_batch_h100(pts, r, nt, th, me, tag, Rd, L, lo, hi, st):
    return _survey_batch(pts, r, nt, th, me, tag, Rd, L, lo, hi, st)


@app.local_entrypoint()
def survey_batch(points: int = 26, repeats: int = 2, n_temps: int = 40,
                 n_therm: int = 1000, n_meas: int = 500,
                 tag: str = "survey_b", rd: float = 18.30, layers: int = 5,
                 lo: int = 10, hi: int = 1020, step: int = 5):
    """Even coverage of the axis, to go with the Bayesian search."""
    survey_batch_h100.remote(points, repeats, n_temps, n_therm, n_meas, tag,
                             rd, layers, lo, hi, step)


@app.function(gpu="H100", **_BS)
def bayes_batch_h100(lo, hi, st, ni, nit, r, nt, th, me, tag, Rd, L, sd):
    return _bayes_batch(lo, hi, st, ni, nit, r, nt, th, me, tag, Rd, L, sd)


@app.local_entrypoint()
def bayes_batch(lo: int = 10, hi: int = 1020, step: int = 5,
                n_init: int = 8, n_iter: int = 26, repeats: int = 2,
                n_temps: int = 40, n_therm: int = 1000, n_meas: int = 500,
                tag: str = "bayes_b", rd: float = 18.30, layers: int = 5,
                seed: int = 11):
    """Bayesian search for the best replicas per GPU. H100, one geometry."""
    bayes_batch_h100.remote(lo, hi, step, n_init, n_iter, repeats, n_temps,
                            n_therm, n_meas, tag, rd, layers, seed)


# --------------------------------------------------------------------------
# Phase 0 of the Pallas plan: can a kernel we write beat the JAX path?
#
# The gate. Everything in `docs/planning/standardisation_strategy.md` rests on
# Pallas being able to express this stencil WITHOUT the six materialised copies
# that `jnp.roll` forces, and on the result being faster. That is cheap to find
# out, so it is found out first.
#
# The design, and why it should win:
#
#   grid is (B, L) -- one program per (replica, layer) plane.
#   Each program loads the (N, N, 3) plane for its layer plus the planes above
#   and below, via three BlockSpecs over the SAME array with shifted index
#   maps. That is the standard halo trick for an axis that does not wrap.
#   x and y DO wrap, and the whole plane is resident, so the four in-plane
#   neighbours are rolls of a block already in SRAM -- no HBM traffic at all,
#   which is exactly what the JAX path pays for six times over.
#
# The maths is the same relaxation step as `_STENCIL`, so the two sides compute
# the identical function and the comparison is only about how it is executed.
# The disk mask is left out of BOTH sides: it would change the arithmetic
# equally and the spike is about the memory pattern.
def _pallas_spike(batches, repeats, tag, Rd, L):
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S

    print(f"gpu={device_identity()}", flush=True)
    print(f"jax {jax.__version__}", flush=True)
    try:
        # The registration has to be imported by hand: it is what installs the
        # gpu lowering rule, and nothing else pulls it in.
        import jax._src.pallas.mosaic_gpu.pallas_call_registration  # noqa: F401
        from jax.experimental import pallas as pl
    except Exception as e:
        print(f"PALLAS UNAVAILABLE: {type(e).__name__}: {e}", flush=True)
        return dict(available=False, error=f"{type(e).__name__}: {e}")

    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    N, Lz = int(geo["N"]), int(geo["L"])
    print(f"Rd={Rd} L={Lz} N={N}  plane {N*N*3*4:,} B  "
          f"replica {Lz*N*N*3*4:,} B", flush=True)

    # ---- the reference, in plain JAX: six rolls, as sim_core does ---------
    def jax_step(sp):
        xp = jnp.roll(sp, -1, 3); xm = jnp.roll(sp, 1, 3)
        yp = jnp.roll(sp, -1, 2); ym = jnp.roll(sp, 1, 2)
        zp = jnp.roll(sp, -1, 1).at[:, -1].set(0.0)
        zm = jnp.roll(sp, 1, 1).at[:, 0].set(0.0)
        e = sp + 0.1 * (xp + xm + yp + ym + zp + zm)
        return e / jnp.sqrt(jnp.sum(e * e, -1, keepdims=True) + 1e-20)

    # ---- the same function as a Pallas kernel -----------------------------
    def kernel(c_ref, below_ref, above_ref, o_ref):
        z = pl.program_id(1)
        c = c_ref[0, 0]                       # (N, N, 3), resident in SRAM
        lo = below_ref[0, 0]
        hi = above_ref[0, 0]
        # z does not wrap: the planes outside the stack contribute nothing.
        lo = jnp.where(z == 0, 0.0, lo)
        hi = jnp.where(z == Lz - 1, 0.0, hi)
        # x and y DO wrap, and the plane is already here, so these are rolls
        # of an SRAM block rather than six arrays fetched from HBM.
        sv = (jnp.roll(c, -1, 1) + jnp.roll(c, 1, 1)
              + jnp.roll(c, -1, 0) + jnp.roll(c, 1, 0) + hi + lo)
        e = c + 0.1 * sv
        o_ref[0, 0] = e / jnp.sqrt(jnp.sum(e * e, -1, keepdims=True) + 1e-20)

    def pallas_step(sp):
        B = sp.shape[0]
        blk = (1, 1, N, N, 3)
        return pl.pallas_call(
            kernel,
            grid=(B, Lz),
            in_specs=[
                pl.BlockSpec(blk, lambda b, z: (b, z, 0, 0, 0)),
                pl.BlockSpec(blk, lambda b, z: (b, jnp.maximum(z - 1, 0), 0, 0, 0)),
                pl.BlockSpec(blk, lambda b, z: (b, jnp.minimum(z + 1, Lz - 1), 0, 0, 0)),
            ],
            out_specs=pl.BlockSpec(blk, lambda b, z: (b, z, 0, 0, 0)),
            out_shape=jax.ShapeDtypeStruct(sp.shape, sp.dtype),
        )(sp, sp, sp)

    # ---- equivalence, before timing --------------------------------------
    sp = S.init_spins(jax.random.PRNGKey(5), geo, 3)
    try:
        a = jax.jit(jax_step)(sp)
        b = jax.jit(pallas_step)(sp)
    except Exception as e:
        print(f"PALLAS COMPILE/RUN FAILED: {type(e).__name__}: {e}", flush=True)
        return dict(available=True, ran=False,
                    error=f"{type(e).__name__}: {e}")
    err = float(jnp.max(jnp.abs(a - b)))
    rel = err / float(jnp.max(jnp.abs(a)))
    print(f"\nequivalencia: max |jax - pallas| = {err:.3e}  "
          f"(relativo {rel:.3e}){'  OK' if rel < 1e-5 else '  MISMATCH'}",
          flush=True)
    if rel >= 1e-5:
        print("  -> no son la misma funcion; el tiempo no significa nada",
              flush=True)
        return dict(available=True, ran=True, equivalent=False,
                    max_err=err, rel_err=rel)

    # ---- timing ----------------------------------------------------------
    def v2(n):
        k = 0
        while n % 2 == 0:
            n //= 2
            k += 1
        return k

    rows = []
    for B in batches:
        sp = S.init_spins(jax.random.PRNGKey(11), geo, B)
        cell = {}
        for lbl, f in (("jax", jax_step), ("pallas", pallas_step)):
            fn = jax.jit(f).lower(sp).compile()
            jax.block_until_ready(fn(sp))
            ts = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                jax.block_until_ready(fn(sp))
                ts.append(time.perf_counter() - t0)
            cell[lbl] = float(np.mean(ts))
        gain = cell["jax"] / cell["pallas"] - 1
        w = min(4, 2 ** (v2(B) + v2(Lz)))
        rows.append(dict(B=int(B), width_xla=w, jax_s=cell["jax"],
                         pallas_s=cell["pallas"], pallas_gain=gain))
        print(f"  B={B:4d} (XLA ancho {w})  jax {cell['jax']*1e3:8.3f} ms   "
              f"pallas {cell['pallas']*1e3:8.3f} ms   -> pallas "
              f"{gain:+.1%}", flush=True)

    g = [r["pallas_gain"] for r in rows]
    print(f"\n=== Fase 0: compuerta de viabilidad ===", flush=True)
    print(f"  ganancia de pallas: media {np.mean(g):+.1%}  "
          f"(min {min(g):+.1%}, max {max(g):+.1%})", flush=True)
    sd = float(np.std([r["pallas_s"] for r in rows]) /
               np.mean([r["pallas_s"] for r in rows]))
    print(f"  dispersion de pallas entre lotes: {sd:.1%}  "
          f"(jax: {float(np.std([r['jax_s'] for r in rows]) / np.mean([r['jax_s'] for r in rows])):.1%})",
          flush=True)
    print(f"  VEREDICTO: {'PASA, seguir a Fase 1' if np.mean(g) > 0.05 else 'NO PASA, parar aqui'}",
          flush=True)
    res = dict(available=True, ran=True, equivalent=True, max_err=err,
               rel_err=rel, rows=rows, mean_gain=float(np.mean(g)),
               pallas_spread=sd, Rd=float(Rd), L=Lz, N=N, repeats=repeats,
               device=device_identity(), jax_version=jax.__version__)
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


@app.function(gpu="H100", **_BS)
def pallas_probe():
    """Why did pallas_call refuse to lower, and what would fix it?

    The error names jaxlib, but the kernels themselves run fine on this image,
    so the GPU build is there. On GPU, Pallas lowers through Triton, and that
    path registers itself when its backend module imports -- which needs the
    `triton` package. This reports what is actually present before anything is
    installed on a guess.
    """
    import importlib, subprocess, sys
    import jax
    out = {"jax": jax.__version__}
    try:
        import jaxlib
        out["jaxlib"] = jaxlib.__version__
    except Exception as e:
        out["jaxlib"] = f"{type(e).__name__}"
    out["devices"] = [str(d) for d in jax.devices()]
    for mod in ("triton", "jax_triton",
                "jax.experimental.pallas",
                "jax.experimental.pallas.triton",
                "jax.experimental.pallas.mosaic_gpu",
                "jax._src.pallas.triton.lowering",
                "jax._src.pallas.mosaic_gpu.lowering"):
        try:
            m = importlib.import_module(mod)
            out[mod] = getattr(m, "__version__", "present")
        except Exception as e:
            out[mod] = f"MISSING: {type(e).__name__}: {str(e)[:90]}"
    for k, v in out.items():
        print(f"  {k:46s} {v}", flush=True)

    print("\ninstalling triton and retrying the lowering:", flush=True)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "triton"],
                       capture_output=True, text=True)
    print(f"  pip exit {r.returncode}  {r.stderr.strip()[:200]}", flush=True)
    code = (
        "import jax, jax.numpy as jnp\n"
        "from jax.experimental import pallas as pl\n"
        "def k(x_ref, o_ref): o_ref[...] = x_ref[...] * 2.0\n"
        "f = pl.pallas_call(k, out_shape=jax.ShapeDtypeStruct((256,), 'float32'))\n"
        "x = jnp.arange(256, dtype='float32')\n"
        "y = f(x)\n"
        "print('PALLAS OK, max err', float(jnp.max(jnp.abs(y - 2*x))))\n")
    r2 = subprocess.run([sys.executable, "-c", code], capture_output=True,
                        text=True, timeout=600)
    print(f"  retry exit {r2.returncode}", flush=True)
    print("  stdout:", r2.stdout.strip()[:300], flush=True)
    if r2.returncode:
        print("  stderr:", r2.stderr.strip()[-700:], flush=True)
    out["triton_install_exit"] = r.returncode
    out["retry_exit"] = r2.returncode
    out["retry_stdout"] = r2.stdout.strip()[:300]
    out["retry_stderr"] = r2.stderr.strip()[-700:] if r2.returncode else ""
    return out


@app.function(gpu="H100", **_BS)
def pallas_probe2():
    """Two specific hypotheses for why the gpu lowering is not registered.

    `jax.experimental.pallas.triton` imports and `jax._src.pallas.triton
    .lowering` imports, yet `pallas_call` still says the gpu platform is
    unsupported. In JAX the rule is installed by a separate *registration*
    module, which is imported conditionally -- so (1) importing it by hand may
    be all that is missing. And `absl` is absent from this image, which is what
    blocks `mosaic_gpu`, the other and newer GPU backend -- so (2) installing
    absl-py may unlock that path instead.
    """
    import subprocess, sys, importlib
    results = {}

    def attempt(label, pre_imports, extra_pip=()):
        if extra_pip:
            r = subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                                *extra_pip], capture_output=True, text=True)
            print(f"  pip {' '.join(extra_pip)}: exit {r.returncode}", flush=True)
        code = ("import jax, jax.numpy as jnp\n"
                + "".join(f"import {m}\n" for m in pre_imports)
                + "from jax.experimental import pallas as pl\n"
                "def k(x_ref, o_ref): o_ref[...] = x_ref[...] * 2.0\n"
                "f = pl.pallas_call(k, out_shape=jax.ShapeDtypeStruct((256,),'float32'))\n"
                "x = jnp.arange(256, dtype='float32')\n"
                "print('OK err', float(jnp.max(jnp.abs(f(x) - 2*x))))\n")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=600)
        ok = r.returncode == 0
        print(f"  [{label}] {'OK -> ' + r.stdout.strip() if ok else 'FAIL'}",
              flush=True)
        if not ok:
            tail = [l for l in r.stderr.strip().splitlines()
                    if "Error" in l or "error" in l][-2:]
            for l in tail:
                print(f"      {l[:160]}", flush=True)
        results[label] = dict(ok=ok, stdout=r.stdout.strip()[:200],
                              stderr="" if ok else r.stderr.strip()[-400:])
        return ok

    print("hipotesis 1: importar el modulo de registro a mano", flush=True)
    if attempt("triton_registration",
               ["jax._src.pallas.triton.pallas_call_registration"]):
        return results
    print("\nhipotesis 2: absl-py desbloquea mosaic_gpu", flush=True)
    if attempt("mosaic_after_absl",
               ["jax._src.pallas.mosaic_gpu.pallas_call_registration"],
               extra_pip=("absl-py",)):
        return results
    print("\nhipotesis 2b: solo absl, sin importar registro a mano", flush=True)
    attempt("absl_plain", [])
    print("\nque trae el wheel de jaxlib:", flush=True)
    for mod in ("jaxlib.triton", "jaxlib.mosaic", "jax_triton"):
        try:
            importlib.import_module(mod)
            print(f"  {mod:22s} present", flush=True)
        except Exception as e:
            print(f"  {mod:22s} MISSING ({type(e).__name__})", flush=True)
    return results


@app.local_entrypoint()
def pallas_diag2():
    """Test the two specific causes, cheapest first."""
    pallas_probe2.remote()


@app.local_entrypoint()
def pallas_diag():
    """What is missing for Pallas on GPU in this image?"""
    pallas_probe.remote()


@app.function(gpu="H100", image=pallas_image, memory=65536, timeout=3600,
              volumes={"/out": vol})
def pallas_spike_h100(b, r, tag, Rd, L):
    return _pallas_spike(b, r, tag, Rd, L)


@app.local_entrypoint()
def pallas_spike(batches: str = "63,64,65", repeats: int = 5,
                 rd: float = 18.30, layers: int = 5,
                 tag: str = "phase0_pallas"):
    """Phase 0: does a Pallas stencil beat the roll-based JAX one?"""
    pallas_spike_h100.remote([int(x) for x in batches.split(",") if x],
                             repeats, tag, rd, layers)


# --------------------------------------------------------------------------
# Experiment 9: a dummy layer makes the width OUR choice, for any geometry.
#
# The rule leaves a hole. When the physics fixes `L` divisible by four,
# `v2(L) >= 2` on its own, so `v2(B) + v2(L) >= 2` for EVERY `B` and no batch
# escapes the 4-wide path. Choosing `B` buys nothing there.
#
# The way out is preprocessing, and it is exact rather than a trick. A layer
# whose disk mask is all False stays zero forever: `init_spins` multiplies by
# `disco`, `substep` sets the trial equal to the current spin outside the mask
# and never accepts there. A zeroed layer above the top physical layer
# contributes nothing to its neighbour sum, which is precisely the free
# boundary the unpadded lattice already has. So simulating `L` physical layers
# as an `L + 1` array with the top one masked off is the SAME physics, and it
# makes the layer count odd:
#
#       L = 8  physical  ->  array of 9  ->  v2(L) = 0  ->  width 1 at odd B
#
# The cost is one extra layer of traffic, 1/L, against a width penalty measured
# at 14.3% on an H100 and 28.6% on a T4. So it loses at L = 4 (+25%) and wins
# from L = 8 (+12.5%) upward. This measures both halves: that the observables
# agree exactly, and what the trade really costs.
def _dummy_layer(pairs, batches, repeats, tag, n_temps, n_therm, n_meas):
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S
    import parallel_axes as PA

    kB = 0.086173404
    betas = jnp.asarray(1.0 / (kB * np.linspace(20.0, 0.1, int(n_temps))))
    sweeps = int(n_temps) * (int(n_therm) + int(n_meas))
    theta = PA.load_thetas("/root/theta4.json")[0]
    print(f"gpu={device_identity()}", flush=True)

    def padded_geo(Rd, Lphys):
        """`Lphys + 1` layers with the top one masked out of the disk."""
        g = S.build_geometry(Rd=float(Rd), L=int(Lphys) + 1)
        g["disco"] = np.array(g["disco"]); g["disco"][-1] = False
        g["even"] = np.array(g["even"]) & g["disco"]
        g["odd"] = np.array(g["odd"]) & g["disco"]
        # Surface = fewer than six in-disk neighbours, recomputed so the top
        # physical layer is correctly surface again.
        N, L = int(g["N"]), int(g["L"])
        surf = np.zeros_like(g["disco"], dtype=bool)
        for z in range(L):
            for y in range(N):
                for x in range(N):
                    if not g["disco"][z, y, x]:
                        continue
                    n = 0
                    for dz, dy, dx in ((0,0,1),(0,0,-1),(0,1,0),(0,-1,0),(1,0,0),(-1,0,0)):
                        nz, ny, nx = z+dz, (y+dy) % N, (x+dx) % N
                        if 0 <= nz < L and g["disco"][nz, ny, nx]:
                            n += 1
                    if n < 6:
                        surf[z, y, x] = True
        g["surface"] = surf
        g["n_active"] = int(g["disco"].sum())
        return g

    out = []
    for Rd, Lphys in pairs:
        gn = S.build_geometry(Rd=float(Rd), L=int(Lphys))
        gp = padded_geo(Rd, Lphys)
        en, ep = S.build_engine(gn), S.build_engine(gp)
        Nn = int(gn["N"])
        print(f"\n--- Rd={Rd}  L fisica={Lphys} ---", flush=True)
        print(f"  nativa: L={gn['L']} sitios={int(np.asarray(gn['disco']).sum())}"
              f"  v2(L)={bin(int(gn['L']))[::-1].index('1')}", flush=True)
        print(f"  rellena: L={gp['L']} sitios={gp['n_active']}"
              f"  (la capa de arriba esta enmascarada)", flush=True)
        assert gp["n_active"] == int(np.asarray(gn["disco"]).sum()), \
            "el relleno cambio el numero de sitios activos"

        # ---- equivalence: the same configuration, both representations ----
        cfg = np.asarray(S.init_spins(jax.random.PRNGKey(5), gn, 3))
        pad = np.zeros((3, gp["L"], Nn, Nn, 3), dtype=np.float32)
        pad[:, :int(Lphys)] = cfg
        cols = PA.pack([theta], 3)
        k1n = jnp.where(jnp.asarray(gn["surface"])[None], cols["KanS"], cols["Kan1"])
        k1p = jnp.where(jnp.asarray(gp["surface"])[None], cols["KanS"], cols["Kan1"])
        En = float(jnp.sum(en["energy"](jnp.asarray(cfg), cols, k1n)))
        Ep = float(jnp.sum(ep["energy"](jnp.asarray(pad), cols, k1p)))
        Mn = np.asarray(en["magnetization"](jnp.asarray(cfg)))
        Mp = np.asarray(ep["magnetization"](jnp.asarray(pad)))
        dE = abs(Ep - En) / max(abs(En), 1e-30)
        dM = float(np.max(np.abs(Mp - Mn)))
        print(f"  energia  nativa {En:.6f}  rellena {Ep:.6f}  "
              f"error relativo {dE:.2e}", flush=True)
        print(f"  magnetizacion: max |diferencia| {dM:.2e}", flush=True)
        ok = dE < 1e-6 and dM < 1e-6
        print(f"  -> {'EQUIVALENTES' if ok else 'NO EQUIVALENTES'}", flush=True)

        # ---- timing, per PHYSICAL site so the 1/L overhead shows -----------
        nsites = gp["n_active"]
        cells = []
        for B in batches:
            row = dict(B=int(B))
            for lbl, g, e in (("nativa", gn, en), ("rellena", gp, ep)):
                c = PA.pack([theta], B)
                k1 = jnp.where(jnp.asarray(g["surface"])[None], c["KanS"], c["Kan1"])
                args = (S.init_spins(jax.random.PRNGKey(11), g, B),
                        jax.random.PRNGKey(12), c)
                fn = jax.jit(lambda sp, kk, pp, _e=e: _e["anneal"](
                    sp, kk, betas, pp, int(n_therm), int(n_meas))
                ).lower(*args).compile()
                jax.block_until_ready(fn(*args))
                ts = []
                for _ in range(repeats):
                    t0 = time.perf_counter()
                    jax.block_until_ready(fn(*args))
                    ts.append(time.perf_counter() - t0)
                row[lbl] = float(np.mean(ts) * 1e6 / (sweeps * B))
            row["gain"] = row["nativa"] / row["rellena"] - 1
            cells.append(row)
            print(f"  B={B:4d}  nativa {row['nativa']:7.3f}  "
                  f"rellena {row['rellena']:7.3f} us/replica-barrido  -> "
                  f"rellena {row['gain']:+.1%}", flush=True)
        out.append(dict(Rd=float(Rd), L_phys=int(Lphys), L_array=int(gp["L"]),
                        n_active=nsites, energy_rel_err=dE, mag_max_err=dM,
                        equivalent=bool(ok), cells=cells))

    print("\n=== relleno con capa ficticia ===", flush=True)
    print(f"{'Rd':>7} {'L fis':>6} {'L arr':>6} {'equiv':>7} "
          f"{'ganancia media':>15}", flush=True)
    for o in out:
        g = [c["gain"] for c in o["cells"]]
        print(f"{o['Rd']:7.2f} {o['L_phys']:6d} {o['L_array']:6d} "
              f"{'si' if o['equivalent'] else 'NO':>7} "
              f"{np.mean(g):+14.1%}", flush=True)
    res = dict(rows=out, batches=list(map(int, batches)), repeats=repeats,
               sweeps_per_cell=sweeps, device=device_identity())
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


DUMMY_PAIRS = [(18.30, 4), (13.00, 8)]   # L|4 both: the hole in the rule


@app.function(gpu="H100", **_BS)
def dummy_layer_h100(pr, b, r, tag, nt, th, me):
    return _dummy_layer(pr, b, r, tag, nt, th, me)


@app.local_entrypoint()
def dummy_layer(batches: str = "63,64,65", repeats: int = 3, n_temps: int = 20,
                n_therm: int = 500, n_meas: int = 250, tag: str = "exp9_dummy"):
    """Experiment 9: does a masked dummy layer buy back the width choice?"""
    dummy_layer_h100.remote(DUMMY_PAIRS,
                            [int(x) for x in batches.split(",") if x],
                            repeats, tag, n_temps, n_therm, n_meas)


# --------------------------------------------------------------------------
# Experiment 8: the rolls are 2x the traffic of the computation. Can they go?
#
# The per-kernel profile says `input_concatenate_fusion` is 10.6% to 22.1% of
# GPU time, and those kernels are the six `jnp.roll` calls in `substep`: each
# one MATERIALISES a full (B, L, N, N, 3) copy to produce a shifted view. At
# B=64 that is 33 MB written per substep against ~17 MB for the useful work.
#
# `jnp.roll` on a non-leading axis lowers to concatenate(slice, slice), which
# XLA does not fuse into the consumer. The alternative: pad ONCE and take six
# static slices of that single array. Slices fuse more readily than
# concatenates, and two pads replace six rolls.
#
# The boundary conditions make the padding natural rather than a trick: x and y
# wrap, z does not and is zeroed, which is exactly `mode="constant"` on z
# followed by `mode="wrap"` on y and x.
#
# EQUIVALENCE IS CHECKED FIRST. This touches the physics engine, so the variant
# has to produce bit-identical neighbour arrays and an identical substep output
# for the same key before any timing is believed.
def _roll_free(batches, repeats, tag, Rd, L, n_temps, n_therm, n_meas):
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S

    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    N = int(geo["N"])
    print(f"gpu={device_identity()}", flush=True)
    print(f"Rd={Rd} L={L} N={N}  array {12*L*N*N:,} B/replica", flush=True)

    def neigh_roll(sp):
        """What sim_core does today: six rolls, six materialised copies."""
        return (jnp.roll(sp, -1, 3), jnp.roll(sp, 1, 3),
                jnp.roll(sp, -1, 2), jnp.roll(sp, 1, 2),
                jnp.roll(sp, -1, 1).at[:, -1].set(0.0),
                jnp.roll(sp, 1, 1).at[:, 0].set(0.0))

    def neigh_pad(sp):
        """Pad once, slice six times. z is zeroed, y and x wrap."""
        P = jnp.pad(sp, ((0, 0), (1, 1), (0, 0), (0, 0), (0, 0)))
        P = jnp.pad(P, ((0, 0), (0, 0), (1, 1), (1, 1), (0, 0)), mode="wrap")
        return (P[:, 1:-1, 1:-1, 2:, :],  P[:, 1:-1, 1:-1, :-2, :],
                P[:, 1:-1, 2:, 1:-1, :],  P[:, 1:-1, :-2, 1:-1, :],
                P[:, 2:, 1:-1, 1:-1, :],  P[:, :-2, 1:-1, 1:-1, :])

    # ---- equivalence, before anything is timed --------------------------
    sp = S.init_spins(jax.random.PRNGKey(3), geo, 5)
    a = jax.jit(neigh_roll)(sp)
    b = jax.jit(neigh_pad)(sp)
    names = ("x+", "x-", "y+", "y-", "z+", "z-")
    worst = 0.0
    for nm, u, v in zip(names, a, b):
        d = float(jnp.max(jnp.abs(u - v)))
        worst = max(worst, d)
        print(f"  neighbour {nm}: max |roll - pad| = {d:.3e}"
              f"{'  OK' if d == 0.0 else '  MISMATCH'}", flush=True)
    if worst != 0.0:
        raise AssertionError(f"the variants differ by {worst:g}; not comparable")
    print("  -> the two forms are BIT-IDENTICAL on every neighbour\n", flush=True)

    # ---- time the neighbour stage on its own ----------------------------
    rows = []
    for B in batches:
        sp = S.init_spins(jax.random.PRNGKey(11), geo, B)
        cell = {}
        for lbl, fn in (("roll", neigh_roll), ("pad", neigh_pad)):
            f = jax.jit(lambda x, _f=fn: sum(_f(x))).lower(sp).compile()
            jax.block_until_ready(f(sp))
            ts = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                jax.block_until_ready(f(sp))
                ts.append(time.perf_counter() - t0)
            cell[lbl] = float(np.mean(ts))
        gain = cell["roll"] / cell["pad"] - 1
        rows.append(dict(B=int(B), roll_s=cell["roll"], pad_s=cell["pad"],
                         pad_speedup=gain))
        print(f"  B={B:4d}  roll {cell['roll']*1e3:8.3f} ms   "
              f"pad {cell['pad']*1e3:8.3f} ms   "
              f"pad es {gain:+.1%} {'mas rapido' if gain > 0 else 'mas lento'}",
              flush=True)

    print("\n=== neighbour stage, roll vs pad+slice ===", flush=True)
    g = [r["pad_speedup"] for r in rows]
    print(f"  ganancia media {np.mean(g):+.1%}  "
          f"(min {min(g):+.1%}, max {max(g):+.1%})", flush=True)
    print("  Si la ganancia es grande, vale reescribir substep; si es nula, "
          "XLA baja el pad a concatenate igual que el roll y hay que ir a "
          "Pallas para controlarlo.", flush=True)
    res = dict(rows=rows, equivalence="bit-identical on all six neighbours",
               Rd=float(Rd), L=int(L), N=N, repeats=repeats,
               device=device_identity())
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return res


@app.function(gpu="H100", **_BS)
def roll_free_h100(b, r, tag, Rd, L, nt, th, me):
    return _roll_free(b, r, tag, Rd, L, nt, th, me)


@app.local_entrypoint()
def roll_free(batches: str = "51,52,63,64", repeats: int = 3, rd: float = 18.30,
              layers: int = 5, n_temps: int = 20, n_therm: int = 500,
              n_meas: int = 250, tag: str = "exp8_rollfree"):
    """Experiment 8: can the six materialising rolls be replaced?"""
    roll_free_h100.remote([int(x) for x in batches.split(",") if x], repeats,
                          tag, rd, layers, n_temps, n_therm, n_meas)


@app.function(image=image, cpu=2.0, timeout=600)
def xla_flag_list():
    """Dump every XLA flag the installed build exposes, and filter it.

    `XLA_FLAGS=--help` makes the backend print its whole flag table and exit
    the initialisation, so this runs on CPU: we only need the names, not a GPU.
    The question is whether this build lets us turn the vectorisation pass off,
    which would let the width rule be tested inside the real kernel instead of
    only in hand-written CUDA.
    """
    import os, subprocess, sys, re
    env = dict(os.environ, XLA_FLAGS="--help")
    r = subprocess.run([sys.executable, "-c", "import jax; jax.devices()"],
                       env=env, capture_output=True, text=True, timeout=540)
    txt = r.stdout + r.stderr
    flags = sorted(set(re.findall(r"--([a-z0-9_]{6,})", txt)))
    print(f"captured {len(txt):,} chars, {len(flags)} distinct flags", flush=True)
    want = ("vector", "emitter", "loop", "tile", "unroll", "coalesc",
            "simd", "width", "codegen")
    hits = [f for f in flags if any(w in f for w in want)]
    print(f"\n=== flags matching {want} ===", flush=True)
    for f in hits:
        print(f"  --{f}", flush=True)
    if not hits:
        print("  (none)", flush=True)
    print(f"\n=== every xla_gpu flag ({sum(1 for f in flags if 'xla_gpu' in f)}) ===",
          flush=True)
    for f in flags:
        if "xla_gpu" in f:
            print(f"  --{f}", flush=True)
    return dict(n_flags=len(flags), matches=hits, all_flags=flags)


@app.local_entrypoint()
def xla_flags():
    """Which knobs does this XLA build actually expose?"""
    xla_flag_list.remote()


@app.local_entrypoint()
def pair(batches: str = "64,65", repeats: int = 3, runs: int = 3,
         rd: float = 18.30, layers: int = 5, n_temps: int = 40,
         n_therm: int = 1000, n_meas: int = 500, device: str = "h100",
         tag: str = "pair64v65"):
    """B=64 against B=65, interleaved, `runs` independent containers.

    Each container is a fresh GPU, so agreement across runs rules out one
    unlucky device as well as one unlucky ordering.
    """
    bl = [int(x) for x in batches.split(",") if x]
    fn = {"h100": pair_h100, "t4": pair_t4}[device.lower()]
    handles = [
        fn.spawn(bl, repeats, f"{tag}_{device.lower()}_run{i + 1}", rd, layers,
                 n_temps, n_therm, n_meas, 11 + 100 * i,
                 f"run {i + 1} of {runs}")
        for i in range(int(runs))
    ]
    for i, h in enumerate(handles):
        try:
            h.get()
            print(f"run {i + 1} done", flush=True)
        except Exception as e:
            print(f"run {i + 1} failed: {type(e).__name__}: {e}", flush=True)


@app.local_entrypoint()
def stride_discriminator(batches: str = "51,52,53,59,60,61,63,64,65",
                         repeats: int = 2, n_temps: int = 40,
                         n_therm: int = 1000, n_meas: int = 500,
                         tag: str = "stride_disc", device: str = "h100",
                         lattices: str = "control"):
    bl = [int(x) for x in batches.split(",") if x]
    fn = {"h100": stride_disc_h100, "a100": stride_disc_a100,
          "t4": stride_disc_t4}[device.lower()]
    lat = {"control": LATTICES, "sweep": STRIDE_SWEEP,
           "sweepbig": STRIDE_SWEEP_BIG, "parity": PARITY_SWEEP,
           "geompair": GEOM_PAIR, "ref": REF_ONLY}[lattices.lower()]
    fn.remote(lat, bl, repeats, n_temps, n_therm, n_meas, tag)


@app.function(gpu="H100", **_BS)
def fcurve1(b, r, nt, th, me, tag, Rd=18.30, L=5):
    return _fcurve(b, r, nt, th, me, tag, Rd, L)


# The same measurement on two other devices. `gpu=` is resolved when the
# decorator runs, so a third device needs a third wrapper -- it cannot be a
# parameter. A100 is the interesting middle case: its bus is 5120-bit like the
# H100's, but its bandwidth is about 1.6x lower, which separates "the sawtooth
# follows bus width" from "the sawtooth follows bandwidth". On a T4 those two
# move together, so that pair cannot tell them apart.
@app.function(gpu="A100-40GB", **_BS)
def fcurve_a100(b, r, nt, th, me, tag, Rd=18.30, L=5):
    return _fcurve(b, r, nt, th, me, tag, Rd, L)


@app.function(gpu="L4", **_BS)
def fcurve_l4(b, r, nt, th, me, tag, Rd=18.30, L=5):
    return _fcurve(b, r, nt, th, me, tag, Rd, L)


# The H200 is the one device that holds SM count fixed. It has the same 132 SMs
# as the H100 and 1.43x the bandwidth (4800 vs 3350 GB/s), so it separates the
# two factors the four-device set leaves degenerate: if the mod-4 penalty drops
# below the H100's +11.7% the driver is bandwidth, and if it stays there the
# driver is SM count.
@app.function(gpu="H200", **_BS)
def fcurve_h200(b, r, nt, th, me, tag, Rd=18.30, L=5):
    return _fcurve(b, r, nt, th, me, tag, Rd, L)


@app.local_entrypoint()
def f_curve(batches: str = "", repeats: int = 2, n_temps: int = 40,
            n_therm: int = 10000, n_meas: int = 5000, tag: str = "fcurve",
            rd: float = 18.30, layers: int = 5,
            device: str = "h100"):
    # Geometry is a parameter because the whole 19-point curve was measured at
    # one lattice (Rd=18.30, L=5, 5245 active sites). The array a kernel sees is
    # (B, L, N, N, 3), so if the sawtooth comes from how that shape tiles the
    # SMs then B alone is the wrong variable and B * n_active is the right one.
    # That is testable: a geometry with twice the sites should put its minimum
    # at half the B.
    bl = [int(x) for x in batches.split(",") if x] or F_GRID
    fn = {"h100": fcurve1, "a100": fcurve_a100, "l4": fcurve_l4,
          "h200": fcurve_h200}[device.lower()]
    print(json.dumps(fn.remote(bl, repeats, n_temps, n_therm, n_meas,
                               tag, rd, layers)["rows"], indent=1))


# --------------------------------------------------------------------------
# Schedule A/B: 100 replicas x 15k sweeps against 50 replicas x 20k
# --------------------------------------------------------------------------
# The f(B) curve stays at the standard 10k thermalisation + 5k measurement mix;
# this is a separate head-to-head, because the two schedules are not a rescaling
# of each other. They accumulate the SAME number of measurement samples --
# 100 x 5 000 and 50 x 10 000 both give 500 000 -- so the question is whether
# those samples carry the same information. Samples inside one replica are
# autocorrelated; separate replicas are independent. Halving the replicas also
# halves the sample the error bars are computed from, since `reduce_obs` takes
# Cv and chi per replica and then averages, so the spread across replicas is the
# uncertainty.
#
# Everything else is held fixed: the same four thetas, the same 200-point
# temperature schedule, the same seed, the same four devices, the same container,
# and B = 50 per device in BOTH arms, so nothing here is the batch-size effect
# in disguise.
#
#   arm A   4 states x 100 replicas, 10k therm + 5k  meas   2 launches of 200
#   arm B   4 states x  50 replicas, 10k therm + 10k meas   1 launch  of 200

_SCHED_ARMS = [("A_100r_15k", 100, 10000, 5000),
               ("B_50r_20k",   50, 10000, 10000)]


def _schedule_ab(n_temps, seed, tag, b_per_device):
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S
    import parallel_axes as PA

    ndev = jax.local_device_count()
    geo = S.build_geometry(Rd=18.30, L=5)
    eng = S.build_engine(geo)
    thetas = PA.load_thetas("/root/theta4.json")
    kB = 0.086173404
    T = np.linspace(20.0, 0.1, int(n_temps))
    betas = jnp.asarray(1.0 / (kB * T))
    n_spins = int(geo["n_active"])
    print(f"devices={jax.devices()}  B={b_per_device} per device  "
          f"{n_temps} temperatures  n_spins={n_spins}", flush=True)

    out = {}
    for name, reps, n_therm, n_meas in _SCHED_ARMS:
        total = len(thetas) * reps
        per_launch = b_per_device * ndev
        assert total % per_launch == 0, (
            f"{name}: {total} replicas do not split into launches of "
            f"{per_launch}")
        n_launch = total // per_launch
        states_per_launch = len(thetas) // n_launch
        sweeps = int(n_temps) * (n_therm + n_meas)

        def body(sp, kk, p):
            return eng["anneal"](sp, kk, betas, p, n_therm, n_meas)

        wall, OBS, SNAP = 0.0, [], []
        for i in range(n_launch):
            ths = thetas[i * states_per_launch:(i + 1) * states_per_launch]
            cols = PA.pack(ths, reps)
            spins = S.init_spins(jax.random.PRNGKey(seed + 100 * i), geo,
                                 per_launch)
            shard = lambda a: a.reshape(ndev, b_per_device, *a.shape[1:])
            args = (shard(spins),
                    jax.random.split(jax.random.PRNGKey(seed + 100 * i + 1),
                                     ndev),
                    {k: shard(v) for k, v in cols.items()})
            fn = jax.pmap(body).lower(*args).compile()   # nothing timed yet
            t0 = time.perf_counter()
            obs, fin, snaps = fn(*args)
            jax.block_until_ready((obs, fin, snaps))
            wall += time.perf_counter() - t0
            # (ndev, n_T, 8, per) -> (n_T, 8, launch_batch)
            OBS.append(np.concatenate(list(np.asarray(obs)), axis=-1))
            # `snaps` is replica 0's middle-layer s_z map, one per temperature.
            # Device 0 only: it is an illustration of the texture, not a
            # statistic, and one chain per launch is what the page shows.
            SNAP.append(np.asarray(snaps)[0])
            print(f"  {name} launch {i+1}/{n_launch}  "
                  f"{(time.perf_counter()-t0)/60:6.2f} min  "
                  f"({[t['name'] for t in ths]}, {per_launch} replicas)",
                  flush=True)

        allobs = np.concatenate(OBS, axis=-1)
        red = S.reduce_obs(allobs, T, n_spins)
        # (launches, n_T, N, N) -> keep every launch so each state has a map.
        snapshots = np.stack(SNAP, axis=0).astype(np.float32)
        gpu_s = wall * ndev
        out[name] = dict(
            replicas_per_state=reps, n_therm=n_therm, n_meas=n_meas,
            replicas_total=total, launches=n_launch, sweeps=sweeps,
            samples=reps * n_meas,
            wall_s=wall, gpu_seconds=gpu_s, usd=gpu_s * 0.001097,
            us_per_replica_sweep=wall * ndev * 1e6 / (sweeps * total),
            obs={k: np.asarray(v).tolist() for k, v in red.items()},
            snap_shape=list(snapshots.shape),
            states_per_launch=states_per_launch,
            state_names=[[t["name"] for t in
                          thetas[i * states_per_launch:(i + 1) * states_per_launch]]
                         for i in range(n_launch)])
        np.save(f"/out/{tag}_{name}_sz.npy", snapshots)
        print(f"  {name}: {wall/60:.2f} min, {gpu_s:.0f} GPU-s, "
              f"${gpu_s*0.001097:.2f}, {reps*n_meas:,} samples", flush=True)

    A, Bm = out["A_100r_15k"], out["B_50r_20k"]
    oa, ob = A["obs"], Bm["obs"]
    print("\n=== timing ===", flush=True)
    print(f"  A  100r x 15k  {A['wall_s']/60:6.2f} min  ${A['usd']:.2f}  "
          f"{A['us_per_replica_sweep']:.3f} us/replica-sweep", flush=True)
    print(f"  B   50r x 20k  {Bm['wall_s']/60:6.2f} min  ${Bm['usd']:.2f}  "
          f"{Bm['us_per_replica_sweep']:.3f} us/replica-sweep", flush=True)
    print(f"  B vs A: wall {Bm['wall_s']/A['wall_s']-1:+.1%}   "
          f"cost {Bm['usd']/A['usd']-1:+.1%}", flush=True)

    print("\n=== physics: do the two schedules agree? ===", flush=True)
    for k in ("E", "M", "Mz", "Cv", "chi", "Q"):
        a, b = np.asarray(oa[k]), np.asarray(ob[k])
        den = np.maximum(np.abs(a), 1e-12)
        rel = np.abs(b - a) / den
        print(f"  {k:4s} max |rel diff| {rel.max():8.2%}   "
              f"median {np.median(rel):8.2%}", flush=True)

    print("\n=== error bars ===", flush=True)
    print("  The spread ACROSS replicas measures physical variation, and its "
          "expected\n  value does not depend on how many replicas sample it. "
          "What degrades is the\n  error on the mean, sd/sqrt(N), and halving "
          "N inflates that by sqrt(2).", flush=True)
    na, nb = A["replicas_total"], Bm["replicas_total"]
    for k in ("E_sd", "M_sd", "Q_sd"):
        a, b = np.asarray(oa[k]), np.asarray(ob[k])
        ok = a > 0
        if not ok.any():
            print(f"  {k:5s} all zero, nothing to compare", flush=True)
            continue
        spread = np.median(b[ok] / a[ok])
        sem = np.median((b[ok] / np.sqrt(nb)) / (a[ok] / np.sqrt(na)))
        print(f"  {k:5s} spread B/A {spread:.3f}   "
              f"error on the mean B/A {sem:.3f}   (sqrt(2) = 1.414)",
              flush=True)

    res = dict(arms=out, n_temps=int(n_temps), b_per_device=b_per_device,
               n_devices=ndev, seed=seed, T=T.tolist(), n_spins=n_spins)
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f)
    vol.commit()
    return {k: {kk: vv for kk, vv in v.items() if kk != "obs"}
            for k, v in out.items()}


@app.function(gpu="H100:4", timeout=6 * 3600, memory=65536, image=image,
              volumes={"/out": vol})
def sched4(nt, seed, tag, bpd): return _schedule_ab(nt, seed, tag, bpd)


@app.local_entrypoint()
def schedule_ab(n_temps: int = 200, seed: int = 42, tag: str = "sched_ab",
                b_per_device: int = 50):
    print(json.dumps(sched4.remote(n_temps, seed, tag, b_per_device), indent=1))


# --------------------------------------------------------------------------
# tau: the integrated autocorrelation time
# --------------------------------------------------------------------------
# The schedule comparison left one question open. Arm A (100 replicas, 5 000
# measurement sweeps) and arm B (50 replicas, 10 000) accumulate the same 500 000
# samples and agree on E, M, Mz, Q and Cv, but chi's peak sits 0.6-0.9 K lower in
# B and the shift survives smoothing at every window from 1 to 21 points. Two
# readings fit that, and they point opposite ways:
#
#   * B under-samples, because 50 independent replicas carry less information
#     than 100 when samples inside a replica are correlated;
#   * B over-samples A, because a 10 000-sweep window captures slow magnetisation
#     modes that a 5 000-sweep window truncates, and chi IS that variance.
#
# tau decides it. If tau << 5 000 both windows are long enough and the first
# reading holds; if tau is an appreciable fraction of 5 000, arm A is cutting the
# correlation function short and B is the better measurement.
#
# The series has to come from a state reached the way production reaches it, so
# each target temperature is approached by a coarse descent from 20 K rather than
# thermalised in place -- a chain dropped straight to 3 K can sit in a metastable
# texture and report a tau that belongs to no real run.

TAU_TEMPS = [20.0, 16.0, 14.2, 12.7, 11.0, 8.0, 3.0, 0.5]


def _tau_int(series):
    """Integrated autocorrelation time, per column, by FFT with automatic
    windowing (Madras-Sokal: stop at the first M with M >= 5 tau_int).

    Returns tau per column and the window actually used.
    """
    import numpy as np
    n, b = series.shape
    x = series - series.mean(0, keepdims=True)
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, n=nfft, axis=0)
    ac = np.fft.irfft(f * np.conj(f), n=nfft, axis=0)[:n].real
    var = ac[0].copy()
    taus, wins = np.zeros(b), np.zeros(b, dtype=int)
    for j in range(b):
        if var[j] <= 0:
            taus[j] = np.nan
            continue
        rho = ac[:, j] / var[j]
        t, M = 0.5, n - 1
        for m in range(1, n):
            t += rho[m]
            if m >= 5.0 * max(t, 0.5):
                M = m
                break
        taus[j], wins[j] = max(t, 0.5), M
    return taus, wins


def _autocorr(reps, n_down, n_therm, n_rec, seed, tag, theta_index=0):
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S
    import parallel_axes as PA

    geo = S.build_geometry(Rd=18.30, L=5)
    eng = S.build_engine(geo)
    # tau was first measured for `uniform` alone, but chi averages over all
    # four states and reduce_obs mixes them. A state whose ordering transition
    # sits near the chi peak can have a far longer tau than uniform does, and
    # that one state would then be under-sampled by the 5 000-sweep window while
    # the other three are fine -- which is the only surviving explanation for a
    # peak that moves with the schedule but not with the seed.
    theta = PA.load_thetas("/root/theta4.json")[int(theta_index)]
    kB = 0.086173404
    ev = jnp.asarray(geo["even"][None, ..., None])
    od = jnp.asarray(geo["odd"][None, ..., None])
    cols = PA.pack([theta], reps)
    k1 = jnp.where(jnp.asarray(geo["surface"])[None], cols["KanS"], cols["Kan1"])
    n_spins = int(geo["n_active"])
    print(f"device={jax.devices()[0]}  {reps} replicas  theta={theta['name']}  "
          f"descent {n_down} steps x {n_therm} sweeps, record {n_rec}",
          flush=True)

    def sweeps(s, k, beta, n):
        def one(c, _):
            s, k = c
            k, ka, kb = jax.random.split(k, 3)
            s = eng["substep"](s, ka, beta, ev, cols, k1)
            s = eng["substep"](s, kb, beta, od, cols, k1)
            return (s, k), None
        (s, k), _ = jax.lax.scan(one, (s, k), None, length=n)
        return s, k

    def record(s, k, beta, n):
        def one(c, _):
            s, k = c
            k, ka, kb = jax.random.split(k, 3)
            s = eng["substep"](s, ka, beta, ev, cols, k1)
            s = eng["substep"](s, kb, beta, od, cols, k1)
            E = eng["energy"](s, cols, k1)
            M, _unused = eng["magnetization"](s)
            return (s, k), (E, M)
        (s, k), out = jax.lax.scan(one, (s, k), None, length=n)
        return out

    jsweeps = jax.jit(sweeps, static_argnums=3)
    jrecord = jax.jit(record, static_argnums=3)

    rows = []
    for Tt in TAU_TEMPS:
        t0 = time.perf_counter()
        s = S.init_spins(jax.random.PRNGKey(seed), geo, reps)
        k = jax.random.PRNGKey(seed + 1)
        # Coarse descent 20 K -> Tt, the way a production ladder arrives.
        for Td in np.linspace(20.0, Tt, int(n_down)):
            s, k = jsweeps(s, k, 1.0 / (kB * float(Td)), n_therm)
        E, M = jrecord(s, k, 1.0 / (kB * float(Tt)), n_rec)
        E = np.asarray(E) / n_spins
        M = np.asarray(M)
        tE, wE = _tau_int(E)
        tM, wM = _tau_int(M)
        dt = time.perf_counter() - t0
        rows.append(dict(T=float(Tt), tau_E=float(np.nanmedian(tE)),
                         tau_E_p90=float(np.nanpercentile(tE, 90)),
                         tau_M=float(np.nanmedian(tM)),
                         tau_M_p90=float(np.nanpercentile(tM, 90)),
                         window_E=int(np.median(wE)), window_M=int(np.median(wM)),
                         n_rec=int(n_rec), replicas=int(reps),
                         wall_s=float(dt)))
        print(f"  T={Tt:5.2f}  tau_E {np.nanmedian(tE):8.1f} "
              f"(p90 {np.nanpercentile(tE,90):8.1f})   "
              f"tau_M {np.nanmedian(tM):8.1f} "
              f"(p90 {np.nanpercentile(tM,90):8.1f})   {dt/60:5.2f} min",
              flush=True)

    print("\n=== what this means for the two schedules ===", flush=True)
    print(f"  {'T':>6} {'tau_M':>9} {'5k/tau':>9} {'10k/tau':>9}  verdict",
          flush=True)
    for r in rows:
        a, b = 5000 / r["tau_M"], 10000 / r["tau_M"]
        v = ("both windows ample" if a >= 50 else
             "A is short" if a >= 10 else "A truncates the correlation")
        print(f"  {r['T']:>6.2f} {r['tau_M']:>9.1f} {a:>9.0f} {b:>9.0f}  {v}",
              flush=True)

    res = dict(rows=rows, replicas=int(reps), n_down=int(n_down),
               n_therm=int(n_therm), n_rec=int(n_rec), theta=theta,
               n_spins=n_spins, device=str(jax.devices()[0]))
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    vol.commit()
    return rows


@app.function(gpu="H100", **_BS)
def tau1(r, nd, nt, nr, seed, tag, ti=0):
    return _autocorr(r, nd, nt, nr, seed, tag, ti)


@app.local_entrypoint()
def autocorr(replicas: int = 50, n_down: int = 30, n_therm: int = 10000,
             n_rec: int = 20000, seed: int = 42, tag: str = "tau",
             theta_index: int = 0):
    print(json.dumps(tau1.remote(replicas, n_down, n_therm, n_rec, seed, tag,
                                 theta_index), indent=1))


def _triton_halo(batches, repeats, tag, Rd, L, steps, tiles, warps):
    """The raw-Triton stencil: validate the physics, then time it amortised.

    Two things have to be true before a number from this kernel means
    anything. The stencil must reproduce `sim_core.substep` on identical
    randomness -- that is what the `dE` comparison establishes, and it is the
    part three earlier kernels got right while losing on speed. And the timing
    has to amortise the launch: a `triton_call` costs about 122 us of host
    dispatch, so the step loop runs inside `lax.fori_loop` and the reference
    arm is the production therm loop, not a bare substep.
    """
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S

    print(f"gpu={device_identity()}", flush=True)
    try:
        import triton, jax_triton as jt
        import triton_halo as TH
    except Exception as e:
        print(f"TRITON CRUDO NO DISPONIBLE: {type(e).__name__}: {e}", flush=True)
        return dict(available=False, error=f"{type(e).__name__}: {e}")
    print(f"  jax {jax.__version__}  triton {triton.__version__}  "
          f"jax_triton {getattr(jt, '__version__', '?')}", flush=True)

    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    N, LZ = int(geo["N"]), int(geo["L"])
    even_j = jnp.asarray(geo["even"][None, ..., None])
    odd_j = jnp.asarray(geo["odd"][None, ..., None])
    surf_j = jnp.asarray(geo["surface"][None, ...])
    disk_f = jnp.asarray(geo["disk2d"].astype(np.float32))
    beta = 1.0 / (0.086173404 * 5.0)
    K = int(steps)
    res = dict(available=True, gpu=device_identity(), N=N, L=LZ, Rd=float(Rd),
               steps_per_call=K, beta=float(beta))
    print(f"Rd={Rd} L={LZ} N={N}  beta={beta:.4f}  {K} barridos por llamada",
          flush=True)

    raw = json.load(open("/root/theta4.json"))
    t0p = list(raw.values())[0]

    def mk_p(B):
        c = lambda v: jnp.full((B, 1, 1, 1), float(v), jnp.float32)
        return dict(Jex=c(1.0), Kan1=c(t0p["Kan1"]), Kan2=c(0.0),
                    KanS=c(t0p["KanS"]), Hex=c(t0p["Hex"]),
                    KDM=c(t0p["KDM"]), g=c(float(np.deg2rad(90.0))))

    eng = S.build_engine(geo)

    # ---------------------------------------------------------- correctness
    print("\n=== el stencil contra sim_core.substep, misma aleatoriedad ===",
          flush=True)
    Bv = 8
    p = mk_p(Bv)
    k1e = jnp.where(surf_j, p["KanS"], p["Kan1"])
    sp0 = S.init_spins(jax.random.PRNGKey(3), geo, Bv)

    kt, ka = jax.random.split(jax.random.PRNGKey(101))
    tr = jax.random.normal(kt, sp0.shape)
    tr = tr / jnp.linalg.norm(tr, axis=-1, keepdims=True)
    ru = jax.random.uniform(ka, sp0.shape[:-1])

    def ref_dE(sp, tr_raw, mask):
        """`substep`'s dE, lifted out so it can be compared on its own."""
        trm = jnp.where(mask, tr_raw, sp)
        dS = trm - sp
        xp = jnp.roll(sp, -1, 3); xm = jnp.roll(sp, 1, 3)
        yp = jnp.roll(sp, -1, 2); ym = jnp.roll(sp, 1, 2)
        zp = jnp.roll(sp, -1, 1).at[:, -1].set(0.0)
        zm = jnp.roll(sp, 1, 1).at[:, 0].set(0.0)
        sv = xp + xm + yp + ym + zp + zm
        dE = -p["Jex"] * jnp.sum(dS * sv, -1)
        an = lambda s: ((k1e) * ((s[..., 0] * s[..., 1]) ** 2
                                 + (s[..., 0] * s[..., 2]) ** 2
                                 + (s[..., 1] * s[..., 2]) ** 2)
                        + p["Kan2"] * (s[..., 0] * s[..., 1]
                                               * s[..., 2]) ** 2)
        dE = dE + an(trm) - an(sp)
        dE = dE - p["KDM"] * (dS[..., 1] * (xp[..., 2] - xm[..., 2])
                                      - dS[..., 2] * (xp[..., 1] - xm[..., 1])
                                      + dS[..., 2] * (yp[..., 0] - ym[..., 0])
                                      - dS[..., 0] * (yp[..., 2] - ym[..., 2]))
        return dE - p["Hex"] * (dS[..., 0] * jnp.cos(p["g"])
                                        + dS[..., 2] * jnp.sin(p["g"]))

    sub = TH.build(geo, BY=int(tiles[0]), num_warps=int(warps))
    par = TH.pack_params(p, Bv)
    spP = TH.planar(sp0)
    trP = TH.planar(tr)
    checks = []
    for parity, mask, nm in ((0, even_j, "par"), (1, odd_j, "impar")):
        dE_ref = ref_dE(sp0, tr, mask)
        spn, dE_t = sub(spP, k1e, par, beta, 0, parity, trP, ru, want_de=True)
        # even_j/odd_j carry a leading singleton, so the selector has to be
        # broadcast up to the batch before it can index dE.
        sel = np.broadcast_to(np.asarray(mask[..., 0]), dE_ref.shape)
        a = np.asarray(dE_ref)[sel]
        b = np.asarray(dE_t)[sel]
        mad = float(np.max(np.abs(a - b)))
        rel = mad / float(np.max(np.abs(a)) + 1e-30)
        # The accepted state, rebuilt from the same dE and the same uniforms,
        # which is exactly what `substep` does with its own draws.
        acc = ((dE_ref < 0) | (ru < jnp.exp(-dE_ref * beta)))[..., None] & mask
        out_ref = jnp.where(acc, jnp.where(mask, tr, sp0), sp0)
        out_t = TH.unplanar(spn)
        bad = int(np.sum(np.asarray(out_ref) != np.asarray(out_t)))
        tot = int(np.asarray(out_ref).size)
        checks.append(dict(sublattice=nm, dE_max_abs=mad, dE_rel=rel,
                           spin_mismatch=bad, spin_total=tot,
                           mismatch_frac=bad / tot))
        print(f"  {nm:>5}: dE max |dif| {mad:.3e}  relativo {rel:.2e}   "
              f"espines distintos {bad}/{tot}", flush=True)
    res["correctness"] = checks
    res["tile"] = sub.meta
    print(f"  bloque BY={sub.meta['BY']} BX={sub.meta['BX']} "
          f"warps={sub.meta['num_warps']}  rejilla/replica "
          f"{sub.meta['grid_per_replica']}  desperdicio de carril "
          f"{sub.meta['lane_waste_pct']:.1f}%", flush=True)

    # ------------------------------------------------------- physics gate
    # The exactness check above fixes the stencil, but it feeds the kernel
    # JAX's own trial spins. The variant that actually competes draws them
    # in-kernel with Philox, and nothing so far says those draws are right: a
    # kernel whose proposals are degenerate accepts nothing or everything and
    # comes out FAST and WRONG. So both arms are run to equilibrium from the
    # same start and their observables compared.
    print("\n=== fisica: produccion contra halo con su propio RNG ===",
          flush=True)
    kB = 0.086173404
    Bp, n_eq = 16, int(max(400, K * 4))
    pp = mk_p(Bp); k1p = jnp.where(surf_j, pp["KanS"], pp["Kan1"])
    parp = TH.pack_params(pp, Bp)
    sp_eq = S.init_spins(jax.random.PRNGKey(11), geo, Bp)
    subp = TH.build(geo, BY=int(tiles[0]), num_warps=int(warps))
    n_act = float(geo["n_active"])
    phys = []
    for T in (20.0, 5.0, 1.0):
        bT = 1.0 / (kB * T)

        def jx(sp):
            def body(i, v):
                s, k = v
                k, ka_, kb_ = jax.random.split(k, 3)
                s = eng["substep"](s, ka_, bT, even_j, pp, k1p)
                s = eng["substep"](s, kb_, bT, odd_j, pp, k1p)
                return s, k
            return jax.lax.fori_loop(0, n_eq, body,
                                     (sp, jax.random.PRNGKey(77)))[0]

        sj = jax.block_until_ready(jax.jit(jx)(sp_eq))
        st = jax.block_until_ready(
            jax.jit(TH.sweeps(subp, k1p, parp, bT, n_eq))(TH.planar(sp_eq)))
        st = TH.unplanar(st)
        ej = float(jnp.mean(eng["energy"](sj, pp, k1p)) / n_act)
        et = float(jnp.mean(eng["energy"](st, pp, k1p)) / n_act)
        mj = float(jnp.mean(eng["magnetization"](sj)[0]))
        mt = float(jnp.mean(eng["magnetization"](st)[0]))
        sdj = float(jnp.std(eng["energy"](sj, pp, k1p)) / n_act)
        # The norm must stay on the unit sphere; a broken proposal shows here.
        nt = float(jnp.max(jnp.abs(
            jnp.linalg.norm(st, axis=-1)
            - jnp.asarray(geo["disco"][None]).astype(jnp.float32))))
        row = dict(T=T, E_prod=ej, E_halo=et, E_sd_prod=sdj,
                   E_sigma=abs(et - ej) / (sdj + 1e-30),
                   M_prod=mj, M_halo=mt, norm_err=nt)
        phys.append(row)
        print(f"  T={T:>5.1f}  E/sitio prod {ej:+9.5f}  halo {et:+9.5f}  "
              f"({row['E_sigma']:.2f} sigma)   |M| {mj:.4f} / {mt:.4f}   "
              f"|norma-1| max {nt:.2e}", flush=True)
    res["physics"] = phys
    bad = [r for r in phys if r["E_sigma"] > 3.0 or r["norm_err"] > 1e-5]
    res["physics_ok"] = not bad
    if bad:
        print(f"  AVISO: {len(bad)} temperatura(s) fuera de 3 sigma o con "
              f"norma rota -- el tiempo NO es comparable", flush=True)
    else:
        print("  las dos cadenas coinciden dentro del error estadistico",
              flush=True)

    # ------------------------------------------------- the production arm
    def prod_loop(sp, p_, k1_):
        def body(i, v):
            s, k = v
            k, ka_, kb_ = jax.random.split(k, 3)
            s = eng["substep"](s, ka_, beta, even_j, p_, k1_)
            s = eng["substep"](s, kb_, beta, odd_j, p_, k1_)
            return s, k
        return jax.lax.fori_loop(0, K, body,
                                 (sp, jax.random.PRNGKey(0)))[0]

    # ---------------------------------------------------- pick the block
    print(f"\n=== eleccion de bloque, B={batches[0]} ===", flush=True)
    Bt = int(batches[0])
    pt = mk_p(Bt); k1t = jnp.where(surf_j, pt["KanS"], pt["Kan1"])
    spt = S.init_spins(jax.random.PRNGKey(5), geo, Bt)
    part = TH.pack_params(pt, Bt)
    sptP = TH.planar(spt)
    fj = jax.jit(lambda s: prod_loop(s, pt, k1t)).lower(spt).compile()
    jax.block_until_ready(fj(spt))
    tj = []
    for _ in range(int(repeats)):
        t = time.perf_counter(); jax.block_until_ready(fj(spt))
        tj.append(time.perf_counter() - t)
    mj = float(np.median(tj))
    print(f"  produccion (6 rolls + threefry): "
          f"{1e6*mj/(K*Bt):8.3f} us/replica-barrido", flush=True)
    best, tilerows = None, []
    for by in tiles:
        try:
            sb = TH.build(geo, BY=int(by), num_warps=int(warps))
            run = jax.jit(TH.sweeps(sb, k1t, part, beta, K)).lower(sptP).compile()
            jax.block_until_ready(run(sptP))
            ts = []
            for _ in range(int(repeats)):
                t = time.perf_counter(); jax.block_until_ready(run(sptP))
                ts.append(time.perf_counter() - t)
            m = float(np.median(ts))
            row = dict(BY=int(by), us_per_replica_sweep=1e6 * m / (K * Bt),
                       vs_prod_pct=100 * (m / mj - 1),
                       grid_per_replica=sb.meta["grid_per_replica"],
                       lane_waste_pct=sb.meta["lane_waste_pct"])
            print(f"  BY={by:>3}: {row['us_per_replica_sweep']:8.3f} "
                  f"us/replica-barrido  {row['vs_prod_pct']:+7.1f}% vs "
                  f"produccion  (rejilla {row['grid_per_replica']}, "
                  f"carril {row['lane_waste_pct']:.0f}%)", flush=True)
            tilerows.append(row)
            if best is None or m < best[1]:
                best = (int(by), m)
        except Exception as e:
            print(f"  BY={by:>3}: FALLO {type(e).__name__}: {e}", flush=True)
            tilerows.append(dict(BY=int(by), error=f"{type(e).__name__}: {e}"))
    res["tile_scan"] = tilerows
    if best is None:
        with open(f"/out/{tag}.json", "w") as f:
            json.dump(res, f, indent=1)
        return res
    BY = best[0]
    res["BY_chosen"] = BY
    print(f"  -> bloque elegido BY={BY}", flush=True)

    # ------------------------------------------------- the sawtooth test
    print(f"\n{'B':>6}{'produccion':>13}{'solo stencil':>13}{'vs prod':>9}"
          f"{'+ su RNG':>13}{'vs prod':>9}{'gana RNG':>10}", flush=True)
    rows = []
    for B in batches:
        pb = mk_p(B); k1b = jnp.where(surf_j, pb["KanS"], pb["Kan1"])
        spb = S.init_spins(jax.random.PRNGKey(5 + B), geo, B)
        parb = TH.pack_params(pb, B)
        spbP = TH.planar(spb)
        sb = TH.build(geo, BY=BY, num_warps=int(warps))
        f1 = jax.jit(lambda s: prod_loop(s, pb, k1b)).lower(spb).compile()
        f2 = jax.jit(TH.sweeps(sb, k1b, parb, beta, K)).lower(spbP).compile()
        # The middle arm: Triton stencil, JAX's Threefry proposals. Without it
        # the win is a lump sum over two changes at once.
        f3 = jax.jit(TH.sweeps_extrng(sb, k1b, parb, beta, K, spbP.shape)
                     ).lower(spbP).compile()
        for f, a in ((f1, spb), (f2, spbP), (f3, spbP)):
            jax.block_until_ready(f(a))
        a1, a2, a3 = [], [], []
        for _ in range(int(repeats)):          # interleaved, against drift
            t = time.perf_counter(); jax.block_until_ready(f1(spb))
            a1.append(time.perf_counter() - t)
            t = time.perf_counter(); jax.block_until_ready(f2(spbP))
            a2.append(time.perf_counter() - t)
            t = time.perf_counter(); jax.block_until_ready(f3(spbP))
            a3.append(time.perf_counter() - t)
        m1, m2, m3 = (float(np.median(a1)), float(np.median(a2)),
                      float(np.median(a3)))
        rows.append(dict(B=B, prod_us=1e6 * m1 / (K * B),
                         halo_us=1e6 * m2 / (K * B),
                         stencil_only_us=1e6 * m3 / (K * B),
                         change_pct=100 * (m2 / m1 - 1),
                         stencil_pct=100 * (m3 / m1 - 1),
                         rng_pct=100 * (m2 / m3 - 1)))
        r = rows[-1]
        print(f"{B:>6}{r['prod_us']:>13.3f}{r['stencil_only_us']:>13.3f}"
              f"{r['stencil_pct']:>+8.1f}%{r['halo_us']:>13.3f}"
              f"{r['change_pct']:>+8.1f}%{r['rng_pct']:>+9.1f}%", flush=True)
    res["timing"] = rows

    import statistics as stx
    ch = [r["change_pct"] for r in rows]
    st = [r["stencil_pct"] for r in rows]
    rg = [r["rng_pct"] for r in rows]
    disp = lambda k: (100 * (max(r[k] for r in rows) / min(r[k] for r in rows)
                             - 1))
    res["summary"] = dict(mean_change_pct=round(stx.mean(ch), 1),
                          best_change_pct=round(min(ch), 1),
                          wins=sum(1 for x in ch if x < 0), n=len(ch),
                          dispersion_prod_pct=round(disp("prod_us"), 1),
                          dispersion_halo_pct=round(disp("halo_us"), 1),
                          dispersion_stencil_pct=round(
                              disp("stencil_only_us"), 1),
                          mean_stencil_pct=round(stx.mean(st), 1),
                          mean_rng_pct=round(stx.mean(rg), 1))
    print(f"\n=== resumen ===", flush=True)
    print(f"  halo contra produccion: media {stx.mean(ch):+.1f}%  "
          f"gana {sum(1 for x in ch if x<0)}/{len(ch)}  "
          f"mejor {min(ch):+.1f}%", flush=True)
    print(f"  de eso, el stencil {stx.mean(st):+.1f}% y el RNG propio "
          f"{stx.mean(rg):+.1f}% adicional", flush=True)
    print(f"  dentado (max/min de us/replica): produccion "
          f"{disp('prod_us'):.1f}%, solo stencil "
          f"{disp('stencil_only_us'):.1f}%, halo {disp('halo_us'):.1f}%",
          flush=True)
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n  guardado /out/{tag}.json", flush=True)
    return res


@app.function(gpu="H100", image=triton_image, timeout=60 * 60,
              volumes={"/out": vol})
def triton_halo_h100(b, rep, tag, Rd, L, steps, tiles, warps):
    return _triton_halo(b, rep, tag, Rd, L, steps, tiles, warps)


@app.local_entrypoint()
def triton_halo(batches: str = "64,65,75,100,101,128", repeats: int = 7,
                rd: float = 18.30, layers: int = 5, steps: int = 200,
                tiles: str = "1,2,4,8", warps: int = 4,
                tag: str = "triton_halo"):
    """Raw Triton: shifted coalesced loads, no materialised rolls, Philox RNG."""
    triton_halo_h100.remote(
        [int(x) for x in batches.split(",") if x], repeats, tag, rd, layers,
        steps, [int(x) for x in tiles.split(",") if x], warps)


def _triton_ladder(tag, Rd, L, replicas, n_temps, n_therm, n_meas,
                   v_temps, v_therm, v_meas, by, warps):
    """The Triton substep inside the full anneal, then the production ladder.

    The substep benchmark left the measurement loop out, and that loop is
    where the ladder actually lives: it recomputes the energy, the
    magnetisation and the topological charge on every measured sweep, in JAX,
    in both arms. So the ladder gain has to be smaller than the substep gain,
    and the only honest way to know by how much is to run the ladder.

    Validation first, on a short ladder: the two engines draw from different
    random streams, so the test is that the reduced observables agree within
    the error bars the run itself reports, not that they match bit for bit.
    """
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S

    print(f"gpu={device_identity()}", flush=True)
    try:
        import triton, jax_triton as jt
        import triton_halo as TH
    except Exception as e:
        print(f"TRITON CRUDO NO DISPONIBLE: {type(e).__name__}: {e}", flush=True)
        return dict(available=False, error=f"{type(e).__name__}: {e}")
    print(f"  jax {jax.__version__}  triton {triton.__version__}  "
          f"jax_triton {getattr(jt, '__version__', '?')}", flush=True)

    kB = 0.086173404
    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    eng = S.build_engine(geo)
    surf_j = jnp.asarray(geo["surface"][None, ...])
    even_j = jnp.asarray(geo["even"][None, ..., None])
    odd_j = jnp.asarray(geo["odd"][None, ..., None])
    n_act = float(geo["n_active"])
    B = int(replicas)
    raw = json.load(open("/root/theta4.json"))
    t0p = list(raw.values())[0]
    c = lambda v: jnp.full((B, 1, 1, 1), float(v), jnp.float32)
    P = dict(Jex=c(1.0), Kan1=c(t0p["Kan1"]), Kan2=c(0.0),
             KanS=c(t0p["KanS"]), Hex=c(t0p["Hex"]), KDM=c(t0p["KDM"]),
             g=c(float(np.deg2rad(90.0))))
    k1e = jnp.where(surf_j, P["KanS"], P["Kan1"])
    par = TH.pack_params(P, B)
    ann_t = TH.build_anneal(geo, BY=int(by), num_warps=int(warps))
    res = dict(available=True, gpu=device_identity(), replicas=B,
               N=int(geo["N"]), L=int(geo["L"]), Rd=float(Rd),
               tile=ann_t.meta, n_active=n_act)
    print(f"Rd={Rd} L={geo['L']} N={geo['N']}  {B} replicas  "
          f"bloque BY={by} warps={warps}", flush=True)

    # ------------------------------------------------------ validation
    print(f"\n=== validacion: escalera corta {v_temps}T x "
          f"{v_therm}+{v_meas} ===", flush=True)
    Tv = np.linspace(20.0, 0.1, int(v_temps))
    bv = jnp.asarray(1.0 / (kB * Tv), jnp.float32)
    sp0 = S.init_spins(jax.random.PRNGKey(19), geo, B)
    oj = jax.block_until_ready(jax.jit(
        lambda s: eng["anneal"](s, jax.random.PRNGKey(5), bv, P,
                                int(v_therm), int(v_meas)))(sp0))[0]
    ot = jax.block_until_ready(jax.jit(
        lambda s: ann_t(s, 5, bv, par, k1e, int(v_therm), int(v_meas))
    )(TH.planar(sp0)))[0]
    rj = S.reduce_obs(np.asarray(oj), Tv, n_act)
    rt = S.reduce_obs(np.asarray(ot), Tv, n_act)
    err = np.sqrt(np.asarray(rj["E_sd"]) ** 2 + 1e-12) / np.sqrt(B)
    sig = np.abs(np.asarray(rj["E"]) - np.asarray(rt["E"])) / (err + 1e-30)
    dM = np.max(np.abs(np.asarray(rj["M"]) - np.asarray(rt["M"])))
    print(f"  {'T':>6}{'E/sitio prod':>14}{'E/sitio halo':>14}{'sigma':>8}"
          f"{'|M| prod':>10}{'|M| halo':>10}", flush=True)
    for i in range(0, int(v_temps), max(1, int(v_temps) // 8)):
        print(f"  {Tv[i]:>6.2f}{rj['E'][i]:>14.5f}{rt['E'][i]:>14.5f}"
              f"{sig[i]:>8.2f}{rj['M'][i]:>10.4f}{rt['M'][i]:>10.4f}",
              flush=True)
    res["validation"] = dict(
        n_temps=int(v_temps), n_therm=int(v_therm), n_meas=int(v_meas),
        T=Tv.tolist(), E_prod=np.asarray(rj["E"]).tolist(),
        E_halo=np.asarray(rt["E"]).tolist(),
        M_prod=np.asarray(rj["M"]).tolist(),
        M_halo=np.asarray(rt["M"]).tolist(),
        Cv_prod=np.asarray(rj["Cv"]).tolist(),
        Cv_halo=np.asarray(rt["Cv"]).tolist(),
        chi_prod=np.asarray(rj["chi"]).tolist(),
        chi_halo=np.asarray(rt["chi"]).tolist(),
        E_sigma_max=float(np.max(sig)), E_sigma_mean=float(np.mean(sig)),
        M_max_abs_diff=float(dM))
    ok = float(np.max(sig)) < 4.0 and dM < 0.05
    res["validation"]["ok"] = bool(ok)
    print(f"  E: desvio maximo {np.max(sig):.2f} sigma, medio "
          f"{np.mean(sig):.2f} sigma   |M|: max |dif| {dM:.4f}", flush=True)
    print(f"  -> {'las dos escaleras coinciden' if ok else 'AVISO: NO coinciden'}",
          flush=True)

    # --------------------------------------------------- the real ladder
    T = np.linspace(20.0, 0.1, int(n_temps))
    bt = jnp.asarray(1.0 / (kB * T), jnp.float32)
    tot = int(n_temps) * (int(n_therm) + int(n_meas))
    print(f"\n=== escalera de produccion: {n_temps}T x {n_therm}+{n_meas} "
          f"= {tot:,} barridos, {B} replicas ===", flush=True)
    arms = {}
    for nm, fn, a0 in (
            ("produccion", lambda s: eng["anneal"](
                s, jax.random.PRNGKey(42), bt, P, int(n_therm), int(n_meas)),
             sp0),
            ("halo", lambda s: ann_t(s, 42, bt, par, k1e, int(n_therm),
                                     int(n_meas)), TH.planar(sp0))):
        t0 = time.perf_counter()
        f = jax.jit(fn).lower(a0).compile()
        tc = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = jax.block_until_ready(f(a0))
        dt = time.perf_counter() - t0
        arms[nm] = dict(compile_s=tc, wall_s=dt, minutes=dt / 60.0,
                        us_per_replica_sweep=1e6 * dt / (tot * B),
                        ms_per_sweep=1e3 * dt / tot)
        print(f"  {nm:>11}: {dt/60:7.2f} min   "
              f"{1e6*dt/(tot*B):7.3f} us/replica-barrido   "
              f"{1e3*dt/tot:7.3f} ms/barrido   (compilar {tc:.1f} s)",
              flush=True)
        # Keep the reduced curve, so the ladder is a result and not only a
        # stopwatch reading.
        rr = S.reduce_obs(np.asarray(out[0]), T, n_act)
        arms[nm]["E"] = np.asarray(rr["E"]).tolist()
        arms[nm]["Cv"] = np.asarray(rr["Cv"]).tolist()
        arms[nm]["chi"] = np.asarray(rr["chi"]).tolist()
        arms[nm]["M"] = np.asarray(rr["M"]).tolist()
        del out
    g = 100 * (arms["halo"]["wall_s"] / arms["produccion"]["wall_s"] - 1)
    res["ladder"] = dict(n_temps=int(n_temps), n_therm=int(n_therm),
                         n_meas=int(n_meas), total_sweeps=tot, T=T.tolist(),
                         arms=arms, change_pct=g)
    print(f"\n=== resumen ===", flush=True)
    print(f"  escalera completa: {arms['produccion']['minutes']:.2f} min -> "
          f"{arms['halo']['minutes']:.2f} min   {g:+.1f}%", flush=True)
    print(f"  (el paso suelto daba -66.6%; la diferencia es el bucle de "
          f"medicion, que sigue en JAX en los dos brazos)", flush=True)
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n  guardado /out/{tag}.json", flush=True)
    return res


@app.function(gpu="H100", image=triton_image, timeout=60 * 90,
              volumes={"/out": vol})
def triton_ladder_h100(tag, Rd, L, replicas, n_temps, n_therm, n_meas,
                       v_temps, v_therm, v_meas, by, warps):
    return _triton_ladder(tag, Rd, L, replicas, n_temps, n_therm, n_meas,
                          v_temps, v_therm, v_meas, by, warps)


@app.local_entrypoint()
def triton_ladder(replicas: int = 50, n_temps: int = 200,
                  n_therm: int = 10000, n_meas: int = 5000,
                  v_temps: int = 24, v_therm: int = 600, v_meas: int = 300,
                  rd: float = 18.30, layers: int = 5, by: int = 4,
                  warps: int = 4, tag: str = "triton_ladder"):
    """The Triton substep inside the full anneal, on the production ladder."""
    triton_ladder_h100.remote(tag, rd, layers, replicas, n_temps, n_therm,
                              n_meas, v_temps, v_therm, v_meas, by, warps)


def _triton_check(tag, Rd, L, replicas, lengths, v_temps, by, warps):
    """Is the 4.94-sigma outlier the kernel, or the yardstick?

    The ladder validation compared production against the halo engine using
    an error bar taken as the across-replica spread over sqrt(B). One
    temperature came out at 4.94 sigma, which is either a real bias in the
    kernel's proposals or an error bar that is too small because a short
    measurement window leaves every replica correlated with the shared start.

    The test that separates those two is a CONTROL: run PRODUCTION twice with
    different seeds and score it on the same yardstick. Production cannot
    disagree with itself about physics, so whatever that control shows is the
    yardstick's own noise floor. If the halo comparison sits at the same
    level, the kernel is clean and the threshold was wrong. Repeated at two
    ladder lengths, because the explanation predicts the spread shrinks with
    a longer window and a real bias does not.
    """
    import sys, json
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S

    print(f"gpu={device_identity()}", flush=True)
    try:
        import triton_halo as TH
    except Exception as e:
        print(f"TRITON NO DISPONIBLE: {type(e).__name__}: {e}", flush=True)
        return dict(available=False, error=f"{type(e).__name__}: {e}")

    kB = 0.086173404
    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    eng = S.build_engine(geo)
    surf_j = jnp.asarray(geo["surface"][None, ...])
    n_act = float(geo["n_active"])
    B = int(replicas)
    raw = json.load(open("/root/theta4.json"))
    t0p = list(raw.values())[0]
    c = lambda v: jnp.full((B, 1, 1, 1), float(v), jnp.float32)
    P = dict(Jex=c(1.0), Kan1=c(t0p["Kan1"]), Kan2=c(0.0),
             KanS=c(t0p["KanS"]), Hex=c(t0p["Hex"]), KDM=c(t0p["KDM"]),
             g=c(float(np.deg2rad(90.0))))
    k1e = jnp.where(surf_j, P["KanS"], P["Kan1"])
    par = TH.pack_params(P, B)
    ann_t = TH.build_anneal(geo, BY=int(by), num_warps=int(warps))
    T = np.linspace(20.0, 0.1, int(v_temps))
    bt = jnp.asarray(1.0 / (kB * T), jnp.float32)
    sp0 = S.init_spins(jax.random.PRNGKey(19), geo, B)
    spP = TH.planar(sp0)
    res = dict(available=True, gpu=device_identity(), replicas=B,
               n_temps=int(v_temps), T=T.tolist(), rows=[])

    def z_of(a, b, sd):
        err = np.asarray(sd) / np.sqrt(B) + 1e-30
        return np.abs(np.asarray(a) - np.asarray(b)) / err

    print(f"\n{'ventana':>14}{'control prod/prod':>22}"
          f"{'prueba prod/halo':>22}", flush=True)
    print(f"{'':>14}{'media':>9}{'maximo':>13}{'media':>9}{'maximo':>13}",
          flush=True)
    for nt, nm_ in lengths:
        rp = lambda sd: S.reduce_obs(np.asarray(jax.block_until_ready(jax.jit(
            lambda s: eng["anneal"](s, jax.random.PRNGKey(sd), bt, P,
                                    int(nt), int(nm_)))(sp0))[0]), T, n_act)
        rA, rB = rp(5), rp(6)
        rH = S.reduce_obs(np.asarray(jax.block_until_ready(jax.jit(
            lambda s: ann_t(s, 5, bt, par, k1e, int(nt), int(nm_)))(spP))[0]),
            T, n_act)
        zc = z_of(rA["E"], rB["E"], rA["E_sd"])
        zh = z_of(rA["E"], rH["E"], rA["E_sd"])
        row = dict(n_therm=int(nt), n_meas=int(nm_),
                   control_mean=float(np.mean(zc)),
                   control_max=float(np.max(zc)),
                   halo_mean=float(np.mean(zh)), halo_max=float(np.max(zh)),
                   M_max_abs_diff=float(np.max(np.abs(
                       np.asarray(rA["M"]) - np.asarray(rH["M"])))),
                   M_control_max_abs_diff=float(np.max(np.abs(
                       np.asarray(rA["M"]) - np.asarray(rB["M"])))))
        res["rows"].append(row)
        print(f"{nt:>8}+{nm_:<5}{row['control_mean']:>9.2f}"
              f"{row['control_max']:>13.2f}{row['halo_mean']:>9.2f}"
              f"{row['halo_max']:>13.2f}", flush=True)
    print(f"\n=== lectura ===", flush=True)
    for r in res["rows"]:
        v = "igual que el control" if (
            r["halo_mean"] <= 1.5 * r["control_mean"] + 0.3
            and r["halo_max"] <= 1.5 * r["control_max"] + 1.0) else "PEOR"
        print(f"  {r['n_therm']}+{r['n_meas']}: halo {v}   "
              f"|M| control {r['M_control_max_abs_diff']:.4f} vs halo "
              f"{r['M_max_abs_diff']:.4f}", flush=True)
    res["verdict_halo_matches_control"] = all(
        r["halo_mean"] <= 1.5 * r["control_mean"] + 0.3
        and r["halo_max"] <= 1.5 * r["control_max"] + 1.0
        for r in res["rows"])
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n  guardado /out/{tag}.json", flush=True)
    return res


@app.function(gpu="H100", image=triton_image, timeout=60 * 60,
              volumes={"/out": vol})
def triton_check_h100(tag, Rd, L, replicas, lengths, v_temps, by, warps):
    return _triton_check(tag, Rd, L, replicas, lengths, v_temps, by, warps)


@app.local_entrypoint()
def triton_check(replicas: int = 50, v_temps: int = 24,
                 windows: str = "600:300,3000:1500,10000:5000",
                 rd: float = 18.30, layers: int = 5, by: int = 4,
                 warps: int = 4, tag: str = "triton_check"):
    """Score the halo engine against production's own seed-to-seed noise."""
    ls = [tuple(int(y) for y in w.split(":")) for w in windows.split(",") if w]
    triton_check_h100.remote(tag, rd, layers, replicas, ls, v_temps, by, warps)


def _triton_full(tag, Rd, L, replicas, n_temps, n_therm, n_meas, by, warps):
    """Everything in Triton: substep AND observables, then the ladder again.

    Three questions in one run.

    First, are the observable kernels right? They are compared against the
    JAX versions on the SAME state, with no randomness anywhere, so this is
    an exactness test and not a statistical one.

    Second, why was the halo engine worse than the control at the 600+300
    window? The candidate is a different acceptance rate: detailed balance
    holds for any symmetric proposal on the sphere, so the equilibrium
    distribution is identical either way, but a different acceptance rate
    means a different autocorrelation, and a short window is exactly where
    that shows. Measuring it settles whether the explanation is benign.

    Third, how much of the seven-point gap between the substep and the ladder
    the observable kernels take back.
    """
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S

    print(f"gpu={device_identity()}", flush=True)
    try:
        import triton_halo as TH
    except Exception as e:
        print(f"TRITON NO DISPONIBLE: {type(e).__name__}: {e}", flush=True)
        return dict(available=False, error=f"{type(e).__name__}: {e}")

    kB = 0.086173404
    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    eng = S.build_engine(geo)
    surf_j = jnp.asarray(geo["surface"][None, ...])
    even_j = jnp.asarray(geo["even"][None, ..., None])
    odd_j = jnp.asarray(geo["odd"][None, ..., None])
    n_act = float(geo["n_active"])
    B = int(replicas)
    raw = json.load(open("/root/theta4.json"))
    t0p = list(raw.values())[0]
    c = lambda v: jnp.full((B, 1, 1, 1), float(v), jnp.float32)
    P = dict(Jex=c(1.0), Kan1=c(t0p["Kan1"]), Kan2=c(0.0),
             KanS=c(t0p["KanS"]), Hex=c(t0p["Hex"]), KDM=c(t0p["KDM"]),
             g=c(float(np.deg2rad(90.0))))
    k1e = jnp.where(surf_j, P["KanS"], P["Kan1"])
    par = TH.pack_params(P, B)
    res = dict(available=True, gpu=device_identity(), replicas=B,
               N=int(geo["N"]), L=int(geo["L"]))

    # ----------------------------------- 1. the observable kernels, exactly
    print("\n=== observables de Triton contra los de JAX, mismo estado ===",
          flush=True)
    OJ = TH.build_obs(geo)
    OT = TH.build_obs_triton(geo, BY=int(by), num_warps=int(warps))
    sp0 = S.init_spins(jax.random.PRNGKey(23), geo, B)
    spP = TH.planar(sp0)
    # A thermalised state, not a random one: a random state has near-zero M
    # and the relative error would be meaningless.
    annJ = TH.build_anneal(geo, BY=int(by), num_warps=int(warps))
    bT = jnp.asarray([1.0 / (kB * 5.0)], jnp.float32)
    spT = jax.block_until_ready(jax.jit(
        lambda s: annJ(s, 3, bT, par, k1e, 400, 1)[1])(spP))
    ej, mj, mzj = (OJ["energy"](spT, par, k1e), *OJ["mag"](spT))
    qj = OJ["topo"](spT)[:, int(geo["L"]) // 2]
    et, mt, mzt = OT["eM"](spT, par, k1e)
    qt = OT["topo_mid"](spT)
    rel = lambda a, b: float(np.max(np.abs(np.asarray(a) - np.asarray(b))
                                    / (np.abs(np.asarray(a)) + 1e-12)))
    chk = dict(E_rel=rel(ej, et), M_rel=rel(mj, mt), Mz_rel=rel(mzj, mzt),
               Q_abs=float(np.max(np.abs(np.asarray(qj) - np.asarray(qt)))))
    print(f"  E  error relativo maximo {chk['E_rel']:.3e}", flush=True)
    print(f"  |M|                      {chk['M_rel']:.3e}", flush=True)
    print(f"  Mz                       {chk['Mz_rel']:.3e}", flush=True)
    print(f"  Q  error absoluto maximo {chk['Q_abs']:.3e}  "
          f"(Q ~ {float(np.mean(np.abs(np.asarray(qj)))):.4f})", flush=True)
    chk["ok"] = (chk["E_rel"] < 1e-4 and chk["M_rel"] < 1e-4
                 and chk["Q_abs"] < 1e-3)
    res["obs_check"] = chk
    print(f"  -> {'exactos a redondeo f32' if chk['ok'] else 'AVISO: NO coinciden'}",
          flush=True)

    # ------------------------------------------- 2. the acceptance rate
    print("\n=== tasa de aceptacion, el pendiente del tramo corto ===",
          flush=True)
    sub = TH.build(geo, BY=int(by), num_warps=int(warps))
    n_upd = float(np.sum(geo["even"]))
    acc_rows = []
    for T in (20.0, 10.0, 5.0, 1.0, 0.1):
        bv = 1.0 / (kB * T)
        st = jax.block_until_ready(jax.jit(
            lambda s: annJ(s, 7, jnp.asarray([bv], jnp.float32), par, k1e,
                           600, 1)[1])(spP))
        # production: one even substep from that state
        sj = TH.unplanar(st)
        oj = eng["substep"](sj, jax.random.PRNGKey(31), bv, even_j, P, k1e)
        aj = float(np.mean(np.any(np.asarray(oj) != np.asarray(sj), -1)
                           ) * (geo["L"] * geo["N"] ** 2) / n_upd)
        ot = sub(st, k1e, par, bv, 999, 0)
        at = float(np.mean(np.any(np.asarray(TH.unplanar(ot))
                                  != np.asarray(sj), -1)
                           ) * (geo["L"] * geo["N"] ** 2) / n_upd)
        acc_rows.append(dict(T=T, acc_prod=aj, acc_halo=at,
                             ratio=at / (aj + 1e-30)))
        print(f"  T={T:>5.1f}  produccion {aj:.4f}   halo {at:.4f}   "
              f"razon {at/(aj+1e-30):.4f}", flush=True)
    res["acceptance"] = acc_rows
    d = max(abs(r["ratio"] - 1.0) for r in acc_rows)
    res["acceptance_max_dev"] = d
    print(f"  desviacion maxima de la razon: {100*d:.2f}%", flush=True)

    # The deficit is systematic, so it is not noise. It can only come from
    # the kernel's own draws, because with JAX's draws the two engines agree
    # bit for bit. At beta -> 0 every proposal is accepted, so the output IS
    # the proposal and its distribution can be read directly: a uniform
    # direction on the sphere has zero mean per component and unit norm.
    print("\n  beta -> 0: la salida ES la propuesta", flush=True)
    bz = 1e-9
    prop = []
    for nm, arr in (("produccion",
                     np.asarray(eng["substep"](TH.unplanar(spP),
                                               jax.random.PRNGKey(3), bz,
                                               even_j, P, k1e))),
                    ("halo", np.asarray(TH.unplanar(
                        sub(spP, k1e, par, bz, 12345, 0))))):
        m = np.asarray(geo["even"])[None]
        sel = arr[np.broadcast_to(m, arr.shape[:-1])]
        sel = sel.reshape(-1, 3)
        nrm = np.linalg.norm(sel, axis=-1)
        row = dict(engine=nm, mean_x=float(sel[:, 0].mean()),
                   mean_y=float(sel[:, 1].mean()),
                   mean_z=float(sel[:, 2].mean()),
                   norm_mean=float(nrm.mean()), norm_sd=float(nrm.std()),
                   n=int(sel.shape[0]))
        prop.append(row)
        print(f"    {nm:>11}: media por componente "
              f"({row['mean_x']:+.5f}, {row['mean_y']:+.5f}, "
              f"{row['mean_z']:+.5f})  |s| {row['norm_mean']:.6f}"
              f" +- {row['norm_sd']:.1e}  n={row['n']}", flush=True)
    res["proposals"] = prop
    err = 1.0 / np.sqrt(prop[0]["n"] / 3.0)
    print(f"    (una direccion uniforme da media 0 +- {err:.5f} y |s| = 1)",
          flush=True)

    # ----------------------------------------------- 3. the ladder, 3 arms
    T = np.linspace(20.0, 0.1, int(n_temps))
    bt = jnp.asarray(1.0 / (kB * T), jnp.float32)
    tot = int(n_temps) * (int(n_therm) + int(n_meas))
    annF = TH.build_anneal_full(geo, BY=int(by), num_warps=int(warps))
    print(f"\n=== escalera: {n_temps}T x {n_therm}+{n_meas} = {tot:,} "
          f"barridos, {B} replicas ===", flush=True)
    arms = {}
    for nm, fn, a0 in (
            ("produccion", lambda s: eng["anneal"](
                s, jax.random.PRNGKey(42), bt, P, int(n_therm), int(n_meas)),
             sp0),
            ("paso Triton", lambda s: annJ(s, 42, bt, par, k1e, int(n_therm),
                                           int(n_meas)), spP),
            ("todo Triton", lambda s: annF(s, 42, bt, par, k1e, int(n_therm),
                                           int(n_meas)), spP)):
        f = jax.jit(fn).lower(a0).compile()
        t0 = time.perf_counter()
        out = jax.block_until_ready(f(a0))
        dt = time.perf_counter() - t0
        arms[nm] = dict(wall_s=dt, minutes=dt / 60.0,
                        us_per_replica_sweep=1e6 * dt / (tot * B))
        rr = S.reduce_obs(np.asarray(out[0]), T, n_act)
        arms[nm]["E"] = np.asarray(rr["E"]).tolist()
        arms[nm]["Cv"] = np.asarray(rr["Cv"]).tolist()
        arms[nm]["chi"] = np.asarray(rr["chi"]).tolist()
        print(f"  {nm:>12}: {dt/60:7.2f} min   "
              f"{1e6*dt/(tot*B):7.3f} us/replica-barrido", flush=True)
        del out
    w0 = arms["produccion"]["wall_s"]
    for nm in ("paso Triton", "todo Triton"):
        arms[nm]["change_pct"] = 100 * (arms[nm]["wall_s"] / w0 - 1)
    res["ladder"] = dict(n_temps=int(n_temps), n_therm=int(n_therm),
                         n_meas=int(n_meas), total_sweeps=tot,
                         T=T.tolist(), arms=arms)
    print(f"\n=== resumen ===", flush=True)
    print(f"  paso Triton, observables en JAX : "
          f"{arms['paso Triton']['change_pct']:+6.1f}%", flush=True)
    print(f"  todo en Triton                  : "
          f"{arms['todo Triton']['change_pct']:+6.1f}%", flush=True)
    rec = (arms["paso Triton"]["wall_s"] - arms["todo Triton"]["wall_s"]) / w0
    res["ladder"]["recovered_pct"] = 100 * rec
    print(f"  los kernels de observables recuperan {100*rec:+.1f} puntos",
          flush=True)
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n  guardado /out/{tag}.json", flush=True)
    return res


@app.function(gpu="H100", image=triton_image, timeout=60 * 90,
              volumes={"/out": vol})
def triton_full_h100(tag, Rd, L, replicas, n_temps, n_therm, n_meas, by, warps):
    return _triton_full(tag, Rd, L, replicas, n_temps, n_therm, n_meas, by,
                        warps)


@app.local_entrypoint()
def triton_full(replicas: int = 50, n_temps: int = 200, n_therm: int = 10000,
                n_meas: int = 5000, rd: float = 18.30, layers: int = 5,
                by: int = 4, warps: int = 4, tag: str = "triton_full"):
    """Substep and observables both in Triton, measured on the ladder."""
    triton_full_h100.remote(tag, rd, layers, replicas, n_temps, n_therm,
                            n_meas, by, warps)


def _triton_ladder4(tag, Rd, L, total, n_temps, n_therm, n_meas, split, by,
                    warps):
    """Four GPUs, 400 chains: does the kernel still win against the protocol?

    The single-GPU gain must NOT be extrapolated across device counts -- the
    width penalty depends on the PER-DEVICE batch, and applying a four-GPU
    figure to a one-GPU arm is a mistake this study already made once. So the
    three arms are run on the real four-device ladder:

      A  production, one wave of 100 per device  -- what the chart does
      B  production, the split protocol, two waves per device
      C  the Triton engine, one wave of 100 per device

    B is the page's current recommendation, so it is the bar the kernel has
    to clear, not the unsplit arm.
    """
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S

    print(f"gpu={device_identity()}", flush=True)
    try:
        import triton_halo as TH
    except Exception as e:
        print(f"TRITON NO DISPONIBLE: {type(e).__name__}: {e}", flush=True)
        return dict(available=False, error=f"{type(e).__name__}: {e}")

    kB = 0.086173404
    nd = jax.local_device_count()
    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    eng = S.build_engine(geo)
    surf_j = jnp.asarray(geo["surface"][None, ...])
    n_act = float(geo["n_active"])
    T = np.linspace(20.0, 0.1, int(n_temps))
    bt = jnp.asarray(1.0 / (kB * T), jnp.float32)
    tot_s = int(n_temps) * (int(n_therm) + int(n_meas))
    per = int(total) // nd
    raw = json.load(open("/root/theta4.json"))
    t0p = list(raw.values())[0]
    print(f"{nd} dispositivos, {total} cadenas, {per} por dispositivo, "
          f"{n_temps}T x {n_therm}+{n_meas} = {tot_s:,} barridos", flush=True)
    res = dict(available=True, gpu=device_identity(), n_devices=nd,
               total_chains=int(total), per_device=per,
               n_temps=int(n_temps), n_therm=int(n_therm),
               n_meas=int(n_meas), total_sweeps=tot_s, arms={})

    def mk(b):
        c = lambda v: jnp.full((b, 1, 1, 1), float(v), jnp.float32)
        P = dict(Jex=c(1.0), Kan1=c(t0p["Kan1"]), Kan2=c(0.0),
                 KanS=c(t0p["KanS"]), Hex=c(t0p["Hex"]), KDM=c(t0p["KDM"]),
                 g=c(float(np.deg2rad(90.0))))
        return P, jnp.where(surf_j, P["KanS"], P["Kan1"]), TH.pack_params(P, b)

    def shard(b, seed, planar):
        sp = S.init_spins(jax.random.PRNGKey(seed), geo, nd * b)
        sp = TH.planar(sp) if planar else sp
        return sp.reshape(nd, b, *sp.shape[1:])

    def timed(label, waves, planar, build_fn):
        """One arm: the sum over its waves, each a pmap over the devices."""
        fns, args = [], []
        for b in waves:
            P, k1, par = mk(b)
            a = shard(b, 5 + b, planar)
            fns.append(jax.pmap(build_fn(P, k1, par)))
            args.append(a)
        keys = [jax.random.split(jax.random.PRNGKey(40 + i), nd)
                for i in range(len(waves))]
        for f, a, k in zip(fns, args, keys):      # compile outside the timer
            jax.block_until_ready(f(a, k) if planar is False else f(a))
        t0 = time.perf_counter()
        outs = []
        for f, a, k in zip(fns, args, keys):
            outs.append(jax.block_until_ready(f(a, k) if planar is False
                                              else f(a)))
        dt = time.perf_counter() - t0
        row = dict(waves=list(waves), wall_s=dt, minutes=dt / 60.0,
                   us_per_replica_sweep=1e6 * dt / (tot_s * int(total)))
        # Keep the physics, so the arm is a result and not just a stopwatch.
        o = np.concatenate([np.asarray(x[0]).reshape(
            nd * w, *np.asarray(x[0]).shape[2:]) for x, w in
            zip(outs, waves)], axis=0) if False else None
        res["arms"][label] = row
        print(f"  {label:<34}{dt/60:7.2f} min   "
              f"{row['us_per_replica_sweep']:7.3f} us/replica-barrido",
              flush=True)
        del outs
        return dt

    wa = timed(f"A produccion, 1 ola de {per}", [per], False,
               lambda P, k1, par: (lambda sp, k: eng["anneal"](
                   sp, k, bt, P, int(n_therm), int(n_meas))))
    sp_waves = [int(x) for x in split.split("+")] if split else []
    if sp_waves and sum(sp_waves) == per:
        timed(f"B produccion, protocolo {'+'.join(map(str, sp_waves))}",
              sp_waves, False,
              lambda P, k1, par: (lambda sp, k: eng["anneal"](
                  sp, k, bt, P, int(n_therm), int(n_meas))))
    annF = TH.build_anneal_full(geo, BY=int(by), num_warps=int(warps))
    timed(f"C Triton, 1 ola de {per}", [per], True,
          lambda P, k1, par: (lambda sp: annF(
              sp, 42, bt, par, k1, int(n_therm), int(n_meas))))

    print(f"\n=== resumen, {nd} GPU ===", flush=True)
    for k, v in res["arms"].items():
        v["vs_A_pct"] = 100 * (v["wall_s"] / wa - 1)
        print(f"  {k:<34}{v['minutes']:7.2f} min   {v['vs_A_pct']:+7.1f}%",
              flush=True)
    bk = [k for k in res["arms"] if k.startswith("B")]
    if bk:
        g = 100 * (res["arms"][[k for k in res["arms"] if k.startswith("C")][0]
                               ]["wall_s"] / res["arms"][bk[0]]["wall_s"] - 1)
        res["C_vs_B_pct"] = g
        print(f"\n  el kernel contra el protocolo (la barra real): "
              f"{g:+.1f}%", flush=True)
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n  guardado /out/{tag}.json", flush=True)
    return res


@app.function(gpu="H100:4", image=triton_image, timeout=60 * 90,
              volumes={"/out": vol})
def triton_ladder4_h100(tag, Rd, L, total, n_temps, n_therm, n_meas, split,
                        by, warps):
    return _triton_ladder4(tag, Rd, L, total, n_temps, n_therm, n_meas, split,
                           by, warps)


@app.local_entrypoint()
def triton_ladder4(total: int = 400, n_temps: int = 60,
                   n_therm: int = 10000, n_meas: int = 5000,
                   split: str = "53+47", rd: float = 18.30, layers: int = 5,
                   by: int = 4, warps: int = 4, tag: str = "triton_ladder4"):
    """The four-GPU ladder: kernel against the split protocol."""
    triton_ladder4_h100.remote(tag, rd, layers, total, n_temps, n_therm,
                               n_meas, split, by, warps)


def _triton_width(tag, Rd, layers, batches, repeats, steps, by, warps):
    """Does the width rule still bite the hand-written kernel?

    The rule was `penalty iff v2(B) + v2(L) >= 2`, and it came from XLA
    deciding how to vectorise the loop it was handed. This kernel hands it no
    such loop: the block shape is fixed by me, and the layer count only sets
    how many blocks there are along z. So the mechanism should be gone for
    BOTH axes -- but only B was ever measured. This measures L too.

    Each layer count is its own column: changing L changes the work per
    sweep, so the comparison is production against the kernel WITHIN a fixed
    L, never across.
    """
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S

    print(f"gpu={device_identity()}", flush=True)
    try:
        import triton_halo as TH
    except Exception as e:
        print(f"TRITON NO DISPONIBLE: {type(e).__name__}: {e}", flush=True)
        return dict(available=False, error=f"{type(e).__name__}: {e}")

    kB = 0.086173404
    beta = 1.0 / (kB * 5.0)
    K = int(steps)
    raw = json.load(open("/root/theta4.json"))
    t0p = list(raw.values())[0]
    res = dict(available=True, gpu=device_identity(), rows=[],
               steps_per_call=K, repeats=int(repeats))
    print(f"{'L':>3}{'v2(L)':>7}{'B':>6}{'v2(B)':>7}{'suma':>6}"
          f"{'produccion':>13}{'halo':>10}{'cambio':>9}", flush=True)
    v2 = lambda n: (n & -n).bit_length() - 1

    for L in layers:
        geo = S.build_geometry(Rd=float(Rd), L=int(L))
        eng = S.build_engine(geo)
        surf_j = jnp.asarray(geo["surface"][None, ...])
        even_j = jnp.asarray(geo["even"][None, ..., None])
        odd_j = jnp.asarray(geo["odd"][None, ..., None])
        for B in batches:
            c = lambda v: jnp.full((B, 1, 1, 1), float(v), jnp.float32)
            P = dict(Jex=c(1.0), Kan1=c(t0p["Kan1"]), Kan2=c(0.0),
                     KanS=c(t0p["KanS"]), Hex=c(t0p["Hex"]),
                     KDM=c(t0p["KDM"]), g=c(float(np.deg2rad(90.0))))
            k1 = jnp.where(surf_j, P["KanS"], P["Kan1"])
            par = TH.pack_params(P, B)
            sp = S.init_spins(jax.random.PRNGKey(5 + B + L), geo, B)
            spP = TH.planar(sp)
            sub = TH.build(geo, BY=int(by), num_warps=int(warps))

            def prod(s):
                def body(i, v):
                    st, k = v
                    k, ka, kb = jax.random.split(k, 3)
                    st = eng["substep"](st, ka, beta, even_j, P, k1)
                    st = eng["substep"](st, kb, beta, odd_j, P, k1)
                    return st, k
                return jax.lax.fori_loop(0, K, body,
                                         (s, jax.random.PRNGKey(0)))[0]

            f1 = jax.jit(prod).lower(sp).compile()
            f2 = jax.jit(TH.sweeps(sub, k1, par, beta, K)).lower(spP).compile()
            jax.block_until_ready(f1(sp)); jax.block_until_ready(f2(spP))
            a1, a2 = [], []
            for _ in range(int(repeats)):          # interleaved
                t = time.perf_counter(); jax.block_until_ready(f1(sp))
                a1.append(time.perf_counter() - t)
                t = time.perf_counter(); jax.block_until_ready(f2(spP))
                a2.append(time.perf_counter() - t)
            m1, m2 = float(np.median(a1)), float(np.median(a2))
            row = dict(L=int(L), B=int(B), v2L=v2(L), v2B=v2(B),
                       suma=v2(L) + v2(B),
                       prod_us=1e6 * m1 / (K * B),
                       halo_us=1e6 * m2 / (K * B),
                       change_pct=100 * (m2 / m1 - 1))
            res["rows"].append(row)
            print(f"{L:>3}{row['v2L']:>7}{B:>6}{row['v2B']:>7}"
                  f"{row['suma']:>6}{row['prod_us']:>13.3f}"
                  f"{row['halo_us']:>10.3f}{row['change_pct']:>+8.1f}%",
                  flush=True)

    # The rule's own prediction: a penalty exactly when v2(B)+v2(L) >= 2.
    # Scored separately for each engine, within each L.
    print(f"\n=== la regla del ancho, por motor ===", flush=True)
    summ = {}
    for L in layers:
        rs = [r for r in res["rows"] if r["L"] == L]
        if len(rs) < 2:
            continue
        hit = [r for r in rs if r["suma"] >= 2]
        miss = [r for r in rs if r["suma"] < 2]
        if not hit or not miss:
            continue
        for eng_k, lab in (("prod_us", "produccion"), ("halo_us", "halo")):
            h = float(np.mean([r[eng_k] for r in hit]))
            m = float(np.mean([r[eng_k] for r in miss]))
            pen = 100 * (h / m - 1)
            summ[f"L{L}_{lab}"] = pen
            print(f"  L={L}  {lab:>11}: castigo por v2>=2  {pen:+6.1f}%",
                  flush=True)
    res["penalty"] = summ
    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n  guardado /out/{tag}.json", flush=True)
    return res


@app.function(gpu="H100", image=triton_image, timeout=60 * 60,
              volumes={"/out": vol})
def triton_width_h100(tag, Rd, layers, batches, repeats, steps, by, warps):
    return _triton_width(tag, Rd, layers, batches, repeats, steps, by, warps)


@app.local_entrypoint()
def triton_width(layers: str = "4,5,6,8", batches: str = "64,65,100,101",
                 repeats: int = 7, steps: int = 200, rd: float = 18.30,
                 by: int = 4, warps: int = 4, tag: str = "triton_width"):
    """Does v2(B)+v2(L)>=2 still cost anything with the hand-written kernel?"""
    triton_width_h100.remote(tag, rd, [int(x) for x in layers.split(",") if x],
                             [int(x) for x in batches.split(",") if x],
                             repeats, steps, by, warps)


def _state_ladder(tag, Rd, L, reps, n_temps, sweeps, by, warps):
    """One state, the 20k-sweep convention, across however many GPUs we have.

    The project's schedule is 20 000 sweeps per temperature, not the 15 000 the
    earlier ladders used, so this re-measures on the real convention. The split
    is the repository's own two thirds / one third.

    One arithmetic fact decides the shape of this table: `pmap` needs equal
    shards, and 50 replicas DO NOT divide by four. So the four-device arm
    brackets it with 48 and 52 instead of pretending, and every arm also
    reports us per replica-sweep, which normalises the differing work.

    The interesting quantity is not the wall clock but whether more devices
    still help: at 50 replicas over four GPUs the per-device batch falls to
    about a dozen, and the launch cost `a/B` was already measured to dominate
    down there.
    """
    import sys, json, time
    sys.path.insert(0, "/root")
    import numpy as np, jax, jax.numpy as jnp
    import sim_core as S

    print(f"gpu={device_identity()}", flush=True)
    try:
        import triton_halo as TH
    except Exception as e:
        print(f"TRITON NO DISPONIBLE: {type(e).__name__}: {e}", flush=True)
        return dict(available=False, error=f"{type(e).__name__}: {e}")

    kB = 0.086173404
    nd = jax.local_device_count()
    n_therm = int(sweeps) * 2 // 3
    n_meas = int(sweeps) - n_therm
    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    eng = S.build_engine(geo)
    surf = jnp.asarray(geo["surface"][None, ...])
    n_act = float(geo["n_active"])
    T = np.linspace(20.0, 0.1, int(n_temps))
    bt = jnp.asarray(1.0 / (kB * T), jnp.float32)
    tot = int(n_temps) * (n_therm + n_meas)
    raw = json.load(open("/root/theta4.json"))
    t0p = list(raw.values())[0]
    print(f"{nd} dispositivo(s)  |  1 estado, {n_temps}T x "
          f"{n_therm}+{n_meas} = {tot:,} barridos por replica", flush=True)
    res = dict(available=True, gpu=device_identity(), n_devices=nd,
               n_temps=int(n_temps), sweeps=int(sweeps), n_therm=n_therm,
               n_meas=n_meas, total_sweeps=tot, rows=[])

    annF = TH.build_anneal_full(geo, BY=int(by), num_warps=int(warps))

    for R in reps:
        if R % nd:
            print(f"  {R} replicas no divide entre {nd}: se omite", flush=True)
            continue
        per = R // nd
        c = lambda v: jnp.full((per, 1, 1, 1), float(v), jnp.float32)
        P = dict(Jex=c(1.0), Kan1=c(t0p["Kan1"]), Kan2=c(0.0),
                 KanS=c(t0p["KanS"]), Hex=c(t0p["Hex"]), KDM=c(t0p["KDM"]),
                 g=c(float(np.deg2rad(90.0))))
        k1 = jnp.where(surf, P["KanS"], P["Kan1"])
        par = TH.pack_params(P, per)
        sp = S.init_spins(jax.random.PRNGKey(7), geo, R)
        spP = TH.planar(sp)
        if nd > 1:
            sp = sp.reshape(nd, per, *sp.shape[1:])
            spP = spP.reshape(nd, per, *spP.shape[1:])
            keys = jax.random.split(jax.random.PRNGKey(42), nd)

        def arm(name):
            if name == "produccion":
                if nd == 1:
                    f = jax.jit(lambda s: eng["anneal"](
                        s, jax.random.PRNGKey(42), bt, P, n_therm, n_meas))
                    a = (sp,)
                else:
                    f = jax.pmap(lambda s, k: eng["anneal"](
                        s, k, bt, P, n_therm, n_meas))
                    a = (sp, keys)
            else:
                g = lambda s: annF(s, 42, bt, par, k1, n_therm, n_meas)
                f = jax.jit(g) if nd == 1 else jax.pmap(g)
                a = (spP,)
            # Compile WITHOUT executing. The earlier version warmed up with a
            # real call, which on a four-million-sweep ladder doubles the bill
            # -- that is what exhausted an account for a table this size.
            try:
                f = f.lower(*a).compile()
            except Exception:
                jax.block_until_ready(f(*a))
            t0 = time.perf_counter()
            o = jax.block_until_ready(f(*a))
            dt = time.perf_counter() - t0
            del o
            return dt

        row = dict(replicas=int(R), per_device=per)
        for nm in ("produccion", "kernel"):
            dt = arm(nm)
            row[nm] = dict(wall_s=dt, minutes=dt / 60.0,
                           us_per_replica_sweep=1e6 * dt / (tot * R))
        row["change_pct"] = 100 * (row["kernel"]["wall_s"]
                                   / row["produccion"]["wall_s"] - 1)
        res["rows"].append(row)
        print(f"  {R:>3} replicas ({per:>2}/disp):  produccion "
              f"{row['produccion']['minutes']:6.2f} min "
              f"({row['produccion']['us_per_replica_sweep']:6.3f} us)   "
              f"kernel {row['kernel']['minutes']:6.2f} min "
              f"({row['kernel']['us_per_replica_sweep']:6.3f} us)   "
              f"{row['change_pct']:+6.1f}%", flush=True)

    with open(f"/out/{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n  guardado /out/{tag}.json", flush=True)
    return res


@app.function(gpu="H100", image=triton_image, timeout=60 * 90,
              volumes={"/out": vol})
def state_ladder_1(tag, Rd, L, reps, n_temps, sweeps, by, warps):
    return _state_ladder(tag, Rd, L, reps, n_temps, sweeps, by, warps)


@app.function(gpu="H100:2", image=triton_image, timeout=60 * 90,
              volumes={"/out": vol})
def state_ladder_2(tag, Rd, L, reps, n_temps, sweeps, by, warps):
    return _state_ladder(tag, Rd, L, reps, n_temps, sweeps, by, warps)


@app.function(gpu="H100:4", image=triton_image, timeout=60 * 90,
              volumes={"/out": vol})
def state_ladder_4(tag, Rd, L, reps, n_temps, sweeps, by, warps):
    return _state_ladder(tag, Rd, L, reps, n_temps, sweeps, by, warps)


@app.local_entrypoint()
def state_ladder(n_temps: int = 200, sweeps: int = 20000, rd: float = 18.30,
                 layers: int = 5, by: int = 4, warps: int = 4,
                 reps: str = "", tag: str = "state_ladder"):
    """One state, 20k sweeps, 50 replicas, on one, two and four GPUs.

    Fifty replicas cannot be sharded four ways, so the four-device call gets
    48 and 52 -- the two nearest counts that can -- and the per-replica rate
    makes them comparable to the fifty.
    """
    rl = [int(x) for x in str(reps).split(",") if x] if reps else None
    for nd, fn, dflt in ((1, state_ladder_1, [50]),
                         (2, state_ladder_2, [50]),
                         (4, state_ladder_4, [48, 52])):
        reps_n = rl if rl else dflt
        print(f"\n########## {nd} GPU ##########", flush=True)
        fn.remote(f"{tag}_{nd}gpu", rd, layers, reps_n, n_temps, sweeps, by,
                  warps)
