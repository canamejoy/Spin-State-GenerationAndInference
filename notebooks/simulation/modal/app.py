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

# The XLA-confound probe needs a kernel that XLA never touches, so this image
# adds CuPy and compiles the stencil with nvrtc instead. It branches off `_base`
# rather than `image` because the local files have to go on last, and it needs
# only `sim_core` for the geometry.
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
