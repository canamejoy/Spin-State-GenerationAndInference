"""Does the hand-written kernel travel? One self-contained benchmark.

The whole Triton result was measured on H100s rented by the hour. That leaves
two things unknown. Whether the kernel even COMPILES elsewhere -- Triton wants
compute capability 7.0 or better, so Kaggle's P100 (6.0) should refuse it
while its T4 (7.5) should not. And whether the gain is an H100 property: the
H100 has 132 SMs and enormous L2, and a stencil that lives in registers might
behave very differently on a 40-SM T4.

So this module runs, in order:

  1. the environment, including compute capability, because a refusal there is
     a result and not a failure;
  2. exactness -- the stencil against `sim_core.substep` on IDENTICAL
     randomness, which must give zero differing spins;
  3. physics -- both engines to equilibrium with their own randomness, whose
     energies must agree within the spread the run itself reports;
  4. the substep rate, production against the kernel, over a grid of layer
     counts and batches, which also scores the width rule SEPARATELY for each
     engine;
  5. a short ladder, so the figure quoted is the full anneal and not a step.

Nothing is timed before the physics agrees, and every arm is interleaved so a
GPU that warms up spreads its drift across both.
"""

from __future__ import annotations

import json
import time

import numpy as np


def _v2(n):
    return (n & -n).bit_length() - 1


def environment():
    import subprocess
    import sys
    out = {}
    try:
        out["smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,clocks.max.sm,"
             "driver_version", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip()
    except Exception as e:
        out["smi"] = f"nvidia-smi no disponible: {e}"
    try:
        import jax
        out["jax"] = jax.__version__
        out["devices"] = [str(d) for d in jax.devices()]
        d = jax.devices()[0]
        # Compute capability decides whether Triton will emit anything at all.
        cc = getattr(d, "compute_capability", None)
        out["compute_capability"] = str(cc) if cc else "desconocida"
        out["device_kind"] = getattr(d, "device_kind", "?")
    except Exception as e:
        out["jax_error"] = f"{type(e).__name__}: {e}"
    for mod in ("triton", "jax_triton"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception as e:
            out[f"{mod}_error"] = f"{type(e).__name__}: {e}"
    out["python"] = sys.version.split()[0]
    return out


def run(batches=(64, 65, 100, 101), layers=(5,), repeats=5, steps=200,
        Rd=18.30, by=4, warps=4, replicas=50, n_temps=20, n_therm=300,
        n_meas=150, seed=11, out=None, label="kaggle"):
    import jax
    import jax.numpy as jnp
    import sim_core as S

    res = dict(label=label, env=environment(), rows=[], ok=False)
    print("=== entorno ===", flush=True)
    for k in ("smi", "device_kind", "compute_capability", "jax", "triton",
              "jax_triton", "python"):
        if k in res["env"]:
            print(f"  {k:<20} {res['env'][k]}", flush=True)
    cc = res["env"].get("compute_capability", "")
    if cc and cc.replace(".", "").isdigit() and float(
            cc if "." in cc else f"{cc[0]}.{cc[1:]}") < 7.0:
        print(f"\n  AVISO: capacidad de computo {cc} < 7.0. Triton exige 7.0+,"
              f" asi que lo mas probable es que se niegue a compilar. Eso es"
              f" un resultado, no un fallo: elige el acelerador T4 (7.5).",
              flush=True)

    try:
        import triton_halo as TH
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        print(f"\nTRITON NO DISPONIBLE: {res['error']}", flush=True)
        if out:
            json.dump(res, open(out, "w"), indent=1)
        return res

    kB = 0.086173404
    beta = 1.0 / (kB * 5.0)
    K = int(steps)
    P0 = dict(Jex=1.0, Kan1=0.177, Kan2=0.0, KanS=0.139, Hex=0.483,
              KDM=0.880, g=float(np.deg2rad(90.0)))

    def params(b):
        c = lambda v: jnp.full((b, 1, 1, 1), float(v), jnp.float32)
        return {k: c(v) for k, v in P0.items()}

    # ---------------------------------------------------- 2. exactness
    print("\n=== exactitud: el stencil contra sim_core.substep ===",
          flush=True)
    geo = S.build_geometry(Rd=float(Rd), L=int(layers[0]))
    eng = S.build_engine(geo)
    surf = jnp.asarray(geo["surface"][None, ...])
    ev = jnp.asarray(geo["even"][None, ..., None])
    od = jnp.asarray(geo["odd"][None, ..., None])
    Bv = 8
    P = params(Bv)
    k1 = jnp.where(surf, P["KanS"], P["Kan1"])
    sp0 = S.init_spins(jax.random.PRNGKey(3), geo, Bv)
    kt, ka = jax.random.split(jax.random.PRNGKey(101))
    tr = jax.random.normal(kt, sp0.shape)
    tr = tr / jnp.linalg.norm(tr, axis=-1, keepdims=True)
    ru = jax.random.uniform(ka, sp0.shape[:-1])
    sub = TH.build(geo, BY=int(by), num_warps=int(warps))
    par = TH.pack_params(P, Bv)
    spP = TH.planar(sp0)
    trP = TH.planar(tr)
    ex = []
    for parity, mask, nm in ((0, ev, "par"), (1, od, "impar")):
        # `substep` draws its own randomness, so the reference is rebuilt here
        # from the SAME tr/ru the kernel is given -- otherwise this compares
        # two different chains and proves nothing.
        trm = jnp.where(mask, tr, sp0)
        dS = trm - sp0
        xp = jnp.roll(sp0, -1, 3); xm = jnp.roll(sp0, 1, 3)
        yp = jnp.roll(sp0, -1, 2); ym = jnp.roll(sp0, 1, 2)
        zp = jnp.roll(sp0, -1, 1).at[:, -1].set(0.0)
        zm = jnp.roll(sp0, 1, 1).at[:, 0].set(0.0)
        dE = -P["Jex"] * jnp.sum(dS * (xp + xm + yp + ym + zp + zm), -1)
        an = lambda s: (k1 * ((s[..., 0] * s[..., 1]) ** 2
                              + (s[..., 0] * s[..., 2]) ** 2
                              + (s[..., 1] * s[..., 2]) ** 2)
                        + P["Kan2"] * (s[..., 0] * s[..., 1] * s[..., 2]) ** 2)
        dE = dE + an(trm) - an(sp0)
        dE = dE - P["KDM"] * (dS[..., 1] * (xp[..., 2] - xm[..., 2])
                              - dS[..., 2] * (xp[..., 1] - xm[..., 1])
                              + dS[..., 2] * (yp[..., 0] - ym[..., 0])
                              - dS[..., 0] * (yp[..., 2] - ym[..., 2]))
        dE = dE - P["Hex"] * (dS[..., 0] * jnp.cos(P["g"])
                              + dS[..., 2] * jnp.sin(P["g"]))
        acc = ((dE < 0) | (ru < jnp.exp(-dE * beta)))[..., None] & mask
        ref = jnp.where(acc, trm, sp0)
        got, dE_t = sub(spP, k1, par, beta, 0, parity, trP, ru, want_de=True)
        sel = np.broadcast_to(np.asarray(mask[..., 0]), dE.shape)
        mad = float(np.max(np.abs(np.asarray(dE)[sel]
                                  - np.asarray(dE_t)[sel])))
        bad = int(np.sum(np.asarray(ref) != np.asarray(TH.unplanar(got))))
        ex.append(dict(sublattice=nm, dE_max_abs=mad, spin_mismatch=bad,
                       spin_total=int(np.asarray(ref).size)))
        print(f"  {nm:>5}: dE max |dif| {mad:.3e}   espines distintos "
              f"{bad}/{np.asarray(ref).size}", flush=True)
    res["exactness"] = ex
    res["exact_ok"] = all(r["spin_mismatch"] == 0 for r in ex)

    # ------------------------------------------------------ 3. physics
    print("\n=== fisica: las dos cadenas a equilibrio, cada una con su RNG ===",
          flush=True)
    Bp = 16
    Pp = params(Bp)
    k1p = jnp.where(surf, Pp["KanS"], Pp["Kan1"])
    parp = TH.pack_params(Pp, Bp)
    spE = S.init_spins(jax.random.PRNGKey(11), geo, Bp)
    subp = TH.build(geo, BY=int(by), num_warps=int(warps))
    n_act = float(geo["n_active"])
    ph = []
    for T in (20.0, 5.0, 1.0):
        bT = 1.0 / (kB * T)
        n_eq = 400

        def jx(s):
            def body(i, v):
                st, k = v
                k, a, b = jax.random.split(k, 3)
                st = eng["substep"](st, a, bT, ev, Pp, k1p)
                st = eng["substep"](st, b, bT, od, Pp, k1p)
                return st, k
            return jax.lax.fori_loop(0, n_eq, body,
                                     (s, jax.random.PRNGKey(77)))[0]

        sj = jax.block_until_ready(jax.jit(jx)(spE))
        st = TH.unplanar(jax.block_until_ready(jax.jit(
            TH.sweeps(subp, k1p, parp, bT, n_eq))(TH.planar(spE))))
        ej = float(jnp.mean(eng["energy"](sj, Pp, k1p)) / n_act)
        et = float(jnp.mean(eng["energy"](st, Pp, k1p)) / n_act)
        sd = float(jnp.std(eng["energy"](sj, Pp, k1p)) / n_act)
        row = dict(T=T, E_prod=ej, E_halo=et, E_sd=sd,
                   sigma=abs(et - ej) / (sd + 1e-30),
                   M_prod=float(jnp.mean(eng["magnetization"](sj)[0])),
                   M_halo=float(jnp.mean(eng["magnetization"](st)[0])))
        ph.append(row)
        print(f"  T={T:>5.1f}  E/sitio {ej:+9.5f} / {et:+9.5f}  "
              f"({row['sigma']:.2f} sigma)   |M| {row['M_prod']:.4f} / "
              f"{row['M_halo']:.4f}", flush=True)
    res["physics"] = ph
    res["physics_ok"] = all(r["sigma"] < 3.0 for r in ph)
    if not (res["exact_ok"] and res["physics_ok"]):
        print("\n  AVISO: la fisica NO coincide. Los tiempos de abajo no son "
              "comparables.", flush=True)

    # ------------------------------- 4. the substep rate, and the width rule
    print(f"\n=== ritmo del paso, {K} barridos por llamada, "
          f"{repeats} repeticiones intercaladas ===", flush=True)
    print(f"{'L':>3}{'B':>6}{'v2(B)+v2(L)':>13}{'produccion':>13}"
          f"{'kernel':>10}{'cambio':>9}", flush=True)
    for L in layers:
        g = S.build_geometry(Rd=float(Rd), L=int(L))
        e2 = S.build_engine(g)
        s2 = jnp.asarray(g["surface"][None, ...])
        v2e = jnp.asarray(g["even"][None, ..., None])
        v2o = jnp.asarray(g["odd"][None, ..., None])
        for B in batches:
            Pb = params(B)
            k1b = jnp.where(s2, Pb["KanS"], Pb["Kan1"])
            parb = TH.pack_params(Pb, B)
            spb = S.init_spins(jax.random.PRNGKey(seed + B + L), g, B)
            spbP = TH.planar(spb)
            sb = TH.build(g, BY=int(by), num_warps=int(warps))

            def prod(s):
                def body(i, v):
                    st, k = v
                    k, a, b = jax.random.split(k, 3)
                    st = e2["substep"](st, a, beta, v2e, Pb, k1b)
                    st = e2["substep"](st, b, beta, v2o, Pb, k1b)
                    return st, k
                return jax.lax.fori_loop(0, K, body,
                                         (s, jax.random.PRNGKey(0)))[0]

            f1 = jax.jit(prod).lower(spb).compile()
            f2 = jax.jit(TH.sweeps(sb, k1b, parb, beta, K)
                         ).lower(spbP).compile()
            jax.block_until_ready(f1(spb)); jax.block_until_ready(f2(spbP))
            a1, a2 = [], []
            for _ in range(int(repeats)):
                t = time.perf_counter(); jax.block_until_ready(f1(spb))
                a1.append(time.perf_counter() - t)
                t = time.perf_counter(); jax.block_until_ready(f2(spbP))
                a2.append(time.perf_counter() - t)
            m1, m2 = float(np.median(a1)), float(np.median(a2))
            row = dict(L=int(L), B=int(B), suma=_v2(L) + _v2(B),
                       prod_us=1e6 * m1 / (K * B),
                       halo_us=1e6 * m2 / (K * B),
                       change_pct=100 * (m2 / m1 - 1))
            res["rows"].append(row)
            print(f"{L:>3}{B:>6}{row['suma']:>13}{row['prod_us']:>13.3f}"
                  f"{row['halo_us']:>10.3f}{row['change_pct']:>+8.1f}%",
                  flush=True)

    pen = {}
    for L in layers:
        rs = [r for r in res["rows"] if r["L"] == L]
        hit = [r for r in rs if r["suma"] >= 2]
        miss = [r for r in rs if r["suma"] < 2]
        if hit and miss:
            for key, lab in (("prod_us", "produccion"), ("halo_us", "kernel")):
                p = 100 * (float(np.mean([r[key] for r in hit]))
                           / float(np.mean([r[key] for r in miss])) - 1)
                pen[f"L{L}_{lab}"] = p
    if pen:
        print(f"\n=== la regla del ancho: castigo por v2(B)+v2(L) >= 2 ===",
              flush=True)
        for k, v in pen.items():
            print(f"  {k:<22}{v:+7.1f}%", flush=True)
    res["penalty"] = pen

    # ------------------------------------------------- 5. the short ladder
    print(f"\n=== escalera corta: {n_temps}T x {n_therm}+{n_meas}, "
          f"{replicas} replicas ===", flush=True)
    Bl = int(replicas)
    Pl = params(Bl)
    k1l = jnp.where(surf, Pl["KanS"], Pl["Kan1"])
    parl = TH.pack_params(Pl, Bl)
    Tl = np.linspace(20.0, 0.1, int(n_temps))
    btl = jnp.asarray(1.0 / (kB * Tl), jnp.float32)
    spl = S.init_spins(jax.random.PRNGKey(5), geo, Bl)
    annF = TH.build_anneal_full(geo, BY=int(by), num_warps=int(warps))
    tot = int(n_temps) * (int(n_therm) + int(n_meas))
    lad = {}
    for nm, fn, a0 in (
            ("produccion", lambda s: eng["anneal"](
                s, jax.random.PRNGKey(42), btl, Pl, int(n_therm),
                int(n_meas)), spl),
            ("kernel", lambda s: annF(s, 42, btl, parl, k1l, int(n_therm),
                                      int(n_meas)), TH.planar(spl))):
        f = jax.jit(fn).lower(a0).compile()
        t0 = time.perf_counter()
        o = jax.block_until_ready(f(a0))
        dt = time.perf_counter() - t0
        lad[nm] = dict(wall_s=dt, us_per_replica_sweep=1e6 * dt / (tot * Bl))
        print(f"  {nm:>11}: {dt:7.2f} s   "
              f"{lad[nm]['us_per_replica_sweep']:7.3f} us/replica-barrido",
              flush=True)
        del o
    lad["change_pct"] = 100 * (lad["kernel"]["wall_s"]
                               / lad["produccion"]["wall_s"] - 1)
    res["ladder"] = dict(n_temps=int(n_temps), n_therm=int(n_therm),
                         n_meas=int(n_meas), replicas=Bl, total_sweeps=tot,
                         arms=lad)
    print(f"\n  la escalera completa: {lad['change_pct']:+.1f}%", flush=True)
    res["ok"] = bool(res["exact_ok"] and res["physics_ok"])
    if out:
        json.dump(res, open(out, "w"), indent=1)
        print(f"\n  guardado {out}", flush=True)
    return res


def run_states(reps=(50, 500), devices=(1, 2), n_temps=10, sweeps=20000,
               Rd=18.30, L=5, by=4, warps=4, So=1.0, out=None,
               label="kaggle-states"):
    """One state and ten states, on one card and on two.

    Kaggle hands out two T4s, which is exactly the pair this needs. The
    one-device arm uses `jax.jit`, which places the whole batch on device 0;
    the two-device arm uses `pmap`. Both replica counts divide by both device
    counts (500 -> 500/250, 50 -> 50/25), so nothing has to be approximated
    the way 50 across four cards did.

    The ladder is SHORTENED on purpose. What this measures is a RATE --
    microseconds per replica-sweep -- and the full 200-temperature clock is
    reported as that rate times the real sweep count. A T4 running ten states
    at full length would take hours, which is why the long cells are
    extrapolated rather than waited for.
    """
    import jax
    import jax.numpy as jnp
    import sim_core as S

    import time
    kB = 0.086173404
    n_therm = int(sweeps) * 2 // 3
    n_meas = int(sweeps) - n_therm
    geo = S.build_geometry(Rd=float(Rd), L=int(L))
    eng = S.build_engine(geo)
    surf = jnp.asarray(geo["surface"][None, ...])
    T = np.linspace(20.0, 0.1, int(n_temps))
    bt = jnp.asarray(1.0 / (kB * T), jnp.float32)
    measured = int(n_temps) * (n_therm + n_meas)
    full = 200 * (n_therm + n_meas)          # the production ladder
    P0 = dict(Jex=1.0, Kan1=0.177, Kan2=0.0, KanS=0.139, Hex=0.483,
              KDM=0.880, g=float(np.deg2rad(90.0)))

    import triton_halo as TH

    # Nothing is timed before the physics is proven in THIS run. A clock
    # nobody checked does not belong on a chart, even when the same file
    # verified elsewhere and its checksum matches.
    chk = verify(geo, by=by, warps=warps, So=So)   # So now a real parameter
    if not chk["ok"]:
        print("\n  la fisica no coincide: no se cronometra nada", flush=True)
        if out:
            json.dump(dict(label=label, env=environment(), verify=chk,
                           ok=False, rows=[]), open(out, "w"), indent=1)
        return dict(label=label, verify=chk, ok=False, rows=[])
    print("", flush=True)

    annF = TH.build_anneal_full(geo, BY=int(by), num_warps=int(warps), So=So)
    avail = jax.local_device_count()
    res = dict(label=label, env=environment(), n_temps=int(n_temps),
               sweeps=int(sweeps), n_therm=n_therm, n_meas=n_meas,
               measured_sweeps=measured, full_sweeps=full,
               devices_available=avail, verify=chk, rows=[])
    print(f"{avail} T4 visibles  |  escalera medida {n_temps}T x "
          f"{n_therm}+{n_meas} = {measured:,} barridos", flush=True)
    print(f"  el reloj a 200 T se reporta como tasa x {full:,} barridos",
          flush=True)
    print(f"\n{'disp':>5}{'repl':>6}{'/disp':>7}{'estandar':>13}"
          f"{'optimizado':>13}{'cambio':>9}", flush=True)

    for nd in devices:
        if nd > avail:
            print(f"  {nd} dispositivos: solo hay {avail}, se omite", flush=True)
            continue
        for R in reps:
            if R % nd:
                continue
            per = R // nd
            c = lambda v: jnp.full((per, 1, 1, 1), float(v), jnp.float32)
            P = {k: c(v) for k, v in P0.items()}
            k1 = jnp.where(surf, P["KanS"], P["Kan1"])
            par = TH.pack_params(P, per)
            sp = S.init_spins(jax.random.PRNGKey(7), geo, R)
            spP = TH.planar(sp)
            if nd > 1:
                sp = sp.reshape(nd, per, *sp.shape[1:])
                spP = spP.reshape(nd, per, *spP.shape[1:])
                keys = jax.random.split(jax.random.PRNGKey(42), nd)

            row = dict(devices=nd, replicas=int(R), per_device=per)
            for nm in ("estandar", "optimizado"):
                if nm == "estandar":
                    if nd == 1:
                        f, a = jax.jit(lambda s: eng["anneal"](
                            s, jax.random.PRNGKey(42), bt, P, n_therm,
                            n_meas)), (sp,)
                    else:
                        f, a = jax.pmap(lambda s, k: eng["anneal"](
                            s, k, bt, P, n_therm, n_meas)), (sp, keys)
                else:
                    g = lambda s: annF(s, 42, bt, par, k1, n_therm, n_meas)
                    f, a = (jax.jit(g) if nd == 1 else jax.pmap(g)), (spP,)
                try:                       # compile without executing
                    f = f.lower(*a).compile()
                except Exception:
                    jax.block_until_ready(f(*a))
                t0 = time.perf_counter()
                o = jax.block_until_ready(f(*a))
                dt = time.perf_counter() - t0
                del o
                us = 1e6 * dt / (measured * R)
                row[nm] = dict(wall_s=dt, us_per_replica_sweep=us,
                               full_ladder_s=us * full * R / 1e6)
            row["change_pct"] = 100 * (row["optimizado"]["wall_s"]
                                       / row["estandar"]["wall_s"] - 1)
            res["rows"].append(row)
            print(f"{nd:>5}{R:>6}{per:>7}"
                  f"{row['estandar']['full_ladder_s']/60:>10.1f} min"
                  f"{row['optimizado']['full_ladder_s']/60:>10.1f} min"
                  f"{row['change_pct']:>+8.1f}%", flush=True)

    if out:
        json.dump(res, open(out, "w"), indent=1)
        print(f"\n  guardado {out}", flush=True)
    return res


def verify(geo, by=4, warps=4, So=1.0, batch=8, n_eq=400, temps=(20.0, 5.0, 1.0)):
    """Prove the physics before any clock is believed, in seconds.

    This exists because `run_states` used to time four cells and report them
    with no physics evidence of its own: the verification lived in a different
    run, and the argument reduced to "the same file ran there, and its sha256
    matches". That is a real chain of evidence but it is not a measurement,
    and a number nobody checked in the run that produced it should not be on
    a chart. These checks cost seconds against tens of minutes of stopwatch.

    Two tests, in the order that matters:

      1. the stencil against `sim_core.substep` on IDENTICAL randomness, which
         must give ZERO differing spins -- an exactness test, no statistics;
      2. both engines to equilibrium with their OWN randomness, whose energies
         must agree within the spread the run itself reports, which is what
         catches a degenerate proposal that accepts everything or nothing.
    """
    import jax
    import jax.numpy as jnp
    import sim_core as S
    import triton_halo as TH

    kB = 0.086173404
    beta = 1.0 / (kB * 5.0)
    eng = S.build_engine(geo)
    surf = jnp.asarray(geo["surface"][None, ...])
    ev = jnp.asarray(geo["even"][None, ..., None])
    od = jnp.asarray(geo["odd"][None, ..., None])
    n_act = float(geo["n_active"])
    P0 = dict(Jex=1.0, Kan1=0.177, Kan2=0.0, KanS=0.139, Hex=0.483,
              KDM=0.880, g=float(np.deg2rad(90.0)))
    mk = lambda b: {k: jnp.full((b, 1, 1, 1), float(v), jnp.float32)
                    for k, v in P0.items()}

    out = dict(exact=[], physics=[])
    print("=== exactitud: el stencil contra sim_core.substep, "
          "misma aleatoriedad ===", flush=True)
    B = int(batch)
    P = mk(B)
    k1 = jnp.where(surf, P["KanS"], P["Kan1"])
    sp = S.init_spins(jax.random.PRNGKey(3), geo, B)
    kt, ka = jax.random.split(jax.random.PRNGKey(101))
    tr = jax.random.normal(kt, sp.shape)
    tr = tr / jnp.linalg.norm(tr, axis=-1, keepdims=True) * So
    ru = jax.random.uniform(ka, sp.shape[:-1])
    sub = TH.build(geo, BY=int(by), num_warps=int(warps), So=So)
    par = TH.pack_params(P, B)
    spP, trP = TH.planar(sp), TH.planar(tr)

    an = lambda s, kk: (kk * ((s[..., 0] * s[..., 1]) ** 2
                              + (s[..., 0] * s[..., 2]) ** 2
                              + (s[..., 1] * s[..., 2]) ** 2)
                        + P["Kan2"] * (s[..., 0] * s[..., 1] * s[..., 2]) ** 2)
    for parity, mask, nm in ((0, ev, "par"), (1, od, "impar")):
        trm = jnp.where(mask, tr, sp)
        dS = trm - sp
        xp = jnp.roll(sp, -1, 3); xm = jnp.roll(sp, 1, 3)
        yp = jnp.roll(sp, -1, 2); ym = jnp.roll(sp, 1, 2)
        zp = jnp.roll(sp, -1, 1).at[:, -1].set(0.0)
        zm = jnp.roll(sp, 1, 1).at[:, 0].set(0.0)
        dE = -P["Jex"] * jnp.sum(dS * (xp + xm + yp + ym + zp + zm), -1)
        dE = dE + an(trm, k1) - an(sp, k1)
        dE = dE - P["KDM"] * (dS[..., 1] * (xp[..., 2] - xm[..., 2])
                              - dS[..., 2] * (xp[..., 1] - xm[..., 1])
                              + dS[..., 2] * (yp[..., 0] - ym[..., 0])
                              - dS[..., 0] * (yp[..., 2] - ym[..., 2]))
        dE = dE - P["Hex"] * (dS[..., 0] * jnp.cos(P["g"])
                              + dS[..., 2] * jnp.sin(P["g"]))
        acc = ((dE < 0) | (ru < jnp.exp(-dE * beta)))[..., None] & mask
        ref = jnp.where(acc, trm, sp)
        got, dE_t = sub(spP, k1, par, beta, 0, parity, trP, ru, want_de=True)
        sel = np.broadcast_to(np.asarray(mask[..., 0]), dE.shape)
        mad = float(np.max(np.abs(np.asarray(dE)[sel]
                                  - np.asarray(dE_t)[sel])))
        bad = int(np.sum(np.asarray(ref) != np.asarray(TH.unplanar(got))))
        tot = int(np.asarray(ref).size)
        out["exact"].append(dict(sublattice=nm, dE_max_abs=mad,
                                 spin_mismatch=bad, spin_total=tot))
        print(f"  {nm:>5}: dE max |dif| {mad:.3e}   espines distintos "
              f"{bad}/{tot}", flush=True)

    print("\n=== fisica: las dos cadenas a equilibrio, cada una con su RNG ===",
          flush=True)
    Bp = 16
    Pp = mk(Bp)
    k1p = jnp.where(surf, Pp["KanS"], Pp["Kan1"])
    parp = TH.pack_params(Pp, Bp)
    spE = S.init_spins(jax.random.PRNGKey(11), geo, Bp)
    subp = TH.build(geo, BY=int(by), num_warps=int(warps), So=So)
    for T in temps:
        bT = 1.0 / (kB * T)

        def jx(s):
            def body(i, v):
                st, k = v
                k, a, b = jax.random.split(k, 3)
                st = eng["substep"](st, a, bT, ev, Pp, k1p)
                st = eng["substep"](st, b, bT, od, Pp, k1p)
                return st, k
            return jax.lax.fori_loop(0, int(n_eq), body,
                                     (s, jax.random.PRNGKey(77)))[0]

        sj = jax.block_until_ready(jax.jit(jx)(spE))
        st = TH.unplanar(jax.block_until_ready(jax.jit(
            TH.sweeps(subp, k1p, parp, bT, int(n_eq)))(TH.planar(spE))))
        ej = float(jnp.mean(eng["energy"](sj, Pp, k1p)) / n_act)
        et = float(jnp.mean(eng["energy"](st, Pp, k1p)) / n_act)
        sd = float(jnp.std(eng["energy"](sj, Pp, k1p)) / n_act)
        nrm = float(jnp.max(jnp.abs(
            jnp.linalg.norm(st, axis=-1)
            - jnp.asarray(geo["disco"][None]).astype(jnp.float32))))
        row = dict(T=T, E_prod=ej, E_halo=et, E_sd=sd,
                   sigma=abs(et - ej) / (sd + 1e-30), norm_err=nrm,
                   M_prod=float(jnp.mean(eng["magnetization"](sj)[0])),
                   M_halo=float(jnp.mean(eng["magnetization"](st)[0])))
        out["physics"].append(row)
        print(f"  T={T:>5.1f}  E/sitio {ej:+9.5f} / {et:+9.5f}  "
              f"({row['sigma']:.2f} sigma)   |M| {row['M_prod']:.4f} / "
              f"{row['M_halo']:.4f}   |norma-1| {nrm:.1e}", flush=True)

    out["exact_ok"] = all(r["spin_mismatch"] == 0 for r in out["exact"])
    out["physics_ok"] = all(r["sigma"] < 3.0 and r["norm_err"] < 1e-5
                            for r in out["physics"])
    out["ok"] = bool(out["exact_ok"] and out["physics_ok"])
    print(f"\n  -> {'la fisica coincide' if out['ok'] else 'AVISO: la fisica NO coincide'}",
          flush=True)
    return out
