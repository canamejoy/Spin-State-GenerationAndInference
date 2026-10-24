"""Full 200-point annealing ladder with a chosen ablation variant, on CPU.

The ablation measured what each acceleration *costs*. This runs the affordable
cells to completion so they can be compared on *physics*: an annealing curve from
the strictly sequential chain, laid over the curve from the checkerboard kernel on
an H100, validates the whole acceleration stack end to end rather than only its
timing.

One replica. The scalar arms do not vectorise over the batch, so cost is strictly
linear in replicas and 100 replicas would be 100x the wall clock. A single chain
still gives every observable: C_v and chi come from the fluctuation of that
chain's measurement sweeps at fixed T, which is the textbook estimator and the
same formula `reduce_obs` applies per replica before averaging.

**Randomness is addressed, not sequential.** Each temperature draws from
`default_rng([seed, t_index])`, so a resumed run produces bit-identical results to
an uninterrupted one -- the property `tests/test_ladder_checkpoint.py` pins,
because a checkpoint bug here costs eleven hours.
"""

from __future__ import annotations

import time

import numpy as np

import ablation

KB = 0.086173404


def observables(s, K1_eff, p, geo):
    """Total energy, |M|/N, M_z/N and the middle layer's topological charge."""
    xp = np.roll(s, -1, 2)
    yp = np.roll(s, -1, 1)
    zp = np.roll(s, -1, 0).copy()
    zp[-1] = 0.0
    E = -p["Jex"] * np.sum(s * (xp + yp + zp), axis=-1)
    sx, sy, sz = s[..., 0], s[..., 1], s[..., 2]
    E = E + (K1_eff * ((sx * sy) ** 2 + (sx * sz) ** 2 + (sy * sz) ** 2)
             + p["Kan2"] * (sx * sy * sz) ** 2)
    E = E - p["KDM"] * (sy * xp[..., 2] - sz * xp[..., 1]
                        + sz * yp[..., 0] - sx * yp[..., 2])
    E = E - p["Hex"] * (sx * np.cos(p["g"]) + sz * np.sin(p["g"]))
    E_tot = float(np.sum(E * geo["disco"]))

    M_vec = s[geo["disco"]].sum(0) / geo["n_active"]
    return E_tot, float(np.linalg.norm(M_vec)), float(M_vec[2]), _topo_mid(s, geo)


def _topo_mid(s, geo):
    m = s[geo["L"] // 2]
    d = geo["disk2d"]
    s00, s10, s01, s11 = m[:-1, :-1], m[:-1, 1:], m[1:, :-1], m[1:, 1:]
    valid = d[:-1, :-1] & d[:-1, 1:] & d[1:, :-1] & d[1:, 1:]

    def om(a, b, c):
        num = np.sum(a * np.cross(b, c), axis=-1)
        den = 1.0 + np.sum(a * b, -1) + np.sum(b * c, -1) + np.sum(c * a, -1)
        return 2.0 * np.arctan2(num, den)

    w = om(s00, s10, s11) + om(s00, s11, s01)
    return float(np.sum(np.where(valid, w, 0.0)) / (4.0 * np.pi))


def _variant(cell, geo):
    """The sweep for one ablation cell, plus whatever it needs built once."""
    if cell == "dp_bp":
        masks = (geo["even"], geo["odd"])
        def sweep(s, pr, u, beta, p, K1, ctx):
            return ablation.sweep_vectorised(s, pr, u, beta, p, K1, geo, masks)
        return sweep, None
    order = (ablation.colour_order(geo) if cell == "nodp_bp"
             else ablation.lattice_order(geo))
    idx = ablation._flat_indices(order, geo["N"])

    def sweep(s, pr, u, beta, p, K1, ctx):
        sx, sy, sz = ablation._flatten(s, geo)
        prf = pr.reshape(-1, 3)
        ablation._sweep_flat(sx, sy, sz, ctx, idx,
                             prf[idx, 0].tolist(), prf[idx, 1].tolist(),
                             prf[idx, 2].tolist(), u.ravel()[idx].tolist(), beta)
        return ablation._unflatten(sx, sy, sz, geo)

    return sweep, "scalar"


# Two namespaces off one seed: 0 for the initial configuration, 1 for the
# temperature streams. A seed sequence takes only non-negative integers, so the
# namespace cannot be a sentinel like -1.
_NS_INIT, _NS_TEMP = 0, 1


def _rng_for(seed, t_index):
    """Per-temperature stream, addressed by index so a resume is exact."""
    return np.random.default_rng([int(seed), _NS_TEMP, int(t_index)])


def _draw(rng, geo):
    pr = rng.normal(size=(geo["L"], geo["N"], geo["N"], 3))
    pr /= np.linalg.norm(pr, axis=-1, keepdims=True)
    pr *= geo["disco"][..., None]
    return pr, rng.random((geo["L"], geo["N"], geo["N"]))


def run_ladder(cell, geo, params, temps, n_therm, n_meas, seed=42,
               checkpoint=None, save=None, log=print):
    """Anneal down `temps`, accumulating observables, checkpointing per step.

    `checkpoint` is a dict from a previous run (or None to start fresh); `save`
    is called with the updated checkpoint after every temperature so an
    interrupted run resumes from the last completed step rather than the start.
    """
    L, N = geo["L"], geo["N"]
    K1 = np.where(geo["surface"], params["KanS"], params["Kan1"])
    sweep, needs = _variant(cell, geo)
    ctx = ablation.prepare_scalar(geo, params, K1) if needs else None

    if checkpoint is None:
        rng0 = np.random.default_rng([int(seed), _NS_INIT, 0])
        s = rng0.normal(size=(L, N, N, 3))
        s /= np.linalg.norm(s, axis=-1, keepdims=True)
        s *= geo["disco"][..., None]
        start, rows, snaps, elapsed = 0, [], [], 0.0
    else:
        s = checkpoint["spins"].astype(np.float64)
        start = int(checkpoint["t_done"])
        rows = [dict(r) for r in checkpoint["rows"]]
        snaps = list(checkpoint["snaps"])
        elapsed = float(checkpoint["elapsed_s"])
        log(f"resuming at T index {start}/{len(temps)} "
            f"({elapsed/3600:.2f} h already spent)")

    for ti in range(start, len(temps)):
        T = float(temps[ti])
        beta = 1.0 / (KB * T)
        rng = _rng_for(seed, ti)
        t0 = time.perf_counter()

        for _ in range(n_therm):
            pr, u = _draw(rng, geo)
            s = sweep(s, pr, u, beta, params, K1, ctx)

        # Reference subtraction, as in the GPU pipeline: accumulating raw E and
        # E^2 loses the variance to cancellation. A constant shift leaves the
        # variance unchanged, so this is exact, and it keeps the two paths
        # computing C_v and chi the same way.
        E_ref, M_ref, _, _ = observables(s, K1, params, geo)
        acc = np.zeros(6)
        for _ in range(n_meas):
            pr, u = _draw(rng, geo)
            s = sweep(s, pr, u, beta, params, K1, ctx)
            E, M, Mz, Q = observables(s, K1, params, geo)
            dE, dM = E - E_ref, M - M_ref
            acc += (dE, dE * dE, dM, dM * dM, Mz, Q)
        acc /= n_meas
        dE, dE2, dM, dM2, Mz, Q = acc
        n = geo["n_active"]
        rows.append(dict(T=T, E=(dE + E_ref) / n, M=dM + M_ref, Mz=Mz, Q=Q,
                         Cv=(dE2 - dE ** 2) / (KB * T ** 2 * n),
                         chi=(dM2 - dM ** 2) / (KB * T)))
        snaps.append(s[L // 2, :, :, 2].astype(np.float32))
        elapsed += time.perf_counter() - t0

        if save is not None:
            # float64, not float32: the whole configuration is 182 kB, and
            # rounding it makes a resumed chain diverge from an uninterrupted
            # one in the eighth digit -- paying exactness for nothing.
            save(dict(spins=s.copy(), t_done=ti + 1, rows=rows,
                      snaps=snaps, elapsed_s=elapsed, cell=cell))
        if ti % 5 == 0 or ti == len(temps) - 1:
            sweeps = (ti + 1 - start) * (n_therm + n_meas)
            rate = 1000 * (elapsed / max(sweeps, 1))
            left = (len(temps) - ti - 1) * (n_therm + n_meas) * rate / 3.6e6
            log(f"  T[{ti:3d}] = {T:6.2f}  E/N={rows[-1]['E']:9.5f}  "
                f"|M|={rows[-1]['M']:.4f}  Q={rows[-1]['Q']:+7.3f}  "
                f"{rate:.2f} ms/sweep  ~{left:.1f} h left")

    total_sweeps = len(temps) * (n_therm + n_meas)
    return dict(cell=cell, rows=rows, snaps=np.asarray(snaps),
                final=s.astype(np.float32), elapsed_s=elapsed,
                ms_per_sweep=1000 * elapsed / total_sweeps,
                n_active=geo["n_active"], replicas=1,
                n_temps=len(temps), n_therm=n_therm, n_meas=n_meas)
