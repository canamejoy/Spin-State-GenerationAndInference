"""Measure the four `DP x BP` cells, and the cost of getting `BP` wrong.

Run at the production geometry (`Rd=18.30`, 5 layers, 5,245 active sites) with
complete lattice sweeps in every cell -- no reduced configuration and no pro
rata scaling. The test plan assumed the un-vectorised arm would need both, on a
per-site cost of 10 us that turned out to be dominated by per-sweep setup the
timed loop should never have carried.

Each cell is repeated with independent seeds and reported as mean +- sd, because
the GPU arm of this study showed ~15% run-to-run scatter and a single number
would invite reading differences that are noise.

The bias section answers the question the timing cannot: the fast-and-wrong
reading of "drop the checkerboard" -- update every site at once -- is compared
against the exact chain on the same observable, so the checkerboard's
contribution is stated as a measured error rather than as an argument.

    python run_ablation.py [--out ablation_results.json] [--quick]
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np

import ablation
import sim_core

PARAMS = dict(Jex=1.0, Kan1=0.177, Kan2=0.034, KanS=0.139, Hex=0.483,
              KDM=0.880, g=float(np.deg2rad(90.0)))
KB = 0.086173404

# Sweep counts chosen so every cell runs for at least ~3 s. A 0.01 s
# measurement is timer noise, which is how the first pass at this came out
# reporting the two scalar cells as 43% apart when they do identical work.
SWEEPS = {"dp_bp": 3000, "nodp_bp": 400, "nodp_nobp": 400, "dp_nobp": 6}
QUICK = {"dp_bp": 200, "nodp_bp": 30, "nodp_nobp": 30, "dp_nobp": 2}


def timing(geo, K1, sweeps, repeats=3):
    rows = []
    for cell in ("dp_bp", "nodp_bp", "nodp_nobp", "dp_nobp"):
        runs = [ablation.time_cell(cell, geo, PARAMS, K1, beta=1.0 / (KB * 3.0),
                                   seed=100 + r, sweeps=sweeps[cell])
                for r in range(repeats)]
        ms = np.array([r["ms_per_sweep"] for r in runs])
        rows.append(dict(cell=cell, ms_per_sweep=float(ms.mean()),
                         ms_sd=float(ms.std()),
                         us_per_site=float(np.mean([r["us_per_site"] for r in runs])),
                         sweeps=sweeps[cell], repeats=repeats,
                         wall_s=float(sum(r["wall_s"] for r in runs))))
        print(f"  {cell:11s} {ms.mean():10.3f} +- {ms.std():6.3f} ms/sweep "
              f"({rows[-1]['us_per_site']:7.3f} us/site)", flush=True)
    return rows


def bias(geo, K1, temps=(1.0, 3.0, 8.0), sweeps=2000, burn=1000, seed=7):
    """Observables from the exact chain and from the un-coloured one.

    Same geometry, same temperatures, same number of sweeps, same starting
    configuration; the only difference is whether neighbours are allowed to
    decide against each other's stale spins.
    """
    L, N = geo["L"], geo["N"]
    out = []
    for T in temps:
        beta = 1.0 / (KB * T)
        row = dict(T=float(T))
        for label, masks in (("checkerboard", (geo["even"], geo["odd"])),
                             ("uncoloured", (geo["disco"],))):
            rng = np.random.default_rng(seed)
            s = rng.normal(size=(L, N, N, 3))
            s /= np.linalg.norm(s, axis=-1, keepdims=True)
            s *= geo["disco"][..., None]
            acc_E, acc_M, acc_Q = [], [], []
            for it in range(sweeps):
                pr, u = ablation._randomness(rng, geo)
                s = ablation.sweep_vectorised(s, pr, u, beta, PARAMS, K1, geo,
                                              masks=masks)
                if it >= burn:
                    acc_E.append(_energy(s, K1, geo))
                    acc_M.append(float(np.linalg.norm(
                        s[geo["disco"]].sum(0) / geo["n_active"])))
                    acc_Q.append(_topo_mid(s, geo))
            E = np.array(acc_E) / geo["n_active"]
            row[label] = dict(E=float(E.mean()), E_sd=float(E.std()),
                              M=float(np.mean(acc_M)),
                              Cv=float(E.var() * geo["n_active"]
                                       / (KB * T ** 2)),
                              Q=float(np.mean(acc_Q)),
                              Q_sd=float(np.std(acc_Q)))
        a, b = row["checkerboard"], row["uncoloured"]
        row["dE_percent"] = 100.0 * (b["E"] - a["E"]) / abs(a["E"])
        row["dM_percent"] = 100.0 * (b["M"] - a["M"]) / abs(a["M"])
        row["E_sigma"] = abs(b["E"] - a["E"]) / (a["E_sd"] + 1e-30)
        row["dQ"] = b["Q"] - a["Q"]
        print(f"  T={T:4.1f}  E {a['E']:9.5f} -> {b['E']:9.5f} "
              f"({row['dE_percent']:+7.2f}%, {row['E_sigma']:7.1f} sigma)  "
              f"|M| {a['M']:.4f} -> {b['M']:.4f} ({row['dM_percent']:+6.2f}%)  "
              f"Q {a['Q']:+6.3f} -> {b['Q']:+6.3f}",
              flush=True)
        out.append(row)
    return out


def _topo_mid(s, geo):
    """Berg-Luscher charge of the middle layer: quantised, unlike a difference.

    The dataset's own state labels are skyrmion counts, so an acceleration that
    shifted this would be changing the thing the paper reports, not just the
    thermodynamics.
    """
    m = s[geo["L"] // 2]
    d = geo["disk2d"]
    s00, s10 = m[:-1, :-1], m[:-1, 1:]
    s01, s11 = m[1:, :-1], m[1:, 1:]
    valid = d[:-1, :-1] & d[:-1, 1:] & d[1:, :-1] & d[1:, 1:]

    def om(a, b, c):
        num = np.sum(a * np.cross(b, c), axis=-1)
        den = (1.0 + np.sum(a * b, -1) + np.sum(b * c, -1) + np.sum(c * a, -1))
        return 2.0 * np.arctan2(num, den)

    w = om(s00, s10, s11) + om(s00, s11, s01)
    return float(np.sum(np.where(valid, w, 0.0)) / (4.0 * np.pi))


def _energy(s, K1, geo):
    """Total energy, the same expression the engine uses, in numpy."""
    xp = np.roll(s, -1, 2)
    yp = np.roll(s, -1, 1)
    zp = np.roll(s, -1, 0).copy()
    zp[-1] = 0.0
    E = -PARAMS["Jex"] * np.sum(s * (xp + yp + zp), axis=-1)
    sx, sy, sz = s[..., 0], s[..., 1], s[..., 2]
    E += (K1 * ((sx * sy) ** 2 + (sx * sz) ** 2 + (sy * sz) ** 2)
          + PARAMS["Kan2"] * (sx * sy * sz) ** 2)
    E -= PARAMS["KDM"] * (sy * xp[..., 2] - sz * xp[..., 1]
                          + sz * yp[..., 0] - sx * yp[..., 2])
    E -= PARAMS["Hex"] * (sx * np.cos(PARAMS["g"]) + sz * np.sin(PARAMS["g"]))
    return float(np.sum(E * geo["disco"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ablation_results.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()

    geo = sim_core.build_geometry(Rd=18.30, L=5)
    K1 = np.where(geo["surface"], PARAMS["KanS"], PARAMS["Kan1"])
    print(f"geometry N={geo['N']} active={geo['n_active']} "
          f"surface={geo['n_surface']}", flush=True)

    t0 = time.perf_counter()
    print("timing the four cells:", flush=True)
    rows = timing(geo, K1, QUICK if a.quick else SWEEPS, a.repeats)
    print("bias of dropping the colouring:", flush=True)
    bi = bias(geo, K1, sweeps=200 if a.quick else 2000,
              burn=100 if a.quick else 1000)

    by = {r["cell"]: r["ms_per_sweep"] for r in rows}
    speedups = dict(
        # What matrix decomposition buys, with the colouring already in place.
        dp_given_bp=by["nodp_bp"] / by["dp_bp"],
        # What the colouring buys in speed, with the decomposition in place:
        # nothing, because a sequential chain cannot use a vectorised energy.
        bp_given_dp=by["dp_nobp"] / by["dp_bp"],
        # What the colouring buys in speed on its own.
        bp_given_nodp=by["nodp_nobp"] / by["nodp_bp"],
        # What the pair buys over a textbook sequential sweep. This, not the
        # `dp_nobp` ratio, is the defensible headline: `dp_nobp` builds a
        # whole-lattice field per site and is nobody's baseline.
        overall_vs_textbook=by["nodp_nobp"] / by["dp_bp"],
        degenerate_cell_vs_dp_bp=by["dp_nobp"] / by["dp_bp"])
    for k, v in speedups.items():
        print(f"  {k:16s} {v:9.2f}x", flush=True)

    res = dict(geometry=dict(Rd=18.30, L=5, N=geo["N"],
                             n_active=geo["n_active"],
                             n_surface=geo["n_surface"]),
               params=PARAMS, backend="numpy float64, single thread",
               machine=dict(processor=platform.processor(),
                            python=platform.python_version(),
                            numpy=np.__version__, platform=platform.platform()),
               cells=rows, speedups=speedups, bias=bi,
               total_wall_s=time.perf_counter() - t0)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {a.out} in {res['total_wall_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
