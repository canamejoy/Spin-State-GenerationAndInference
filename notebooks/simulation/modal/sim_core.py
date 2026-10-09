"""Monte Carlo engine for the nanodisk, faithful to the project's reference notebook.

Taken from `00_benchmark_mc_observables.ipynb` -- the same checkerboard, the same
uniform-sphere proposal, and in particular the same DMI: **in-plane bonds only
(x and y), with a minus sign**. That is interfacial DMI, the physics of a thin
film whose DM vector comes from an interface, and it is not interchangeable with
the all-axes form: on the same configuration the total energy differs by 50%.

Two terms the reference notebook does not carry are added here because the
tutor's dataset varies them:

- `KanS`, the surface anisotropy, applied where a site has fewer than six
  in-disk nearest neighbours (the coordination test used in the group's
  multi-GPU notebooks).
- `Kan2`, read as the SECOND-ORDER constant of the cubic anisotropy expansion,

      E_an = (Kan1 / So^4)(sx^2 sy^2 + sx^2 sz^2 + sy^2 sz^2)
           + (Kan2 / So^6)(sx^2 sy^2 sz^2)

  which is the textbook K1/K2 series and matches the naming. It is a single-site
  term, so unlike a second-neighbour coupling it leaves the checkerboard exact.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import random


def build_geometry(Rd: float = 18.30, L: int = 5):
    """Disk mask, checkerboard sublattices and the surface mask."""
    L2 = int(Rd) + 1
    N = 2 * L2 + 1
    x = np.arange(-L2, L2 + 1)
    xg, yg = np.meshgrid(x, x, indexing="xy")
    disk2d = (xg ** 2 + yg ** 2) <= Rd ** 2
    disco = np.broadcast_to(disk2d, (L, N, N)).copy()

    zc, yc, xc = np.indices((L, N, N))
    checker = (xc + yc + zc) % 2
    even = (checker == 0) & disco
    odd = (checker == 1) & disco

    # Surface = fewer than six in-disk nearest neighbours. x and y wrap, z does
    # not, so the top and bottom layers are surface everywhere.
    surface = np.zeros_like(disco, dtype=bool)
    for z in range(L):
        for y in range(N):
            for xx in range(N):
                if not disco[z, y, xx]:
                    continue
                n = 0
                for dz, dy, dx in ((0, 0, 1), (0, 0, -1), (0, 1, 0),
                                   (0, -1, 0), (1, 0, 0), (-1, 0, 0)):
                    nz, ny, nx = z + dz, (y + dy) % N, (xx + dx) % N
                    if 0 <= nz < L and disco[nz, ny, nx]:
                        n += 1
                if n < 6:
                    surface[z, y, xx] = True
    return dict(L=L, N=N, Rd=Rd, disco=disco, even=even, odd=odd,
                surface=surface, disk2d=disk2d, n_active=int(disco.sum()),
                n_surface=int(surface.sum()))


def build_engine(geo, So=1.0):
    """Returns (substep, energy, magnetization, topological_charge, anneal)."""
    L, N = geo["L"], geo["N"]
    even_j = jnp.asarray(geo["even"][None, ..., None])
    odd_j = jnp.asarray(geo["odd"][None, ..., None])
    disco_j = jnp.asarray(geo["disco"][None, ..., None])
    surf_j = jnp.asarray(geo["surface"][None, ...])
    disk2d_j = jnp.asarray(geo["disk2d"])
    n_act = geo["n_active"]

    def _an(s, Kan1, Kan2, K1_eff):
        """Cubic anisotropy: the K1 invariant plus the K2 (sixth-order) one."""
        sx, sy, sz = s[..., 0], s[..., 1], s[..., 2]
        e = (K1_eff / So ** 4) * ((sx * sy) ** 2 + (sx * sz) ** 2 + (sy * sz) ** 2)
        return e + (Kan2 / So ** 6) * (sx * sy * sz) ** 2

    def substep(spins, key, beta, mask, p, K1_eff):
        S_xp = jnp.roll(spins, -1, 3); S_xm = jnp.roll(spins, 1, 3)
        S_yp = jnp.roll(spins, -1, 2); S_ym = jnp.roll(spins, 1, 2)
        S_zp = jnp.roll(spins, -1, 1).at[:, -1].set(0.0)
        S_zm = jnp.roll(spins, 1, 1).at[:, 0].set(0.0)

        kt, ka = random.split(key)
        tr = random.normal(kt, spins.shape)
        tr = tr / jnp.linalg.norm(tr, axis=-1, keepdims=True) * So
        tr = jnp.where(mask, tr, spins)
        dS = tr - spins

        sv = S_xp + S_xm + S_yp + S_ym + S_zp + S_zm
        dE = -p["Jex"] * jnp.sum(dS * sv, axis=-1)
        dE = dE + _an(tr, p["Kan1"], p["Kan2"], K1_eff) - _an(spins, p["Kan1"], p["Kan2"], K1_eff)
        # Interfacial DMI: in-plane bonds only, minus sign. See module docstring.
        dE = dE - p["KDM"] * (dS[..., 1] * (S_xp[..., 2] - S_xm[..., 2])
                              - dS[..., 2] * (S_xp[..., 1] - S_xm[..., 1])
                              + dS[..., 2] * (S_yp[..., 0] - S_ym[..., 0])
                              - dS[..., 0] * (S_yp[..., 2] - S_ym[..., 2]))
        dE = dE - p["Hex"] * (dS[..., 0] * jnp.cos(p["g"]) + dS[..., 2] * jnp.sin(p["g"]))

        r = random.uniform(ka, dE.shape)
        acc = ((dE < 0) | (r < jnp.exp(-dE * beta)))[..., None] & mask
        return jnp.where(acc, tr, spins)

    def energy(spins, p, K1_eff):
        """Total energy; each bond counted once via the +x, +y, +z shifts only."""
        S_xp = jnp.roll(spins, -1, 3)
        S_yp = jnp.roll(spins, -1, 2)
        S_zp = jnp.roll(spins, -1, 1).at[:, -1].set(0.0)
        E = -p["Jex"] * jnp.sum(spins * (S_xp + S_yp + S_zp), axis=-1)
        sx, sy, sz = spins[..., 0], spins[..., 1], spins[..., 2]
        E = E + _an(spins, p["Kan1"], p["Kan2"], K1_eff)
        E = E - p["KDM"] * (sy * S_xp[..., 2] - sz * S_xp[..., 1]
                            + sz * S_yp[..., 0] - sx * S_yp[..., 2])
        E = E - p["Hex"] * (sx * jnp.cos(p["g"]) + sz * jnp.sin(p["g"]))
        return jnp.sum(E * disco_j[..., 0], axis=(-3, -2, -1))

    def magnetization(spins):
        M = jnp.sum(spins * disco_j, axis=(1, 2, 3)) / n_act
        return jnp.linalg.norm(M, axis=-1), M[:, 2]

    def _omega(s1, s2, s3):
        num = jnp.sum(s1 * jnp.cross(s2, s3), axis=-1)
        den = (1.0 + jnp.sum(s1 * s2, -1) + jnp.sum(s2 * s3, -1) + jnp.sum(s3 * s1, -1))
        return 2.0 * jnp.arctan2(num, den)

    def topo_charge(spins):
        """Berg-Luscher charge of every layer: quantised, unlike a finite difference."""
        s00 = spins[..., :-1, :-1, :]; s10 = spins[..., :-1, 1:, :]
        s01 = spins[..., 1:, :-1, :];  s11 = spins[..., 1:, 1:, :]
        valid = (disk2d_j[:-1, :-1] & disk2d_j[:-1, 1:]
                 & disk2d_j[1:, :-1] & disk2d_j[1:, 1:])
        w = _omega(s00, s10, s11) + _omega(s00, s11, s01)
        return jnp.sum(jnp.where(valid, w, 0.0), axis=(-2, -1)) / (4.0 * jnp.pi)

    def anneal(spins, key, betas, p, n_therm, n_meas):
        K1_eff = jnp.where(surf_j, p["KanS"], p["Kan1"])

        def per_T(carry, beta):
            s, k = carry

            def therm(_, v):
                s, k = v
                k, k1, k2 = random.split(k, 3)
                s = substep(s, k1, beta, even_j, p, K1_eff)
                s = substep(s, k2, beta, odd_j, p, K1_eff)
                return s, k
            s, k = jax.lax.fori_loop(0, n_therm, therm, (s, k))

            b = s.shape[0]
            z0 = jnp.zeros(b)

            # C_v and chi are variances, and accumulating raw E and E^2
            # destroys them in float32. At T = 0.2 this run has <E> ~ -1.7e4, so
            # <E^2> ~ 2.9e8, where one float32 ulp is ~34 -- larger than the
            # variance itself. <E^2> - <E>^2 then comes out NEGATIVE and C_v
            # reads -437: not a small error but pure cancellation.
            #
            # Measuring the fluctuation about a reference taken once, before the
            # loop, keeps every accumulated quantity at the scale of the
            # fluctuation rather than of the total. A constant shift leaves the
            # variance unchanged, so this is exact. Both the centred moments and
            # the reference are returned, and `reduce_obs` recombines them.
            E_ref = energy(s, p, K1_eff)
            M_ref, _ = magnetization(s)

            def meas(_, v):
                s, k, sE, sE2, sM, sM2, sMz, sQ = v
                k, k1, k2 = random.split(k, 3)
                s = substep(s, k1, beta, even_j, p, K1_eff)
                s = substep(s, k2, beta, odd_j, p, K1_eff)
                dE = energy(s, p, K1_eff) - E_ref
                M, Mz = magnetization(s)
                dM = M - M_ref
                Q = topo_charge(s)[:, L // 2]
                return (s, k, sE + dE, sE2 + dE * dE,
                        sM + dM, sM2 + dM * dM, sMz + Mz, sQ + Q)

            s, k, sE, sE2, sM, sM2, sMz, sQ = jax.lax.fori_loop(
                0, n_meas, meas, (s, k, z0, z0, z0, z0, z0, z0))
            obs = jnp.stack([sE / n_meas, sE2 / n_meas,
                             sM / n_meas, sM2 / n_meas,
                             sMz / n_meas, sQ / n_meas, E_ref, M_ref])
            # One middle-layer snapshot per temperature, replica 0 only. The
            # whole batch would be 30 x 100 x 5 x 39 x 39 x 3 floats (~2.7 GB);
            # one replica's middle layer is 182 kB and is what the state-
            # transition figure needs.
            snap = s[0, L // 2, :, :, 2]
            return (s, k), (obs, snap)

        (s, _), (all_obs, snaps) = jax.lax.scan(per_T, (spins, key), betas)
        return all_obs, s, snaps

    return dict(substep=substep, energy=energy, magnetization=magnetization,
                topo_charge=topo_charge, anneal=anneal)


def init_spins(key, geo, batch, So=1.0):
    L, N = geo["L"], geo["N"]
    x = random.normal(key, (batch, L, N, N, 3))
    x = x / jnp.linalg.norm(x, axis=-1, keepdims=True) * So
    return x * jnp.asarray(geo["disco"][None, ..., None])


def reduce_obs(obs, temps, n_spins, kB=0.086173404):
    """Accumulators -> per-temperature scalars.

    `lax.scan` stacks each step's output along a NEW LEADING axis, so `anneal`
    returns `(n_T, 6, batch)` and the observable index is the SECOND axis. The
    obvious `obs[i]` indexes temperatures instead, silently mixing the two --
    and a test with six temperatures and six observables cannot tell the
    difference, since both shapes read `(6, 6, batch)`.
    """
    obs = np.asarray(obs)
    assert obs.ndim == 3 and obs.shape[1] == 8, (
        f"expected (n_T, 8, batch) from lax.scan, got {obs.shape}")
    dE, dE2, dM, dM2, Mz, Q, E_ref, M_ref = [obs[:, i] for i in range(8)]
    T = np.asarray(temps)[:, None]
    # Variances of the CENTRED quantities: both terms sit at the scale of the
    # fluctuation, so there is nothing to cancel.
    var_E = dE2 - dE ** 2
    var_M = dM2 - dM ** 2
    E = dE + E_ref
    M = dM + M_ref
    Cv = var_E / (kB * T ** 2 * n_spins)
    chi = var_M / (kB * T)
    return dict(T=np.asarray(temps),
                E=E.mean(1) / n_spins, E_sd=E.std(1) / n_spins,
                M=M.mean(1), M_sd=M.std(1),
                Mz=Mz.mean(1), Cv=Cv.mean(1), chi=chi.mean(1),
                Q=Q.mean(1), Q_sd=Q.std(1))
