"""The Monte Carlo engine generalised to the first four exchange shells.

`sim_core` carries first-shell exchange only, and for that case a two-colour
checkerboard is an EXACT reordering of the sequential sweep: every first
neighbour sits at an odd displacement, so it always has the opposite parity of
``(x+y+z)``. Turning on a shell whose displacements sum to an even number breaks
that, and the fix is not to forbid the shell -- it is to use more sublattices.

The minimal colouring is decided by which shells are active, and it is VERIFIED
here rather than asserted (see `check_colouring`):

    active shells        colouring                     sublattices
    {1}, {1,3}           (x+y+z) mod 2                      2
    anything with 2      (x mod 2, y mod 2, z mod 2)        8
    anything with 4      (x mod 3, y mod 3, z mod 3)       27

Shells 1 and 3 have odd displacement sums and keep the graph bipartite; shells
2 and 4 do not. Shell 4 reaches distance 2 along an axis, which no stride-2
colouring can separate, so it needs stride 3 on all three axes.

The cost of a wider colouring is immediate and worth stating: a sweep becomes
one pass per colour, and each pass updates only 1/n of the sites. Going from
two sublattices to twenty-seven does not make a sweep thirteen times slower --
the work per sweep is the same -- but it does cut the parallelism available in
each pass by that factor, which on a GPU is what the cost follows.

The DMI term stays on first-shell in-plane bonds whatever else is switched on:
it is interfacial by construction, not a generic chiral coupling.
"""

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

# The displacement vectors of each shell on a simple cubic lattice, by
# squared distance: 1, 2, 3 and 4.
_SHELL = {
    1: [d for d in itertools.product((-1, 0, 1), repeat=3)
        if sum(x * x for x in d) == 1],
    2: [d for d in itertools.product((-1, 0, 1), repeat=3)
        if sum(x * x for x in d) == 2],
    3: [d for d in itertools.product((-1, 0, 1), repeat=3)
        if sum(x * x for x in d) == 3],
    4: [d for d in itertools.product((-2, 0, 2), repeat=3)
        if sum(x * x for x in d) == 4],
}

_COLOURINGS = [
    ("tablero", 2, lambda x, y, z: (x + y + z) % 2),
    ("paridad por eje", 8, lambda x, y, z: (x % 2) * 4 + (y % 2) * 2 + z % 2),
    ("paso 3 por eje", 27, lambda x, y, z: (x % 3) * 9 + (y % 3) * 3 + z % 3),
]


def shells_of(Jex2=0.0, Jex3=0.0, Jex4=0.0):
    """Which shells are active. The first is always on and normalised to 1."""
    out = [1]
    for k, v in ((2, Jex2), (3, Jex3), (4, Jex4)):
        if abs(float(v)) > 0.0:
            out.append(k)
    return out


def check_colouring(colour, shells, extent=6):
    """True when no displacement of any active shell preserves the colour.

    Checked over a window of the lattice rather than argued from parity, so a
    colouring that happens to fail somewhere is caught here and not in the
    physics.
    """
    for k in shells:
        for d in _SHELL[k]:
            for p in itertools.product(range(extent), repeat=3):
                q = tuple(p[i] + d[i] for i in range(3))
                if colour(*p) == colour(*q):
                    return False
    return True


def pick_colouring(shells):
    """The narrowest verified colouring for those shells."""
    for name, n, f in _COLOURINGS:
        if check_colouring(f, shells):
            return name, n, f
    raise ValueError(f"ninguno de los coloreos conocidos sirve para {shells}")


def build_geometry(Rd: float = 18.30, L: int = 5, shells=(1,)):
    """Disc mask, the per-shell neighbour offsets, and the sublattice masks."""
    shells = tuple(sorted(set(int(s) for s in shells)))
    L2 = int(Rd) + 1
    N = 2 * L2 + 1
    x = np.arange(-L2, L2 + 1)
    xg, yg = np.meshgrid(x, x, indexing="xy")
    disk2d = (xg ** 2 + yg ** 2) <= Rd ** 2
    disco = np.broadcast_to(disk2d, (L, N, N)).copy()

    name, ncol, f = pick_colouring(shells)
    zc, yc, xc = np.indices((L, N, N))
    colour = f(xc, yc, zc)
    masks = [((colour == c) & disco) for c in range(ncol)]
    # A colour class empty inside the disc would be a wasted pass.
    masks = [m for m in masks if m.any()]

    # Surface = fewer than six in-disc FIRST neighbours, as in `sim_core`.
    surface = np.zeros_like(disco, dtype=bool)
    for z in range(L):
        for yy in range(N):
            for xx in range(N):
                if not disco[z, yy, xx]:
                    continue
                n = 0
                for dz, dy, dx in _SHELL[1]:
                    nz, ny, nx = z + dz, (yy + dy) % N, (xx + dx) % N
                    if 0 <= nz < L and disco[nz, ny, nx]:
                        n += 1
                if n < 6:
                    surface[z, yy, xx] = True

    return dict(L=L, N=N, Rd=Rd, disco=disco, surface=surface, disk2d=disk2d,
                shells=shells, colouring=name, n_sublattices=len(masks),
                masks=masks, n_active=int(disco.sum()),
                n_surface=int(surface.sum()))


def _shift(spins, d, L):
    """One neighbour translate. x and y wrap; z is open, so it is zeroed."""
    dz, dy, dx = d
    out = spins
    if dx:
        out = jnp.roll(out, -dx, 3)
    if dy:
        out = jnp.roll(out, -dy, 2)
    if dz:
        out = jnp.roll(out, -dz, 1)
        # the |dz| layers that rolled in from the other end have no neighbour
        if dz > 0:
            out = out.at[:, L - dz:].set(0.0)
        else:
            out = out.at[:, :(-dz)].set(0.0)
    return out


def build_engine(geo, So=1.0):
    """Returns (substep, energy, magnetization, topo_charge, anneal).

    `substep` takes the index of the sublattice to update rather than a mask,
    so the caller loops over `geo["n_sublattices"]` instead of over two
    hardcoded colours.
    """
    L, N = geo["L"], geo["N"]
    shells = geo["shells"]
    disco_j = jnp.asarray(geo["disco"][None, ..., None])
    surf_j = jnp.asarray(geo["surface"][None, ...])
    disk2d_j = jnp.asarray(geo["disk2d"])
    masks = [jnp.asarray(m[None, ..., None]) for m in geo["masks"]]
    n_act = geo["n_active"]

    def _an(s, Kan2, K1_eff):
        sx, sy, sz = s[..., 0], s[..., 1], s[..., 2]
        e = (K1_eff / So ** 4) * ((sx * sy) ** 2 + (sx * sz) ** 2
                                  + (sy * sz) ** 2)
        return e + (Kan2 / So ** 6) * (sx * sy * sz) ** 2

    def _coupling(p, k):
        """The exchange constant of shell k. The first is normalised to 1."""
        return p["Jex"] if k == 1 else p[f"Jex{k}"]

    def substep(spins, key, beta, sub, p, K1_eff):
        mask = masks[sub]
        # the weighted neighbour sum over every active shell
        sv = 0.0
        for k in shells:
            acc = 0.0
            for d in _SHELL[k]:
                acc = acc + _shift(spins, d, L)
            # the coupling columns are (B,1,1,1) and `acc` carries the
            # component axis, so the coupling needs a trailing axis to broadcast
            sv = sv + _coupling(p, k)[..., None] * acc

        kt, ka = random.split(key)
        tr = random.normal(kt, spins.shape)
        tr = tr / jnp.linalg.norm(tr, axis=-1, keepdims=True) * So
        tr = jnp.where(mask, tr, spins)
        dS = tr - spins

        dE = -jnp.sum(dS * sv, axis=-1)
        dE = dE + _an(tr, p["Kan2"], K1_eff) - _an(spins, p["Kan2"], K1_eff)
        # Interfacial DMI: first-shell in-plane bonds only, whatever else is on.
        xp = _shift(spins, (0, 0, 1), L); xm = _shift(spins, (0, 0, -1), L)
        yp = _shift(spins, (0, 1, 0), L); ym = _shift(spins, (0, -1, 0), L)
        dE = dE - p["KDM"] * (dS[..., 1] * (xp[..., 2] - xm[..., 2])
                              - dS[..., 2] * (xp[..., 1] - xm[..., 1])
                              + dS[..., 2] * (yp[..., 0] - ym[..., 0])
                              - dS[..., 0] * (yp[..., 2] - ym[..., 2]))
        dE = dE - p["Hex"] * (dS[..., 0] * jnp.cos(p["g"])
                              + dS[..., 2] * jnp.sin(p["g"]))

        r = random.uniform(ka, dE.shape)
        acc = ((dE < 0) | (r < jnp.exp(-dE * beta)))[..., None] & mask
        return jnp.where(acc, tr, spins)

    def energy(spins, p, K1_eff):
        """Total energy, each bond counted once: only the positive half of each
        shell's displacement set is summed."""
        E = 0.0
        for k in shells:
            half = [d for d in _SHELL[k] if d > (0, 0, 0)]
            acc = 0.0
            for d in half:
                acc = acc + _shift(spins, d, L)
            E = E - _coupling(p, k) * jnp.sum(spins * acc, axis=-1)
        sx, sy, sz = spins[..., 0], spins[..., 1], spins[..., 2]
        E = E + _an(spins, p["Kan2"], K1_eff)
        xp = _shift(spins, (0, 0, 1), L); yp = _shift(spins, (0, 1, 0), L)
        E = E - p["KDM"] * (sy * xp[..., 2] - sz * xp[..., 1]
                            + sz * yp[..., 0] - sx * yp[..., 2])
        E = E - p["Hex"] * (sx * jnp.cos(p["g"]) + sz * jnp.sin(p["g"]))
        return jnp.sum(E * disco_j[..., 0], axis=(-3, -2, -1))

    def magnetization(spins):
        M = jnp.sum(spins * disco_j, axis=(1, 2, 3)) / n_act
        return jnp.linalg.norm(M, axis=-1), M[:, 2]

    def _omega(s1, s2, s3):
        num = jnp.sum(s1 * jnp.cross(s2, s3), axis=-1)
        den = (1.0 + jnp.sum(s1 * s2, -1) + jnp.sum(s2 * s3, -1)
               + jnp.sum(s3 * s1, -1))
        return 2.0 * jnp.arctan2(num, den)

    def topo_charge(spins):
        s00 = spins[..., :-1, :-1, :]; s10 = spins[..., :-1, 1:, :]
        s01 = spins[..., 1:, :-1, :];  s11 = spins[..., 1:, 1:, :]
        valid = (disk2d_j[:-1, :-1] & disk2d_j[:-1, 1:]
                 & disk2d_j[1:, :-1] & disk2d_j[1:, 1:])
        w = _omega(s00, s10, s11) + _omega(s00, s11, s01)
        return jnp.sum(jnp.where(valid, w, 0.0), axis=(-2, -1)) / (4.0 * jnp.pi)

    def anneal(spins, key, betas, p, n_therm, n_meas):
        K1_eff = jnp.where(surf_j, p["KanS"], p["Kan1"])
        nsub = len(masks)

        def sweep(s, k, beta):
            """One sweep is one pass per sublattice, in order."""
            for c in range(nsub):
                k, kc = random.split(k)
                s = substep(s, kc, beta, c, p, K1_eff)
            return s, k

        def per_T(carry, beta):
            s, k = carry

            def therm(_, v):
                return sweep(v[0], v[1], beta)
            s, k = jax.lax.fori_loop(0, n_therm, therm, (s, k))

            b = s.shape[0]
            z0 = jnp.zeros(b)
            E_ref = energy(s, p, K1_eff)
            M_ref, _ = magnetization(s)

            def meas(_, v):
                s, k, sE, sE2, sM, sM2, sMz, sQ = v
                s, k = sweep(s, k, beta)
                dE = energy(s, p, K1_eff) - E_ref
                M, Mz = magnetization(s)
                dM = M - M_ref
                Q = topo_charge(s)[:, L // 2]
                return (s, k, sE + dE, sE2 + dE * dE, sM + dM,
                        sM2 + dM * dM, sMz + Mz, sQ + Q)

            s, k, sE, sE2, sM, sM2, sMz, sQ = jax.lax.fori_loop(
                0, n_meas, meas, (s, k, z0, z0, z0, z0, z0, z0))
            obs = jnp.stack([sE / n_meas, sE2 / n_meas, sM / n_meas,
                             sM2 / n_meas, sMz / n_meas, sQ / n_meas,
                             E_ref, M_ref])
            return (s, k), (obs, s[0, L // 2, :, :, 2])

        (s, _), (all_obs, snaps) = jax.lax.scan(per_T, (spins, key), betas)
        return all_obs, s, snaps

    return dict(substep=substep, energy=energy, magnetization=magnetization,
                topo_charge=topo_charge, anneal=anneal,
                n_sublattices=len(masks))


def init_spins(key, geo, batch, So=1.0):
    L, N = geo["L"], geo["N"]
    x = random.normal(key, (batch, L, N, N, 3))
    x = x / jnp.linalg.norm(x, axis=-1, keepdims=True) * So
    return x * jnp.asarray(geo["disco"][None, ..., None])
