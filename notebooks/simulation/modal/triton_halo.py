"""A raw-Triton substep that keeps the neighbourhood in registers.

Why this exists
---------------
The JAX substep builds six shifted copies of the lattice with `jnp.roll` and
then does arithmetic on them. HLO attributes 31.5% of the anneal to those six
rolls alone. Three earlier attempts to remove them went through Pallas and all
lost, by +15% to +31%, for one reason recorded in
`the-cost-is-the-gather-not-the-lanes`: each lane computed its own neighbour
addresses, so the compiler could no longer prove the accesses were contiguous
and emitted scattered loads. 1521 sites x 7 neighbours x 3 components is about
32k address computations per site-step, against seven coalesced array reads in
the JAX path.

This kernel is the version that was never writable in Pallas. The difference
is not fewer loads -- it still issues 21 of them -- but their SHAPE:

    Pallas gather   : one address per lane, computed per lane  -> scattered
    this kernel     : one contiguous run per row, shifted      -> coalesced

A shifted contiguous load is the same access pattern a roll produces, except
the shifted plane is never materialised. It lands in registers, the whole
update happens there, and only the result is stored. So the six rolls'
round-trip through memory disappears while the coalescing survives. In Triton
terms that is "the halo in SRAM": the overlapping shifted loads of adjacent
blocks hit L1/L2, and the stencil never writes an intermediate.

Why the halo needs no coordination: the update is a checkerboard, so every
neighbour of an updated site has the OPPOSITE parity and no block writes it
during this substep. A block can therefore read its halo straight from global
memory with no barrier -- which is the whole reason the checkerboard exists.

Layout
------
Production spins are `(B, L, N, N, 3)`, component-minor, so a run along x has
stride 3. This kernel takes `(B, 3, L, N, N)`, component-planar, where a run
along x is stride 1 and a row load is one contiguous 39-float segment. The
transpose is paid once outside the step loop, so over thousands of steps it
costs nothing.

Two variants, for two different jobs
------------------------------------
`HAS_RNG=False` takes the trial spins and the Metropolis uniforms as inputs,
so it can be compared against `sim_core.substep` on identical randomness --
that validates the stencil. `HAS_RNG=True` draws them in-kernel with Philox,
which is the real candidate: it also absorbs the 17.1% of the anneal that
HLO attributes to Threefry.
"""

from __future__ import annotations

import numpy as np

import triton
import triton.language as tl


# ---------------------------------------------------------------- the kernel

@triton.jit
def _substep_kernel(
    sp_ptr,          # (B, 3, LZ, N, N)  f32   spins in
    k1_ptr,          # (B, LZ, N, N)     f32   K1_eff, surface-dependent
    par_ptr,         # (B, 6)            f32   Jex, Kan2, Hex, KDM, cos g, sin g
    disk_ptr,        # (N, N)            f32   1.0 inside the disk
    beta_ptr,        # (1,)              f32
    step_ptr,        # (1,)              i32   mixes into the Philox seed
    tr_ptr,          # (B, 3, LZ, N, N)  f32   trial spins   (HAS_RNG=False)
    ru_ptr,          # (B, LZ, N, N)     f32   uniforms      (HAS_RNG=False)
    out_ptr,         # (B, 3, LZ, N, N)  f32   spins out
    de_ptr,          # (B, LZ, N, N)     f32   dE out        (WRITE_DE=True)
    N: tl.constexpr,
    LZ: tl.constexpr,
    PARITY: tl.constexpr,     # 0 updates the even sublattice, 1 the odd
    BY: tl.constexpr,         # rows of the block
    BX: tl.constexpr,         # lanes along x, a power of two >= N
    HAS_RNG: tl.constexpr,
    WRITE_DE: tl.constexpr,
    SO: tl.constexpr,
):
    b = tl.program_id(0)
    z = tl.program_id(1)
    ty = tl.program_id(2)

    NN = N * N
    sb = 3 * LZ * NN          # stride between replicas
    sc = LZ * NN              # stride between components

    # ---- the block, and its neighbour index vectors.
    # These are VECTORS over the block, not per-lane addresses: `xc + 1` is a
    # contiguous run shifted by one element, which is what keeps the loads
    # coalesced. That is the single difference from the gather attempts.
    ys = ty * BY + tl.arange(0, BY)
    xs = tl.arange(0, BX)
    live = (ys < N)[:, None] & (xs < N)[None, :]

    yc = tl.where(ys < N, ys, 0)
    xc = tl.where(xs < N, xs, 0)

    # x and y wrap; z does not (open boundary, handled by the gates below).
    ypi = tl.where(yc + 1 >= N, 0, yc + 1)
    ymi = tl.where(yc - 1 < 0, N - 1, yc - 1)
    xpi = tl.where(xc + 1 >= N, 0, xc + 1)
    xmi = tl.where(xc - 1 < 0, N - 1, xc - 1)

    o_c = yc[:, None] * N + xc[None, :]
    o_yp = ypi[:, None] * N + xc[None, :]
    o_ym = ymi[:, None] * N + xc[None, :]
    o_xp = yc[:, None] * N + xpi[None, :]
    o_xm = yc[:, None] * N + xmi[None, :]

    zp = tl.minimum(z + 1, LZ - 1)
    zm = tl.maximum(z - 1, 0)
    # `jnp.roll(...).at[:, -1].set(0.0)` in the reference: the plane past the
    # last layer contributes zero rather than wrapping.
    gp = tl.where(z < LZ - 1, 1.0, 0.0)
    gm = tl.where(z > 0, 1.0, 0.0)

    # ---- the site mask: inside the disk, and on the sublattice being updated.
    disk = tl.load(disk_ptr + o_c, mask=live, other=0.0)
    par = (xc[None, :] + yc[:, None] + z) % 2
    upd = live & (disk > 0.5) & (par == PARITY)

    # ---- per-chain parameters.
    Jex = tl.load(par_ptr + b * 6 + 0)
    Kan2 = tl.load(par_ptr + b * 6 + 1)
    Hex = tl.load(par_ptr + b * 6 + 2)
    KDM = tl.load(par_ptr + b * 6 + 3)
    cosg = tl.load(par_ptr + b * 6 + 4)
    sing = tl.load(par_ptr + b * 6 + 5)
    beta = tl.load(beta_ptr)

    sca = sp_ptr + b * sb
    k1 = tl.load(k1_ptr + b * LZ * NN + z * NN + o_c, mask=live, other=0.0)

    # ---- the spin at the site.
    s0 = tl.load(sca + 0 * sc + z * NN + o_c, mask=live, other=0.0)
    s1 = tl.load(sca + 1 * sc + z * NN + o_c, mask=live, other=0.0)
    s2 = tl.load(sca + 2 * sc + z * NN + o_c, mask=live, other=0.0)

    # ---- the trial spin: a uniform point on the sphere.
    if HAS_RNG:
        off = ((b * LZ + z) * N + yc[:, None]) * N + xc[None, :]
        # Four single-tensor draws rather than one `randn4x`: the 4-tuple form
        # needs tuple unpacking, and four consecutive seeds are independent
        # Philox streams anyway. Still far cheaper than Threefry, which HLO
        # charges 17.1% of the anneal.
        sd = tl.load(step_ptr) * 4
        g0 = tl.randn(sd + 0, off)
        g1 = tl.randn(sd + 1, off)
        g2 = tl.randn(sd + 2, off)
        nrm = tl.sqrt(g0 * g0 + g1 * g1 + g2 * g2) + 1e-30
        t0 = g0 / nrm * SO
        t1 = g1 / nrm * SO
        t2 = g2 / nrm * SO
        ru = tl.rand(sd + 3, off)
    else:
        trb = tr_ptr + b * sb
        t0 = tl.load(trb + 0 * sc + z * NN + o_c, mask=live, other=0.0)
        t1 = tl.load(trb + 1 * sc + z * NN + o_c, mask=live, other=0.0)
        t2 = tl.load(trb + 2 * sc + z * NN + o_c, mask=live, other=0.0)
        ru = tl.load(ru_ptr + b * LZ * NN + z * NN + o_c, mask=live, other=0.0)

    d0 = t0 - s0
    d1 = t1 - s1
    d2 = t2 - s2

    # ---- the stencil, one component at a time so the six neighbour blocks of
    # a component die before the next component's are loaded. Holding all 21
    # at once spills at BY >= 8.
    p0 = sca + 0 * sc
    n_xp = tl.load(p0 + z * NN + o_xp, mask=live, other=0.0)
    n_xm = tl.load(p0 + z * NN + o_xm, mask=live, other=0.0)
    n_yp = tl.load(p0 + z * NN + o_yp, mask=live, other=0.0)
    n_ym = tl.load(p0 + z * NN + o_ym, mask=live, other=0.0)
    sv0 = (n_xp + n_xm + n_yp + n_ym
           + gp * tl.load(p0 + zp * NN + o_c, mask=live, other=0.0)
           + gm * tl.load(p0 + zm * NN + o_c, mask=live, other=0.0))
    dyx = n_yp - n_ym                     # DMI needs this difference only

    p1 = sca + 1 * sc
    n_xp = tl.load(p1 + z * NN + o_xp, mask=live, other=0.0)
    n_xm = tl.load(p1 + z * NN + o_xm, mask=live, other=0.0)
    n_yp = tl.load(p1 + z * NN + o_yp, mask=live, other=0.0)
    n_ym = tl.load(p1 + z * NN + o_ym, mask=live, other=0.0)
    sv1 = (n_xp + n_xm + n_yp + n_ym
           + gp * tl.load(p1 + zp * NN + o_c, mask=live, other=0.0)
           + gm * tl.load(p1 + zm * NN + o_c, mask=live, other=0.0))
    dxy = n_xp - n_xm

    p2 = sca + 2 * sc
    n_xp = tl.load(p2 + z * NN + o_xp, mask=live, other=0.0)
    n_xm = tl.load(p2 + z * NN + o_xm, mask=live, other=0.0)
    n_yp = tl.load(p2 + z * NN + o_yp, mask=live, other=0.0)
    n_ym = tl.load(p2 + z * NN + o_ym, mask=live, other=0.0)
    sv2 = (n_xp + n_xm + n_yp + n_ym
           + gp * tl.load(p2 + zp * NN + o_c, mask=live, other=0.0)
           + gm * tl.load(p2 + zm * NN + o_c, mask=live, other=0.0))
    dxz = n_xp - n_xm
    dyz = n_yp - n_ym

    # ---- the energy difference, term for term as in `sim_core.substep`.
    dE = -Jex * (d0 * sv0 + d1 * sv1 + d2 * sv2)

    i4 = 1.0 / (SO * SO * SO * SO)
    i6 = i4 / (SO * SO)
    an_t = (k1 * i4 * ((t0 * t1) * (t0 * t1) + (t0 * t2) * (t0 * t2)
                       + (t1 * t2) * (t1 * t2))
            + Kan2 * i6 * (t0 * t1 * t2) * (t0 * t1 * t2))
    an_s = (k1 * i4 * ((s0 * s1) * (s0 * s1) + (s0 * s2) * (s0 * s2)
                       + (s1 * s2) * (s1 * s2))
            + Kan2 * i6 * (s0 * s1 * s2) * (s0 * s1 * s2))
    dE += an_t - an_s

    # Interfacial DMI: in-plane bonds only, with the minus sign.
    dE -= KDM * (d1 * dxz - d2 * dxy + d2 * dyx - d0 * dyz)
    dE -= Hex * (d0 * cosg + d2 * sing)

    acc = ((dE < 0.0) | (ru < tl.exp(-dE * beta))) & upd

    ob = out_ptr + b * sb + z * NN + o_c
    tl.store(ob + 0 * sc, tl.where(acc, t0, s0), mask=live)
    tl.store(ob + 1 * sc, tl.where(acc, t1, s1), mask=live)
    tl.store(ob + 2 * sc, tl.where(acc, t2, s2), mask=live)
    if WRITE_DE:
        tl.store(de_ptr + b * LZ * NN + z * NN + o_c, dE, mask=live)


# ------------------------------------------------------------- JAX plumbing

def planar(x):
    """(B, L, N, N, 3) -> (B, 3, L, N, N), paid once outside the step loop.

    No explicit contiguity call is needed: `jax_triton` pins default row-major
    layouts on its operands, so XLA has to materialise the transpose before
    the custom call rather than passing a strided view.
    """
    import jax.numpy as jnp
    return jnp.transpose(x, (0, 4, 1, 2, 3))


def unplanar(x):
    import jax.numpy as jnp
    return jnp.transpose(x, (0, 2, 3, 4, 1))


def pack_params(p, batch):
    """The six scalars the kernel reads, as (B, 6).

    `Kan1` is absent on purpose: `sim_core._an` takes it as an argument and
    never uses it -- the K1 invariant is carried entirely by `K1_eff`, which
    is already the surface-resolved per-site field.
    """
    import jax.numpy as jnp
    def col(v):
        a = jnp.asarray(v, dtype=jnp.float32).reshape(-1)
        return jnp.broadcast_to(a, (batch,)) if a.size == 1 else a
    g = col(p["g"])
    return jnp.stack([col(p["Jex"]), col(p["Kan2"]), col(p["Hex"]),
                      col(p["KDM"]), jnp.cos(g), jnp.sin(g)], axis=1)


def build(geo, BY=4, BX=None, num_warps=4, So=1.0):
    """Returns `substep(spins_planar, k1, par, beta, step, parity, tr, ru)`.

    `spins_planar` is component-planar. `tr`/`ru` are `None` for the in-kernel
    RNG variant. `de` comes back only when `want_de` is set, for validation.
    """
    import jax
    import jax.numpy as jnp
    import jax_triton as jt

    N, LZ = int(geo["N"]), int(geo["L"])
    BX = int(BX) if BX else 1 << (N - 1).bit_length()
    BY = int(BY)
    n_strip = -(-N // BY)
    disk = jnp.asarray(geo["disk2d"].astype(np.float32))

    def substep(sp, k1, par, beta, step, parity, tr=None, ru=None,
                want_de=False):
        B = sp.shape[0]
        has_rng = tr is None
        shape = jax.ShapeDtypeStruct(sp.shape, jnp.float32)
        # The kernel's argument list has to stay FIXED: `triton_call` appends
        # one pointer per entry of `out_shape`, so dropping the dE output left
        # `de_ptr` unbound and Triton failed to compile the signature. The
        # output is always declared and shrinks to one element when unwanted.
        de_shape = jax.ShapeDtypeStruct(
            (B, LZ, N, N) if want_de else (1,), jnp.float32)
        outs = (shape, de_shape)
        # Likewise the trial/uniform inputs: unread under HAS_RNG, so they
        # shrink to one element instead of allocating a full lattice of zeros
        # on every iteration of the step loop.
        dum = jnp.zeros((1,), jnp.float32)
        res = jt.triton_call(
            sp, k1, par, disk,
            jnp.asarray(beta, jnp.float32).reshape(1),
            jnp.asarray(step, jnp.int32).reshape(1),
            dum if has_rng else tr,
            dum if has_rng else ru,
            kernel=_substep_kernel,
            out_shape=outs,
            grid=(B, LZ, n_strip),
            num_warps=int(num_warps),
            N=N, LZ=LZ, PARITY=int(parity), BY=BY, BX=BX,
            HAS_RNG=has_rng, WRITE_DE=bool(want_de), SO=float(So),
        )
        return res if want_de else res[0]

    substep.meta = dict(N=N, LZ=LZ, BY=BY, BX=BX, n_strip=n_strip,
                        num_warps=int(num_warps),
                        grid_per_replica=LZ * n_strip,
                        lanes_per_block=BY * BX,
                        lane_waste_pct=100.0 * (1.0 - (N * N) / float(
                            BX * BY * n_strip)))
    return substep


def sweeps(substep, k1, par, beta, n, base_step=0):
    """`n` sweeps in one XLA program: even then odd, inside a device loop.

    The loop has to live in `lax.fori_loop`. A Python loop would pay the
    ~122 us host dispatch of a `triton_call` per substep, which is what faked
    an earlier Triton win and reversed on amortisation.
    """
    import jax
    import jax.numpy as jnp

    def body(i, sp):
        st = base_step + 2 * i
        sp = substep(sp, k1, par, beta, st, 0)
        return substep(sp, k1, par, beta, st + 1, 1)

    def run(sp):
        return jax.lax.fori_loop(0, int(n), body, sp)
    return run


def sweeps_extrng(substep, k1, par, beta, n, shape, base_step=0, So=1.0):
    """`n` sweeps with the trial spins drawn by JAX, not by the kernel.

    This arm exists to split the win. The in-kernel variant changes two things
    at once -- the stencil and the random source -- so on its own it cannot
    say which one paid. Here the stencil is the Triton one while the proposals
    still come from JAX's Threefry, exactly as production draws them, so the
    difference against production isolates the stencil and the difference
    against the in-kernel variant isolates the RNG.

    `shape` is the component-planar spin shape; drawing the normals directly
    in that layout and normalising along the component axis is the same
    distribution as drawing them component-minor and transposing.
    """
    import jax
    import jax.numpy as jnp

    B, _, LZ, N, _ = shape

    def half(sp, k, parity):
        kt, ka = jax.random.split(k)
        tr = jax.random.normal(kt, shape)
        tr = tr / jnp.linalg.norm(tr, axis=1, keepdims=True) * So
        ru = jax.random.uniform(ka, (B, LZ, N, N))
        return substep(sp, k1, par, beta, 0, parity, tr, ru)

    def body(i, v):
        sp, k = v
        k, k0, k1_ = jax.random.split(k, 3)
        sp = half(sp, k0, 0)
        return half(sp, k1_, 1), k

    def run(sp):
        return jax.lax.fori_loop(0, int(n), body,
                                 (sp, jax.random.PRNGKey(base_step)))[0]
    return run


# ------------------------------------------- observables, component-planar
#
# These exist because of where the measurement loop spends its time. The
# kernel works on `(B, 3, L, N, N)` while `sim_core`'s observables expect
# `(B, L, N, N, 3)`, and transposing once per measured sweep would copy the
# whole lattice as often as the substep touches it -- enough to eat the win.
# So the observables are rewritten against the component axis at position 1.
# Every formula is the one in `sim_core`, term for term; only the axes move.

def build_obs(geo):
    import jax.numpy as jnp

    LZ, N = int(geo["L"]), int(geo["N"])
    disco = jnp.asarray(geo["disco"].astype(np.float32))          # (L, N, N)
    disk2d = jnp.asarray(geo["disk2d"])
    n_act = float(geo["n_active"])

    def c4(par, k):
        """One per-chain scalar, shaped to broadcast against (B, L, N, N)."""
        return par[:, k].reshape(-1, 1, 1, 1)

    def energy(sp, par, k1):
        # Each bond counted once, via the +x, +y, +z shifts only.
        xp = jnp.roll(sp, -1, 4)
        yp = jnp.roll(sp, -1, 3)
        zp = jnp.roll(sp, -1, 2).at[:, :, -1].set(0.0)
        E = -c4(par, 0) * jnp.sum(sp * (xp + yp + zp), axis=1)
        sx, sy, sz = sp[:, 0], sp[:, 1], sp[:, 2]
        E = E + (k1 * ((sx * sy) ** 2 + (sx * sz) ** 2 + (sy * sz) ** 2)
                 + c4(par, 1) * (sx * sy * sz) ** 2)
        E = E - c4(par, 3) * (sy * xp[:, 2] - sz * xp[:, 1]
                              + sz * yp[:, 0] - sx * yp[:, 2])
        E = E - c4(par, 2) * (sx * par[:, 4].reshape(-1, 1, 1, 1)
                              + sz * par[:, 5].reshape(-1, 1, 1, 1))
        return jnp.sum(E * disco[None], axis=(-3, -2, -1))

    def mag(sp):
        M = jnp.sum(sp * disco[None, None], axis=(2, 3, 4)) / n_act   # (B, 3)
        return jnp.linalg.norm(M, axis=1), M[:, 2]

    def _omega(a, b, c):
        num = jnp.sum(a * jnp.cross(b, c, axis=1), axis=1)
        den = (1.0 + jnp.sum(a * b, 1) + jnp.sum(b * c, 1)
               + jnp.sum(c * a, 1))
        return 2.0 * jnp.arctan2(num, den)

    valid = (disk2d[:-1, :-1] & disk2d[:-1, 1:]
             & disk2d[1:, :-1] & disk2d[1:, 1:])

    def topo(sp):
        """Berg-Luscher charge per layer: quantised, unlike a difference."""
        s00 = sp[:, :, :, :-1, :-1]; s10 = sp[:, :, :, :-1, 1:]
        s01 = sp[:, :, :, 1:, :-1];  s11 = sp[:, :, :, 1:, 1:]
        w = _omega(s00, s10, s11) + _omega(s00, s11, s01)
        return jnp.sum(jnp.where(valid, w, 0.0), axis=(-2, -1)) / (
            4.0 * jnp.pi)

    return dict(energy=energy, mag=mag, topo=topo, n_active=n_act)


def build_anneal(geo, BY=4, num_warps=4, So=1.0):
    """The full temperature scan, with the Triton substep inside.

    Structurally identical to `sim_core.build_engine`'s `anneal`: a `lax.scan`
    over betas, a thermalisation `fori_loop`, then a measurement `fori_loop`
    accumulating the SAME eight quantities in the SAME order, so the result
    feeds `sim_core.reduce_obs` unchanged.

    The one difference is the random stream. `sim_core` splits a Threefry key
    down the loop; the kernel is counter-based, so a step counter rides the
    carry and every substep gets its own Philox offset. That makes the chain
    reproducible from `seed` without carrying key state through the scan.
    """
    import jax
    import jax.numpy as jnp

    sub = build(geo, BY=BY, num_warps=num_warps, So=So)
    O = build_obs(geo)
    LZ = int(geo["L"])

    def anneal(sp, seed, betas, par, k1, n_therm, n_meas):
        n_therm, n_meas = int(n_therm), int(n_meas)

        def pair(s, c, bt):
            s = sub(s, k1, par, bt, c, 0)
            return sub(s, k1, par, bt, c + 1, 1), c + 2

        def per_T(carry, bt):
            s, c = carry

            def therm(_, v):
                s, c = v
                return pair(s, c, bt)
            s, c = jax.lax.fori_loop(0, n_therm, therm, (s, c))

            # Fluctuations are taken about a reference fixed once per
            # temperature. Accumulating raw E and E^2 in float32 destroys the
            # variance by cancellation -- one ulp of <E^2> can exceed the
            # variance itself, and Cv comes out negative. A constant shift
            # leaves the variance unchanged, so this is exact.
            E_ref = O["energy"](s, par, k1)
            M_ref, _ = O["mag"](s)
            z0 = jnp.zeros(s.shape[0], jnp.float32)

            def meas(_, v):
                s, c, sE, sE2, sM, sM2, sMz, sQ = v
                s, c = pair(s, c, bt)
                dE = O["energy"](s, par, k1) - E_ref
                M, Mz = O["mag"](s)
                dM = M - M_ref
                Q = O["topo"](s)[:, LZ // 2]
                return (s, c, sE + dE, sE2 + dE * dE, sM + dM,
                        sM2 + dM * dM, sMz + Mz, sQ + Q)

            s, c, sE, sE2, sM, sM2, sMz, sQ = jax.lax.fori_loop(
                0, n_meas, meas, (s, c, z0, z0, z0, z0, z0, z0))
            out = jnp.stack([sE / n_meas, sE2 / n_meas,
                             sM / n_meas, sM2 / n_meas,
                             sMz / n_meas, sQ / n_meas, E_ref, M_ref])
            # Replica 0, middle layer, z component -- the same snapshot
            # `sim_core` takes, read off the planar layout.
            snap = s[0, 2, LZ // 2]
            return (s, c), (out, snap)

        c0 = jnp.asarray(int(seed), jnp.int32) * jnp.int32(1000003)
        (s, _), (all_obs, snaps) = jax.lax.scan(per_T, (sp, c0), betas)
        return all_obs, s, snaps

    anneal.meta = dict(sub.meta)
    return anneal
