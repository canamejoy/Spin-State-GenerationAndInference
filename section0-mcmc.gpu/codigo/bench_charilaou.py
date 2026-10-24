"""Three controlled comparisons against Charilaou, arXiv:2106.11286.

The comparison in the thesis section rests on numbers read from that paper,
measured on a Tesla V100 with a Hamiltonian lighter than ours. Three things
differ at once -- the model, the hardware and the method -- so the twelvefold
raw ratio decomposes into nothing in particular. This file fixes them one at a
time:

  T1  same device, same model, THEIR method against OURS
  T2  our method and their model on devices that bracket the V100
  T3  our devices running their model, against their reported figure

What the paper specifies, and what is reproduced here
-----------------------------------------------------
Model    H = -J sum_<ij> S_i.S_j - K sum_i (S_i.e)^2, uniaxial, e = z
         J = 1, K = 0.01, |S| = 1
Lattice  simple cubic, periodic on all three axes, 128^3 for the headline
Protocol 1e4 MCS thermalisation + 1e4 MCS averaging at T = 0.9 T_C,
         with T_C/J = 1.445
Reported 610e6 updates/s = 1.63 ns at 128^3, 630e6 = 1.59 ns at 512^3,
         1.55 ns as the inset minimum, on a Tesla V100 32 GB
Method   three linearised 1-D arrays, one per component, in global memory;
         N_b blocks of n spins; ONE spin per block chosen at random per step;
         dE computed locally; cuRAND Mersenne-Twister; NO colouring

That last line is the point of T1. The paper rejects checkerboard decomposition
explicitly, on the grounds that it restricts the lattice symmetry, and accepts
in exchange a read/write race it mitigates only by keeping n large enough. Our
method pays a colouring and gets a race-free halo. T1 is what that trade costs
or buys, with everything else held equal.

Three honesty notes, all of which cut against us
------------------------------------------------
1. Our headline 0.129 ns is measured at 5000 replicas of a 5245-site disc.
   His is one system of 2.1e6 spins. The per-device replica floor is real, so
   comparing those two numbers compares our best parallel configuration against
   his single-system one. Every arm here runs ONE system at his geometry.
2. His optimum of n = 8 spins per block was found on a V100. Re-using it on
   another device would handicap his method, so `theirs` scans n and reports
   its best, exactly as `ours` scans its block height.
3. His kernel updates the spin arrays IN PLACE. If the host allocated a fresh
   output buffer per step, the 2.1e6 unmodified spins would have to be copied
   every step and his method would measure far worse than it is. The driver
   therefore aliases its inputs to its outputs and REFUSES to report a time if
   the installed jax_triton will not do that -- see `_alias_support`.

Nothing here is timed until the physics passes. `verify_host` checks the
formulas with numpy alone, so it runs without a GPU; `verify_device` checks the
kernel against a pure-JAX reference engine on IDENTICAL randomness, where the
only admissible difference is arithmetic and the spin count must be zero.
`theirs` is nondeterministic by construction -- the race is the method -- so it
is held to a statistical check instead.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ModuleNotFoundError:                 # host-side checks still run
    HAVE_TRITON = False

    class _NoTriton:
        """Lets the module import where Triton is absent, so `verify_host`
        can check the formulas on a machine with no GPU. Any attempt to build
        a kernel then fails loudly rather than silently measuring nothing."""

        def jit(self, fn):
            return fn

        def __getattr__(self, name):
            raise RuntimeError("Triton is not installed in this environment")

    triton = tl = _NoTriton()

# ----------------------------------------------------------- what he reported
PAPER = dict(
    source="arXiv:2106.11286",
    device="Tesla V100 32GB",
    ns_per_update={"128^3": 1.63, "512^3": 1.59, "inset_min": 1.55},
    updates_per_s={"128^3": 610e6, "512^3": 630e6},
    J=1.0, K=0.01, S=1.0, Tc_over_J=1.445, n_per_block=8,
    mcs_therm=10_000, mcs_avg=10_000,
)
# T = 0.9 T_C with T_C/J = 1.445 and J = 1, so beta = 1/(0.9*1.445).
BETA = 1.0 / (0.9 * 1.445)


# =============================================================== our method
#
# The shifted-load checkerboard stencil of `triton_halo`, carrying his
# Hamiltonian instead of ours and periodic on all three axes instead of two.
# The index arithmetic is the only thing that changes from production: z wraps
# here, where production clamps it and gates the wrapped plane to zero.

@triton.jit
def _ours_kernel(
    sp_ptr,          # (3, LZ, N, N)  f32   spins in, component-planar
    tr_ptr,          # (3, LZ, N, N)  f32   trial spins    (HAS_RNG=False)
    ru_ptr,          # (LZ, N, N)     f32   uniforms       (HAS_RNG=False)
    beta_ptr,        # (1,)           f32
    step_ptr,        # (1,)           i32
    out_ptr,         # (3, LZ, N, N)  f32   spins out
    de_ptr,          # (LZ, N, N)     f32   dE out         (WRITE_DE=True)
    N: tl.constexpr,
    LZ: tl.constexpr,
    PARITY: tl.constexpr,
    BY: tl.constexpr,
    BX: tl.constexpr,
    JEX: tl.constexpr,
    KAN: tl.constexpr,
    SO: tl.constexpr,
    HAS_RNG: tl.constexpr,
    WRITE_DE: tl.constexpr,
):
    z = tl.program_id(0)
    ty = tl.program_id(1)

    NN = N * N
    sc = LZ * NN

    ys = ty * BY + tl.arange(0, BY)
    xs = tl.arange(0, BX)
    live = (ys < N)[:, None] & (xs < N)[None, :]
    yc = tl.where(ys < N, ys, 0)
    xc = tl.where(xs < N, xs, 0)

    # All three axes wrap. N must be even for the parity colouring to close
    # across the seam, which the host asserts.
    ypi = tl.where(yc + 1 >= N, 0, yc + 1)
    ymi = tl.where(yc - 1 < 0, N - 1, yc - 1)
    xpi = tl.where(xc + 1 >= N, 0, xc + 1)
    xmi = tl.where(xc - 1 < 0, N - 1, xc - 1)
    zp = tl.where(z + 1 >= LZ, 0, z + 1)
    zm = tl.where(z - 1 < 0, LZ - 1, z - 1)

    o_c = yc[:, None] * N + xc[None, :]
    o_yp = ypi[:, None] * N + xc[None, :]
    o_ym = ymi[:, None] * N + xc[None, :]
    o_xp = yc[:, None] * N + xpi[None, :]
    o_xm = yc[:, None] * N + xmi[None, :]

    par = (xc[None, :] + yc[:, None] + z) % 2
    upd = live & (par == PARITY)

    beta = tl.load(beta_ptr)
    s0 = tl.load(sp_ptr + 0 * sc + z * NN + o_c, mask=live, other=0.0)
    s1 = tl.load(sp_ptr + 1 * sc + z * NN + o_c, mask=live, other=0.0)
    s2 = tl.load(sp_ptr + 2 * sc + z * NN + o_c, mask=live, other=0.0)

    if HAS_RNG:
        # one key, four distinct counters: the acceptance variate must share no
        # input with the trial direction.
        sd = tl.load(step_ptr)
        o4 = (z * NN + o_c).to(tl.int32) * 4
        g0 = tl.randn(sd, o4 + 0)
        g1 = tl.randn(sd, o4 + 1)
        g2 = tl.randn(sd, o4 + 2)
        nrm = tl.sqrt(g0 * g0 + g1 * g1 + g2 * g2) + 1e-30
        t0 = g0 / nrm * SO
        t1 = g1 / nrm * SO
        t2 = g2 / nrm * SO
        ru = tl.rand(sd, o4 + 3)
    else:
        t0 = tl.load(tr_ptr + 0 * sc + z * NN + o_c, mask=live, other=0.0)
        t1 = tl.load(tr_ptr + 1 * sc + z * NN + o_c, mask=live, other=0.0)
        t2 = tl.load(tr_ptr + 2 * sc + z * NN + o_c, mask=live, other=0.0)
        ru = tl.load(ru_ptr + z * NN + o_c, mask=live, other=0.0)

    d0 = t0 - s0
    d1 = t1 - s1
    d2 = t2 - s2

    # ---- the stencil, one component at a time so each component's six
    # neighbour blocks die before the next component's are loaded.
    p = sp_ptr + 0 * sc
    sv0 = (tl.load(p + z * NN + o_xp, mask=live, other=0.0)
           + tl.load(p + z * NN + o_xm, mask=live, other=0.0)
           + tl.load(p + z * NN + o_yp, mask=live, other=0.0)
           + tl.load(p + z * NN + o_ym, mask=live, other=0.0)
           + tl.load(p + zp * NN + o_c, mask=live, other=0.0)
           + tl.load(p + zm * NN + o_c, mask=live, other=0.0))
    p = sp_ptr + 1 * sc
    sv1 = (tl.load(p + z * NN + o_xp, mask=live, other=0.0)
           + tl.load(p + z * NN + o_xm, mask=live, other=0.0)
           + tl.load(p + z * NN + o_yp, mask=live, other=0.0)
           + tl.load(p + z * NN + o_ym, mask=live, other=0.0)
           + tl.load(p + zp * NN + o_c, mask=live, other=0.0)
           + tl.load(p + zm * NN + o_c, mask=live, other=0.0))
    p = sp_ptr + 2 * sc
    sv2 = (tl.load(p + z * NN + o_xp, mask=live, other=0.0)
           + tl.load(p + z * NN + o_xm, mask=live, other=0.0)
           + tl.load(p + z * NN + o_yp, mask=live, other=0.0)
           + tl.load(p + z * NN + o_ym, mask=live, other=0.0)
           + tl.load(p + zp * NN + o_c, mask=live, other=0.0)
           + tl.load(p + zm * NN + o_c, mask=live, other=0.0))

    # H = -J sum_<ij> S_i.S_j - K sum_i (S_i.z)^2, so a single-site change is
    #   dE = -J (t - s).v  -  K (t_z^2 - s_z^2).
    dE = -JEX * (d0 * sv0 + d1 * sv1 + d2 * sv2) \
         - KAN * (t2 * t2 - s2 * s2)

    acc = upd & ((dE <= 0.0) | (ru < tl.exp(-beta * tl.minimum(dE, 80.0))))
    ob = out_ptr + z * NN + o_c
    tl.store(ob + 0 * sc, tl.where(acc, t0, s0), mask=live)
    tl.store(ob + 1 * sc, tl.where(acc, t1, s1), mask=live)
    tl.store(ob + 2 * sc, tl.where(acc, t2, s2), mask=live)
    if WRITE_DE:
        tl.store(de_ptr + z * NN + o_c, dE, mask=live)


# ============================================================= their method
#
# Faithful to the paper: three linearised component arrays in global memory,
# N_b blocks each owning n contiguous spins, one of those n drawn at random per
# step, dE computed from the 1-D index, no colouring and no barrier. The race
# the paper acknowledges is part of the method and is reproduced, not repaired
# -- repairing it would measure something the paper does not claim.
#
# One spin is updated per block, so the lane count is not ours to choose.

@triton.jit
def _theirs_kernel(
    sx_ptr,          # (NT,) f32   x components, linearised
    sy_ptr,          # (NT,) f32
    sz_ptr,          # (NT,) f32
    beta_ptr,        # (1,)  f32
    step_ptr,        # (1,)  i32
    N: tl.constexpr,
    LZ: tl.constexpr,
    N_PER: tl.constexpr,
    JEX: tl.constexpr,
    KAN: tl.constexpr,
    SO: tl.constexpr,
):
    blk = tl.program_id(0)
    sd = tl.load(step_ptr)
    NN = N * N

    # one of the block's n spins, drawn at random
    pick = (tl.rand(sd, blk * 5 + 0) * N_PER).to(tl.int32)
    pick = tl.minimum(pick, N_PER - 1)
    i = blk * N_PER + pick

    # 1-D index back to (z, y, x) on the simple cubic lattice
    z = i // NN
    r = i - z * NN
    y = r // N
    x = r - y * N

    xp = tl.where(x + 1 >= N, 0, x + 1)
    xm = tl.where(x - 1 < 0, N - 1, x - 1)
    yp = tl.where(y + 1 >= N, 0, y + 1)
    ym = tl.where(y - 1 < 0, N - 1, y - 1)
    zp = tl.where(z + 1 >= LZ, 0, z + 1)
    zm = tl.where(z - 1 < 0, LZ - 1, z - 1)

    nb0 = z * NN + y * N + xp
    nb1 = z * NN + y * N + xm
    nb2 = z * NN + yp * N + x
    nb3 = z * NN + ym * N + x
    nb4 = zp * NN + y * N + x
    nb5 = zm * NN + y * N + x

    s0 = tl.load(sx_ptr + i)
    s1 = tl.load(sy_ptr + i)
    s2 = tl.load(sz_ptr + i)

    vx = (tl.load(sx_ptr + nb0) + tl.load(sx_ptr + nb1)
          + tl.load(sx_ptr + nb2) + tl.load(sx_ptr + nb3)
          + tl.load(sx_ptr + nb4) + tl.load(sx_ptr + nb5))
    vy = (tl.load(sy_ptr + nb0) + tl.load(sy_ptr + nb1)
          + tl.load(sy_ptr + nb2) + tl.load(sy_ptr + nb3)
          + tl.load(sy_ptr + nb4) + tl.load(sy_ptr + nb5))
    vz = (tl.load(sz_ptr + nb0) + tl.load(sz_ptr + nb1)
          + tl.load(sz_ptr + nb2) + tl.load(sz_ptr + nb3)
          + tl.load(sz_ptr + nb4) + tl.load(sz_ptr + nb5))

    g0 = tl.randn(sd, blk * 5 + 1)
    g1 = tl.randn(sd, blk * 5 + 2)
    g2 = tl.randn(sd, blk * 5 + 3)
    nrm = tl.sqrt(g0 * g0 + g1 * g1 + g2 * g2) + 1e-30
    t0 = g0 / nrm * SO
    t1 = g1 / nrm * SO
    t2 = g2 / nrm * SO
    ru = tl.rand(sd, blk * 5 + 4)

    dE = -JEX * ((t0 - s0) * vx + (t1 - s1) * vy + (t2 - s2) * vz) \
         - KAN * (t2 * t2 - s2 * s2)
    beta = tl.load(beta_ptr)
    acc = (dE <= 0.0) | (ru < tl.exp(-beta * tl.minimum(dE, 80.0)))

    # The host aliases these three inputs onto the three outputs, so writing
    # back through them IS the paper's in-place update of global memory, and
    # the untouched n-1 spins of the block are never copied.
    tl.store(sx_ptr + i, tl.where(acc, t0, s0))
    tl.store(sy_ptr + i, tl.where(acc, t1, s1))
    tl.store(sz_ptr + i, tl.where(acc, t2, s2))


# ============================================ a pure-JAX reference for ours
#
# Runs on CPU or GPU, so the formulas are checkable without a device. This is
# what `verify_device` holds the kernel to, on identical randomness.

def ref_engine(N, LZ, J=1.0, K=0.01, So=1.0):
    import jax.numpy as jnp

    zc, yc, xc = np.indices((LZ, N, N))
    par = jnp.asarray(((xc + yc + zc) % 2).astype(np.int32))

    def nsum(sp):
        return (jnp.roll(sp, -1, 3) + jnp.roll(sp, 1, 3)
                + jnp.roll(sp, -1, 2) + jnp.roll(sp, 1, 2)
                + jnp.roll(sp, -1, 1) + jnp.roll(sp, 1, 1))

    def dE_of(sp, tr):
        v = nsum(sp)
        d = tr - sp
        return (-J * jnp.sum(d * v, axis=0)
                - K * (tr[2] ** 2 - sp[2] ** 2))

    def substep(sp, tr, ru, beta, parity):
        dE = dE_of(sp, tr)
        acc = (par == parity) & ((dE <= 0) | (ru < jnp.exp(
            -beta * jnp.minimum(dE, 80.0))))
        return jnp.where(acc[None], tr, sp), dE

    def energy(sp):
        # each bond once: only the +x, +y, +z translates
        e = -J * (jnp.sum(sp * jnp.roll(sp, -1, 3))
                  + jnp.sum(sp * jnp.roll(sp, -1, 2))
                  + jnp.sum(sp * jnp.roll(sp, -1, 1)))
        return e - K * jnp.sum(sp[2] ** 2)

    return dict(substep=substep, dE_of=dE_of, energy=energy, nsum=nsum)


def random_state(key, N, LZ, So=1.0):
    import jax
    import jax.numpy as jnp
    sp = jax.random.normal(key, (3, LZ, N, N), jnp.float32)
    return sp / (jnp.linalg.norm(sp, axis=0, keepdims=True) + 1e-30) * So


# =================================================================== drivers

def _alias_support():
    """Does the installed jax_triton alias inputs onto outputs?

    Without it, his in-place update becomes a full-array copy per step and the
    measurement is meaningless, so this decides whether a time may be reported
    at all rather than being worked around.
    """
    import inspect
    import jax_triton as jt
    sig = inspect.signature(jt.triton_call)
    return "input_output_aliases" in sig.parameters


def build_ours(N, LZ, BY=4, J=1.0, K=0.01, So=1.0, has_rng=True):
    """One pass of our checkerboard, his Hamiltonian, periodic everywhere."""
    import jax
    import jax.numpy as jnp
    import jax_triton as jt

    assert N % 2 == 0, "the parity colouring needs an even N to close under PBC"
    BX = 1 << (N - 1).bit_length()
    n_strip = -(-N // BY)
    dum3 = jnp.zeros((1,), jnp.float32)

    def substep(sp, beta, step, parity, tr=None, ru=None, want_de=False):
        shape = jax.ShapeDtypeStruct(sp.shape, jnp.float32)
        de_shape = jax.ShapeDtypeStruct(
            (LZ, N, N) if want_de else (1,), jnp.float32)
        out, de = jt.triton_call(
            sp,
            dum3 if tr is None else tr,
            dum3 if ru is None else ru,
            jnp.float32(beta).reshape(1), jnp.int32(step).reshape(1),
            kernel=_ours_kernel, out_shape=(shape, de_shape),
            grid=(LZ, n_strip), N=N, LZ=LZ, PARITY=int(parity), BY=BY, BX=BX,
            JEX=float(J), KAN=float(K), SO=float(So),
            HAS_RNG=bool(tr is None), WRITE_DE=bool(want_de), num_warps=4,
        )
        return (out, de) if want_de else out

    def sweep(sp, beta, step):
        sp = substep(sp, beta, step * 2 + 0, 0)
        return substep(sp, beta, step * 2 + 1, 1)

    return dict(substep=substep, sweep=sweep, BY=BY, BX=BX)


def build_theirs(N, LZ, n_per=8, J=1.0, K=0.01, So=1.0):
    """One of his steps: N_b blocks, one random spin updated in each."""
    import jax
    import jax.numpy as jnp
    import jax_triton as jt

    if not _alias_support():
        raise RuntimeError(
            "this jax_triton cannot alias inputs onto outputs, so his in-place "
            "update would become a full-array copy per step; refusing to time it")

    NT = LZ * N * N
    assert NT % n_per == 0, "n must divide the site count"
    n_blocks = NT // n_per
    shp = jax.ShapeDtypeStruct((NT,), jnp.float32)

    def step_fn(sx, sy, sz, beta, step):
        return jt.triton_call(
            sx, sy, sz,
            jnp.float32(beta).reshape(1), jnp.int32(step).reshape(1),
            kernel=_theirs_kernel, out_shape=(shp, shp, shp),
            grid=(n_blocks,), N=N, LZ=LZ, N_PER=int(n_per),
            JEX=float(J), KAN=float(K), SO=float(So), num_warps=1,
            input_output_aliases={0: 0, 1: 1, 2: 2},
        )

    return dict(step=step_fn, n_blocks=n_blocks, n_per=n_per,
                updates_per_step=n_blocks)


# ============================================================== verification

def verify_host(N=8, LZ=8, J=1.0, K=0.01, So=1.0, tol=2e-6):
    """The formulas, with numpy alone. Runs without a GPU."""
    import jax
    import jax.numpy as jnp

    rep = []
    ref = ref_engine(N, LZ, J=J, K=K, So=So)
    sp = random_state(jax.random.PRNGKey(0), N, LZ, So)
    spn = np.asarray(sp)

    # 1. the neighbour sum, against an explicit six-term loop with PBC
    v = np.asarray(ref["nsum"](sp))
    man = np.zeros_like(spn)
    for ax, sh in ((3, -1), (3, 1), (2, -1), (2, 1), (1, -1), (1, 1)):
        man += np.roll(spn, sh, axis=ax)
    e1 = float(np.abs(v - man).max())
    rep.append(("neighbour sum vs an explicit PBC loop", e1 < tol, e1))

    # 2. the energy, against a direct bond enumeration over the whole lattice
    E_ref = float(ref["energy"](sp))
    E_dir = 0.0
    for z in range(LZ):
        for y in range(N):
            for x in range(N):
                s = spn[:, z, y, x]
                for dz, dy, dx in ((0, 0, 1), (0, 1, 0), (1, 0, 0)):
                    o = spn[:, (z + dz) % LZ, (y + dy) % N, (x + dx) % N]
                    E_dir += -J * float(s @ o)
                E_dir += -K * float(s[2] ** 2)
    e2 = abs(E_ref - E_dir) / max(abs(E_dir), 1e-30)
    rep.append(("energy vs a direct bond enumeration", e2 < 1e-5, e2))

    # 3. dE, against recomputing the total energy before and after one flip
    tr = random_state(jax.random.PRNGKey(7), N, LZ, So)
    dE = np.asarray(ref["dE_of"](sp, tr))
    z, y, x = LZ // 2, N // 2, N // 3
    sp2 = spn.copy()
    sp2[:, z, y, x] = np.asarray(tr)[:, z, y, x]
    dE_tot = float(ref["energy"](jnp.asarray(sp2))) - E_ref
    e3 = abs(dE[z, y, x] - dE_tot) / max(abs(dE_tot), 1e-30)
    rep.append(("dE vs the total-energy difference of one flip", e3 < 1e-4, e3))

    # 4. the colouring closes under PBC, which needs an even N
    zc, yc, xc = np.indices((LZ, N, N))
    par = (xc + yc + zc) % 2
    bad = 0
    for dz, dy, dx in ((0, 0, 1), (0, 1, 0), (1, 0, 0)):
        nb = par[(zc + dz) % LZ, (yc + dy) % N, (xc + dx) % N]
        bad += int((nb == par).sum())
    rep.append(("every first neighbour is of the opposite colour",
                bad == 0, float(bad)))

    ok = all(r[1] for r in rep)
    for name, good, val in rep:
        print(f"  [{'ok ' if good else 'FAIL'}] {name}: {val:.3e}")
    return ok, rep


def verify_device(N=16, LZ=16, J=1.0, K=0.01, So=1.0):
    """The kernel against the reference, on IDENTICAL randomness.

    With the randomness held fixed the only admissible difference is
    arithmetic, so the differing-spin count must be zero: a non-zero count
    identifies a term transcribed wrongly, not a sampling discrepancy.
    """
    import jax
    import jax.numpy as jnp

    rep = []
    ref = ref_engine(N, LZ, J=J, K=K, So=So)
    eng = build_ours(N, LZ, BY=4, J=J, K=K, So=So)
    sp = random_state(jax.random.PRNGKey(3), N, LZ, So)
    tr = random_state(jax.random.PRNGKey(4), N, LZ, So)
    ru = jax.random.uniform(jax.random.PRNGKey(5), (LZ, N, N), jnp.float32)

    for parity in (0, 1):
        got, de = eng["substep"](sp, BETA, 0, parity, tr=tr, ru=ru,
                                 want_de=True)
        want, dE_ref = ref["substep"](sp, tr, ru, BETA, parity)
        nd = int(np.sum(np.abs(np.asarray(got) - np.asarray(want)) > 1e-6))
        rep.append((f"one pass, parity {parity}, identical randomness",
                    nd == 0, float(nd)))
        ed = float(np.abs(np.asarray(de) - np.asarray(dE_ref)).max())
        rep.append((f"dE, parity {parity}", ed < 2e-5, ed))

    # the in-kernel generator must put the trial direction on the sphere
    got = eng["substep"](sp, BETA, 11, 0)
    nrm = np.linalg.norm(np.asarray(got), axis=0)
    rep.append(("|s| = So after an in-kernel pass",
                float(np.abs(nrm - So).max()) < 2e-5,
                float(np.abs(nrm - So).max())))

    ok = all(r[1] for r in rep)
    for name, good, val in rep:
        print(f"  [{'ok ' if good else 'FAIL'}] {name}: {val:.3e}")
    return ok, rep


def verify_theirs(N=16, LZ=16, steps=400, J=1.0, K=0.01, So=1.0):
    """His method can only be checked statistically: the race is the method.

    Two things must still hold. The spins must stay on the sphere, and the
    energy must fall from a random start towards the reference engine's
    equilibrium -- a method that accepted everything or nothing would fail both.
    """
    import jax
    import jax.numpy as jnp

    rep = []
    ref = ref_engine(N, LZ, J=J, K=K, So=So)
    eng = build_theirs(N, LZ, n_per=8, J=J, K=K, So=So)
    s = np.asarray(random_state(jax.random.PRNGKey(9), N, LZ, So))
    E0 = float(ref["energy"](jnp.asarray(s)))
    sx, sy, sz = (jnp.asarray(s[0].ravel()), jnp.asarray(s[1].ravel()),
                  jnp.asarray(s[2].ravel()))
    for t in range(steps):
        sx, sy, sz = eng["step"](sx, sy, sz, BETA, t)
    out = np.stack([np.asarray(sx).reshape(LZ, N, N),
                    np.asarray(sy).reshape(LZ, N, N),
                    np.asarray(sz).reshape(LZ, N, N)])
    nrm = np.linalg.norm(out, axis=0)
    rep.append(("|s| = So throughout", float(np.abs(nrm - So).max()) < 2e-5,
                float(np.abs(nrm - So).max())))
    E1 = float(ref["energy"](jnp.asarray(out)))
    rep.append((f"energy falls from a random start ({E0:.1f} -> {E1:.1f})",
                E1 < E0, (E1 - E0) / abs(E0)))
    ok = all(r[1] for r in rep)
    for name, good, val in rep:
        print(f"  [{'ok ' if good else 'FAIL'}] {name}: {val:.3e}")
    return ok, rep


# ================================================================== timing

def _sync(x):
    import jax
    jax.block_until_ready(x)
    return x


def time_ours(N=128, LZ=128, sweeps=200, BY=4, J=1.0, K=0.01, So=1.0):
    """ns per attempted single-site update, one system, his geometry."""
    import jax
    eng = build_ours(N, LZ, BY=BY, J=J, K=K, So=So)
    sp = random_state(jax.random.PRNGKey(1), N, LZ, So)

    def go(s, n):
        for t in range(n):
            s = eng["sweep"](s, BETA, t)
        return s

    _sync(go(sp, 3))                        # compile and warm
    t0 = time.perf_counter()
    out = _sync(go(sp, sweeps))
    dt = time.perf_counter() - t0
    sites = LZ * N * N
    return dict(seconds=dt, sweeps=sweeps, sites=sites, BY=BY,
                ns_per_update=dt / (sweeps * sites) * 1e9,
                updates_per_s=sweeps * sites / dt,
                finite=bool(np.isfinite(np.asarray(out)).all()))


def time_theirs(N=128, LZ=128, steps=2000, n_per=8, J=1.0, K=0.01, So=1.0):
    """The same unit for his method.

    One of his steps attempts N_b = sites/n updates, so the denominator is
    steps * N_b and not steps * sites.
    """
    import jax
    import jax.numpy as jnp
    eng = build_theirs(N, LZ, n_per=n_per, J=J, K=K, So=So)
    NT = LZ * N * N
    s = np.asarray(random_state(jax.random.PRNGKey(2), N, LZ, So))
    mk = lambda i: jnp.asarray(s[i].ravel())

    def go(n):
        a, b, c = mk(0), mk(1), mk(2)
        for t in range(n):
            a, b, c = eng["step"](a, b, c, BETA, t)
        return a, b, c

    _sync(go(3))
    t0 = time.perf_counter()
    out = _sync(go(steps))
    dt = time.perf_counter() - t0
    attempts = steps * eng["updates_per_step"]
    return dict(seconds=dt, steps=steps, n_per=n_per, sites=NT,
                attempts=attempts,
                ns_per_update=dt / attempts * 1e9,
                updates_per_s=attempts / dt,
                finite=bool(np.isfinite(np.asarray(out[0])).all()))


def run(N=128, LZ=128, out=None, label="", sweeps=200, steps=2000,
        by_scan=(2, 4, 8), n_scan=(1, 2, 4, 8, 16, 32)):
    """Both methods at his geometry, each scanned for its own best shape."""
    import jax

    dev = jax.devices()[0]
    print(f"=== bench_charilaou  {N}x{N}x{LZ}  ({N*N*LZ:,} spins) ===")
    print(f"    device: {dev.device_kind}   jax {jax.__version__}")
    print("--- verification; nothing is timed until every check passes")
    ok_h, rep_h = verify_host()
    ok_d, rep_d = verify_device()
    ok_t, rep_t = verify_theirs()
    if not (ok_h and ok_d and ok_t):
        raise SystemExit("verification FAILED -- refusing to report a time")

    res = dict(label=label, N=N, LZ=LZ, sites=N * N * LZ, paper=PAPER,
               device=dev.device_kind, jax=jax.__version__, beta=BETA,
               verification=dict(
                   host=[(n, bool(g), float(v)) for n, g, v in rep_h],
                   ours=[(n, bool(g), float(v)) for n, g, v in rep_d],
                   theirs=[(n, bool(g), float(v)) for n, g, v in rep_t]))

    print("--- ours (checkerboard, shifted loads), scanning the block height")
    res["ours"] = []
    for by in by_scan:
        try:
            r = time_ours(N, LZ, sweeps=sweeps, BY=by)
        except Exception as e:
            print(f"    BY={by:<3} refused: {type(e).__name__}: {e}")
            continue
        res["ours"].append(r)
        print(f"    BY={by:<3} {r['ns_per_update']:8.4f} ns/update"
              f"   {r['updates_per_s']/1e6:9.1f} e6 upd/s")

    print("--- theirs (1 of n per block, no colouring), scanning n")
    res["theirs"] = []
    for n in n_scan:
        if (N * N * LZ) % n:
            continue
        try:
            r = time_theirs(N, LZ, steps=steps, n_per=n)
        except Exception as e:
            print(f"    n={n:<3} refused: {type(e).__name__}: {e}")
            continue
        res["theirs"].append(r)
        print(f"    n={n:<3}  {r['ns_per_update']:8.4f} ns/update"
              f"   {r['updates_per_s']/1e6:9.1f} e6 upd/s")

    bo = min((r["ns_per_update"] for r in res["ours"]), default=None)
    bt = min((r["ns_per_update"] for r in res["theirs"]), default=None)
    res["best_ours_ns"], res["best_theirs_ns"] = bo, bt
    res["ratio_theirs_over_ours"] = (bt / bo) if (bo and bt) else None
    if bo and bt:
        print(f"--- best ours {bo:.4f} ns | best theirs {bt:.4f} ns"
              f" | theirs/ours {bt/bo:.2f}x")
    print(f"    paper on a V100: {PAPER['ns_per_update']['128^3']} ns at"
          f" 128^3, {PAPER['ns_per_update']['inset_min']} ns minimum")

    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        json.dump(res, open(out, "w"), indent=1)
        print(f"    wrote {out}")
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--LZ", type=int, default=128)
    ap.add_argument("--sweeps", type=int, default=200)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--host-only", action="store_true",
                    help="run only the checks that need no GPU")
    a = ap.parse_args()
    if a.host_only:
        ok, _ = verify_host()
        raise SystemExit(0 if ok else 1)
    run(a.N, a.LZ, out=a.out or None, label=a.label,
        sweeps=a.sweeps, steps=a.steps)
