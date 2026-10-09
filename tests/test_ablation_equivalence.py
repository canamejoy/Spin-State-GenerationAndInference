"""The ablation variants must differ in speed only -- never in physics.

An ablation that compares two implementations of *different* Hamiltonians
measures nothing, so these tests pin every variant to the reference engine in
``sim_core`` before any timing is believed:

1. the scalar single-site ``dE`` reproduces the reference total-energy
   difference, which is what makes the un-vectorised arm the same model;
2. a sublattice updated all at once equals the same sites updated one by one,
   given identical proposals and identical uniforms -- the exactness claim of
   the checkerboard, which has to hold to machine precision;
3. updating the *whole* disk at once does **not** equal the sequential chain,
   which is why switching the checkerboard off is a correctness change and not
   a speed setting.

Test 3 asserts a difference rather than an agreement. That is deliberate: the
test plan's ``SIN BP`` column is only meaningful if the naive parallel update is
demonstrably wrong, and a test that merely allowed it to be wrong would pass
just as happily on a silently correct implementation.
"""

import numpy as np
import pytest

import ablation
import sim_core

# Small enough to run in well under a second, large enough that the disk has an
# interior (sites with all six neighbours) as well as a surface.
RD, L = 4.3, 3

P = dict(Jex=1.0, Kan1=0.177, Kan2=0.034, KanS=0.139, Hex=0.483, KDM=0.880,
         g=float(np.deg2rad(90.0)))


@pytest.fixture(scope="module")
def setup():
    geo = sim_core.build_geometry(Rd=RD, L=L)
    K1 = np.where(geo["surface"], P["KanS"], P["Kan1"])
    rng = np.random.default_rng(0)
    s = rng.normal(size=(L, geo["N"], geo["N"], 3))
    s /= np.linalg.norm(s, axis=-1, keepdims=True)
    s *= geo["disco"][..., None]
    return geo, K1, s


def _proposal(geo, seed):
    """A trial direction and an acceptance uniform for every site."""
    rng = np.random.default_rng(seed)
    pr = rng.normal(size=(geo["L"], geo["N"], geo["N"], 3))
    pr /= np.linalg.norm(pr, axis=-1, keepdims=True)
    pr *= geo["disco"][..., None]
    return pr, rng.random((geo["L"], geo["N"], geo["N"]))


def test_scalar_dE_matches_reference_total_energy(setup):
    """The single-site dE is the reference Hamiltonian's difference, exactly."""
    import jax.numpy as jnp
    geo, K1, s = setup
    eng = sim_core.build_engine(geo)
    pr, _ = _proposal(geo, 1)

    # Every eighth active site, so interior and surface are both covered.
    sites = ablation.active_sites(geo)[::8]
    assert len(sites) > 5

    for idx in sites:
        z, y, x = idx
        trial = pr[z, y, x]
        dE = ablation.local_dE(s, (z, y, x), trial, P, K1, geo)

        after = s.copy()
        after[z, y, x] = trial
        E0 = float(eng["energy"](jnp.asarray(s[None]), P, jnp.asarray(K1))[0])
        E1 = float(eng["energy"](jnp.asarray(after[None]), P, jnp.asarray(K1))[0])
        assert dE == pytest.approx(E1 - E0, abs=2e-5), f"site {idx}"


def test_checkerboard_equals_sequential(setup):
    """Colour-by-colour in parallel == the same sites one at a time."""
    geo, K1, s = setup
    pr, u = _proposal(geo, 2)
    beta = 3.0

    par = ablation.sweep_vectorised(s, pr, u, beta, P, K1, geo,
                                    masks=(geo["even"], geo["odd"]))
    seq = ablation.sweep_scalar(s, pr, u, beta, P, K1, geo,
                                order=ablation.colour_order(geo))

    assert np.allclose(par, seq, atol=1e-6), (
        "the checkerboard update is not an exact reordering of the sequential "
        f"one; max deviation {np.abs(par - seq).max():.3e}")


def test_naive_parallel_update_is_not_the_sequential_chain(setup):
    """Dropping the colouring changes the chain, not just its speed."""
    geo, K1, s = setup
    pr, u = _proposal(geo, 3)
    beta = 3.0

    naive = ablation.sweep_vectorised(s, pr, u, beta, P, K1, geo,
                                      masks=(geo["disco"],))
    seq = ablation.sweep_scalar(s, pr, u, beta, P, K1, geo,
                                order=ablation.lattice_order(geo))

    assert not np.allclose(naive, seq, atol=1e-6), (
        "the un-coloured parallel update agreed with the sequential chain; "
        "either the proposals are not shared or the lattice has no interacting "
        "same-colour pairs, and the SIN BP arm would be measuring nothing")


def test_variant_registry_covers_the_four_cells(setup):
    """All four DP x BP cells exist and advance the state."""
    geo, K1, s = setup
    assert set(ablation.VARIANTS) == {"dp_bp", "dp_nobp", "nodp_bp", "nodp_nobp"}
    for name, fn in ablation.VARIANTS.items():
        out = fn(s, np.random.default_rng(7), 3.0, P, K1, geo)
        assert out.shape == s.shape, name
        assert np.isfinite(out).all(), name
        # Spins outside the disk stay zero, and inside they stay on the sphere.
        assert np.allclose(out[~geo["disco"]], 0.0), name
        n = np.linalg.norm(out[geo["disco"]], axis=-1)
        assert np.allclose(n, 1.0, atol=1e-6), name


def test_sitewise_batched_matches_the_scalar_chain(setup):
    """Batching over replicas changes the cost, never the chain.

    With one replica the batched site-by-site loop must reproduce the scalar one
    exactly. This is what lets the reference notebook's CPU mode be compared
    against the production kernel: the two differ only in where the array axis
    is, not in what is sampled.
    """
    geo, K1, s = setup
    pr, u = _proposal(geo, 11)
    beta = 3.0

    batched = ablation.sweep_sitewise_batched(
        s[None], pr[None], u[None], beta, P, K1, geo,
        order=ablation.lattice_order(geo))
    scalar = ablation.sweep_scalar(s, pr, u, beta, P, K1, geo,
                                   order=ablation.lattice_order(geo))

    assert np.allclose(batched[0], scalar, atol=1e-9), (
        "the replica-batched site loop is not the scalar chain; max deviation "
        f"{np.abs(batched[0] - scalar).max():.3e}")


def test_replicas_do_not_interact(setup):
    """Independent chains stay independent under the batched loop."""
    geo, K1, s = setup
    pr, u = _proposal(geo, 12)
    pr2, u2 = _proposal(geo, 13)
    beta = 3.0

    both = ablation.sweep_sitewise_batched(
        np.stack([s, s]), np.stack([pr, pr2]), np.stack([u, u2]),
        beta, P, K1, geo, order=ablation.lattice_order(geo))
    alone = ablation.sweep_sitewise_batched(
        s[None], pr[None], u[None], beta, P, K1, geo,
        order=ablation.lattice_order(geo))

    assert np.allclose(both[0], alone[0], atol=1e-12), (
        "replica 0 changed when a second replica was present, so the batch axis "
        "is coupling chains that must be independent")
