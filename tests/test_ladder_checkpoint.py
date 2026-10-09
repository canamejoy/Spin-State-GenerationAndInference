"""A resumed ladder must be bit-identical to an uninterrupted one.

This is the load-bearing property of the CPU arms: those runs take 11 hours, and
a checkpoint that silently restarts the chain, or reuses a random stream, wastes
all of it and produces a curve that looks plausible. So the test interrupts a
ladder mid-way and demands the *same numbers*, not similar ones.

It is exact rather than statistical because randomness is addressed by
temperature index -- `default_rng([seed, t_index])` -- instead of carried along
with the chain. That design choice is what makes an exact assertion possible, and
this test is what keeps it.
"""

import numpy as np
import pytest

import ladder_cpu
import sim_core

RD, L = 4.3, 3
TEMPS = [12.0, 8.0, 4.0, 1.0]
N_THERM, N_MEAS = 3, 4

PARAMS = dict(Jex=1.0, Kan1=0.177, Kan2=0.034, KanS=0.139, Hex=0.483,
              KDM=0.880, g=float(np.deg2rad(90.0)))


@pytest.fixture(scope="module")
def geo():
    return sim_core.build_geometry(Rd=RD, L=L)


def _run(geo, cell, checkpoint=None, stop_after=None):
    """Run the ladder, optionally aborting once `stop_after` steps are saved."""
    saved = {}

    class Stop(Exception):
        pass

    def save(ck):
        saved.clear()
        saved.update(ck)
        if stop_after is not None and ck["t_done"] >= stop_after:
            raise Stop

    try:
        out = ladder_cpu.run_ladder(cell, geo, PARAMS, TEMPS, N_THERM, N_MEAS,
                                    seed=7, checkpoint=checkpoint, save=save,
                                    log=lambda *a: None)
    except Stop:
        return None, saved
    return out, saved


@pytest.mark.parametrize("cell", ["dp_bp", "nodp_bp", "nodp_nobp"])
def test_resume_is_identical_to_uninterrupted(geo, cell):
    whole, _ = _run(geo, cell)
    _, mid = _run(geo, cell, stop_after=2)
    assert int(mid["t_done"]) == 2, "the interruption did not land where expected"

    resumed, _ = _run(geo, cell, checkpoint=mid)

    assert len(resumed["rows"]) == len(whole["rows"]) == len(TEMPS)
    for i, (a, b) in enumerate(zip(whole["rows"], resumed["rows"])):
        for k in ("T", "E", "M", "Mz", "Q", "Cv", "chi"):
            assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-9), (
                f"{cell}: observable {k} differs at T index {i} "
                f"({a[k]!r} vs {b[k]!r}) -- the resume is not reproducing the run")
    # The checkpoint stores float64, so this is exact too. An earlier version
    # stored float32 and the resumed chain diverged in the eighth digit.
    assert np.array_equal(whole["final"], resumed["final"]), (
        f"{cell}: the final configuration differs after a resume")


def test_checkpoint_carries_the_clock(geo):
    """Elapsed time accumulates across a resume instead of restarting at zero."""
    _, mid = _run(geo, "dp_bp", stop_after=2)
    resumed, _ = _run(geo, "dp_bp", checkpoint=mid)
    assert resumed["elapsed_s"] > mid["elapsed_s"] > 0.0, (
        "a resumed run reported less wall clock than the checkpoint it came from, "
        "so the reported ms/sweep would be wrong")


def test_observables_match_the_reference_engine(geo):
    """`observables` computes the same energy as `sim_core`, not its own."""
    import jax.numpy as jnp
    eng = sim_core.build_engine(geo)
    K1 = np.where(geo["surface"], PARAMS["KanS"], PARAMS["Kan1"])

    rng = np.random.default_rng(3)
    s = rng.normal(size=(L, geo["N"], geo["N"], 3))
    s /= np.linalg.norm(s, axis=-1, keepdims=True)
    s *= geo["disco"][..., None]

    E, M, Mz, Q = ladder_cpu.observables(s, K1, PARAMS, geo)
    E_ref = float(eng["energy"](jnp.asarray(s[None]), PARAMS, jnp.asarray(K1))[0])
    M_ref, Mz_ref = eng["magnetization"](jnp.asarray(s[None]))
    Q_ref = float(eng["topo_charge"](jnp.asarray(s[None]))[0, L // 2])

    assert E == pytest.approx(E_ref, rel=1e-5)
    assert M == pytest.approx(float(M_ref[0]), rel=1e-5)
    assert Mz == pytest.approx(float(Mz_ref[0]), rel=1e-5)
    assert Q == pytest.approx(Q_ref, abs=1e-4)
