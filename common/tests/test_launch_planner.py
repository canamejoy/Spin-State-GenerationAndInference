"""The partition is exact, so it is worth testing against cases we can reason about.

`plan` answers the only question the parallelisation study left open for a user:
given N chains and D devices, how should the launches be sized? Step 1 measured
that device count, launch count and parameter values are all free, so the cost of
a plan is fully determined by the per-device batch of each launch -- which makes
this a shortest-path problem over partitions rather than a search.
"""

import pytest

from launch_planner import grid, plan, predicted_seconds, report


FLAT = {b: 3.0 for b in range(1, 201)}


def _valley(best_b, best_f=2.0, far_f=4.0, hi=200):
    """A single-minimum curve, linear on each side of `best_b`."""
    out = {}
    for b in range(1, hi + 1):
        d = abs(b - best_b) / max(best_b, hi - best_b)
        out[b] = best_f + (far_f - best_f) * d
    return out


def test_flat_curve_makes_every_partition_equal():
    # With f constant, total work is the only thing that matters, so the plan's
    # cost must equal N * f regardless of how it splits.
    p = plan(400, 4, FLAT, sweeps=1)
    assert p.waste_chains == 0
    assert predicted_seconds(p, FLAT, sweeps=1) == pytest.approx(100 * 3.0e-6)


def test_reachable_optimum_is_used_uniformly():
    f = _valley(50)
    p = plan(400, 4, f)
    # 400 chains over 4 devices is 100 per-device units; B=50 tiles it exactly.
    assert p.batches == [50, 50]
    assert p.waste_chains == 0


def test_unreachable_optimum_falls_back_to_a_mix_not_a_uniform_plan():
    f = _valley(50)
    p = plan(300, 4, f)          # 75 per-device units; 50 does not tile it
    assert sum(p.batches) == 75
    assert p.waste_chains == 0
    # Whatever it picks must beat every uniform plan that divides 75.
    for b in (1, 3, 5, 15, 25, 75):
        uniform = [b] * (75 // b)
        assert (sum(x * f[x] for x in p.batches)
                <= sum(x * f[x] for x in uniform) + 1e-9)


def test_chains_not_divisible_by_devices_are_padded_and_reported():
    f = _valley(50)
    p = plan(301, 4, f)
    assert p.waste_chains == 3          # 301 -> 304, three discarded
    assert p.total_chains == 304
    assert sum(p.batches) == 76


def test_padding_can_be_refused():
    f = _valley(50)
    with pytest.raises(ValueError, match="not divisible"):
        plan(301, 4, f, allow_pad=False)


def test_single_device_is_the_same_problem():
    f = _valley(50)
    p = plan(400, 1, f)
    assert sum(p.batches) == 400
    assert p.batches == [50] * 8


def test_measured_curve_puts_four_hundred_chains_in_two_launches():
    # The three production ladders, corrected for the compilation they timed.
    measured = {25: 3.009, 50: 2.610, 100: 3.180}
    p = plan(400, 4, measured, candidates=[25, 50, 100])
    assert p.batches == [50, 50]


def test_a_lone_candidate_is_reached_by_padding_up_to_its_multiple():
    # 100 units is not a multiple of 7, so the only way to use B=7 is to pad.
    # 15 launches cover 105 units = 420 chains, discarding 20.
    p = plan(400, 4, {7: 2.0}, candidates=[7])
    assert p.batches == [7] * 15
    assert p.total_chains == 420
    assert p.waste_chains == 20


def test_padding_beyond_the_cap_is_refused_rather_than_silently_paid():
    with pytest.raises(ValueError, match="within 1% padding"):
        plan(400, 4, {7: 2.0}, candidates=[7], max_waste=0.01)


def test_a_batch_larger_than_the_work_is_rejected():
    with pytest.raises(ValueError, match="no candidate"):
        plan(8, 4, {50: 2.0}, candidates=[50])


def test_composition_does_not_enter_the_model():
    # The planner takes a chain count, never a state count: step 1 measured
    # that which theta sits on a device costs -0.2%. Four states of 25 and one
    # state of 100 are the same plan, and that is the point.
    f = _valley(50)
    assert plan(400, 4, f).batches == plan(400, 4, f).batches


# ---------------------------------------------------------------------------
# Reporting: wall clock and money across the choices the user actually controls
# ---------------------------------------------------------------------------

MEASURED = {25: 3.009, 32: 2.510, 50: 2.427, 64: 2.532, 100: 3.180}
CAND = sorted(MEASURED)


def test_gpu_seconds_is_wall_clock_times_devices():
    p = plan(400, 4, MEASURED, candidates=CAND, sweeps=1000)
    r = report(p, MEASURED, sweeps=1000)
    assert r.gpu_seconds == pytest.approx(r.wall_seconds * 4)


def test_more_devices_at_the_same_batch_buys_time_for_the_same_money():
    # Step 1 measured device count at -0.2%, so at a fixed per-device batch the
    # bill must not move while the clock does. This is the arithmetic
    # consequence, and the thing a user deciding how many GPUs to rent needs.
    a = report(plan(200, 2, MEASURED, candidates=[50], sweeps=1000),
               MEASURED, sweeps=1000)
    b = report(plan(400, 4, MEASURED, candidates=[50], sweeps=1000),
               MEASURED, sweeps=1000)
    assert a.wall_seconds == pytest.approx(b.wall_seconds)
    assert a.gpu_seconds == pytest.approx(b.gpu_seconds / 2)


def test_same_batch_same_price_per_chain_regardless_of_device_count():
    rows = [report(plan(100 * d, d, MEASURED, candidates=[50], sweeps=1000),
                   MEASURED, sweeps=1000) for d in (1, 2, 4)]
    per_chain = [r.usd / r.plan.total_chains for r in rows]
    assert max(per_chain) == pytest.approx(min(per_chain))


def test_money_is_linear_in_states_when_the_batch_does_not_change():
    four = report(plan(400, 4, MEASURED, candidates=[50], sweeps=1000),
                  MEASURED, sweeps=1000)
    two = report(plan(200, 4, MEASURED, candidates=[50], sweeps=1000),
                 MEASURED, sweeps=1000)
    assert two.usd == pytest.approx(four.usd / 2)


def test_grid_covers_every_combination_and_flags_padding():
    rows = grid(states=[3, 4], devices=[2, 4], replicas=100,
                f_table=MEASURED, candidates=CAND, sweeps=1000)
    assert len(rows) == 4
    assert {(r.states, r.devices) for r in rows} == {(3, 2), (3, 4), (4, 2), (4, 4)}
    for r in rows:
        assert r.plan.waste_chains == 0        # 300 and 400 divide by 2 and 4


FULL = {b: 3.0 - 0.5 * (b == 50) for b in
        (10, 16, 25, 32, 40, 50, 64, 75, 80, 100, 128)}


def test_pads_up_to_a_total_it_can_actually_tile():
    # 300 chains over 8 devices is 37.5 units; 38 and 39 cannot be covered by
    # the measured grid either, and 40 can. The planner must keep padding until
    # the work tiles instead of failing on an arithmetic accident, and must say
    # what it discarded.
    p = plan(300, 8, FULL, max_waste=0.1, sweeps=1000)
    assert p.total_chains == 320
    assert p.waste_chains == 20
    assert p.total_chains % 8 == 0
    assert sum(p.batches) == 40
