"""How to size the launches, given how many chains there are and how many GPUs.

Step 1 of the parallelisation study measured that three things are free: the
number of devices (-0.2% over 1/2/4 GPUs at fixed per-device batch), the number
of launches (+0.9%), and the parameter values (-0.2%). What is not free is the
per-device batch B, which varies by 27% across the measured range. So the cost
of a plan is

    seconds = sum over launches of  B_i * sweeps * f(B_i)

because the devices of one launch work in parallel, and the launches run one
after another. Minimising that over partitions of the work is a shortest-path
problem, not a search: `plan` solves it exactly with dynamic programming.

This replaces the rule of thumb "aim for 50 chains per GPU", which is only
right when 50 happens to tile the work. With three states instead of four it
does not, and the best plan is a mix of two batch sizes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Plan:
    """A sequence of launches, each given by its per-device batch."""

    batches: list[int]
    devices: int
    total_chains: int          # after padding
    requested_chains: int
    waste_chains: int = 0
    f_table: dict = field(default_factory=dict, repr=False)

    @property
    def launches(self) -> list[int]:
        """Chains per launch, which is what the runner actually takes."""
        return [b * self.devices for b in self.batches]


def predicted_seconds(p: Plan, f_table: dict, sweeps: int) -> float:
    """Wall clock the cost model predicts, with f in microseconds."""
    return sum(b * sweeps * f_table[b] for b in p.batches) * 1e-6


def plan(chains: int, devices: int, f_table: dict, sweeps: int = 3_000_000,
         candidates: list[int] | None = None, allow_pad: bool = True,
         max_waste: float = 0.05) -> Plan:
    """Partition `chains` into launches that minimise predicted wall clock.

    `f_table` maps a per-device batch to its cost in microseconds per
    chain-sweep, as measured by `app.py::f_curve` in production column form.
    Only batches present in the table can be chosen, since this refuses to
    extrapolate a curve whose whole point is that it is not monotonic.
    """
    if devices < 1:
        raise ValueError(f"devices must be at least 1, got {devices}")
    if chains < 1:
        raise ValueError(f"chains must be at least 1, got {chains}")

    requested = chains
    base = ((chains + devices - 1) // devices) * devices
    if base != chains and not allow_pad:
        raise ValueError(
            f"{chains} chains is not divisible by {devices} devices; pass "
            "allow_pad=True to round up and discard the remainder")

    cand = sorted(set(candidates if candidates is not None else f_table))
    cand = [b for b in cand if b in f_table and b >= 1]
    if not cand:
        raise ValueError("no candidate batch is present in the f table")

    # A sparse grid leaves many unit counts unreachable: with {25,32,50,64,100}
    # there is no way to cover 38. So pad upward to the first total that both
    # divides by the device count AND can be tiled, rather than failing on an
    # arithmetic accident. The cap keeps the padding honest -- discarding a
    # tenth of the chains to hit a round batch is not a saving.
    limit = base + max(devices, int(max_waste * max(chains, 1)))
    cost = pick = units = None
    for total in range(base, limit + 1, devices):
        u = total // devices
        usable = [b for b in cand if b <= u]
        if not usable:
            continue
        INF = float("inf")
        c = [0.0] + [INF] * u
        pk = [0] * (u + 1)
        for n in range(1, u + 1):
            for b in usable:
                if b > n:
                    break
                v = b * f_table[b] + c[n - b]
                if v < c[n]:
                    c[n], pk[n] = v, b
        if c[u] < INF:
            cost, pick, units, chains = c, pk, u, total
            break
    if units is None:
        if not any(b <= base // devices for b in cand):
            raise ValueError(
                f"no candidate batch in {cand} fits {base // devices} "
                f"per-device units")
        raise ValueError(
            f"no combination of {cand} tiles {base // devices} per-device "
            f"units within {max_waste:.0%} padding")
    waste = chains - requested

    batches, n = [], units
    while n:
        batches.append(pick[n])
        n -= pick[n]
    # Largest first: it is the order a human reads as "the big launch, then the
    # remainder", and the cost is order-independent.
    batches.sort(reverse=True)
    return Plan(batches=batches, devices=devices, total_chains=chains,
                requested_chains=requested, waste_chains=waste,
                f_table=dict(f_table))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
# A plan's clock and its bill are different questions, and step 1 makes the
# second one surprising: GPU-seconds are
#
#     sum over launches of  f(B_i) * chain_sweeps_i
#
# with no device count in it. Devices divide the wall clock and leave the bill
# alone, because they were measured to cost -0.2% each. So renting more GPUs
# buys time at the same price, and the only thing that moves the bill is f(B)
# and how much work there is. Both numbers are reported because which one
# matters is the user's call, not the model's.

H100_USD_PER_GPU_SECOND = 0.001097      # modal.com/pricing, verified


@dataclass
class Report:
    plan: Plan
    states: int = 0
    devices: int = 0
    wall_seconds: float = 0.0
    gpu_seconds: float = 0.0
    usd: float = 0.0

    @property
    def usd_per_chain(self) -> float:
        return self.usd / self.plan.total_chains


def report(p: Plan, f_table: dict, sweeps: int,
           usd_per_gpu_second: float = H100_USD_PER_GPU_SECOND,
           states: int = 0) -> Report:
    """Wall clock, GPU-seconds and dollars for a plan."""
    wall = predicted_seconds(p, f_table, sweeps)
    gpu_s = wall * p.devices
    return Report(plan=p, states=states, devices=p.devices,
                  wall_seconds=wall, gpu_seconds=gpu_s,
                  usd=gpu_s * usd_per_gpu_second)


def grid(states: list[int], devices: list[int], replicas: int, f_table: dict,
         sweeps: int = 3_000_000, candidates: list[int] | None = None,
         usd_per_gpu_second: float = H100_USD_PER_GPU_SECOND) -> list[Report]:
    """The best plan for every combination of state count and device count.

    This is the table to read before committing hardware: each row is a choice
    the user can actually make, with the clock and the bill it implies and the
    chains it throws away when the work does not tile.
    """
    out = []
    for s in states:
        for d in devices:
            p = plan(s * replicas, d, f_table, sweeps=sweeps,
                     candidates=candidates)
            out.append(report(p, f_table, sweeps, usd_per_gpu_second, states=s))
    return out
