"""The four bar charts: one state and ten, with the CPU and without it.

Every bar is `us per replica-sweep x 4,000,000 sweeps x replicas`, so the
clock is the production ladder (200 T x 20 000) in every case, whatever the
length of the cell that was actually measured. That rate is the only thing
comparable across a CPU, a T4 and an H100, and the scaling in sweeps is exact
by construction.

What is measured and what is extrapolated is marked on every bar, because the
two deserve different trust: a cell measured at 40 of the 200 temperatures
carries the short-cell noise this project put at about +-2.5%.

Names: `estandar` is `sim_core`'s JAX engine, six shifted lattices and
Threefry. `optimizado` is the hand-written Triton stencil with its own Philox.
"""

from __future__ import annotations

import io
import json
import math

ROOT = "/home/andres/Spin-State-GenerationAndInference"
CUR = f"{ROOT}/notebooks/simulation/results/curves"
FULL = 200 * 20000              # the production ladder, per replica
SW_MEASURED_H100_1 = 200 * 20000        # the 50-replica arms ran full length
SW_MEASURED_H100_10 = 40 * 20000        # the 500-replica arms ran 40 of 200


def load():
    d = {}
    c = json.load(open(f"{ROOT}/notebooks/simulation/results/cpu_rate_repeat.json"))
    d["cpu_us"] = min(r["us_per_replica_sweep"] for r in c["rows"])
    d["cpu_threads"] = c["threads"]
    t4 = json.load(open(f"{ROOT}/results/kaggle/triton_bench_t4.json"))
    a = t4["ladder"]["arms"]
    d["t4_1_50"] = (a["produccion"]["us_per_replica_sweep"],
                    a["kernel"]["us_per_replica_sweep"])
    for n in (1, 2, 4):
        r = json.load(open(f"{CUR}/state_ladder_{n}gpu.json"))["rows"][0]
        d[f"h100_{n}_50"] = (r["produccion"]["us_per_replica_sweep"],
                             r["kernel"]["us_per_replica_sweep"])
        d[f"h100_{n}_50_repl"] = r["replicas"]
        s = json.load(open(f"{CUR}/states10_{n}gpu.json"))["rows"][0]
        d[f"h100_{n}_500"] = (s["produccion"]["us_per_replica_sweep"],
                              s["kernel"]["us_per_replica_sweep"])
    try:
        st = json.load(open(f"{ROOT}/results/kaggle/triton_states_t4.json"))
        for r in st["rows"]:
            k = f"t4_{r['devices']}_{r['replicas']}"
            d[k] = (r["estandar"]["us_per_replica_sweep"],
                    r["optimizado"]["us_per_replica_sweep"])
    except FileNotFoundError:
        print("  AVISO: aun no hay datos de T4 para estados; se omiten")
    return d


def clock(us, repl):
    """Seconds for the full production ladder at that rate."""
    return us * FULL * repl / 1e6


def fmt(s):
    if s < 60:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.1f} min"
    if s < 86400 * 2:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.1f} d"


def chart(rows, title, sub, log, w=1000, pad_l=150, pad_r=215):
    """One horizontal grouped bar chart as inline SVG.

    Returns a self-contained <svg>. A log axis is used only when the CPU bar
    is present, since it is three to four orders of magnitude above the rest
    and a linear axis would collapse everything else onto the spine.
    """
    bh, gap, grp = 20, 6, 16
    y = 58
    heights = []
    for label, bars in rows:
        heights.append(len(bars) * bh + (len(bars) - 1) * gap)
    h = y + sum(heights) + grp * len(rows) + 46
    plot_w = w - pad_l - pad_r
    vals = [v for _, bars in rows for _, v, _ in bars]
    if log:
        lo = 10 ** math.floor(math.log10(min(vals)))
        hi = 10 ** math.ceil(math.log10(max(vals)))
        xf = lambda v: pad_l + plot_w * (math.log10(v) - math.log10(lo)) / (
            math.log10(hi) - math.log10(lo))
        ticks = []
        t = lo
        while t <= hi + 1e-9:
            ticks.append(t)
            t *= 10
    else:
        hi = max(vals) * 1.02
        xf = lambda v: pad_l + plot_w * v / hi
        step = 10 ** math.floor(math.log10(hi))
        if hi / step < 2:
            step /= 5
        elif hi / step < 5:
            step /= 2
        ticks = [i * step for i in range(int(hi / step) + 2) if i * step <= hi]

    COL = {"estandar": "#b00020", "optimizado": "#2e7d32", "cpu": "#9e9e9e"}
    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" '
         f'style="max-width:100%;font-family:system-ui,sans-serif" '
         f'role="img" aria-label="{title}">']
    o.append(f'<text x="8" y="22" font-size="17" font-weight="600" '
             f'fill="currentColor">{title}</text>')
    o.append(f'<text x="8" y="42" font-size="12.5" fill="currentColor" '
             f'opacity=".72">{sub}</text>')
    for t in ticks:
        x = xf(t)
        o.append(f'<line x1="{x:.1f}" y1="{y - 6}" x2="{x:.1f}" y2="{h - 34}" '
                 f'stroke="currentColor" stroke-opacity=".14"/>')
        o.append(f'<text x="{x:.1f}" y="{h - 18}" font-size="11.5" '
                 f'text-anchor="middle" fill="currentColor" opacity=".7">'
                 f'{fmt(t)}</text>')
    cy = y
    for (label, bars), gh in zip(rows, heights):
        o.append(f'<text x="{pad_l - 10}" y="{cy + gh / 2 + 4:.0f}" '
                 f'font-size="13" text-anchor="end" font-weight="600" '
                 f'fill="currentColor">{label}</text>')
        for name, v, measured in bars:
            x = xf(v)
            # An extrapolated bar is drawn translucent and tagged with a
            # tilde, so a reader never takes one for a stopwatch reading.
            op = "" if measured else ' fill-opacity="0.6"'
            o.append(f'<rect x="{pad_l}" y="{cy}" '
                     f'width="{max(x - pad_l, 1):.1f}" height="{bh}" '
                     f'rx="2.5" fill="{COL[name]}"{op}/>')
            tag = "" if measured else " ≈"
            o.append(f'<text x="{x + 7:.1f}" y="{cy + bh - 5}" font-size="12" '
                     f'fill="currentColor">{fmt(v)}{tag}'
                     f'<tspan opacity=".62" font-size="11">  {name}</tspan>'
                     f'</text>')
            cy += bh + gap
        cy += grp - gap
    o.append('</svg>')
    return "\n".join(o)


# Which cells were timed at the full 200-temperature length and which are a
# rate scaled up to it. Only the 50-replica H100 arms ran full length.
MEASURED = {"h100_1_50", "h100_2_50", "h100_4_50"}


def build(d, repl, with_cpu):
    """The six bar groups for one replica count."""
    rows = []
    if with_cpu:
        rows.append((f"CPU, {d['cpu_threads']} hilos",
                     [("cpu", clock(d["cpu_us"], repl), False)]))
    for key, label in (("t4_1", "1x T4"), ("t4_2", "2x T4"),
                       ("h100_1", "1x H100"), ("h100_2", "2x H100"),
                       ("h100_4", "4x H100")):
        k = f"{key}_{repl}"
        if k not in d:
            continue
        e, o = d[k]
        m = k in MEASURED
        rows.append((label, [("estandar", clock(e, repl), m),
                             ("optimizado", clock(o, repl), m)]))
    return rows


def main():
    d = load()
    out = {}
    for repl, states in ((50, 1), (500, 10)):
        for with_cpu in (True, False):
            rows = build(d, repl, with_cpu)
            n = f"{states}estado{'s' if states > 1 else ''}_" \
                f"{'con' if with_cpu else 'sin'}cpu"
            sub = (f"{states} estado{'s' if states > 1 else ''}, "
                   f"{repl} réplicas, escalera completa de producción "
                   f"(200 T × 20 000 barridos). "
                   f"≈ = extrapolado desde una celda más corta")
            title = (f"{states} estado{'s' if states > 1 else ''} "
                     f"({repl} réplicas)"
                     f"{'' if with_cpu else ' — solo GPU'}")
            out[n] = chart(rows, title, sub, log=with_cpu)
    for n, svg in out.items():
        with io.open(f"/tmp/chart_{n}.svg", "w", encoding="utf-8") as f:
            f.write(svg)
    print("generadas:", ", ".join(out))
    # The numbers, printed so they can be checked against the bars.
    for repl, states in ((50, 1), (500, 10)):
        print(f"\n=== {states} estado(s), {repl} réplicas ===")
        for label, bars in build(d, repl, True):
            for name, v, m in bars:
                print(f"  {label:<16}{name:<12}{fmt(v):>9}"
                      f"{'' if m else '   (extrapolado)'}")
    return out


if __name__ == "__main__":
    main()
