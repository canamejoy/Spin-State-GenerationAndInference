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
SEC = "section0-mcmc.gpu"
CUR = f"{ROOT}/{SEC}/resultados/curvas"
FULL = 200 * 20000              # the production ladder, per replica
SW_MEASURED_H100_1 = 200 * 20000        # the 50-replica arms ran full length
SW_MEASURED_H100_10 = 40 * 20000        # the 500-replica arms ran 40 of 200


# How far each cell was scaled to reach the 200-temperature ladder. 1 means
# the stopwatch read it directly; `load` sets 1 on any cell a full-length run
# later measured. Carrying the FACTOR rather than a boolean matters, since a
# x5 stretch and a x20 one do not deserve the same trust and a bare tilde
# hides the difference.
FACTOR = {
    "h100_1_50": 1, "h100_2_50": 1, "h100_4_50": 1,      # 200 T, measured
    "h100_1_500": 5, "h100_2_500": 5, "h100_4_500": 5,   # 40 T
    "t4_1_50": 20, "t4_2_50": 20,                        # 10 T, overridden
    "t4_1_500": 20, "t4_2_500": 20,
    "h100_1_5000": 100, "h100_2_5000": 100,                # 2 T, 100 states
    "h100_4_5000": 100,
    "t4_1_5000": 100, "t4_2_5000": 100,                    # 2 T, 100 states
}
# A cell where only one of the two arms was timed at full length.
PARTIAL = {}

CPU_FACTOR = "tasa"      # the CPU bar comes from a rate, not a ladder run


def load():
    d = {}
    c = json.load(open(f"{ROOT}/{SEC}/resultados/cpu_rate_repeat.json"))
    d["cpu_us"] = min(r["us_per_replica_sweep"] for r in c["rows"])
    d["cpu_threads"] = c["threads"]
    t4 = json.load(open(f"{ROOT}/{SEC}/resultados/kaggle/triton_bench_t4.json"))
    a = t4["ladder"]["arms"]
    d["t4_1_50"] = (a["produccion"]["us_per_replica_sweep"],
                    a["kernel"]["us_per_replica_sweep"])
    for n in (1, 2, 4):
        r = json.load(open(f"{CUR}/state_ladder_{n}gpu.json"))["rows"][0]
        d[f"h100_{n}_50"] = (r["produccion"]["us_per_replica_sweep"],
                             r["kernel"]["us_per_replica_sweep"])
        d[f"h100_{n}_50_repl"] = r["replicas"]
        s = json.load(open(f"{CUR}/states10_{n}gpu.json"))["rows"][0]
        e, o = (s["produccion"]["us_per_replica_sweep"],
                s["kernel"]["us_per_replica_sweep"])
        # The 200-temperature run overrides each cell it actually timed. Its
        # one-device call skipped the standard arm, which alone cost more than
        # the rest of the table, so that cell keeps its x5 value and factor.
        try:
            f = json.load(open(f"{CUR}/states10full_{n}gpu.json"))["rows"][0]
            got = []
            if "produccion" in f:
                e = f["produccion"]["us_per_replica_sweep"]; got.append("e")
            if "kernel" in f:
                o = f["kernel"]["us_per_replica_sweep"]; got.append("o")
            if len(got) == 2:
                FACTOR[f"h100_{n}_500"] = 1
            else:
                PARTIAL[f"h100_{n}_500"] = got
        except FileNotFoundError:
            pass
        if n == 1:
            # the arm `state_ladder` was told to skip, timed on its own at
            # 80 of 200 temperatures
            try:
                m1 = json.load(open(f"{CUR}/missing_1gpu.json"))
                r1 = m1["rows"][0]
                e = r1["produccion"]["us_per_replica_sweep"]
                # ONLY the standard arm came from this 80-temperature run;
                # the optimised one was already timed over all 200, and
                # PARTIAL is what records that, so it must not be cleared.
                FACTOR["h100_1_500"] = round(200 / m1["n_temps"], 1)
            except FileNotFoundError:
                pass
        d[f"h100_{n}_500"] = (e, o)
        # 100 states
        try:
            c = json.load(open(f"{CUR}/states100_{n}gpu.json"))
            r = c["rows"][0]
            d[f"h100_{n}_5000"] = (r["produccion"]["us_per_replica_sweep"],
                                   r["kernel"]["us_per_replica_sweep"])
            FACTOR[f"h100_{n}_5000"] = round(200 / c["n_temps"])
        except FileNotFoundError:
            pass
    # The ten-temperature run supplies every T4 cell...
    try:
        st = json.load(open(f"{ROOT}/{SEC}/resultados/kaggle/triton_states_t4.json"))
        for r in st["rows"]:
            k = f"t4_{r['devices']}_{r['replicas']}"
            d[k] = (r["estandar"]["us_per_replica_sweep"],
                    r["optimizado"]["us_per_replica_sweep"])
    except FileNotFoundError:
        print("  AVISO: aun no hay datos de T4 para estados; se omiten")
    # ...and the full 200-temperature run overrides the one-state ones, which
    # it measured outright. Those keys then drop out of FACTOR, so the chart
    # draws them solid.
    # 100 states on the T4, measured over two temperatures
    try:
        c1 = json.load(open(f"{ROOT}/{SEC}/resultados/kaggle/triton_states_t4_100.json"))
        for r in c1["rows"]:
            d[f"t4_{r['devices']}_{r['replicas']}"] = (
                r["estandar"]["us_per_replica_sweep"],
                r["optimizado"]["us_per_replica_sweep"])
    except FileNotFoundError:
        pass
    try:
        fu = json.load(open(f"{ROOT}/{SEC}/resultados/kaggle/triton_states_t4_full.json"))
        for r in fu["rows"]:
            k = f"t4_{r['devices']}_{r['replicas']}"
            d[k] = (r["estandar"]["us_per_replica_sweep"],
                    r["optimizado"]["us_per_replica_sweep"])
            FACTOR[k] = 1
    except FileNotFoundError:
        pass
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


def chart(rows, title, sub, log, w=1000, h=470):
    """One grouped VERTICAL bar chart as inline SVG.

    A log axis is used only when the CPU bar is present, since it sits three
    to four orders of magnitude above the rest and a linear axis would leave
    everything else as a flat line on the floor. With a log axis the bars grow
    from the axis minimum rather than from zero, which is the only honest way
    to draw them.
    """
    pad_t, pad_b, pad_l, pad_r = 66, 62, 76, 18
    ph = h - pad_t - pad_b
    pw = w - pad_l - pad_r
    vals = [v for _, bars in rows for _, v, _ in bars]

    if log:
        lo = 10 ** math.floor(math.log10(min(vals)))
        hi = 10 ** math.ceil(math.log10(max(vals)))
        lg = math.log10
        yf = lambda v: pad_t + ph * (1 - (lg(v) - lg(lo)) / (lg(hi) - lg(lo)))
        ticks, t = [], lo
        while t <= hi + 1e-9:
            ticks.append(t)
            t *= 10
        base = pad_t + ph
    else:
        lo, hi = 0.0, max(vals) * 1.12
        yf = lambda v: pad_t + ph * (1 - v / hi)
        step = 10 ** math.floor(math.log10(hi))
        if hi / step < 2:
            step /= 5
        elif hi / step < 5:
            step /= 2
        ticks = [i * step for i in range(int(hi / step) + 2)
                 if i * step <= hi]
        base = pad_t + ph

    COL = {"estandar": "#b00020", "optimizado": "#2e7d32", "cpu": "#9e9e9e"}
    gw = pw / len(rows)
    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" '
         f'style="max-width:100%;font-family:system-ui,sans-serif" '
         f'role="img" aria-label="{title}">']
    o.append(f'<text x="8" y="22" font-size="17" font-weight="600" '
             f'fill="currentColor">{title}</text>')
    o.append(f'<text x="8" y="41" font-size="12.5" fill="currentColor" '
             f'opacity=".72">{sub}</text>')

    # legend, top right
    lx = w - pad_r
    for name in ("optimizado", "estandar"):
        if not any(n == name for _, bars in rows for n, _, _ in bars):
            continue
        tw = 8 * len(name) + 22
        lx -= tw
        o.append(f'<rect x="{lx}" y="48" width="11" height="11" rx="2" '
                 f'fill="{COL[name]}"/>')
        o.append(f'<text x="{lx + 16}" y="58" font-size="11.5" '
                 f'fill="currentColor" opacity=".8">{name}</text>')

    for t in ticks:
        y = yf(t)
        o.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w - pad_r}" '
                 f'y2="{y:.1f}" stroke="currentColor" stroke-opacity=".14"/>')
        o.append(f'<text x="{pad_l - 9}" y="{y + 4:.1f}" font-size="11.5" '
                 f'text-anchor="end" fill="currentColor" opacity=".7">'
                 f'{fmt(t) if t > 0 else "0"}</text>')
    o.append(f'<line x1="{pad_l}" y1="{base:.1f}" x2="{w - pad_r}" '
             f'y2="{base:.1f}" stroke="currentColor" stroke-opacity=".45"/>')

    for gi, (label, bars) in enumerate(rows):
        gx = pad_l + gi * gw
        bw = min(58.0, (gw - 24) / len(bars))
        span = bw * len(bars) + 7 * (len(bars) - 1)
        x0 = gx + (gw - span) / 2
        for bi, (name, v, fac) in enumerate(bars):
            x = x0 + bi * (bw + 7)
            y = yf(v)
            direct = fac == 1
            # A measured bar is solid. An extrapolated one is translucent AND
            # carries its factor, so the reader sees how far it was stretched.
            op = "" if direct else ' fill-opacity="0.6"'
            o.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" '
                     f'height="{max(base - y, 1):.1f}" rx="2.5" '
                     f'fill="{COL[name]}"{op}/>')
            o.append(f'<text x="{x + bw / 2:.1f}" y="{y - 6:.1f}" '
                     f'font-size="11.5" text-anchor="middle" '
                     f'fill="currentColor">{fmt(v)}</text>')
            if not direct:
                lab = fac if isinstance(fac, str) else f"\u00d7{fac}"
                o.append(f'<text x="{x + bw / 2:.1f}" y="{y - 19:.1f}" '
                         f'font-size="10" text-anchor="middle" '
                         f'fill="currentColor" opacity=".55">{lab}</text>')
        o.append(f'<text x="{gx + gw / 2:.1f}" y="{base + 20:.0f}" '
                 f'font-size="13" text-anchor="middle" font-weight="600" '
                 f'fill="currentColor">{label}</text>')
    o.append(f'<text x="{pad_l}" y="{h - 10}" font-size="11.5" '
             f'fill="currentColor" opacity=".6">'
             f'reloj de la escalera completa'
             f'{" (escala logar\u00edtmica)" if log else ""}</text>')
    o.append('</svg>')
    return "\n".join(o)


def build(d, repl, with_cpu):
    """The six bar groups for one replica count."""
    rows = []
    if with_cpu:
        rows.append((f"CPU, {d['cpu_threads']} hilos",
                     [("cpu", clock(d["cpu_us"], repl), CPU_FACTOR)]))
    for key, label in (("t4_1", "1x T4"), ("t4_2", "2x T4"),
                       ("h100_1", "1x H100"), ("h100_2", "2x H100"),
                       ("h100_4", "4x H100")):
        k = f"{key}_{repl}"
        if k not in d:
            continue
        e, o = d[k]
        m = FACTOR.get(k, 1)
        pa = PARTIAL.get(k, [])
        rows.append((label, [("estandar", clock(e, repl),
                              1 if "e" in pa else m),
                             ("optimizado", clock(o, repl),
                              1 if "o" in pa else m)]))
    return rows


def main():
    d = load()
    out = {}
    for repl, states in ((50, 1), (500, 10), (5000, 100)):
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
    for repl, states in ((50, 1), (500, 10), (5000, 100)):
        print(f"\n=== {states} estado(s), {repl} réplicas ===")
        for label, bars in build(d, repl, True):
            for name, v, fac in bars:
                mark = "" if fac == 1 else (f"   (extrapolado {fac})"
                                            if isinstance(fac, str)
                                            else f"   (extrapolado x{fac})")
                print(f"  {label:<16}{name:<12}{fmt(v):>9}{mark}")
    return out


if __name__ == "__main__":
    main()
