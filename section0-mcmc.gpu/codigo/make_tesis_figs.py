"""The five figures of the thesis section, as PDF for LaTeX.

Every number is read from the measurement files in `resultados/`; nothing is
typed in by hand except the labels. Figure 1 is the only drawn one -- it is the
summary diagram the Materials and Methods contract makes mandatory, and it
carries the four components in the order that section defines them.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
SEC = os.path.normpath(os.path.join(HERE, ".."))
CUR = os.path.join(SEC, "resultados", "curvas")
KAG = os.path.join(SEC, "resultados", "kaggle")
OUT = os.path.join(SEC, "tesis", "figuras")
SITES, FULL = 5245, 200 * 20000

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9.5, "legend.fontsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
GREEN, GREY, BLUE, SAND = "#2e7d32", "#8a8a8a", "#2b5d8a", "#b8860b"
J = lambda p: json.load(open(p))


def fig1_diagram():
    """The summary diagram: the four components of one fused pass."""
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
    # the kernel boundary
    ax.add_patch(FancyBboxPatch((0.45, 0.35), 9.1, 3.5,
                                boxstyle="round,pad=0.12", linestyle="--",
                                linewidth=1.0, edgecolor="#555",
                                facecolor="none"))
    ax.text(5.0, 4.25, "one pass of one colour class, inside the kernel",
            ha="center", fontsize=8.5, color="#444")
    boxes = [
        (1.0, 2.35, 2.2, 1.0, BLUE,
         "(1) contiguous\nneighbour access", "component-planar\n$(B,3,L,N,N)$"),
        (4.0, 2.35, 2.2, 1.0, GREEN,
         "(2) synchronisation-\nfree halo", "shifted contiguous\nreads, no barrier"),
        (7.0, 2.35, 2.2, 1.0, SAND,
         "(3) counter-based\nproposals", "one key,\nfour counters"),
        (4.0, 0.65, 2.2, 0.95, GREY,
         "(4) deterministic\nobservables", "per-block partials"),
    ]
    for x, y, w, h, c, title, sub in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    linewidth=1.3, edgecolor=c,
                                    facecolor=c, alpha=0.11))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="center",
                fontsize=8.4, color=c, weight="bold")
        ax.text(x + w / 2, y + 0.26, sub, ha="center", va="center",
                fontsize=7.4, color="#333")
    for a, b in ((3.2, 4.0), (6.2, 7.0)):
        ax.add_patch(FancyArrowPatch((a, 2.85), (b, 2.85),
                                     arrowstyle="-|>", mutation_scale=11,
                                     linewidth=1.0, color="#555"))
    ax.add_patch(FancyArrowPatch((5.1, 2.35), (5.1, 1.60), arrowstyle="-|>",
                                 mutation_scale=11, linewidth=1.0,
                                 color="#555"))
    ax.text(5.1, 1.98, "accepted configuration", ha="center", fontsize=7.2,
            color="#555")
    fig.savefig(os.path.join(OUT, "fig1_diagram.pdf"))
    plt.close(fig)


def _ladder_rows(repl):
    """Full-ladder seconds for one replica count, per hardware arm."""
    tag = {50: "state_ladder", 500: "states10full", 5000: "states100"}[repl]
    out = {}
    for nd in (1, 2, 4):
        p = os.path.join(CUR, f"{tag}_{nd}gpu.json")
        if not os.path.exists(p):
            continue
        r = J(p)["rows"][0]
        if "kernel" in r:
            out[f"{nd}x H100"] = r["kernel"]["us_per_replica_sweep"] \
                * FULL * repl / 1e6
    kf = {50: "triton_states_t4_full", 5000: "triton_states_t4_100"}.get(repl)
    if kf:
        for r in J(os.path.join(KAG, kf + ".json"))["rows"]:
            out[f"{r['devices']}x T4"] = \
                r["optimizado"]["us_per_replica_sweep"] * FULL * repl / 1e6
    cpu = min(x["us_per_replica_sweep"]
              for x in J(os.path.join(SEC, "resultados",
                                      "cpu_rate_repeat.json"))["rows"])
    out["CPU, 24 thr"] = cpu * FULL * repl / 1e6
    return out


def _fmt(s):
    if s < 5400:
        return f"{s/60:.1f} min"
    if s < 86400 * 2:
        return f"{s/3600:.1f} h"
    return f"{s/86400:.1f} d"


def fig23_flagship():
    """The production ladder at a hundred states, with the CPU and without."""
    d = _ladder_rows(5000)
    order = ["CPU, 24 thr", "1x T4", "2x T4", "1x H100", "2x H100", "4x H100"]
    for name, keys, logy in (("fig2_flagship_cpu", order, True),
                             ("fig3_flagship_gpu", order[1:], False)):
        ks = [k for k in keys if k in d]
        v = [d[k] for k in ks]                      # seconds
        h = [x / 3600 for x in v]                   # the axis reads in hours
        fig, ax = plt.subplots(figsize=(5.4, 2.9))
        col = [GREY if "CPU" in k else (BLUE if "T4" in k else GREEN)
               for k in ks]
        b = ax.bar(np.arange(len(ks)), h, color=col, width=.62, alpha=.88)
        ax.set_xticks(np.arange(len(ks)))
        ax.set_xticklabels([k.replace(", ", "\n") for k in ks])
        if logy:
            # a log axis cannot start at zero, and leaving the limit unset
            # lets the bars run past their own labels
            ax.set_yscale("log")
            ax.set_ylim(min(h) * 0.30, max(h) * 6.0)
        else:
            ax.set_ylim(0, max(h) * 1.22)
        for r, sec in zip(b, v):
            y = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2,
                    y * 1.18 if logy else y + max(h) * .025,
                    _fmt(sec), ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("wall clock of the full ladder (hours"
                      + (", log)" if logy else ")"))
        fig.savefig(os.path.join(OUT, name + ".pdf"))
        plt.close(fig)


def fig4_floor():
    """Per-site cost per device against replicas per device."""
    pts = {"H100": [], "T4": []}
    for tag, repl in (("state_ladder", 50), ("states10full", 500),
                      ("states100", 5000)):
        for nd in (1, 2, 4):
            p = os.path.join(CUR, f"{tag}_{nd}gpu.json")
            if not os.path.exists(p):
                continue
            dd = J(p)
            for r in dd["rows"]:
                if "kernel" not in r:
                    continue
                pts["H100"].append((r["per_device"],
                                    r["kernel"]["us_per_replica_sweep"]
                                    * dd["n_devices"] * 1000 / SITES))
    for kf in ("triton_states_t4_full", "triton_states_t4_100"):
        for r in J(os.path.join(KAG, kf + ".json"))["rows"]:
            pts["T4"].append((r["per_device"],
                              r["optimizado"]["us_per_replica_sweep"]
                              * r["devices"] * 1000 / SITES))
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    for k, c, m in (("H100", GREEN, "o"), ("T4", BLUE, "s")):
        p = sorted(pts[k])
        ax.plot([x for x, _ in p], [y for _, y in p], m + "-", color=c,
                lw=1.3, ms=4.2, label=k)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("replicas per device")
    ax.set_ylabel("ns per site update, per device")
    ax.axvspan(10, 25, color=GREY, alpha=.13)
    ax.text(16, ax.get_ylim()[0] * 1.22, "floor", ha="center", fontsize=7.5,
            color="#555")
    ax.grid(alpha=.22, which="both"); ax.legend()
    fig.savefig(os.path.join(OUT, "fig4_floor.pdf"))
    plt.close(fig)


def fig5_shape():
    """Cost against the replica count, by layer count: the extent is immaterial."""
    rows = J(os.path.join(CUR, "triton_width.json"))["rows"]
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    for Lv, c in zip(sorted({r["L"] for r in rows}),
                     (GREEN, BLUE, SAND, GREY)):
        rr = sorted((r for r in rows if r["L"] == Lv), key=lambda r: r["B"])
        ax.plot([r["B"] for r in rr], [r["halo_us"] for r in rr], "o-",
                color=c, lw=1.3, ms=4.2, label=f"$L={Lv}$")
    ax.set_xlabel("replicas per device $B$")
    ax.set_ylabel(r"$\mu$s per replica-sweep")
    ax.grid(alpha=.22)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              frameon=False, handlelength=1.6, columnspacing=1.4)
    fig.savefig(os.path.join(OUT, "fig5_shape.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig1_diagram(); fig23_flagship(); fig4_floor(); fig5_shape()
    for f in sorted(os.listdir(OUT)):
        print(f"  {f:<28}{os.path.getsize(os.path.join(OUT, f))//1024} KB")
