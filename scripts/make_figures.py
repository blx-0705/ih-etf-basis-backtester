"""README figures from the outputs of run_pipeline.py (synthetic data)."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
IMG = ROOT / "docs" / "img"
BLUE, ORANGE, INK, MUTED, GRID, SURFACE = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "lines.linewidth": 2, "axes.axisbelow": True,
})


def day_view(leader="IH"):
    pos = pd.read_csv(OUT / f"leader_{leader}" / "minute_positions.csv", index_col=0, parse_dates=True)
    sig = pd.read_csv(OUT / f"leader_{leader}" / "signal.csv", index_col=0, parse_dates=True)
    day = pos.index.normalize().unique()[-2]
    p, s = pos.loc[pos.index.normalize() == day], sig.loc[sig.index.normalize() == day]
    x = range(len(p))                                  # bar number: no lunch-break gap
    ticks = [0, 60, 119, 180, 239]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, height_ratios=[1, 1.2])
    a1.plot(x, s["basis"], color=BLUE)
    a1.set_ylabel("basis (index pts)")
    a1.set_title(f"One session, framework {'F1' if leader == 'IH' else 'F2'}: signal target vs. executed position",
                 loc="left", fontsize=11)
    a2.step(x, p["target_ih_qty"], where="post", color=MUTED, linewidth=1.5, linestyle="--", label="target")
    a2.step(x, p["ih_pos"], where="post", color=ORANGE, label="executed IH position")
    a2.set_ylabel("IH contracts")
    a2.legend(frameon=False, loc="lower left")
    a2.set_xticks(ticks, [p.index[i].strftime("%H:%M") for i in ticks])
    fig.tight_layout()
    fig.savefig(IMG / "session_target_vs_position.png", dpi=150)


def cost_breakdown():
    segs = ["delay", "reprice", "queue", "chase"]
    data = {lead: pd.read_csv(OUT / f"leader_{lead}" / "cost_summary_points.csv", index_col=0)["ALL"]
            for lead in ("IH", "ETF")}
    fig, ax = plt.subplots(figsize=(8, 3.6))
    w = 0.38
    for k, (lead, color, name) in enumerate((("IH", BLUE, "F1: IH passive"), ("ETF", ORANGE, "F2: ETF passive"))):
        vals = [data[lead][s] for s in segs] + [data[lead]["total_cost"]]
        xs = [i + (k - 0.5) * (w + 0.02) for i in range(len(vals))]
        ax.bar(xs, vals, width=w, color=color, label=name)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(range(len(segs) + 1), segs + ["total"])
    ax.set_ylabel("index pts per hedge unit")
    ax.set_title("Implementation shortfall by segment (positive = cost)", loc="left", fontsize=11)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(IMG / "cost_breakdown.png", dpi=150)


if __name__ == "__main__":
    IMG.mkdir(parents=True, exist_ok=True)
    day_view()
    cost_breakdown()
    print(f"figures written to {IMG}")
