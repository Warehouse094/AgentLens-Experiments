"""Regenerate fig_ablation_study.pdf with the unified style."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import apply_style, PALETTE, panel_label

apply_style()
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "fig_ablation_study.pdf"

configs = ["Full\nsystem",
           "w/o Appeal\nMechanism",
           "w/o Verification\nLayer",
           "w/o Temporal\nCalibration"]
scores = [66.95, 53.46, 76.44, 64.21]
deltas = [0.0, -13.49, +9.49, -2.74]

# colour scheme: full system in primary; degradation in red; inflation in orange; minor in grey
colors = [PALETTE["primary"], PALETTE["danger"], PALETTE["secondary"], PALETTE["neutral"]]

fig, ax = plt.subplots(figsize=(5.6, 3.2))
fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.22)
x = np.arange(len(configs))
bars = ax.bar(x, scores, color=colors, edgecolor="#333", linewidth=0.5, width=0.62)

ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=9)
ax.set_ylabel("Mean composite score")
ax.set_ylim(45, 88)
ax.set_title("Ablation study: component contribution (Exp12)", pad=8)

# value labels (score + delta)
for bar, s, d in zip(bars, scores, deltas):
    ax.text(bar.get_x() + bar.get_width()/2, s + 1.6,
            f"{s:.2f}", ha="center", fontsize=9.5, fontweight="bold", color="#222")
    if d != 0:
        sign = "+" if d > 0 else ""
        col = PALETTE["secondary"] if d > 0 else PALETTE["danger"]
        # delta inside the bar near the top
        ax.annotate(f"Δ {sign}{d:.2f}",
                    xy=(bar.get_x() + bar.get_width()/2, s - 2.5),
                    ha="center", fontsize=8.4,
                    color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18", fc=col, ec="none"))

# baseline line at full system
ax.axhline(scores[0], color=PALETTE["primary"], linewidth=0.8,
           linestyle="--", alpha=0.55)
ax.grid(axis="x", visible=False)

panel_label(ax, "", x=-0.10, y=1.04)
fig.savefig(OUT)
fig.savefig(OUT.with_suffix(".png"), dpi=200)
print(f"saved: {OUT}")
