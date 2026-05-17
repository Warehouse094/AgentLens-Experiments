"""Regenerate fig_failure_taxonomy.pdf with the unified style."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import apply_style, PALETTE, panel_label

apply_style()
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "fig_failure_taxonomy.pdf"

# Data (Exp17, n=306)
modes = [
    "Environment Timeout",
    "Unauthorized Tool Escalation",
    "Policy Boundary Violation",
    "Missing Resource",
    "Reasoning Loop",
    "Safety Check Bypass",
    "Incomplete Task",
    "Score Dispute",
    "Other",
]
counts = [70, 58, 52, 36, 30, 24, 15, 12, 9]
total = sum(counts)
percents = [c / total * 100 for c in counts]

# Sorted descending
order = np.argsort(counts)[::-1]
modes_s = [modes[i] for i in order]
counts_s = [counts[i] for i in order]
percents_s = [percents[i] for i in order]

# Colors: severity-tinted palette
mode_colors = [
    PALETTE["danger"],     # timeout (env)
    PALETTE["accent"],     # tool escalation
    PALETTE["secondary"],  # policy violation
    PALETTE["lavender"],   # missing resource
    PALETTE["primary"],    # reasoning loop
    PALETTE["teal"],       # safety bypass
    PALETTE["olive"],      # incomplete task
    PALETTE["success"],    # score dispute
    PALETTE["neutral"],    # other
]

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6),
                         gridspec_kw={"width_ratios": [1, 1.25],
                                      "wspace": 0.45})
fig.subplots_adjust(left=0.04, right=0.97, top=0.86, bottom=0.16)

# Panel A: Donut chart
axA = axes[0]
wedges, _ = axA.pie(counts_s, colors=mode_colors,
                    startangle=90, counterclock=False,
                    wedgeprops=dict(width=0.42, edgecolor="white",
                                    linewidth=1.4))
# percentage labels around the donut
for w, p, m in zip(wedges, percents_s, modes_s):
    ang = (w.theta2 + w.theta1) / 2
    x = 1.10 * np.cos(np.deg2rad(ang))
    y = 1.10 * np.sin(np.deg2rad(ang))
    if p < 4:  # too small to label cleanly outside
        continue
    axA.text(x, y, f"{p:.0f}%", ha="center", va="center",
             fontsize=8.5, fontweight="bold", color="#222")
axA.set_title("Failure mode distribution\n(Exp17, n=306 cases)", pad=4)
# centre count
axA.text(0, 0, f"{total}\nfailures", ha="center", va="center",
         fontsize=11, fontweight="bold", color="#333")
panel_label(axA, "A", x=-0.05, y=1.02)
axA.grid(False)

# Panel B: Horizontal bar (sorted)
axB = axes[1]
ypos = np.arange(len(modes_s))[::-1]
bars = axB.barh(ypos, counts_s, color=mode_colors, edgecolor="#333",
                linewidth=0.4, height=0.65)
axB.set_yticks(ypos)
axB.set_yticklabels(modes_s, fontsize=9)
axB.set_xlabel("Number of failure cases")
axB.set_xlim(0, max(counts_s) * 1.18)
axB.set_title("Failure mode frequency (sorted)", pad=4)
axB.grid(axis="x", linestyle=":", color="#E5E5E5")
axB.grid(axis="y", visible=False)
for bar, c, p in zip(bars, counts_s, percents_s):
    axB.text(bar.get_width() + 1.2,
             bar.get_y() + bar.get_height() / 2,
             f"{c} ({p:.1f}%)",
             va="center", fontsize=8.6, color="#333")
panel_label(axB, "B", x=-0.16, y=1.02)

fig.savefig(OUT)
fig.savefig(OUT.with_suffix(".png"), dpi=200)
print(f"saved: {OUT}")
