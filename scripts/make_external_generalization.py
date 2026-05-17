"""Regenerate fig_external_generalization.pdf with the unified style."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import apply_style, PALETTE, panel_label

apply_style()
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "fig_external_generalization.pdf"

domains = ["DAO Ops", "DeFi Risk", "Contract Review",
           "Incident Response", "Wallet Guard"]
# Cross-domain transfer (Exp25-27): rows = source, cols = target
# Diagonals are highest (within-domain), off-diagonals show transfer
M = np.array([
    [0.87, 0.74, 0.71, 0.69, 0.72],
    [0.73, 0.89, 0.71, 0.71, 0.71],
    [0.70, 0.69, 0.85, 0.72, 0.71],
    [0.68, 0.67, 0.72, 0.86, 0.70],
    [0.71, 0.70, 0.69, 0.73, 0.88],
])

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.7),
                         gridspec_kw={"width_ratios": [1.05, 1.0],
                                      "wspace": 0.40})
fig.subplots_adjust(left=0.10, right=0.96, top=0.86, bottom=0.18)

# --- Panel A: heatmap ---
axA = axes[0]
cmap = LinearSegmentedColormap.from_list(
    "cool_to_warm",
    ["#F5F5F5", "#A8C7E8", "#2E5C99", "#1F3A6B"],
)
im = axA.imshow(M, cmap=cmap, vmin=0.55, vmax=0.95, aspect="auto")
axA.set_xticks(range(len(domains)))
axA.set_yticks(range(len(domains)))
axA.set_xticklabels(domains, rotation=30, ha="right", fontsize=9)
axA.set_yticklabels(domains, fontsize=9)
axA.set_xlabel("Target domain")
axA.set_ylabel("Source domain")
axA.set_title("Cross-domain transfer agreement\n(Exp25–27, threshold transfer)",
              pad=4)
# annotate cells
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        col = "white" if v > 0.78 else "#222"
        axA.text(j, i, f"{v:.2f}", ha="center", va="center",
                 fontsize=8.6, color=col,
                 fontweight="bold" if i == j else "normal")
cbar = fig.colorbar(im, ax=axA, fraction=0.045, pad=0.03)
cbar.ax.tick_params(labelsize=8)
cbar.set_label("Agreement", fontsize=9)
axA.grid(False)
panel_label(axA, "A", x=-0.18, y=1.04)

# --- Panel B: External agent generalisation ---
axB = axes[1]
agents = ["GPT-4 Turbo", "Claude-3 Opus", "Gemini Pro"]
raw_corr = [0.682, 0.701, 0.693]
trans_agr = [0.745, 0.745, 0.739]

x = np.arange(len(agents))
w = 0.36
b1 = axB.bar(x - w/2, raw_corr, w, color=PALETTE["primary_light"],
             label="Raw score correlation", edgecolor="#333", linewidth=0.4)
b2 = axB.bar(x + w/2, trans_agr, w, color=PALETTE["primary"],
             label="Threshold-transferred agreement",
             edgecolor="#333", linewidth=0.4)
axB.set_xticks(x)
axB.set_xticklabels(agents, fontsize=9)
axB.set_ylim(0.55, 0.85)
axB.set_ylabel("Agreement / Correlation")
axB.set_title("External agent generalization\n(Exp06, Exp21)", pad=4)
for bar in list(b1) + list(b2):
    axB.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f"{bar.get_height():.3f}",
             ha="center", fontsize=8.6, color="#222")
axB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
           ncol=2, frameon=False, fontsize=8.6)
panel_label(axB, "B", x=-0.16, y=1.04)

fig.savefig(OUT)
fig.savefig(OUT.with_suffix(".png"), dpi=200)
print(f"saved: {OUT}")
