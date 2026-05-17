"""Unified visual style for AgentEval-Gov figures.

Designed to match the look-and-feel of fig2_audit_validity_composite.pdf
and fig3_prediction_composite.pdf.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Color palette ---
PALETTE = {
    "primary": "#2E5C99",      # deep blue (proposed/main metric)
    "primary_light": "#7FA8D8",
    "secondary": "#E89B2C",    # orange (highlight/contrast)
    "accent": "#A23B72",       # plum / appeal
    "danger": "#C0392B",       # red (failure / warning)
    "success": "#3B7A57",      # green (normal / safe)
    "neutral": "#6C757D",      # mid grey
    "neutral_light": "#D5D5D5",
    "teal": "#2C8C99",
    "lavender": "#9B86C9",
    "olive": "#9CB46B",
}

# Ordered palette (used for categorical bar/bar groups)
PALETTE_LIST = [
    PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"],
    PALETTE["success"], PALETTE["teal"], PALETTE["danger"],
    PALETTE["lavender"], PALETTE["olive"], PALETTE["neutral"],
]

VARIANT_COLORS = {
    "Normal": PALETTE["success"],
    "OverPermissioned": PALETTE["secondary"],
    "HighTaskLowSecurity": PALETTE["primary"],
    "EnvironmentBrittle": PALETTE["lavender"],
    "PolicyViolating": PALETTE["accent"],
}


def apply_style():
    """Apply the unified matplotlib style globally."""
    plt.rcdefaults()
    mpl.rcParams.update({
        # Font
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        # Spines: keep only left and bottom
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#333333",
        # Grid
        "axes.grid": True,
        "grid.color": "#E5E5E5",
        "grid.linestyle": ":",
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        # Background
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        # Saving
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        # Lines
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
    })


def panel_label(ax, label, x=-0.16, y=1.10, fontsize=13):
    """Add a bold (A)/(B)/(C) panel label in the top-left corner."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold",
            ha="left", va="bottom", color="#1a1a1a")


def hide_spines(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def value_labels_on_bars(ax, bars, fmt="{:.1f}", offset=0.5, fontsize=9, color="#333"):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                fmt.format(h), ha="center", va="bottom",
                fontsize=fontsize, color=color)


def value_labels_on_hbars(ax, bars, fmt="{:.0f}%", offset=0.5, fontsize=9, color="#333"):
    for bar in bars:
        w = bar.get_width()
        ax.text(w + offset, bar.get_y() + bar.get_height() / 2,
                fmt.format(w), ha="left", va="center",
                fontsize=fontsize, color=color)
