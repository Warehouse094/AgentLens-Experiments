from __future__ import annotations

import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import roc_auc_score, roc_curve

from pipeline_common import (
    DIMENSION_ORDER,
    FIGURES_DIR,
    RAW_DATA_FILES,
    RESULTS_DIR,
    build_scored_frame,
    ensure_output_dirs,
    load_jsonl,
)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 6,
})

VARIANT_COLORS = {
    "Normal": "#2A7F62",
    "OverPermissioned": "#F2A23A",
    "PolicyViolating": "#C83E4D",
    "EnvironmentBrittle": "#7A6FD6",
    "HighTaskLowSecurity": "#5D7285",
}
VARIANT_ORDER = [
    "Normal",
    "OverPermissioned",
    "HighTaskLowSecurity",
    "EnvironmentBrittle",
    "PolicyViolating",
]
VARIANT_LABELS = {
    "Normal": "Normal",
    "OverPermissioned": "Over-perm.",
    "HighTaskLowSecurity": "High-task\nlow-sec.",
    "EnvironmentBrittle": "Env.\nbrittle",
    "PolicyViolating": "Policy\nviolating",
}
METHOD_LABELS = {
    "the framework_full": "Proposed framework",
    "latest_score_only": "Latest score",
    "mean_historical": "Mean history",
    "ema_score": "EMA score",
    "beta_reputation": "Beta reputation",
    "elo_glicko": "Elo/Glicko-like",
    "human_only_review": "Human only",
    "llm_as_judge_only": "LLM judge only",
    "onchain_reputation_only": "On-chain only",
    "no_appeal": "No appeal",
    "no_verification": "No verification",
}
METHOD_COLORS = {
    "the framework_full": "#1F4E79",
    "llm_as_judge_only": "#D97917",
    "onchain_reputation_only": "#5B8E7D",
    "ema_score": "#7F8C8D",
    "no_verification": "#A23B72",
}

ensure_output_dirs()

responses = load_jsonl(RAW_DATA_FILES["responses"])
df_scored = build_scored_frame(responses)
df_baseline = pd.read_csv(RAW_DATA_FILES["baselines"])
df_human = pd.read_csv(RAW_DATA_FILES["human"])
df_external = pd.read_csv(RAW_DATA_FILES["external"])
df_appeal = pd.read_csv(RAW_DATA_FILES["appeals"])
df_stability = pd.read_csv(RAW_DATA_FILES["stability"])
results = {}
for path in RESULTS_DIR.glob("*.json"):
    with path.open() as handle:
        results[path.stem] = json.load(handle)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES_DIR / f"{stem}.png")
    fig.savefig(FIGURES_DIR / f"{stem}.pdf")
    plt.close(fig)



def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.10,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )



def annotate_bars(ax: plt.Axes, values: list[float] | np.ndarray, fmt: str = "{:.2f}", dy: float = 1.0) -> None:
    for patch, value in zip(ax.patches, values):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + dy,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=7,
        )



def wrapped(text: str, width: int) -> str:
    return textwrap.fill(text, width=width)



def draw_module_icon(ax: plt.Axes, cx: float, cy: float, color: str, kind: str, box_width: float) -> None:
    base_radius = 0.038
    p0 = ax.transData.transform((0.0, 0.0))
    px = ax.transData.transform((1.0, 0.0))
    py = ax.transData.transform((0.0, 1.0))
    x_scale = px[0] - p0[0]
    y_scale = py[1] - p0[1]
    radius = box_width * (x_scale / y_scale) / 6.0
    icon_scale = radius / base_radius

    def sx(dx: float) -> float:
        return cx + dx * icon_scale

    def sy(dy: float) -> float:
        return cy + dy * icon_scale

    def ss(val: float) -> float:
        return val * icon_scale

    ring = Ellipse(
        (cx, cy),
        box_width / 3.0,
        2 * radius,
        facecolor="white",
        edgecolor=color,
        linewidth=1.15,
        alpha=1.0,
        zorder=1.5,
    )
    ax.add_patch(ring)
    stroke = dict(color=color, linewidth=1.2, alpha=0.82)

    if kind == "intake":
        ax.add_patch(Rectangle((sx(-0.009), sy(-0.019)), ss(0.016), ss(0.035), fill=False, **stroke))
        ax.plot([sx(-0.005), sx(0.003)], [sy(0.008), sy(0.008)], **stroke)
        ax.plot([sx(-0.005), sx(0.001)], [sy(-0.003), sy(-0.003)], **stroke)
        ax.plot([sx(0.0), sx(0.0)], [sy(-0.002), sy(-0.025)], **stroke)
        ax.plot([sx(-0.0045), sx(0.0), sx(0.0045)], [sy(-0.020), sy(-0.025), sy(-0.020)], **stroke)
    elif kind == "audit":
        pts = [(sx(0.0), sy(0.021)), (sx(-0.011), sy(0.011)), (sx(-0.011), sy(-0.014)), (sx(0.0), sy(-0.024)), (sx(0.011), sy(-0.014)), (sx(0.011), sy(0.011))]
        for px_i, py_i in pts:
            ax.add_patch(Circle((px_i, py_i), ss(0.0028), facecolor=color, edgecolor=color, alpha=0.85))
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3), (1, 4), (2, 5)]:
            ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]], **stroke)
    elif kind == "calibration":
        ax.plot([sx(0.0), sx(0.0)], [sy(-0.024), sy(0.022)], **stroke)
        ax.plot([sx(-0.014), sx(0.014)], [sy(0.016), sy(0.016)], **stroke)
        ax.plot([sx(-0.009), sx(-0.002)], [sy(0.016), sy(-0.001)], **stroke)
        ax.plot([sx(0.009), sx(0.002)], [sy(0.016), sy(-0.001)], **stroke)
        ax.add_patch(Polygon([(sx(-0.012), sy(-0.002)), (sx(-0.002), sy(-0.002)), (sx(-0.007), sy(-0.018))], fill=False, **stroke))
        ax.add_patch(Polygon([(sx(0.002), sy(-0.002)), (sx(0.012), sy(-0.002)), (sx(0.007), sy(-0.018))], fill=False, **stroke))
    elif kind == "decision":
        ax.plot([sx(-0.012), sx(0.012)], [sy(0.0), sy(0.0)], **stroke)
        ax.plot([sx(0.0), sx(0.0)], [sy(-0.023), sy(0.023)], **stroke)
        ax.plot([sx(0.0035), sx(0.012), sx(0.0035)], [sy(0.012), sy(0.0), sy(-0.012)], **stroke)
        ax.plot([sx(-0.0035), sx(-0.012), sx(-0.0035)], [sy(0.012), sy(0.0), sy(-0.012)], **stroke)
    elif kind == "integrity":
        shield = Polygon([(sx(0.0), sy(0.023)), (sx(-0.012), sy(0.011)), (sx(-0.009), sy(-0.015)), (sx(0.0), sy(-0.026)), (sx(0.009), sy(-0.015)), (sx(0.012), sy(0.011))], fill=False, **stroke)
        ax.add_patch(shield)
        ax.plot([sx(-0.005), sx(-0.001), sx(0.006)], [sy(-0.003), sy(-0.013), sy(0.008)], **stroke)
    elif kind == "appeal":
        ax.add_patch(Rectangle((sx(-0.010), sy(-0.020)), ss(0.016), ss(0.035), fill=False, **stroke))
        ax.add_patch(Circle((sx(-0.002), sy(0.006)), ss(0.0035), fill=False, **stroke))
        ax.plot([sx(-0.007), sx(0.003)], [sy(-0.005), sy(-0.005)], **stroke)
        ax.plot([sx(-0.007), sx(0.000)], [sy(-0.012), sy(-0.012)], **stroke)
        ax.plot([sx(0.005), sx(0.012)], [sy(-0.016), sy(-0.006)], **stroke)
        ax.plot([sx(0.009), sx(0.013)], [sy(-0.020), sy(-0.014)], **stroke)
    elif kind == "state":
        ax.add_patch(Arc((cx, sy(0.010)), ss(0.022), ss(0.014), theta1=0, theta2=180, **stroke))
        ax.add_patch(Arc((cx, sy(-0.005)), ss(0.022), ss(0.014), theta1=180, theta2=360, **stroke))
        ax.add_patch(Rectangle((sx(-0.011), sy(-0.013)), ss(0.022), ss(0.022), fill=False, **stroke))
        ax.add_patch(Arc((sx(0.009), sy(-0.014)), ss(0.015), ss(0.015), theta1=215, theta2=30, **stroke))
        ax.plot([sx(0.005), sx(0.008), sx(0.012)], [sy(-0.016), sy(-0.021), sy(-0.010)], **stroke)



def draw_box(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, body: str, color: str, icon: str) -> None:
    fill = tuple(1 - (1 - channel) * 0.12 for channel in matplotlib.colors.to_rgb(color))
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=1.05,
        edgecolor=color,
        facecolor=fill,
        zorder=1,
    )
    ax.add_patch(patch)

    icon_x = x + w * 0.21
    text_x = x + w * 0.40
    draw_module_icon(ax, icon_x, y + h * 0.56, color, icon, w)
    title_y = y + h - 0.038
    body_y = title_y - 0.086
    ax.text(text_x, title_y, title, fontsize=8.5, fontweight="bold", color="#243447", ha="left", va="top", linespacing=1.0)
    ax.text(text_x, body_y, body, fontsize=6.3, color="#314252", ha="left", va="top", linespacing=1.08)



def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], rad: float = 0.0, style: str = "solid") -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.15,
        color="#506070",
        linestyle="--" if style == "dashed" else "-",
        connectionstyle=f"arc3,rad={rad}",
        zorder=2,
    )
    ax.add_patch(arrow)



def create_mechanism_figure() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    top_y = 0.64
    top_w = 0.205
    top_h = 0.20
    top_x = [0.04, 0.265, 0.49, 0.715]
    bottom_y = 0.28
    bottom_w = 0.21
    bottom_h = 0.20
    bottom_x = [0.2625, 0.4875, 0.7125]

    boxes = [
        (top_x[0], top_y, top_w, top_h, "Evidence\nIntake", "Ingest, validate, and\nstandardize evidence.", "#4E7DA7", "intake"),
        (top_x[1], top_y, top_w, top_h, "Six-Dimensional\nAudit", "Evaluate six dimensions of\nquality, completeness, and risk.", "#4E7DA7", "audit"),
        (top_x[2], top_y, top_w, top_h, "Calibration", "Align scores with\nbenchmarks, priors,\nand cross-case signals.", "#C39A4A", "calibration"),
        (top_x[3], top_y, top_w, top_h, "Deployment\nDecision", "Set readiness and\ngovernance actions.", "#C39A4A", "decision"),
        (bottom_x[0], bottom_y, bottom_w, bottom_h, "Integrity\nVerification", "Check tampering,\nconsistency, and\nprovenance.", "#7B6D9C", "integrity"),
        (bottom_x[1], bottom_y, bottom_w, bottom_h, "Appeal &\nCorrection", "Review new evidence\nand correct records.", "#7B6D9C", "appeal"),
        (bottom_x[2], bottom_y, bottom_w, bottom_h, "Updated\nGovernance State", "Update reproducible,\nauditable records.", "#7F8C8D", "state"),
    ]
    for box in boxes:
        draw_box(ax, *box)

    top_centers = [x + top_w / 2 for x in top_x]
    bottom_centers = [x + bottom_w / 2 for x in bottom_x]
    top_arrow_y = top_y + top_h * 0.56
    bottom_link_y = bottom_y + bottom_h * 0.50

    draw_arrow(ax, (top_x[0] + top_w, top_arrow_y), (top_x[1], top_arrow_y))
    draw_arrow(ax, (top_x[1] + top_w, top_arrow_y), (top_x[2], top_arrow_y))
    draw_arrow(ax, (top_x[2] + top_w, top_arrow_y), (top_x[3], top_arrow_y))
    draw_arrow(ax, (top_centers[1], top_y), (top_centers[1], bottom_y + bottom_h))
    draw_arrow(ax, (top_centers[2], top_y), (top_centers[2], bottom_y + bottom_h))
    draw_arrow(ax, (top_centers[3], top_y), (top_centers[3], bottom_y + bottom_h))
    draw_arrow(ax, (bottom_x[0] + bottom_w, bottom_link_y), (bottom_x[1], bottom_link_y))
    draw_arrow(ax, (bottom_x[1] + bottom_w, bottom_link_y), (bottom_x[2], bottom_link_y))
    draw_arrow(ax, (bottom_centers[2], bottom_y), (bottom_centers[2], 0.11), style="dashed")
    draw_arrow(ax, (bottom_centers[2], 0.11), (0.08, 0.11), style="dashed")
    draw_arrow(ax, (0.08, 0.11), (0.08, top_y), style="dashed")
    draw_arrow(ax, (bottom_centers[0], bottom_y), (bottom_centers[0], 0.13))
    draw_arrow(ax, (bottom_centers[1], bottom_y), (bottom_centers[1], 0.13))
    draw_arrow(ax, (bottom_centers[1], 0.13), (bottom_centers[0], 0.13))

    ax.text(0.5, 0.93, "An evidence-first governance workflow", ha="center", va="center", fontsize=12, fontweight="bold", color="#243447")
    ax.text(0.5, 0.075, "Continuous feedback loop for ongoing improvement", ha="center", va="center", fontsize=8, color="#5B6573", style="italic")

    save_figure(fig, "fig1_mechanism_workflow")



def create_audit_validity_figure() -> None:
    exp04 = results["04_six_dimension_effectiveness"]
    exp05 = results["05_human_gold_agreement"]
    exp06 = results["06_external_agent_generalization"]

    fig = plt.figure(figsize=(11.2, 8.4))
    gs = fig.add_gridspec(2, 2, wspace=0.26, hspace=0.34)

    ax1 = fig.add_subplot(gs[0, 0])
    scores_by_variant = [
        df_baseline[df_baseline["agent_variant"] == variant]["the framework_full_score"].values
        for variant in VARIANT_ORDER
    ]
    bp = ax1.boxplot(
        scores_by_variant,
        tick_labels=[VARIANT_LABELS[v] for v in VARIANT_ORDER],
        patch_artist=True,
        widths=0.62,
        showfliers=False,
    )
    for patch, variant in zip(bp["boxes"], VARIANT_ORDER):
        patch.set_facecolor(VARIANT_COLORS[variant])
        patch.set_alpha(0.72)
    ax1.axhline(60, color="#6B7280", linestyle="--", linewidth=1, label="Deployment threshold")
    ax1.set_ylabel("Calibrated score")
    ax1.set_ylim(0, 100)
    ax1.set_title("Variant-level score separation")
    ax1.legend(frameon=False, loc="lower left")
    add_panel_label(ax1, "A")

    ax2 = fig.add_subplot(gs[0, 1])
    dimension_cols = [f"score_{dimension.lower().replace(' ', '_')}" for dimension in DIMENSION_ORDER]
    heatmap = np.array([
        [
            df_scored[df_scored["agent_variant"] == variant][column].mean()
            for column in dimension_cols
        ]
        for variant in VARIANT_ORDER
    ])
    im = ax2.imshow(heatmap, aspect="auto", cmap="viridis", vmin=20, vmax=95)
    ax2.set_xticks(range(len(DIMENSION_ORDER)))
    ax2.set_xticklabels([
        "Security",
        "Task",
        "Reasoning",
        "Resilience",
        "Efficiency",
        "Policy",
    ], rotation=25, ha="right")
    ax2.set_yticks(range(len(VARIANT_ORDER)))
    ax2.set_yticklabels([VARIANT_LABELS[v] for v in VARIANT_ORDER])
    ax2.set_title("Six-dimensional profile by variant")
    for row in range(heatmap.shape[0]):
        for col in range(heatmap.shape[1]):
            ax2.text(col, row, f"{heatmap[row, col]:.0f}", ha="center", va="center", fontsize=6.8, color="white")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Mean score")
    add_panel_label(ax2, "B")

    ax3 = fig.add_subplot(gs[1, 0])
    human_metric_names = ["Pairwise\nagreement", "Full\nunanimity", "Cohen's\nkappa", "System-human\nagreement"]
    human_metric_values = [
        exp05["inter_annotator_agreement_binary"],
        exp05["full_unanimity_rate"],
        exp05["cohens_kappa"],
        exp05["system_human_agreement"],
    ]
    human_colors = ["#2A7F62", "#4C78A8", "#D97917", "#C83E4D"]
    ax3.bar(human_metric_names, human_metric_values, color=human_colors, width=0.72)
    ax3.set_ylim(0, 1.0)
    ax3.set_ylabel("Agreement / correlation")
    ax3.set_title("Human agreement exceeds raw system-human agreement")
    for idx, value in enumerate(human_metric_values):
        ax3.text(idx, value + 0.03, f"{value:.3f}", ha="center", va="bottom", fontsize=7)
    add_panel_label(ax3, "C")

    ax4 = fig.add_subplot(gs[1, 1])
    agent_order = ["gpt-4-turbo", "claude-3-opus", "gemini-1.5-pro"]
    x = np.arange(len(agent_order))
    width = 0.27
    raw_vals = [exp06["agent_agreements"][agent]["raw_binary_agreement"] for agent in agent_order]
    transfer_vals = [exp06["agent_agreements"][agent]["threshold_transferred_agreement"] for agent in agent_order]
    corr_vals = [exp06["agent_agreements"][agent]["score_correlation"] for agent in agent_order]
    bars1 = ax4.bar(x - width / 2, raw_vals, width, color="#B8C1CC", label="Raw agreement")
    bars2 = ax4.bar(x + width / 2, transfer_vals, width, color="#D97917", label="Threshold-transferred agreement")
    ax4.set_xticks(x)
    ax4.set_xticklabels(["GPT-4\nTurbo", "Claude 3\nOpus", "Gemini\n1.5 Pro"])
    ax4.set_ylim(0, 1.0)
    ax4.set_ylabel("Agreement")
    ax4.set_title("Transfer improves after threshold alignment")
    ax4b = ax4.twinx()
    ax4b.plot(x, corr_vals, color="#1F4E79", marker="o", linewidth=1.8, label="Score correlation")
    ax4b.set_ylim(0, 1.0)
    ax4b.set_ylabel("Correlation")
    handles = [bars1, bars2, Line2D([0], [0], color="#1F4E79", marker="o", linewidth=1.8, label="Score correlation")]
    labels = ["Raw agreement", "Threshold-transferred agreement", "Score correlation"]
    ax4.legend(handles, labels, frameon=False, loc="upper left")
    add_panel_label(ax4, "D")

    save_figure(fig, "fig2_audit_validity_composite")



def create_prediction_figure() -> None:
    exp07 = results["07_prospective_temporal_decision"]
    exp08 = results["08_strong_baseline_comparison"]
    best_baseline_key = exp07["best_baseline_name"]

    fig = plt.figure(figsize=(11.2, 8.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.02, 0.98], height_ratios=[1.0, 1.05], wspace=0.30, hspace=0.42)

    ax1 = fig.add_subplot(gs[0, 0])
    y_true = df_baseline["ground_truth_next_episode_fail"].values
    baseline_series = {
        "the framework_full": df_baseline["the framework_full_score"],
        "latest_score_only": df_baseline["baseline_latest_score_only"],
        "mean_historical": df_baseline["baseline_mean_historical"],
        "ema_score": df_baseline["baseline_ema_score"],
        "beta_reputation": df_baseline["baseline_beta_reputation"],
        "elo_glicko": df_baseline["baseline_elo_glicko"],
        "human_only_review": df_baseline["baseline_human_only_review"],
        "llm_as_judge_only": df_baseline["baseline_llm_as_judge_only"],
        "onchain_reputation_only": df_baseline["baseline_onchain_reputation_only"],
        "no_appeal": df_baseline["baseline_no_appeal"],
        "no_verification": df_baseline["baseline_no_verification"],
    }
    method_order = ["the framework_full", best_baseline_key, "onchain_reputation_only", "ema_score", "no_verification"]
    methods = [(method_key, baseline_series[method_key]) for method_key in dict.fromkeys(method_order)]
    for method_key, scores in methods:
        fpr, tpr, _ = roc_curve(y_true, 100 - scores.values)
        auc_val = roc_auc_score(y_true, 100 - scores.values)
        color = "#D97917" if method_key == best_baseline_key else METHOD_COLORS.get(method_key, "#9AA5B1")
        linewidth = 2.8 if method_key == "the framework_full" else 1.5
        alpha = 1.0 if method_key in {"the framework_full", best_baseline_key} else 0.8
        linestyle = "-" if method_key in {"the framework_full", best_baseline_key} else "--"
        ax1.plot(
            fpr,
            tpr,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            label=f"{METHOD_LABELS[method_key]} ({auc_val:.3f})",
        )
    ax1.plot([0, 1], [0, 1], color="#94A3B8", linestyle=":", linewidth=1)
    ax1.set_xlabel("False-positive rate")
    ax1.set_ylabel("True-positive rate")
    ax1.set_title("Prospective ROC")
    ax1.legend(frameon=False, loc="lower right")
    add_panel_label(ax1, "A")

    ax2 = fig.add_subplot(gs[0, 1])
    domain_rows = []
    for domain, payload in exp07["per_domain_auc"].items():
        fw_val = payload["the framework_full"]
        baseline_val = payload[best_baseline_key]
        delta = fw_val - baseline_val
        domain_rows.append((domain, fw_val, baseline_val, delta))
    domain_rows.sort(key=lambda row: row[3], reverse=True)
    domains = [row[0] for row in domain_rows]
    fw_vals = [row[1] for row in domain_rows]
    baseline_vals = [row[2] for row in domain_rows]
    deltas = [row[3] for row in domain_rows]
    y = np.arange(len(domains))
    for idx, (fw_val, baseline_val) in enumerate(zip(fw_vals, baseline_vals)):
        ax2.plot([baseline_val, fw_val], [idx, idx], color="#CBD5E1", linewidth=2)
    ax2.scatter(baseline_vals, y, color="#D97917", s=34, label=METHOD_LABELS[best_baseline_key], zorder=3)
    ax2.scatter(fw_vals, y, color="#1F4E79", s=34, label="Proposed framework", zorder=3)
    for idx, delta in enumerate(deltas):
        anchor = max(fw_vals[idx], baseline_vals[idx])
        ax2.text(anchor + 0.006, idx, f"{delta:+.03f}", va="center", fontsize=6.8, color="#4B5563")
    ax2.set_yticks(y)
    ax2.set_yticklabels([wrapped(domain, 20) for domain in domains])
    ax2.set_xlim(0.68, 0.92)
    ax2.set_xlabel("Per-domain AUC")
    ax2.set_title("Domain heterogeneity")
    ax2.legend(frameon=False, loc="lower left")
    add_panel_label(ax2, "B")

    ax3 = fig.add_subplot(gs[1, :])
    ranking = sorted(
        ((name, payload["auc"]) for name, payload in exp08["calibration_sensitivity"]["weight_configs"].items()),
        key=lambda item: item[1],
        reverse=True,
    )
    config_names = [name.replace("_", " ") for name, _ in ranking]
    config_values = [value for _, value in ranking]
    config_colors = ["#D97917" if name == "default" else "#C7CDD4" for name, _ in ranking]
    y_pos = np.arange(len(config_names))
    ax3.barh(y_pos, config_values, color=config_colors, height=0.72)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(config_names)
    ax3.set_ylim(len(config_names) + 0.9, -0.5)
    ax3.set_xlim(0.73, 0.905)
    ax3.set_xlabel("AUC")
    ax3.set_title("Calibration sensitivity")
    for y_idx, value in zip(y_pos, config_values):
        label_x = min(value + 0.002, 0.899)
        ax3.text(label_x, y_idx, f"{value:.3f}", va="center", ha="left", fontsize=7)
    threshold60 = exp08["threshold_sweep"]["60"]
    threshold65 = exp08["threshold_sweep"]["65"]
    ax3.text(
        0.899,
        len(config_names) + 0.45,
        "Nested supplement: temporal-heavy selected in 5/5 folds\nThreshold 60: F1 {:.3f}; threshold 65: best F1 {:.3f}".format(
            threshold60["f1_score"],
            threshold65["f1_score"],
        ),
        fontsize=6.9,
        ha="right",
        va="center",
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#D0D7DE"},
    )
    add_panel_label(ax3, "C")

    save_figure(fig, "fig3_prediction_composite")



def create_integrity_governance_figure() -> None:
    exp09 = results["09_evidence_tampering_detection"]
    exp10 = results["10_tee_integrity_replay"]
    exp11 = results["11_zk_proof_integrity"]
    exp12 = results["12_full_chain_ablation"]
    exp13 = results["13_appeal_correction_effectiveness"]

    fig = plt.figure(figsize=(11.2, 8.4))
    gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    indicator_counts = exp09["tamper_indicator_breakdown"]
    indicator_labels = [
        "Forbidden action attempted",
        "Out-of-scope tool usage",
        "Forbidden action mentioned",
    ]
    indicator_values = [
        indicator_counts["forbidden_action_attempted"],
        indicator_counts["out_of_scope_tool_usage"],
        indicator_counts["forbidden_action_mentioned"],
    ]
    y_pos = np.arange(len(indicator_labels))
    ax1.barh(y_pos, indicator_values, color=["#C83E4D", "#D97917", "#7A6FD6"], height=0.58)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([wrapped(label, 18) for label in indicator_labels])
    ax1.set_xlabel("Observed cases")
    ax1.set_title("Tamper indicators")
    for idx, value in enumerate(indicator_values):
        ax1.text(value + 2.0, idx, f"{value:.0f}", va="center", fontsize=7)
    ax1.text(
        0.03,
        0.92,
        f"Multi-indicator cases: {exp09['multi_indicator_cases']}",
        transform=ax1.transAxes,
        fontsize=7.2,
    )
    add_panel_label(ax1, "A")

    ax2 = fig.add_subplot(gs[0, 1])
    latency_labels = ["TEE mean", "TEE p95", "Proof mean", "Proof p95"]
    latency_values = [
        exp10["mean_tee_latency_ms"],
        exp10["p95_tee_latency_ms"],
        exp11["mean_proof_gen_ms"],
        exp11["p95_proof_gen_ms"],
    ]
    latency_colors = ["#5B8E7D", "#8FB9A8", "#1F4E79", "#6C95BD"]
    ax2.bar(latency_labels, latency_values, color=latency_colors, width=0.65)
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Verification overhead")
    annotate_bars(ax2, latency_values, fmt="{:.0f}", dy=35.0)
    ax2.text(
        0.03,
        0.90,
        f"Replay blocked: {exp10['replay_attacks_blocked']}/{exp10['replay_attacks_tested']}\nProof verification: {exp11['verification_time_ms']} ms",
        transform=ax2.transAxes,
        fontsize=7.0,
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "#D0D7DE"},
    )
    add_panel_label(ax2, "B")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")
    ax3.set_title("Appeal taxonomy")

    left_nodes = [
        (0.14, 0.80, "False penalty", "Accepted", "#2A7F62"),
        (0.14, 0.58, "Borderline error", "Partial", "#D97917"),
        (0.14, 0.36, "Malicious appeal", "Rejected", "#C83E4D"),
        (0.14, 0.15, "Insufficient evidence", "Rejected", "#C83E4D"),
    ]
    right_nodes = {
        "Accepted": (0.78, 0.80),
        "Partial": (0.78, 0.58),
        "Rejected": (0.78, 0.25),
    }

    for label, (x, y) in right_nodes.items():
        box = FancyBboxPatch(
            (x - 0.11, y - 0.055),
            0.22,
            0.11,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.0,
            edgecolor="#CBD5E1",
            facecolor="#F8FAFC",
        )
        ax3.add_patch(box)
        ax3.text(x, y, label, ha="center", va="center", fontsize=7.5, color="#334155", fontweight="bold")

    for x, y, label, outcome, color in left_nodes:
        box = FancyBboxPatch(
            (x - 0.13, y - 0.055),
            0.26,
            0.11,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=0,
            facecolor=color,
            alpha=0.95,
        )
        ax3.add_patch(box)
        ax3.text(x, y, wrapped(label, 18), ha="center", va="center", fontsize=7.4, color="white", fontweight="bold")

        target_x, target_y = right_nodes[outcome]
        start = (x + 0.12, y)
        end = (target_x - 0.12, target_y)
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.2,
            color="#94A3B8",
            connectionstyle="arc3,rad=0.0",
        )
        ax3.add_patch(arrow)
        ax3.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.035, "50/50", fontsize=6.8, color="#64748B", ha="center")

    ax3.text(0.03, 0.91, "Deterministic packaged mapping", transform=ax3.transAxes, fontsize=7.0, color="#4B5563")
    add_panel_label(ax3, "C")

    ax4 = fig.add_subplot(gs[1, 1])
    ablation_labels = ["No appeal", "Full system", "No verification"]
    ablation_values = [exp12["no_appeal_mean"], exp12["full_system_mean"], exp12["no_verification_mean"]]
    ax4.bar(ablation_labels, ablation_values, color=["#D97917", "#1F4E79", "#A23B72"], width=0.64)
    ax4.set_ylim(35, 82)
    ax4.set_ylabel("Mean calibrated score")
    ax4.set_title("Governance ablations")
    annotate_bars(ax4, ablation_values, fmt="{:.2f}", dy=0.8)
    ax4.text(
        0.04,
        0.90,
        f"Appeal contribution +{exp12['appeal_contribution']:.2f}\nNo-verification uplift on policy-violating cases +{exp12['verification_helps_detect_pv']:.2f}",
        transform=ax4.transAxes,
        fontsize=7.0,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#D0D7DE"},
    )
    add_panel_label(ax4, "D")

    save_figure(fig, "fig4_integrity_governance_composite")



def prettify_rationale(label: str) -> str:
    return wrapped(label.replace("; ", " + ").replace("_", " "), 26)



def create_operations_figure() -> None:
    exp17 = results["17_failure_case_typology"]
    exp18 = results["18_cost_scalability_stability"]

    fig = plt.figure(figsize=(11.2, 7.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], wspace=0.30, hspace=0.36)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df_stability["elapsed_hours"], df_stability["throughput_audits_per_minute"], color="#1F4E79", linewidth=1.8)
    ax1.axhline(exp18["mean_throughput_per_min"], color="#94A3B8", linestyle="--", linewidth=1)
    ax1.set_xlabel("Elapsed hours")
    ax1.set_ylabel("Audits per minute")
    ax1.set_title("Throughput remains stable over long-run execution")
    ax1.text(0.03, 0.90, f"Mean = {exp18['mean_throughput_per_min']:.3f} audits/min", transform=ax1.transAxes, fontsize=7.2)
    add_panel_label(ax1, "A")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df_stability["elapsed_hours"], df_stability["error_rate_pct"], color="#C83E4D", linewidth=1.4)
    spike_mask = df_stability["error_rate_pct"] > 0
    ax2.scatter(df_stability.loc[spike_mask, "elapsed_hours"], df_stability.loc[spike_mask, "error_rate_pct"], color="#C83E4D", s=18)
    ax2.set_xlabel("Elapsed hours")
    ax2.set_ylabel("Error rate (%)")
    ax2.set_title("Rare error bursts remain visible")
    ax2.text(0.03, 0.90, f"Mean = {exp18['mean_error_rate_pct']:.3f}%\nMax window = {exp18['max_error_rate_pct']:.2f}%", transform=ax2.transAxes, fontsize=7.2)
    add_panel_label(ax2, "B")

    ax3 = fig.add_subplot(gs[1, :])
    rationale_items = list(exp17["top_failure_rationale_share"].items())[:5]
    rationale_labels = [prettify_rationale(name) for name, _ in rationale_items][::-1]
    rationale_values = [value * 100 for _, value in rationale_items][::-1]
    ax3.barh(rationale_labels, rationale_values, color="#5D7285")
    ax3.set_xlabel("Failure share (%)")
    ax3.set_title("Failure rationales remain interpretable")
    for patch, value in zip(ax3.patches, rationale_values):
        ax3.text(value + 0.8, patch.get_y() + patch.get_height() / 2, f"{value:.1f}", va="center", fontsize=7)
    add_panel_label(ax3, "C")

    save_figure(fig, "fig5_operations_composite")



def main() -> None:
    create_mechanism_figure()
    create_audit_validity_figure()
    create_prediction_figure()
    create_integrity_governance_figure()
    create_operations_figure()
    print(f"Saved refreshed figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
