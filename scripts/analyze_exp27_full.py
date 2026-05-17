"""
Exp27 analysis and figure generation.
Reads human-system divergence data and produces the 6-panel taxonomy figure.
"""
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

csv_path    = '../experiments/07_revision_bundle/external_validation/data/human_system_divergence_cases.csv'
json_path   = '../experiments/07_revision_bundle/external_validation/results/27_human_system_divergence_taxonomy.json'
scores_csv  = '../experiments/07_revision_bundle/external_validation/results/27_human_system_divergence_taxonomy_scores.csv'

df = pd.read_csv(csv_path)
with open(json_path) as f:
    tax = json.load(f)

n_total      = tax['n_total_units']
n_div        = tax['n_divergence_cases']
div_rate     = tax['divergence_rate']
mismatch     = tax['mismatch_direction']
tax_counts   = tax['taxonomy_counts']
mean_gap_div = tax['mean_abs_score_gap_divergence']
mean_gap_mat = tax['mean_abs_score_gap_matched']
mean_ext_div = tax['mean_external_pass_rate_divergence']
mean_ext_mat = tax['mean_external_pass_rate_matched']
by_domain    = tax['by_domain']
by_taxonomy  = tax['by_taxonomy']

# Identify relevant columns
div_col        = [c for c in df.columns if 'divergence_taxonomy' in c or 'taxonomy' in c]
dir_col        = [c for c in df.columns if 'direction' in c or 'mismatch' in c]
domain_col     = [c for c in df.columns if 'domain' in c]
dim_col        = [c for c in df.columns if 'dimension' in c]
gap_col        = [c for c in df.columns if 'gap' in c.lower() or 'score_gap' in c]
sys_label_col  = [c for c in df.columns if 'system_label' in c]
human_label_col= [c for c in df.columns if 'consensus_pass' in c or 'human_label' in c]
ext_col        = [c for c in df.columns if 'external_pass' in c]

COLORS = {
    'primary':   '#1B4F72',
    'secondary': '#E67E22',
    'accent1':   '#C0392B',
    'accent2':   '#27AE60',
    'accent3':   '#8E44AD',
    'light_bg':  '#F8FAFC',
    'grid':      '#E8EDF2',
    'text':      '#2C3E50',
}

plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         9,
    'axes.titlesize':    10,
    'axes.labelsize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.4,
    'grid.color':        COLORS['grid'],
    'figure.facecolor':  'white',
    'axes.facecolor':    COLORS['light_bg'],
})

fig = plt.figure(figsize=(16, 12))
gs  = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.42,
               top=0.93, bottom=0.08, left=0.07, right=0.97)

# Panel A: mismatch direction
ax_a = fig.add_subplot(gs[0, 0])
directions = {'Sys-Fail\nHuman-Pass': 292, 'Sys-Pass\nHuman-Fail': 3}
bars = ax_a.bar(directions.keys(), directions.values(),
                color=[COLORS['accent1'], COLORS['accent2']],
                width=0.5, edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, directions.values()):
    ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
              str(val), ha='center', va='bottom', fontweight='bold', fontsize=10)
ax_a.set_ylabel('Number of Cases')
ax_a.set_title('A. Mismatch Direction\n(n=295 divergence cases)', fontweight='bold', pad=6)
ax_a.set_ylim(0, 330)

# Panel B: taxonomy distribution
ax_b = fig.add_subplot(gs[0, 1])
tax_labels = {
    'environment_or_execution_uncertainty': 'Env/Exec\nUncertainty',
    'threshold_or_uncertainty_mismatch':    'Threshold\nMismatch',
    'policy_over_penalization':             'Policy\nOver-Penalty',
    'generic_system_over_penalization':     'Generic\nOver-Penalty',
    'human_flagged_risk_not_captured':      'Human-Flagged\nRisk (System Miss)',
}
mapped = {}
for k, v in tax_counts.items():
    label = tax_labels.get(k, k.replace('_', '\n'))
    mapped[label] = v
sorted_items = sorted(mapped.items(), key=lambda x: x[1], reverse=True)
labels_b     = [x[0] for x in sorted_items]
vals_b       = [x[1] for x in sorted_items]
bar_colors   = [COLORS['accent1'], COLORS['secondary'], COLORS['primary'],
                COLORS['accent3'], COLORS['accent2']][:len(labels_b)]
bars_b = ax_b.barh(range(len(labels_b)), vals_b, color=bar_colors,
                   edgecolor='white', linewidth=0.8)
ax_b.set_yticks(range(len(labels_b)))
ax_b.set_yticklabels(labels_b, fontsize=7.5)
ax_b.set_xlabel('Count')
ax_b.set_title('B. Divergence Taxonomy\n(System-Fail / Human-Pass)', fontweight='bold', pad=6)
for bar, val in zip(bars_b, vals_b):
    ax_b.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
              str(val), va='center', fontsize=8, fontweight='bold')
ax_b.set_xlim(0, max(vals_b) * 1.18)

# Panel C: external pass rate comparison
ax_c = fig.add_subplot(gs[0, 2])
categories = ['Matched\n(Agreed)', 'Divergence\n(Sys-Fail\nHuman-Pass)']
ext_rates  = [mean_ext_mat, mean_ext_div]
bar_c = ax_c.bar(categories, ext_rates,
                 color=[COLORS['accent2'], COLORS['accent1']],
                 width=0.5, edgecolor='white', linewidth=0.8)
for bar, val in zip(bar_c, ext_rates):
    ax_c.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
              f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
ax_c.set_ylabel('Mean External Pass Rate')
ax_c.set_title('C. External Evaluator Validation\n(Independent 3rd-party)', fontweight='bold', pad=6)
ax_c.set_ylim(0, 1.1)
ax_c.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
ax_c.annotate('External also\nflag as risky',
              xy=(1, mean_ext_div), xytext=(0.55, 0.28),
              fontsize=8, color=COLORS['accent1'],
              arrowprops=dict(arrowstyle='->', color=COLORS['accent1'], lw=1.2))

# Panel D: domain distribution
ax_d = fig.add_subplot(gs[1, 0])
domain_data = {}
if domain_col:
    dc = domain_col[0]
    tc = div_col[0] if div_col else None
    if tc:
        domain_counts = df.groupby(dc).size()
        domain_data   = domain_counts.to_dict()
if domain_data:
    dom_labels = [k.replace(' ', '\n') for k in domain_data.keys()]
    dom_vals   = list(domain_data.values())
    dom_colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent1'],
                  COLORS['accent2'], COLORS['accent3']][:len(dom_labels)]
    bars_d = ax_d.bar(range(len(dom_labels)), dom_vals, color=dom_colors,
                      edgecolor='white', linewidth=0.8)
    ax_d.set_xticks(range(len(dom_labels)))
    ax_d.set_xticklabels(dom_labels, fontsize=7.5)
    ax_d.set_ylabel('Divergence Cases')
    ax_d.set_title('D. Divergence by Domain', fontweight='bold', pad=6)
    for bar, val in zip(bars_d, dom_vals):
        ax_d.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                  str(val), ha='center', va='bottom', fontsize=8, fontweight='bold')

# Panel E: dimension distribution
ax_e = fig.add_subplot(gs[1, 1])
if dim_col:
    dimc = dim_col[0]
    dim_counts = df.groupby(dimc).size().sort_values(ascending=True)
    dim_labels_short = {
        'Security Robustness':    'Sec. Robustness',
        'Task Success':           'Task Success',
        'Reasoning Quality':      'Reasoning Quality',
        'Environmental Resilience':'Env. Resilience',
        'Operational Efficiency': 'Operational Eff.',
        'Policy Compliance':      'Policy Compliance',
    }
    dim_labels_plot = [dim_labels_short.get(k, k) for k in dim_counts.index]
    dim_colors_e    = [COLORS['primary'], COLORS['secondary'], COLORS['accent1'],
                       COLORS['accent2'], COLORS['accent3'], COLORS['primary']][:len(dim_labels_plot)]
    bars_e = ax_e.barh(range(len(dim_labels_plot)), dim_counts.values,
                       color=dim_colors_e, edgecolor='white', linewidth=0.8, height=0.6)
    ax_e.set_yticks(range(len(dim_labels_plot)))
    ax_e.set_yticklabels(dim_labels_plot, fontsize=8.5)
    ax_e.set_xlabel('Divergence Cases', fontsize=9)
    ax_e.set_title('E. Divergence by Dimension', fontweight='bold', pad=8)
    for bar, val in zip(bars_e, dim_counts.values):
        ax_e.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                  str(val), va='center', fontsize=9, fontweight='bold')
    ax_e.set_xlim(0, max(dim_counts.values) * 1.22)

# Panel F: score gap distribution
ax_f = fig.add_subplot(gs[1, 2])
if gap_col:
    gc   = gap_col[0]
    gaps = df[gc].dropna()
    ax_f.hist(gaps, bins=25, color=COLORS['accent1'], edgecolor='white',
              linewidth=0.6, alpha=0.85)
    ax_f.axvline(x=gaps.mean(), color=COLORS['primary'], linestyle='--',
                 linewidth=1.5, label=f'Mean={gaps.mean():.1f}')
    ax_f.axvline(x=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax_f.set_xlabel('Score Gap (System − Human Consensus)')
    ax_f.set_ylabel('Frequency')
    ax_f.set_title('F. Score Gap Distribution\n(Divergence Cases)', fontweight='bold', pad=6)
    ax_f.legend(fontsize=8)

fig.suptitle('Human-System Divergence Analysis (Exp27, n=295 cases)',
             fontsize=12, fontweight='bold', y=1.01, color=COLORS['text'])

out_path = '../figures/fig_human_system_divergence.pdf'
out_png  = '../figures/fig_human_system_divergence.png'
plt.savefig(out_path, bbox_inches='tight', dpi=300)
plt.savefig(out_png,  bbox_inches='tight', dpi=200)
print(f"Saved: {out_path}")
print(f"Saved: {out_png}")
