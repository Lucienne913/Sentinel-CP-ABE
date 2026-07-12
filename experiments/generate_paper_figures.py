#!/usr/bin/env python3
"""Generate SCI Q1 compliant paper figures from experiment results.
   IEEE IoT Journal format: STIX fonts, professional color scheme,
   colors with distinct brightness for B&W compatibility, error bars."""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ============================================================
# Configuration
# ============================================================
FIGS_DIR = '/app/paper/figures'
os.makedirs(FIGS_DIR, exist_ok=True)

# STIXGeneral ~= Times New Roman, guaranteed available in Docker
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIXGeneral', 'DejaVu Serif', 'serif'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'lines.linewidth': 1.0,
    'lines.markersize': 3.5,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3.0,
    'ytick.major.size': 3.0,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
})

# Professional IEEE/MATLAB-style color palette
# Blue, Orange, Yellow, Purple, Green, Cyan, Red
COLORS = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E',
          '#77AC30', '#4DBEEE', '#A2142F']
# Colors are chosen with distinct brightness for grayscale compatibility


def load_json(rel_path):
    full = os.path.join('/app/experiments/results', rel_path)
    with open(full) as f:
        return json.load(f)


# ============================================================
# FIG 1: Hash Chain Verification Latency  (single column)
# Save as: fig04_hash_chain_opt.pdf
# ============================================================
def fig_hash_chain_optimization():
    hc = load_json('ablation/hash_chain_performance.json')
    chain = hc['chains'][1]  # 8760-chain

    positions = [0, 875, 2189, 4379, 6569, 7883, 8671]
    orig_lat = np.array([float(chain['orig'][f'pos_{p}']['avg_latency_us']) for p in positions])
    opt_lat  = np.array([float(chain['opt'][f'pos_{p}']['avg_latency_us']) for p in positions])
    orig_err = np.array([max(float(chain['orig'][f'pos_{p}']['p95_latency_us'])
                         - float(chain['orig'][f'pos_{p}']['avg_latency_us']), 0.0) for p in positions])
    opt_err  = np.array([max(float(chain['opt'][f'pos_{p}']['p95_latency_us'])
                         - float(chain['opt'][f'pos_{p}']['avg_latency_us']), 0.0) for p in positions])

    orig_lat /= 1000.0; opt_lat /= 1000.0
    orig_err /= 1000.0; opt_err /= 1000.0

    x_labels = ['Start\n(pos 0)', '875', '2189', 'Mid\n(pos 4379)',
                '6569', '7883', 'End\n(pos 8671)']

    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    x = np.arange(len(positions))
    w = 0.30

    # Color: Orange for Original, Blue for Optimized
    bars1 = ax.bar(x - w/2, orig_lat, w, label='Original O(N)',
                   color='#D95319', edgecolor='black', linewidth=0.6)
    bars2 = ax.bar(x + w/2, opt_lat, w, label='Optimized O(1)',
                   color='#0072BD', edgecolor='black', linewidth=0.6)

    ax.errorbar(x - w/2, orig_lat, yerr=orig_err, fmt='none',
                ecolor='#D95319', capsize=1.8, capthick=0.5, elinewidth=0.5)
    ax.errorbar(x + w/2, opt_lat, yerr=opt_err, fmt='none',
                ecolor='#0072BD', capsize=1.8, capthick=0.5, elinewidth=0.5)

    ax.set_ylabel('Latency (ms)', fontsize=8)
    ax.set_xlabel('Verification Position', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=6.5)
    ax.set_yscale('log')
    ax.set_ylim(bottom=0.0001, top=1e4)
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='black',
              fontsize=6.5, handlelength=2.5)
    ax.grid(axis='y', alpha=0.25, linestyle=':', linewidth=0.4)
    ax.minorticks_on()
    ax.grid(axis='y', which='minor', alpha=0.08, linestyle=':', linewidth=0.3)

    plt.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIGS_DIR, 'fig04_hash_chain_opt.pdf'),
                bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print('[OK] fig04_hash_chain_opt.pdf  (Orange-Blue, error bars)')


# ============================================================
# FIG 2: Anomaly Detection - Baseline Comparison (2x2 layout)
# Save as: fig05_anomaly_results.pdf
# ============================================================
def fig_anomaly_evaluation():
    sota = load_json('comparison/sota_comparison.json')
    models_data = {}
    for m in sota['models']:
        name = m['model'].replace('DiffusionModel(Ours)', 'Diffusion (Ours)')
        models_data[name] = m

    order = ['IsolationForest', 'OneClassSVM', 'Autoencoder', 'Diffusion (Ours)']
    metrics = ['auc', 'f1', 'precision', 'recall']
    metric_labels = ['AUC', 'F1-Score', 'Precision', 'Recall']
    # Per-model colors: Blue, Orange, Yellow, Green (our method highlighted)
    model_colors = ['#0072BD', '#D95319', '#EDB120', '#77AC30']

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.2))
    axes_flat = axes.flatten()

    for i, (metric, mlabel, ax) in enumerate(zip(metrics, metric_labels, axes_flat)):
        vals = [models_data[n][metric] for n in order]
        bars = ax.bar(order, vals, color=model_colors, edgecolor='black',
                      linewidth=0.6, width=0.55)
        ax.set_title(mlabel, fontsize=9, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Score', fontsize=7.5)
        ax.tick_params(axis='x', labelsize=6.5, rotation=20)
        ax.grid(axis='y', alpha=0.25, linestyle=':', linewidth=0.4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=6.5)

    fig.suptitle('Anomaly Detection Performance (UNSW-NB15)',
                 fontsize=10, fontweight='bold', y=1.01)
    plt.tight_layout(pad=0.8)
    fig.savefig(os.path.join(FIGS_DIR, 'fig05_anomaly_results.pdf'),
                bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print('[OK] fig05_anomaly_results.pdf  (2x2, colored per-model, annotated)')


# ============================================================
# FIG 3: Ablation Study - Stacked Component Breakdown
# Save as: fig06_ablation.pdf
# ============================================================
def fig_ablation_study():
    abl = load_json('ablation/full_ablation_study.json')

    keys = ['Minimal', 'NoTimePredicate', 'NoCache', 'NoSubprocess',
            'NoDiffusion', 'NoDigitalTwin', 'Full']
    labels = ['Minimal', 'No Time\nPredicate', 'No Cache',
              'No\nSubprocess', 'No\nDiffusion', 'No Digital\nTwin', 'Full']

    keygen  = np.array([abl[k]['keygen_time'] * 1000 for k in keys])
    encrypt = np.array([abl[k]['encrypt_time'] * 1000 for k in keys])
    decrypt = np.array([abl[k]['decrypt_time'] * 1000 for k in keys])
    total   = keygen + encrypt + decrypt

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    x = np.arange(len(keys))
    w = 0.55

    # Stacked: Blue (KeyGen) -> Orange (Encrypt) -> Cyan (Decrypt)
    b1 = ax.bar(x, keygen, w, label='KeyGen',
                color='#0072BD', edgecolor='black', linewidth=0.5)
    b2 = ax.bar(x, encrypt, w, bottom=keygen, label='Encrypt',
                color='#D95319', edgecolor='black', linewidth=0.5)
    b3 = ax.bar(x, decrypt, w, bottom=keygen + encrypt, label='Decrypt',
                color='#4DBEEE', edgecolor='black', linewidth=0.5)

    for xi, t in zip(x, total):
        ax.text(xi, t + 3, f'{t:.1f}', ha='center', va='bottom', fontsize=6.5,
                fontweight='bold')

    ax.set_ylabel('Total Time (ms)', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.legend(loc='upper left', framealpha=0.9, edgecolor='black',
              fontsize=6.5, handlelength=2.0)
    ax.set_ylim(0, max(total) * 1.25)
    ax.grid(axis='y', alpha=0.25, linestyle=':', linewidth=0.4)

    plt.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIGS_DIR, 'fig06_ablation.pdf'),
                bbox_inches='tight', pad_inches=0.02)
    plt.close()
    print('[OK] fig06_ablation.pdf  (Blue-Orange-Cyan stacked, annotated)')


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    print('=== Generating SCI Q1 Colored Figures ===')
    fig_hash_chain_optimization()
    fig_anomaly_evaluation()
    fig_ablation_study()
    print('\nAll figures generated successfully.')
