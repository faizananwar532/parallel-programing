#!/usr/bin/env python3
"""
Performance comparison of 3 versions of MPI Matrix Multiplication (Strategy 2)
  V1: Ring + blocking MPI_Sendrecv
  V2: Ring + non-blocking MPI_Isend/MPI_Irecv (overlap attempt)
  V3: MPI_Allgatherv collective (final optimized)
Matrix size: 8000x8000, Seed: 42, 5 runs per configuration
"""

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
#  Raw timing data from Fulda HPC Cluster runs
# ============================================================

nodes = [1, 2, 4, 6, 8]
ranks = [64, 128, 256, 384, 512]

# V1: Ring + blocking MPI_Sendrecv
v1_times = {
    1: [38.83, 38.80, 38.71, 38.88, 38.72],
    2: [23.64, 24.38, 23.20, 24.03, 24.48],
    4: [18.77, 18.00, 17.87, 18.25, 17.96],
    6: [14.75, 15.16, 15.15, 15.16, 15.49],
    8: [16.34, 16.26, 16.46, 16.43, 16.61],
}

# V2: Ring + non-blocking MPI_Isend/MPI_Irecv
v2_times = {
    1: [38.71, 38.65, 38.77, 39.14, 38.96],
    2: [24.71, 24.73, 24.19, 24.72, 24.95],
    4: [19.79, 18.70, 20.09, 19.77, 19.30],
    6: [14.95, 15.15, 15.17, 15.13, 15.28],
    8: [16.37, 16.45, 16.22, 16.24, 16.33],
}

# V3: MPI_Allgatherv collective (optimized)
v3_times = {
    1: [35.33, 34.89, 35.39, 35.24, 35.20],
    2: [19.36, 19.14, 19.14, 19.84, 19.22],
    4: [11.11, 11.61, 11.21, 11.14, 11.16],
    6: [8.39, 8.57, 8.39, 8.40, 8.92],
    8: [7.67, 7.84, 7.86, 7.87, 7.79],
}

versions = [
    ("V1: Ring + Sendrecv",     v1_times, "#E53935"),
    ("V2: Ring + Isend/Irecv",  v2_times, "#FB8C00"),
    ("V3: Allgatherv",          v3_times, "#43A047"),
]

# ============================================================
#  Compute statistics
# ============================================================
def compute_stats(times_dict):
    avg  = [np.mean(times_dict[n]) for n in nodes]
    std  = [np.std(times_dict[n])  for n in nodes]
    base = avg[0]
    sp   = [base / t for t in avg]
    return avg, std, sp

v1_avg, v1_std, v1_sp = compute_stats(v1_times)
v2_avg, v2_std, v2_sp = compute_stats(v2_times)
v3_avg, v3_std, v3_sp = compute_stats(v3_times)

ideal_sp = [n / nodes[0] for n in nodes]

# ============================================================
#  Print summary tables
# ============================================================
print("=" * 75)
print("  PERFORMANCE COMPARISON — MPI Matrix Multiplication (8000×8000)")
print("=" * 75)

for label, times_dict, _ in versions:
    avg, std, sp = compute_stats(times_dict)
    eff = [s / ideal * 100 for s, ideal in zip(sp, ideal_sp)]
    print(f"\n  {label}")
    print(f"  {'Nodes':>5} {'Ranks':>6} {'Avg(s)':>8} {'±σ':>7} {'Speedup':>8} {'Efficiency':>10}")
    print("  " + "-" * 48)
    for i, n in enumerate(nodes):
        print(f"  {n:>5} {ranks[i]:>6} {avg[i]:>8.2f} {std[i]:>7.2f} "
              f"{sp[i]:>7.2f}x {eff[i]:>9.1f}%")

print(f"\n{'=' * 75}")
print(f"  V3 improvement over V1 at 8 nodes: "
      f"{v1_avg[4]:.2f}s → {v3_avg[4]:.2f}s "
      f"({v1_avg[4]/v3_avg[4]:.1f}x faster)")
print(f"{'=' * 75}\n")

# ============================================================
#  Figure 1: Three-panel comparison (Time, Speedup, Efficiency)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("MPI Matrix Multiplication — Strategy 2 Evolution (8000×8000)",
             fontsize=15, fontweight='bold', y=1.02)

# --- Panel 1: Execution Time ---
ax1 = axes[0]
for label, times_dict, color in versions:
    avg = [np.mean(times_dict[n]) for n in nodes]
    std = [np.std(times_dict[n]) for n in nodes]
    ax1.errorbar(nodes, avg, yerr=std, fmt='o-', color=color,
                 linewidth=2, markersize=7, capsize=4, label=label)

ax1.set_xlabel('Nodes', fontsize=12)
ax1.set_ylabel('Execution Time (s)', fontsize=12)
ax1.set_title('Execution Time', fontsize=12, fontweight='bold')
ax1.set_xticks(nodes)
ax1.set_xticklabels([f"{n}\n({r})" for n, r in zip(nodes, ranks)], fontsize=9)
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(True, alpha=0.3)

# Annotate V1 degradation
ax1.annotate('degradation!', xy=(8, v1_avg[4]), fontsize=8, color='#E53935',
             xytext=(6.5, v1_avg[4]+2),
             arrowprops=dict(arrowstyle='->', color='#E53935', lw=1.5))

# --- Panel 2: Speedup ---
ax2 = axes[1]
ax2.plot(nodes, ideal_sp, 'k--', linewidth=1.5, alpha=0.4, label='Ideal (linear)')
for (label, _, color), sp in zip(versions, [v1_sp, v2_sp, v3_sp]):
    ax2.plot(nodes, sp, 'o-', color=color, linewidth=2, markersize=7, label=label)

ax2.set_xlabel('Nodes', fontsize=12)
ax2.set_ylabel('Speedup', fontsize=12)
ax2.set_title('Speedup (relative to own 1-node baseline)', fontsize=12, fontweight='bold')
ax2.set_xticks(nodes)
ax2.set_xticklabels([f"{n}\n({r})" for n, r in zip(nodes, ranks)], fontsize=9)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

for i, n in enumerate(nodes):
    if i > 0:
        ax2.annotate(f'{v3_sp[i]:.2f}x', (n, v3_sp[i]),
                     textcoords="offset points", xytext=(8, 5),
                     fontsize=8, color='#43A047', fontweight='bold')

# --- Panel 3: Bar chart — 8-node comparison ---
ax3 = axes[2]
labels_short = ['V1\nSendrecv', 'V2\nIsend/Irecv', 'V3\nAllgatherv']
times_8node = [v1_avg[4], v2_avg[4], v3_avg[4]]
bar_colors = ['#E53935', '#FB8C00', '#43A047']
bars = ax3.bar(labels_short, times_8node, color=bar_colors,
               edgecolor='white', linewidth=2, width=0.5)

for bar, t in zip(bars, times_8node):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{t:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax3.set_ylabel('Execution Time at 8 nodes (s)', fontsize=12)
ax3.set_title('8-Node Comparison (512 ranks)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0, max(times_8node) * 1.15)

plt.tight_layout()
plt.savefig('performance_analysis.png', dpi=150, bbox_inches='tight')
plt.savefig('performance_analysis.pdf', bbox_inches='tight')
print("Saved: performance_analysis.png and performance_analysis.pdf")

# ============================================================
#  Figure 2: Detailed V3 analysis (standalone)
# ============================================================
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle("V3 Allgatherv — Detailed Analysis (8000×8000)",
              fontsize=14, fontweight='bold', y=1.02)

v3_eff = [s / ideal * 100 for s, ideal in zip(v3_sp, ideal_sp)]

# Individual runs scatter + avg
ax = axes2[0]
for i, n in enumerate(nodes):
    ax.scatter([n]*5, v3_times[n], color='#66BB6A', alpha=0.5, s=30, zorder=3)
ax.errorbar(nodes, v3_avg, yerr=v3_std, fmt='o-', color='#2E7D32',
            linewidth=2, markersize=8, capsize=5, capthick=2, zorder=4, label='Avg ± σ')
ax.set_xlabel('Nodes', fontsize=12)
ax.set_ylabel('Execution Time (s)', fontsize=12)
ax.set_title('Execution Time', fontsize=12, fontweight='bold')
ax.set_xticks(nodes)
ax.set_xticklabels([f"{n}\n({r})" for n, r in zip(nodes, ranks)], fontsize=9)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
for i, n in enumerate(nodes):
    ax.annotate(f'{v3_avg[i]:.2f}s', (n, v3_avg[i]),
                textcoords="offset points", xytext=(10, 5), fontsize=9)

# Speedup
ax = axes2[1]
ax.plot(nodes, ideal_sp, 'k--', linewidth=1.5, alpha=0.4, label='Ideal')
ax.plot(nodes, v3_sp, 'o-', color='#43A047', linewidth=2, markersize=8, label='V3 Measured')
ax.fill_between(nodes, v3_sp, alpha=0.15, color='#43A047')
ax.set_xlabel('Nodes', fontsize=12)
ax.set_ylabel('Speedup', fontsize=12)
ax.set_title('Speedup', fontsize=12, fontweight='bold')
ax.set_xticks(nodes)
ax.set_xticklabels([f"{n}\n({r})" for n, r in zip(nodes, ranks)], fontsize=9)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
for i, n in enumerate(nodes):
    ax.annotate(f'{v3_sp[i]:.2f}x', (n, v3_sp[i]),
                textcoords="offset points", xytext=(10, 5), fontsize=9)

# Efficiency
ax = axes2[2]
bar_colors_v3 = ['#66BB6A', '#4CAF50', '#43A047', '#388E3C', '#2E7D32']
bars = ax.bar(range(len(nodes)), v3_eff, color=bar_colors_v3,
              edgecolor='white', linewidth=1.5, width=0.6)
ax.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.4, label='100% (ideal)')
ax.set_xlabel('Nodes', fontsize=12)
ax.set_ylabel('Parallel Efficiency (%)', fontsize=12)
ax.set_title('Parallel Efficiency', fontsize=12, fontweight='bold')
ax.set_xticks(range(len(nodes)))
ax.set_xticklabels([f"{n}\n({r})" for n, r in zip(nodes, ranks)], fontsize=9)
ax.set_ylim(0, 120)
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=10)
for bar, eff in zip(bars, v3_eff):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{eff:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('performance_analysis_v3_detail.png', dpi=150, bbox_inches='tight')
plt.savefig('performance_analysis_v3_detail.pdf', bbox_inches='tight')
print("Saved: performance_analysis_v3_detail.png and performance_analysis_v3_detail.pdf")

plt.show()
