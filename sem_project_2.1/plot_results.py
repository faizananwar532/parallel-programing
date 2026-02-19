#!/usr/bin/env python3
"""
Visualization of MPI Matrix Multiplication Performance
Strategy 2: Row-block distribution with ring-based B shifting
Matrix size: 8000x8000, 5 runs per configuration
"""

import matplotlib.pyplot as plt
import numpy as np

# === Raw data ===
nodes     = [1,     2,     4,     6,     8]
ranks     = [64,    128,   256,   384,   512]
times_all = {
    1: [38.83, 38.80, 38.71, 38.88, 38.72],
    2: [23.64, 24.38, 23.20, 24.03, 24.48],
    4: [18.77, 18.00, 17.87, 18.25, 17.96],
    6: [14.75, 15.16, 15.15, 15.16, 15.49],
    8: [16.34, 16.26, 16.46, 16.43, 16.61],
}

# === Compute statistics ===
avg_times  = [np.mean(times_all[n]) for n in nodes]
std_times  = [np.std(times_all[n])  for n in nodes]
min_times  = [np.min(times_all[n])  for n in nodes]
max_times  = [np.max(times_all[n])  for n in nodes]

baseline   = avg_times[0]  # 1-node baseline
speedup    = [baseline / t for t in avg_times]
ideal_sp   = [n / nodes[0] for n in nodes]
efficiency = [s / ideal * 100 for s, ideal in zip(speedup, ideal_sp)]

# === Print summary table ===
print(f"{'Nodes':>5} {'Ranks':>6} {'Avg(s)':>8} {'Std(s)':>7} {'Speedup':>8} {'Ideal':>6} {'Eff%':>6}")
print("-" * 50)
for i, n in enumerate(nodes):
    print(f"{n:>5} {ranks[i]:>6} {avg_times[i]:>8.2f} {std_times[i]:>7.2f} "
          f"{speedup[i]:>8.2f}x {ideal_sp[i]:>5.1f}x {efficiency[i]:>5.1f}%")

# === Create figure with 3 subplots ===
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("MPI Matrix Multiplication — Ring-based B Shifting (8000×8000)",
             fontsize=14, fontweight='bold', y=1.02)

colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']

# --- 1. Execution Time with error bars and individual runs ---
ax1 = axes[0]
for i, n in enumerate(nodes):
    ax1.scatter([n]*5, times_all[n], color=colors[i], alpha=0.4, s=30, zorder=3)
ax1.errorbar(nodes, avg_times, yerr=std_times, fmt='o-', color='#1565C0',
             linewidth=2, markersize=8, capsize=5, capthick=2, zorder=4, label='Avg ± σ')
ax1.set_xlabel('Nodes', fontsize=12)
ax1.set_ylabel('Execution Time (s)', fontsize=12)
ax1.set_title('Execution Time vs Nodes', fontsize=12, fontweight='bold')
ax1.set_xticks(nodes)
ax1.set_xticklabels([f"{n}\n({r} ranks)" for n, r in zip(nodes, ranks)])
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Annotate avg times
for i, n in enumerate(nodes):
    ax1.annotate(f'{avg_times[i]:.2f}s', (n, avg_times[i]),
                 textcoords="offset points", xytext=(12, 5), fontsize=9)

# --- 2. Speedup ---
ax2 = axes[1]
ax2.plot(nodes, ideal_sp, 'k--', linewidth=1.5, label='Ideal (linear)', alpha=0.5)
ax2.plot(nodes, speedup, 'o-', color='#E91E63', linewidth=2, markersize=8, label='Measured')
ax2.fill_between(nodes, speedup, alpha=0.15, color='#E91E63')
ax2.set_xlabel('Nodes', fontsize=12)
ax2.set_ylabel('Speedup', fontsize=12)
ax2.set_title('Speedup (relative to 1 node)', fontsize=12, fontweight='bold')
ax2.set_xticks(nodes)
ax2.set_xticklabels([f"{n}\n({r} ranks)" for n, r in zip(nodes, ranks)])
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

for i, n in enumerate(nodes):
    ax2.annotate(f'{speedup[i]:.2f}x', (n, speedup[i]),
                 textcoords="offset points", xytext=(10, 5), fontsize=9)

# --- 3. Parallel Efficiency ---
ax3 = axes[2]
bars = ax3.bar(range(len(nodes)), efficiency, color=colors, edgecolor='white', linewidth=1.5, width=0.6)
ax3.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.4, label='100% (ideal)')
ax3.set_xlabel('Nodes', fontsize=12)
ax3.set_ylabel('Parallel Efficiency (%)', fontsize=12)
ax3.set_title('Parallel Efficiency', fontsize=12, fontweight='bold')
ax3.set_xticks(range(len(nodes)))
ax3.set_xticklabels([f"{n}\n({r} ranks)" for n, r in zip(nodes, ranks)])
ax3.set_ylim(0, 120)
ax3.grid(True, alpha=0.3, axis='y')
ax3.legend(fontsize=10)

for bar, eff in zip(bars, efficiency):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f'{eff:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/fd0002007/Desktop/parallel-programing/sem_project_2.1/performance_analysis.png',
            dpi=150, bbox_inches='tight')
plt.savefig('/home/fd0002007/Desktop/parallel-programing/sem_project_2.1/performance_analysis.pdf',
            bbox_inches='tight')
print("\nSaved: performance_analysis.png and performance_analysis.pdf")
plt.show()
