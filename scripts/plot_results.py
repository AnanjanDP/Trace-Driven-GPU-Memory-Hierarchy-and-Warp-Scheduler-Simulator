#!/usr/bin/env python3
"""
Plotting Script for GPU Memory Hierarchy and Warp Scheduler Simulator
Generates publication-ready figures using matplotlib.
"""

import json
import os
import matplotlib.pyplot as plt


def set_plot_style():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 14


# 1. Cache Capacity Plot
def plot_exp1():
    with open("results/exp1_cache_capacity.json", "r") as f:
        data = json.load(f)

    caps = [d["capacity_kb"] for d in data]
    hit_rates = [d["l1_hit_rate"] * 100 for d in data]
    ipcs = [d["ipc"] for d in data]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    color = '#1f77b4'
    ax1.set_xlabel('L1 Cache Capacity (KB)')
    ax1.set_ylabel('L1 Hit Rate (%)', color=color)
    ax1.plot(caps, hit_rates, color=color, marker='o', linewidth=2.5, markersize=7, label='Hit Rate')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-5, 105)

    ax2 = ax1.twinx()
    color = '#d62728'
    ax2.set_ylabel('Instructions Per Cycle (IPC)', color=color)
    ax2.plot(caps, ipcs, color=color, marker='s', linestyle='--', linewidth=2.5, markersize=7, label='IPC')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Experiment 1: Effect of Cache Capacity on Hit Rate & IPC')
    fig.tight_layout()
    out = "results/exp1_cache_capacity.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[PLOT] Saved: {out}")


# 2. Associativity Plot
def plot_exp2():
    with open("results/exp2_associativity.json", "r") as f:
        data = json.load(f)

    assocs = [str(d["associativity"]) + "-way" for d in data]
    hit_rates = [d["l1_hit_rate"] * 100 for d in data]
    evictions = [d["l1_evictions"] for d in data]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    color = '#2ca02c'
    ax1.set_xlabel('Cache Set Associativity')
    ax1.set_ylabel('L1 Hit Rate (%)', color=color)
    bars = ax1.bar(assocs, hit_rates, color=color, alpha=0.7, width=0.4, label='Hit Rate')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 100)

    ax2 = ax1.twinx()
    color = '#ff7f0e'
    ax2.set_ylabel('L1 Evictions (Conflict Misses)', color=color)
    ax2.plot(assocs, evictions, color=color, marker='D', linewidth=2.5, markersize=8, label='Evictions')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Experiment 2: Cache Conflict Misses vs. Set Associativity')
    fig.tight_layout()
    out = "results/exp2_associativity.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[PLOT] Saved: {out}")


# 3. Warp Count & Latency Hiding Plot
def plot_exp3():
    with open("results/exp3_warp_count.json", "r") as f:
        data = json.load(f)

    warps = [d["warp_count"] for d in data]
    ipcs = [d["ipc"] for d in data]
    eligible = [d["avg_eligible_warps"] for d in data]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    color = '#9467bd'
    ax1.set_xlabel('Active Warp Count')
    ax1.set_ylabel('IPC (Throughput)', color=color)
    ax1.plot(warps, ipcs, color=color, marker='o', linewidth=2.5, markersize=7)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = '#8c564b'
    ax2.set_ylabel('Avg Eligible Warps / Cycle', color=color)
    ax2.plot(warps, eligible, color=color, marker='^', linestyle=':', linewidth=2.5, markersize=7)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Experiment 3: Latency Hiding & Diminishing Returns (Little's Law)")
    fig.tight_layout()
    out = "results/exp3_warp_latency_hiding.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[PLOT] Saved: {out}")


# 4. Memory Coalescing Plot
def plot_exp4():
    with open("results/exp4_coalescing.json", "r") as f:
        data = json.load(f)

    strides = [str(d["stride_words"]) for d in data]
    txns = [d["transactions_per_mem_inst"] for d in data]
    eff = [d["coalescing_efficiency"] * 100 for d in data]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    color = '#e377c2'
    ax1.set_xlabel('Memory Access Stride (32-bit Words)')
    ax1.set_ylabel('Transactions per Memory Instruction', color=color)
    ax1.plot(strides, txns, color=color, marker='s', linewidth=2.5, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 36)

    ax2 = ax1.twinx()
    color = '#17becf'
    ax2.set_ylabel('Coalescing Efficiency (%)', color=color)
    ax2.plot(strides, eff, color=color, marker='o', linestyle='--', linewidth=2.5, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 110)

    plt.title('Experiment 4: Memory Divergence & Coalescing Efficiency')
    fig.tight_layout()
    out = "results/exp4_coalescing.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[PLOT] Saved: {out}")


# 5. Scheduling Policy (RR vs GTO) Plot
def plot_exp5():
    with open("results/exp5_scheduling.json", "r") as f:
        data = json.load(f)

    workloads = ["matmul", "repeated", "streaming"]
    rr_ipc = [next(d["ipc"] for d in data if d["workload"] == wl and d["policy"] == "ROUND_ROBIN") for wl in workloads]
    gto_ipc = [next(d["ipc"] for d in data if d["workload"] == wl and d["policy"] == "GTO") for wl in workloads]

    import numpy as np
    x = np.arange(len(workloads))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width/2, rr_ipc, width, label='Round-Robin', color='#4a7bb7')
    ax.bar(x + width/2, gto_ipc, width, label='Greedy-Then-Oldest (GTO)', color='#e86c5f')

    ax.set_ylabel('Instructions Per Cycle (IPC)')
    ax.set_xlabel('Workload Access Pattern')
    ax.set_title('Experiment 5: Scheduling Policy Comparison (RR vs. GTO)')
    ax.set_xticks(x)
    ax.set_xticklabels(workloads)
    ax.legend()

    fig.tight_layout()
    out = "results/exp5_scheduling_rr_vs_gto.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[PLOT] Saved: {out}")


# 6. Replacement Policy (LRU vs FIFO) Plot
def plot_exp6():
    with open("results/exp6_replacement.json", "r") as f:
        data = json.load(f)

    workloads = ["repeated", "random"]
    lru_hit = [next(d["l1_hit_rate"] * 100 for d in data if d["workload"] == wl and d["replacement_policy"] == "LRU") for wl in workloads]
    fifo_hit = [next(d["l1_hit_rate"] * 100 for d in data if d["workload"] == wl and d["replacement_policy"] == "FIFO") for wl in workloads]

    import numpy as np
    x = np.arange(len(workloads))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width/2, lru_hit, width, label='LRU', color='#2ca02c')
    ax.bar(x + width/2, fifo_hit, width, label='FIFO', color='#bcbd22')

    ax.set_ylabel('L1 Hit Rate (%)')
    ax.set_xlabel('Workload Access Pattern')
    ax.set_title('Experiment 6: Cache Replacement Policy (LRU vs. FIFO)')
    ax.set_xticks(x)
    ax.set_xticklabels(workloads)
    ax.set_ylim(0, 100)
    ax.legend()

    fig.tight_layout()
    out = "results/exp6_replacement_lru_vs_fifo.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[PLOT] Saved: {out}")


# 7. Latency vs Bandwidth Plot
def plot_exp7():
    with open("results/exp7_latency_bandwidth.json", "r") as f:
        data = json.load(f)

    lat_data = data["latency_sweep"]
    bw_data = data["bandwidth_sweep"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Latency sweep
    lats = [d["latency"] for d in lat_data]
    lat_ipcs = [d["ipc"] for d in lat_data]
    ax1.plot(lats, lat_ipcs, color='#d62728', marker='o', linewidth=2.5)
    ax1.set_xlabel('DRAM Fixed Latency (Cycles)')
    ax1.set_ylabel('IPC')
    ax1.set_title('Latency Sensitivity (Fixed BW = 32 B/cyc)')

    # Bandwidth sweep
    bws = [d["bandwidth"] for d in bw_data]
    bw_ipcs = [d["ipc"] for d in bw_data]
    ax2.plot(bws, bw_ipcs, color='#1f77b4', marker='s', linewidth=2.5)
    ax2.set_xlabel('DRAM Sustained Bandwidth (B/cycle)')
    ax2.set_ylabel('IPC')
    ax2.set_title('Bandwidth Sensitivity (Fixed Latency = 200 cyc)')

    fig.suptitle('Experiment 7: Latency vs. Bandwidth Sensitivity')
    fig.tight_layout()
    out = "results/exp7_latency_vs_bandwidth.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[PLOT] Saved: {out}")


def main():
    set_plot_style()
    os.makedirs("results", exist_ok=True)

    plot_exp1()
    plot_exp2()
    plot_exp3()
    plot_exp4()
    plot_exp5()
    plot_exp6()
    plot_exp7()

    print("\n[SUCCESS] Generated all 7 microarchitecture experiment figures!")


if __name__ == "__main__":
    main()
