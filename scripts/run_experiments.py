#!/usr/bin/env python3
"""
Automated Experiment Runner for GPU Memory Hierarchy and Warp Scheduler Simulator
Runs 7 microarchitectural parameter sweeps and saves raw structured results.
"""

import json
import os
import subprocess
import sys

from generate_traces import generate_trace


def find_simulator_binary():
    candidates = [
        os.path.join("build", "Release", "gpu_simulator.exe"),
        os.path.join("build", "gpu_simulator.exe"),
        os.path.join("build", "gpu_simulator"),
        "gpu_simulator.exe"
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError("Could not locate gpu_simulator executable. Please build the project first.")


def run_single_simulation(sim_bin: str, config_data: dict, trace_path: str, stats_path: str):
    config_temp = "configs/temp_run.json"
    config_data["trace_file"] = trace_path
    config_data["output_stats_file"] = stats_path

    with open(config_temp, "w") as f:
        json.dump(config_data, f, indent=4)

    cmd = [sim_bin, "--config", config_temp, "--trace", trace_path, "--stats", stats_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Simulation failed:\n{result.stderr}")
        sys.exit(1)

    with open(stats_path, "r") as f:
        return json.load(f)


def get_base_config():
    with open("configs/default.json", "r") as f:
        return json.load(f)


# ==========================================
# Experiment 1: Cache Capacity Sweep
# ==========================================
def run_exp1_cache_capacity(sim_bin):
    print("\n[EXPERIMENT 1] Cache Capacity Sweep (L1: 4KB -> 64KB)...")
    trace_path = "traces/exp1_capacity.trace"
    # Working set: 16 KB working set repeated access
    generate_trace("repeated", num_warps=8, num_instructions=80, stride_words=1,
                   access_size=4, alu_ratio=0.2, seed=101, output_path=trace_path)

    capacities = [1024, 2048, 4096, 8192, 16384, 32768]
    results = []

    for cap in capacities:
        cfg = get_base_config()
        cfg["warp_count"] = 8
        cfg["l1_cache"]["capacity_bytes"] = cap
        cfg["l1_cache"]["associativity"] = 4
        cfg["l1_cache"]["line_size_bytes"] = 128

        stats_path = f"results/exp1_cap_{cap}.json"
        stats = run_single_simulation(sim_bin, cfg, trace_path, stats_path)

        results.append({
            "capacity_bytes": cap,
            "capacity_kb": cap // 1024,
            "ipc": stats["ipc"],
            "cycles": stats["total_simulated_cycles"],
            "l1_hit_rate": stats["l1_cache"]["hit_rate"],
            "l1_hits": stats["l1_cache"]["total_hits"],
            "l1_misses": stats["l1_cache"]["total_misses"],
            "dram_requests": stats["dram"]["total_requests"]
        })
        print(f"  Capacity: {cap//1024:2d} KB -> L1 Hit Rate: {stats['l1_cache']['hit_rate']*100:.1f}%, IPC: {stats['ipc']:.4f}")

    with open("results/exp1_cache_capacity.json", "w") as f:
        json.dump(results, f, indent=4)


# ==========================================
# Experiment 2: Associativity Sweep
# ==========================================
def run_exp2_associativity(sim_bin):
    print("\n[EXPERIMENT 2] Associativity Sweep (1-way to 16-way)...")
    trace_path = "traces/exp2_associativity.trace"
    # Strided pattern designed to stress set conflicts
    generate_trace("strided", num_warps=4, num_instructions=60, stride_words=16,
                   access_size=4, alu_ratio=0.2, seed=202, output_path=trace_path)

    associativities = [1, 2, 4, 8, 16]
    results = []

    for assoc in associativities:
        cfg = get_base_config()
        cfg["warp_count"] = 4
        cfg["l1_cache"]["capacity_bytes"] = 32768
        cfg["l1_cache"]["associativity"] = assoc
        cfg["l1_cache"]["line_size_bytes"] = 128

        stats_path = f"results/exp2_assoc_{assoc}.json"
        stats = run_single_simulation(sim_bin, cfg, trace_path, stats_path)

        results.append({
            "associativity": assoc,
            "ipc": stats["ipc"],
            "cycles": stats["total_simulated_cycles"],
            "l1_hit_rate": stats["l1_cache"]["hit_rate"],
            "l1_evictions": stats["l1_cache"]["evictions"],
            "l1_misses": stats["l1_cache"]["total_misses"]
        })
        print(f"  Associativity: {assoc:2d}-way -> Hit Rate: {stats['l1_cache']['hit_rate']*100:.1f}%, Evictions: {stats['l1_cache']['evictions']}")

    with open("results/exp2_associativity.json", "w") as f:
        json.dump(results, f, indent=4)


# ==========================================
# Experiment 3: Warp Count & Latency Hiding
# ==========================================
def run_exp3_warp_count(sim_bin):
    print("\n[EXPERIMENT 3] Warp Count & Latency Hiding Sweep (1 to 64 Warps)...")
    warp_counts = [1, 2, 4, 8, 16, 32, 48, 64]
    results = []

    for w in warp_counts:
        trace_path = f"traces/exp3_warps_{w}.trace"
        generate_trace("sequential", num_warps=w, num_instructions=40, stride_words=1,
                       access_size=4, alu_ratio=0.4, seed=303, output_path=trace_path)

        cfg = get_base_config()
        cfg["warp_count"] = w
        cfg["dram"]["latency"] = 150
        cfg["dram"]["bandwidth_bytes_per_cycle"] = 32.0

        stats_path = f"results/exp3_warps_{w}.json"
        stats = run_single_simulation(sim_bin, cfg, trace_path, stats_path)

        results.append({
            "warp_count": w,
            "ipc": stats["ipc"],
            "cycles": stats["total_simulated_cycles"],
            "memory_stall_cycles": stats["total_memory_stall_cycles"],
            "scheduler_utilization": stats["scheduler"]["scheduler_utilization"],
            "avg_eligible_warps": stats["scheduler"]["average_eligible_warps"]
        })
        print(f"  Warps: {w:2d} -> IPC: {stats['ipc']:.4f}, Avg Eligible: {stats['scheduler']['average_eligible_warps']:.2f}, Sched Util: {stats['scheduler']['scheduler_utilization']*100:.1f}%")

    with open("results/exp3_warp_count.json", "w") as f:
        json.dump(results, f, indent=4)


# ==========================================
# Experiment 4: Memory Coalescing
# ==========================================
def run_exp4_coalescing(sim_bin):
    print("\n[EXPERIMENT 4] Memory Coalescing Evaluation (Stride 1 to 32 words)...")
    strides = [1, 2, 4, 8, 16, 32]
    results = []

    for s in strides:
        trace_path = f"traces/exp4_stride_{s}.trace"
        generate_trace("strided", num_warps=8, num_instructions=50, stride_words=s,
                       access_size=4, alu_ratio=0.1, seed=404, output_path=trace_path)

        cfg = get_base_config()
        cfg["warp_count"] = 8
        stats_path = f"results/exp4_stride_{s}.json"
        stats = run_single_simulation(sim_bin, cfg, trace_path, stats_path)

        results.append({
            "stride_words": s,
            "stride_bytes": s * 4,
            "transactions_per_mem_inst": stats["transactions_per_mem_inst"],
            "coalescing_efficiency": stats["coalescing_efficiency"],
            "total_coalesced_transactions": stats["total_coalesced_transactions"],
            "dram_transferred_bytes": stats["dram"]["total_bytes_transferred"],
            "ipc": stats["ipc"],
            "cycles": stats["total_simulated_cycles"]
        })
        print(f"  Stride: {s:2d} words -> Txns/Inst: {stats['transactions_per_mem_inst']:.1f}, Efficiency: {stats['coalescing_efficiency']*100:.1f}%, IPC: {stats['ipc']:.4f}")

    with open("results/exp4_coalescing.json", "w") as f:
        json.dump(results, f, indent=4)


# ==========================================
# Experiment 5: Scheduling Policy (RR vs GTO)
# ==========================================
def run_exp5_scheduling(sim_bin):
    print("\n[EXPERIMENT 5] Scheduling Policy Comparison (Round-Robin vs GTO)...")
    workloads = ["repeated", "matmul", "streaming"]
    results = []

    for wl in workloads:
        trace_path = f"traces/exp5_{wl}.trace"
        generate_trace(wl, num_warps=16, num_instructions=50, stride_words=1,
                       access_size=4, alu_ratio=0.3, seed=505, output_path=trace_path)

        for pol in ["ROUND_ROBIN", "GTO"]:
            cfg = get_base_config()
            cfg["warp_count"] = 16
            cfg["scheduling_policy"] = pol
            stats_path = f"results/exp5_{wl}_{pol}.json"
            stats = run_single_simulation(sim_bin, cfg, trace_path, stats_path)

            results.append({
                "workload": wl,
                "policy": pol,
                "ipc": stats["ipc"],
                "cycles": stats["total_simulated_cycles"],
                "l1_hit_rate": stats["l1_cache"]["hit_rate"],
                "total_memory_stalls": stats["total_memory_stall_cycles"],
                "scheduler_utilization": stats["scheduler"]["scheduler_utilization"]
            })
            print(f"  Workload: {wl:10s} | Policy: {pol:12s} -> IPC: {stats['ipc']:.4f}, L1 Hit: {stats['l1_cache']['hit_rate']*100:.1f}%, Cycles: {stats['total_simulated_cycles']}")

    with open("results/exp5_scheduling.json", "w") as f:
        json.dump(results, f, indent=4)


# ==========================================
# Experiment 6: Replacement Policy (LRU vs FIFO)
# ==========================================
def run_exp6_replacement(sim_bin):
    print("\n[EXPERIMENT 6] Cache Replacement Policies (LRU vs FIFO)...")
    workloads = ["repeated", "random"]
    results = []

    for wl in workloads:
        trace_path = f"traces/exp6_{wl}.trace"
        generate_trace(wl, num_warps=8, num_instructions=60, stride_words=1,
                       access_size=4, alu_ratio=0.2, seed=606, output_path=trace_path)

        for rep in ["LRU", "FIFO"]:
            cfg = get_base_config()
            cfg["warp_count"] = 8
            cfg["l1_cache"]["replacement_policy"] = rep
            cfg["l2_cache"]["replacement_policy"] = rep
            stats_path = f"results/exp6_{wl}_{rep}.json"
            stats = run_single_simulation(sim_bin, cfg, trace_path, stats_path)

            results.append({
                "workload": wl,
                "replacement_policy": rep,
                "ipc": stats["ipc"],
                "cycles": stats["total_simulated_cycles"],
                "l1_hit_rate": stats["l1_cache"]["hit_rate"],
                "l1_evictions": stats["l1_cache"]["evictions"]
            })
            print(f"  Workload: {wl:10s} | Replacement: {rep:4s} -> L1 Hit: {stats['l1_cache']['hit_rate']*100:.1f}%, Evictions: {stats['l1_cache']['evictions']}")

    with open("results/exp6_replacement.json", "w") as f:
        json.dump(results, f, indent=4)


# ==========================================
# Experiment 7: Latency vs Bandwidth
# ==========================================
def run_exp7_latency_bandwidth(sim_bin):
    print("\n[EXPERIMENT 7] Latency vs Bandwidth Sensitivity...")
    trace_path = "traces/exp7_workload.trace"
    generate_trace("streaming", num_warps=16, num_instructions=40, stride_words=1,
                   access_size=4, alu_ratio=0.2, seed=707, output_path=trace_path)

    # 1. Sweep Latency (fixed BW = 32)
    latencies = [50, 100, 200, 400]
    lat_results = []
    for lat in latencies:
        cfg = get_base_config()
        cfg["warp_count"] = 16
        cfg["dram"]["latency"] = lat
        cfg["dram"]["bandwidth_bytes_per_cycle"] = 32.0
        stats = run_single_simulation(sim_bin, cfg, trace_path, f"results/exp7_lat_{lat}.json")
        lat_results.append({
            "latency": lat,
            "ipc": stats["ipc"],
            "cycles": stats["total_simulated_cycles"],
            "dram_bw_util": stats["dram"]["bandwidth_utilization"]
        })
        print(f"  Latency: {lat:3d} cyc -> IPC: {stats['ipc']:.4f}, Cycles: {stats['total_simulated_cycles']}")

    # 2. Sweep Bandwidth (fixed Latency = 200)
    bandwidths = [8.0, 16.0, 32.0, 64.0, 128.0]
    bw_results = []
    for bw in bandwidths:
        cfg = get_base_config()
        cfg["warp_count"] = 16
        cfg["dram"]["latency"] = 200
        cfg["dram"]["bandwidth_bytes_per_cycle"] = bw
        stats = run_single_simulation(sim_bin, cfg, trace_path, f"results/exp7_bw_{int(bw)}.json")
        bw_results.append({
            "bandwidth": bw,
            "ipc": stats["ipc"],
            "cycles": stats["total_simulated_cycles"],
            "dram_bw_util": stats["dram"]["bandwidth_utilization"]
        })
        print(f"  Bandwidth: {bw:5.1f} B/cyc -> IPC: {stats['ipc']:.4f}, BW Util: {stats['dram']['bandwidth_utilization']*100:.1f}%")

    with open("results/exp7_latency_bandwidth.json", "w") as f:
        json.dump({"latency_sweep": lat_results, "bandwidth_sweep": bw_results}, f, indent=4)


def main():
    sim_bin = find_simulator_binary()
    print(f"[INFO] Using simulator binary: {sim_bin}")

    os.makedirs("results", exist_ok=True)
    os.makedirs("traces", exist_ok=True)

    run_exp1_cache_capacity(sim_bin)
    run_exp2_associativity(sim_bin)
    run_exp3_warp_count(sim_bin)
    run_exp4_coalescing(sim_bin)
    run_exp5_scheduling(sim_bin)
    run_exp6_replacement(sim_bin)
    run_exp7_latency_bandwidth(sim_bin)

    print("\n[SUCCESS] All 7 experiments completed successfully! Raw results saved in results/")


if __name__ == "__main__":
    main()
