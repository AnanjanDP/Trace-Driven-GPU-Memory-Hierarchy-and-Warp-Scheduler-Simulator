#!/usr/bin/env python3
"""
Performance Profiling and Throughput Benchmark for GPU Architecture Simulator
Measures execution time, throughput (KIPS and MCycles/s), and scalability.
"""

import os
import sys
import time
import json
import subprocess
from typing import Dict, Any, List

BUILD_DIR = "build/Release"
EXE = os.path.join(BUILD_DIR, "gpu_simulator.exe" if os.name == "nt" else "gpu_simulator")
RESULTS_DIR = "results"
TRACES_DIR = "traces"


def run_command(cmd: List[str]) -> Dict[str, Any]:
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.perf_counter()
    wall_clock = t1 - t0

    if p.returncode != 0:
        raise RuntimeError(f"Command failed (code {p.returncode}):\n{p.stderr}")

    return {
        "wall_clock_sec": wall_clock,
        "stdout": p.stdout,
        "stderr": p.stderr,
        "code": p.returncode
    }


def ensure_trace(pattern: str, warps: int, instructions: int, output_path: str):
    if not os.path.exists(output_path):
        print(f"[GEN] Generating trace: {output_path} ({pattern}, {warps} warps, {instructions} inst/warp)...")
        cmd = [
            sys.executable, "scripts/generate_traces.py",
            "--pattern", pattern,
            "--warps", str(warps),
            "--instructions", str(instructions),
            "--output", output_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Trace generation failed:\n{res.stderr}")


def benchmark_suite(suite_name: str) -> Dict[str, Any]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TRACES_DIR, exist_ok=True)

    benchmarks = [
        {
            "name": "Sequential_Coalesced_80K",
            "pattern": "sequential",
            "warps": 16,
            "instructions": 5000,
            "trace": "traces/bench_seq_80k.trace",
            "config": "configs/default.json",
            "expected_instructions": 80000
        },
        {
            "name": "Random_Memory_Bound_96K",
            "pattern": "random",
            "warps": 32,
            "instructions": 3000,
            "trace": "traces/bench_rand_96k.trace",
            "config": "configs/default.json",
            "expected_instructions": 96000
        },
        {
            "name": "Matmul_GTO_High_Contention_64K",
            "pattern": "matmul",
            "warps": 32,
            "instructions": 2000,
            "trace": "traces/bench_matmul_64k.trace",
            "config": "configs/gto.json",
            "expected_instructions": 64000
        }
    ]

    # Create configs/gto.json if needed
    if not os.path.exists("configs/gto.json"):
        with open("configs/default.json", "r") as f:
            cfg = json.load(f)
        cfg["scheduling_policy"] = "GTO"
        with open("configs/gto.json", "w") as f:
            json.dump(cfg, f, indent=4)

    results = []

    print(f"\n========================================================")
    print(f"  RUNNING BENCHMARK SUITE: {suite_name}")
    print(f"========================================================")

    for bench in benchmarks:
        ensure_trace(bench["pattern"], bench["warps"], bench["instructions"], bench["trace"])
        stats_out = os.path.join(RESULTS_DIR, f"prof_{bench['name']}.json")

        cmd = [
            EXE,
            "--config", bench["config"],
            "--trace", bench["trace"],
            "--stats", stats_out
        ]

        # Warmup run
        _ = run_command(cmd)

        # Timed runs (3 repetitions)
        times = []
        for rep in range(3):
            run_res = run_command(cmd)
            times.append(run_res["wall_clock_sec"])

        avg_time = sum(times) / len(times)
        min_time = min(times)

        # Parse simulator stats
        with open(stats_out, "r") as f:
            stats = json.load(f)

        instructions = stats.get("total_instructions_issued", bench["expected_instructions"])
        cycles = stats.get("total_simulated_cycles", 1)
        ipc = stats.get("ipc", 0.0)

        kips = (instructions / min_time) / 1000.0
        mcycles_per_sec = (cycles / min_time) / 1e6

        record = {
            "name": bench["name"],
            "pattern": bench["pattern"],
            "warps": bench["warps"],
            "instructions": instructions,
            "cycles": cycles,
            "ipc": ipc,
            "min_time_sec": min_time,
            "avg_time_sec": avg_time,
            "kips": kips,
            "mcycles_per_sec": mcycles_per_sec
        }
        results.append(record)

        print(f"[{bench['name']}]")
        print(f"  Instructions:     {instructions:,}")
        print(f"  Simulated Cycles: {cycles:,} (IPC: {ipc:.4f})")
        print(f"  Best Time:        {min_time*1000.0:.2f} ms (Avg: {avg_time*1000.0:.2f} ms)")
        print(f"  Throughput:       {kips:.2f} KIPS ({kips/1000.0:.3f} MIPS)")
        print(f"  Cycle Sim Rate:   {mcycles_per_sec:.2f} Mcycles/sec\n")

    output_summary = {
        "suite_name": suite_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": results
    }

    out_file = os.path.join(RESULTS_DIR, f"{suite_name}.json")
    with open(out_file, "w") as f:
        json.dump(output_summary, f, indent=4)
    print(f"[SUCCESS] Saved profiling results to: {out_file}\n")
    return output_summary


if __name__ == "__main__":
    suite_id = sys.argv[1] if len(sys.argv) > 1 else "profiling_baseline"
    benchmark_suite(suite_id)
