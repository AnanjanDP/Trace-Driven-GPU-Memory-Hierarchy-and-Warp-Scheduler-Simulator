# Trace-Driven GPU Memory Hierarchy and Warp Scheduler Simulator

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B17)
[![CMake](https://img.shields.io/badge/CMake-3.15%2B-064F8C.svg)](https://cmake.org/)
[![Tests](https://img.shields.io/badge/Tests-28%2F28%20Passing%20(211%20assertions)-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A cycle-stepped, trace-driven GPU architectural simulator written in **C++17** with a Python experimental framework. Models the execution timing, warp scheduling policies, SIMT memory coalescing, multi-level sectored cache hierarchy, and DRAM queuing behavior of a modern GPU Streaming Multiprocessor (SM).

---

## Architecture Overview

```
+---------------------------------------------------------------------------------------+
|                                STREAMING MULTIPROCESSOR                               |
|                                                                                       |
|   +-------------------+    +--------------------+    +----------------------------+   |
|   | Warp Pool [0..W-1]|--->|   Warp Scheduler   |--->|    SIMT Issue & Decode     |   |
|   | (State, Stall cyc)|    |   (RR / GTO)       |    | (ALU / Memory Instructions)|   |
|   +-------------------+    +--------------------+    +--------------+-------------+   |
|                                                                     |                 |
|                                                                     v                 |
|                                                     +-------------------------------+ |
|                                                     | Memory Coalescing Unit        | |
|                                                     | - 32-lane mask evaluation     | |
|                                                     | - 32B Sector segmentation     | |
|                                                     | - 128B Cache line grouping    | |
|                                                     +---------------+---------------+ |
+---------------------------------------------------------------------|-----------------+
                                                                      | Coalesced Transactions
                                                                      v
                                                      +-------------------------------+
                                                      | L1 Data Cache (per-SM)        |
                                                      | - 32 KB, 4-way, 64 sets       |
                                                      | - Sector valid/dirty tracking |
                                                      | - LRU / FIFO replacement      |
                                                      +---------------+---------------+
                                                                      | L1 Misses / Write-backs
                                                                      v
                                                      +-------------------------------+
                                                      | L2 Shared Cache               |
                                                      | - 256 KB, 16-way, 128 sets    |
                                                      | - Unified sector management   |
                                                      +---------------+---------------+
                                                                      | L2 Misses / Write-backs
                                                                      v
                                                      +-------------------------------+
                                                      | DRAM Subsystem Model          |
                                                      | - Fixed burst latency (200 c) |
                                                      | - Channel BW serialization    |
                                                      | - FIFO request queuing        |
                                                      +-------------------------------+
```

---

## Key Microarchitectural Features

### 1. Warp Schedulers
- **Round-Robin (RR)**: Rotating priority pointer distributing issue slots evenly across warps.
- **Greedy-Then-Oldest (GTO)**: Prioritizes retaining the currently executing warp as long as it remains ready; falls back to the oldest waiting warp (`min(last_issued_cycle)`) with deterministic `WarpId` tie-breaking. Desynchronizes warp execution phases to mitigate memory burst contention.

### 2. SIMT Memory Coalescing Unit
- Evaluates 32 active thread lanes per warp memory instruction.
- Groups thread addresses into **32-byte sectors** within **128-byte cache lines**.
- Emits coalesced `MemoryTransaction` packets with lane contributing bitmasks.
- Quantifies **Coalescing Efficiency**:
  $$\eta_{\text{coalescing}} = \frac{\sum_{\text{active lanes}} \text{Access Size}}{\sum_{\text{transactions}} \text{Sector Size}}$$
  Measures bandwidth amplification from 4 transactions ($100\%$ efficiency) to 32 transactions ($12.5\%$ efficiency) under stride divergence.

### 3. Multi-Level Sectored Cache Hierarchy
- **L1 Data Cache**: 32 KB, 4-way set-associative, 128B line size with 4 independent 32B sectors.
- **L2 Unified Cache**: 256 KB, 16-way set-associative, 128B line size with 4 independent 32B sectors.
- **Replacement Policies**: Bit-exact implementations of **LRU** (Least Recently Used) and **FIFO** (First-In, First-Out).
- **Decoupled Operations**: Separate `probe()`, `access()`, and `fill()` methods cleanly separating hit latency from line installation.

### 4. DRAM Channel & Queuing Model
- Combines a fixed propagation latency ($t_{\text{latency}} = 200 \text{ cycles}$) with an analytical bandwidth transfer serialization queue:
  $$\Delta t_{\text{transfer}} = \left\lceil \frac{\text{Sector Size (32 B)}}{\text{Bandwidth (Bytes/Cycle)}} \right\rceil$$
- Accurately captures queuing delay and latency inflation under heavy memory contention.

---

## Architectural Experiments & Empirical Findings

All experiments are fully automated via `scripts/run_experiments.py` and plotted via `scripts/plot_results.py`. Raw JSON metrics are stored in `results/`.

| Experiment | Focus | Key Architectural Finding | Empirical Plot |
|---|---|---|---|
| **Exp 1: Cache Capacity** | 1 KB – 32 KB L1 Capacity Sweep | Identified working-set knee at 8 KB; hit rate rose from 0% (capacity thrashing) to 75%, driving a $3.7\times$ IPC increase. | `results/exp1_cache_capacity.png` |
| **Exp 2: Cache Associativity** | 1-way (Direct Mapped) to 16-way | Direct-mapped cache experienced severe conflict misses (0% hit rate); 4-way associativity captured 75% hit rate, fully resolving conflict thrashing. | `results/exp2_associativity.png` |
| **Exp 3: Warp Concurrency** | 1 – 64 Warps (Little's Law) | Single warp achieved $\text{IPC} = 0.0074$ ($99.3\%$ memory stall). Increasing concurrency to 64 warps scaled IPC to $0.4552$ ($61.5\times$ scaling), proving Little's Law ($N = L / R$). | `results/exp3_warp_latency_hiding.png` |
| **Exp 4: Memory Coalescing** | Stride 1 – 32 Words (Divergence) | Unit stride achieved 4 transactions/instruction ($100\%$ efficiency). Strides $\ge 8$ caused maximum divergence ($8.0\times$ bandwidth inflation, 32 txns/inst, $12.5\%$ efficiency). | `results/exp4_coalescing.png` |
| **Exp 5: Scheduling Policy** | Round-Robin vs. GTO | GTO outperformed Round-Robin in matrix multiplication ($0.926$ vs. $0.901$ IPC) by desynchronizing warp memory phases and reducing peak bus queuing. | `results/exp5_scheduling_rr_vs_gto.png` |
| **Exp 6: Replacement Policy** | LRU vs. FIFO | LRU outperformed FIFO on repeated temporal access streams by protecting active loop working sets from premature eviction. | `results/exp6_replacement_lru_vs_fifo.png` |
| **Exp 7: Latency vs. Bandwidth** | Latency (50–400c) & BW (8–128 B/c) | Identified memory wall: beyond 32 B/cycle, performance plateaus as execution shifts from bandwidth-limited to latency-bound. | `results/exp7_latency_vs_bandwidth.png` |

---

## Performance & Optimization Profile

Profiling measurements conducted on Release builds ($10^5+$ instructions) before and after microarchitectural simulator optimizations:

| Workload | Baseline Time | Optimized Time | Speedup | Simulation Throughput | Cycle Sim Rate |
|---|---|---|---|---|---|
| **Sequential Coalesced (80K inst)** | 350.42 ms | **233.86 ms** | **1.50x** (+49.8%) | **342.08 KIPS** | **4.13 MCycles/s** |
| **Random Memory-Bound (96K inst, 2.4M cyc)** | 1,824.96 ms | **1,147.48 ms** | **1.59x** (+59.0%) | **83.66 KIPS** | **2.08 MCycles/s** |
| **Matmul GTO Contention (166K inst)** | 467.56 ms | **376.07 ms** | **1.24x** (+24.3%) | **441.63 KIPS** | **0.44 MCycles/s** |

### Key Optimizations:
1. **Direct Warp Indexing**: Replaced hash map (`std::unordered_map`) in `TraceReader` with direct $O(1)$ vector indexing, eliminating 76+ million hash lookups.
2. **Zero-Allocation Coalescing**: Replaced `std::set` and `std::map` node allocations with fixed stack vectors and linear scanning over 32 lanes, eliminating heap churn on every memory instruction.
3. **Completion Buffer Reuse**: Added early-exit checks and persistent vector reuse for memory completion events, eliminating millions of vector heap allocations per second.

---

## Validation & Testing

The simulator includes a comprehensive unit testing and analytical validation suite built on **Catch2 (v2.13.10)**:

```bash
# Direct test execution
./build/Release/gpu_simulator_tests.exe
```

Output:
```text
===============================================================================
All tests passed (211 assertions in 28 test cases)
```

### Key Verification Corner Cases:
- **Pure ALU IPC Proof**: Hand-calculated trace confirms $\text{IPC} = 1.0000$ and 0 stall cycles.
- **Single-Warp Memory Stall**: Verifies exact wake-up cycle matching analytical DRAM + L1/L2 access time ($T_{\text{stall}} = 28 \text{ cycles}$).
- **Direct-Mapped Conflict Thrashing**: Provable 0% hit rate on index aliasing addresses.
- **FIFO Replacement Invariance**: Verifies re-access does not reset eviction order in FIFO.
- **Little's Law Validation**: Confirms linear throughput scaling with active warp concurrency.

---

## Quick Start & Build Instructions

### Prerequisites
- C++17 compatible compiler (MSVC 2019+, GCC 9+, or Clang 10+)
- CMake 3.15+
- Python 3.8+ (with `numpy`, `matplotlib`, and `pandas` for experiments)

### Build
```bash
# Clone the repository
git clone https://github.com/your-username/gpu-memory-hierarchy-simulator.git
cd gpu-memory-hierarchy-simulator

# Configure and compile
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### Run Simulator
```bash
# Run simulator with default configuration
./build/Release/gpu_simulator.exe --config configs/default.json --trace traces/sequential.trace --stats results/run_stats.json
```

### Run Experiments & Generate Plots
```bash
# Generate synthetic benchmark traces
python scripts/generate_traces.py --pattern sequential --warps 16 --instructions 500 --output traces/sequential.trace

# Run all 7 architectural parameter sweeps
python scripts/run_experiments.py

# Generate publication-grade figures
python scripts/plot_results.py
```

---

## Repository Structure

```
├── configs/                  # Microarchitectural configuration profiles
│   ├── default.json          # Baseline SM: 32 warps, 32KB L1, 256KB L2, RR
│   ├── gto.json              # Greedy-Then-Oldest configuration
│   └── no_l1.json            # Bypassed L1 configuration
├── docs/                     # Technical documentation
│   ├── architecture.md       # Microarchitecture and SIMT timing specification
│   └── validation_methodology.md # Mathematical proofs and test suite design
├── include/                  # Simulator C++ headers
│   ├── cache.hpp             # Set-associative sectored cache model
│   ├── coalescer.hpp         # SIMT 32-lane memory coalescer
│   ├── config.hpp            # JSON configuration parser & validator
│   ├── dram_model.hpp        # DRAM latency & bandwidth queuing model
│   ├── instruction.hpp       # Instruction definitions & active masks
│   ├── memory_hierarchy.hpp  # L1/L2/DRAM memory hierarchy coordinator
│   ├── scheduler.hpp         # Round-Robin & GTO warp schedulers
│   ├── simulator.hpp         # Cycle-stepped execution engine
│   ├── trace_reader.hpp      # Streaming trace parser & warp queue manager
│   ├── types.hpp             # Architectural types, enums, & bitmasks
│   └── nlohmann/json.hpp     # JSON parser
├── results/                  # Empirical experiment outputs (JSON and PNG)
├── scripts/                  # Python experimental framework
│   ├── generate_traces.py    # Synthetic GPU trace generator
│   ├── run_experiments.py    # Automated parameter sweep framework
│   ├── plot_results.py       # Publication-ready matplotlib plotting suite
│   └── profile_simulator.py  # Performance and throughput benchmark harness
├── src/                      # C++ implementation files
├── tests/                    # Catch2 unit and validation test suite
└── CMakeLists.txt            # CMake build configuration
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
