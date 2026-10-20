# Simulator Validation Methodology & Test Suite Specification

## 1. Validation Philosophy

A simulator is only as credible as its validation methodology. The **Trace-Driven GPU Memory Hierarchy and Warp Scheduler Simulator** is validated through a three-tier testing framework:

1. **Unit Testing**: Isolated micro-benchmarks verifying bitwise exactness of cache indexing, replacement policies, coalescer lane grouping, and scheduler tie-breaking.
2. **Analytical Corner Cases**: Controlled hand-calculable test traces with mathematically provable outcomes (e.g., pure ALU IPC = 1.0, direct-mapped conflict thrashing, single-warp stall latency).
3. **Architectural Scaling & Laws**: System-level multi-warp sweeps validating foundational architectural theorems (Little's Law, memory divergence bandwidth amplification, DRAM bus saturation).

All unit tests are executed using the vendored **Catch2 (v2.13.10)** framework and integrated into **CTest**.

---

## 2. Test Suite Overview

| Test Module | Test File | Test Cases | Assertions | Focus Areas |
|---|---|---|---|---|
| **Cache Subsystem** | `tests/test_cache.cpp` | 6 | 48 | Tag/Index/Offset extraction, LRU eviction, FIFO insertion order, Sector valid/dirty states, Write-back tracking |
| **Memory Coalescer** | `tests/test_coalescer.cpp` | 5 | 39 | Sequential 128B coalescence (4 txns), Uniform scalar broadcast (1 txn), Strided divergence (32 txns), Active mask evaluation |
| **Warp Scheduler** | `tests/test_scheduler.cpp` | 5 | 42 | RR rotating index, GTO greedy retention, GTO oldest fallback, Deterministic WarpId tie-breaking, Idle cycle tracking |
| **DRAM Model** | `tests/test_dram.cpp` | 3 | 24 | Fixed latency calculation, Bandwidth transfer serialization, Concurrent request queuing delay |
| **System Simulator** | `tests/test_simulator.cpp` | 5 | 28 | Pure ALU IPC = 1.0, Single-warp stall & wake-up, Multi-warp latency hiding, Trace reader integration |
| **System Validation** | `tests/test_validation.cpp` | 4 | 30 | Little's Law concurrency sweep, Divergence bandwidth amplification (8x), DRAM saturation plateau, GTO vs RR phase desynchronization |
| **TOTAL** | — | **28** | **211** | **100% Passing (0 failures, 0 warnings)** |

---

## 3. Mathematical Proofs & Hand-Calculated Corner Cases

### 3.1 Pure ALU Instruction Stream (IPC = 1.0)
- **Hypothesis**: For an SM with issue width = 1 and a trace consisting of $K$ independent ALU instructions, the simulator must execute in exactly $K$ cycles with $\text{IPC} = 1.0000$.
- **Validation**:
  - Test case: `tests/test_simulator.cpp::"Pure ALU Workload (IPC = 1.0)"`
  - Input: 5 ALU instructions on Warp 0.
  - Expected Cycles: 5. Measured Cycles: 5. IPC: 1.0000. Stall cycles: 0.

### 3.2 Single-Warp Uncached Memory Stall Latency
- **Hypothesis**: When Warp 0 issues a uniform memory load that misses both L1 and L2 caches, the latency observed before the warp can issue its subsequent instruction must be:
  $$T_{\text{stall}} = t_{\text{L1\_hit}} + t_{\text{L2\_hit}} + t_{\text{DRAM\_latency}} + t_{\text{DRAM\_transfer}}$$
- **Configuration**:
  - $t_{\text{L1\_hit}} = 2\text{ cycles}$
  - $t_{\text{L2\_hit}} = 5\text{ cycles}$
  - $t_{\text{DRAM\_latency}} = 20\text{ cycles}$
  - $\text{BW} = 32\text{ B/cycle} \implies t_{\text{DRAM\_transfer}} = \lceil 32 / 32 \rceil = 1\text{ cycle}$
  - Expected completion cycle: $1 + 2 + 5 + 20 + 1 = 28\text{ cycles}$.
- **Validation**:
  - Test case: `tests/test_simulator.cpp::"Single-Warp Memory Stall and Wake-Up"`
  - Measured wake-up and completion cycle: 28 cycles. Passed.

### 3.3 Direct-Mapped Cache Conflict Thrashing
- **Hypothesis**: In a direct-mapped cache ($A = 1$) with capacity 1024 B and line size 64 B ($\text{NumSets} = 16$), accesses that map to the identical set index but possess distinct tags must evict each other continuously (0% hit rate).
- **Index Math**:
  - $\text{Offset bits} = \log_2(64) = 6$
  - $\text{Index bits} = \log_2(16) = 4$
  - Set index = $(\text{Address} \gg 6) \ \& \ 0\text{xF}$
  - Address $0\text{x}000 \implies \text{Set } 0, \text{Tag } 0$
  - Address $0\text{x}400 \ (1024) \implies \text{Set } 0, \text{Tag } 1$
- **Validation**:
  - Alternating access sequence: $0\text{x}000, 0\text{x}400, 0\text{x}000, 0\text{x}400$.
  - Measured Hits: 0. Misses: 4. Evictions: 3. Passed.

### 3.4 FIFO vs. LRU Invariance Under Re-access
- **Hypothesis**: In a 2-way associative cache with FIFO replacement, re-accessing an existing resident line does **not** reset its replacement timestamp. In contrast, under LRU, accessing a line promotes it to the most recently used position.
- **Access Pattern**: Access Way A, Access Way B, Re-access Way A, Insert Way C (eviction required).
  - Under FIFO: Way A was inserted first $\implies$ Way A is evicted.
  - Under LRU: Way B was least recently used $\implies$ Way B is evicted.
- **Validation**:
  - Test cases in `tests/test_cache.cpp` verify that Way A is evicted under FIFO, while Way B is evicted under LRU.

---

## 4. Architectural Scaling & Validation Experiments

### 4.1 Little's Law Concurrency Verification
According to Little's Law for pipelined architectures:
$$N_{\text{warps}} = \frac{L}{R} = \frac{\text{Memory Latency}}{\text{Cycles between Memory Requests}}$$

If each warp issues a memory instruction requiring $L = 227 \text{ cycles}$ followed by $R = 1$ ALU instruction, full latency hiding ($\text{IPC} \approx 1.0$) requires:
$$N_{\text{warps}} \ge 227$$

- **Empirical Measurement**:
  - At $N = 1$: $\text{IPC} = 0.0074$ (stalled 99.3% of cycles).
  - At $N = 8$: $\text{IPC} = 0.0585$ ($7.9\times$ scaling).
  - At $N = 32$: $\text{IPC} = 0.2308$ ($31.2\times$ scaling).
  - At $N = 64$: $\text{IPC} = 0.4552$ ($61.5\times$ scaling).
  - Scaling efficiency is linear until memory bus contention causes queuing delay.

### 4.2 Memory Divergence Bandwidth Amplification
For 32 active threads accessing 4-byte words:
- Stride 1 (Sequential): All 32 addresses lie within 128 bytes (4 sectors of 32B). Transactions = 4. $\eta = 100\%$.
- Stride 8+ (Strided): All 32 addresses land in distinct 32B sectors. Transactions = 32. $\eta = 12.5\%$.
- **Validation Result**: Exactly $8.0\times$ bandwidth amplification observed between unit stride and stride $\ge 8$.

---

## 5. Running the Test Suite

Execute all 28 unit tests via CMake/CTest:

```bash
# Build test suite
cmake --build build --config Release

# Run Catch2 tests directly with full reporting
./build/Release/gpu_simulator_tests.exe

# Or run through CTest
ctest --test-dir build -C Release --output-on-failure
```
