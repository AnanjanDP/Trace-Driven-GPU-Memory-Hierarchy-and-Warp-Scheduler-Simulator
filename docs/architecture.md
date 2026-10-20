# GPU Microarchitecture & Simulator Design Specification

## 1. System Overview

This document details the microarchitectural timing model, SIMT execution semantics, memory coalescing logic, multi-level cache hierarchy, warp scheduling policies, and DRAM queuing model implemented in the **Trace-Driven GPU Memory Hierarchy and Warp Scheduler Simulator**.

The simulator models a Single Instruction, Multiple Threads (SIMT) streaming multiprocessor (SM / Compute Unit) executing memory-intensive and compute workloads driven by trace records.

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

## 2. Microarchitectural Timing Model

The simulator operates on a cycle-by-cycle stepped timing model. Each cycle executes the following ordered sequence:

1. **Memory Completion Event Processing (`step_completions`)**:
   - The event queue is inspected for events completing on or before `current_cycle`.
   - On completion, data fills into L2 (if missed at L2) and L1 (if missed at L1).
   - The associated warp receives a completion notification:
     $$\text{pending\_memory\_ops} \leftarrow \text{pending\_memory\_ops} - 1$$
   - If `pending_memory_ops == 0`, the warp transitions from `WAITING_ON_MEMORY` to `READY`.

2. **Stall Cycle Tracking**:
   - Every warp in state `WAITING_ON_MEMORY` increments its `stall_cycles` and `memory_stall_cycles`.
   - Aggregate memory stall counters are updated for Little's Law validation.

3. **Warp Scheduling (`select_warps`)**:
   - The scheduler queries all warps for eligibility.
   - Eligibility condition:
     $$\text{is\_eligible}(w, c) = (w.\text{state} == \text{READY}) \land (w.\text{ready\_cycle} \le c) \land \text{has\_instruction}(w.\text{id})$$
   - Schedulers select up to `issue_width` warps per cycle based on configured policy (Round-Robin or Greedy-Then-Oldest).

4. **Instruction Issue & Execution**:
   - Instructions are popped from the trace reader.
   - If **ALU**: 1-cycle latency; warp remains `READY` for subsequent cycles.
   - If **MEMORY**: Instruction is passed to the Coalescing Unit.
     - Coalesced into $N$ sector transactions ($N \in [1, 32]$).
     - Warp enters `WAITING_ON_MEMORY` with `pending_memory_ops = N`.
     - Transactions enter the memory hierarchy pipeline.

5. **Cycle Advancement & Termination**:
   - `current_cycle` increments by 1.
   - Simulation terminates when all warps have exhausted their instruction streams and all in-flight memory transactions have drained.

---

## 3. Memory Coalescing Architecture

Modern GPUs enforce memory coalescing to minimize memory traffic and maximize memory bus efficiency.

### 3.1 Addressing & Bit Alignment
- **Word / Access Size ($A$)**: 4 bytes (e.g., `float32`).
- **Sector Size ($S$)**: 32 bytes (NVIDIA Volta/Turing/Ampere/Hopper standard).
- **Cache Line Size ($L$)**: 128 bytes (composed of 4 sectors).

### 3.2 Coalescing Algorithm
For an active thread lane $i \in [0, 31]$ with thread address $\text{Addr}_i$:
$$\text{SectorAddr}_i = \text{Addr}_i \ \& \ \sim(S - 1)$$
$$\text{LineAddr}_i = \text{Addr}_i \ \& \ \sim(L - 1)$$

Lanes accessing the same $\text{SectorAddr}$ are merged into a single `MemoryTransaction`:
- `contributing_mask`: Bitmask of lanes participating in the transaction:
  $$\text{contributing\_mask} = \sum_{i \in \text{lanes}} 2^i$$
- `size`: Exactly 32 bytes (sector granularity).

### 3.3 Coalescing Metrics
- **Transactions Per Memory Instruction**:
  $$\text{Txn/Inst} = \frac{\text{Total Coalesced Transactions}}{\text{Total Memory Instructions}}$$
  Optimal coalescing yields 4.0 transactions for 128 contiguous bytes across 32 threads.
- **Coalescing Efficiency**:
  $$\eta_{\text{coalescing}} = \frac{\sum \text{Active Threads} \times \text{Access Size}}{\sum \text{Transactions} \times \text{Sector Size}} = \frac{32 \times 4}{4 \times 32} = 1.0 \ (100\%)$$
  For uncoalesced accesses (e.g., stride $\ge 8$ words), $\eta_{\text{coalescing}} = 12.5\%$ (32 transactions of 32B to transfer 128B of payload).

---

## 4. Multi-Level Cache Hierarchy

### 4.1 Address Decomposition
For an address with tag, set index, and block offset:
- Offset bits: $b = \log_2(\text{LineSize}) = \log_2(128) = 7$
- Index bits: $s = \log_2(\text{NumSets}) = \log_2\left(\frac{\text{Capacity}}{\text{LineSize} \times \text{Associativity}}\right)$
- Tag bits: $t = 64 - s - b$

```
+------------------------------------+----------------+---------------+
|             Tag Bits               |   Index Bits   |  Offset Bits  |
|            (64 - s - b)            |      (s)       |      (b)      |
+------------------------------------+----------------+---------------+
```

### 4.2 Sector Valid/Dirty Tracking
Each cache line maintains:
- `tag`: Physical tag of the 128B block.
- `valid`: Line-level valid bit.
- `sectors_valid[4]`: Independent valid bit per 32B sector.
- `sectors_dirty[4]`: Independent dirty bit per 32B sector.
- `last_access_time`: Updated on hit/fill for LRU.
- `insertion_time`: Set on allocation for FIFO.

A cache hit requires:
$$\text{Hit} = \text{line.valid} \land (\text{line.tag} == \text{Tag}) \land \text{line.sectors\_valid}[\text{sector\_idx}]$$

If the line is allocated but the specific sector is invalid, a **sector miss** occurs, initiating a sector fill from lower memory levels without evicting the surrounding line.

### 4.3 Replacement Policies
- **LRU (Least Recently Used)**: Evicts the way in the set with minimum `last_access_time`. Updated on both reads and writes.
- **FIFO (First-In, First-Out)**: Evicts the way in the set with minimum `insertion_time`. Access hits do not alter eviction priority.

---

## 5. Warp Scheduling Policies

### 5.1 Round-Robin (RR)
- Maintains a circular scheduling pointer: `last_rr_index_`.
- Evaluates warps sequentially starting from `(last_rr_index_ + 1) % num_warps`.
- Distributes issue slots evenly across all warps.
- **Microarchitectural Effect**: Warps progress lockstep, causing memory requests to cluster into burst intervals, maximizing instantaneous memory contention.

### 5.2 Greedy-Then-Oldest (GTO)
- **Greedy Phase**: If the warp issued in cycle $c-1$ remains eligible in cycle $c$, it retains scheduling priority and is re-issued immediately.
- **Oldest Phase**: When the current warp stalls on a dependency or memory miss, the scheduler selects the eligible warp with the minimum `last_issued_cycle` (oldest).
- **Tie-Breaking**: Smallest `WarpId` breaks ties deterministically.
- **Microarchitectural Effect**: Desynchronizes warp execution phases. While the leading warp is stalled waiting for memory, other warps advance through ALU and cache-friendly sections, smoothing memory bandwidth utilization and increasing IPC.

---

## 6. DRAM Controller & Bus Modeling

DRAM is modeled as a latency pipeline coupled with an analytical bandwidth serialization queue:

1. **Fixed Burst Latency**: $t_{\text{latency}} = 200 \text{ cycles}$ (CAS, RAS, tRCD, tCL, channel transit).
2. **Channel Serialization Delay**:
   $$\Delta t_{\text{transfer}} = \left\lceil \frac{\text{Transaction Size (32 B)}}{\text{Bandwidth (Bytes/Cycle)}} \right\rceil$$
3. **Queue Serialization Tracking**:
   The controller tracks `next_available_cycle_`:
   $$\text{Start Cycle} = \max(\text{Request Arrival Cycle}, \text{next\_available\_cycle\_})$$
   $$\text{next\_available\_cycle\_} \leftarrow \text{Start Cycle} + \Delta t_{\text{transfer}}$$
   $$\text{Completion Cycle} = \text{Start Cycle} + t_{\text{latency}} + \Delta t_{\text{transfer}}$$

This models queuing delays under heavy memory contention: if thousands of sector requests arrive simultaneously, they serialize on the DRAM bus, naturally inflating memory response latency.

---

## 7. What This Simulator Models vs. Approximates

| Architectural Feature | Modeled Realistically | Approximated / Abstracted | Justification |
|---|---|---|---|
| SIMT Thread Divergence & Masking | Yes (32-bit lane active mask) | Control divergence reconvergence stack | Focus is on memory hierarchy and scheduling |
| Memory Coalescing | Yes (Exact 32B sector / 128B line math) | Shared memory bank conflicts | Evaluates global memory pipeline behavior |
| Cache Tag & Index Matching | Yes (Exact bit masking and shifting) | MSHR entry count limit | Modeled via DRAM queue depth limit |
| Sector Cache Operations | Yes (4 sectors/line, per-sector valid/dirty) | Cross-SM cache coherency | Single SM focus with unified L2 cache |
| Warp Scheduling Policies | Yes (Deterministic RR & GTO with tie-breaks) | Register scoreboard dependency checks | Trace provides pre-scheduled dependency streams |
| DRAM System | Yes (Queuing delay + sustained bandwidth limits) | Row buffer hits/misses (tRP/tRAS/tRC) | Sufficient for architectural bandwidth saturation studies |
