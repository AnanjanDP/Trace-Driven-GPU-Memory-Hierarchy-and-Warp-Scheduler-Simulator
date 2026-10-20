#pragma once

#include "types.hpp"
#include "config.hpp"
#include "warp.hpp"
#include "trace_reader.hpp"
#include "coalescer.hpp"
#include "scheduler.hpp"
#include "memory_hierarchy.hpp"
#include "nlohmann/json.hpp"
#include <vector>
#include <string>

namespace gpu_sim {

class Simulator {
public:
    explicit Simulator(const SimConfig& config);

    // Loads a trace from file
    void load_trace(const std::string& filepath);

    // Executes a single cycle. Returns false if simulation has finished.
    bool step();

    // Runs simulation until all warps complete and in-flight memory requests finish
    uint64_t run(uint64_t max_cycles = 10000000ULL);

    // Results & Statistics
    Cycle current_cycle() const { return current_cycle_; }
    uint64_t total_instructions_issued() const { return total_instructions_issued_; }
    double ipc() const {
        return current_cycle_ == 0 ? 0.0 : static_cast<double>(total_instructions_issued_) / static_cast<double>(current_cycle_);
    }

    nlohmann::json get_statistics_json() const;
    void dump_statistics_to_file(const std::string& filepath) const;

    // Component accessors for testing
    const std::vector<Warp>& warps() const { return warps_; }
    std::vector<Warp>& warps() { return warps_; }
    const WarpScheduler& scheduler() const { return scheduler_; }
    const MemoryHierarchy& memory() const { return memory_hierarchy_; }
    MemoryHierarchy& memory() { return memory_hierarchy_; }
    const TraceReader& trace_reader() const { return trace_reader_; }
    TraceReader& trace_reader() { return trace_reader_; }

private:
    SimConfig config_;
    std::vector<Warp> warps_;
    TraceReader trace_reader_;
    Coalescer coalescer_;
    WarpScheduler scheduler_;
    MemoryHierarchy memory_hierarchy_;

    Cycle current_cycle_{0};
    uint64_t total_instructions_issued_{0};
    uint64_t total_alu_instructions_{0};
    uint64_t total_memory_instructions_{0};
    uint64_t total_coalesced_transactions_{0};
    uint64_t total_requested_bytes_{0};
    uint64_t total_transferred_bytes_{0};
    uint64_t total_memory_stall_cycles_{0};
    std::vector<MemoryCompletionEvent> completed_events_buffer_;

    void init_warps();
    bool is_simulation_complete() const;
};

} // namespace gpu_sim
