#pragma once

#include "types.hpp"
#include "config.hpp"
#include "cache.hpp"
#include "dram_model.hpp"
#include "coalescer.hpp"
#include <memory>
#include <map>
#include <vector>

namespace gpu_sim {

struct MemoryCompletionEvent {
    Cycle completion_cycle{0};
    Cycle issue_cycle{0};
    WarpId warp_id{0};
    Address addr{0};
    bool is_write{false};
    bool fills_l1{false};
    bool fills_l2{false};
};

class MemoryHierarchy {
public:
    explicit MemoryHierarchy(const SimConfig& config);

    // Processes a coalesced sector transaction arriving at current_cycle
    void access(const MemoryTransaction& txn, Cycle current_cycle);

    // Pops all memory events completing on or before current_cycle, performs cache fills
    std::vector<MemoryCompletionEvent> step_completions(Cycle current_cycle);
    void step_completions(Cycle current_cycle, std::vector<MemoryCompletionEvent>& out_completed);

    // Status queries
    bool has_in_flight_events() const { return !event_queue_.empty(); }
    size_t in_flight_event_count() const { return event_queue_.size(); }

    double average_memory_latency() const {
        return completed_requests_count_ == 0 ? 0.0 : static_cast<double>(total_latency_cycles_) / static_cast<double>(completed_requests_count_);
    }
    uint64_t completed_requests_count() const { return completed_requests_count_; }

    // Cache & DRAM accessors
    Cache* l1_cache() { return l1_cache_.get(); }
    const Cache* l1_cache() const { return l1_cache_.get(); }
    Cache* l2_cache() { return l2_cache_.get(); }
    const Cache* l2_cache() const { return l2_cache_.get(); }
    DRAMModel& dram() { return dram_model_; }
    const DRAMModel& dram() const { return dram_model_; }

private:
    SimConfig config_;
    std::unique_ptr<Cache> l1_cache_;
    std::unique_ptr<Cache> l2_cache_;
    DRAMModel dram_model_;

    uint64_t total_latency_cycles_{0};
    uint64_t completed_requests_count_{0};

    // Multimap keyed by completion_cycle for efficient chronological event handling
    std::multimap<Cycle, MemoryCompletionEvent> event_queue_;
};

} // namespace gpu_sim
