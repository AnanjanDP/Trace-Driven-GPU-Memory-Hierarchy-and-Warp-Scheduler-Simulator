#include "memory_hierarchy.hpp"

namespace gpu_sim {

MemoryHierarchy::MemoryHierarchy(const SimConfig& config)
    : config_(config),
      dram_model_(config.dram_latency, config.dram_bandwidth_bytes_per_cycle, config.dram_max_queue_size) {

    if (config_.l1_enabled) {
        l1_cache_ = std::make_unique<Cache>(
            "L1_Cache",
            config_.l1_capacity_bytes,
            config_.l1_line_size_bytes,
            config_.l1_associativity,
            config_.sector_size_bytes,
            config_.l1_replacement_policy,
            config_.l1_hit_latency
        );
    }

    l2_cache_ = std::make_unique<Cache>(
        "L2_Cache",
        config_.l2_capacity_bytes,
        config_.l2_line_size_bytes,
        config_.l2_associativity,
        config_.sector_size_bytes,
        config_.l2_replacement_policy,
        config_.l2_hit_latency
    );
}

void MemoryHierarchy::access(const MemoryTransaction& txn, Cycle current_cycle) {
    MemoryCompletionEvent event;
    event.warp_id = txn.warp_id;
    event.addr = txn.sector_addr;
    event.is_write = txn.is_write;
    event.issue_cycle = current_cycle;

    Cycle l1_lat = (l1_cache_ != nullptr) ? config_.l1_hit_latency : 0;

    // 1. Check L1 Cache
    if (l1_cache_) {
        bool l1_hit = l1_cache_->probe(txn.sector_addr, txn.is_write);
        l1_cache_->access(txn.sector_addr, txn.is_write);
        if (l1_hit) {
            event.completion_cycle = current_cycle + config_.l1_hit_latency;
            event.fills_l1 = false;
            event.fills_l2 = false;
            event_queue_.insert({event.completion_cycle, event});
            return;
        }
    }

    // 2. Check L2 Cache
    bool l2_hit = l2_cache_->probe(txn.sector_addr, txn.is_write);
    l2_cache_->access(txn.sector_addr, txn.is_write);
    if (l2_hit) {
        event.completion_cycle = current_cycle + l1_lat + config_.l2_hit_latency;
        event.fills_l1 = (l1_cache_ != nullptr);
        event.fills_l2 = false;
        event_queue_.insert({event.completion_cycle, event});
        return;
    }

    // 3. L2 Miss: Access DRAM
    Cycle dram_done = dram_model_.schedule_request(txn.sector_addr, txn.size, current_cycle);
    event.completion_cycle = dram_done + l1_lat + config_.l2_hit_latency;
    event.fills_l1 = (l1_cache_ != nullptr);
    event.fills_l2 = true;
    event_queue_.insert({event.completion_cycle, event});
}

void MemoryHierarchy::step_completions(Cycle current_cycle, std::vector<MemoryCompletionEvent>& out_completed) {
    out_completed.clear();
    if (event_queue_.empty() || event_queue_.begin()->first > current_cycle) {
        return;
    }

    auto it = event_queue_.begin();
    while (it != event_queue_.end() && it->first <= current_cycle) {
        const auto& event = it->second;

        // Cache fills when data returns from lower levels
        if (event.fills_l2 && l2_cache_) {
            l2_cache_->fill(event.addr, event.is_write);
        }
        if (event.fills_l1 && l1_cache_) {
            l1_cache_->fill(event.addr, event.is_write);
        }

        total_latency_cycles_ += (event.completion_cycle - event.issue_cycle);
        completed_requests_count_++;

        out_completed.push_back(event);
        it = event_queue_.erase(it);
    }
}

std::vector<MemoryCompletionEvent> MemoryHierarchy::step_completions(Cycle current_cycle) {
    std::vector<MemoryCompletionEvent> completed;
    step_completions(current_cycle, completed);
    return completed;
}

} // namespace gpu_sim
