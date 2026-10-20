#pragma once

#include "types.hpp"
#include "nlohmann/json.hpp"
#include <cstdint>
#include <algorithm>

namespace gpu_sim {

struct DRAMStats {
    uint64_t total_requests{0};
    uint64_t total_bytes_transferred{0};
    uint64_t total_busy_cycles{0};

    double bandwidth_utilization(Cycle total_simulated_cycles) const {
        return total_simulated_cycles == 0 ? 0.0 : static_cast<double>(total_busy_cycles) / static_cast<double>(total_simulated_cycles);
    }

    nlohmann::json to_json(Cycle total_simulated_cycles) const {
        return {
            {"total_requests", total_requests},
            {"total_bytes_transferred", total_bytes_transferred},
            {"total_busy_cycles", total_busy_cycles},
            {"bandwidth_utilization", bandwidth_utilization(total_simulated_cycles)}
        };
    }
};

class DRAMModel {
public:
    DRAMModel(Cycle fixed_latency = 200,
              double bandwidth_bytes_per_cycle = 32.0,
              uint32_t max_queue_size = 64);

    // Schedules a DRAM request arriving at current_cycle with given size in bytes
    // Returns the cycle when the DRAM access completes
    Cycle schedule_request(Address addr, uint32_t size_bytes, Cycle current_cycle);

    // Accessors
    Cycle fixed_latency() const { return fixed_latency_; }
    double bandwidth_bytes_per_cycle() const { return bandwidth_bytes_per_cycle_; }
    uint32_t max_queue_size() const { return max_queue_size_; }
    const DRAMStats& stats() const { return stats_; }
    void reset_stats() { stats_ = DRAMStats{}; bus_available_cycle_ = 0; }

private:
    Cycle fixed_latency_;
    double bandwidth_bytes_per_cycle_;
    uint32_t max_queue_size_;

    Cycle bus_available_cycle_{0};
    DRAMStats stats_;
};

} // namespace gpu_sim
