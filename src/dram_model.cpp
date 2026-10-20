#include "dram_model.hpp"
#include <cmath>
#include <stdexcept>

namespace gpu_sim {

DRAMModel::DRAMModel(Cycle fixed_latency,
                     double bandwidth_bytes_per_cycle,
                     uint32_t max_queue_size)
    : fixed_latency_(fixed_latency),
      bandwidth_bytes_per_cycle_(bandwidth_bytes_per_cycle),
      max_queue_size_(max_queue_size) {

    if (fixed_latency_ == 0) {
        throw std::invalid_argument("DRAMModel: fixed_latency must be >= 1 cycle");
    }
    if (bandwidth_bytes_per_cycle_ <= 0.0) {
        throw std::invalid_argument("DRAMModel: bandwidth_bytes_per_cycle must be > 0.0");
    }
}

Cycle DRAMModel::schedule_request(Address /*addr*/, uint32_t size_bytes, Cycle current_cycle) {
    // 1. Calculate transfer duration based on sustained channel bandwidth
    Cycle transfer_cycles = static_cast<Cycle>(std::ceil(static_cast<double>(size_bytes) / bandwidth_bytes_per_cycle_));
    if (transfer_cycles == 0) {
        transfer_cycles = 1;
    }

    // 2. Bus serialization: transfer begins when bus becomes available or at current_cycle (whichever is later)
    Cycle transfer_start = std::max(current_cycle, bus_available_cycle_);
    Cycle transfer_end = transfer_start + transfer_cycles;
    bus_available_cycle_ = transfer_end;

    // 3. Total round-trip completion = transfer end + access propagation latency
    Cycle completion_cycle = transfer_end + fixed_latency_;

    // 4. Update statistics
    stats_.total_requests++;
    stats_.total_bytes_transferred += size_bytes;
    stats_.total_busy_cycles += transfer_cycles;

    return completion_cycle;
}

} // namespace gpu_sim
