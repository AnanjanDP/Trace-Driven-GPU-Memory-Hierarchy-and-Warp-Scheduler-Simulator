#pragma once

#include "types.hpp"
#include "nlohmann/json.hpp"
#include <string>
#include <cstdint>

namespace gpu_sim {

struct SimConfig {
    // SM Architecture
    uint32_t warp_count{32};
    uint32_t scheduler_count{1};
    uint32_t issue_width{1};
    SchedulingPolicy scheduling_policy{SchedulingPolicy::ROUND_ROBIN};

    // Coalescing
    uint32_t sector_size_bytes{32};

    // L1 Data Cache
    bool l1_enabled{true};
    uint64_t l1_capacity_bytes{32768};      // 32 KB
    uint32_t l1_line_size_bytes{128};       // 128 B
    uint32_t l1_associativity{4};           // 4-way set associative
    Cycle l1_hit_latency{4};                // 4 cycles
    ReplacementPolicy l1_replacement_policy{ReplacementPolicy::LRU};

    // L2 Data Cache
    uint64_t l2_capacity_bytes{262144};     // 256 KB
    uint32_t l2_line_size_bytes{128};       // 128 B
    uint32_t l2_associativity{16};          // 16-way set associative
    Cycle l2_hit_latency{32};               // 32 cycles
    ReplacementPolicy l2_replacement_policy{ReplacementPolicy::LRU};

    // DRAM Model
    Cycle dram_latency{200};                // 200 cycles fixed access latency
    double dram_bandwidth_bytes_per_cycle{32.0}; // e.g. 32 B/cycle
    uint32_t dram_max_queue_size{64};       // Maximum outstanding DRAM requests

    // Trace & Stats
    std::string trace_file{""};
    std::string output_stats_file{"stats.json"};

    // Helpers
    uint64_t l1_num_sets() const;
    uint64_t l2_num_sets() const;

    // Validation
    void validate() const;

    // Serialization
    static SimConfig from_json(const nlohmann::json& j);
    nlohmann::json to_json() const;
    static SimConfig from_file(const std::string& filepath);
    void to_file(const std::string& filepath) const;
};

} // namespace gpu_sim
