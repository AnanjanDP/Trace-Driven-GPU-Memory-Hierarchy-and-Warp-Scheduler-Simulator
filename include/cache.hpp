#pragma once

#include "types.hpp"
#include "nlohmann/json.hpp"
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>

namespace gpu_sim {

struct CacheLine {
    bool valid{false};
    bool dirty{false};
    Address tag{0};
    uint32_t sector_mask{0};         // Bitmask for valid sectors (e.g., 4 sectors in 128B line)
    uint64_t last_access_time{0};    // Timestamp for LRU
    uint64_t insertion_time{0};      // Timestamp for FIFO
};

struct CacheSet {
    std::vector<CacheLine> lines;
};

struct CacheAccessResult {
    bool hit{false};
    bool sector_hit{false};          // True if line hit and sector was already present
    bool evicted{false};
    Address evicted_addr{0};
    bool evicted_dirty{false};
    uint32_t set_index{0};
    uint32_t way_index{0};
};

struct CacheStats {
    uint64_t accesses{0};
    uint64_t reads{0};
    uint64_t writes{0};
    uint64_t read_hits{0};
    uint64_t read_misses{0};
    uint64_t write_hits{0};
    uint64_t write_misses{0};
    uint64_t evictions{0};
    uint64_t writebacks{0};

    uint64_t total_hits() const { return read_hits + write_hits; }
    uint64_t total_misses() const { return read_misses + write_misses; }

    double hit_rate() const {
        return accesses == 0 ? 0.0 : static_cast<double>(total_hits()) / static_cast<double>(accesses);
    }

    double miss_rate() const {
        return accesses == 0 ? 0.0 : static_cast<double>(total_misses()) / static_cast<double>(accesses);
    }

    void reset() {
        *this = CacheStats{};
    }

    nlohmann::json to_json() const {
        return {
            {"accesses", accesses},
            {"reads", reads},
            {"writes", writes},
            {"total_hits", total_hits()},
            {"total_misses", total_misses()},
            {"read_hits", read_hits},
            {"read_misses", read_misses},
            {"write_hits", write_hits},
            {"write_misses", write_misses},
            {"evictions", evictions},
            {"writebacks", writebacks},
            {"hit_rate", hit_rate()},
            {"miss_rate", miss_rate()}
        };
    }
};

class Cache {
public:
    Cache(const std::string& name,
          uint64_t capacity_bytes,
          uint32_t line_size_bytes,
          uint32_t associativity,
          uint32_t sector_size_bytes,
          ReplacementPolicy replacement_policy,
          Cycle hit_latency);

    // Primary interface: synchronous access (probe + allocate on miss)
    CacheAccessResult access(Address addr, bool is_write);

    // Decoupled interface for multi-cycle / non-blocking memory modeling
    bool probe(Address addr, bool is_write = false);
    CacheAccessResult fill(Address addr, bool is_write);

    // Address decomposition helpers
    Address get_tag(Address addr) const;
    uint32_t get_set_index(Address addr) const;
    uint32_t get_block_offset(Address addr) const;
    uint32_t get_sector_index(Address addr) const;
    Address get_line_address(Address addr) const;

    // Flush & Reset
    void flush();
    void reset_stats() { stats_.reset(); }

    // Getters
    const std::string& name() const { return name_; }
    uint64_t capacity_bytes() const { return capacity_bytes_; }
    uint32_t line_size_bytes() const { return line_size_bytes_; }
    uint32_t associativity() const { return associativity_; }
    uint32_t sector_size_bytes() const { return sector_size_bytes_; }
    uint32_t sectors_per_line() const { return sectors_per_line_; }
    uint64_t num_sets() const { return num_sets_; }
    ReplacementPolicy replacement_policy() const { return replacement_policy_; }
    Cycle hit_latency() const { return hit_latency_; }
    const CacheStats& stats() const { return stats_; }

private:
    std::string name_;
    uint64_t capacity_bytes_;
    uint32_t line_size_bytes_;
    uint32_t associativity_;
    uint32_t sector_size_bytes_;
    uint32_t sectors_per_line_;
    uint64_t num_sets_;
    ReplacementPolicy replacement_policy_;
    Cycle hit_latency_;

    uint32_t offset_bits_{0};
    uint32_t index_bits_{0};
    Address offset_mask_{0};
    Address index_mask_{0};

    std::vector<CacheSet> sets_;
    CacheStats stats_;
    uint64_t access_counter_{0};

    uint32_t find_victim_way(uint32_t set_idx) const;
};

} // namespace gpu_sim
