#pragma once

#include "types.hpp"
#include "instruction.hpp"
#include <vector>
#include <cstdint>

namespace gpu_sim {

struct MemoryTransaction {
    Address sector_addr{0};      // Base address of the 32B sector
    Address line_addr{0};        // Base address of the cache line containing the sector
    uint32_t size{32};           // Sector size (typically 32 bytes)
    bool is_write{false};
    WarpId warp_id{0};
    uint64_t pc{0};
    ThreadMask contributing_mask{0}; // Bitmask of lanes that access this sector
};

struct CoalescingResult {
    std::vector<MemoryTransaction> transactions;
    uint32_t unique_addresses_count{0};
    uint32_t unique_sectors_count{0};
    uint32_t unique_lines_count{0};
    uint32_t active_threads_count{0};
    uint64_t total_requested_bytes{0};    // active_threads * access_size
    uint64_t total_transferred_bytes{0};  // unique_sectors * sector_size

    double coalescing_efficiency() const {
        if (total_transferred_bytes == 0) return 0.0;
        return static_cast<double>(total_requested_bytes) / static_cast<double>(total_transferred_bytes);
    }
};

class Coalescer {
public:
    Coalescer(uint32_t sector_size_bytes = 32, uint32_t line_size_bytes = 128);

    CoalescingResult coalesce(const Instruction& inst) const;

    uint32_t sector_size_bytes() const { return sector_size_bytes_; }
    uint32_t line_size_bytes() const { return line_size_bytes_; }

private:
    uint32_t sector_size_bytes_;
    uint32_t line_size_bytes_;
    Address sector_mask_;
    Address line_mask_;
};

} // namespace gpu_sim
