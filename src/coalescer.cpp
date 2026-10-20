#include "coalescer.hpp"
#include <algorithm>
#include <stdexcept>

namespace gpu_sim {

namespace {

bool is_power_of_two(uint64_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

} // anonymous namespace

Coalescer::Coalescer(uint32_t sector_size_bytes, uint32_t line_size_bytes)
    : sector_size_bytes_(sector_size_bytes), line_size_bytes_(line_size_bytes) {

    if (!is_power_of_two(sector_size_bytes_)) {
        throw std::invalid_argument("Coalescer: sector_size_bytes must be a power of 2");
    }
    if (!is_power_of_two(line_size_bytes_)) {
        throw std::invalid_argument("Coalescer: line_size_bytes must be a power of 2");
    }
    if (line_size_bytes_ < sector_size_bytes_ || (line_size_bytes_ % sector_size_bytes_) != 0) {
        throw std::invalid_argument("Coalescer: line_size_bytes must be a multiple of sector_size_bytes");
    }

    sector_mask_ = ~(static_cast<Address>(sector_size_bytes_) - 1ULL);
    line_mask_ = ~(static_cast<Address>(line_size_bytes_) - 1ULL);
}

CoalescingResult Coalescer::coalesce(const Instruction& inst) const {
    CoalescingResult result;

    if (!inst.is_memory()) {
        return result;
    }

    bool is_write = (inst.type == InstructionType::STORE);
    std::vector<Address> unique_addresses;
    std::vector<Address> unique_lines;
    unique_addresses.reserve(WARP_SIZE);
    unique_lines.reserve(WARP_SIZE);
    result.transactions.reserve(WARP_SIZE);

    for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
        if (!inst.is_lane_active(lane)) {
            continue;
        }

        result.active_threads_count++;
        Address addr = inst.addresses[lane];
        if (std::find(unique_addresses.begin(), unique_addresses.end(), addr) == unique_addresses.end()) {
            unique_addresses.push_back(addr);
        }

        Address sector_addr = addr & sector_mask_;
        Address line_addr = addr & line_mask_;
        if (std::find(unique_lines.begin(), unique_lines.end(), line_addr) == unique_lines.end()) {
            unique_lines.push_back(line_addr);
        }

        auto it = std::find_if(result.transactions.begin(), result.transactions.end(),
            [sector_addr](const MemoryTransaction& t) { return t.sector_addr == sector_addr; });

        if (it == result.transactions.end()) {
            MemoryTransaction txn;
            txn.sector_addr = sector_addr;
            txn.line_addr = line_addr;
            txn.size = sector_size_bytes_;
            txn.is_write = is_write;
            txn.warp_id = inst.warp_id;
            txn.pc = inst.pc;
            txn.contributing_mask = (1U << lane);
            result.transactions.push_back(txn);
        } else {
            it->contributing_mask |= (1U << lane);
        }
    }

    // Maintain deterministic ascending order of sector addresses matching map semantics
    std::sort(result.transactions.begin(), result.transactions.end(),
        [](const MemoryTransaction& a, const MemoryTransaction& b) {
            return a.sector_addr < b.sector_addr;
        });

    result.unique_addresses_count = static_cast<uint32_t>(unique_addresses.size());
    result.unique_sectors_count = static_cast<uint32_t>(result.transactions.size());
    result.unique_lines_count = static_cast<uint32_t>(unique_lines.size());
    result.total_requested_bytes = static_cast<uint64_t>(result.active_threads_count) * inst.access_size;
    result.total_transferred_bytes = static_cast<uint64_t>(result.unique_sectors_count) * sector_size_bytes_;

    return result;
}

} // namespace gpu_sim
