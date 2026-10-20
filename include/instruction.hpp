#pragma once

#include "types.hpp"
#include <array>
#include <cstdint>

namespace gpu_sim {

struct Instruction {
    WarpId warp_id{0};
    uint64_t pc{0};
    InstructionType type{InstructionType::ALU};
    ThreadMask active_mask{FULL_WARP_MASK};
    std::array<Address, WARP_SIZE> addresses{};
    uint32_t access_size{4}; // Bytes per active lane (e.g. 4 for 32-bit word)

    bool is_memory() const {
        return type == InstructionType::LOAD || type == InstructionType::STORE;
    }

    bool is_lane_active(LaneId lane) const {
        if (lane >= WARP_SIZE) return false;
        return (active_mask & (1U << lane)) != 0;
    }

    uint32_t active_lane_count() const {
        uint32_t count = 0;
        for (uint32_t i = 0; i < WARP_SIZE; ++i) {
            if ((active_mask & (1U << i)) != 0) {
                count++;
            }
        }
        return count;
    }
};

} // namespace gpu_sim
