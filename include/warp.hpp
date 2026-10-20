#pragma once

#include "types.hpp"
#include "instruction.hpp"
#include <vector>

namespace gpu_sim {

struct Warp {
    WarpId id{0};
    WarpState state{WarpState::READY};
    uint64_t current_trace_idx{0};
    Cycle last_issued_cycle{0};
    Cycle ready_cycle{0};
    uint32_t pending_memory_ops{0};

    // Tracking metrics
    uint64_t instructions_issued{0};
    uint64_t stall_cycles{0};
    uint64_t memory_stall_cycles{0};

    Warp() = default;
    explicit Warp(WarpId warp_id) : id(warp_id) {}

    bool is_eligible(Cycle current_cycle) const {
        return (state == WarpState::READY) && (current_cycle >= ready_cycle);
    }

    bool has_completed() const {
        return state == WarpState::COMPLETED;
    }

    void set_waiting_on_memory(uint32_t num_transactions) {
        pending_memory_ops += num_transactions;
        if (pending_memory_ops > 0) {
            state = WarpState::WAITING_ON_MEMORY;
        }
    }

    void notify_memory_completion() {
        if (pending_memory_ops > 0) {
            pending_memory_ops--;
        }
        if (pending_memory_ops == 0 && state == WarpState::WAITING_ON_MEMORY) {
            state = WarpState::READY;
        }
    }

    void set_completed() {
        state = WarpState::COMPLETED;
    }
};

} // namespace gpu_sim
