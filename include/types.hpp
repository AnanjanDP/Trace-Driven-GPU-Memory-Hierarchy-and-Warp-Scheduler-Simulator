#pragma once

#include <cstdint>
#include <string>
#include <stdexcept>

namespace gpu_sim {

using Address = uint64_t;
using Cycle = uint64_t;
using WarpId = uint32_t;
using LaneId = uint32_t;
using ThreadMask = uint32_t;

constexpr uint32_t WARP_SIZE = 32;
constexpr ThreadMask FULL_WARP_MASK = 0xFFFFFFFFU;

enum class InstructionType {
    ALU,
    LOAD,
    STORE,
    BRANCH,
    BARRIER
};

inline std::string to_string(InstructionType type) {
    switch (type) {
        case InstructionType::ALU: return "ALU";
        case InstructionType::LOAD: return "LOAD";
        case InstructionType::STORE: return "STORE";
        case InstructionType::BRANCH: return "BRANCH";
        case InstructionType::BARRIER: return "BARRIER";
        default: return "UNKNOWN";
    }
}

inline InstructionType instruction_type_from_string(const std::string& s) {
    if (s == "ALU") return InstructionType::ALU;
    if (s == "LOAD") return InstructionType::LOAD;
    if (s == "STORE") return InstructionType::STORE;
    if (s == "BRANCH") return InstructionType::BRANCH;
    if (s == "BARRIER") return InstructionType::BARRIER;
    throw std::invalid_argument("Unknown InstructionType: " + s);
}

enum class WarpState {
    READY,
    WAITING_ON_MEMORY,
    COMPLETED
};

inline std::string to_string(WarpState state) {
    switch (state) {
        case WarpState::READY: return "READY";
        case WarpState::WAITING_ON_MEMORY: return "WAITING_ON_MEMORY";
        case WarpState::COMPLETED: return "COMPLETED";
        default: return "UNKNOWN";
    }
}

enum class SchedulingPolicy {
    ROUND_ROBIN,
    GTO  // Greedy-Then-Oldest
};

inline std::string to_string(SchedulingPolicy policy) {
    switch (policy) {
        case SchedulingPolicy::ROUND_ROBIN: return "ROUND_ROBIN";
        case SchedulingPolicy::GTO: return "GTO";
        default: return "UNKNOWN";
    }
}

inline SchedulingPolicy scheduling_policy_from_string(const std::string& s) {
    if (s == "ROUND_ROBIN") return SchedulingPolicy::ROUND_ROBIN;
    if (s == "GTO") return SchedulingPolicy::GTO;
    throw std::invalid_argument("Unknown SchedulingPolicy: " + s);
}

enum class ReplacementPolicy {
    LRU,
    FIFO
};

inline std::string to_string(ReplacementPolicy policy) {
    switch (policy) {
        case ReplacementPolicy::LRU: return "LRU";
        case ReplacementPolicy::FIFO: return "FIFO";
        default: return "UNKNOWN";
    }
}

inline ReplacementPolicy replacement_policy_from_string(const std::string& s) {
    if (s == "LRU") return ReplacementPolicy::LRU;
    if (s == "FIFO") return ReplacementPolicy::FIFO;
    throw std::invalid_argument("Unknown ReplacementPolicy: " + s);
}

} // namespace gpu_sim
