#pragma once

#include "types.hpp"
#include "instruction.hpp"
#include <string>
#include <vector>
#include <deque>
#include <unordered_map>

namespace gpu_sim {

class TraceReader {
public:
    TraceReader() = default;
    explicit TraceReader(const std::string& filepath);

    void load_trace(const std::string& filepath);

    bool has_instruction(WarpId warp_id) const;
    Instruction pop_next_instruction(WarpId warp_id);
    const Instruction& peek_next_instruction(WarpId warp_id) const;

    size_t total_instructions_loaded() const { return total_instructions_loaded_; }
    size_t remaining_instructions(WarpId warp_id) const;
    bool has_any_remaining_instructions() const;

    const std::vector<WarpId>& active_warps() const { return active_warps_; }

private:
    std::vector<std::deque<Instruction>> warp_queues_;
    std::vector<WarpId> active_warps_;
    size_t total_instructions_loaded_{0};

    void parse_line(const std::string& line, size_t line_number);
};

} // namespace gpu_sim
