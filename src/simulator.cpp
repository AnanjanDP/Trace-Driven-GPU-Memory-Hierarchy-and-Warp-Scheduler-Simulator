#include "simulator.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

namespace gpu_sim {

Simulator::Simulator(const SimConfig& config)
    : config_(config),
      coalescer_(config.sector_size_bytes, config.l1_line_size_bytes),
      scheduler_(config.scheduling_policy, config.issue_width),
      memory_hierarchy_(config) {

    if (!config_.trace_file.empty()) {
        try {
            load_trace(config_.trace_file);
        } catch (const std::exception& e) {
            std::cerr << "[WARN] Could not automatically load trace: " << e.what() << "\n";
        }
    } else {
        init_warps();
    }
}

void Simulator::load_trace(const std::string& filepath) {
    trace_reader_.load_trace(filepath);
    init_warps();
}

void Simulator::init_warps() {
    warps_.clear();
    uint32_t count = config_.warp_count;

    // If trace has warps, ensure we allocate enough slots
    for (WarpId wid : trace_reader_.active_warps()) {
        if (wid >= count) {
            count = wid + 1;
        }
    }

    warps_.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        warps_[i] = Warp(i);
    }
}

bool Simulator::step() {
    current_cycle_++;

    // 1. Process memory completion events arriving at or before current_cycle
    memory_hierarchy_.step_completions(current_cycle_, completed_events_buffer_);
    for (const auto& ev : completed_events_buffer_) {
        if (ev.warp_id < warps_.size()) {
            warps_[ev.warp_id].notify_memory_completion();
        }
    }

    // 2. Accumulate stall statistics for warps waiting on memory
    for (auto& w : warps_) {
        if (w.state == WarpState::WAITING_ON_MEMORY) {
            w.stall_cycles++;
            w.memory_stall_cycles++;
            total_memory_stall_cycles_++;
        }
    }

    // 3. Scheduler selects eligible warps for issue
    std::vector<WarpId> selected_warps = scheduler_.select_warps(warps_, trace_reader_, current_cycle_);

    // 4. Issue instructions for selected warps
    for (WarpId wid : selected_warps) {
        Warp& warp = warps_[wid];

        if (!trace_reader_.has_instruction(wid)) {
            warp.set_completed();
            continue;
        }

        Instruction inst = trace_reader_.pop_next_instruction(wid);
        warp.instructions_issued++;
        total_instructions_issued_++;
        warp.last_issued_cycle = current_cycle_;

        if (inst.is_memory()) {
            total_memory_instructions_++;
            CoalescingResult coal_res = coalescer_.coalesce(inst);

            total_coalesced_transactions_ += coal_res.transactions.size();
            total_requested_bytes_ += coal_res.total_requested_bytes;
            total_transferred_bytes_ += coal_res.total_transferred_bytes;

            warp.set_waiting_on_memory(static_cast<uint32_t>(coal_res.transactions.size()));

            for (const auto& txn : coal_res.transactions) {
                memory_hierarchy_.access(txn, current_cycle_);
            }
        } else {
            total_alu_instructions_++;
            // If warp has no instructions remaining after this ALU op, mark completed
            if (!trace_reader_.has_instruction(wid) && warp.pending_memory_ops == 0) {
                warp.set_completed();
            }
        }
    }

    // 5. Check if simulation is complete
    if (is_simulation_complete()) {
        for (auto& w : warps_) {
            w.set_completed();
        }
        return false;
    }

    return true;
}

bool Simulator::is_simulation_complete() const {
    if (memory_hierarchy_.has_in_flight_events()) {
        return false;
    }

    for (const auto& w : warps_) {
        if (w.pending_memory_ops > 0) {
            return false;
        }
        if (trace_reader_.has_instruction(w.id)) {
            return false;
        }
    }

    return true;
}

uint64_t Simulator::run(uint64_t max_cycles) {
    while (step()) {
        if (current_cycle_ >= max_cycles) {
            std::cerr << "[WARN] Reached maximum cycle limit: " << max_cycles << "\n";
            break;
        }
    }
    return current_cycle_;
}

nlohmann::json Simulator::get_statistics_json() const {
    nlohmann::json j;
    j["total_simulated_cycles"] = current_cycle_;
    j["total_instructions_issued"] = total_instructions_issued_;
    j["ipc"] = ipc();
    j["alu_instructions"] = total_alu_instructions_;
    j["memory_instructions"] = total_memory_instructions_;
    j["total_memory_stall_cycles"] = total_memory_stall_cycles_;
    j["total_coalesced_transactions"] = total_coalesced_transactions_;
    j["transactions_per_mem_inst"] = (total_memory_instructions_ == 0) ? 0.0 :
        static_cast<double>(total_coalesced_transactions_) / static_cast<double>(total_memory_instructions_);

    double coal_eff = (total_transferred_bytes_ == 0) ? 0.0 :
        static_cast<double>(total_requested_bytes_) / static_cast<double>(total_transferred_bytes_);
    j["coalescing_efficiency"] = coal_eff;
    j["average_memory_latency_cycles"] = memory_hierarchy_.average_memory_latency();

    j["stall_breakdown"] = {
        {"warp_memory_stall_cycles", total_memory_stall_cycles_},
        {"scheduler_idle_cycles", scheduler_.stats().cycles_idle}
    };

    if (memory_hierarchy_.l1_cache()) {
        j["l1_cache"] = memory_hierarchy_.l1_cache()->stats().to_json();
    }
    if (memory_hierarchy_.l2_cache()) {
        j["l2_cache"] = memory_hierarchy_.l2_cache()->stats().to_json();
    }

    j["dram"] = memory_hierarchy_.dram().stats().to_json(current_cycle_);
    j["scheduler"] = scheduler_.stats().to_json();

    nlohmann::json warp_arr = nlohmann::json::array();
    for (const auto& w : warps_) {
        warp_arr.push_back({
            {"warp_id", w.id},
            {"instructions_issued", w.instructions_issued},
            {"stall_cycles", w.stall_cycles},
            {"memory_stall_cycles", w.memory_stall_cycles}
        });
    }
    j["warps"] = warp_arr;

    return j;
}

void Simulator::dump_statistics_to_file(const std::string& filepath) const {
    std::ofstream out(filepath);
    if (!out.is_open()) {
        throw std::runtime_error("Simulator: Could not open output file: " + filepath);
    }
    out << get_statistics_json().dump(4) << std::endl;
}

} // namespace gpu_sim
