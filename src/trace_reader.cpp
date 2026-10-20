#include "trace_reader.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>

namespace gpu_sim {

TraceReader::TraceReader(const std::string& filepath) {
    load_trace(filepath);
}

void TraceReader::load_trace(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("TraceReader: Failed to open trace file: " + filepath);
    }

    warp_queues_.clear();
    active_warps_.clear();
    total_instructions_loaded_ = 0;

    std::string line;
    size_t line_number = 0;
    while (std::getline(file, line)) {
        line_number++;
        // Remove leading/trailing spaces
        size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') {
            continue; // Empty line or comment
        }
        parse_line(line.substr(first), line_number);
    }

    for (size_t wid = 0; wid < warp_queues_.size(); ++wid) {
        if (!warp_queues_[wid].empty()) {
            active_warps_.push_back(static_cast<WarpId>(wid));
        }
    }
}

void TraceReader::parse_line(const std::string& line, size_t line_number) {
    std::istringstream iss(line);
    std::string opcode;
    if (!(iss >> opcode)) return;

    Instruction inst;
    WarpId warp_id = 0;
    std::string pc_str;

    if (!(iss >> warp_id >> pc_str)) {
        throw std::runtime_error("Trace format error at line " + std::to_string(line_number) +
                                 ": Expected opcode, warp_id, and PC");
    }

    inst.warp_id = warp_id;
    inst.pc = std::stoull(pc_str, nullptr, 0);

    if (opcode == "ALU") {
        inst.type = InstructionType::ALU;
        inst.active_mask = FULL_WARP_MASK;
    } else if (opcode == "BRANCH") {
        inst.type = InstructionType::BRANCH;
        inst.active_mask = FULL_WARP_MASK;
    } else if (opcode == "BARRIER") {
        inst.type = InstructionType::BARRIER;
        inst.active_mask = FULL_WARP_MASK;
    } else if (opcode == "LOAD" || opcode == "STORE" ||
               opcode == "LOAD_STRIDED" || opcode == "STORE_STRIDED" ||
               opcode == "LOAD_UNIFORM" || opcode == "STORE_UNIFORM") {

        bool is_load = (opcode.find("LOAD") != std::string::npos);
        inst.type = is_load ? InstructionType::LOAD : InstructionType::STORE;

        std::string mask_str;
        uint32_t access_size = 4;
        if (!(iss >> mask_str >> access_size)) {
            throw std::runtime_error("Trace format error at line " + std::to_string(line_number) +
                                     ": Expected active_mask and access_size for memory op");
        }
        inst.active_mask = static_cast<ThreadMask>(std::stoul(mask_str, nullptr, 16));
        inst.access_size = access_size;

        if (opcode == "LOAD_STRIDED" || opcode == "STORE_STRIDED") {
            std::string base_str;
            int64_t stride = 0;
            if (!(iss >> base_str >> stride)) {
                throw std::runtime_error("Trace format error at line " + std::to_string(line_number) +
                                         ": Expected base_addr and stride for strided memory op");
            }
            Address base_addr = std::stoull(base_str, nullptr, 0);
            for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
                inst.addresses[lane] = static_cast<Address>(static_cast<int64_t>(base_addr) + static_cast<int64_t>(lane) * stride);
            }
        } else if (opcode == "LOAD_UNIFORM" || opcode == "STORE_UNIFORM") {
            std::string base_str;
            if (!(iss >> base_str)) {
                throw std::runtime_error("Trace format error at line " + std::to_string(line_number) +
                                         ": Expected base_addr for uniform memory op");
            }
            Address base_addr = std::stoull(base_str, nullptr, 0);
            for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
                inst.addresses[lane] = base_addr;
            }
        } else {
            // Full 32 per-lane addresses
            for (uint32_t lane = 0; lane < WARP_SIZE; ++lane) {
                std::string addr_str;
                if (!(iss >> addr_str)) {
                    throw std::runtime_error("Trace format error at line " + std::to_string(line_number) +
                                             ": Expected 32 per-lane addresses, found fewer");
                }
                inst.addresses[lane] = std::stoull(addr_str, nullptr, 0);
            }
        }
    } else {
        throw std::runtime_error("Trace format error at line " + std::to_string(line_number) +
                                 ": Unknown instruction opcode '" + opcode + "'");
    }

    if (inst.warp_id >= warp_queues_.size()) {
        warp_queues_.resize(inst.warp_id + 1);
    }
    warp_queues_[inst.warp_id].push_back(std::move(inst));
    total_instructions_loaded_++;
}

bool TraceReader::has_instruction(WarpId warp_id) const {
    if (warp_id >= warp_queues_.size()) return false;
    return !warp_queues_[warp_id].empty();
}

Instruction TraceReader::pop_next_instruction(WarpId warp_id) {
    if (warp_id >= warp_queues_.size() || warp_queues_[warp_id].empty()) {
        throw std::runtime_error("TraceReader: No instructions remaining for warp " + std::to_string(warp_id));
    }
    Instruction inst = std::move(warp_queues_[warp_id].front());
    warp_queues_[warp_id].pop_front();
    return inst;
}

const Instruction& TraceReader::peek_next_instruction(WarpId warp_id) const {
    if (warp_id >= warp_queues_.size() || warp_queues_[warp_id].empty()) {
        throw std::runtime_error("TraceReader: No instructions remaining to peek for warp " + std::to_string(warp_id));
    }
    return warp_queues_[warp_id].front();
}

size_t TraceReader::remaining_instructions(WarpId warp_id) const {
    if (warp_id >= warp_queues_.size()) return 0;
    return warp_queues_[warp_id].size();
}

bool TraceReader::has_any_remaining_instructions() const {
    for (const auto& q : warp_queues_) {
        if (!q.empty()) return true;
    }
    return false;
}

} // namespace gpu_sim
