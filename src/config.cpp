#include "config.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace gpu_sim {

namespace {

bool is_power_of_two(uint64_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

} // anonymous namespace

uint64_t SimConfig::l1_num_sets() const {
    if (!l1_enabled) return 0;
    return l1_capacity_bytes / (static_cast<uint64_t>(l1_line_size_bytes) * l1_associativity);
}

uint64_t SimConfig::l2_num_sets() const {
    return l2_capacity_bytes / (static_cast<uint64_t>(l2_line_size_bytes) * l2_associativity);
}

void SimConfig::validate() const {
    // SM validation
    if (warp_count == 0) {
        throw std::invalid_argument("Validation error: warp_count must be > 0");
    }
    if (scheduler_count == 0) {
        throw std::invalid_argument("Validation error: scheduler_count must be > 0");
    }
    if (issue_width == 0) {
        throw std::invalid_argument("Validation error: issue_width must be > 0");
    }

    // Coalescing validation
    if (!is_power_of_two(sector_size_bytes)) {
        throw std::invalid_argument("Validation error: sector_size_bytes must be a power of 2 (e.g. 32)");
    }

    // L1 validation
    if (l1_enabled) {
        if (!is_power_of_two(l1_capacity_bytes)) {
            throw std::invalid_argument("Validation error: l1_capacity_bytes must be a power of 2");
        }
        if (!is_power_of_two(l1_line_size_bytes)) {
            throw std::invalid_argument("Validation error: l1_line_size_bytes must be a power of 2");
        }
        if (!is_power_of_two(l1_associativity)) {
            throw std::invalid_argument("Validation error: l1_associativity must be a power of 2");
        }
        if (l1_line_size_bytes < sector_size_bytes || (l1_line_size_bytes % sector_size_bytes) != 0) {
            throw std::invalid_argument("Validation error: l1_line_size_bytes must be >= sector_size_bytes and a multiple of sector_size_bytes");
        }
        uint64_t min_l1_capacity = static_cast<uint64_t>(l1_line_size_bytes) * l1_associativity;
        if (l1_capacity_bytes < min_l1_capacity) {
            throw std::invalid_argument("Validation error: l1_capacity_bytes (" + std::to_string(l1_capacity_bytes) +
                                        ") is smaller than line_size * associativity (" + std::to_string(min_l1_capacity) + ")");
        }
    }

    // L2 validation
    if (!is_power_of_two(l2_capacity_bytes)) {
        throw std::invalid_argument("Validation error: l2_capacity_bytes must be a power of 2");
    }
    if (!is_power_of_two(l2_line_size_bytes)) {
        throw std::invalid_argument("Validation error: l2_line_size_bytes must be a power of 2");
    }
    if (!is_power_of_two(l2_associativity)) {
        throw std::invalid_argument("Validation error: l2_associativity must be a power of 2");
    }
    if (l2_line_size_bytes < sector_size_bytes || (l2_line_size_bytes % sector_size_bytes) != 0) {
        throw std::invalid_argument("Validation error: l2_line_size_bytes must be >= sector_size_bytes and a multiple of sector_size_bytes");
    }
    uint64_t min_l2_capacity = static_cast<uint64_t>(l2_line_size_bytes) * l2_associativity;
    if (l2_capacity_bytes < min_l2_capacity) {
        throw std::invalid_argument("Validation error: l2_capacity_bytes (" + std::to_string(l2_capacity_bytes) +
                                    ") is smaller than line_size * associativity (" + std::to_string(min_l2_capacity) + ")");
    }

    // DRAM validation
    if (dram_latency == 0) {
        throw std::invalid_argument("Validation error: dram_latency must be >= 1 cycle");
    }
    if (dram_bandwidth_bytes_per_cycle <= 0.0) {
        throw std::invalid_argument("Validation error: dram_bandwidth_bytes_per_cycle must be > 0.0");
    }
    if (dram_max_queue_size == 0) {
        throw std::invalid_argument("Validation error: dram_max_queue_size must be > 0");
    }
}

SimConfig SimConfig::from_json(const nlohmann::json& j) {
    SimConfig cfg;

    if (j.contains("warp_count")) cfg.warp_count = j.at("warp_count").get<uint32_t>();
    if (j.contains("scheduler_count")) cfg.scheduler_count = j.at("scheduler_count").get<uint32_t>();
    if (j.contains("issue_width")) cfg.issue_width = j.at("issue_width").get<uint32_t>();
    if (j.contains("scheduling_policy")) {
        cfg.scheduling_policy = scheduling_policy_from_string(j.at("scheduling_policy").get<std::string>());
    }

    if (j.contains("sector_size_bytes")) cfg.sector_size_bytes = j.at("sector_size_bytes").get<uint32_t>();

    if (j.contains("l1_cache")) {
        const auto& l1 = j.at("l1_cache");
        if (l1.contains("enabled")) cfg.l1_enabled = l1.at("enabled").get<bool>();
        if (l1.contains("capacity_bytes")) cfg.l1_capacity_bytes = l1.at("capacity_bytes").get<uint64_t>();
        if (l1.contains("line_size_bytes")) cfg.l1_line_size_bytes = l1.at("line_size_bytes").get<uint32_t>();
        if (l1.contains("associativity")) cfg.l1_associativity = l1.at("associativity").get<uint32_t>();
        if (l1.contains("hit_latency")) cfg.l1_hit_latency = l1.at("hit_latency").get<Cycle>();
        if (l1.contains("replacement_policy")) {
            cfg.l1_replacement_policy = replacement_policy_from_string(l1.at("replacement_policy").get<std::string>());
        }
    }

    if (j.contains("l2_cache")) {
        const auto& l2 = j.at("l2_cache");
        if (l2.contains("capacity_bytes")) cfg.l2_capacity_bytes = l2.at("capacity_bytes").get<uint64_t>();
        if (l2.contains("line_size_bytes")) cfg.l2_line_size_bytes = l2.at("line_size_bytes").get<uint32_t>();
        if (l2.contains("associativity")) cfg.l2_associativity = l2.at("associativity").get<uint32_t>();
        if (l2.contains("hit_latency")) cfg.l2_hit_latency = l2.at("hit_latency").get<Cycle>();
        if (l2.contains("replacement_policy")) {
            cfg.l2_replacement_policy = replacement_policy_from_string(l2.at("replacement_policy").get<std::string>());
        }
    }

    if (j.contains("dram")) {
        const auto& d = j.at("dram");
        if (d.contains("latency")) cfg.dram_latency = d.at("latency").get<Cycle>();
        if (d.contains("bandwidth_bytes_per_cycle")) cfg.dram_bandwidth_bytes_per_cycle = d.at("bandwidth_bytes_per_cycle").get<double>();
        if (d.contains("max_queue_size")) cfg.dram_max_queue_size = d.at("max_queue_size").get<uint32_t>();
    }

    if (j.contains("trace_file")) cfg.trace_file = j.at("trace_file").get<std::string>();
    if (j.contains("output_stats_file")) cfg.output_stats_file = j.at("output_stats_file").get<std::string>();

    cfg.validate();
    return cfg;
}

nlohmann::json SimConfig::to_json() const {
    nlohmann::json j;
    j["warp_count"] = warp_count;
    j["scheduler_count"] = scheduler_count;
    j["issue_width"] = issue_width;
    j["scheduling_policy"] = to_string(scheduling_policy);
    j["sector_size_bytes"] = sector_size_bytes;

    j["l1_cache"] = {
        {"enabled", l1_enabled},
        {"capacity_bytes", l1_capacity_bytes},
        {"line_size_bytes", l1_line_size_bytes},
        {"associativity", l1_associativity},
        {"hit_latency", l1_hit_latency},
        {"replacement_policy", to_string(l1_replacement_policy)},
        {"num_sets", l1_num_sets()}
    };

    j["l2_cache"] = {
        {"capacity_bytes", l2_capacity_bytes},
        {"line_size_bytes", l2_line_size_bytes},
        {"associativity", l2_associativity},
        {"hit_latency", l2_hit_latency},
        {"replacement_policy", to_string(l2_replacement_policy)},
        {"num_sets", l2_num_sets()}
    };

    j["dram"] = {
        {"latency", dram_latency},
        {"bandwidth_bytes_per_cycle", dram_bandwidth_bytes_per_cycle},
        {"max_queue_size", dram_max_queue_size}
    };

    j["trace_file"] = trace_file;
    j["output_stats_file"] = output_stats_file;

    return j;
}

SimConfig SimConfig::from_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open configuration file: " + filepath);
    }
    nlohmann::json j;
    try {
        file >> j;
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error("JSON parse error in configuration file " + filepath + ": " + e.what());
    }
    return from_json(j);
}

void SimConfig::to_file(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not write to file: " + filepath);
    }
    file << to_json().dump(4) << std::endl;
}

} // namespace gpu_sim
