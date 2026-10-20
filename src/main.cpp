#include "config.hpp"
#include "simulator.hpp"
#include <iostream>
#include <string>
#include <vector>

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n"
              << "Options:\n"
              << "  --config <path>    Path to JSON configuration file (default: configs/default.json)\n"
              << "  --trace  <path>    Path to input trace file (overrides config)\n"
              << "  --stats  <path>    Path to output statistics file (overrides config)\n"
              << "  --help             Show this help message\n";
}

int main(int argc, char* argv[]) {
    std::string config_path = "configs/default.json";
    std::string trace_override = "";
    std::string stats_override = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else if (arg == "--trace" && i + 1 < argc) {
            trace_override = argv[++i];
        } else if (arg == "--stats" && i + 1 < argc) {
            stats_override = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    std::cout << "========================================================\n"
              << "  Trace-Driven GPU Memory & Warp Scheduler Simulator    \n"
              << "========================================================\n";

    gpu_sim::SimConfig config;
    try {
        std::cout << "[INFO] Loading configuration from: " << config_path << "\n";
        config = gpu_sim::SimConfig::from_file(config_path);
        if (!trace_override.empty()) config.trace_file = trace_override;
        if (!stats_override.empty()) config.output_stats_file = stats_override;
        config.validate();
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Configuration error: " << e.what() << "\n";
        return 1;
    }

    std::cout << "[INFO] Configuration validated successfully:\n"
              << "  - Warps:            " << config.warp_count << "\n"
              << "  - Schedulers:       " << config.scheduler_count << " (Issue width: " << config.issue_width << ")\n"
              << "  - Policy:           " << to_string(config.scheduling_policy) << "\n"
              << "  - Sector Size:      " << config.sector_size_bytes << " B\n"
              << "  - L1 Cache:         " << (config.l1_enabled ? "Enabled" : "Disabled")
              << " (" << (config.l1_capacity_bytes / 1024) << " KB, "
              << config.l1_associativity << "-way, "
              << config.l1_num_sets() << " sets, "
              << config.l1_hit_latency << " cyc latency, "
              << to_string(config.l1_replacement_policy) << ")\n"
              << "  - L2 Cache:         " << (config.l2_capacity_bytes / 1024) << " KB, "
              << config.l2_associativity << "-way, "
              << config.l2_num_sets() << " sets, "
              << config.l2_hit_latency << " cyc latency, "
              << to_string(config.l2_replacement_policy) << "\n"
              << "  - DRAM:             " << config.dram_latency << " cyc latency, "
              << config.dram_bandwidth_bytes_per_cycle << " B/cycle sustained BW\n"
              << "  - Trace File:       " << (config.trace_file.empty() ? "<none>" : config.trace_file) << "\n"
              << "  - Stats File:       " << config.output_stats_file << "\n"
              << "========================================================\n";

    if (config.trace_file.empty()) {
        std::cout << "[INFO] No trace file specified. Exiting cleanly.\n";
        return 0;
    }

    std::cout << "[INFO] Running simulation on trace: " << config.trace_file << "\n";
    gpu_sim::Simulator sim(config);
    uint64_t total_cycles = sim.run();

    std::cout << "\n================ SIMULATION RESULTS ================\n"
              << "  Total Simulated Cycles:     " << total_cycles << "\n"
              << "  Total Instructions Issued:  " << sim.total_instructions_issued() << "\n"
              << "  Overall IPC:                " << sim.ipc() << "\n";

    if (sim.memory().l1_cache()) {
        std::cout << "  L1 Hit Rate:                "
                  << (sim.memory().l1_cache()->stats().hit_rate() * 100.0) << "%\n";
    }
    if (sim.memory().l2_cache()) {
        std::cout << "  L2 Hit Rate:                "
                  << (sim.memory().l2_cache()->stats().hit_rate() * 100.0) << "%\n";
    }
    std::cout << "  DRAM Requests:              " << sim.memory().dram().stats().total_requests << "\n"
              << "  DRAM BW Utilization:        "
              << (sim.memory().dram().stats().bandwidth_utilization(total_cycles) * 100.0) << "%\n"
              << "====================================================\n";

    try {
        sim.dump_statistics_to_file(config.output_stats_file);
        std::cout << "[INFO] Successfully saved statistics to: " << config.output_stats_file << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[WARN] Could not save stats to file: " << e.what() << "\n";
    }

    return 0;
}
