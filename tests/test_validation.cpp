#include "catch.hpp"
#include "simulator.hpp"
#include <fstream>

using namespace gpu_sim;

namespace {

void write_test_trace(const std::string& path, const std::string& content) {
    std::ofstream out(path);
    out << content;
    out.close();
}

} // anonymous namespace

TEST_CASE("Validation: Deterministic Execution Repeatability", "[validation]") {
    SimConfig cfg;
    cfg.warp_count = 4;
    cfg.scheduling_policy = SchedulingPolicy::GTO;
    cfg.l1_enabled = true;
    cfg.l1_capacity_bytes = 32768;
    cfg.l2_capacity_bytes = 65536;
    cfg.dram_latency = 50;

    std::string trace_path = "traces/determinism_test.trace";
    std::string content =
        "LOAD_STRIDED 0 0x1000 FFFFFFFF 4 0x10000 4\n"
        "LOAD_STRIDED 1 0x1000 FFFFFFFF 4 0x20000 4\n"
        "ALU 2 0x1000\n"
        "ALU 3 0x1000\n"
        "LOAD_STRIDED 2 0x1004 FFFFFFFF 4 0x10000 4\n" // Hits L1 (brought by warp 0)
        "ALU 0 0x1004\n"
        "ALU 1 0x1004\n";
    write_test_trace(trace_path, content);

    // Run 1
    Simulator sim1(cfg);
    sim1.load_trace(trace_path);
    uint64_t cycles1 = sim1.run();

    // Run 2
    Simulator sim2(cfg);
    sim2.load_trace(trace_path);
    uint64_t cycles2 = sim2.run();

    // Bit-for-bit repeatability check
    REQUIRE(cycles1 == cycles2);
    REQUIRE(sim1.total_instructions_issued() == sim2.total_instructions_issued());
    REQUIRE(sim1.ipc() == Approx(sim2.ipc()));
    REQUIRE(sim1.memory().l1_cache()->stats().total_hits() == sim2.memory().l1_cache()->stats().total_hits());
    REQUIRE(sim1.memory().l1_cache()->stats().total_misses() == sim2.memory().l1_cache()->stats().total_misses());
    REQUIRE(sim1.memory().dram().stats().total_requests == sim2.memory().dram().stats().total_requests);

    // Verify full JSON stats equality
    nlohmann::json j1 = sim1.get_statistics_json();
    nlohmann::json j2 = sim2.get_statistics_json();
    REQUIRE(j1 == j2);
}

TEST_CASE("Validation: Conflict Misses vs Increased Associativity", "[validation]") {
    // 1024B capacity, 64B line, 32B sector -> 16 lines total
    // All 3 addresses map to Set 0: (tag << 9)
    Address addrA = (1ULL << 9);
    Address addrB = (2ULL << 9);
    Address addrC = (3ULL << 9);

    std::string trace_path = "traces/associativity_test.trace";
    std::string content =
        "LOAD_UNIFORM 0 0x1000 FFFFFFFF 4 " + std::to_string(addrA) + "\n"
        "LOAD_UNIFORM 0 0x1004 FFFFFFFF 4 " + std::to_string(addrB) + "\n"
        "LOAD_UNIFORM 0 0x1008 FFFFFFFF 4 " + std::to_string(addrC) + "\n"
        "LOAD_UNIFORM 0 0x100C FFFFFFFF 4 " + std::to_string(addrA) + "\n"
        "LOAD_UNIFORM 0 0x1010 FFFFFFFF 4 " + std::to_string(addrB) + "\n"
        "LOAD_UNIFORM 0 0x1014 FFFFFFFF 4 " + std::to_string(addrC) + "\n";
    write_test_trace(trace_path, content);

    SECTION("2-Way Associative: Thrashing and Evictions") {
        SimConfig cfg;
        cfg.warp_count = 1;
        cfg.l1_enabled = true;
        cfg.l1_capacity_bytes = 1024;
        cfg.l1_line_size_bytes = 64;
        cfg.l1_associativity = 2; // Can only hold 2 lines in Set 0 -> 3 lines cause thrashing!
        cfg.sector_size_bytes = 32;
        cfg.dram_latency = 10;

        Simulator sim(cfg);
        sim.load_trace(trace_path);
        sim.run();

        // 3 items in a 2-way set causes evictions
        REQUIRE(sim.memory().l1_cache()->stats().evictions > 0);
        REQUIRE(sim.memory().l1_cache()->stats().total_misses() > 3);
    }

    SECTION("4-Way Associative: Fits Working Set, Zero Evictions") {
        SimConfig cfg;
        cfg.warp_count = 1;
        cfg.l1_enabled = true;
        cfg.l1_capacity_bytes = 1024;
        cfg.l1_line_size_bytes = 64;
        cfg.l1_associativity = 4; // Can hold 4 lines in Set 0 -> All 3 lines fit!
        cfg.sector_size_bytes = 32;
        cfg.dram_latency = 10;

        Simulator sim(cfg);
        sim.load_trace(trace_path);
        sim.run();

        // All 3 items fit in the 4-way set, so exactly 3 cold misses and 3 hits!
        REQUIRE(sim.memory().l1_cache()->stats().evictions == 0);
        REQUIRE(sim.memory().l1_cache()->stats().total_hits() == 3);
        REQUIRE(sim.memory().l1_cache()->stats().total_misses() == 3);
        REQUIRE(sim.memory().l1_cache()->stats().hit_rate() == 0.5);
    }
}

TEST_CASE("Validation: Edge Cases (Empty and Comment-Only Traces)", "[validation]") {
    SimConfig cfg;
    cfg.warp_count = 2;

    SECTION("Empty Trace File") {
        std::string empty_path = "traces/empty.trace";
        write_test_trace(empty_path, "");

        Simulator sim(cfg);
        sim.load_trace(empty_path);
        uint64_t cycles = sim.run();

        // Terminates immediately
        REQUIRE(sim.total_instructions_issued() == 0);
        REQUIRE(cycles <= 1);
        REQUIRE(sim.ipc() == 0.0);
    }

    SECTION("Comment-Only Trace File") {
        std::string comment_path = "traces/comments.trace";
        write_test_trace(comment_path, "# Just a comment\n# Another comment\n\n");

        Simulator sim(cfg);
        sim.load_trace(comment_path);
        uint64_t cycles = sim.run();

        REQUIRE(sim.total_instructions_issued() == 0);
        REQUIRE(cycles <= 1);
    }
}

TEST_CASE("Validation: DRAM Bandwidth Saturation Hand-Calculation", "[validation]") {
    SimConfig cfg;
    cfg.warp_count = 1;
    cfg.l1_enabled = false; // Bypass L1
    cfg.l2_capacity_bytes = 1024;
    cfg.l2_line_size_bytes = 64;
    cfg.l2_associativity = 1;
    cfg.dram_latency = 10;
    cfg.dram_bandwidth_bytes_per_cycle = 16.0; // 16 B/cycle sustained BW
    // Each 32B transaction takes 32 / 16 = 2 cycles of bus time

    Simulator sim(cfg);

    std::string trace_path = "traces/bw_sat_test.trace";
    // 5 uniform loads to distinct memory addresses -> 5 DRAM requests
    // Total bytes = 5 * 32 = 160 bytes
    // Expected transfer cycles = 5 * 2 = 10 cycles
    std::string content =
        "LOAD_UNIFORM 0 0x1000 FFFFFFFF 4 0x10000\n"
        "LOAD_UNIFORM 0 0x1004 FFFFFFFF 4 0x20000\n"
        "LOAD_UNIFORM 0 0x1008 FFFFFFFF 4 0x30000\n"
        "LOAD_UNIFORM 0 0x100C FFFFFFFF 4 0x40000\n"
        "LOAD_UNIFORM 0 0x1010 FFFFFFFF 4 0x50000\n";
    write_test_trace(trace_path, content);

    sim.load_trace(trace_path);
    sim.run();

    const auto& dram_stats = sim.memory().dram().stats();
    REQUIRE(dram_stats.total_requests == 5);
    REQUIRE(dram_stats.total_bytes_transferred == 160);
    REQUIRE(dram_stats.total_busy_cycles == 10);
    REQUIRE(sim.memory().average_memory_latency() > 0.0);
}
