#include "catch.hpp"
#include "simulator.hpp"
#include <fstream>

using namespace gpu_sim;

namespace {

void write_trace(const std::string& path, const std::string& content) {
    std::ofstream out(path);
    out << content;
    out.close();
}

} // anonymous namespace

TEST_CASE("Simulator: Pure ALU Workload (IPC = 1.0)", "[simulator]") {
    SimConfig cfg;
    cfg.warp_count = 1;
    cfg.scheduler_count = 1;
    cfg.issue_width = 1;
    cfg.trace_file = ""; // We will load directly

    Simulator sim(cfg);

    std::string trace_path = "traces/alu_test.trace";
    std::string content =
        "ALU 0 0x1000\n"
        "ALU 0 0x1004\n"
        "ALU 0 0x1008\n"
        "ALU 0 0x100C\n"
        "ALU 0 0x1010\n";
    write_trace(trace_path, content);

    sim.load_trace(trace_path);

    uint64_t cycles = sim.run();

    REQUIRE(cycles == 5);
    REQUIRE(sim.total_instructions_issued() == 5);
    REQUIRE(sim.ipc() == Approx(1.0));
    REQUIRE(sim.warps()[0].instructions_issued == 5);
    REQUIRE(sim.warps()[0].stall_cycles == 0);
}

TEST_CASE("Simulator: Single-Warp Memory Stall and Wake-Up", "[simulator]") {
    SimConfig cfg;
    cfg.warp_count = 1;
    cfg.l1_enabled = true;
    cfg.l1_hit_latency = 2;
    cfg.l2_hit_latency = 5;
    cfg.dram_latency = 20;
    cfg.dram_bandwidth_bytes_per_cycle = 32.0; // 32B sector transfer = 1 cycle

    Simulator sim(cfg);

    std::string trace_path = "traces/mem_stall_test.trace";
    // Warp 0 issues a uniform load (1 transaction of 32B), then an ALU instruction
    std::string content =
        "LOAD_UNIFORM 0 0x1000 FFFFFFFF 4 0x5000\n"
        "ALU 0 0x1004\n";
    write_trace(trace_path, content);

    sim.load_trace(trace_path);

    // Hand calculation:
    // Cycle 1: Warp 0 issues LOAD_UNIFORM.
    //          L1 misses, L2 misses.
    //          DRAM: transfer takes 1 cycle (32B / 32B/cyc).
    //          Bus transfer: cycle 1 to 2.
    //          DRAM done: transfer_end(2) + dram_latency(20) = 22.
    //          Event completion: 22 + L1_lat(2) + L2_lat(5) = 29.
    // Cycles 2..28: Warp 0 stalled on memory (27 stall cycles).
    // Cycle 29: Memory completes, fills L2 and L1, notifies warp 0.
    //           Warp 0 wakes up, is selected by scheduler, and issues ALU!
    // Total instructions: 2.
    uint64_t cycles = sim.run();

    REQUIRE(cycles == 29);
    REQUIRE(sim.total_instructions_issued() == 2);
    REQUIRE(sim.warps()[0].instructions_issued == 2);
    REQUIRE(sim.warps()[0].memory_stall_cycles == 27);
}

TEST_CASE("Simulator: Latency Hiding Between Two Warps", "[simulator]") {
    SimConfig cfg;
    cfg.warp_count = 2;
    cfg.scheduling_policy = SchedulingPolicy::ROUND_ROBIN;
    cfg.l1_enabled = true;
    cfg.l1_hit_latency = 2;
    cfg.l2_hit_latency = 5;
    cfg.dram_latency = 10;

    Simulator sim(cfg);

    std::string trace_path = "traces/latency_hiding_test.trace";
    // Warp 0 does a load, then an ALU
    // Warp 1 does 5 ALU ops
    std::string content =
        "LOAD_UNIFORM 0 0x1000 FFFFFFFF 4 0x6000\n"
        "ALU 1 0x2000\n"
        "ALU 1 0x2004\n"
        "ALU 1 0x2008\n"
        "ALU 1 0x200C\n"
        "ALU 1 0x2010\n"
        "ALU 0 0x1004\n";
    write_trace(trace_path, content);

    sim.load_trace(trace_path);

    // Cycle 1: Warp 1 is chosen by RR ((0+1)%2 = 1), issues ALU 1.
    // Cycle 2: Warp 0 is chosen by RR, issues LOAD_UNIFORM and stalls.
    // Cycles 3..6: Warp 0 is stalled on memory, but Warp 1 CONTINUES issuing ALU ops!
    // This demonstrates simultaneous latency hiding!
    sim.run();

    REQUIRE(sim.warps()[1].instructions_issued == 5);
    REQUIRE(sim.warps()[0].instructions_issued == 2);
    // Warp 1 had zero stalls because it had independent ALU work
    REQUIRE(sim.warps()[1].stall_cycles == 0);
}

TEST_CASE("Simulator: L1 Hit on Repeated Memory Access", "[simulator]") {
    SimConfig cfg;
    cfg.warp_count = 1;
    cfg.l1_enabled = true;
    cfg.l1_hit_latency = 2;
    cfg.l2_hit_latency = 5;
    cfg.dram_latency = 20;

    Simulator sim(cfg);

    std::string trace_path = "traces/l1_hit_test.trace";
    // 1st load: compulsory miss (goes to DRAM)
    // 2nd load: same address -> hits L1!
    std::string content =
        "LOAD_UNIFORM 0 0x1000 FFFFFFFF 4 0x7000\n"
        "LOAD_UNIFORM 0 0x1004 FFFFFFFF 4 0x7000\n";
    write_trace(trace_path, content);

    sim.load_trace(trace_path);
    sim.run();

    // 1st access missed L1, 2nd access hit L1!
    REQUIRE(sim.memory().l1_cache()->stats().accesses == 2);
    REQUIRE(sim.memory().l1_cache()->stats().read_hits == 1);
    REQUIRE(sim.memory().l1_cache()->stats().read_misses == 1);
    REQUIRE(sim.memory().l1_cache()->stats().hit_rate() == 0.5);

    // L2 saw only 1 access (the first miss), because the second access was filtered by L1!
    REQUIRE(sim.memory().l2_cache()->stats().accesses == 1);
}
