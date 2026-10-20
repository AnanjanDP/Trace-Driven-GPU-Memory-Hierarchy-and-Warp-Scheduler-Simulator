#include "catch.hpp"
#include "scheduler.hpp"
#include "warp.hpp"
#include "trace_reader.hpp"
#include <fstream>

using namespace gpu_sim;

namespace {

// Helper to construct a synthetic TraceReader with known instruction counts per warp
TraceReader make_synthetic_reader(const std::vector<std::pair<WarpId, uint32_t>>& warp_counts) {
    TraceReader reader;
    // We can write to a temporary trace file or use an empty trace and inject instructions
    // Let's create a minimal synthetic trace in memory and load it
    std::string content = "# Synthetic trace\n";
    for (const auto& pair : warp_counts) {
        WarpId wid = pair.first;
        uint32_t count = pair.second;
        for (uint32_t i = 0; i < count; ++i) {
            content += "ALU " + std::to_string(wid) + " " + std::to_string(0x1000 + i * 4) + "\n";
        }
    }
    // Write to a temporary file
    std::string path = "traces/temp_sched_test.trace";
    std::ofstream out(path);
    if (!out.is_open()) {
        path = "../traces/temp_sched_test.trace";
        out.open(path);
    }
    out << content;
    out.close();

    reader.load_trace(path);
    return reader;
}

} // anonymous namespace

TEST_CASE("Scheduler: Round-Robin Rotation Across Eligible Warps", "[scheduler]") {
    WarpScheduler scheduler(SchedulingPolicy::ROUND_ROBIN, /*issue_width=*/1);

    std::vector<Warp> warps;
    for (WarpId w = 0; w < 4; ++w) {
        warps.emplace_back(w);
    }

    TraceReader reader = make_synthetic_reader({{0, 10}, {1, 10}, {2, 10}, {3, 10}});

    // Cycle 1: should select warp 1 (since last_rr_index starts at 0, (0+1)%4 = 1)
    auto sel1 = scheduler.select_warps(warps, reader, 1);
    REQUIRE(sel1.size() == 1);
    REQUIRE(sel1[0] == 1);

    // Cycle 2: should select warp 2
    auto sel2 = scheduler.select_warps(warps, reader, 2);
    REQUIRE(sel2.size() == 1);
    REQUIRE(sel2[0] == 2);

    // Cycle 3: should select warp 3
    auto sel3 = scheduler.select_warps(warps, reader, 3);
    REQUIRE(sel3.size() == 1);
    REQUIRE(sel3[0] == 3);

    // Cycle 4: should wrap around and select warp 0
    auto sel4 = scheduler.select_warps(warps, reader, 4);
    REQUIRE(sel4.size() == 1);
    REQUIRE(sel4[0] == 0);

    // Cycle 5: should select warp 1 again
    auto sel5 = scheduler.select_warps(warps, reader, 5);
    REQUIRE(sel5.size() == 1);
    REQUIRE(sel5[0] == 1);

    REQUIRE(scheduler.stats().cycles_evaluated == 5);
    REQUIRE(scheduler.stats().cycles_issued == 5);
    REQUIRE(scheduler.stats().cycles_idle == 0);
}

TEST_CASE("Scheduler: Round-Robin Skips Stalled Warps", "[scheduler]") {
    WarpScheduler scheduler(SchedulingPolicy::ROUND_ROBIN, 1);

    std::vector<Warp> warps;
    for (WarpId w = 0; w < 4; ++w) {
        warps.emplace_back(w);
    }

    // Warp 1 and Warp 2 stall on memory
    warps[1].set_waiting_on_memory(1);
    warps[2].set_waiting_on_memory(1);

    TraceReader reader = make_synthetic_reader({{0, 10}, {1, 10}, {2, 10}, {3, 10}});

    // Starting from 0, RR should skip 1 and 2, selecting 3!
    auto sel1 = scheduler.select_warps(warps, reader, 1);
    REQUIRE(sel1.size() == 1);
    REQUIRE(sel1[0] == 3);

    // Next cycle: should wrap around, select 0
    auto sel2 = scheduler.select_warps(warps, reader, 2);
    REQUIRE(sel2.size() == 1);
    REQUIRE(sel2[0] == 0);

    // Next cycle: skips 1 and 2 again, selects 3
    auto sel3 = scheduler.select_warps(warps, reader, 3);
    REQUIRE(sel3.size() == 1);
    REQUIRE(sel3[0] == 3);
}

TEST_CASE("Scheduler: GTO Policy Keeps Issuing Greedy Warp", "[scheduler]") {
    WarpScheduler scheduler(SchedulingPolicy::GTO, 1);

    std::vector<Warp> warps;
    for (WarpId w = 0; w < 3; ++w) {
        warps.emplace_back(w);
    }

    TraceReader reader = make_synthetic_reader({{0, 10}, {1, 10}, {2, 10}});

    // Cycle 1: Initially no last issued warp. Picks oldest (all last_issued=0, tie-break by smallest ID = 0)
    auto sel1 = scheduler.select_warps(warps, reader, 1);
    REQUIRE(sel1.size() == 1);
    REQUIRE(sel1[0] == 0);

    // Cycle 2: Warp 0 remains eligible. GTO must GREEDILY select warp 0 again!
    auto sel2 = scheduler.select_warps(warps, reader, 2);
    REQUIRE(sel2.size() == 1);
    REQUIRE(sel2[0] == 0);

    // Cycle 3: Still greedily warp 0
    auto sel3 = scheduler.select_warps(warps, reader, 3);
    REQUIRE(sel3.size() == 1);
    REQUIRE(sel3[0] == 0);
}

TEST_CASE("Scheduler: GTO Switches to Oldest Warp When Active Warp Stalls", "[scheduler]") {
    WarpScheduler scheduler(SchedulingPolicy::GTO, 1);

    std::vector<Warp> warps;
    for (WarpId w = 0; w < 3; ++w) {
        warps.emplace_back(w);
    }

    TraceReader reader = make_synthetic_reader({{0, 10}, {1, 10}, {2, 10}});

    // Cycle 1: Selects Warp 0
    auto sel1 = scheduler.select_warps(warps, reader, 1);
    REQUIRE(sel1[0] == 0);
    warps[0].last_issued_cycle = 1;

    // Warp 0 now stalls on memory
    warps[0].set_waiting_on_memory(1);

    // Configure ages for warps 1 and 2:
    // Warp 1 was issued at cycle 50 (younger)
    // Warp 2 was issued at cycle 20 (older)
    warps[1].last_issued_cycle = 50;
    warps[2].last_issued_cycle = 20;

    // Cycle 2: Greedy warp 0 is stalled. GTO must pick OLDEST eligible warp -> Warp 2!
    auto sel2 = scheduler.select_warps(warps, reader, 2);
    REQUIRE(sel2.size() == 1);
    REQUIRE(sel2[0] == 2);

    // Now Warp 2 is the active greedy warp!
    auto sel3 = scheduler.select_warps(warps, reader, 3);
    REQUIRE(sel3[0] == 2);
}

TEST_CASE("Scheduler: Idle When All Warps Stalled or Completed", "[scheduler]") {
    WarpScheduler scheduler(SchedulingPolicy::ROUND_ROBIN, 1);

    std::vector<Warp> warps;
    warps.emplace_back(0);
    warps.emplace_back(1);

    TraceReader reader = make_synthetic_reader({{0, 5}, {1, 5}});

    // Both warps stall
    warps[0].set_waiting_on_memory(1);
    warps[1].set_waiting_on_memory(2);

    auto sel = scheduler.select_warps(warps, reader, 10);
    REQUIRE(sel.empty());
    REQUIRE(scheduler.stats().cycles_evaluated == 1);
    REQUIRE(scheduler.stats().cycles_idle == 1);
    REQUIRE(scheduler.stats().cycles_issued == 0);
    REQUIRE(scheduler.stats().scheduler_utilization() == 0.0);
}
