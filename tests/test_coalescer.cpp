#include "catch.hpp"
#include "coalescer.hpp"
#include "trace_reader.hpp"
#include <fstream>

using namespace gpu_sim;

TEST_CASE("Coalescer: Contiguous Aligned Access (100% Efficiency)", "[coalescer]") {
    Coalescer coalescer(32, 128); // 32B sector, 128B cache line

    Instruction inst;
    inst.warp_id = 0;
    inst.pc = 0x100;
    inst.type = InstructionType::LOAD;
    inst.active_mask = FULL_WARP_MASK;
    inst.access_size = 4;

    Address base = 0x1000; // Aligned to 128 bytes
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        inst.addresses[i] = base + i * 4;
    }

    CoalescingResult res = coalescer.coalesce(inst);

    REQUIRE(res.active_threads_count == 32);
    REQUIRE(res.unique_addresses_count == 32);
    REQUIRE(res.unique_sectors_count == 4);
    REQUIRE(res.unique_lines_count == 1);
    REQUIRE(res.transactions.size() == 4);
    REQUIRE(res.total_requested_bytes == 128);
    REQUIRE(res.total_transferred_bytes == 128);
    REQUIRE(res.coalescing_efficiency() == Approx(1.0));

    // Verify 4 sectors: 0x1000, 0x1020, 0x1040, 0x1060
    REQUIRE(res.transactions[0].sector_addr == 0x1000);
    REQUIRE(res.transactions[1].sector_addr == 0x1020);
    REQUIRE(res.transactions[2].sector_addr == 0x1040);
    REQUIRE(res.transactions[3].sector_addr == 0x1060);

    // Each sector should have 8 contributing lanes
    REQUIRE(res.transactions[0].contributing_mask == 0x000000FFU); // lanes 0..7
    REQUIRE(res.transactions[1].contributing_mask == 0x0000FF00U); // lanes 8..15
    REQUIRE(res.transactions[2].contributing_mask == 0x00FF0000U); // lanes 16..23
    REQUIRE(res.transactions[3].contributing_mask == 0xFF000000U); // lanes 24..31
}

TEST_CASE("Coalescer: Strided Access with Large Stride (Memory Divergence)", "[coalescer]") {
    Coalescer coalescer(32, 128);

    Instruction inst;
    inst.warp_id = 1;
    inst.pc = 0x200;
    inst.type = InstructionType::LOAD;
    inst.active_mask = FULL_WARP_MASK;
    inst.access_size = 4;

    Address base = 0x2000;
    // Each thread accesses a separate 128-byte line (stride = 128 bytes)
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        inst.addresses[i] = base + i * 128;
    }

    CoalescingResult res = coalescer.coalesce(inst);

    REQUIRE(res.active_threads_count == 32);
    REQUIRE(res.unique_sectors_count == 32);
    REQUIRE(res.unique_lines_count == 32);
    REQUIRE(res.transactions.size() == 32);
    REQUIRE(res.total_requested_bytes == 128);
    REQUIRE(res.total_transferred_bytes == 32 * 32); // 1024 bytes transferred
    REQUIRE(res.coalescing_efficiency() == Approx(128.0 / 1024.0)); // 12.5% efficiency
}

TEST_CASE("Coalescer: Partially Active Warp", "[coalescer]") {
    Coalescer coalescer(32, 128);

    Instruction inst;
    inst.warp_id = 2;
    inst.pc = 0x300;
    inst.type = InstructionType::STORE;
    inst.active_mask = 0x0000FFFFU; // Only lower 16 lanes active
    inst.access_size = 4;

    Address base = 0x3000;
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        inst.addresses[i] = base + i * 4;
    }

    CoalescingResult res = coalescer.coalesce(inst);

    REQUIRE(res.active_threads_count == 16);
    REQUIRE(res.unique_sectors_count == 2);
    REQUIRE(res.unique_lines_count == 1);
    REQUIRE(res.transactions.size() == 2);
    REQUIRE(res.transactions[0].is_write == true);
    REQUIRE(res.total_requested_bytes == 64);
    REQUIRE(res.total_transferred_bytes == 64);
    REQUIRE(res.coalescing_efficiency() == Approx(1.0));
}

TEST_CASE("Coalescer: Uniform / Broadcast Access", "[coalescer]") {
    Coalescer coalescer(32, 128);

    Instruction inst;
    inst.warp_id = 3;
    inst.pc = 0x400;
    inst.type = InstructionType::LOAD;
    inst.active_mask = FULL_WARP_MASK;
    inst.access_size = 4;

    Address broadcast_addr = 0x4000;
    for (uint32_t i = 0; i < WARP_SIZE; ++i) {
        inst.addresses[i] = broadcast_addr;
    }

    CoalescingResult res = coalescer.coalesce(inst);

    REQUIRE(res.active_threads_count == 32);
    REQUIRE(res.unique_addresses_count == 1);
    REQUIRE(res.unique_sectors_count == 1);
    REQUIRE(res.unique_lines_count == 1);
    REQUIRE(res.transactions.size() == 1);
    REQUIRE(res.transactions[0].contributing_mask == FULL_WARP_MASK);
    REQUIRE(res.total_requested_bytes == 128);
    REQUIRE(res.total_transferred_bytes == 32);
    // Efficiency: requested 128 bytes from a 32-byte sector -> 4.0
    REQUIRE(res.coalescing_efficiency() == Approx(4.0));
}

TEST_CASE("TraceReader: Synthetic Trace Parsing and Per-Warp Queuing", "[trace_reader]") {
    TraceReader reader;
    std::string trace_path = "traces/sample.trace";
    std::ifstream check_file(trace_path);
    if (!check_file.is_open()) {
        trace_path = "../traces/sample.trace";
    }
    REQUIRE_NOTHROW(reader.load_trace(trace_path));

    REQUIRE(reader.total_instructions_loaded() > 0);
    REQUIRE(reader.active_warps().size() == 4);

    for (WarpId w = 0; w < 4; ++w) {
        REQUIRE(reader.has_instruction(w));
        REQUIRE(reader.remaining_instructions(w) == 20);
    }

    Instruction first_inst = reader.pop_next_instruction(0);
    REQUIRE(first_inst.warp_id == 0);
    REQUIRE(first_inst.pc == 0x8000);
    REQUIRE(first_inst.type == InstructionType::LOAD);
    REQUIRE(first_inst.active_mask == FULL_WARP_MASK);
    REQUIRE(first_inst.access_size == 4);
    REQUIRE(first_inst.addresses[0] == 0x10000000);
    REQUIRE(first_inst.addresses[1] == 0x10000004);

    REQUIRE(reader.remaining_instructions(0) == 19);
}
