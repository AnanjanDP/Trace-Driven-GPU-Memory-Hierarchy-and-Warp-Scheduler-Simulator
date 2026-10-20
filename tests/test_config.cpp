#include "catch.hpp"
#include "config.hpp"

using namespace gpu_sim;

TEST_CASE("SimConfig Default Validation", "[config]") {
    SimConfig cfg;
    REQUIRE_NOTHROW(cfg.validate());
    REQUIRE(cfg.l1_num_sets() == 64);   // 32768 / (128 * 4) = 64
    REQUIRE(cfg.l2_num_sets() == 128);  // 262144 / (128 * 16) = 128
}

TEST_CASE("SimConfig Geometry Calculations", "[config]") {
    SimConfig cfg;
    cfg.l1_capacity_bytes = 65536; // 64 KB
    cfg.l1_line_size_bytes = 64;
    cfg.l1_associativity = 8;
    cfg.sector_size_bytes = 32;

    REQUIRE_NOTHROW(cfg.validate());
    // 65536 / (64 * 8) = 65536 / 512 = 128 sets
    REQUIRE(cfg.l1_num_sets() == 128);
}

TEST_CASE("SimConfig Invalid Cache Geometry Rejection", "[config]") {
    SECTION("Non-power-of-2 L1 capacity") {
        SimConfig cfg;
        cfg.l1_capacity_bytes = 50000;
        REQUIRE_THROWS_AS(cfg.validate(), std::invalid_argument);
    }

    SECTION("Non-power-of-2 L1 associativity") {
        SimConfig cfg;
        cfg.l1_associativity = 3;
        REQUIRE_THROWS_AS(cfg.validate(), std::invalid_argument);
    }

    SECTION("Non-power-of-2 line size") {
        SimConfig cfg;
        cfg.l1_line_size_bytes = 100;
        REQUIRE_THROWS_AS(cfg.validate(), std::invalid_argument);
    }

    SECTION("Capacity smaller than 1 set (line_size * assoc)") {
        SimConfig cfg;
        cfg.l1_capacity_bytes = 128;
        cfg.l1_line_size_bytes = 128;
        cfg.l1_associativity = 4; // needs at least 512 bytes
        REQUIRE_THROWS_AS(cfg.validate(), std::invalid_argument);
    }

    SECTION("Line size smaller than sector size") {
        SimConfig cfg;
        cfg.sector_size_bytes = 64;
        cfg.l1_line_size_bytes = 32;
        REQUIRE_THROWS_AS(cfg.validate(), std::invalid_argument);
    }
}

TEST_CASE("SimConfig Invalid Execution Parameters", "[config]") {
    SECTION("Zero warp count") {
        SimConfig cfg;
        cfg.warp_count = 0;
        REQUIRE_THROWS_AS(cfg.validate(), std::invalid_argument);
    }

    SECTION("Zero scheduler count") {
        SimConfig cfg;
        cfg.scheduler_count = 0;
        REQUIRE_THROWS_AS(cfg.validate(), std::invalid_argument);
    }

    SECTION("Zero issue width") {
        SimConfig cfg;
        cfg.issue_width = 0;
        REQUIRE_THROWS_AS(cfg.validate(), std::invalid_argument);
    }

    SECTION("Zero or negative DRAM bandwidth") {
        SimConfig cfg;
        cfg.dram_bandwidth_bytes_per_cycle = 0.0;
        REQUIRE_THROWS_AS(cfg.validate(), std::invalid_argument);
        cfg.dram_bandwidth_bytes_per_cycle = -5.0;
        REQUIRE_THROWS_AS(cfg.validate(), std::invalid_argument);
    }

    SECTION("Zero DRAM latency") {
        SimConfig cfg;
        cfg.dram_latency = 0;
        REQUIRE_THROWS_AS(cfg.validate(), std::invalid_argument);
    }
}

TEST_CASE("SimConfig JSON Serialization Roundtrip", "[config]") {
    SimConfig original;
    original.warp_count = 16;
    original.scheduling_policy = SchedulingPolicy::GTO;
    original.l1_capacity_bytes = 16384;
    original.l1_associativity = 2;
    original.l1_replacement_policy = ReplacementPolicy::FIFO;
    original.dram_bandwidth_bytes_per_cycle = 64.0;
    original.trace_file = "test.trace";

    nlohmann::json j = original.to_json();
    SimConfig restored = SimConfig::from_json(j);

    REQUIRE(restored.warp_count == 16);
    REQUIRE(restored.scheduling_policy == SchedulingPolicy::GTO);
    REQUIRE(restored.l1_capacity_bytes == 16384);
    REQUIRE(restored.l1_associativity == 2);
    REQUIRE(restored.l1_replacement_policy == ReplacementPolicy::FIFO);
    REQUIRE(restored.dram_bandwidth_bytes_per_cycle == 64.0);
    REQUIRE(restored.trace_file == "test.trace");
    REQUIRE(restored.l1_num_sets() == original.l1_num_sets());
}
