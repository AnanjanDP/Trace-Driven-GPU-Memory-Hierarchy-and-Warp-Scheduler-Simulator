#include "catch.hpp"
#include "cache.hpp"

using namespace gpu_sim;

TEST_CASE("Cache Bit Decomposition and Indexing", "[cache]") {
    // 1 KB capacity, 64-byte line, 2-way associative, 32-byte sectors
    // num_sets = 1024 / (64 * 2) = 8 sets
    // offset_bits = 6, index_bits = 3, tag_bits = 55
    Cache cache("TestL1", 1024, 64, 2, 32, ReplacementPolicy::LRU, 4);

    REQUIRE(cache.num_sets() == 8);
    REQUIRE(cache.sectors_per_line() == 2);

    Address addr = 0x1A2B; // 6699 in decimal
    // offset = 6699 & 0x3F = 43 (0x2B)
    REQUIRE(cache.get_block_offset(addr) == 43);
    // sector_index = 43 / 32 = 1
    REQUIRE(cache.get_sector_index(addr) == 1);
    // set_index = (6699 >> 6) & 0x7 = 104 & 7 = 0
    REQUIRE(cache.get_set_index(addr) == 0);
    // tag = 6699 >> 9 = 13 (0xD)
    REQUIRE(cache.get_tag(addr) == 13);
    // line base address = addr & ~0x3F = 6699 - 43 = 6656 (0x1A00)
    REQUIRE(cache.get_line_address(addr) == 0x1A00);
}

TEST_CASE("Cache Compulsory Misses and Hits", "[cache]") {
    Cache cache("TestL1", 1024, 64, 2, 32, ReplacementPolicy::LRU, 4);

    Address addr1 = 0x1000;
    Address addr2 = 0x1004; // Same cache line as addr1

    // 1st access: Cold/compulsory miss
    auto res1 = cache.access(addr1, /*is_write=*/false);
    REQUIRE(res1.hit == false);
    REQUIRE(res1.evicted == false);
    REQUIRE(cache.stats().accesses == 1);
    REQUIRE(cache.stats().read_misses == 1);
    REQUIRE(cache.stats().read_hits == 0);

    // 2nd access: Same address -> Hit
    auto res2 = cache.access(addr1, /*is_write=*/false);
    REQUIRE(res2.hit == true);
    REQUIRE(res2.sector_hit == true);
    REQUIRE(cache.stats().accesses == 2);
    REQUIRE(cache.stats().read_hits == 1);
    REQUIRE(cache.stats().hit_rate() == 0.5);

    // 3rd access: Same cache line, same sector -> Hit
    auto res3 = cache.access(addr2, /*is_write=*/false);
    REQUIRE(res3.hit == true);
    REQUIRE(res3.sector_hit == true);
    REQUIRE(cache.stats().accesses == 3);
    REQUIRE(cache.stats().read_hits == 2);
    REQUIRE(cache.stats().hit_rate() == Approx(2.0 / 3.0));
}

TEST_CASE("Cache Replacement Policy: LRU vs FIFO Divergence", "[cache]") {
    // 1 KB capacity, 64-byte line, 2-way associative -> 8 sets (offset=6, index=3)
    // All 3 addresses map to Set 0:
    Address addrA = (1ULL << 9); // tag=1, set=0
    Address addrB = (2ULL << 9); // tag=2, set=0
    Address addrC = (3ULL << 9); // tag=3, set=0

    SECTION("LRU Policy: Replaces Least Recently Used Block") {
        Cache cache_lru("L1_LRU", 1024, 64, 2, 32, ReplacementPolicy::LRU, 4);

        // Fill Set 0 with A and B
        auto rA1 = cache_lru.access(addrA, false); // Miss, installed
        auto rB1 = cache_lru.access(addrB, false); // Miss, installed
        REQUIRE(rA1.hit == false);
        REQUIRE(rB1.hit == false);

        // Access A again -> Hit! Now A is the MOST recently used, B is LEAST recently used
        auto rA2 = cache_lru.access(addrA, false);
        REQUIRE(rA2.hit == true);

        // Access C -> Set is full! Must evict LRU block (which is B)
        auto rC1 = cache_lru.access(addrC, false);
        REQUIRE(rC1.hit == false);
        REQUIRE(rC1.evicted == true);
        REQUIRE(rC1.evicted_addr == addrB); // B was evicted!

        // Access A again -> Under LRU, A is still in the cache -> HIT!
        auto rA3 = cache_lru.access(addrA, false);
        REQUIRE(rA3.hit == true);

        // Access B -> B was evicted, so it must be a MISS
        auto rB2 = cache_lru.access(addrB, false);
        REQUIRE(rB2.hit == false);
    }

    SECTION("FIFO Policy: Replaces First-In Block Regardless of Accesses") {
        Cache cache_fifo("L1_FIFO", 1024, 64, 2, 32, ReplacementPolicy::FIFO, 4);

        // Fill Set 0 with A (inserted 1st) and B (inserted 2nd)
        cache_fifo.access(addrA, false);
        cache_fifo.access(addrB, false);

        // Access A again -> Hit!
        // Under FIFO, access does NOT refresh insertion time; A remains oldest inserted!
        auto rA2 = cache_fifo.access(addrA, false);
        REQUIRE(rA2.hit == true);

        // Access C -> Set is full! FIFO must evict oldest inserted (which is A)
        auto rC1 = cache_fifo.access(addrC, false);
        REQUIRE(rC1.hit == false);
        REQUIRE(rC1.evicted == true);
        REQUIRE(rC1.evicted_addr == addrA); // A was evicted under FIFO! Set now holds [C, B]

        // Access B -> In FIFO, B was NOT evicted -> HIT!
        auto rB2 = cache_fifo.access(addrB, false);
        REQUIRE(rB2.hit == true);

        // Access A -> A was evicted in the previous step -> MISS!
        auto rA3 = cache_fifo.access(addrA, false);
        REQUIRE(rA3.hit == false);
        REQUIRE(rA3.evicted == true);
        REQUIRE(rA3.evicted_addr == addrB); // B was oldest in [C, B], so A evicts B!
    }
}

TEST_CASE("Cache Dirty Line Tracking and Writebacks", "[cache]") {
    // 1024 bytes, 64-byte line, 1-way (direct mapped) -> 16 sets
    // offset_bits = 6, index_bits = 4 -> tag begins at bit 10 (1024)
    Cache cache("TestL1", 1024, 64, 1, 32, ReplacementPolicy::LRU, 4);

    Address addrA = 0x000; // set 0, tag 0
    Address addrB = 0x400; // set 0, tag 1 (conflicts with A in set 0)

    REQUIRE(cache.get_set_index(addrA) == 0);
    REQUIRE(cache.get_set_index(addrB) == 0);
    REQUIRE(cache.get_tag(addrA) != cache.get_tag(addrB));

    // Write to A -> Miss, allocates dirty line
    auto r1 = cache.access(addrA, /*is_write=*/true);
    REQUIRE(r1.hit == false);
    REQUIRE(cache.stats().write_misses == 1);

    // Read B -> Conflicts with A in set 0, causes eviction of dirty line A
    auto r2 = cache.access(addrB, /*is_write=*/false);
    REQUIRE(r2.hit == false);
    REQUIRE(r2.evicted == true);
    REQUIRE(r2.evicted_dirty == true);
    REQUIRE(r2.evicted_addr == addrA);
    REQUIRE(cache.stats().evictions == 1);
    REQUIRE(cache.stats().writebacks == 1);
}

TEST_CASE("Cache Sector Granularity Tracking", "[cache]") {
    // 64B line, 32B sector -> 2 sectors per line
    Cache cache("TestL1", 1024, 64, 2, 32, ReplacementPolicy::LRU, 4);

    Address addr_sec0 = 0x1000;      // offset 0 -> sector 0
    Address addr_sec1 = 0x1000 + 32; // offset 32 -> sector 1 (same 64B cache line)

    // 1st access to sector 0: line miss, sector miss
    auto r1 = cache.access(addr_sec0, false);
    REQUIRE(r1.hit == false);
    REQUIRE(r1.sector_hit == false);

    // 2nd access to sector 1 of the same line: line HIT, but sector MISS (new sector)
    auto r2 = cache.access(addr_sec1, false);
    REQUIRE(r2.hit == true);
    REQUIRE(r2.sector_hit == false);

    // 3rd access to sector 1 again: line HIT and sector HIT
    auto r3 = cache.access(addr_sec1, false);
    REQUIRE(r3.hit == true);
    REQUIRE(r3.sector_hit == true);
}
