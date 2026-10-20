#include "cache.hpp"
#include <limits>
#include <stdexcept>

namespace gpu_sim {

namespace {

uint32_t log2_u64(uint64_t n) {
    uint32_t bits = 0;
    while ((n >>= 1) > 0) {
        bits++;
    }
    return bits;
}

bool is_power_of_two(uint64_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

} // anonymous namespace

Cache::Cache(const std::string& name,
             uint64_t capacity_bytes,
             uint32_t line_size_bytes,
             uint32_t associativity,
             uint32_t sector_size_bytes,
             ReplacementPolicy replacement_policy,
             Cycle hit_latency)
    : name_(name),
      capacity_bytes_(capacity_bytes),
      line_size_bytes_(line_size_bytes),
      associativity_(associativity),
      sector_size_bytes_(sector_size_bytes),
      replacement_policy_(replacement_policy),
      hit_latency_(hit_latency) {

    if (!is_power_of_two(capacity_bytes_)) {
        throw std::invalid_argument(name_ + ": capacity_bytes must be a power of 2");
    }
    if (!is_power_of_two(line_size_bytes_)) {
        throw std::invalid_argument(name_ + ": line_size_bytes must be a power of 2");
    }
    if (!is_power_of_two(associativity_)) {
        throw std::invalid_argument(name_ + ": associativity must be a power of 2");
    }
    if (!is_power_of_two(sector_size_bytes_)) {
        throw std::invalid_argument(name_ + ": sector_size_bytes must be a power of 2");
    }
    if (line_size_bytes_ < sector_size_bytes_ || (line_size_bytes_ % sector_size_bytes_) != 0) {
        throw std::invalid_argument(name_ + ": line_size_bytes must be a multiple of sector_size_bytes");
    }

    uint64_t min_capacity = static_cast<uint64_t>(line_size_bytes_) * associativity_;
    if (capacity_bytes_ < min_capacity) {
        throw std::invalid_argument(name_ + ": capacity_bytes must be >= line_size_bytes * associativity");
    }

    sectors_per_line_ = line_size_bytes_ / sector_size_bytes_;
    num_sets_ = capacity_bytes_ / (static_cast<uint64_t>(line_size_bytes_) * associativity_);

    offset_bits_ = log2_u64(line_size_bytes_);
    index_bits_ = log2_u64(num_sets_);

    offset_mask_ = (1ULL << offset_bits_) - 1ULL;
    index_mask_ = (index_bits_ == 0) ? 0 : ((1ULL << index_bits_) - 1ULL);

    sets_.resize(num_sets_);
    for (auto& set : sets_) {
        set.lines.resize(associativity_);
    }
}

Address Cache::get_tag(Address addr) const {
    return addr >> (offset_bits_ + index_bits_);
}

uint32_t Cache::get_set_index(Address addr) const {
    if (index_bits_ == 0) return 0;
    return static_cast<uint32_t>((addr >> offset_bits_) & index_mask_);
}

uint32_t Cache::get_block_offset(Address addr) const {
    return static_cast<uint32_t>(addr & offset_mask_);
}

uint32_t Cache::get_sector_index(Address addr) const {
    return get_block_offset(addr) / sector_size_bytes_;
}

Address Cache::get_line_address(Address addr) const {
    return addr & ~offset_mask_;
}

void Cache::flush() {
    for (auto& set : sets_) {
        for (auto& line : set.lines) {
            line = CacheLine{};
        }
    }
    access_counter_ = 0;
}

uint32_t Cache::find_victim_way(uint32_t set_idx) const {
    const auto& lines = sets_[set_idx].lines;
    uint32_t victim_way = 0;

    if (replacement_policy_ == ReplacementPolicy::LRU) {
        uint64_t min_access_time = std::numeric_limits<uint64_t>::max();
        for (uint32_t w = 0; w < associativity_; ++w) {
            if (lines[w].last_access_time < min_access_time) {
                min_access_time = lines[w].last_access_time;
                victim_way = w;
            }
        }
    } else { // FIFO
        uint64_t min_insertion_time = std::numeric_limits<uint64_t>::max();
        for (uint32_t w = 0; w < associativity_; ++w) {
            if (lines[w].insertion_time < min_insertion_time) {
                min_insertion_time = lines[w].insertion_time;
                victim_way = w;
            }
        }
    }

    return victim_way;
}

bool Cache::probe(Address addr, bool is_write) {
    uint32_t set_idx = get_set_index(addr);
    Address tag = get_tag(addr);
    uint32_t sector_idx = get_sector_index(addr);

    for (uint32_t w = 0; w < associativity_; ++w) {
        auto& line = sets_[set_idx].lines[w];
        if (line.valid && line.tag == tag) {
            // Tag matches
            line.last_access_time = ++access_counter_;
            line.sector_mask |= (1U << sector_idx);
            if (is_write) {
                line.dirty = true;
            }
            return true;
        }
    }
    return false;
}

CacheAccessResult Cache::fill(Address addr, bool is_write) {
    uint32_t set_idx = get_set_index(addr);
    Address tag = get_tag(addr);
    uint32_t sector_idx = get_sector_index(addr);

    CacheAccessResult result;
    result.set_index = set_idx;

    // First check if already present (e.g. coalesced fill or existing line)
    for (uint32_t w = 0; w < associativity_; ++w) {
        auto& line = sets_[set_idx].lines[w];
        if (line.valid && line.tag == tag) {
            result.hit = true;
            result.sector_hit = (line.sector_mask & (1U << sector_idx)) != 0;
            result.way_index = w;
            line.sector_mask |= (1U << sector_idx);
            line.last_access_time = ++access_counter_;
            if (is_write) line.dirty = true;
            return result;
        }
    }

    // Line not present: check for an empty way
    for (uint32_t w = 0; w < associativity_; ++w) {
        auto& line = sets_[set_idx].lines[w];
        if (!line.valid) {
            line.valid = true;
            line.dirty = is_write;
            line.tag = tag;
            line.sector_mask = (1U << sector_idx);
            line.last_access_time = ++access_counter_;
            line.insertion_time = access_counter_;
            result.hit = false;
            result.evicted = false;
            result.way_index = w;
            return result;
        }
    }

    // Set is full: evict victim
    uint32_t victim_way = find_victim_way(set_idx);
    auto& victim = sets_[set_idx].lines[victim_way];

    result.hit = false;
    result.evicted = true;
    result.evicted_dirty = victim.dirty;
    result.evicted_addr = (victim.tag << (offset_bits_ + index_bits_)) | (static_cast<Address>(set_idx) << offset_bits_);
    result.way_index = victim_way;

    stats_.evictions++;
    if (victim.dirty) {
        stats_.writebacks++;
    }

    // Install new line into victim way
    victim.valid = true;
    victim.dirty = is_write;
    victim.tag = tag;
    victim.sector_mask = (1U << sector_idx);
    victim.last_access_time = ++access_counter_;
    victim.insertion_time = access_counter_;

    return result;
}

CacheAccessResult Cache::access(Address addr, bool is_write) {
    stats_.accesses++;
    if (is_write) {
        stats_.writes++;
    } else {
        stats_.reads++;
    }

    uint32_t set_idx = get_set_index(addr);
    Address tag = get_tag(addr);
    uint32_t sector_idx = get_sector_index(addr);

    CacheAccessResult result;
    result.set_index = set_idx;

    // Check for hit
    for (uint32_t w = 0; w < associativity_; ++w) {
        auto& line = sets_[set_idx].lines[w];
        if (line.valid && line.tag == tag) {
            result.hit = true;
            result.sector_hit = (line.sector_mask & (1U << sector_idx)) != 0;
            result.way_index = w;

            line.sector_mask |= (1U << sector_idx);
            line.last_access_time = ++access_counter_;
            if (is_write) {
                line.dirty = true;
                stats_.write_hits++;
            } else {
                stats_.read_hits++;
            }
            return result;
        }
    }

    // Miss occurred
    if (is_write) {
        stats_.write_misses++;
    } else {
        stats_.read_misses++;
    }

    // Allocate in empty way or evict
    for (uint32_t w = 0; w < associativity_; ++w) {
        auto& line = sets_[set_idx].lines[w];
        if (!line.valid) {
            line.valid = true;
            line.dirty = is_write;
            line.tag = tag;
            line.sector_mask = (1U << sector_idx);
            line.last_access_time = ++access_counter_;
            line.insertion_time = access_counter_;

            result.hit = false;
            result.evicted = false;
            result.way_index = w;
            return result;
        }
    }

    // All ways valid: evict victim
    uint32_t victim_way = find_victim_way(set_idx);
    auto& victim = sets_[set_idx].lines[victim_way];

    result.hit = false;
    result.evicted = true;
    result.evicted_dirty = victim.dirty;
    result.evicted_addr = (victim.tag << (offset_bits_ + index_bits_)) | (static_cast<Address>(set_idx) << offset_bits_);
    result.way_index = victim_way;

    stats_.evictions++;
    if (victim.dirty) {
        stats_.writebacks++;
    }

    victim.valid = true;
    victim.dirty = is_write;
    victim.tag = tag;
    victim.sector_mask = (1U << sector_idx);
    victim.last_access_time = ++access_counter_;
    victim.insertion_time = access_counter_;

    return result;
}

} // namespace gpu_sim
