#include "scheduler.hpp"
#include <algorithm>

namespace gpu_sim {

WarpScheduler::WarpScheduler(SchedulingPolicy policy, uint32_t issue_width)
    : policy_(policy), issue_width_(issue_width) {}

bool WarpScheduler::is_eligible(const Warp& warp, const TraceReader& trace_reader, Cycle current_cycle) const {
    return warp.is_eligible(current_cycle) && trace_reader.has_instruction(warp.id);
}

uint32_t WarpScheduler::count_eligible_warps(const std::vector<Warp>& warps,
                                             const TraceReader& trace_reader,
                                             Cycle current_cycle) const {
    uint32_t count = 0;
    for (const auto& warp : warps) {
        if (is_eligible(warp, trace_reader, current_cycle)) {
            count++;
        }
    }
    return count;
}

std::vector<WarpId> WarpScheduler::select_warps(const std::vector<Warp>& warps,
                                                const TraceReader& trace_reader,
                                                Cycle current_cycle) {
    stats_.cycles_evaluated++;
    uint32_t eligible_count = count_eligible_warps(warps, trace_reader, current_cycle);
    stats_.cumulative_eligible_warps += eligible_count;

    if (eligible_count == 0) {
        stats_.cycles_idle++;
        return {};
    }

    std::vector<WarpId> selected;
    if (policy_ == SchedulingPolicy::ROUND_ROBIN) {
        selected = select_round_robin(warps, trace_reader, current_cycle);
    } else {
        selected = select_gto(warps, trace_reader, current_cycle);
    }

    if (!selected.empty()) {
        stats_.cycles_issued++;
        stats_.total_issues += selected.size();
        for (WarpId wid : selected) {
            notify_issued(wid, current_cycle);
        }
    }

    return selected;
}

void WarpScheduler::notify_issued(WarpId warp_id, Cycle /*current_cycle*/) {
    last_issued_warp_ = warp_id;
    has_last_issued_ = true;
}

std::vector<WarpId> WarpScheduler::select_round_robin(const std::vector<Warp>& warps,
                                                      const TraceReader& trace_reader,
                                                      Cycle current_cycle) {
    std::vector<WarpId> selected;
    if (warps.empty()) return selected;

    size_t n = warps.size();
    for (size_t i = 1; i <= n; ++i) {
        size_t idx = (last_rr_index_ + i) % n;
        const auto& warp = warps[idx];
        if (is_eligible(warp, trace_reader, current_cycle)) {
            selected.push_back(warp.id);
            last_rr_index_ = static_cast<uint32_t>(idx);
            if (selected.size() == issue_width_) {
                break;
            }
        }
    }

    return selected;
}

std::vector<WarpId> WarpScheduler::select_gto(const std::vector<Warp>& warps,
                                              const TraceReader& trace_reader,
                                              Cycle current_cycle) {
    std::vector<WarpId> selected;
    if (warps.empty()) return selected;

    // 1. Greedy Phase: If the last issued warp is still eligible, keep issuing it!
    if (has_last_issued_) {
        // Find the warp in the pool
        for (const auto& warp : warps) {
            if (warp.id == last_issued_warp_) {
                if (is_eligible(warp, trace_reader, current_cycle)) {
                    selected.push_back(warp.id);
                }
                break;
            }
        }
    }

    if (selected.size() == issue_width_) {
        return selected;
    }

    // 2. Oldest Phase: Select the oldest eligible warp (minimum last_issued_cycle)
    struct Candidate {
        WarpId id;
        Cycle last_issued;
    };
    std::vector<Candidate> candidates;

    for (const auto& warp : warps) {
        // Skip if already selected in the greedy step
        if (!selected.empty() && warp.id == selected.front()) {
            continue;
        }
        if (is_eligible(warp, trace_reader, current_cycle)) {
            candidates.push_back({warp.id, warp.last_issued_cycle});
        }
    }

    // Sort: smallest last_issued_cycle first (oldest). Tie-break by smallest WarpId
    std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        if (a.last_issued != b.last_issued) {
            return a.last_issued < b.last_issued;
        }
        return a.id < b.id;
    });

    for (const auto& cand : candidates) {
        selected.push_back(cand.id);
        if (selected.size() == issue_width_) {
            break;
        }
    }

    return selected;
}

} // namespace gpu_sim
