#pragma once

#include "types.hpp"
#include "warp.hpp"
#include "trace_reader.hpp"
#include "nlohmann/json.hpp"
#include <vector>
#include <cstdint>

namespace gpu_sim {

struct SchedulerStats {
    uint64_t cycles_evaluated{0};
    uint64_t cycles_issued{0};
    uint64_t cycles_idle{0};
    uint64_t total_issues{0};
    uint64_t cumulative_eligible_warps{0};

    double scheduler_utilization() const {
        return cycles_evaluated == 0 ? 0.0 : static_cast<double>(cycles_issued) / static_cast<double>(cycles_evaluated);
    }

    double average_eligible_warps() const {
        return cycles_evaluated == 0 ? 0.0 : static_cast<double>(cumulative_eligible_warps) / static_cast<double>(cycles_evaluated);
    }

    nlohmann::json to_json() const {
        return {
            {"cycles_evaluated", cycles_evaluated},
            {"cycles_issued", cycles_issued},
            {"cycles_idle", cycles_idle},
            {"total_issues", total_issues},
            {"scheduler_utilization", scheduler_utilization()},
            {"average_eligible_warps", average_eligible_warps()}
        };
    }
};

class WarpScheduler {
public:
    WarpScheduler(SchedulingPolicy policy = SchedulingPolicy::ROUND_ROBIN, uint32_t issue_width = 1);

    // Determines if a warp can be issued on current_cycle
    bool is_eligible(const Warp& warp, const TraceReader& trace_reader, Cycle current_cycle) const;

    // Counts total eligible warps on current_cycle
    uint32_t count_eligible_warps(const std::vector<Warp>& warps,
                                  const TraceReader& trace_reader,
                                  Cycle current_cycle) const;

    // Selects up to issue_width warps according to the scheduling policy
    std::vector<WarpId> select_warps(const std::vector<Warp>& warps,
                                     const TraceReader& trace_reader,
                                     Cycle current_cycle);

    // Notifies the scheduler that a warp was issued
    void notify_issued(WarpId warp_id, Cycle current_cycle);

    // Accessors
    SchedulingPolicy policy() const { return policy_; }
    void set_policy(SchedulingPolicy policy) { policy_ = policy; }
    uint32_t issue_width() const { return issue_width_; }
    const SchedulerStats& stats() const { return stats_; }
    void reset_stats() { stats_ = SchedulerStats{}; }

private:
    SchedulingPolicy policy_;
    uint32_t issue_width_;

    // Round-Robin state
    uint32_t last_rr_index_{0};

    // GTO state
    WarpId last_issued_warp_{0};
    bool has_last_issued_{false};

    SchedulerStats stats_;

    std::vector<WarpId> select_round_robin(const std::vector<Warp>& warps,
                                           const TraceReader& trace_reader,
                                           Cycle current_cycle);

    std::vector<WarpId> select_gto(const std::vector<Warp>& warps,
                                   const TraceReader& trace_reader,
                                   Cycle current_cycle);
};

} // namespace gpu_sim
