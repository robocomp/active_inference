/*
 * rt_memo.h — the per-tick memo of get_transformation_matrix queries (see tick_memo.h).
 *
 * rc::g_rt_memo is set by SpecificWorker::compute() for the duration of one tick and null otherwise; with it null
 * every query goes straight to cortex (counted in rt_calls). With it set, an identical (from,to,ts) query inside the
 * tick is answered from the memo (counted in rt_hits) — the whole tick sees one pose for ts=0 queries.
 */
#pragma once

#include "tick_memo.h"
#include "tick_timing.h"

#include <cstdint>
#include <string>

#include <Eigen/Dense>
#include <dsr/api/dsr_inner_eigen_api.h>

namespace rc
{
struct RtMemo
{
    TickMemo<Mat::RTMat>   rt;          // (from|to|ts) -> transform
    TickMemo<Eigen::Matrix4d> room_zed; // (room frame|pose ts) -> camera->room
    void clear() { rt.clear(); room_zed.clear(); }
};

inline thread_local RtMemo* g_rt_memo = nullptr;

inline void count_rt_hit() { if (g_tick_timing) ++g_tick_timing->rt_hits; }

// get_transformation_matrix(from, to, ts) memoised for the current tick. Same result as the direct call.
template <class Api>   // raw pointer or unique_ptr<InnerEigenAPI>
inline std::optional<Mat::RTMat> cached_rt(const Api& api_in, const std::string& from, const std::string& to,
                                           std::uint64_t ts = 0)
{
    DSR::InnerEigenAPI* const api = &*api_in;
    const auto direct = [&]
    {
        count_rt_call();
        return api->get_transformation_matrix(from, to, ts);
    };
    if (g_rt_memo == nullptr)
        return direct();
    bool hit = false;
    const auto r = g_rt_memo->rt.get(from + '|' + to + '|' + std::to_string(ts), direct, hit);
    if (hit) count_rt_hit();
    return r;
}

// Clears the memo and activates it for one tick; deactivates on scope exit (every return path of compute()).
class RtMemoTickScope
{
public:
    explicit RtMemoTickScope(RtMemo& m) : prev_(g_rt_memo) { m.clear(); g_rt_memo = &m; }
    ~RtMemoTickScope() { g_rt_memo = prev_; }
    RtMemoTickScope(const RtMemoTickScope&) = delete;
    RtMemoTickScope& operator=(const RtMemoTickScope&) = delete;
private:
    RtMemo* prev_;
};
}  // namespace rc
