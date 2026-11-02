// reloc_proposer.h — background PROPOSER of rival poses (spec 2026-11-01-relocaliser-redesign §4.5, §7).
// It only PROPOSES: it never moves the pose. One search in flight, one job slot, everything deep-copied in, the worker
// touches nothing but its copies (no DSR, no Qt, no ts==0 transforms). The compute thread calls submit()/poll() and
// drops a result whose epoch differs from the current room epoch.
// Standalone test build (from room_concept/); add -fsanitize=thread for the TSAN run:
//   g++ -std=c++23 -O1 -g -Isrc -I/usr/include/eigen3 tools/reloc_proposer_test.cpp src/reloc_proposer.cpp src/reloc_search.cpp src/wall_segmenter.cpp -pthread -o /tmp/reloc_proposer_test && /tmp/reloc_proposer_test
#pragma once
#include "reloc_search.h"
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace rc::rivals
{
class Proposer
{
public:
    /// The search the worker runs. Default: rc::reloc::search. Test hook (set before the first submit).
    using SearchFn = std::function<rc::reloc::Result(const rc::reloc::LikelihoodRaster&,
                                                     const std::vector<Eigen::Vector2f>& polygon,
                                                     const std::vector<Eigen::Vector2f>& pts_robot,
                                                     const std::vector<Eigen::Vector3f>& seeds,
                                                     const rc::reloc::Params&)>;

    Proposer();
    ~Proposer();
    Proposer(const Proposer&) = delete;
    Proposer& operator=(const Proposer&) = delete;

    void set_search_fn(SearchFn f);   ///< test hook; call while idle

    /// Copy-in; false if a search is already RUNNING. `epoch` = room-generation counter. `around` seeds the search ONLY; it is
    /// never added as a prior mode (a continuity prior would bias proposals toward the pose being challenged and hide a
    /// kidnap; that prior belongs to birth_lambda). `sampling_cov` is currently unused: search() has no sampling-only hook.
    /// An unpolled older result is replaced by the new job's.
    bool submit(std::uint64_t epoch, rc::reloc::LikelihoodRaster raster, std::vector<Eigen::Vector2f> polygon,
                std::vector<Eigen::Vector2f> pts_robot, rc::reloc::Params params,
                Eigen::Vector3f around, Eigen::Matrix3f sampling_cov);

    /// Non-blocking. Returns a finished result once (epoch, Result); the caller drops it if epoch != current.
    std::optional<std::pair<std::uint64_t, rc::reloc::Result>> poll();

    bool busy() const;

private:
    struct Job
    {
        std::uint64_t epoch = 0;
        rc::reloc::LikelihoodRaster raster;
        std::vector<Eigen::Vector2f> polygon, pts;
        rc::reloc::Params params;
        std::vector<Eigen::Vector3f> seeds;
    };
    void run(std::stop_token st);

    mutable std::mutex m_;
    std::condition_variable_any cv_;
    std::optional<Job> job_;
    bool running_ = false;
    std::optional<std::pair<std::uint64_t, rc::reloc::Result>> done_;
    SearchFn fn_;
    std::jthread worker_;   // last member: starts after the rest is built, joins first
};
}
