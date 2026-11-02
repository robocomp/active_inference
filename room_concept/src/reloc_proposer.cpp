#include "reloc_proposer.h"

namespace rc::rivals
{
Proposer::Proposer()
{
    fn_ = [](const rc::reloc::LikelihoodRaster& r, const auto& poly, const auto& pts, const auto& seeds,
             const rc::reloc::Params& p) { return rc::reloc::search(r, poly, pts, seeds, p); };
    worker_ = std::jthread([this](std::stop_token st) { run(st); });
}

Proposer::~Proposer()
{
    worker_.request_stop();
    cv_.notify_all();
}

void Proposer::set_search_fn(SearchFn f)
{
    std::scoped_lock l(m_);
    fn_ = std::move(f);
}

bool Proposer::submit(std::uint64_t epoch, rc::reloc::LikelihoodRaster raster, std::vector<Eigen::Vector2f> polygon,
                      std::vector<Eigen::Vector2f> pts_robot, rc::reloc::Params params,
                      Eigen::Vector3f around, Eigen::Matrix3f sampling_cov)
{
    {
        std::scoped_lock l(m_);
        if (running_ or job_) return false;
        (void)sampling_cov;   // search() has no sampling-only hook; the pose prior belongs to birth_lambda, never here
        job_ = Job{epoch, std::move(raster), std::move(polygon), std::move(pts_robot), std::move(params), {around}};
        running_ = true;
        done_.reset();
    }
    cv_.notify_one();
    return true;
}

std::optional<std::pair<std::uint64_t, rc::reloc::Result>> Proposer::poll()
{
    std::scoped_lock l(m_);
    auto r = std::move(done_);
    done_.reset();
    return r;
}

bool Proposer::busy() const
{
    std::scoped_lock l(m_);
    return running_;
}

void Proposer::run(std::stop_token st)
{
    for (;;)
    {
        Job job;
        SearchFn fn;
        {
            std::unique_lock l(m_);
            if (not cv_.wait(l, st, [this] { return job_.has_value(); }) or st.stop_requested()) return;
            job = std::move(*job_);
            job_.reset();
            fn = fn_;
        }
        rc::reloc::Result res;
        try { res = fn(job.raster, job.polygon, job.pts, job.seeds, job.params); }
        catch (...) { res = {}; }
        std::scoped_lock l(m_);
        done_ = std::make_pair(job.epoch, std::move(res));
        running_ = false;
    }
}
}
