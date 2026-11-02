// Build: see src/reloc_proposer.h
#include "reloc_proposer.h"
#include <chrono>
#include <print>
#include <thread>
using namespace rc::rivals;
using namespace std::chrono;
static int failures = 0;
#define CHECK(c) do { if (not (c)) { ++failures; std::println("FAIL line {}: {}", __LINE__, #c); } } while (0)

static bool wait_poll(Proposer& p, std::optional<std::pair<std::uint64_t, rc::reloc::Result>>& out, int ms)
{
    for (int i = 0; i < ms; ++i) { if ((out = p.poll())) return true; std::this_thread::sleep_for(1ms); }
    return false;
}
static Proposer::SearchFn fake(int ms)
{
    return [ms](const rc::reloc::LikelihoodRaster&, const auto&, const auto&, const auto&, const rc::reloc::Params&)
    { std::this_thread::sleep_for(milliseconds(ms)); rc::reloc::Result r; r.n_evals = 7; return r; };
}
static bool sub(Proposer& p, std::uint64_t e)
{
    return p.submit(e, {}, {}, {}, {}, Eigen::Vector3f::Zero(), Eigen::Matrix3f::Identity());
}

int main()
{
    std::optional<std::pair<std::uint64_t, rc::reloc::Result>> out;
    {   // one_in_flight
        Proposer p; p.set_search_fn(fake(100));
        CHECK(sub(p, 1)); CHECK(not sub(p, 2)); CHECK(not p.poll());
        CHECK(wait_poll(p, out, 2000)); CHECK(out->first == 1 and out->second.n_evals == 7);
        CHECK(sub(p, 3)); CHECK(wait_poll(p, out, 2000));
    }
    {   // stale_epoch: result carries the submitted epoch, the caller compares
        Proposer p; p.set_search_fn(fake(20));
        std::uint64_t current = 1;
        CHECK(sub(p, current)); current = 2;
        CHECK(wait_poll(p, out, 2000)); CHECK(out->first == 1 and out->first != current);
        CHECK(not p.poll());   // delivered once
    }
    {   // never_blocks
        Proposer p; p.set_search_fn(fake(200));
        const auto t0 = steady_clock::now();
        CHECK(sub(p, 1)); const bool second = sub(p, 2); (void)p.poll(); (void)p.busy();
        const auto us = duration_cast<microseconds>(steady_clock::now() - t0).count();
        CHECK(not second); CHECK(us < 1000);
        std::println("submit+submit+poll took {} us", us);
    }   // destructor joins mid-search: must not hang
    {   // real search path on an empty job returns without crashing
        Proposer p; CHECK(sub(p, 5)); CHECK(wait_poll(p, out, 5000)); CHECK(out->first == 5);
    }
    {   // no_incumbent_prior: prior_modes reaches the search untouched
        Proposer p; std::size_t seen = 99; std::size_t nseeds = 0;
        p.set_search_fn([&](const rc::reloc::LikelihoodRaster&, const auto&, const auto&, const auto& seeds, const rc::reloc::Params& pr)
        { seen = pr.prior_modes.size(); nseeds = seeds.size(); return rc::reloc::Result{}; });
        CHECK(sub(p, 1)); CHECK(wait_poll(p, out, 2000)); CHECK(seen == 0 and nseeds == 1);
    }
    {   // exception clears running_
        Proposer p; p.set_search_fn([](const rc::reloc::LikelihoodRaster&, const auto&, const auto&, const auto&, const rc::reloc::Params&)
        -> rc::reloc::Result { throw 1; });
        CHECK(sub(p, 1)); CHECK(wait_poll(p, out, 2000)); CHECK(out->second.modes.empty()); CHECK(not p.busy()); CHECK(sub(p, 2));
    }
    std::println("{}", failures ? "FAILED" : "all passed");
    return failures != 0;
}
