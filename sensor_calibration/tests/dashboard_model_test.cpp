// dashboard_model_test -- pure aggregation behind the dashboard (no Qt, no DDS). make -C build dashboard_model_test
#include "../src/dashboard_model.h"

#include <cmath>
#include <cstdio>

static int fails = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main()
{
    using namespace sc;
    calib::HarvesterConfig cfg;
    const dash::Priors pr{cfg.prior_sigma_xy * cfg.prior_sigma_xy, cfg.prior_sigma_yaw * cfg.prior_sigma_yaw};

    // straight-only: dth = 0, chord forward 1 m  -> only the yaw row lives
    auto straight = calib::make_pair_from_motion(0.0, 1.0, 0.0, 1.0, 0.0, cfg);
    // pure turn: dth = pi/2, chord ~0 (in place) -> lever rows
    auto turn = calib::make_pair_from_motion(M_PI / 2, 0.0, 0.0, 0.0, M_PI / 2, cfg);
    CHECK(dash::classify(straight.info, pr) == dash::Label::yaw);
    CHECK(dash::classify(turn.info, pr) == dash::Label::lever);

    dash::DashboardModel m(pr);
    straight.stamp_a_ms = 1000; straight.stamp_b_ms = 3000;
    turn.stamp_a_ms = 5000;     turn.stamp_b_ms = 6000;
    m.add_pair(straight);
    m.add_pair(turn);
    CHECK(m.pairs_total(dash::Label::yaw) == 1 and m.pairs_total(dash::Label::lever) == 1);
    CHECK(m.pairs_episode() == 2);
    m.close_episode();
    CHECK(m.pairs_episode() == 0 and m.pairs_total() == 2);

    // ratios = accumulated info diagonal * prior var
    const auto r = m.ratios();
    const Eigen::Matrix3d S = straight.info + turn.info;
    CHECK(std::abs(r[0] - S(0, 0) * pr.var_xy) < 1e-9);
    CHECK(std::abs(r[1] - S(1, 1) * pr.var_xy) < 1e-9);
    CHECK(std::abs(r[2] - S(2, 2) * pr.var_yaw) < 1e-9);
    CHECK(r[2] > 0.0 and r[0] > 0.0);

    // keyframe counters
    m.note_keyframe(true, false); m.note_keyframe(false, false); m.note_keyframe(false, true);
    CHECK(m.keyframes() == 1 and m.dropped_stalled() == 1);

    // pruning: a fragment older than 5 min behind the newest time goes, a recent one stays
    auto late = turn; late.stamp_a_ms = 400000; late.stamp_b_ms = 401000;   // 401 s: cut = 101 s
    m.add_pair(late);
    CHECK(m.fragments().size() == 1 and m.fragments().front().t0_ms == 400000);   // [1,3]s and [5,6]s are older than 101 s: pruned
    auto later = turn; later.stamp_a_ms = 700000; later.stamp_b_ms = 701000;  // cut = 401 s
    m.add_pair(later);
    CHECK(m.fragments().size() == 2 and m.fragments().front().t1_ms == 401000);
    auto latest = turn; latest.stamp_a_ms = 701000; latest.stamp_b_ms = 702000;
    m.add_pair(latest);
    CHECK(m.fragments().size() == 2 and m.fragments().front().t0_ms == 700000);   // newest 702 s, cut 402 s: the 401 s one goes
    // totals are cumulative, not pruned
    CHECK(m.pairs_total() == 5);

    // stall intervals on the helios clock via the wall offset
    dash::DashboardModel s(pr);
    s.sample_stall(100, true);                  // no offset yet: ignored
    CHECK(s.stalls().empty());
    s.note_lidar_stamp(10000, 100);             // offset = 9900
    s.sample_stall(200, false);
    s.sample_stall(300, true); s.sample_stall(400, true); s.sample_stall(500, true);
    s.sample_stall(600, false);
    s.sample_stall(700, true); s.sample_stall(800, true);
    CHECK(s.stalls().size() == 2);
    CHECK(s.stalls()[0].t0_ms == 10200 and s.stalls()[0].t1_ms == 10400);   // wall 300..500 -> helios 10200..10400
    CHECK(s.stalls()[1].t0_ms == 10600 and s.stalls()[1].t1_ms == 10700);
    // old stall intervals are pruned too
    s.note_lidar_stamp(10000 + 400000, 900);
    s.sample_stall(1000, false);
    CHECK(s.stalls().empty());

    std::printf(fails ? "dashboard_model_test: %d FAILED\n" : "dashboard_model_test: all passed\n", fails);
    return fails ? 1 : 0;
}
