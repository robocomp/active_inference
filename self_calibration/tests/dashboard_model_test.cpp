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
    // no helios sweep yet: the pair is assumed to end "now" (wall 10 s)
    straight.stamp_a_ms = 1000; straight.stamp_b_ms = 3000;
    turn.stamp_a_ms = 5000;     turn.stamp_b_ms = 6000;
    m.note_lidar_stamp(6000, 10000);            // offset = -4000: helios 5000..6000 -> wall 9000..10000
    m.add_pair(straight, 10000);                // helios 1000..3000 -> wall 5000..7000
    m.add_pair(turn, 10000);
    CHECK(m.pairs_total(dash::Label::yaw) == 1 and m.pairs_total(dash::Label::lever) == 1);
    CHECK(m.pairs_recent() == 2);

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

    // a fragment converted with one offset STAYS where it was placed when the offset changes (sim vs wall rate)
    CHECK(m.fragments().size() == 2 and m.fragments()[0].t0_ms == 5000 and m.fragments()[0].t1_ms == 7000);
    m.note_lidar_stamp(6000, 9000);             // offset now -3000 (sim clock slipped)
    CHECK(m.fragments()[0].t0_ms == 5000 and m.fragments()[0].t1_ms == 7000);
    auto moved = turn; moved.stamp_a_ms = 7000; moved.stamp_b_ms = 8000;
    m.add_pair(moved, 11000);                   // new offset: wall 10000..11000
    CHECK(m.fragments().back().t0_ms == 10000 and m.fragments().back().t1_ms == 11000);
    CHECK(m.fragments()[0].t0_ms == 5000);

    // rolling 60 s: a fragment drops out of the count after 60 s, the total keeps it
    CHECK(m.pairs_recent() == 3 and m.pairs_total() == 3);
    dash::DashboardModel q(pr);
    q.note_lidar_stamp(0, 0);
    auto one = turn; one.stamp_a_ms = 0; one.stamp_b_ms = 1000;
    q.add_pair(one, 1000);
    CHECK(q.pairs_recent(dash::Label::lever) == 1);
    q.sample_stall(1000 + 59000, false);
    CHECK(q.pairs_recent() == 1);
    q.sample_stall(1000 + 61000, false);
    CHECK(q.pairs_recent() == 0 and q.pairs_total() == 1 and q.fragments().size() == 1);

    // pruning: older than 5 min behind the newest wall time goes; totals stay
    auto late = turn; late.stamp_a_ms = 400000; late.stamp_b_ms = 401000;
    m.note_lidar_stamp(401000, 401000);
    m.add_pair(late, 401000);                   // cut = 101 s: everything before is pruned
    CHECK(m.fragments().size() == 1 and m.fragments().front().t0_ms == 400000);
    CHECK(m.pairs_total() == 4);

    // a backward wall-clock step resets the timeline
    m.sample_stall(1000, false);
    CHECK(m.timeline_resets() == 1 and m.fragments().empty() and m.pairs_total() == 4);

    // stream rows: media + odometry + pose present; never-seen = waiting (not stalled); rates over >= 1 s windows
    {
        dash::StreamRates sr;
        auto samples = [](std::uint64_t t) {
            return std::vector<dash::StreamSample>{{"helios", t, true, false, true}, {"zed/rgb", 0, false, false, true},
                                                   {"imu", 5, true, true, true}, {"odom", 2 * t, true, false, true},
                                                   {"pose", t, true, false, true}, {"polygon", 4, true, false, false}};
        };
        (void)sr.update(0, samples(0));
        const auto early = sr.update(500, samples(5));          // < 1 s: window not rolled
        CHECK(early.size() == 6);
        const auto rows = sr.update(2000, samples(20));         // 20 arrivals in 2 s
        auto find = [&](const char* k) { for (const auto& x : rows) if (x.key == k) return x; return dash::StreamRow{}; };
        CHECK(find("odom").key == "odom" and find("pose").key == "pose" and find("polygon").key == "polygon");
        CHECK(std::abs(find("helios").rate_hz - 10.0) < 1e-9 and std::abs(find("odom").rate_hz - 20.0) < 1e-9);
        CHECK(find("zed/rgb").status == dash::StreamStatus::waiting);
        CHECK(find("imu").status == dash::StreamStatus::stalled);
        CHECK(find("helios").status == dash::StreamStatus::live);
        CHECK(not find("polygon").show_rate);
    }

    // stall intervals on the helios clock via the wall offset
    dash::DashboardModel s(pr);
    s.sample_stall(100, false);
    s.sample_stall(200, false);
    s.sample_stall(300, true); s.sample_stall(400, true); s.sample_stall(500, true);
    s.sample_stall(600, false);
    s.sample_stall(700, true); s.sample_stall(800, true);
    CHECK(s.stalls().size() == 2);
    CHECK(s.stalls()[0].t0_ms == 300 and s.stalls()[0].t1_ms == 500);   // wall-clock intervals
    CHECK(s.stalls()[1].t0_ms == 700 and s.stalls()[1].t1_ms == 800);
    // old stall intervals are pruned too
    s.sample_stall(400900, false);
    CHECK(s.stalls().empty());

    std::printf(fails ? "dashboard_model_test: %d FAILED\n" : "dashboard_model_test: all passed\n", fails);
    return fails ? 1 : 0;
}
