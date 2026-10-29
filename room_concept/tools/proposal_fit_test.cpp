// Task 6 fix round (room-transition protocol C2.7): judging ltsm's proposal must cost the survey NOTHING.
// RoomConcept driven through update(), surveying a 6 x 4 m room with plain SLAM (as a room entered through a door is).
// A proposal's layout K is fitted (begin_hypothesis_fit = the late-adoption fit, snapshot first). Asserted:
//   * while the fit runs, the pose the agent PUBLISHES (survey_pose_during_fit → PosePublisher) stays the survey's,
//     even when the fit's own pose is dragged metres away by a wrong prior;
//   * a REJECTED fit and a WITHDRAWN one leave pose, WallMap, map_ready, derived polygon, layout flags and the learnt
//     tracking emission exactly as they were;
//   * the true layout is still ACCEPTED and kept.
//   make -C build proposal_fit_test && ../bin/proposal_fit_test
#include "room_concept.h"
#include "reloc_search.h"
#include "published_pose.h"

#include <clocale>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <random>
#include <string>
#include <vector>

namespace rc
{
struct RelocSelftestAccess   // friend of RoomConcept (one per test binary)
{
    static std::size_t walls(RoomConcept& r) { return r.wall_map_.walls.size(); }
    static bool map_ready_raw(RoomConcept& r) { return r.map_ready_.load(); }
    static float track_mu(RoomConcept& r) { return r.track_mu_ln_s_; }
    static float track_var(RoomConcept& r) { return r.track_var_ln_s_; }
    static void evidence(RoomConcept& r, float s) { r.accumulate_late_evidence(s, 1.f); }
    static double log_odds(RoomConcept& r) { return r.late_log_odds_; }
    // The agent's run loop calls relocalisation_step after update() (room_concept.cpp, step 8); update() alone does not.
    static std::uint64_t epoch(RoomConcept& r) { return r.reloc_epoch_.load(); }
    static void reloc(RoomConcept& r, const RoomConcept::UpdateResult& res, const std::vector<Eigen::Vector3f>& pts)
    { r.relocalisation_step(res, pts); }
};
}   // namespace rc
using Acc = rc::RelocSelftestAccess;
using rc::RoomConcept;
using Poly = std::vector<Eigen::Vector2f>;

static int failures = 0;
static void check(bool ok, const std::string& what)
{ if (not ok) ++failures; std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what.c_str()); }

static constexpr float kPi = std::numbers::pi_v<float>;
static const Poly kRoom{{-3.f, -2.f}, {3.f, -2.f}, {3.f, 2.f}, {-3.f, 2.f}};
static const Poly kCloset{{-1.2f, -1.2f}, {1.2f, -1.2f}, {1.2f, 1.2f}, {-1.2f, 1.2f}};
static const Eigen::Vector3f kTruth(-0.7f, 0.4f, 0.6f);

static std::vector<Eigen::Vector3f> scan_at(const Poly& poly, const Eigen::Vector3f& pose, std::mt19937& rng)
{
    std::normal_distribution<float> nz(0.f, 0.01f);
    std::vector<Eigen::Vector3f> out;
    for (int i = 0; i < 360; ++i)
    {
        const float b = -kPi + 2.f * kPi * static_cast<float>(i) / 360.f;
        const Eigen::Vector2f dir(std::cos(pose.z() + b), std::sin(pose.z() + b));
        float best = 1e9f;
        for (std::size_t k = 0; k < poly.size(); ++k)
        {
            const Eigen::Vector2f a = poly[k], e = poly[(k + 1) % poly.size()] - a;
            const float den = dir.x() * e.y() - dir.y() * e.x();
            if (std::abs(den) < 1e-9f) continue;
            const Eigen::Vector2f w = a - pose.head<2>();
            const float s = (w.x() * e.y() - w.y() * e.x()) / den, t = (w.x() * dir.y() - w.y() * dir.x()) / den;
            if (s > 1e-4f and t >= 0.f and t <= 1.f) best = std::min(best, s);
        }
        if (best > 1e8f) continue;
        const float r = best + nz(rng);
        out.emplace_back(r * std::cos(b), r * std::sin(b), 0.f);
    }
    return out;
}

// What PosePublisher::maybe_publish_corrected_pose does with each result, on the SAME functions it calls
// (rc::publish::substitute_survey + clamp_correction) and the same state it carries between frames
// (last published pose / pre-clamp estimate / epoch). Limits as live: 1 m/s, 2 rad/s.
struct Publisher
{
    std::optional<Eigen::Affine2f> last_pose;
    std::optional<Eigen::Vector3f> last_est;
    std::uint64_t last_epoch = 0;
    std::int64_t last_ts = 0;
    bool fired = false;
    bool use_survey_pred = true;   // false = the reviewed bug: survey pose substituted, fit prediction kept
    Eigen::Vector3f publish(RoomConcept& r, rc::loc::UpdateResult res)
    {
        if (auto sp = r.survey_pose_during_fit(); sp.has_value())
        {
            if (not use_survey_pred) sp->pred = Eigen::Vector3f(res.pred_x, res.pred_y, res.pred_theta);
            rc::publish::substitute_survey(res, sp.value(), last_epoch);
        }
        const Eigen::Affine2f est = res.robot_pose;
        const bool relocalised = res.reloc_epoch != last_epoch;
        fired = false;
        if (not relocalised and last_pose.has_value() and res.timestamp_ms > last_ts)
        {
            const float dt = static_cast<float>(res.timestamp_ms - last_ts) / 1000.f;
            const auto cl = rc::publish::clamp_correction(*last_pose, last_est, res.robot_pose,
                                                          Eigen::Vector3f(res.pred_x, res.pred_y, res.pred_theta), 1.f * dt, 2.f * dt);
            if (cl.fired) { res.robot_pose = cl.pose; fired = true; }
        }
        last_pose = res.robot_pose;
        last_est = Eigen::Vector3f(est.translation().x(), est.translation().y(), std::atan2(est.linear()(1, 0), est.linear()(0, 0)));
        last_epoch = res.reloc_epoch;
        last_ts = res.timestamp_ms;
        const auto& P = res.robot_pose;
        return {P.translation().x(), P.translation().y(), std::atan2(P.linear()(1, 0), P.linear()(0, 0))};
    }
};

struct Rig
{
    RoomConcept rcn;
    Publisher pub;
    Eigen::Vector3f published = Eigen::Vector3f::Zero();
    std::mt19937 rng{2026};
    std::int64_t ts = 1000;
    void frame()
    {
        ts += 50;
        std::vector<rc::OdometryReading> odom;   // parked: a zero-velocity stream covering the frame
        for (std::int64_t t = ts - 100; t <= ts; t += 10)
        {
            rc::OdometryReading o;
            o.source_ts_ms = t; o.recv_ts_ms = t;
            o.var_adv = 1e-4f; o.var_side = 1e-4f;
            odom.push_back(o);
        }
        const auto pts = scan_at(kRoom, kTruth, rng);
        auto res = rcn.update(rc::LidarData{pts, ts}, {}, odom);
        res.reloc_epoch = Acc::epoch(rcn);          // as the run loop stamps it
        published = pub.publish(rcn, res);          // on_result_ready → PosePublisher
        Acc::reloc(rcn, res, pts);                  // then section 8
    }
    Rig()
    {
        rcn.params.plain_slam = true;
        rcn.set_polygon_room(kRoom);                         // the room left behind (given) ...
        rcn.set_robot_pose(kTruth.x(), kTruth.y(), kTruth.z(), false);
        rcn.begin_room_estimation("proposal_fit_test: a room entered through a door is surveyed");   // ... P surveyed
        for (int f = 0; f < 60; ++f) frame();
    }
};

static Eigen::Vector3f pose_of(RoomConcept& r) { const auto s = r.get_current_state(); return {s[2], s[3], s[4]}; }
static float dpose(const Eigen::Vector3f& a, const Eigen::Vector3f& b)
{ return (a.head<2>() - b.head<2>()).norm() + std::abs(std::remainder(a.z() - b.z(), 2.f * kPi)); }

struct Before
{
    Eigen::Vector3f pose; std::size_t walls; bool map_ready, adopted, driving; float mu, var; Poly derived;
};
static Before snap(RoomConcept& r)
{
    return {pose_of(r), Acc::walls(r), Acc::map_ready_raw(r), r.map_layout_adopted(), r.plain_slam_driving(),
            Acc::track_mu(r), Acc::track_var(r), r.nominal_room_polygon()};
}
static bool same_f(float a, float b) { return (std::isnan(a) and std::isnan(b)) or a == b; }
static void unchanged(RoomConcept& r, const Before& b, const std::string& tag)
{
    const auto a = snap(r);
    check(dpose(a.pose, b.pose) < 0.05f, tag + ": pose back on the survey's (" + std::to_string(dpose(a.pose, b.pose)) + ")");
    check(a.walls == b.walls, tag + ": WallMap untouched (" + std::to_string(a.walls) + " walls)");
    check(a.map_ready == b.map_ready, tag + ": map_ready unchanged");
    check(a.derived == b.derived, tag + ": derived polygon unchanged");
    check(a.adopted == b.adopted and a.driving == b.driving and not r.hypothesis_fit_active(),
          tag + ": SEARCHING with plain SLAM driving again");
    check(same_f(a.mu, b.mu) and same_f(a.var, b.var), tag + ": the learnt tracking emission is the survey's");
}

int main()
{
    std::setlocale(LC_ALL, "");
    rc::reloc::PriorMode wrong;   // the try32 shape: ltsm's prior metres and ~90 deg off
    wrong.mu = Eigen::Vector3f(2.f, -1.f, kTruth.z() + kPi / 2);
    wrong.S = Eigen::Vector3f(0.04f, 0.04f, 0.02f).asDiagonal();

    std::printf("survey precondition\n");
    {
        Rig rig;
        check(rig.rcn.plain_slam_driving() and dpose(pose_of(rig.rcn), kTruth) < 0.05f, "plain SLAM drives the survey, pose on truth");
    }

    std::printf("rejected_fit_restores_the_survey\n");
    {
        Rig rig;
        auto& r = rig.rcn;
        const auto b = snap(r);
        check(r.begin_hypothesis_fit(kCloset, {wrong}, 0.0), "the fit starts (closet layout, prior 3 m / 90 deg off)");
        const Eigen::Vector3f published_before = rig.published;
        float published_dev = 0.f, fit_dev = 0.f, vs_shadow = 0.f;
        bool any_clamp = false;
        for (int k = 0; k < 60 and r.late_recall_verdict() == RoomConcept::LateVerdict::Pending; ++k)
        {
            rig.frame();
            const auto sp = r.survey_pose_during_fit();
            if (k == 0)
                check(sp.has_value() and dpose(rig.published, sp->pose) < 1e-4f and not rig.pub.fired,
                      "FIRST fit frame: the PUBLISHED pose is exactly the shadow's, no clamp ("
                      + std::to_string(sp.has_value() ? dpose(rig.published, sp->pose) : -1.f) + ")");
            if (sp.has_value()) vs_shadow = std::max(vs_shadow, dpose(rig.published, sp->pose));
            any_clamp = any_clamp or rig.pub.fired;
            published_dev = std::max(published_dev, dpose(rig.published, published_before));
            fit_dev = std::max(fit_dev, dpose(pose_of(r), b.pose));
        }
        check(vs_shadow < 1e-4f and not any_clamp, "every fit frame publishes the shadow's pose exactly (max "
              + std::to_string(vs_shadow) + ")");
        // Parked, repeated views count once (novelty-weighted): the forced search's scan took the log-odds to ~-1.9 and
        // further identical views add nothing. Two more DISTINCT contrary views (what driving would add) are fed
        // straight into the sequential test — the rejection itself is the fit's, the views are the harness's.
        const double after_scans = Acc::log_odds(r);
        for (int k = 0; k < 10 and r.late_recall_verdict() == RoomConcept::LateVerdict::Pending; ++k) Acc::evidence(r, 1.0f);
        check(after_scans < 0.0, "the scans alone already argue against K (log-odds " + std::to_string(after_scans) + ")");
        check(fit_dev > 0.5f, "the fit's own pose was dragged away (" + std::to_string(fit_dev) + ") ...");
        check(published_dev < 0.05f, "... while the PUBLISHED pose (publisher path) stayed where it was (" + std::to_string(published_dev) + ")");
        check(r.late_recall_verdict() == RoomConcept::LateVerdict::Rejected, "verdict: REJECTED");
        r.end_hypothesis_fit();
        unchanged(r, b, "rejected");
        rig.frame();
        check(dpose(rig.published, published_before) < 0.05f and dpose(rig.published, pose_of(r)) < 1e-4f,
              "first frame after the reject: PUBLISHED = the survey's pose, where it was ("
              + std::to_string(dpose(rig.published, published_before)) + ")");
        for (int f = 0; f < 10; ++f) rig.frame();
        check(dpose(pose_of(r), kTruth) < 0.05f, "the survey carries on from where it was");
    }

    std::printf("withdrawn_fit_restores_the_survey\n");
    {
        Rig rig;
        auto& r = rig.rcn;
        const auto b = snap(r);
        check(r.begin_hypothesis_fit(kCloset, {wrong}, 3.0), "the fit starts");
        float vs = 0.f;
        for (int f = 0; f < 3; ++f)
        {
            rig.frame();
            if (const auto sp = r.survey_pose_during_fit(); sp.has_value()) vs = std::max(vs, dpose(rig.published, sp->pose));
        }
        check(vs < 1e-4f, "while pending, PUBLISHED = the shadow's pose (max " + std::to_string(vs) + ")");
        check(r.late_recall_verdict() == RoomConcept::LateVerdict::Pending, "no verdict yet (ltsm withdraws / re-posts)");
        r.end_hypothesis_fit();
        unchanged(r, b, "withdrawn");
        const Eigen::Vector3f before_pub = b.pose;
        rig.frame();
        check(dpose(rig.published, before_pub) < 0.05f and dpose(rig.published, pose_of(r)) < 1e-4f,
              "first frame after the withdrawal: PUBLISHED = the survey's pose (" + std::to_string(dpose(rig.published, before_pub)) + ")");
        check(r.late_recall_verdict() == RoomConcept::LateVerdict::None, "the late fit is disarmed");
    }

    std::printf("survey_prediction_is_what_keeps_the_clamp_quiet\n");
    {
        Rig rig;
        auto& r = rig.rcn;
        rig.pub.use_survey_pred = false;   // the reviewed bug: survey pose, but the FIT tracker's prediction
        const Eigen::Vector3f before = rig.published;
        check(r.begin_hypothesis_fit(kCloset, {wrong}, 0.0), "the fit starts");
        rig.frame();
        check(rig.pub.fired and dpose(rig.published, before) > 0.1f,
              "with the fit's prediction the clamp FIRES and drags the published pose toward the fit ("
              + std::to_string(dpose(rig.published, before)) + ") — why the survey's prediction is substituted");
        r.end_hypothesis_fit();
    }

    std::printf("true_layout_accepted_and_kept\n");
    {
        Rig rig;
        auto& r = rig.rcn;
        rc::reloc::PriorMode right;
        right.mu = kTruth + Eigen::Vector3f(0.05f, -0.03f, 0.02f);
        right.S = Eigen::Vector3f(0.04f, 0.04f, 0.02f).asDiagonal();
        // Parked, repeated views are ONE observation (novelty-weighted), so the verdict needs the recogniser's prior as it
        // does live: ln P(K)/P(new) = 1.5 (posterior ~0.82), below the bound on its own.
        check(r.begin_hypothesis_fit(kRoom, {right}, 1.5), "the fit starts (the true layout, recogniser log-odds 1.5)");
        for (int k = 0; k < 60 and r.late_recall_verdict() == RoomConcept::LateVerdict::Pending; ++k) rig.frame();
        check(r.late_recall_verdict() == RoomConcept::LateVerdict::Accepted, "verdict: ACCEPTED (" + std::to_string(static_cast<int>(r.late_recall_verdict())) + ", log-odds " + std::to_string(Acc::log_odds(r)) + ")");
        r.keep_hypothesis_fit();
        check(not r.hypothesis_fit_active() and r.map_layout_adopted() and r.localizing(), "K stays the given room");
        check(dpose(pose_of(r), kTruth) < 0.1f, "pose on truth in K");
    }

    std::printf(failures == 0 ? "all ok (0 failures)\n" : "FAILED (%d failures)\n", failures);
    return failures == 0 ? 0 : 1;
}
