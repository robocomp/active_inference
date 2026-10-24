/*
 * room_posterior_test.cpp — algebra of the joint (room k, pose) + new posterior. Standalone:
 *     g++ -std=c++23 -O1 -I/usr/include/eigen3 room_posterior_test.cpp -o room_posterior_test && ./room_posterior_test
 */
#include "room_posterior.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace rc::recog;

static int g_fail = 0;
#define CHECK(c) do { if (not (c)) { std::printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)
#define CHECK_NEAR(a, b, tol) do { const double A_ = (a), B_ = (b); if (not (std::abs(A_ - B_) <= (tol))) { \
    std::printf("  FAIL %s:%d  %s=%.6f vs %s=%.6f\n", __FILE__, __LINE__, #a, A_, #b, B_); ++g_fail; } } while (0)

static PoseMix one(float x, float y, float th, float sx, float sth)
{
    PoseGauss g; g.mu = {x, y, th}; g.S = Eigen::Vector3f(sx * sx, sx * sx, sth * sth).asDiagonal(); g.w = 1.f;
    return {g};
}
static double p_of(const std::vector<Hypothesis>& h, int i) { return std::exp(h[i].log_post); }
static double total(const std::vector<Hypothesis>& h) { double s = 0; for (const auto& x : h) s += std::exp(x.log_post); return s; }

static MetricEvidence M(const PoseMix& p, float li) { return {p, li, true}; }
static AppearanceEvidence A(const PoseMix& p, float l) { return {p, l, true}; }

static void no_evidence_equals_prior()
{
    std::printf("no_evidence_equals_prior\n");
    std::vector<MetricEvidence> m(3); std::vector<AppearanceEvidence> a(3);
    std::vector<float> area{10.f, 20.f, 30.f};
    const auto h = posterior(m, a, area, 0.f);
    CHECK(h.size() == 4);
    CHECK(h.back().room == -1);
    for (const auto& x : h) { CHECK(std::isfinite(x.log_post)); CHECK_NEAR(std::exp(x.log_post), 0.25, 1e-6); }
}

// metric alone gives P(k)=0.7 against new (one known room).
static const float kLiK = std::log(0.7f / 0.3f);   // with metric_log_new = 0
static void agreeing_experts_beat_one()
{
    std::printf("agreeing_experts_beat_one\n");
    std::vector<float> area{20.f};
    const auto mp = one(2, 3, 0.1f, 0.1f, 0.05f);
    std::vector<MetricEvidence> m{M(mp, kLiK)};
    std::vector<AppearanceEvidence> a0{AppearanceEvidence{}};
    const auto h0 = posterior(m, a0, area, 0.f);
    CHECK_NEAR(p_of(h0, 0), 0.7, 1e-5);
    std::vector<AppearanceEvidence> a1{A(one(2.05f, 3.05f, 0.12f, 0.1f, 0.05f), 0.f)};
    const auto h1 = posterior(m, a1, area, 0.f);
    CHECK(p_of(h1, 0) > 0.7);
    CHECK_NEAR(total(h1), 1.0, 1e-5);
}

static void disagreeing_poses_lower_k()
{
    std::printf("disagreeing_poses_lower_k\n");
    std::vector<float> area{20.f};
    std::vector<MetricEvidence> m{M(one(2, 3, 0.f, 0.1f, 0.05f), kLiK)};
    // pair sigma = sqrt(2)*0.1; "4 sigma" of the combined spread along x
    std::vector<AppearanceEvidence> agree{A(one(2.f, 3.f, 0.f, 0.1f, 0.05f), 1.f)};
    std::vector<AppearanceEvidence> dis{A(one(2.f + 4.f * 0.1f * std::sqrt(2.f), 3.f, 0.f, 0.1f, 0.05f), 1.f)};
    const auto ha = posterior(m, agree, area, 0.f);
    const auto hd = posterior(m, dis, area, 0.f);
    CHECK(p_of(hd, 0) < p_of(ha, 0));
}

static void no_double_counted_prior()
{
    std::printf("no_double_counted_prior\n");
    // A pose mixture that IS the polygon-wide prior: 2S chosen so N(0;0,2S) = 1/(2*pi*area).
    const float A_m2 = 20.f;
    const float s = 6.f;                                              // sx = sy
    const double det2S_target = double(A_m2) * A_m2 / (2.0 * M_PI);   // det(2S)
    // det(2S) = 2^3 * s^2 * s^2 * sth^2 = 8 s^4 sth^2
    const float sth = float(std::sqrt(det2S_target / (8.0 * s * s * s * s)));
    std::vector<float> area{A_m2, A_m2};
    PoseMix pm = one(0, 0, 0, s, sth);
    std::vector<MetricEvidence> m{M(pm, 0.f), M(pm, 0.f)};
    std::vector<AppearanceEvidence> a{A(pm, 0.f), A(pm, 0.f)};
    const auto h = posterior(m, a, area, 0.f);
    for (const auto& x : h) CHECK_NEAR(std::exp(x.log_post), 1.0 / 3.0, 1e-4);
}

static void missing_map_uses_metric_only()
{
    std::printf("missing_map_uses_metric_only\n");
    std::vector<float> area{20.f, 15.f};
    std::vector<MetricEvidence> m{M(one(1, 1, 0, 0.2f, 0.1f), 0.5f), M(one(2, 2, 0, 0.2f, 0.1f), -1.f)};
    std::vector<AppearanceEvidence> none(2);
    std::vector<AppearanceEvidence> junk(2);
    junk[0].pose_in_k = one(9, 9, 1, 0.1f, 0.1f); junk[0].loglr = 7.f; junk[0].available = false;   // ignored
    const auto h0 = posterior(m, none, area, -0.3f);
    const auto h1 = posterior(m, junk, area, -0.3f);
    for (size_t i = 0; i < h0.size(); ++i) CHECK(h0[i].log_post == h1[i].log_post);
    // and equals softmax of the metric terms
    const double e0 = std::exp(0.5), e1 = std::exp(-1.0), en = std::exp(-0.3), z = e0 + e1 + en;
    CHECK_NEAR(p_of(h0, 0), e0 / z, 1e-5);
    CHECK_NEAR(p_of(h0, 2), en / z, 1e-5);
}

static void symmetric_rooms_split()
{
    std::printf("symmetric_rooms_split\n");
    std::vector<float> area{20.f, 20.f};
    const auto mp = one(1, 1, 0, 0.2f, 0.1f);
    std::vector<MetricEvidence> m{M(mp, 0.f), M(mp, 0.f)};
    std::vector<AppearanceEvidence> a{A(mp, 4.f), A(mp, 4.f)};
    const auto h = posterior(m, a, area, -8.f);
    CHECK(h[0].log_post == h[1].log_post);
    CHECK_NEAR(p_of(h, 0), 0.5, 1e-3);
    const Commit c = commit_step(Commit{Commit::State::Entering, -1}, h, h);
    CHECK(c.state == Commit::State::Entering);
}

static void correlated_frames_not_overconfident()
{
    std::printf("correlated_frames_not_overconfident\n");
    std::vector<float> llr(200, 0.3f);
    CHECK_NEAR(discounted_loglr(llr, 100.f, 0.2f), 6.0, 1e-4);
    CHECK(discounted_loglr({}, 1.f, 0.2f) == 0.f);
}

static void undo_uses_only_post_commit_evidence()
{
    std::printf("undo_uses_only_post_commit_evidence\n");
    std::vector<float> area{20.f};
    std::vector<MetricEvidence> m1(1), m2(1);
    std::vector<AppearanceEvidence> a1{A(PoseMix{}, 8.f)}, a2{A(PoseMix{}, -8.f)};
    const auto pre = posterior(m1, a1, area, 0.f);
    const auto post = posterior(m2, a2, area, 0.f);
    const Commit c1 = commit_step(Commit{Commit::State::Entering, -1}, pre, post);
    CHECK(c1.state == Commit::State::Committed and c1.room == 0);
    const Commit c2 = commit_step(c1, pre, post);       // pre is strongly k but must be ignored
    CHECK(c2.state == Commit::State::Entering);
    const Commit c3 = commit_step(c1, pre, pre);        // post-commit agrees: stays
    CHECK(c3.state == Commit::State::Committed and c3.room == 0);
}

static void prior_sensitivity()
{
    std::printf("prior_sensitivity\n");
    // The prior is uniform; halving pi_new is the same as metric_log_new - log 2.
    std::vector<float> area{20.f};
    std::vector<MetricEvidence> m(1);
    for (float l : {5.f, -5.f, 7.f, -7.f}) {
        std::vector<AppearanceEvidence> a{A(PoseMix{}, l)};
        const auto h0 = posterior(m, a, area, 0.f);
        const auto h1 = posterior(m, a, area, -std::log(2.f));
        CHECK(std::abs(p_of(h0, 0) - p_of(h1, 0)) < 0.05);
    }
}

int main()
{
    no_evidence_equals_prior();
    agreeing_experts_beat_one();
    disagreeing_poses_lower_k();
    no_double_counted_prior();
    missing_map_uses_metric_only();
    symmetric_rooms_split();
    correlated_frames_not_overconfident();
    undo_uses_only_post_commit_evidence();
    prior_sensitivity();
    std::printf(g_fail == 0 ? "room_posterior_test: ALL PASSED\n" : "room_posterior_test: %d FAILED\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
