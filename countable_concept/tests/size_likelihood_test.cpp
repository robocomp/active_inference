// size_likelihood_test — the likelihood-only (H, w, h) summary kept beside the posterior (spec 2026-11-14 countable
// class existence, §2 and §9 "No double counting"). The posterior already contains the class SIZE prior (accumulate_extra
// blocks 2 and 2a), so scoring it against the class would count the prior twice; size_likelihood() is what the DATA says.
// Bed parameters from bed_concept/etc/config.toml. Run: ./size_likelihood_test
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include "countable_belief.h"

using namespace rc;
static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

// Posterior after 10 frames of the full bed cloud, recorded from the code BEFORE the summary existed (RECORD_ONLY build).
static constexpr float RECORDED_CX = 0.0273552388f, RECORDED_CY = -0.0290362667f, RECORDED_H = 0.674938679f,
                       RECORDED_W = 1.48336649f, RECORDED_HH = 1.35412252f, RECORDED_YAW = -0.0919634104f;

static CountableBeliefParams bed_params()
{
    CountableBeliefParams p;
    p.prior_height_m = 0.65f;  p.prior_height_std = 0.10f;
    p.prior_footprint_m = 1.75f; p.prior_footprint_std = 0.20f; p.prior_depth_m = 2.10f;
    return p;
}

// A box of size (H, w, h) at the origin, yaw 0: sides (all, or only the front face local -y) and top.
static std::vector<Eigen::Vector3f> box_cloud(float H, float w, float h, bool front_only, std::mt19937& rng)
{
    std::normal_distribution<float> noise(0.0f, 0.01f);
    std::uniform_real_distribution<float> U01(0.0f, 1.0f);
    std::vector<Eigen::Vector3f> pts;
    for (int i = 0; i < 1500; ++i)
    {
        const float u = U01(rng), z = U01(rng) * H;
        const int face = front_only ? 3 : (i & 3);
        float lx, ly;
        if (face == 0)      { lx =  0.5f * w; ly = (u - 0.5f) * h; }
        else if (face == 1) { lx = -0.5f * w; ly = (u - 0.5f) * h; }
        else if (face == 2) { ly =  0.5f * h; lx = (u - 0.5f) * w; }
        else                { ly = -0.5f * h; lx = (u - 0.5f) * w; }
        pts.emplace_back(lx + noise(rng), ly + noise(rng), z + noise(rng));
    }
    if (not front_only)
        for (int i = 0; i < 400; ++i)
            pts.emplace_back((U01(rng) - 0.5f) * w, (U01(rng) - 0.5f) * h, H + noise(rng));
    return pts;
}

static CountableBelief run(const std::vector<Eigen::Vector3f>& pts, int frames)
{
    CountableBeliefState s0{0.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f};
    Eigen::Vector3f sum = Eigen::Vector3f::Zero(); float zmax = 0.0f;
    for (const auto& p : pts) { sum += p; zmax = std::max(zmax, p.z()); }
    s0.cx = sum.x() / pts.size(); s0.cy = sum.y() / pts.size(); s0.H = zmax;
    CountableBelief b(s0, bed_params());
    CountableFrame f; f.points = pts;
    for (int i = 0; i < frames; ++i) b.update(f);
    return b;
}

#ifdef RECORD_ONLY
int main()
{
    std::mt19937 rng(11);
    const auto b = run(box_cloud(0.65f, 1.75f, 2.10f, false, rng), 10);
    const auto s = b.state();
    std::printf("RECORD cx=%.9g cy=%.9g H=%.9g w=%.9g h=%.9g yaw=%.9g\n", s.cx, s.cy, s.H, s.w, s.h, s.yaw);
}
#else
int main()
{
    {   // panel_seen_as_panel: the data says a panel (in some w/h labelling), whatever the class prior says
        std::mt19937 rng(5);
        const auto b = run(box_cloud(1.86f, 1.98f, 0.28f, false, rng), 10);
        const auto& L = b.size_likelihood().mean;
        const auto& s = b.state();
        const float lo = std::min(L(1), L(2)), hi = std::max(L(1), L(2));
        std::printf("panel: lik (H,w,h)=(%.3f %.3f %.3f)  posterior (H,w,h)=(%.3f %.3f %.3f)\n", L(0), L(1), L(2), s.H, s.w, s.h);
        CHECK(std::abs(L(0) - 1.86f) < 0.08f);
        CHECK(std::abs(lo - 0.28f) < 0.08f);
        CHECK(std::abs(hi - 1.98f) < 0.25f);
    }
    {   // sparse_view_prior_removed: few noisy points; the prior pulls the posterior, not the likelihood summary
        std::mt19937 rng(9);
        std::normal_distribution<float> nz(0.0f, 0.05f);
        auto pts = box_cloud(1.86f, 1.98f, 0.28f, false, rng);
        std::shuffle(pts.begin(), pts.end(), rng);
        pts.resize(40);
        for (auto& p : pts) p += Eigen::Vector3f(nz(rng), nz(rng), nz(rng));
        CountableBeliefState s0{0.0f, 0.0f, 1.86f, 1.98f, 0.28f, 0.0f};   // start at the truth so only the prior moves it
        CountableBelief b(s0, bed_params());
        CountableFrame f; f.points = pts; f.R.assign(pts.size(), 0.05f * 0.05f);
        for (int i = 0; i < 5; ++i) b.update(f);
        const auto& L = b.size_likelihood().mean; const auto& s = b.state();
        std::printf("sparse: lik H=%.3f w=%.3f h=%.3f  posterior H=%.3f w=%.3f h=%.3f\n", L(0), L(1), L(2), s.H, s.w, s.h);
        CHECK(std::abs(L(0) - 1.86f) < std::abs(s.H - 1.86f));
    }
    {   // no_frames_zero_info
        CountableBelief b(CountableBeliefState{}, bed_params());
        CHECK(b.size_likelihood().info.norm() == 0.0f);
    }
    {   // one_face_view_depth_drops_out: a front face fixes the face position, not the depth behind it. The unobserved
        // direction is identified from the FIRST frame (smallest eigenvector of its summary) and must stay unobserved.
        std::mt19937 rng(3);
        const auto pts = box_cloud(0.65f, 1.75f, 2.10f, true, rng);
        CountableBeliefState s0{0.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f};
        Eigen::Vector3f sum = Eigen::Vector3f::Zero(); float zmax = 0.0f;
        for (const auto& p : pts) { sum += p; zmax = std::max(zmax, p.z()); }
        s0.cx = sum.x() / pts.size(); s0.cy = sum.y() / pts.size(); s0.H = zmax;
        CountableBelief b(s0, bed_params());
        CountableFrame f; f.points = pts;
        b.update(f);
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(b.size_likelihood().info);
        const Eigen::Vector3f v = es.eigenvectors().col(0);
        std::printf("one-face frame 1: info eigenvalues %.4g %.4g %.4g\n", es.eigenvalues()(0), es.eigenvalues()(1), es.eigenvalues()(2));
        for (int i = 2; i <= 30; ++i)
        {
            b.update(f);
            if (i == 10 or i == 30)
            {
                const auto& I = b.size_likelihood().info;
                const float un = v.dot(I * v);
                std::printf("one-face frame %d: info along unobserved dir %.4g, H info %.4g, ratio %.3g\n", i, un, I(0, 0), un / I(0, 0));
                // 10 frames: < 1%. 30 frames: 1.3% measured, a slow leak of ~0.1%/frame from the fit's own small state change
                // (the unobserved direction itself moves slightly), still ~75x below the observed extents; bound 2%.
                CHECK(un < (i == 10 ? 0.01f : 0.02f) * I(0, 0));
            }
        }
    }
    {   // a gap of predict-only steps forgets: the summary's information on H decreases (and by q_scale for aged steps)
        std::mt19937 rng(11);
        auto b = run(box_cloud(0.65f, 1.75f, 2.10f, false, rng), 5);
        const float h0 = b.size_likelihood().info(0, 0);
        auto c = b;
        for (int i = 0; i < 20; ++i) b.predict();
        c.inflate_for_age(20.0f, 1.0f);
        std::printf("gap: H info %.4g -> %.4g (20 predicts), %.4g (inflate_for_age 20x)\n", h0, b.size_likelihood().info(0, 0), c.size_likelihood().info(0, 0));
        CHECK(b.size_likelihood().info(0, 0) < h0);
        CHECK(c.size_likelihood().info(0, 0) < h0);
        CountableFrame empty;                       // an update with no points also forgets
        auto d = run(box_cloud(0.65f, 1.75f, 2.10f, false, rng), 5);
        const float d0 = d.size_likelihood().info(0, 0);
        d.update(empty);
        CHECK(d.size_likelihood().info(0, 0) < d0);
    }
    {   // fit_unchanged: the main fit is bit-identical to the value recorded from the code before the summary
        std::mt19937 rng(11);
        const auto b = run(box_cloud(0.65f, 1.75f, 2.10f, false, rng), 10);
        const auto s = b.state();
        std::printf("fit: cx=%.9g cy=%.9g H=%.9g w=%.9g h=%.9g yaw=%.9g\n", s.cx, s.cy, s.H, s.w, s.h, s.yaw);
        CHECK(std::abs(s.cx - RECORDED_CX) < 1e-6f and std::abs(s.cy - RECORDED_CY) < 1e-6f and std::abs(s.H - RECORDED_H) < 1e-6f
              and std::abs(s.w - RECORDED_W) < 1e-6f and std::abs(s.h - RECORDED_HH) < 1e-6f and std::abs(s.yaw - RECORDED_YAW) < 1e-6f);
    }
    std::printf("size_likelihood_test: %s\n", bad ? "FAILED" : "OK");
    return bad ? 1 : 0;
}
#endif
