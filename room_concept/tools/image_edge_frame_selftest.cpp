// g++ -std=c++23 -O2 -Isrc -I/usr/include/eigen3 tools/image_edge_frame_selftest.cpp -o /tmp/x && /tmp/x
// The RGB edge term's FRAME-level marginalisation (image_edge_accumulate.h accumulate_frame).
// 1. One frame of S single-sample segments that all share one boresight yaw: the heading information
//    must SATURATE at ~1/sigma_yaw^2 as S grows. The per-segment form (accumulate_segment summed) gives
//    every segment its own copy of the yaw and grows ~linearly — the defect this replaces.
// 2. Residual noise inflated by s: the frame precision pi must come out ~1/s^2 (and ~1 when honest).
// 3. With no shared nuisance and pi held at 1, the frame form reproduces the per-segment sum exactly.
#include "image_edge_accumulate.h"
#include <cstdio>
#include <random>
#include <vector>

namespace
{
    constexpr float kSigmaPx = 0.5f, kYawSens = 448.f * 0.0035f;   // px per unit-variance yaw nuisance

    std::vector<rc::ImageEdgeSegment> frame(int S, bool shared_yaw, float h_local = 0.5f)
    {
        std::vector<rc::ImageEdgeSegment> segs(S);
        for (auto &seg : segs)
        {
            rc::ImageEdgeSample s;
            s.sigma_px = kSigmaPx; s.pi_vis = 1.f; s.search_L = 50.f;
            s.h.setZero();
            if (shared_yaw) s.h(2) = kYawSens;
            s.h(4) = h_local;                    // the contour's own map position, per segment
            seg.samples.push_back(s);
        }
        return segs;
    }
}

int main()
{
    int fails = 0;
    std::mt19937 rng(5);
    std::normal_distribution<float> n01;

    // ── 1. saturation ──
    {
        float h_frame[2], h_seg[2];
        const int Ss[2] = {8, 128};
        for (int i = 0; i < 2; ++i)
        {
            auto segs = frame(Ss[i], true);
            // J: heading sensitivity ~ the same 448 px/rad for every sample (a yaw rotates the whole image)
            const auto J = [](std::size_t, std::size_t, Eigen::Matrix<float, 1, 3> &Jr) { Jr << 30.f, 5.f, 448.f; return 0.f; };
            const auto fa = rc::img::accumulate_frame(segs, J, [](std::size_t, std::size_t) { return 0.f; }, 1e9, 1e9);
            h_frame[i] = fa.H(2, 2);
            float hs = 0.f;
            for (const auto &seg : segs)
                hs += rc::img::accumulate_segment(seg, [&](std::size_t k, Eigen::Matrix<float, 1, 3> &Jr) { return J(0, k, Jr); },
                                                  [](std::size_t) { return 0.f; }).H(2, 2);
            h_seg[i] = hs;
        }
        const float bound = 448.f * 448.f / (kYawSens * kYawSens);   // heading info a yaw nuisance allows
        const bool ok = h_frame[1] / h_frame[0] < 1.2f and h_frame[1] < 1.05f * bound
                    and h_seg[1] / h_seg[0] > 10.f;
        std::printf("1. heading info 8 -> 128 segments: frame %.3g -> %.3g (bound %.3g); per-segment %.3g -> %.3g  (%s)\n",
                    h_frame[0], h_frame[1], bound, h_seg[0], h_seg[1], ok ? "ok" : "FAIL");
        if (not ok) ++fails;
    }

    // ── 2. precision hyperparameter ──
    {
        bool ok = true;
        for (const float s : {1.f, 3.f})
        {
            // only the PIXEL noise is inflated (no nuisance columns): the responsibility's inlier variance
            // is sigma_px^2, so inflating an unmodelled column would just get the samples rejected as
            // outliers by the mixture before pi ever sees them
            auto segs = frame(200, false, 0.f);
            std::vector<float> rs;
            for (std::size_t i = 0; i < segs.size(); ++i)
                rs.push_back(s * kSigmaPx * n01(rng));
            const auto fa = rc::img::accumulate_frame(segs,
                [&](std::size_t si, std::size_t, Eigen::Matrix<float, 1, 3> &Jr) { Jr << 10.f, 10.f, 10.f; return rs[si]; },
                [](std::size_t, std::size_t) { return 0.f; });
            const float want = 1.f / (s * s);
            const bool o = std::abs(fa.pi - want) < 0.3f * want;
            std::printf("2. residuals x%.0f: pi %.3f (want ~%.3f)  (%s)\n", s, fa.pi, want, o ? "ok" : "FAIL");
            ok = ok and o;
        }
        if (not ok) ++fails;
    }

    // ── 3. reduces to the per-segment sum without shared nuisances ──
    {
        auto segs = frame(20, false);
        std::vector<float> rs; for (int i = 0; i < 20; ++i) rs.push_back(0.3f * n01(rng));
        const auto J = [&](std::size_t si, std::size_t, Eigen::Matrix<float, 1, 3> &Jr) { Jr << 1.f + si, 2.f, -0.5f * si; return rs[si]; };
        const auto fa = rc::img::accumulate_frame(segs, J, [](std::size_t, std::size_t) { return 0.f; }, 1e9, 1e9);
        Eigen::Matrix3f Hs = Eigen::Matrix3f::Zero(); Eigen::Vector3f bs = Eigen::Vector3f::Zero();
        for (std::size_t si = 0; si < segs.size(); ++si)
        {
            const auto a = rc::img::accumulate_segment(segs[si], [&](std::size_t k, Eigen::Matrix<float, 1, 3> &Jr) { return J(si, k, Jr); },
                                                       [](std::size_t) { return 0.f; });
            Hs += a.H; bs += a.b;
        }
        const float e = (fa.H - Hs).norm() / Hs.norm() + (fa.b - bs).norm() / std::max(bs.norm(), 1e-9f);
        const bool ok = e < 1e-4f;
        std::printf("3. no shared nuisance: frame == per-segment sum to %.1e  (%s)\n", e, ok ? "ok" : "FAIL");
        if (not ok) ++fails;
    }
    std::printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails;
}
