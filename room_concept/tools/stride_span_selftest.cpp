// g++ -std=c++23 -O2 -Isrc -I/usr/include/eigen3 tools/stride_span_selftest.cpp -o /tmp/x && /tmp/x
// The strided window's motion factors. Every frame builds a slot, so the NEWEST slot is always the
// previous frame. A slot's motion factor links it to the slot BEFORE it in the window, and must carry
// exactly the motion between those two frames. Simulates a drive, applies the admit/replace rule, and
// checks that invariant for every slot — once with the pre-2026-10-05 accumulator (which fails: an
// appended slot carried the whole stride, the next replaced one only a frame) and once with StrideSpan.
#include "stride_span.h"
#include <cstdio>
#include <random>
#include <vector>

namespace
{
    struct Slot { int frame; Eigen::Vector3f delta; };

    /// Max over slots of |factor delta - true motion between the linked frames|.
    template <class Policy>
    float worst_span_error(Policy policy)
    {
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> step(0.01f, 0.04f);   // 1-4 cm per frame, driving
        const int frames = 400;
        std::vector<Eigen::Vector3f> pos(frames + 1, Eigen::Vector3f::Zero());   // pose after frame k
        std::vector<Slot> window;
        Eigen::Vector3f last_admitted = Eigen::Vector3f::Zero();
        bool has_admitted = false;
        float worst = 0.f;
        for (int k = 1; k <= frames; ++k)
        {
            const Eigen::Vector3f d(step(rng), 0.3f * step(rng), 0.01f * step(rng));
            pos[k] = pos[k - 1] + d;
            const bool replace = has_admitted and window.size() > 1
                                 and (pos[k] - last_admitted).head<2>().norm() < 0.10f;
            const Eigen::Vector3f fd = policy(replace, d);
            if (replace) window.back() = {k, fd};
            else { window.push_back({k, fd}); last_admitted = pos[k]; has_admitted = true; }
            // Checked EVERY frame, because the window is solved every frame: an appended slot that is
            // wrong lives for exactly one solve before the next frame overwrites it, so a check of the
            // final window alone never sees it.
            if (window.size() > 1)
            {
                const auto &a = window[window.size() - 2], &b = window.back();
                worst = std::max(worst, (b.delta - (pos[b.frame] - pos[a.frame])).norm());
            }
        }
        return worst;
    }
}

int main()
{
    int fails = 0;
    const Eigen::Matrix3f c = Eigen::Matrix3f::Identity() * 1e-4f;

    // The accumulator as it was: += every frame, used for both cases, zeroed after an append.
    Eigen::Vector3f acc = Eigen::Vector3f::Zero();
    const float old_err = worst_span_error([&](bool replace, const Eigen::Vector3f &d)
    {
        acc += d;
        const Eigen::Vector3f out = acc;
        if (not replace) acc.setZero();
        return out;
    });
    std::printf("old accumulator : worst factor-vs-truth error %.4f m  (%s)\n", old_err,
                old_err > 0.05f ? "reproduces the over-span, as expected" : "UNEXPECTED: no defect");
    if (not (old_err > 0.05f)) ++fails;

    rc::StrideSpan span;
    const float new_err = worst_span_error([&](bool replace, const Eigen::Vector3f &d)
    {
        return (replace ? span.replace(d, c, nullptr) : span.append(d, c, nullptr)).delta;
    });
    std::printf("StrideSpan      : worst factor-vs-truth error %.2e m  (%s)\n", new_err,
                new_err < 1e-5f ? "ok" : "FAIL");
    if (not (new_err < 1e-5f)) ++fails;

    // Covariance follows the same span: a replaced slot carries newest + frame, an appended one the frame.
    rc::StrideSpan s2;
    s2.append(Eigen::Vector3f::Zero(), c, nullptr);
    const auto r = s2.replace(Eigen::Vector3f::Zero(), c, nullptr);
    const auto a = s2.append(Eigen::Vector3f::Zero(), c, nullptr);
    const bool cov_ok = r.cov.isApprox(2.f * c) and a.cov.isApprox(c);
    std::printf("covariance span : replace %.1e (want 2e-04), append %.1e (want 1e-04)  (%s)\n",
                r.cov(0, 0), a.cov(0, 0), cov_ok ? "ok" : "FAIL");
    if (not cov_ok) ++fails;

    std::printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails;
}
