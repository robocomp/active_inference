// spec 2026-11-15 lidar-measures §7c.8 rev 3.3.14 Pass (1), (2): birth size at the class means; the cloud's extent beyond
// the class size widens the position covariance, σ² = σ_pos² + (E − L)²/12 per wall axis; F1.3: along the
// wall either sign of E − L (a clipped cloud too). Run: ./birth_footprint_test
#include <cstdio>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include "countable_birth_footprint.h"

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

// A uniform grid of points over [t0, t1] along the wall × [n0, n1] into the room, wall y = 0 with normal +y.
static std::vector<Eigen::Vector2f> slab(float t0, float t1, float n0, float n1, int nt, int nn)
{
    std::vector<Eigen::Vector2f> v;
    for (int i = 0; i < nt; ++i)
        for (int j = 0; j < nn; ++j)
            v.emplace_back(t0 + (t1 - t0) * (i + 0.5f) / nt, n0 + (n1 - n0) * (j + 0.5f) / nn);
    return v;
}

int main()
{
    std::printf("birth_footprint_test (spec 7c.8 rev 3.3.14)\n");
    const Eigen::Vector2f wp(0.0f, 0.0f), wn(0.0f, 1.0f);       // wall along x, room at +y; along t = (−1, 0)
    constexpr float cw = 0.51f, cd = 0.51f, sp = 0.30f;

    // (1) merged: a 0.51 m nightstand front face at 0.5 m from the wall, plus a 1.6 m stretch of the neighbour (bed) reaching 2.1 m
    {
        auto xy = slab(-0.25f, 0.25f, 0.48f, 0.52f, 30, 4);
        const auto bed = slab(-0.25f, 0.25f, 0.52f, 2.10f, 30, 60);
        xy.insert(xy.end(), bed.begin(), bed.end());
        const auto bf = rc::birth_footprint(xy, wp, wn, cw, cd, sp);
        const float ex_n = bf.extent_n - cd;
        const float want_n = sp * sp + ex_n * ex_n / 12.0f;
        std::printf("  merged: w %.3f h %.3f centre (%.3f, %.3f) extent along %.3f across %.3f ; sigma along %.3f across %.3f\n",
                    bf.w, bf.h, bf.centre.x(), bf.centre.y(), bf.extent_t, bf.extent_n, std::sqrt(bf.var_t), std::sqrt(bf.var_n));
        CHECK(bf.ok);
        CHECK(bf.w == cw and bf.h == cd);                        // the class means, not the 2 m cloud
        CHECK(std::abs(bf.var_n - want_n) <= 1e-6f * want_n);    // σ_n² = σ_pos² + (E_n − h̄)²/12
        CHECK(bf.extent_n > 1.5f);                                // the cloud really is long across
        const float ex_t = bf.extent_t - cw;                       // along: (w̄ − E_t)²/12, either sign (F1.3)
        CHECK(std::abs(bf.var_t - (sp * sp + ex_t * ex_t / 12.0f)) <= 1e-6f * bf.var_t);
        CHECK(std::abs(bf.centre.y() - 0.5f * cd) < 1e-5f);       // flush: h̄/2 in from the wall
        // room-frame covariance: the wall's n is +y, so the across variance sits on (y, y)
        CHECK(std::abs(bf.pos_cov(1, 1) - bf.var_n) < 1e-6f and std::abs(bf.pos_cov(0, 0) - bf.var_t) < 1e-6f);
        CHECK(std::abs(bf.pos_cov(0, 1)) < 1e-6f);
    }
    // (2) clean: one 0.51 m front face (a line, 4 cm deep) — no excess on either axis
    {
        const auto xy = slab(-0.25f, 0.25f, 0.49f, 0.53f, 30, 4);
        const auto bf = rc::birth_footprint(xy, wp, wn, cw, cd, sp);
        std::printf("  clean: w %.3f h %.3f extent along %.3f across %.3f ; sigma along %.3f across %.3f\n",
                    bf.w, bf.h, bf.extent_t, bf.extent_n, std::sqrt(bf.var_t), std::sqrt(bf.var_n));
        CHECK(bf.ok and bf.w == cw and bf.h == cd);
        const float ex_t = bf.extent_t - cw;
        CHECK(std::abs(bf.var_t - (sp * sp + ex_t * ex_t / 12.0f)) <= 1e-6f * bf.var_t);   // the seen width ≈ w̄: ~σ_pos
        CHECK(std::abs(bf.var_n - sp * sp) <= 1e-6f);             // across: one face, shallower than h̄ ⇒ σ_pos only
    }
    // (3) clipped (plan rev 2 F1.3): the image border cut the front face to its left 0.20 m. The seen extent is SHORTER than
    // the class width, so the centre along the wall lies anywhere over w̄ − E_t: (w̄ − E_t)²/12, no switch, no gain.
    {
        const auto xy = slab(-0.25f, -0.05f, 0.49f, 0.53f, 12, 4);
        const auto bf = rc::birth_footprint(xy, wp, wn, cw, cd, sp);
        const float ex_t = cw - bf.extent_t;
        const float want_t = sp * sp + ex_t * ex_t / 12.0f;
        std::printf("  clipped: extent along %.3f (class %.3f) ; sigma along %.3f (sigma_pos %.3f) across %.3f\n",
                    bf.extent_t, cw, std::sqrt(bf.var_t), sp, std::sqrt(bf.var_n));
        CHECK(bf.ok and bf.w == cw and bf.h == cd);
        CHECK(bf.extent_t < 0.25f);                                // the cloud really is short of the class width
        CHECK(std::abs(bf.var_t - want_t) <= 1e-6f * want_t);
        CHECK(bf.var_t > sp * sp + 0.0045f);                      // (0.51 − ~0.2)²/12 ≈ 0.008: visibly wider than σ_pos² alone
        CHECK(std::abs(bf.var_n - sp * sp) <= 1e-6f);
    }
    // a rotated wall: the covariance rotates with it (trace and determinant kept)
    {
        const float a = 0.6f;
        const Eigen::Vector2f n(std::sin(a), std::cos(a));
        const Eigen::Vector2f t(-n.y(), n.x());
        std::vector<Eigen::Vector2f> xy;
        for (const auto& q : slab(-0.25f, 0.25f, 0.48f, 2.10f, 30, 60)) xy.push_back(-q.x() * t + q.y() * n);
        const auto bf = rc::birth_footprint(xy, wp, n, cw, cd, sp);
        const float tr = bf.pos_cov.trace(), det = bf.pos_cov.determinant();
        std::printf("  rotated: trace %.5f (want %.5f) det %.6e (want %.6e)\n", tr, bf.var_t + bf.var_n, det, bf.var_t * bf.var_n);
        CHECK(std::abs(tr - (bf.var_t + bf.var_n)) < 1e-5f);
        CHECK(std::abs(det - bf.var_t * bf.var_n) < 1e-5f * bf.var_t * bf.var_n);
        const Eigen::Vector2f Pn = bf.pos_cov * n;                // n is an eigenvector with eigenvalue var_n
        CHECK((Pn - bf.var_n * n).norm() < 1e-5f);
    }
    std::printf(bad ? "birth_footprint_test: %d FAILED\n" : "birth_footprint_test: OK\n", bad);
    return bad != 0;
}
