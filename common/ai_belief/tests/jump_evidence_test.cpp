// jump_evidence_test — the move evidence of one update (F1 round 6c; persistent_engine.h detail::jump_evidence).
// (1) THE CLOSED FORM. With the data's information diagonal, each θ direction's contribution to log BF(moved : stayed) is
//     ½ g² (1/s − 1/(s + W²M)) − ½ log(1 + W²M/s),  s = W + W² P̄  (g ~ N(0, s) stayed, N(0, s + W²M) moved).
//     The code's log BF is asserted equal to the sum of these over the six directions (1e-9 relative), for W_HH ∈ {1e-6, 1e-9,
//     1e-12} and a biased g_H ∈ {0, 1.9, 1e3} on the barely-informed H. The 6.1b form (each Gaussian's pseudo-determinant on its
//     own numerical rank) disagreed with it: at nightstand_2 update 177 (W_HH 4.8e-6, g_H 1.9) it kept H in S_s and dropped it
//     from S_m, giving log BF = +3.8e5 (p = 1) where the closed form gives +0.12 on H.
// (2) For the biases the replay shows (|g_H| ≤ 1.9 at W_HH ≤ 1.3e-5): the H share of log BF is ≤ ½ g_H² M_H = 0.018, so p moves
//     by at most a factor e^0.018 from what the informed directions say. NOTE what the closed form does NOT say: as W → 0 at a
//     FIXED g the H share tends to ½ g² M, not to 0 — a consistent channel's g shrinks with W, a biased one's need not, and a
//     large enough bias on an uninformed direction is evidence in the exact Gaussian model too. At W = 0 exactly the share is 0.
// (3) An informed direction: the data put cx 0.30 m away (g = W·0.30, W = 1e4): p ≈ 1, E[v_cx] ≈ 0.30.
// (4) Round 7 A2: an exactly uninformed direction with a stray gradient (bed_5 update 140): finite, its share the closed form.
// Run: ./jump_evidence_test
#include <cmath>
#include <cstdio>
#include "ai_belief/persistent_engine.h"

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

using M6 = Eigen::Matrix<double, 6, 6>;
using V6 = Eigen::Matrix<double, 6, 1>;

int main()
{
    std::printf("jump_evidence_test (F1 round 6c)\n");
    const double pi = 3.125e-9;                                     // the bed class's per-step move probability (0.0027/day, 0.1 s)
    V6 m; m << 0.25, 0.25, 0.01, 0.04, 0.04, 0.25;                 // the bed class's move variance (0.5 m, size stds, 0.5 rad)
    V6 pbar; pbar << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;           // a converged belief, σ 1 cm
    const M6 P = pbar.asDiagonal();                                 // the predicted covariance (θ only here: Λ = W, η = g)
    const auto closed = [&](const V6& w, const V6& g)
    {
        double l = 0.0;
        for (int k = 0; k < 6; ++k)
        {
            const double sk = w(k) + w(k) * w(k) * pbar(k), d = w(k) * w(k) * m(k);
            if (sk > 0.0) l += 0.5 * g(k) * g(k) * d / (sk * (sk + d)) - 0.5 * std::log1p(d / sk);   // 1/s − 1/(s+d) without cancellation
        }
        return l;
    };
    for (const double wH : {1e-6, 1e-9, 1e-12})
        for (const double gH : {0.0, 1.9, 1e3})
        {
            V6 w; w << 1e4, 1e4, wH, 1e4, 1e4, 1e3;
            const M6 W = w.asDiagonal();
            V6 g = V6::Zero(); g(2) = gH;                           // the bias sits on the barely-informed H only
            const auto je = rc::ai::engine::detail::jump_evidence<6, 6>(W, g, P, m, pi);
            const double want = closed(w, g);
            V6 g0 = V6::Zero();
            const double share = je.log_bf - rc::ai::engine::detail::jump_evidence<6, 6>(W, g0, P, m, pi).log_bf;
            std::printf("  W_HH %.0e g_H %.1e: log BF %+.6e (closed form %+.6e) ; H share %+.3e (<= 0.5 g^2 M = %.3e)\n", wH, gH,
                        je.log_bf, want, share, 0.5 * gH * gH * m(2));
            CHECK(std::abs(je.log_bf - want) <= 1e-9 * std::max(1.0, std::abs(want)));
            CHECK(share <= 0.5 * gH * gH * m(2) + 1e-9);
            if (gH == 1.9) CHECK(share <= 0.5 * 1.9 * 1.9 * m(2) + 1e-9 and std::abs(std::log(je.p) - std::log(
                rc::ai::engine::detail::jump_evidence<6, 6>(W, g0, P, m, pi).p)) <= 0.5 * 1.9 * 1.9 * m(2) + 1e-9);
        }
    {   // (3) an informed direction: the data put cx 0.30 m away (g = W·0.30 on a W = 1e4 channel)
        V6 w; w << 1e4, 1e4, 1e4, 1e4, 1e4, 1e3;
        const M6 W = w.asDiagonal();
        V6 g = V6::Zero(); g(0) = 1e4 * 0.30;
        const auto je = rc::ai::engine::detail::jump_evidence<6, 6>(W, g, P, m, pi);
        std::printf("  informed cx, 0.30 m: log BF %.3e, p %.6f, E[v_cx] %.4f\n", je.log_bf, je.p, je.v_mean(0));
        CHECK(je.p > 0.99);
        CHECK(std::abs(je.v_mean(0) - 0.30) < 0.01);
    }
    {   // (4) round 7 A2 (bed_5 update 140, p1_bed): the depth h UNINFORMED (W_hh = 0, W_hH a float denormal) while the gradient
        // carries a stray −3.4e-6 there. The g-space form solved S_s = W + W P W, with an h eigenvalue ~1e-93: on these inputs
        // (BASE 75b3fd8a) log BF 3.1e54, P(moved) 1, |E[v]| 2.5e25; live, log BF 1.3e62, ω → +134, DEGENERATE P. The information form: finite, and h's share is the difference of the
        // ½ η_h² P_hh terms between the hypotheses, ½ η_h²·M_h = 2.3e-13 (no Λ on h to shrink it).
        V6 w; w << 1.69e4, 6.6e-2, 9.26e5, 4.23e3, 0.0, 2.38e3;        // the update's W diagonal (h = index 4 uninformed)
        M6 W = w.asDiagonal();
        W(4, 2) = W(2, 4) = 2.803e-45;                                  // the live W_hH: the smallest float denormal (the
                                                                        // channel's float information cast to double)
        V6 g; g << 37.4, 0.074, -5605.0, 39.9, -3.4e-6, -0.82;
        const auto je = rc::ai::engine::detail::jump_evidence<6, 6>(W, g, P, m, pi);
        V6 g0 = g; g0(4) = 0.0;
        const auto j0 = rc::ai::engine::detail::jump_evidence<6, 6>(W, g0, P, m, pi);
        const double share = je.log_bf - j0.log_bf, want = 0.5 * g(4) * g(4) * m(4);
        std::printf("  uninformed h with a stray gradient: log BF %.6e (finite %d), h share %.3e (closed form %.3e), |E[v]| max %.3e\n",
                    je.log_bf, int(std::isfinite(je.log_bf)), share, want, je.v_mean.cwiseAbs().maxCoeff());
        CHECK(std::isfinite(je.log_bf) and std::isfinite(je.p) and je.v_mean.allFinite() and je.v_cov.allFinite());
        CHECK(std::abs(share - want) <= 1e-12);
        CHECK(je.v_mean.cwiseAbs().maxCoeff() <= 1.0);                 // a posterior move within a few √M, not 1e30
    }
    std::printf(bad ? "jump_evidence_test: %d FAILED\n" : "jump_evidence_test: OK\n", bad);
    return bad != 0;
}
