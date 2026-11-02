// Build (from room_concept/):
//   g++ -std=c++23 -O2 -Isrc -I/usr/include/eigen3 tools/reloc_rivals_test.cpp src/reloc_rival_score.cpp src/reloc_search.cpp src/wall_segmenter.cpp -o /tmp/reloc_rivals_test && /tmp/reloc_rivals_test
#include "reloc_rivals.h"
#include "reloc_rival_score.h"
#include <chrono>
#include <cstdio>
#include <print>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, int line, const char* what)
{
    ++checks;
    if (not ok) { ++failures; std::println("FAIL line {}: {}", line, what); }
}
#define CHECK(c) check((c), __LINE__, #c)
using namespace rc::rivals;
using V3 = Eigen::Vector3f;
using M3 = Eigen::Matrix3f;


// ---- synthetic room + scan for the scorer tests -------------------------------------------------------------
using V2 = Eigen::Vector2f;
std::vector<V2> room_poly() { return {V2(-3, -2), V2(3, -2), V2(3, 2), V2(-3, 2)}; }

// ray from o along angle a to the rectangle's boundary
float ray_range(const V2& o, float a)
{
    const V2 d(std::cos(a), std::sin(a));
    float t = 1e9f;
    if (std::abs(d.x()) > 1e-6f) for (float X : {-3.f, 3.f}) { const float s = (X - o.x()) / d.x(); if (s > 0) t = std::min(t, s); }
    if (std::abs(d.y()) > 1e-6f) for (float Y : {-2.f, 2.f}) { const float s = (Y - o.y()) / d.y(); if (s > 0) t = std::min(t, s); }
    return t;
}
// door_half > 0: beams crossing the x=3 wall within |y|<=half go through the door and hit the NEXT room's wall at x = door_far
std::vector<V2> make_scan(const V3& pose, int n, float door_half = 0.f, float door_far = 0.f)
{
    std::vector<V2> pts;
    for (int i = 0; i < n; ++i)
    {
        const float a = -std::numbers::pi_v<float> + 2.f * std::numbers::pi_v<float> * (i + 0.5f) / n;
        const float wa = a + pose.z();
        float r = ray_range(pose.head<2>(), wa);
        const V2 hit = pose.head<2>() + r * V2(std::cos(wa), std::sin(wa));
        if (door_half > 0.f and std::abs(hit.x() - 3.f) < 1e-3f and std::abs(hit.y()) <= door_half) r = (door_far - pose.x()) / std::cos(wa);
        pts.push_back(r * V2(std::cos(a), std::sin(a)));
    }
    return pts;
}

Rival make(float x, float lam, std::uint64_t id = 1)
{
    Rival r; r.pose = V3(x, 0.f, 0.f); r.cov = M3(Eigen::Vector3f(4e-4f, 4e-4f, 1e-4f).asDiagonal());
    r.lambda = r.lambda_birth = lam; r.id = id; return r;
}
}

int main()
{
    Params P;
    { // birth_price
        const M3 cr = M3(V3(4e-4f, 4e-4f, 1e-4f).asDiagonal());
        const M3 Pp = M3(V3(1e-2f, 1e-2f, 2.5e-3f).asDiagonal());
        const float l = birth_lambda(P, -100.f, cr, -100.f, V3(1, 1, 0), cr, V3(1, 1, 0), Pp, 40.f * 2.f * std::numbers::pi_v<float>);
        std::println("birth lambda = {:.2f}", l);
        CHECK(l >= -18.f and l <= -14.f);
        CHECK(std::abs(l - (-17.2776f)) < 1e-2f);   // hand-computed sec 4.2 value
        const float V = 40.f * 2.f * std::numbers::pi_v<float>;
        const float lo = birth_lambda(P, -100.f, cr, -100.f, V3(1.1f, 1, 0), cr, V3(1, 1, 0), Pp, V);
        CHECK(std::abs(lo - (-16.7776f)) < 1e-2f);  // +1/2 d'P^-1 d = +0.5 for d = 0.1 m at sigma 0.1
        const float lw = birth_lambda(P, -100.f, cr, -100.f, V3(1, 1, std::numbers::pi_v<float> - 0.01f), cr,
                                      V3(1, 1, -std::numbers::pi_v<float> + 0.01f), Pp, V);
        CHECK(std::abs(lw - (-17.1976f)) < 1e-2f);  // yaw diff wraps to 0.02 rad, not 2pi-0.02
    }
    { // twin_never_switches
        Belief b(P); b.add(make(2.f, -16.f));
        for (int i = 0; i < 1000; ++i) b.update({0.f}, 1.f);
        CHECK(not b.winner().has_value());
        CHECK(b.rivals()[0].lambda == b.rivals()[0].lambda_birth);
    }
    { // mismatch_cancels
        Belief b(P); b.add(make(2.f, -10.f));
        const float pen = 30.f, ll0 = -100.f - pen, llr = -100.f - pen;   // same penalty on both
        b.update({llr - ll0}, 1.f);
        CHECK(b.rivals()[0].lambda == -10.f);
    }
    { // kidnap_two_scans: two novel scans, +40 each
        Belief b(P); b.add(make(2.f, -16.f));
        b.update({40.f}, 1.f);
        CHECK(not b.winner().has_value() or b.rivals()[0].lambda >= P.log_band);
        b.update({40.f}, 1.f);
        CHECK(b.winner().has_value() and *b.winner() == 0);
    }
    { // parked_no_accumulation
        Belief b(P); b.add(make(2.f, -16.f));
        for (int i = 0; i < 500; ++i) b.update({5.f}, 0.f);
        CHECK(not b.winner().has_value());
    }
    { // outlier_scan_bounded
        Belief b(P); b.add(make(2.f, -16.f));
        b.update({10.f}, 1.f);
        const float before = b.rivals()[0].lambda;
        b.update({-1e6f}, 1.f);
        CHECK(before - b.rivals()[0].lambda <= std::abs(P.ln_eps) + 1e-4f);
        CHECK(b.rivals()[0].lambda >= b.rivals()[0].lambda_birth);
        for (int i = 0; i < 10; ++i) b.update({-1e6f}, 1.f);
        CHECK(b.rivals()[0].lambda == -16.f);
    }
    { // swap_is_honest
        Belief b(P); b.add(make(2.f, -16.f));
        b.update({19.f}, 1.f);   // lambda = 3: just past the band
        const float li = b.rivals()[0].lambda;
        const M3 c0 = M3::Identity() * 1e-3f;
        const Rival won = b.swap(0, V3(0, 0, 0), c0, -16.f);
        CHECK(won.lambda == li and won.pose.x() == 2.f);
        CHECK(b.rivals().size() == 1 and b.rivals()[0].lambda == -li and b.rivals()[0].pose.norm() == 0.f);
        CHECK(b.rivals()[0].lambda_birth == -16.f and b.rivals()[0].cov == c0);
        b.update({-4.f}, 1.f);
        CHECK(b.rivals()[0].lambda == -li - 4.f);   // decays from -li ...
        for (int i = 0; i < 10; ++i) b.update({-4.f}, 1.f);
        CHECK(b.rivals()[0].lambda == -16.f);       // ... down to the birth floor, not stuck at -li
    }
    { // kmax_and_merge
        Belief b(P);
        for (int i = 0; i < 6; ++i) b.add(make(2.f * (i + 1), -16.f + i, i + 1));
        CHECK(b.rivals().size() == 4);
        float mn = 1e9f; for (const auto& r : b.rivals()) mn = std::min(mn, r.lambda);
        CHECK(mn == -14.f);   // the two lowest (-16,-15) were dropped
        const float x0 = b.rivals()[0].pose.x();
        b.add(make(x0 + 0.01f, 5.f, 99));   // higher lambda: the merged rival must be the new one
        CHECK(b.rivals().size() == 4);
        bool ok = false; for (const auto& r : b.rivals()) ok = ok or (r.id == 99 and r.lambda == 5.f);
        CHECK(ok);
    }
    { // p_rival, spread, carry sanity
        Belief b(P); b.add(make(2.f, 0.f));
        CHECK(std::abs(b.p_rival() - 0.5f) < 1e-5f);
        CHECK(std::abs(b.spread(V3::Zero())(0, 0) - 2.f) < 1e-4f);   // w=0.5, d=2 -> 0.5*4
        b.carry(V3(1.f, 0.f, 0.f));
        CHECK(std::abs(b.rivals()[0].pose.x() - 3.f) < 1e-5f);
        b.clear(); CHECK(b.rivals().empty() and b.p_rival() == 0.f);
    }
    { // score_matches_search: the scorer IS the search's per-pose likelihood
        rc::reloc::Params rp;
        const auto poly = room_poly();
        rc::reloc::LikelihoodRaster raster; raster.build(poly, rp);
        const V3 truth(0.7f, -0.4f, 0.5f);
        const auto pts = make_scan(truth, 360);
        const auto res = rc::reloc::search(raster, poly, pts, {truth}, rp);
        CHECK(not res.modes.empty());
        if (not res.modes.empty())
        {
            const auto& m = res.modes.front();
            const float s = rc::rivals::score(raster, poly, pts, m.pose, rp);
            std::println("score {:.4f} vs Mode::log_lik {:.4f}", s, m.log_lik);
            CHECK(std::abs(s - m.log_lik) <= 1e-3f * std::max(1.f, std::abs(m.log_lik)));
        }
        // dll: truth beats a rival 0.5 m away; the incumbent's own dll is 0
        const std::vector<V3> riv = {V3(1.2f, -0.4f, 0.5f), V3(0.7f, -0.4f, 0.5f)};
        const auto d = rc::rivals::dll(raster, poly, pts, truth, riv, rp);
        CHECK(d.size() == 2 and d[0] < -5.f and std::abs(d[1]) < 1e-4f);
        // timing: 4 rivals x <=128 points
        const std::vector<V3> r4 = {V3(1.2f, -0.4f, 0.5f), V3(0.7f, 0.1f, 0.5f), V3(0.7f, -0.4f, 0.9f), V3(-1.f, 0.f, 0.f)};
        const auto p128 = make_scan(truth, 128);
        const int N = 2000;
        const auto t0 = std::chrono::steady_clock::now();
        float acc = 0.f;
        for (int i = 0; i < N; ++i) acc += rc::rivals::dll(raster, poly, p128, truth, r4, rp)[0];
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / N;
        std::println("dll 4 rivals x 128 pts: {:.4f} ms (acc {:.1f})", ms, acc);
        CHECK(ms < 0.1);
    }
    { // apertures_mask_both: through-door returns lower H0 and a rival alike (they are explained as "beyond"), so they add
      // no spurious evidence. The invariant that holds in the real model: dll on the open-door scan ~ dll on the SAME scan with
      // those returns deleted (<6 nats of ~90), where without the aperture model the through returns are charged as clutter
      // to whichever pose places them worse. (The closed-door scan is NOT the reference: its 26 wall returns on x=3 carry real
      // information about x, which an open door removes -- dll differs by ~25 nats by information, not by model error.)
        rc::reloc::Params rp;
        rc::reloc::Params pd = rp;
        rc::reloc::Aperture ap;
        ap.a = V2(3.f, -0.8f); ap.b = V2(3.f, 0.8f); ap.n_out = V2(1.f, 0.f); ap.open = 0.9f; ap.p_shut = 0.1f; ap.sigma_along = 0.05f;
        pd.apertures = {ap};
        const auto poly = room_poly();
        rc::reloc::LikelihoodRaster raster; raster.build(poly, pd);
        const V3 truth(-0.5f, 0.2f, 0.f), rival(-0.3f, 0.2f, 0.f);
        const auto closed = make_scan(truth, 360);
        const auto open = make_scan(truth, 360, 0.8f, 9.f);
        std::vector<V2> deleted;
        for (std::size_t i = 0; i < closed.size(); ++i) if ((closed[i] - open[i]).norm() < 1.f) deleted.push_back(open[i]);
        std::println("through-door points: {} of {}", closed.size() - deleted.size(), closed.size());
        CHECK(closed.size() - deleted.size() >= 20);
        const float d_open = rc::rivals::dll(raster, poly, open, truth, {rival}, pd)[0];
        const float d_del = rc::rivals::dll(raster, poly, deleted, truth, {rival}, rp)[0];
        const float d_open_noap = rc::rivals::dll(raster, poly, open, truth, {rival}, rp)[0];
        std::println("dll open(aperture) {:.2f}  deleted {:.2f}  open(no aperture) {:.2f}", d_open, d_del, d_open_noap);
        CHECK(std::abs(d_open - d_del) < 6.f);
        // both poses are lowered by the through-door returns (H0 and rival alike)
        const float h_c = rc::rivals::score(raster, poly, closed, truth, pd), h_o = rc::rivals::score(raster, poly, open, truth, pd);
        const float r_c = rc::rivals::score(raster, poly, closed, rival, pd), r_o = rc::rivals::score(raster, poly, open, rival, pd);
        CHECK(h_o < h_c and r_o < r_c);
    }
    std::println("{} checks, {} failures", checks, failures);
    return failures ? 1 : 0;
}
