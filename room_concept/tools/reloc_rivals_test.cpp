// Build (from room_concept/):
//   g++ -std=c++23 -O2 -Isrc -I/usr/include/eigen3 tools/reloc_rivals_test.cpp -o /tmp/reloc_rivals_test && /tmp/reloc_rivals_test
#include "reloc_rivals.h"
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
    }
    { // twin_never_switches
        Belief b(P); b.add(make(2.f, -16.f));
        for (int i = 0; i < 1000; ++i) b.update({0.f}, 1.f);
        CHECK(not b.winner().has_value());
        CHECK(b.rivals()[0].lambda == b.rivals()[0].lambda_birth);
    }
    { // mismatch_cancels
        Belief b(P); b.add(make(2.f, -10.f));
        const float ll0 = -30.f, llr = -30.f;
        b.update({llr - ll0}, 1.f);
        CHECK(b.rivals()[0].lambda == -10.f);
    }
    { // kidnap_two_scans
        Belief b(P); b.add(make(2.f, -16.f));
        b.update({60.f}, 1.f);
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
        b.update({60.f}, 1.f);
        const float li = b.rivals()[0].lambda;
        const Rival won = b.swap(0, V3(0, 0, 0), M3::Identity() * 1e-3f);
        CHECK(won.lambda == li and won.pose.x() == 2.f);
        CHECK(b.rivals().size() == 1 and b.rivals()[0].lambda == -li and b.rivals()[0].pose.norm() == 0.f);
    }
    { // kmax_and_merge
        Belief b(P);
        for (int i = 0; i < 6; ++i) b.add(make(2.f * (i + 1), -16.f + i, i + 1));
        CHECK(b.rivals().size() == 4);
        float mn = 1e9f; for (const auto& r : b.rivals()) mn = std::min(mn, r.lambda);
        CHECK(mn == -14.f);   // the two lowest (-16,-15) were dropped
        b.add(make(b.rivals()[0].pose.x() + 0.01f, -16.f, 99));
        CHECK(b.rivals().size() == 4);
    }
    { // p_rival, spread, carry sanity
        Belief b(P); b.add(make(2.f, 0.f));
        CHECK(std::abs(b.p_rival() - 0.5f) < 1e-5f);
        CHECK(std::abs(b.spread(V3::Zero())(0, 0) - 2.f) < 1e-4f);   // w=0.5, d=2 -> 0.5*4
        b.carry(V3(1.f, 0.f, 0.f));
        CHECK(std::abs(b.rivals()[0].pose.x() - 3.f) < 1e-5f);
        b.clear(); CHECK(b.rivals().empty() and b.p_rival() == 0.f);
    }
    std::println("{} checks, {} failures", checks, failures);
    return failures ? 1 : 0;
}
