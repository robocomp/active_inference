// test_view_ledger.cpp — Task 4: the view ledger.
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -w -I/usr/include/eigen3 -I/usr/include/opencv4 -I../common tests/test_view_ledger.cpp -o /tmp/test_view_ledger && /tmp/test_view_ledger
#include <clocale>
#include <cmath>
#include <cstdio>
#include "../src/core/view_ledger.h"
using namespace rc::door::core;
static int fails = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)
static const DetectorModel M = default_detectors().zed;
static const Vec2f DOOR(2.f, 1.f);
static const Vec2f NRM(0.f, 1.f);
static Vec2f cam_at(float deg, float d = 3.f) { const float a = deg * float(M_PI) / 180.f; return DOOR + d * Vec2f(std::cos(a), std::sin(a)); }
static void stare(ViewLedger& L, const Vec2f& cam, int n, float q, float hit_w, float spread = 0.f, int visit = 0)
{
    std::size_t v = 0;
    for (int i = 0; i < n; ++i) { v = L.observe(cam, DOOR, NRM, 1.0f, spread, visit); L.add_bernoulli(v, Channel::Zed, q, M, hit_w); }
    L.recompute(v, false);
}
int main()
{
    std::setlocale(LC_ALL, "");
    const float one_miss = miss_llr(0.6f, false_alarm_eff(M, 0.6f));
    { ViewLedger L; stare(L, cam_at(0), 200, 0.6f, 0.f);   // parked_stare_is_one_view
      CHECK(L.views().size() == 1); CHECK(std::fabs(L.sum() - one_miss) < 1e-5f); CHECK(one_miss < 0.f); }
    { ViewLedger L; stare(L, cam_at(0), 20, 0.6f, 0.f); stare(L, cam_at(60), 20, 0.6f, 0.f);   // new_standpoint_adds_view
      CHECK(L.views().size() == 2); CHECK(std::fabs(L.sum() - 2 * one_miss) < 1e-5f); }
    { ViewLedger L; stare(L, cam_at(0), 20, 0.6f, 0.f, 4.f); stare(L, cam_at(60), 20, 0.6f, 0.f, 4.f);   // pose_spread_merges
      CHECK(L.views().size() == 1); }
    { ViewLedger L; std::size_t v = 0;   // hit_then_miss_same_view_is_mean
      for (int i = 0; i < 5; ++i) { v = L.observe(cam_at(0), DOOR, NRM, 1, 0, 0); L.add_bernoulli(v, Channel::Zed, 0.6f, M, 1.f); }
      for (int i = 0; i < 5; ++i) { v = L.observe(cam_at(0), DOOR, NRM, 1, 0, 0); L.add_bernoulli(v, Channel::Zed, 0.6f, M, 0.f); }
      L.recompute(v, false);
      const float e = 0.5f * hit_llr(0.6f, M, 1.f) + 0.5f * one_miss;
      CHECK(L.views().size() == 1); CHECK(std::fabs(L.sum() - e) < 1e-5f); }
    { ViewLedger L; stare(L, cam_at(0), 3, 0.f, 0.f);   // all-q==0 view is exactly 0
      CHECK(L.sum() == 0.f); }
    for (bool may : {false, true})   // geometry_up_only
    { ViewLedger L; const auto v = L.observe(cam_at(0), DOOR, NRM, 1, 0, 0);
      L.add_geometry(v, 0.f, 1.f, 0.5f, 1.f, 0.25f);   // x nearer the absent mean => negative
      const float ch = L.recompute(v, may);
      const float neg = ((0.f-0.5f)*(0.f-0.5f) - (0.f-1.f)*(0.f-1.f)) / (2*0.25f);
      if (may) { CHECK(std::fabs(ch - neg) < 1e-5f); CHECK(neg < 0); } else CHECK(ch == 0.f && L.sum() == 0.f); }
    { // ledger_invariant_to_room_frame
      const float th = 0.7f; const Vec2f t(5.f, -3.f);
      Eigen::Rotation2Df Rm(th);
      ViewLedger A, B;
      for (float deg : {0.f, 60.f})
          for (int i = 0; i < 5; ++i)
          {
              const Vec2f c = cam_at(deg); const Vec2f c2 = Rm * c + t, d2 = Rm * DOOR + t, n2 = Rm * NRM;
              auto va = A.observe(c, DOOR, NRM, 1, 0, 0); A.add_bernoulli(va, Channel::Zed, 0.6f, M, 0.f); A.recompute(va, false);
              auto vb = B.observe(c2, d2, n2, 1, 0, 0); B.add_bernoulli(vb, Channel::Zed, 0.6f, M, 0.f); B.recompute(vb, false);
          }
      CHECK(A.views().size() == B.views().size()); CHECK(std::fabs(A.sum() - B.sum()) < 1e-5f);
      for (std::size_t i = 0; i < A.views().size(); ++i)
      { CHECK(std::fabs(A.views()[i].key.range - B.views()[i].key.range) < 1e-5f);
        CHECK(std::fabs(A.views()[i].key.bearing - B.views()[i].key.bearing) < 1e-5f); } }
    { // persisted_keys_survive_reregistration
      ViewLedger L; stare(L, cam_at(0), 5, 0.6f, 0.f); stare(L, cam_at(60), 5, 0.6f, 0.f);
      auto P = *ViewLedger::parse(L.serialise(), false);
      Eigen::Rotation2Df Rm(1.9f); const Vec2f t(-7.f, 4.f);
      const auto v = P.observe(Rm * cam_at(60) + t, Rm * DOOR + t, Rm * NRM, 1, 0, 0);
      CHECK(P.views().size() == 2 and v == 1); }
    { // domain: R = 0 and s2 = 0 contribute exactly 0
      ViewLedger L; const auto bad = L.observe(cam_at(0), DOOR, NRM, 0.f, 0.f, 0);
      CHECK(bad == ViewLedger::npos and L.views().empty());
      L.add_bernoulli(bad, Channel::Zed, 0.6f, M, 0.f); CHECK(L.recompute(bad, true) == 0.f and L.sum() == 0.f);
      const auto v = L.observe(cam_at(0), DOOR, NRM, 1, 0, 0);
      L.add_geometry(v, 0.f, 1.f, 0.5f, 1.f, 0.f); L.recompute(v, true);
      CHECK(L.sum() == 0.f and std::isfinite(L.sum())); }
    { // parse rejects non-finite, recomputes a lying llr
      ViewLedger L; stare(L, cam_at(0), 5, 0.6f, 0.f); auto s = L.serialise();
      CHECK(ViewLedger::parse("VL1 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1", false).has_value());   // control: same layout parses
      CHECK(not ViewLedger::parse("VL1 1 nan 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1", false).has_value());
      CHECK(not ViewLedger::parse("VL1 1 inf 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1", false).has_value());
      const auto pos = s.find(' ', s.find(' ', s.find(' ', s.find(' ', s.find(' ', 4) + 1) + 1) + 1) + 1);   // after actuation
      (void)pos;
      auto P = *ViewLedger::parse(s, false); CHECK(std::fabs(P.sum() - L.sum()) < 1e-6f); }
    { // serialise_roundtrip_es_locale
      std::setlocale(LC_ALL, "es_ES.UTF-8");
      ViewLedger L; stare(L, cam_at(0), 7, 0.6f, 0.3f); stare(L, cam_at(90), 4, 0.35f, 1.f);
      { auto v = L.observe(cam_at(0), DOOR, NRM, 1, 0, 0); L.add_geometry(v, 0.3f, 0.8f, 0.f, 1.f, 0.2f); L.add_actuation(v, 2.3f); L.recompute(v, true); }
      const auto s = L.serialise();
      CHECK(s.find(',') == std::string::npos and s.find('\n') == std::string::npos);
      const auto P = ViewLedger::parse(s, true);
      CHECK(P.has_value()); CHECK(P->serialise() == s); CHECK(P->views().size() == 2); CHECK(P->sum() == L.sum());
      CHECK(not ViewLedger::parse("VL1 2 1 2", false).has_value()); CHECK(not ViewLedger::parse("garbage", false).has_value());
      std::setlocale(LC_ALL, ""); }
    { // support_one_llr_per_view: 200 frames of one view of x are ONE ℓ, the mean x scored once; a new view adds another
      const SupportModel m = default_detectors().support;
      ViewLedger L;
      const auto one = [&](float deg, float x, int n) { std::size_t v = 0; for (int i = 0; i < n; ++i) { v = L.observe(cam_at(deg), DOOR, NRM, 1, 0, 0); L.add_support(v, x, 1.f, m); L.recompute(v, false); } };
      one(0, 0.95f, 200);
      CHECK(L.views().size() == 1); CHECK(std::fabs(L.sum() - support_llr(0.95f, m)) < 1e-4f);
      one(0, 0.05f, 200);   // the same view recovers: its ℓ moves continuously with the view's mean x
      CHECK(L.views().size() == 1); CHECK(std::fabs(L.sum() - support_llr(0.5f, m)) < 1e-3f);
      one(90, 0.95f, 1);
      CHECK(L.views().size() == 2); CHECK(std::fabs(L.sum() - support_llr(0.5f, m) - support_llr(0.95f, m)) < 1e-3f); }
    { // support_weighted_by_kappa: κ scales the view's ℓ; κ = 0 contributes exactly 0 (pose lost, Review Focus 1)
      const SupportModel m = default_detectors().support;
      ViewLedger L; auto v = L.observe(cam_at(0), DOOR, NRM, 1, 0, 0);
      L.add_support(v, 0.95f, 0.f, m); L.recompute(v, false); CHECK(L.sum() == 0.f);
      L.add_support(v, 0.95f, 0.5f, m); L.recompute(v, false);   // one frame of κ 0 and one of κ 0.5 at the same x
      CHECK(std::fabs(L.sum() - 0.25f * support_llr(0.95f, m)) < 1e-4f); }
    { // support_serialise: VL2 round-trips the support channel; a VL1 row (no support block) still loads, its channel empty
      const SupportModel m = default_detectors().support;
      ViewLedger L; { auto v = L.observe(cam_at(0), DOOR, NRM, 1, 0, 0); L.add_support(v, 0.95f, 1.f, m); L.recompute(v, false); }
      const auto s = L.serialise();
      CHECK(s.starts_with("VL2 ")); CHECK(s.find(',') == std::string::npos);
      const auto P = ViewLedger::parse(s, false);
      CHECK(P.has_value()); CHECK(P->serialise() == s); CHECK(std::fabs(P->sum() - L.sum()) < 1e-5f);
      const auto old = ViewLedger::parse("VL1 1 0 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1", false);
      CHECK(old.has_value()); CHECK(old->views().size() == 1); CHECK(old->views()[0].sup.n == 0); CHECK(old->sum() == 0.f); }
    { // ricoh_miss_counts_under_flag: a Ricoh view with q 0.8 and no hit lowers the view llr by miss_llr(q, f); flag off: unchanged (hits only)
      const DetectorModel R = default_detectors().ricoh;
      for (bool flag : {false, true})
      { ViewLedger L; L.ricoh_counts_misses = flag; const auto v = L.observe(cam_at(0), DOOR, NRM, 1, 0, 0);
        L.add_bernoulli(v, Channel::Ricoh, 0.8f, R, 0.f); L.recompute(v, false);
        const float m = miss_llr(0.8f, false_alarm_eff(R, 0.8f));
        if (flag) { CHECK(m < 0.f); CHECK(std::fabs(L.sum() - m) < 1e-5f); } else CHECK(L.sum() == 0.f); } }
    { // geometry_may_refute_under_flag: x at mu0 - 2 s is evidence AGAINST (one-model, no clamp); the old path stays 0 with the clamp
      const float mu0 = 0.5f, s = 0.1f;
      ViewLedger L; const auto v = L.observe(cam_at(0), DOOR, NRM, 1, 0, 0);
      L.add_geometry_t(v, mu0 - 2 * s, 1.f, mu0, s * s, 2.f, 5); L.recompute(v, false);
      CHECK(L.sum() < 0.f);
      ViewLedger O; const auto w = O.observe(cam_at(0), DOOR, NRM, 1, 0, 0);
      O.add_geometry(w, mu0 - 2 * s, 1.f, mu0, mu0 + 2 * s, s * s); O.recompute(w, false); CHECK(O.sum() == 0.f); }
    { // geometry_no_repricing: constant x, s alternating 0.015 / 0.25 — each frame priced by its OWN s, so the view's term moves by at most
      // (|this frame's term| + |the previous view term|) / n per added frame; the old running-mean-with-overwritten-s form jumps by ~1/s
      const float x = 0.5f, mu0 = 0.f; ViewLedger L; const auto v = L.observe(cam_at(0), DOOR, NRM, 1, 0, 0);
      float prev = 0.f;
      for (int i = 0; i < 8; ++i)
      {
          const float s = i % 2 ? 0.25f : 0.015f;
          ViewLedger one; const auto o = one.observe(cam_at(0), DOOR, NRM, 1, 0, 0); one.add_geometry_t(o, x, 1.f, mu0, s * s, 2.f, 3); one.recompute(o, false);
          const float own = one.sum();   // this frame's term alone (n = 1)
          L.add_geometry_t(v, x, 1.f, mu0, s * s, 2.f, 3); L.recompute(v, false);
          CHECK(std::fabs(L.sum() - prev) <= (std::fabs(own) + std::fabs(prev)) / float(i + 1) + 1e-4f);
          CHECK(std::fabs(own) < 10.f);   // Student-t: log growth, never the 1/s^2 of a Gaussian
          prev = L.sum();
      }
      ViewLedger G; const auto g = G.observe(cam_at(0), DOOR, NRM, 1, 0, 0);   // the old form, same inputs, for contrast
      float gprev = 0.f, gmax = 0.f;
      for (int i = 0; i < 8; ++i) { const float s = i % 2 ? 0.25f : 0.015f; G.add_geometry(g, x, 1.f, mu0, mu0 + 2 * s, s * s); G.recompute(g, true); gmax = std::max(gmax, std::fabs(G.sum() - gprev)); gprev = G.sum(); }
      CHECK(gmax > 50.f); }
    { // geometry_zero_when_noninformative: mu1_sd = 0 (a fitted "no information") => exactly 0
      ViewLedger L; const auto v = L.observe(cam_at(0), DOOR, NRM, 1, 0, 0);
      for (int i = 0; i < 5; ++i) { L.add_geometry_t(v, 0.9f - 0.1f * i, 1.f, 0.2f, 0.01f, 0.f, 4); L.recompute(v, false); CHECK(L.sum() == 0.f); } }
    { // VL4: one-model geometry round-trips; a ledger without it still writes VL3
      ViewLedger L; const auto v = L.observe(cam_at(0), DOOR, NRM, 1, 0, 0);
      L.add_geometry_t(v, 0.7f, 0.8f, 0.2f, 0.04f, 2.f, 4); L.recompute(v, false);
      const auto s = L.serialise(); CHECK(s.starts_with("VL4 ")); CHECK(s.find(',') == std::string::npos);
      const auto P = ViewLedger::parse(s, false, true); CHECK(P.has_value()); CHECK(P->serialise() == s); CHECK(std::fabs(P->sum() - L.sum()) < 1e-4f);
      ViewLedger N; stare(N, cam_at(0), 3, 0.6f, 0.f); CHECK(N.serialise().starts_with("VL3 ")); }
    { // revisit_replaces_not_adds
      ViewLedger L; stare(L, cam_at(0), 20, 0.6f, 0.f);
      auto P = *ViewLedger::parse(L.serialise(), false); const float s0 = P.sum();
      const auto v = P.observe(cam_at(0), DOOR, NRM, 1, 0, 0); P.add_bernoulli(v, Channel::Zed, 0.6f, M, 0.f);
      CHECK(P.views().size() == 1); CHECK(P.sum() == s0);
      P.recompute(v, false); CHECK(std::fabs(P.sum() - s0) < 1e-4f);   // replaced, not added
      const auto v2 = P.observe(cam_at(0), DOOR, NRM, 1, 0, 1); CHECK(P.views().size() == 2 and v2 == 1); }
    std::printf(fails ? "FAILED %d\n" : "ALL PASS\n", fails);
    return fails != 0;
}
