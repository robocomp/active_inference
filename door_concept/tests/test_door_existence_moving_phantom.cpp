// test_door_existence_moving_phantom — Task 5b item 4, fix round (I-2): the door's silhouette POOLED BY VIEW
// (door_view_pool.h add_logodds — the same view model and scale as the image channel: one view's looks are one
// observation at their mean increment, renewed in proportion to each look's novelty), then rc::exist::ExistenceBelief
// (clamp, armed with ρ 0: no second discount) and decide_removal (the shared policy).
// ★RED with the shared class's own run (ρ 0.747 armed into ExistenceBelief, fdb40c3e): a moving phantom at P(detect) 0.3
//   was never removed within 400 looks, and a real door seen from new viewpoints stopped at L +2.42. CONCEPT_AGENT_LIFECYCLE: "occupancy confirms / absence removes / occlusion HOLDS, absence
// weighted by P(detect)". The weighting must keep that: a phantom seen from NEW viewpoints is removed, in a time that
// falls with P(detect); one seen from a single parked pose is not executed on repeats of one look.
//   1. MOVING phantom (each look a new bearing, many of the door's angular half-sizes apart), P(detect) 0.3 and 0.6:
//      removed, and faster at 0.6 — the looks-to-removal are printed (the "time to removal" the review asked for);
//   2. the same misses from ONE parked pose: not removed within the same number of looks (one look repeated);
//   3. a REAL door seen from new viewpoints, P(detect) 0.3: confirmed to the clamp (the weighting is symmetric).
// The evidence per look is the door's own: mask_evidence over a fully unlit 300-cell silhouette (or fully lit), scaled
// by P(detect) exactly as core/door_existence.cpp does, integrated with integrate(ev, p_detect), decided by decide_removal
// with the door's policy (RemovalProb 0.12, RemoveFrames 15, MaxLogodds 4, birth L 1).
//
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -I/usr/include/eigen3 tests/test_door_existence_moving_phantom.cpp -o /tmp/test_door_existence_moving_phantom && /tmp/test_door_existence_moving_phantom
#include <cmath>
#include <print>

#include "../../common/existence_belief/existence_belief.h"
#include "../src/door_config.h"
#include "../src/door_view_pool.h"

namespace
{
int g_fail = 0, g_checks = 0;
void check(bool ok, const char* what)
{
    ++g_checks;
    if (not ok) { ++g_fail; std::println("  FAIL: {}", what); }
}

// looks until the shared policy removes the door (or -1 within max_looks); `moving` ⇒ the camera steps 0.6 rad around a
// door 4 m away each look (its angular half-size atan(0.5/4) = 0.12 rad), else it stays put.
int looks_to_removal(float p_detect, bool moving, bool present, int max_looks, float* L_end = nullptr)
{
    const rc::DoorConfig cfg;
    rc::exist::RemovalPolicy pol;
    pol.logodds_max = cfg.exist_max_logodds; pol.removal_prob = cfg.exist_removal_prob;
    pol.remove_frames = static_cast<float>(cfg.exist_remove_frames);
    pol.frame_correlation = 0.0f;   // the door pools its silhouette itself (door_view_pool.h): applied once
    rc::exist::SensorModel sm;
    sm.detection_prob = cfg.exist_detection_prob; sm.clutter_prob = cfg.exist_clutter_prob; sm.sensor_sigma_m = cfg.exist_sensor_sigma_m;
    rc::exist::ExistenceBelief b(cfg.exist_birth_logodds, cfg.exist_max_logodds);
    rc::exist::arm(b, pol);
    rc::exist::RemovalDebounce deb;
    rc::door::ViewPool pool;
    for (int k = 0; k < max_looks; ++k)
    {
        const float ang = moving ? 0.6f * static_cast<float>(k % 5) - 1.2f : 0.0f;
        const Eigen::Vector2f cam(4.0f * std::sin(ang), -4.0f * std::cos(ang));
        const float e_occ = present ? 300.f : 0.f, e_free = present ? 0.f : 300.f;
        rc::exist::Evidence ev;
        ev.e_occ = e_occ; ev.e_free = e_free; ev.n_reached = 300;
        ev.log_odds_delta = p_detect * rc::exist::mask_evidence(e_occ, e_free, 300, sm).log_odds_delta;
        ev.log_odds_delta = pool.add_logodds(cam, {0.f, 0.f}, 0.5f * std::hypot(1.0f, 2.1f), ev.log_odds_delta);
        const float L0 = b.logodds();
        b.integrate(ev, p_detect);
        pool.commit(b.logodds() - L0);
        if (rc::exist::decide_removal(b, deb, pol, p_detect).remove) { if (L_end) *L_end = b.logodds(); return k + 1; }
    }
    if (L_end) *L_end = b.logodds();
    return -1;
}
}  // namespace

int main()
{
    const rc::DoorConfig cfg;

    std::println("1. a MOVING phantom");
    float L3 = 0, L6 = 0;
    const int n3 = looks_to_removal(0.3f, true, false, 400, &L3), n6 = looks_to_removal(0.6f, true, false, 400, &L6);
    std::println("    P(detect) 0.3: removed after {} looks ({:.1f} s at 10 Hz); P(detect) 0.6: after {} looks ({:.1f} s)",
                 n3, n3 * 0.1f, n6, n6 * 0.1f);
    check(n3 > 0, "a moving phantom at P(detect) 0.3 is never removed (absence must remove)");
    check(n6 > 0, "a moving phantom at P(detect) 0.6 is never removed");
    check(n6 > 0 and n3 > 0 and n6 <= n3, "removal is not faster at the higher P(detect)");

    std::println("2. the same misses from ONE parked pose");
    float Lp = 0;
    const int np = looks_to_removal(0.6f, false, false, std::max(n6, 1), &Lp);
    std::println("    parked, P(detect) 0.6, {} looks: {} (L {:+.2f})", std::max(n6, 1), np < 0 ? "kept" : "removed", Lp);
    check(np < 0, "a parked robot's repeats of one look removed the door as fast as new looks");

    std::println("3. a REAL door from new viewpoints, P(detect) 0.3");
    float Lr = 0;
    (void)looks_to_removal(0.3f, true, true, 60, &Lr);
    std::println("    L after 60 looks {:+.2f} (clamp {:.1f})", Lr, cfg.exist_max_logodds);
    check(Lr >= cfg.exist_max_logodds - 1e-3f, "a real door seen from new viewpoints is not confirmed to the clamp");
    std::println("{} / {} checks passed", g_checks - g_fail, g_checks);
    return g_fail == 0 ? 0 : 1;
}
