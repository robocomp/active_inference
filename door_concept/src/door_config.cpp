/*
 * door_config.cpp — fill DoorConfig from a RoboComp ConfigLoader.
 */

#include "door_config.h"
#include "door_config_fields.h"   // read_door_config_fields — every key, shared with tools/door_replay

#include "../../common/config_report/config_read.h"   // rc::cfg::Reader (SHARED)

#include <print>

#include "../../common/concept_manifest/concept_manifest.h"   // rc::manifest (SHARED)

#include <genericworker.h>   // ConfigLoader

namespace rc {

DoorConfig load_door_config(const ConfigLoader& cfg)
{
    DoorConfig out;


    // The one place this path is written. Relative to the agent's CWD (<agent>/), not to src/ — the same
    // string was right in one place and wrong in the other for a week, and the manifest was inert the whole time.
    static constexpr const char* kManifestPath = "../common/concept_manifest/door.concept.toml";
    // ★★AN INHERITED WORLD FACT IS FATAL — rc::manifest::provenance_ok. `from = "inherited"` means the number
    // arrived by a rename and nobody chose it for THIS object; hood shipped ten such defects in a week with
    // several of them declared, in writing, in its manifest. A note stopped nothing, so this stops the agent.
    if (not rc::manifest::provenance_ok(kManifestPath, "door"))
        std::exit(EXIT_FAILURE);

    // ConfigLoader::get has no default overload; TOML numeric floats are stored as double.
    // Every read below registers its key, its CODE DEFAULT and a one-line description,
    // so the startup table can say where each value came from - not just what it is.
    // (The four local lambdas now forward to the shared registry; the call sites are
    // unchanged except for that description.)
    rc::cfg::Reader reader(cfg, "door_concept");
    const auto getf = [&](std::string_view k, float def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.f(k, def, what, o); };
    const auto geti = [&](std::string_view k, int def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.i(k, def, what, o); };
    const auto gets = [&](std::string_view k, std::string def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.s(k, std::move(def), what, o); };
    const auto getb = [&](std::string_view k, bool def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.b(k, def, what, o); };

    // Every key read: door_config_fields.h (one template, shared with the replay tool's CSV reader).
    read_door_config_fields(out, getf, geti, gets, getb);

    std::print("door_concept: configuration loaded.\n");
    // ── THE DECLARED VERTICAL SPAN, ADOPTED (shared: rc::manifest::adopt_span) ─────────────────────
    // The manifest states the anchoring ONCE, as a fact about the object, and the span is derived from it —
    // instead of every site that needs a z-band restating the same assumption in its own arithmetic. That
    // restating is how hood_concept, cloned from a floor-anchored parent, ran for days with a LiDAR band
    // DISJOINT from its body while every channel reported full coverage.
    // ★z_top is the door-leaf height prior (2.00), matching the manifest. A door has no LiDAR sweep, so
    // there is no selection band to check — only point ownership.
    {
        const auto span = rc::manifest::adopt_span(kManifestPath, "door",
                                                  rc::manifest::Support::floor_anchored,
                                                  out.door_prior_h_m, out.door_prior_h_m);
        rc::manifest::Geometry decl;
        decl.support  = span.support;
        decl.z_top_m  = out.door_prior_h_m;
        decl.extent_m = out.door_prior_h_m;
        decl.valid    = true;
        bool ok_bands = true;
        ok_bands &= rc::manifest::band_contains_body("door ", "point_ownership",
                        span.z0 - out.support_select_height_margin_m, span.z1 + out.support_select_height_margin_m, decl);
        if (ok_bands)
            std::print("[manifest] door ✓ every derived z-band contains the declared body\n");
    }

    // ── the three pragmatic affordances: approach / open / cross ([DoorAffordance]) ───────────────
    {
        auto& pg = out.pragmatic;
        pg.enabled               = getb("DoorAffordance.Enabled", pg.enabled,
                "distance or an angle");
        pg.actuation_reach_m     = getf("DoorAffordance.ActuationReachM", pg.actuation_reach_m,
                "⚠ApproachStandoffM IS GONE, not renamed");
        pg.cross_standoff_m      = getf("DoorAffordance.CrossStandoffM", pg.cross_standoff_m,
                "how far PAST the aperture the cross target sits");
        pg.cross_clearance_m     = getf("DoorAffordance.CrossClearanceM", pg.cross_clearance_m,
                "distance past the aperture plane that counts as fully through");
        pg.robot_passage_width_m = getf("DoorAffordance.RobotPassageWidthM", pg.robot_passage_width_m,
                "★A MEASUREMENT OF THE ROBOT, not a tuning knob — and the one direction in which a doorway closes");
        pg.passage_margin_m      = getf("DoorAffordance.PassageMarginM", pg.passage_margin_m,
                "clearance demanded on top of the body width");
        pg.pose_sigma_floor_m    = getf("DoorAffordance.PoseSigmaFloorM", pg.pose_sigma_floor_m,
                "Pose-uncertainty floor for door_reach_prob");
        pg.swing_clearance_m     = getf("DoorAffordance.SwingClearanceM", pg.swing_clearance_m,
                "★CLEARANCE FOR THE LEAF'S OWN SWEPT ARC — a SAFETY quantity, not a preference");
        pg.actuation_swing_rate  = getf("DoorAffordance.ActuationSwingRate", pg.actuation_swing_rate,
                "rad/s; 90 deg in ~3.1 s");
        pg.phi_sigma_rad         = getf("DoorAffordance.PhiSigmaRad", pg.phi_sigma_rad,
                "★THE WIDTH OF THE LEAF-ANGLE BELIEF WHEN THE IMAGE IS NOT RESOLVING IT. estimate_phi() refills phi_curve only from a fresh mask, and this agent's own…");
        pg.value_approach        = getf("DoorAffordance.ValueApproach", pg.value_approach,
                "★An epistemic gain is a measured entropy reduction in nats");
        pg.value_open            = getf("DoorAffordance.ValueOpen", pg.value_open,
                "");
        pg.value_cross           = getf("DoorAffordance.ValueCross", pg.value_cross,
                "");
        pg.approach_timeout_s    = getf("DoorAffordance.ApproachTimeoutS", pg.approach_timeout_s,
                "");
        pg.open_timeout_s        = getf("DoorAffordance.OpenTimeoutS", pg.open_timeout_s,
                "provider latency + swing time, not a navigation budget");
        pg.cross_timeout_s       = getf("DoorAffordance.CrossTimeoutS", pg.cross_timeout_s,
                "");
        pg.offer_prob            = getf("DoorAffordance.OfferProb", pg.offer_prob,
                "Offer policy (rc::pragmatic::PragmaticAffordance::Policy): offer above `offer_prob`, withdraw below `withdraw_prob`, and only after the decision has…");
        pg.withdraw_prob         = getf("DoorAffordance.WithdrawProb", pg.withdraw_prob,
                "… and withdraw once it falls below THIS (must be < OfferProb; refused at load)");
        pg.stable_cycles         = geti("DoorAffordance.StableCycles", pg.stable_cycles,
                "… and only after the decision has held this many MEASURED cycles");
        pg.transitable_prob      = getf("DoorAffordance.TransitableProb", pg.transitable_prob,
                "assert once P(passable) rises above this …");
        pg.not_transitable_prob  = getf("DoorAffordance.NotTransitableProb", pg.not_transitable_prob,
                "… and retract once it falls below THIS (must be < the above)");
        pg.transitable_stable_cycles = geti("DoorAffordance.TransitableStableCycles", pg.transitable_stable_cycles,
                "… and only after the decision has held this many MEASURED cycles");
        pg.autonomous_actuation  = getb("DoorAffordance.AutonomousActuation", pg.autonomous_actuation,
                "★MAY THIS AGENT ASK A PROVIDER TO MOVE A DOOR WITHOUT A HUMAN? The `open` affordance's whole point is that it can, and the consumer's claim is the…");
        pg.actuation_provider_id = gets("DoorAffordance.ActuationProviderId", pg.actuation_provider_id,
                "FALLBACK provider door id, used only when no world frame resolves to match our door by place");
        pg.actuation_match_max_m = getf("DoorAffordance.ActuationMatchMaxM", pg.actuation_match_max_m,
                "max distance (m, provider world) from our door to the nearest advertised door to accept the match");

        // ★REFUSE AN INVERTED SCHMITT BAND AT LOAD. withdraw_prob >= offer_prob is not a band, it is a
        // single line with the two decisions on the wrong sides of it: the affordance would be offered and
        // withdrawn on the same probability and chatter at the compute rate, writing a node into the shared
        // graph every few cycles. Caught here because the symptom (CRDT churn in somebody else's agent) is
        // nowhere near the cause. See [[dsr-crdt-dot-cloud-unbounded]].
        // The `transitable` band gets the same refusal as the offer band above, and for the same reason:
        // an inverted band asserts and retracts on one probability, which would flap a PUBLIC edge in the
        // shared graph at the compute rate — CRDT churn in somebody else's agent, nowhere near the cause.
        if (pg.not_transitable_prob >= pg.transitable_prob)
        {
            std::print("[config] DoorAffordance.NotTransitableProb ({:.2f}) must be BELOW TransitableProb "
                       "({:.2f}) — that is not a hysteresis band. Falling back to 0.35/0.60.\n",
                       pg.not_transitable_prob, pg.transitable_prob);
            pg.not_transitable_prob = 0.35f;
            pg.transitable_prob     = 0.60f;
        }
        if (pg.withdraw_prob >= pg.offer_prob)
        {
            std::print("[config] DoorAffordance.WithdrawProb ({:.2f}) must be BELOW OfferProb ({:.2f}) — "
                       "that is not a hysteresis band. Falling back to 0.35/0.60.\n",
                       pg.withdraw_prob, pg.offer_prob);
            pg.withdraw_prob = 0.35f;
            pg.offer_prob    = 0.60f;
        }
        // A passage the robot cannot fit through can never be crossed, and a body width of 0 says the
        // robot is a point — both are configuration mistakes worth a line rather than a silent behaviour.
        if (pg.robot_passage_width_m <= 0.0f)
            std::print("[config] DoorAffordance.RobotPassageWidthM is {:.2f} m — the robot is not a point; "
                       "the `cross` affordance will be offered for ANY ajar door\n", pg.robot_passage_width_m);
    }

    return out;
}

}  // namespace rc
