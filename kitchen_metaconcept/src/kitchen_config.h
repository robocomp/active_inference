/*
 * kitchen_config.h
 *
 * Plain-data configuration for the kitchen_metaconcept agent (level-2 "concept of concepts", schema
 * = RECTILINEAR: members share an axis grid, a worktop plane and a carcass depth). Mirrors the
 * per-agent config pattern (ring/bottle/chair/table_config): a POD struct + a loader that fills it
 * from a RoboComp ConfigLoader, so a sibling meta-concept can copy this file and edit only its keys.
 */

#pragma once

#include <string>
#include <vector>

class ConfigLoader;   // RoboComp config façade (defined in genericworker.h)

namespace rc {

struct KitchenConfig
{
    // ── Member CLASSES this frame groups ─────────────────────────────────────────
    // ★These are object CLASSES, not DSR node types. Since the object-type migration every concept
    // agent publishes its instances as the generic type "object" with the class in `object_subtype`
    // (name prefixes cabinet_*/… unchanged). The graph-reader therefore does ONE
    // get_nodes_by_type("object") and filters by class — polling get_nodes_by_type("cabinet")
    // returns nothing and would silently starve the agent.
    std::vector<std::string> member_classes{"cabinet"};
    // Parallel to member_classes: log[P(class | kitchen unit) / P(class | not)]. A likelihood ratio,
    // NOT a set test — an unexpected class is down-weighted, never excluded. Authored priors; see the
    // per-class justification and the producer-status table in etc/config.toml. Missing/short entries
    // default to 0 (uninformative), which is the safe direction.
    std::vector<float> member_class_logodds{0.0f};
    // Parallel to member_classes: which TIER this class is, as "base" | "tall" | "wall" | "auto".
    // ★This is the strongest use of the subtype channel — far stronger than the membership prior.
    // The class does not merely say "kitchen-ish"; it says WHICH shared quantities apply. A
    // refrigerator is a TALL unit: it shares the axis and a tall depth, and has no worktop at all.
    // Getting that wrong is what made the live fridge read membership 0.00. "auto" defers to
    // geometry (a plain "cabinet" can be any of the three).
    std::vector<std::string> member_class_tiers{"auto"};


    // ── The node this agent will OWN (M3+; nothing is created at M2) ─────────────
    // DSR type `metaconcept`, NOT `object`: a frame is a belief about a RELATION among nodes, so it
    // must stay out of everyone's get_nodes_by_type("object") sweep (else it is drawn as a solid,
    // carved out of the residual grid, and associated against).
    std::string node_subtype = "kitchen";
    std::string node_prefix  = "kitchen_";

    // ── Rectilinear-frame belief ─────────────────────────────────────────────────
    // Intrinsic model spread per shared DOF: how much genuine variation the hypothesis "one kitchen"
    // allows. Added in quadrature to the geometric propagation, and it is what BOUNDS the down-prior
    // precision. NOT confidence knobs.
    float axis_model_std_deg  = 1.5f;
    float worktop_model_std_m = 0.02f;
    float depth_model_std_m   = 0.03f;

    // ── ESTIMATION noise of a member's own size, at a 1 m reference length ───────────────────────
    // A DIFFERENT quantity from the model spread above: that says how much real kitchens vary, this
    // says how well cabinet_concept currently KNOWS each value. Live, two runs of the same kitchen
    // reported worktops of 0.758 and 0.896 m — a 14 cm disagreement, which the 2 cm model spread
    // calls a 7σ contradiction and which floors every member on the clutter column.
    // ⚠STAND-IN: cabinet_concept publishes a POSE covariance only. It HAS the real numbers (the
    // belief's S(4,4) for z1, S(2,2) for depth) — publishing them needs new cortex attributes, and
    // then these two keys should be deleted rather than tuned.
    float worktop_meas_std_m = 0.06f;   // at 1 m of run; scaled by 1/length below
    float depth_meas_std_m   = 0.05f;

    float clutter_frac       = 0.20f;   // prior mass on "this member belongs to no kitchen frame"
    float evidence_ema_alpha = 0.05f;   // low-pass on frame-vs-independent evidence (NOT a sum)

    // ── Axis evidence source (the structural cavity) ─────────────────────────────
    // Estimate the shared axis ONLY from members whose yaw is pinned by the room polygon. Those can
    // never adopt a prior this agent pushes, so the leave-one-out down-date is structural rather
    // than bolted on — see SCHEMA_GENERALITY_TODO.md §2.5.
    bool  axis_from_pinned_only  = true;
    float pinned_yaw_std_max_deg = 3.5f;

    // ── Diagnostics ──────────────────────────────────────────────────────────────
    int         log_period_frames = 25;
    std::string csv_path;       // per-member snapshot; empty disables
    std::string fit_csv_path;   // frame fit + the down-prior it WOULD publish; empty disables
};

// Fill a KitchenConfig from a RoboComp ConfigLoader (all keys optional, defaults above).
KitchenConfig load_kitchen_config(const ConfigLoader& cfg);

}  // namespace rc
