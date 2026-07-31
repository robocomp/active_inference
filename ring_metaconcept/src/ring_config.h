/*
 * ring_config.h
 *
 * Plain-data configuration for the ring_metaconcept agent (the level-2 "concept of
 * concepts": a radial arrangement — the dining_set is its first instance). Mirrors the
 * per-agent config pattern (bottle/chair/table_config): a POD struct + a loader that
 * fills it from a RoboComp ConfigLoader, so a new meta-concept can copy this file and
 * edit only the keys it needs.
 *
 * Milestone 0 (graph-reader) needs the member CLASSES the reader polls, the owned-node
 * naming, and the diagnostics channel; the ring-belief / evidence / down-prior keys are
 * added in later milestones.
 */

#pragma once

#include <string>

class ConfigLoader;   // RoboComp config façade (defined in genericworker.h)

namespace rc {

struct RingConfig
{
    // ── Member CLASSES this radial arrangement groups (schema = ring) ─────────────
    // ★These are object CLASSES, not DSR node types. Since the object-type migration every
    // concept agent publishes its instances as the generic type "object" with the class in the
    // `object_subtype` string attr (name prefixes table_*/chair_* unchanged). The graph-reader
    // therefore does ONE get_nodes_by_type("object") and filters by class — polling
    // get_nodes_by_type("table") returns nothing and would silently starve the agent.
    std::string anchor_class = "table";      // central member (dining table)
    std::string ring_class   = "chair";      // ring members arranged around the anchor

    // ── The node this agent OWNS (births in M3) ──────────────────────────────────
    // DSR type `metaconcept` — NOT `object` like the members above. A rig is a belief about a
    // RELATION among nodes, so it must stay out of everyone's get_nodes_by_type("object") sweep
    // (else it is drawn as a solid, carved out of the residual grid, and associated against). The
    // class still lives in object_subtype and the name carries the prefix, which the sweep keys on.
    std::string node_subtype = "dining_set";     // object_subtype written on the owned node
    std::string node_prefix  = "dining_set_";    // owned-node NAME prefix (sweep/cleanup key)

    // ── Arrangement belief ────────────────────────────────────────────────────────
    // Slack between a seat and the chair on it (NOT sensor noise — that arrives per-member as the
    // peer's own published Σ). Chairs get pushed in and out, so this is deliberately loose.
    float sigma_slot_m = 0.20f;
    float clutter_frac = 0.20f;    // prior mass on "this member belongs to no rig"
    float occupancy_q  = 0.70f;    // P(a seat of a real rig is occupied)
    // Added to the inward-facing direction to express the prior in the MEMBER's yaw convention.
    // −90° for the current chair model (backrest on −y ⇒ the chair faces its local +y). ★Verified
    // live 2026-07-26; see RingBeliefParams::member_yaw_offset. Re-derive per member class.
    float member_yaw_offset_deg = -90.0f;
    // Intrinsic spread of "a chair faces the table" — a tendency, not a law. Added in quadrature to
    // the geometric propagation, and it is what BOUNDS how hard the rig can push a member's yaw
    // (κ = p_ring / facing_var). ★Live estimate from three real chairs: residuals −7°, +16°.
    float facing_model_std_deg = 12.0f;
    // Low-pass rate for the ring-vs-null estimate. NOT an accumulator — static furniture re-observed
    // is not new evidence; summing saturated the clamp on cycle 1 and pinned p_ring at ~1.
    float evidence_ema_alpha = 0.05f;

    // ── Slot visibility (gates the empty-seat evidence) ───────────────────────────
    // An EMPTY seat is only evidence against a slot count where the robot has actually LOOKED. These
    // are physical camera properties, not tuning knobs; the soft edge keeps observability continuous
    // (a seat at the very rim of the frame is partially observed, not a step from seen to unseen).
    bool  slot_visibility_enabled = true;
    float zed_hfov_deg      = 100.0f;   // ZED horizontal field of view
    float zed_max_range_m   = 6.0f;     // beyond this a seat is not reliably resolvable
    float zed_fov_soft_deg  = 12.0f;    // soft edge width on the FoV boundary
    float zed_range_soft_m  = 1.0f;     // soft edge width on the range boundary
    // Fallback support for the independent-objects null when the room polygon is unavailable. The
    // null density is 1/area, so this SCALES the ring-vs-null log-Bayes factor — prefer the polygon.
    float room_area_m2_fallback = 50.0f;

    // ── Diagnostics ───────────────────────────────────────────────────────────────
    int         log_period_frames = 30;   // per-cycle stdout log throttle for the graph-reader
    std::string csv_path;                 // per-member snapshot CSV; empty disables
    std::string fit_csv_path;             // arrangement fit + would-be down-prior CSV; empty disables
};

// Fill a RingConfig from a RoboComp ConfigLoader (all keys optional, defaults above).
RingConfig load_ring_config(const ConfigLoader& cfg);

}  // namespace rc
