#pragma once

/*
 * affordance_protocol.h — shared affordance EXECUTION contract.
 *
 * The controller is a generic executor; each producing agent (table_concept, bottle_concept, …)
 * owns the *semantics* of its affordance: when is it done, and what local behaviour completes it.
 * This header is the data contract by which a producer declares that, per affordance node, and the
 * controller executes it without any object-specific code.
 *
 * A Contract has four parts:
 *   1. policy        — which reusable control primitive the executor runs after reaching the pose
 *                      (Reach = navigate only; Servo = feedback micro-search / "lock-on").
 *   2. feedback bind — NAMES of attributes the producer publishes on the feedback node (its parent
 *                      object), so the executor reads them generically: err_vec_attr (normalised
 *                      error to null → base yaw/side), scalar_attr→scalar_target (→ advance),
 *                      valid_attr (gate).
 *   3. completion c  — AND-ed clauses over the feedback node's attributes, held `stable_n` cycles,
 *                      bounded by `timeout_ms`; `on_fail` says what to do on timeout.
 *   4. (selection/claim stay in the existing epistemic_* / active / pending protocol; this is the
 *      additive behaviour+completion layer.)
 *
 * Resolution is "type-level defaults + per-node overrides": read_contract() starts from
 * default_contract_for(<affordance node name>) and overrides with any aff_* attributes present on
 * the node. So a producer can rely on defaults (write nothing) or tailor a single affordance.
 *
 * Wire format: compact runtime attributes ("aff_*") on the affordance node, mirroring the
 * '|'-joined-string + parallel-float-vec style used elsewhere (masks). Header-only; depends only on
 * libdsr, so every producer and the controller share one definition with no extra build wiring.
 */

#include <dsr/api/dsr_api.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "affordance_goal_parse.h"   // GoalClause, CompareOp, parse_goal_clauses, validate_contract

namespace rc::affordance
{

// Reach  = navigate to the pose, then consume. Servo = navigate to the pose, then run the lock-on
// micro-search to satisfy the completion predicate. Orient = do NOT navigate; rotate in place toward the
// affordance's target yaw, then satisfy the predicate (a peripheral "glance" / saccade — no (x,y) target).
enum class Policy    { Reach, Servo, Orient };
enum class OnFail    { Consume, Abandon };   // Consume = give up & mark done; Abandon = release for retry
// CompareOp and GoalClause live in affordance_goal_parse.h — they are pure, and they are the part
// that had to become testable (see that file's header for the two silent mis-parses it exists to stop).

class ContractBuilder;   // fluent authoring builder (defined just below)

struct Contract
{
    // Fluent authoring: Contract::servo()/reach() return a ContractBuilder (see below). Sugar only —
    // they produce this same struct, so the wire format and executor are unaffected.
    static ContractBuilder reach();
    static ContractBuilder servo();
    static ContractBuilder orient();

    Policy policy = Policy::Reach;

    // Servo feedback binding (attribute names on the feedback node = the affordance's parent object).
    std::string   err_vec_attr;             // vector<float> normalised error to null → base yaw/(side)
    std::string   scalar_attr;              // scalar driven to scalar_target           → advance
    float         scalar_target = 0.0f;
    std::string   valid_attr;               // gate (nonzero = valid); empty = always valid
    std::uint64_t feedback_node_id = 0;     // filled by the executor (defaults to the affordance parent)

    // Completion predicate c: AND-ed clauses on the feedback node, held stable_n cycles, with timeout.
    std::vector<GoalClause> goal;
    int    stable_n   = 3;
    float  timeout_ms = 10000.0f;
    OnFail on_fail    = OnFail::Consume;

    // Observation-stillness precondition: the producer asks the executor to hold the base below these
    // speeds for the look to count (a moving/rotating capture blurs the RGB and smears the deprojected
    // mask points by ≈ω·lag·range, biasing the fit). 0 = not required. The angular term is the one that
    // matters most. ANDed into completion by the executor, which knows the base velocity it commands.
    float max_observe_vel   = 0.0f;   // m/s
    float max_observe_omega = 0.0f;   // rad/s
};

// ─── object-relative viewpoint constraint (the epistemic "where to look") ───────
// A producer's NBV planner declares WHAT view it needs to shrink its own uncertainty, expressed in the
// parent OBJECT's frame — NOT as a world pose, because the producer cannot see global occupancy or
// reachability (the old model published an absolute (x,y,yaw) with a p_observable=1 stub). It offers a
// RANKED set of candidate faces, each with its expected posterior-entropy reduction (nats), plus the
// sensor-model stand-off band and framing target it needs, and its precision demand Σ* (per belief DOF,
// the adequacy-gap target). The controller looks up the object footprint, generates viewpoints on the
// requested faces across the stand-off band, tests them against the scene polygons + reachability, and
// picks the best FEASIBLE face (argmax gain over reachable faces) — so a blocked argmax face falls back
// to the next-best reachable one. object_relative=false ⇒ absent/legacy ⇒ the controller uses the
// epistemic_target_* hint pose instead. Carried on the affordance node as aff_view_* attributes.
struct ViewpointConstraint
{
    bool  object_relative = false;          // true ⇒ the fields below are authoritative (object frame)
    std::vector<std::string> faces;         // candidate faces in object frame, e.g. "+x","-x","+y","-y"
    std::vector<float>       face_gains;     // parallel to faces: expected ΔH (nats) from that face's view
    float standoff_min_m = 0.0f;            // sensor-model stand-off band (framing sweet spot ← FoV geometry)
    float standoff_max_m = 0.0f;
    float framing_fill   = 0.0f;            // desired projected fill fraction (drive fill→this after arrival)
    std::vector<float> sigma_star;          // per-DOF precision demand [·]; empty = unset (adequacy-gap target)
};

// ─── fluent authoring builder ─────────────────────────────────────────────────
// Ergonomic alternative to filling Contract field-by-field. Produces the SAME Contract, so
// write_contract/read_contract and the executor are untouched — authoring sugar only.
//
//   write_contract(G, node, Contract::servo()
//       .center ("table_roi_offset")             // err_vec  → base yaw/side centring
//       .advance("table_roi_fill", 0.45f)        // scalar   → advance to stand-off
//       .valid  ("table_roi_valid")              // gate
//       .until  ("table_detection_alive",      CompareOp::GE, 0.5f)
//       .and_   ("table_detection_confidence", CompareOp::GE, 0.20f)
//       .stable(1).timeout_s(25).on_fail(OnFail::Consume));
class ContractBuilder
{
public:
    static ContractBuilder reach()  { ContractBuilder b; b.c_.policy = Policy::Reach;  return b; }
    static ContractBuilder servo()  { ContractBuilder b; b.c_.policy = Policy::Servo;  return b; }
    static ContractBuilder orient() { ContractBuilder b; b.c_.policy = Policy::Orient; return b; }

    // Servo feedback bindings (ignored under Reach).
    ContractBuilder& center (std::string a)            { c_.err_vec_attr = std::move(a); return *this; }
    ContractBuilder& advance(std::string a, float tgt) { c_.scalar_attr  = std::move(a); c_.scalar_target = tgt; return *this; }
    ContractBuilder& valid  (std::string a)            { c_.valid_attr   = std::move(a); return *this; }
    ContractBuilder& feedback_node(std::uint64_t id)   { c_.feedback_node_id = id; return *this; }

    // Completion predicate — clauses are AND-ed; `until` then `and_` reads naturally.
    ContractBuilder& until(std::string a, CompareOp op, float v) { c_.goal.push_back({std::move(a), op, v}); return *this; }
    ContractBuilder& and_ (std::string a, CompareOp op, float v) { c_.goal.push_back({std::move(a), op, v}); return *this; }
    ContractBuilder& stable(int n)        { c_.stable_n   = n;           return *this; }
    ContractBuilder& timeout_ms(float ms) { c_.timeout_ms = ms;          return *this; }
    ContractBuilder& timeout_s (float s)  { c_.timeout_ms = s * 1000.0f; return *this; }
    ContractBuilder& on_fail(OnFail f)    { c_.on_fail    = f;           return *this; }
    // Observation-stillness precondition: hold base speed below (v m/s, ω rad/s) for the look to count.
    ContractBuilder& still(float v, float omega) { c_.max_observe_vel = v; c_.max_observe_omega = omega; return *this; }

    // Validate-and-return. Warns (never aborts — this runs in a crash-sensitive graph path) on the
    // one structural mistake that makes a Servo affordance silently never complete: an empty
    // completion predicate. (A servo with no .center/.advance binding is NOT an error: the executor's
    // distance-sweep is its primary motion and needs no binding — the bindings only add centring.)
    // Fires once at authoring (write_contract) and once when the controller reads the contract on
    // reach — not per cycle.
    Contract build() const
    {
        const ContractShape shape{
            .is_reach          = (c_.policy == Policy::Reach),
            .has_servo_binding = (not c_.err_vec_attr.empty() or not c_.scalar_attr.empty()),
            .stable_n          = c_.stable_n,
            .timeout_ms        = c_.timeout_ms,
            .max_observe_omega = c_.max_observe_omega,
            .is_orient         = (c_.policy == Policy::Orient)};
        for (const auto& why : validate_contract(shape, c_.goal))
            std::fprintf(stderr, "[affordance] WARNING: %s\n", why.c_str());
        return c_;
    }
    operator Contract() const { return build(); }   // pass a builder straight into write_contract(const Contract&)

private:
    Contract c_;
};

inline ContractBuilder Contract::reach()  { return ContractBuilder::reach(); }
inline ContractBuilder Contract::servo()  { return ContractBuilder::servo(); }
inline ContractBuilder Contract::orient() { return ContractBuilder::orient(); }

// ─── enum ⇄ string ────────────────────────────────────────────────────────────
inline std::string_view to_string(Policy p)
{
    switch (p) { case Policy::Servo: return "servo"; case Policy::Orient: return "orient"; default: return "reach"; }
}
inline Policy policy_from(std::string_view s)
{
    if (s == "servo")  return Policy::Servo;
    if (s == "orient") return Policy::Orient;
    return Policy::Reach;
}
// to_string(CompareOp) / op_from_strict: affordance_goal_parse.h.
inline std::string_view to_string(OnFail f)     { return f == OnFail::Abandon ? "abandon" : "consume"; }
inline OnFail           onfail_from(std::string_view s) { return s == "abandon" ? OnFail::Abandon : OnFail::Consume; }

// ─── type-level defaults ──────────────────────────────────────────────────────
// Baseline contract keyed by the PARENT OBJECT's node type (robust: the affordance node name can be
// renamed by DSR on a restart collision, but the parent type is stable). Per-node aff_* overrides win.
inline Contract default_contract_for(std::string_view object_type)
{
    using enum CompareOp;
    using enum OnFail;
    if (object_type == "table")
    {
        // Table: RGB-mask lock-on. Centre the projected ROI (yaw) + reach the stand-off fill
        // (advance); done when YOLO is firing on the table. "Found a YOLO position" = a fresh table
        // mask exists (detection_alive); raw confidence tops out ~0.3 here so a high-confidence clause
        // would be unsatisfiable — use a low floor that just rejects junk. Sweep takes time → 25 s.
        return Contract::servo()
            .center ("table_roi_offset")
            .advance("table_roi_fill", 0.45f)
            .valid  ("table_roi_valid")
            .until  ("table_detection_alive",      GE, 0.5f)
            .and_   ("table_detection_confidence", GE, 0.20f)
            .still  (0.10f, 0.15f)   // a table fills the frame → base rotation smears it most; keep ω tight
            .stable(1).timeout_s(25).on_fail(Consume);
    }
    // ★"bottle", the SUBTYPE — not "cylinder", which was bottle's node TYPE before the fleet moved to
    // type "object" + object_subtype. Every other agent already keys on its subtype ("table", "chair",
    // "door", …); bottle alone still passed the retired type name. The key is matched by STRING, so the
    // mismatch was invisible: this case simply stopped being reachable the day bottle's node became an
    // "object", and default_contract_for fell through to the valid-LOOKING generic reach() the header
    // warns about — the robot arriving with no detection goal to satisfy.
    if (object_type == "bottle")
    {
        // Bottle: verify YOLO is firing on this bottle before completing. No ROI binding (bottle has
        // no projected-ROI feedback) → the controller's distance-sweep finds a stand-off where the
        // detection holds; the goal then confirms it. Parent (bottle node) publishes the attrs.
        return Contract::servo()
            .until("bottle_detection_alive",      GE, 0.5f)
            .and_ ("bottle_detection_confidence", GE, 0.20f)
            .still(0.12f, 0.20f)   // a quiet rotational dwell to resolve the radius; a touch more v allowed
            .stable(1).timeout_s(20).on_fail(Consume);
    }
    if (object_type == "person")   // human_concept's object node type
    {
        // Person: reduce-occlusion look. No projected-ROI binding (the controller's distance-sweep
        // finds a stand-off where the skeleton track holds); the goal confirms the body is detected
        // with a clean, motion-free capture before completing. The .still dwell is essential here — a
        // moving/rotating base blurs the RGB and smears the deprojected joints, biasing the pose fit.
        return Contract::servo()
            .until("human_detection_alive",      GE, 0.5f)
            .and_ ("human_detection_confidence", GE, 0.20f)
            .still(0.10f, 0.15f)   // hold still for a clean, blur-free skeleton observation
            .stable(2).timeout_s(20).on_fail(Consume);
    }
    if (object_type == "chair")
    {
        // Chair: verify YOLO is firing on this chair before completing. No ROI binding (the chair
        // publishes no projected-ROI yet) → the controller's distance-sweep finds a stand-off where
        // the detection holds; the goal then confirms it. Parent (chair node) publishes the attrs.
        return Contract::servo()
            .until("chair_detection_alive",      GE, 0.5f)
            .and_ ("chair_detection_confidence", GE, 0.20f)
            .still(0.12f, 0.20f)
            .stable(1).timeout_s(20).on_fail(Consume);
    }
    if (object_type == "refrigerator")   // refrigerator_concept's node = generic "object", class key "refrigerator"
    {
        // Refrigerator: same RGB-mask lock-on as the table (its scene-graph writes the same table_roi_*/
        // table_detection_* active-perception channel). Centre the projected ROI + reach the stand-off fill;
        // done when YOLO is firing on the fridge. A tall box fills the frame → keep base rotation (ω) tight
        // for a clean, motion-free look (the .still dwell).
        return Contract::servo()
            .center ("table_roi_offset")
            .advance("table_roi_fill", 0.45f)
            .valid  ("table_roi_valid")
            .until  ("table_detection_alive",      GE, 0.5f)
            .and_   ("table_detection_confidence", GE, 0.20f)
            .still  (0.10f, 0.15f)
            .stable(1).timeout_s(25).on_fail(Consume);
    }
    if (object_type == "hood")
    {
        // Hood (ADE20K "hood" — the extractor over a hob). Wall-mounted and HIGH: ~1.5-2 m, well above the
        // zed's optical centre, so unlike every sibling the VERTICAL framing is what binds and the robot
        // must stand back far enough to fit it rather than close enough to fill the frame. Bound to the
        // channel hood_scene_graph publishes (hood_roi_* / hood_detection_*), not the table_* channel the
        // refrigerator borrows — a hood is not a fridge and must not lock onto a fridge's ROI.
        //
        // ★advance() fill is 0.30, not the fridge's 0.45: at 0.45 a 2 m-tall hood would have to be
        // approached to ~1 m, where it leaves the top of the frame entirely and the mask truncates. Written
        // as a starting point tied to the geometry, NOT a measurement — recalibrate from the agent's own
        // ai2_log once it has one (common/detectability/tools/fit_envelope).
        return Contract::servo()
            .center ("hood_roi_offset")
            .advance("hood_roi_fill", 0.30f)
            .valid  ("hood_roi_valid")
            .until  ("hood_detection_alive",      GE, 0.5f)
            .and_   ("hood_detection_confidence", GE, 0.20f)
            .still  (0.10f, 0.15f)
            .stable(1).timeout_s(25).on_fail(Consume);
    }
    if (object_type == "cabinet")
    {
        // Cabinet: wall-anchored like the refrigerator, and like the door it was asking for a contract that
        // did not exist — cabinet_affordance.cpp calls default_contract_for("cabinet"), which fell through to
        // the Contract::reach() below ("navigate to the pose, done on arrival"). Since
        // ControllerSession::wants_final_facing honours the affordance's target yaw only for Servo/Orient,
        // the robot arrived at the right stand-off pointing the wrong way and no mask could form.
        // Bound to the channel cabinet_scene_graph.cpp:172-177 actually publishes — cabinet_roi_* /
        // cabinet_detection_*, NOT the table_* names the refrigerator reuses. A contract bound to attributes
        // that do not exist never satisfies its predicate and times out on every attempt.
        // No .advance(): the NBV already put the stand-off at the detector envelope's argmax.
        return Contract::servo()
            .center ("cabinet_roi_offset")
            .valid  ("cabinet_roi_valid")
            .until  ("cabinet_detection_alive",      GE, 0.5f)
            .and_   ("cabinet_detection_confidence", GE, 0.20f)
            .still  (0.10f, 0.15f)   // a tall run fills the frame → keep ω tight for a clean look
            .stable(1).timeout_s(25).on_fail(Consume);
    }
    if (object_type == "door")
    {
        // Door: a leaf is a PLANE, so arriving at the stand-off is only half the manoeuvre — the robot must
        // also end up FACING it, or the ZED points along the wall and never sees the panel it was sent to
        // look at. That facing is not a preference: `epistemic_target_yaw_rad` on the affordance node is the
        // face normal the NBV chose, and the controller honours it at arrival only for Servo/Orient
        // (ControllerSession::wants_final_facing). This case previously fell through to the Contract::reach()
        // below — "navigate to the pose, done on arrival" — so a fitted door was driven to the right spot
        // pointing the wrong way, while a bearing HYPOTHESIS (which takes the Orient branch in
        // DoorAffordance::write_policy_contract) turned correctly. Same object, opposite behaviour, purely
        // because one path had a contract and the other did not.
        //
        // Feedback channel is the one door_scene_graph.cpp already publishes (door_roi_offset / door_roi_valid
        // / door_detection_alive / door_detection_confidence), so this binds to existing attributes.
        //
        // Deliberately NO .advance(): the NBV already places the stand-off at the argmax of the detector
        // envelope (common/nbv), so a second hard-coded fill target here would fight it — that is exactly the
        // retired framing_fill = 0.45 constant. Range is the planner's job; this contract owns orientation
        // and the completion predicate.
        return Contract::servo()
            .center("door_roi_offset")
            .valid ("door_roi_valid")
            .until ("door_detection_alive",      GE, 0.5f)
            .and_  ("door_detection_confidence", GE, 0.20f)
            .still (0.10f, 0.15f)   // a grazing view smears the leaf most: keep ω tight for a clean look
            .stable(2).timeout_s(20).on_fail(Consume);
    }
    return Contract::reach();   // default: navigate to the pose, done on arrival
}

// ─── internal helpers ─────────────────────────────────────────────────────────
namespace detail
{
inline std::vector<std::string> split(const std::string& s, char d)
{
    std::vector<std::string> out; std::stringstream ss(s); std::string t;
    while (std::getline(ss, t, d)) out.push_back(t);
    return out;
}
inline std::string join(const std::vector<std::string>& v, char d)
{
    std::string out; for (std::size_t i = 0; i < v.size(); ++i) { if (i) out += d; out += v[i]; } return out;
}
// Read an int/float/bool attribute as a float (Value variant indices: 1=int32, 2=float, 4=bool).
inline std::optional<float> attr_scalar(const DSR::Attribute& a)
{
    switch (a.selected())
    {
        case 1:  return static_cast<float>(a.dec());
        case 2:  return a.fl();
        case 4:  return a.bl() ? 1.0f : 0.0f;
        default: return std::nullopt;
    }
}
inline std::optional<std::string> attr_string(const DSR::Attribute& a)
{
    if (a.selected() == 0) return a.str();
    return std::nullopt;
}
}  // namespace detail

// ─── producer: write per-node overrides onto the affordance node ──────────────
// (Caller does G.update_node(node) afterwards, as usual.)
inline void write_contract(DSR::DSRGraph& G, DSR::Node& node, const Contract& c)
{
    G.add_or_modify_attrib_local<aff_policy_att>      (node, std::string(to_string(c.policy)));
    G.add_or_modify_attrib_local<aff_err_vec_attr_att>(node, c.err_vec_attr);
    G.add_or_modify_attrib_local<aff_scalar_attr_att> (node, c.scalar_attr);
    G.add_or_modify_attrib_local<aff_scalar_target_att>(node, c.scalar_target);
    G.add_or_modify_attrib_local<aff_valid_attr_att>  (node, c.valid_attr);

    std::vector<std::string> attrs, ops; std::vector<float> vals;
    attrs.reserve(c.goal.size()); ops.reserve(c.goal.size()); vals.reserve(c.goal.size());
    for (const auto& g : c.goal) { attrs.push_back(g.attr); ops.emplace_back(to_string(g.op)); vals.push_back(g.value); }
    G.add_or_modify_attrib_local<aff_goal_attrs_att>  (node, detail::join(attrs, '|'));
    G.add_or_modify_attrib_local<aff_goal_ops_att>    (node, detail::join(ops, '|'));
    G.add_or_modify_attrib_local<aff_goal_values_att> (node, vals);
    // Refuse-at-source: emitting a clause set our own reader would reject means the consumer silently
    // runs the type default while the producer believes its contract is live. Checked here, once, at
    // authoring — the same check the reader applies, so the two cannot drift.
    if (const auto rt = parse_goal_clauses(attrs, ops, vals); not rt.ok())
    {
        std::fprintf(stderr, "[affordance] WARNING: this contract will NOT round-trip; the consumer "
                             "will fall back to the type default:\n");
        for (const auto& why : rt.problems) std::fprintf(stderr, "[affordance]   - %s\n", why.c_str());
    }
    G.add_or_modify_attrib_local<aff_goal_stable_n_att>(node, c.stable_n);
    G.add_or_modify_attrib_local<aff_timeout_ms_att>  (node, c.timeout_ms);
    G.add_or_modify_attrib_local<aff_on_fail_att>     (node, std::string(to_string(c.on_fail)));
    G.add_or_modify_attrib_local<aff_max_vel_att>     (node, c.max_observe_vel);
    G.add_or_modify_attrib_local<aff_max_omega_att>   (node, c.max_observe_omega);
}

// ─── executor: type defaults + per-node overrides ─────────────────────────────
// parent_type = the affordance's parent object SUBTYPE (e.g. "table", "bottle"); the
// executor resolves it from the graph and passes it so the default base survives node-name renames.
inline Contract read_contract(const DSR::Node& node, std::string_view parent_type = {})
{
    Contract c = default_contract_for(parent_type);
    const auto& attrs = node.attrs();

    const auto gets = [&](const char* k, std::string& out)
    { if (const auto it = attrs.find(k); it != attrs.end()) if (auto s = detail::attr_string(it->second)) out = *s; };
    const auto getf = [&](const char* k, float& out)
    { if (const auto it = attrs.find(k); it != attrs.end()) if (auto v = detail::attr_scalar(it->second)) out = *v; };
    const auto geti = [&](const char* k, int& out)
    { if (const auto it = attrs.find(k); it != attrs.end()) if (auto v = detail::attr_scalar(it->second)) out = static_cast<int>(*v); };

    if (const auto it = attrs.find("aff_policy"); it != attrs.end())
        if (auto s = detail::attr_string(it->second)) c.policy = policy_from(*s);
    gets("aff_err_vec_attr", c.err_vec_attr);
    gets("aff_scalar_attr",  c.scalar_attr);
    getf("aff_scalar_target", c.scalar_target);
    gets("aff_valid_attr",   c.valid_attr);

    // Goal clauses override the defaults only if explicitly present.
    if (const auto ita = attrs.find("aff_goal_attrs"), itv = attrs.find("aff_goal_values");
        ita != attrs.end() && itv != attrs.end())
    {
        if (auto names_s = detail::attr_string(ita->second))
        {
            const auto names = detail::split(*names_s, kClauseDelim);
            std::vector<std::string> ops;
            if (const auto io = attrs.find("aff_goal_ops"); io != attrs.end())
                if (auto ops_s = detail::attr_string(io->second)) ops = detail::split(*ops_s, kClauseDelim);
            const auto& vals = itv->second.float_vec();

            // ★ ALL OR NOTHING. This used to zip to the SHORTER array and default a missing operator
            // to GE, both silently. Dropping a clause weakens the predicate; defaulting an absent
            // operator INVERTS an 'le' clause into 'ge' — either way the affordance completes without
            // the thing having happened, and it looks exactly like success. A set that does not
            // round-trip exactly is refused whole and the conservative type default stands.
            const auto parsed = parse_goal_clauses(names, ops, vals);
            if (not parsed.ok())
            {
                std::fprintf(stderr, "[affordance] REFUSED the goal clauses on this node; keeping the "
                                     "'%.*s' type default (%zu clause(s)). Reasons:\n",
                             static_cast<int>(parent_type.size()), parent_type.data(), c.goal.size());
                for (const auto& why : parsed.problems)
                    std::fprintf(stderr, "[affordance]   - %s\n", why.c_str());
            }
            else if (not parsed.clauses.empty())
                c.goal = parsed.clauses;
        }
    }
    geti("aff_goal_stable_n", c.stable_n);
    getf("aff_timeout_ms", c.timeout_ms);
    getf("aff_max_vel",   c.max_observe_vel);
    getf("aff_max_omega", c.max_observe_omega);
    if (const auto it = attrs.find("aff_on_fail"); it != attrs.end())
        if (auto s = detail::attr_string(it->second)) c.on_fail = onfail_from(*s);
    return c;
}

// ─── viewpoint constraint ⇄ affordance node (aff_view_* attributes) ────────────
// Producer stamps the object-relative viewpoint constraint; the controller reads it and resolves a
// collision-free reachable pose. object_relative is wired as an int flag (0/1) so it rides the same
// scalar-attr path as the rest. (Caller does G.update_node(node) afterwards, as usual.)
inline void write_viewpoint(DSR::DSRGraph& G, DSR::Node& node, const ViewpointConstraint& v)
{
    G.add_or_modify_attrib_local<aff_view_object_relative_att>(node, v.object_relative ? 1 : 0);
    G.add_or_modify_attrib_local<aff_view_faces_att>       (node, detail::join(v.faces, '|'));
    G.add_or_modify_attrib_local<aff_view_face_gains_att>  (node, v.face_gains);
    G.add_or_modify_attrib_local<aff_view_standoff_min_att>(node, v.standoff_min_m);
    G.add_or_modify_attrib_local<aff_view_standoff_max_att>(node, v.standoff_max_m);
    G.add_or_modify_attrib_local<aff_view_framing_fill_att>(node, v.framing_fill);
    G.add_or_modify_attrib_local<aff_view_sigma_star_att>  (node, v.sigma_star);
}

inline ViewpointConstraint read_viewpoint(const DSR::Node& node)
{
    ViewpointConstraint v;
    const auto& attrs = node.attrs();
    if (const auto it = attrs.find("aff_view_object_relative"); it != attrs.end())
        if (auto s = detail::attr_scalar(it->second)) v.object_relative = (*s != 0.0f);
    if (const auto it = attrs.find("aff_view_faces"); it != attrs.end())
        if (auto s = detail::attr_string(it->second)) v.faces = detail::split(*s, '|');
    if (const auto it = attrs.find("aff_view_face_gains"); it != attrs.end())
        v.face_gains = it->second.float_vec();
    // ★ faces and face_gains are PARALLEL, and the consumer zips them to pick the best face. A length
    // mismatch re-pairs every face with the wrong gain, so the argmax names a face nobody scored — and
    // the robot drives confidently to it. An empty gains list is the legal "unscored" encoding; any
    // other mismatch drops the constraint back to the legacy hint pose rather than acting on a
    // corrupted ranking.
    if (not parallel_ok(v.faces.size(), v.face_gains.size()))
    {
        std::fprintf(stderr, "[affordance] viewpoint REFUSED: %zu face(s) but %zu gain(s) — the ranking "
                             "cannot be zipped; falling back to the hint pose\n",
                     v.faces.size(), v.face_gains.size());
        v = ViewpointConstraint{};
    }
    if (const auto it = attrs.find("aff_view_standoff_min"); it != attrs.end())
        if (auto s = detail::attr_scalar(it->second)) v.standoff_min_m = *s;
    if (const auto it = attrs.find("aff_view_standoff_max"); it != attrs.end())
        if (auto s = detail::attr_scalar(it->second)) v.standoff_max_m = *s;
    if (const auto it = attrs.find("aff_view_framing_fill"); it != attrs.end())
        if (auto s = detail::attr_scalar(it->second)) v.framing_fill = *s;
    if (const auto it = attrs.find("aff_view_sigma_star"); it != attrs.end())
        v.sigma_star = it->second.float_vec();
    return v;
}

// ─── executor: stateless predicate evaluation ─────────────────────────────────
inline bool clause_ok(float v, CompareOp op, float thr)
{
    switch (op)
    {
        case CompareOp::GE: return v >= thr;
        case CompareOp::LE: return v <= thr;
        case CompareOp::EQ: return std::abs(v - thr) < 1e-3f;
        case CompareOp::NE: return std::abs(v - thr) >= 1e-3f;
    }
    return false;
}
// Observation-stillness precondition: is the base quiet enough for a clean look? The caller supplies
// the MEASURED base speeds (m/s, rad/s); 0 thresholds → not required. ANDed into completion by the
// executor so the controller dwells for the observation instead of grabbing masks in motion.
inline bool stillness_ok(float linear_speed, float angular_speed, const Contract& c)
{
    const bool v_ok = c.max_observe_vel   <= 0.0f or linear_speed  <= c.max_observe_vel;
    const bool w_ok = c.max_observe_omega <= 0.0f or angular_speed <= c.max_observe_omega;
    return v_ok and w_ok;
}

// Are all clauses satisfied right now? (Caller counts stability across cycles.)
inline bool evaluate_goal(const DSR::Node& feedback_node, const std::vector<GoalClause>& goal)
{
    if (goal.empty()) return true;
    const auto& attrs = feedback_node.attrs();
    for (const auto& g : goal)
    {
        const auto it = attrs.find(g.attr);
        if (it == attrs.end()) return false;
        const auto v = detail::attr_scalar(it->second);
        if (!v || !clause_ok(*v, g.op, g.value)) return false;
    }
    return true;
}

}  // namespace rc::affordance
