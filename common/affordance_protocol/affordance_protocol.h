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

namespace rc::affordance
{

enum class Policy    { Reach, Servo };
enum class CompareOp { GE, LE, EQ, NE };
enum class OnFail    { Consume, Abandon };   // Consume = give up & mark done; Abandon = release for retry

struct GoalClause
{
    std::string attr;                 // attribute name on the feedback node
    CompareOp   op    = CompareOp::GE;
    float       value = 0.0f;
};

class ContractBuilder;   // fluent authoring builder (defined just below)

struct Contract
{
    // Fluent authoring: Contract::servo()/reach() return a ContractBuilder (see below). Sugar only —
    // they produce this same struct, so the wire format and executor are unaffected.
    static ContractBuilder reach();
    static ContractBuilder servo();

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
    static ContractBuilder reach() { ContractBuilder b; b.c_.policy = Policy::Reach; return b; }
    static ContractBuilder servo() { ContractBuilder b; b.c_.policy = Policy::Servo; return b; }

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
        if (c_.policy == Policy::Servo && c_.goal.empty())
            std::fprintf(stderr, "[affordance] WARNING: servo contract has no completion clause "
                                 "(.until/.and_) — it can never complete\n");
        return c_;
    }
    operator Contract() const { return build(); }   // pass a builder straight into write_contract(const Contract&)

private:
    Contract c_;
};

inline ContractBuilder Contract::reach() { return ContractBuilder::reach(); }
inline ContractBuilder Contract::servo() { return ContractBuilder::servo(); }

// ─── enum ⇄ string ────────────────────────────────────────────────────────────
inline std::string_view to_string(Policy p)    { return p == Policy::Servo ? "servo" : "reach"; }
inline Policy           policy_from(std::string_view s) { return s == "servo" ? Policy::Servo : Policy::Reach; }
inline std::string_view to_string(CompareOp o)
{
    switch (o) { case CompareOp::LE: return "le"; case CompareOp::EQ: return "eq"; case CompareOp::NE: return "ne"; default: return "ge"; }
}
inline CompareOp op_from(std::string_view s)
{
    if (s == "le") return CompareOp::LE;
    if (s == "eq") return CompareOp::EQ;
    if (s == "ne") return CompareOp::NE;
    return CompareOp::GE;
}
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
    if (object_type == "cylinder")   // bottle_concept's object node type
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
    G.runtime_checked_add_or_modify_attrib_local(node, "aff_policy",       std::string(to_string(c.policy)));
    G.runtime_checked_add_or_modify_attrib_local(node, "aff_err_vec_attr", c.err_vec_attr);
    G.runtime_checked_add_or_modify_attrib_local(node, "aff_scalar_attr",  c.scalar_attr);
    G.runtime_checked_add_or_modify_attrib_local(node, "aff_scalar_target", c.scalar_target);
    G.runtime_checked_add_or_modify_attrib_local(node, "aff_valid_attr",   c.valid_attr);

    std::vector<std::string> attrs, ops; std::vector<float> vals;
    attrs.reserve(c.goal.size()); ops.reserve(c.goal.size()); vals.reserve(c.goal.size());
    for (const auto& g : c.goal) { attrs.push_back(g.attr); ops.emplace_back(to_string(g.op)); vals.push_back(g.value); }
    G.runtime_checked_add_or_modify_attrib_local(node, "aff_goal_attrs",  detail::join(attrs, '|'));
    G.runtime_checked_add_or_modify_attrib_local(node, "aff_goal_ops",    detail::join(ops, '|'));
    G.runtime_checked_add_or_modify_attrib_local(node, "aff_goal_values", vals);
    G.runtime_checked_add_or_modify_attrib_local(node, "aff_goal_stable_n", c.stable_n);
    G.runtime_checked_add_or_modify_attrib_local(node, "aff_timeout_ms",  c.timeout_ms);
    G.runtime_checked_add_or_modify_attrib_local(node, "aff_on_fail",     std::string(to_string(c.on_fail)));
    G.runtime_checked_add_or_modify_attrib_local(node, "aff_max_vel",    c.max_observe_vel);
    G.runtime_checked_add_or_modify_attrib_local(node, "aff_max_omega",  c.max_observe_omega);
}

// ─── executor: type defaults + per-node overrides ─────────────────────────────
// parent_type = the type of the affordance's parent object node (e.g. "table", "cylinder"); the
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
            const auto names = detail::split(*names_s, '|');
            std::vector<std::string> ops;
            if (const auto io = attrs.find("aff_goal_ops"); io != attrs.end())
                if (auto ops_s = detail::attr_string(io->second)) ops = detail::split(*ops_s, '|');
            const auto& vals = itv->second.float_vec();
            std::vector<GoalClause> g;
            for (std::size_t i = 0; i < names.size() && i < vals.size(); ++i)
                g.push_back({names[i], i < ops.size() ? op_from(ops[i]) : CompareOp::GE, vals[i]});
            if (!g.empty()) c.goal = std::move(g);
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
