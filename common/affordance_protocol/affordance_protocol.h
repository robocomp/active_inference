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

struct Contract
{
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
};

// ─── enum ⇄ string ────────────────────────────────────────────────────────────
inline std::string_view to_string(Policy p)    { return p == Policy::Servo ? "servo" : "reach"; }
inline Policy           policy_from(std::string_view s) { return s == "servo" ? Policy::Servo : Policy::Reach; }
inline std::string_view to_string(CompareOp o)
{
    switch (o) { case CompareOp::LE: return "le"; case CompareOp::EQ: return "eq"; case CompareOp::NE: return "ne"; default: return "ge"; }
}
inline CompareOp op_from(std::string_view s)
{
    if (s == "le") return CompareOp::LE; if (s == "eq") return CompareOp::EQ; if (s == "ne") return CompareOp::NE; return CompareOp::GE;
}
inline std::string_view to_string(OnFail f)     { return f == OnFail::Abandon ? "abandon" : "consume"; }
inline OnFail           onfail_from(std::string_view s) { return s == "abandon" ? OnFail::Abandon : OnFail::Consume; }

// ─── type-level defaults ──────────────────────────────────────────────────────
// Baseline contract keyed by the affordance node name; per-node aff_* attributes override these.
inline Contract default_contract_for(std::string_view affordance_name)
{
    Contract c;   // default policy = Reach (navigate to the pose, done on arrival)
    if (affordance_name.rfind("aff_table", 0) == 0)   // table_concept names its node "aff_table_*"
    {
        // Table: RGB-mask lock-on. Centre the projected ROI (yaw) + reach the stand-off fill
        // (advance); done when YOLO is firing confidently for the table.
        c.policy        = Policy::Servo;
        c.err_vec_attr  = "table_roi_offset";
        c.scalar_attr   = "table_roi_fill";
        c.scalar_target = 0.45f;
        c.valid_attr    = "table_roi_valid";
        // "Found a YOLO position" = a fresh table mask exists (detection_alive). The raw confidence
        // tops out low (~0.3) on this table, so a high-confidence clause is unsatisfiable; lock on a
        // recent detection during a settled measurement instead. A low confidence floor rejects junk.
        c.goal          = { {"table_detection_alive",      CompareOp::GE, 0.5f},
                            {"table_detection_confidence", CompareOp::GE, 0.20f} };
        c.stable_n      = 1;
        c.timeout_ms    = 25000.0f;   // a full distance sweep takes time
        c.on_fail       = OnFail::Consume;
    }
    return c;
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
}

// ─── executor: type defaults + per-node overrides ─────────────────────────────
inline Contract read_contract(const DSR::Node& node)
{
    Contract c = default_contract_for(node.name());
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
