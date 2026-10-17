#pragma once
/*
 * mission_ir.h — the GROUNDED mission: the language chat_agent and the controller speak (2026-10-17).
 * Design: chat_agent/docs/mission_language_2026-10-13.md, §7 (extensibility, limits, syntax, v1 reservations).
 *
 * The IR is the step tree AFTER resolution: every reference is a node name, a passage id, an anchor or a pose; every
 * predicate names a graph fact and, when it is a probability, carries the decision boundary the user stated. The LLM
 * never writes it; the controller never sees anything less grounded. It travels as JSON in chat_intention's
 * `current_intention` (mission_json.h) and is shown to people as a DSL (later: mission_text.h). Both syntaxes are
 * driven by the registry below, so an op cannot exist in one place and not another.
 *
 * ★ OPS MAP TO CONTROLLER MECHANISMS, NOT TO OBJECT TYPES (door passage, point target, selector prior, orient, wait,
 *   relative motion). A new concept agent becomes reachable without an IR change; ops multiplying per object type is
 *   the failure mode this registry exists to prevent.
 *
 * ★ COMPATIBILITY (§7.1): an unknown op / ref kind / predicate rejects the WHOLE mission before anything moves; an
 *   unknown optional field is ignored; a field listed in `requires_fields` that this build does not know rejects;
 *   `v` must equal kVersion.
 *
 * ★ v1 RESERVATIONS (§7.4): step ids, `bind`, Ref::Kind::Var and `prefs` are in the format so adding bindings,
 *   for_each and preferences later does not change the wire; v1's validate() refuses them as `unsupported:<name>`.
 *
 * ★ TERMINATION IS CHECKED BEFORE RUNNING: every loop carries a budget (repeat-until needs budget_s, explore needs
 *   budget_s). A `wait until` may still last until the person answers — `resume` ends it (§7.5).
 *
 * Pure: no graph, no Qt.
 */

#include <cstdint>
#include <format>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace rc::mission
{

inline constexpr int kVersion = 1;

// ── Types ─────────────────────────────────────────────────────────────────────────────────────────────────────────

/// A pose in a room's frame, with its xy uncertainty (an anchor carries σ so the arrival tolerance can follow it).
struct Pose
{
    std::string room;
    double x = 0, y = 0, yaw = 0, sigma_xy = 0;
    bool operator==(const Pose&) const = default;
};

/// A grounded reference.
struct Ref
{
    enum class Kind
    {
        Node,      ///< a live node, by name: door_27, room_2, table_1
        Passage,   ///< "the live door of the room I am in that faces this passage" — resolved by the controller on arrival
        Anchor,    ///< a named pose: in Mission::anchors, or captured by an earlier `anchor` step
        Pose,      ///< an explicit pose in a room frame
        Var,       ///< RESERVED (bindings, §7.2 L1): refused by v1
    };
    Kind kind = Kind::Node;
    std::string name{};   ///< node name · passage id · anchor name · variable name
    std::string from{};   ///< Passage: the room the robot is in when the face is resolved
    Pose pose{};          ///< Pose
    bool operator==(const Ref&) const = default;
};

[[nodiscard]] constexpr std::uint8_t kind_bit(Ref::Kind k) { return std::uint8_t(1u << unsigned(k)); }

/// A decision on a probability, in the user's own terms ("if it is shut" → door_open lt 0.5).
struct Comparison
{
    enum class Op { Ge, Lt };
    Op op = Op::Ge;
    double value = 0.5;
    bool operator==(const Comparison&) const = default;
};

/// A predicate over beliefs. Probabilistic ones carry `cmp`; fact ones must not.
struct Pred
{
    std::string name{};
    std::map<std::string, Ref, std::less<>> refs{};
    std::optional<Comparison> cmp{};
    bool operator==(const Pred&) const = default;
};

using Value = std::variant<double, std::int64_t, std::string, Ref, Pred>;

/// What a failed step does to its parent. Default: the failure propagates (abort).
struct OnFail
{
    enum class Kind { Abort, Skip, Retry };
    Kind kind = Kind::Abort;
    int retries = 0;   ///< Retry: further attempts after the first
    bool operator==(const OnFail&) const = default;
};

struct Step
{
    std::string id{};   ///< unique in the mission ("1", "2.3"); what progress, bind and replies refer to
    std::string op{};
    std::map<std::string, Value, std::less<>> args{};
    std::vector<Step> steps{};   ///< body of a composite op
    OnFail on_fail{};
    std::optional<std::string> bind{};   ///< RESERVED (§7.2 L1): refused by v1

    bool operator==(const Step&) const = default;

    template <class T>
    [[nodiscard]] std::optional<T> get(std::string_view key) const
    {
        const auto it = args.find(key);
        if (it == args.end()) return std::nullopt;
        if (const T* v = std::get_if<T>(&it->second)) return *v;
        return std::nullopt;
    }
    [[nodiscard]] std::optional<double> number(std::string_view k) const { return get<double>(k); }
    [[nodiscard]] std::optional<std::int64_t> integer(std::string_view k) const { return get<std::int64_t>(k); }
    [[nodiscard]] std::optional<std::string> string(std::string_view k) const { return get<std::string>(k); }
    [[nodiscard]] std::optional<Ref> ref(std::string_view k) const { return get<Ref>(k); }
    [[nodiscard]] std::optional<Pred> pred(std::string_view k) const { return get<Pred>(k); }
    [[nodiscard]] bool has(std::string_view k) const { return args.contains(k); }
};

struct Mission
{
    int v = kVersion;
    std::string title;
    std::map<std::string, Pose, std::less<>> anchors;
    std::map<std::string, double, std::less<>> prefs;   ///< RESERVED (§7.2 L4): must be empty in v1
    std::vector<std::string> requires_fields;            ///< fields whose meaning this build must know
    std::vector<Step> steps;
    bool operator==(const Mission&) const = default;
};

/// Why a mission was refused. status() is what the controller puts after `failed|` on the edge.
struct Error
{
    std::string code{};    ///< malformed · unsupported_version · unsupported · missing_field · bad_type · bad_value · duplicate_id
    std::string name{};    ///< unsupported: what was not supported (fly · ref:region · pred:battery_low · bind · speed)
    std::string where{};   ///< "step 2.1 door"
    std::string text{};    ///< for people and logs
    [[nodiscard]] std::string status() const { return code == "unsupported" ? code + ":" + name : code; }
    [[nodiscard]] std::string message() const
    {
        return std::format("{}{}{}", status(), where.empty() ? "" : " at " + where, text.empty() ? "" : ": " + text);
    }
};

// ── Registry: the single source of the schema, validation, dispatch and DSL ──────────────────────────────────────

enum class FieldType { Number, Integer, String, Ref, Pred };

/// Who a step occupies while it runs. At most one step owns the base (a future `parallel` relies on it, §7.4).
enum class Resource { Base, Observer, Instant, Composite };

struct FieldSpec
{
    std::string_view name;
    FieldType type;
    bool required = false;
    std::uint8_t ref_kinds = 0;   ///< Ref fields: the kinds accepted
    bool positive = false;        ///< Number fields: must be > 0
};

struct OpSpec
{
    std::string_view name;
    Resource resource;
    bool has_body;
    std::vector<FieldSpec> fields;
    std::optional<Error> (*check)(const Step&) = nullptr;   ///< constraints across fields
};

struct PredSpec
{
    std::string_view name;
    bool probabilistic;   ///< a belief with a probability → needs the boundary; else a fact → must not have one
    std::vector<FieldSpec> fields;
};

namespace detail
{
inline Error err(std::string code, const Step& s, std::string_view field, std::string text)
{
    return {.code = std::move(code), .where = std::format("step {}{}{}", s.id, field.empty() ? "" : " ", field),
            .text = std::move(text)};
}

inline std::optional<Error> check_repeat(const Step& s)
{
    if (s.has("times") == s.has("until"))
        return err("bad_value", s, "", "repeat needs exactly one of times / until");
    if (s.has("until") and not s.has("budget_s"))
        return err("missing_field", s, "budget_s", "repeat-until needs a budget so it terminates");
    if (const auto n = s.integer("times"); n.has_value() and *n < 1) return err("bad_value", s, "times", "must be >= 1");
    return std::nullopt;
}
inline std::optional<Error> check_wait(const Step& s)
{
    if (not s.has("s") and not s.has("until")) return err("bad_value", s, "", "wait for what? give s or until");
    return std::nullopt;
}
inline std::optional<Error> check_move(const Step& s)
{
    if (not s.has("distance_m") and not s.has("turn_deg"))
        return err("bad_value", s, "", "move needs distance_m and/or turn_deg");
    return std::nullopt;
}

constexpr std::uint8_t kDoor = kind_bit(Ref::Kind::Node) | kind_bit(Ref::Kind::Passage);
constexpr std::uint8_t kNode = kind_bit(Ref::Kind::Node);
constexpr std::uint8_t kPlace = kind_bit(Ref::Kind::Anchor) | kind_bit(Ref::Kind::Pose);
}   // namespace detail

/// Phase-1 ops (doc §6.2 + amendment 3). Adding an op = one entry here + its interpreter case.
inline const std::vector<OpSpec>& ops()
{
    using enum FieldType;
    using namespace detail;
    static const std::vector<OpSpec> table = {
        {.name = "seq", .resource = Resource::Composite, .has_body = true, .fields = {}},
        {.name = "repeat", .resource = Resource::Composite, .has_body = true,
         .fields = {{.name = "times", .type = Integer},
                    {.name = "until", .type = Pred},
                    {.name = "budget_s", .type = Number, .positive = true}},
         .check = check_repeat},
        {.name = "cross", .resource = Resource::Base, .has_body = false,
         .fields = {{.name = "door", .type = Ref, .required = true, .ref_kinds = kDoor},
                    {.name = "expect", .type = String}}},
        {.name = "open", .resource = Resource::Base, .has_body = false,
         .fields = {{.name = "door", .type = Ref, .required = true, .ref_kinds = kDoor}}},
        {.name = "go_to", .resource = Resource::Base, .has_body = false,
         .fields = {{.name = "to", .type = Ref, .required = true, .ref_kinds = kPlace},
                    {.name = "standoff_m", .type = Number, .positive = true}}},
        {.name = "explore", .resource = Resource::Base, .has_body = false,
         .fields = {{.name = "room", .type = Ref, .required = true, .ref_kinds = kNode},
                    {.name = "until", .type = Pred},
                    {.name = "budget_s", .type = Number, .required = true, .positive = true}}},
        {.name = "wait", .resource = Resource::Base, .has_body = false,
         .fields = {{.name = "s", .type = Number, .positive = true}, {.name = "until", .type = Pred}},
         .check = check_wait},
        {.name = "say", .resource = Resource::Instant, .has_body = false,
         .fields = {{.name = "text", .type = String, .required = true}}},
        {.name = "anchor", .resource = Resource::Instant, .has_body = false,
         .fields = {{.name = "name", .type = String, .required = true}}},
        {.name = "move", .resource = Resource::Base, .has_body = false,
         .fields = {{.name = "distance_m", .type = Number}, {.name = "turn_deg", .type = Number}},
         .check = check_move},
        {.name = "stop", .resource = Resource::Instant, .has_body = false, .fields = {}},
    };
    return table;
}

/// Phase-1 predicates. Adding one = one entry here + its evaluator.
inline const std::vector<PredSpec>& preds()
{
    using enum FieldType;
    using namespace detail;
    static const std::vector<PredSpec> table = {
        {.name = "explored", .probabilistic = true,   // room_concept's explore readiness (Phase 0 attribute)
         .fields = {{.name = "room", .type = Ref, .required = true, .ref_kinds = kNode}}},
        {.name = "in_room", .probabilistic = false,   // rc::room::current_room == room
         .fields = {{.name = "room", .type = Ref, .required = true, .ref_kinds = kNode}}},
        {.name = "door_open", .probabilistic = true,  // door_open_prob
         .fields = {{.name = "door", .type = Ref, .required = true, .ref_kinds = kDoor}}},
    };
    return table;
}

[[nodiscard]] inline const OpSpec* op_spec(std::string_view name)
{
    for (const auto& s : ops())
        if (s.name == name) return &s;
    return nullptr;
}
[[nodiscard]] inline const PredSpec* pred_spec(std::string_view name)
{
    for (const auto& s : preds())
        if (s.name == name) return &s;
    return nullptr;
}

/// Every field name this build understands (what `requires_fields` is checked against).
[[nodiscard]] inline bool known_field(std::string_view name)
{
    for (const auto k : {"id", "op", "steps", "on_fail", "bind"})
        if (name == k) return true;
    for (const auto& o : ops())
        for (const auto& f : o.fields)
            if (f.name == name) return true;
    for (const auto& p : preds())
        for (const auto& f : p.fields)
            if (f.name == name) return true;
    return false;
}

// ── Validation ────────────────────────────────────────────────────────────────────────────────────────────────────

namespace detail
{
inline bool type_matches(const Value& v, FieldType t)
{
    switch (t)
    {
        case FieldType::Number: return std::holds_alternative<double>(v);
        case FieldType::Integer: return std::holds_alternative<std::int64_t>(v);
        case FieldType::String: return std::holds_alternative<std::string>(v);
        case FieldType::Ref: return std::holds_alternative<rc::mission::Ref>(v);
        case FieldType::Pred: return std::holds_alternative<rc::mission::Pred>(v);
    }
    return false;
}

class Validator
{
public:
    explicit Validator(const Mission& m)
    {
        for (const auto& [name, _] : m.anchors) anchors_.insert(name);
    }

    std::optional<Error> steps(const std::vector<Step>& list)
    {
        for (const auto& s : list)
            if (auto e = step(s); e.has_value()) return e;
        return std::nullopt;
    }

private:
    std::set<std::string, std::less<>> ids_, anchors_;

    std::optional<Error> step(const Step& s)
    {
        if (s.id.empty()) return err("missing_field", s, "id", "every step has an id");
        if (not ids_.insert(s.id).second) return err("duplicate_id", s, "", "step ids must be unique");
        const OpSpec* spec = op_spec(s.op);
        if (spec == nullptr) return unsupported(s.op, s);
        if (s.bind.has_value()) return unsupported("bind", s);
        for (const auto& f : spec->fields)
            if (auto e = field(s, f, s.args); e.has_value()) return e;
        if (spec->has_body and s.steps.empty()) return err("bad_value", s, "steps", s.op + " needs a body");
        if (not spec->has_body and not s.steps.empty()) return err("bad_value", s, "steps", s.op + " takes no body");
        if (s.on_fail.kind == OnFail::Kind::Retry and s.on_fail.retries < 1)
            return err("bad_value", s, "on_fail", "retry needs >= 1");
        if (spec->check != nullptr)
            if (auto e = spec->check(s); e.has_value()) return e;
        if (s.op == "anchor") anchors_.insert(*s.string("name"));   // defined from here on, in execution order
        return steps(s.steps);
    }

    template <class Args>
    std::optional<Error> field(const Step& s, const FieldSpec& f, const Args& args)
    {
        const auto it = args.find(f.name);
        if (it == args.end())
        {
            if (f.required) return err("missing_field", s, f.name, "required");
            return std::nullopt;
        }
        if (not type_matches(it->second, f.type)) return err("bad_type", s, f.name, "wrong type");
        if (f.positive)
            if (const double* d = std::get_if<double>(&it->second); d != nullptr and not(*d > 0))
                return err("bad_value", s, f.name, "must be > 0");
        if (const auto* r = std::get_if<rc::mission::Ref>(&it->second)) return ref(s, f, *r);
        if (const auto* p = std::get_if<rc::mission::Pred>(&it->second)) return pred(s, f.name, *p);
        return std::nullopt;
    }

    std::optional<Error> ref(const Step& s, const FieldSpec& f, const rc::mission::Ref& r)
    {
        if (r.kind == Ref::Kind::Var) return unsupported("var", s);
        if ((f.ref_kinds & kind_bit(r.kind)) == 0) return err("bad_value", s, f.name, "this kind of reference is not accepted here");
        if (r.kind == Ref::Kind::Passage and r.from.empty())
            return err("missing_field", s, f.name, "a passage reference needs `from`");
        if (r.kind == Ref::Kind::Pose and r.pose.room.empty())
            return err("missing_field", s, f.name, "a pose needs its room");
        if (r.kind != Ref::Kind::Pose and r.name.empty()) return err("missing_field", s, f.name, "empty reference");
        if (r.kind == Ref::Kind::Anchor and not anchors_.contains(r.name))
            return err("bad_value", s, f.name, std::format("anchor '{}' is not defined before this step", r.name));
        return std::nullopt;
    }

    std::optional<Error> pred(const Step& s, std::string_view field_name, const rc::mission::Pred& p)
    {
        const PredSpec* spec = pred_spec(p.name);
        if (spec == nullptr) return unsupported("pred:" + p.name, s);
        for (const auto& f : spec->fields)
        {
            const auto it = p.refs.find(f.name);
            if (it == p.refs.end())
            {
                if (f.required) return err("missing_field", s, f.name, std::format("{} needs it", p.name));
                continue;
            }
            if (auto e = ref(s, f, it->second); e.has_value()) return e;
        }
        if (spec->probabilistic and not p.cmp.has_value())
            return err("missing_field", s, field_name, p.name + " is a probability: state the decision (ge / lt)");
        if (not spec->probabilistic and p.cmp.has_value())
            return err("bad_value", s, field_name, p.name + " is a fact: it takes no ge / lt");
        if (p.cmp.has_value() and not(p.cmp->value >= 0.0 and p.cmp->value <= 1.0))
            return err("bad_value", s, field_name, "a probability boundary is in [0, 1]");
        return std::nullopt;
    }

    static Error unsupported(std::string name, const Step& s)
    {
        return {.code = "unsupported", .name = std::move(name), .where = std::format("step {}", s.id),
                .text = "not supported by this build"};
    }
};
}   // namespace detail

/// Check a mission against the registry and the v1 rules. nullopt = it may run.
[[nodiscard]] inline std::optional<Error> validate(const Mission& m)
{
    if (m.v != kVersion)
        return Error{.code = "unsupported_version", .text = std::format("mission v{}, this build speaks v{}", m.v, kVersion)};
    if (not m.prefs.empty()) return Error{.code = "unsupported", .name = "prefs", .text = "reserved for a later version"};
    for (const auto& f : m.requires_fields)
        if (not known_field(f)) return Error{.code = "unsupported", .name = f, .text = "required field not known to this build"};
    if (m.steps.empty()) return Error{.code = "bad_value", .where = "steps", .text = "a mission needs at least one step"};
    return detail::Validator(m).steps(m.steps);
}

/// Hierarchical ids ("1", "2", "2.1") for a tree built by the resolver.
inline void assign_ids(std::vector<Step>& steps, const std::string& prefix = "")
{
    for (std::size_t i = 0; i < steps.size(); ++i)
    {
        steps[i].id = prefix + std::to_string(i + 1);
        assign_ids(steps[i].steps, steps[i].id + ".");
    }
}
inline void assign_ids(Mission& m) { assign_ids(m.steps); }

}   // namespace rc::mission
