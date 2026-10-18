#pragma once
/*
 * mission_json.h — the wire form of the grounded mission (mission_ir.h): what chat_agent writes into
 * chat_intention's `current_intention` and the controller decodes (2026-10-17).
 *
 *   {"v":1, "title":…, "anchors":{name:{room,x,y,yaw,sigma_xy}}, "prefs":{}, "requires":[…],
 *    "steps":[{"id":"1", "op":"cross", "door":{"node":"door_27"}, "expect":"room_2", "on_fail":{"retry":2}}, …]}
 *
 *   Ref   {"node":n} · {"passage":p,"from":room} · {"anchor":a} · {"pose":{room,x,y,yaw,sigma_xy}} · {"var":v}
 *   Pred  {"p":name, <ref fields>…, "ge":x | "lt":x}
 *   on_fail  "abort" (default, omitted) · "skip" · {"retry":n}
 *
 * DECODING IS DRIVEN BY THE REGISTRY: a field is read with the type its OpSpec/PredSpec declares; a field the
 * registry does not list is dropped (an unknown OPTIONAL field is ignored — compatibility rule, §7.1). The version is
 * checked before any step is read, so a newer mission is refused as unsupported_version, not as an unknown op.
 * decode_json() ends with validate(): a decoded mission is a runnable one.
 *
 * Qt's JSON (nlohmann is not installed here; both agents link Qt6Core). QJsonDocument parses and prints numbers
 * independently of the C locale, so es_ES's decimal comma cannot truncate a pose (CLAUDE.md).
 */

#include <expected>
#include <string>
#include <string_view>

#include <QByteArray>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QString>

#include "mission_ir.h"

namespace rc::mission
{

namespace json_detail
{
inline QString q(std::string_view s) { return QString::fromUtf8(s.data(), qsizetype(s.size())); }

inline Error bad(std::string code, std::string where, std::string text)
{
    return {.code = std::move(code), .where = std::move(where), .text = std::move(text)};
}

inline std::expected<double, Error> number(const QJsonValue& v, const std::string& where)
{
    if (not v.isDouble()) return std::unexpected(bad("bad_type", where, "expected a number"));
    return v.toDouble();
}

inline std::expected<Pose, Error> pose(const QJsonValue& v, const std::string& where)
{
    if (not v.isObject()) return std::unexpected(bad("bad_type", where, "expected a pose object"));
    const QJsonObject o = v.toObject();
    Pose p;
    if (not o.value("room").isString()) return std::unexpected(bad("missing_field", where + " room", "a pose needs its room"));
    p.room = o.value("room").toString().toStdString();
    for (const auto& [key, dst] : {std::pair{"x", &p.x}, {"y", &p.y}, {"yaw", &p.yaw}, {"sigma_xy", &p.sigma_xy}})
        if (o.contains(key))
        {
            const auto d = number(o.value(key), where + " " + key);
            if (not d.has_value()) return std::unexpected(d.error());
            *dst = *d;
        }
    return p;
}

inline std::expected<Ref, Error> ref(const QJsonValue& v, const std::string& where)
{
    if (not v.isObject()) return std::unexpected(bad("bad_type", where, "expected a reference object"));
    const QJsonObject o = v.toObject();
    static constexpr std::pair<const char*, Ref::Kind> kinds[] = {
        {"node", Ref::Kind::Node}, {"passage", Ref::Kind::Passage}, {"anchor", Ref::Kind::Anchor},
        {"pose", Ref::Kind::Pose}, {"var", Ref::Kind::Var}};
    std::optional<std::pair<const char*, Ref::Kind>> found;
    for (const auto& k : kinds)
        if (o.contains(k.first))
        {
            if (found.has_value())
                return std::unexpected(bad("malformed", where, std::format("both '{}' and '{}'", found->first, k.first)));
            found = k;
        }
    if (not found.has_value())
    {
        if (o.isEmpty()) return std::unexpected(bad("malformed", where, "empty reference"));
        return std::unexpected(Error{.code = "unsupported", .name = "ref:" + o.keys().front().toStdString(),
                                     .where = where, .text = "unknown reference kind"});
    }
    Ref r{.kind = found->second};
    if (r.kind == Ref::Kind::Pose)
    {
        auto p = pose(o.value("pose"), where);
        if (not p.has_value()) return std::unexpected(p.error());
        r.pose = std::move(*p);
        return r;
    }
    const QJsonValue name = o.value(found->first);
    if (not name.isString()) return std::unexpected(bad("bad_type", where, "a reference names a string"));
    r.name = name.toString().toStdString();
    if (o.contains("from"))
    {
        if (not o.value("from").isString()) return std::unexpected(bad("bad_type", where + " from", "expected a room name"));
        r.from = o.value("from").toString().toStdString();
    }
    return r;
}

inline std::expected<Pred, Error> pred(const QJsonValue& v, const std::string& where)
{
    if (not v.isObject()) return std::unexpected(bad("bad_type", where, "expected a predicate object"));
    const QJsonObject o = v.toObject();
    if (not o.contains("p")) return std::unexpected(bad("missing_field", where + " p", "a predicate names itself in p"));
    if (not o.value("p").isString()) return std::unexpected(bad("bad_type", where + " p", "expected a name"));
    Pred p{.name = o.value("p").toString().toStdString()};
    const PredSpec* spec = pred_spec(p.name);
    if (spec == nullptr)
        return std::unexpected(Error{.code = "unsupported", .name = "pred:" + p.name, .where = where,
                                     .text = "unknown predicate"});
    for (const auto& f : spec->fields)
        if (o.contains(q(f.name)))
        {
            auto r = ref(o.value(q(f.name)), where + " " + std::string(f.name));
            if (not r.has_value()) return std::unexpected(r.error());
            p.refs.emplace(std::string(f.name), std::move(*r));
        }
    if (o.contains("ge") and o.contains("lt")) return std::unexpected(bad("bad_value", where, "one decision: ge or lt"));
    for (const auto& [key, op] : {std::pair{"ge", Comparison::Op::Ge}, {"lt", Comparison::Op::Lt}})
        if (o.contains(key))
        {
            const auto d = number(o.value(key), where + " " + key);
            if (not d.has_value()) return std::unexpected(d.error());
            p.cmp = Comparison{.op = op, .value = *d};
        }
    return p;
}

inline std::expected<Value, Error> value(const QJsonValue& v, FieldType t, const std::string& where)
{
    switch (t)
    {
        case FieldType::Number:
        {
            auto d = number(v, where);
            if (not d.has_value()) return std::unexpected(d.error());
            return Value{*d};
        }
        case FieldType::Integer:
        {
            const double d = v.toDouble();
            if (not v.isDouble() or d != double(std::int64_t(d)))
                return std::unexpected(bad("bad_type", where, "expected an integer"));
            return Value{std::int64_t(d)};
        }
        case FieldType::String:
            if (not v.isString()) return std::unexpected(bad("bad_type", where, "expected a string"));
            return Value{v.toString().toStdString()};
        case FieldType::Ref:
        {
            auto r = ref(v, where);
            if (not r.has_value()) return std::unexpected(r.error());
            return Value{std::move(*r)};
        }
        case FieldType::Pred:
        {
            auto p = pred(v, where);
            if (not p.has_value()) return std::unexpected(p.error());
            return Value{std::move(*p)};
        }
    }
    return std::unexpected(bad("bad_type", where, "unknown field type"));
}

inline std::expected<OnFail, Error> on_fail(const QJsonValue& v, const std::string& where)
{
    if (v.isString())
    {
        if (v.toString() == "abort") return OnFail{};
        if (v.toString() == "skip") return OnFail{.kind = OnFail::Kind::Skip};
    }
    else if (v.isObject() and v.toObject().value("retry").isDouble())
    {
        const double n = v.toObject().value("retry").toDouble();
        if (n == double(int(n))) return OnFail{.kind = OnFail::Kind::Retry, .retries = int(n)};
    }
    return std::unexpected(bad("bad_value", where, "on_fail is \"abort\", \"skip\" or {\"retry\":n}"));
}

inline std::expected<std::vector<Step>, Error> steps(const QJsonValue& v, const std::string& where);

inline std::expected<Step, Error> step(const QJsonValue& v, const std::string& where)
{
    if (not v.isObject()) return std::unexpected(bad("bad_type", where, "a step is an object"));
    const QJsonObject o = v.toObject();
    Step s;
    if (o.contains("id"))
    {
        if (not o.value("id").isString()) return std::unexpected(bad("bad_type", where + " id", "expected a string"));
        s.id = o.value("id").toString().toStdString();
    }
    const std::string at = s.id.empty() ? where : "step " + s.id;
    if (not o.contains("op")) return std::unexpected(bad("missing_field", at + " op", "every step has an op"));
    if (not o.value("op").isString()) return std::unexpected(bad("bad_type", at + " op", "expected a name"));
    s.op = o.value("op").toString().toStdString();
    const OpSpec* spec = op_spec(s.op);
    if (spec == nullptr)
        return std::unexpected(Error{.code = "unsupported", .name = s.op, .where = at, .text = "unknown op"});
    for (const auto& f : spec->fields)
        if (o.contains(q(f.name)))
        {
            auto val = value(o.value(q(f.name)), f.type, at + " " + std::string(f.name));
            if (not val.has_value()) return std::unexpected(val.error());
            s.args.emplace(std::string(f.name), std::move(*val));
        }
    if (o.contains("on_fail"))
    {
        auto of = on_fail(o.value("on_fail"), at + " on_fail");
        if (not of.has_value()) return std::unexpected(of.error());
        s.on_fail = *of;
    }
    if (o.contains("bind"))
    {
        if (not o.value("bind").isString()) return std::unexpected(bad("bad_type", at + " bind", "expected a name"));
        s.bind = o.value("bind").toString().toStdString();
    }
    if (o.contains("steps"))
    {
        auto body = steps(o.value("steps"), at + " steps");
        if (not body.has_value()) return std::unexpected(body.error());
        s.steps = std::move(*body);
    }
    return s;
}

inline std::expected<std::vector<Step>, Error> steps(const QJsonValue& v, const std::string& where)
{
    if (not v.isArray()) return std::unexpected(bad("bad_type", where, "expected an array of steps"));
    std::vector<Step> out;
    const QJsonArray a = v.toArray();
    for (qsizetype i = 0; i < a.size(); ++i)
    {
        auto s = step(a.at(i), std::format("{}[{}]", where, i));
        if (not s.has_value()) return std::unexpected(s.error());
        out.push_back(std::move(*s));
    }
    return out;
}

// ── encoding ─────────────────────────────────────────────────────────────────────────────────────────────────────

inline QJsonObject pose_json(const Pose& p)
{
    return {{"room", q(p.room)}, {"x", p.x}, {"y", p.y}, {"yaw", p.yaw}, {"sigma_xy", p.sigma_xy}};
}

inline QJsonObject ref_json(const Ref& r)
{
    switch (r.kind)
    {
        case Ref::Kind::Node: return {{"node", q(r.name)}};
        case Ref::Kind::Passage: return {{"passage", q(r.name)}, {"from", q(r.from)}};
        case Ref::Kind::Anchor: return {{"anchor", q(r.name)}};
        case Ref::Kind::Pose: return {{"pose", pose_json(r.pose)}};
        case Ref::Kind::Var: return {{"var", q(r.name)}};
    }
    return {};
}

inline QJsonObject pred_json(const Pred& p)
{
    QJsonObject o{{"p", q(p.name)}};
    for (const auto& [name, r] : p.refs) o.insert(q(name), ref_json(r));
    if (p.cmp.has_value()) o.insert(p.cmp->op == Comparison::Op::Ge ? "ge" : "lt", p.cmp->value);
    return o;
}

inline QJsonValue value_json(const Value& v)
{
    return std::visit(
        [](const auto& x) -> QJsonValue
        {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, double>) return x;
            else if constexpr (std::is_same_v<T, std::int64_t>) return qint64(x);
            else if constexpr (std::is_same_v<T, std::string>) return q(x);
            else if constexpr (std::is_same_v<T, Ref>) return ref_json(x);
            else return pred_json(x);
        },
        v);
}

inline QJsonArray steps_json(const std::vector<Step>& list)
{
    QJsonArray a;
    for (const auto& s : list)
    {
        QJsonObject o{{"id", q(s.id)}, {"op", q(s.op)}};
        for (const auto& [name, v] : s.args) o.insert(q(name), value_json(v));
        switch (s.on_fail.kind)
        {
            case OnFail::Kind::Abort: break;   // the default, omitted
            case OnFail::Kind::Skip: o.insert("on_fail", "skip"); break;
            case OnFail::Kind::Retry: o.insert("on_fail", QJsonObject{{"retry", s.on_fail.retries}}); break;
        }
        if (s.bind.has_value()) o.insert("bind", q(*s.bind));
        if (not s.steps.empty()) o.insert("steps", steps_json(s.steps));
        a.append(o);
    }
    return a;
}
}   // namespace json_detail

/// Parse + validate. A returned mission is one the controller may run.
[[nodiscard]] inline std::expected<Mission, Error> decode_json(std::string_view text)
{
    using namespace json_detail;
    QJsonParseError perr{};
    const QJsonDocument doc = QJsonDocument::fromJson(QByteArray(text.data(), qsizetype(text.size())), &perr);
    if (perr.error != QJsonParseError::NoError or not doc.isObject())
        return std::unexpected(bad("malformed", "", perr.error != QJsonParseError::NoError
                                                        ? perr.errorString().toStdString() : "not a JSON object"));
    const QJsonObject o = doc.object();
    Mission m;
    if (not o.value("v").isDouble()) return std::unexpected(bad("malformed", "v", "a mission states its version"));
    m.v = int(o.value("v").toDouble());
    if (m.v != kVersion or o.value("v").toDouble() != double(m.v))
        return std::unexpected(Error{.code = "unsupported_version",
                                     .text = std::format("mission v{}, this build speaks v{}", o.value("v").toDouble(), kVersion)});
    if (o.contains("title"))
    {
        if (not o.value("title").isString()) return std::unexpected(bad("bad_type", "title", "expected a string"));
        m.title = o.value("title").toString().toStdString();
    }
    if (o.contains("anchors"))
    {
        if (not o.value("anchors").isObject()) return std::unexpected(bad("bad_type", "anchors", "expected an object"));
        const QJsonObject a = o.value("anchors").toObject();
        for (auto it = a.begin(); it != a.end(); ++it)
        {
            auto p = pose(it.value(), "anchor " + it.key().toStdString());
            if (not p.has_value()) return std::unexpected(p.error());
            m.anchors.emplace(it.key().toStdString(), std::move(*p));
        }
    }
    if (o.contains("prefs"))
    {
        if (not o.value("prefs").isObject()) return std::unexpected(bad("bad_type", "prefs", "expected an object"));
        const QJsonObject p = o.value("prefs").toObject();
        for (auto it = p.begin(); it != p.end(); ++it)
        {
            const auto d = number(it.value(), "prefs " + it.key().toStdString());
            if (not d.has_value()) return std::unexpected(d.error());
            m.prefs.emplace(it.key().toStdString(), *d);
        }
    }
    if (o.contains("requires"))
    {
        if (not o.value("requires").isArray()) return std::unexpected(bad("bad_type", "requires", "expected an array"));
        for (const auto& r : o.value("requires").toArray())
        {
            if (not r.isString()) return std::unexpected(bad("bad_type", "requires", "expected field names"));
            m.requires_fields.push_back(r.toString().toStdString());
        }
    }
    if (not o.contains("steps")) return std::unexpected(bad("malformed", "steps", "a mission has steps"));
    auto body = steps(o.value("steps"), "steps");
    if (not body.has_value()) return std::unexpected(body.error());
    m.steps = std::move(*body);
    if (auto e = validate(m); e.has_value()) return std::unexpected(*e);
    return m;
}

/// Compact JSON. decode_json(encode_json(m)) == m for every mission validate() accepts.
[[nodiscard]] inline std::string encode_json(const Mission& m)
{
    using namespace json_detail;
    QJsonObject o{{"v", m.v}, {"steps", steps_json(m.steps)}};
    if (not m.title.empty()) o.insert("title", q(m.title));
    if (not m.anchors.empty())
    {
        QJsonObject a;
        for (const auto& [name, p] : m.anchors) a.insert(q(name), pose_json(p));
        o.insert("anchors", a);
    }
    if (not m.prefs.empty())
    {
        QJsonObject p;
        for (const auto& [name, w] : m.prefs) p.insert(q(name), w);
        o.insert("prefs", p);
    }
    if (not m.requires_fields.empty())
    {
        QJsonArray r;
        for (const auto& f : m.requires_fields) r.append(q(f));
        o.insert("requires", r);
    }
    return QJsonDocument(o).toJson(QJsonDocument::Compact).toStdString();
}

}   // namespace rc::mission
