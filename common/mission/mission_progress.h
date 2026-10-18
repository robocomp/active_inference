#pragma once
/*
 * mission_progress.h — how a running mission is reported OUT of the controller (2026-10-17): the JSON snapshot on the
 * controller's `controller_mission` node (cortex attribute mission_progress).
 *
 *   mission_progress       {"serial":3, "source":"remote", "title":…, "status":"in_progress", "t0":…, "t1":0,
 *                           "line":"<Interpreter::progress()>",
 *                           "steps":[{"id":"2.1","op":"cross","state":"running","outcome":"","attempt":1,
 *                                     "iteration":0,"t0":…,"t1":0}, …]}          — a snapshot, pre-order
 *
 * The snapshot is what chat's plan view animates (it already holds the IR, from current_intention, and matches steps
 * by id). What EM records is the controller's journal (common/episodic/em_journal.h), not this.
 *
 * Qt's JSON, as mission_json.h: numbers are printed and parsed independently of the C locale (es_ES, CLAUDE.md).
 * Times are ms since the epoch; a double holds them exactly (< 2^53).
 */

#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include "mission_interpreter.h"
#include "mission_json.h"

namespace rc::mission
{

[[nodiscard]] inline const char* step_state_word(Interpreter::StepState s)
{
    using SS = Interpreter::StepState;
    switch (s)
    {
        case SS::Pending: return "pending";
        case SS::Running: return "running";
        case SS::Satisfied: return "satisfied";
        case SS::Failed: return "failed";
        case SS::Skipped: return "skipped";
        case SS::Cancelled: return "cancelled";
    }
    return "pending";
}

[[nodiscard]] inline std::optional<Interpreter::StepState> step_state_from(std::string_view w)
{
    using SS = Interpreter::StepState;
    for (const SS s : {SS::Pending, SS::Running, SS::Satisfied, SS::Failed, SS::Skipped, SS::Cancelled})
        if (w == step_state_word(s)) return s;
    return std::nullopt;
}

/// The decoded snapshot.
struct Progress
{
    std::uint64_t serial = 0;
    std::string source, title;
    std::string status;   ///< Interpreter::status(): "" (idle) · in_progress · completed · failed|… · aborted|…
    std::uint64_t t0 = 0, t1 = 0;
    std::string line;     ///< Interpreter::progress()
    std::vector<Interpreter::StepView> steps;
};

namespace progress_detail
{
using json_detail::q;
inline std::string s(const QJsonValue& v) { return v.toString().toStdString(); }
inline std::uint64_t ms(const QJsonValue& v) { return std::uint64_t(v.toDouble()); }
inline std::string compact(const QJsonObject& o)
{
    return QJsonDocument(o).toJson(QJsonDocument::Compact).toStdString();
}
inline std::expected<QJsonObject, std::string> object(std::string_view text)
{
    QJsonParseError err{};
    const auto doc = QJsonDocument::fromJson(QByteArray(text.data(), qsizetype(text.size())), &err);
    if (err.error != QJsonParseError::NoError) return std::unexpected("bad json: " + err.errorString().toStdString());
    if (not doc.isObject()) return std::unexpected(std::string("bad json: an object is expected"));
    return doc.object();
}
inline std::expected<Interpreter::StepState, std::string> state(const QJsonValue& v)
{
    const auto st = step_state_from(s(v));
    if (not st.has_value()) return std::unexpected("bad step state '" + s(v) + "'");
    return *st;
}
}   // namespace progress_detail

[[nodiscard]] inline std::string encode_progress(const Interpreter& in, std::string_view source)
{
    using namespace progress_detail;
    QJsonArray steps;
    for (const auto& v : in.steps())
        steps.append(QJsonObject{{"id", q(v.id)},
                                 {"op", q(v.op)},
                                 {"state", step_state_word(v.state)},
                                 {"outcome", q(v.outcome)},
                                 {"attempt", v.attempt},
                                 {"iteration", v.iteration},
                                 {"t0", double(v.t0)},
                                 {"t1", double(v.t1)}});
    const auto [t0, t1] = in.span();
    const bool idle = in.state() == Interpreter::State::Idle;
    return compact(QJsonObject{{"serial", double(in.serial())},
                               {"source", q(source)},
                               {"title", q(in.title())},
                               {"status", q(in.status())},
                               {"t0", double(t0)},
                               {"t1", double(t1)},
                               {"line", idle ? QString() : q(in.progress())},
                               {"steps", steps}});
}

[[nodiscard]] inline std::expected<Progress, std::string> decode_progress(std::string_view text)
{
    using namespace progress_detail;
    const auto o = object(text);
    if (not o.has_value()) return std::unexpected(o.error());
    Progress p{.serial = ms((*o)["serial"]),
               .source = s((*o)["source"]),
               .title = s((*o)["title"]),
               .status = s((*o)["status"]),
               .t0 = ms((*o)["t0"]),
               .t1 = ms((*o)["t1"]),
               .line = s((*o)["line"]),
               .steps = {}};
    for (const auto& e : (*o)["steps"].toArray())
    {
        const auto k = e.toObject();
        const auto st = state(k["state"]);
        if (not st.has_value()) return std::unexpected(st.error());
        p.steps.push_back({.id = s(k["id"]), .op = s(k["op"]), .state = *st, .outcome = s(k["outcome"]),
                           .t0 = ms(k["t0"]), .t1 = ms(k["t1"]), .attempt = k["attempt"].toInt(),
                           .iteration = k["iteration"].toInt()});
    }
    return p;
}

}   // namespace rc::mission
