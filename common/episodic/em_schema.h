#pragma once
/*
 * em_schema.h — the episodic memory's record (2026-10-17): one entry per mission, step or affordance claim,
 * and its JSON form. Header-only, Qt Core only, no DSR.
 *
 * ABSENT IS NOT ZERO: an unread fact, an unset promise, a still-open record are `std::optional` / a missing key,
 * never a defaulted 0. to_json omits them and record_from_json leaves them empty.
 *
 * Fact.values keys (ruling R2): the attribute names in kFactAttrs, plus "x","y","yaw" (pose in the parent room
 * frame, metres / rad) and "cov_xx","cov_yy" (the rt_covariance xy diagonal). Only attributes actually read are
 * present. A claim's delivered = 0.5*ln((cov_xx_pre*cov_yy_pre)/(cov_xx_post*cov_yy_post)).
 *
 * JSON keys are exactly the field names. Numbers go through QJsonValue (locale-independent; CLAUDE.md).
 */

#include <array>
#include <chrono>
#include <cstdint>
#include <format>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QString>

namespace rc::em
{

enum class Level { Mission, Step, Claim };      // "mission" | "step" | "claim"  (reserved: "arm")
enum class Author { Human, Self };              // "human" | "self"

struct Fact
{
    std::string node, type, ltsm_id;                        // ltsm_id empty = none
    std::map<std::string, double, std::less<>> values;      // only attrs actually read
    bool operator==(const Fact&) const = default;
};

struct Event
{
    std::int64_t dt_ms = 0;                                 // since the record's t0
    std::string kind, node;                                 // kind: born|died|current_moved|offered|withdrawn
    bool operator==(const Event&) const = default;
};

/// THE SCENE (2026-10-18): what the robot believes is there when an action starts and once its consequences settled —
/// the context a world model conditions on. things = names of the live nodes of kEventNodeTypes except affordances
/// (rooms, doors, objects...); offered = names of the affordance nodes present. Both sorted. Names, not types: the
/// type is the name's stem ("door_5" → door), and the instance keeps identity for whoever needs it.
/// An RT edge in the scene (2026-10-23): robot<->room and room<-parent only, as stored (from -> to). t = (x,y,z),
/// r = euler xyz; cov = the 3x3 over [x,y,yaw] row-major (9 values, from the 6x6 rt_covariance entries
/// [0,1,5; 6,7,11; 30,31,35]); empty when the edge carries no (or a short) covariance. ABSENT IS NOT ZERO.
struct RtEdge
{
    std::string from, from_type, to, to_type;
    std::vector<double> t, r, cov;
    bool operator==(const RtEdge&) const = default;
};

struct Scene
{
    std::vector<std::string> things, offered;
    std::vector<RtEdge> rt;                                 // omitted from JSON when empty (old records unchanged)
    bool operator==(const Scene&) const = default;
};

struct Record
{
    std::string id, parent;                     // id "<YYYYMMDD>-<seq>", parent "" for a mission
    Level level = Level::Mission;
    Author author = Author::Human;
    std::string kind, target, target_ltsm;      // kind = op / affordance kind / "mission"; target = live node name
    std::string action_json;                    // IR / command text / claim contract, verbatim JSON
    std::string source;                         // mission only: chat|remote|panel|self
    std::uint64_t t0 = 0;
    std::optional<std::uint64_t> t1;            // empty = still open
    std::string room;                           // current room at t0
    std::vector<Fact> pre, post;
    std::string outcome;                        // "" while open
    int attempt = 0, iteration = 0;             // step only
    std::optional<double> promised, delivered;  // claim only
    std::string alternatives_json;              // claim only: selector candidates, verbatim JSON
    std::vector<Event> events;
    // Back-references into the controller's journal (em_journal.h): its mission key ("m12" / "c3"; "" for a self
    // mission) and the serials of the entries that opened / closed this record — ABSENT when EM itself opened or
    // closed it (a self mission, `lost`, `interrupted_by_em_shutdown`).
    std::string mission_key;
    std::string step_id;                        // step only: the IR step id ("2.1"; single commands "1")
    std::optional<std::uint64_t> serial_start, serial_end;
    // THE STATE AFTER (2026-10-18). EM stores the consequences of applying affordances, and they settle AFTER the close:
    // door_concept raises a door belief a cycle after the open claim closes, the current room switches ~0.2 s after a
    // crossing step closes. So post, room_after and delivered are read at the close AND again kSettleMs later
    // (em_builder.h), and the record is re-emitted with settled = true. t_post = when post / room_after were read
    // (absent on records older than this field); room_after "" = unknown. A record EM closed at its own shutdown is
    // never settled.
    std::string room_after;
    std::optional<std::uint64_t> t_post;
    bool settled = false;
    // Steps and claims only: the scene at open (scene_pre) and with the state after (scene_post, re-read at settling).
    // Absent = not read (missions, records older than this field, a FactSource without a scene).
    std::optional<Scene> scene_pre, scene_post;
    bool operator==(const Record&) const = default;
};

inline constexpr std::array kFactAttrs{"door_open_prob", "door_reach_prob", "door_crossing_progress",
                                       "obj_exist_prob", "passage_passable"};   // + pose x,y,yaw + rt_covariance diag
inline constexpr std::array kEventNodeTypes{"room", "door", "object", "person", "table", "chair", "bottle",
                                            "refrigerator", "box", "affordance"};

/// "YYYYMMDD", the local date of a wall-clock instant (ms since epoch).
inline std::string day_of(std::uint64_t t_ms)
{
    using namespace std::chrono;
    const sys_time<milliseconds> t{milliseconds{std::int64_t(t_ms)}};
    const year_month_day ymd{floor<days>(zoned_time{current_zone(), t}.get_local_time())};
    return std::format("{:04}{:02}{:02}", int(ymd.year()), unsigned(ymd.month()), unsigned(ymd.day()));
}

namespace detail
{
inline QString q(const std::string& s) { return QString::fromUtf8(s.data(), qsizetype(s.size())); }
inline std::string s(const QJsonValue& v) { return v.toString().toStdString(); }

inline const char* name(Level l) { return l == Level::Mission ? "mission" : l == Level::Step ? "step" : "claim"; }

inline QJsonObject to_json(const Fact& f)
{
    QJsonObject o{{"node", q(f.node)}, {"type", q(f.type)}};
    if (not f.ltsm_id.empty()) o["ltsm_id"] = q(f.ltsm_id);
    QJsonObject v;
    for (const auto& [k, x] : f.values) v[q(k)] = x;
    o["values"] = v;
    return o;
}

inline Fact fact_from_json(const QJsonObject& o)
{
    Fact f{.node = s(o["node"]), .type = s(o["type"]), .ltsm_id = s(o["ltsm_id"]), .values = {}};
    const QJsonObject v = o["values"].toObject();
    for (auto it = v.begin(); it != v.end(); ++it)
        if (it.value().isDouble()) f.values[it.key().toStdString()] = it.value().toDouble();
    return f;
}

inline QJsonArray facts_json(const std::vector<Fact>& fs)
{
    QJsonArray a;
    for (const Fact& f : fs) a.append(to_json(f));
    return a;
}

inline QJsonArray strings_json(const std::vector<std::string>& v)
{
    QJsonArray a;
    for (const std::string& x : v) a.append(q(x));
    return a;
}

inline std::vector<std::string> strings_from(const QJsonValue& v)
{
    std::vector<std::string> out;
    for (const QJsonValue& e : v.toArray()) out.push_back(s(e));
    return out;
}

inline QJsonArray doubles_json(const std::vector<double>& v)
{
    QJsonArray a;
    for (const double x : v) a.append(x);
    return a;
}

inline std::vector<double> doubles_from(const QJsonValue& v)
{
    std::vector<double> out;
    for (const QJsonValue& e : v.toArray()) out.push_back(e.toDouble());
    return out;
}

inline QJsonObject to_json(const Scene& sc)
{
    QJsonObject o{{"things", strings_json(sc.things)}, {"offered", strings_json(sc.offered)}};
    if (not sc.rt.empty())
    {
        QJsonArray a;
        for (const RtEdge& e : sc.rt)
            a.append(QJsonObject{{"from", q(e.from)}, {"from_type", q(e.from_type)}, {"to", q(e.to)},
                                 {"to_type", q(e.to_type)}, {"t", doubles_json(e.t)}, {"r", doubles_json(e.r)},
                                 {"cov", doubles_json(e.cov)}});
        o["rt"] = a;
    }
    return o;
}

inline std::optional<Scene> scene_from(const QJsonValue& v)
{
    if (not v.isObject()) return std::nullopt;
    const QJsonObject o = v.toObject();
    Scene sc{.things = strings_from(o["things"]), .offered = strings_from(o["offered"]), .rt = {}};
    for (const QJsonValue& v2 : o["rt"].toArray())
    {
        const QJsonObject e = v2.toObject();
        sc.rt.push_back(RtEdge{.from = s(e["from"]), .from_type = s(e["from_type"]), .to = s(e["to"]),
                               .to_type = s(e["to_type"]), .t = doubles_from(e["t"]), .r = doubles_from(e["r"]),
                               .cov = doubles_from(e["cov"])});
    }
    return sc;
}

inline std::vector<Fact> facts_from(const QJsonValue& v)
{
    std::vector<Fact> out;
    for (const QJsonValue& e : v.toArray()) out.push_back(fact_from_json(e.toObject()));
    return out;
}
}   // namespace detail

inline QJsonObject to_json(const Record& r)
{
    using namespace detail;
    QJsonObject o;
    o["id"] = q(r.id);
    o["parent"] = q(r.parent);
    o["level"] = name(r.level);
    o["author"] = r.author == Author::Human ? "human" : "self";
    o["kind"] = q(r.kind);
    o["target"] = q(r.target);
    o["target_ltsm"] = q(r.target_ltsm);
    o["action_json"] = q(r.action_json);
    o["source"] = q(r.source);
    o["t0"] = double(r.t0);
    if (r.t1) o["t1"] = double(*r.t1);
    o["room"] = q(r.room);
    o["pre"] = facts_json(r.pre);
    o["post"] = facts_json(r.post);
    o["outcome"] = q(r.outcome);
    o["attempt"] = r.attempt;
    o["iteration"] = r.iteration;
    if (r.promised) o["promised"] = *r.promised;
    if (r.delivered) o["delivered"] = *r.delivered;
    o["alternatives_json"] = q(r.alternatives_json);
    QJsonArray ev;
    for (const Event& e : r.events)
        ev.append(QJsonObject{{"dt_ms", double(e.dt_ms)}, {"kind", q(e.kind)}, {"node", q(e.node)}});
    o["events"] = ev;
    o["mission_key"] = q(r.mission_key);
    o["step_id"] = q(r.step_id);
    if (r.serial_start) o["serial_start"] = double(*r.serial_start);
    if (r.serial_end) o["serial_end"] = double(*r.serial_end);
    if (not r.room_after.empty()) o["room_after"] = q(r.room_after);
    if (r.t_post) o["t_post"] = double(*r.t_post);
    if (r.settled) o["settled"] = true;
    if (r.scene_pre) o["scene_pre"] = to_json(*r.scene_pre);
    if (r.scene_post) o["scene_post"] = to_json(*r.scene_post);
    return o;
}

/// nullopt when level or author is not one of the known spellings (a reserved/unknown level is refused, not guessed).
inline std::optional<Record> record_from_json(const QJsonObject& o)
{
    using namespace detail;
    Record r;
    const std::string lv = s(o["level"]), au = s(o["author"]);
    if (lv == "mission") r.level = Level::Mission;
    else if (lv == "step") r.level = Level::Step;
    else if (lv == "claim") r.level = Level::Claim;
    else return std::nullopt;
    if (au == "human") r.author = Author::Human;
    else if (au == "self") r.author = Author::Self;
    else return std::nullopt;
    r.id = s(o["id"]);
    r.parent = s(o["parent"]);
    r.kind = s(o["kind"]);
    r.target = s(o["target"]);
    r.target_ltsm = s(o["target_ltsm"]);
    r.action_json = s(o["action_json"]);
    r.source = s(o["source"]);
    r.t0 = std::uint64_t(o["t0"].toDouble());
    if (o["t1"].isDouble()) r.t1 = std::uint64_t(o["t1"].toDouble());
    r.room = s(o["room"]);
    r.pre = facts_from(o["pre"]);
    r.post = facts_from(o["post"]);
    r.outcome = s(o["outcome"]);
    r.attempt = o["attempt"].toInt();
    r.iteration = o["iteration"].toInt();
    if (o["promised"].isDouble()) r.promised = o["promised"].toDouble();
    if (o["delivered"].isDouble()) r.delivered = o["delivered"].toDouble();
    r.alternatives_json = s(o["alternatives_json"]);
    for (const QJsonValue& v : o["events"].toArray())
    {
        const QJsonObject e = v.toObject();
        r.events.push_back(Event{.dt_ms = std::int64_t(e["dt_ms"].toDouble()), .kind = s(e["kind"]), .node = s(e["node"])});
    }
    r.mission_key = s(o["mission_key"]);
    r.step_id = s(o["step_id"]);
    if (o["serial_start"].isDouble()) r.serial_start = std::uint64_t(o["serial_start"].toDouble());
    if (o["serial_end"].isDouble()) r.serial_end = std::uint64_t(o["serial_end"].toDouble());
    r.room_after = s(o["room_after"]);
    if (o["t_post"].isDouble()) r.t_post = std::uint64_t(o["t_post"].toDouble());
    r.settled = o["settled"].toBool(false);
    r.scene_pre = scene_from(o["scene_pre"]);
    r.scene_post = scene_from(o["scene_post"]);
    return r;
}

/// All-time statistics of the closed records sharing a (level, kind, author, identity) key (see em_consolidate.h).
/// level = "mission"|"step"|"claim" (a step op and a claim kind may share a name); author = "human"|"self" for
/// missions, "" otherwise; identity = target_ltsm, "" = per-kind only. n counts every record (outcomes too); dur_* is
/// Welford over t1-t0 in seconds over the n_dur records whose t1 is the commitment's own end — `lost` and
/// `interrupted_by_em_shutdown` are EM's stop time, not a duration. n_gain counts only records carrying BOTH promised
/// and delivered (absent is not zero).
struct Stats
{
    std::string key_level, key_kind, key_author, key_identity;
    std::map<std::string, int, std::less<>> outcomes;       // outcome word -> count
    int n = 0, n_dur = 0;
    double dur_mean_s = 0, dur_m2 = 0;
    int n_gain = 0;
    double promised_sum = 0, delivered_sum = 0;
    std::uint64_t first_t0 = 0, last_t0 = 0;
    bool operator==(const Stats&) const = default;
};

inline QJsonObject to_json(const Stats& st)
{
    using namespace detail;
    QJsonObject o;
    o["key_level"] = q(st.key_level);
    o["key_kind"] = q(st.key_kind);
    o["key_author"] = q(st.key_author);
    o["key_identity"] = q(st.key_identity);
    QJsonObject oc;
    for (const auto& [k, c] : st.outcomes) oc[q(k)] = c;
    o["outcomes"] = oc;
    o["n"] = st.n;
    o["n_dur"] = st.n_dur;
    o["dur_mean_s"] = st.dur_mean_s;
    o["dur_m2"] = st.dur_m2;
    o["n_gain"] = st.n_gain;
    o["promised_sum"] = st.promised_sum;
    o["delivered_sum"] = st.delivered_sum;
    o["first_t0"] = double(st.first_t0);
    o["last_t0"] = double(st.last_t0);
    return o;
}

/// nullopt when the mandatory key_kind is missing.
inline std::optional<Stats> stats_from_json(const QJsonObject& o)
{
    using namespace detail;
    if (not o["key_kind"].isString()) return std::nullopt;
    Stats st;
    st.key_level = s(o["key_level"]);
    st.key_kind = s(o["key_kind"]);
    st.key_author = s(o["key_author"]);
    st.key_identity = s(o["key_identity"]);
    const QJsonObject oc = o["outcomes"].toObject();
    for (auto it = oc.begin(); it != oc.end(); ++it) st.outcomes[it.key().toStdString()] = it.value().toInt();
    st.n = o["n"].toInt();
    st.n_dur = o["n_dur"].toInt();
    st.dur_mean_s = o["dur_mean_s"].toDouble();
    st.dur_m2 = o["dur_m2"].toDouble();
    st.n_gain = o["n_gain"].toInt();
    st.promised_sum = o["promised_sum"].toDouble();
    st.delivered_sum = o["delivered_sum"].toDouble();
    st.first_t0 = std::uint64_t(o["first_t0"].toDouble());
    st.last_t0 = std::uint64_t(o["last_t0"].toDouble());
    return st;
}

}   // namespace rc::em
