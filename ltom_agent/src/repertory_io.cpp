#include "repertory_io.h"

#include <array>
#include <QByteArray>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QString>

#include <charconv>
#include <chrono>
#include <cmath>
#include <fstream>
#include <locale>
#include <print>
#include <system_error>

namespace fs = std::filesystem;

namespace ltom
{
namespace
{
const char* end_name(End e)
{
    switch (e)
    {
        case End::alive: return "alive";
        case End::death: return "death";
        case End::owner_exit: return "owner_exit";
        case End::gap: return "gap";
    }
    return "alive";
}

bool end_from(const QString& s, End& out)
{
    for (End e : {End::alive, End::death, End::owner_exit, End::gap})
        if (s == end_name(e)) { out = e; return true; }
    return false;
}

QJsonObject to_json(const Entry& e)
{
    QJsonObject o;
    o["uid"] = QString::number(e.uid);                 // decimal string: a double loses uint64 precision
    o["cls"] = QString::fromStdString(e.cls);
    o["name"] = QString::fromStdString(e.name);
    o["w"] = static_cast<double>(e.w);
    o["d"] = static_cast<double>(e.d);
    o["h"] = static_cast<double>(e.h);
    o["n_sightings"] = static_cast<double>(e.n_sightings);
    QJsonArray lives;
    for (const Life& l : e.lives)
    {
        QJsonObject j;
        j["born_ms"] = static_cast<double>(l.born_ms);
        j["first_seen_ms"] = static_cast<double>(l.first_seen_ms);
        j["last_seen_ms"] = static_cast<double>(l.last_seen_ms);
        j["end_ms"] = static_cast<double>(l.end_ms);
        j["end"] = end_name(l.end);
        lives.append(j);
    }
    o["lives"] = lives;
    QJsonArray kfs;
    for (const Keyframe& k : e.keyframes)
    {
        QJsonObject j;
        j["t_ms"] = static_cast<double>(k.t_ms);
        j["room_key"] = static_cast<double>(k.room_key);
        j["x"] = k.pose.x; j["y"] = k.pose.y; j["yaw"] = k.pose.yaw;
        j["vx"] = k.pose.vx; j["vy"] = k.pose.vy; j["vyaw"] = k.pose.vyaw;
        kfs.append(j);
    }
    o["keyframes"] = kfs;
    return o;
}

bool parse_uid(const QString& s, std::uint64_t& out)
{
    const std::string b = s.toStdString();
    const char* first = b.data();
    const char* last = first + b.size();
    const auto [p, ec] = std::from_chars(first, last, out);
    return ec == std::errc{} and p == last and first != last;
}

bool from_json(const QJsonObject& o, Entry& e)
{
    if (not parse_uid(o["uid"].toString(), e.uid)) return false;
    e.cls = o["cls"].toString().toStdString();
    e.name = o["name"].toString().toStdString();
    e.w = static_cast<float>(o["w"].toDouble());
    e.d = static_cast<float>(o["d"].toDouble());
    e.h = static_cast<float>(o["h"].toDouble());
    e.n_sightings = static_cast<std::int64_t>(o["n_sightings"].toDouble());
    for (const QJsonValue& v : o["lives"].toArray())
    {
        const QJsonObject j = v.toObject();
        Life l;
        l.born_ms = static_cast<std::int64_t>(j["born_ms"].toDouble());
        l.first_seen_ms = static_cast<std::int64_t>(j["first_seen_ms"].toDouble());
        l.last_seen_ms = static_cast<std::int64_t>(j["last_seen_ms"].toDouble());
        l.end_ms = static_cast<std::int64_t>(j["end_ms"].toDouble(-1));
        if (not end_from(j["end"].toString(), l.end)) return false;
        e.lives.push_back(l);
    }
    for (const QJsonValue& v : o["keyframes"].toArray())
    {
        const QJsonObject j = v.toObject();
        Keyframe k;
        k.t_ms = static_cast<std::int64_t>(j["t_ms"].toDouble());
        k.room_key = static_cast<std::int64_t>(j["room_key"].toDouble(-1));
        k.pose = Pose2{j["x"].toDouble(), j["y"].toDouble(), j["yaw"].toDouble(),
                       j["vx"].toDouble(), j["vy"].toDouble(), j["vyaw"].toDouble()};
        e.keyframes.push_back(k);
    }
    return true;
}

// Parses one file into a scratch list; nothing is inserted unless the whole file is good.
bool try_parse(const fs::path& file, std::vector<Entry>& out)
{
    std::ifstream f(file, std::ios::binary);
    if (not f) return false;
    const std::string text{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    QJsonParseError err;
    const QJsonDocument doc = QJsonDocument::fromJson(QByteArray(text.data(), static_cast<qsizetype>(text.size())), &err);
    if (err.error != QJsonParseError::NoError or not doc.isObject()) return false;
    const QJsonObject root = doc.object();
    if (root["version"].toInt(-1) != 1 or not root["entries"].isArray()) return false;
    for (const QJsonValue& v : root["entries"].toArray())
    {
        Entry e;
        if (not from_json(v.toObject(), e)) return false;
        out.push_back(std::move(e));
    }
    return true;
}

std::string fmt_double(double v)
{
    char buf[40];
    const auto r = std::to_chars(buf, buf + sizeof buf, v);
    return std::string(buf, r.ptr);
}
}

LoadResult load_repertory(const fs::path& file, Repertory& into)
{
    std::error_code ec;
    fs::path prev = file; prev += ".prev";
    const bool have_main = fs::exists(file, ec);
    const bool have_prev = fs::exists(prev, ec);
    if (not have_main and not have_prev) return LoadResult::missing;

    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    // A bad file is never left where the next save would rotate it over a good copy.
    bool blocked = false;
    auto move_aside = [&](const fs::path& p)
    {
        fs::path bad = p; bad += ".bad." + std::to_string(ms);
        fs::rename(p, bad, ec);
        if (ec)
        {
            blocked = true;
            std::println(stderr, "⚠⚠ ltom: CANNOT move bad {} aside ({}): saving is disabled so it cannot clobber the good copy",
                         p.string(), ec.message());
        }
        return bad;
    };

    std::vector<Entry> got;
    if (have_main and try_parse(file, got))
    {
        for (Entry& e : got) into.insert_loaded(std::move(e));
        return LoadResult::loaded;
    }
    got.clear();
    if (have_prev and try_parse(prev, got))
    {
        std::println(stderr, "⚠ ltom: {} unreadable or missing, falling back to {}", file.string(), prev.string());
        if (have_main)
        {
            const auto bad = move_aside(file);
            if (not blocked) std::println(stderr, "⚠ ltom: bad main moved aside to {}", bad.string());
        }
        for (Entry& e : got) into.insert_loaded(std::move(e));
        return blocked ? LoadResult::blocked : LoadResult::loaded_prev;
    }
    for (const fs::path* p : std::array<const fs::path*, 2>{&file, &prev})
        if (fs::exists(*p, ec))
        {
            const auto bad = move_aside(*p);
            if (not blocked) std::println(stderr, "⚠⚠ ltom: unreadable {} moved aside to {}", p->string(), bad.string());
        }
    std::println(stderr, "⚠⚠ ltom: starting EMPTY");
    return blocked ? LoadResult::blocked : LoadResult::unreadable;
}

bool save_repertory(const fs::path& file, const Repertory& r)
{
    QJsonArray entries;
    for (const auto& [uid, e] : r.entries()) entries.append(to_json(e));
    QJsonObject root;
    root["version"] = 1;
    root["entries"] = entries;
    const QByteArray bytes = QJsonDocument(root).toJson(QJsonDocument::Indented);

    fs::path tmp = file; tmp += ".tmp";
    fs::path prev = file; prev += ".prev";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f.imbue(std::locale::classic());
        f.write(bytes.constData(), bytes.size());
        f.flush();
        if (not f) { std::println(stderr, "⚠ ltom: cannot write {}", tmp.string()); return false; }
    }
    std::error_code ec;
    if (fs::exists(file, ec))
    {
        fs::rename(file, prev, ec);
        if (ec) { std::println(stderr, "⚠ ltom: cannot rotate {} to .prev: {}", file.string(), ec.message()); return false; }
    }
    fs::rename(tmp, file, ec);
    if (ec) { std::println(stderr, "⚠ ltom: cannot rename {} into place: {}", tmp.string(), ec.message()); return false; }
    return true;
}

bool write_csv(const fs::path& file, const Repertory& r)
{
    std::ofstream f(file, std::ios::trunc);
    f.imbue(std::locale::classic());
    f << "uid,owner,class,name,lives,state,born_ms,first_seen_ms,last_seen_ms,end_ms,room_key,x,y,yaw,sx,sy,w,d,h,n_sightings,n_keyframes\n";
    for (const auto& [uid, e] : r.entries())
    {
        Life l; Keyframe k;
        if (not e.lives.empty()) l = e.lives.back();
        if (not e.keyframes.empty()) k = e.keyframes.back();
        f << uid << ',' << owner_of(uid) << ',' << e.cls << ',' << e.name << ',' << e.lives.size() << ','
          << end_name(l.end) << ',' << l.born_ms << ',' << l.first_seen_ms << ',' << l.last_seen_ms << ',' << l.end_ms << ','
          << k.room_key << ',' << fmt_double(k.pose.x) << ',' << fmt_double(k.pose.y) << ',' << fmt_double(k.pose.yaw) << ','
          << fmt_double(std::sqrt(k.pose.vx)) << ',' << fmt_double(std::sqrt(k.pose.vy)) << ','
          << fmt_double(e.w) << ',' << fmt_double(e.d) << ',' << fmt_double(e.h) << ','
          << e.n_sightings << ',' << e.keyframes.size() << '\n';
    }
    f.flush();
    return static_cast<bool>(f);
}
}
