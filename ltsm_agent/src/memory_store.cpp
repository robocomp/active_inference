#include "memory_store.h"

#include <QDebug>
#include <QFile>
#include <QJsonDocument>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <format>
#include <fstream>
#include <system_error>
#include <vector>

namespace ltsm
{

namespace
{
constexpr std::string_view kPrefix = "ltsm_", kLayoutSuffix = "_layout.json";

/// Local wall-clock stamp YYYYMMDD_HHMMSS_mmm: sorts as time, and a human reads it as a date.
std::string stamp_now()
{
    const auto now = std::chrono::system_clock::now();
    const auto t = std::chrono::system_clock::to_time_t(now);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count() % 1000;
    std::tm tm{};
    localtime_r(&t, &tm);
    return std::format("{:04}{:02}{:02}_{:02}{:02}{:02}_{:03}", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                       tm.tm_hour, tm.tm_min, tm.tm_sec, ms);
}

/// The memory generations in `dir`, newest first: ltsm_<stamp>.json, never a sidecar or a temp file.
std::vector<std::filesystem::path> generations(const std::filesystem::path &dir)
{
    std::vector<std::filesystem::path> out;
    std::error_code ec;
    for (const auto &e : std::filesystem::directory_iterator(dir, ec))
    {
        const auto name = e.path().filename().string();
        if (e.is_regular_file(ec) and name.starts_with(kPrefix) and name.ends_with(".json")
            and not name.ends_with(kLayoutSuffix))
            out.push_back(e.path());
    }
    std::ranges::sort(out, std::greater{});
    return out;
}

/// True when the file is a whole JSON document. Checked BEFORE read_from_json_file, which inserts as it goes and so
/// would leave half a graph behind on a truncated file.
bool parses(const std::filesystem::path &p)
{
    QFile f(QString::fromStdString(p.string()));
    if (not f.open(QIODevice::ReadOnly)) return false;
    QJsonParseError err{};
    QJsonDocument::fromJson(f.readAll(), &err);
    return err.error == QJsonParseError::NoError;
}
}   // namespace

MemoryStore::MemoryStore(std::shared_ptr<DSR::DSRGraph> memory, std::filesystem::path path, bool enabled,
                         std::filesystem::path history_dir)
    : mem_(std::move(memory)), path_(std::move(path)), history_dir_(std::move(history_dir)), enabled_(enabled) {}

std::filesystem::path MemoryStore::layout_path() const
{
    if (generation_.empty()) return {};
    auto p = generation_;
    return p.replace_filename(generation_.stem().string() + std::string(kLayoutSuffix));
}

int MemoryStore::load()
{
    if (not enabled_) return 0;

    std::error_code ec;
    std::filesystem::path source;
    if (not history_dir_.empty())
    {
        // Newest first; a generation that does not parse is skipped LOUDLY, never silently replaced by an empty memory.
        for (const auto &g : generations(history_dir_))
        {
            if (parses(g)) { source = g; break; }
            qWarning() << "[memory] generation" << g.c_str() << "does NOT parse -- skipped, trying the one before it."
                       << "The file is left in place for inspection.";
        }
        if (source.empty())
            qInfo() << "[memory] no generation in" << history_dir_.c_str() << "-- trying the single file" << path_.c_str();
    }
    if (source.empty())
    {
        if (not std::filesystem::exists(path_, ec))
        {
            qInfo() << "[memory] no persisted graph at" << path_.c_str()
                    << "-- starting from the seed. This is normal on a first run.";
            return 0;
        }
        source = path_;
    }

    const auto before = static_cast<int>(mem_->size());
    mem_->read_from_json_file(source.string());       // insert_node_with_id: ids survive the round trip
    const auto after = static_cast<int>(mem_->size());
    generation_ = source;

    // ★ An alarm that should read ZERO. Node-side uint64 round-trips correctly
    // (dsr_utils.cpp:105); it was the EDGE copy of the same switch that truncated. So a stamp
    // coming back zeroed now means something regressed, not business as usual.
    int zeroed = 0, with_text = 0;
    for (const auto &n : mem_->get_nodes())
    {
        const auto ts = mem_->get_attrib_by_name<timestamp_creation_att>(n);
        if (not ts.has_value()) continue;
        if (ts.value() == 0)
        {
            ++zeroed;
            if (mem_->get_attrib_by_name<creation_datetime_att>(n).has_value()) ++with_text;
        }
    }

    qInfo() << "[memory] restored" << (after - before) << "nodes from" << source.c_str()
            << "-- graph now holds" << after;
    if (zeroed > 0)
        qWarning() << "[memory] ALARM:" << zeroed << "node(s) came back with timestamp_creation == 0."
                   << "Node-side uint64 survives this round trip (dsr_utils.cpp:105), so this is a"
                   << "regression, not the known edge-copy truncation." << with_text
                   << "of them still carry creation_datetime, which is where the birth time can be"
                   << "recovered from. Eviction is unaffected either way: it reads birth stamps off"
                   << "the LIVE graph, never from this file.";
    return after - before;
}

bool MemoryStore::save(const std::string &reason, bool force)
{
    if (not enabled_) return false;
    if (not dirty_ and not force) return false;

    std::error_code ec;
    const bool dated = not history_dir_.empty();
    const auto target = dated ? history_dir_ / (std::string(kPrefix) + stamp_now() + ".json") : path_;
    if (const auto dir = target.parent_path(); not dir.empty())
        std::filesystem::create_directories(dir, ec);

    const auto tmp  = std::filesystem::path(target).concat(".tmp");
    const auto prev = std::filesystem::path(target).concat(".prev");

    mem_->write_to_json_file(tmp.string());

    // A zero-length or missing temp file means the write failed; keep the previous generation
    // rather than renaming a truncated one over the only copy of the robot's memory.
    if (not std::filesystem::exists(tmp, ec) or std::filesystem::file_size(tmp, ec) == 0)
    {
        qWarning() << "[memory] save ABORTED (" << reason.c_str()
                   << ") -- the temp file is missing or empty; the existing memory file is untouched.";
        std::filesystem::remove(tmp, ec);
        return false;
    }

    // Single-file mode keeps one previous generation; dated mode overwrites nothing, so it needs none.
    if (not dated and std::filesystem::exists(target, ec))
    {
        std::filesystem::remove(prev, ec);
        std::filesystem::rename(target, prev, ec);    // one good generation always survives
    }
    std::filesystem::rename(tmp, target, ec);         // atomic within one filesystem
    if (ec)
    {
        qWarning() << "[memory] save FAILED (" << reason.c_str() << "):" << ec.message().c_str();
        return false;
    }

    ++saves_;
    dirty_ = false;
    generation_ = target;
    qInfo().noquote() << QString::fromStdString(std::format(
        "[memory] saved generation #{} ({}) -- {} nodes to {}{}", saves_, reason, static_cast<int>(mem_->size()),
        target.string(), dated ? "" : std::format(" (previous kept as {})", prev.string())));
    if (layout_provider_)
        if (const auto json = layout_provider_(); json.has_value()) write_layout(json.value());
    return true;
}

bool MemoryStore::write_layout(const std::string &json)
{
    if (not enabled_ or generation_.empty()) return false;
    const auto target = layout_path();
    const auto tmp = std::filesystem::path(target).concat(".tmp");
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        f.imbue(std::locale::classic());   // the JSON is already text; this only keeps the stream itself locale-free
        f << json;
        if (not f) { qWarning() << "[memory] layout sidecar write FAILED:" << tmp.c_str(); return false; }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, target, ec);
    if (ec)
    {
        qWarning() << "[memory] layout sidecar rename FAILED:" << ec.message().c_str();
        return false;
    }
    qInfo() << "[memory] layout written beside the generation:" << target.c_str();
    return true;
}

}   // namespace ltsm
