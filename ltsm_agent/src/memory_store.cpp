#include "memory_store.h"

#include <QDebug>

#include <format>
#include <system_error>

namespace ltsm
{

MemoryStore::MemoryStore(std::shared_ptr<DSR::DSRGraph> memory, std::filesystem::path path, bool enabled)
    : mem_(std::move(memory)), path_(std::move(path)), enabled_(enabled) {}

int MemoryStore::load()
{
    if (not enabled_) return 0;

    std::error_code ec;
    if (not std::filesystem::exists(path_, ec))
    {
        qInfo() << "[memory] no persisted graph at" << path_.c_str()
                << "-- starting from the seed. This is normal on a first run.";
        return 0;
    }

    const auto before = static_cast<int>(mem_->size());
    mem_->read_from_json_file(path_.string());       // insert_node_with_id: ids survive the round trip
    const auto after = static_cast<int>(mem_->size());

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

    qInfo() << "[memory] restored" << (after - before) << "nodes from" << path_.c_str()
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
    if (const auto dir = path_.parent_path(); not dir.empty())
        std::filesystem::create_directories(dir, ec);

    const auto tmp  = std::filesystem::path(path_).concat(".tmp");
    const auto prev = std::filesystem::path(path_).concat(".prev");

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

    if (std::filesystem::exists(path_, ec))
    {
        std::filesystem::remove(prev, ec);
        std::filesystem::rename(path_, prev, ec);    // one good generation always survives
    }
    std::filesystem::rename(tmp, path_, ec);         // atomic within one filesystem
    if (ec)
    {
        qWarning() << "[memory] save FAILED (" << reason.c_str() << "):" << ec.message().c_str();
        return false;
    }

    ++saves_;
    dirty_ = false;
    qInfo().noquote() << QString::fromStdString(std::format(
        "[memory] saved generation #{} ({}) -- {} nodes to {} (previous kept as {})",
        saves_, reason, static_cast<int>(mem_->size()), path_.string(), prev.string()));
    return true;
}

}   // namespace ltsm
