// The window onto the long-term object memory: the repertory projected onto ltom's own DSR graph (DDS domain 5), shown in
// the standard graph viewer. root -> room_<key> -> one `object` node per retained object, at its newest keyframe.
// The repertory file stays the store of record; nothing reads this graph back. Main thread only.
#pragma once
#include "repertory.h"

#include <dsr/api/dsr_api.h>

#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>

namespace ltom
{
// What a click on the object's node shows of its past: sightings, keyframes, one line per life.
std::string history_text(const Entry& e);

class MemoryGraph
{
public:
    explicit MemoryGraph(std::shared_ptr<DSR::DSRGraph> G);
    // Make the graph show every entry. `present` = the uids on the live graph right now (`state`: live or remembered).
    void sync(const std::map<std::uint64_t, Entry>& entries, const std::set<std::uint64_t>& present);

private:
    struct Written { std::string room, state, history; Keyframe kf; bool has_kf = false; };

    std::shared_ptr<DSR::DSRGraph> G_;
    std::unique_ptr<DSR::RT_API> rt_api_;
    std::map<std::uint64_t, Written> written_;
    std::set<std::string> warned_;

    void warn_once(const std::string& name, const char* what);
    std::uint64_t ensure_room(std::int64_t key);
    void show(const Entry& e, std::uint64_t room_id, const std::string& room, const std::string& state);
    void write_pose(std::uint64_t room_id, std::uint64_t id, const std::string& name, const Keyframe& kf);
};
}
