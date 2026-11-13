#include "recall_select.h"
#include <algorithm>

namespace ltom
{
std::vector<Recall> select_recalls(const Repertory& r, std::int64_t room_key,
                                   const std::set<std::uint64_t>& present, std::span<const std::string> classes)
{
    std::vector<Recall> out;
    for (const auto& [uid, e] : r.entries())      // std::map: already ordered by uid
    {
        if (e.lives.empty() or e.keyframes.empty()) continue;
        if (e.lives.back().end == End::alive or present.contains(uid)) continue;
        if (std::ranges::find(classes, e.cls) == classes.end()) continue;
        const auto last = std::ranges::max_element(e.keyframes, {}, &Keyframe::t_ms);
        if (last->room_key != room_key) continue;
        out.push_back(Recall{uid, e.cls, *last, e.exist_prob});
    }
    return out;
}
}
