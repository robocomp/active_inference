#pragma once
/*
 * place_words.h — where a point in a room is, in words, for the changed-object question ("by door_25", "in the
 * corner", "near the middle of the room"). The NEAREST landmark by distance; no cutoff (spec 2026-11-15-changed-object).
 * Pure, header-only: room frame, metres.
 */

#include <cmath>
#include <limits>
#include <span>
#include <string>
#include <utility>

namespace rc::chat
{
struct P2 { float x = 0.f, y = 0.f; };

[[nodiscard]] inline std::string place_words(P2 xy, std::span<const P2> room_polygon,
                                             std::span<const std::pair<std::string, P2>> doors)
{
    const auto dist = [&](P2 p) { return std::hypot(p.x - xy.x, p.y - xy.y); };
    P2 centre{};
    for (const P2& v : room_polygon) { centre.x += v.x; centre.y += v.y; }
    if (not room_polygon.empty())
    {
        centre.x /= static_cast<float>(room_polygon.size());
        centre.y /= static_cast<float>(room_polygon.size());
    }
    std::string best = "near the middle of the room";
    float best_d = room_polygon.empty() ? std::numeric_limits<float>::infinity() : dist(centre);
    for (const auto& [name, p] : doors)
        if (const float d = dist(p); d < best_d) { best_d = d; best = "by " + name; }
    for (const P2& v : room_polygon)
        if (const float d = dist(v); d < best_d) { best_d = d; best = "in the corner"; }
    return best;
}
}   // namespace rc::chat
