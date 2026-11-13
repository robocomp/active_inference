#pragma once
/*
 * door_memory_note.h — a door's memory as the attributes a door node (or an LTSM note) carries (2026-11-13,
 * door memory through ltsm). Pure (Eigen + std, no DSR): the translation IdentityRow <-> NoteAttrs and nothing else.
 *
 * The leaf is stored as the two probabilities of door_leaf_memory.h (P into the room, P hinge at the node's +x end)
 * plus phi / phi_sd, and rebuilt from them and the pose:
 *   - the node's x axis is the wall tangent (cos yaw, sin yaw); its normal n = (-sin yaw, cos yaw) is across_u, so
 *     open_dir = swing * n (swing +1 = into the room, as identity_of() builds it) and the encoded swing is sign(open_dir . n);
 *   - the hinge is the jamb xy +- (width/2) x; "at +x" is p_hinge_plus's MAP flag. Decoding says the node's +x end is the
 *     Far jamb (plus_is_far = true): the node's own frame is the frame of the rebuilt row, so hinge_from_plus names the jamb.
 * hinge_llr / swing_llr come back relative to the MAP side (>= 0), saturated at leafmem::logit's numerical clamp (13.8 nat).
 * removed_ms is never encoded: a removed door has no node to carry it.
 */
#include "door_identity_table.h"
#include "door_leaf_memory.h"

#include <Eigen/Dense>

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace rc::door::memnote
{
struct NoteAttrs   // exactly what a door node / note carries
{
    Eigen::Vector2f xy = Eigen::Vector2f::Zero();   ///< the RT edge room -> node (room frame)
    float yaw = 0.f;
    float width = 0.f, height = 0.f, depth = 0.f;   ///< width_m / height_m / depth_m
    std::optional<float> exist_evidence, exist_physical, place_sd, height_sd, phi, phi_sd;
    std::optional<float> leaf_into_room_prob, leaf_hinge_plus_prob;
    std::optional<bool> crossed;
    std::optional<std::uint64_t> seen_ms, sighted_ms;
    std::optional<std::string> ledger;
    std::optional<int> visit;
};

namespace detail
{
inline Eigen::Vector2f axis_x(float yaw) { return {std::cos(yaw), std::sin(yaw)}; }
inline Eigen::Vector2f axis_n(float yaw) { return {-std::sin(yaw), std::cos(yaw)}; }
inline std::optional<std::uint64_t> ms_of(const std::optional<std::int64_t>& v)
{
    return v.has_value() ? std::optional<std::uint64_t>(static_cast<std::uint64_t>(*v)) : std::nullopt;
}
inline std::optional<std::int64_t> ms_of(const std::optional<std::uint64_t>& v)
{
    return v.has_value() ? std::optional<std::int64_t>(static_cast<std::int64_t>(*v)) : std::nullopt;
}
}  // namespace detail

/// A row that has `pose` (a door node always has geometry); without it the geometry is zero.
[[nodiscard]] inline NoteAttrs note_of(const IdentityRow& r, std::optional<float> height_sd)
{
    NoteAttrs a;
    a.xy = r.xy;
    const RestorePose p = r.pose.value_or(RestorePose{});
    a.yaw = p.yaw;  a.width = p.width;  a.height = p.height;  a.depth = p.depth;
    a.exist_evidence = r.exist_logodds;
    a.exist_physical = r.exist_physical;
    a.place_sd = r.place_sd;
    a.height_sd = height_sd;
    a.crossed = r.crossed;
    a.seen_ms = detail::ms_of(r.seen_wall_ms);
    a.sighted_ms = detail::ms_of(r.sighted_ms);
    a.ledger = r.ledger;
    a.visit = r.visit;
    if (r.leaf.has_value())
    {
        const LeafMemory& L = *r.leaf;
        const float swing = detail::axis_n(p.yaw).dot(L.open_dir) >= 0.0f ? 1.0f : -1.0f;
        const bool hinge_plus = detail::axis_x(p.yaw).dot(L.hinge_xy - r.xy) >= 0.0f;
        a.leaf_into_room_prob = leafmem::p_into_room(swing, L.swing_llr);
        a.leaf_hinge_plus_prob = leafmem::p_hinge_plus(hinge_plus, L.hinge_llr);
        a.phi = L.phi;
        a.phi_sd = L.phi_sigma;
    }
    return a;
}

[[nodiscard]] inline IdentityRow row_of(const NoteAttrs& a, std::string room, std::string name)
{
    IdentityRow r;
    r.room = std::move(room);  r.name = std::move(name);
    r.xy = a.xy;
    r.pose = RestorePose{a.yaw, a.width, a.height, a.depth};
    r.exist_logodds = a.exist_evidence;
    r.exist_physical = a.exist_physical;
    r.place_sd = a.place_sd;
    r.crossed = a.crossed;
    r.seen_wall_ms = detail::ms_of(a.seen_ms);
    r.sighted_ms = detail::ms_of(a.sighted_ms);
    r.ledger = a.ledger;
    r.visit = a.visit;
    if (a.leaf_into_room_prob.has_value() or a.leaf_hinge_plus_prob.has_value())
    {
        const auto sw = leafmem::swing_from_into_room(a.leaf_into_room_prob.value_or(0.5f));
        const auto hg = leafmem::hinge_from_plus(a.leaf_hinge_plus_prob.value_or(0.5f), /*plus_is_far=*/true);
        LeafMemory L;
        L.hinge_xy = a.xy + (hg.far ? 1.0f : -1.0f) * 0.5f * a.width * detail::axis_x(a.yaw);
        L.open_dir = sw.swing * detail::axis_n(a.yaw);
        L.hinge_llr = hg.llr;  L.swing_llr = sw.llr;
        L.phi = a.phi.value_or(0.f);  L.phi_sigma = a.phi_sd.value_or(0.f);
        r.leaf = L;
    }
    return r;
}

/// The live door a remembered note stands for: "remembered_door_5" -> "door_5"; any other name is returned unchanged.
[[nodiscard]] inline std::string face_live_name(std::string_view note_name)
{
    constexpr std::string_view prefix = "remembered_";
    if (note_name.starts_with(prefix)) note_name.remove_prefix(prefix.size());
    return std::string(note_name);
}
}  // namespace rc::door::memnote
