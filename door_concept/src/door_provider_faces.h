#pragma once
/*
 * door_provider_faces.h — ONE PHYSICAL DOOR, ONE LEAF (warm34, 2026-11-02).
 *
 * THE DEFECT. door_concept holds a belief per FACE of a doorway: the same hole is `door_1` seen from room_1 and `door_24`
 * seen from room_2 (names are opaque; the faces are linked only while the crossing's twin edge lives). Both faces' actuation
 * resolves, by place, to ONE provider door ('DOOR_1'). Leg 1 opened it through door_1, crossed, and CLOSED it through
 * door_24. door_1's remembered leaf (phi 105 deg) was never told: on re-entering room_1 in leg 4 it was restored "open", the
 * mission's `open room_2` step was satisfied by a belief nobody had refreshed, and `cross` found a shut door.
 *
 * THE MODEL. The provider door id is the identity of the PHYSICAL leaf. Every accepted actuation through any face is
 * evidence about the leaf angle, and it is evidence for every other face of the same provider door:
 *   - the angle phi is measured from the shut position, a property of the leaf, the same from either side (a carried
 *     mirror already inherits it unchanged), so no frame mapping is needed for phi — only the swing SIDE and hinge are
 *     face-relative, and an actuation says nothing about them (left untouched);
 *   - the command is obeyed with probability q (kActuationObeyed) and otherwise the leaf is where it was:
 *     p(phi) = q N(target, 0) + (1-q) p_old(phi), moment-matched to the (phi, sigma) a row stores, exactly as
 *     restore_leaf_phi does for the change-point mixture. The old belief is first AGED to now (restore_leaf_phi), so an
 *     unobserved face that was already relaxing toward "unknown" is mixed from where it has got to, not from the stale value.
 *   - completion stays PERCEPTUAL: this moves a prior, it never asserts the door IS open (CROSSING contract D4).
 *
 * ⚑ MODEL CONSTANT, flagged (CLAUDE.md): kActuationObeyed = 0.9 — ASSUMED, not measured. The provider acknowledged the
 * request ("accepted"), and a person or an obstruction can still defeat it; 0.9 keeps 10 % on "the leaf did not move" so the
 * robot still looks. No model-level term yields it: it is the reliability of an actuator this agent has no history for.
 *
 * Pure (Eigen + std); tests/test_door_provider_faces.cpp.
 */
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "door_identity_table.h"

namespace rc::door
{

inline constexpr double kActuationObeyed = 0.9;

/// A face of a physical door: the room it hangs in and the aperture centre in THAT room's frame.
struct FaceRef
{
    std::string     room;
    Eigen::Vector2f xy = Eigen::Vector2f::Zero();
};

/// provider door id -> the faces (at most one per room) that have been actuated through it.
class ProviderFaces
{
public:
    /// Record that `room`'s face at `xy` is the provider door `provider`. One face per room per provider: a later note for the
    /// same room replaces the earlier (the aperture's place moves a little between fits).
    void note(const std::string& provider, const std::string& room, const Eigen::Vector2f& xy)
    {
        if (provider.empty() or room.empty() or not xy.allFinite()) return;
        auto& v = faces_[provider];
        for (auto& f : v)
            if (f.room == room) { f.xy = xy; return; }
        v.push_back({room, xy});
    }
    /// The other faces of the physical door (every recorded face not in `room`).
    [[nodiscard]] std::vector<FaceRef> others(const std::string& provider, const std::string& room) const
    {
        std::vector<FaceRef> out;
        if (const auto it = faces_.find(provider); it != faces_.end())
            for (const auto& f : it->second)
                if (f.room != room) out.push_back(f);
        return out;
    }
private:
    std::map<std::string, std::vector<FaceRef>> faces_;
};

/// The leaf belief (phi, sigma), moment-matched, after a command to `target` obeyed with probability q.
[[nodiscard]] inline std::pair<float, float> leaf_after_actuation(float phi, float sigma, float target, double q = kActuationObeyed)
{
    const double mean = q * target + (1.0 - q) * phi;
    const double m2 = q * double(target) * target + (1.0 - q) * (double(sigma) * sigma + double(phi) * phi);
    return {static_cast<float>(mean), static_cast<float>(std::sqrt(std::max(0.0, m2 - mean * mean)))};
}

/// Tell the remembered row of `face` that its physical leaf was commanded to `target` (open: the commanded angle, close: 0).
/// The row's leaf is aged to `now_ms` first, then mixed with the command; the stamp moves to now so the restore does not age it
/// twice, and the existence belief — which shares the stamp — is aged to now with the PLACE TABLE's hazard (PlaceTable::hazard_per_s,
/// learned from revisits: `existence_hazard_per_s`) so it loses nothing. The leaf keeps its own (leaf-channel) hazard.
/// Returns whether a row with a leaf memory on that hole was found and updated.
inline bool apply_actuation_to_face(std::vector<IdentityRow>& rows, const FaceRef& face, float target, float phi_max,
                                    std::int64_t now_ms, float reacquire_radius_m, double existence_hazard_per_s,
                                    double q = kActuationObeyed)
{
    bool any = false;
    for (auto& r : rows)
    {
        const float same_hole = reacquire_radius_m + (r.pose.has_value() ? r.pose->width : 0.f);
        if (r.room != face.room or not r.leaf.has_value() or (r.xy - face.xy).norm() >= same_hole) continue;
        const auto [pa, sa] = restore_leaf_phi(r.leaf->phi, r.leaf->phi_sigma, r.seen_wall_ms, now_ms, phi_max);
        const auto [pn, sn] = leaf_after_actuation(pa, sa, target, q);
        if (r.exist_logodds.has_value() and r.seen_wall_ms.has_value())   // age the evidence to now (toward 0 = ignorance) before the stamp moves
        {
            const double gamma = std::exp(-existence_hazard_per_s * std::max<double>(0.0, double(now_ms - *r.seen_wall_ms) * 1e-3));
            r.exist_logodds = float(double(*r.exist_logodds) * gamma);
            if (r.exist_physical.has_value()) r.exist_physical = float(double(*r.exist_physical) * gamma);
        }
        if (not r.sighted_ms.has_value()) r.sighted_ms = r.seen_wall_ms;   // the command is no sighting: the hazard's exposure keeps counting from the last one
        r.leaf->phi = pn;
        r.leaf->phi_sigma = sn;
        r.seen_wall_ms = now_ms;
        any = true;
    }
    return any;
}

}   // namespace rc::door
