/*
 * door_world_registration.h — learn room→world from doors the operator has already identified.
 *
 * ★THE PROBLEM. DoorControl offers two ways to say which door is meant: quote an id the provider
 * advertised, or send a pose and let it match by PLACE. Only the second works for an autonomous robot —
 * nobody is standing there to pick "DOOR_1" from a list when the robot decides on its own to open a
 * door. But place-matching needs our frame and the provider's world frame registered to each other, and
 * measured 2026-09-09 they are not: door_3 at room (-0.431, 4.610) m is the same physical door as
 * DOOR_1 at world (-4.769, 9.380) m — 6.45 m apart against the provider's 400 mm gate, and the room
 * frame turns out to be rotated ~90.3 deg from the world frame, so no translation-only guess and no
 * widening of the gate could ever have closed it.
 *
 * ★THE SOURCE OF TRUTH IS THE OPERATOR, AND IT IS FREE. Every time somebody picks an advertised id and
 * the provider ACCEPTS, they have asserted a correspondence: "the door I believe here is the one you
 * call DOOR_1". That is a labelled pair of poses in the two frames, produced as a side effect of using
 * the button. Two such pairs determine a 2-D rigid transform outright. So the registration is learned
 * from ordinary use rather than from a calibration procedure nobody will run.
 *
 * ★AND IT IS CHECKABLE. With three or more pairs the fit is over-determined and the residual says
 * whether the correspondences are consistent. A pair recorded against the wrong door shows up as metres
 * of residual, not as a quietly-wrong transform — which is the failure mode that matters, since a wrong
 * registration would send a request that opens SOME OTHER door.
 *
 * ⚠A single pair is NOT enough on its own. It fixes position but leaves rotation to be inferred from the
 * two yaws, and the two sides do not necessarily mean the same thing by "the door's angle" (ours is the
 * aperture normal; the provider's is a Webots rotation field). One-pair fits are therefore marked
 * PROVISIONAL and reported as such, never silently used as though they were measured.
 *
 * Persisted so the registration survives a restart; the correspondences are facts about this apartment,
 * not about this session.
 */

#pragma once

#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Dense>

namespace rc
{

class DoorWorldRegistration
{
public:
    struct Pair
    {
        Eigen::Vector2f room_m;    // our fitted aperture centre, room frame, metres
        Eigen::Vector2f world_m;   // the provider's pose for the door the operator named, metres
        std::string     our_name;  // door_3
        std::string     their_id;  // DOOR_1
        // ★BOTH YAWS ARE PART OF THE PAIR, and they are persisted. A one-pair fit can only recover
        // rotation by comparing them, so leaving them out of the file meant a restart rebuilt the fit
        // with rotation ASSUMED ZERO — against a frame rotated 90 degrees, which puts the door metres
        // away. NaN = not recorded (an older file), and a one-pair fit is then simply unusable rather
        // than silently wrong.
        float yaw_room  = std::numeric_limits<float>::quiet_NaN();
        float yaw_world = std::numeric_limits<float>::quiet_NaN();
    };

    struct Fit
    {
        bool  valid       = false;
        bool  provisional = false;   // one pair only: rotation inferred from yaws, not measured
        float theta       = 0.0f;    // room→world rotation (rad)
        Eigen::Vector2f t = Eigen::Vector2f::Zero();   // room→world translation (m)
        float residual_m  = 0.0f;    // mean |world - (R*room + t)| over the pairs; 0 when n < 3
        int   n_pairs     = 0;
    };

    void load(const std::string& path);

    // Record a correspondence the operator just asserted by naming a provider door. Replaces any earlier
    // pair for the same provider id — the newest fit of the same door is the better estimate of it.
    void add(const Pair& p, const std::string& path);

    // Recompute from the stored pairs. The one-pair case uses the yaws carried BY the pair, so the
    // result is the same whether it was just recorded or reloaded from disk.
    const Fit& solve();

    [[nodiscard]] const Fit& fit() const noexcept { return fit_; }
    [[nodiscard]] const std::vector<Pair>& pairs() const noexcept { return pairs_; }

    // world_T_room as a 4x4, for the request builder. nullopt when there is no usable fit.
    [[nodiscard]] std::optional<Eigen::Matrix4d> world_T_room() const;

private:
    std::vector<Pair> pairs_;
    Fit               fit_;
};

}   // namespace rc
