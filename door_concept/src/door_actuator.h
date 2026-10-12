/*
 * door_actuator.h — ask somebody to open or close a door. Client side of RoboCompDoorControl.
 *
 * ★WHY A CAPABILITY AND NOT "THE SIMULATOR". On the real robot there is no Webots, so this agent must
 * not depend on one. DoorControl.idsl is provider-agnostic on purpose: the same request is served by a
 * Webots supervisor, by home automation, or by a person asked over TTS. This class knows only that
 * *something* out there may be able to move a door, and prices the request accordingly.
 *
 * Three properties of the interface this client must preserve — they are the reason it exists:
 *
 *  1. ★NOTHING HERE EVER REPORTS THAT A DOOR IS OPEN. `Delivered` means the provider did what it was
 *     asked, not that the world changed. Whether the door is actually open stays a PERCEPTUAL question,
 *     answered by the same existence/shape machinery as everything else. Wiring "Delivered" straight
 *     into a belief would be exactly the mistake the affordance work spent a month removing: a protocol
 *     event is not evidence.
 *
 *  2. ★THE CALLER OWNS THE ASSOCIATION. A provider advertises opaque ids (a Webots DEF, a Z-Wave node)
 *     with poses in mm, WORLD frame. We hold fitted apertures in metres, ROOM frame. Matching the two
 *     is our job — we are the ones doing perception and frame algebra — and a provider is never asked
 *     to understand rooms. Hence match_door(), and hence the conversion in the request builder.
 *
 *  3. ★A REFUSAL IS INFORMATION, and the kinds differ. UnknownDoor/NotActuable mean never ask again;
 *     TemporarilyBusy means later; Declined means a person said no. Three different beliefs and three
 *     different policies, so the reason is surfaced rather than collapsed into "it didn't work".
 *
 * Polled, not evented (matches this agent's compute loop): requestDoor -> RequestAck, then
 * getRequestStatus(id) until it settles.
 *
 * ⚠Everything here is best-effort. No proxy, a dead provider, or an Ice exception leaves the agent
 * running and simply means "this affordance is not available", which is a legitimate state and not an
 * error — a door the robot cannot open is still a door.
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace Ice { class Communicator; }
namespace RoboCompDoorControl { class DoorControlPrx; }

namespace rc
{

class DoorActuator
{
public:
    // What the provider can do, read once. `available` false means: no proxy, or the provider itself
    // says canActuate=false — which is the honest answer for a world whose doors are scenery, and the
    // UI must show it rather than offering a button that can only fail.
    struct Caps
    {
        bool  available        = false;
        bool  continuous_angle = false;
        bool  requires_human   = false;   // every request is then a social act with a real cost
        float typical_latency_s = 0.0f;
        int   n_doors          = 0;       // advertised ids; 0 = "ask me about any door, I resolve it"
        std::string note;                 // human-readable state for the UI
        // ★The provider's OWN ids and where it thinks they are (mm, its world frame). Kept because the
        // place-matching path needs both frames registered to each other, and here they are not: the
        // room frame is room-local, so our door at (-373, 4440) mm is 4 m from DOOR_0 at (-3583, 7166)
        // and the provider correctly answers UnknownDoor. Quoting an advertised id resolves by NAME and
        // needs no shared frame at all — which is what a human pressing a button can supply and an
        // autonomous caller cannot. Autonomy still needs the registration; this does not pretend to fix it.
        // ★THE ANGLE IS PART OF THE POSE. Storing only (id, xy) threw it away, and the one-pair
        // registration then had nothing to derive rotation from — it was called with a hardcoded 0,
        // producing theta ~ +0.01 rad where the truth is ~ -1.576. A pure-translation fit against a
        // 90-degree-rotated frame puts the door metres away, which is exactly the failure it was
        // supposed to fix.
        struct Door { std::string id; Eigen::Vector2f xy_mm; float angle; };
        std::vector<Door> doors;
    };

    // One outstanding request, as far as this agent knows.
    struct Pending
    {
        int         id = -1;
        std::string door_name;
        bool        opening = true;
        std::string state = "—";          // Queued / InProgress / Delivered / Aborted / refused
        bool        settled = false;
    };

    DoorActuator();
    ~DoorActuator();

    // `endpoint` is a full Ice proxy string, e.g. "doorcontrol:tcp -h localhost -p 10017".
    // Lazily connects on first use; never throws.
    void configure(std::string endpoint);

    // Re-reads capabilities from the provider (cheap; call on UI open or on a retry button).
    const Caps& refresh_caps();
    [[nodiscard]] const Caps& caps() const noexcept { return caps_; }

    // Ask for `door_name` (our own persistent identity) at room-frame `xy_m` / `yaw` to be opened or
    // closed. `room_T_world` converts our room frame to the provider's world frame; identity if they
    // are the same. Returns the pending record, or nullopt when there is no usable provider.
    // `provider_id` empty ⇒ let the provider resolve by PLACE from the pose we send (requires the two
    // frames to be registered). Non-empty ⇒ resolve by NAME, which needs no shared frame.
    std::optional<Pending> request(const std::string& door_name,
                                   const Eigen::Vector2f& xy_m, float yaw, float width_m,
                                   bool open,
                                   const Eigen::Matrix4d& world_T_room,
                                   const std::string& purpose,
                                   const std::string& provider_id = {});

    // Poll a pending request. Updates `p.state` / `p.settled` in place.
    void poll(Pending& p);

    void cancel(const Pending& p);

private:
    bool ensure_proxy();

    std::string                                   endpoint_;
    std::shared_ptr<Ice::Communicator>            comm_;
    std::shared_ptr<RoboCompDoorControl::DoorControlPrx> prx_;
    Caps                                          caps_;
    std::chrono::steady_clock::time_point         last_connect_attempt_{};
};

// ─── WHICH PROVIDER DOOR IS OURS — pure, header-only so an offline harness can test it ───────────
//
// ★THE ASSOCIATION IS OURS (property 2 above). Given our door's aperture centre already carried into the
// provider's WORLD frame (metres), pick the advertised door nearest to it. Every distance is returned so
// the request log can say WHY a door was chosen (nearest, how far, and how far the runner-up was), which
// is what was missing when every request silently went to one configured id.
//
// ⚠THE GATE IS A THRESHOLD, and it is kept deliberately: this decides an ACT IN THE WORLD (a door swings),
// not a belief, so "none of the advertised doors" must be a possible answer, and nothing in the model gives
// a probability for an unadvertised door. The value is set by geometry, not tuned: advertised doors here
// are >= 2.5 m apart, and the error budget of a correct match is the localisation error (cm) + the aperture
// fit (~0.1-0.2 m) + the provider's door origin vs the doorway centre (~0.1 m, measured door_3/19 ↔ DOOR_1).
struct ProviderDoorMatch
{
    std::string id;                 // the provider's advertised id (a Webots DEF)
    float dist_m      = 0.0f;       // our door -> that door, world frame
    float runner_up_m = -1.0f;      // the next-nearest advertised door; -1 when there is only one
    Eigen::Vector2f door_world_m = Eigen::Vector2f::Zero();
};

[[nodiscard]] inline std::optional<ProviderDoorMatch>
nearest_provider_door(const std::vector<DoorActuator::Caps::Door>& doors, const Eigen::Vector2f& ours_world_m)
{
    if (doors.empty())
        return std::nullopt;
    std::vector<std::pair<float, std::size_t>> by_dist;
    by_dist.reserve(doors.size());
    for (std::size_t i = 0; i < doors.size(); ++i)
        by_dist.emplace_back((doors[i].xy_mm / 1000.0f - ours_world_m).norm(), i);
    std::ranges::sort(by_dist);
    const auto& d = doors[by_dist.front().second];
    return ProviderDoorMatch{d.id, by_dist.front().first,
                             by_dist.size() > 1 ? by_dist[1].first : -1.0f, d.xy_mm / 1000.0f};
}

// ─── world←body FROM THE SIMULATOR'S WITNESS ─────────────────────────────────────────────────────
// robot_concept writes robot_gt_x/y/angle on the "robot" node ONLY while the producer says simulated (the
// Webots supervisor's own node pose), so on the real robot these attributes simply do not exist and this
// path cannot be taken by accident.
//   • x, y are METRES in the Webots world (the bridge's "metros → mm" comment is wrong; it sends metres).
//   • the angle is NOT the Webots yaw. The bridge sends atan2(R01, R00) - pi/2 = -theta_w - pi/2, so
//     theta_w = -(raw + pi/2). Verified 2026-10-12 on 18031 rows of room_concept's gt_error.csv: with this
//     convention theta_w - est_theta has a circular mean of 0.14 deg against a POSITION-ONLY Procrustes fit
//     of 0.24 deg, i.e. the robocomp body frame (x right, y forward) IS the Webots Shadow node frame
//     rotated by theta_w. (ground_truth_log.h documents the same inverted sign from the other side.)
// Returns world_T_body as a 4x4 (planar; z/roll/pitch are irrelevant to a door's floor position).
[[nodiscard]] inline Eigen::Matrix4d world_T_body_from_sim_gt(double x_m, double y_m, double raw_angle)
{
    const double th = -(raw_angle + M_PI / 2.0);
    Eigen::Matrix4d M = Eigen::Matrix4d::Identity();
    M(0, 0) = std::cos(th); M(0, 1) = -std::sin(th);
    M(1, 0) = std::sin(th); M(1, 1) =  std::cos(th);
    M(0, 3) = x_m;          M(1, 3) = y_m;
    return M;
}

}   // namespace rc
