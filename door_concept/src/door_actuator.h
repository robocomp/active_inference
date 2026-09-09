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

#include <chrono>
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

}   // namespace rc
