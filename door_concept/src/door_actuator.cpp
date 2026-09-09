#include "door_actuator.h"

#include <Ice/Ice.h>

#include <cmath>
#include <print>

#include "../generated/DoorControl.h"

namespace rc
{

namespace
{
const char* refusal_text(RoboCompDoorControl::RefusalReason r)
{
    using R = RoboCompDoorControl::RefusalReason;
    switch (r)
    {
        case R::UnknownDoor:     return "unknown door";      // never ask again
        case R::NotActuable:     return "not actuable";      // never ask again
        case R::TemporarilyBusy: return "busy";              // ask later
        case R::Declined:        return "declined";          // a person said no
        default:                 return "none";
    }
}

const char* state_text(RoboCompDoorControl::RequestState s)
{
    using S = RoboCompDoorControl::RequestState;
    switch (s)
    {
        case S::Queued:     return "queued";
        case S::InProgress: return "in progress";
        case S::Delivered:  return "delivered";
        case S::Aborted:    return "aborted";
        default:            return "?";
    }
}
}   // namespace

DoorActuator::DoorActuator() = default;
DoorActuator::~DoorActuator()
{
    // Destroy the communicator explicitly: Ice threads outliving the process teardown is a documented
    // way to hang on exit, and this agent's shutdown path must stay clean (it deletes owned DSR nodes).
    if (comm_)
        try { comm_->destroy(); } catch (...) {}
}

void DoorActuator::configure(std::string endpoint)
{
    endpoint_ = std::move(endpoint);
    caps_ = Caps{};
    caps_.note = endpoint_.empty() ? "disabled (no endpoint configured)" : "not connected yet";
}

bool DoorActuator::ensure_proxy()
{
    if (prx_)
        return true;
    if (endpoint_.empty())
        return false;
    // Self-throttled: a provider that is not running must not cost a connect attempt every cycle.
    const auto now = std::chrono::steady_clock::now();
    if (now - last_connect_attempt_ < std::chrono::seconds(2))
        return false;
    last_connect_attempt_ = now;
    try
    {
        if (not comm_)
            comm_ = Ice::initialize();
        auto base = comm_->stringToProxy(endpoint_);
        prx_ = Ice::checkedCast<RoboCompDoorControl::DoorControlPrx>(base);
        if (not prx_)
        {
            caps_.note = "no DoorControl provider at " + endpoint_;
            return false;
        }
    }
    catch (const std::exception& e)
    {
        // Not an error: a robot with no door provider is a normal robot.
        caps_.note = std::string{"provider unreachable: "} + e.what();
        prx_ = nullptr;
        return false;
    }
    return true;
}

const DoorActuator::Caps& DoorActuator::refresh_caps()
{
    if (not ensure_proxy())
    {
        caps_.available = false;
        return caps_;
    }
    try
    {
        const auto c = prx_->getCapabilities();
        caps_.available         = c.canActuate;
        caps_.continuous_angle  = c.continuousAngle;
        caps_.requires_human    = c.requiresHuman;
        caps_.typical_latency_s = c.typicalLatencySec;
        caps_.n_doors           = static_cast<int>(c.doors.size());
        caps_.doors.clear();
        for (const auto& d : c.doors)
            caps_.doors.push_back(Caps::Door{d.id, Eigen::Vector2f(d.pose.x, d.pose.y), d.pose.angle});
        // ★"Connected" and "can actuate" are different facts, and the UI must not conflate them: the
        // Webots provider answers canActuate=false whenever its [DoorControl] Doors list is absent,
        // which is the honest reply for a world whose doors are scenery. A button that is enabled
        // because a peer exists, and then always fails, teaches the operator to distrust the button.
        caps_.note = c.canActuate
            ? std::format("ready · {} door(s) advertised · ~{:.1f}s{}", caps_.n_doors,
                          caps_.typical_latency_s, caps_.requires_human ? " · asks a person" : "")
            : "provider connected but reports canActuate = false (no actuable doors configured)";
    }
    catch (const std::exception& e)
    {
        caps_.available = false;
        caps_.note = std::string{"getCapabilities failed: "} + e.what();
        prx_ = nullptr;   // force a reconnect next time
    }
    return caps_;
}

std::optional<DoorActuator::Pending>
DoorActuator::request(const std::string& door_name, const Eigen::Vector2f& xy_m, float yaw,
                      float width_m, bool open, const Eigen::Matrix4d& world_T_room,
                      const std::string& purpose, const std::string& provider_id)
{
    if (not ensure_proxy())
        return std::nullopt;

    // ★ROOM (metres) -> WORLD (millimetres). The interface says mm, world frame, "matching
    // getObjectPose", and we hold metres in the room frame. Getting this wrong does not fail loudly —
    // it asks a provider to open a door 1000x too far away, which comes back as UnknownDoor and reads
    // like a broken interface rather than a unit bug. The yaw is rotated by the same transform.
    const Eigen::Vector4d p_room(xy_m.x(), xy_m.y(), 0.0, 1.0);
    const Eigen::Vector4d p_world = world_T_room * p_room;
    const double yaw_world = yaw + std::atan2(world_T_room(1, 0), world_T_room(0, 0));

    RoboCompDoorControl::DoorRequest req;
    req.doorId = provider_id;              // empty ⇒ the provider resolves by pose instead
    req.door.id = door_name;               // OUR persistent identity, for the provider to echo back
    req.door.pose.x = static_cast<float>(p_world.x() * 1000.0);
    req.door.pose.y = static_cast<float>(p_world.y() * 1000.0);
    req.door.pose.angle = static_cast<float>(yaw_world);
    req.door.width = width_m * 1000.0f;
    req.action = open ? RoboCompDoorControl::DoorAction::OpenDoor
                      : RoboCompDoorControl::DoorAction::CloseDoor;
    req.apertureHint = 0.0f;               // binary providers ignore it; we do not claim an angle
    req.purpose = purpose;

    Pending p;
    p.door_name = door_name;
    p.opening = open;
    try
    {
        const auto ack = prx_->requestDoor(req);
        p.id = ack.requestId;
        if (not ack.accepted)
        {
            p.state = std::string{"refused: "} + refusal_text(ack.refusal);
            p.settled = true;
        }
        else
            p.state = std::format("accepted (~{:.1f}s)", ack.expectedLatencySec);
        // ★Print the WORLD-frame millimetres actually sent. The provider matches by place within a few
        // hundred mm, so when it answers UnknownDoor the only useful question is "where did we say the
        // door was" — and without this line that number exists nowhere.
        std::println("door_concept: [actuator] {} '{}' as '{}' at world ({:.0f}, {:.0f}) mm w={:.0f} mm "
                     "-> id {} : {}",
                     open ? "OPEN" : "CLOSE", door_name,
                     provider_id.empty() ? "(by place)" : provider_id,
                     req.door.pose.x, req.door.pose.y, req.door.width, p.id, p.state);
    }
    catch (const std::exception& e)
    {
        p.state = std::string{"call failed: "} + e.what();
        p.settled = true;
        prx_ = nullptr;
    }
    return p;
}

void DoorActuator::poll(Pending& p)
{
    if (p.settled or p.id < 0 or not prx_)
        return;
    try
    {
        const auto st = prx_->getRequestStatus(p.id);
        p.state = state_text(st.state);
        using S = RoboCompDoorControl::RequestState;
        if (st.state == S::Delivered or st.state == S::Aborted)
        {
            p.settled = true;
            if (st.state == S::Aborted)
                p.state += std::string{" ("} + refusal_text(st.refusal) + ")";
            // ★Deliberately NOTHING is written to any belief here. Delivered means the provider acted,
            // not that the door moved. Whether it is open is settled by looking, like everything else.
        }
    }
    catch (const std::exception& e)
    {
        p.state = std::string{"status failed: "} + e.what();
        p.settled = true;
        prx_ = nullptr;
    }
}

void DoorActuator::cancel(const Pending& p)
{
    if (p.id < 0 or not prx_)
        return;
    try { prx_->cancelRequest(p.id); } catch (...) {}
}

}   // namespace rc
