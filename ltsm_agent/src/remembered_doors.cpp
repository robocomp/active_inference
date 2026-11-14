#include "remembered_doors.h"
#include "room_eviction.h"
#include "room_recall.h"
#include "../../common/room_resolve/room_resolve.h"

#include <dsr/core/types/type_checking/dsr_attr_name.h>

#include <QDebug>
#include <QString>

#include <chrono>
#include <cmath>
#include <format>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

namespace ltsm
{
namespace
{
constexpr const char *kSubtype = "remembered_door";

bool is_note(DSR::DSRGraph &g, const DSR::Node &n)
{
    return n.type() == "metaconcept" and n.name().starts_with("remembered_")
           and g.get_attrib_by_name<object_subtype_att>(n).value_or(std::string{}) == kSubtype;
}

/// A memory door face is an `object` named r<k>_door_*.
bool is_face(const DSR::Node &n)
{
    return n.type() == "object" and RoomRecall::live_name_of(n.name()).starts_with("door_");
}

/// Whether `face` hangs, through any chain of parents (wall, floor, ...), from `ancestor`. The walk ends with the chain
/// itself: at the ancestor, at a root, or at a node already seen (a cycle).
bool descends_from(DSR::DSRGraph &g, const DSR::Node &face, std::uint64_t ancestor)
{
    std::unordered_set<std::uint64_t> seen{face.id()};
    auto p = g.get_attrib_by_name<parent_att>(face);
    while (p.has_value() and p.value() != 0)
    {
        if (p.value() == ancestor) return true;
        if (not seen.insert(p.value()).second) return false;
        const auto up = g.get_node(p.value());
        if (not up.has_value()) return false;
        p = g.get_attrib_by_name<parent_att>(up.value());
    }
    return false;
}
}   // namespace

RememberedDoors::RememberedDoors(std::shared_ptr<DSR::DSRGraph> live, std::shared_ptr<DSR::DSRGraph> memory)
    : live_(std::move(live)), mem_(std::move(memory)) {}

int RememberedDoors::remove_all()
{
    int n = 0;
    for (const auto &node : live_->get_nodes_by_type("metaconcept"))
        if (is_note(*live_, node) and live_->delete_node(node.id())) ++n;
    return n;
}

int RememberedDoors::retire_where_parent_is_not(std::uint64_t room_id)
{
    int n = 0;
    for (const auto &node : live_->get_nodes_by_type("metaconcept"))
        if (is_note(*live_, node) and live_->get_attrib_by_name<parent_att>(node).value_or(0) != room_id
            and live_->delete_node(node.id())) ++n;
    return n;
}

int RememberedDoors::step()
{
    // one-off cost report: the first kTimedCalls calls (door memory must not slow the main cycle unnoticed)
    const auto t0 = std::chrono::steady_clock::now();
    const int n = step_impl();
    if (timed_calls_ < kTimedCalls)
    {
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        sum_us_ += us;
        max_us_ = std::max(max_us_, us);
        if (++timed_calls_ == kTimedCalls)
            qInfo().noquote() << QString::fromStdString(std::format("[remembered] step cost over first {} calls: mean {:.0f} us, max {:.0f} us",
                                                                    kTimedCalls, sum_us_ / kTimedCalls, max_us_));
    }
    return n;
}

int RememberedDoors::step_impl()
{
    const auto cur = rc::room::current_room(*live_);
    if (not cur.has_value())
    {
        retire_where_parent_is_not(0);
        current_ = 0;
        adopted_.clear();
        return 0;
    }
    if (cur.value() != current_)
    {
        current_ = cur.value();
        adopted_.clear();
    }
    retire_where_parent_is_not(current_);

    const auto room = live_->get_node(current_);
    if (not room.has_value()) return 0;
    const auto rid = live_->get_attrib_by_name<room_id_att>(room.value());
    const auto twin = rid.has_value() ? mem_->get_node(std::format("r{}_{}", rid.value(), room->name())) : std::nullopt;
    if (not twin.has_value() or is_staged_room(*mem_, twin.value()))
    {
        retire_where_parent_is_not(0);
        return 0;
    }

    auto mie = mem_->get_inner_eigen_api();   // created here, on the thread that owns it
    const int level = live_->get_node_level(room.value()).value_or(0) + 1;
    for (const auto &face : mem_->get_nodes_by_type("object"))
    {
        if (not is_face(face) or not descends_from(*mem_, face, twin->id())) continue;
        // A face without its own geometry (a derived `_entry` face) can never be read back, hence never adopted: not posted.
        if (not (face.attrs().contains(width_m_str.data()) and face.attrs().contains(height_m_str.data())
                 and face.attrs().contains(depth_m_str.data()))) continue;
        if (adopted_.contains(face.id())) continue;

        // The note's pose is the face's pose IN THE ROOM's frame, T(room <- face): the face hangs from a wall (or the room) in memory.
        const auto pose = mie->get_transformation_matrix(twin->name(), face.name());
        if (not pose.has_value()) continue;
        const std::string name = "remembered_" + RoomRecall::live_name_of(face.name());
        auto note = live_->get_node(name);
        if (note.has_value() and live_->get_attrib_by_name<door_adopted_att>(note.value()).value_or(false))
        {
            live_->delete_node(note->id());
            adopted_.insert(face.id());
            continue;
        }

        // The face's attributes come over as RoomEviction::twin took them in, minus what is the face's place in ITS tree.
        auto attrs = face.attrs();
        attrs.erase(parent_str.data());
        attrs.erase(level_str.data());
        attrs.erase(collapsed_str.data());
        attrs.erase(object_subtype_str.data());   // the note's own subtype is set below; copying "door" would rewrite it every cycle
        attrs.erase(door_adopted_str.data());   // the adoption flag belongs to the note alone

        const bool fresh = not note.has_value();
        if (fresh)
        {
            DSR::Node n;
            n.type("metaconcept");
            n.name(name);
            note = n;
        }
        bool changed = fresh;
        for (const auto &[k, v] : attrs)
        {
            const auto it = note->attrs().find(k);
            if (it == note->attrs().end() or not(it->second == v)) { note->attrs()[k] = v; changed = true; }
        }
        const auto set = [&](auto tag, auto value)
        {
            using A = decltype(tag);
            const auto old = live_->get_attrib_by_name<A>(note.value());
            if (old.has_value() and old.value() == value) return;
            live_->add_or_modify_attrib_local<A>(note.value(), value);
            changed = true;
        };
        set(object_subtype_att{}, std::string(kSubtype));
        set(parent_att{}, current_);
        set(level_att{}, level);
        set(room_id_att{}, rid.value());

        std::uint64_t id = note->id();
        if (fresh)
        {
            const auto ins = live_->insert_node(note.value());
            if (not ins.has_value())
            {
                if (failed_insert_.insert(name).second)
                    qWarning().noquote() << QString::fromStdString(std::format("[remembered] insert_node({}) failed", name));
                continue;
            }
            failed_insert_.erase(name);
            id = ins.value();
            qInfo().noquote() << QString::fromStdString(std::format("[remembered] {} posted under {}", name, room->name()));
        }
        else if (changed)
        {
            live_->update_node(note.value());
            ++updates_;
        }

        // Planar: a door is a vertical plane, so the note carries (0, 0, yaw) written directly. Going through rt_attributes would
        // run Eigen's eulerAngles(2,1,0), which returns (pi, -pi, yaw+pi) for a negative yaw: a reader of rz would be off by pi.
        const auto& M = pose.value();
        auto rt = RoomEviction::rt_attributes(M);
        const float yaw = static_cast<float>(std::atan2(M.rotation()(1, 0), M.rotation()(0, 0)));
        rt.insert_or_assign(rt_rotation_euler_xyz_str.data(), DSR::Attribute{std::vector<float>{0.f, 0.f, yaw}, 0, 0});
        const auto old = live_->get_edge(current_, id, "RT");
        if (not old.has_value() or not(old->attrs() == rt))
            live_->insert_or_assign_edge(DSR::Edge::create<RT_edge_type>(current_, id, rt));
    }

    int n = 0;
    for (const auto &node : live_->get_nodes_by_type("metaconcept")) if (is_note(*live_, node)) ++n;
    return n;
}
}   // namespace ltsm
