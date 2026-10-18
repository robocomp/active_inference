#include "canvas_snapshot.h"

#include <chrono>
#include <cmath>
#include <ctime>
#include <format>
#include <numbers>

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>

namespace rc::chat
{

namespace
{
QString q(const std::string& s) { return QString::fromStdString(s); }

QJsonValue opt(const std::optional<float>& v) { return v.has_value() ? QJsonValue(double(*v)) : QJsonValue(); }

double yaw_of(const Eigen::Affine3d& T) { return std::atan2(T.matrix()(1, 0), T.matrix()(0, 0)); }

QJsonValue room_json(DSR::DSRGraph& G, std::uint64_t room_id, const std::string& room_name)
{
    if (room_id == 0) return QJsonValue();
    QJsonObject room{{"name", q(room_name)}};
    QJsonArray poly;
    if (const auto rn = G.get_node(room_id); rn.has_value())
    {
        const auto px = G.get_attrib_by_name<delimiting_polygon_x_att>(rn.value());
        const auto py = G.get_attrib_by_name<delimiting_polygon_y_att>(rn.value());
        if (px.has_value() and py.has_value())
        {
            const auto& xs = px.value().get();
            const auto& ys = py.value().get();
            for (std::size_t i = 0; i < std::min(xs.size(), ys.size()); ++i)
                poly.append(QJsonArray{double(xs[i]), double(ys[i])});
        }
    }
    room.insert("polygon", poly);
    return room;
}

QJsonValue motion_json(DSR::DSRGraph& G)
{
    const auto robots = G.get_nodes_by_type("robot");
    if (robots.empty()) return QJsonValue();
    const auto& rn = robots.front();
    const auto adv = G.get_attrib_by_name<robot_current_advance_speed_att>(rn);
    if (not adv.has_value()) return QJsonValue();
    return QJsonObject{{"advance", double(*adv)},
                       {"side", double(G.get_attrib_by_name<robot_current_side_speed_att>(rn).value_or(0.f))},
                       {"angular", double(G.get_attrib_by_name<robot_current_angular_speed_att>(rn).value_or(0.f))}};
}

// The executing affordance's goal, as world_digest.cpp reads it: the consumer's claim edge, else the proposal.
QJsonValue target_json(DSR::DSRGraph& G)
{
    for (const auto& a : G.get_nodes_by_type("affordance"))
    {
        if (not G.get_attrib_by_name<active_att>(a).value_or(false) or
            not G.get_attrib_by_name<epistemic_pending_att>(a).value_or(false))
            continue;
        std::optional<float> gx, gy;
        for (const auto& e : G.get_edges_to_id(a.id()))
            if (e.type() == "executing")
            {
                gx = G.get_attrib_by_name<executing_target_x_m_att>(e);
                gy = G.get_attrib_by_name<executing_target_y_m_att>(e);
            }
        if (not gx.has_value() or not gy.has_value())
        {
            gx = G.get_attrib_by_name<epistemic_target_x_m_att>(a);
            gy = G.get_attrib_by_name<epistemic_target_y_m_att>(a);
        }
        if (not gx.has_value() or not gy.has_value()) return QJsonValue();
        return QJsonObject{{"x", double(*gx)}, {"y", double(*gy)}, {"affordance", q(a.name())}};
    }
    return QJsonValue();
}

std::string clock_now()
{
    const auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tmv{};
    localtime_r(&t, &tmv);
    return std::format("{:02}:{:02}:{:02}", tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
}
}   // namespace

QJsonObject build_canvas_snapshot(DSR::DSRGraph& G, DSR::InnerEigenAPI& inner, const Digest& digest,
                                  const rc::mission::Progress& mission, double robot_radius_m, SnapshotSize canvas)
{
    const std::string& room = digest.facts.room;
    QJsonObject s{{"version", 1}, {"time", q(clock_now())},
                  {"canvas", QJsonObject{{"w", canvas.w}, {"h", canvas.h}}},
                  {"room", room_json(G, digest.room_id, room)}, {"motion", motion_json(G)}, {"target", target_json(G)}};

    QJsonValue robot;
    if (not room.empty())
        if (const auto T = inner.get_transformation_matrix(room, "body"); T.has_value())
        {
            const double yaw = yaw_of(*T);
            robot = QJsonObject{{"x", T->translation().x()}, {"y", T->translation().y()}, {"yaw", yaw},
                                {"heading", yaw + std::numbers::pi / 2.0}, {"radius", robot_radius_m}};
        }
    s.insert("robot", robot);

    QJsonArray doors;
    for (const auto& name : digest.facts.doors)
    {
        QJsonObject d{{"name", q(name)}, {"x", QJsonValue()}, {"y", QJsonValue()}, {"yaw", QJsonValue()},
                      {"width", QJsonValue()}, {"a", QJsonValue()}, {"b", QJsonValue()}, {"p_open", QJsonValue()}};
        const auto dn = G.get_node(name);
        if (dn.has_value())
        {
            d["width"] = opt(G.get_attrib_by_name<width_m_att>(dn.value()));
            d["p_open"] = opt(G.get_attrib_by_name<door_open_prob_att>(dn.value()));
        }
        if (const auto T = inner.get_transformation_matrix(room, name); T.has_value())
        {
            const double x = T->translation().x(), y = T->translation().y(), yaw = yaw_of(*T);
            d["x"] = x;
            d["y"] = y;
            d["yaw"] = yaw;
            if (d["width"].isDouble())
            {
                const double h = d["width"].toDouble() / 2.0;
                d["a"] = QJsonArray{x - h * std::cos(yaw), y - h * std::sin(yaw)};
                d["b"] = QJsonArray{x + h * std::cos(yaw), y + h * std::sin(yaw)};
            }
        }
        doors.append(d);
    }
    s.insert("doors", doors);

    QJsonArray objects;
    for (const auto& o : digest.objects)
    {
        if (not o.in_current_room) continue;
        QJsonObject j{{"name", q(o.name)}, {"type", q(o.subtype)}, {"x", QJsonValue()}, {"y", QJsonValue()},
                      {"yaw", QJsonValue()}, {"width", opt(o.width_m)}, {"depth", opt(o.depth_m)},
                      {"height", opt(o.height_m)},
                      {"dist", o.dist_m.has_value() ? QJsonValue(*o.dist_m) : QJsonValue()},
                      {"bearing_deg", o.bearing_deg.has_value() ? QJsonValue(*o.bearing_deg) : QJsonValue()}};
        if (const auto T = inner.get_transformation_matrix(room, o.name); T.has_value())
        {
            j["x"] = T->translation().x();
            j["y"] = T->translation().y();
            j["yaw"] = yaw_of(*T);
        }
        objects.append(j);
    }
    s.insert("objects", objects);

    s.insert("mission", mission.status.empty() and mission.title.empty()
                            ? QJsonValue()
                            : QJsonValue(QJsonObject{{"title", q(mission.title)}, {"status", q(mission.status)},
                                                     {"line", q(mission.line)}}));
    return s;
}

bool write_canvas_snapshot(const QString& dir, const QJsonObject& snapshot)
{
    const QString tmp = dir + QStringLiteral("/snapshot.json.tmp");
    QFile f(tmp);
    if (not f.open(QIODevice::WriteOnly | QIODevice::Truncate)) return false;
    f.write(QJsonDocument(snapshot).toJson(QJsonDocument::Compact));
    f.close();
    const QString dst = dir + QStringLiteral("/snapshot.json");
    QFile::remove(dst);
    return QFile::rename(tmp, dst);
}

}   // namespace rc::chat
