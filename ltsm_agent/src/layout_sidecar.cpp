/*
 * layout_sidecar.cpp — see layout_sidecar.h.
 */
#include "layout_sidecar.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

namespace ltsm::topo
{

QByteArray layout_to_json(const LayoutCache &c)
{
    const auto terms = [](const layout::Terms &t)
    {
        return QJsonObject{{"seam", t.seam}, {"manhattan", t.manhattan}, {"overlap", t.overlap}, {"wall", t.wall},
                           {"total", t.total()}};
    };
    QJsonArray rooms;
    for (const auto &[name, p] : c.pose)
    {
        QJsonArray cov;
        if (const auto it = c.cov.find(name); it != c.cov.end())
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) cov.append(it->second(i, j));
        const auto pm = c.p_manhattan.find(name);
        rooms.append(QJsonObject{{"name", QString::fromStdString(name)}, {"x", p.x()}, {"y", p.y()}, {"yaw", p.z()},
                                 {"cov", cov}, {"p_manhattan", pm != c.p_manhattan.end() ? pm->second : -1.0}});
    }
    QJsonArray passages;
    for (const auto &[name, d2] : c.seam_d2)
    {
        const auto sf = c.stated_faces.find(name);
        const auto pi = c.seam_p_inlier.find(name);
        passages.append(QJsonObject{{"name", QString::fromStdString(name)}, {"seam_d2", d2},
                                    {"p_inlier", pi != c.seam_p_inlier.end() ? pi->second : -1.0},
                                    {"stated_faces", sf != c.stated_faces.end() ? sf->second : 0}});
    }
    QJsonObject o{{"key", QString::fromStdString(c.key)}, {"rooms", rooms}, {"passages", passages},
                  {"energy_before", terms(c.before)}, {"energy_after", terms(c.after)},
                  {"iterations", c.iterations}, {"converged", c.converged}, {"ms", c.ms}, {"solves", c.solves},
                  {"starts", c.starts}, {"restarted", c.restarted}};
    if (c.building_axis.has_value()) o["building_axis"] = c.building_axis.value();
    return QJsonDocument(o).toJson(QJsonDocument::Indented);
}

std::optional<LayoutCache> layout_from_json(const std::filesystem::path &path)
{
    QFile f(QString::fromStdString(path.string()));
    if (not f.open(QIODevice::ReadOnly)) return std::nullopt;
    QJsonParseError err{};
    const auto doc = QJsonDocument::fromJson(f.readAll(), &err);
    if (err.error != QJsonParseError::NoError or not doc.isObject()) return std::nullopt;
    const auto o = doc.object();
    LayoutCache c;   // key stays empty: the graph decides whether this still describes it
    for (const auto &v : o["rooms"].toArray())
    {
        const auto r = v.toObject();
        const auto name = r["name"].toString().toStdString();
        if (name.empty()) continue;
        c.pose[name] = layout::Vec3(r["x"].toDouble(), r["y"].toDouble(), r["yaw"].toDouble());
        if (const auto cov = r["cov"].toArray(); cov.size() == 9)
        {
            layout::Mat3 m;
            for (int i = 0; i < 9; ++i) m(i / 3, i % 3) = cov[i].toDouble();
            c.cov[name] = m;
        }
        c.p_manhattan[name] = r["p_manhattan"].toDouble(-1.0);
    }
    for (const auto &v : o["passages"].toArray())
    {
        const auto p = v.toObject();
        c.seam_d2[p["name"].toString().toStdString()] = p["seam_d2"].toDouble(-1.0);
        c.seam_p_inlier[p["name"].toString().toStdString()] = p["p_inlier"].toDouble(-1.0);
    }
    if (o.contains("building_axis")) c.building_axis = o["building_axis"].toDouble();
    c.solves = o["solves"].toInt(0);
    return c;
}

}   // namespace ltsm::topo
