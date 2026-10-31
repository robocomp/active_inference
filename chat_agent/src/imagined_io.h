// imagined_io.h -- the data path from the card's imagined observations to chat_agent's side canvas.
// Writes <run_dir>/imagined.json and <run_dir>/imagined_zed.png; canvas/rcanvas.py reads them (imagined(),
// draw_imagined_zed(), draw_imagined_lidar()).
//
// Determinism: identical inputs give byte-identical output. Keys come out sorted (QJsonObject), the writer is
// QJsonDocument::Compact (shortest round-trip doubles, never locale dependent), LiDAR points and poses are rounded to
// a millimetre, nothing carries a timestamp.
// PNG: R = per-pixel frequency of "visible" over the samples, G = of "occluded", both scaled to 0..255, B = 0.
#pragma once

#include "perception_card/generative.h"
#include "perception_card/render.h"

#include <array>
#include <cmath>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <QSaveFile>
#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

namespace rc::chat
{
namespace imagined_detail
{
inline double mm(double v) { return std::round(v * 1000.0) / 1000.0; }

// The JSON stays bounded: at most kMaxPoints per list, taken by a fixed stride (deterministic). Non-finite points are
// skipped (counted in `skipped`); n_total is the finite count before decimation.
constexpr std::size_t kMaxPoints = 500;

inline QJsonArray points(const std::vector<Eigen::Vector3f>& pts, std::size_t& n_total, std::size_t& skipped)
{
    std::vector<const Eigen::Vector3f*> ok;
    for (const auto& p : pts)
    {
        if (std::isfinite(p.x()) and std::isfinite(p.y()) and std::isfinite(p.z())) ok.push_back(&p);
        else ++skipped;
    }
    n_total = ok.size();
    const std::size_t stride = std::max<std::size_t>(1, (ok.size() + kMaxPoints - 1) / kMaxPoints);
    QJsonArray a;
    for (std::size_t i = 0; i < ok.size(); i += stride)
        a.append(QJsonArray{mm(ok[i]->x()), mm(ok[i]->y()), mm(ok[i]->z())});
    return a;
}
}   // namespace imagined_detail

// images[i] and scans[k][i] belong to samples[i]; a shorter vector just leaves the later samples without that part.
inline bool write_imagined(const std::filesystem::path& run_dir, const rc::card::SampleSet& set,
                           const std::vector<rc::card::ImaginedImage>& images,
                           const std::array<std::vector<rc::card::ImaginedScan>, 2>& scans,
                           const rc::card::McResult& mc, std::string_view caption,
                           const std::array<std::string_view, 2>& lidar_names = {"lidar 0", "lidar 1"})
{
    using namespace imagined_detail;
    QJsonArray samples;
    for (std::size_t i = 0; i < set.samples.size(); ++i)
    {
        const auto& s = set.samples[i];
        QJsonObject state;
        for (const auto& [k, v] : s.state) state.insert(QString::fromStdString(k), mm(v));
        QJsonArray parts;
        for (const auto& b : s.parts.boxes)
            parts.append(QJsonObject{{"cx", mm(b.cx)}, {"cy", mm(b.cy)}, {"w", mm(b.w)}, {"h", mm(b.h)},
                                     {"yaw", std::round(double(b.yaw) * 1e4) / 1e4}, {"z0", mm(b.z0)}, {"z1", mm(b.z1)}});
        QJsonObject o{{"x", mm(s.x)}, {"y", mm(s.y)}, {"yaw", std::round(double(s.yaw) * 1e4) / 1e4},
                      {"weight", mm(s.weight)}, {"state", state}, {"parts", parts}};
        if (i < images.size())
        {
            o.insert("visible_frac", std::round(double(images[i].visible_frac) * 1e4) / 1e4);
            o.insert("fill_max", std::round(double(images[i].fill_max) * 1e4) / 1e4);
        }
        samples.append(o);
    }
    QJsonArray lidar;
    for (const auto& per_sample : scans)
    {
        QJsonArray sensor;
        for (const auto& sc : per_sample)
        {
            QJsonArray ring;
            std::size_t ring_skipped = 0;
            for (const float e : sc.ring_elev_room_deg)
            {
                if (std::isfinite(e)) ring.append(std::round(double(e) * 100.0) / 100.0);
                else ++ring_skipped;
            }
            QJsonArray notes;
            for (const auto& n : sc.notes) notes.append(QString::fromStdString(n));
            if (ring_skipped > 0) notes.append(QString("%1 non-finite ring elevation(s) skipped").arg(ring_skipped));
            std::size_t n_target = 0, n_other = 0, skipped = 0;
            auto on_target = points(sc.on_target, n_target, skipped);
            auto on_other = points(sc.on_other, n_other, skipped);
            if (skipped > 0) notes.append(QString("%1 non-finite points skipped").arg(skipped));
            const bool mount_ok = std::isfinite(sc.mount_xyz.x()) and std::isfinite(sc.mount_xyz.y()) and std::isfinite(sc.mount_xyz.z());
            sensor.append(QJsonObject{{"on_target", on_target}, {"on_other", on_other},
                                      {"n_total_target", int(n_target)}, {"n_total_other", int(n_other)},
                                      {"mount_xyz", mount_ok ? QJsonArray{mm(sc.mount_xyz.x()), mm(sc.mount_xyz.y()), mm(sc.mount_xyz.z())} : QJsonArray{0, 0, 0}},
                                      {"bearing_rad", std::isfinite(sc.bearing_rad) ? std::round(double(sc.bearing_rad) * 1e4) / 1e4 : 0.0},
                                      {"ring_elev_room_deg", ring}, {"lambda0", mm(sc.lambda0)},
                                      {"modelled", sc.modelled}, {"notes", notes}});
        }
        lidar.append(sensor);
    }
    QJsonArray notes;
    for (const auto& n : set.notes) notes.append(QString::fromStdString(n));
    int iw = 0, ih = 0;
    for (const auto& im : images) if (im.w > 0 and im.h > 0) { iw = im.w; ih = im.h; break; }
    const QJsonObject root{
        {"version", 1}, {"caption", QString::fromUtf8(caption.data(), int(caption.size()))},
        {"seed", QString::number(set.seed)},   // 64-bit: a string, a JSON double would round it
        {"drawn", set.drawn}, {"rejected", set.rejected}, {"outside_polygon", set.outside_polygon}, {"notes", notes},
        {"mc", QJsonObject{{"p_notice_mean", std::round(double(mc.p_notice_mean) * 1e4) / 1e4},
                           {"p_notice_sd", std::round(double(mc.p_notice_sd) * 1e4) / 1e4},
                           {"p_recognise_mean", std::round(double(mc.p_recognise_mean) * 1e4) / 1e4},
                           {"p_recognise_sd", std::round(double(mc.p_recognise_sd) * 1e4) / 1e4},
                           {"modal_limit", int(mc.modal_limit)}, {"n", mc.n}}},
        {"lidar_names", QJsonArray{QString::fromUtf8(lidar_names[0].data(), int(lidar_names[0].size())),
                                   QString::fromUtf8(lidar_names[1].data(), int(lidar_names[1].size()))}},
        {"image", QJsonObject{{"w", iw}, {"h", ih}}}, {"samples", samples}, {"lidar", lidar}};

    std::error_code ec;
    std::filesystem::create_directories(run_dir, ec);
    const QString json_path = QString::fromStdString((run_dir / "imagined.json").string());
    const QString png_path = QString::fromStdString((run_dir / "imagined_zed.png").string());
    {
        QSaveFile f(json_path);   // atomic: a reader never sees half a file
        if (not f.open(QIODevice::WriteOnly)) return false;
        const QByteArray bytes = QJsonDocument(root).toJson(QJsonDocument::Compact);
        if (f.write(bytes) != bytes.size() or not f.commit()) return false;
    }

    if (iw == 0)   // nothing rendered: no PNG, and a stale one from an earlier run must not outlive this write
    {
        QFile::remove(png_path);
        return true;
    }
    std::vector<int> vis(std::size_t(iw) * ih, 0), occ(std::size_t(iw) * ih, 0);
    int n = 0;
    for (const auto& im : images)
    {
        if (im.w != iw or im.h != ih or im.visible.size() != vis.size() or im.occluded.size() != occ.size()) continue;
        for (std::size_t p = 0; p < vis.size(); ++p) { vis[p] += im.visible[p] ? 1 : 0; occ[p] += im.occluded[p] ? 1 : 0; }
        ++n;
    }
    QImage png(iw, ih, QImage::Format_RGB888);
    png.fill(Qt::black);
    for (int y = 0; y < ih; ++y)
    {
        uchar* row = png.scanLine(y);
        for (int x = 0; x < iw; ++x)
        {
            const std::size_t p = std::size_t(y) * iw + x;
            row[3 * x + 0] = n ? uchar(std::lround(255.0 * vis[p] / n)) : 0;
            row[3 * x + 1] = n ? uchar(std::lround(255.0 * occ[p] / n)) : 0;
            row[3 * x + 2] = 0;
        }
    }
    QSaveFile pf(png_path);
    if (not pf.open(QIODevice::WriteOnly)) return false;
    return png.save(&pf, "PNG") and pf.commit();
}
}   // namespace rc::chat
