#include "mask_hover_info.h"

#include "yolo_seg_detector.h"   // SegDetection

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace rc
{

std::int64_t MaskHoverSnapshot::steady_now_ms()
{
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

MaskHoverInfo MaskHoverSnapshot::from_detection(const SegDetection& d, int index, bool whitelist_active,
                                                bool accepted)
{
    MaskHoverInfo m;
    m.index = index;
    m.label = d.label;
    m.raw_label = d.raw_label;
    m.class_id = d.class_id;
    m.confidence = d.confidence;
    if (d.second_class_id >= 0)
    {
        m.second_label = d.second_label.empty() ? std::to_string(d.second_class_id) : d.second_label;
        m.second_confidence = d.second_confidence;
    }
    m.whitelist_active = whitelist_active;
    m.accepted = accepted;
    m.bbox = d.bbox;
    if (not d.mask.empty())
    {
        m.pixel_area = cv::countNonZero(d.mask > 127);
        const cv::Rect box = d.bbox & cv::Rect(0, 0, d.mask.cols, d.mask.rows);
        if (box.area() > 0)
        {
            const double s = std::min(1.0, 128.0 / std::max(box.width, box.height));
            cv::Mat small;
            cv::resize(d.mask(box), small,
                       cv::Size(std::max(1, static_cast<int>(std::lround(box.width * s))),
                                std::max(1, static_cast<int>(std::lround(box.height * s)))),
                       0, 0, cv::INTER_AREA);
            m.roi_mask = small > 127;   // comparison allocates a new buffer: aliases nothing
            m.bbox = box;
        }
    }
    return m;
}

namespace
{
QString num(const std::optional<float>& v, int prec, const char* unit = "")
{
    return v ? QString("%1%2").arg(*v, 0, 'f', prec).arg(unit) : QStringLiteral("—");
}
}

QString MaskHoverSnapshot::tooltip_at(double rx, double ry) const
{
    if (src_size.width <= 0 or src_size.height <= 0)
        return {};
    const int x = std::clamp(static_cast<int>(rx * src_size.width), 0, src_size.width - 1);
    const int y = std::clamp(static_cast<int>(ry * src_size.height), 0, src_size.height - 1);

    std::vector<const MaskHoverInfo*> hits;
    for (const auto& m : masks)
    {
        if (m.roi_mask.empty() or not m.bbox.contains(cv::Point(x, y)))
            continue;
        const int u = std::clamp((x - m.bbox.x) * m.roi_mask.cols / std::max(1, m.bbox.width), 0, m.roi_mask.cols - 1);
        const int v = std::clamp((y - m.bbox.y) * m.roi_mask.rows / std::max(1, m.bbox.height), 0, m.roi_mask.rows - 1);
        if (m.roi_mask.at<std::uint8_t>(v, u))
            hits.push_back(&m);
    }
    if (hits.empty())
        return {};
    std::ranges::stable_sort(hits, [](const auto* a, const auto* b) { return a->pixel_area < b->pixel_area; });

    const double age_ms = static_cast<double>(steady_now_ms() - created_steady_ms);
    QStringList out;
    out << QString("── masks under cursor: %1 ──").arg(hits.size());
    for (const auto* m : hits)
    {
        out << QString("%1[%2] %3 %4").arg(QString::fromStdString(node)).arg(m->index)
                   .arg(QString::fromStdString(m->label))
                   .arg(m->raw_label.empty() or m->raw_label == m->label
                            ? QString() : QString("(raw: %1)").arg(QString::fromStdString(m->raw_label)));
        out << QString("class id %1  conf %2").arg(m->class_id).arg(m->confidence, 0, 'f', 3);
        out << (m->second_label.empty()
                    ? QStringLiteral("runner-up —")
                    : QString("runner-up %1  %2").arg(QString::fromStdString(m->second_label))
                          .arg(m->second_confidence, 0, 'f', 3));
        out << QString("accepted %1").arg(m->whitelist_active ? (m->accepted ? "yes (in accepted_labels)" : "NO")
                                                              : "yes (no whitelist)");
        out << QString("source %1  has_depth %2").arg(QString::fromStdString(source)).arg(m->has_depth ? "yes" : "no");
        out << QString("area %1 px  bbox x%2 y%3 %4x%5").arg(m->pixel_area).arg(m->bbox.x).arg(m->bbox.y)
                   .arg(m->bbox.width).arg(m->bbox.height);
        out << (m->centroid_cam
                    ? QString("centroid (%1, %2, %3) m  range %4")
                          .arg((*m->centroid_cam)[0], 0, 'f', 2).arg((*m->centroid_cam)[1], 0, 'f', 2)
                          .arg((*m->centroid_cam)[2], 0, 'f', 2).arg(num(m->range_m, 2, " m"))
                    : QStringLiteral("centroid —  range —"));
        out << QString("support pts %1  depth σ %2")
                   .arg(m->support_points ? QString::number(*m->support_points) : QStringLiteral("—"))
                   .arg(num(m->depth_sigma_cm, 1, " cm"));
        out << QString("azimuth %1").arg(num(m->azimuth_deg, 1, "°"));
        out << (m->color_rgb and m->color_neff
                    ? QString("colour rgb (%1, %2, %3)  n_eff %4")
                          .arg((*m->color_rgb)[0], 0, 'f', 2).arg((*m->color_rgb)[1], 0, 'f', 2)
                          .arg((*m->color_rgb)[2], 0, 'f', 2).arg(*m->color_neff, 0, 'f', 0)
                    : QStringLiteral("colour —"));
        out << QString("motion dotd %1  bias %2").arg(num(m->motion_dotd, 3, " m/s")).arg(num(m->motion_bias, 3, " m"));
        out << QString("       σ %1  trunc %2  c.radius %3").arg(num(m->motion_sigma_cm, 1, " cm"))
                   .arg(num(m->trunc_frac, 2)).arg(num(m->centroid_radius, 2));
    }
    out << QString("frame %1  stamp %2 ms  age %3 ms").arg(frame_id).arg(stamp_ms).arg(age_ms, 0, 'f', 0);
    return out.join('\n');
}

} // namespace rc
