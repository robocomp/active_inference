#include "yolo_viewer.h"
#include "yolo_processor.h"

#include <opencv2/imgproc.hpp>
#include <QImage>
#include <QResizeEvent>

namespace rc
{

// ─── Deterministic 10-colour palette (RGB order) ─────────────────────────────
cv::Vec3b YoloViewer::class_color(int class_id)
{
    static const std::array<cv::Vec3b, 10> palette = {{
        {255,  56,  56}, {255, 157, 151}, {255, 112,  31}, {255, 178,  29},
        {207, 210,  49}, { 72, 249,  10}, {146, 204,  23}, { 61, 219, 134},
        { 26, 147,  52}, {  0, 212, 187}
    }};
    return palette[static_cast<std::size_t>(class_id) % palette.size()];
}

// ─── Construction ─────────────────────────────────────────────────────────────
YoloViewer::YoloViewer(QWidget* parent)
    : QLabel(parent)
{
    setAlignment(Qt::AlignCenter);
    setMinimumSize(320, 240);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    setText("Waiting for YOLO frame…");
}

// ─── Rendering ───────────────────────────────────────────────────────────────
QPixmap YoloViewer::render_frame(const cv::Mat& rgb,
                                 const std::vector<SegDetection>& detections) const
{
    cv::Mat canvas = rgb.clone();   // RGB, 8UC3

    for (const auto& det : detections)
    {
        const cv::Vec3b color = class_color(det.class_id);
        const cv::Scalar color_s(color[0], color[1], color[2]);

        // ── Mask overlay (50 % alpha blend in mask region) ────────────────────
        if (!det.mask.empty())
        {
            cv::Mat mask;
            if (det.mask.size() != canvas.size())
                cv::resize(det.mask, mask, canvas.size(), 0, 0, cv::INTER_NEAREST);
            else
                mask = det.mask;

            // Blend only inside the mask: pixel = 0.5 * orig + 0.5 * color
            for (int r = 0; r < canvas.rows; ++r)
            {
                const uint8_t* mr  = mask.ptr<uint8_t>(r);
                const cv::Vec3b* sr = rgb.ptr<cv::Vec3b>(r);
                cv::Vec3b*       dr = canvas.ptr<cv::Vec3b>(r);
                for (int c = 0; c < canvas.cols; ++c)
                {
                    if (mr[c] == 0) continue;
                    dr[c][0] = static_cast<uint8_t>(sr[c][0] / 2 + color[0] / 2);
                    dr[c][1] = static_cast<uint8_t>(sr[c][1] / 2 + color[1] / 2);
                    dr[c][2] = static_cast<uint8_t>(sr[c][2] / 2 + color[2] / 2);
                }
            }
        }

        // ── Bounding box ──────────────────────────────────────────────────────
        cv::rectangle(canvas, det.bbox, color_s, 2, cv::LINE_AA);

        // ── Label chip ───────────────────────────────────────────────────────
        const std::string text = det.label + " " + cv::format("%.2f", det.confidence);
        int baseline = 0;
        const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
        const cv::Point tl = det.bbox.tl();
        const cv::Point chip_tl(tl.x, std::max(0, tl.y - ts.height - 6));
        const cv::Point chip_br(tl.x + ts.width + 4, tl.y);
        cv::rectangle(canvas, chip_tl, chip_br, color_s, cv::FILLED);
        cv::putText(canvas, text, cv::Point(tl.x + 2, std::max(ts.height, tl.y - 4)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    // ── Convert RGB cv::Mat → QImage (no copy) → QPixmap ─────────────────────
    const QImage qimg(canvas.data, canvas.cols, canvas.rows,
                      static_cast<int>(canvas.step),
                      QImage::Format_RGB888);
    return QPixmap::fromImage(qimg.copy());  // .copy() detaches from canvas memory
}

void YoloViewer::update_frame(const cv::Mat& rgb,
                               const std::vector<SegDetection>& detections)
{
    if (rgb.empty())
        return;
    last_pixmap_ = render_frame(rgb, detections);
    setPixmap(last_pixmap_.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void YoloViewer::resizeEvent(QResizeEvent* event)
{
    QLabel::resizeEvent(event);
    if (!last_pixmap_.isNull())
        setPixmap(last_pixmap_.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

} // namespace rc
