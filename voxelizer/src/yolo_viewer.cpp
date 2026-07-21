#include "yolo_viewer.h"
#include "yolo_processor.h"

#include <opencv2/imgproc.hpp>
#include <QImage>
#include <QResizeEvent>
#include <QMouseEvent>
#include <QToolTip>

#include <algorithm>

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
    setMouseTracking(true);   // fire mouseMoveEvent without a pressed button (for the hover readout)
}

// ─── Rendering ───────────────────────────────────────────────────────────────
QPixmap YoloViewer::render_frame(const cv::Mat& rgb,
                                 const std::vector<SegDetection>& detections) const
{
    cv::Mat canvas = rgb.clone();   // RGB, 8UC3

    for (const auto& det : detections)
    {
        // YOLO-sem-derived furniture masks use a class_id offset of 1000 (SemanticMaskStage) so they are
        // clear of the COCO range. Render them DISTINCTLY from the YOLO-seg masks: a fixed bright cyan, a
        // thicker box, and a "sem:" label prefix — so it's obvious what the semantic branch contributes.
        const bool is_semantic = det.class_id >= 1000;
        const cv::Vec3b color = is_semantic ? cv::Vec3b{0, 229, 255} : class_color(det.class_id);
        const cv::Scalar color_s(color[0], color[1], color[2]);
        const int box_thickness = is_semantic ? 3 : 2;

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
        cv::rectangle(canvas, det.bbox, color_s, box_thickness, cv::LINE_AA);

        // ── Label chip ───────────────────────────────────────────────────────
        const std::string text = (is_semantic ? "sem:" : "") + det.label + " " + cv::format("%.2f", det.confidence);
        int baseline = 0;
        const cv::Size ts = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);
        const cv::Point tl = det.bbox.tl();
        const cv::Point chip_tl(tl.x, std::max(0, tl.y - ts.height - 6));
        const cv::Point chip_br(tl.x + ts.width + 4, tl.y);
        cv::rectangle(canvas, chip_tl, chip_br, color_s, cv::FILLED);
        cv::putText(canvas, text, cv::Point(tl.x + 2, std::max(ts.height, tl.y - 4)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }

    // ── FPS overlay (top-left, green on a dark chip for legibility) ───────────
    const std::string fps_text = cv::format("%.1f FPS", fps_ema_);
    int fps_baseline = 0;
    const cv::Size fts = cv::getTextSize(fps_text, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &fps_baseline);
    cv::rectangle(canvas, cv::Point(6, 6),
                  cv::Point(6 + fts.width + 10, 6 + fts.height + 12),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(canvas, fps_text, cv::Point(11, 6 + fts.height + 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);  // RGB → green

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

    // Update the display-rate EMA from the inter-call interval (drawn by render_frame).
    const auto now = std::chrono::steady_clock::now();
    if (last_frame_time_.time_since_epoch().count() != 0)
    {
        const double dt_ms = std::chrono::duration<double, std::milli>(now - last_frame_time_).count();
        if (dt_ms > 0.0)
        {
            const float inst = static_cast<float>(1000.0 / dt_ms);
            fps_ema_ = (fps_ema_ > 0.f) ? (0.9f * fps_ema_ + 0.1f * inst) : inst;
        }
    }
    last_frame_time_ = now;

    last_pixmap_ = render_frame(rgb, detections);
    setPixmap(last_pixmap_.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void YoloViewer::resizeEvent(QResizeEvent* event)
{
    QLabel::resizeEvent(event);
    if (!last_pixmap_.isNull())
        setPixmap(last_pixmap_.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void YoloViewer::update_semantic(const cv::Mat& labels, bool active)
{
    semantic_active_ = active;
    // Keep our own copy: the source map is recomputed in place each (decimated) cycle.
    if (active and not labels.empty())
        semantic_labels_ = labels.clone();
    else
        semantic_labels_.release();
}

void YoloViewer::mouseMoveEvent(QMouseEvent* event)
{
    QLabel::mouseMoveEvent(event);

    if (!semantic_active_ or semantic_labels_.empty() or last_pixmap_.isNull())
    {
        QToolTip::hideText();
        return;
    }

    // The pixmap is shown scaled-to-fit (KeepAspectRatio) and centred (AlignCenter), so undo that
    // letterbox transform to recover the image pixel under the cursor.
    const QSize shown = last_pixmap_.size().scaled(size(), Qt::KeepAspectRatio);
    const int off_x = (width()  - shown.width())  / 2;
    const int off_y = (height() - shown.height()) / 2;

    const QPoint p = event->position().toPoint();
    const double rx = static_cast<double>(p.x() - off_x) / shown.width();
    const double ry = static_cast<double>(p.y() - off_y) / shown.height();
    if (rx < 0.0 or rx >= 1.0 or ry < 0.0 or ry >= 1.0)   // cursor in the letterbox margin
    {
        QToolTip::hideText();
        return;
    }

    const int ix = std::clamp(static_cast<int>(rx * semantic_labels_.cols), 0, semantic_labels_.cols - 1);
    const int iy = std::clamp(static_cast<int>(ry * semantic_labels_.rows), 0, semantic_labels_.rows - 1);
    const int id = semantic_labels_.at<unsigned char>(iy, ix);

    QString text;
    if (id >= 0 and id < static_cast<int>(class_names_.size()))
        text = QString::fromStdString(class_names_[static_cast<std::size_t>(id)]);
    else
        text = QStringLiteral("(unlabelled)");   // IGNORE_LABEL (255) / below confidence

    QToolTip::showText(event->globalPosition().toPoint(), text, this);
}

} // namespace rc
