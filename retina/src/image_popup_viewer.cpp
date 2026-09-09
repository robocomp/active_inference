#include "image_popup_viewer.h"

#include "yolo_semantic.h"   // rc::semantic::SemanticMap (graded posteriors)

#include <opencv2/imgproc.hpp>
#include <QImage>
#include <QMouseEvent>
#include <QResizeEvent>
#include <QStringList>
#include <QToolTip>
#include <algorithm>
#include <cmath>
#include <limits>

namespace rc
{

ImagePopupViewer::ImagePopupViewer(QWidget* parent)
    : QLabel(parent)
{
    setAlignment(Qt::AlignCenter);
    setMinimumSize(320, 160);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    setText("Waiting for Ricoh 360 frame…");
}

void ImagePopupViewer::update_image(const cv::Mat& bgr)
{
    if (bgr.empty() || bgr.type() != CV_8UC3)
        return;

    // Display-rate EMA from the inter-call interval.
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

    // BGR → RGB for Qt, with an FPS chip drawn on a copy.
    cv::Mat canvas;
    cv::cvtColor(bgr, canvas, cv::COLOR_BGR2RGB);
    const std::string fps_text = cv::format("%.1f FPS  %dx%d", fps_ema_, canvas.cols, canvas.rows);
    int baseline = 0;
    const cv::Size ts = cv::getTextSize(fps_text, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
    cv::rectangle(canvas, cv::Point(6, 6), cv::Point(6 + ts.width + 10, 6 + ts.height + 12),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(canvas, fps_text, cv::Point(11, 6 + ts.height + 4),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

    const QImage qimg(canvas.data, canvas.cols, canvas.rows,
                      static_cast<int>(canvas.step), QImage::Format_RGB888);
    last_pixmap_ = QPixmap::fromImage(qimg.copy());   // .copy() detaches from canvas memory
    setPixmap(last_pixmap_.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void ImagePopupViewer::resizeEvent(QResizeEvent* event)
{
    QLabel::resizeEvent(event);
    if (!last_pixmap_.isNull())
        setPixmap(last_pixmap_.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void ImagePopupViewer::set_depth_readout(const cv::Mat& room_log_range,
                                         const cv::Mat& model_log_range, bool active)
{
    depth_active_ = active and (not room_log_range.empty() or not model_log_range.empty());
    if (depth_active_)
    {
        // clone() on an empty Mat yields an empty Mat, so the "only one field present" case needs
        // no special casing here — mouseMoveEvent asks each field whether it exists.
        room_log_range_  = room_log_range.clone();
        model_log_range_ = model_log_range.clone();
    }
    else
    {
        room_log_range_.release();
        model_log_range_.release();
        QToolTip::hideText();
    }
    sync_mouse_tracking();
}

ImagePopupViewer::~ImagePopupViewer() = default;

void ImagePopupViewer::sync_mouse_tracking()
{
    setMouseTracking(depth_active_ or probs_active_);   // fire mouseMoveEvent with no button pressed
}

void ImagePopupViewer::set_prob_readout(const rc::semantic::SemanticMap& map, bool active)
{
    probs_active_ = active and map.graded();
    if (probs_active_)
    {
        auto owned = std::make_unique<rc::semantic::SemanticMap>();
        owned->prob_class_ids = map.prob_class_ids;
        owned->probs_src_size = map.probs_src_size;
        owned->probs.reserve(map.probs.size());
        for (const auto& plane : map.probs)
            owned->probs.push_back(plane.empty() ? cv::Mat{} : plane.clone());
        probs_map_ = std::move(owned);
    }
    else
    {
        probs_map_.reset();
        QToolTip::hideText();
    }
    sync_mouse_tracking();
}

void ImagePopupViewer::mouseMoveEvent(QMouseEvent* event)
{
    QLabel::mouseMoveEvent(event);
    const bool want_depth = depth_active_ and (not room_log_range_.empty()
                                               or not model_log_range_.empty());
    const bool want_probs = probs_active_ and probs_map_ and probs_map_->graded();
    if ((not want_depth and not want_probs) or last_pixmap_.isNull())
    {
        QToolTip::hideText();
        return;
    }

    // The pixmap is shown scaled-to-fit (KeepAspectRatio) and centred, so undo that letterbox
    // transform to recover the panorama pixel under the cursor. Same maths as the ZED window's
    // semantic hover readout (yolo_viewer.cpp) — keep the two in step if either changes.
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

    // Each field is sampled through its OWN dimensions, so the two need not share a resolution —
    // the envelope is ray-cast at panorama size while the model's map may be decimated.
    const auto sample = [rx, ry](const cv::Mat& f) -> float
    {
        if (f.empty())
            return std::numeric_limits<float>::quiet_NaN();
        const int ix = std::clamp(static_cast<int>(rx * f.cols), 0, f.cols - 1);
        const int iy = std::clamp(static_cast<int>(ry * f.rows), 0, f.rows - 1);
        const float lr = f.at<float>(iy, ix);
        return std::isfinite(lr) ? std::exp(lr) : std::numeric_limits<float>::quiet_NaN();
    };
    const float room  = sample(room_log_range_);
    const float model = sample(model_log_range_);

    // NaN is not "0 m", it is NO ESTIMATE — outside the elevation band the model was never asked, and
    // outside the envelope the ray escapes the room. Saying so beats printing a number that looks
    // like a measurement. A field that was never HANDED to us is simply not mentioned.
    QStringList lines;

    if (want_probs)
    {
        const cv::Size fs = probs_map_->probs_src_size;
        const int ix = std::clamp(static_cast<int>(rx * fs.width),  0, std::max(0, fs.width  - 1));
        const int iy = std::clamp(static_cast<int>(ry * fs.height), 0, std::max(0, fs.height - 1));
        const auto ranked = probs_map_->probs_at(ix, iy);
        for (const auto& [id, prob] : ranked)
        {
            const QString name = (id >= 0 and id < static_cast<int>(class_names_.size()))
                                     ? QString::fromStdString(class_names_[static_cast<std::size_t>(id)])
                                     : QString("class %1").arg(id);
            lines << (std::isnan(prob) ? QString("P(%1)  —  not looked at").arg(name)
                                       : QString("P(%1)  %2").arg(name).arg(prob, 0, 'f', 3));
        }
        if (ranked.size() >= 2 and not std::isnan(ranked[0].second) and not std::isnan(ranked[1].second))
            lines << QString("margin %1").arg(ranked[0].second - ranked[1].second, 0, 'f', 3);
    }

    if (want_depth and not room_log_range_.empty())
        lines << (std::isfinite(room) ? QString("room   %1 m").arg(room, 0, 'f', 2)
                                      : QStringLiteral("room   —"));
    if (want_depth and not model_log_range_.empty())
        lines << (std::isfinite(model) ? QString("model  %1 m").arg(model, 0, 'f', 2)
                                       : QStringLiteral("model  —"));
    // Sign matches compose_difference (model − reference): + ⇒ the model reads FARTHER than the room
    // believes, which is the same thing the red channel of the DIFF overlay is showing.
    if (want_depth and std::isfinite(room) and std::isfinite(model))
        lines << QString("Δ      %1%2 m").arg(model - room >= 0.f ? "+" : "")
                                         .arg(model - room, 0, 'f', 2);

    if (lines.isEmpty())
        QToolTip::hideText();
    else
        QToolTip::showText(event->globalPosition().toPoint(), lines.join('\n'), this);
}

} // namespace rc
