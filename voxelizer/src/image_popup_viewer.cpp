#include "image_popup_viewer.h"

#include <opencv2/imgproc.hpp>
#include <QImage>
#include <QMouseEvent>
#include <QResizeEvent>
#include <QToolTip>
#include <algorithm>
#include <cmath>

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

void ImagePopupViewer::set_depth_readout(const cv::Mat& metric_log_range, bool active)
{
    depth_active_ = active and not metric_log_range.empty();
    if (depth_active_)
    {
        depth_log_range_ = metric_log_range.clone();
        setMouseTracking(true);    // fire mouseMoveEvent with no button pressed
    }
    else
    {
        depth_log_range_.release();
        setMouseTracking(false);
        QToolTip::hideText();
    }
}

void ImagePopupViewer::mouseMoveEvent(QMouseEvent* event)
{
    QLabel::mouseMoveEvent(event);
    if (not depth_active_ or depth_log_range_.empty() or last_pixmap_.isNull())
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

    const int ix = std::clamp(static_cast<int>(rx * depth_log_range_.cols), 0, depth_log_range_.cols - 1);
    const int iy = std::clamp(static_cast<int>(ry * depth_log_range_.rows), 0, depth_log_range_.rows - 1);
    const float lr = depth_log_range_.at<float>(iy, ix);

    // NaN is not "0 m", it is NO ESTIMATE — outside the elevation band the model was never asked.
    // Saying so beats printing a number that looks like a measurement.
    QToolTip::showText(event->globalPosition().toPoint(),
                       std::isfinite(lr) ? QString("%1 m").arg(std::exp(lr), 0, 'f', 2)
                                         : QStringLiteral("no depth here"),
                       this);
}

} // namespace rc
