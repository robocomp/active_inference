#include "self_projection_viewer.h"

#include <QPainter>
#include <QPainterPath>
#include <QPen>

#include <cmath>
#include <numbers>
#include <utility>

SelfProjectionViewer::SelfProjectionViewer(QWidget* parent)
    : QWidget(parent)
{
    setWindowTitle("Self-projection (RGB + FK aura)");
    setMinimumSize(320, 240);
}

SelfProjectionViewer::~SelfProjectionViewer() = default;

void SelfProjectionViewer::set_overlay(const QImage& rgb, std::vector<ProjCapsule> capsules,
                                       std::vector<QPointF> skeleton)
{
    image_         = rgb.copy();          // own the bytes (caller's buffer is reused)
    proj_capsules_ = std::move(capsules);
    skeleton_      = std::move(skeleton);
    update();
}

void SelfProjectionViewer::paintEvent(QPaintEvent* /*event*/)
{
    QPainter painter(this);
    painter.fillRect(rect(), Qt::black);

    if (image_.isNull())
    {
        painter.setPen(Qt::gray);
        painter.drawText(rect(), Qt::AlignCenter, "waiting for camera stream…");
        return;
    }

    // Letterbox-fit the frame into the widget, preserving aspect ratio.
    const QSize target = image_.size().scaled(size(), Qt::KeepAspectRatio);
    const QRect dst(QPoint((width() - target.width()) / 2, (height() - target.height()) / 2), target);
    painter.drawImage(dst, image_);

    // Map image-pixel coords -> widget coords through the SAME letterbox transform.
    const double sx = static_cast<double>(dst.width())  / image_.width();
    const double sy = static_cast<double>(dst.height()) / image_.height();
    const auto to_widget = [&](const QPointF& px)
    {
        return QPointF(dst.x() + px.x() * sx, dst.y() + px.y() * sy);
    };
    const double rscale = 0.5 * (sx + sy);   // px radius in image space -> widget space

    painter.setRenderHint(QPainter::Antialiasing, true);

    // Crisp FK skeleton (projection check): a thin polyline through the joint centers
    // + a dot at each. In a static pose this should trace the arm's centerline — any
    // offset is a calibration/projection error, independent of the capsule thickness.
    if (skeleton_.size() >= 2)
    {
        painter.setPen(QPen(QColor(255, 230, 0, 230), 1.5));   // yellow centerline
        for (std::size_t i = 1; i < skeleton_.size(); ++i)
            painter.drawLine(to_widget(skeleton_[i - 1]), to_widget(skeleton_[i]));
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(255, 80, 80, 230));            // red joint dots
        for (const auto& p : skeleton_)
            painter.drawEllipse(to_widget(p), 2.6, 2.6);
    }

    if (proj_capsules_.empty())
        return;

    // Capsule aura: each link is the TRUE silhouette of its fitted capsule — the convex
    // hull of its two end-discs = two outer tangent lines + only the two OUTER cap arcs.
    // (Drawing two FULL end-circles instead ballooned the near links and stacked into
    // "beads" at every joint.) All links union into ONE path, then simplified() collapses
    // it to just the outer boundary, so the chain reads as a single continuous contour
    // with the real arm showing through.
    painter.setRenderHint(QPainter::Antialiasing, true);

    const auto add_capsule = [](QPainterPath& path, QPointF c0, double r0, QPointF c1, double r1)
    {
        const double dx = c1.x() - c0.x(), dy = c1.y() - c0.y();
        const double d  = std::hypot(dx, dy);

        // One disc swallows the other (a foreshortened, near end-on link) ⇒ just the larger circle.
        if (d <= std::abs(r1 - r0) + 1e-3)
        {
            path.addEllipse((r0 >= r1) ? c0 : c1, std::max(r0, r1), std::max(r0, r1));
            return;
        }

        const double g   = std::atan2(dy, dx);                              // center-line angle
        const double psi = std::acos(std::clamp((r1 - r0) / d, -1.0, 1.0)); // tangent normal offset
        const auto   dir = [](double a){ return QPointF(std::cos(a), std::sin(a)); };
        constexpr int CAP_SEG = 10;

        path.moveTo(c0 + r0 * dir(g + psi));                 // tangent line a
        path.lineTo(c1 + r1 * dir(g + psi));
        for (int i = 1; i <= CAP_SEG; ++i)                   // c1 cap: g+psi → g-psi, the near way (through +u)
            path.lineTo(c1 + r1 * dir((g + psi) - (2.0 * psi) * (double(i) / CAP_SEG)));
        path.lineTo(c0 + r0 * dir(g - psi));                 // tangent line b
        for (int i = 1; i <= CAP_SEG; ++i)                   // c0 cap: g-psi → g+psi, the far way (through -u)
            path.lineTo(c0 + r0 * dir((g - psi) - (2.0 * std::numbers::pi - 2.0 * psi) * (double(i) / CAP_SEG)));
        path.closeSubpath();
    };

    QPainterPath aura;
    aura.setFillRule(Qt::WindingFill);
    for (const auto& pc : proj_capsules_)
        add_capsule(aura, to_widget(pc.a), pc.ra * rscale, to_widget(pc.b), pc.rb * rscale);
    aura = aura.simplified();   // union of all links → outer contour only (drops internal joint arcs)

    painter.setPen(QPen(QColor(0, 220, 255, 210), 1.6));   // crisp cyan contour
    painter.setBrush(QColor(0, 200, 255, 28));             // faint translucent body — arm shows through
    painter.drawPath(aura);
}
