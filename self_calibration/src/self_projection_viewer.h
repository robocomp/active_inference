#pragma once

#include <QImage>
#include <QPointF>
#include <QWidget>

#include <vector>

// Self-projection viewer (display-only).
//
// The self_calibration agent already holds the RGB frame, the FK link poses, the
// arm→camera extrinsic and the intrinsics, so it computes the projected per-link
// capsule "aura" itself and pushes (image + capsules) here each compute() cycle.
// This widget only paints — no DDS, no FK, no threads. The agent's compute() and
// this widget run on the same Qt thread, so set_overlay() is a direct call.
// Registered with add_custom_widget_in_own_window so its repaints live in their own
// backing store, isolated from the graph view's churn repaint storm.
class SelfProjectionViewer : public QWidget
{
    Q_OBJECT
public:
    // One projected per-link capsule: endpoints a,b and their perspective radii
    // ra,rb in IMAGE-pixel coords (mapped to widget coords in paintEvent through the
    // same letterbox transform as the image).
    struct ProjCapsule
    {
        QPointF a, b;
        double  ra = 0.0, rb = 0.0;
    };

    explicit SelfProjectionViewer(QWidget* parent = nullptr);
    ~SelfProjectionViewer() override;

    // Push the latest frame + overlay (called from the agent, same thread). The
    // image is deep-copied so the widget owns its pixels. `capsules` is the fat
    // translucent aura (the model); `skeleton` is the crisp FK joint-center polyline
    // (base→joint_1..7→tool, image px) drawn on top — the clean projection check in a
    // static pose (it should trace the arm's centerline). Either may be empty.
    void set_overlay(const QImage& rgb, std::vector<ProjCapsule> capsules,
                     std::vector<QPointF> skeleton);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QImage                   image_;
    std::vector<ProjCapsule> proj_capsules_;
    std::vector<QPointF>     skeleton_;   // FK joint-center chain, image-pixel coords
};
