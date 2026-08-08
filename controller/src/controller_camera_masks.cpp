/*
 *  controller_camera_masks.cpp — see controller_camera_masks.h
 */

#include "controller_camera_masks.h"

#include <QFont>
#include <QFontMetrics>
#include <QPainter>
#include <QPen>

#include <algorithm>
#include <print>
#include <array>
#include <cctype>
#include <cmath>
#include <span>

#include <Eigen/Geometry>
#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include "../../common/mask_ingestor/mask_ingestor.h"
#include "../../common/media_transport/media_transport.h"

namespace rc
{
namespace
{

// Recompose no faster than this. The control loop runs at 20 Hz and the camera at ~30, but a human
// reading silhouettes gains nothing past ~6 Hz and every recompose is a full-frame paint.
constexpr std::uint64_t kComposePeriodMs = 160;
constexpr std::uint64_t kDiscoveryPeriodMs = 1000;
// Past this, the masks node is stale rather than merely quiet. Only used to LABEL the view — nothing
// here gates behaviour, so it is a readout threshold, not a model one.
constexpr std::int64_t kMasksStaleMs = 1500;
// Time-alignment window. Frames older than this cannot be the one a live mask set was computed from,
// and holding more of them only costs memory: at 30 Hz this is ~18 frames, which covers any plausible
// capture→YOLO→graph latency with room to spare.
// 400 ms at 30 Hz is ~12 frames, comfortably longer than any capture→YOLO→graph latency, and the
// bound matters: these are FULL-RESOLUTION frames (~2.7 MB each at 720p) because the mask pixels index
// the full image, so the ring is tens of megabytes. It only exists while the affordance window is open.
constexpr std::uint64_t kRingSpanMs = 400;
constexpr std::size_t   kRingMaxFrames = 12;
// Past this the pairing is a guess rather than a match, and the view says so instead of pretending.
constexpr std::uint64_t kAlignTolMs = 60;

// Distinct outline per instance, in first-seen order. Deliberately NOT keyed on the label: two chairs
// must be distinguishable, and that is the whole question when an affordance goes to look at one.
const QColor kPalette[] = {QColor(0x4a, 0xa3, 0xe0), QColor(0xf1, 0xc4, 0x0f), QColor(0x9b, 0x59, 0xb6),
                           QColor(0x1a, 0xbc, 0x9c), QColor(0xe6, 0x7e, 0x22), QColor(0xec, 0x87, 0xc0),
                           QColor(0x95, 0xa5, 0xa6), QColor(0x2e, 0xcc, 0x71)};
constexpr std::size_t kPaletteSize = sizeof(kPalette) / sizeof(kPalette[0]);

std::string normalise(std::string s)
{
    std::string out;
    out.reserve(s.size());
    for (const char c : s)
        if (std::isalpha(static_cast<unsigned char>(c)))
            out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    return out;
}

QImage qimage_from_media(const std::uint8_t *data, int w, int h, int step, std::uint32_t fmt)
{
    if (data == nullptr or w <= 0 or h <= 0)
        return {};
    // Deep-copied on every path (.copy()/.rgbSwapped() both allocate): the source is a loaned view into
    // the SHM segment and is invalid the moment the poll callback returns.
    switch (fmt)
    {
        case rc::media::FORMAT_RGB8:  return QImage(data, w, h, step, QImage::Format_RGB888).copy();
        case rc::media::FORMAT_BGR8:  return QImage(data, w, h, step, QImage::Format_RGB888).rgbSwapped();
        case rc::media::FORMAT_GRAY8: return QImage(data, w, h, step, QImage::Format_Grayscale8)
                                                 .convertToFormat(QImage::Format_RGB888);
        default: return {};
    }
}

}   // namespace

// See the declaration for why this is shared rather than local to the composer.
bool mask_label_matches_object(const std::string &object, const std::string &label)
{
    if (object.empty() or label.empty()) return false;
    const std::string o = normalise(object), l = normalise(label);
    if (o.empty() or l.empty()) return false;
    return o.find(l) != std::string::npos or l.find(o) != std::string::npos;
}

ControllerCameraMasks::ControllerCameraMasks(std::shared_ptr<DSR::DSRGraph> graph, std::string camera_node)
    : graph_(std::move(graph)), camera_node_(std::move(camera_node))
{
    // Graph-side only. The DDS subscriber comes up in pump(), from the Operating control thread.
    if (graph_)
    {
        masks_ = std::make_unique<rc::MaskIngestor>(graph_);
        inner_ = graph_->get_inner_eigen_api();
    }
}

bool ControllerCameraMasks::ensure_intrinsics()
{
    if (fx_ > 0.f and fy_ > 0.f)
        return true;
    if (not graph_)
        return false;
    const auto node = graph_->get_node(camera_node_);
    if (not node.has_value())
        return false;
    // The DEPTH intrinsics, not the rgb ones — matching the voxelizer, which computes mask pixels on
    // the RGBD frame these describe. Using a different pair would put the projected box in a slightly
    // different image from the silhouettes it is meant to be compared against, which is the one error
    // this overlay must not make.
    const auto fx = graph_->get_attrib_by_name<cam_depth_focalx_att>(node.value());
    const auto fy = graph_->get_attrib_by_name<cam_depth_focaly_att>(node.value());
    if (not fx.has_value() or not fy.has_value())
        return false;
    fx_ = static_cast<float>(fx.value());
    fy_ = static_cast<float>(fy.value());
    return fx_ > 0.f and fy_ > 0.f;
}

bool ControllerCameraMasks::draw_target_pose(QPainter &p, const std::string &object,
                                             const std::optional<ControllerStandpoint> &standpoint,
                                             const QSize &canvas, std::uint64_t stamp_ms, QString &note)
{
    if ((object.empty() and not standpoint.has_value()) or not graph_ or not inner_)
        return false;
    if (not ensure_intrinsics())
    {
        note = QStringLiteral("no zed intrinsics — cannot project the target pose");
        return false;
    }
    // PINNED TO THE CAPTURE STAMP, always. ts==0 would take InnerEigenAPI's unlocked cache path, which
    // is single-thread-only, and this runs on the control thread; it would also answer with the pose
    // NOW rather than the pose when the shutter opened, which for a moving base is exactly the error
    // the overlay exists to reveal. No stamp ⇒ no overlay, and say so.
    if (stamp_ms == 0)
    {
        note = QStringLiteral("masks carry no capture stamp — target pose not projected");
        return false;
    }
    const auto node = object.empty() ? std::nullopt : graph_->get_node(object);
    if (not object.empty() and not node.has_value())
        note = QStringLiteral("object node '%1' is not in the graph")
                   .arg(QString::fromStdString(object));

    // ── THE CHAIN HAS TO BE BUILT IN TWO PIECES, WITH TWO DIFFERENT TIME QUERIES ──────────────────
    // ★THIS IS WHY THE MARKER LANDED IN THE WRONG PLACE. Asking for zed←object in ONE call pins the
    // WHOLE chain to the capture stamp, and that chain contains the rigid robot→zed camera mount,
    // which carries only its bootstrap timestamp — a per-frame Nearest query against it is answered
    // from a single ancient entry, or not at all. The voxelizer composes room_T_zed the same way and
    // says so in as many words (SceneProcessor::room_T_zed_extrapolated / get_room_zed_transform);
    // this overlay has to agree with the producer whose mask pixels it is drawing on, or the model and
    // the measurement are quoted in two different frames and comparing them is meaningless.
    //   room←robot  : DYNAMIC, pinned to the capture stamp (ts!=0, no cache, thread-safe)
    //   robot←zed   : STATIC mount, "latest" (ts==0), resolved once and cached here
    //   room←object : DYNAMIC, pinned to the capture stamp
    if (room_name_.empty() or robot_name_.empty())
    {
        note = QStringLiteral("room/robot frame names not known yet");
        return false;
    }
    if (not robot_T_zed_.has_value())
    {
        if (const auto m = inner_->get_transformation_matrix(robot_name_, camera_node_, 0); m.has_value())
            robot_T_zed_ = m.value();
        else
        {
            note = QStringLiteral("no static %1←%2 mount transform")
                       .arg(QString::fromStdString(robot_name_), QString::fromStdString(camera_node_));
            return false;
        }
    }
    // ALWAYS check the optional — get_transformation_matrix bails to {} at every missing node/edge in
    // the chain, and dereferencing that is the documented crash mode.
    const auto room_T_robot = inner_->get_transformation_matrix(room_name_, robot_name_, stamp_ms);
    if (not room_T_robot.has_value())
    {
        note = QStringLiteral("no %1←%2 RT at the capture stamp")
                   .arg(QString::fromStdString(room_name_), QString::fromStdString(robot_name_));
        return false;
    }
    // ONE camera pose, both marks. The object is quoted in its own frame and the standpoint in the
    // room's, so the shared quantity is cam←room; deriving each from its own chain would let the two
    // marks disagree about where the camera was, which is the only thing they must agree on.
    const Eigen::Affine3d cam_T_room = (room_T_robot.value() * robot_T_zed_.value()).inverse();
    const auto room_T_obj = object.empty()
                                ? std::nullopt
                                : inner_->get_transformation_matrix(room_name_, object, stamp_ms);
    if (not object.empty() and not room_T_obj.has_value() and note.isEmpty())
        note = QStringLiteral("no %1←%2 RT at the capture stamp")
                   .arg(QString::fromStdString(room_name_), QString::fromStdString(object));

    constexpr double kNearY = 0.05;   // 5 cm: nearer than this the projection diverges
    const double cx = 0.5 * canvas.width(), cy = 0.5 * canvas.height();
    // zed frame: x right, y DEPTH, z up (CLAUDE.md / ROBOT_GEOMETRY.md). Exactly the inverse of the
    // voxelizer's unprojection, so a point it turned into a mask pixel lands back on that pixel.
    // Takes a point ALREADY IN CAMERA COORDINATES: the two marks live in different frames and each
    // brings its own transform, so the projection itself must not assume either one.
    const auto project = [&](const Eigen::Vector3d &c, QPointF &out) -> bool
    {
        if (c.y() < kNearY) return false;
        out = QPointF(cx + fx_ * c.x() / c.y(), cy - fy_ * c.z() / c.y());
        return true;
    };

    // MAGENTA, and nothing else in this image is magenta. The silhouettes are the MEASUREMENT and this
    // is the MODEL; drawn in related colours, the one comparison the overlay exists for would be the
    // one it made hardest.
    const QColor kModel(0xff, 0x4d, 0xd2);
    p.setBrush(Qt::NoBrush);

    // ★EVERY SIZE BELOW IS IN CANVAS PIXELS, AND THE CANVAS IS NOT WHAT YOU LOOK AT. This composes on
    // the full camera frame (720 rows) and the panel then scales it into a ~170-row peek — a factor of
    // four. A 2.5 px pen arrives as 0.6 px and a 16 px cross as 4: drawn correctly, and invisible. So
    // the overlay is dimensioned for the DISPLAYED size and multiplied up, which keeps it legible
    // whatever the camera resolution and whatever the panel is scaled to.
    const double k = std::max(1.0, canvas.height() / 180.0);

    // ── AND IT MUST NOT SWALLOW THE IMAGE UP CLOSE ────────────────────────────────────────────────
    // Everything projected from METRES grows as 1/range, so the axes and the stick that read nicely at
    // three metres span the whole frame at half a metre — exactly when the robot is doing the thing
    // worth watching. Every projected segment is therefore clipped to a fraction of the frame: the
    // DIRECTION it points is the information, and past a certain length that direction is already said.
    const double kMaxSeg = 0.13 * canvas.width();
    const auto shortened = [&](const QPointF &from, const QPointF &to)
    {
        const QPointF d = to - from;
        const double len = std::hypot(d.x(), d.y());
        return len <= kMaxSeg or len < 1e-6 ? to : from + d * (kMaxSeg / len);
    };

    const double kArm = 18.0 * k;
    QFont f = p.font();
    f.setBold(true);
    f.setPointSizeF(std::max(9.0, canvas.height() / 22.0));
    p.setFont(f);
    // Notes accumulate: two independent marks can fail for two independent reasons, and a single slot
    // that the second overwrites hides the first.
    const auto add_note = [&note](const QString &s)
    { note = note.isEmpty() ? s : note + QStringLiteral(" · ") + s; };
    const auto off_frame = [&canvas](const QPointF &px)
    { return px.x() < 0 or px.y() < 0 or px.x() >= canvas.width() or px.y() >= canvas.height(); };

    bool drew = false;

    // ── 1. THE OBJECT'S BELIEF: A STICK AND ITS AXES, ON THE OBJECT'S VERTICAL ────────────────────
    // ★THIS IS WHY THE BOX AND THE CIRCLE DREW NOTHING. A furniture node's RT origin sits at the FLOOR
    // (the base-origin convention — see the z_lo note in the voxelizer's build_graph_object_box), the
    // zed is mounted about 1.5 m up, and row = cy − fy·z/y. For an object 1 m away that puts the origin
    // roughly fy·1.5 pixels BELOW the image — comfortably off the bottom of a 720-row frame. Every
    // marker anchored at the origin was being drawn correctly, off-screen.
    // So the mark climbs the object's own vertical axis to where the object actually is in frame, and
    // the STICK from the base to it is drawn too: the stick says where the belief is standing, the axes
    // ride at eye height where they can be seen, and neither pretends the other is unnecessary.
    // NO CROSS HERE. The cross means "the affordance is aimed at this" and the object is not what the
    // affordance is aimed AT — it is what the aim is FOR. The axes already meet at the anchor, so the
    // belief's position is marked without borrowing the mark that belongs to the standpoint.
    if (node.has_value() and room_T_obj.has_value())
    {
        const Eigen::Affine3d cam_T_obj = cam_T_room * room_T_obj.value();
        const auto w = graph_->get_attrib_by_name<width_m_att>(node.value());
        const auto d = graph_->get_attrib_by_name<depth_m_att>(node.value());
        const auto h = graph_->get_attrib_by_name<height_m_att>(node.value());
        // Where to float it: the object's MID-HEIGHT. The top was wrong in the obvious way — it put the
        // marker above a 1.8 m fridge and hovering off the edge of a table, reading as "not in place"
        // when the pose was fine. Mid-height is where the object visually is, for a low table and a tall
        // box alike, and it is still well clear of the floor origin that was projecting off the bottom
        // of the frame. Never zero: that is the original off-screen failure.
        const double z_obj = (h.has_value() and h.value() > 0.05f) ? static_cast<double>(h.value()) : 1.0;
        const double z_mid = 0.5 * z_obj;
        const auto obj_px = [&](double lx, double ly, double lz, QPointF &out)
        { return project(cam_T_obj * Eigen::Vector3d(lx, ly, lz), out); };

        QPointF base_px, top_px;
        const bool have_base = obj_px(0.0, 0.0, 0.0, base_px);
        const bool have_top  = obj_px(0.0, 0.0, z_mid, top_px);
        if (not have_base and not have_top)
            add_note(QStringLiteral("the object is BEHIND the camera — the robot is not looking at it"));
        else
        {
            const QPointF anchor = have_top ? top_px : base_px;
            // The vertical stick down toward the base. Drawn thin: it is context, not the mark.
            if (have_base and have_top)
            {
                p.setPen(QPen(kModel, 1.6 * k, Qt::DashLine));
                p.drawLine(top_px, shortened(top_px, base_px));
            }
            // The object's own x/y axes through the anchor, half-extent long, projected. These carry the
            // believed ORIENTATION and the foreshortening that says how obliquely the robot is looking
            // at it. The +x arm is thicker, so which way the object thinks it faces is readable.
            const double ax = (w.has_value() and w.value() > 0.f) ? 0.5 * static_cast<double>(w.value()) : 0.0;
            const double ay = (d.has_value() and d.value() > 0.f) ? 0.5 * static_cast<double>(d.value()) : 0.0;
            const auto axis = [&](double lx, double ly, double width_px)
            {
                QPointF tip;
                if (not obj_px(lx, ly, have_top ? z_mid : 0.0, tip)) return;
                p.setPen(QPen(kModel, width_px * k));
                p.drawLine(anchor, shortened(anchor, tip));
            };
            if (ax > 0.0) { axis(+ax, 0.0, 3.0); axis(-ax, 0.0, 1.4); }
            if (ay > 0.0) { axis(0.0, +ay, 1.4); axis(0.0, -ay, 1.4); }

            p.setPen(kModel);
            const Eigen::Vector3d o_cam = cam_T_obj * Eigen::Vector3d::Zero();
            p.drawText(anchor + QPointF(kArm + 6, -6),
                       QStringLiteral("%1 @%2 m").arg(QString::fromStdString(object))
                           .arg(o_cam.norm(), 0, 'f', 2));
            drew = true;
            // OFF-FRAME IS NOT THE SAME AS ABSENT: a marker drawn at row 1400 of a 720-row image is
            // working perfectly and looks identical to one that never ran. Say where it went, so
            // "nothing on the image" can never mean two different things.
            if (off_frame(anchor))
                add_note(QStringLiteral("the object projects OFF-FRAME at (%1,%2) of %3x%4")
                             .arg(anchor.x(), 0, 'f', 0).arg(anchor.y(), 0, 'f', 0)
                             .arg(canvas.width()).arg(canvas.height()));
        }
    }

    // ── 2. THE CROSS: THE AFFORDANCE'S TARGET, WHICH IS THE STANDPOINT ────────────────────────────
    // A FLOOR point in the ROOM frame, not a point on the object: the affordance's target is a place
    // for the ROBOT, chosen at the sensor's stand-off distance off the face it wants to see. Marking
    // the object's centre instead put the cross metres from anything the action was aimed at, and made
    // the one failure worth seeing — driving to a standpoint that is not where the affordance asked for
    // — invisible, because the mark tracked the object no matter where the robot went.
    if (standpoint.has_value())
    {
        const Eigen::Vector3d sp_room(static_cast<double>(standpoint->room_pos.x()),
                                      static_cast<double>(standpoint->room_pos.y()), 0.0);
        // Distance ALONG THE FLOOR from the robot, not from the camera: this number is "how far is
        // there still to go", and a camera 1.5 m up would inflate it by its own mount height.
        const Eigen::Vector3d robot_room = room_T_robot.value().translation();
        const double to_go = std::hypot(sp_room.x() - robot_room.x(), sp_room.y() - robot_room.y());

        QPointF sp_px;
        if (not project(cam_T_room * sp_room, sp_px))
            // Expected, and NOT a fault, once the robot has arrived: it is standing on it. Said in
            // those words, because an orange band reading "behind the camera" during the dwell would
            // report the success of the drive as a failure of the overlay.
            add_note(to_go < 0.6
                         ? QStringLiteral("standpoint REACHED (%1 m) — it is underfoot, out of view")
                               .arg(to_go, 0, 'f', 2)
                         : QStringLiteral("the standpoint is BEHIND the camera (%1 m away)")
                               .arg(to_go, 0, 'f', 2));
        else
        {
            // Sized in pixels, not metres: this is a MARKER, and one that shrinks to nothing at range
            // stops doing its job exactly when the robot is far enough away for the question to matter.
            p.setPen(QPen(kModel, 3.0 * k));
            p.drawLine(sp_px + QPointF(-kArm, 0), sp_px + QPointF(kArm, 0));
            p.drawLine(sp_px + QPointF(0, -kArm), sp_px + QPointF(0, kArm));

            // The heading arm, along the floor — ONLY when the contract designed a final orientation.
            // A Reach ends wherever the drive ends, and drawing an arrow for it would claim a facing
            // the executor never turns to.
            if (standpoint->has_facing)
            {
                constexpr double kHeadingArm_m = 0.6;
                QPointF tip_px;
                const Eigen::Vector3d tip_room =
                    sp_room + kHeadingArm_m * Eigen::Vector3d(std::cos(standpoint->yaw_rad),
                                                              std::sin(standpoint->yaw_rad), 0.0);
                if (project(cam_T_room * tip_room, tip_px))
                {
                    p.setPen(QPen(kModel, 2.2 * k));
                    p.drawLine(sp_px, shortened(sp_px, tip_px));
                }
            }

            p.setPen(kModel);
            p.drawText(sp_px + QPointF(kArm + 6, -6),
                       QStringLiteral("standpoint %1 m%2").arg(to_go, 0, 'f', 2)
                           .arg(standpoint->has_facing ? QStringLiteral(" ↷") : QString()));
            drew = true;
            if (off_frame(sp_px))
                add_note(QStringLiteral("the standpoint projects OFF-FRAME at (%1,%2) of %3x%4")
                             .arg(sp_px.x(), 0, 'f', 0).arg(sp_px.y(), 0, 'f', 0)
                             .arg(canvas.width()).arg(canvas.height()));
        }
    }

    return drew;
}

ControllerCameraMasks::~ControllerCameraMasks() = default;

bool ControllerCameraMasks::try_discover()
{
    if (rgb_sub_ or not graph_)
        return false;
    // Descriptor-driven shared factory: it verifies the node + stream exist and takes the domain/topic
    // from the producer's JSON (media domain, never the Agent domain). Returns nullptr until then, so
    // this is safe to retry — throttled, because each failure logs.
    rgb_sub_ = rc::media::make_image_subscriber_from_graph(*graph_, camera_node_, "rgb");
    return rgb_sub_ != nullptr;
}

bool ControllerCameraMasks::poll_camera()
{
    if (not rgb_sub_)
        return false;
    bool got = false;
    // Drain to the NEWEST sample: a backlog of stale frames beside a live mask set would be worse than
    // no picture, because the silhouettes would not sit on the pixels they were computed from.
    rgb_sub_->poll([this, &got](const rc::media::ImageFrame &f, std::int64_t)
    {
        const int w = static_cast<int>(f.width());
        const int h = static_cast<int>(f.height());
        if (w <= 0 or h <= 0)
            return;
        const std::size_t npix = static_cast<std::size_t>(w) * static_cast<std::size_t>(h);
        const std::uint32_t fmt = f.format();
        std::size_t expected = 0;
        switch (fmt)
        {
            case rc::media::FORMAT_BGR8:
            case rc::media::FORMAT_RGB8:  expected = npix * 3; break;
            case rc::media::FORMAT_GRAY8: expected = npix;     break;
            default: return;                                   // depth streams are not a backdrop
        }
        if (f.size() < expected)
            return;
        const auto *data = reinterpret_cast<const std::uint8_t *>(f.data().data());
        const int step = static_cast<int>(expected / static_cast<std::size_t>(h));
        QImage img = qimage_from_media(data, w, h, step, fmt);
        if (img.isNull())
            return;
        // KEEP IT, don't replace: the frame the masks will refer to is usually NOT the newest one. A
        // ring of the last few hundred ms is what makes the pairing below possible at all.
        const std::uint64_t stamp = f.stamp_ms();
        camera_ring_.emplace_back(stamp, std::move(img));
        camera_newest_stamp_ms_ = std::max(camera_newest_stamp_ms_, stamp);
        while (camera_ring_.size() > kRingMaxFrames
               or (camera_ring_.size() > 1 and camera_ring_.front().first + kRingSpanMs
                                                   < camera_newest_stamp_ms_))
            camera_ring_.pop_front();
        got = true;
    });
    return got;
}

// The buffered frame closest in time to `stamp_ms` — the one the detector actually ran on. Falls back
// to the newest when the masks carry no stamp (a producer predating it), which is the old behaviour and
// the old drift, so the caller reports which of the two it got.
const QImage *ControllerCameraMasks::frame_for(std::uint64_t stamp_ms, std::int64_t &err_ms) const
{
    if (camera_ring_.empty())
        return nullptr;
    if (stamp_ms == 0)
    {
        err_ms = -1;
        return &camera_ring_.back().second;
    }
    const QImage *best = nullptr;
    std::int64_t best_err = 0;
    for (const auto &[t, img] : camera_ring_)
    {
        const std::int64_t e = static_cast<std::int64_t>(t) - static_cast<std::int64_t>(stamp_ms);
        if (best == nullptr or std::abs(e) < std::abs(best_err)) { best = &img; best_err = e; }
    }
    err_ms = best_err;
    return best;
}

bool ControllerCameraMasks::pump(const std::string &target_object,
                                 const std::optional<ControllerStandpoint> &standpoint,
                                 std::uint64_t now_ms, bool draw_rois)
{
    if (not masks_)
        return false;

    if (not rgb_sub_ and now_ms - last_discovery_ms_ >= kDiscoveryPeriodMs)
    {
        last_discovery_ms_ = now_ms;
        try_discover();
    }
    const bool fresh_camera = poll_camera();
    const bool fresh_masks  = masks_->refresh();

    // Recompose on new evidence OR on a period tick — the highlight depends on which affordance is
    // live, which changes without either stream moving.
    if (not fresh_masks and not fresh_camera and draw_rois == last_draw_rois_
        and now_ms - last_compose_ms_ < kComposePeriodMs)
        return false;
    last_draw_rois_ = draw_rois;
    last_compose_ms_ = now_ms;

    const auto &packet = masks_->packet();
    const std::int64_t age = masks_->ms_since_last_frame();

    // ── PAIR THE MASKS WITH THE FRAME THEY WERE COMPUTED FROM ─────────────────────────────────────
    std::int64_t align_err_ms = 0;
    const QImage *paired = frame_for(packet.timestamp_ms, align_err_ms);

    CameraMasksView v;
    v.camera_live = paired != nullptr and not paired->isNull();
    v.frame_id    = packet.frame_id;
    v.age_ms      = age;
    v.masks_live  = age >= 0 and age < kMasksStaleMs;
    v.target_label = target_object.empty() ? QStringLiteral("—")
                                           : QString::fromStdString(target_object);

    // Which slices came from the ZED. The panorama's masks index a DIFFERENT image, so drawing them on
    // this one would put silhouettes at coordinates that mean nothing here — read straight off the node
    // because MaskSlice does not carry the source (0 = zed, 1 = ricoh).
    std::vector<float> source;
    if (const auto node = graph_->get_node("masks"); node.has_value())
        if (const auto s = graph_->get_attrib_by_name<mask_source_att>(node.value()); s.has_value())
            source = s.value().get();
    const auto is_zed = [&source](std::size_t i)
    { return i >= source.size() or source[i] < 0.5f; };   // absent attribute ⇒ a zed-only producer

    // ── COMPOSE ───────────────────────────────────────────────────────────────────────────────────
    // The canvas is the camera frame when there is one, and a plain ground the same size as the mask
    // pixel span when there is not. The silhouettes are the payload; the picture is the backdrop.
    QImage canvas;
    if (v.camera_live)
        canvas = paired->convertToFormat(QImage::Format_RGB32);
    else
    {
        int max_x = 640, max_y = 480;
        for (const auto &p : packet.mask_pixels)
        {
            max_x = std::max(max_x, static_cast<int>(p.x()) + 1);
            max_y = std::max(max_y, static_cast<int>(p.y()) + 1);
        }
        canvas = QImage(max_x, max_y, QImage::Format_RGB32);
        canvas.fill(QColor(0x1a, 0x1c, 0x1e));
        v.note = QStringLiteral("no camera stream — silhouettes only");
    }

    QPainter p(&canvas);
    p.setRenderHint(QPainter::Antialiasing, false);
    // See the note in draw_target_pose: this composes at camera resolution and is DISPLAYED at roughly
    // 180 rows, so every stroke is dimensioned for the displayed size and multiplied up.
    const double k = std::max(1.0, canvas.height() / 180.0);
    // Dim the backdrop ONLY when silhouettes are going on top of it — the dimming exists to make them
    // read as an overlay, and with nothing overlaid it is just a darker picture.
    if (v.camera_live and draw_rois)
        p.fillRect(canvas.rect(), QColor(0, 0, 0, 70));

    std::size_t colour_next = 0;
    for (std::size_t i = 0; i < packet.slices.size(); ++i)
    {
        const auto &s = packet.slices[i];
        const bool target = mask_label_matches_object(target_object, s.label);
        CameraMasksView::Item item;
        item.label      = QString::fromStdString(s.label);
        item.confidence = s.confidence;
        item.pixels     = static_cast<int>(s.pixel_end - s.pixel_begin);
        item.range_m    = s.range;
        item.is_target  = target;
        v.items.push_back(item);
        if (target) ++v.target_count;

        // The item list above stays complete whatever happens here: what the producer saw is recorded
        // even when it is not drawn, so nothing downstream has to care about a display decision.
        if (not draw_rois)
            continue;
        if (not is_zed(i) or s.pixel_end <= s.pixel_begin)
            continue;   // ricoh slice, or a producer that sent no silhouette pixels

        const QColor c = target ? QColor(0x2e, 0xcc, 0x71) : kPalette[colour_next++ % kPaletteSize];
        // Every foreground pixel, not a hull: a YOLO mask is ragged, and a tidy outline would claim a
        // precision the segmentation does not have. Cheap — these are a few thousand points.
        p.setPen(QPen(QColor(c.red(), c.green(), c.blue(), target ? 200 : 130), 1.0));
        int min_x = canvas.width(), min_y = canvas.height(), max_x = 0, max_y = 0;
        for (std::size_t k = s.pixel_begin; k < s.pixel_end and k < packet.mask_pixels.size(); ++k)
        {
            const int px = static_cast<int>(packet.mask_pixels[k].x());
            const int py = static_cast<int>(packet.mask_pixels[k].y());
            p.drawPoint(px, py);
            min_x = std::min(min_x, px); max_x = std::max(max_x, px);
            min_y = std::min(min_y, py); max_y = std::max(max_y, py);
        }
        if (min_x > max_x)
            continue;

        const QRect box(min_x, min_y, max_x - min_x, max_y - min_y);
        p.setPen(QPen(c, (target ? 2.5 : 1.5) * k));
        p.setBrush(Qt::NoBrush);
        p.drawRect(box);

        QFont f = p.font();
        f.setBold(true);
        f.setPointSizeF(std::max(9.0, canvas.height() / 22.0));
        p.setFont(f);
        const QString tag = QStringLiteral("%1 %2%3")
                                .arg(item.label)
                                .arg(static_cast<double>(s.confidence), 0, 'f', 2)
                                .arg(target ? QStringLiteral("  ←") : QString());
        const int tag_h = QFontMetrics(f).height() + 4;
        const QRect tag_box(box.left(), std::max(0, box.top() - tag_h),
                            std::max(static_cast<int>(200 * k), box.width()), tag_h);
        p.fillRect(tag_box, QColor(c.red(), c.green(), c.blue(), 190));
        p.setPen(QColor(0x10, 0x10, 0x10));
        p.drawText(tag_box.adjusted(3, 0, 0, 0), Qt::AlignLeft | Qt::AlignVCenter, tag);
    }
    // THE MODEL ON TOP OF THE MEASUREMENT — drawn last, so it sits over the silhouettes it is meant to
    // be compared with. Pinned to the mask capture stamp, so the box is where the belief says the object
    // was WHEN THE SHUTTER OPENED, not where it is now: comparing a live pose against an old frame would
    // manufacture an offset on every moving cycle and read as a pose error that is not there.
    QString project_note;
    v.target_projected = draw_target_pose(p, target_object, standpoint, canvas.size(),
                                          packet.timestamp_ms, project_note);
    // "There is no affordance running" is the commonest reason for no marker, and the only one
    // draw_target_pose cannot report — it has nothing to report ABOUT. Say it here, or an absent
    // overlay looks like a broken overlay.
    if (not v.target_projected and project_note.isEmpty()
        and target_object.empty() and not standpoint.has_value())
        project_note = QStringLiteral("no affordance target — nothing to project");

    // An alignment the pairing could not achieve is worth saying: silhouettes drawn on a frame 200 ms
    // from their own are not a perception fault, they are this view's fault, and the two must never be
    // confused when someone is using the overlay to judge the perception.
    QString align_note;
    if (v.camera_live and align_err_ms < 0 and packet.timestamp_ms == 0)
        align_note = QStringLiteral("masks carry no stamp — drawn on the NEWEST frame (may drift)");
    else if (v.camera_live and static_cast<std::uint64_t>(std::abs(align_err_ms)) > kAlignTolMs)
        align_note = QStringLiteral("no frame within %1 ms of the masks (off by %2 ms)")
                         .arg(kAlignTolMs).arg(align_err_ms);

    QString stream_note;
    if (v.camera_live and not v.masks_live)
        stream_note = age < 0 ? QStringLiteral("no masks frame yet — is the voxelizer running?")
                              : QStringLiteral("masks STALE (%1 ms) — producer stopped").arg(age);
    // BOTH, when both apply. The stream note used to overwrite the projection note outright, so a
    // stalled producer hid the reason the target box was missing — two independent faults, one slot.
    if (not v.note.isEmpty() and not stream_note.isEmpty())
        v.note += QStringLiteral(" · ") + stream_note;
    else if (not stream_note.isEmpty())
        v.note = stream_note;
    if (not align_note.isEmpty())
        v.note = v.note.isEmpty() ? align_note : v.note + QStringLiteral(" · ") + align_note;
    if (not project_note.isEmpty())
        v.note = v.note.isEmpty() ? project_note : project_note + QStringLiteral(" · ") + v.note;

    // FAULTS ARE DRAWN ON THE FRAME, not beside it. The panel has no text under the image any more, and
    // a picture that silently shows nothing is indistinguishable from a scene with nothing in it — which
    // is the one confusion this view exists to prevent. Only the note: what IS detected is already
    // labelled in place, so restating it here would be a second rendering of the same frame.
    if (not v.note.isEmpty())
    {
        QFont f = p.font();
        f.setBold(true);
        f.setPointSizeF(std::max(9.0, canvas.height() / 22.0));
        p.setFont(f);
        const QRect band(0, 0, canvas.width(), QFontMetrics(f).height() + 6);
        p.fillRect(band, QColor(0x9a, 0x5b, 0x10, 210));
        p.setPen(QColor(0xff, 0xff, 0xff));
        p.drawText(band.adjusted(6, 0, -6, 0), Qt::AlignLeft | Qt::AlignVCenter, v.note);
    }
    p.end();

    // ★SAY IT OUT LOUD, ONCE EVERY FEW SECONDS. Three rounds went into "nothing is on the image"
    // without knowing whether the marker was absent, off-frame, or drawn four pixels wide — and the
    // note band is itself part of the image, so a rendering fault hides its own explanation. The log
    // is outside that loop.
    if (now_ms - last_report_ms_ >= 3000)
    {
        last_report_ms_ = now_ms;
        std::print("[aff-view] object='{}' standpoint={} projected={} camera={}x{} masks={} stamp={} "
                   "align_err={}ms ring={} note='{}'\n",
                   target_object.empty() ? std::string("(none)") : target_object,
                   standpoint.has_value()
                       ? std::format("({:.2f},{:.2f}) yaw {:.0f}deg{}", standpoint->room_pos.x(),
                                     standpoint->room_pos.y(),
                                     standpoint->yaw_rad * 180.0 / M_PI,
                                     standpoint->has_facing ? "" : " (no facing)")
                       : std::string("(none)"),
                   v.target_projected, canvas.width(), canvas.height(),
                   packet.slices.size(), packet.timestamp_ms, align_err_ms,
                   camera_ring_.size(), v.note.toStdString());
        if (not draw_rois)
            std::print("[aff-view] ROIs hidden — not at a step that reads them\n");
        std::fflush(stdout);
    }

    v.image = std::move(canvas);
    view_ = std::move(v);
    last_composed_frame_ = packet.frame_id;
    return true;
}

}   // namespace rc
