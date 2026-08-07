#pragma once

/*
 * controller_camera_masks.h — what the camera is looking at, with the YOLO masks on it.
 *
 * WHY THE CONTROLLER NEEDS THIS AT ALL. An affordance is an EPISTEMIC action: the robot drives to a
 * standpoint and turns to face an object so that perception can get a look at it. Whether that paid off
 * is a question about the MASKS — did a silhouette of the thing appear, at what confidence — and until
 * now the controller could only report that it had arrived. Arriving is not the goal; being able to SEE
 * is. So the affordance panel shows the last camera frame with the silhouettes drawn on it, and the
 * session holds the robot still for a moment afterwards (see ControllerParams::affordance_dwell_ms) so
 * there is something to look at.
 *
 * TWO INDEPENDENT SOURCES, DELIBERATELY:
 *   • the SILHOUETTES come from the "masks" DSR node (shared MaskIngestor — the same parse every concept
 *     agent uses, so what is drawn here is exactly what the beliefs were fed);
 *   • the PICTURE comes from the zero-copy media plane (zed/rgb descriptor).
 * The masks are the point and the picture is the backdrop: if the camera stream is absent the
 * silhouettes are still drawn, on a plain ground, and the panel says the picture is missing. A viewer
 * that goes blank when the OPTIONAL half is unavailable would hide the half that matters.
 *
 * FRAMES ARE COMPOSED ON THE CONTROL THREAD and handed over as a finished QImage. QImage is a raster
 * paint device (painting one off the GUI thread is supported) and it is deep-copied at the boundary, so
 * no pixel buffer is ever shared with the GUI — the cv::Mat hazard in CLAUDE.md, avoided by not having
 * a shared buffer in the first place. Nothing here touches DDS from a constructor: the subscriber comes
 * up lazily from the already-Operating control thread, the sanctioned media-plane consumer pattern.
 */

#include <QImage>
#include <QSize>
#include <QString>

#include <Eigen/Geometry>

#include <cstdint>
#include <deque>
#include <optional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class QPainter;

namespace DSR { class DSRGraph; class InnerEigenAPI; }
namespace rc { class MaskIngestor; }
namespace rc::media { class MediaSubscriber; }

namespace rc
{

// Does this mask show the thing the affordance went to look at? The object is a graph NODE name
// ("table_3", "chair_1") and the mask carries a YOLO CLASS name ("dining table", "chair"), so the
// match is on the alphabetic stem of each with the other's contained in it. Loose ON PURPOSE: it
// decides a highlight and how many confirming looks the dwell has counted, never what the robot does
// next. SHARED with ControllerSession's dwell counter deliberately — two definitions of "this mask is
// the object" would let the panel and the wait disagree about the same frame.
bool mask_label_matches_object(const std::string &object, const std::string &label);

// The finished snapshot the panel renders. Pure data + one composed image; no live handles, so it can
// cross to the GUI thread by value like every other display snapshot field.
struct CameraMasksView
{
    struct Item
    {
        QString label;
        float   confidence = 0.f;
        int     pixels = 0;        // silhouette size, the honest proxy for "how much of it is visible"
        float   range_m = 0.f;     // mean camera→mask depth; 0 when the producer sent none
        bool    is_target = false; // matches the object the affordance is servicing
    };

    QImage      image;                 // camera + silhouettes, ready to draw. Null = nothing composed yet
    bool        camera_live = false;   // false ⇒ silhouettes only, on a plain ground
    bool        masks_live = false;    // the producer's frame id is advancing
    int         frame_id = -1;
    std::int64_t age_ms = -1;          // since the last NEW masks frame (-1 = never)
    std::vector<Item> items;
    int         target_count = 0;      // how many of `items` are the affordance's object
    QString     target_label;          // what we were looking for ("—" when no affordance is live)
    QString     note;                  // why the picture or the masks are missing, when they are
    // The BELIEF's pose, projected into the same image as the measurement it came from — see
    // draw_target_pose. False when it could not be drawn, which is itself worth knowing.
    bool        target_projected = false;
};

// Owns the mask ingestor + the rgb subscriber and composes the view. One instance, pumped from the
// control loop; construct once the graph is loaded.
class ControllerCameraMasks
{
public:
    explicit ControllerCameraMasks(std::shared_ptr<DSR::DSRGraph> graph, std::string camera_node = "zed");
    ~ControllerCameraMasks();

    // Drain the camera plane, re-read the masks node, and recompose when either moved. `target_object`
    // is the affordance's object node name ("" when none) and only decides which items are HIGHLIGHTED —
    // never which are drawn, because "the mask you were not looking for showed up instead" is a finding.
    // Returns true when the view changed. Cheap to call at the control rate; it self-throttles the
    // recompose, which is the only part that costs anything.
    // `draw_rois` = the affordance is at a step whose completion reads the masks (see
    // ControllerSession::look_step_active). When false the silhouettes are NOT drawn: before the look
    // begins they are clutter over the marker that matters, and a detection visible at a step that
    // ignores it reads as progress that is not happening. The masks are still ingested — the frame
    // counter and the item list stay live — only the drawing is withheld.
    bool pump(const std::string &target_object, std::uint64_t now_ms, bool draw_rois);

    // The room and robot frame names, pushed each cycle by the worker (they are not known until the
    // graph has loaded). REQUIRED for the target projection: room←zed cannot be asked for in one query
    // because the two legs need DIFFERENT time queries — see draw_target_pose.
    void set_frames(std::string room, std::string robot)
    { room_name_ = std::move(room); robot_name_ = std::move(robot); }

    const CameraMasksView &view() const { return view_; }

private:
    bool try_discover();
    bool poll_camera();
    // ── THE MODEL, DRAWN ON TOP OF THE MEASUREMENT ────────────────────────────────────────────────
    // Project the affordance object's BELIEVED 3D box into this image, where the belief says it should
    // appear. The comparison is the whole value: a mask with no box on it is a detection the model does
    // not know about; a box with no mask in it is a model asserting something the camera will not
    // confirm; and a box sitting BESIDE its mask is a pose error you can read off in pixels instead of
    // inferring from a residual. Returns false when it could not be drawn, and says why in `note`.
    bool draw_target_pose(QPainter &p, const std::string &object, const QSize &canvas,
                          std::uint64_t stamp_ms, QString &note);
    // Static zed intrinsics, read once off the camera node. False until they are available.
    bool ensure_intrinsics();
    // Buffered frame nearest `stamp_ms`; `err_ms` receives (frame − masks) in ms, or -1 when the masks
    // carried no stamp and the newest frame was used instead. nullptr when nothing is buffered.
    const QImage *frame_for(std::uint64_t stamp_ms, std::int64_t &err_ms) const;

    std::shared_ptr<DSR::DSRGraph> graph_;
    std::string camera_node_;
    std::unique_ptr<rc::MaskIngestor> masks_;
    std::unique_ptr<rc::media::MediaSubscriber> rgb_sub_;
    // Its OWN InnerEigenAPI instance. get_inner_eigen_api() hands out a fresh unique_ptr per call, and
    // sharing one across threads is only safe on the ts!=0 path — which this uses exclusively (every
    // query is pinned to the mask capture stamp, so nothing here touches the unlocked ts==0 cache).
    // Owning one anyway costs nothing and removes the question. See CLAUDE.md, DSR thread-safety.
    std::unique_ptr<DSR::InnerEigenAPI> inner_;
    float fx_ = 0.f, fy_ = 0.f;        // zed intrinsics (px); 0 = not read yet
    std::string room_name_, robot_name_;
    // robot←zed, the RIGID camera mount. Resolved once with ts==0 and kept: it never changes, and the
    // ts==0 path is the one InnerEigenAPI cache that is not thread-safe to share — caching the RESULT
    // means this instance touches it once instead of at frame rate.
    std::optional<Eigen::Affine3d> robot_T_zed_;

    // ── TIME ALIGNMENT: A SHORT HISTORY, NOT THE NEWEST FRAME ─────────────────────────────────────
    // The masks describe the frame the detector RAN ON, which by the time they reach the graph is
    // several tens of milliseconds old (capture → YOLO → mask node). Painting them on the newest image
    // therefore draws yesterday's silhouettes on today's pixels, and every centimetre the robot moved
    // in between shows up as the masks sliding off their objects — the drift.
    // Both sides carry the SAME stamp: the media ImageFrame's stamp_ms and the producer's
    // mask_timestamp_ms are the same camera capture clock, because the voxelizer copies it straight
    // through. So keep a few hundred ms of frames and pair the masks with the one they were computed
    // from. Bounded by age AND count: this is a debug view, not a recorder.
    std::deque<std::pair<std::uint64_t, QImage>> camera_ring_;
    std::uint64_t camera_newest_stamp_ms_ = 0;
    std::uint64_t last_discovery_ms_ = 0;
    std::uint64_t last_compose_ms_ = 0;
    std::uint64_t last_report_ms_ = 0;   // throttles the diagnostic line (see pump)
    bool last_draw_rois_ = false;        // recompose when the draw decision flips, not only on new data
    int last_composed_frame_ = -1;
    CameraMasksView view_;
};

}   // namespace rc
