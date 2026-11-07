// door_visibility.h — "should this door be visible in this image, and how big does it project?" for ANY camera the core
// knows (the ZED pinhole and the ricoh 360). Task 2 of the door-existence model (.superpowers/sdd/2026-11-07-door-existence-model).
//
// ★MOVED, NOT REWRITTEN. The sample classification below is the old silhouette probe's `classify` lambda (door_existence.cpp
// as of a316097f), line for line: a leaf-face sample is NOT SEEN when it is behind the near clip or outside the image (for a
// 360 camera: outside every looked window), OCCLUDED when a nearer non-door mask, a wall (the door's own excluded), another
// door's cone, or the aperture itself (it lies past the room-side face: door_room_band.h) hides it, and VISIBLE otherwise,
// weighted by 1 − P(beyond the far face). the old probe walked its samples through ViewRig and kept only
// its occupied / free bookkeeping; door_view() walks the same samples to answer the camera-generic question the existence
// model needs (how visible, how large, from what angle).
//
// The only change from the ZED-only code is the camera: the projection is camera_model.h's project(CameraSpec), identical to
// camera_project() for the pinhole (tests/test_door_visibility.cpp pins the silhouette's outputs).
#pragma once
#include <cstdint>
#include <span>
#include <unordered_set>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include "camera_model.h"
#include "door_existence.h"

namespace rc::door::core
{
// Hashed pixel-cell coverage of the current YOLO foreground, split door (occupancy) vs every other label (occluder). A
// CELL-px cell absorbs mask-boundary jitter and makes membership O(1); the key packs the two cell indices. CELL is a
// discretization constant (quadrature), not a belief gate — no config key. Used by the leaf silhouette, the door view and
// the aperture-frame probe, so "what hides a door" is one definition.
struct MaskCells
{
    static constexpr float CELL = 6.0f;
    std::unordered_set<std::int64_t> door, occluder;
    [[nodiscard]] static std::int64_t key(float col, float row)
    {
        const std::int64_t kx = static_cast<std::int64_t>(std::floor(col / CELL));
        const std::int64_t ky = static_cast<std::int64_t>(std::floor(row / CELL));
        return (kx << 32) ^ (ky & 0xffffffffLL);
    }
    // A pixel covered by some NON-door mask and by no door mask: a nearer masked object hides it.
    [[nodiscard]] bool hides(float col, float row) const
    {
        const std::int64_t k = key(col, row);
        return occluder.contains(k) and not door.contains(k);
    }
};

/// Which slices of the packet describe THIS camera's pixels: the ZED path has always read every slice (a packet with no ricoh
/// slice is all ZED); the ricoh view reads only the slices whose source is not the ZED (MaskSlice::source != 0).
enum class MaskScope : std::uint8_t { All, RicohOnly };
[[nodiscard]] MaskCells build_mask_cells(const MaskPacket& pkt, MaskScope scope = MaskScope::All);

/// Task D-C: the run of polygon edges the doorway is cut in (every OTHER wall occludes).
[[nodiscard]] std::vector<std::size_t> own_wall_edges(const rc::DoorFitState& inst, const std::vector<Eigen::Vector2f>& polygon);

/// One camera looking at the room. `room_T_cam`: camera→room, Y forward / Z up (CameraAPI's frame).
struct CameraView
{
    const CameraSpec& cam;
    Mat4d room_T_cam;
    /// ROOM-frame azimuth intervals (from, to; rad, counter-clockwise) the 360 detector actually processed this frame
    /// (MaskPacket::ricoh_windows). Used by non-pinhole cameras only. A bearing outside every window was NOT LOOKED AT.
    std::span<const Vec2f> looked_windows;
    /// false ⇒ the windows are unknown (a producer without the field): a 360 view then counts NOTHING as looked at, so it adds
    /// no miss evidence (Review Focus 3). The caller may set it true with a known window it derived another way.
    bool windows_known = false;
    /// How many vertical strips the 360 detector splits the panorama into; a frame processes ONE, so the image extent a door
    /// fills is one strip wide. retina/etc/config.toml [Ricoh] yolo_n_strips = 3 (yolo_strip_overlap_px = 200 is ignored: the
    /// overlap only widens the strip, and the fill is a ratio of the nominal extent). Used by non-pinhole cameras only.
    int strips = 3;
};

enum class SampleFate : std::uint8_t
{
    Unseen,     // behind the near clip, outside the image, or outside every looked window: no vote, not counted
    Occluded,   // projects, but a mask / wall / another door / the aperture itself hides it
    Visible,
};

struct SampleView
{
    SampleFate fate = SampleFate::Unseen;
    double X = 0, Y = 0, Z = 0;          // camera-frame position of the sample
    float col = 0.f, row = 0.f;          // pixel (valid when fate != Unseen... and for Visible always)
    std::int64_t key = 0;                // MaskCells key of (col,row) (Visible only)
    float w_room = 0.f;                  // 1 − P(seen through the opening) (Visible only)
    Eigen::Vector3f Pw = Eigen::Vector3f::Zero();   // the sample in the room frame
};

/// The quadrature grid over the leaf face (numeric SAMPLING RESOLUTION, not a threshold).
inline constexpr int kLeafGridNX = 14, kLeafGridNZ = 30;

/// Everything about a camera+scene that does not depend on the leaf's opening angle, built once and asked per sample.
/// Holds references to its arguments: they must outlive it.
class ViewRig
{
public:
    ViewRig(const CameraView& v, const ExistenceScene& sc, const rc::DoorFitState& inst, const MaskCells& cells);
    /// Classify the leaf-face sample at local x (hinge→free edge) and absolute z, for the leaf in pose `L`.
    [[nodiscard]] SampleView at(const door::LeafPose& L, float lx, float lz) const;
    [[nodiscard]] const Eigen::Vector3f& camera_position() const { return O_; }
    [[nodiscard]] const Eigen::Matrix4d& cam_T_room() const { return cam_T_room_; }
private:
    const CameraView& v_;
    const ExistenceScene& sc_;
    const rc::DoorFitState& inst_;
    const MaskCells& cells_;
    Eigen::Matrix4d cam_T_room_;
    Eigen::Vector3f O_;
    Eigen::Vector2f Oxy_;
    std::vector<std::size_t> own_edges_;
    float occ_margin_ = 0.f;
    bool has_poly_ = false;
    bool band_known_ = false;
    Eigen::Vector2f n_out_ = Eigen::Vector2f::Zero();
    float r0_ = 0.f;
    std::vector<Eigen::Matrix<float, 2, 1>> windows_;
    float W_ = 0.f, H_ = 0.f;
};

struct ViewGeometry
{
    float vis = 0.f;        // E_phi[visible fraction of leaf-face samples]
    float fill_h = 0.f;     // projected extent / image extent, horizontal (a 360 camera: / one detector strip)
    float fill_v = 0.f;     // ... vertical; > 1 means the door overflows
    float aspect = 0.f;     // |cos| of the view angle to the aperture plane's normal
    float range_m = 0.f;
    std::vector<cv::Point2f> samples;   // pixel samples of the modal rendering that are visible (for mask lookup)
    bool projectable = false;           // false => nothing is computed, every likelihood is 0
};

/// Would this camera see the door, and how big does it project? Marginalised over the leaf-angle posterior (inst.phi_curve)
/// exactly as the silhouette's absence is. No pose / uninitialised door / the door's centre has no pixel ⇒ projectable false.
[[nodiscard]] ViewGeometry door_view(const CameraView& v, const ExistenceScene& sc, const rc::DoorFitState& inst);
}  // namespace rc::door::core
