/*
 * contour_edge_project.h — build the contour the RGB/depth check is scored on, and the NULL it is
 * scored against, from a belief's own geometry.
 *
 * ★WHY THIS IS A SHARED UNIT AND NOT A LINE OF EACH AGENT'S FITTER.
 * The scorer next door (contour_edge_check.h) is the easy half: given a polygon and some control
 * polygons it measures which one the image prefers. The half that decides whether the answer MEANS
 * anything is the construction of those polygons, and door_concept paid for that twice in two days:
 *
 *   · Controls displaced sideways IN IMAGE PIXELS are a valid null only while the object lies in the
 *     surface its neighbours are made of. A door leaf that swings out of the wall keeps its image-space
 *     neighbours, but those neighbours are now the open doorway and the room beyond — both edge-rich —
 *     so s_true fell below s_control and the channel voted to DELETE the door, most confidently exactly
 *     when it was open (measured 2026-09-09: dL −1.4…−2.6 EVERY cycle at phi 35-45°).
 *   · The repair was to slide the quad in 3-D along the object's OWN plane and reproject. Identical at
 *     phi = 0, still valid at every other angle.
 *
 * That repair is geometry, not door geometry, and every concept agent that wants this channel needs
 * exactly it. Hence: the object states where it is in 3-D, this unit produces the contour and a null of
 * the form "the same shape, somewhere it could equally have been", and no agent re-derives the trap.
 *
 * ★THE NULL IS A SLIDE ALONG THE SUPPORT PLANE. For an upright box standing on a floor (fridge,
 * cabinet, chair) or a leaf hanging in a wall (door), the honest "somewhere it could equally have been"
 * is the same shape translated horizontally along the surface that carries it: same range, same
 * lighting, same wall or floor, same exposure. Sliding VERTICALLY would straddle floor and ceiling and
 * compare against a different kind of surface entirely — which is a different question, not a control.
 *
 * ★THE PROJECTION ITSELF STAYS WITH THE AGENT, as a callback. Near-clip, which timestamp the room→camera
 * hop is pinned to, and whether the delivered frame's size matches the CameraAPI's intrinsics are all
 * things only the agent knows, and getting them wrong is silent. This unit refuses to guess: it hands
 * the caller 3-D points in the world frame and takes back pixels plus the DEPTH it projected them at.
 *
 * ★RANGE COMES BACK WITH THE PIXEL because the depth channel needs it. A depth check that only looked
 * for a step would be another unsigned edge detector; the belief predicts an actual distance at every
 * point of its own silhouette, and checking THAT is what separates a door from a photograph of a door.
 * See contour_depth_check.h.
 */

#pragma once

#include <array>
#include <functional>
#include <optional>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace rc::edges
{

// One projected vertex: where the belief says it lands, and how far away it says it is.
struct ProjectedVertex
{
    cv::Point2f px{};
    // ★THE SAME QUANTITY THE DEPTH PLANE STORES, which on the ZED is the camera-frame FORWARD
    // coordinate (y here) — NOT the Euclidean norm of the point. retina deprojects with `py = depth`
    // (graph_publisher.cpp), so a norm supplied here would read high by up to a few percent at the
    // edges of a wide field and would look exactly like a belief sitting slightly too far away. The
    // caller owns the projection and therefore owns this convention; < 0 = unknown.
    float       depth_m = -1.0f;
};

// The caller's projection: world point → pixel + range, or nullopt for "behind the camera / not
// finite / outside this camera's model". The caller owns the near-clip and the stamp the transform is
// pinned to. Returning nullopt for ANY corner invalidates the whole contour, which is the intent — a
// quad with a corner behind the camera has no meaningful projection to score.
using PointProjector = std::function<std::optional<ProjectedVertex>(const Eigen::Vector3d&)>;

// A closed contour in image space that remembers what the belief predicted at each of its vertices.
// `px` is what the RGB scorer consumes (it ignores the rest); `depth_m` is what the depth scorer needs.
struct Contour
{
    std::vector<cv::Point> px;        // integer pixels, in closed-loop order
    std::vector<float>     depth_m;   // predicted depth per vertex (see ProjectedVertex), parallel to px

    [[nodiscard]] bool valid() const { return px.size() >= 3 and depth_m.size() == px.size(); }
};

// A contour plus the null it must beat. Empty `face` = nothing projectable this frame, which a caller
// must read as NOT MEASURED and not as a refutation.
struct ContourSet
{
    Contour              face;
    std::vector<Contour> controls;
    Eigen::Vector2f      outward_normal{0.0f, 0.0f};   // room-frame outward normal of the chosen face
    float                area_px = 0.0f;               // |projected area| of `face`, for diagnostics
    int                  n_faces_visible = 0;          // camera-facing faces considered (box entry only)
};

// The control offsets, in multiples of the contour's own width. ±1 and ±1.6: far enough not to overlap
// the object, near enough to be the same surface and the same light. Two distances so one control that
// happens to land on a second door (or a window, or the fridge's neighbour) cannot alone decide the
// comparison. Same values as make_side_controls' image-space fallback, so the two agree by construction.
inline constexpr std::array<float, 4> kControlOffsets{-1.6f, -1.0f, 1.0f, 1.6f};

// ── ENTRY 1: an arbitrary planar quad, slid along a direction of the caller's choosing ───────────────
//
// The general form, and what door_concept's leaf uses: `corners` in world order (a closed loop — points
// out of loop order make a bow-tie whose "edges" cross the object and score whatever lies under the
// diagonals), `slide_dir` a unit vector IN the quad's plane along which a displaced copy is still a
// like-for-like comparison, and `slide_span_m` the quad's own width along that direction.
[[nodiscard]] ContourSet project_quad(const std::array<Eigen::Vector3d, 4>& corners,
                                      const Eigen::Vector3d& slide_dir,
                                      float slide_span_m,
                                      const PointProjector& project);

// ── ENTRY 2: an upright box standing on a support plane ──────────────────────────────────────────────
//
// The form every box-shaped concept has (fridge, cabinet, table, chair). Picks the camera-facing
// vertical side face with the LARGEST projected area — the most measurable boundary this view offers,
// chosen by the geometry rather than by a preferred side — and slides it along its own horizontal
// in-plane axis, which for an upright box IS the support plane.
struct BoxFootprint
{
    float cx = 0.0f, cy = 0.0f;    // centre of the footprint, world frame
    float yaw = 0.0f;              // footprint orientation
    float w = 0.0f, d = 0.0f;      // full extent along the box's local x and y
    float z_min = 0.0f, z_max = 0.0f;   // vertical extent, world frame
};

[[nodiscard]] ContourSet project_box_face(const BoxFootprint& box,
                                          const Eigen::Vector3d& camera_pos_world,
                                          const PointProjector& project);

}   // namespace rc::edges
