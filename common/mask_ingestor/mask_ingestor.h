/*
 * common/mask_ingestor/mask_ingestor.h
 *
 * Shared perception input layer for the concept agents (bottle/table/chair/…): reads the YOLO
 * "masks" DSR node (written by the retina), parses it into per-instance mask slices + support
 * points + raw 2D silhouette pixels, and serves the nearest slice of a requested label to a query
 * centroid. Object-AGNOSTIC: the caller passes the query point + label, so this is a single shared
 * file (no <Obj>Instance dependency). Also reads the per-node candidate/residual point attributes.
 *
 * Owns the parsed MasksPacket (and the last-seen frame guard); consumers read it read-only via
 * packet(). Plain class (no Q_OBJECT) constructed by SpecificWorker once G is ready.
 *
 * FRAME CONTRACT (this class IS the single agreement point with the retina producer):
 *   The producer publishes mask support points in the ZED/CAMERA frame — the raw deprojection, before
 *   any pose is applied ("mask_support_points_cam"). THE ROOM TRANSFORM HAPPENS HERE, not at the
 *   producer: this class reads the camera points and transforms them zed→"room" via inner_eigen,
 *   pinned to the CAPTURE stamp, recomputing per-slice centroids/bboxes in the target frame. No *_cam
 *   centroid/bbox attribute exists or is needed for that reason.
 *
 *   ★WHY THE CONSUMER PAYS FOR THE TRANSFORM, NOT THE PRODUCER. Baking room_T_zed at the producer made
 *   a RAW-PERCEPTION component depend on a LOCALIZATION component: the retina could not run at all
 *   without room_concept, it grew a forward pose-extrapolator purely to beat the room→robot RT lag, and
 *   when the RT chain failed to resolve it published camera-frame points LABELLED as room frame (an
 *   identity fallback with no attribute to distinguish it). It also hid the localization uncertainty:
 *   the room transform is a measurement channel WITH a covariance, and only whoever applies it can add
 *   the J·Σ_chain·Jᵀ term (door/table/chair/cabinet already do, via InnerGaussianAPI). Skeletons were
 *   already published camera-frame for exactly this reason ("so a consumer can servo robot-relative
 *   without paying localization noise"); masks were the last holdout.
 *
 *   Legacy: "mask_support_points" (ROOM frame) is still read when the producer publishes it and the
 *   camera array is absent, so an OLD retina keeps working. Set MASK_INGESTOR_LEGACY_ROOM=1 in the
 *   environment to force that path for an A/B against the transform done here — the two must agree to
 *   float precision, since it is the same matrix at the same timestamp, one process later.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <Eigen/Dense>
#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

namespace rc {

class MaskIngestor
{
public:
    // One detected instance: label + pose + [begin,end) ranges into the packet's shared point/pixel arrays,
    // plus the per-mask corruption/range/bearing channels the consumers fold into R and the common-mode.
    // Which sensor produced a mask. Values match the retina's `mask_source` attribute exactly.
    enum class MaskSource : int { zed = 0, ricoh = 1 };

    struct MaskSlice
    {
        std::string label;
        float class_id = -1.0f;
        float confidence = 0.0f;
        Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
        Eigen::Vector3f bbox_min = Eigen::Vector3f::Zero();
        Eigen::Vector3f bbox_max = Eigen::Vector3f::Zero();
        std::size_t support_begin = 0;
        std::size_t support_end = 0;
        std::size_t pixel_begin = 0;   // raw 2D mask pixel range (into MasksPacket::mask_pixels)
        std::size_t pixel_end = 0;
        // Ego-motion capture-corruption channel (see ../MASK_MOTION_CORRUPTION.md). Per-mask, written by
        // the retina producer; 0 when the producer predates the feature (backward-compatible).
        float motion_var      = 0.0f;  // variance to ADD to R (m²): exposure blur + timing jitter, ×peripheral
        float motion_bias     = 0.0f;  // systematic displacement from a known timing offset (m) → GATE if large
        float motion_dotd     = 0.0f;  // metric position-corruption speed Z·‖ṡ‖ (m/s), diagnostic
        float trunc_frac      = 0.0f;  // fraction of silhouette pixels on the image border → unobserved face
        float centroid_radius = 0.0f;  // normalised centroid radius from the principal point (periphery)
        float range           = 0.0f;  // mean camera→mask depth Z (m): STATIC range — consumer grows R + pose
                                       // common-mode with it so a distant view can't resolve pose/orientation
        // RGB-360 (ricoh) peripheral evidence (Part B, RICOH_360_PERIPHERAL_DETECTION.md). A no-depth slice
        // carries NO 3D points (support_begin==support_end, centroid/bbox are NaN) — only a room-frame
        // BEARING. Consumers MUST skip !has_depth slices until a bearing confirm/birth path exists (Part C).
        // Defaults match the zed contract, so a producer that predates the field reads back has_depth=true.
        bool  has_depth        = true;   // false ⇒ ricoh bearing-only slice
        float azimuth_room_rad = 0.0f;   // room-frame bearing; meaningful only when has_depth==false
        // ★★WHICH CAMERA SAW IT — AND `has_depth` IS NOT THAT QUESTION. The two were the same question only
        // while ricoh slices were bearing-only. Once the producer began depth-filling them from reprojected
        // LiDAR it started publishing them as FULL 3D slices with has_depth = 1 (graph_publisher.cpp: a
        // lidar-depth ricoh mask pushes has_depth 1.0 and source 1.0), so every consumer that wrote
        // `if (has_depth)` to mean "from the ZED" silently began accepting 360° detections from behind the
        // robot. The producer says so in its own comment — "Unlike mask_has_depth this is unambiguous" —
        // and published `mask_source` for exactly this; the ingestor simply never read it, so no agent
        // COULD ask. Reported live: a bottle moving and cloning with the robot facing away, 3 m off.
        //
        // The rule this restores: AN OBJECT MAY ONLY BE CREATED OR UPDATED FROM THE FRONT RGB-D CAMERA.
        // A ricoh slice may still CONFIRM a live instance (common/peripheral_channel) — that is evidence the
        // thing is still there, not a licence to move it — and may raise a proto-object to go and look at.
        // Defaults to zed so a producer predating the field reads back exactly as before.
        MaskSource source = MaskSource::zed;
        bool  is_zed()   const { return source == MaskSource::zed; }
        bool  is_ricoh() const { return source == MaskSource::ricoh; }
        // The one predicate a fit/birth path should ask. Named for the RULE, not for the sensor, so a third
        // camera lands here rather than in seven `== zed` comparisons.
        bool  may_fit_geometry() const { return is_zed() and has_depth; }

        // ★★INSTANCE (YOLO-seg) vs SEMANTIC (YOLO-sem), and it was published all along. The retina tags a
        // semantic detection with `class_id = 1000 + id` — "class_id offset by 1000 keeps semantic ids clear
        // of the COCO id range" (retina/src/semantic_mask_stage.cpp) — and the viewer prefixes those labels
        // with "sem:" ON SCREEN ONLY. So the distinction is visible to a human watching the overlay and, until
        // now, to NO consumer: `grep class_id` across every agent found the ingestor writing it and nobody
        // reading it. Exactly the shape of the mask_source defect: the producer said it, the ingestor carried
        // it, no agent COULD ask.
        //
        // The rule it enables (the user's): AN INSTANCE MASK PREVAILS OVER A SEMANTIC ONE where they overlap.
        // Measured live: the semantic `cabinet` mask covers the lower half of the refrigerator, so cabinet's
        // wall run grew over the fridge and then out-claimed its birth — "[fridge-filter] birth cand slice=0
        // CLAIMED by 'cabinet_w13_base' (35%)" — and no refrigerator was ever created.
        //
        // A producer predating the semantic branch publishes class_id < 1000 (or the -1 default), which reads
        // as an instance and keeps every existing consumer's behaviour.
        bool  is_semantic() const { return class_id >= 1000.0f; }
        bool  is_instance() const { return not is_semantic(); }
        // Depth-uncertainty channel (common/depth_projection). σ_range² (m²) to ADD to R along the mask
        // ray, SAME currency as motion_var — sum them. 0 for dense-depth zed masks; the scored range
        // variance for ricoh masks depth-filled from reprojected lidar (grows as hits get sparse/scattered,
        // → the mask degrades back toward bearing-only). 0 when the producer predates the field.
        float depth_var        = 0.0f;
        // Appearance channel (retina MaskColor.*, consumed by common/appearance_belief for the agent's
        // DISPLAY-mesh tint — no geometric fit reads it). color_chroma is the slice's median CHROMATICITY
        // (R,G,B)/(R+G+B), which is invariant to the per-frame illumination gain by construction; the
        // renderer applies its own shading, so raw RGB would be double-shaded. color_var is the scatter
        // BETWEEN grid cells, not between pixels — mask pixels on one surface are massively correlated,
        // and a per-pixel variance would collapse like 1/sqrt(N) and hugely overstate the precision.
        // color_neff counts contributing interior cells; 0 means NO colour information this frame (ricoh
        // bearing slices, tiny/distant masks, or a producer predating the feature) and consumers should
        // simply gain nothing rather than branch. Frame-independent: colour is a 2D image quantity, so
        // enable_frame_transform() does not affect it.
        Eigen::Vector3f color_chroma = Eigen::Vector3f::Zero();
        Eigen::Vector3f color_var    = Eigen::Vector3f::Zero();
        float           color_neff   = 0.0f;
    };

    // One ingested masks frame: every slice plus the shared support-point and raw-pixel arrays they index into.
    struct MasksPacket
    {
        bool valid = false;
        int frame_id = -1;
        std::uint64_t timestamp_ms = 0;   // capture stamp of the source RGBD frame (0 = producer didn't publish one)
        std::vector<MaskSlice> slices;
        std::vector<Eigen::Vector3f> support_points;
        // RAW camera/source-frame support points (from "mask_support_points_cam"), 1-to-1 with
        // support_points' indexing. Empty if the producer didn't dual-publish it. Independent of
        // the robot pose — used to build pose-independent observations (object-anchor z_o).
        std::vector<Eigen::Vector3f> support_points_cam;
        std::vector<Eigen::Vector2f> mask_pixels;   // raw YOLO foreground (col,row), depth-independent
        // Per-frame ego-motion (optical frame), written by the producer; empty/0 if absent.
        std::array<float, 6> cam_twist{};   // [vx,vy,vz,wx,wy,wz]
        float                frame_dt_s = 0.0f;
    };

    // Constructs its OWN InnerEigenAPI (a fresh instance per call by design) so every consumer gets the
    // zed→room transform without wiring one in. Safe: the lookup is pinned to the capture stamp (ts!=0),
    // which bypasses InnerEigenAPI's unlocked ts==0 cache entirely (CLAUDE.md, DSR thread-safety).
    explicit MaskIngestor(std::shared_ptr<DSR::DSRGraph> graph);

    // Override the default zed→room transform (e.g. a consumer that fits in a different frame). Passing
    // a null inner_eigen or an empty frame name reverts to reading the legacy ROOM-frame array as-is.
    void enable_frame_transform(DSR::InnerEigenAPI* inner_eigen,
                                std::string source_frame, std::string target_frame);

    // Frames the transform resolves; defaults "zed" → "room".
    void set_frames(std::string source_frame, std::string target_frame);

    // A/B control: read the producer's legacy ROOM-frame array verbatim instead of transforming the
    // camera-frame one here. Exists ONLY so the two paths can be compared on the same producer while
    // both arrays still exist; it becomes a no-op once the producer stops publishing the room array
    // (Phase 2). Also settable via MASK_INGESTOR_LEGACY_ROOM=1, which this overrides.
    void set_legacy_room_frame(bool on) { force_legacy_room_ = on; }

    // Forward-extrapolate the target←robot pose from the newest RT block to the mask capture stamp before
    // composing the transform (default ON, horizon 0.2 s — the producer's settings, ported verbatim).
    // See resolve_transform() for why this is not optional in practice.
    void set_pose_extrapolation(bool enabled, float max_dt_s = 0.2f);

    // Frames whose zed→room lookup failed (chain not resolvable at the capture stamp). These are DROPPED,
    // never passed through with an identity transform — identity would yield camera-frame points labelled
    // room, which is the exact defect this class was moved here to remove. A consumer skipping a cycle is
    // correct; a consumer fitting a corrupt cycle is not.
    [[nodiscard]] std::uint64_t dropped_no_transform() const noexcept { return dropped_no_transform_; }

    // Re-read the "masks" node. Returns true only when a NEW frame was ingested.
    bool refresh();

    // ── Producer-liveness probes (for a consumer's primary-input stream gate; see CONCEPT_AGENT_RECIPE.md) ──
    // Wall-clock ms since the last NEW masks frame was ingested (mask_frame_id advanced), or -1 if none ever
    // has. The producer's mask_frame_id is a monotonic PUBLISH counter that advances on every fresh camera
    // frame even with zero masks, so a non-advancing id reliably means producer-dead/stalled (an empty scene
    // never false-trips). Single-threaded (main-thread refresh + read), so no atomic.
    [[nodiscard]] std::int64_t ms_since_last_frame() const noexcept;

    // Graph-only readiness probe (usable from Waiting, before any frame): is the "masks" DSR node present and
    // carrying a mask_frame_id? True ⇒ refresh() will start yielding frames once the producer publishes.
    // `detail` (optional) receives a human-readable reason when false. Object-agnostic.
    [[nodiscard]] bool stream_ready(std::string* detail = nullptr) const;

    // The most recently ingested packet (read-only).
    const MasksPacket& packet() const { return masks_packet_; }
    // Mutable access for a consumer that post-processes the freshly-refreshed packet in place (e.g. splitting
    // an L-shaped mask into two sub-slices). All packet() readers then see the modified packet this cycle.
    MasksPacket& mutable_packet() { return masks_packet_; }

    // Nearest `label`-labelled slice to `query_centroid` (room frame). nullopt if none. The caller
    // builds the centroid from its model state, so this stays object-agnostic.
    std::optional<MaskSlice> select_nearest(const Eigen::Vector3f& query_centroid,
                                            std::string_view label) const;

    // Read a flat float3 point attribute (candidate_pts_att / residual_pts_att) off a node.
    std::vector<Eigen::Vector3f> read_pts_attrib(const DSR::Node& node, const std::string& att_name) const;

private:
    // tgt←src at the mask capture stamp, reproducing the chain the retina used to apply internally:
    // tgt←robot pinned to the stamp and forward-extrapolated over the RT lag, composed with the STATIC
    // robot←src camera mount. nullopt when the chain is not resolvable — callers must DROP the frame,
    // never substitute identity. Falls back to a direct tgt←src lookup when the chain cannot be split.
    [[nodiscard]] std::optional<Eigen::Matrix4d> resolve_transform(std::uint64_t stamp) const;

    std::shared_ptr<DSR::DSRGraph> G_;
    MasksPacket                    masks_packet_;
    int                            last_masks_frame_seen_ = -1;
    std::int64_t                   last_fresh_wall_ms_    = 0;   // wall-clock ms of the last NEW-frame ingest (0 = never)

    // src→tgt frame transform (see the FRAME CONTRACT above). `inner_eigen_` points either at
    // `owned_inner_eigen_` (the default) or at an instance handed in by enable_frame_transform.
    std::unique_ptr<DSR::InnerEigenAPI> owned_inner_eigen_;
    DSR::InnerEigenAPI* inner_eigen_       = nullptr;
    std::string         src_frame_         = "zed";
    std::string         tgt_frame_         = "room";
    bool                transform_enabled_ = false;
    bool                force_legacy_room_ = false;   // MASK_INGESTOR_LEGACY_ROOM=1 (A/B escape hatch)
    bool                pose_extrapolate_  = true;    // beat the room→robot RT lag (producer default)
    float               pose_extrap_max_dt_s_ = 0.2f; // extrapolation horizon clamp (producer default)
    // Memoized get_nodes_by_type("robot").front().name(), DROPPED as soon as that node leaves the
    // graph — a write-once memo here fed cortex a dead name once per mask frame. See resolve_transform.
    mutable std::string robot_name_;
    std::uint64_t       dropped_no_transform_ = 0;
};

}  // namespace rc
