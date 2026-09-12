#pragma once

#include <QObject>
#include <QWidget>
#include <QRectF>
#include <QTransform>
#include <QPointF>
#include <QPen>
#include <QBrush>
#include <QColor>
#include <QGraphicsScene>
#include <QGraphicsPolygonItem>
#include <QGraphicsEllipseItem>
#include <QGraphicsLineItem>
#include <QGraphicsRectItem>
#include <QGraphicsTextItem>
#include <QTimer>
#include <Eigen/Dense>
#include <cstdint>
#include <optional>
#include <vector>
#include <deque>
#include <memory>
#include <unordered_map>
#include "corner_detector.h"
#include "image_edge_types.h"   // rc::TriplePoint — RGB corners drawn beside the LiDAR ones
#include <map>
#include "object_anchor_types.h"
#include "wall_map.h"

class AbstractGraphicViewer;
namespace DSR { class DSRGraph; }

namespace rc {

/// What the wall-SLAM estimator knows about ITSELF this frame — the HUD line's inputs.
/// Namespace scope, not nested in Viewer2D: a nested class's default member initializers cannot be
/// used by a default argument declared inside the same (still incomplete) enclosing class.
struct WallMapStatus
{
    bool  have_result = false;   ///< a localiser frame exists at all (false ⇒ seed only)
    int   candidates  = 0;       ///< lines under trial, not yet born
    int   births      = 0;       ///< walls born on this frame
    bool  theta0_born = false;
    float theta0      = 0.f;     ///< rad, the room's reference direction
    /// Per-segment association: index of the wall it was matched to, or −1 for a segment no wall
    /// explains. A raw pointer, not a reference, so the struct keeps a default: it is borrowed from
    /// the caller's WallView for the duration of one draw_wall_map() call and never stored.
    const std::vector<int>* seg_to_wall = nullptr;
};

/**
 * @brief 2D scene viewer for the SLAMO component.
 *
 * Wraps AbstractGraphicViewer and owns all QGraphicsItem* scene objects:
 * robot covariance ellipse, room polygon, furniture polygons, path lines,
 * trajectory debug overlays, and temporary obstacle polygons.
 *
 * All drawing is performed via typed update methods so that SpecificWorker
 * never touches QGraphicsScene directly.
 *
 * Usage:
 *   viewer_2d_ = std::make_unique<rc::Viewer2D>(this->frame, grid_rect, true);
 *   viewer_2d_->add_robot(w, l, 0.0, 0.2, QColor("Blue"));
 *   viewer_2d_->show();
 *   layout->addWidget(viewer_2d_->get_widget());
 *
 *   // per frame:
 *   viewer_2d_->update_robot(robot_pose);
 */
class Viewer2D : public QObject
{
    Q_OBJECT
    public:
        explicit Viewer2D(QWidget* parent, const QRectF& grid_dim, bool show_axis = true);
        ~Viewer2D() override = default;

        // ----- Widget / view management -----
        QWidget* get_widget() const;
        void add_robot(float w, float l, float offset, float rotation, QColor color);
        void show();
        QTransform transform() const;
        void set_transform(const QTransform& t);
        void fit_to_scene(const QRectF& r);
        void fit_view(float margin_ratio = 0.05f);
        void fit_to_robot_and_points(const Eigen::Affine2f& robot_pose,
                         const std::vector<Eigen::Vector3f>& lidar_points,
                         float fit_radius,
                         float margin_ratio = 0.12f);
        void center_on(float x, float y);

        // ----- Robot -----
        void update_robot(const Eigen::Affine2f& robot_pose);

        // ----- Covariance ellipse -----
        void update_covariance_ellipse(float cx, float cy,
                                    float radius_x, float radius_y,
                                    float angle_deg);
        // ----- Estimated room rect (shown when room polygon is absent) -----
        void update_estimated_room_rect(float width, float length, bool has_polygon);

        // ----- Room polygon outline -----
        void draw_room_polygon(const std::vector<Eigen::Vector2f>& verts, bool is_capturing);

        // ----- Lidar -----
        void draw_lidar_points(const std::vector<Eigen::Vector3f>& points_high,
                       const std::vector<Eigen::Vector3f>& points_low,
                       const Eigen::Affine2f& robot_pose,
                       int max_points_high);
        void set_lidar_points_visible(bool visible);
        bool lidar_points_visible() const;

        // ----- Composite per-frame update -----
        struct FrameData
        {
            const std::vector<Eigen::Vector3f>& lidar_points;
            Eigen::Affine2f display_pose;
            Eigen::Matrix3f covariance = Eigen::Matrix3f::Identity();
            int max_lidar_points;
            bool have_loc;
            bool is_initialized;
            bool has_room_polygon;
            float room_width;
            float room_length;
            Eigen::Affine2f loc_pose;    // Raw localization pose (no forward projection)
            bool use_loc_pose;           // If true, draw lidar with loc_pose
        };
        void update_frame(const FrameData& fd);
    
        // ----- Path -----
        struct PathDrawData
        {
            std::vector<Eigen::Vector2f> path;
            std::vector<Eigen::Vector2f> orig_poly_verts;
            std::vector<Eigen::Vector2f> inner_poly;
            std::vector<Eigen::Vector2f> nav_poly;
            std::vector<std::vector<Eigen::Vector2f>> expanded_obstacles;
        };
        void draw_path(const PathDrawData& data);
        /// Remove path lines / waypoints / expanded-obstacle overlays; hide target marker.
        void clear_path_items();

        // ----- Epistemic target marker -----
        /// Show a target marker at the given room-frame position, or hide it.
        void update_target_marker(float x, float y, bool visible);

        /// Draw the selected arc trajectory as a polyline on the canvas.
        void draw_trajectory(const std::vector<Eigen::Vector3f>& predicted_states);

        /// Draw all candidate trajectories as faded polylines, then the best one on top.
        void draw_all_trajectories(const std::vector<std::vector<Eigen::Vector3f>>& candidates,
                                   const std::vector<Eigen::Vector3f>& best);

        /// Draw the object-anchor landmarks (fridge, …): the PINNED map anchor p_o, this frame's
        /// observation z_o carried to world, the sight line robot→z_o, and the innovation covariance
        /// S = Λ⁻¹ as an oriented 1σ ellipse at the observation. The p_o→z_o gap IS the residual the
        /// factor is minimising, so it is drawn explicitly rather than left to be inferred.
        /// `landmarks` (node id → room-frame position) is room's PRIVATE estimate when it is optimising
        /// them. Drawn as a separate marker joined to the producer's published pose, because the whole
        /// justification for keeping the estimate private is that their disagreement stays visible.
        void draw_object_anchors(const std::vector<rc::ObjectAnchorObs>& anchors,
                                 const Eigen::Affine2f& robot_pose,
                                 const std::map<std::uint64_t, Eigen::Vector2f>& landmarks = {});

        /// Draw detected corners: green circles for accepted, yellow for predicted/in-FOV.
        /// Also draws a line from the robot to each detected corner.
        /// RGB triple points (floor+wall+wall) beside the LiDAR corners. Same physical (x, y) seen
        /// by a different sensor, which is the whole reason to draw them together: the gap between a
        /// magenta square and its cyan neighbour IS the RGB-vs-LiDAR disagreement, in metres, on the
        /// canvas. Squares not circles, and magenta not cyan, so the two remain separable in a
        /// screenshot and for a viewer who cannot tell the hues apart.
        /// Each drawn corner also gets a DOTTED sight line back to the robot, so the RGB channel's
        /// geometry can be read the same way the LiDAR one is. Dotted, because the LiDAR sight lines
        /// are solid (live) or dashed (retired) — the stroke pattern alone says which sensor it is.
        /// `robot_pose` is only used for the robot→corner sight lines; the corner positions
        /// themselves are already in the room frame.
        void draw_rgb_corners(const std::vector<rc::TriplePoint>& points,
                              const Eigen::Affine2f& robot_pose);

        void draw_corners(const std::vector<rc::CornerDetector::CornerMatch>& matches,
                          const Eigen::Affine2f& robot_pose);

        /// Draw a line from the robot to each pinned-landmark object (room-frame positions). The
        /// sight line is shown only when `measured[i]` is true (object actively detected this frame);
        /// the landmark marker is always shown. `measured` may be shorter/empty → treated as measured.
        void draw_landmark_lines(const std::vector<Eigen::Vector2f>& landmarks_world,
                                 const std::vector<char>& measured,
                                 const Eigen::Vector2f& robot_xy);

        /// Wall-SLAM overlay (Estimate mode): this frame's segments (robot frame, drawn in the room
        /// frame through `robot_pose`), the wall landmarks clipped to their observed extents with
        /// their class, the derived corners (filled = observed, hollow = inferred) and the derived
        /// polygon (dashed until it is published, solid after).
        /// `publish_bar` is Params::publish_corner_sigma — the σ every corner must reach to be
        /// publishable. It is drawn, not just tested: a corner disc is green under it and orange
        /// over it, so "is the layout trustworthy yet" is answered on the canvas rather than in a log.
        ///
        /// EVERY layer here survives an OPEN polygon. WallMap::build_from() fills verts/corners only
        /// inside its `if (closed)` branch, so before the cycle closes the polygon carries nothing at
        /// all — and gating the outline, the edge bands and the HUD on `polygon.closed` left the first
        /// minutes of an Estimate run with the wall landmarks as the only thing on screen. The walls
        /// now carry their OWN offset band (σ_d = 1/√Λ_dd), which exists from the frame a wall is born,
        /// and the HUD reports the open state instead of hiding until it is over.
        void draw_wall_map(const std::vector<rc::wallseg::WallSegment>& segments,
                           const std::vector<rc::wallmap::WallLandmark>& walls,
                           const rc::wallmap::Polygon& polygon, bool map_ready,
                           const Eigen::Affine2f& robot_pose, float publish_bar = 0.06f,
                           const WallMapStatus& status = WallMapStatus{});

        /// Draw the epistemic score grid as semi-transparent coloured cells.
        /// cell_size is in world-frame meters.
        void draw_score_grid(const std::vector<std::pair<Eigen::Vector2f, float>>& cells,
                             float cell_size);

        /// Draw the IoR inhibition overlay: warm-red cells whose alpha encodes
        /// freshness (1=just visited → opaque, 0=stale → transparent).
        void draw_ior_grid(const std::vector<std::pair<Eigen::Vector2f, float>>& cells,
                           float cell_size);

        /// Highlight the currently selected target cell.
        /// Pass std::nullopt to hide the highlight.
        void draw_selected_grid_cell(const std::optional<Eigen::Vector2f>& center,
                         float cell_size);

        /// Low-rate poll of DSR graph nodes of type object/obstacle and draw their BBs.
        /// Missing nodes are automatically removed from the canvas.
        void refresh_semantic_bboxes(const std::shared_ptr<DSR::DSRGraph>& graph);

        /// Thread-safe, self-driven object/obstacle BB overlay: starts a QTimer owned by
        /// this viewer (so it fires on the GUI thread, where the DSR reads and the
        /// QGraphicsScene mutation are both safe) that polls the graph every period_ms.
        /// Decoupled from the compute loop; call once after construction (main thread).
        void start_semantic_bbox_overlay(std::shared_ptr<DSR::DSRGraph> graph, int period_ms = 1000);

    Q_SIGNALS:
        void robot_moved(QPointF);
        void robot_rotate(QPointF);
        void robot_dragging(QPointF);
        void robot_drag_end(QPointF);
        void new_mouse_coordinates(QPointF);
        void right_click(QPointF);

    private:
        AbstractGraphicViewer* agv_ = nullptr;

        // Covariance ellipse
        QGraphicsEllipseItem* cov_ellipse_item_ = nullptr;

        // Estimated room rect (shown when no polygon is loaded)
        QGraphicsRectItem* estimated_room_item_ = nullptr;

        // Room polygon capture
        std::vector<QGraphicsEllipseItem*> capture_vertex_items_;

        // Room polygon outline
        // Wall-SLAM overlay pools (see draw_wall_map)
        std::vector<QGraphicsLineItem*>       wall_seg_items_;
        std::vector<QGraphicsLineItem*>       wall_line_items_;
        std::vector<QGraphicsEllipseItem*>    wall_corner_items_;
        std::vector<QGraphicsSimpleTextItem*> wall_label_items_;
        QGraphicsPolygonItem*                 wall_poly_item_ = nullptr;
        // ── UNCERTAINTY LAYER: what the model knows about the outline it is drawing ──────────────
        // A corner is never measured — it is where two estimated lines cross — so it inherits their
        // covariance, and a corner whose two walls are never seen well together keeps a large σ no
        // matter how long the robot looks. Drawn as a disc of radius σ. The edge bands are each
        // wall's own offset σ_d = 1/√Λ_dd, so a thinly-observed wall visibly blurs. The ghosts are
        // the last few published outlines, which is how CHURN becomes visible on a live canvas:
        // a settled map shows one outline, a churning one shows a fan.
        std::vector<QGraphicsEllipseItem*>    wall_sigma_items_;
        std::vector<QGraphicsSimpleTextItem*> wall_sigma_label_items_;   // σ of a corner drawn off the scale
        std::vector<QGraphicsLineItem*>       wall_band_items_;
        // The per-WALL band, drawn from the frame a wall is born and long before any polygon closes:
        // same σ_d = 1/√Λ_dd as the per-edge band, but attached to the landmark's own observed extent
        // instead of to a polygon edge that does not exist yet.
        std::vector<QGraphicsLineItem*>       wall_lm_band_items_;
        // The polygon as an OPEN chain: QGraphicsPolygonItem always closes its outline, so a partial
        // cycle needs its own line pool. Visible only while the cycle is not closed — the two never
        // draw the same edge twice.
        std::vector<QGraphicsLineItem*>       wall_chain_items_;
        std::vector<QGraphicsPolygonItem*>    wall_ghost_items_;
        std::deque<QPolygonF>                 wall_ghosts_;
        QGraphicsSimpleTextItem*              wall_hud_item_ = nullptr;
        int                                   wall_ghost_tick_ = 0;
        QGraphicsPolygonItem* polygon_item_         = nullptr;
        QGraphicsPolygonItem* polygon_fill_item_    = nullptr;   // warm floor fill (interior only)
        QGraphicsPolygonItem* polygon_item_backup_  = nullptr;
        QGraphicsLineItem* room_axis_x_item_        = nullptr;
        QGraphicsLineItem* room_axis_y_item_        = nullptr;
        QGraphicsTextItem* room_axis_x_label_       = nullptr;
        QGraphicsTextItem* room_axis_y_label_       = nullptr;

        // Furniture
        std::vector<QGraphicsPolygonItem*> furniture_draw_items_;

        // Temporary obstacles
        std::vector<QGraphicsPolygonItem*> temp_obstacle_draw_items_;

        // Lidar point pools (reused each frame)
        std::vector<QGraphicsEllipseItem*> lidar_pool_high_;
        std::vector<QGraphicsEllipseItem*> lidar_pool_low_;
        bool lidar_points_visible_ = false;   // LIDAR overlay starts OFF; toggled on via btn_lidar_points_viz

        // Path
        std::vector<QGraphicsItem*>         path_draw_items_;
        QGraphicsEllipseItem*               target_marker_        = nullptr;
        QGraphicsPolygonItem*               navigable_poly_item_  = nullptr;
        std::vector<QGraphicsPolygonItem*>  obstacle_expanded_items_;

        // Epistemic target
        QGraphicsEllipseItem* epistemic_target_item_ = nullptr;

        // ── Corner detection markers ─────────────────────────────────────────────────────────────
        // One grammar across both channels, so a screenshot is readable without a legend:
        //   SHAPE = which sensor  — circle = LiDAR, square = RGB.
        //   FILL  = what it is    — filled = MEASURED (what the sensor says), hollow = MODEL (where
        //                           the room polygon predicts it), the residual line joining the pair.
        //   HUE   = the channel   — LiDAR blues (bright cyan measured, dark blue model),
        //                           RGB warms (orange measured, dark brown model).
        std::vector<QGraphicsRectItem*>    rgb_corner_items_;
        std::vector<QGraphicsRectItem*>    rgb_model_items_;             // model vertex, hollow dark-brown square
        std::vector<QGraphicsLineItem*>    rgb_corner_line_items_;       // model vertex → measured crossing (the residual)
        std::vector<QGraphicsLineItem*>    rgb_corner_robot_line_items_; // robot → measured RGB corner (sight line)
        std::vector<QGraphicsEllipseItem*> corner_detected_items_;
        std::vector<QGraphicsEllipseItem*> corner_predicted_items_;
        std::vector<QGraphicsLineItem*>    corner_line_items_;
        std::vector<QGraphicsLineItem*>    corner_robot_line_items_;   // robot → detected corner

        // Landmark markers (pinned objects): robot → landmark lines
        std::vector<QGraphicsLineItem*>    landmark_line_items_;
        std::vector<QGraphicsEllipseItem*> landmark_marker_items_;

        // ----- Object anchors (fridge, …) -----
        std::vector<QGraphicsEllipseItem*> anchor_pin_items_;      // pinned map anchor p_o
        std::vector<QGraphicsEllipseItem*> anchor_obs_items_;      // this frame's observation z_o
        std::vector<QGraphicsEllipseItem*> anchor_cov_items_;      // 1σ innovation ellipse at z_o
        std::vector<QGraphicsLineItem*>    anchor_sight_items_;    // robot → z_o
        std::vector<QGraphicsLineItem*>    anchor_resid_items_;    // z_o → p_o (the residual)
        std::vector<QGraphicsTextItem*>    anchor_text_items_;
        std::vector<QGraphicsEllipseItem*> anchor_lm_items_;      // room's private landmark estimate
        std::vector<QGraphicsLineItem*>    anchor_lm_link_items_; // private estimate ↔ published pose

        // Trajectory overlay
        std::vector<QGraphicsLineItem*>   traj_line_items_;
        std::vector<QGraphicsEllipseItem*> traj_dot_items_;

        // Candidate trajectory overlay (faded)
        std::vector<QGraphicsLineItem*>   cand_line_items_;

        // Score grid overlay
        std::vector<QGraphicsRectItem*> score_grid_items_;
        QGraphicsRectItem* selected_grid_cell_item_ = nullptr;

        // IoR inhibition overlay (warm-red freshness map)
        std::vector<QGraphicsRectItem*> ior_grid_items_;

        // Object/obstacle semantic BB overlay
        std::unordered_map<std::uint64_t, QGraphicsPolygonItem*> object_bbox_items_;
        std::unordered_map<std::uint64_t, QGraphicsPolygonItem*> obstacle_bbox_items_;
        std::unordered_map<std::uint64_t, QGraphicsPolygonItem*> table_bbox_items_;
        std::shared_ptr<DSR::DSRGraph> semantic_graph_;   // graph polled by the BB overlay timer
        QTimer* semantic_bbox_timer_ = nullptr;           // self-driven 1 Hz overlay (GUI thread)

        void update_room_axes(const QRectF& room_bounds);

    };

} // namespace rc
