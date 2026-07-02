/*
 * bottle_model.h
 *
 * Generative model for a bottle instance — a single vertical cylinder.
 *
 * State vector θ = [cx, cy, cz, radius, height]  (5 params)
 *   No yaw: a cylinder is rotationally symmetric about its vertical axis, so the
 *   azimuth is unobservable and is dropped (one fewer free parameter, no gauge
 *   ambiguity for the optimiser to chase).
 *   (cx, cy, cz) is the cylinder centre in the room frame; the bottle stands on
 *   the table, so cz ≈ table_top + height/2 — the host anchors cz's prior there.
 *
 * Cylinder SDF (axis = +Z): d_radial = ‖(x,y)−(cx,cy)‖ − radius,
 *                           d_vertical = |z−cz| − height/2.
 * Free Energy  = Σᵢ wᵢ · ρ(SDF(θ,yᵢ)) / σ²  +  KL(q‖p)
 * Gradient computed with PyTorch autograd and optimised with Adam/SGD,
 * mirroring TableModel so the perception loop is identical bar the geometry.
 *
 * pose_covariance() returns the 3×3 position covariance from the Laplace
 * (curvature) approximation of F at the optimum: many well-spread observations
 * ⇒ tight covariance; few/no points ⇒ only the prior curvature survives ⇒ the
 * covariance grows. This is the P_bottle the controller consumes off the RT edge
 * to decide how slowly/often to look.
 */

#pragma once

#include <array>
#include <cmath>
#include <utility>
#include <vector>
#include <Eigen/Dense>

namespace rc {

// ─── Parameter structs ────────────────────────────────────────────────────────

struct BottleState
{
    float cx     = 0.0f;    // Room-frame X of cylinder centre
    float cy     = 0.0f;    // Room-frame Y of cylinder centre
    float cz     = 0.85f;   // Room-frame Z of cylinder centre (table_top + height/2)
    float radius = 0.035f;  // Cylinder radius
    float height = 0.20f;   // Cylinder height (full, not half)

    // Serialise/deserialise as 5-vector
    std::array<float, 5> to_array() const
    {
        return {cx, cy, cz, radius, height};
    }

    static BottleState from_array(const std::array<float, 5>& a)
    {
        return {a[0], a[1], a[2], a[3], a[4]};
    }
};

// Prior geometry that seeds a new instance's state_ (radius/height/size std). The fit itself is the AI2
// belief (bottle_belief.h); this struct only carries the cold-start prior + physical extents.
struct BottleModelParams
{
    float prior_radius   = 0.035f;
    float prior_height   = 0.20f;
    float prior_size_std = 0.03f;
};

// ─── BottleModel ───────────────────────────────────────────────────────────────

class BottleModel
{
    public:
        using State = BottleState;

        // Minimum physical extents
        static constexpr float MIN_RADIUS = 0.005f;
        static constexpr float MIN_HEIGHT = 0.02f;

        BottleModel() = default;
        BottleModel(const BottleState& prior, const BottleModelParams& params);

        // ── SDF ──────────────────────────────────────────────────────────────────
        // BottleModel is the STATE CARRIER + cylinder SDF: the AI2 belief (bottle_belief.h) writes its
        // posterior back into state_, observe() uses the SDF to split candidate vs residual mask points,
        // and feed_silhouette stores the mask edge rays here for the belief's silhouette factor.

        /** SDF for a single 3-D point (room frame). */
        float sdf_point(const Eigen::Vector3f& p) const;

        /** SDF for a batch of points. Returns one value per point. */
        std::vector<float> compute_sdf(const std::vector<Eigen::Vector3f>& points) const;

        // ── State access ─────────────────────────────────────────────────────────

        const BottleState& state()  const { return state_; }
        const BottleState& prior()  const { return prior_; }
        const BottleModelParams& params() const { return params_; }

        void set_state(const BottleState& s) { state_ = s; apply_constraints(); }
        void set_prior(const BottleState& p) { prior_ = p; }

        // ── Silhouette (RGB-mask) observation ──────────────────────────────────────
        // Edge rays from the mask's left/right silhouette, in the ROOM frame: a shared camera
        // centre (Cx,Cy) and one horizontal direction (dx,dy) per edge pixel. The occluding-
        // contour condition is "ray tangent to the vertical cylinder" → δ(ray;cx,cy) = radius.
        // Cleared each cycle; set when a fresh bottle mask + camera model are available.
        // confidence ∈ [0,1] (YOLO detection score) scales the effective silhouette precision: an
        // unreliable / over-segmented mask (low conf) must not pull radius toward its bloat.
        void set_silhouette(const Eigen::Vector2f& cam_xy, std::vector<Eigen::Vector2f> edge_dirs_xy,
                            float confidence = 1.0f)
        { sil_cam_xy_ = cam_xy; sil_dirs_ = std::move(edge_dirs_xy); sil_conf_ = std::clamp(confidence, 0.0f, 1.0f); }
        void clear_silhouette() { sil_dirs_.clear(); }
        std::size_t silhouette_ray_count() const { return sil_dirs_.size(); }
        // Read-only access for the AI2 belief path (bottle_belief's occluding-contour tangent factor).
        const Eigen::Vector2f&              silhouette_cam_xy() const { return sil_cam_xy_; }
        const std::vector<Eigen::Vector2f>& silhouette_dirs()   const { return sil_dirs_; }
        float                               silhouette_conf()   const { return sil_conf_; }

        /** Enforce positive, physically plausible radius and height. */
        void apply_constraints();

        /**
         * Axis-aligned bounding box of the cylinder in room frame.
         * Returns {min_corner, max_corner}.
         */
        std::pair<Eigen::Vector3f, Eigen::Vector3f> bounding_box() const;

    private:
        // SDF evaluated for a given explicit state
        float sdf_point_at(const Eigen::Vector3f& p, const BottleState& s) const;

        BottleState        state_;
        BottleState        prior_;
        BottleModelParams  params_;

        // Silhouette observation (room frame); empty ⇒ term inactive.
        Eigen::Vector2f                 sil_cam_xy_ = Eigen::Vector2f::Zero();
        std::vector<Eigen::Vector2f>    sil_dirs_;
        float                           sil_conf_ = 1.0f;   // YOLO confidence → precision scale
};

}  // namespace rc
