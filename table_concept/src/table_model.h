/*
 * table_model.h
 *
 * Generative model for a table instance.
 *
 * State vector θ = [cx, cy, w, h, table_height, leg_length, yaw]  (7 params)
 * Fixed geometry: TOP_THICKNESS = 0.03 m, LEG_RADIUS = 0.025 m
 *
 * Compound SDF = min(SDF_top, min_k SDF_leg_k)
 * Free Energy  = Σᵢ wᵢ · ρ(SDF(θ,yᵢ)) / σ²  +  λ‖θ−θ_prior‖²_Σ
 * Gradient computed with PyTorch autograd and optimised with Adam/SGD.
 */

#pragma once

#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <Eigen/Dense>

#include "../../common/robust_metrics/robust_metrics.h"

// ─── Parameter structs ────────────────────────────────────────────────────────


namespace rc {
struct TableState
{
    float cx           = 0.0f;   // Room-frame X of table centre
    float cy           = 0.0f;   // Room-frame Y of table centre
    float w            = 1.0f;   // Width  (table-local X)
    float h            = 0.6f;   // Depth  (table-local Y)
    float table_height = 0.75f;  // Height of table surface from floor
    float leg_length   = 0.72f;  // Length of legs from floor to underside of top
    float yaw          = 0.0f;   // Rotation around Z axis (room frame)
    float leg_inset    = 0.045f; // Distance from each top edge to the leg centre (top overhang)

    // Serialise/deserialise as 8-vector
    std::array<float, 8> to_array() const
    {
        return {cx, cy, w, h, table_height, leg_length, yaw, leg_inset};
    }

    static TableState from_array(const std::array<float, 8>& a)
    {
        return {a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]};
    }
};

// Footprint-extent diagnostic: the half-extent of the observed point span in table-local frame at
// a few percentile pairs (so we can see how much trimming changes it), the local centre offset of
// that span, and how the points split between the top slab and the legs. Used to diagnose whether a
// w/h over-estimate comes from off-table margin points inflating the span vs the model overshooting.
struct ExtentDiagnostics
{
    float half_ex[3] = {0, 0, 0};   // half-extent in local X at [2-98, 5-95, 10-90] percentile
    float half_ey[3] = {0, 0, 0};   // half-extent in local Y
    float off_x = 0.0f, off_y = 0.0f;  // local-frame centre of the 2-98 span (0 = centred on the model)
    int   n_top = 0, n_leg = 0, n_total = 0;  // point attribution (compound SDF: top slab vs nearest leg)
};

struct TableModelParams
{
    // Observation noise (SDF likelihood width)
    float sigma_obs = 0.05f;

    // KL regulariser weights (toward prior)
    float lambda_size  = 0.5f;   // Prior table dimensions
    float lambda_pos   = 0.05f;  // Position stays close to prior
    float lambda_state = 0.02f;  // Size state transition
    float lambda_angle = 0.01f;  // Angle transition
    // Footprint-extent term: pull the top-face rectangle (centre + half-extents) to the observed
    // point extent in table-local frame. Breaks the flat-interior translation/size degeneracy that
    // lets an oversized, off-centre box sit at ~zero data energy. 0 disables.
    float lambda_extent = 2.0f;
    // Percentile pair defining the observed point span the extent term pins the footprint to. Tighter
    // (e.g. 0.05/0.95) trims the depth-noise margin that overshoots the true edge; looser (0.02/0.98)
    // is more inclusive under partial views. Symmetric pair in [0,1].
    float extent_pct_lo = 0.05f;
    float extent_pct_hi = 0.95f;

    // Prior geometry (used by KL term)
    float prior_w            = 1.0f;
    float prior_h            = 0.6f;
    float prior_table_height = 0.75f;
    float prior_size_std     = 0.15f;

    // Gradient-descent optimiser
    int   optimization_iters = 10;
    float optimization_lr    = 0.05f;
    float grad_clip          = 2.0f;

    // Optimizer selection for autograd loop: "adam" or "sgd"
    std::string optimizer_type = "adam";
    // Momentum used when optimizer_type == "sgd"
    float sgd_momentum = 0.9f;
    // Robust loss applied to SDF residuals before scaling by 1 / sigma_obs^2
    RobustLossType robust_loss = RobustLossType::Quadratic;
    float robust_loss_scale = 0.10f;
    // RGB-mask silhouette precision (0 disables). Weights the differentiable term that pulls the
    // table-top rectangle onto the back-projected 2D mask contour — constrains the footprint
    // (incl. occluded faces & yaw) from the image where depth is missing.
    float mask_precision = 0.0f;
    // Silhouette mode: >0 = height-agnostic occluding-contour (sample each back-projected mask-edge
    // ray over the table's vertical band, drive the closest approach to the box+legs SDF → 0, so the
    // ray grazes the model — top, sides AND legs). 0 = legacy table-top-plane intersection only.
    int   sil_tangent_samples = 8;
    // Graduated non-convexity (GNC): within each gradient_step, anneal the robust-loss scale
    // geometrically from robust_gnc_start_scale down to robust_loss_scale across the iterations.
    // A wide start keeps far points (e.g. a table half-offset past the box edge) inside the
    // robust band so they still produce gradient during coarse alignment; the scale then
    // sharpens to the target so the final pose is precise. <= robust_loss_scale disables it.
    float robust_gnc_start_scale = 0.0f;
    // Clamp on the per-point finite-difference slope |∂SDF/∂θ| inside observation_information, to
    // suppress the spurious ~1000× Fisher spikes a point on a non-smooth min()-SDF kink would inject.
    // For a well-behaved SDF point the slope is O(1). 0 disables the clamp.
    float fisher_grad_clamp = 2.0f;
};

struct FreeEnergyDecomposition
{
    float likelihood_current   = 0.0f;
    float likelihood_historical = 0.0f;
    float prior                = 0.0f;
    float total_fe             = 0.0f;
    float effective_weight_mass = 0.0f;
    float current_weight_mass  = 0.0f;
    float historical_weight_mass = 0.0f;
};

// ─── TableModel ──────────────────────────────────────────────────────────────

class TableModel
{
public:
    using IterationObserver = std::function<void(int, const TableState&, const FreeEnergyDecomposition&)>;

    static constexpr float TOP_THICKNESS = 0.03f;
    static constexpr float LEG_RADIUS    = 0.025f;
    static constexpr float SDF_SMOOTH_K  = 0.02f;
    static constexpr float SDF_INSIDE_SCALE = 0.3f;

    TableModel() = default;
    TableModel(const TableState& prior, const TableModelParams& params);

    // ── SDF ──────────────────────────────────────────────────────────────────

    /** SDF for a single 3-D point (room frame). */
    float sdf_point(const Eigen::Vector3f& p) const;

    /** SDF for a batch of points. Returns one value per point. */
    std::vector<float> compute_sdf(const std::vector<Eigen::Vector3f>& points) const;

    /**
     * Per-DOF observation Fisher information (diagonal of the Gauss-Newton Hessian of the SDF
     * data-likelihood) at the current state, in the state order [cx,cy,w,h,H,leg,yaw,inset].
     *
     *   I_jj = (1/σ²) · Σᵢ wᵢ · ω(rᵢ) · (∂SDF(yᵢ)/∂θ_j)²
     *
     * with rᵢ = SDF(yᵢ) the residual, ω the robust IRLS weight (so outliers contribute little
     * curvature), and ∂SDF/∂θ_j a central finite difference on the scalar SDF path. UN-normalised
     * (no /N) so it accumulates with the number of supporting points — the calibrated "amount of
     * evidence this viewpoint provides about each DOF". This is the per-frame measurement of the
     * Fisher information filter that hardens the belief as the robot orbits the table.
     */
    std::array<float, 8> observation_information(const std::vector<Eigen::Vector3f>& points,
                                                 const std::vector<float>& weights) const;

    /** Footprint-extent + top/leg attribution diagnostic at the current state (see ExtentDiagnostics). */
    ExtentDiagnostics extent_diagnostics(const std::vector<Eigen::Vector3f>& points) const;

    // ── Free Energy ──────────────────────────────────────────────────────────

    /**
    * F(θ) = Σᵢ wᵢ·ρ(SDF(yᵢ))/σ²  +  KL(q‖p)
     *
     * @param points   Room-frame 3-D observations.
     * @param weights  Per-point weights (1.0 = equal); pass empty for uniform.
     */
    float compute_free_energy(const std::vector<Eigen::Vector3f>& points,
                              const std::vector<float>& weights) const;

    FreeEnergyDecomposition compute_free_energy_decomposition(
        const std::vector<Eigen::Vector3f>& points,
        const std::vector<float>& weights,
        std::size_t historical_count) const;

    // ── Gradient step ────────────────────────────────────────────────────────

    /**
     * Perform `params_.optimization_iters` gradient-descent steps on θ.
     * Returns the free energy after the last step.
     */
    float gradient_step(const std::vector<Eigen::Vector3f>& points,
                        const std::vector<float>& weights,
                        std::size_t historical_count = 0,
                        const IterationObserver& observer = {},
                        float gnc_start_override = -1.0f);

    // ── State access ─────────────────────────────────────────────────────────

    const TableState& state()  const { return state_; }
    const TableState& prior()  const { return prior_; }
    const TableModelParams& params() const { return params_; }

    void set_state(const TableState& s) { state_ = s; apply_constraints(); }
    void set_prior(const TableState& p) { prior_ = p; }

    // RGB-mask silhouette: camera centre (room) + back-projected 3-D contour ray directions (room).
    // Each ray is intersected with the table-top plane (z=table_height) at fit time; the resulting
    // room-XY point should lie on the top rectangle boundary. Set per-frame by the fitter.
    void set_silhouette(const Eigen::Vector3f& cam_room, std::vector<Eigen::Vector3f> ray_dirs_room, float confidence)
    { sil_cam_ = cam_room; sil_dirs_ = std::move(ray_dirs_room); sil_conf_ = std::clamp(confidence, 0.0f, 1.0f); }
    void clear_silhouette() { sil_dirs_.clear(); }
    std::size_t silhouette_ray_count() const { return sil_dirs_.size(); }
    // Raw mean closest-approach of the silhouette rays to the model at the current state (metres);
    // 0 = perfectly on the contour. Diagnostic, independent of mask_precision / robust loss.
    float silhouette_residual() const;

    /**
     * Clamp leg_length so that leg_length ≤ table_height − TOP_THICKNESS,
     * and enforce positive dimensions.
     */
    void apply_constraints();

    /**
     * Axis-aligned bounding box of the full table (top + legs) in room frame.
     * Returns {min_corner, max_corner}.
     */
    std::pair<Eigen::Vector3f, Eigen::Vector3f> bounding_box() const;

private:
    // Evaluate free energy for an arbitrary state (scalar float path)
    float fe_at(const TableState& s,
                const std::vector<Eigen::Vector3f>& pts,
                const std::vector<float>& weights) const;

    FreeEnergyDecomposition fe_terms_at(const TableState& s,
                                        const std::vector<Eigen::Vector3f>& pts,
                                        const std::vector<float>& weights,
                                        std::size_t historical_count) const;

    // Prior (size regularisation) energy for a given state
    float prior_energy(const TableState& s) const;

    // RGB-mask silhouette energy for a given state (scalar mirror of the autograd term).
    float silhouette_energy_at(const TableState& s) const;
    // Per-ray silhouette residual (closest approach, tangent mode; or top-plane 2-D box-SDF).
    float silhouette_ray_metric(const Eigen::Vector3f& d, const TableState& s) const;

    // SDF evaluated for a given explicit state
    float sdf_point_at(const Eigen::Vector3f& p, const TableState& s) const;

    TableState        state_;
    TableState        prior_;
    TableModelParams  params_;

    // RGB-mask silhouette evidence (room frame), set per-frame; empty ⇒ term inactive.
    Eigen::Vector3f              sil_cam_ = Eigen::Vector3f::Zero();
    std::vector<Eigen::Vector3f> sil_dirs_;
    float                        sil_conf_ = 1.0f;
};

}  // namespace rc
