#include "corner_detector.h"
#include "corner_visibility.h"
#include <cmath>
#include <algorithm>
#include <ranges>

namespace rc {

// ---------------------------------------------------------------------------
//  set_model_corners — keep only real corners and store wall edge directions
// ---------------------------------------------------------------------------
void CornerDetector::set_model_corners(const std::vector<Eigen::Vector2f>& polygon_vertices)
{
    model_corners_.clear();
    model_dups_dropped_ = 0;
    short_wall_dropped_.clear();
    yield_.clear();
    slot_of_original_.assign(polygon_vertices.size(), -1);
    polygon_ = polygon_vertices;   // retained for the occlusion/visibility ray-cast in detect()
    const int N = static_cast<int>(polygon_vertices.size());
    if (N < 3) return;

    // Shortest adjacent wall a vertex must have to be claimed as a LANDMARK — see
    // Params::min_wall_map_sigmas. Note polygon_ above already holds every vertex: what follows
    // decides landmark status only, never geometry.
    const float min_wall_for_landmark = params_.min_wall_map_sigmas > 0.f
                                      ? params_.min_wall_map_sigmas * params_.map_sigma
                                      : 0.f;

    for (int i = 0; i < N; ++i)
    {
        const Eigen::Vector2f& prev = polygon_vertices[(i + N - 1) % N];
        const Eigen::Vector2f& curr = polygon_vertices[i];
        const Eigen::Vector2f& next = polygon_vertices[(i + 1) % N];

        // Exclusion at the MODEL level: a repeated/duplicated polygon vertex (SVG authoring artefact)
        // produces a zero-length edge whose .normalized() is NaN — the angle test then silently passes
        // or fails at random and, when it passes, two model corners sit on the same physical point and
        // BOTH predict + detect the same wall intersection. Drop degenerate edges outright.
        if ((prev - curr).squaredNorm() < 1e-8f or (next - curr).squaredNorm() < 1e-8f)
            { model_dups_dropped_++; continue; }

        // Landmark admissibility (see min_wall_for_landmark above). Applied BEFORE the angle test so
        // a 6 cm trace wobble never reaches prediction, association or the loss: it cannot be
        // distinguished from a straight wall by a map with map_sigma of error, and its neighbour
        // vertex sits well inside the distance the sensor can resolve, so the pair can only alias.
        const float min_adjacent_wall = std::min((prev - curr).norm(), (next - curr).norm());
        if (min_adjacent_wall < min_wall_for_landmark)
            { short_wall_dropped_.push_back(i); continue; }

        const Eigen::Vector2f d1 = (prev - curr).normalized();
        const Eigen::Vector2f d2 = (next - curr).normalized();
        const float dot = std::clamp(d1.dot(d2), -1.0f, 1.0f);
        const float angle_deg = std::acos(dot) * 180.0f / static_cast<float>(M_PI);

        if (angle_deg >= params_.min_corner_angle && angle_deg <= params_.max_corner_angle)
        {
            ModelCorner mc;
            mc.position       = curr;
            mc.edge_in_dir    = (curr - prev).normalized();  // wall arriving at corner
            mc.edge_out_dir   = (next - curr).normalized();  // wall leaving corner
            mc.convexity_sign = mc.edge_in_dir.x() * mc.edge_out_dir.y()
                              - mc.edge_in_dir.y() * mc.edge_out_dir.x();
            mc.wall_in_length  = (curr - prev).norm();
            mc.wall_out_length = (next - curr).norm();
            mc.original_index = i;

            // Exclusion at the MODEL level (2): two kept corners must not coincide. Non-adjacent
            // vertices can still land on the same point in a self-touching polygon; keeping both makes
            // the Hungarian arbitrate a tie it cannot resolve and yields the duplicate pair seen in the UI.
            const bool coincides = std::ranges::any_of(model_corners_, [&](const ModelCorner& k)
                { return (k.position - mc.position).squaredNorm() < 1e-6f; });
            if (coincides) { model_dups_dropped_++; continue; }

            slot_of_original_[i] = static_cast<int>(model_corners_.size());
            model_corners_.push_back(mc);
            yield_.emplace_back();
        }
    }
}

// ---------------------------------------------------------------------------
//  detect — model-guided partitioning + PCA line fit + intersect
// ---------------------------------------------------------------------------
//  solve_hungarian — minimum-cost bipartite assignment (Kuhn-Munkres, O(N³))
//
//  Returns assignment[row] = col, or -1 if that row remains unassigned.
//  Infeasible pairs must be encoded as INFEASIBLE (defined below).
//  Rows with no feasible column end up unassigned (-1).
// ---------------------------------------------------------------------------
static constexpr float INFEASIBLE = 1e9f;

static std::vector<int> solve_hungarian(
        const std::vector<std::vector<float>>& cost, int R, int C)
{
    if (R == 0 || C == 0) return std::vector<int>(R, -1);

    const int N = std::max(R, C);   // pad to square

    auto cell = [&](int i, int j) -> float {
        return (i < R && j < C) ? cost[i][j] : INFEASIBLE;
    };

    // u[i]/v[j] — row/column potentials (1-indexed internally)
    // p[j]      — row currently matched to column j  (0 = free)
    // way[j]    — predecessor column on the shortest-path tree
    std::vector<float> u(N + 1, 0.f), v(N + 1, 0.f);
    std::vector<int>   p(N + 1, 0),   way(N + 1, 0);

    for (int i = 1; i <= N; ++i)
    {
        p[0] = i;
        int j0 = 0;
        std::vector<float> minv(N + 1, INFEASIBLE);
        std::vector<bool>  used(N + 1, false);

        bool augmented = true;
        do {
            used[j0] = true;
            const int i0 = p[j0];
            int   j1    = -1;          // -1 ⇒ no unused column found this step
            float delta = INFEASIBLE;
            for (int j = 1; j <= N; ++j)
            {
                if (not used[j])
                {
                    const float c = cell(i0 - 1, j - 1) - u[i0] - v[j];
                    if (c < minv[j]) { minv[j] = c; way[j] = j0; }
                    if (minv[j] < delta) { delta = minv[j]; j1 = j; }
                }
            }
            // Termination guard: a valid augmenting step must reach a FEASIBLE free column. If none
            // exists (row i0 reaches only INFEASIBLE/padded columns — e.g. corner candidates far from
            // every model corner), the classic Kuhn-Munkres loop leaves delta==INFEASIBLE, never
            // advances j0 to a free column, and spins FOREVER (caught live via gdb: this thread at
            // 100% CPU, the whole agent hung, needing kill -9). Bail here WITHOUT touching the
            // potentials so the matching stays consistent; row i then stays unassigned (-1) — exactly
            // this function's documented contract for a row with no feasible column.
            if (j1 < 0 or delta >= INFEASIBLE * 0.5f) { augmented = false; break; }
            for (int j = 0; j <= N; ++j)
            {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else          { minv[j] -= delta; }
            }
            j0 = j1;
        } while (p[j0] != 0);

        // Augment along the discovered path only when we actually reached a free column. On a bail
        // the loop above never wrote p[], so previously matched rows/columns are left intact.
        if (augmented)
        {
            do {
                const int j1 = way[j0];
                p[j0] = p[j1];
                j0 = j1;
            } while (j0);
        }
    }

    // Extract: p[j] = 1-indexed row assigned to column j
    // Reject padded or infeasible pairs.
    std::vector<int> assignment(R, -1);
    for (int j = 1; j <= C; ++j)
    {
        const int r = p[j] - 1;   // 0-based
        if (r >= 0 && r < R && cost[r][j - 1] < INFEASIBLE * 0.5f)
            assignment[r] = j - 1;
    }
    return assignment;
}

// ---------------------------------------------------------------------------
//  detect — two-phase pipeline:
//    Phase 1: per-model-corner PCA candidate generation
//    Phase 2: Hungarian assignment — prevents two model corners from claiming
//             the same physical detection when search discs overlap
// ---------------------------------------------------------------------------
CornerDetector::DetectionResult CornerDetector::detect(
        const std::vector<Eigen::Vector3f>& lidar_points,
        float robot_x, float robot_y, float robot_theta,
        const Eigen::Matrix3f& pose_cov,
        float max_range)
{
    DetectionResult result;
    result.model_dup_dropped = model_dups_dropped_;
    result.model_short_wall_dropped = static_cast<int>(short_wall_dropped_.size());
    if (model_corners_.empty() || lidar_points.empty())
        return result;

    // Build 2D lidar points in robot frame (drop z)
    std::vector<Eigen::Vector2f> pts2d;
    pts2d.reserve(lidar_points.size());
    for (const auto& p : lidar_points)
        pts2d.emplace_back(p.x(), p.y());

    // Rotation world→robot:  R(-θ)
    const float cos_t = std::cos(robot_theta);
    const float sin_t = std::sin(robot_theta);
    const Eigen::Vector2f t_world(robot_x, robot_y);

    auto to_robot = [&](const Eigen::Vector2f& v) -> Eigen::Vector2f {
        return {cos_t * v.x() + sin_t * v.y(),
               -sin_t * v.x() + cos_t * v.y()};
    };

    // (the gather disc is per-corner now — see corner_r below; search_radius is only its upper bound)
    const float max_range2 = max_range * max_range;

    // ── Phase 1: per-model-corner detection attempt ───────────────────────
    // We record every in-FOV corner's predicted position for the cost matrix,
    // then collect all candidates that survive all quality filters.

    struct FOVCorner {
        const ModelCorner* mc;
        Eigen::Vector2f    predicted;   // robot-frame expected position
        Eigen::Matrix2f    pred_cov;    // H·P_pose·Hᵀ — how much the PREDICTION itself can move
    };
    std::vector<FOVCorner>   fov_corners;
    std::vector<CornerMatch> candidates;

    // Prediction Jacobian for a corner at robot-frame p = R(-θ)·(m_w - t):
    //   ∂p/∂x, ∂p/∂y = −R(-θ) columns;   ∂p/∂θ = (p_y, −p_x).
    // So H·P·Hᵀ is the corner's positional uncertainty induced by the pose uncertainty — the term that
    // makes the association gate breathe with localization quality instead of being a fixed radius.
    // Refuse to build the association algebra on a poisoned pose covariance: S = Σ_det + H P Hᵀ would
    // inherit the NaN, every Mahalanobis cost would be NaN, and the PDA weights with it. Treating an
    // unusable P as ZERO degrades to "gate on detection + map error alone", which is conservative
    // (tighter gate) rather than silently wrong.
    const Eigen::Matrix3f P = pose_cov.allFinite() ? pose_cov : Eigen::Matrix3f::Zero();
    auto prediction_cov = [&](const Eigen::Vector2f& p) -> Eigen::Matrix2f {
        Eigen::Matrix<float, 2, 3> H;
        H(0, 0) = -cos_t; H(0, 1) = -sin_t; H(0, 2) =  p.y();
        H(1, 0) =  sin_t; H(1, 1) = -cos_t; H(1, 2) = -p.x();
        return H * P * H.transpose();
    };

    // Σ of a detection = inverse of its graded information, regularised by the weak "the corner lies
    // inside the search disc" prior so a rank-1 (shallow-corner) Λ_det stays invertible. Shared by the
    // exclusion test and the association gate, so both speak the same units.
    const float merge_prior_sigma = params_.merge_prior_sigma > 0.f ? params_.merge_prior_sigma
                                                                    : params_.search_radius;
    const Eigen::Matrix2f prior_info =
        Eigen::Matrix2f::Identity() / (merge_prior_sigma * merge_prior_sigma);
    auto pos_cov = [&](const Eigen::Matrix2f& Lambda) -> Eigen::Matrix2f {
        return (Lambda + prior_info).inverse();
    };
    // Map error: the layout is a traced hypothesis, so a detection can legitimately sit ~0.1-0.3 m off
    // its predicted corner with nothing wrong. Enters every innovation covariance isotropically.
    const Eigen::Matrix2f map_cov =
        Eigen::Matrix2f::Identity() * (params_.map_sigma * params_.map_sigma);
    // Squared Mahalanobis distance of δ under S, plus |S| for the PDA likelihood normalisation.
    auto mahalanobis2 = [](const Eigen::Vector2f& delta, const Eigen::Matrix2f& S, float* det_out) {
        const Eigen::Matrix2f Sinv = S.inverse();
        if (det_out) *det_out = std::max(1e-12f, S.determinant());
        return delta.dot(Sinv * delta);
    };

    for (const auto& mc : model_corners_)
    {
        const Eigen::Vector2f dw = mc.position - t_world;
        if (dw.squaredNorm() > max_range2)
            continue;

        // Occlusion/visibility gate: skip a corner whose sight line from the robot is blocked by a
        // non-adjacent wall (the notch step occludes its neighbour from most of the room). An occluded
        // corner is UNREACHABLE — it must not be matched (→ spurious residual), must not enter the loss,
        // and must not count toward the early-exit decision. This is the real cause of the lone red-circle
        // corner that pinned early-exit at 0% (see corners_1.png): it was occluded, not mismatched.
        if (!corner_visibility::is_corner_visible(t_world, mc.original_index, polygon_, max_range))
            { result.rej_occluded++; result.occluded_indices.push_back(mc.original_index); continue; }

        const Eigen::Vector2f predicted = to_robot(dw);
        result.corners_in_fov++;
        result.in_fov_indices.push_back(mc.original_index);
        fov_corners.push_back({&mc, predicted, prediction_cov(predicted)});

        const Eigen::Vector2f dir_in  = to_robot(mc.edge_in_dir);
        const Eigen::Vector2f dir_out = to_robot(mc.edge_out_dir);
        const Eigen::Vector2f normal_in (-dir_in.y(),  dir_in.x());
        const Eigen::Vector2f normal_out(-dir_out.y(), dir_out.x());

        // Gather neighbourhood, clipped to each wall's actual length
        // (+ 0.2 m slack for localisation error).
        std::vector<Eigen::Vector2f> group_in, group_out;
        group_in.reserve(128);
        group_out.reserve(128);
        const float in_limit  = mc.wall_in_length  + 0.2f;
        const float out_limit = mc.wall_out_length + 0.2f;
        // ── Neighbourhood scaled to THIS corner's own walls ────────────────────────────────────────
        // A fixed 1.5 m disc + 0.35 m band is fine for a corner between multi-metre walls, but the
        // pillars introduced corners whose walls are 0.40 m and 0.60 m long — and whose parallel
        // neighbour (the room's right wall) sits only 0.385 m away, barely outside a 0.35 m band. Lidar
        // noise then leaks far, near-PARALLEL structure into a wall group, both PCA fits land on
        // almost the same line, and their intersection shoots ~1.2 m down the degenerate direction
        // (observed live: svg_v20 resid 1.17→1.26 m climbing, Σ collapsed onto the prior floor).
        //
        // The corner's own geometry sets its scale: the band may not exceed half the shorter adjacent
        // wall, so a structure further away than that wall is long cannot be mistaken for it. Bounded
        // BELOW by map_sigma — a band tighter than the map is wrong can never gather the true wall —
        // and above by the configured wall_band, so multi-metre room corners are untouched.
        const float min_wall  = std::min(mc.wall_in_length, mc.wall_out_length);
        const float wall_band = std::clamp(0.5f * min_wall, params_.map_sigma, params_.wall_band);
        // Likewise cap the gather disc: no point reaching 1.5 m out for a corner defined by 0.4 m walls.
        const float corner_r  = std::min(params_.search_radius,
                                         std::max(mc.wall_in_length, mc.wall_out_length) + wall_band);
        const float corner_r2 = corner_r * corner_r;

        for (const auto& p : pts2d)
        {
            const Eigen::Vector2f d = p - predicted;
            if (d.squaredNorm() > corner_r2)
                continue;

            const float dist_to_in  = std::abs(normal_in.dot(d));
            const float dist_to_out = std::abs(normal_out.dot(d));

            const float along_in = -dir_in.dot(d);
            const float along_out = dir_out.dot(d);
            const bool in_candidate = (along_in >= -0.2f && along_in <= in_limit &&
                                       dist_to_in <= wall_band);
            const bool out_candidate = (along_out >= -0.2f && along_out <= out_limit &&
                                        dist_to_out <= wall_band);

            if (in_candidate && (!out_candidate || dist_to_in < dist_to_out))
                group_in.push_back(p);
            else if (out_candidate)
                group_out.push_back(p);
        }

        if (static_cast<int>(group_in.size())  < params_.min_points_per_line ||
            static_cast<int>(group_out.size()) < params_.min_points_per_line)
            { result.rej_fewpoints++; continue; }   // FORMATION failure — see counter doc

        auto line_in  = fit_line_pca(group_in,  params_.min_points_per_line);
        auto line_out = fit_line_pca(group_out, params_.min_points_per_line);
        if (!line_in || !line_out)
            continue;

        result.corners_detected++;

        float angle_deg = 0.f;
        auto intersection = intersect(*line_in, *line_out, &angle_deg);
        if (!intersection)
            continue;   // truly parallel (det≈0) → no intersection point exists at all

        // ── Coarse pre-filter only: keep the cost matrix small. The REAL gate is the Mahalanobis
        //    test below, once Λ_det (and therefore Σ_det) exists. ──
        const float isect_dist = (*intersection - predicted).norm();
        if (isect_dist > params_.max_match_distance)
            { result.rej_dist++; continue; }

        // ── Convexity: the rot180 disambiguator. |dir·model_dir| below is blind to a 180° edge flip,
        //    so a rotated-by-π hypothesis passes the orientation test; the SIGNED convexity (reflex vs
        //    convex — e.g. the notch, the pillar roots) is the one feature that breaks the tie.
        //
        //    This used to read `sign_model·cross_det < 0.50·|sign_model|`, which divides out to
        //    `sign_model·cross_det < 0.50` — i.e. it demanded |sin(turn)| ≥ 0.5, an undocumented
        //    "detected interior angle must be ≤ 150°" cutoff bolted onto the sign test. Two separate
        //    claims in one comparison, and the wedge-sharpness half is a hard threshold this codebase
        //    forbids: shallow wedges are ALREADY handled continuously, because near-parallel walls make
        //    Λ_det collapse to rank-1 and the corner contributes almost nothing on its own. Worse, the
        //    layout's two chamfer outer ends sit at |cross| 0.65 / 0.63 — barely 0.13 above the cliff,
        //    so ~10° of line-fit noise silently deleted exactly the corners the rounding introduced.
        //
        //    Keep ONLY the topological claim: the detected turn must not have the opposite sign to the
        //    model's. Magnitude is evidence strength, and that is Λ_det's job, not a gate's.
        const float raw_dot_in  = line_in->direction().dot(dir_in);
        const float raw_dot_out = line_out->direction().dot(dir_out);
        const Eigen::Vector2f ori_in  = (raw_dot_in  >= 0.f ? 1.f : -1.f) * line_in->direction();
        const Eigen::Vector2f ori_out = (raw_dot_out >= 0.f ? 1.f : -1.f) * line_out->direction();
        {
            const float detected_cross = ori_in.x() * ori_out.y() - ori_in.y() * ori_out.x();
            // Normalised agreement in [-1,1]: +1 = same turn direction and sharp, 0 = wedge too shallow
            // to tell, -1 = confidently the opposite convexity (the 180°-flipped hypothesis).
            const float agree = detected_cross * (mc.convexity_sign >= 0.f ? 1.f : -1.f);
            if (agree < 0.f)
            {
                result.rej_convex++;
                result.convex_rej_agree_sum += agree;   // near 0 ⇒ shallow/noisy, near -1 ⇒ real flip
                continue;
            }
        }

        // ── GRADED per-detection information matrix Λ_det (robot frame) ────────────────────────
        // Each fitted wall line L contributes a rank-1 constraint (1/σ_L²)·n_L n_Lᵀ along its normal,
        // scaled by a smooth orientation-trust weight. Summed:
        //     Λ_det = Σ_L (ori_scale_L / σ_L²) n_L n_Lᵀ
        //   • σ_L²      = base_sigma² + resid_var_L   — fit scatter inflates σ (clutter → distrust).
        //   • ori_scale = exp(−(dev_L/τ)²)            — smooth wall-orientation trust (no hard cut).
        // Near-parallel walls (shallow corner) → n_in ≈ ±n_out → Λ_det collapses to rank-1: the
        // bisector direction is left UNCONSTRAINED (aperture ambiguity falls out of the geometry,
        // replacing the old min/max angle gate). Perpendicular walls, clean fit → ~isotropic.
        const float base_var = params_.base_sigma * params_.base_sigma;
        const float tau = std::max(1e-3f, params_.orient_tau_deg * static_cast<float>(M_PI) / 180.f);
        auto line_info = [&](const Line2D& L, float raw_dot) -> Eigen::Matrix2f {
            const float cos_dev = std::min(1.0f, std::abs(raw_dot));
            const float dev = std::acos(cos_dev);                 // 0 = aligned with model edge
            const float ori_scale = std::exp(-(dev * dev) / (tau * tau));
            const float sigma2 = base_var + L.resid_var;
            return (ori_scale / sigma2) * (L.normal * L.normal.transpose());
        };
        const Eigen::Matrix2f Lambda = line_info(*line_in, raw_dot_in)
                                     + line_info(*line_out, raw_dot_out);
        {   // observability: heavily-downweighted (near-orthogonal to model) detections
            const float dev_in  = std::acos(std::min(1.0f, std::abs(raw_dot_in)));
            const float dev_out = std::acos(std::min(1.0f, std::abs(raw_dot_out)));
            const float s = std::exp(-(std::max(dev_in, dev_out) * std::max(dev_in, dev_out)) / (tau * tau));
            if (s < 0.05f) result.soft_orient++;
        }

        // ── Mahalanobis self-consistency gate: is this intersection plausibly THIS model corner,
        //    given both the detection noise and how far the pose itself could be wrong? ──
        {
            const Eigen::Matrix2f S = pos_cov(Lambda) + fov_corners.back().pred_cov + map_cov;
            const float d2 = mahalanobis2(*intersection - predicted, S, nullptr);
            if (params_.assoc_chi2 > 0.f and d2 > params_.assoc_chi2)
                { result.rej_dist++; continue; }
        }

        CornerMatch cand;
        cand.detected    = *intersection;
        cand.predicted   = predicted;          // overwritten by Phase 2 winner
        cand.model_world = mc.position;        // overwritten by Phase 2 winner
        cand.model_index = mc.original_index;  // overwritten by Phase 2 winner
        cand.distance    = isect_dist;
        cand.angle_deg   = angle_deg;
        cand.information = Lambda;
        // Legacy display covariance = pseudo-inverse of Λ_det (falls back to a large isotropic value
        // along any unconstrained/rank-deficient direction). Not used by the loss.
        {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(Lambda);
            Eigen::Vector2f ev = es.eigenvalues();
            Eigen::Matrix2f Vd = es.eigenvectors();
            Eigen::Vector2f inv;
            for (int i = 0; i < 2; ++i) inv(i) = ev(i) > 1e-6f ? 1.f / ev(i) : 1e6f;
            cand.covariance = Vd * inv.asDiagonal() * Vd.transpose();
        }
        candidates.push_back(cand);
    }

    if (candidates.empty())
        return result;

    // ── Phase 1.5: EXCLUSION — two corners cannot occupy the same physical space ──────────────
    // Phase 1 runs INDEPENDENTLY per model corner, so two model corners whose search discs overlap
    // (adjacent vertices of a short wall, or the two lips of the notch) can each fit the SAME pair of
    // walls and emit their own candidate at essentially the same intersection. The Hungarian below
    // only enforces "one model corner per candidate OBJECT" — it happily accepts two distinct objects
    // sitting on top of each other, which is what shows up in the UI as duplicated/near-coincident
    // corners, and what double-counts that corner's precision in the RFE loss.
    //
    // The test is statistical, not metric: candidates a and b are the same physical corner when their
    // separation is not resolvable given their own uncertainties,
    //     d² = δᵀ (Σ_a + Σ_b)⁻¹ δ  <  χ²₂,     δ = x_a − x_b,
    // with Σ = (Λ_det + I/σ_prior²)⁻¹ — the detection information regularised by the weak prior "the
    // corner lies inside the search disc", so a rank-1 (shallow-corner) detection gets a large but
    // FINITE covariance along its unconstrained direction instead of an infinite one. Consequence:
    // two crisp detections 10 cm apart stay separate (a genuine narrow notch survives), two vague or
    // mutually unconstrained ones fuse. No metric cutoff, no discarded evidence — coincident
    // candidates are FUSED in information form (Λ = ΣΛ_i, x = Λ⁻¹ΣΛ_i x_i), which is exactly the
    // posterior of the two observations under the identity hypothesis.
    if (params_.merge_chi2 > 0.f and candidates.size() > 1)
    {
        std::vector<CornerMatch> fused;
        std::vector<Eigen::Matrix2f> fused_cov;   // Σ of the fused estimate, for the next comparisons
        std::vector<Eigen::Vector2f> fused_info_x;// Λ·x accumulator (information vector)
        fused.reserve(candidates.size());

        for (const auto& cand : candidates)
        {
            const Eigen::Matrix2f cov_c = pos_cov(cand.information);

            int host = -1;
            float best_d2 = params_.merge_chi2;
            for (int k = 0; k < static_cast<int>(fused.size()); ++k)
            {
                const Eigen::Vector2f delta = cand.detected - fused[k].detected;
                const Eigen::Matrix2f S = cov_c + fused_cov[k];
                const float d2 = delta.dot(S.inverse() * delta);
                if (d2 < best_d2) { best_d2 = d2; host = k; }
            }

            if (host < 0)   // resolvably distinct from every kept candidate → a new physical corner
            {
                fused.push_back(cand);
                fused_cov.push_back(cov_c);
                fused_info_x.push_back(cand.information * cand.detected);
                continue;
            }

            // Same physical corner → information-form fusion into the host.
            result.merged_coincident++;
            auto& h = fused[host];
            h.information += cand.information;
            fused_info_x[host] += cand.information * cand.detected;
            // Recover the fused mean; fall back to the host's own estimate if the summed information
            // is still singular (both detections rank-1 along the same direction).
            const Eigen::Matrix2f H = h.information + prior_info;
            h.detected  = H.inverse() * (fused_info_x[host] + prior_info * h.detected);
            h.angle_deg = std::max(h.angle_deg, cand.angle_deg);   // keep the better-conditioned wedge
            fused_cov[host] = pos_cov(h.information);
            {   // refresh the legacy display covariance from the fused information
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(h.information);
                const Eigen::Vector2f ev = es.eigenvalues();
                Eigen::Vector2f inv;
                for (int i = 0; i < 2; ++i) inv(i) = ev(i) > 1e-6f ? 1.f / ev(i) : 1e6f;
                h.covariance = es.eigenvectors() * inv.asDiagonal() * es.eigenvectors().transpose();
            }
        }
        candidates.swap(fused);
    }

    // ── Phase 2: Hungarian assignment on MAHALANOBIS cost ────────────────────────────────────────
    // The solver only needs a cost matrix, so the statistically correct distance drops straight in:
    //   cost[r][c] = δᵀ S_rc⁻¹ δ,   S_rc = Σ_det(c) + H_r·P_pose·H_rᵀ,   δ = x_c − p_r
    // gated at χ²₂ (assoc_chi2). Beyond the gate the pair is INFEASIBLE, so the solver's existing
    // "row with no feasible column stays unassigned" contract does the rejecting. Unlike the old
    // metric cap this shrinks as the pose sharpens, which is exactly what stops a 0.3 m-spaced pillar
    // corner from being claimed by its neighbour once the robot knows where it is.
    const int R = static_cast<int>(fov_corners.size());
    const int C = static_cast<int>(candidates.size());

    std::vector<std::vector<float>> cost(R, std::vector<float>(C, INFEASIBLE));
    std::vector<std::vector<float>> lik (R, std::vector<float>(C, 0.f));   // PDA likelihoods
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c)
        {
            const Eigen::Matrix2f S = pos_cov(candidates[c].information) + fov_corners[r].pred_cov + map_cov;
            float detS = 1.f;
            const float d2 = mahalanobis2(candidates[c].detected - fov_corners[r].predicted, S, &detS);
            if (params_.assoc_chi2 <= 0.f or d2 <= params_.assoc_chi2)
            {
                cost[r][c] = d2;
                // Gaussian association likelihood (shared 2π factor cancels in the normalisation).
                lik[r][c] = std::exp(-0.5f * d2) / std::sqrt(detS);
            }
        }

    const std::vector<int> assignment = solve_hungarian(cost, R, C);

    float assoc_prob_sum = 0.f;
    for (int r = 0; r < R; ++r)
    {
        const int c = assignment[r];
        if (c < 0) continue;

        // ── Association posterior (PDA): given detection c, how probable is it that model corner r
        //    produced it rather than any OTHER in-gate model corner? Normalising the likelihoods down
        //    column c answers exactly that. Two corners that explain the detection equally well each
        //    get ~0.5, three get ~0.33 — the corner stops voting instead of voting wrongly. This is
        //    the ambiguity fix; the Hungarian above only picks the best guess, it cannot know the
        //    guess was a coin flip.
        float denom = 0.f;
        float runnerup = INFEASIBLE;
        for (int rr = 0; rr < R; ++rr)
        {
            denom += lik[rr][c];
            if (rr != r) runnerup = std::min(runnerup, cost[rr][c]);   // best RIVAL for this detection
        }
        // ── NO EVIDENCE ⇒ NO FACTOR ────────────────────────────────────────────────────────────
        // The posterior can legitimately reach 0 (every in-gate likelihood underflowed) and NaN if any
        // covariance upstream was already poisoned. Emitting the match anyway with Λ *= 0 hands the
        // optimizer a ZERO-precision observation: it constrains nothing, yet it still enters the
        // Hessian assembly and can drive min_ev to 0 ⇒ cond_num sentinel 1e8 ⇒ NaN losses ⇒ NaN pose
        // (root cause of the 2026-07-21 divergence). A corner we cannot associate is not weak
        // evidence, it is ABSENT evidence — so it must not become a factor at all.
        // NOTE std::clamp does NOT sanitise NaN (NaN<lo and hi<NaN are both false ⇒ NaN passes
        // through), so the finiteness test has to come first and explicitly.
        const float w_raw = denom > 1e-20f ? lik[r][c] / denom : 0.f;
        if (not std::isfinite(w_raw) or w_raw <= 0.f)
            { result.rej_noninformative++; continue; }
        const float w = std::clamp(w_raw, params_.assoc_min_weight, 1.f);

        CornerMatch m    = candidates[c];
        m.model_index    = fov_corners[r].mc->original_index;
        m.predicted      = fov_corners[r].predicted;
        m.model_world    = fov_corners[r].mc->position;
        m.distance       = (m.detected - m.predicted).norm();
        m.assoc_prob     = w;
        m.assoc_chi2_val = cost[r][c];
        m.runnerup_chi2  = runnerup;
        // Precision IS the confidence in the correspondence: an ambiguous match carries proportionally
        // less information into the loss. The loss needs no change — it already consumes `information`.
        m.information   *= w;
        // Final contract check before this leaves the detector: the emitted precision must be finite
        // and must actually constrain at least one direction. Rank-1 is legitimate (a shallow corner
        // leaves its bisector free); rank-0 or non-finite is not, and must never reach the optimizer.
        {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> chk(m.information);
            if (not m.information.allFinite() or chk.eigenvalues().maxCoeff() <= 1e-9f)
                { result.rej_noninformative++; continue; }
        }
        {   // keep the display covariance consistent with the down-weighted information
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(m.information);
            const Eigen::Vector2f ev = es.eigenvalues();
            Eigen::Vector2f inv;
            for (int i = 0; i < 2; ++i) inv(i) = ev(i) > 1e-6f ? 1.f / ev(i) : 1e6f;
            m.covariance = es.eigenvectors() * inv.asDiagonal() * es.eigenvectors().transpose();
        }
        assoc_prob_sum += w;
        result.matches.push_back(m);
    }
    result.corners_accepted = static_cast<int>(result.matches.size());
    result.mean_assoc_prob  = result.corners_accepted > 0
                            ? assoc_prob_sum / static_cast<float>(result.corners_accepted) : 1.f;

    // Residual + ambiguity distribution over the accepted set. resid_* measures the real model misfit
    // (so map_sigma stops being a guess); runnerup/min_assoc identify WHICH corners are coin flips.
    if (result.corners_accepted > 0)
    {
        float rs = 0.f, rmax = 0.f, chi = 0.f, ru = 0.f, pmin = 1.f;
        int   nru = 0;
        for (const auto& m : result.matches)
        {
            rs   += m.distance;
            rmax  = std::max(rmax, m.distance);
            chi  += m.assoc_chi2_val;
            pmin  = std::min(pmin, m.assoc_prob);
            // Only corners that actually HAVE a rival in gate contribute to the rival statistic.
            if (m.runnerup_chi2 < INFEASIBLE * 0.5f) { ru += m.runnerup_chi2; ++nru; }
        }
        const float inv_n = 1.f / static_cast<float>(result.corners_accepted);
        result.resid_mean         = rs * inv_n;
        result.resid_max          = rmax;
        result.resid_chi2_mean    = chi * inv_n;
        result.corners_with_rival = nru;
        result.runnerup_chi2_mean = nru > 0 ? ru / static_cast<float>(nru) : 0.f;
        result.min_assoc_prob     = pmin;
    }
    // ── Landmark retirement by observed information (Params::min_yield_map_sigmas) ────────────────
    // Fold THIS frame's evidence in first, then judge, so a corner that just became observable is
    // released the same frame it recovers.
    //
    // Scored ONLY on frames where the corner actually produced a match — silence is deliberately NOT
    // counted as zero information. A corner that fails to form a detection contributes nothing to the
    // loss, so it does no harm; the failure mode being targeted is a corner that DOES fire and says
    // something uninformative. The question is therefore "when this landmark speaks, is what it says
    // worth hearing", not "how often does it speak". Verified against 23400 live frames: scoring
    // silence would also retire v1 (λ≈67 when it fires, well above the bar, but only on 1.9% of
    // frames) and v29 (λ≈121, two samples in the whole run) — both merely quiet, neither harmful.
    {
        const float leak = std::clamp(params_.yield_leak, 0.f, 1.f);
        for (const auto& m : result.matches)
        {
            if (m.model_index < 0 || m.model_index >= static_cast<int>(slot_of_original_.size()))
                continue;
            const int slot = slot_of_original_[m.model_index];
            if (slot < 0) continue;
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(m.information);
            if (es.info() != Eigen::Success) continue;
            const float lam = std::max(0.f, es.eigenvalues()(0));
            auto& y = yield_[slot];
            // SEED on the first sample instead of leaking up from zero. Leaking from zero biases the
            // estimate low by ~1/leak for the first samples (live: a corner whose single match had
            // λ=91 read as yield=1.82), which would retire a good corner the moment the warmup ends.
            y.lambda_min = (y.samples == 0) ? lam : y.lambda_min + leak * (lam - y.lambda_min);
            ++y.samples;
        }

        if (params_.min_yield_map_sigmas > 0.f)
        {
            const float sigma_bar = params_.min_yield_map_sigmas * params_.map_sigma;
            const float lambda_bar = 1.f / std::max(1e-9f, sigma_bar * sigma_bar);
            // Release needs to CLEAR the bar, not merely touch it (Params::yield_release_factor).
            const float release_bar = std::max(1.f, params_.yield_release_factor) * lambda_bar;
            for (auto& m : result.matches)
            {
                const int slot = (m.model_index >= 0 && m.model_index < static_cast<int>(slot_of_original_.size()))
                               ? slot_of_original_[m.model_index] : -1;
                if (slot < 0) continue;
                auto& y = yield_[slot];
                m.yield = y.lambda_min;

                if (y.retired)
                {
                    // Still measured while retired — that is the whole point of keeping it detected —
                    // so it releases itself once the view genuinely improves.
                    if (y.lambda_min > release_bar)
                        y.retired = false;
                }
                // Warmup guard: never retire on a handful of samples. A corner that has spoken only
                // twice has told us nothing about itself either way, so it keeps its vote.
                else if (y.samples >= params_.yield_warmup && y.lambda_min < lambda_bar)
                    y.retired = true;

                if (y.retired)
                {
                    m.suppressed = true;
                    result.corners_suppressed++;
                }
            }
        }
    }

    // Candidates that passed every quality gate but lost the 1-to-1 assignment.
    result.rej_unassigned = C - result.corners_accepted;

    return result;
}

// ---------------------------------------------------------------------------
//  fit_line_pca — least-squares line fit via PCA
// ---------------------------------------------------------------------------
std::optional<CornerDetector::Line2D> CornerDetector::fit_line_pca(
        const std::vector<Eigen::Vector2f>& pts, int min_points)
{
    if (static_cast<int>(pts.size()) < min_points)
        return std::nullopt;

    Eigen::Vector2f centroid = Eigen::Vector2f::Zero();
    for (const auto& p : pts)
        centroid += p;
    centroid /= static_cast<float>(pts.size());

    Eigen::Matrix2f scatter = Eigen::Matrix2f::Zero();
    for (const auto& p : pts)
    {
        Eigen::Vector2f dp = p - centroid;
        scatter += dp * dp.transpose();
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eig(scatter);
    Line2D line;
    line.normal = eig.eigenvectors().col(0);  // smallest eigenvalue = line normal
    line.d = line.normal.dot(centroid);
    // Smallest eigenvalue = Σ perpendicular² of the points about the fitted line.
    // Per-point mean square scatter → a graded quality signal for σ_L (clean wall ≈ sensor noise).
    line.npts = static_cast<int>(pts.size());
    line.resid_var = std::max(0.f, eig.eigenvalues()(0)) / static_cast<float>(line.npts);
    return line;
}

// ---------------------------------------------------------------------------
//  intersect
// ---------------------------------------------------------------------------
std::optional<Eigen::Vector2f> CornerDetector::intersect(
        const Line2D& a, const Line2D& b, float* angle_deg)
{
    const float det = a.normal.x() * b.normal.y() - a.normal.y() * b.normal.x();
    if (std::abs(det) < 1e-6f)
        return std::nullopt;

    const float x = (b.normal.y() * a.d - a.normal.y() * b.d) / det;
    const float y = (a.normal.x() * b.d - b.normal.x() * a.d) / det;

    if (angle_deg)
    {
        // Angle between the two wall directions (= 180° - angle between normals)
        const float ndot = std::abs(a.normal.dot(b.normal));
        const float clamped = std::min(1.0f, ndot);
        *angle_deg = 180.0f - std::acos(clamped) * 180.0f / static_cast<float>(M_PI);
    }

    return Eigen::Vector2f(x, y);
}

} // namespace rc
