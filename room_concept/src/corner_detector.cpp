#include "corner_detector.h"
#include "corner_visibility.h"
#include "assignment.h"
#include <cmath>
#include <cstdlib>
#include <limits>
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
using rc::assign::solve_hungarian;
using rc::assign::INFEASIBLE;

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

    // ── Model edges in the robot frame: let every wall compete for a return ──────────────────────
    // A gather is a band around ONE model edge, and a band wide enough to reach the real wall through
    // the layout's own error is also wide enough to swallow a face PERPENDICULAR to it — the notch
    // step, the pillar sides. A PCA over those returns a direction ACROSS the wall instead of along
    // it. Measured over a 4389-frame tour (55803 evaluations, tmp/corner_probe.csv): model corner 13
    // fitted a line orthogonal to its own model edge on 98% of its evaluations, corner 11 on 98%,
    // corner 27 on 76%. Both groups then land on the same physical wall, the fitted pair comes out
    // parallel (|n_in × n_out| = 0), the propagation refuses, and the rank-1 fallback returns
    // pos_cov of a rank-deficient Λ — the merge prior, 1.06 m. That is not a covariance, it is "no
    // information", handed to the loss as though it were a measurement. 15.6% of all evaluations.
    //
    // The layout already knows those faces are there, so the fix needs no new test: a return is
    // evidence about SOME wall, and the walls compete for it.
    //     w_e(p) = q_e / (q_e + Σ_{f≠e} q_f·(1 − |t_e·t_f|)),   q_f = exp(−dist(p,f)²/2σ²), σ = base_sigma
    // A competitor steals only to the extent its ORIENTATION differs. That factor is the whole idea:
    // what can corrupt a line fit is a return off a surface pointing a different way, and a collinear
    // edge — a wall the trace subdivided, or its continuation past a vertex that lost landmark status
    // — is the SAME surface and carries the same direction evidence, so it must not take anything.
    // Measured with a plain softmax over edges (which does let collinear edges split the point): the
    // fits kept a median 64% of their returns and corner 1 starved on 5.6%, collapsing on 91% of its
    // evaluations — the cure's own failure mode, and worse than the disease it fixed.
    // base_sigma is exactly the right scale here — it is how far the layout can be from the surface,
    // so it is also the distance below which the layout cannot tell two surfaces apart. A return on
    // the step is explained by the step and contributes ~0 to its neighbour; one on the wall itself
    // has no competitor and contributes ~1; one in the wedge where two walls genuinely meet is shared
    // ~½ / ~½, which is what it is worth. No angle test and no cut-off: the fit simply stops being
    // told about surfaces that belong to somebody else, and where that leaves it with little to go on,
    // Σw falls and info_phi_d reports the large σ_φ that is then the truth.
    // A duplicated polygon vertex (SVG authoring artefact — set_model_corners counts them in
    // model_dup_dropped) makes a ZERO-LENGTH edge that sits exactly ON the wall it degenerated from.
    // Left in the competition it would be a perfect explanation for every point near it and would
    // steal half their responsibility from the real wall. Mark them non-competing instead of
    // renumbering, so edge i still means vertex i → i+1 and the corner's own indices stay valid.
    std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> edges_r;
    std::vector<char> edge_ok;
    const std::size_t nv = polygon_.size();
    if (nv >= 3)
    {
        edges_r.reserve(nv); edge_ok.reserve(nv);
        for (std::size_t i = 0; i < nv; ++i)
        {
            const Eigen::Vector2f a = to_robot(polygon_[i] - t_world);
            const Eigen::Vector2f b = to_robot(polygon_[(i + 1) % nv] - t_world);
            edges_r.emplace_back(a, b);
            edge_ok.push_back((b - a).squaredNorm() > 1e-8f ? 1 : 0);
        }
    }
    const float assign_var = 2.f * std::max(1e-6f, params_.base_sigma * params_.base_sigma);
    const auto seg_dist2 = [](const Eigen::Vector2f& p, const Eigen::Vector2f& a, const Eigen::Vector2f& b)
    {
        const Eigen::Vector2f ab = b - a;
        const float L2 = ab.squaredNorm();
        const float u = L2 > 1e-9f ? std::clamp(ab.dot(p - a) / L2, 0.f, 1.f) : 0.f;
        return (p - (a + u * ab)).squaredNorm();
    };
    std::vector<Eigen::Vector2f> edge_t;               // unit direction of each edge, robot frame
    edge_t.reserve(edges_r.size());
    for (const auto& e : edges_r)
    {
        const Eigen::Vector2f ab = e.second - e.first;
        edge_t.push_back(ab.squaredNorm() > 1e-9f ? ab.normalized() : Eigen::Vector2f(1.f, 0.f));
    }
    std::vector<float> edge_q(edges_r.size(), 0.f);    // scratch, reused per point

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
        std::vector<float> w_in_v, w_out_v;   // per-point responsibility for this corner's two edges
        group_in.reserve(128);
        group_out.reserve(128);
        w_in_v.reserve(128);
        w_out_v.reserve(128);
        // This corner's own two edges in the polygon's edge numbering (edge i runs vertex i → i+1),
        // so the competition below can tell which share belongs to the incoming and outgoing wall.
        const std::size_t vi     = static_cast<std::size_t>(std::max(0, mc.original_index)) % std::max<std::size_t>(1, nv);
        const std::size_t ei_in  = (vi + nv - 1) % std::max<std::size_t>(1, nv);
        const std::size_t ei_out = vi;
        // ⚠ NEGATIVE RESULT — OFF BY DEFAULT, AND DO NOT RE-TRY IT WITHOUT READING THIS.
        // The explain-away removes returns that ANOTHER MODEL EDGE explains better. It was built for
        // a diagnosis — "the gather band swallows a surface at right angles" — that the live data does
        // not support. Three arms, same room, same statistic (share of evaluations whose detection
        // covariance collapsed to the merge prior):
        //     committed gather            15.6%
        //     softmax over edges          21.9%
        //     + direction disagreement    24.0%
        // No arm ever beat doing nothing. If the contaminant were a modelled face, competition between
        // model edges would remove it; it does not, which is evidence against the mechanism. The
        // contaminant is not in the model — clutter, or a traced vertex with no physical wall.
        //
        // And the corners it was aimed at are ALREADY HANDLED. etc/corner_stats.csv, same run:
        // vertex 28 retired outright (yield 2e-4 against a bar of 11.1), 11 and 13 suppressed on 40%
        // and 24% of their frames. `160d36d` established on 23400 frames that NO GEOMETRIC RULE
        // separates a good pillar corner from a bad one — wall length has counterexamples in BOTH
        // directions (v27/v28 and v12/v13 have identical walls and yields differing by 10^7), because
        // which corners alias is a property of the TRAJECTORY'S visibility, not of the layout. The
        // rule that works is retirement on observed information yield, and it is already running.
        // A bad corner producing a merge-prior covariance is that rule's INPUT, not a bug.
        //
        // What survives: the weighted fit itself. fit_line_pca and info_phi_d take responsibilities,
        // so Σw replaces N, and with the flag unset every weight is 1 and each reduces exactly to the
        // arithmetic it replaced. tools/corner_gather_test.cpp keeps both arms runnable — note it
        // shows ZERO collapses on the committed path in all four synthetic rooms while the live room
        // collapses on 15.6%, i.e. the harness does not contain the live defect and cannot be used to
        // justify switching this on. That mistake cost two tours.
        static const bool explain_on = (std::getenv("RC_CORNER_EXPLAIN_AWAY") != nullptr);
        const bool can_explain   = explain_on and (edges_r.size() == nv and nv >= 3);
        // 1 − |t_claim·t_f| per edge: 0 for a collinear surface (takes nothing), 1 for a perpendicular
        // face (takes everything it can explain). Fixed for this corner, so it is computed once here
        // rather than per point.
        std::vector<float> dis_in(edges_r.size(), 1.f), dis_out(edges_r.size(), 1.f);
        std::vector<float> steal_e_in(edges_r.size(), 0.f), steal_e_out(edges_r.size(), 0.f);
        int probe_rival_in = -1, probe_rival_out = -1;
        float probe_rival_share_in = 0.f, probe_rival_share_out = 0.f;
        if (can_explain)
            for (std::size_t e = 0; e < edges_r.size(); ++e)
            {
                dis_in[e]  = 1.f - std::min(1.f, std::abs(edge_t[e].dot(edge_t[ei_in])));
                dis_out[e] = 1.f - std::min(1.f, std::abs(edge_t[e].dot(edge_t[ei_out])));
            }
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

            if (not in_candidate and not out_candidate)
                continue;
            bool use_in = in_candidate, use_out = out_candidate;

            // Responsibility of this corner's two edges for the return, against every OTHER wall in
            // the layout. Without a competitor this reduces to 1 for the only edge that claims the
            // point; the old rule (nearest band wins, exclusively) is its argmax, and it is exactly
            // the argmax that put a perpendicular face into a wall's fit at full weight.
            float w_in = in_candidate ? 1.f : 0.f, w_out = out_candidate ? 1.f : 0.f;
            if (can_explain)
            {
                float dmin = std::numeric_limits<float>::max();
                for (std::size_t e = 0; e < edges_r.size(); ++e)
                {
                    edge_q[e] = edge_ok[e] ? seg_dist2(p, edges_r[e].first, edges_r[e].second) : 1e30f;
                    dmin = std::min(dmin, edge_q[e]);
                }
                for (std::size_t e = 0; e < edges_r.size(); ++e)
                    edge_q[e] = std::exp(-(edge_q[e] - dmin) / assign_var);   // max-subtracted
                // THIRD-PARTY SURFACES ONLY. The corner's own two edges are excluded from each other's
                // competition, and that exclusion is the whole correction: a corner's two walls are not
                // rival explanations, they are the two things being fitted, and letting them bid for
                // each other's returns starved the corners this was built to rescue. Measured live over
                // 826 frames with them bidding: in EVERY failing corner the top thief was the corner's
                // own partner edge — #1 slot 0 lost 73% of its theft to edge 1 and kept 0.23 of its
                // returns, #6 slot 1 lost 89% to edge 5 and kept 0.18, #12 and #13 lost 100% to their
                // own partners — and the degenerate share rose 15.6% → 24.0%, worse than no fix at all.
                float steal_in = 0.f, steal_out = 0.f;
                for (std::size_t e = 0; e < edges_r.size(); ++e)
                {
                    if (e == ei_in or e == ei_out) continue;
                    const float ti = edge_q[e] * dis_in[e];  steal_in  += ti; steal_e_in[e]  += ti;
                    const float to = edge_q[e] * dis_out[e]; steal_out += to; steal_e_out[e] += to;
                }
                // Between the corner's OWN pair, keep the committed rule: nearest band takes the point,
                // whole. A return in the wedge is ambiguous between two walls that meet there, and
                // splitting it would also hand the same return to both line fits — which the corner
                // covariance, a sum of two INDEPENDENT line contributions, is not entitled to assume.
                const bool in_wins = in_candidate and (not out_candidate or dist_to_in < dist_to_out);
                const float qi = edge_q[ei_in], qo = edge_q[ei_out];
                use_in  = in_wins;
                use_out = (not in_wins) and out_candidate;
                w_in  = (use_in  and qi + steal_in  > 1e-12f) ? qi / (qi + steal_in)  : (use_in  ? 1.f : 0.f);
                w_out = (use_out and qo + steal_out > 1e-12f) ? qo / (qo + steal_out) : (use_out ? 1.f : 0.f);
            }
            else
            {   // Committed behaviour: nearest band wins, exclusively, at full weight.
                if (in_candidate and (not out_candidate or dist_to_in < dist_to_out))
                    { use_in = true;  use_out = false; w_in = 1.f; w_out = 0.f; }
                else
                    { use_in = false; use_out = true;  w_in = 0.f; w_out = 1.f; }
            }
            // With the explain-away on, a point may belong PARTLY to both adjacent walls — near the
            // vertex it genuinely does — so it is offered to both groups with its share rather than
            // awarded whole to one of them by a coin flip.
            if (use_in)  { group_in.push_back(p);  w_in_v.push_back(w_in); }
            if (use_out) { group_out.push_back(p); w_out_v.push_back(w_out); }
            // KNOWN CONSERVATISM, logged rather than hidden: two COLLINEAR edges (a wall the trace
            // subdivided, or a continuation past a vertex that lost landmark status) are one physical
            // surface, and the competition splits a point's responsibility between them even though
            // both carry the same direction evidence. That costs Σw — σ_φ grows by ~√2 on a wall
            // split in two — but it is symmetric, so it inflates the covariance without biasing the
            // fit. nraw vs npts in the probe measures exactly how much is being given up.
        }

        // Name the rival. When a fit starves, the question is always WHICH surface took its returns —
        // a collinear neighbour means the disagreement factor is not doing its job, a perpendicular
        // face means it is. Recording it costs an argmax and saves a tour of guessing.
        {
            const auto worst = [](const std::vector<float>& st, int& idx, float& share) {
                float tot = 0.f, best = -1.f; idx = -1;
                for (std::size_t e = 0; e < st.size(); ++e)
                    { tot += st[e]; if (st[e] > best) { best = st[e]; idx = static_cast<int>(e); } }
                share = tot > 1e-12f ? best / tot : 0.f;
            };
            worst(steal_e_in,  probe_rival_in,  probe_rival_share_in);
            worst(steal_e_out, probe_rival_out, probe_rival_share_out);
        }

        // Counted in RESPONSIBILITY, not in returns: a hundred points that all belong to the step
        // next door are not a wall, and the formation counter should say so rather than waving them
        // through to a fit that cannot use them.
        const float min_eff = static_cast<float>(params_.min_points_per_line);
        const auto wsum = [](const std::vector<float>& w) { float s = 0.f; for (const float x : w) s += x; return s; };
        if (wsum(w_in_v) < min_eff or wsum(w_out_v) < min_eff)
            { result.rej_fewpoints++; continue; }   // FORMATION failure — see counter doc

        auto line_in  = linefit::fit_line_pca(group_in,  w_in_v,  min_eff);
        auto line_out = linefit::fit_line_pca(group_out, w_out_v, min_eff);
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
        const auto ori_of = [&](float raw_dot) {
            const float dev = std::acos(std::min(1.0f, std::abs(raw_dot)));   // 0 = aligned with the model edge
            return std::exp(-(dev * dev) / (tau * tau));
        };
        // ── PROPAGATED, NOT ASSUMED: points → line → corner ───────────────────────────────────────
        // The rank-1 form below used σ_L² = base_sigma² + resid_var, so a line through 400 returns and
        // one through 12 were trusted equally as long as their scatter matched — and the fitted line's
        // uncertainty falls as 1/N. The segmenter has always known better: linefit::info_phi_d gives
        // Λ(φ,d) from the points themselves, and its φ row is dominated by how far they spread ALONG
        // the wall, which is exactly what an intersection needs to know. The association gate already
        // used that covariance; only the LOSS was still reading a constant, so the same segment was a
        // full covariance when deciding WHETHER it matched and a 4 cm floor when deciding how hard it
        // pulled the pose.
        // Corner covariance is then the textbook propagation through the intersection (the same
        // construction as WallMap::intersect_walls): Σ = Jₐ Cₐ Jₐᵀ + J_b C_b J_bᵀ. The shallow-corner
        // degeneracy the rank-1 form was built to produce now falls out of the geometry instead — Jₐ
        // carries 1/sin(θ) between the lines, so a near-parallel pair yields a huge covariance along
        // the bisector and almost no precision there, with no angle term of its own.
        // The base_sigma floor stays, added to the corner covariance rather than to each line: it is
        // MODEL error between views, and without it a wall with thousands of points would claim a
        // sub-millimetre corner and refuse its own re-observations (the map_sigma lesson).
        // Raw per-candidate record (see CornerProbe): every input the propagation reads, kept
        // unaggregated so the calibration question can be answered off a file.
        CornerProbe probe;
        probe.model_index = mc.original_index;
        probe.angle_deg   = angle_deg;
        probe.rival[0] = probe_rival_in;  probe.rival_share[0] = probe_rival_share_in;
        probe.rival[1] = probe_rival_out; probe.rival_share[1] = probe_rival_share_out;
        const auto line_cov = [&](const Line2D& L, const std::vector<Eigen::Vector2f>& pts,
                                  const std::vector<float>& w,
                                  float raw_dot, int slot) -> std::optional<Eigen::Matrix2f> {
            const float ori = ori_of(raw_dot);
            probe.ori[slot]       = ori;
            probe.npts[slot]      = L.npts;                          // EFFECTIVE count Σw
            probe.nraw[slot]      = static_cast<int>(pts.size());    // returns the band offered it
            probe.resid_sig[slot] = std::sqrt(std::max(0.f, L.resid_var));
            probe.lever[slot]     = std::abs(L.direction().dot(*intersection));
            {   // Tangent coordinates s = t·p as info_phi_d sees them — FROM THE ROBOT ORIGIN.
                // C(φ,φ) = σ²/(N·Var(s)) and C(d,d) = (σ²/N)(1 + mean(s)²/Var(s)), so the spread
                // of s is the entire scale of the line covariance and the offset of s is what
                // makes the two marginals hugely correlated. Neither is recoverable from the
                // covariance after the fact; both are one pass over the points.
                const Eigen::Vector2f t = L.direction();
                double m = 0.0, m2 = 0.0, wt = 0.0; float lo = 1e30f, hi = -1e30f;
                for (std::size_t i = 0; i < pts.size(); ++i)
                {
                    const float sv = t.dot(pts[i]);
                    const double wi = (i < w.size() ? w[i] : 1.f);
                    m += wi * sv; m2 += wi * static_cast<double>(sv) * sv; wt += wi;
                    if (wi >= 0.5) { lo = std::min(lo, sv); hi = std::max(hi, sv); }   // extent this wall OWNS
                }
                const double n = std::max(1e-6, wt);
                probe.s_mean[slot] = static_cast<float>(m / n);
                probe.s_std[slot]  = static_cast<float>(std::sqrt(std::max(0.0, m2 / n - (m / n) * (m / n))));
                probe.s_span[slot] = (hi > lo ? hi - lo : 0.f);
            }
            result.ori_sum += static_cast<double>(ori); result.ori_n++;
            result.ori_min = std::min(result.ori_min, static_cast<double>(ori));
            if (not (ori > 1e-4f)) return std::nullopt;           // orthogonal to the model edge: no trust
            // ⚠ PER-POINT VARIANCE IS THE OBSERVED SCATTER, AND NOTHING ELSE. This read
            // base_var + L.resid_var, which double-counted base_sigma: it inflated the point-level
            // fit AND was added again to the corner covariance below. base_sigma is between-view
            // MODEL error, not sensor noise on a point, so it belongs only at the corner. The
            // inflation was not small — on a clean wall resid_var ≈ (2 cm)² against base_var =
            // (4 cm)², so the per-point variance was ~5x too large and the line covariance with it.
            // Measured over a full tour before the fix: NIS/dof 0.48 ± 0.01 over 28884 evaluations
            // (a covariance 2.1x too large), with this term carrying 87% of it at σ = 0.189 m
            // against map 0.060 and pose 0.041. The segmenter has always passed resid_var alone.
            const Eigen::Matrix2f Lam = linefit::info_phi_d(pts, w, L, std::max(L.resid_var, 1e-6f)) * ori;
            if (not Lam.allFinite() or Lam.determinant() < 1e-12f) return std::nullopt;
            const Eigen::Matrix2f C = Lam.inverse();
            probe.c00[slot] = C(0, 0); probe.c01[slot] = C(0, 1); probe.c11[slot] = C(1, 1);
            if (C.allFinite())
            {
                result.sphi_sum      += std::sqrt(std::max(0.f, C(0, 0))) * 180.0 / M_PI;
                result.sd_sum        += std::sqrt(std::max(0.f, C(1, 1)));
                result.npts_sum      += static_cast<double>(L.npts);
                result.resid_sig_sum += std::sqrt(std::max(0.f, L.resid_var));
                result.lever_sum     += std::abs(static_cast<double>(L.direction().dot(*intersection)));
                result.line_n++;
            }
            return C.allFinite() ? std::optional<Eigen::Matrix2f>(C) : std::nullopt;
        };
        Eigen::Matrix2f Lambda = Eigen::Matrix2f::Zero();
        {
            result.angle_sum += static_cast<double>(angle_deg); result.angle_n++;
            const auto Cin  = line_cov(*line_in,  group_in,  w_in_v,  raw_dot_in,  0);
            const auto Cout = line_cov(*line_out, group_out, w_out_v, raw_dot_out, 1);
            bool propagated = false;
            if (Cin and Cout)
            {
                Eigen::Matrix2f M;
                M.row(0) = line_in->normal.transpose();
                M.row(1) = line_out->normal.transpose();
                probe.sin_theta = std::abs(M.determinant());   // |n_in × n_out| = sin of the wall angle
                if (std::abs(M.determinant()) > 1e-6f)
                {
                    const Eigen::Matrix2f Mi = M.inverse();
                    const auto Jof = [&](const Line2D& L, int row) {
                        Eigen::Matrix2f J;
                        const Eigen::Vector2f col = Mi.col(row);
                        J.col(0) = -col * L.direction().dot(*intersection);   // ∂p/∂φ
                        J.col(1) =  col;                                      // ∂p/∂d
                        return J;
                    };
                    const Eigen::Matrix2f Ja = Jof(*line_in, 0), Jb = Jof(*line_out, 1);
                    Eigen::Matrix2f S = Ja * (*Cin) * Ja.transpose() + Jb * (*Cout) * Jb.transpose();
                    S += base_var * Eigen::Matrix2f::Identity();              // between-view model error
                    if (S.allFinite() and S.determinant() > 1e-12f)
                    {
                        const Eigen::Matrix2f L2 = S.inverse();
                        if (L2.allFinite()) { Lambda = L2; propagated = true; probe.propagated = 1; }
                    }
                }
            }
            if (not propagated)
            {
                // Fallback: the previous rank-1 sum. Reached when a line's fit is degenerate (all its
                // points at one tangent coordinate), when the pair is parallel to numerical precision,
                // or when a line is orthogonal to its model edge — cases where the propagation has
                // nothing to say and silence would be worse than a coarse estimate.
                const auto rank1 = [&](const Line2D& L, float raw_dot) {
                    return (ori_of(raw_dot) / (base_var + L.resid_var)) * (L.normal * L.normal.transpose());
                };
                Lambda = rank1(*line_in, raw_dot_in) + rank1(*line_out, raw_dot_out);
            }
        }
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
            {   // The raw record, emitted whether or not the gate keeps this candidate — a covariance
                // audited only on what its own gate admitted is not audited at all.
                const Eigen::Matrix2f Sd = pos_cov(Lambda);
                const Eigen::Matrix2f Sp = fov_corners.back().pred_cov;
                const Eigen::Vector2f nu = *intersection - predicted;
                probe.det_x = intersection->x();  probe.det_y = intersection->y();
                probe.pred_x = predicted.x();     probe.pred_y = predicted.y();
                probe.nu_x = nu.x();              probe.nu_y = nu.y();
                probe.d2 = d2;
                probe.over_gate = (params_.assoc_chi2 > 0.f and d2 > params_.assoc_chi2) ? 1 : 0;
                probe.sdet_xx = Sd(0,0); probe.sdet_xy = Sd(0,1); probe.sdet_yy = Sd(1,1);
                probe.sprd_xx = Sp(0,0); probe.sprd_xy = Sp(0,1); probe.sprd_yy = Sp(1,1);
                probe.smap_xx = map_cov(0,0); probe.smap_yy = map_cov(1,1);
                result.probes.push_back(probe);
            }
            // NIS, recorded BEFORE the gate can truncate it (see DetectionResult::nis_pre_*), and
            // the three terms of S separately, because a NIS below 1 says the SUM is too big and
            // cannot say whose fault that is.
            if (std::isfinite(d2))
            {
                result.nis_pre_sum += static_cast<double>(d2) * 0.5;   // /dof, dof = 2
                result.nis_pre_n++;
                if (params_.assoc_chi2 > 0.f and d2 > params_.assoc_chi2) result.nis_pre_over++;
                const Eigen::Matrix2f Sdet = pos_cov(Lambda);
                const auto rms = [](const Eigen::Matrix2f& M) { return std::sqrt(std::max(0.f, M.trace() * 0.5f)); };
                if (Sdet.allFinite())
                {
                    result.s_det_sum  += rms(Sdet);
                    result.s_pred_sum += rms(fov_corners.back().pred_cov);
                    result.s_map_sum  += rms(map_cov);
                    result.s_terms_n++;
                }
            }
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
        int   n_rivals = 0;
        for (int rr = 0; rr < R; ++rr)
        {
            denom += lik[rr][c];
            if (rr == r) continue;
            runnerup = std::min(runnerup, cost[rr][c]);                // best RIVAL for this detection
            // In gate at all? Count it. This is what the association had to CHOOSE BETWEEN, and it
            // is the only part of the decision that survives into the log: the winning cost is
            // truncated at the gate by construction, so it cannot report how hard the choice was.
            if (cost[rr][c] < INFEASIBLE * 0.5f) ++n_rivals;
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
        m.n_rivals       = n_rivals;
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
            if (std::isfinite(m.assoc_chi2_val))
            { result.nis_acc_sum += static_cast<double>(m.assoc_chi2_val) * 0.5; result.nis_acc_n++; }
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

    // ── Fold this frame into the run-long totals ────────────────────────────────────────────────
    // A frame carries ~14 gate evaluations; χ²₂/2 has unit variance, so the standard error on a mean
    // of 14 is 0.27 and a single frame cannot separate 0.5 from 1.0. The tour can.
    tour_.nis_pre_sum += result.nis_pre_sum; tour_.nis_pre_n += result.nis_pre_n;
    tour_.nis_pre_over += result.nis_pre_over;
    tour_.nis_acc_sum += result.nis_acc_sum; tour_.nis_acc_n += result.nis_acc_n;
    tour_.s_det_sum += result.s_det_sum; tour_.s_pred_sum += result.s_pred_sum;
    tour_.s_map_sum += result.s_map_sum; tour_.s_terms_n += result.s_terms_n;
    tour_.ori_sum += result.ori_sum; tour_.ori_n += result.ori_n;
    tour_.ori_min = std::min(tour_.ori_min, result.ori_min);
    tour_.sphi_sum += result.sphi_sum; tour_.sd_sum += result.sd_sum;
    tour_.npts_sum += result.npts_sum; tour_.resid_sig_sum += result.resid_sig_sum;
    tour_.lever_sum += result.lever_sum; tour_.line_n += result.line_n;
    tour_.angle_sum += result.angle_sum; tour_.angle_n += result.angle_n;

    return result;
}

} // namespace rc
