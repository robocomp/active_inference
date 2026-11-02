#pragma once
/*
 * door_leaf_memory.h — the leaf's side as a PUBLISHED, REMEMBERED probability (2026-11-02, chained-actions spec
 * "Leaf direction in memory"). Pure (Eigen-free): the writer, the seeder, the standoff and their test share it.
 *
 * Two attributes on every door node (cortex dsr_attr_name.h):
 *   door_leaf_into_room_prob   P(the leaf swings INTO the room this door node hangs from);  0.5 = unknown
 *   door_leaf_hinge_plus_prob  P(the hinge is at the +x end of the aperture, in THIS node's own frame); 0.5 = unknown
 *
 * SIGN CONVENTIONS (derive both from the instance's own variables — nothing else is authoritative):
 *   - LeafState::swing = +1 opens along across_u = the wall normal that associate_wall points INTO the room. So
 *     P(into room) = P(swing = +1), and the same hole seen from the room on its OTHER side is the complement: 1 - p.
 *   - A door node's x axis is the wall tangent (ap_yaw = wall_u); HingeSide::Near is the jamb at s (the -x end),
 *     Far the jamb at s + w (the +x end). The two faces of one passage are rotated by pi (opposite wall tangents), so
 *     hinge_plus in the other face's frame is 1 - p; in general it flips with the sign of (x_here . x_there).
 *   - The instance keeps log-odds RELATIVE to its MAP side (swing_llr / hinge_llr, >= 0 once settled; a re-signed
 *     value may be negative). The published probability is the sigmoid of that, mapped to the "+" side.
 */
#include <algorithm>
#include <cmath>
#include <optional>
#include <utility>

namespace rc::door::leafmem
{
[[nodiscard]] inline float sigmoid(float l) { return 1.0f / (1.0f + std::exp(-l)); }

/// log-odds of p. The clamp is NUMERICAL only (a probability that saturated to exactly 0 or 1 in float has no finite
/// logit): 1e-6 is a +-13.8 nat belief, far beyond anything the evidence accumulates to a decision.
[[nodiscard]] inline float logit(float p)
{
    constexpr float eps = 1e-6f;
    const float q = std::clamp(p, eps, 1.0f - eps);
    return std::log(q / (1.0f - q));
}

/// P(the leaf swings into the instance's room) from the MAP swing side (+-1, sign of LeafState::swing) and the log-odds of that MAP side.
[[nodiscard]] inline float p_into_room(float swing_map, float swing_llr_map)
{
    const float p_map = sigmoid(swing_llr_map);
    return swing_map >= 0.0f ? p_map : 1.0f - p_map;
}

/// The inverse: the (MAP swing, log-odds of the MAP side) a remembered P(into room) stands for. p = 0.5 -> (+1, 0).
struct SwingBelief { float swing = 1.0f; float llr = 0.0f; };
[[nodiscard]] inline SwingBelief swing_from_into_room(float p_into)
{
    const float l = logit(p_into);
    return {l >= 0.0f ? 1.0f : -1.0f, std::abs(l)};
}

/// P(hinge at the node's +x end) from the MAP hinge (is it at +x?) and the log-odds of the MAP jamb.
[[nodiscard]] inline float p_hinge_plus(bool map_at_plus, float hinge_llr_map)
{
    const float p_map = sigmoid(hinge_llr_map);
    return map_at_plus ? p_map : 1.0f - p_map;
}

/// The inverse. `plus_is_far` = the node's +x end is the belief's FAR jamb (x_node . wall_u >= 0). Returns (hinge_is_far, llr of that jamb).
struct HingeBelief { bool far = false; float llr = 0.0f; };
[[nodiscard]] inline HingeBelief hinge_from_plus(float p_plus, bool plus_is_far)
{
    const float l = logit(p_plus);
    const bool map_plus = l >= 0.0f;
    return {map_plus == plus_is_far, std::abs(l)};
}

/// The same hole's other face: swing flips (the room on the other side), the hinge flips with the frame.
[[nodiscard]] inline float other_face_into_room(float p_into) { return 1.0f - p_into; }
[[nodiscard]] inline float other_face_hinge_plus(float p_plus, bool frames_flipped) { return frames_flipped ? 1.0f - p_plus : p_plus; }

/// What a recall node (ltsm, hung under a live door) carries: the leaf attributes of the remembered face IN THE CURRENT ROOM
/// (the face the live door matches), as {P(into), P(hinge at +x)}. A face "has" them when either attribute is present and the
/// pair is not the unknown {0.5, 0.5}. Near face first; otherwise the FAR face's, complemented (other_face_*: the far face's
/// frame is rotated by pi, hinge flips with it); neither -> nullopt (write nothing). A single missing attribute reads 0.5.
using LeafProbs = std::pair<float, float>;
using LeafAttrs = std::pair<std::optional<float>, std::optional<float>>;
[[nodiscard]] inline std::optional<LeafProbs> recall_leaf_probs(const LeafAttrs& near_face, const LeafAttrs& far_face)
{
    const auto known = [](const LeafAttrs& a) {
        return (a.first.has_value() or a.second.has_value()) and (a.first.value_or(0.5f) != 0.5f or a.second.value_or(0.5f) != 0.5f);
    };
    if (known(near_face)) return LeafProbs{near_face.first.value_or(0.5f), near_face.second.value_or(0.5f)};
    if (known(far_face))
    {
        const float pi = far_face.first.value_or(0.5f), ph = far_face.second.value_or(0.5f);
        return LeafProbs{other_face_into_room(pi), other_face_hinge_plus(ph, /*frames_flipped=*/true)};
    }
    return std::nullopt;
}

/// A remembered side arriving AFTER the belief was built (a recall posted once the room is current): the memory is one more
/// independent observation of each binary latent, so it ADDS its signed log-odds to the belief's (naive Bayes, flat prior — the
/// same evidence the pre-belief seed would have been, only it must not discard what was already gathered). Inputs are the
/// belief's MAP side and its EFFECTIVE log-odds (>= 0 when settled; negative if the reference side differs). Output is the
/// new MAP side and |log-odds|; a tie keeps the +/far side.
struct FusedLeaf { float swing = 1.0f; float swing_llr = 0.0f; bool hinge_far = false; float hinge_llr = 0.0f; };
[[nodiscard]] inline FusedLeaf fuse_recall(float swing, float swing_llr_eff, bool hinge_far, float hinge_llr_eff,
                                           const LeafProbs& mem, bool plus_is_far)
{
    FusedLeaf o;
    const float ls = (swing >= 0.0f ? 1.0f : -1.0f) * swing_llr_eff + logit(mem.first);
    o.swing = ls >= 0.0f ? 1.0f : -1.0f;  o.swing_llr = std::abs(ls);
    const auto hm = hinge_from_plus(mem.second, plus_is_far);
    const float lh = (hinge_far ? 1.0f : -1.0f) * hinge_llr_eff + (hm.far ? 1.0f : -1.0f) * hm.llr;
    o.hinge_far = lh >= 0.0f;  o.hinge_llr = std::abs(lh);
    return o;
}

/// P(the leaf swings toward the side of the wall the robot stands on). `side` = +1 when the robot is on the across_u
/// (room-interior) side, -1 beyond. A continuous mix over the two sides when `side` is a probability: pass P(robot on +).
[[nodiscard]] inline float p_leaf_toward(float p_into, float p_robot_on_plus)
{
    return p_into * p_robot_on_plus + (1.0f - p_into) * (1.0f - p_robot_on_plus);
}

/// Phi^-1(p), Acklam's rational approximation (|relative error| < 1.2e-9); p in (0,1).
[[nodiscard]] inline float normal_quantile(float p)
{
    constexpr double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
    constexpr double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01};
    constexpr double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
    constexpr double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00};
    constexpr double lo = 0.02425;
    const double q = std::clamp(static_cast<double>(p), 1e-12, 1.0 - 1e-12);   // numerical domain guard only
    if (q < lo)
    {
        const double r = std::sqrt(-2.0 * std::log(q));
        return static_cast<float>((((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / ((((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1.0));
    }
    if (q > 1.0 - lo)
    {
        const double r = std::sqrt(-2.0 * std::log(1.0 - q));
        return static_cast<float>(-(((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / ((((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1.0));
    }
    const double r = (q - 0.5) * (q - 0.5), t = (q - 0.5);
    return static_cast<float>((((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*t / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0));
}

/// THE SWEPT ARC THE ROBOT MUST STAY OUT OF, weighted by the leaf's side (a chance constraint, no gate).
/// `arc` = the full swept radius + body (door_pragmatics.h safe_standoff), the requirement when the leaf surely swings toward
/// the robot. The robot is hit iff the leaf swings toward it (p_toward) AND it stands inside the arc, so P(hit) = p_toward *
/// Phi((arc - r)/sigma) must stay <= risk (the motion policy's own, door_close_clear.h kCloseContactRiskMax). Solving for r:
///     r >= arc - sigma * ( z(1 - risk) - z(1 - risk / p_toward) )        (z = Phi^-1; equals `arc` at p_toward = 1)
/// which falls continuously as the leaf is more likely to swing AWAY, and bottoms at the body alone (`body_half`) once
/// p_toward <= risk (no arc left to be hit by). A leaf of unknown side (0.5) keeps ~the whole arc: the risk budget is 1%, not 50%.
[[nodiscard]] inline float swing_weighted_arc(float arc, float body_half, float sigma, float p_toward, float risk)
{
    if (p_toward <= risk) return std::min(arc, body_half);
    const float shrink = sigma * (normal_quantile(1.0f - risk) - normal_quantile(1.0f - risk / p_toward));
    return std::clamp(arc - shrink, std::min(arc, body_half), arc);
}
}  // namespace rc::door::leafmem
