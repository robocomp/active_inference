/*
 * rt_covariance.h — publish an object's pose uncertainty on its parent→object RT edge. SHARED, header-only.
 *
 * This is the seam where a concept agent tells the CONTROLLER how sure it is about where a thing is. The
 * controller's speed governor, its obstacle inflation and its affordance ranking all read this block, so a
 * mistake here does not crash anything — it quietly makes the robot plan around the wrong uncertainty. That
 * is the worst kind of bug to leave duplicated, and it was duplicated five times.
 *
 * ★THE LAYOUT IS PROTOCOL, NOT PREFERENCE. 6×6 row-major over [x, y, z, rx, ry, rz], 36 floats, diagonal
 * only (cross-terms unmodelled). The consumer side indexes it by hand — the controller once read yaw from
 * cov[14] instead of cov[5*6+5] = cov[35], i.e. it was reading the z↔pitch cross-term and calling it
 * heading uncertainty. Writing the block in ONE place is what stops the producer half of that class of bug.
 *
 * ★AND THE FLOOR PRIOR IS A MODELLING STATEMENT, which is why it is a named constant here rather than a
 * literal in five files. An object resting on the floor has its roll and pitch pinned by that geometry:
 * they are CONFIDENTLY KNOWN to be ~0, not unknown. Publishing the old "unobservable ⇒ 1e3" told the
 * controller the table might be standing on its side, and — because it is a variance, on the same plot
 * scale as everything else — it dominated the published-covariance display so the real sub-1 DOF read as
 * zero. Measured 2026-08-11: table, refrigerator and cabinet had been corrected to 5e-4 (σ ≈ 1.3°); chair
 * and door were still publishing 1e3. Same fix, three agents, two left behind — a factor of 2 000 000
 * disagreement about the same physical claim, invisible because nobody was comparing.
 *
 * What stays with the AGENT is the mapping from its own DOF to these six variances — a table's z comes from
 * H/2 so var_z = var_H/4, a chair's is the floor height, a door's yaw carries discrete-mode entropy. That
 * mapping is genuinely per-object and belongs in the agent. Everything after it is this file.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <print>
#include <string_view>
#include <vector>

#include <Eigen/Dense>

#include <dsr/api/dsr_api.h>

namespace rc::rtcov {

inline constexpr int   kBlockSize = 36;      // 6×6 row-major, matches RT_COVARIANCE_BLOCK_SIZE in cortex
// Roll/pitch of an object standing on the floor: pinned by the geometry, σ ≈ 1.3°. See the header note.
inline constexpr float kFlatRollPitchVar = 5e-4f;

// The six diagonal variances, in the order the block expects. Units: m² for x/y/z, rad² for roll/pitch/yaw.
struct Se3Var
{
    float x = 0.0f, y = 0.0f, z = 0.0f;
    float roll = kFlatRollPitchVar, pitch = kFlatRollPitchVar, yaw = 0.0f;

    // ★OPTIONAL FULL TRANSLATION BLOCK, cross-terms included. A belief's position covariance is NOT
    // diagonal: an object seen down a long sightline is far more uncertain ALONG the ray than across it,
    // and that correlation is precisely what a planner needs in order to inflate in the right direction
    // instead of inflating a circle. Four agents published the diagonal only and threw the correlation
    // away; bottle_concept was already publishing the whole 3×3 and would have LOST that had this been
    // unified downward to the common shape. Union, not intersection — so the block is carried here and
    // every agent can start supplying one.
    Eigen::Matrix3f xyz = Eigen::Matrix3f::Zero();
    bool            has_xyz = false;

    // Build from a full translation covariance (m²). yaw_var is the object's own rotational uncertainty;
    // roll/pitch default to the floor prior, which is the right claim for anything standing on a surface.
    static Se3Var from_translation_block(const Eigen::Matrix3f& C, float yaw_var,
                                         float rp_var = kFlatRollPitchVar)
    {
        Se3Var v;
        v.x = C(0, 0); v.y = C(1, 1); v.z = C(2, 2);
        v.roll = rp_var; v.pitch = rp_var; v.yaw = yaw_var;
        v.xyz = C; v.has_xyz = true;
        return v;
    }

    // The self-gate reads the TRANSLATION+YAW trace only: roll/pitch are a constant prior here, so folding
    // them in would add a fixed offset that dilutes the relative change test below.
    float trace() const { return x + y + z + yaw; }
};

// Write the covariance onto the parent→object RT edge, self-gated. Returns true if it wrote.
//
// The gate: publish on a geometry republish (`force`) or when the uncertainty moved more than `change_frac`
// since the last publish — which covers a frozen pose whose belief keeps tightening. Without it a settled
// object rewrites its edge every cycle, and an RT edge write is a CRDT delta that every peer must merge.
inline bool publish(DSR::DSRGraph& G, std::uint64_t parent_id, std::uint64_t node_id,
                    const Se3Var& v, float& last_trace, bool force,
                    std::string_view node_name, float change_frac = 0.05f)
{
    if (parent_id == 0 or node_id == 0)
        return false;

    const float trace = v.trace();
    const float prev  = last_trace;
    const bool changed = not std::isfinite(prev) or prev <= 0.0f
                         or std::abs(trace - prev) > change_frac * prev;
    if (not force and not changed)
        return false;

    auto edge = G.get_edge(parent_id, node_id, "RT");
    if (not edge.has_value())
        return false;   // no RT edge yet ⇒ nothing to annotate; the pose write creates it

    std::vector<float> cov(kBlockSize, 0.0f);
    if (v.has_xyz)                       // full translation block, correlations preserved
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                cov[r * 6 + c] = v.xyz(r, c);
    else                                 // diagonal only (cross-terms unmodelled by this agent)
    {
        cov[0 * 6 + 0] = v.x;
        cov[1 * 6 + 1] = v.y;
        cov[2 * 6 + 2] = v.z;
    }
    cov[3 * 6 + 3] = v.roll;
    cov[4 * 6 + 4] = v.pitch;
    cov[5 * 6 + 5] = v.yaw;

    G.add_or_modify_attrib_local<rt_covariance_att>(edge.value(), cov);
    G.insert_or_assign_edge(edge.value());
    last_trace = trace;

    // Verification readout in the units a human checks (cm and degrees), naturally throttled by the gate.
    std::print("[{}] RT-cov σ x={:.1f}cm y={:.1f}cm z={:.1f}cm yaw={:.2f}° (src=Σ)\n",
               node_name,
               100.0f * std::sqrt(std::max(0.0f, v.x)), 100.0f * std::sqrt(std::max(0.0f, v.y)),
               100.0f * std::sqrt(std::max(0.0f, v.z)),
               57.2958f * std::sqrt(std::max(0.0f, v.yaw)));
    return true;
}

}  // namespace rc::rtcov
