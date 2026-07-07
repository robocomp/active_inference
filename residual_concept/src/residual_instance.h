/*
 * residual_instance.h
 *
 * Per-obstacle runtime state shared by the residual_concept collaborators (ResidualFitter fits it,
 * ResidualSceneGraph publishes it). One ResidualInstance per "residual_N" obstacle node in the DSR graph.
 * Much slimmer than BottleInstance: no camera/mask/silhouette, no voxel bank, no support-surface, no
 * affordance/epistemic — a residual is fit from a LiDAR cluster and published as a footprint box + hull.
 */

#pragma once

#include <cstdint>
#include <limits>
#include <string>

#include <Eigen/Dense>

#include "residual_model.h"
#include "residual_belief.h"   // rc::ResidualBelief (AI2 box-footprint belief — the fit)

namespace rc
{

struct ResidualInstance
{
    std::uint64_t node_id = 0;
    std::string   node_name;

    // The fit: the recursive-Laplace full-covariance box-footprint belief on the shared engine. Lazily
    // initialised from the first cluster; run_inference writes its posterior back into `model` (the state
    // carrier). Its Σ drives the tracker gate + the published rt_covariance.
    ResidualModel  model;
    ResidualBelief belief;
    bool           belief_initialized = false;

    // ── tracker / lifecycle bookkeeping ──
    int  matched_frames       = 0;      // frames with fresh cluster data
    int  processed_cycles     = 0;      // per-instance compute cycles (log throttle)
    int  assigned_cluster_idx = -1;     // cluster assigned by the InstanceTracker this cycle (-1 = none)
    // Negative-info death gate. LiDAR is 360°, so "expected visible" = the centre is within sensor range
    // and not occluded (worker sets it). Held false when the obstacle is out of range → it persists.
    bool expected_visible     = true;
    int  frames_converged     = 0;
    int  frames_diverged      = 0;
    float prev_free_energy    = std::numeric_limits<float>::max();

    // ── occupancy evidence for REMOVAL (Bayesian ray-carving, replaces the death-frame counter) ──
    // Log-odds that the obstacle is really there. Each cycle the worker carves the LiDAR sweep against this
    // box: beams returning INSIDE raise it (still there), beams passing THROUGH lower it (compelling evidence
    // it is gone), occluded/out-of-range cycles leave it unchanged (HOLD). Starts uninformative (0 = P=0.5);
    // the birth cluster's own occupancy immediately raises it. Removed when it drops past the decision bound.
    float occupancy_logodds   = 0.0f;

    // ── dissolve-on-explained-interior (the fission policy) ──
    // Fraction of the instance's interior cluster support that a specialist SDF now explains. When a
    // specialist claims the core of this obstacle (e.g. a table modelled inside a table+chairs blob), the
    // worker invalidates the instance and lets the next cycle's residual re-cluster rebuild it fresh.
    float explained_frac = 0.0f;

    // Inferred SUPPORT-surface height (room z): what this obstacle rests on — the floor (0) by default, or a
    // tabletop when it sits on one. The box bottom is anchored here (height = z_max − support_z), so a
    // floor-standing object rises from the floor and an on-table object rises from the table, not the floor.
    float support_z = 0.0f;

    // Publish-adjusted footprint centre (room xy). The raw fit centres the box on the VISIBLE (near) face, so
    // a symmetric box pokes half its depth TOWARD the sensor into verified-free space ("closer than it should
    // be"). The published centre is shifted AWAY from the sensor so the box's near face sits on the OBSERVED
    // near surface (nf below) and all depth extends into the occluded far side. Set by the scene-graph each
    // publish; the internal fit/carve keep using s.cx/s.cy.
    float pub_cx = 0.0f, pub_cy = 0.0f;
    // Observed near-surface point (room xy): the closest cluster returns along the sensor→object axis (a low
    // percentile, outlier-robust). The published box's near face is pinned here. Set by the fitter each fit.
    float nf_x = 0.0f, nf_y = 0.0f;
    bool  nf_valid = false;

    // ── chain-cov diagnostics (published RT-edge cov, room frame, m²) ──
    float chain_cov_xx = 0.0f, chain_cov_yy = 0.0f;

    // Dense ZED-depth subcloud for THIS instance this cycle (room frame): the backprojected depth inside the
    // box region. Appended to the belief frame's points by the fitter. Empty when out of the ZED FoV / off.
    std::vector<Eigen::Vector3f> zed_points;

    std::uint64_t last_obs_timestamp_ms = 0;   // capture stamp of the last consumed cluster (RT/cov pinning)
    int  model_generation = 0;

    // ── publish dead-band traces (suppress churn once settled) ──
    float last_pub_cx  = std::numeric_limits<float>::max();
    float last_pub_cy  = std::numeric_limits<float>::max();
    float last_pub_yaw = std::numeric_limits<float>::max();
    float last_pub_w   = std::numeric_limits<float>::max();
    float last_pub_d   = std::numeric_limits<float>::max();
};

}  // namespace rc
