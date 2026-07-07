/*
 * residual_clusterer.h  —  the residual-concept PERCEPTION FRONT-END (replaces MaskIngestor)
 *
 * This is the one genuine deviation from the concept-agent recipe: instead of a YOLO mask, the residual
 * concept's detections come from RAW LiDAR. Each cycle the clusterer:
 *   1. RESIDUAL FILTER — keep only returns that NO specialist explains and that aren't floor/ceiling, the
 *      robot itself, or a wall. "Explained" is a 3D SDF-RESIDUAL test against each specialist's published
 *      generative geometry (|sdf(p)| < explain_margin), NOT a 2D footprint/bounding-box test — that is what
 *      lets an unknown object resting ON a table (above the tabletop SDF) survive as a residual instead of
 *      being swallowed by the table's footprint column.
 *   2. 3D DBSCAN — density-cluster the surviving points in xyz (NOT XY-projected): a cluster sitting on a
 *      table is separable in z from the tabletop, and — crucially — when a specialist later claims the
 *      connecting mass (e.g. a table modelled between four pushed-in chairs), the residual disconnects and
 *      the next DBSCAN pass re-partitions one big cluster into several (the FISSION dynamic).
 *   3. OBSERVATION — per cluster, a PCA oriented-box footprint SEED (for the belief) + the 2D CONVEX HULL of
 *      the floor projection + the [z_min,z_max] band (Layer B: the geometry the planner actually consumes).
 *
 * Pure Eigen + STL, DSR-free → unit-testable in isolation (see self_test()). The worker builds the
 * SpecialistSdf list from the graph's table/chair/bottle/human nodes and feeds room-frame LiDAR in.
 *
 * NOTE on thresholds (CLAUDE.md): the z-band, robot-footprint and wall-margin rejects ARE geometric masks,
 * not inference — flagged as the unavoidable exception. The CLAIMING itself (explain_margin) stays an
 * evidential SDF-residual, and instance birth/death lives in the shared evidential instance_tracker.
 */

#pragma once

#include <cstdint>
#include <functional>
#include <vector>
#include <Eigen/Dense>

#include "../../common/existence_belief/existence_belief.h"   // shared occupancy carve + existence log-odds

namespace rc
{

// A specialist's explanation of space: its generative SDF (min over the specialist's primitives), room
// frame, metres. A point is "explained" iff |sdf(p)| < explain_margin_m. Used by the DISSOLVE test.
using SpecialistSdf = std::function<float(const Eigen::Vector3f&)>;

// A unified point EXPLAINER: "does the concept I model account for this LiDAR return?". The residual is
// everything NO explainer accounts for. The room's FLOOR/CEILING/WALLS (from room_concept) and the
// ROBOT itself are explainers exactly like tables/chairs — so there are no geometric-threshold masks, only
// evidential model explanation. Each explainer owns its own tolerance (e.g. the floor's grows with range).
using SpecialistExplainer = std::function<bool(const Eigen::Vector3f&)>;

struct ResidualClusterParams
{
    // ── geometric masks (the flagged threshold exception; not inference) ──
    // RANGE-DEPENDENT floor band (not a flat z gate): a LiDAR beam grazing the floor lands HIGHER as range
    // grows (beam pitch/noise × distance) + the ~3 cm floor-datum offset, so a flat cutoff can't separate a
    // far floor return (z≈0.22 @6 m) from a close low obstacle. Reject p if z < floor_z0 + floor_slope·range
    // (range = horizontal distance from the sensor). Mirrors the codebase's range-grows-covariance pattern.
    float floor_z0         = 0.06f;   // floor band at the sensor
    float floor_slope      = 0.04f;   // + this per metre of horizontal range (≈beam grazing at distance)
    float ceil_z           = 1.80f;   // drop returns above this (ceiling / overhead)
    float robot_radius_m   = 0.40f;   // drop returns within this of the robot centre (self-returns)
    float wall_margin_m    = 0.30f;   // drop returns within this of the room delimiting polygon boundary
    // A residual OBSTACLE must have vertical extent OR be an elevated surface. Drop a cluster only if it is
    // BOTH flat (extent < min_vertical_extent_m) AND low (top < min_top_z_m) — that is floor. An ELEVATED
    // flat sheet (a tabletop/shelf, top ≥ min_top_z_m) is a REAL obstacle and is KEPT: a downward LiDAR sees
    // a table mostly as its (leg-occluded) flat top, so dropping it would lose the table. z-anisotropic
    // clustering merges a desk's top into its body, so a *standalone* elevated flat sheet really is a table.
    float min_vertical_extent_m = 0.10f;
    float min_top_z_m           = 0.30f;
    // ── evidential claiming (SDF-residual) ──
    float explain_margin_m = 0.04f;   // a point within this of a specialist SURFACE is claimed (≈sensor noise;
                                      //   tight so an on-table object is NOT swallowed by the tabletop slab)
    // ── DBSCAN ──
    // eps admits LARGER ensembles: a downward (helios) LiDAR paints a big surface (tabletop) as concentric
    // rings with gaps, so a too-small eps shatters it into arcs. 0.20 connects the rings into one cluster.
    float dbscan_eps_m     = 0.20f;   // neighbourhood radius (m)
    int   dbscan_min_pts   = 5;       // core-point density
    int   min_cluster_pts  = 12;      // drop clusters smaller than this (margin leakage / sparse noise)
    // Anisotropic z: z-distance is scaled by this before clustering, so vertically-separated LiDAR layers
    // of ONE object (a pillar's rings, a surface directly above its body) connect into a SINGLE cluster
    // instead of splitting into per-layer sheets. <1 = compress z (bridge vertical gaps); xy is unchanged,
    // so horizontally-distinct objects (and the fission chairs) stay separate.
    float dbscan_z_scale   = 0.35f;
    // ── navigation fragment-merge (post-DBSCAN) ──
    // The robot cannot pass between two obstacles closer than its own width, so for a NAVIGATION map they are
    // effectively ONE obstacle. After clustering, union any two clusters whose closest FOOTPRINT points are
    // within this gap (= robot width + an excess). This coalesces a LiDAR-shattered tabletop (concentric
    // ring-arcs, each gap ≪ robot width) into ONE coherent obstacle, LiDAR-only — so it also holds when the
    // ZED FoV swings away (the arcs are still within a robot-width each cycle → they re-merge, no re-fragment).
    float merge_gap_m      = 0.0f;    // 0 = off
};

// One residual cluster ready for the belief (seed) + the scene graph (hull/z-band).
struct ResidualCluster
{
    std::vector<Eigen::Vector3f> points;                     // room-frame member points
    Eigen::Vector2f              centroid = Eigen::Vector2f::Zero();  // OBB centre (xy)
    float                        yaw_seed = 0.0f;            // PCA principal-axis orientation (rad)
    float                        w_seed   = 0.20f;           // OBB extent along the principal axis (m)
    float                        d_seed   = 0.20f;           // OBB extent along the secondary axis (m)
    float                        z_min    = 0.0f;            // vertical band → anchored cz, height
    float                        z_max    = 0.0f;
    std::vector<Eigen::Vector2f> hull;                       // 2D convex-hull footprint (Layer B)
};

class ResidualClusterer
{
public:
    ResidualClusterer() = default;
    explicit ResidualClusterer(const ResidualClusterParams& p) : params_(p) {}
    void set_params(const ResidualClusterParams& p) { params_ = p; }
    const ResidualClusterParams& params() const { return params_; }

    // Full pipeline: residual filter (keep points NO explainer accounts for) → 3D DBSCAN → per-cluster
    // observation. The worker builds `explainers` = room floor/ceiling/walls + robot + modelled objects.
    std::vector<ResidualCluster> extract(const std::vector<Eigen::Vector3f>&      room_points,
                                         const std::vector<SpecialistExplainer>&  explainers) const;

    // ── pieces (static, pure — exposed for reuse + unit tests) ──
    std::vector<Eigen::Vector3f> residual_filter(const std::vector<Eigen::Vector3f>&     room_points,
                                                 const std::vector<SpecialistExplainer>& explainers) const;
    // Grid-accelerated 3D DBSCAN; returns one index-list per cluster (noise points dropped).
    static std::vector<std::vector<int>> dbscan_3d(const std::vector<Eigen::Vector3f>& pts, float eps, int min_pts);
    static ResidualCluster make_cluster(const std::vector<Eigen::Vector3f>& pts);   // PCA OBB seed + hull + z
    // Navigation fragment-merge: union clusters whose closest footprint (xy) points are within gap_m, then
    // re-fit ONE box+hull to each merged group (transitive: A–B–C chains collapse). gap_m ≤ 0 → no-op.
    static std::vector<ResidualCluster> merge_close_clusters(std::vector<ResidualCluster> clusters, float gap_m);
    static std::vector<Eigen::Vector2f> convex_hull_2d(std::vector<Eigen::Vector2f> pts);
    static bool  point_in_polygon(const std::vector<Eigen::Vector2f>& poly, const Eigen::Vector2f& p);
    static float dist_to_polygon_boundary(const std::vector<Eigen::Vector2f>& poly, const Eigen::Vector2f& p);

    // ── Robust infrastructure subtraction for a DENSE cloud (ZED) ──────────────────────────────────────
    // The floor/ceiling/wall subtraction that works for sparse LiDAR LEAKS on a dense stereo cloud: a small
    // depth-calibration offset or the ZED's range-degrading depth noise lets a continuous SHEET of floor/wall
    // points survive and BRIDGE distinct objects into one blob. Fix (no arbitrary threshold): scale the
    // removal band by the cloud's OWN range-dependent depth uncertainty. Stereo depth noise grows with
    // range²: σ(r) = σ0 + q·r² (a physical sensor property). A point is FLOOR if z < floor_z0 + k·σ(r),
    // CEILING if z > ceil_z − k·σ(r); WALL if it is OUTSIDE the room polygon (a hard boundary — nothing real
    // is beyond the room) or within (wall_margin + k·σ(r)) of it. What survives is the ZED residual. Pure.
    struct DepthInfraParams
    {
        float floor_z0     = 0.06f;   // floor height at the sensor (m)
        float ceil_z       = 1.80f;   // ceiling height (m)
        float wall_margin_m = 0.10f;  // base wall tolerance (grown by k·σ(r))
        float sigma0_m     = 0.03f;   // depth-noise floor (m) — near-range + calibration offset
        float sigma_quad   = 0.006f;  // depth-noise range² coeff (m/m²): σ = σ0 + sigma_quad·r²  (stereo physics)
        float k            = 3.0f;    // how many σ of band (confidence)
    };
    static std::vector<Eigen::Vector3f> subtract_infrastructure(
        const std::vector<Eigen::Vector3f>& cloud_room, const Eigen::Vector3f& sensor_origin,
        const std::vector<Eigen::Vector2f>& room_polygon, const DepthInfraParams& p);

    // A convenience 3D oriented-box SDF the worker uses to build a SpecialistSdf from a published box node
    // (table/chair/generic). centre + half-extents (hx,hy,hz), yaw about z.
    static float box_sdf_3d(const Eigen::Vector3f& p, const Eigen::Vector3f& centre, float yaw,
                            float hx, float hy, float hz);

    // ── Free-space (ray-carving) occupancy evidence for REMOVAL — Bayesian, no arbitrary thresholds ─────
    // Along each LiDAR beam (sensor origin → return), compare where the return LANDED to where the predicted
    // box SURFACES are. There are no hard margins: the surface position is uncertain (belief Σ projected +
    // sensor range noise), so the classification is SOFT via the Gaussian CDF Φ. For a beam crossing the box:
    //   p_reached = Φ((t_ret − t_near)/σ)   — prob the return got AT/BEYOND the near surface (not occluded)
    //   p_through = Φ((t_ret − t_far )/σ)   — prob the return is BEYOND the far surface (passed clean through)
    //   e_occ = p_reached − p_through (returned INSIDE ⇒ still there),  e_free = p_through (see-through ⇒ gone)
    // and the z-band membership is likewise a soft Φ weight (a beam over the top contributes ~0). The per-beam
    // occupancy LOG-ODDS increment uses only the sensor's own detection/clutter probabilities (interpretable
    // physical rates, not gates): ΔL = e_occ·log(p_det/p_clut) + e_free·log((1−p_det)/(1−p_clut)). The cycle's
    // beams share a common-mode (registration) error, so the summed ΔL is soft-SATURATED (tanh) to one
    // confident observation's worth — the same "N correlated points can't collapse σ" discipline as the belief.
    // The worker integrates ΔL into a per-instance log-odds and removes when P(occupied) crosses the decision
    // boundary. Pure geometry + Eigen → unit-tested.
    // The occupancy carve + evidence types now live in the shared existence belief (common/existence_belief).
    // Kept as class-scoped aliases + a thin forwarder so existing residual call sites are UNCHANGED (regression).
    using CarveParams   = rc::exist::SensorModel;
    using CarveEvidence = rc::exist::Evidence;
    static CarveEvidence carve_box(const Eigen::Vector3f& origin, const std::vector<Eigen::Vector3f>& sweep,
                                   float cx, float cy, float yaw, float w, float d, float z_min, float z_max,
                                   float surface_sigma_m, const CarveParams& p)
    { return rc::exist::carve_box(origin, sweep, cx, cy, yaw, w, d, z_min, z_max, surface_sigma_m, p); }

    static bool self_test();

private:
    ResidualClusterParams params_;
};

}  // namespace rc
