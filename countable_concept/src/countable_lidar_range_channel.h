/*
 * countable_lidar_range_channel.h — YOLO-independent LiDAR range channel for countable_concept (extracted from CountableFitter).
 *
 * Owns the staged per-cycle sweep (room frame + sensor origin) and selects the returns landing on ONE bed,
 * staging them on the frame's LiDAR channel for the first-hit range factor / free-space VACATE term:
 * set_sweep (stage this cycle's sweep + origin, from CountableFitter fed by ConceptLidarIngestor), clear (drop the
 * staged sweep, called each cycle before set_sweep so a stale sweep never leaks), and feed (select this
 * bed's returns and populate frame.lidar). Holds the CountableConfig by reference. Plain class (no Q_OBJECT).
 */

#pragma once

#include <functional>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "countable_config.h"     // rc::CountableConfig
#include "countable_instance.h"   // rc::CountableInstance
#include "countable_belief.h"     // rc::CountableFrame
#include "countable_beam_staging.h"   // rc::stage_beam_cycle (spec 7c.8 a0)

namespace rc {

class CountableLidarRangeChannel
{
public:
    explicit CountableLidarRangeChannel(const CountableConfig& cfg) : cfg_(cfg) {}

    // Stage this cycle's primary (helios) sweep (room frame) + sensor origin for the range factor / VACATE.
    void set_sweep(const std::vector<Eigen::Vector3f>& sweep_room, const Eigen::Vector3f& origin_room)
    { lidar_sweep_room_ = sweep_room; lidar_origin_room_ = origin_room; lidar_have_sweep_ = true; }
    // Stage this cycle's low (bpearl) sweep + its OWN origin — a SEPARATE per-device ray-set (own occlusion-aware
    // first-hit). Range factor only (the VACATE term stays on the primary helios sweep).
    void set_sweep_bpearl(const std::vector<Eigen::Vector3f>& sweep_room, const Eigen::Vector3f& origin_room)
    { lidar_sweep_bpearl_ = sweep_room; lidar_origin_bpearl_ = origin_room; lidar_have_bpearl_ = true; }
    // The staged helios sweep's sensor origin (room frame, from the RT chain at the sweep's capture), or nullopt when no
    // fresh sweep is staged this cycle.
    std::optional<Eigen::Vector3f> helios_origin() const
    { return lidar_have_sweep_ ? std::optional<Eigen::Vector3f>(lidar_origin_room_) : std::nullopt; }
    // Any sweep (helios or bpearl) staged this cycle: without one, feed() stages nothing.
    bool has_sweep() const { return lidar_have_sweep_ or lidar_have_bpearl_; }
    // clear() each cycle first so a stale sweep never leaks into a frame with no fresh LiDAR.
    void clear() { lidar_have_sweep_ = false; lidar_have_bpearl_ = false; }

    // Select this cycle's LiDAR returns landing on THIS bed and stage them on the frame's range channel(s). The select box
    // is anchored on `anchor_cloud` (this cycle's mask cloud; on a LiDAR-only frame the camera points that are NOT fed to
    // the engine, spec §7b). An empty `membership` ⇒ select_into (today's selection, bit for bit); else every return
    // carries the camera's m(p) as a weight (select_into_weighted) and Σm goes to inst.dbg_lidar_msum.
    void feed(CountableInstance& inst, CountableFrame& frame, std::span<const Eigen::Vector3f> anchor_cloud,
              const std::function<float(const Eigen::Vector3f&)>& membership = {}) const;

    // The persistent path's ray-set (spec 2026-11-15 lidar-measures §7c.8 a0; flag on). HELIOS only (the bpearl set is not
    // staged on this path, spec §9). Stages frame.beam by RAY GEOMETRY through rc::stage_beam_cycle (countable_beam_staging.h) on
    // the same select box (anchored on `anchor_cloud`) and weights as feed(), with the persistent `label` ({m, q}; nullopt = no
    // camera information ⇒ that ray is not staged). An EMPTY label is a cycle without membership: nothing is staged and
    // inst.beam_log counts it (owner 11-18). σ_s = cfg.model_sigma_m, p_f against the floor (Model.FloorSigmaM) and the room
    // walls set by set_room_walls. feed() still runs for the instruments (probe, dbg_lidar_rays, coverage): they keep reading
    // the endpoint-in-box subset. Logs into inst.beam_log (staged, nolabel, the κ instrument, Σq, nomem_cycle); call beam_log.begin_cycle() first.
    void feed_beam(CountableInstance& inst, CountableFrame& frame, std::span<const Eigen::Vector3f> anchor_cloud,
                   const std::function<std::optional<std::pair<double, float>>(const Eigen::Vector3f&)>& label) const;
    // The beam config's Student-t (for p_f); set once by the fitter from the BeamConfig it built. Unset ⇒ feed_beam stages nothing.
    void set_beam_density(const rc::ai::beam::StudentT& st) { beam_st_ = st; }
    // The parts of an offer's LiDAR (shared-engine G5) feed_beam would stage with: ρ_max, the model σ, the room's known
    // surfaces (floor, walls, the neighbours), the beam density. nullopt without a beam density (flag off).
    std::optional<rc::ai::offer::OfferLidar> offer_lidar_base(std::vector<rc::ai::beam::ForeignBox> neighbours) const
    {
        if (not beam_st_) return std::nullopt;
        rc::ai::offer::OfferLidar L;
        L.rho_max = static_cast<double>(rc::ai::LidarRays{}.max_range_m);
        L.sigma_s = static_cast<double>(cfg_.model_sigma_m);
        L.room.floor = true;
        L.room.floor_z = 0.0;
        L.room.floor_sigma = cfg_.floor_sigma_m;
        L.room.walls = room_walls_;
        L.room.neighbours = std::move(neighbours);
        L.st = *beam_st_;
        return L;
    }
    // The room polygon's walls as known surfaces of p_f, each with its position σ (the WallRef σ; room_concept publishes none).
    void set_room_walls(const std::vector<Eigen::Vector2f>& polygon, double sigma_m)
    {
        room_walls_.clear();
        const std::size_t n = polygon.size();
        for (std::size_t i = 0; n >= 3 and i < n; ++i)
            room_walls_.push_back({polygon[i].cast<double>(), polygon[(i + 1) % n].cast<double>(), sigma_m});
    }

    // Measurement only (label dump): the staged HELIOS returns inside this instance's select box, anchored on the same
    // mask-cloud centroid feed() uses. Empty without a fresh helios sweep or an anchor.
    // `margin_m` grows the XY radius and the z-band (dump only; the fit's selection is unchanged); dist_box_m is the
    // horizontal distance beyond the ORIGINAL select radius (0 inside).
    struct DumpReturn { Eigen::Vector3f p; float dist_box_m; };
    std::vector<DumpReturn> helios_returns_in_select_box(std::span<const Eigen::Vector3f> anchor_cloud, float margin_m = 0.f) const;

private:
    static Eigen::Vector3f cloud_centroid(std::span<const Eigen::Vector3f> cloud);

    // Select returns from `sweep` (sensor at `origin`) that land on this bed into `out`, with the two
    // informativeness down-weights (ray-count + angular coverage) applied to `precision_base`. Returns the
    // staged ray count; writes the angular-coverage weight to *cov_ang (if non-null). Shared by helios + bpearl.
    int select_into(const CountableInstance& inst, const Eigen::Vector3f& centroid,
                    const std::vector<Eigen::Vector3f>& sweep, const Eigen::Vector3f& origin,
                    float precision_base, rc::ai::LidarRays& out, float* cov_ang = nullptr,
                    const std::function<float(const Eigen::Vector3f&)>* membership = nullptr,
                    float* m_sum = nullptr) const;

    // Birth-footprint select box: rotation-agnostic circumscribed XY radius + floor-referenced z-band (spans the
    // legs + rim). Fixed from the BIRTH dims (never the fitted w/h, so a blown-up extent can't explode the region).
    struct SelectBox { float rxy, z_lo, z_hi; };
    SelectBox select_box() const;

    const CountableConfig&             cfg_;
    std::optional<rc::ai::beam::StudentT> beam_st_;            // the beam density (set_beam_density); unset ⇒ no beam staging
    std::vector<rc::ai::beam::RoomWall>   room_walls_;         // the room polygon's walls (set_room_walls)
    // Staged LiDAR sweeps for the range factor (room frame). Refreshed each compute() cycle; have flag false
    // ⇒ no fresh sweep this cycle ⇒ that plane is a no-op (never reuses a stale sweep).
    std::vector<Eigen::Vector3f>   lidar_sweep_room_;
    Eigen::Vector3f                lidar_origin_room_ = Eigen::Vector3f::Zero();
    bool                           lidar_have_sweep_  = false;
    std::vector<Eigen::Vector3f>   lidar_sweep_bpearl_;
    Eigen::Vector3f                lidar_origin_bpearl_ = Eigen::Vector3f::Zero();
    bool                           lidar_have_bpearl_ = false;
};

}  // namespace rc
