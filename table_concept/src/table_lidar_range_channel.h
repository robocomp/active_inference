/*
 * table_lidar_range_channel.h — YOLO-independent LiDAR range channel for table_concept (extracted from TableFitter).
 *
 * Owns the staged per-cycle sweep (room frame + sensor origin) and selects the returns landing on ONE table,
 * staging them on the frame's LiDAR channel for the first-hit range factor / free-space VACATE term:
 * set_sweep (stage this cycle's sweep + origin, from TableFitter fed by TableLidarIngestor), clear (drop the
 * staged sweep, called each cycle before set_sweep so a stale sweep never leaks), and feed (select this
 * table's returns and populate frame.lidar). Holds the TableConfig by reference. Plain class (no Q_OBJECT).
 */

#pragma once

#include <vector>

#include <Eigen/Dense>

#include "table_config.h"     // rc::TableConfig
#include "table_instance.h"   // rc::TableInstance
#include "table_belief.h"     // rc::TableFrame

namespace rc {

class TableLidarRangeChannel
{
public:
    explicit TableLidarRangeChannel(const TableConfig& cfg) : cfg_(cfg) {}

    // Stage this cycle's sweep (room frame) + sensor origin for the range factor.
    void set_sweep(const std::vector<Eigen::Vector3f>& sweep_room, const Eigen::Vector3f& origin_room)
    { lidar_sweep_room_ = sweep_room; lidar_origin_room_ = origin_room; lidar_have_sweep_ = true; }
    // clear() each cycle first so a stale sweep never leaks into a frame with no fresh LiDAR.
    void clear() { lidar_have_sweep_ = false; }

    // Select this cycle's LiDAR returns landing on THIS table and stage them on the frame's range channel.
    void feed(TableInstance& inst, TableFrame& frame) const;

private:
    const TableConfig&             cfg_;
    // Staged LiDAR sweep for the range factor (room frame). Refreshed each compute() cycle; have flag false
    // ⇒ no fresh sweep this cycle ⇒ feed is a no-op (never reuses a stale sweep).
    std::vector<Eigen::Vector3f>   lidar_sweep_room_;
    Eigen::Vector3f                lidar_origin_room_ = Eigen::Vector3f::Zero();
    bool                           lidar_have_sweep_  = false;
};

}  // namespace rc
