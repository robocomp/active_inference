// residual_card.h -- this agent's PERCEPTION CARD, built from the config the agent actually runs with.
// Header-only, framework-free (needs only residual_config.h): the unit test builds it without Qt / DSR.
// Contract: common/perception_card/card.h.
//
// residual_concept models UNEXPLAINED OCCUPANCY as stuff (a log-odds grid of cells), not as objects, so the
// card is level Mass: the only term is `coverage` (the visible fraction of the region, frustum ^ walls ^
// obstacles). The evaluator's mass path returns visible_fraction x the card's terms and applies NO detector
// envelope, so SubtypeParams::env stays at its defaults and is UNUSED (it is not residual's, and nothing is fitted).
#pragma once

#include "residual_config.h"
#include "../../common/perception_card/card.h"

namespace rc
{
// cell_size_m: the occupancy grid's resolution (OccupancyGrid::Params::cell_size_m, 0.05 m; it is a grid parameter,
// not a ResidualConfig key, so the caller passes grid_.params().cell_size_m).
inline card::Card make_card(const ResidualConfig& cfg, float cell_size_m = 0.05f)
{
    card::Card c;
    c.concept_ = "residual";
    c.graph_type = "obstacle";   // DSR node type of the residual_<N> nodes (obstacle_node_type in residual_scene_graph.cpp)
    c.level = card::Level::Mass;
    // All three are Occupancy: a return / depth hit only says "something is here", never what.
    //  helios: always fused (residual_lidar_ingestor.cpp planes{"helios"}).
    //  bpearl: fused when ResidualModel.UseBpearl (cfg.use_bpearl, ON in etc/config.toml).
    //  zed:    dense depth feeds the grid (ZedBoost.FeedGrid -> grid_zed_enabled) and the belief (ZedBoost.Enabled -> zed_boost_enabled); detection
    //          from ZED is a separate flag (zed_detection_enabled, OFF) and is not what makes the channel modelled.
    c.channels = {{"helios", card::Role::Occupancy, true},
                  {"bpearl", card::Role::Occupancy, cfg.use_bpearl},
                  {"zed", card::Role::Occupancy, cfg.grid_zed_enabled or cfg.zed_boost_enabled}};
    // A mass has no natural size. Footprint = one grid cell (the unit residual reasons in); z = the nav band the grid
    // marks hits in: Clusterer.FloorZ0 .. Clusterer.CeilZ (floor returns below it, ceiling above it are not residual).
    c.geometry = {cell_size_m, cell_size_m, cfg.cluster.ceil_z - cfg.cluster.floor_z0, cfg.cluster.floor_z0, cfg.cluster.ceil_z};
    // Level 1: residual has NO wall line-of-sight test of its own: its grid is fed by real beams, which the physical
    // walls stop. So: walls as lines (where the beams stop), no own-wall skip. (Returns within the wall margin of the
    // polygon are DROPPED, which is a blind spot, not an occlusion: see blind_spots.)
    c.walls = card::WallOcclusion{0.0f, -1.0f, 0.0f};
    card::SubtypeParams sp;
    // q_return stays at the default 1.0: UNCALIBRATED. residual has CarveParams detection/clutter rates, but those
    // are the old carve's per-box rates (and the grid's inverse-sensor model), not a measured per-beam return probability.
    sp.terms = {{"coverage", {}}};
    c.subtypes[""] = sp;
    c.blind_spots = {
        "A mass has no identity and no size prior: the card gives only how much of a region the sensors can see (coverage).",
        "Nothing thinner than the beam spacing can hide between rings; the nav band excludes returns below Clusterer.FloorZ0 and above Clusterer.CeilZ.",
        "q_return is uncalibrated (1.0); no per-beam return probability is measured for residual.",
        "Returns within Clusterer.WallMarginM of the room polygon are dropped as wall: anything standing against a wall is invisible to it."};
    c.calibration = "prior";
    return c;
}
}  // namespace rc
