/*
 * ring_config.cpp — fill RingConfig from a RoboComp ConfigLoader.
 */

#include "ring_config.h"

#include <print>

#include <genericworker.h>   // ConfigLoader

namespace rc {

RingConfig load_ring_config(const ConfigLoader& cfg)
{
    RingConfig out;

    // ConfigLoader::get has no default overload; guard every key with exists().
    auto geti = [&](const std::string& k, int def) -> int {
        return cfg.exists(k) ? cfg.get<int>(k) : def;
    };
    auto gets = [&](const std::string& k, std::string def) -> std::string {
        return cfg.exists(k) ? cfg.get<std::string>(k) : def;
    };
    auto getf = [&](const std::string& k, float def) -> float {
        return cfg.exists(k) ? static_cast<float>(cfg.get<double>(k)) : def;
    };

    out.anchor_class      = gets("RingMetaconcept.AnchorClass", "table");
    out.ring_class        = gets("RingMetaconcept.RingClass",   "chair");
    out.node_subtype      = gets("RingMetaconcept.NodeSubtype", "dining_set");
    out.node_prefix       = gets("RingMetaconcept.NodePrefix",  "dining_set_");
    out.publish           = cfg.exists("RingMetaconcept.Publish")
                                ? cfg.get<bool>("RingMetaconcept.Publish") : true;
    out.sigma_slot_m          = getf("RingMetaconcept.SigmaSlotM", 0.20f);
    out.clutter_frac          = getf("RingMetaconcept.ClutterFrac", 0.20f);
    out.occupancy_q           = getf("RingMetaconcept.OccupancyQ", 0.70f);
    out.member_yaw_offset_deg = getf("RingMetaconcept.MemberYawOffsetDeg", -90.0f);
    out.facing_model_std_deg  = getf("RingMetaconcept.FacingModelStdDeg", 12.0f);
    out.evidence_ema_alpha    = getf("RingMetaconcept.EvidenceEmaAlpha", 0.05f);

    out.slot_visibility_enabled = cfg.exists("RingMetaconcept.SlotVisibilityEnabled")
                                      ? cfg.get<bool>("RingMetaconcept.SlotVisibilityEnabled") : true;
    out.zed_hfov_deg      = getf("RingMetaconcept.ZedHFovDeg", 100.0f);
    out.zed_max_range_m   = getf("RingMetaconcept.ZedMaxRangeM", 6.0f);
    out.zed_fov_soft_deg  = getf("RingMetaconcept.ZedFovSoftDeg", 12.0f);
    out.zed_range_soft_m  = getf("RingMetaconcept.ZedRangeSoftM", 1.0f);
    out.room_area_m2_fallback = getf("RingMetaconcept.RoomAreaM2Fallback", 50.0f);

    out.log_period_frames = geti("RingMetaconcept.LogPeriodFrames", 30);
    out.csv_path          = gets("RingMetaconcept.CsvPath", "");
    out.fit_csv_path      = gets("RingMetaconcept.FitCsvPath", "");

    std::print("ring_metaconcept: configuration loaded (anchor='{}' ring='{}' owns='{}*' csv='{}') PUBLISH={}.\n",
               out.anchor_class, out.ring_class, out.node_prefix,
               out.csv_path.empty() ? "<disabled>" : out.csv_path,
               out.publish ? "true (steering)" : "false (observe-only)");
    return out;
}

}  // namespace rc
