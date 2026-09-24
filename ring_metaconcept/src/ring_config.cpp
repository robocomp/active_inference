/*
 * ring_config.cpp — fill RingConfig from a RoboComp ConfigLoader.
 */

#include "ring_config.h"

#include "../../common/config_report/config_read.h"   // rc::cfg::Reader (SHARED)

#include <print>

#include <genericworker.h>   // ConfigLoader

namespace rc {

RingConfig load_ring_config(const ConfigLoader& cfg)
{
    RingConfig out;

    // ConfigLoader::get has no default overload; guard every key with exists().
    // Every read below registers its key, its CODE DEFAULT and a one-line description,
    // so the startup table can say where each value came from - not just what it is.
    // (The four local lambdas now forward to the shared registry; the call sites are
    // unchanged except for that description.)
    rc::cfg::Reader reader(cfg, "ring_metaconcept");
    const auto geti = [&](std::string_view k, int def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.i(k, def, what, o); };
    const auto gets = [&](std::string_view k, std::string def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.s(k, std::move(def), what, o); };
    const auto getf = [&](std::string_view k, float def, std::string_view what,
                          rc::cfg::Opts o = {}) { return reader.f(k, def, what, o); };

    out.anchor_class      = gets("RingMetaconcept.AnchorClass", "table",
            "central member (dining table)");
    out.ring_class        = gets("RingMetaconcept.RingClass", "chair",
            "ring members arranged around the anchor");
    out.node_subtype      = gets("RingMetaconcept.NodeSubtype", "dining_set",
            "object_subtype written on the owned node");
    out.node_prefix       = gets("RingMetaconcept.NodePrefix", "dining_set_",
            "owned-node NAME prefix (sweep/cleanup key)");
    out.publish = reader.b("RingMetaconcept.Publish", true,
            "publish the ring metaconcept node onto the graph");
    out.sigma_slot_m          = getf("RingMetaconcept.SigmaSlotM", 0.20f,
            "sigma_slot_m is the SLACK between a slot and the chair occupying it — chairs get pushed in and out, so this is deliberately loose (it is not sensor…");
    out.clutter_frac          = getf("RingMetaconcept.ClutterFrac", 0.20f,
            "prior mass on 'this member belongs to no rig'");
    out.occupancy_q           = getf("RingMetaconcept.OccupancyQ", 0.70f,
            "P(a slot of a real rig is occupied)");
    out.member_yaw_offset_deg = getf("RingMetaconcept.MemberYawOffsetDeg", -90.0f,
            "Added to the inward-facing direction to express the prior in the MEMBER's yaw convention. −90° for the current chair model (backrest on −y ⇒ the…");
    out.facing_model_std_deg  = getf("RingMetaconcept.FacingModelStdDeg", 12.0f,
            "Intrinsic spread of 'a chair faces the table' — a tendency, not a law");
    out.evidence_ema_alpha    = getf("RingMetaconcept.EvidenceEmaAlpha", 0.05f,
            "★Rate at which the ring-vs-null estimate tracks the instantaneous evidence");

    out.slot_visibility_enabled = reader.b("RingMetaconcept.SlotVisibilityEnabled", true,
            "weight each slot's evidence by whether that slot was actually visible");
    out.zed_hfov_deg      = getf("RingMetaconcept.ZedHFovDeg", 100.0f,
            "ZED horizontal field of view");
    out.zed_max_range_m   = getf("RingMetaconcept.ZedMaxRangeM", 6.0f,
            "beyond this a seat is not reliably resolvable");
    out.zed_fov_soft_deg  = getf("RingMetaconcept.ZedFovSoftDeg", 12.0f,
            "soft edge width on the FoV boundary");
    out.zed_range_soft_m  = getf("RingMetaconcept.ZedRangeSoftM", 1.0f,
            "soft edge width on the range boundary");
    out.room_area_m2_fallback = getf("RingMetaconcept.RoomAreaM2Fallback", 50.0f,
            "Fallback support for the independent-objects null when the room polygon is unavailable");

    out.log_period_frames = geti("RingMetaconcept.LogPeriodFrames", 30,
            "per-cycle stdout log throttle for the graph-reader");
    out.csv_path          = gets("RingMetaconcept.CsvPath", "",
            "per-member snapshot CSV; empty disables");
    out.fit_csv_path      = gets("RingMetaconcept.FitCsvPath", "",
            "arrangement fit + would-be down-prior CSV; empty disables");

    std::print("ring_metaconcept: configuration loaded (anchor='{}' ring='{}' owns='{}*' csv='{}') PUBLISH={}.\n",
               out.anchor_class, out.ring_class, out.node_prefix,
               out.csv_path.empty() ? "<disabled>" : out.csv_path,
               out.publish ? "true (steering)" : "false (observe-only)");
    return out;
}

}  // namespace rc
