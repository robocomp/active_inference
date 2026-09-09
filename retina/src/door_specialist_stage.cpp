#include "door_specialist_stage.h"

#include "yolo_seg_detector.h"   // SegDetection (what the cheap channel puts in out.masks)

namespace rc
{

DoorSpecialistStage::DoorSpecialistStage(const rc::doors::DoorSpecialist::Config& cfg, int decimation)
    : decimation_(std::max(1, decimation))
{
    spec_.configure(cfg);
}

void DoorSpecialistStage::run(const PerceptionFrame& in, PerceptionResult& out)
{
    if (not ready() or in.rgbd.bgr.empty() or in.is_360)
        return;   // ZED only: the panorama's geometry is a different question and a different model

    // Did the CHEAP channel already speak about doors this frame? This stage must run after seg and
    // semantic_masks, which the worker's stage order guarantees.
    bool cheap_has_door = false;
    if (out.masks)
        for (const auto& m : *out.masks)
            if (m.label == "door") { cheap_has_door = true; break; }

    DoorSpecialistResult res;
    res.eligible = not cheap_has_door;
    if (cheap_has_door)
    {
        // Not asked, and deliberately so — the door lives on the cheap channel's own evidence. Reported
        // rather than silently skipped, so a consumer never reads this frame as a denial.
        out.door_specialist = std::move(res);
        return;
    }

    // Decimate INSIDE the silence. Every silent run still gets its first frame (the counter is 0 at the
    // start of a run only if it happens to align, so the cost of N is LATENCY, bounded by the removal
    // budget, never a run that goes entirely unexamined at steady state).
    const bool run_now = (silent_seen_++ % static_cast<std::uint64_t>(decimation_) == 0);
    if (not run_now)
    {
        out.door_specialist = std::move(res);
        return;
    }

    res.ran = true;
    res.detections = spec_.detect(in.rgbd.bgr);
    out.door_specialist = std::move(res);
}

}   // namespace rc
