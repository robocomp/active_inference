#pragma once

// Second-opinion door detector, run ONLY on frames where the cheap channel said nothing about doors.
//
// ★THE CASCADE ARGUMENT IS STRUCTURAL, NOT EMPIRICAL. When the semantic path already yields a door mask
// the door lives anyway; the specialist could not change that outcome. It can only ever change one by
// refuting a SILENCE. So inference is spent exactly where a null result would otherwise be charged as
// absence, and nowhere else. Measured over a 40.9 min tour: silence is 36.8% of frames, and decimating
// inside each silent run to ~1.5-2 Hz costs ~10-12% duty with the first call within 0.62 s — inside
// door_concept's ~1.6 s removal budget.
//
// ★THE GATE IS OBSERVED DATA, NOT A BELIEF. This stage reads what the cheap channel produced THIS frame;
// it never consults any agent's belief. That keeps the selection rule a function of already-observed
// data, which does not bias a likelihood-ratio update — it changes which looks you have, not what each
// look means. (A belief-conditioned trigger is also defensible, but it belongs to the consumer, not
// here, and it would need the provenance below to stay honest.)
//
// ★PROVENANCE IS MANDATORY. A channel running below 100% duty MUST publish whether it ran, or a
// consumer cannot tell "the specialist denied a door here" from "the specialist was never asked" — and
// those two must move a belief in opposite directions. `ran` is therefore reported separately from the
// detections and is set even when the list is empty.

#include <memory>
#include <vector>

#include "door_specialist.h"
#include "perception_stage.h"

namespace rc
{

class DoorSpecialistStage : public Stage
{
public:
    // `decimation` counts SILENT frames, not all frames: the clock only advances while the specialist
    // is actually eligible, so a long confident stretch does not "bank" credit that fires a burst the
    // moment the cheap channel blinks.
    DoorSpecialistStage(const rc::doors::DoorSpecialist::Config& cfg, int decimation);

    const char* name() const override { return "door_specialist"; }
    bool ready() const override { return spec_.ready(); }
    void run(const PerceptionFrame& in, PerceptionResult& out) override;

    [[nodiscard]] const rc::doors::DoorSpecialist& detector() const noexcept { return spec_; }

private:
    rc::doors::DoorSpecialist spec_;
    int           decimation_ = 1;
    std::uint64_t silent_seen_ = 0;   // silent frames encountered, the decimation clock
};

}   // namespace rc
