#pragma once

// SAM2 mask-refinement stage. Runs AFTER SegStage in the worker's stage list: it reads the YOLO
// detections already in PerceptionResult::masks, keeps only the configured target classes, and asks
// SAM2 for a tight mask per object → PerceptionResult::refined_masks. Decimated + runtime-gated (the
// heavy 1024² encoder only fires when enabled AND there is at least one target object in the frame).

#include "perception_stage.h"
#include "sam2_processor.h"

#include <chrono>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace rc
{

class Sam2Stage : public Stage
{
public:
    // `refine_labels` (empty → all) filters which SegDetection labels get a SAM2 pass. `decimation`
    // runs the model every Nth enabled frame (held-last reused between). `metrics_log` writes a per-object
    // refinement-quality CSV (etc/sam2_refine_metrics.csv) comparing the YOLO vs SAM2 mask.
    Sam2Stage(const rc::sam2::Config& cfg, std::vector<std::string> refine_labels, int decimation,
              bool metrics_log = false);

    const char* name() const override { return "sam2"; }
    bool ready() const override { return refiner_ && refiner_->ready(); }
    void run(const PerceptionFrame& in, PerceptionResult& out) override;

    rc::sam2::Sam2Refiner* refiner() { return refiner_.get(); }

private:
    // Compare each YOLO mask against its SAM2 refinement using the frame depth and append a CSV row per
    // object; the key column is the depth-BLEED drop (in-mask pixels whose depth is off the object).
    void log_metrics(const PerceptionFrame& in,
                     const std::vector<SegDetection>& yolo,
                     const std::vector<SegDetection>& sam2);

    std::unique_ptr<rc::sam2::Sam2Refiner> refiner_;
    std::vector<std::string>               refine_labels_;
    int                                    decimation_ = 1;
    std::uint64_t                          counter_ = 0;
    std::optional<std::vector<SegDetection>> last_refined_;   // held-last between decimated runs

    // metrics logging
    bool                                          metrics_log_ = false;
    std::ofstream                                 metrics_csv_;
    std::chrono::steady_clock::time_point         metrics_t0_{};
    double                                        sum_yolo_mad_ = 0.0, sum_sam2_mad_ = 0.0;
    double                                        sum_yolo_far_ = 0.0, sum_sam2_far_ = 0.0;
    long                                          n_rows_ = 0;
};

} // namespace rc
