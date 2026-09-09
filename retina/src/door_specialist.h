/*
 * door_specialist.h — a second opinion on doors, asked only when the cheap channel has nothing to say.
 *
 * WHY. The ADE20K semantic path is the only source of `door` masks, and measured over an approach on
 * 2026-09-08/09 its posterior collapses as the robot closes: median P(door) 0.995 at 5.5-6 m, 0.450 at
 * 4.0-4.5 m, 0.110 at 3.5-4.0 m, 0.048 at 2.0-2.5 m, with the mask present on 100%, 42%, 12% and 2% of
 * frames respectively. Below ~3.5 m the classifier does not merely hesitate — it confidently calls a
 * plainly visible closed door `wall` (0.676 against 0.265). No reweighting of that output recovers the
 * door, which is why the contour check built against the same posterior (door_semantic_field.h) holds the
 * 4-5 m band and cannot reach inside it. Only an INDEPENDENT model can.
 *
 * ★WHY THIS IS NOT THE TOP-DOWN LOOP THAT WAS REMOVED. The belief never sees this model's input and never
 * edits its output; the only thing conditioned on the belief state is WHETHER THE QUESTION IS ASKED. The
 * specialist runs unmodified and is free to DENY. That is the line the removed table loop crossed — there
 * the belief suppressed evidence before the estimator saw it, so it could never be contradicted — and it
 * is why the decouple plan permits top-down strictly as "a gain on the budget".
 *
 * ★AND THE DENIAL MUST COST AS MUCH AS THE CONFIRMATION. Run it only where a door is expected and act
 * only on its confirmations and this becomes circular: a test that can only ever help. Whatever consumes
 * this must integrate "the specialist looked here and found nothing" with the same weight it gives
 * "the specialist found one". This class therefore reports `ran` separately from `detections`: a caller
 * must be able to tell "not asked" from "asked and denied", and those two are the whole point.
 *
 * Cost. The trigger is "the cheap channel produced no door mask", which measured 36.8% duty over a 40.9
 * min tour; decimated to roughly 1.5-2 Hz inside each silent run it is ~10-12% duty with the first call
 * within 0.62 s — well inside door_concept's ~1.6 s removal budget. ⚠The per-call cost on THIS box with
 * TRT/CUDA is unmeasured; the 0.28 s/frame in the offline study was CPU-only.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace Ort { class Env; class Session; class SessionOptions; }

namespace rc::doors
{

struct DoorDetection
{
    cv::Rect    bbox;        // in ORIGINAL frame pixels (letterbox undone)
    float       confidence = 0.0f;
    int         class_id   = -1;
    std::string label;       // open_door | closed_door | semi_door
};

class DoorSpecialist
{
public:
    struct Config
    {
        std::string model_path;
        int   input_size = 640;
        bool  use_gpu    = true;
        bool  use_trt    = false;
        // ★A FLOOR, NOT A DECISION. Everything above this is REPORTED with its score; nothing here
        // decides whether a door exists. The offline study measured this model's confidences as modest
        // even when correct (median 0.403, p90 0.648), so a 0.25-style gate would discard most of the
        // evidence it was called to provide. The consumer weights by score; this only bounds the list.
        float score_floor = 0.05f;
    };

    DoorSpecialist() ;
    ~DoorSpecialist();
    DoorSpecialist(const DoorSpecialist&) = delete;
    DoorSpecialist& operator=(const DoorSpecialist&) = delete;

    // Loads the model. Returns false and leaves the object !ready() on any failure — a missing specialist
    // must never take the agent down, it must only mean "no second opinion available".
    bool configure(const Config& cfg);
    [[nodiscard]] bool ready() const noexcept { return session_ != nullptr; }

    // bgr: CV_8UC3. Returns every box above score_floor, in original-frame pixels.
    [[nodiscard]] std::vector<DoorDetection> detect(const cv::Mat& bgr) const;

    [[nodiscard]] const std::vector<std::string>& class_names() const noexcept { return class_names_; }

private:
    Config                       cfg_;
    std::unique_ptr<Ort::Env>            env_;
    std::unique_ptr<Ort::SessionOptions> opts_;
    std::unique_ptr<Ort::Session>        session_;
    std::string                  input_name_, output_name_;
    std::vector<std::string>     class_names_{"open_door", "closed_door", "semi_door"};
};

}   // namespace rc::doors
