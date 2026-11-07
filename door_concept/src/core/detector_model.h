// detector_model.h — Task 3 of the door-existence model: the per-camera DETECTOR model. q = P(this camera's detector reports
// the door from this view | the door is there); f = P(it reports a door here | none is there). Exact Bernoulli hit / miss
// log-likelihood ratios, and the persisted per-camera parameters (etc/door_detectors.csv).
//
// ★Every number in default_detectors() is a SENSOR-MODEL CONSTANT, to be refit from data by tools/fit_detector_envelopes.py
// (no per-view log exists yet; see the Task 3 report). The shoulders are rc::detect::DetectorEnvelope's own parameters.
#pragma once
#include "detector_llr.h"
#include "door_visibility.h"

namespace rc::door::core
{
/// q_c = env(fill_h) * env(fill_v) * aspect * vis * kappa * hit_rate. Each axis uses the envelope's full unimodal form at that fill.
/// Not projectable or vis == 0 gives exactly 0.
[[nodiscard]] inline float detect_prob(const DetectorModel& m, const ViewGeometry& g, float kappa)
{
    if (not g.projectable or g.vis <= 0.0f)
        return 0.0f;
    const float eh = rc::detect::p_detect(g.fill_h, 1.0f, m.env);
    const float ev = rc::detect::p_detect(g.fill_v, 1.0f, m.env);
    return eh * ev * g.aspect * g.vis * kappa * m.hit_rate;
}

}  // namespace rc::door::core
