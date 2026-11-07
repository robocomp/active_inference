// detector_model.h — Task 3 of the door-existence model: the per-camera DETECTOR model. q = P(this camera's detector reports
// the door from this view | the door is there); f = P(it reports a door here | none is there). Exact Bernoulli hit / miss
// log-likelihood ratios, and the persisted per-camera parameters (etc/door_detectors.csv).
//
// ★Every number in default_detectors() is a SENSOR-MODEL CONSTANT, to be refit from data by tools/fit_detector_envelopes.py
// (no per-view log exists yet; see the Task 3 report). The shoulders are rc::detect::DetectorEnvelope's own parameters.
#pragma once
#include <algorithm>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <locale>
#include <sstream>
#include <string>
#include <vector>
#include "door_visibility.h"
#include "detectability/detectability.h"

namespace rc::door::core
{
struct DetectorModel
{
    rc::detect::DetectorEnvelope env;   // shoulders, per axis (fill_h and fill_v)
    float hit_rate = 1.0f;              // P(report | present, ideally framed, face-on)
    float false_alarm = 0.0f;           // P(report | absent), per view at full visibility
    float n_hit_trials = 0.0f;          // evidence behind hit_rate (0 = a compiled default)
};

struct DetectorTable { DetectorModel zed, ricoh; };

// SENSOR-MODEL CONSTANTS (refit from data). ZED: today's aperture_envelope shoulders (DoorModel.Detect{Min,Max}Fill/Soft),
// hit 0.9 / clutter 0.02 are placeholders for the ZED (no ZED hit-rate was ever measured per view).
// Ricoh: today's birth model rc::birth::Detectability{periph, 2.5 m, power 2} weights a look by min(1,(2.5/range)^2), i.e. it
// falls under 2.5 m-equivalent framing and never has a near cut-off. In fill units (door 2.05 m tall, 1.4 m <-> fill_v 0.386)
// the 2.5 m knee is fill_v ~ 0.216, so min_fill = 0.216 with the same soft; max_fill = 1.5 (nothing overflows a strip).
// hit_rate 0.5 and clutter 0.05 are the ricoh's current values.
[[nodiscard]] inline DetectorTable default_detectors()
{
    DetectorTable t;
    t.zed   = {{0.10f, 0.60f, 0.06f}, 0.9f, 0.02f, 0.0f};
    t.ricoh = {{0.216f, 1.5f, 0.06f}, 0.5f, 0.05f, 0.0f};
    return t;
}

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

/// Presence: a * ln(q / (f*vis)); f is the false-alarm rate (the caller's model.false_alarm), vis the view's visibility.
/// 0 when vis*q == 0. (f*vis == 0 with q > 0 is an unbounded ratio; callers keep false_alarm > 0.)
[[nodiscard]] inline float hit_llr(float q, float f, float vis, float assoc_w)
{
    if (vis * q == 0.0f)
        return 0.0f;
    return assoc_w * std::log(q / (f * vis));
}

/// Absence: ln((1 - q) / (1 - f*vis)). Falls as q rises; exactly 0 when q == 0 and f*vis == 0 (an unseen door says nothing).
[[nodiscard]] inline float miss_llr(float q, float f, float vis)
{
    return std::log((1.0f - q) / (1.0f - f * vis));
}

// CSV: camera,param,value,n,source — params min_fill,max_fill,soft,hit_rate (n = n_hit_trials),false_alarm.
[[nodiscard]] inline DetectorTable load_detectors(const std::filesystem::path& p)
{
    DetectorTable t = default_detectors();
    std::ifstream f(p);
    if (not f)
        return t;
    std::string line;
    std::getline(f, line);   // header
    while (std::getline(f, line))
    {
        std::vector<std::string> c;
        std::size_t s = 0;
        for (std::size_t i = 0; i <= line.size(); ++i)
            if (i == line.size() or line[i] == ',') { c.push_back(line.substr(s, i - s)); s = i + 1; }
        if (c.size() < 4) continue;
        const auto num = [](const std::string& x, float& out) {
            return std::from_chars(x.data(), x.data() + x.size(), out).ec == std::errc{};
        };
        float v = 0, n = 0;
        if (not num(c[2], v) or not num(c[3], n)) continue;
        DetectorModel* m = c[0] == "zed" ? &t.zed : c[0] == "ricoh" ? &t.ricoh : nullptr;
        if (m == nullptr) continue;
        if (c[1] == "min_fill") m->env.min_fill = v;
        else if (c[1] == "max_fill") m->env.max_fill = v;
        else if (c[1] == "soft") m->env.soft = v;
        else if (c[1] == "hit_rate") { m->hit_rate = v; m->n_hit_trials = n; }
        else if (c[1] == "false_alarm") m->false_alarm = v;
    }
    return t;
}

inline void save_detectors(const std::filesystem::path& p, const DetectorTable& t, const std::string& source = "compiled_defaults")
{
    std::ofstream f(p);
    f.imbue(std::locale::classic());
    f << "camera,param,value,n,source\n";
    const auto emit = [&](const char* cam, const DetectorModel& m) {
        f << std::format("{},min_fill,{},0,{}\n{},max_fill,{},0,{}\n{},soft,{},0,{}\n{},hit_rate,{},{},{}\n{},false_alarm,{},0,{}\n",
                         cam, m.env.min_fill, source, cam, m.env.max_fill, source, cam, m.env.soft, source,
                         cam, m.hit_rate, m.n_hit_trials, source, cam, m.false_alarm, source);
    };
    emit("zed", t.zed);
    emit("ricoh", t.ricoh);
}
}  // namespace rc::door::core
