// detector_llr.h — split from detector_model.h (Task 8: view_ledger.h is included by door_fit_state.h, which door_visibility.h
// includes, so the model + likelihoods may not include door_visibility.h). Task 3 of the door-existence model: the per-camera DETECTOR model. q = P(this camera's detector reports
// the door from this view | the door is there); f = P(it reports a door here | none is there). Exact Bernoulli hit / miss
// log-likelihood ratios, and the persisted per-camera parameters (etc/door_detectors.csv).
//
// ★Every number in default_detectors() is a SENSOR-MODEL CONSTANT, to be refit from data by tools/fit_detector_envelopes.py
// (no per-view log exists yet; see the Task 3 report). The shoulders are rc::detect::DetectorEnvelope's own parameters.
#pragma once
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <locale>
#include <sstream>
#include <string>
#include <vector>
#include "../../../common/detectability/detectability.h"

namespace rc::door::core
{
struct DetectorModel
{
    rc::detect::DetectorEnvelope env;   // shoulders, per axis (fill_h and fill_v)
    float hit_rate = 1.0f;              // P(report | present, ideally framed, face-on)
    float false_alarm = 0.0f;           // P(report | absent), per view at full visibility
    float n_hit_trials = 0.0f;          // evidence behind hit_rate (0 = a compiled default)
    float n_fa_trials = 0.0f;           // evidence behind false_alarm (Σ front-end factor of removed places' views; 0 = default)
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

/// The false-alarm rate AT THIS VIEW: a false alarm passes through the detector's own front end exactly as the door does
/// (envelope, aspect, visibility, kappa), so f_eff = false_alarm * q / hit_rate = false_alarm * env * aspect * vis * kappa.
/// 0 when q == 0 (a view the detector cannot report from can neither confirm nor refute).
[[nodiscard]] inline float false_alarm_eff(const DetectorModel& m, float q)
{
    return m.hit_rate > 0.0f ? m.false_alarm * q / m.hit_rate : 0.0f;
}

/// Presence: a * ln(hit_rate / false_alarm) when q > 0 (q and f_eff share the front end, so it cancels in the ratio q / f_eff);
/// exactly 0 when q == 0 (nothing was reportable from this view). false_alarm must be > 0.
[[nodiscard]] inline float hit_llr(float q, const DetectorModel& m, float assoc_w)
{
    return q > 0.0f ? assoc_w * std::log(m.hit_rate / m.false_alarm) : 0.0f;
}

/// Absence: ln((1 - q) / (1 - f_eff)), f_eff = false_alarm_eff(m, q) (or a per-view mean from a ledger). Falls as q rises;
/// exactly 0 when q == 0.
[[nodiscard]] inline float miss_llr(float q, float f_eff)
{
    return q > 0.0f ? std::log((1.0f - q) / (1.0f - f_eff)) : 0.0f;
}

// CSV: camera,param,value,n,source — params min_fill,max_fill,soft,hit_rate (n = n_hit_trials),false_alarm (n = n_fa_trials).
// from_chars must consume the WHOLE field; a malformed row keeps the default for that param and is counted (one warning line).
// Domain validation (a probability must be a probability, NOT a model threshold): hit_llr divides by false_alarm, f_eff by
// hit_rate and miss_llr takes log(1 - q), so a loaded camera must satisfy 0 < false_alarm < hit_rate < 1 and a valid envelope;
// otherwise that camera falls back to its compiled defaults with a warning. kappa is in [0,1] by construction
// (pose_overlap_factor), so q <= hit_rate < 1 and 1 - q > 0.
[[nodiscard]] inline bool detector_in_domain(const DetectorModel& m)
{
    return m.false_alarm > 0.0f and m.false_alarm < m.hit_rate and m.hit_rate < 1.0f and m.env.valid()
           and std::isfinite(m.env.min_fill) and std::isfinite(m.env.max_fill) and std::isfinite(m.env.soft)
           and std::isfinite(m.hit_rate) and std::isfinite(m.false_alarm);
}

[[nodiscard]] inline DetectorTable load_detectors(const std::filesystem::path& p)
{
    const DetectorTable defaults = default_detectors();
    DetectorTable t = defaults;
    std::ifstream f(p);
    if (not f)
        return t;
    std::string line;
    std::getline(f, line);   // header
    int malformed = 0;
    while (std::getline(f, line))
    {
        std::vector<std::string> c;
        std::size_t s = 0;
        for (std::size_t i = 0; i <= line.size(); ++i)
            if (i == line.size() or line[i] == ',') { c.push_back(line.substr(s, i - s)); s = i + 1; }
        if (line.empty()) continue;
        const auto num = [](const std::string& x, float& out) {
            const char* e = x.data() + x.size();
            const auto r = std::from_chars(x.data(), e, out);
            return r.ec == std::errc{} and r.ptr == e;
        };
        float v = 0, n = 0;
        if (c.size() < 4 or not num(c[2], v) or not num(c[3], n)) { ++malformed; continue; }
        DetectorModel* m = c[0] == "zed" ? &t.zed : c[0] == "ricoh" ? &t.ricoh : nullptr;
        if (m == nullptr) { ++malformed; continue; }
        if (c[1] == "min_fill") m->env.min_fill = v;
        else if (c[1] == "max_fill") m->env.max_fill = v;
        else if (c[1] == "soft") m->env.soft = v;
        else if (c[1] == "hit_rate") { m->hit_rate = v; m->n_hit_trials = n; }
        else if (c[1] == "false_alarm") { m->false_alarm = v; m->n_fa_trials = n; }
        else ++malformed;
    }
    if (malformed > 0)
        std::fprintf(stderr, "[detector_model] %s: %d malformed row(s) ignored (defaults kept for those params)\n",
                     p.string().c_str(), malformed);
    if (not detector_in_domain(t.zed))
    {
        std::fprintf(stderr, "[detector_model] %s: zed out of domain (need 0 < false_alarm < hit_rate < 1): compiled defaults\n", p.string().c_str());
        t.zed = defaults.zed;
    }
    if (not detector_in_domain(t.ricoh))
    {
        std::fprintf(stderr, "[detector_model] %s: ricoh out of domain (need 0 < false_alarm < hit_rate < 1): compiled defaults\n", p.string().c_str());
        t.ricoh = defaults.ricoh;
    }
    return t;
}

inline void save_detectors(const std::filesystem::path& p, const DetectorTable& t, const std::string& source = "compiled_defaults")
{
    std::ofstream f(p);
    f.imbue(std::locale::classic());
    f << "camera,param,value,n,source\n";
    const auto write_cam = [&](const char* cam, const DetectorModel& m) {   // (not "emit": Qt #defines it)
        f << std::format("{},min_fill,{},0,{}\n{},max_fill,{},0,{}\n{},soft,{},0,{}\n{},hit_rate,{},{},{}\n{},false_alarm,{},{},{}\n",
                         cam, m.env.min_fill, source, cam, m.env.max_fill, source, cam, m.env.soft, source,
                         cam, m.hit_rate, m.n_hit_trials, source, cam, m.false_alarm, m.n_fa_trials, source);
    };
    write_cam("zed", t.zed);
    write_cam("ricoh", t.ricoh);
}
}  // namespace rc::door::core
