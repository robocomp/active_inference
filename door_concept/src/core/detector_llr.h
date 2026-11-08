// detector_llr.h — split from detector_model.h (Task 8: view_ledger.h is included by door_fit_state.h, which door_visibility.h
// includes, so the model + likelihoods may not include door_visibility.h). Task 3 of the door-existence model: the per-camera DETECTOR model. q = P(this camera's detector reports
// the door from this view | the door is there); f = P(it reports a door here | none is there). Exact Bernoulli hit / miss
// log-likelihood ratios, and the persisted per-camera parameters (etc/door_detectors.csv).
//
// ★Every number in default_detectors() is a SENSOR-MODEL CONSTANT, to be refit from data by tools/fit_detector_envelopes.py
// (no per-view log exists yet; see the Task 3 report). The shoulders are rc::detect::DetectorEnvelope's own parameters.
#pragma once
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>
#include <locale>
#include <sstream>
#include <string>
#include <utility>
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
};

// ═══ THE FIT'S OWN SUPPORT (door-existence model, channel `sup`) ═══════════════════════════════════════════════════════════════
// x = the fraction of the door's own support points this cycle's fit explains as CLUTTER (DoorBelief::clutter_fraction). What a
// fit says about the place it landed on: a fit that explains none of its support did not land on the door the support came from.
// ℓ_sup(x) = ln p(x | a door at the fitted place) − ln p(x | no door there), two fixed densities on [0,1], loaded with the detector
// rates (door_detectors.csv, camera `support`) and fixed within the session; refit offline.
//   p(x | H) = (1 − o) Beta(x; a, b) + o          (o = the uniform outlier weight of H; uniform on [0,1] has density 1)
// The outlier weight is what keeps the term finite on the whole closed interval with no cutoff on x: a Beta alone sends ℓ to ±∞ at
// the ends, and x reaches them (a mask that lands nowhere on the model gives exactly 1). Physically: a real door now and then has a
// fit that explains nothing (o_present, small), and a place with no door often has masks a fit explains anyway (o_absent, large).
// With present (a = 1, decreasing) and absent (b = 1, increasing) the ratio is monotone decreasing in x, continuous everywhere.
// ★SENSOR-MODEL CONSTANTS, flagged (same standing as hit_rate/false_alarm): present_b = 4 (mass at x = 0.05-0.3), absent_a = 12
// (mass near 1), o_present = 0.002, o_absent = 0.8. From the 34 first fits of the comparison tours: x >= 0.9 → median 1.65 m from
// the detection, x < 0.9 → median 0.11 m, no overlap. No per-view fit yet (tools/fit_detector_envelopes.py carries the rows through).
struct SupportModel
{
    float present_a = 1.0f, present_b = 4.0f, present_outlier = 0.002f;
    float absent_a = 12.0f, absent_b = 1.0f, absent_outlier = 0.8f;
    float n_trials = 0.0f;      // evidence behind the values (0 = compiled defaults)
};

/// The density of a uniform-outlier Beta at x (x is the mean of per-point responsibilities, so it is in [0,1] by construction;
/// the clamp is the domain guard against float rounding of that mean, not a model threshold).
[[nodiscard]] inline double support_density(float x, float a, float b, float outlier)
{
    const double t = std::clamp(static_cast<double>(x), 0.0, 1.0);   // clamp: x's domain [0,1], rounding only (not a threshold)
    const double norm = std::lgamma(static_cast<double>(a) + b) - std::lgamma(static_cast<double>(a)) - std::lgamma(static_cast<double>(b));
    const double beta = std::exp(norm) * std::pow(t, a - 1.0) * std::pow(1.0 - t, b - 1.0);   // pow(0, 0) == 1 (a or b == 1)
    return (1.0 - outlier) * beta + outlier;
}

/// ℓ_sup(x) = ln p(x | present) − ln p(x | absent). Finite on [0,1] for any model in domain (support_in_domain).
[[nodiscard]] inline float support_llr(float x, const SupportModel& m)
{
    return static_cast<float>(std::log(support_density(x, m.present_a, m.present_b, m.present_outlier))
                              - std::log(support_density(x, m.absent_a, m.absent_b, m.absent_outlier)));
}

/// A probability must be a probability and the densities finite on [0,1] (a, b >= 1): domain, not a model threshold.
[[nodiscard]] inline bool support_in_domain(const SupportModel& m)
{
    return std::isfinite(m.present_a) and std::isfinite(m.present_b) and std::isfinite(m.absent_a) and std::isfinite(m.absent_b)
           and std::isfinite(m.present_outlier) and std::isfinite(m.absent_outlier)
           and m.present_a >= 1.0f and m.present_b >= 1.0f and m.absent_a >= 1.0f and m.absent_b >= 1.0f
           and m.present_outlier > 0.0f and m.present_outlier < 1.0f and m.absent_outlier > 0.0f and m.absent_outlier < 1.0f;
}

struct DetectorTable { DetectorModel zed, ricoh; SupportModel support; };

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

// CSV: camera,param,value,n,source — params min_fill,max_fill,soft,hit_rate (n = n_hit_trials),false_alarm.
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
    int support_seen = 0;   // bit per support param; all six must be present
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
        // n: the evidence behind the value (views, written by tools/fit_detector_envelopes.py; 0 = a compiled default). A negative
        // or non-finite count is not a count: the row is malformed.
        if (c.size() < 4 or not num(c[2], v) or not num(c[3], n) or not std::isfinite(n) or n < 0.0f) { ++malformed; continue; }
        if (c[0] == "support")
        {
            const std::array<std::pair<const char*, float SupportModel::*>, 6> fields{{
                {"present_a", &SupportModel::present_a}, {"present_b", &SupportModel::present_b}, {"present_outlier", &SupportModel::present_outlier},
                {"absent_a", &SupportModel::absent_a}, {"absent_b", &SupportModel::absent_b}, {"absent_outlier", &SupportModel::absent_outlier}}};
            bool found = false;
            for (std::size_t i = 0; i < fields.size(); ++i)
                if (c[1] == fields[i].first) { t.support.*(fields[i].second) = v; t.support.n_trials = n; support_seen |= 1 << i; found = true; }
            if (not found) ++malformed;
            continue;
        }
        DetectorModel* m = c[0] == "zed" ? &t.zed : c[0] == "ricoh" ? &t.ricoh : nullptr;
        if (m == nullptr) { ++malformed; continue; }
        if (c[1] == "min_fill") m->env.min_fill = v;
        else if (c[1] == "max_fill") m->env.max_fill = v;
        else if (c[1] == "soft") m->env.soft = v;
        else if (c[1] == "hit_rate") { m->hit_rate = v; m->n_hit_trials = n; }
        else if (c[1] == "false_alarm") m->false_alarm = v;
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
    if (support_seen != 0b111111 or not support_in_domain(t.support))
    {
        std::fprintf(stderr, "[detector_model] %s: support rows %s: compiled defaults (need present_a/b, present_outlier, absent_a/b, absent_outlier)\n",
                     p.string().c_str(), support_seen != 0b111111 ? "missing" : "out of domain");
        t.support = defaults.support;
    }
    return t;
}

inline void save_detectors(const std::filesystem::path& p, const DetectorTable& t, const std::string& source = "compiled_defaults")
{
    std::ofstream f(p);
    f.imbue(std::locale::classic());
    f << "camera,param,value,n,source\n";
    const auto write_cam = [&](const char* cam, const DetectorModel& m) {   // (not "emit": Qt #defines it)
        f << std::format("{},min_fill,{},0,{}\n{},max_fill,{},0,{}\n{},soft,{},0,{}\n{},hit_rate,{},{},{}\n{},false_alarm,{},0,{}\n",
                         cam, m.env.min_fill, source, cam, m.env.max_fill, source, cam, m.env.soft, source,
                         cam, m.hit_rate, m.n_hit_trials, source, cam, m.false_alarm, source);
    };
    write_cam("zed", t.zed);
    write_cam("ricoh", t.ricoh);
    const auto& s = t.support;
    f << std::format("support,present_a,{},{},{}\nsupport,present_b,{},{},{}\nsupport,present_outlier,{},{},{}\n"
                     "support,absent_a,{},{},{}\nsupport,absent_b,{},{},{}\nsupport,absent_outlier,{},{},{}\n",
                     s.present_a, s.n_trials, source, s.present_b, s.n_trials, source, s.present_outlier, s.n_trials, source,
                     s.absent_a, s.n_trials, source, s.absent_b, s.n_trials, source, s.absent_outlier, s.n_trials, source);
}
}  // namespace rc::door::core
