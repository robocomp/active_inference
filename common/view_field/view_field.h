/*
 * view_field.h — a LEARNT belief about the sensor, keyed on (world cell x view bearing x label).
 *
 * ★WHAT IT IS FOR. The existence channel is a log-odds ratio built from two numbers:
 *     llr_occ  = log(pd / pc)          a detection confirms this much
 *     llr_free = log((1-pd) / (1-pc))  a miss refutes this much
 * and BOTH have been configured constants since the beginning: detection_prob = 0.85, clutter_prob = 0.05
 * in every agent. pd too high deletes real objects; pc too low invents them. They are the same defect with
 * opposite sign, and neither has ever been measured against the place it was applied.
 *
 * ★WHY (world cell x bearing) AND NOT ROBOT POSE. A false alarm is a CLASSIFIER failure, and classifier
 * failures are viewpoint-dependent: the same patch of scene fools the detector from one direction and not
 * another. Keyed on the ROBOT POSE the field cannot generalise — two positions that view the same spot the
 * same way would learn separately. Keyed on PLACE alone it degenerates into "nothing may be born here" and
 * would suppress a genuine object later put at that spot, seen from any direction. The pair is the thing
 * that is actually stable. (This keying was settled in a4145c2, which is also why PhantomEvent already
 * carries view_bearing: the recording half of this was built before the field was.)
 *
 * ★ONE CLASS, TWO INSTANCES — the symmetry is the point.
 *     p_FA     ← supervised by CONFIDENT DISCONFIRMATIONS. Rare, retrospective: the robot comes back,
 *                looks closely, and the object is denied. That death retro-labels the detections that
 *                gave birth to it as clutter.
 *     p_detect ← supervised by EVERY IN-VIEW CYCLE of a live object. Dense and immediate: a Bernoulli
 *                trial per frame, already recorded by common/detect_probe.
 * Same key, same estimator, same decay; only the event that feeds them differs.
 *
 * ★THE GEOMETRIC MODEL REMAINS THE PRIOR — this does not replace common/detectability, it corrects it
 * where it is wrong. estimate() takes the prior mean from the caller (the envelope for pd, the configured
 * constant for pc), so a cell never visited behaves EXACTLY as today and the field is a learnt residual
 * on top. That is also the honest answer to "should we re-fit the envelope": measured 2026-08-17 the
 * envelope's low end is wrong by ~7x and its empirical curve is not even monotonic in fill, so the missing
 * information is not in its three parameters. A field keyed on place-and-direction can capture "from here,
 * looking that way, this is hard to see" WITHOUT having to name the cause (obliquity, an occluder, glare).
 *
 * ★CONFIDENCE IS BOUNDED ON PURPOSE. Counts are exponentially forgotten, so the total pseudo-count
 * saturates at 1/(1-forget). The field can never become certain, old evidence fades, and a cell that has
 * changed (furniture moved, lighting different) recovers instead of being condemned forever.
 *
 * ★★THE HAZARD THIS FILE CANNOT FIX BY ITSELF — READ BEFORE COUPLING IT TO ANYTHING ELSE.
 * A field learnt from the actor's own decisions can lock in its own mistakes: if p_FA rises in a cell,
 * detections there confirm less, so objects there are less likely to survive, so the evidence that would
 * correct the field stops being generated. The mitigation is NOT in the estimator — it is that UNCERTAINTY
 * must RAISE the drive to go and look. A cell the field is unsure about should become an epistemic target,
 * not a write-off. confidence() is exposed for exactly that consumer; it is not wired yet, and until it is,
 * this field must only ever shrink a likelihood ratio and must NEVER gate a birth.
 *
 * Header-only, no DSR, no Qt. Locale-safe on both ends (from_chars to read, classic locale to write) —
 * these machines run es_ES and this file persists across runs, so it round-trips its own data. See CLAUDE.md.
 */

#pragma once

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <locale>
#include <string>
#include <string_view>
#include <unordered_map>

namespace rc::field
{

struct Params
{
    // ★CELL SIZE is bracketed by physics, not taste: it must be LARGER than the localisation noise that
    // smears the same spot across cells (pose sigma is a few cm) and SMALLER than the objects it has to
    // tell apart (0.3-2 m). 0.25 m sits an order of magnitude clear of both ends.
    float cell_m = 0.25f;
    // ★BEARING BINS: a classifier's behaviour varies over tens of degrees, not degrees — a 1-degree bin
    // would never accumulate two samples in the same key, and a 90-degree bin would average a front view
    // with a side view. 12 bins = 30 degrees.
    int   bearing_bins = 12;
    // Pseudo-count of the prior. 20 ⇒ a single event moves an estimate from 0.05 to ~0.09: enough to
    // register, far too little to act on alone.
    float prior_strength = 20.0f;
    // Exponential forgetting per observation. 0.995 ⇒ an effective window of ~200 observations and a total
    // pseudo-count that saturates there, which is what BOUNDS confidence (see the header note).
    float forget = 0.995f;
};

class ViewField
{
public:
    explicit ViewField(Params p = {}) : p_(p) {}

    // One observation. `positive` means the event this field counts happened (a confirmed false alarm for
    // p_FA; a successful detection for p_detect).
    //
    // ★WEIGHT, NOT A CUTOFF. "Did this event really tell us something?" is a continuous question and it
    // belongs in the likelihood, not in an `if`. A death whose killing look could not have resolved the
    // object (p_detect -> 0) is not weak evidence of a false alarm — it is NO evidence, and it must
    // contribute nothing rather than be excluded by a hand-placed threshold. Passing p_detect * in_fov_frac
    // as the weight makes that fall out: the useless event lands with weight 0 on its own.
    // ★This is the same discipline the existence channel already uses (ΔL = p_vis * log-ratio) and the
    // reason it matters here is sharper: the events that would be thrown away by a threshold are exactly
    // OUR OWN removal bugs, and a field that learnt from those would be recording our failures as the
    // detector's.
    void observe(std::string_view label, float x, float y, float view_bearing_rad, bool positive,
                 float weight = 1.0f)
    {
        const float w = std::clamp(weight, 0.0f, 1.0f);
        if (w <= 0.0f)
            return;                        // told us nothing — not even a fractional count
        auto& c = cells_[key(label, x, y, view_bearing_rad)];
        c.a *= p_.forget;
        c.b *= p_.forget;
        (positive ? c.a : c.b) += w;
    }

    // Posterior mean, shrunk toward `prior_mean`. An unvisited cell returns prior_mean EXACTLY, so a fleet
    // with an empty field behaves precisely as it did before the field existed.
    [[nodiscard]] float estimate(std::string_view label, float x, float y, float view_bearing_rad,
                                 float prior_mean) const
    {
        const auto it = cells_.find(key(label, x, y, view_bearing_rad));
        if (it == cells_.end())
            return prior_mean;
        const float k = p_.prior_strength;
        return std::clamp((k * prior_mean + it->second.a) / (k + it->second.a + it->second.b),
                          1e-3f, 1.0f - 1e-3f);
    }

    // How much this cell's estimate rests on evidence rather than on the prior, in [0,1). The intended
    // consumer is the epistemic drive: LOW confidence should attract a verifying look. Never use it to
    // gate a birth — see the hazard note in the header.
    [[nodiscard]] float confidence(std::string_view label, float x, float y, float view_bearing_rad) const
    {
        const auto it = cells_.find(key(label, x, y, view_bearing_rad));
        if (it == cells_.end())
            return 0.0f;
        const float n = it->second.a + it->second.b;
        return n / (n + p_.prior_strength);
    }

    // The field's own bearing discretisation, exposed so a caller can ask "is this a DIFFERENT direction?"
    // in exactly the units the field is keyed on — rather than inventing a second angle somewhere else.
    [[nodiscard]] int bearing_bin(float bearing_rad) const
    {
        const float two_pi = 2.0f * static_cast<float>(M_PI);
        float b = std::fmod(bearing_rad, two_pi);
        if (b < 0.0f) b += two_pi;
        return std::clamp(static_cast<int>(b / two_pi * static_cast<float>(p_.bearing_bins)),
                          0, p_.bearing_bins - 1);
    }

    [[nodiscard]] std::size_t size() const { return cells_.size(); }

    // ── Persistence. The whole value of this field is that it OUTLIVES the run that learnt it. ──
    bool save(const std::string& path) const
    {
        std::ofstream f(path, std::ios::out | std::ios::trunc);
        if (not f.is_open())
            return false;
        f.imbue(std::locale::classic());   // never emit a comma decimal separator (CLAUDE.md)
        f << "label,cell_x,cell_y,bearing_bin,alpha,beta\n";
        for (const auto& [k, c] : cells_)
            f << k.label << ',' << k.cx << ',' << k.cy << ',' << k.bb << ','
              << c.a << ',' << c.b << '\n';
        return true;
    }

    bool load(const std::string& path)
    {
        std::ifstream f(path);
        if (not f.is_open())
            return false;
        std::string line;
        std::getline(f, line);            // header
        while (std::getline(f, line))
        {
            std::string_view s{line};
            std::string_view fld[6];
            std::size_t n = 0, start = 0;
            for (std::size_t i = 0; i <= s.size() and n < 6; ++i)
                if (i == s.size() or s[i] == ',')
                { fld[n++] = s.substr(start, i - start); start = i + 1; }
            if (n < 6) continue;
            Key k;
            k.label = std::string(fld[0]);
            // ★from_chars ONLY. strtof/atof read through LC_NUMERIC and these machines run es_ES, where
            // they stop dead at the '.' of a value THIS CLASS WROTE with a point. Silent, no error flag.
            if (not num(fld[1], k.cx) or not num(fld[2], k.cy) or not num(fld[3], k.bb)) continue;
            Cell c;
            if (not num(fld[4], c.a) or not num(fld[5], c.b)) continue;
            cells_[k] = c;
        }
        return true;
    }

private:
    struct Key
    {
        std::string label;
        int cx = 0, cy = 0, bb = 0;
        bool operator==(const Key& o) const
        { return cx == o.cx and cy == o.cy and bb == o.bb and label == o.label; }
    };
    struct KeyHash
    {
        std::size_t operator()(const Key& k) const
        {
            std::size_t h = std::hash<std::string>{}(k.label);
            const auto mix = [&h](int v)
            { h ^= std::hash<int>{}(v) + 0x9e3779b9u + (h << 6) + (h >> 2); };
            mix(k.cx); mix(k.cy); mix(k.bb);
            return h;
        }
    };
    struct Cell { float a = 0.0f, b = 0.0f; };

    [[nodiscard]] Key key(std::string_view label, float x, float y, float bearing) const
    {
        const float two_pi = 2.0f * static_cast<float>(M_PI);
        float b = std::fmod(bearing, two_pi);
        if (b < 0.0f) b += two_pi;
        Key k;
        k.label = std::string(label);
        k.cx = static_cast<int>(std::floor(x / p_.cell_m));
        k.cy = static_cast<int>(std::floor(y / p_.cell_m));
        k.bb = std::clamp(static_cast<int>(b / two_pi * static_cast<float>(p_.bearing_bins)),
                          0, p_.bearing_bins - 1);
        return k;
    }
    static bool num(std::string_view s, int& out)
    { return std::from_chars(s.data(), s.data() + s.size(), out).ec == std::errc{}; }
    static bool num(std::string_view s, float& out)
    { return std::from_chars(s.data(), s.data() + s.size(), out).ec == std::errc{}; }

    Params p_;
    std::unordered_map<Key, Cell, KeyHash> cells_;
};

}   // namespace rc::field
