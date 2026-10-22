/*
 * blind_spot_map.h — a LEARNED P(detect | object, viewpoint), the detector's blind spots. Shared, header-only.
 *
 * The geometric envelope (detectability.h) is the prior; this map corrects it with what the robot actually
 * saw. Labels come from the classifier-free contour channel (the image proves the object is there), never
 * from the existence belief. Frames are pooled into VISITS so one pose stared at for 1000 cycles counts as one
 * observation. Kernel Beta-Bernoulli, two levels (class from other objects, instance from its own samples).
 * Design and every constant: BLIND_SPOT_MAP.md (same directory).
 *
 * Pure std + <charconv>; no graph, no camera — the caller computes the viewpoint.
 */
#pragma once

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <locale>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rc::detect
{

// Where the camera is relative to the object (BLIND_SPOT_MAP.md §2).
struct Viewpoint
{
    float az = 0.0f;   // bearing around the object, object frame, folded by its symmetry (rad)
    float el = 0.0f;   // elevation of the camera above the object's top (rad)
    float lr = 0.0f;   // log(camera→object distance in metres)
};

struct BlindSpotParams
{
    float ell_az = 0.30f, ell_el = 0.12f, ell_lr = 0.25f;   // kernel length scales
    float kappa0 = 1.0f;                                    // geometric prior strength (observations)
    float beta   = 0.5f;                                    // how much the class predicts an individual
    std::size_t max_samples = 20000;                        // oldest dropped beyond this
};

struct BlindSpotAnswer
{
    float p     = 0.0f;   // P(detect | present, visible, viewpoint)
    float n_c   = 0.0f;   // class evidence near this viewpoint (observations, other objects)
    float n_own = 0.0f;   // this object's own evidence near this viewpoint
    float p_c   = 0.0f;   // the class-level answer (the instance's prior)
};

// Fold a bearing by the object's symmetry. A rectangle (C2v: views from ±x alike, ±y alike, mirror images
// alike) folds to [0, π/2] — 0 = looking along its local x axis, π/2 = along y. A round object's bearing
// carries nothing, so it folds to 0.
inline float fold_bearing(float az, bool round)
{
    if (round)
        return 0.0f;
    const float pi = static_cast<float>(M_PI);
    float a = std::fmod(std::abs(az), pi);
    if (a > 0.5f * pi)
        a = pi - a;
    return a;
}

class BlindSpotMap
{
public:
    explicit BlindSpotMap(BlindSpotParams p = {}) : p_(p) {}
    void set_params(const BlindSpotParams& p) { p_ = p; }

    // One labelled frame (BLIND_SPOT_MAP.md §3-4): `hit` = YOLO assigned a mask to the object this frame,
    // `w` = how sure the image is that the object was there and visible (0 ⇒ ignored).
    void observe(std::uint64_t owner, const Viewpoint& v, bool hit, float w)
    {
        if (not (w > 0.0f) or not std::isfinite(w))
            return;
        auto& vis = open_[owner];
        if (vis.w_sum > 0.0f and dist2(vis.anchor, v) > 0.25f)   // moved > ½ kernel length: close the visit
        {
            close(owner, vis);
            vis = {};
        }
        if (vis.w_sum == 0.0f)
            vis.anchor = v;
        vis.w_sum   += w;
        vis.hit_sum += hit ? w : 0.0f;
    }

    // Close an object's open visit (it died, or the caller is about to save).
    void flush(std::uint64_t owner)
    {
        if (const auto it = open_.find(owner); it != open_.end())
        {
            close(owner, it->second);
            open_.erase(it);
        }
    }
    void flush_all()
    {
        for (auto& [owner, vis] : open_)
            close(owner, vis);
        open_.clear();
    }

    // P(detect) at viewpoint v for `owner`, given the geometric prior p_env (BLIND_SPOT_MAP.md §5).
    [[nodiscard]] BlindSpotAnswer query(std::uint64_t owner, const Viewpoint& v, float p_env) const
    {
        p_env = std::clamp(p_env, 0.0f, 1.0f);
        float nc = 0.0f, hc = 0.0f, no = 0.0f, ho = 0.0f;
        const auto add = [&](const Viewpoint& s, float r, float w, std::uint64_t o)
        {
            const float k = std::exp(-0.5f * dist2(v, s)) * w;
            if (o == owner and owner != 0) { no += k; ho += k * r; }
            else                           { nc += k; hc += k * r; }
        };
        for (const auto& s : samples_)
            add(s.v, s.r, s.w, s.owner);
        for (const auto& [o, vis] : open_)
            if (vis.w_sum > 0.0f)
                add(vis.anchor, vis.hit_sum / vis.w_sum, visit_weight(vis.w_sum), o);
        BlindSpotAnswer a;
        a.n_c = nc; a.n_own = no;
        a.p_c = (p_.kappa0 * p_env + hc) / (p_.kappa0 + nc);
        const float kappa = p_.kappa0 + p_.beta * nc;
        a.p = (kappa * a.p_c + ho) / (kappa + no);
        return a;
    }

    [[nodiscard]] std::size_t size() const { return samples_.size(); }

    // Persistence (BLIND_SPOT_MAP.md §7). Owners are dropped on save: next run, every loaded sample is
    // "another object" of the class.
    bool save(const std::string& path)
    {
        flush_all();
        std::ofstream f(path, std::ios::trunc);
        if (not f)
            return false;
        f.imbue(std::locale::classic());
        f << "az,el,lr,r,w\n";
        for (const auto& s : samples_)
            f << s.v.az << ',' << s.v.el << ',' << s.v.lr << ',' << s.r << ',' << s.w << '\n';
        return static_cast<bool>(f);
    }
    std::size_t load(const std::string& path)
    {
        std::ifstream f(path);
        if (not f)
            return 0;
        std::string line;
        std::getline(f, line);   // header
        std::size_t n = 0;
        while (std::getline(f, line))
        {
            float x[5];
            const char* b = line.data();
            const char* e = line.data() + line.size();
            bool ok = true;
            for (int i = 0; i < 5 and ok; ++i)
            {
                const auto [ptr, ec] = std::from_chars(b, e, x[i]);
                ok = ec == std::errc{};
                b = (ptr < e and *ptr == ',') ? ptr + 1 : ptr;
            }
            if (not ok or not (x[4] > 0.0f))
                continue;
            push({{x[0], x[1], x[2]}, std::clamp(x[3], 0.0f, 1.0f), std::min(x[4], 1.0f), 0});
            ++n;
        }
        return n;
    }

private:
    struct Sample { Viewpoint v; float r = 0.0f, w = 0.0f; std::uint64_t owner = 0; };
    struct Visit  { Viewpoint anchor; float w_sum = 0.0f, hit_sum = 0.0f; };

    float dist2(const Viewpoint& a, const Viewpoint& b) const
    {
        const float da = (a.az - b.az) / p_.ell_az, de = (a.el - b.el) / p_.ell_el, dl = (a.lr - b.lr) / p_.ell_lr;
        return da * da + de * de + dl * dl;
    }
    // A visit is worth at most ONE observation, however long it lasted (§4).
    static float visit_weight(float w_sum) { return 1.0f - std::exp(-w_sum); }
    void close(std::uint64_t owner, const Visit& vis)
    {
        if (vis.w_sum > 0.0f)
            push({vis.anchor, vis.hit_sum / vis.w_sum, visit_weight(vis.w_sum), owner});
    }
    void push(const Sample& s)
    {
        samples_.push_back(s);
        while (samples_.size() > p_.max_samples)
            samples_.pop_front();
    }

    BlindSpotParams                          p_;
    std::deque<Sample>                       samples_;
    std::unordered_map<std::uint64_t, Visit> open_;
};

}  // namespace rc::detect
