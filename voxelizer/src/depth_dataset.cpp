#include "depth_dataset.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <charconv>
#include <fstream>
#include <locale>
#include <print>
#include <sstream>

namespace rc::depth
{

namespace
{
// ★LOCALE-INDEPENDENT PARSING. Do NOT use strtof/atof/atoi here, and do not reach for a stringstream
// either. This CSV is written by std::ofstream, which formats through the C++ GLOBAL locale (still
// "C", so a decimal POINT), but strtof reads through the C locale — and Qt calls setlocale(LC_ALL,"")
// at startup, which on this machine is es_ES.UTF-8 where the decimal separator is a COMMA. strtof then
// stops at the '.' and every float silently truncates to its integer part. Measured 2026-08-03: poses
// collapsed to integers so dedup threw away 334 of 406 frames, `t` became 0 so ct fitted to exactly
// zero, values in scientific notation ("-9.2e-05") parsed as -9 giving a "supported range" of
// 0.00012..8103 m, and the fit came out a=0.153 instead of 0.193. std::from_chars is locale
// independent BY DEFINITION, which is the only property that makes a data file portable across
// whatever locale the process happens to boot into.
template <typename T>
const char* parse_field(const char* p, const char* end, T& out)
{
    while (p < end and (*p == ' ' or *p == '\t')) ++p;
    const auto r = std::from_chars(p, end, out);
    if (r.ec != std::errc{})
        out = T{};
    p = (r.ec == std::errc{}) ? r.ptr : p;
    while (p < end and *p != ',') ++p;      // skip to the separator (and past any unconsumed tail)
    return (p < end) ? p + 1 : end;
}

// Pose novelty. Deliberately NOT a model quantity — this decides what the dataset CONTAINS, not what
// anything believes, so a plain distance test is the honest tool rather than something dressed up as
// inference. Angle wrapped so 359° and 1° are 2° apart.
bool pose_close(const DepthFrame& a, const DepthFrame& b, float dm, float dth)
{
    const float dx = a.rx - b.rx, dy = a.ry - b.ry;
    float da = std::fmod(std::abs(a.rtheta - b.rtheta), 2.f * 3.14159265f);
    if (da > 3.14159265f) da = 2.f * 3.14159265f - da;
    return (dx * dx + dy * dy) < dm * dm and da < dth;
}
}   // namespace

// ─── DepthFitMap I/O ─────────────────────────────────────────────────────────────────────────────

bool DepthFitMap::save(const std::string& path) const
{
    std::ofstream f(path, std::ios::trunc);
    if (not f.is_open())
        return false;
    f.imbue(std::locale::classic());   // decimal POINT always — the reader assumes it
    f << "# ricoh monocular-depth correction map\n"
      << "# log_range = a*lm + SUM hinge[k]*max(0, lm - knot[k]) + b[view] + cs*s^2   (lm = log_model)\n"
      << "# ct*t^2 was DROPPED: the ceiling half of the band has no lidar, so it only extrapolated.\n"
      << "# fitted on " << n_samples << " samples / " << n_frames << " frames; "
      << "resid_rms(log)=" << resid_rms << " (anchored " << resid_anchored << ") r=" << r << "\n"
      << "# ★SUPPORTED RANGE " << range_lo << " .. " << range_hi << " m — outside that the fit is\n"
      << "# extrapolation, because helios only anchors the horizon stripe at short range.\n"
      << "a," << a << "\ncs," << cs << "\nn_views," << n_views << "\nn_knots," << n_knots << "\n"
      << "n_samples," << n_samples << "\nn_frames," << n_frames << "\nresid_rms," << resid_rms
      << "\nresid_anchored," << resid_anchored
      << "\nmed_rel," << med_rel << "\nmed_abs_m," << med_abs_m << "\ndelta125," << delta125
      << "\nr," << r << "\nrange_lo," << range_lo << "\nrange_hi," << range_hi << "\n";
    for (int v = 0; v < n_views and v < kMaxViews; ++v)
        f << "b" << v << ',' << b[static_cast<std::size_t>(v)] << '\n';
    for (int k = 0; k < n_knots and k < kSplineKnots; ++k)
        f << "knot" << k << ',' << knot[static_cast<std::size_t>(k)]
          << "\nhinge" << k << ',' << hinge[static_cast<std::size_t>(k)] << '\n';
    return true;
}

bool DepthFitMap::load(const std::string& path)
{
    std::ifstream f(path);
    if (not f.is_open())
        return false;
    *this = DepthFitMap{};
    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty() or line[0] == '#')
            continue;
        const auto comma = line.find(',');
        if (comma == std::string::npos)
            continue;
        const std::string key = line.substr(0, comma);
        float val = 0.f;                       // from_chars, not strtof — same locale trap (see above)
        std::from_chars(line.data() + comma + 1, line.data() + line.size(), val);
        if      (key == "a")         a  = val;
        else if (key == "cs")        cs = val;
        else if (key == "ct")        ct = val;
        else if (key == "n_views")   n_views   = static_cast<int>(val);
        else if (key == "n_knots")   n_knots   = static_cast<int>(val);
        else if (key.rfind("knot",0)==0 and key.size()>4)
        { const int k=std::atoi(key.c_str()+4); if(k>=0 and k<kSplineKnots) knot[static_cast<std::size_t>(k)]=val; }
        else if (key.rfind("hinge",0)==0 and key.size()>5)
        { const int k=std::atoi(key.c_str()+5); if(k>=0 and k<kSplineKnots) hinge[static_cast<std::size_t>(k)]=val; }
        else if (key == "n_samples") n_samples = static_cast<long>(val);
        else if (key == "n_frames")  n_frames  = static_cast<long>(val);
        else if (key == "resid_rms") resid_rms = val;
        else if (key == "resid_anchored") resid_anchored = val;
        else if (key == "med_rel")   med_rel   = val;
        else if (key == "med_abs_m") med_abs_m = val;
        else if (key == "delta125")  delta125  = val;
        else if (key == "r")         r  = val;
        else if (key == "range_lo")  range_lo = val;
        else if (key == "range_hi")  range_hi = val;
        else if (key.size() > 1 and key[0] == 'b')
        {
            const int v = std::atoi(key.c_str() + 1);
            if (v >= 0 and v < kMaxViews)
                b[static_cast<std::size_t>(v)] = val;
        }
    }
    valid = (n_views > 0 and n_samples > 0 and std::isfinite(a));
    return valid;
}

// ─── DepthDataset ────────────────────────────────────────────────────────────────────────────────

bool DepthDataset::add_frame(DepthFrame&& f, float min_move_m, float min_turn_rad)
{
    if (f.samples.empty())
        return false;
    if (not frames_.empty() and pose_close(f, frames_.back(), min_move_m, min_turn_rad))
        return false;
    frames_.push_back(std::move(f));
    ++pending_;
    return true;
}

std::size_t DepthDataset::sample_count() const
{
    std::size_t n = 0;
    for (const auto& f : frames_)
        n += f.samples.size();
    return n;
}

bool DepthDataset::append_csv(const std::string& path)
{
    if (pending_ == 0)
        return true;
    const bool fresh = not std::ifstream(path).good();
    std::ofstream f(path, std::ios::app);
    if (not f.is_open())
        return false;
    f.imbue(std::locale::classic());   // decimal POINT always — the reader assumes it
    if (fresh)
        f << "frame,stamp_ms,rx,ry,rtheta,view,s,t,log_model,log_range\n";
    // Only the frames added since the last append — the file is the accumulating set across runs.
    const std::size_t first = frames_.size() - pending_;
    for (std::size_t i = first; i < frames_.size(); ++i)
    {
        const auto& fr = frames_[i];
        for (const auto& s : fr.samples)
            f << i << ',' << fr.stamp_ms << ',' << fr.rx << ',' << fr.ry << ',' << fr.rtheta << ','
              << static_cast<int>(s.view) << ',' << s.s << ',' << s.t << ','
              << s.log_model << ',' << s.log_range << '\n';
    }
    pending_ = 0;
    return true;
}

bool DepthDataset::load_csv(const std::string& path)
{
    std::ifstream f(path);
    if (not f.is_open())
        return false;
    frames_.clear();
    pending_ = 0;
    std::string line;
    std::getline(f, line);   // header
    long cur_id = -1;
    while (std::getline(f, line))
    {
        if (line.empty())
            continue;
        const char* p   = line.data();
        const char* end = line.data() + line.size();
        long id = 0; unsigned long long stamp = 0; int view = 0;
        DepthFrame tmp;
        DepthSample s;
        p = parse_field(p, end, id);
        p = parse_field(p, end, stamp);
        p = parse_field(p, end, tmp.rx);
        p = parse_field(p, end, tmp.ry);
        p = parse_field(p, end, tmp.rtheta);
        p = parse_field(p, end, view);
        p = parse_field(p, end, s.s);
        p = parse_field(p, end, s.t);
        p = parse_field(p, end, s.log_model);
        p = parse_field(p, end, s.log_range);
        tmp.stamp_ms = static_cast<std::uint64_t>(stamp);
        s.view = static_cast<std::uint8_t>(view);
        if (id != cur_id)
        {
            frames_.push_back(std::move(tmp));
            cur_id = id;
        }
        frames_.back().samples.push_back(s);
    }
    return true;
}

std::size_t DepthDataset::dedup(float min_move_m, float min_turn_rad)
{
    std::vector<DepthFrame> keep;
    keep.reserve(frames_.size());
    for (auto& f : frames_)
    {
        const bool dup = std::any_of(keep.begin(), keep.end(), [&](const DepthFrame& k)
                                     { return pose_close(f, k, min_move_m, min_turn_rad); });
        if (not dup)
            keep.push_back(std::move(f));
    }
    const std::size_t removed = frames_.size() - keep.size();
    frames_ = std::move(keep);
    pending_ = 0;
    return removed;
}

DepthFitMap DepthDataset::fit(int n_views) const
{
    DepthFitMap m;
    const int V = std::clamp(n_views, 1, kMaxViews);
    const std::size_t n = sample_count();

    // Knots at EQUALLY-SPACED PERCENTILES of log_model, so every segment carries the same number of
    // samples. Placing them on a uniform grid instead would put most of the freedom where there is no
    // data and leave the dense middle under-modelled.
    const int K = (n > 20000) ? kSplineKnots : 0;   // too few rows ⇒ stay linear rather than overfit
    if (K > 0)
    {
        std::vector<float> lms;
        lms.reserve(n);
        for (const auto& fr : frames_)
            for (const auto& s : fr.samples)
                if (s.view < V)
                    lms.push_back(s.log_model);
        if (lms.size() > static_cast<std::size_t>(K) * 100)
        {
            for (int k = 0; k < K; ++k)
            {
                const auto pos = static_cast<std::size_t>((k + 1.0) / (K + 1.0) * (lms.size() - 1));
                std::nth_element(lms.begin(), lms.begin() + static_cast<std::ptrdiff_t>(pos), lms.end());
                m.knot[static_cast<std::size_t>(k)] = lms[pos];
            }
            m.n_knots = K;
        }
    }

    const int P = 1 + m.n_knots + 1 + V;           // a, hinge[K], cs, b[V]   (ct dropped)
    if (n < static_cast<std::size_t>(P) * 20)      // enough rows for the parameters to be determined
        return m;

    // Normal equations, accumulated in one pass — the design matrix would be millions of rows and
    // there is no reason to materialise it.
    Eigen::MatrixXd AtA = Eigen::MatrixXd::Zero(P, P);
    Eigen::VectorXd Atb = Eigen::VectorXd::Zero(P);
    Eigen::VectorXd x(P);
    double sy = 0.0, syy = 0.0;
    float  rmin = 1e9f, rmax = -1e9f;

    for (const auto& fr : frames_)
        for (const auto& s : fr.samples)
        {
            if (s.view >= V)
                continue;
            x.setZero();
            x(0) = s.log_model;
            for (int k = 0; k < m.n_knots; ++k)
                x(1 + k) = std::max(0.0f, s.log_model - m.knot[static_cast<std::size_t>(k)]);
            x(1 + m.n_knots) = static_cast<double>(s.s) * s.s;
            x(2 + m.n_knots + s.view) = 1.0;
            AtA.selfadjointView<Eigen::Lower>().rankUpdate(x);
            Atb += x * static_cast<double>(s.log_range);
            sy  += s.log_range;
            syy += static_cast<double>(s.log_range) * s.log_range;
            const float rr = std::exp(s.log_range);
            rmin = std::min(rmin, rr);
            rmax = std::max(rmax, rr);
        }
    // A tiny ridge keeps the solve well-posed when a view contributed almost nothing this session —
    // its b would otherwise be determined by a handful of points, or not at all.
    AtA.diagonal().array() += 1e-6;
    // Solve through the selfadjoint view rather than mirroring the triangle first: `AtA.triangularView
    // <Upper>() = AtA.transpose()` aliases AtA on both sides, which Eigen does not guarantee for a
    // triangular assignment. rankUpdate only ever wrote the lower triangle, so let the view read it.
    const Eigen::VectorXd sol = AtA.selfadjointView<Eigen::Lower>().ldlt().solve(Atb);
    if (not sol.allFinite())
        return m;

    m.a = static_cast<float>(sol(0));
    for (int k = 0; k < m.n_knots; ++k)
        m.hinge[static_cast<std::size_t>(k)] = static_cast<float>(sol(1 + k));
    m.cs = static_cast<float>(sol(1 + m.n_knots));
    for (int v = 0; v < V; ++v)
        m.b[static_cast<std::size_t>(v)] = static_cast<float>(sol(2 + m.n_knots + v));
    m.ct = 0.f;                                   // deprecated, never applied
    m.n_views = V;

    // Residual + correlation of fitted vs measured, on the data itself. This is a FIT quality number,
    // not a validation score — nothing here is held out, so read it as "did the form fit", never as
    // "will it generalise to poses we never visited".
    double sse = 0.0, sf = 0.0, sff = 0.0, sfy = 0.0;
    long   cnt = 0;
    for (const auto& fr : frames_)
        for (const auto& s : fr.samples)
        {
            if (s.view >= V)
                continue;
            const double yhat = m.apply(s.log_model, s.view, s.s, s.t);
            const double d = yhat - s.log_range;
            sse += d * d; sf += yhat; sff += yhat * yhat; sfy += yhat * s.log_range;
            ++cnt;
        }
    if (cnt > 1)
    {
        const double N = static_cast<double>(cnt);
        m.resid_rms = static_cast<float>(std::sqrt(sse / N));
        const double cov = sfy / N - (sf / N) * (sy / N);
        const double vf  = sff / N - (sf / N) * (sf / N);
        const double vy  = syy / N - (sy / N) * (sy / N);
        m.r = (vf > 1e-12 and vy > 1e-12) ? static_cast<float>(cov / std::sqrt(vf * vy)) : 0.f;
    }
    // Residual as the RUNTIME will see it: within each (frame, view) the offset is re-solved from that
    // frame's LiDAR, so charge the fit only for what the anchor cannot absorb. Two passes per group —
    // mean of (y - a*lm - cs*s^2 - ct*t^2), then the spread about it.
    {
        double sse_a = 0.0;
        long   cnt_a = 0;
        for (const auto& fr : frames_)
        {
            std::array<double, kMaxViews> sum{}, num{};
            for (const auto& s : fr.samples)
            {
                if (s.view >= V) continue;
                sum[s.view] += static_cast<double>(s.log_range)
                             - m.base(s.log_model, s.s);
                num[s.view] += 1.0;
            }
            for (const auto& s : fr.samples)
            {
                if (s.view >= V or num[s.view] < 1.0) continue;
                const double bg = sum[s.view] / num[s.view];
                const double d  = m.apply_with(s.log_model, static_cast<float>(bg), s.s, s.t) - s.log_range;
                sse_a += d * d;
                ++cnt_a;
            }
        }
        if (cnt_a > 0)
            m.resid_anchored = static_cast<float>(std::sqrt(sse_a / static_cast<double>(cnt_a)));

        // Second pass for the human-readable measures, on the SAME anchored prediction.
        std::vector<float> rel, absm;
        rel.reserve(static_cast<std::size_t>(cnt_a));
        absm.reserve(static_cast<std::size_t>(cnt_a));
        long within = 0;
        for (const auto& fr : frames_)
        {
            std::array<double, kMaxViews> sum{}, num{};
            for (const auto& s : fr.samples)
            {
                if (s.view >= V) continue;
                sum[s.view] += static_cast<double>(s.log_range)
                             - m.base(s.log_model, s.s);
                num[s.view] += 1.0;
            }
            for (const auto& s : fr.samples)
            {
                if (s.view >= V or num[s.view] < 1.0) continue;
                const double bg = sum[s.view] / num[s.view];
                const double d  = std::exp(m.apply_with(s.log_model, static_cast<float>(bg), s.s, s.t));
                const double tr = std::exp(static_cast<double>(s.log_range));
                if (not (d > 0.0) or not (tr > 0.0)) continue;
                rel.push_back(static_cast<float>(std::abs(d - tr) / tr));
                absm.push_back(static_cast<float>(std::abs(d - tr)));
                if (std::max(d / tr, tr / d) < 1.25) ++within;
            }
        }
        if (rel.size() > 32)
        {
            const auto med = [](std::vector<float>& v)
            { std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end()); return v[v.size() / 2]; };
            m.med_rel   = med(rel);
            m.med_abs_m = med(absm);
            m.delta125  = static_cast<float>(within) / static_cast<float>(rel.size());
        }
    }

    m.n_samples = static_cast<long>(n);
    m.n_frames  = static_cast<long>(frames_.size());
    m.range_lo  = rmin;
    m.range_hi  = rmax;
    m.valid     = std::isfinite(m.a) and cnt > 0;
    return m;
}

}   // namespace rc::depth
