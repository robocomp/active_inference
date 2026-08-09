#include "depth_dataset.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <charconv>
#include <fstream>
#include <locale>
#include <print>
#include <ranges>
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
      << "# log_range = a*lm + SUM hinge[k]*max(0, lm - knot[k]) + b[view] + cs*s^2 [+ ct*t^2]  (lm = log_model)\n"
      << "# ct*t^2 is applied ONLY when ct_active=1: a LiDAR-only set has no samples above the horizon,\n"
      << "# so the term is unidentifiable there. Enrichment (room-envelope ceiling rows) is what gives\n"
      << "# it support, and fit() keeps it only when dBIC = t_ratio^2 - ln(N_eff) > 0.\n"
      << "# fitted on " << n_samples << " samples (" << n_synth << " synthetic) / " << n_frames
      << " frames; resid_rms(log)=" << resid_rms << " (anchored " << resid_anchored << ") r=" << r << "\n"
      << "# ★SUPPORTED RANGE " << range_lo << " .. " << range_hi << " m, band t " << t_lo << " .. "
      << t_hi << " — outside that the fit is extrapolation.\n"
      << "a," << a << "\ncs," << cs << "\nct," << ct << "\nct_active," << (ct_active ? 1 : 0)
      << "\nct_delta_bic," << ct_delta_bic
      << "\nn_views," << n_views << "\nn_knots," << n_knots << "\n"
      << "n_samples," << n_samples << "\nn_synth," << n_synth
      << "\nt_lo," << t_lo << "\nt_hi," << t_hi
      << "\nn_frames," << n_frames << "\nresid_rms," << resid_rms
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
        // ★Read BEFORE the generic "starts with 'c'"/"starts with 'b'" arms below — key order in the
        // else-if chain is the only thing separating `ct_active` from a stray view offset.
        else if (key == "ct_active") ct_active = (val != 0.f);
        else if (key == "ct_delta_bic") ct_delta_bic = val;
        else if (key == "n_synth")   n_synth   = static_cast<long>(val);
        else if (key == "t_lo")      t_lo = val;
        else if (key == "t_hi")      t_hi = val;
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

// ─── Shared design row + information selector ────────────────────────────────────────────────────

int map_n_params(const DepthFitMap& m, int n_views)
{
    return 1 + std::max(0, m.n_knots) + 1 + (m.ct_active ? 1 : 0) + std::clamp(n_views, 1, kMaxViews);
}

void design_row(const DepthSample& s, const DepthFitMap& m, int n_views, Eigen::VectorXd& x)
{
    const int V = std::clamp(n_views, 1, kMaxViews);
    const int K = std::max(0, m.n_knots);
    const int T = m.ct_active ? 1 : 0;         // the t² column exists only when the map carries ct
    const int P = 1 + K + 1 + T + V;
    if (x.size() != P)
        x.resize(P);
    x.setZero();
    x(0) = s.log_model;
    for (int k = 0; k < K; ++k)
        x(1 + k) = std::max(0.0f, s.log_model - m.knot[static_cast<std::size_t>(k)]);
    x(1 + K) = static_cast<double>(s.s) * s.s;
    if (T > 0)
        x(2 + K) = static_cast<double>(s.t) * s.t;
    if (s.view < V)
        x(2 + K + T + s.view) = 1.0;
}

void InfoSelector::reset(int n_params, double ridge)
{
    const int P = std::max(1, n_params);
    lambda_ = Eigen::MatrixXd::Identity(P, P) * ridge;
    logdet_ = static_cast<double>(P) * std::log(ridge);
    n_accepted_ = 0;
}

void InfoSelector::accumulate(const std::vector<DepthSample>& samples, const DepthFitMap& m, int n_views)
{
    if (not ready() or samples.empty())
        return;
    Eigen::VectorXd x;
    for (const auto& s : samples)
    {
        design_row(s, m, n_views, x);
        if (x.size() != lambda_.rows())
            return;                                   // geometry changed under us — caller must reset
        lambda_.selfadjointView<Eigen::Lower>().rankUpdate(x, static_cast<double>(s.w));
    }
    const Eigen::LLT<Eigen::MatrixXd> llt(lambda_.selfadjointView<Eigen::Lower>());
    if (llt.info() == Eigen::Success)
        logdet_ = 2.0 * llt.matrixLLT().diagonal().array().log().sum();
    ++n_accepted_;
}

double InfoSelector::gain(const std::vector<DepthSample>& samples, const DepthFitMap& m, int n_views) const
{
    if (not ready() or samples.empty())
        return 0.0;
    Eigen::MatrixXd cand = lambda_;                   // Λ + ΔΛ, without touching the running state
    Eigen::VectorXd x;
    for (const auto& s : samples)
    {
        design_row(s, m, n_views, x);
        if (x.size() != cand.rows())
            return 0.0;
        cand.selfadjointView<Eigen::Lower>().rankUpdate(x, static_cast<double>(s.w));
    }
    const Eigen::LLT<Eigen::MatrixXd> llt(cand.selfadjointView<Eigen::Lower>());
    if (llt.info() != Eigen::Success)
        return 0.0;
    const double ld = 2.0 * llt.matrixLLT().diagonal().array().log().sum();
    return 0.5 * (ld - logdet_);                      // nats
}

double InfoSelector::residual(const std::vector<DepthSample>& samples, const DepthFitMap& m, int n_views) const
{
    if (not m.valid or samples.empty())
        return -1.0;
    const int V = std::clamp(n_views, 1, kMaxViews);
    // Solve this frame's own per-view offset first, exactly as the runtime anchor does — otherwise a
    // frame would look "inconsistent" purely because the model's arbitrary per-image scale differs,
    // which is the one thing we already know it does and do not want to penalise.
    std::array<double, kMaxViews> sum{}, num{};
    for (const auto& s : samples)
        if (s.view < V)
        {
            sum[s.view] += static_cast<double>(s.log_range) - m.base(s.log_model, s.s, s.t);
            num[s.view] += 1.0;
        }
    std::vector<double> err;
    err.reserve(samples.size());
    for (const auto& s : samples)
        if (s.view < V and num[s.view] > 0.0)
            err.push_back(std::abs(m.apply_with(s.log_model,
                                                static_cast<float>(sum[s.view] / num[s.view]),
                                                s.s, s.t) - s.log_range));
    if (err.empty())
        return -1.0;
    std::nth_element(err.begin(), err.begin() + err.size() / 2, err.end());
    return err[err.size() / 2];
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

const DepthFrame& DepthDataset::frame(std::size_t i) const
{
    static const DepthFrame none;
    return i < frames_.size() ? frames_[i] : none;
}

bool DepthDataset::add_samples_to_frame(std::uint64_t stamp_ms, const std::vector<DepthSample>& extra)
{
    if (extra.empty())
        return false;
    const auto it = std::ranges::find_if(frames_, [stamp_ms](const DepthFrame& f)
                                         { return f.stamp_ms == stamp_ms; });
    if (it == frames_.end())
        return false;
    it->samples.insert(it->samples.end(), extra.begin(), extra.end());
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

bool DepthDataset::save_csv(const std::string& path) const
{
    std::ofstream f(path, std::ios::trunc);
    if (not f.is_open())
        return false;
    f.imbue(std::locale::classic());   // decimal POINT always — the reader assumes it
    f << "frame,stamp_ms,rx,ry,rtheta,view,s,t,log_model,log_range,src,region,w,h\n";
    for (std::size_t i = 0; i < frames_.size(); ++i)
    {
        const auto& fr = frames_[i];
        for (const auto& s : fr.samples)
            f << i << ',' << fr.stamp_ms << ',' << fr.rx << ',' << fr.ry << ',' << fr.rtheta << ','
              << static_cast<int>(s.view) << ',' << s.s << ',' << s.t << ','
              << s.log_model << ',' << s.log_range << ','
              << static_cast<int>(s.src) << ',' << static_cast<int>(s.region) << ','
              << s.w << ',' << s.h << '\n';
    }
    return f.good();
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
    // ★SCHEMA DETECTION ON THE HEADER, never on the field count of a row. See load_csv()'s
    // declaration: defaulting a missing `w` to 0 instead of 1 would silently zero every legacy row's
    // weight, and a fit over an all-zero-weight set fails with no message anywhere.
    const bool extended = line.find(",w,") != std::string::npos;
    // ★Group by STAMP, not by the frame-id column. `append_csv` writes the in-memory index, which
    // restarts at 0 every session, so ids COLLIDE across runs — a file with 53 real frames was
    // observed carrying only 42 distinct ids. Grouping on id changes happens to recover that, but
    // fails silently in one case: a session contributing exactly ONE frame (id 0) followed by another
    // starting at 0 merges two panoramas, from two different poses, into a single frame — corrupting
    // both its anchor and its information score. `stamp_ms` is the panorama's own capture stamp, is
    // unique by construction (RicohSource dedups on it, and collection admits each stamp once), and
    // is already the join key to the saved image. Fall back to the id only for stamp-less legacy rows.
    long          cur_id    = -1;
    std::uint64_t cur_stamp = 0;
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
        if (extended)
        {
            int src = 0, region = 0;
            p = parse_field(p, end, src);
            p = parse_field(p, end, region);
            p = parse_field(p, end, s.w);
            p = parse_field(p, end, s.h);
            s.src    = static_cast<std::uint8_t>(src);
            s.region = static_cast<std::uint16_t>(region);
            if (not (s.w > 0.f) or not std::isfinite(s.w))
                s.w = 1.f;
        }
        tmp.stamp_ms = static_cast<std::uint64_t>(stamp);
        s.view = static_cast<std::uint8_t>(view);
        const bool new_frame = frames_.empty()
                            or (tmp.stamp_ms != 0 ? tmp.stamp_ms != cur_stamp : id != cur_id);
        if (new_frame)
        {
            frames_.push_back(std::move(tmp));
            cur_id    = id;
            cur_stamp = frames_.back().stamp_ms;
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


DepthFitMap DepthDataset::fit(int n_views, bool measured_only) const
{
    DepthFitMap m;
    const int V = std::clamp(n_views, 1, kMaxViews);

    // Which rows take part. `measured_only` is the honest A/B against enrichment: same code, same
    // frames, same knot rule — only the synthetic rows withheld.
    const auto use = [&](const DepthSample& s)
    { return s.view < V and (not measured_only or s.src == kSrcLidar); };

    std::size_t n = 0, n_syn = 0;
    for (const auto& fr : frames_)
        for (const auto& s : fr.samples)
            if (use(s))
            {
                ++n;
                if (s.src != kSrcLidar)
                    ++n_syn;
            }

    // Knots at EQUALLY-SPACED PERCENTILES of log_model, so every segment carries the same number of
    // samples. Placing them on a uniform grid instead would put most of the freedom where there is no
    // data and leave the dense middle under-modelled.
    // ★Knot COUNT is proposed here; whether the spline SURVIVES is decided by conditioning below, not
    // by a row count. A row count was the old gate and it is the wrong question: the information
    // selector's whole purpose is to make each row worth more, so counting rows systematically
    // under-rates an information-selected set (13,604 selected samples constrain these 16 parameters
    // far better than 13,604 redundant ones from a single viewpoint, and a count cannot tell them
    // apart). The floor below is only "enough rows to place K percentiles at all".
    const int K = (n > static_cast<std::size_t>(kSplineKnots) * 100) ? kSplineKnots : 0;
    if (K > 0)
    {
        std::vector<float> lms;
        lms.reserve(n);
        for (const auto& fr : frames_)
            for (const auto& s : fr.samples)
                if (use(s))
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

    // ★PROPOSE ct only when there are rows that could possibly identify it. A LiDAR-only set lives on
    // the horizon stripe (t ≈ −0.14..+0.67 against a band of ±0.667), so the ceiling half is empty and
    // the term would be pure extrapolation — which is exactly why it was deleted. The proposal is
    // adjudicated by ΔBIC after the solve; this only decides whether the question is worth asking.
    m.ct_active = (n_syn > 0);

    double sy = 0.0, syy = 0.0;      // MEASURED rows only — the descriptive stats behind `r`
    long   n_meas = 0;
    double syy_w = 0.0;              // yᵀR⁻¹y, for the residual variance the ΔBIC test needs
    double sumw = 0.0, sumw2 = 0.0;  // Kish effective sample size = sumw² / sumw2
    long   n_rows = 0;
    float  rmin = 1e9f, rmax = -1e9f, tlo = 1e9f, thi = -1e9f;
    Eigen::MatrixXd AtA;
    Eigen::VectorXd Atb;

    // One COMMON-MODE group inside one frame: the rank-1 direction its samples share.
    struct CommonMode { Eigen::VectorXd u; double q = 0.0, z = 0.0; };

    // Normal equations, accumulated in one pass — the design matrix would be millions of rows and
    // there is no reason to materialise it. Factored out so every refit below reuses it verbatim.
    const auto accumulate = [&](const DepthFitMap& mm)
    {
        const int PP = map_n_params(mm, V);
        AtA = Eigen::MatrixXd::Zero(PP, PP);
        Atb = Eigen::VectorXd::Zero(PP);
        Eigen::VectorXd x(PP);
        sy = syy = syy_w = 0.0;
        sumw = sumw2 = 0.0;
        n_meas = 0; n_rows = 0;
        rmin = 1e9f; rmax = -1e9f; tlo = 1e9f; thi = -1e9f;
        std::vector<CommonMode> reg;
        for (const auto& fr : frames_)
        {
            reg.clear();
            for (const auto& s : fr.samples)
            {
                if (not use(s))
                    continue;
                // w is a PRECISION relative to a LiDAR row (1.0 by definition), so an untouched
                // dataset accumulates exactly the unit-weight normal equations it always did.
                const double w = (std::isfinite(s.w) and s.w > 0.f) ? static_cast<double>(s.w) : 0.0;
                if (w <= 0.0)
                    continue;
                const double y = static_cast<double>(s.log_range);
                design_row(s, mm, V, x);   // the SHARED definition — see design_row()
                AtA.selfadjointView<Eigen::Lower>().rankUpdate(x, w);
                Atb.noalias() += (w * y) * x;
                syy_w += w * y * y;
                sumw  += w;
                sumw2 += w * w;
                ++n_rows;
                if (s.src == kSrcLidar)
                {
                    sy  += y;
                    syy += y * y;
                    ++n_meas;
                }
                const float rr = std::exp(s.log_range);
                rmin = std::min(rmin, rr);
                rmax = std::max(rmax, rr);
                tlo  = std::min(tlo, s.t);
                thi  = std::max(thi, s.t);
                if (s.region != 0 and s.h > 0.f and std::isfinite(s.h))
                {
                    if (s.region >= reg.size())
                        reg.resize(static_cast<std::size_t>(s.region) + 1);
                    CommonMode& cm = reg[s.region];
                    if (cm.u.size() != PP)
                        cm.u = Eigen::VectorXd::Zero(PP);
                    const double wh = w * static_cast<double>(s.h);
                    cm.u.noalias() += wh * x;
                    cm.q += wh * static_cast<double>(s.h);
                    cm.z += wh * y;
                }
            }
            // ★COMMON-MODE MARGINALISATION (Woodbury), the reason a wall does not out-vote the LiDAR.
            // A region's samples are NOT independent: they all sit on one believed plane, and when that
            // plane moves by δn every one of them moves with it. Their covariance is therefore
            // R = D + h·hᵀ (D diagonal = the per-pixel independent part, h = the shared displacement's
            // per-sample effect δlnR = δn/(R·cosθ), which is why h is per-sample and not a constant).
            // With W = D⁻¹,  R⁻¹ = W − W·h·hᵀ·W / (1 + hᵀWh), so the normal equations only need the
            // three quantities already accumulated above:
            //     XᵀR⁻¹X = XᵀWX − u·uᵀ/(1+q),   XᵀR⁻¹y = XᵀWy − u·z/(1+q),   u = ΣwhX, q = Σwh², z = Σwhy.
            // The effect is exactly right: the region's INTERNAL contrasts keep their full weight (they
            // are what pins the spline's shape), while its MEAN saturates at one δn-limited observation
            // however many pixels it contains. A diagonal σ-inflation cannot do this — it would kill
            // both, and 5,000 wall pixels would end up worth less than a single LiDAR return.
            for (auto& cm : reg)
            {
                if (cm.u.size() != PP or not (cm.q > 0.0))
                    continue;
                const double k = 1.0 / (1.0 + cm.q);
                AtA.selfadjointView<Eigen::Lower>().rankUpdate(cm.u, -k);
                Atb.noalias() -= (k * cm.z) * cm.u;
                syy_w -= k * cm.z * cm.z;
            }
        }
        // A tiny ridge keeps the solve well-posed when a view contributed almost nothing this session
        // — its b would otherwise be determined by a handful of points, or not at all.
        AtA.diagonal().array() += 1e-6;
    };

    // ★IS THE SPLINE IDENTIFIABLE? Ask the data, not the row count. Scale AtA by its diagonal (so the
    // answer is about information content, not about the units each column happens to carry) and take
    // the eigenvalue ratio. If the worst-determined direction carries less than `kMinRcond` of the
    // best, the hinge coefficients are noise that the solve would amplify — drop to the linear form
    // and refit rather than publish parameters the data cannot support.
    constexpr double kMinRcond = 1e-6;
    const auto rcond_of = [](const Eigen::MatrixXd& A)
    {
        // Materialise the full symmetric matrix first — rankUpdate only wrote the lower triangle, and
        // a SelfAdjointView is not a plain matrix expression the diagonal scaling can multiply.
        const Eigen::MatrixXd F = A.selfadjointView<Eigen::Lower>();
        const Eigen::VectorXd d = F.diagonal().cwiseMax(1e-300).cwiseSqrt().cwiseInverse();
        const Eigen::MatrixXd S = d.asDiagonal() * F * d.asDiagonal();
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S, Eigen::EigenvaluesOnly);
        if (es.info() != Eigen::Success)
            return 0.0;
        const double lo = es.eigenvalues().minCoeff(), hi = es.eigenvalues().maxCoeff();
        return (hi > 0.0) ? std::max(0.0, lo) / hi : 0.0;
    };

    accumulate(m);
    if (n < static_cast<std::size_t>(map_n_params(m, V)) * 20)   // not enough rows for ANY fit
        return m;
    if (m.n_knots > 0)
    {
        const double rc = rcond_of(AtA);
        if (rc < kMinRcond)
        {
            std::println("[depth-map] spline NOT identifiable (rcond {:.2e} < {:.0e}) — refitting "
                         "linear. Collect more varied log-depth, not merely more rows.", rc, kMinRcond);
            m.n_knots = 0;
            m.knot.fill(0.f);
            m.hinge.fill(0.f);
            accumulate(m);
        }
    }

    // Solve through the selfadjoint view rather than mirroring the triangle first: `AtA.triangularView
    // <Upper>() = AtA.transpose()` aliases AtA on both sides, which Eigen does not guarantee for a
    // triangular assignment. rankUpdate only ever wrote the lower triangle, so let the view read it.
    Eigen::VectorXd sol = AtA.selfadjointView<Eigen::Lower>().ldlt().solve(Atb);
    if (not sol.allFinite())
        return m;

    // ── ★DOES ct EARN ITS PARAMETER? Bayesian model selection, not a row count ────────────────────
    // M0 = no vertical term, M1 = with it. Under a Gaussian likelihood the log evidence ratio is, to
    // BIC order, ΔlnZ ≈ ½·t_ratio² − ½·ln(N_eff): the term must explain more than one parameter's
    // worth of the data. t_ratio = |ct| / se(ct) with se² = σ0²·(XᵀR⁻¹X)⁻¹_{ct,ct}, and BOTH the
    // variance estimate and the penalty use the KISH EFFECTIVE SAMPLE SIZE (Σw)²/Σw² rather than the
    // raw row count — the synthetic rows are correlated by construction and counting them as
    // independent would let a single wall talk the model into a parameter it cannot see.
    if (m.ct_active)
    {
        const int    PP    = static_cast<int>(sol.size());
        const int    j     = 2 + m.n_knots;                  // the t² column — see design_row()
        const double n_eff = (sumw2 > 0.0) ? (sumw * sumw) / sumw2 : 0.0;
        const double dof   = n_eff - static_cast<double>(PP);
        const double sse   = std::max(0.0, syy_w - 2.0 * sol.dot(Atb)
                                        + sol.dot(AtA.selfadjointView<Eigen::Lower>() * sol));
        double delta_bic = -1.0;
        if (dof > 1.0 and j < PP)
        {
            Eigen::VectorXd e = Eigen::VectorXd::Zero(PP);
            e(j) = 1.0;
            const Eigen::VectorXd col = AtA.selfadjointView<Eigen::Lower>().ldlt().solve(e);
            const double var_ct = (sse / dof) * std::max(0.0, col(j));
            if (var_ct > 0.0 and n_eff > 1.0)
            {
                const double t_ratio = static_cast<double>(sol(j)) / std::sqrt(var_ct);
                delta_bic = t_ratio * t_ratio - std::log(n_eff);
            }
        }
        m.ct_delta_bic = static_cast<float>(delta_bic);
        std::println("[depth-map] ct*t^2: dBIC = {:+.1f} (N_eff {:.0f}, t span {:+.3f}..{:+.3f}) — {}",
                     delta_bic, n_eff, tlo, thi, delta_bic > 0.0 ? "KEPT" : "dropped as unidentifiable");
        if (not (delta_bic > 0.0))
        {
            const float keep = m.ct_delta_bic;
            m.ct_active = false;
            accumulate(m);
            sol = AtA.selfadjointView<Eigen::Lower>().ldlt().solve(Atb);
            m.ct_delta_bic = keep;
            if (not sol.allFinite())
                return m;
        }
    }

    const int K_ = m.n_knots;
    const int T_ = m.ct_active ? 1 : 0;
    m.a = static_cast<float>(sol(0));
    for (int k = 0; k < K_; ++k)
        m.hinge[static_cast<std::size_t>(k)] = static_cast<float>(sol(1 + k));
    m.cs = static_cast<float>(sol(1 + K_));
    m.ct = T_ > 0 ? static_cast<float>(sol(2 + K_)) : 0.f;
    for (int v = 0; v < V; ++v)
        m.b[static_cast<std::size_t>(v)] = static_cast<float>(sol(2 + K_ + T_ + v));
    m.n_views = V;

    // ── Quality metrics: on the MEASURED rows only ───────────────────────────────────────────────
    // The synthetic rows are what the map was HELPED by, not what it is judged against — scoring the
    // fit on a belief it was fitted to would be circular, and it would also make every number
    // incomparable with the pre-enrichment ones. So resid/med_rel/δ1/r all speak about how well the
    // map predicts the LiDAR, exactly as before. (If a set somehow has no measured rows at all, fall
    // back to every row rather than reporting zeros.)
    const bool have_meas = (n_meas > 0);
    const auto scored = [&](const DepthSample& s)
    { return use(s) and (not have_meas or s.src == kSrcLidar); };
    if (not have_meas)
    {
        sy = syy = 0.0;
        for (const auto& fr : frames_)
            for (const auto& s : fr.samples)
                if (scored(s)) { sy += s.log_range; syy += static_cast<double>(s.log_range) * s.log_range; }
    }

    // Residual + correlation of fitted vs measured, on the data itself. This is a FIT quality number,
    // not a validation score — nothing here is held out, so read it as "did the form fit", never as
    // "will it generalise to poses we never visited".
    double sse = 0.0, sf = 0.0, sff = 0.0, sfy = 0.0;
    long   cnt = 0;
    for (const auto& fr : frames_)
        for (const auto& s : fr.samples)
        {
            if (not scored(s))
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
    // mean of (y - base(...)), then the spread about it. ★The anchor is solved on MEASURED rows only,
    // because that is all the live anchor ever has.
    {
        double sse_a = 0.0;
        long   cnt_a = 0;
        for (const auto& fr : frames_)
        {
            std::array<double, kMaxViews> sum{}, num{};
            for (const auto& s : fr.samples)
            {
                if (not scored(s)) continue;
                sum[s.view] += static_cast<double>(s.log_range) - m.base(s.log_model, s.s, s.t);
                num[s.view] += 1.0;
            }
            for (const auto& s : fr.samples)
            {
                if (not scored(s) or num[s.view] < 1.0) continue;
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
                if (not scored(s)) continue;
                sum[s.view] += static_cast<double>(s.log_range) - m.base(s.log_model, s.s, s.t);
                num[s.view] += 1.0;
            }
            for (const auto& s : fr.samples)
            {
                if (not scored(s) or num[s.view] < 1.0) continue;
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
    m.n_synth   = static_cast<long>(n_syn);
    m.n_frames  = static_cast<long>(frames_.size());
    // ★Support spans EVERY row, synthetic included — extending it is the whole point of enrichment.
    m.range_lo  = rmin;
    m.range_hi  = rmax;
    m.t_lo      = (tlo <= thi) ? tlo : 0.f;
    m.t_hi      = (tlo <= thi) ? thi : 0.f;
    m.valid     = std::isfinite(m.a) and cnt > 0;
    return m;
}

}   // namespace rc::depth
