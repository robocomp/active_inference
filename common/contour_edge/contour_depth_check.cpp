#include "contour_depth_check.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace rc::edges
{

namespace
{

constexpr float kMinValidDepth = 0.05f;   // below this a ZED read is not a near surface, it is a decode
                                          // artefact; the same floor camera_ingestor's depth patch uses.

// Φ(x) = ½·erfc(−x/√2). Same form as rc::exist's soft occ/free split, so "graded, signed, no cutoff"
// means the same thing in both files.
[[nodiscard]] inline float phi_cdf(float x) { return 0.5f * std::erfc(-x * 0.70710678f); }

[[nodiscard]] inline bool valid_depth(float d) { return std::isfinite(d) and d > kMinValidDepth; }

// Median of the valid depths along the normal between offset_min and offset_max, on the given side.
// MEDIAN and not mean, for camera_ingestor's reason: a run of pixels crossing an edge mixes two
// populations and a mean lands between them, on a surface that is not there.
[[nodiscard]] float side_depth(const cv::Mat& depth, float px, float py, float nx, float ny,
                               int off_min, int off_max, float sx, float sy)
{
    std::vector<float> vals;
    vals.reserve(static_cast<std::size_t>(std::max(1, off_max - off_min + 1)));
    for (int k = off_min; k <= off_max; ++k)
    {
        const int qx = static_cast<int>(std::lround((px + k * nx) * sx));
        const int qy = static_cast<int>(std::lround((py + k * ny) * sy));
        if (qx < 0 or qy < 0 or qx >= depth.cols or qy >= depth.rows)
            continue;
        const float d = depth.at<float>(qy, qx);
        if (valid_depth(d))
            vals.push_back(d);
    }
    if (vals.empty())
        return -1.0f;
    std::nth_element(vals.begin(), vals.begin() + vals.size() / 2, vals.end());
    return vals[vals.size() / 2];
}

struct SideScore
{
    double acc = 0.0;        // sum of A·R          (unsigned, for reading)
    double verdict_acc = 0.0;// sum of (2·A·R − 1)  (SIGNED — the evidence)
    double bias_acc = 0.0;
    int n = 0;
    int n_no_return = 0;
};

// Score one contour: walk its edges, and at each step ask the model's own question — is the surface at
// the predicted range, and does the world recede behind the boundary?
[[nodiscard]] SideScore score_contour(const cv::Mat& depth, const Contour& c, const ContourDepthParams& p)
{
    SideScore out;
    if (not c.valid())
        return out;

    // Winding sign, so the normal below points OUTWARD rather than into the object on half the edges —
    // which would swap d_in and d_out and turn recession into approach on those edges.
    double area2 = 0.0;
    for (std::size_t i = 0; i < c.px.size(); ++i)
    {
        const cv::Point a = c.px[i], b = c.px[(i + 1) % c.px.size()];
        area2 += static_cast<double>(a.x) * b.y - static_cast<double>(b.x) * a.y;
    }
    const double wind = area2 >= 0.0 ? 1.0 : -1.0;

    for (std::size_t i = 0; i < c.px.size(); ++i)
    {
        const std::size_t j = (i + 1) % c.px.size();
        const cv::Point a = c.px[i], b = c.px[j];
        const float ra = c.depth_m[i], rb = c.depth_m[j];
        const double len = std::hypot(static_cast<double>(b.x - a.x), static_cast<double>(b.y - a.y));
        if (len < 4.0)
            continue;
        // Outward unit normal of this edge, under the contour's own winding.
        const double nx = wind * (b.y - a.y) / len, ny = -wind * (b.x - a.x) / len;
        const int steps = std::min(160, std::max(8, static_cast<int>(len / 3.0)));

        // Perspective-correct depth along the edge: what is linear in image space is INVERSE depth, not
        // depth. Over a 2 m face seen obliquely the difference is tens of centimetres — enough to make
        // a correct belief look wrong at the far end of its own silhouette.
        const bool have_r = ra > 0.0f and rb > 0.0f;
        const double ia = have_r ? 1.0 / ra : 0.0, ib = have_r ? 1.0 / rb : 0.0;

        for (int s = 0; s < steps; ++s)
        {
            const double t = (s + 0.5) / steps;
            const float px = static_cast<float>(a.x + t * (b.x - a.x));
            const float py = static_cast<float>(a.y + t * (b.y - a.y));
            if (not have_r)
                continue;                       // no prediction here ⇒ nothing to test against
            const double inv = ia + t * (ib - ia);
            if (inv <= 1e-6)
                continue;
            const float d_hat = static_cast<float>(1.0 / inv);

            const float d_in  = side_depth(depth, px, py, static_cast<float>(-nx), static_cast<float>(-ny),
                                           p.offset_px_min, p.offset_px_max, p.depth_scale_x, p.depth_scale_y);
            if (not valid_depth(d_in))
                continue;                       // NOT MEASURED — no read, so no evidence either way
            const float d_out = side_depth(depth, px, py, static_cast<float>(nx), static_cast<float>(ny),
                                           p.offset_px_min, p.offset_px_max, p.depth_scale_x, p.depth_scale_y);

            const float sigma_d = std::hypot(p.sigma_depth_m, p.sigma_depth_rel * d_hat);
            const float z = (d_in - d_hat) / std::max(sigma_d, 1e-3f);
            const float agreement = std::exp(-0.5f * z * z);
            // No return beyond the boundary is NOT recession — see the header. 0.5 says nothing, which
            // is different from saying no.
            const float recession = valid_depth(d_out)
                                  ? phi_cdf((d_out - d_in) / std::max(p.sigma_step_m, 1e-3f))
                                  : 0.5f;
            if (not valid_depth(d_out))
                ++out.n_no_return;

            // ★SIGNED BY CONSTRUCTION. 2·A·R − 1 confirms a surface that is where the belief said it
            // is with space behind it, refutes one we can see straight past, and abstains on a flat
            // wall at the right distance — see the header's table. No cutoff separates the three.
            const double ar = static_cast<double>(agreement) * recession;
            out.acc += ar;
            out.verdict_acc += 2.0 * ar - 1.0;
            out.bias_acc += static_cast<double>(d_in - d_hat);
            ++out.n;
        }
    }
    return out;
}

}   // namespace

ContourDepthScore contour_depth_support(const cv::Mat& depth_img,
                                        const Contour& face,
                                        const std::vector<Contour>& controls,
                                        const ContourDepthParams& p)
{
    ContourDepthScore out;
    if (depth_img.empty() or depth_img.type() != CV_32F or not face.valid())
        return out;

    const SideScore t = score_contour(depth_img, face, p);
    out.n_samples   = t.n;
    out.n_no_return = t.n_no_return;
    if (t.n == 0)
        return out;                       // nothing measured; the caller must not read this as "no"
    out.s_true      = static_cast<float>(t.acc / t.n);
    out.verdict     = static_cast<float>(t.verdict_acc / t.n);
    out.mean_bias_m = static_cast<float>(t.bias_acc / t.n);

    double ctl_acc = 0.0;
    int    ctl_n   = 0;
    for (const auto& c : controls)
    {
        const SideScore s = score_contour(depth_img, c, p);
        if (s.n > 0) { ctl_acc += s.acc / s.n; ++ctl_n; }
    }
    out.n_controls = ctl_n;
    out.s_control  = ctl_n > 0 ? static_cast<float>(ctl_acc / ctl_n) : 0.0f;
    // ★`verdict` IS NOT ADJUSTED BY s_control, and that is deliberate — see the header. Subtracting the
    // control is what made the first version unable to refute: with the object gone, the believed
    // contour and every control alike score nothing at the predicted depth, and the difference of two
    // silences is silence. The control is logged, not applied.
    return out;
}

}   // namespace rc::edges
