// mask_color_summary.h — retina's per-mask colour summary (chromaticity + brightness). Moved out of
// graph_publisher.cpp 2026-11-11 so it can be tested (retina/tests/test_mask_color_summary.cpp).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace rc::retina
{
// Per-slice APPEARANCE summary (MaskColor.*): what the concept agents fold into their per-instance
// appearance belief to tint the display mesh. Three modelling decisions live here, all deliberate:
//
//  1. CHROMATICITY, not raw RGB. Observed = albedo x irradiance + specular. The viewer applies its own
//     ambient+diffuse shading, so publishing observed RGB would bake the room's lighting into the asset
//     and then shade it a second time. Normalising by (R+G+B) is invariant to the per-frame illumination
//     gain by construction — the dominant nuisance removed without inferring a light model.
//  2. GRID CELLS, not pixels. Neighbouring pixels on one surface are massively correlated, so treating
//     each as an independent sample would shrink the variance like 1/sqrt(N) and make this channel
//     absurdly overconfident. One sample per cell, and the BETWEEN-CELL scatter is the reported variance.
//  3. INTERIOR cells only. A cell contributes only if it is fully foreground AND its four neighbours are
//     too — silhouette erosion at cell granularity, which keeps background bleed at the mask edge out of
//     the estimate. Per-cell MEDIAN (not mean) additionally rejects specular highlights inside a cell.
//
// Note the variance floor is not a magic number: it is first-order propagation of 8-bit QUANTISATION
// through the normalisation, so it grows for dark pixels — which is correct, since a dark surface really
// does have poorly determined chromaticity. It also keeps a single-cell slice from claiming zero variance.
struct MaskColorSummary
{
    Eigen::Vector3f chroma = Eigen::Vector3f::Zero();   // median (r,g,b), each in [0,1], summing to 1
    Eigen::Vector3f var    = Eigen::Vector3f::Zero();   // robust between-cell variance, per channel
    float           n_eff  = 0.0f;                      // contributing interior cells (0 ⇒ no information)
    // BRIGHTNESS (2026-11-11, colour names): (R+G+B)/(3*255) — median over each interior cell, then median over
    // cells, with the between-cell robust variance plus the quantisation floor. Counted over its OWN cells:
    // a pure-black cell has no chromaticity but a perfectly good brightness, and a black chair must be nameable.
    float lum = 0.0f, lum_var = 0.0f, lum_n = 0.0f;
};

// bgr: CV_8UC3 in BGR order (RGBDData::rgb is BGR despite the field name — media_plane_source converts
// the RGB8 media-plane view on arrival). mask_bin: full-frame 0/255 silhouette. Bounds are the detection
// bbox already clamped to the image.
inline MaskColorSummary mask_color_summary(const cv::Mat& bgr, const cv::Mat& mask_bin,
                                    int min_x, int min_y, int max_x, int max_y, int cell_px)
{
    MaskColorSummary out;
    if (bgr.empty() or bgr.type() != CV_8UC3 or cell_px <= 0)
        return out;

    const int nx = (max_x - min_x) / cell_px;
    const int ny = (max_y - min_y) / cell_px;
    if (nx <= 0 or ny <= 0)
        return out;

    // Pass 1: a cell is FULL iff every one of its pixels is foreground. That is a definition, not a tuned
    // cutoff — it is what makes the neighbour test below a clean erosion.
    std::vector<char> full(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny), 0);
    for (int cy = 0; cy < ny; ++cy)
        for (int cx = 0; cx < nx; ++cx)
        {
            const int x0 = min_x + cx * cell_px, y0 = min_y + cy * cell_px;
            bool all_fg = true;
            for (int r = y0; r < y0 + cell_px and all_fg; ++r)
                for (int c = x0; c < x0 + cell_px; ++c)
                    if (mask_bin.at<std::uint8_t>(r, c) == 0) { all_fg = false; break; }
            full[static_cast<std::size_t>(cy) * nx + cx] = all_fg ? 1 : 0;
        }

    // Pass 2: one median-chromaticity sample per INTERIOR cell (full, with all four neighbours full).
    std::vector<Eigen::Vector3f> cell_chroma;
    cell_chroma.reserve(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));
    double sum_intensity = 0.0;   // mean (R+G+B)/255 over contributing cells, for the quantisation floor
    std::array<std::vector<float>, 3> ch;
    std::vector<float> lum_px, cell_lum;
    for (int cy = 1; cy < ny - 1; ++cy)
        for (int cx = 1; cx < nx - 1; ++cx)
        {
            const std::size_t idx = static_cast<std::size_t>(cy) * nx + cx;
            if (!full[idx] or !full[idx - 1] or !full[idx + 1]
                or !full[idx - nx] or !full[idx + nx])
                continue;

            const int x0 = min_x + cx * cell_px, y0 = min_y + cy * cell_px;
            for (auto& v : ch) { v.clear(); v.reserve(static_cast<std::size_t>(cell_px) * cell_px); }
            double cell_intensity = 0.0;
            lum_px.clear();
            for (int r = y0; r < y0 + cell_px; ++r)
                for (int c = x0; c < x0 + cell_px; ++c)
                {
                    const cv::Vec3b& p = bgr.at<cv::Vec3b>(r, c);
                    const float b = p[0], g = p[1], rr = p[2];   // BGR order
                    const float s = rr + g + b;
                    lum_px.push_back(s / (3.0f * 255.0f));   // brightness is defined even where chromaticity is not
                    if (s <= 0.0f)          // a pure-black pixel has no defined chromaticity at all
                        continue;
                    ch[0].push_back(rr / s);
                    ch[1].push_back(g  / s);
                    ch[2].push_back(b  / s);
                    cell_intensity += s / 255.0;
                }
            if (not lum_px.empty())
            {
                const std::size_t lm = lum_px.size() / 2;
                std::nth_element(lum_px.begin(), lum_px.begin() + static_cast<std::ptrdiff_t>(lm), lum_px.end());
                cell_lum.push_back(lum_px[lm]);
            }
            if (ch[0].empty())
                continue;

            const std::size_t mid = ch[0].size() / 2;
            Eigen::Vector3f med;
            for (int k = 0; k < 3; ++k)
            {
                std::nth_element(ch[k].begin(), ch[k].begin() + static_cast<std::ptrdiff_t>(mid), ch[k].end());
                med[k] = ch[k][mid];
            }
            cell_chroma.push_back(med);
            sum_intensity += cell_intensity / static_cast<double>(ch[0].size());
        }

    if (not cell_lum.empty())
    {   // Brightness: same robust centre + scale; floor = one LSB of the MEAN of three channels, uniform ⇒ /12, /3.
        const std::size_t lm = cell_lum.size() / 2;
        std::vector<float> lc = cell_lum;
        std::nth_element(lc.begin(), lc.begin() + static_cast<std::ptrdiff_t>(lm), lc.end());
        out.lum = lc[lm];
        for (auto& v : lc) v = std::abs(v - out.lum);
        std::nth_element(lc.begin(), lc.begin() + static_cast<std::ptrdiff_t>(lm), lc.end());
        const float sd = 1.4826f * lc[lm];
        out.lum_var = sd * sd + static_cast<float>((1.0 / 255.0) * (1.0 / 255.0) / 36.0);
        out.lum_n = static_cast<float>(cell_lum.size());
    }
    if (cell_chroma.empty())
        return out;

    // Robust centre + scale across cells: median and (1.4826*MAD)^2, which is the variance consistent with
    // a median centre. A handful of cells landing on a highlight or a thin occluder cannot drag either.
    std::vector<float> comp(cell_chroma.size());
    Eigen::Vector3f mad = Eigen::Vector3f::Zero();
    const std::size_t mid = cell_chroma.size() / 2;
    for (int k = 0; k < 3; ++k)
    {
        for (std::size_t i = 0; i < cell_chroma.size(); ++i) comp[i] = cell_chroma[i][k];
        std::nth_element(comp.begin(), comp.begin() + static_cast<std::ptrdiff_t>(mid), comp.end());
        out.chroma[k] = comp[mid];
        for (std::size_t i = 0; i < cell_chroma.size(); ++i) comp[i] = std::abs(cell_chroma[i][k] - out.chroma[k]);
        std::nth_element(comp.begin(), comp.begin() + static_cast<std::ptrdiff_t>(mid), comp.end());
        mad[k] = comp[mid];
    }

    // Quantisation floor: one LSB is 1/255 of a channel; through r = R/S its first-order effect is
    // (1/255)/S, with S the mean channel sum in [0,3]. Uniform over an LSB ⇒ divide the square by 12.
    const double mean_intensity = std::max(1e-3, sum_intensity / static_cast<double>(cell_chroma.size()));
    const float  quant_var = static_cast<float>((1.0 / (255.0 * mean_intensity)) * (1.0 / (255.0 * mean_intensity)) / 12.0);

    for (int k = 0; k < 3; ++k)
    {
        const float robust_sd = 1.4826f * mad[k];
        out.var[k] = robust_sd * robust_sd + quant_var;
    }
    out.n_eff = static_cast<float>(cell_chroma.size());
    return out;
}
}   // namespace rc::retina
