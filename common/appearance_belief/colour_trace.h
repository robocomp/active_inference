/*
 * colour_trace.h — one diagnostic CSV row per colour update: the slice's colour as it came in, and the belief and
 * its three most probable names after it.
 *
 * Why (2026-11-11, live chat_e2e5): chair_34 (magenta in the world) was named magenta at p 0.34, then drifted to
 * "white" p 0.50 while the robot stood next to it. Nothing recorded the numbers behind that, so it could not be
 * told whether the mask bled onto the wall, the brightness pulled the name, or the term widths are wrong. This
 * row carries the input (chromaticity, its spread and cell count, brightness, range), the frame's probability of
 * being this object (`w`, AppearanceBelief::observe) and the belief it produced, so each can be read off the file.
 *
 * Header-only, Eigen only. The caller imbues the stream with the classic locale (rc::diag::open_rotating does).
 */
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <ostream>
#include <string_view>

#include <Eigen/Dense>

#include "appearance_belief.h"
#include "colour_name.h"

namespace rc::appearance
{

/// What one mask slice contributed to the colour belief (retina's per-mask summary, as ingested).
struct ColourSample
{
    Eigen::Vector3f chroma = Eigen::Vector3f::Zero();
    Eigen::Vector3f chroma_var = Eigen::Vector3f::Zero();
    float chroma_n = 0.0f;
    float lum = 0.0f, lum_var = 0.0f, lum_n = 0.0f;
    float range = 0.0f;   // camera→mask depth (m)
};

inline void write_colour_trace_header(std::ostream& os)
{
    os << "t_ms,node,in_r,in_g,in_b,in_var_r,in_var_g,in_var_b,in_n,in_L,in_var_L,in_n_L,range,w,"
          "bel_r,bel_g,bel_b,bel_var_r,bel_var_g,bel_var_b,bel_L,bel_var_L,name1,p1,name2,p2,name3,p3\n";
}

inline void write_colour_trace_row(std::ostream& os, std::uint64_t t_ms, std::string_view node, const ColourSample& in,
                                   const AppearanceBelief& a)
{
    const auto cn = name_colour(a);
    std::array<std::size_t, kPalette.size()> idx{};
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    std::ranges::partial_sort(idx, idx.begin() + 3, [&](std::size_t x, std::size_t y) { return cn.probs[x] > cn.probs[y]; });

    os << t_ms << ',' << node << ','
       << in.chroma[0] << ',' << in.chroma[1] << ',' << in.chroma[2] << ','
       << in.chroma_var[0] << ',' << in.chroma_var[1] << ',' << in.chroma_var[2] << ',' << in.chroma_n << ','
       << in.lum << ',' << in.lum_var << ',' << in.lum_n << ',' << in.range << ',' << a.last_weight() << ','
       << a.map()[0] << ',' << a.map()[1] << ',' << a.map()[2] << ','
       << a.variance()[0] << ',' << a.variance()[1] << ',' << a.variance()[2] << ','
       << a.lum() << ',' << a.lum_variance();
    for (int k = 0; k < 3; ++k)
    {
        if (cn.name.empty()) os << ",,";
        else os << ',' << kPalette[idx[k]].name << ',' << cn.probs[idx[k]];
    }
    os << '\n';
}
}   // namespace rc::appearance
