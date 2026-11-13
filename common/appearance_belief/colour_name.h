/*
 * colour_name.h — an object's colour NAME from its appearance belief: a posterior over a fixed palette.
 *
 * Why (2026-11-10, live chat test chat_e2e2): asked about colours, the vision model named whatever the question
 * suggested ("yellow" for two magenta chairs). A colour said to a person must be a measurement attached to the
 * object. The measurement already exists (retina's per-mask chromaticity + brightness, fused per instance by
 * appearance_belief.h); this header only NAMES it. Design: docs/superpowers/specs/2026-11-10-object-colour-design.md.
 *
 * MODEL — each of 12 basic terms is a Gaussian over (r, g, L): r, g the chromaticity (b = 1 − r − g is redundant)
 * and L the brightness (R+G+B)/3, centred on the term's sRGB prototype. Its covariance is the BELIEF's variance
 * plus the TERM's own width (how wide the word is). Equal priors; the posterior is normalised over the palette.
 * No boundary anywhere: the reported name is the argmax and its probability goes with it, so a colour between two
 * words comes out as one of them at a low probability, which the chat says.
 *
 * Darkness is NOT put into the term's width: retina's chromaticity floor already grows for dark pixels (8-bit
 * quantisation through the normalisation), so a dark object's hue arrives uncertain through the BELIEF. Widening the
 * dark words as well (tried first, 2026-11-11) spread "black" so thin that a narrower "brown" won on a near-black
 * object. Without a brightness measurement the L dimension is simply marginalised out (dropped), and the neutral
 * terms share the probability rather than one of them being guessed.
 *
 * Header-only, Eigen only.
 */
#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <string>
#include <string_view>

#include <Eigen/Dense>

#include "appearance_belief.h"
#include "colour_terms.h"   // kPalette (Eigen-free, shared with language_agent)

namespace rc::appearance
{

/// How wide a word is, the same for every word. ★ Builder's call (design doc): chromaticity 0.04 (about the gap
/// between neighbouring prototypes, e.g. magenta↔purple 0.09, is two widths); brightness 0.15 (shading across one
/// object spans about that). To be checked against one chair's measured spread over a tour.
struct TermWidth
{
    float chroma_sigma = 0.04f;
    float lum_sigma = 0.15f;
};

struct ColourName
{
    std::string name;                    // "" = no colour measured
    float prob = 0.0f;                   // posterior probability of `name`
    std::array<float, kPalette.size()> probs{};   // over kPalette, in its order
};

/// The colour name of an appearance belief. Empty name when the belief has no measurement; a missing chromaticity
/// or brightness is marginalised out (that dimension is dropped), never guessed.
[[nodiscard]] inline ColourName name_colour(const AppearanceBelief& a, const TermWidth& w = {})
{
    ColourName out;
    if (not a.initialized() and not a.lum_initialized())
        return out;
    const bool has_chroma = a.initialized();   // a pure-black object has brightness only
    const Eigen::Vector3f& m = a.map();
    const Eigen::Vector3f& v = a.variance();
    const bool has_lum = a.lum_initialized();

    std::array<double, kPalette.size()> logp{};
    for (std::size_t i = 0; i < kPalette.size(); ++i)
    {
        const auto& t = kPalette[i];
        const float s = t.r + t.g + t.b;
        const float L = s / 3.0f;
        const float sc = w.chroma_sigma;
        double lp = 0.0;
        const auto add = [&lp](double x, double mu, double var)
        { lp += -0.5 * ((x - mu) * (x - mu) / var + std::log(2.0 * std::numbers::pi * var)); };
        if (has_chroma)
        {
            add(m[0], t.r / s, v[0] + sc * sc);
            add(m[1], t.g / s, v[1] + sc * sc);
        }
        if (has_lum)
            add(a.lum(), L, a.lum_variance() + w.lum_sigma * w.lum_sigma);
        logp[i] = lp;
    }
    double mx = logp[0];
    for (const double l : logp) mx = std::max(mx, l);
    double z = 0.0;
    for (const double l : logp) z += std::exp(l - mx);
    std::size_t best = 0;
    for (std::size_t i = 0; i < kPalette.size(); ++i)
    {
        out.probs[i] = static_cast<float>(std::exp(logp[i] - mx) / z);
        if (out.probs[i] > out.probs[best]) best = i;
    }
    out.name = std::string(kPalette[best].name);
    out.prob = out.probs[best];
    return out;
}
}   // namespace rc::appearance
