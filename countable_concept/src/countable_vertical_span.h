/*
 * countable_vertical_span.h — WHERE the class's box stands vertically, stated once (header-only, pure).
 *
 * θ = [cx, cy, H, w, h, yaw]: H is the box TOP above the floor for every class. What differs is the base:
 *   floor_anchored (bed, nightstand, toilet, refrigerator): z ∈ [0, H]           — the extent is H itself;
 *   hangs          (hood):                                  z ∈ [H − e, H]       — e = the manifest's [model.geometry]
 *                                                                                 extent_m, a class PARAMETER, not a DOF.
 * Every site that needs the body's z-band (the SDF, the beam part, the point admission and moment bands, the silhouette,
 * the existence carve, the NBV target, the published RT origin / height_m / mesh) asks this struct instead of writing
 * 0 and H, so a hanging class cannot inherit a floor-anchored assumption site by site (the hood_concept clone lesson,
 * hood.concept.toml [model.geometry]).
 *
 * extent_m ≤ 0 is floor-anchored, the default: every existing class is bit-identical to before.
 */
#pragma once

namespace rc::countable
{
struct VerticalSpan
{
    float extent_m = -1.0f;   // > 0: the box hangs with this vertical extent below its top H; ≤ 0: it stands on the floor

    // Templated on the scalar so the double paths (the beam part) keep their own precision: the floor-anchored values are
    // the exact expressions the code used before (0, H, H/2), in the caller's type.
    bool hangs() const { return extent_m > 0.0f; }
    template <class T> T z0(T H) const { return hangs() ? H - T(extent_m) : T(0); }                // the body's underside
    template <class T> T z1(T H) const { return H; }                                                // the body's top
    template <class T> T zc(T H) const { return hangs() ? H - T(0.5) * T(extent_m) : T(0.5) * H; }
    template <class T> T half(T H) const { return hangs() ? T(0.5) * T(extent_m) : T(0.5) * H; }  // vertical half extent
    template <class T> T height(T H) const { return hangs() ? T(extent_m) : H; }                    // the body's vertical extent
    // ∂zc/∂H and ∂half/∂H: a hanging box translates with its top, a floor-anchored one stretches.
    double dzc_dH() const { return hangs() ? 1.0 : 0.5; }
    double dhalf_dH() const { return hangs() ? 0.0 : 0.5; }
};
}  // namespace rc::countable
