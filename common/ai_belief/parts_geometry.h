/*
 * parts_geometry.h — a box-part class's SOLID, laid out on θ = (cx, cy, H, w, h, yaw), from ONE manifest table
 * (shared-engine K2a; spec 2026-11-23 "beam_parts(θ)").
 *
 * The table is the class manifest's [[shape.part]] rows (fractions of the fitted w, h and of the body band [z0, H]), its back
 * panel ([prior.headboard], absolute height) and its [shape] solid key. Three consumers read it, and before K2a each had its
 * own copy of the geometry:
 *   - the camera SDF            sdf()          (was CountableBelief::sdf_box / sdf_headboard / sdf_compound);
 *   - the beam model's parts    beam_parts()   (was CountableBelief::beam_part, with its analytic J);
 *   - the existence carve       carve_parts()  (was countable_shape_parts.h carve_parts_for).
 *
 * [shape] solid says what the MODEL's solid is (the SDF and the beam parts):
 *   "box"   (default; bed, nightstand, refrigerator, hood): the whole fitted box over the body band, ∪ the back panel. The
 *           [[shape.part]] rows of such a class describe its parts for the CARVE only (legs, mattress layer, footboard),
 *           exactly as before K2a — every one of these classes is bit-identical.
 *   "parts" (toilet; chair and table in K2b/K2c): the union of the [[shape.part]] boxes, ∪ the back panel. The carve reads
 *           the same boxes, so model and carve are one geometry.
 * The carve is always the parts (or the whole box without any), ∪ the panel: what it always was.
 *
 * Local frame (as everywhere in the countable code): x across the width w, +y toward the BACK face, z from the floor;
 * local = R(−yaw)(p − c). The body band [z0, H] comes from the caller's span (countable_vertical_span.h: z0 = 0 floor-anchored,
 * H − e hanging); the part z fractions scale over that band, the panel height is absolute.
 *
 * Header-only, pure (Eigen + the textual manifest structs); the arithmetic of the "box" path is the pre-K2a code's, operation for
 * operation, so the floor classes and the hood replay bit-identically.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <vector>
#include <Eigen/Dense>
#include "beam_ray.h"                                   // rc::ai::beam::BeamPart
#include "../existence_belief/existence_belief.h"       // rc::exist::BoxPart
#include "../concept_manifest/shape_parts.h"            // rc::manifest::ShapePart

namespace rc::ai::parts
{
// The table, as a view (no copy per SDF evaluation).
struct Table
{
    std::span<const rc::manifest::ShapePart> parts;   // [[shape.part]]
    bool  solid_is_parts = false;                      // [shape] solid = "parts"
    float panel_h_m = -1.0f;                           // back panel top (m, from the floor); ≤ 0 = no panel
    float panel_t_m = 0.10f;                           // back panel thickness (m)
    // K2a-L: the class proportions' CURRENT values (the model's latent mean, or what the engine linearises at). A row whose
    // z0_ref / z1_ref names proportion k reads its boundary from q[k]; empty ⇒ every row reads its fixed fraction.
    std::span<const double> q;
    int n_q = 0;                                       // the model's proportion count (the beam parts' Jq columns); 0 ⇒ no Jq

    float z0_of(const rc::manifest::ShapePart& P) const { return P.z0_ref >= 0 and std::size_t(P.z0_ref) < q.size() ? float(q[std::size_t(P.z0_ref)]) : P.z0_f; }
    float z1_of(const rc::manifest::ShapePart& P) const { return P.z1_ref >= 0 and std::size_t(P.z1_ref) < q.size() ? float(q[std::size_t(P.z1_ref)]) : P.z1_f; }

    bool has_panel() const { return panel_h_m > 0.0f; }
    bool model_uses_parts() const { return solid_is_parts and not parts.empty(); }
    // The number of convex boxes in the MODEL's solid (the SDF / beam union). 1 ⇔ the solid is one convex box.
    int  n_model_boxes() const { return (model_uses_parts() ? int(parts.size()) : 1) + (has_panel() ? 1 : 0); }
};

// Box SDF given the point's distances-to-face along each axis (|local| − half extent).
inline float box_sdf(float dx, float dy, float dz)
{
    const float ox = std::max(dx, 0.0f), oy = std::max(dy, 0.0f), oz = std::max(dz, 0.0f);
    const float outside = std::sqrt(ox * ox + oy * oy + oz * oz);
    const float inside  = std::min(std::max(dx, std::max(dy, dz)), 0.0f);
    return outside + inside;
}

// A point in the object's local frame.
struct Local { float lx, ly, z; };
inline Local to_local(const Eigen::Vector3f& p, float cx, float cy, float yaw)
{
    const float c = std::cos(-yaw), sn = std::sin(-yaw);
    const float px = p.x() - cx, py = p.y() - cy;
    return {px * c - py * sn, px * sn + py * c, p.z()};
}

// ── The camera SDF ────────────────────────────────────────────────────────────────────────────────────────────────────────
// The whole fitted box over the body band (Span: zc(H), half(H)).
template <class Span>
float envelope_sdf(const Local& q, float w, float h, float H, const Span& span)
{
    return box_sdf(std::abs(q.lx) - 0.5f * w, std::abs(q.ly) - 0.5f * h, std::abs(q.z - span.zc(H)) - span.half(H));
}
// The back panel: local x ∈ [−w/2, w/2], y ∈ [h/2 − t, h/2], t = min(t_panel, h), z ∈ [0, panel_h_m]. +inf without a panel.
inline float panel_sdf(const Local& q, float w, float h, const Table& T)
{
    if (not T.has_panel()) return std::numeric_limits<float>::infinity();
    const float t  = std::min(T.panel_t_m, h);
    const float yc = 0.5f * h - 0.5f * t;
    const float hz = 0.5f * T.panel_h_m;
    return box_sdf(std::abs(q.lx) - 0.5f * w, std::abs(q.ly - yc) - 0.5f * t, std::abs(q.z - hz) - hz);
}
// One [[shape.part]] row on the fitted box: centre (x_f·w, y_f·h), size (w_f·w, h_f·h), z ∈ z0 + [z0_f, z1_f]·(H − z0).
inline float part_sdf(const Local& q, const rc::manifest::ShapePart& P, float w, float h, float z0, float H, const Table& T)
{
    const float dz = H - z0;
    const float za = z0 + T.z0_of(P) * dz, zb = z0 + T.z1_of(P) * dz;
    const float zc = 0.5f * (za + zb), hz = 0.5f * (zb - za);
    return box_sdf(std::abs(q.lx - P.x_f * w) - 0.5f * (P.w_f * w), std::abs(q.ly - P.y_f * h) - 0.5f * (P.h_f * h),
                   std::abs(q.z - zc) - hz);
}
// The model solid's SDF: min over its boxes (the union).
template <class Span>
float sdf(const Eigen::Vector3f& p, float cx, float cy, float H, float w, float h, float yaw, const Table& T, const Span& span)
{
    const Local q = to_local(p, cx, cy, yaw);
    float d = std::numeric_limits<float>::infinity();
    if (T.model_uses_parts())
    {
        const float z0 = span.z0(H);
        for (const auto& P : T.parts) d = std::min(d, part_sdf(q, P, w, h, z0, H, T));
    }
    else d = envelope_sdf(q, w, h, H, span);
    return std::min(d, panel_sdf(q, w, h, T));
}

// ── The beam model's parts ────────────────────────────────────────────────────────────────────────────────────────────────
// Rows of J: ∂(cx, cy, cz, yaw, hx, hy, hz)/∂θ, θ = (cx, cy, H, w, h, yaw).
// The envelope: c = (cx, cy, zc), half = (w/2, h/2, hz), (zc, hz) = (H/2, H/2) floor-anchored, (H − e/2, e/2) hanging.
template <class Span>
rc::ai::beam::BeamPart envelope_beam_part(double cx, double cy, double H, double w, double h, double yaw, const Span& span)
{
    rc::ai::beam::BeamPart P;
    P.yaw = yaw;
    P.J.setZero();
    P.c = {cx, cy, span.zc(H)};
    P.half = {0.5 * w, 0.5 * h, span.half(H)};
    P.J(0, 0) = 1.0; P.J(1, 1) = 1.0; P.J(2, 2) = span.dzc_dH(); P.J(3, 5) = 1.0; P.J(4, 3) = 0.5; P.J(5, 4) = 0.5;
    P.J(6, 2) = span.dhalf_dH();
    return P;
}
// The panel: t = min(t_panel, h), local centre y_c = h/2 − t/2, so c = (cx − sin ψ·y_c, cy + cos ψ·y_c, H_p/2), half = (w/2, t/2,
// H_p/2); while t_panel < h, ∂y_c/∂h = ½ and ∂t/∂h = 0; else (t = h) y_c = 0 and ∂(t/2)/∂h = ½ (one-sided at t_panel = h, as
// the SDF's min).
inline rc::ai::beam::BeamPart panel_beam_part(double cx, double cy, double w, double h, double yaw, const Table& T)
{
    rc::ai::beam::BeamPart P;
    P.yaw = yaw;
    P.J.setZero();
    const double t_hb = T.panel_t_m, H_hb = T.panel_h_m;
    const bool thin = t_hb < h;
    const double t = thin ? t_hb : h, yc = 0.5 * h - 0.5 * t, dyc = thin ? 0.5 : 0.0, dhalf_t = thin ? 0.0 : 0.5;
    const double c = std::cos(yaw), sn = std::sin(yaw);
    P.c = {cx - sn * yc, cy + c * yc, 0.5 * H_hb};
    P.half = {0.5 * w, 0.5 * t, 0.5 * H_hb};
    P.J(0, 0) = 1.0; P.J(0, 4) = -sn * dyc; P.J(0, 5) = -c * yc;
    P.J(1, 1) = 1.0; P.J(1, 4) =  c * dyc;  P.J(1, 5) = -sn * yc;
    P.J(3, 5) = 1.0; P.J(4, 3) = 0.5; P.J(5, 4) = dhalf_t;
    return P;
}
// A [[shape.part]] row: local centre (xc, yc) = (x_f·w, y_f·h), world centre (cx + cos ψ·xc − sin ψ·yc, cy + sin ψ·xc + cos ψ·yc);
// z band z0 + [z0_f, z1_f]·(H − z0), so zc = z0 + m·(H − z0), hz = ½(z1_f − z0_f)(H − z0), m = ½(z0_f + z1_f), with
// ∂z0/∂H = ∂zc/∂H − ∂half/∂H of the span (0 floor-anchored, 1 hanging).
// K2a-L: Jq = ∂(…)/∂q for the rows whose boundary is a proportion: z0 = zb + f₀·dz, z1 = zb + f₁·dz ⇒ ∂zc/∂q_k = ½dz·([ref₀ = k]
// + [ref₁ = k]) and ∂hz/∂q_k = ½dz·([ref₁ = k] − [ref₀ = k]) (rows 2 and 6); n_q = 0 ⇒ no Jq.
template <class Span>
rc::ai::beam::BeamPart shape_beam_part(const rc::manifest::ShapePart& S, double cx, double cy, double H, double w, double h,
                                       double yaw, const Span& span, const Table& T)
{
    rc::ai::beam::BeamPart P;
    P.yaw = yaw;
    P.J.setZero();
    const double c = std::cos(yaw), sn = std::sin(yaw);
    const double xf = S.x_f, yf = S.y_f, xc = xf * w, yc = yf * h;
    const double z0 = span.z0(H), dz = H - z0, dz0 = span.dzc_dH() - span.dhalf_dH();
    const double f0 = T.z0_of(S), f1 = T.z1_of(S);
    const double m = 0.5 * (f0 + f1), e = 0.5 * (f1 - f0);
    P.c = {cx + c * xc - sn * yc, cy + sn * xc + c * yc, z0 + m * dz};
    P.half = {0.5 * double(S.w_f) * w, 0.5 * double(S.h_f) * h, e * dz};
    P.J(0, 0) = 1.0; P.J(0, 3) = c * xf;  P.J(0, 4) = -sn * yf; P.J(0, 5) = -sn * xc - c * yc;
    P.J(1, 1) = 1.0; P.J(1, 3) = sn * xf; P.J(1, 4) =  c * yf;  P.J(1, 5) =  c * xc - sn * yc;
    P.J(2, 2) = dz0 + m * (1.0 - dz0);
    P.J(3, 5) = 1.0;
    P.J(4, 3) = 0.5 * double(S.w_f);
    P.J(5, 4) = 0.5 * double(S.h_f);
    P.J(6, 2) = e * (1.0 - dz0);
    if (T.n_q > 0)
    {
        P.Jq = Eigen::Matrix<double, 7, Eigen::Dynamic>::Zero(7, T.n_q);
        for (int k = 0; k < T.n_q; ++k)
        {
            const double a0 = S.z0_ref == k ? 1.0 : 0.0, a1 = S.z1_ref == k ? 1.0 : 0.0;
            P.Jq(2, k) = 0.5 * dz * (a0 + a1);
            P.Jq(6, k) = 0.5 * dz * (a1 - a0);
        }
    }
    return P;
}
// The model solid's convex parts, in the order the SDF takes its min: the envelope or the [[shape.part]] rows, then the panel.
template <class Span>
std::vector<rc::ai::beam::BeamPart> beam_parts(double cx, double cy, double H, double w, double h, double yaw, const Table& T,
                                               const Span& span)
{
    std::vector<rc::ai::beam::BeamPart> out;
    out.reserve(std::size_t(T.n_model_boxes()));
    if (T.model_uses_parts())
        for (const auto& S : T.parts) out.push_back(shape_beam_part(S, cx, cy, H, w, h, yaw, span, T));
    else out.push_back(envelope_beam_part(cx, cy, H, w, h, yaw, span));
    if (T.has_panel()) out.push_back(panel_beam_part(cx, cy, w, h, yaw, T));
    return out;
}

// ── The existence carve ───────────────────────────────────────────────────────────────────────────────────────────────────
// No rows ⇒ one whole-box part over [z0, H]. The panel is appended as {0, h/2 − t/2, w, t, 0, panel_h_m} (absolute height).
// z0 = the body's underside (K1): 0 floor-anchored (0 + f·(H − 0) = f·H, bit-identical), H − e hanging; the rows' z fractions
// scale over [z0, H], never over the floor → top column.
inline std::vector<rc::exist::BoxPart> carve_parts(float w, float h, float H, const Table& T, float z0 = 0.0f)
{
    std::vector<rc::exist::BoxPart> out;
    const float dz = H - z0;
    if (T.parts.empty()) out.push_back({0.0f, 0.0f, w, h, z0, H});
    for (const auto& p : T.parts)
        out.push_back({p.x_f * w, p.y_f * h, p.w_f * w, p.h_f * h, z0 + T.z0_of(p) * dz, z0 + T.z1_of(p) * dz});
    if (T.has_panel())
    {
        const float t = std::min(T.panel_t_m, h);
        out.push_back({0.0f, h * 0.5f - t * 0.5f, w, t, 0.0f, T.panel_h_m});
    }
    return out;
}
}  // namespace rc::ai::parts
