/*
 *  reloc_search.cpp — see reloc_search.h for the model and why it replaces the lattice.
 */
#include "reloc_search.h"
#include "wall_segmenter.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numbers>
#include <random>
#include <ranges>
#include <unordered_map>
#include <cstdlib>
#include <charconv>
#include <cstring>

namespace rc::reloc
{
    namespace
    {
        constexpr float kPi  = std::numbers::pi_v<float>;
        constexpr float kInf = std::numeric_limits<float>::infinity();

        float wrap(float a) { return std::remainder(a, 2.f * kPi); }

        /// log[(1−ε)·2·N(d; 0, σ²) + ε/L] — the per-point term of ℓ, and the inlier responsibility.
        struct PointTerm
        {
            float log_in_norm;   // log((1−ε)·2/(√(2π)σ))
            float log_out;       // log(ε/L)
            float inv_2s2;
            PointTerm(const Params& p, float extent)
                : log_in_norm(std::log((1.f - p.outlier_frac) * 2.f / (std::sqrt(2.f * kPi) * p.sigma_sdf))),
                  log_out(std::log(p.outlier_frac / std::max(extent, 1e-3f))),
                  inv_2s2(1.f / (2.f * p.sigma_sdf * p.sigma_sdf)) {}
            float ll(float d, float* gamma = nullptr, float* li_out = nullptr) const
            {
                const float li = log_in_norm - d * d * inv_2s2;
                const float hi = std::max(li, log_out), lo = std::min(li, log_out);
                const float out = hi + std::log1p(std::exp(lo - hi));
                if (gamma != nullptr) *gamma = std::exp(li - out);
                if (li_out != nullptr) *li_out = li;
                return out;
            }
        };

        /// The opening model is on when the room lists a door.
        bool openings_on(const Params& p) { return not p.apertures.empty(); }

        float lse(float x, float y)
        {
            if (x == -kInf) return y;
            if (y == -kInf) return x;
            const float hi = std::max(x, y), lo = std::min(x, y);
            return hi + std::log1p(std::exp(lo - hi));
        }

        /// Do the segments p→p+r and q→q+s cross, with the crossing strictly before p+r? (DoorApertures' test.)
        bool crosses(const Eigen::Vector2f& p, const Eigen::Vector2f& r, const Eigen::Vector2f& q, const Eigen::Vector2f& s)
        {
            const float rxs = r.x() * s.y() - r.y() * s.x();
            if (std::abs(rxs) < 1e-9f) return false;
            const Eigen::Vector2f qp = q - p;
            const float t = (qp.x() * s.y() - qp.y() * s.x()) / rxs;
            const float u = (qp.x() * r.y() - qp.y() * r.x()) / rxs;
            return t > 1e-4f and t < 1.f - 1e-4f and u >= 0.f and u <= 1.f;
        }

        /// Weighted median of d (weights ≥ 0); the plain median when every weight is equal.
        float weighted_median(std::vector<std::pair<float, float>>& dw)
        {
            if (dw.empty()) return std::numeric_limits<float>::quiet_NaN();
            std::ranges::sort(dw, {}, &std::pair<float, float>::first);
            // Equal weights (no doorway in play): the plain median, the same element the unweighted code returned.
            if (std::ranges::all_of(dw, [&](const auto& x) { return x.second == dw.front().second; }))
                return dw[dw.size() / 2].first;
            double tot = 0.0;
            for (const auto& [d, w] : dw) tot += w;
            if (not (tot > 0.0)) return dw[dw.size() / 2].first;
            double acc = 0.0;
            for (const auto& [d, w] : dw)
                if ((acc += w) >= 0.5 * tot) return d;
            return dw.back().first;
        }

        /// ── EACH LISTED DOOR IS ONE OPENING OF UNCERTAIN PLACE (fix round 1, I-1) ───────────────────────────────────
        /// The boundary is cut into cells of about Params::opening_cell_m (a doorway's width). Door k is shut (q_k) or
        /// open in exactly ONE cell c with prior π_k(c) = (1 − w)·cover_k(c)/Σ cover_k + w/N_cells: where door_concept
        /// lists it, smeared by its own position σ along the wall (cover), or — w, "the list is wrong here" — anywhere.
        struct Cells
        {
            std::vector<int>   first, count;          ///< per polygon edge: its first cell and how many
            std::vector<float> s0, s1;                ///< per cell: its span along its edge (m)
            std::vector<float> leaf;                  ///< per cell: P(leaf | open), from the doors that list it
            std::vector<std::vector<float>> pi;       ///< per door: open in cell c alone, where door_concept lists it
            std::vector<std::pair<int, int>> windows; ///< generic openings: cells [first, last] of one edge, ~one door wide
            std::vector<float> wgen;                  ///< per door: prior mass of ONE generic window
            std::vector<float> q;                     ///< per door: P(shut)
        };
        const Cells kNoCells{};

        /// ── WHERE AN OPENING CAN BE (Task 7c) ───────────────────────────────────────────────────────────────────────────
        /// ★ A listed door's MOUTH is one cell: the cells are cut at its ends. With a uniform 1 m grid a door straddled two
        ///   cells and, as ONE opening in ONE cell, explained only one side's beams; the rest paid "through a mapped wall"
        ///   (reloc_selftest Q: a 1 m door at x ∈ [0.7, 1.7] split 0.3 / 0.7, 61 nats lost at the truth). The beam model
        ///   charges a return long of a wall honestly, so the opening must be where the door is.
        /// ★ A door "anywhere" (list_wrong) is an opening ~opening_cell_m wide at an UNKNOWN position: a WINDOW of consecutive
        ///   cells of one edge, the edge cut into cells of opening_cell_m / kSub so that the position is marginalised at that
        ///   resolution (a fixed 1 m grid made the result depend on how a pose's grid happened to align with the real door:
        ///   reloc_selftest S, the truth beaten by the 180° twin). The list_wrong mass is uniform over the windows. Inside a
        ///   window there is no wall (the window IS the opening); the door's position is ONE fact for all its beams.
        /// ⚠ kSub is the resolution of a numerical integral over the door's position, not a model constant: the residual
        ///   misalignment is ≤ opening_cell_m / (2·kSub) per side.
        constexpr int kSub = 4;
        Cells build_cells(const std::vector<Eigen::Vector2f>& poly, const Params& p)
        {
            Cells c;
            const std::size_t ne = poly.size();
            c.first.resize(ne); c.count.resize(ne);
            const float cell = std::max(p.opening_cell_m, 1e-2f), sub = cell / static_cast<float>(kSub);
            std::vector<std::size_t> cell_edge;
            struct Mouth { std::size_t e; float t0, t1; };
            std::vector<Mouth> mouths;
            for (const auto& ap : p.apertures)
            {
                // The door belongs to the edge nearest its centre (where seat_in_room put it), as [t0, t1] along it.
                const Eigen::Vector2f m = 0.5f * (ap.a + ap.b);
                float best = kInf; std::size_t be = 0;
                for (std::size_t e = 0; e < ne; ++e)
                {
                    const Eigen::Vector2f e0 = poly[e], ab = poly[(e + 1) % ne] - e0;
                    const float L2 = ab.squaredNorm();
                    if (L2 < 1e-12f) continue;
                    const float t = std::clamp((m - e0).dot(ab) / L2, 0.f, 1.f);
                    if (const float d = (m - e0 - t * ab).norm(); d < best) { best = d; be = e; }
                }
                const Eigen::Vector2f e0 = poly[be], u = (poly[(be + 1) % ne] - e0).normalized();
                mouths.push_back({be, std::min((ap.a - e0).dot(u), (ap.b - e0).dot(u)), std::max((ap.a - e0).dot(u), (ap.b - e0).dot(u))});
            }
            for (std::size_t e = 0; e < ne; ++e)
            {
                const float L = (poly[(e + 1) % ne] - poly[e]).norm();
                std::vector<float> br{0.f, L};
                for (const auto& mo : mouths)
                    if (mo.e == e)
                        for (const float t : {mo.t0, mo.t1}) br.push_back(std::clamp(t, 0.f, L));
                std::ranges::sort(br);
                c.first[e] = static_cast<int>(c.s0.size());
                for (std::size_t k = 0; k + 1 < br.size(); ++k)
                {
                    const float a = br[k], b = br[k + 1];
                    if (b - a < 1e-3f) continue;   // numerical: coincident breakpoints
                    const float mid = 0.5f * (a + b);
                    const bool in_mouth = std::ranges::any_of(mouths, [&](const Mouth& mo) { return mo.e == e and mid > mo.t0 and mid < mo.t1; });
                    const int n = in_mouth ? 1 : std::max(1, static_cast<int>(std::ceil((b - a) / sub)));
                    for (int j = 0; j < n; ++j)
                    {
                        c.s0.push_back(a + (b - a) * static_cast<float>(j) / static_cast<float>(n));
                        c.s1.push_back(a + (b - a) * static_cast<float>(j + 1) / static_cast<float>(n));
                        cell_edge.push_back(e);
                    }
                }
                if (static_cast<int>(c.s0.size()) == c.first[e])   // a degenerate edge still owns one cell
                {
                    c.s0.push_back(0.f); c.s1.push_back(L); cell_edge.push_back(e);
                }
                c.count[e] = static_cast<int>(c.s0.size()) - c.first[e];
                // generic windows of this edge: from each cell, consecutive cells until they span a door's width (the
                // edge's end cuts a window short: a wall shorter than a door is opened whole)
                for (int i = c.first[e]; i < c.first[e] + c.count[e]; ++i)
                {
                    int j = i;
                    while (j + 1 < c.first[e] + c.count[e] and c.s1[static_cast<std::size_t>(j)] - c.s0[static_cast<std::size_t>(i)] < cell - 0.5f * sub) ++j;
                    c.windows.emplace_back(i, j);
                    if (c.s1[static_cast<std::size_t>(j)] - c.s0[static_cast<std::size_t>(i)] < cell - 0.5f * sub) break;   // the rest are shorter
                }
            }
            const std::size_t N = c.s0.size();
            c.leaf.assign(N, 0.f);
            std::vector<float> leaf_w(N, 0.f);
            auto Phi = [](float z) { return 0.5f * std::erfc(-z / std::sqrt(2.f)); };
            const float w = std::clamp(p.list_wrong, 0.f, 1.f);
            const float n_win = static_cast<float>(std::max<std::size_t>(c.windows.size(), 1));
            for (std::size_t k = 0; k < p.apertures.size(); ++k)
            {
                const auto& ap = p.apertures[k];
                const auto& mo = mouths[k];
                std::vector<float> cover(N, 0.f);
                float csum = 0.f;
                for (std::size_t ci = 0; ci < N; ++ci)
                {
                    if (cell_edge[ci] != mo.e) continue;
                    // mean over the cell of P(this point of wall is inside the mouth), the mouth uncertain by door_concept's
                    // own σ along the wall (8-point quadrature).
                    float cv = 0.f;
                    for (int j = 0; j < 8; ++j)
                    {
                        const float sx = c.s0[ci] + (static_cast<float>(j) + 0.5f) / 8.f * (c.s1[ci] - c.s0[ci]);
                        if (ap.sigma_along > 1e-4f) cv += Phi((mo.t1 - sx) / ap.sigma_along) - Phi((mo.t0 - sx) / ap.sigma_along);
                        else cv += (sx >= mo.t0 and sx <= mo.t1) ? 1.f : 0.f;
                    }
                    cover[ci] = cv / 8.f;
                    csum += cover[ci];
                }
                std::vector<float> pi(N, 0.f);
                for (std::size_t ci = 0; ci < N; ++ci)
                {
                    pi[ci] = csum > 0.f ? (1.f - w) * cover[ci] / csum : 0.f;
                    c.leaf[ci] += cover[ci] * std::clamp(1.f - ap.open, 0.f, 1.f);
                    leaf_w[ci] += cover[ci];
                }
                c.pi.push_back(std::move(pi));
                c.wgen.push_back((csum > 0.f ? w : 1.f) / n_win);
                c.q.push_back(std::clamp(ap.p_shut, 0.f, 1.f));
            }
            for (std::size_t ci = 0; ci < N; ++ci) c.leaf[ci] = leaf_w[ci] > 0.f ? c.leaf[ci] / leaf_w[ci] : 0.f;
            return c;
        }

        /// build_cells for the last (polygon, doors) this THREAD scored: every ℓ of a frame — the incumbent, each rival, each
        /// polish step — shares them, and with quarter-door cells rebuilding them was half of ℓ's cost. A cache, not state:
        /// the key is the full input, compared exactly.
        const Cells& cells_for(const std::vector<Eigen::Vector2f>& poly, const Params& p)
        {
            struct Key
            {
                std::vector<Eigen::Vector2f> poly;
                std::vector<Aperture> aps;
                float cell = 0.f, list_wrong = 0.f;
            };
            thread_local Key key;
            thread_local Cells cached;
            thread_local bool valid = false;
            auto same_ap = [](const Aperture& x, const Aperture& y)
            {
                return x.a == y.a and x.b == y.b and x.n_out == y.n_out and x.open == y.open and x.p_shut == y.p_shut
                       and x.sigma_along == y.sigma_along;
            };
            const bool hit = valid and key.cell == p.opening_cell_m and key.list_wrong == p.list_wrong and key.poly == poly
                             and key.aps.size() == p.apertures.size()
                             and std::ranges::equal(key.aps, p.apertures, same_ap);
            if (not hit)
            {
                cached = build_cells(poly, p);
                key = {poly, p.apertures, p.opening_cell_m, p.list_wrong};
                valid = true;
            }
            return cached;
        }

        /// ── ONE BEAM, CAST FROM THE SENSOR AT THE HYPOTHESISED POSE (Task 7c) ──────────────────────────────────────────
        /// The first crossing of the RAY origin→q with the room boundary, at any distance (also past the return): z = the
        /// measured range, zs = z* = the range the map predicts (∞ when the ray meets no wall), the wall it meets, |cos| of
        /// the incidence angle, its unit normal, `passed` = the return lies PAST that wall (z > z*), and the boundary cell
        /// it meets (−1 without cells or with the origin outside the room). Every beam belongs to the cell its ray meets:
        /// an open cell predicts that its beams pass it.
        struct Ray
        {
            float z = 0.f, zs = kInf, cosa = 1.f;
            float z2 = kInf, cos2 = 1.f;   ///< the NEXT mapped wall along the ray past z* (its back face seen from beyond)
            int edge = -1, cell = -1;
            bool passed = false;
            Eigen::Vector2f n{0.f, 0.f};
        };
        Ray cast_ray(const std::vector<Eigen::Vector2f>& poly, const Cells* c, const Eigen::Vector2f& origin,
                     bool origin_inside, const Eigen::Vector2f& q);

        /// ── THE BEAM MODEL OF ONE RETURN (Task 7c; Thrun, Burgard & Fox, Probabilistic Robotics §6.3, on the room map) ──
        /// Densities over the measured range z along the beam, z* its predicted range (cast_ray), σ_r = σ/|cos α| the range
        /// noise of a wall whose PERPENDICULAR noise is σ (the map's wall-band σ; its per-wall common-mode offset is
        /// marginalised from each mode's covariance, polish()):
        ///   p_short(z) = λ·e^{−λz} on [0, z*)       an UNMAPPED surface (furniture, a leaf, a person) meets the beam first:
        ///                                           surfaces scattered along the ray at λ per metre, the first one at z.
        ///                                           Its mass 1 − e^{−λz*} is the chance the beam meets one before the wall;
        ///                                           normalised on [0, z*) it is λe^{−λz}/(1 − e^{−λz*}) times that mass.
        ///   p_hit(z)   = e^{−λz*}·N(z; z*, σ_r²)    nothing unmapped in the way (e^{−λz*}), and the beam returns from the wall.
        ///   p_beyond(z)= e^{−λz}·[N(z; z_line, σ_r'²) + λ],  z > z*
        ///                                           the wall is an OPEN doorway: the beam goes on into the next room and the
        ///                                           SAME beam model applies there — it survives z metres of unmapped
        ///                                           surfaces (e^{−λz}) and returns from a next-room wall, a line fitted to the
        ///                                           scan (BeyondLines; σ_r' = σ/|cos| on that line; each line pays its Occam
        ///                                           marginal in cell_mixture), or from an unmapped surface (λ). The next
        ///                                           room ends at the next MAPPED wall on the ray, z2 (this room's wall seen
        ///                                           from behind): support [z*, z2), plus e^{−λz2}·N(z; z2, σ_r2²) for the
        ///                                           return from that wall. Only an open cell admits it (cell_mixture).
        ///   p_rand(z)  = 1/L                        uniform clutter over the room's extent (RelocOutlierFrac's mixture).
        /// Shut wall:  p = (1−ε)·[p_short + p_hit] + ε/L.
        /// Open cell:  p = (1−c)·o + c·shut, o = (1−ε)·[p_short + p_beyond] + ε/L (c = P(leaf | open), cell_mixture).
        /// p_short + p_hit integrates to 1 over z (up to the Gaussian's tail below 0); so does the open alternative.
        /// ★ WHY (warm44, 7b report): a return SHORT of the wall is cheap under the true pose (furniture has a proper
        ///   density, not uniform clutter); a return LONG of a wall — through a wall the map says is there — costs the
        ///   Gaussian tail or clutter unless a doorway admits it. Under a twin standing at a wall, the truth's furniture
        ///   returns land long of other walls and pay for it, where the old per-point SDF band let a return near ANY wall,
        ///   even one its ray had already crossed, score as a wall hit.
        struct BeamTerm
        {
            float l1e, lclut, lam, loglam, lhit, inv_2s2;
            BeamTerm(const Params& p, float sigma, float extent)
                : l1e(std::log1p(-std::clamp(p.outlier_frac, 0.f, 1.f - 1e-6f))),
                  lclut(std::log(std::max(p.outlier_frac, 1e-6f) / std::max(extent, 1e-3f))),
                  lam(std::max(p.unmapped_rate, 0.f)),
                  loglam(p.unmapped_rate > 0.f ? std::log(p.unmapped_rate) : -kInf),
                  lhit(-std::log(std::sqrt(2.f * kPi) * sigma)),
                  inv_2s2(1.f / (2.f * sigma * sigma)) {}
            /// shut = log p(z | wall shut); open = log o (the alternative an open cell offers); hit = log (1−ε)·p_hit
            /// (the wall component, for the IRLS responsibility). r_line: distance to the return's own fitted line (∞: none),
            /// cos_line: |cos| of the beam's incidence on that line.
            void eval(const Ray& ry, float r_line, float cos_line, float& shut, float& open, float& hit) const
            {
                const float ls = (ry.z < ry.zs) ? loglam - lam * ry.z : -kInf;
                float lh = -kInf, lb = -kInf;
                if (std::isfinite(ry.zs))
                {
                    const float cz = std::max(ry.cosa, 1e-3f);   // numerical guard: σ_r = σ/cos stays finite at grazing
                    const float d = (ry.z - ry.zs) * cz;          // the return's perpendicular offset from the wall
                    lh = -lam * ry.zs + std::log(cz) + lhit - d * d * inv_2s2;
                    if (ry.passed)
                    {
                        // The next room is a room: the SAME beam model past the doorway, with its walls fitted to the scan.
                        // The beam crosses z − z* metres of it before returning, so it survives its unmapped surfaces with
                        // e^{−λ(z−z*)} — exactly as a beam in this room pays e^{−λz*} to reach its wall. Then it returns from a
                        // next-room wall, N(z; z_line, σ²/cos²) = cos·N(r; 0, σ²) (r = the return's offset from its fitted
                        // line), or from an unmapped surface there, λ. Per return "beyond" can never out-score "on a known
                        // wall"; the line's own parameters pay their Occam marginal in cell_mixture.
                        // ★ The next room ENDS at the next mapped wall on the ray (z2: the back of a wall of THIS room, the beam
                        //   re-entering it): its surfaces lie on [z*, z2), and the beam returns from that wall if nothing came
                        //   first, e^{−λz2}·N(z; z2, σ²/cos2²). A return past z2 went through a second wall the map says is there
                        //   — no doorway admitted it (a beam is assigned to the first cell it meets), so only clutter explains
                        //   it. Without this a pose at a wall opened the cell at its feet and called returns from the far side
                        //   of the ROOM ITSELF "the next room" (warm44 f677: 26 of the twin's beams crossed 2-3 mapped walls).
                        const float cl = std::max(cos_line, 1e-3f);   // numerical guard, as for p_hit
                        const float lwall = std::isfinite(r_line) ? std::log(cl) + lhit - r_line * r_line * inv_2s2 : -kInf;
                        const float lnext = ry.z < ry.z2 ? -lam * ry.z + lse(lwall, loglam) : -kInf;   // e^{−λz*}·e^{−λ(z−z*)}
                        float lback = -kInf;
                        if (std::isfinite(ry.z2))
                        {
                            const float c2 = std::max(ry.cos2, 1e-3f);
                            const float d2 = (ry.z - ry.z2) * c2;
                            lback = -lam * ry.z2 + std::log(c2) + lhit - d2 * d2 * inv_2s2;
                        }
                        lb = lse(lnext, lback);
                    }
                }
                hit  = l1e + lh;
                shut = lse(l1e + lse(ls, lh), lclut);
                open = lse(l1e + lse(ls, lb), lclut);
            }
        };

        /// ℓ over a scan given each point's room term and the cell its ray reaches; fills room_post[i] = P(room return).
        ///   Per cell C with beams S_C: shut ⇒ Π room_i; open ⇒ Π_i ((1−c)·o_i + c·room_i)·e^{−Occam(lines)}, o_i =
        ///   the open alternative of BeamTerm (an unmapped surface short of the cell, or the next room past it).
        ///   Δ_C = log open − log shut.
        ///   ℓ = Σ_i room_i + log Σ_{c_1..c_K} Π_k prior_k(c_k)·Π_{C ∈ ∪_k c_k} e^{Δ_C}   (cells without beams: Δ = 0;
        ///   prior_k(shut) = q_k, prior_k(C) = (1 − q_k)·π_k(C)). Each open cell's beams count ONCE (Task 7b, below).
        /// ★ ONE OPENING PER LISTED DOOR, COMMON TO ITS BEAMS. A door's state is one fact shared by every beam through it
        ///   (a per-beam state charged a shut door wrongly listed open 199 nats, Q's exploration), and the room has as
        ///   many openings as doors are listed: a wrong pose cannot open the boundary wherever its scan crosses it. A
        ///   generic "any wall cell may be open" prior was tried in fix round 1 and measured: with the next-room line
        ///   density a wrong pose explained its boundary violations as openings for ~14 nats each against ~5 nats per
        ///   return, and kidnap recovery fell (F on L 14/15 → 8/15; J rect/L/notched 20/20 → 15–17/20).
        float cell_mixture(const std::vector<float>& room_ll, const std::vector<float>& open_ll, const std::vector<int>& cell_of,
                           const std::vector<char>& passed, const Cells& cells, const Params& p, float sigma,
                           const BeyondLines& bl, std::vector<float>& room_post)
        {
            const std::size_t n = room_ll.size();
            room_post.assign(n, 1.f);
            const float sig2 = sigma * sigma;
            double ll = 0.0;
            std::unordered_map<int, std::vector<std::size_t>> groups;
            for (std::size_t i = 0; i < n; ++i)
            {
                ll += room_ll[i];
                if (cell_of[i] >= 0) groups[cell_of[i]].push_back(i);
            }
            std::unordered_map<int, double> delta;
            for (const auto& [cid, idx] : groups)
            {
                const float cf = cells.leaf[static_cast<std::size_t>(cid)];
                const float lc = cf > 0.f ? std::log(cf) : -kInf, l1c = cf < 1.f ? std::log1p(-cf) : -kInf;
                double so = 0.0, ss = 0.0;
                struct LineUse { int n = 0; double st = 0.0, st2 = 0.0; };
                std::unordered_map<int, LineUse> lines;
                for (const std::size_t i : idx)
                {
                    so += lse(l1c + open_ll[i], lc + room_ll[i]);
                    ss += room_ll[i];
                    if (passed[i] and i < bl.seg.size() and bl.seg[i] >= 0)
                    {
                        auto& u = lines[bl.seg[i]];
                        ++u.n; u.st += bl.t[i]; u.st2 += static_cast<double>(bl.t[i]) * bl.t[i];
                    }
                }
                // The full marginal of each next-room line these beams use (BeyondLines; fix round 1, I-2).
                for (const auto& [line, u] : lines)
                {
                    const double nn = u.n, mean = u.st / nn;
                    const double var_t = std::max(u.st2 / nn - mean * mean, static_cast<double>(sig2) / 12.0);
                    so -= std::log(M_PI * p.beyond_extent) - std::log(2.0 * M_PI)
                          + 0.5 * std::log(nn / sig2) + 0.5 * std::log(nn * var_t / sig2);
                }
                delta[cid] = so - ss;
            }
            // THE DOORS' OPENINGS, MARGINALISED JOINTLY (Task 7b). Door k is shut (q_k) or open in ONE cell c_k (prior
            // (1 − q_k)·π_k(c_k)); the boundary is open on the UNION U of the doors' cells, and a cell's beams are evidence
            // ONCE, however many doors name it:  Z = Σ_{c_1..c_K} Π_k prior_k(c_k) · Π_{C∈U} e^{Δ_C}.
            // ⚠ The old Σ_k log Z_k (each door marginalised on its own) is this sum only when no two doors can reach the same
            //   cell. Every door reaches EVERY cell through list_wrong, so the cell a wrong pose stands next to had its e^{Δ}
            //   multiplied once per listed door: warm43 f358's 178° twin took Δ = 219 nats twice (ℓ 293 vs the truth's 191),
            //   f459's Δ = 391 twice, warm45 f1651's Δ = 285 twice — and listing one doorway under two door nodes doubled its
            //   evidence even at the truth. reloc_selftest X.
            // Exact, by a forward-backward recursion over the cells with beams whose state is the SET of doors already
            // placed (2^K states, 3^K transitions a cell; K = the room's listed doors, a handful). Doors whose cell has no
            // beam contribute their mass  off_k = q_k + (1 − q_k)·Σ_{C without beams} π_k(C)  (Δ = 0 there). With every
            // Δ = 0, Z = Π_k Σ prior_k = 1: a scan that says nothing about the doorways leaves ℓ at the room term.
            // Linear domain, no exp/log in the inner loops (this runs on every ℓ, the rivals' hot path): each cell's weights
            // are scaled by e^{−s_m}, s_m = max(Δ_m, 0), so none exceeds 1, and the forward/backward vectors are renormalised
            // at every cell with the log of the scale carried — the dominant placement never underflows.
            // ★ Task 7c: an opening is a listed door's mouth cell or a generic WINDOW of consecutive cells (build_cells). The
            //   recursion runs over the cells with beams in boundary order; its state is (S, f): the doors S already placed,
            //   and f = how many cells from the current one on are already open by a window started earlier (windows of
            //   different doors may overlap: the boundary is open on their UNION, each cell's beams counted once). A placement
            //   touches the scan from its first cell with beams on; one touching none adds to off_k. Exact for this prior.
            std::vector<int> act;
            act.reserve(delta.size());
            for (const auto& [cid, d] : delta) act.push_back(cid);
            std::ranges::sort(act);   // boundary order
            const std::size_t K = cells.pi.size(), M = act.size(), NS = std::size_t{1} << K, full = NS - 1;
            const std::size_t NC = cells.leaf.size();
            std::vector<int> act_of(NC, -1);
            for (std::size_t m = 0; m < M; ++m) act_of[static_cast<std::size_t>(act[m])] = static_cast<int>(m);
            // the first cell with beams at or after each cell (to map a placement to where it first touches the scan)
            std::vector<int> next_act(NC + 1, -1);
            for (std::size_t ci = NC; ci-- > 0;) next_act[ci] = act_of[ci] >= 0 ? act_of[ci] : next_act[ci + 1];
            // L = number of cells a placement keeps open counting from its first cell with beams (≥ 1)
            int LMAX = 1;
            for (const auto& [i, j] : cells.windows) LMAX = std::max(LMAX, j - i + 1);
            const std::size_t NL = static_cast<std::size_t>(LMAX) + 1, NF = NL;   // L ∈ 1..LMAX, f ∈ 0..LMAX
            std::vector<double> e(K * M * NL, 0.0), offk(K), off_of(NS, 1.0);
            {
                std::vector<double> touched(K, 0.0);
                auto add = [&](int first, int last, std::size_t k, double mass)
                {
                    const int m = next_act[static_cast<std::size_t>(first)];
                    if (m < 0 or act[static_cast<std::size_t>(m)] > last or mass <= 0.0) return;
                    const int L = last - act[static_cast<std::size_t>(m)] + 1;
                    e[(k * M + static_cast<std::size_t>(m)) * NL + static_cast<std::size_t>(L)] += mass;
                    touched[k] += mass;
                };
                for (std::size_t k = 0; k < K; ++k)
                {
                    const double open = 1.0 - cells.q[k];
                    for (std::size_t ci = 0; ci < NC; ++ci) add(static_cast<int>(ci), static_cast<int>(ci), k, open * cells.pi[k][ci]);
                    for (const auto& [i, j] : cells.windows) add(i, j, k, open * cells.wgen[k]);
                    offk[k] = cells.q[k] + std::max(0.0, open - touched[k]);
                }
            }
            for (std::size_t S = 0; S < NS; ++S)
                for (std::size_t k = 0; k < K; ++k)
                    if (not (S >> k & 1u)) off_of[S] *= offk[k];
            // per cell: es = e^{−s_m} (shut), eo = e^{Δ_m − s_m} (open); wx[m][T][x] = Σ over the T doors' placements here
            // whose longest keeps x cells open (x = 0 for T = ∅)
            std::vector<double> es(M), eo(M), wx(M * NS * NL, 0.0);
            double scale = 0.0;
            for (std::size_t m = 0; m < M; ++m)
            {
                const double dm = delta[act[m]], sm = std::max(dm, 0.0);
                scale += sm;
                es[m] = std::exp(-sm);
                eo[m] = std::exp(dm - sm);
                wx[(m * NS) * NL] = 1.0;
                for (std::size_t k = 0; k < K; ++k)   // e → its cumulative over L, in place: C_k(x) = Σ_{L ≤ x} e_k(L)
                    for (std::size_t x = 2; x < NL; ++x) e[(k * M + m) * NL + x] += e[(k * M + m) * NL + x - 1];
                for (std::size_t T = 1; T < NS; ++T)
                {
                    double prev = 0.0;
                    for (std::size_t x = 1; x < NL; ++x)
                    {
                        double cum = 1.0;   // Π_k C_k(x) — the T doors' placements all keeping ≤ x cells open
                        for (std::size_t k = 0; k < K; ++k)
                            if (T >> k & 1u) cum *= e[(k * M + m) * NL + x];
                        wx[(m * NS + T) * NL + x] = std::max(0.0, cum - prev);
                        prev = cum;
                    }
                }
            }
            // after cell m with h cells open from m on, the count entering the next cell with beams
            auto carry = [&](std::size_t m, std::size_t h) -> std::size_t
            {
                if (m + 1 >= M or h == 0) return 0;
                const auto gap = static_cast<std::size_t>(act[m + 1] - act[m]);
                return h > gap ? h - gap : 0;
            };
            const std::size_t N2 = NS * NF;
            std::vector<double> F((M + 1) * N2, 0.0), B((M + 1) * N2, 0.0), lF(M + 1, 0.0), lB(M + 1, 0.0);
            F[0] = 1.0;
            B[M * N2] = 1.0;
            auto renorm = [&](double* v, double& lscale)
            {
                double mx = 0.0;
                for (std::size_t i = 0; i < N2; ++i) mx = std::max(mx, v[i]);
                if (mx > 0.0) { for (std::size_t i = 0; i < N2; ++i) v[i] /= mx; lscale += std::log(mx); }
            };
            // one step: from (S, f) at cell m, the doors T start here keeping x open; calls emit(S|T, f', weight)
            auto step = [&](std::size_t m, std::size_t S, std::size_t f, auto&& emit)
            {
                const std::size_t fr = full & ~S;
                for (std::size_t T = fr;; T = (T - 1) & fr)
                {
                    for (std::size_t x = (T == 0 ? 0 : 1); x < (T == 0 ? 1 : NL); ++x)
                    {
                        const double wt = wx[(m * NS + T) * NL + x];
                        if (wt == 0.0) continue;
                        const std::size_t h = std::max(f, x);
                        emit(S | T, T, carry(m, h), wt * (h > 0 ? eo[m] : es[m]));
                    }
                    if (T == 0) break;
                }
            };
            for (std::size_t m = 0; m < M; ++m)
            {
                for (std::size_t st = 0; st < N2; ++st)
                {
                    const double fv = F[m * N2 + st];
                    if (fv == 0.0) continue;
                    step(m, st % NS, st / NS, [&](std::size_t S2, std::size_t, std::size_t f2, double wt)
                         { F[(m + 1) * N2 + f2 * NS + S2] += fv * wt; });
                }
                lF[m + 1] = lF[m];
                renorm(&F[(m + 1) * N2], lF[m + 1]);
            }
            for (std::size_t m = M; m-- > 0;)
            {
                // B[m][(R, f)] = Σ_T,x w · B[m+1][(R \ T, f')]: R are the doors placed at cells ≥ m
                for (std::size_t f = 0; f < NF; ++f)
                    for (std::size_t R2 = 0; R2 < NS; ++R2)   // doors placed after m
                        step(m, R2, f, [&](std::size_t R, std::size_t, std::size_t f2, double wt)
                             {
                                 const double bv = B[(m + 1) * N2 + f2 * NS + R2];
                                 if (bv != 0.0) B[m * N2 + f * NS + R] += bv * wt;
                             });
                lB[m] = lB[m + 1];
                renorm(&B[m * N2], lB[m]);
            }
            double zt = 0.0;
            for (std::size_t S = 0; S < NS; ++S) zt += F[M * N2 + S] * off_of[S];   // nothing open past the last cell: f = 0
            if (zt > 0.0) ll += lF[M] + std::log(zt) + scale;
            // P(cell m shut | scan): entered with f = 0, no door starts here (then f stays 0 into the next cell).
            std::unordered_map<int, double> stay;
            for (std::size_t m = 0; m < M and zt > 0.0; ++m)
            {
                double num = 0.0;
                for (std::size_t S = 0; S < NS; ++S)
                {
                    const double fv = F[m * N2 + S];
                    if (fv == 0.0) continue;
                    const std::size_t fr = full & ~S;
                    for (std::size_t R = fr;; R = (R - 1) & fr)
                    {
                        num += fv * B[(m + 1) * N2 + R] * off_of[S | R];
                        if (R == 0) break;
                    }
                }
                stay[act[m]] = std::clamp(num * es[m] * std::exp(lF[m] + lB[m + 1] - lF[M]) / zt, 0.0, 1.0);
            }
            for (const auto& [cid, idx] : groups)
            {
                const auto it = stay.find(cid);
                const float rho = it == stay.end() ? 1.f : static_cast<float>(it->second);   // P(cell shut | scan)
                if (rho >= 1.f) continue;
                const float cf = cells.leaf[static_cast<std::size_t>(cid)];
                const float lc = cf > 0.f ? std::log(cf) : -kInf, l1c = cf < 1.f ? std::log1p(-cf) : -kInf;
                for (const std::size_t i : idx)
                {
                    if (not passed[i]) continue;
                    const float leaf = std::exp(lc + room_ll[i] - lse(l1c + open_ll[i], lc + room_ll[i]));
                    room_post[i] = rho + (1.f - rho) * leaf;
                }
            }
            return static_cast<float>(ll);
        }

        float polygon_extent(const std::vector<Eigen::Vector2f>& poly)
        {
            Eigen::Vector2f lo = poly.front(), hi = poly.front();
            for (const auto& v : poly) { lo = lo.cwiseMin(v); hi = hi.cwiseMax(v); }
            return (hi - lo).norm();
        }

        /// Closest boundary point of the polygon to q (and, optionally, which edge it lies on).
        Eigen::Vector2f closest_on_polygon(const std::vector<Eigen::Vector2f>& poly, const Eigen::Vector2f& q,
                                           int* edge = nullptr)
        {
            float best = kInf;
            Eigen::Vector2f best_c = poly.front();
            const auto n = poly.size();
            for (std::size_t i = 0; i < n; ++i)
            {
                const Eigen::Vector2f& a = poly[i];
                const Eigen::Vector2f ab = poly[(i + 1) % n] - a;
                const float len2 = ab.squaredNorm();
                const float t = len2 > 1e-12f ? std::clamp((q - a).dot(ab) / len2, 0.f, 1.f) : 0.f;
                const Eigen::Vector2f c = a + t * ab;
                const float d2 = (q - c).squaredNorm();
                if (d2 < best) { best = d2; best_c = c; if (edge != nullptr) *edge = static_cast<int>(i); }
            }
            return best_c;
        }

        float log_det3(const Eigen::Matrix3f& m)
        {
            const Eigen::LLT<Eigen::Matrix3d> llt(m.cast<double>());
            if (llt.info() != Eigen::Success) return 0.f;
            const auto& L = llt.matrixL();
            return static_cast<float>(2.0 * (std::log(L(0, 0)) + std::log(L(1, 1)) + std::log(L(2, 2))));
        }

        std::vector<Eigen::Vector2f> subsample(const std::vector<Eigen::Vector2f>& pts, int max_pts)
        {
            if (max_pts <= 0 or static_cast<int>(pts.size()) <= max_pts) return pts;
            std::vector<Eigen::Vector2f> out;
            out.reserve(static_cast<std::size_t>(max_pts));
            const double stride = static_cast<double>(pts.size()) / max_pts;
            for (int i = 0; i < max_pts; ++i)
                out.push_back(pts[static_cast<std::size_t>(i * stride)]);
            return out;
        }
    } // namespace

    // ── geometry ────────────────────────────────────────────────────────────────────────────────────
    namespace
    {
        bool goes_through(const Aperture& ap, const Eigen::Vector2f& origin, const Eigen::Vector2f& hit)
        {
            const Eigen::Vector2f ab = ap.b - ap.a;
            const float w2 = ab.squaredNorm();
            if (w2 < 1e-12f) return false;
            const float u_org = (origin - ap.a).dot(ab) / w2;
            if (ap.n_out.squaredNorm() > 0.f and ap.n_out.dot(origin - ap.a) > 0.f and u_org >= 0.f and u_org <= 1.f)
                return ap.n_out.dot(hit - ap.a) > 0.f;          // the sensor stands in the opening
            return crosses(origin, hit - origin, ap.a, ab);     // the sensor is in the room
        }
        Ray cast_ray(const std::vector<Eigen::Vector2f>& poly, const Cells* c, const Eigen::Vector2f& origin,
                     bool origin_inside, const Eigen::Vector2f& q)
        {
            Ray ry;
            const std::size_t ne = poly.size();
            const Eigen::Vector2f r = q - origin;
            ry.z = r.norm();
            float best_t = kInf, best_u = 0.f, t2 = kInf;
            int e2 = -1;
            for (std::size_t e = 0; e < ne; ++e)
            {
                const Eigen::Vector2f e0 = poly[e], s = poly[(e + 1) % ne] - e0;
                const float rxs = r.x() * s.y() - r.y() * s.x();
                if (std::abs(rxs) < 1e-9f) continue;
                const Eigen::Vector2f qp = e0 - origin;
                const float t = (qp.x() * s.y() - qp.y() * s.x()) / rxs;   // along the RAY (t > 1: past the return)
                const float u = (qp.x() * r.y() - qp.y() * r.x()) / rxs;
                if (t <= 0.f or u < 0.f or u > 1.f or t >= t2) continue;
                if (t < best_t) { t2 = best_t; e2 = ry.edge; best_t = t; best_u = u; ry.edge = static_cast<int>(e); }
                else { t2 = t; e2 = static_cast<int>(e); }
            }
            if (ry.edge < 0) return ry;
            if (e2 >= 0 and ry.z > 0.f)
            {
                const auto f = static_cast<std::size_t>(e2);
                const Eigen::Vector2f s2 = (poly[(f + 1) % ne] - poly[f]).normalized();
                ry.z2 = t2 * ry.z;
                ry.cos2 = std::abs(s2.x() * r.y() - s2.y() * r.x()) / ry.z;
            }
            const auto e = static_cast<std::size_t>(ry.edge);
            const Eigen::Vector2f s = (poly[(e + 1) % ne] - poly[e]).normalized();
            ry.n = Eigen::Vector2f(-s.y(), s.x());
            ry.zs = best_t * ry.z;
            ry.cosa = ry.z > 0.f ? std::abs(ry.n.dot(r)) / ry.z : 1.f;
            ry.passed = best_t < 1.f;
            if (c != nullptr and origin_inside)
            {
                const float sx = best_u * (poly[(e + 1) % ne] - poly[e]).norm();   // where along the edge the ray meets it
                int k = c->first[e];
                while (k + 1 < c->first[e] + c->count[e] and sx >= c->s1[static_cast<std::size_t>(k)]) ++k;
                ry.cell = k;
            }
            return ry;
        }
    }   // namespace

    float exit_prior(const std::vector<Aperture>& apertures, const Eigen::Vector2f& origin, const Eigen::Vector2f& hit)
    {
        float stay = 1.f;   // Π (1 − P(open state)·(1 − coverage) · through_k)
        for (const auto& ap : apertures)
            if (goes_through(ap, origin, hit))
                stay *= 1.f - std::clamp(1.f - ap.p_shut, 0.f, 1.f) * std::clamp(ap.open, 0.f, 1.f);
        return 1.f - stay;
    }

    float polygon_distance(const std::vector<Eigen::Vector2f>& polygon, const Eigen::Vector2f& q)
    {
        return (q - closest_on_polygon(polygon, q)).norm();
    }

    bool inside_polygon(const std::vector<Eigen::Vector2f>& poly, const Eigen::Vector2f& q)
    {
        bool in = false;
        const auto n = poly.size();
        for (std::size_t i = 0, j = n - 1; i < n; j = i++)
        {
            const auto& a = poly[i];
            const auto& b = poly[j];
            if (((a.y() > q.y()) != (b.y() > q.y()))
                and (q.x() < (b.x() - a.x()) * (q.y() - a.y()) / (b.y() - a.y()) + a.x()))
                in = not in;
        }
        return in;
    }

    // ── raster ──────────────────────────────────────────────────────────────────────────────────────
    void LikelihoodRaster::build(const std::vector<Eigen::Vector2f>& polygon, const Params& p)
    {
        nx_ = ny_ = 0;
        ll_.clear(); dist_.clear();
        if (polygon.size() < 3) return;
        extent_ = polygon_extent(polygon);
        if (not (extent_ > 0.f)) return;

        // Pad by 4σ: beyond that the wall term is exp(−8) of its peak and a point is pure clutter, so
        // off-raster lookups lose nothing by returning the clutter term.
        const float pad = 4.f * p.sigma_sdf;
        Eigen::Vector2f lo = polygon.front(), hi = polygon.front();
        for (const auto& v : polygon) { lo = lo.cwiseMin(v); hi = hi.cwiseMax(v); }
        res_ = p.raster_res;
        x0_ = lo.x() - pad;
        y0_ = lo.y() - pad;
        nx_ = static_cast<int>(std::ceil((hi.x() - lo.x() + 2.f * pad) / res_)) + 1;
        ny_ = static_cast<int>(std::ceil((hi.y() - lo.y() + 2.f * pad) / res_)) + 1;

        const PointTerm term(p, extent_);
        ll_out_ = term.log_out;
        ll_.resize(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_));
        dist_.resize(ll_.size());
        for (int j = 0; j < ny_; ++j)
            for (int i = 0; i < nx_; ++i)
            {
                // Cell CENTRE, so nearest-cell lookup (floor) is unbiased.
                const Eigen::Vector2f q(x0_ + (i + 0.5f) * res_, y0_ + (j + 0.5f) * res_);
                const float d = polygon_distance(polygon, q);
                const auto k = static_cast<std::size_t>(j) * static_cast<std::size_t>(nx_) + static_cast<std::size_t>(i);
                dist_[k] = d;
                ll_[k] = term.ll(d);
            }
    }

    float LikelihoodRaster::ll_at(float x, float y) const
    {
        const int i = static_cast<int>(std::floor((x - x0_) / res_));
        const int j = static_cast<int>(std::floor((y - y0_) / res_));
        if (i < 0 or j < 0 or i >= nx_ or j >= ny_) return ll_out_;
        return ll_[static_cast<std::size_t>(j) * static_cast<std::size_t>(nx_) + static_cast<std::size_t>(i)];
    }

    // ── exact likelihood ────────────────────────────────────────────────────────────────────────────
    BeyondLines beyond_lines(const std::vector<Eigen::Vector2f>& pts_robot, const Params& p)
    {
        BeyondLines bl;
        if (not openings_on(p) or pts_robot.size() < 10) return bl;
        bl.r.assign(pts_robot.size(), kInf);
        bl.t.assign(pts_robot.size(), 0.f);
        bl.seg.assign(pts_robot.size(), -1);
        bl.n.assign(pts_robot.size(), Eigen::Vector2f::Zero());
        std::mt19937 rng(20261021u);   // fixed: the same point set must always give the same lines
        const wallseg::Params sp;
        const auto res = wallseg::segment(pts_robot, sp, rng);
        for (std::size_t j = 0; j < res.segments.size(); ++j)
        {
            const auto& sg = res.segments[j];
            for (const int i : sg.inliers)
            {
                if (i < 0 or i >= static_cast<int>(pts_robot.size())) continue;
                const Eigen::Vector2f& pt = pts_robot[static_cast<std::size_t>(i)];
                bl.r[static_cast<std::size_t>(i)] = std::abs(sg.normal.dot(pt) - sg.d);
                bl.t[static_cast<std::size_t>(i)] = Eigen::Vector2f(-sg.normal.y(), sg.normal.x()).dot(pt);
                bl.seg[static_cast<std::size_t>(i)] = static_cast<int>(j);
                bl.n[static_cast<std::size_t>(i)] = sg.normal;
            }
        }
        return bl;
    }

    namespace
    {
        /// Per point under one pose: its ray (cast_ray) and its beam terms (BeamTerm). `origins` (robot frame, one per point)
        /// places the sensor of a point taken from another pose (multi_scan_fit's stacked keyframes); nullptr ⇒ the robot.
        /// `o_in` caches inside_polygon for the robot's own origin.
        void beam_pass(const std::vector<Eigen::Vector2f>& polygon, const std::vector<Eigen::Vector2f>& pts_robot,
                       const Eigen::Vector3f& pose, const Cells* cells, const BeamTerm& bt, const BeyondLines& bl,
                       const std::vector<Eigen::Vector2f>* origins, std::vector<Ray>& rays, std::vector<float>& shut,
                       std::vector<float>& open, std::vector<float>& hit, std::vector<Eigen::Vector2f>* rp_out = nullptr)
        {
            const float c = std::cos(pose.z()), s = std::sin(pose.z());
            const Eigen::Vector2f t = pose.head<2>();
            const std::size_t n = pts_robot.size();
            rays.resize(n); shut.resize(n); open.resize(n); hit.resize(n);
            if (rp_out != nullptr) rp_out->resize(n);
            const bool own_in = cells != nullptr and inside_polygon(polygon, t);
            const bool per_point = origins != nullptr and origins->size() == n;
            for (std::size_t i = 0; i < n; ++i)
            {
                const auto& pr = pts_robot[i];
                const Eigen::Vector2f rp(c * pr.x() - s * pr.y(), s * pr.x() + c * pr.y());
                if (rp_out != nullptr) (*rp_out)[i] = rp;
                Eigen::Vector2f o = t;
                bool o_in = own_in;
                if (per_point)
                {
                    const auto& orr = (*origins)[i];
                    o = t + Eigen::Vector2f(c * orr.x() - s * orr.y(), s * orr.x() + c * orr.y());
                    o_in = cells != nullptr and inside_polygon(polygon, o);
                }
                rays[i] = cast_ray(polygon, cells, o, o_in, rp + t);
                float rl = kInf, cl = 1.f;
                if (i < bl.r.size() and std::isfinite(bl.r[i]))
                {
                    rl = bl.r[i];
                    const Eigen::Vector2f dir = per_point ? Eigen::Vector2f(pr - (*origins)[i]) : pr;   // the beam, robot frame
                    const float dn = dir.norm();
                    cl = dn > 0.f ? std::abs(bl.n[i].dot(dir)) / dn : 1.f;
                }
                bt.eval(rays[i], rl, cl, shut[i], open[i], hit[i]);
            }
        }
    }

    float log_likelihood(const std::vector<Eigen::Vector2f>& polygon,
                         const std::vector<Eigen::Vector2f>& pts_robot,
                         const Eigen::Vector3f& pose, const Params& p, float extent, float* median_abs,
                         const BeyondLines* beyond, const std::vector<Eigen::Vector2f>* origins)
    {
        BeyondLines own;
        if (beyond == nullptr and openings_on(p)) { own = beyond_lines(pts_robot, p); beyond = &own; }
        const BeyondLines& bl = beyond != nullptr ? *beyond : own;
        const BeamTerm bt(p, p.sigma_sdf, extent);
        const bool on = openings_on(p);
        const Cells& cells = on ? cells_for(polygon, p) : kNoCells;
        const std::size_t n = pts_robot.size();
        std::vector<Ray> rays;
        std::vector<float> shut, open, hit, room_post;
        beam_pass(polygon, pts_robot, pose, on ? &cells : nullptr, bt, bl, origins, rays, shut, open, hit);
        std::vector<int> cell_of(n, -1);
        std::vector<char> passed(n, 0);
        for (std::size_t i = 0; i < n; ++i) { cell_of[i] = rays[i].cell; passed[i] = rays[i].passed ? 1 : 0; }
        const float ll = cell_mixture(shut, open, cell_of, passed, cells, p, p.sigma_sdf, bl, room_post);
        if (median_abs != nullptr)
        {
            // The legacy trigger's statistic, unchanged: |distance to the polygon|, weighted by P(room return).
            const float c = std::cos(pose.z()), s = std::sin(pose.z());
            std::vector<std::pair<float, float>> dw;
            dw.reserve(n);
            for (std::size_t i = 0; i < n; ++i)
            {
                const auto& pr = pts_robot[i];
                const Eigen::Vector2f q(c * pr.x() - s * pr.y() + pose.x(), s * pr.x() + c * pr.y() + pose.y());
                dw.emplace_back(polygon_distance(polygon, q), room_post[i]);
            }
            *median_abs = weighted_median(dw);
        }
        return ll;
    }

    // ── yaw hypotheses from structure ───────────────────────────────────────────────────────────────
    std::vector<float> structural_yaws(const std::vector<Eigen::Vector2f>& polygon,
                                       const std::vector<Eigen::Vector2f>& pts_robot,
                                       const Params& p, std::uint32_t seed, int* n_segments)
    {
        if (n_segments != nullptr) *n_segments = 0;
        if (polygon.size() < 3 or pts_robot.size() < 10) return {};

        std::mt19937 rng(seed);
        const wallseg::Params sp;
        const auto seg = wallseg::segment(pts_robot, sp, rng);
        if (n_segments != nullptr) *n_segments = static_cast<int>(seg.segments.size());
        if (seg.segments.empty()) return {};

        // Inward normals of the polygon edges. The sign of the area fixes which side is inside.
        float area2 = 0.f;
        const auto n = polygon.size();
        for (std::size_t i = 0; i < n; ++i)
        {
            const auto& a = polygon[i];
            const auto& b = polygon[(i + 1) % n];
            area2 += a.x() * b.y() - b.x() * a.y();
        }
        const float ccw = area2 > 0.f ? 1.f : -1.f;
        struct Edge { float psi; float len; };
        std::vector<Edge> edges;
        for (std::size_t i = 0; i < n; ++i)
        {
            const Eigen::Vector2f ab = polygon[(i + 1) % n] - polygon[i];
            const float len = ab.norm();
            if (len < 1e-4f) continue;
            // CCW polygon: interior on the LEFT of a→b, so the inward normal is (−ab.y, ab.x).
            edges.push_back({std::atan2(ccw * ab.x(), -ccw * ab.y()), len});
        }

        // Kernel density on the circle of the implied yaw θ = ψ − φ. Each kernel is as wide as the
        // segment's own σ_φ (from its information matrix), weighted by the points it explains and the
        // length of the edge it is paired with (a long wall is more likely to be the one in view).
        constexpr int kBins = 720;   // 0.5°
        const float bin = 2.f * kPi / kBins;
        std::vector<double> dens(kBins, 0.0);
        for (const auto& sgm : seg.segments)
        {
            const Eigen::Matrix2f info = sgm.info_phi_d;
            float var_phi = 1e-2f;
            if (std::abs(info.determinant()) > 1e-12f)
                var_phi = std::max(info.inverse()(0, 0), 0.f);
            // A kernel narrower than a bin would alias; widen to the bin, which is the density's resolution.
            const float sig = std::sqrt(std::max(var_phi, bin * bin));
            const int half = std::min(kBins / 2, static_cast<int>(std::ceil(3.f * sig / bin)));
            for (const auto& e : edges)
            {
                const float theta = wrap(e.psi - sgm.phi);
                const int c = static_cast<int>(std::lround((theta + kPi) / bin)) % kBins;
                const double w = static_cast<double>(sgm.npts) * e.len;
                for (int k = -half; k <= half; ++k)
                {
                    const float dth = k * bin;
                    dens[static_cast<std::size_t>((c + k + kBins) % kBins)] +=
                        w * std::exp(-0.5 * dth * dth / (sig * sig));
                }
            }
        }

        struct Peak { double v; float theta; };
        std::vector<Peak> peaks;
        for (int i = 0; i < kBins; ++i)
        {
            const double v = dens[static_cast<std::size_t>(i)];
            if (v <= 0.0) continue;
            if (v >= dens[static_cast<std::size_t>((i + 1) % kBins)]
                and v > dens[static_cast<std::size_t>((i - 1 + kBins) % kBins)])
                peaks.push_back({v, -kPi + i * bin});
        }
        std::ranges::sort(peaks, std::ranges::greater{}, &Peak::v);
        std::vector<float> yaws;
        for (const auto& pk : peaks | std::views::take(p.max_yaws))
            yaws.push_back(pk.theta);
        return yaws;
    }

    // ── polish ──────────────────────────────────────────────────────────────────────────────────────
    Mode polish(const std::vector<Eigen::Vector2f>& polygon,
                const std::vector<Eigen::Vector2f>& pts_robot,
                const Eigen::Vector3f& start, const Params& p, float extent, const BeyondLines* beyond,
                const std::vector<Eigen::Vector2f>* origins)
    {
        BeyondLines own;
        if (beyond == nullptr and openings_on(p)) { own = beyond_lines(pts_robot, p); beyond = &own; }
        const BeyondLines& bl = beyond != nullptr ? *beyond : own;
        const bool on = openings_on(p);
        const Cells& cells = on ? cells_for(polygon, p) : kNoCells;
        // σ is ESTIMATED here, by EM on the inlier responsibilities (σ² ← Σγd²/Σγ), annealing down from the wide
        // σ_sdf the lattice needs for its basins. A polish at the fixed σ_sdf = 0.15 m — the agent's model-
        // mismatch noise, ~7x a scan's real spread — left the likelihood flat over ±10 cm and let one-sided
        // clutter drag the mode: measured in reloc_selftest, right-basin modes polished to median |d| 0.056 m
        // where the truth scored 0.019, and the NEES tail. Never raised above σ_sdf: annealing only.
        Params pl = p;
        BeamTerm term(pl, pl.sigma_sdf, extent);
        float inv_s2 = 1.f / (pl.sigma_sdf * pl.sigma_sdf);
        double em_wd2 = 0.0, em_w = 0.0;
        // The search's own prior: uniform over the room and the circle. Its information (1/extent² on
        // x,y, 1/π² on θ) keeps H invertible along a direction the scan does not constrain — a corridor
        // then reports a covariance as large as the room, which is the honest answer, not a NaN.
        Eigen::Matrix3f prior = Eigen::Matrix3f::Zero();
        prior(0, 0) = prior(1, 1) = 1.f / std::max(extent * extent, 1e-3f);
        prior(2, 2) = 1.f / (kPi * kPi);

        Eigen::Vector3f x = start;
        Eigen::Matrix3f H = prior;
        float ll_cur = -kInf;
        float lambda = 1e-3f;
        // Per-edge sums for the common-mode marginalisation below: Σw and Σw·J over the points on each edge.
        struct EdgeSum { float sw = 0.f; Eigen::Vector3f swj = Eigen::Vector3f::Zero(); };
        std::vector<EdgeSum> edge_sums;
        std::vector<Eigen::Vector2f> pt_rp;   // per-point scratch, reused across evaluations
        std::vector<Ray> pt_ray;
        std::vector<int> pt_cell;
        std::vector<char> pt_passed;
        std::vector<float> pt_shut, pt_open, pt_hit, pt_post;
        auto evaluate = [&](const Eigen::Vector3f& pose, Eigen::Matrix3f* Hout, Eigen::Vector3f* gout) -> float
        {
            Eigen::Matrix3f Hs = prior;
            Eigen::Vector3f g = Eigen::Vector3f::Zero();
            if (Hout != nullptr) { edge_sums.assign(polygon.size(), EdgeSum{}); em_wd2 = 0.0; em_w = 0.0; }
            // Pass 1: each beam cast from the sensor under THIS pose (Task 7c): the wall it meets, its range terms and the
            // doorway cell it reaches. Which wall a beam meets is a step function of the pose, held constant within a GN
            // step like γ and re-evaluated at every accepted pose. Pass 2 (below) needs each cell's posterior state.
            const std::size_t n = pts_robot.size();
            beam_pass(polygon, pts_robot, pose, on ? &cells : nullptr, term, bl, origins, pt_ray, pt_shut, pt_open, pt_hit, &pt_rp);
            pt_cell.resize(n); pt_passed.resize(n);
            for (std::size_t i = 0; i < n; ++i) { pt_cell[i] = pt_ray[i].cell; pt_passed[i] = pt_ray[i].passed ? 1 : 0; }
            const float ll = cell_mixture(pt_shut, pt_open, pt_cell, pt_passed, cells, p, pl.sigma_sdf, bl, pt_post);
            for (std::size_t i = 0; i < n and Hout != nullptr; ++i)
            {
                const Ray& ry = pt_ray[i];
                if (ry.edge < 0) continue;   // the beam meets no wall: nothing to align
                // The hit term in the pose: −d²/2σ², d = the return's signed perpendicular offset from the wall its ray
                // meets (= (z − z*)·|cos α|), with that wall held fixed within the step.
                const Eigen::Vector2f q = pt_rp[i] + pose.head<2>();
                const float d = ry.n.dot(q - polygon[static_cast<std::size_t>(ry.edge)]);
                // inlier responsibility = P(room return) · P(wall hit | room return)
                const float gamma = pt_post[i] * std::exp(pt_hit[i] - pt_shut[i]);
                em_wd2 += static_cast<double>(gamma) * d * d;
                em_w   += gamma;
                const Eigen::Vector2f& u = ry.n;
                const Eigen::Vector2f& rp = pt_rp[i];
                // ∂q/∂θ = R'·p = (−rp.y, rp.x)
                const Eigen::Vector3f J(u.x(), u.y(), -u.x() * rp.y() + u.y() * rp.x());
                const float w = gamma * inv_s2;
                Hs += w * J * J.transpose();
                g  += (w * d) * J;     // ∂(−ℓ)/∂x
                auto& es = edge_sums[static_cast<std::size_t>(ry.edge)];
                es.sw += w;
                es.swj += w * J;
            }
            if (Hout != nullptr) *Hout = Hs;
            if (gout != nullptr) *gout = g;
            return ll;
        };

        Eigen::Vector3f g;
        ll_cur = evaluate(x, &H, &g);
        for (int it = 0; it < p.polish_iters; ++it)
        {
            Eigen::Matrix3f A = H;
            A.diagonal() *= (1.f + lambda);
            const Eigen::Vector3f step = A.ldlt().solve(-g);
            if (not step.allFinite()) break;
            Eigen::Vector3f xn = x + step;
            xn.z() = wrap(xn.z());
            Eigen::Matrix3f Hn;
            Eigen::Vector3f gn;
            const float ll_new = evaluate(xn, &Hn, &gn);
            if (ll_new >= ll_cur)
            {
                x = xn; H = Hn; g = gn; ll_cur = ll_new;
                lambda = std::max(lambda * 0.3f, 1e-6f);
                // EM step on σ at the accepted pose (the sums evaluate() just filled belong to it).
                bool sigma_moved = false;
                if (em_w > 3.0)
                {
                    const float sig_em = static_cast<float>(std::sqrt(em_wd2 / em_w));
                    if (std::isfinite(sig_em) and sig_em > 0.f and sig_em < 0.99f * pl.sigma_sdf)
                    {
                        pl.sigma_sdf = sig_em;
                        term = BeamTerm(pl, pl.sigma_sdf, extent);
                        inv_s2 = 1.f / (pl.sigma_sdf * pl.sigma_sdf);
                        ll_cur = evaluate(x, &H, &g);   // the objective changed with σ: re-linearise
                        sigma_moved = true;
                    }
                }
                if (sigma_moved) continue;
                // Converged when the UNDAMPED Newton step is negligible against the pose's own uncertainty
                // (Mahalanobis ≪ 1). Testing the change in ℓ instead stopped on a tiny step taken under
                // heavy damping — measured in reloc_selftest: modes left 5–12 cm short with ℓ 14–50 nats
                // below the truth's.
                const Eigen::Vector3f newton = H.ldlt().solve(-g);
                if (newton.allFinite() and newton.dot(H * newton) < 1e-6f) break;
            }
            else
                lambda *= 10.f;
        }

        // COVARIANCE WITH EACH WALL'S COMMON-MODE OFFSET MARGINALISED OUT.
        // The points on one wall do not err independently: clutter, a mis-placed map wall or an
        // unmodelled bias shifts them TOGETHER, and that shared part does not average down with more
        // points. Model it as an offset δ_e ~ N(0, σ_c²) added to every residual on edge e and integrate
        // it out (Woodbury, exact for the linearised model):
        //     H_e  =  Σ w J Jᵀ  −  (Σ w J)(Σ w J)ᵀ / (1/σ_c² + Σ w)
        // A wall seen by n points then contributes at most ~1/σ_c² along its normal instead of n/σ².
        // Measured in reloc_selftest before this term: NEES 4–6 (covariance 2–2.5x too tight in σ),
        // growing as points were ADDED — the signature of an error that does not average down.
        Eigen::Matrix3f H_marg = H;
        if (p.wall_offset_sigma > 0.f)
        {
            // Recompute the per-edge sums at the final pose (evaluate() filled them for the last accepted pose
            // only if that was the last call; make it explicit).
            evaluate(x, &H, nullptr);
            H_marg = H;
            const float inv_c2 = 1.f / (p.wall_offset_sigma * p.wall_offset_sigma);
            for (const auto& es : edge_sums)
                if (es.sw > 0.f)
                    H_marg -= (es.swj * es.swj.transpose()) / (inv_c2 + es.sw);
        }

        Mode m;
        m.pose = x;
        m.pose.z() = wrap(m.pose.z());
        m.cov = H_marg.inverse();
        float med = 0.f;
        m.log_lik = log_likelihood(polygon, pts_robot, m.pose, p, extent, &med, &bl, origins);
        m.median_abs = med;
        return m;
    }

    // ── search ──────────────────────────────────────────────────────────────────────────────────────
    Result search(const LikelihoodRaster& raster,
                  const std::vector<Eigen::Vector2f>& polygon,
                  const std::vector<Eigen::Vector2f>& pts_all,
                  const std::vector<Eigen::Vector3f>& seed_poses,
                  const Params& p, std::uint32_t seed)
    {
        const auto t0 = std::chrono::steady_clock::now();
        Result res;
        if (not raster.valid() or polygon.size() < 3 or pts_all.empty()) return res;

        const float extent = raster.extent();
        const auto pts = subsample(pts_all, p.max_points);

        // 1. yaws: structure, else a uniform sweep; seed yaws and their half-turn always included.
        std::vector<float> yaws = structural_yaws(polygon, subsample(pts_all, 4 * p.max_points), p, seed,
                                                  &res.n_segments);
        if (yaws.empty())
            for (int k = 0; k < p.fallback_yaws; ++k)
                yaws.push_back(-kPi + 2.f * kPi * k / p.fallback_yaws);
        res.n_yaws = static_cast<int>(yaws.size());

        // 2. translation lattice, aligned to the raster so a lattice offset is an integer cell offset.
        //    ⚠ The lattice scores the ROOM-ONLY raster (no doorway mixture: it would need a per-point crossing test per
        //    candidate). It only PROPOSES seeds; every mode is then polished and weighed on the full ℓ, doorways
        //    included, and the caller's seeds (the incumbent) are always polished — so the doorway model decides.
        const int m = std::max(1, static_cast<int>(std::lround(p.lattice_step / raster.res())));
        const float step = m * raster.res();
        const int lx = raster.nx() / m, ly = raster.ny() / m;
        // Valid centres: inside the polygon and clear of every wall by the body radius.
        std::vector<char> valid(static_cast<std::size_t>(lx) * static_cast<std::size_t>(ly), 0);
        for (int j = 0; j < ly; ++j)
            for (int i = 0; i < lx; ++i)
            {
                const Eigen::Vector2f t(raster.x0() + i * step, raster.y0() + j * step);
                const int ci = i * m, cj = j * m;
                const float d = raster.dist()[static_cast<std::size_t>(cj) * static_cast<std::size_t>(raster.nx())
                                              + static_cast<std::size_t>(ci)];
                valid[static_cast<std::size_t>(j) * static_cast<std::size_t>(lx) + static_cast<std::size_t>(i)] =
                    (d >= p.body_clearance and inside_polygon(polygon, t)) ? 1 : 0;
            }

        struct Cand { float ll; Eigen::Vector3f pose; };
        std::vector<Cand> maxima;
        std::vector<float> score(valid.size());
        std::vector<int> ci(pts.size()), cj(pts.size());
        const auto& ll = raster.ll();
        const int nx = raster.nx(), ny = raster.ny();
        for (const float yaw : yaws)
        {
            const float c = std::cos(yaw), s = std::sin(yaw);
            // Lattice point (i,j) sits at x0 + i·step; a point rotated to rp lands in cell
            // floor((rp.x + i·step)/res) = floor(rp.x/res + i·m) — the per-point part is precomputed.
            for (std::size_t k = 0; k < pts.size(); ++k)
            {
                ci[k] = static_cast<int>(std::floor((c * pts[k].x() - s * pts[k].y()) / raster.res()));
                cj[k] = static_cast<int>(std::floor((s * pts[k].x() + c * pts[k].y()) / raster.res()));
            }
            for (int j = 0; j < ly; ++j)
                for (int i = 0; i < lx; ++i)
                {
                    const auto idx = static_cast<std::size_t>(j) * static_cast<std::size_t>(lx) + static_cast<std::size_t>(i);
                    if (not valid[idx]) { score[idx] = -kInf; continue; }
                    float acc = 0.f;
                    const int oi = i * m, oj = j * m;
                    for (std::size_t k = 0; k < pts.size(); ++k)
                    {
                        const int a = ci[k] + oi, b = cj[k] + oj;
                        acc += (a < 0 or b < 0 or a >= nx or b >= ny)
                                   ? raster.ll_outside()
                                   : ll[static_cast<std::size_t>(b) * static_cast<std::size_t>(nx) + static_cast<std::size_t>(a)];
                    }
                    score[idx] = acc;
                    ++res.n_evals;
                }
            // Local maxima over the 8-neighbourhood (ties broken towards the lower index).
            for (int j = 0; j < ly; ++j)
                for (int i = 0; i < lx; ++i)
                {
                    const auto idx = static_cast<std::size_t>(j) * static_cast<std::size_t>(lx) + static_cast<std::size_t>(i);
                    const float v = score[idx];
                    if (v == -kInf) continue;
                    bool is_max = true;
                    for (int dj = -1; dj <= 1 and is_max; ++dj)
                        for (int di = -1; di <= 1; ++di)
                        {
                            if (di == 0 and dj == 0) continue;
                            const int a = i + di, b = j + dj;
                            if (a < 0 or b < 0 or a >= lx or b >= ly) continue;
                            const float w = score[static_cast<std::size_t>(b) * static_cast<std::size_t>(lx) + static_cast<std::size_t>(a)];
                            if (w > v or (w == v and (dj < 0 or (dj == 0 and di < 0)))) { is_max = false; break; }
                        }
                    if (is_max)
                        maxima.push_back({v, Eigen::Vector3f(raster.x0() + i * step, raster.y0() + j * step, yaw)});
                }
        }
        std::ranges::sort(maxima, std::ranges::greater{}, &Cand::ll);
        if (static_cast<int>(maxima.size()) > p.max_seeds) maxima.resize(static_cast<std::size_t>(p.max_seeds));

        // 3. polish every seed (lattice maxima + caller's seeds), then merge seeds that converged together.
        std::vector<Eigen::Vector3f> starts;
        if (not (p.prior_only and not p.prior_modes.empty()))
        {
            if (p.local_seeds.empty()) for (const auto& c : maxima) starts.push_back(c.pose);
            else starts.insert(starts.end(), p.local_seeds.begin(), p.local_seeds.end());   // the entry area only
            for (const auto& sp : seed_poses) starts.push_back(sp);
        }
        for (const auto& pm : p.prior_modes) starts.push_back(pm.mu);   // the prior's own hypotheses are always polished

        // The lattice ran on the subsample for speed; the polish is per seed and cheap, so it uses the
        // denser set — mode PRECISION comes from here, and a 150-point polish left modes ~10 cm off.
        const auto pts_polish = subsample(pts_all, 4 * p.max_points);
        const BeyondLines bl = beyond_lines(pts_polish, p);   // pose-independent: once for every seed
        std::vector<Mode> modes;
        for (const auto& st : starts)
        {
            Mode md = polish(polygon, pts_polish, st, p, extent, &bl);
            if (not md.pose.allFinite() or not std::isfinite(md.log_lik)) continue;
            bool merged = false;
            for (auto& other : modes)
            {
                Eigen::Vector3f dlt = md.pose - other.pose;
                dlt.z() = wrap(dlt.z());
                const Eigen::Matrix3f S = md.cov + other.cov;
                const float chi2 = dlt.dot(S.ldlt().solve(dlt));
                if (std::isfinite(chi2) and chi2 < p.chi2_merge)
                {
                    if (md.log_lik > other.log_lik) other = md;
                    merged = true;
                    break;
                }
            }
            if (not merged) modes.push_back(md);
        }

        // 4. Laplace evidence  ℓ* + ½log|Σ|  (+ the pose prior's log-density when one is given), normalised by log-sum-exp.
        //    The mode ordering, the max_modes cut, the weights and the ESS all include the prior: it is part of the evidence.
        const double area = p.prior_modes.empty() ? 0.0 : polygon_area(polygon);
        for (auto& md : modes)
        {
            md.log_weight = md.log_lik + 0.5f * log_det3(md.cov);
            // A pose prior from outside the scan is one more factor of the mode's evidence (empty: nothing added).
            if (not p.prior_modes.empty())
                md.log_weight += static_cast<float>(log_prior_modes(p, md.pose, md.cov, area));
        }
        std::ranges::sort(modes, std::ranges::greater{}, &Mode::log_weight);
        if (static_cast<int>(modes.size()) > p.max_modes) modes.resize(static_cast<std::size_t>(p.max_modes));
        if (not modes.empty())
        {
            const float top = modes.front().log_weight;
            double z = 0.0;
            for (const auto& md : modes) z += std::exp(static_cast<double>(md.log_weight - top));
            const float logz = top + static_cast<float>(std::log(z));
            double w2 = 0.0;
            for (auto& md : modes)
            {
                md.log_weight -= logz;
                md.weight = std::exp(md.log_weight);
                w2 += static_cast<double>(md.weight) * md.weight;
                if (md.weight > 0.f) res.entropy -= md.weight * md.log_weight;
            }
            res.ess = w2 > 0.0 ? static_cast<float>(1.0 / w2) : 1.f;
        }
        res.modes = std::move(modes);
        res.duration_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return res;
    }

    double polygon_area(const std::vector<Eigen::Vector2f>& polygon)
    {
        double a2 = 0.0;
        for (std::size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++)
            a2 += static_cast<double>(polygon[j].x()) * polygon[i].y() - static_cast<double>(polygon[i].x()) * polygon[j].y();
        return 0.5 * std::abs(a2);
    }

    double log_prior_modes(const Params& p, const Eigen::Vector3f& pose, const Eigen::Matrix3f& cov, double area)
    {
        if (p.prior_modes.empty()) return 0.0;
        // 1e-9 / 1e-6 / 1e-30 below are numerical guards (log of 0, division by 0), not model thresholds.
        const double wrong = std::clamp(static_cast<double>(p.prior_wrong), 1e-9, 1.0 - 1e-9);
        const double log_u = -std::log(std::max(area, 1e-6) * 2.0 * M_PI);
        double wsum = 0.0;
        for (const auto& c : p.prior_modes) wsum += std::max(0.f, c.w);
        double lg = -std::numeric_limits<double>::infinity();
        for (const auto& c : p.prior_modes)
        {
            if (not (c.w > 0.f) or wsum <= 0.0) continue;
            Eigen::Vector3f d = pose - c.mu;
            d.z() = wrap(d.z());
            const Eigen::LDLT<Eigen::Matrix3f> ldlt(Eigen::Matrix3f(c.S + cov));
            const double maha = d.dot(ldlt.solve(d));
            const double logdet = ldlt.vectorD().array().abs().max(1e-30f).log().sum();
            if (not std::isfinite(maha)) continue;
            const double lc = std::log(static_cast<double>(c.w) / wsum) - 0.5 * maha
                              - 0.5 * (3.0 * std::log(2.0 * M_PI) + logdet);
            const double hm = std::max(lg, lc);
            lg = hm + std::log(std::exp(lg - hm) + std::exp(lc - hm));
        }
        const double a = std::log1p(-wrong) + lg, b = std::log(wrong) + log_u;
        const double hi = std::max(a, b);
        return hi + std::log(std::exp(a - hi) + std::exp(b - hi));
    }

    std::vector<Eigen::Vector2f> polish_points(const std::vector<Eigen::Vector2f>& pts_all, const Params& p)
    {
        return subsample(pts_all, 4 * p.max_points);   // the same set search() polishes and scores its modes on
    }

    float log_evidence(const Mode& m)
    {
        return m.log_lik + 0.5f * log_det3(m.cov);      // the same Laplace evidence search() weighs modes by
    }
} // namespace rc::reloc
