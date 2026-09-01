#include "wall_map.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#include "corner_visibility.h"

namespace rc::wallmap
{
    namespace
    {
        constexpr float kPi = static_cast<float>(M_PI);
        using linefit::wrap_pi;

        Eigen::Matrix2f rot2(float th)
        {
            Eigen::Matrix2f R;
            R << std::cos(th), -std::sin(th), std::sin(th), std::cos(th);
            return R;
        }

        /// Covariance of a (φ, d) pair from its information, or nullopt when not positive definite.
        std::optional<Eigen::Matrix2f> cov_of(const Eigen::Matrix2f& info)
        {
            if (not info.allFinite()) return std::nullopt;
            const float det = info.determinant();
            if (not (det > 0.f) or info(0, 0) <= 0.f or info(1, 1) <= 0.f) return std::nullopt;
            Eigen::Matrix2f c = info.inverse();
            if (not c.allFinite()) return std::nullopt;
            return c;
        }

        float chi2_of(const Eigen::Vector2f& r, const Eigen::Matrix2f& S)
        {
            const float det = S.determinant();
            if (not (det > 0.f)) return std::numeric_limits<float>::infinity();
            const float v = r.dot(S.inverse() * r);
            return std::isfinite(v) ? v : std::numeric_limits<float>::infinity();
        }

        /// Information-weighted fusion of two (φ, d) estimates. φ_b is first brought onto φ_a's branch.
        void fuse(float& phi, float& d, Eigen::Matrix2f& info,
                  float phi_b, float d_b, const Eigen::Matrix2f& info_b)
        {
            const float phib = phi + wrap_pi(phi_b - phi);
            const Eigen::Matrix2f L = info + info_b;
            const Eigen::Vector2f xa(phi, d), xb(phib, d_b);
            const float det = L.determinant();
            if (det > 0.f)
            {
                const Eigen::Vector2f x = L.inverse() * (info * xa + info_b * xb);
                if (x.allFinite()) { phi = wrap_pi(x(0)); d = x(1); }
            }
            else if (info_b.trace() > info.trace()) { phi = wrap_pi(phib); d = d_b; }
            info = L;
        }

        /// The systematic (model-error) covariance every innovation carries — see Params::map_sigma_d.
        Eigen::Matrix2f sys_cov(const Params& p)
        {
            return Eigen::Vector2f(p.map_sigma_phi_rad * p.map_sigma_phi_rad,
                                   p.map_sigma_d * p.map_sigma_d).asDiagonal();
        }
    } // namespace

    float point_to_segment_local(const Eigen::Vector2f& p, const Eigen::Vector2f& a, const Eigen::Vector2f& b)
    {
        const Eigen::Vector2f ab = b - a;
        const float t = std::clamp((p - a).dot(ab) / std::max(1e-9f, ab.squaredNorm()), 0.f, 1.f);
        return (p - (a + t * ab)).norm();
    }

    const WallLandmark* WallMap::find(std::uint64_t id) const
    {
        for (const auto& w : walls) if (w.id == id) return &w;
        return nullptr;
    }
    WallLandmark* WallMap::find(std::uint64_t id)
    {
        for (auto& w : walls) if (w.id == id) return &w;
        return nullptr;
    }
    int WallMap::index_of(std::uint64_t id) const
    {
        for (size_t i = 0; i < walls.size(); ++i) if (walls[i].id == id) return static_cast<int>(i);
        return -1;
    }

    void WallMap::to_map(float phi_r, float d_r, const Eigen::Vector3f& pose,
                         float& phi_m, float& d_m, Eigen::Matrix<float, 2, 3>& H)
    {
        phi_m = wrap_pi(phi_r + pose.z());
        const Eigen::Vector2f n = linefit::normal_of(phi_m);
        const Eigen::Vector2f tv = linefit::tangent_of(phi_m);
        const Eigen::Vector2f t = pose.head<2>();
        d_m = d_r + n.dot(t);
        H << 0.f, 0.f, 1.f,
             n.x(), n.y(), tv.dot(t);
    }

    WallMap::ClassChoice WallMap::classify(float phi) const
    {
        ClassChoice c;
        if (not theta0_born) { c.k = 0; c.eps = 0.f; c.cost = 0.f; return c; }
        const float var = params.manhattan_sigma_rad * params.manhattan_sigma_rad;
        float best = std::numeric_limits<float>::infinity();
        for (int k = 0; k < 4; ++k)
        {
            const float eps = wrap_pi(phi - theta0 - static_cast<float>(k) * kPi * 0.5f);
            const float cost = 0.5f * eps * eps / var;
            if (cost < best) { best = cost; c.k = k; c.eps = eps; c.cost = cost; }
        }
        const float off_cost = -std::log(std::clamp(params.manhattan_off_prior, 1e-6f, 1.f - 1e-6f));
        if (best > off_cost) { c.k = -1; c.eps = 0.f; c.cost = off_cost; }
        return c;
    }

    void WallMap::reclassify_all()
    {
        for (auto& w : walls)
        {
            const auto cls = classify(w.phi);
            w.k = cls.k;
            w.manhattan_var = (w.k >= 0) ? params.manhattan_sigma_rad * params.manhattan_sigma_rad : 0.f;
        }
    }

    WallLandmark WallMap::make_wall(float phi, float d, const Eigen::Matrix2f& info, float exist_seed,
                                    std::int64_t ts)
    {
        WallLandmark w;
        w.id = next_id_++;
        w.phi = wrap_pi(phi);
        w.d = d;
        w.information = info;
        w.exist_lodds = exist_seed;
        w.last_seen_ms = ts;
        return w;
    }

    Corner WallMap::intersect_walls(const WallLandmark& a, const WallLandmark& b, bool inferred)
    {
        Corner c;
        c.wall_a = a.id; c.wall_b = b.id; c.inferred = inferred;
        c.sigma = std::numeric_limits<float>::infinity();
        const auto p = linefit::intersect(a.line(), b.line());
        if (not p) return c;
        c.p = *p;
        Eigen::Matrix2f M; M.row(0) = a.normal().transpose(); M.row(1) = b.normal().transpose();
        const Eigen::Matrix2f Mi = M.inverse();
        const auto Ja = [&](const WallLandmark& w, int row)
        {
            Eigen::Matrix2f J;
            const Eigen::Vector2f col_d = Mi.col(row);
            J.col(0) = -col_d * w.tangent().dot(*p);   // ∂p/∂φ
            J.col(1) =  col_d;                         // ∂p/∂d
            return J;
        };
        const auto ca = cov_of(a.information), cb = cov_of(b.information);
        if (not ca or not cb) return c;
        const Eigen::Matrix2f Sp = Ja(a, 0) * (*ca) * Ja(a, 0).transpose() + Ja(b, 1) * (*cb) * Ja(b, 1).transpose();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eig(Sp);
        const float lmax = eig.eigenvalues().maxCoeff();
        c.sigma = (std::isfinite(lmax) and lmax >= 0.f) ? std::sqrt(lmax) : std::numeric_limits<float>::infinity();
        return c;
    }

    void WallMap::initialize_rect(const std::vector<Eigen::Vector2f>& rect_ccw)
    {
        walls.clear(); candidates.clear(); order.clear();
        if (rect_ccw.size() < 3) return;
        const Eigen::Matrix2f prior_info =
            Eigen::Vector2f(1.f / (params.rect_prior_sigma_phi_rad * params.rect_prior_sigma_phi_rad),
                            1.f / (params.rect_prior_sigma_d * params.rect_prior_sigma_d)).asDiagonal();
        const int N = static_cast<int>(rect_ccw.size());
        for (int i = 0; i < N; ++i)
        {
            const Eigen::Vector2f a = rect_ccw[static_cast<size_t>(i)];
            const Eigen::Vector2f b = rect_ccw[static_cast<size_t>((i + 1) % N)];
            const Eigen::Vector2f e = (b - a).normalized();
            const Eigen::Vector2f n(-e.y(), e.x());     // left of the CCW edge = inside
            WallLandmark w = make_wall(std::atan2(n.y(), n.x()), n.dot(a), prior_info, params.birth_nats, 0);
            const Eigen::Vector2f tv = w.tangent();
            w.s_min = std::min(tv.dot(a), tv.dot(b));
            w.s_max = std::max(tv.dot(a), tv.dot(b));
            w.has_extent = true;
            walls.push_back(w);
            order.push_back(w.id);
        }
        theta0 = walls.front().phi;
        theta0_born = true;
        theta0_information = 1.f / (params.rect_prior_sigma_phi_rad * params.rect_prior_sigma_phi_rad);
        reclassify_all();
    }

    // ─────────────────────────────────────────────────────────────────────────────────────────────
    //  Existence (the step-back operator), per extent bin.
    // ─────────────────────────────────────────────────────────────────────────────────────────────
    void WallMap::update_existence(const std::vector<Eigen::Vector2f>& pts_robot, const Eigen::Vector3f& pose,
                                   FrameResult& fr)
    {
        if (walls.empty() or pts_robot.empty()) return;
        const Eigen::Matrix2f Rm = rot2(pose.z());
        const Eigen::Matrix2f Rt = Rm.transpose();
        const Eigen::Vector2f t = pose.head<2>();
        const Eigen::Vector2f origin = Eigen::Vector2f::Zero();
        const float bw = std::max(0.05f, params.exist_bin_m);
        const float lo_clamp = -1.5f * params.birth_nats, hi_clamp = 2.f * params.birth_nats;

        for (auto& w : walls)
        {
            if (not w.has_extent or w.s_max - w.s_min < bw) continue;
            const float seed = std::clamp(w.exist_lodds, 0.f, params.birth_nats);
            if (w.exist_bins.empty())
            {
                w.bins_s0 = w.s_min;
                w.exist_bins.assign(static_cast<size_t>(std::max(1.f, std::ceil((w.s_max - w.s_min) / bw))), seed);
            }
            while (w.bins_s0 > w.s_min)
            { w.exist_bins.insert(w.exist_bins.begin(), seed); w.bins_s0 -= bw; }
            while (w.bins_s0 + bw * static_cast<float>(w.exist_bins.size()) < w.s_max - 0.5f * bw)
                w.exist_bins.push_back(seed);
            const int nb = static_cast<int>(w.exist_bins.size());

            const Eigen::Vector2f n = w.normal(), tv = w.tangent();
            const Eigen::Vector2f a_r = Rt * ((n * w.d + tv * w.s_min) - t);
            const Eigen::Vector2f b_r = Rt * ((n * w.d + tv * w.s_max) - t);
            const Eigen::Vector2f n_r = Rt * n;

            std::vector<float> sup(static_cast<size_t>(nb), 0.f), ref(static_cast<size_t>(nb), 0.f);
            for (const auto& p : pts_robot)
            {
                const float r = p.norm();
                if (r < 1e-3f) continue;
                const Eigen::Vector2f dir = p / r;
                const auto hit = corner_visibility::ray_segment_t(origin, dir, a_r, b_r);
                if (not hit) continue;
                const float s_hit = tv.dot(Rm * (dir * *hit) + t);
                const int bin = std::clamp(static_cast<int>(std::floor((s_hit - w.bins_s0) / bw)), 0, nb - 1);
                const float inc = std::abs(n_r.dot(dir));
                const float dr  = r - *hit;
                if (std::abs(dr) <= params.huber_delta)      sup[static_cast<size_t>(bin)] += inc;
                else if (dr > params.huber_delta)            ref[static_cast<size_t>(bin)] += inc;
            }
            for (int bin = 0; bin < nb; ++bin)
            {
                const float denom = sup[static_cast<size_t>(bin)] + params.exist_refute_pdet * ref[static_cast<size_t>(bin)];
                if (denom <= 0.f) continue;
                const float delta = std::clamp((sup[static_cast<size_t>(bin)] - params.exist_refute_pdet * ref[static_cast<size_t>(bin)]) / std::max(denom, 1.f), -1.f, 1.f);
                w.exist_bins[static_cast<size_t>(bin)] = std::clamp(w.exist_bins[static_cast<size_t>(bin)] + delta, lo_clamp, hi_clamp);
            }
            int first = 0, last = nb - 1;
            while (first < nb and w.exist_bins[static_cast<size_t>(first)] < -params.birth_nats) ++first;
            while (last >= first and w.exist_bins[static_cast<size_t>(last)] < -params.birth_nats) --last;
            const float testable = (first > last) ? 0.f
                : std::min(w.s_max, w.bins_s0 + bw * static_cast<float>(last + 1))
                  - std::max(w.s_min, w.bins_s0 + bw * static_cast<float>(first));
            if (first > last or testable < 2.f * bw)
            { w.exist_lodds = lo_clamp; continue; }
            if (first > 0 or last < nb - 1)
            {
                w.exist_bins.assign(w.exist_bins.begin() + first, w.exist_bins.begin() + last + 1);
                w.bins_s0 += bw * static_cast<float>(first);
                w.s_min = std::max(w.s_min, w.bins_s0);
                w.s_max = std::min(w.s_max, w.bins_s0 + bw * static_cast<float>(w.exist_bins.size()));
            }
            w.exist_lodds = *std::max_element(w.exist_bins.begin(), w.exist_bins.end());
        }

        // Deaths: refuted through, and spliced OUT so the cycle heals immediately.
        for (int i = static_cast<int>(walls.size()) - 1; i >= 0; --i)
        {
            if (walls[static_cast<size_t>(i)].exist_lodds > lo_clamp + 1e-3f) continue;
            const auto& w = walls[static_cast<size_t>(i)];
            fr.deaths_info.push_back({w.id, w.exist_lodds, w.frames_seen, w.points_seen});
            fr.deaths++;
            splice_out(w.id);   // erases from walls too — index i is not reused after this
        }
    }

    void WallMap::splice_out(std::uint64_t id)
    {
        order.erase(std::remove(order.begin(), order.end(), id), order.end());
        if (const int i = index_of(id); i >= 0) walls.erase(walls.begin() + i);
        heal_order();
    }

    void WallMap::heal_order()
    {
        // Heal: adjacent duplicates collapse; adjacent near-parallel edges have no corner — drop the
        // weaker one; an A,B,A sandwich is a zero-width spike (its two corners coincide) — drop B.
        // Runs after every death, merge and commit: a merge that fuses a jog into a real wall is what
        // creates the sandwiches.
        bool again = true;
        while (again and order.size() >= 2)
        {
            again = false;
            const int N = static_cast<int>(order.size());
            for (int i = 0; i < N; ++i)
            {
                const std::uint64_t ida = order[static_cast<size_t>(i)];
                const std::uint64_t idb = order[static_cast<size_t>((i + 1) % N)];
                if (ida == idb)
                { order.erase(order.begin() + i); again = true; break; }
                const auto* A = find(ida); const auto* B = find(idb);
                if (A == nullptr or B == nullptr)
                { order.erase(order.begin() + ((A == nullptr) ? i : (i + 1) % N)); again = true; break; }
                const float sin_ab = std::abs(A->normal().x() * B->normal().y() - A->normal().y() * B->normal().x());
                if (sin_ab < 5e-2f)
                {
                    const bool drop_a = A->points_seen < B->points_seen;
                    order.erase(order.begin() + (drop_a ? i : (i + 1) % N));
                    again = true; break;
                }
                // A,B,A — a zero-width spike (both corners coincide): drop B, the duplicate collapses next pass.
                const std::uint64_t idc = order[static_cast<size_t>((i + 2) % N)];
                if (ida == idc and N >= 3)
                { order.erase(order.begin() + (i + 1) % N); again = true; break; }
            }
        }
        // Walls no longer referenced by the order are not part of the model any more.
        for (int i = static_cast<int>(walls.size()) - 1; i >= 0; --i)
            if (std::find(order.begin(), order.end(), walls[static_cast<size_t>(i)].id) == order.end())
                walls.erase(walls.begin() + i);
    }

    // ─────────────────────────────────────────────────────────────────────────────────────────────
    //  Splice jumps.
    // ─────────────────────────────────────────────────────────────────────────────────────────────
    int WallMap::try_splice(const Candidate& c, FrameResult& fr, std::int64_t ts)
    {
        if (order.size() < 3) return -1;
        const int N = static_cast<int>(order.size());
        // Chosen across ALL hosts and cases by SCORE, not first-valid: a valid polygon cannot tell
        // the right corner from a wrong one (the chamfer once replaced a dead ghost at the ghost's
        // position, wherever that was). The score is how much of C's OBSERVED extent lies on the
        // polygon edge C would get — evidence placing the jump, not geometry alone.
        struct Best { float score = -1.f; std::vector<WallLandmark> new_walls; std::vector<std::uint64_t> ord; std::uint64_t host = 0; };
        Best best_v;
        // Occam on ties: a notch side wall and a free-standing stub both start at the host and run
        // interior, and both variants place the candidate's full extent — the SCORE ties. At equal
        // evidence the simpler structure (fewer new walls) must win, or the stub (listed first)
        // hijacks every notch. Lexicographic (score, −new walls).
        const auto better = [&](float score, size_t n_new)
        {
            if (score > best_v.score + 1e-3f) return true;
            if (score < best_v.score - 1e-3f) return false;
            return best_v.score >= 0.f and n_new < best_v.new_walls.size();
        };
        const Eigen::Vector2f n_c = linefit::normal_of(c.phi), t_c = linefit::tangent_of(c.phi);
        const Eigen::Vector2f c0 = n_c * c.d + t_c * c.s_min, c1 = n_c * c.d + t_c * c.s_max;
        const float tol = params.splice_end_tol;

        for (int i = 0; i < N; ++i)
        {
            WallLandmark* E = find(order[static_cast<size_t>(i)]);
            if (E == nullptr or not E->has_extent) continue;
            const float dphi = wrap_pi(c.phi - E->phi);
            const bool parallel = std::abs(dphi) < kPi / 4.f;
            // Anti-parallel (a face looking the other way) is not an edge of THIS boundary.
            if (std::abs(dphi) > 3.f * kPi / 4.f) continue;

            const Eigen::Vector2f tE = E->tangent();
            float a = tE.dot(c0), b = tE.dot(c1);
            if (a > b) std::swap(a, b);
            if (b < E->s_min - tol or a > E->s_max + tol) continue;   // no overlap with the host span
            if (parallel)
            {
                // A parallel candidate at (nearly) the host's own offset is a twin, not a jump.
                const float dd = std::abs(c.d - E->d);
                if (dd < 3.f * params.map_sigma_d) continue;
            }

            const std::uint64_t prev_id = order[static_cast<size_t>((i + N - 1) % N)];
            const std::uint64_t next_id = order[static_cast<size_t>((i + 1) % N)];
            const bool at_lo = a <= E->s_min + tol;   // reaches the corner shared with NEXT (s decreasing walk)
            const bool at_hi = b >= E->s_max - tol;   // reaches the corner shared with PREV

            // Jog line ⊥ to E through tangent coordinate s0: normal ±t_E, offset ±s0. The normal must
            // point INTO the room; both signs are tried and the polygon validation plus an interior
            // test picks the survivor.
            const auto make_jog = [&](float s0, int sign) -> WallLandmark
            {
                const float phi_j = wrap_pi(E->phi + static_cast<float>(sign) * kPi * 0.5f);
                const Eigen::Vector2f n_j = linefit::normal_of(phi_j);
                const float d_j = n_j.dot(tE * s0);
                const Eigen::Matrix2f prior_info =
                    Eigen::Vector2f(1.f / (params.rect_prior_sigma_phi_rad * params.rect_prior_sigma_phi_rad),
                                    1.f / (params.rect_prior_sigma_d * params.rect_prior_sigma_d)).asDiagonal();
                return make_wall(phi_j, d_j, prior_info, 0.5f * params.birth_nats, ts);
            };

            // Enumerate order variants (walls listed by VALUE first; committed only on validation).
            struct Variant { std::vector<WallLandmark> new_walls; std::vector<std::uint64_t> ord; };
            std::vector<Variant> variants;
            const auto splice_at = [&](const std::vector<std::uint64_t>& inserted, bool replace_host_after) -> std::vector<std::uint64_t>
            {
                std::vector<std::uint64_t> o;
                for (int j = 0; j < N; ++j)
                {
                    o.push_back(order[static_cast<size_t>(j)]);
                    if (j == i)
                    {
                        for (auto id : inserted) o.push_back(id);
                        if (replace_host_after) o.push_back(E->id);   // the host line resumes (notch)
                    }
                }
                return o;
            };

            WallLandmark C = make_wall(c.phi, c.d, c.information, params.birth_nats, ts);
            C.s_min = c.s_min; C.s_max = c.s_max; C.has_extent = c.npts > 0;
            C.frames_seen = c.frames; C.points_seen = c.npts;

            // RUN REPLACEMENT: C substitutes a consecutive RUN of edges starting at E. The single-host
            // moves cannot express "this 8-metre wall replaces the four junk edges currently standing
            // in for it" — every local variant still crosses the rest of the run and fails validation
            // (measured: a 25k-point real wall homeless through 792 rejections). Evidence-gated: the
            // run's combined support must be LESS than the candidate's own, so a supported stretch of
            // boundary can never be amputated by one loud line.
            for (int L = 1; L <= std::min(5, N - 3); ++L)
            {
                int run_pts = 0;
                bool run_ok = true;
                for (int j = 0; j < L; ++j)
                {
                    const auto* Wj = find(order[static_cast<size_t>((i + j) % N)]);
                    if (Wj == nullptr) { run_ok = false; break; }
                    run_pts += Wj->points_seen;
                }
                if (not run_ok or run_pts >= c.npts) continue;
                Variant v;
                v.new_walls = {C};
                std::vector<std::uint64_t> o;
                for (int j = 0; j < N; ++j)
                {
                    const int rel = (j - i + N) % N;
                    if (rel == 0) o.push_back(C.id);
                    if (rel < L) continue;               // the run is gone
                    o.push_back(order[static_cast<size_t>(j)]);
                }
                v.ord = std::move(o);
                variants.push_back(std::move(v));
            }

            // REPLACEMENT: C substitutes E outright. Allowed on the evidence, not on geometry alone —
            // validation cannot tell "the prior's side was wrong" from "amputate a supported wall"
            // (swapping the east wall for a notch line also builds a valid polygon). A host that has
            // NEVER been confirmed (frames_seen == 0: a tilted OBB side the gate refused for ever) may
            // be substituted; a supported host only when the candidate covers both its ends.
            if (E->frames_seen == 0 or (at_lo and at_hi))
            {
                Variant v;
                v.new_walls = {C};
                std::vector<std::uint64_t> o = order;
                o[static_cast<size_t>(i)] = C.id;
                v.ord = std::move(o);
                variants.push_back(std::move(v));
            }
            for (int sign : {+1, -1})
            {
                if (parallel and not at_lo and not at_hi)
                {
                    // Middle notch/extension: E, jog(b), C, jog(a), E.
                    Variant v;
                    v.new_walls = {make_jog(b, sign), C, make_jog(a, -sign)};
                    v.ord = splice_at({v.new_walls[0].id, v.new_walls[1].id, v.new_walls[2].id}, true);
                    variants.push_back(std::move(v));
                }
                if (parallel and at_lo and not at_hi)
                {
                    // Corner cut with NEXT: E, jog(b), C, next.
                    Variant v;
                    v.new_walls = {make_jog(b, sign), C};
                    v.ord = splice_at({v.new_walls[0].id, v.new_walls[1].id}, false);
                    variants.push_back(std::move(v));
                }
                if (parallel and at_hi and not at_lo)
                {
                    // Corner cut with PREV: prev, C, jog(a), E.
                    Variant v;
                    v.new_walls = {C, make_jog(a, sign)};
                    std::vector<std::uint64_t> o;
                    for (int j = 0; j < N; ++j)
                    {
                        if (j == i) { o.push_back(v.new_walls[0].id); o.push_back(v.new_walls[1].id); }
                        o.push_back(order[static_cast<size_t>(j)]);
                    }
                    v.ord = std::move(o);
                    variants.push_back(std::move(v));
                }
            }
            // STUB: a candidate roughly ⊥ to the host whose extent runs from the host's line INTO
            // the interior is not a T-junction to ignore — it is a WALL IN THE MIDDLE of the space,
            // and the boundary wraps around it: [E, near-face, tip, far-face, E]. The observed
            // candidate is one face; its mirror starts at a thin-wall prior thickness and the tip at
            // the deep end; evidence (and the other face's own candidate, which twin-fuses into the
            // mirror) refines both. This is what makes the room CONCAVE around interior walls.
            if (not parallel and params.enable_stub_jumps)
            {
                const float e0 = E->normal().dot(c0) - E->d;   // distances from the host line,
                const float e1 = E->normal().dot(c1) - E->d;   // positive = interior side
                const float near_e = std::min(e0, e1), far_e = std::max(e0, e1);
                const Eigen::Vector2f near_p = (e0 <= e1) ? c0 : c1;
                const Eigen::Vector2f far_p  = (e0 <= e1) ? c1 : c0;
                const float s_att = tE.dot(near_p);
                if (near_e >= -tol and near_e <= tol
                    and far_e >= 4.f * params.exist_bin_m
                    and s_att > E->s_min + tol and s_att < E->s_max - tol)
                {
                    const float t0 = 0.12f;   // ⚠ thin-wall thickness prior (m); evidence refines it
                    const Eigen::Matrix2f prior_info =
                        Eigen::Vector2f(1.f / (params.rect_prior_sigma_phi_rad * params.rect_prior_sigma_phi_rad),
                                        1.f / (params.rect_prior_sigma_d * params.rect_prior_sigma_d)).asDiagonal();
                    // Mirror face: parallel to C at t0 on the side OPPOSITE C's normal, facing back.
                    WallLandmark M = make_wall(wrap_pi(c.phi + kPi), t0 - c.d, prior_info,
                                               0.5f * params.birth_nats, ts);
                    // Tip: ⊥ to C through the deep end, both normal signs tried via variants.
                    const float s_far = linefit::tangent_of(c.phi).dot(far_p);
                    for (int tsign : {+1, -1})
                    {
                        const float phi_t = wrap_pi(c.phi + static_cast<float>(tsign) * kPi * 0.5f);
                        WallLandmark T = make_wall(phi_t,
                                                   linefit::normal_of(phi_t).dot(linefit::tangent_of(c.phi) * s_far),
                                                   prior_info, 0.5f * params.birth_nats, ts);
                        for (bool c_first : {true, false})
                        {
                            Variant v;
                            WallLandmark Cw = C, Mw = M, Tw = T;
                            v.new_walls = c_first ? std::vector<WallLandmark>{Cw, Tw, Mw}
                                                  : std::vector<WallLandmark>{Mw, Tw, Cw};
                            v.ord = splice_at({v.new_walls[0].id, v.new_walls[1].id, v.new_walls[2].id}, true);
                            variants.push_back(std::move(v));
                        }
                    }
                }
            }

            // Direct connection, no jog — for ANY angle: a 45° chamfer sits exactly on the
            // parallel/oblique boundary, and when the candidate's endpoint lies ON the host line the
            // jog is zero-length by geometry. Validation + score decide against the jog variants.
            if (at_lo)
            {
                Variant v; v.new_walls = {C};
                v.ord = splice_at({C.id}, false);
                variants.push_back(std::move(v));
            }
            if (at_hi)
            {
                Variant v; v.new_walls = {C};
                std::vector<std::uint64_t> o;
                for (int j = 0; j < N; ++j)
                {
                    if (j == i) o.push_back(C.id);
                    o.push_back(order[static_cast<size_t>(j)]);
                }
                v.ord = std::move(o);
                variants.push_back(std::move(v));
            }
            (void)prev_id; (void)next_id;

            for (auto& v : variants)
            {
                // Trial: add the new walls, build, validate, score, roll back.
                const size_t walls_before = walls.size();
                for (const auto& w : v.new_walls) walls.push_back(w);
                Polygon poly = build_from(v.ord);
                bool ok = poly.closed;
                std::uint64_t c_id = v.new_walls.back().id;
                for (const auto& w : v.new_walls) if (std::abs(wrap_pi(w.phi - c.phi)) < 1e-4f) c_id = w.id;
                float score = -1.f;
                if (ok)
                {
                    // LOCAL interior test, exact for any simple polygon: on a CCW cycle the interior
                    // is LEFT of travel, so each edge's walk direction must have the wall's inward
                    // normal on its left. The old "normal points at the centroid" test was a
                    // convexity assumption in disguise — it would refuse every deep concavity (the
                    // mid-space walls the room wraps around) by construction.
                    for (size_t e = 0; e < poly.verts.size() and ok; ++e)
                    {
                        const Eigen::Vector2f dirv = poly.verts[(e + 1) % poly.verts.size()] - poly.verts[e];
                        if (dirv.norm() < 0.05f) { ok = false; break; }
                        const auto* We = find(poly.wall_of_edge[e]);
                        if (We == nullptr) { ok = false; break; }
                        const Eigen::Vector2f left(-dirv.y(), dirv.x());
                        if (left.dot(We->normal()) <= 0.f) { ok = false; break; }
                    }
                }
                if (ok)
                {
                    // Every corner of the trial cycle must sit NEAR the observed evidence of the two
                    // edges forming it. A cycle can look valid while a near-parallel adjacency throws
                    // a corner kilometres out (run replacement produced a 4e6 m "corner"): a corner
                    // beyond any observed extent is speculation, and commits are evidence-gated.
                    const int np = static_cast<int>(poly.verts.size());
                    for (int e2 = 0; e2 < np and ok; ++e2)
                    {
                        const auto* W2 = find(poly.wall_of_edge[static_cast<size_t>(e2)]);
                        if (W2 == nullptr) { ok = false; break; }
                        float smin = 0.f, smax = 0.f; bool have = false;
                        if (W2->has_extent) { smin = W2->s_min; smax = W2->s_max; have = true; }
                        else if (W2->id == c_id) { smin = c.s_min; smax = c.s_max; have = true; }
                        if (not have) continue;
                        const Eigen::Vector2f tv2 = W2->tangent();
                        for (const Eigen::Vector2f& vv : {poly.verts[static_cast<size_t>(e2)],
                                                          poly.verts[static_cast<size_t>((e2 + 1) % np)]})
                        {
                            const float sc = tv2.dot(vv);
                            if (sc < smin - 1.0f or sc > smax + 1.0f) { ok = false; break; }
                        }
                    }
                }
                if (ok)
                {
                    // Score: length of C's observed extent that lies on C's polygon edge.
                    for (size_t e = 0; e < poly.wall_of_edge.size(); ++e)
                        if (poly.wall_of_edge[e] == c_id)
                        {
                            const Eigen::Vector2f tc = linefit::tangent_of(c.phi);
                            float s0 = tc.dot(poly.verts[e]);
                            float s1 = tc.dot(poly.verts[(e + 1) % poly.verts.size()]);
                            if (s0 > s1) std::swap(s0, s1);
                            score = std::max(score, std::min(s1, c.s_max) - std::max(s0, c.s_min));
                        }
                }
                walls.resize(walls_before);
                // ⚠ fraction bar: a jump must place the MAJORITY of its candidate's observed extent on
                // the resulting edge — "some overlap" let wrong-position commits win over nothing.
                if (ok and score > 0.5f * std::max(params.exist_bin_m, c.s_max - c.s_min) and better(score, v.new_walls.size()))
                    best_v = Best{score, v.new_walls, v.ord, E->id};
            }
        }
        if (best_v.score < 0.f) return -1;
        // Commit the winner.
        for (const auto& w : best_v.new_walls) walls.push_back(w);
        order = best_v.ord;
        for (int wi = static_cast<int>(walls.size()) - 1; wi >= 0; --wi)
            if (std::find(order.begin(), order.end(), walls[static_cast<size_t>(wi)].id) == order.end())
                walls.erase(walls.begin() + wi);
        heal_order();
        reclassify_all();
        seed_extents_from_polygon();
        BirthInfo bi;
        bi.id = best_v.new_walls.back().id;
        for (const auto& w : best_v.new_walls) if (std::abs(wrap_pi(w.phi - c.phi)) < 1e-4f) bi.id = w.id;
        bi.phi = c.phi; bi.d = c.d; bi.npts = c.npts; bi.frames = c.frames;
        bi.nearest_wall = best_v.host;
        fr.births_info.push_back(bi);
        fr.births++;
        return index_of(bi.id);
    }

    void WallMap::seed_extents_from_polygon()
    {
        const Polygon poly = build_polygon();
        if (not poly.closed) return;
        const int n = static_cast<int>(poly.verts.size());
        for (int i = 0; i < n; ++i)
        {
            auto* w = find(poly.wall_of_edge[static_cast<size_t>(i)]);
            if (w == nullptr or w->has_extent) continue;
            const Eigen::Vector2f tv = w->tangent();
            const float s0 = tv.dot(poly.verts[static_cast<size_t>(i)]);
            const float s1 = tv.dot(poly.verts[static_cast<size_t>((i + 1) % n)]);
            w->s_min = std::min(s0, s1);
            w->s_max = std::max(s0, s1);
            w->has_extent = true;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────────────────────
    //  The per-frame pipeline.
    // ─────────────────────────────────────────────────────────────────────────────────────────────
    FrameResult WallMap::observe(const wallseg::Result& seg, const std::vector<Eigen::Vector2f>& pts_robot,
                                 const Eigen::VectorXf& weights, const Eigen::Vector3f& pose,
                                 const Eigen::Matrix3f& pose_cov, std::int64_t timestamp_ms)
    {
        FrameResult fr;
        {
            // Free-space evidence first: beams traverse free space and end on matter.
            if (not fgrid.ready()) fgrid.init(pose.head<2>(), params.sensor_range);
            const Eigen::Matrix2f Rm = rot2(pose.z());
            std::vector<Eigen::Vector2f> pm;
            pm.reserve(pts_robot.size());
            for (const auto& pr : pts_robot)
                pm.push_back(Rm * pr + pose.head<2>());
            fgrid.mark(pose.head<2>(), pm);
        }
        update_existence(pts_robot, pose, fr);

        const int S = static_cast<int>(seg.segments.size());
        const int W = static_cast<int>(walls.size());
        fr.seg_to_wall.assign(static_cast<size_t>(S), -1);
        fr.seg_pda.assign(static_cast<size_t>(S), 0.f);
        fr.seg_to_candidate.assign(static_cast<size_t>(S), -1);
        fr.unexplained_points = static_cast<int>(seg.unexplained.size());
        const Eigen::Matrix2f R = rot2(pose.z());
        const Eigen::Vector2f t = pose.head<2>();

        // ── Every segment in the map frame, with the covariance the association test needs ───────
        struct SegMap { float phi = 0.f, d = 0.f; Eigen::Matrix2f Sigma = Eigen::Matrix2f::Zero(); bool ok = false; };
        std::vector<SegMap> sm(static_cast<size_t>(S));
        for (int s = 0; s < S; ++s)
        {
            const auto& sg = seg.segments[static_cast<size_t>(s)];
            Eigen::Matrix<float, 2, 3> H;
            to_map(sg.phi, sg.d, pose, sm[s].phi, sm[s].d, H);
            const auto cov_r = cov_of(sg.info_phi_d);
            if (not cov_r) continue;
            Eigen::Matrix2f J = Eigen::Matrix2f::Identity();
            J(1, 0) = linefit::tangent_of(sm[s].phi).dot(t);
            sm[s].Sigma = J * (*cov_r) * J.transpose() + H * pose_cov * H.transpose();
            sm[s].ok = sm[s].Sigma.allFinite();
        }

        // ── Association: Mahalanobis under the innovation covariance, MANY-TO-ONE, PDA ───────────
        std::vector<std::vector<float>> chi2(static_cast<size_t>(S), std::vector<float>(static_cast<size_t>(W), std::numeric_limits<float>::infinity()));
        for (int s = 0; s < S; ++s)
        {
            if (not sm[s].ok) continue;
            for (int w = 0; w < W; ++w)
            {
                const auto& wl = walls[static_cast<size_t>(w)];
                const auto cov_w = cov_of(wl.information);
                const Eigen::Matrix2f Sm = sys_cov(params)
                                         + (cov_w ? (sm[s].Sigma + *cov_w).eval() : sm[s].Sigma);
                const Eigen::Vector2f r(wrap_pi(sm[s].phi - wl.phi), sm[s].d - wl.d);
                chi2[s][w] = chi2_of(r, Sm);
            }
        }
        for (int s = 0; s < S; ++s)
        {
            int w = -1;
            float best = std::numeric_limits<float>::infinity();
            for (int w2 = 0; w2 < W; ++w2)
                if (chi2[s][w2] <= params.assoc_chi2 and chi2[s][w2] < best) { best = chi2[s][w2]; w = w2; }
            if (w < 0) continue;
            fr.segments_associated++;
            float denom = 0.f;
            for (int w2 = 0; w2 < W; ++w2)
                if (chi2[s][w2] <= params.assoc_chi2) denom += std::exp(-0.5f * chi2[s][w2]);
            const float pda = (denom > 0.f) ? std::exp(-0.5f * chi2[s][w]) / denom : 1.f;
            fr.seg_to_wall[static_cast<size_t>(s)] = w;
            fr.seg_pda[static_cast<size_t>(s)] = pda;

            const auto& sg = seg.segments[static_cast<size_t>(s)];
            auto& wl = walls[static_cast<size_t>(w)];
            WallAssoc a;
            a.wall_id = wl.id;
            a.pda = pda;
            a.pts.resize(static_cast<long>(sg.inliers.size()), 2);
            a.weights.resize(static_cast<long>(sg.inliers.size()));
            const Eigen::Vector2f tv = wl.tangent();
            for (size_t i = 0; i < sg.inliers.size(); ++i)
            {
                const Eigen::Vector2f& p = pts_robot[static_cast<size_t>(sg.inliers[i])];
                a.pts(static_cast<long>(i), 0) = p.x();
                a.pts(static_cast<long>(i), 1) = p.y();
                a.weights(static_cast<long>(i)) = (weights.size() > sg.inliers[i]) ? weights(sg.inliers[i]) : 1.f;
                const float sc = tv.dot(R * p + t);
                if (not wl.has_extent) { wl.s_min = wl.s_max = sc; wl.has_extent = true; }
                else { wl.s_min = std::min(wl.s_min, sc); wl.s_max = std::max(wl.s_max, sc); }
            }
            wl.frames_seen++;
            wl.points_seen += sg.npts;
            wl.last_seen_ms = timestamp_ms;
            fr.assoc.push_back(std::move(a));
        }

        // ── Unexplained segments → candidates ───────────────────────────────────────────────────
        for (auto& c : candidates) c.this_frame_seg = -1;
        for (int s = 0; s < S; ++s)
        {
            if (fr.seg_to_wall[static_cast<size_t>(s)] >= 0 or not sm[s].ok) continue;
            const auto& sg = seg.segments[static_cast<size_t>(s)];
            const Eigen::Matrix2f info_m = sm[s].Sigma.inverse();
            if (not info_m.allFinite()) continue;

            int best = -1; float best_chi2 = std::numeric_limits<float>::infinity();
            for (size_t ci = 0; ci < candidates.size(); ++ci)
            {
                const auto& c = candidates[ci];
                if (c.this_frame_seg >= 0) continue;
                const auto cov_c = cov_of(c.information);
                const Eigen::Matrix2f Sm = sys_cov(params)
                                         + (cov_c ? (sm[s].Sigma + *cov_c).eval() : sm[s].Sigma);
                const Eigen::Vector2f r(wrap_pi(sm[s].phi - c.phi), sm[s].d - c.d);
                const float c2 = chi2_of(r, Sm);
                if (c2 <= params.assoc_chi2 and c2 < best_chi2) { best_chi2 = c2; best = static_cast<int>(ci); }
            }
            if (best < 0)
            {
                Candidate c;
                c.phi = sm[s].phi; c.d = sm[s].d;
                c.information = info_m;
                c.first_ms = timestamp_ms;
                candidates.push_back(c);
                best = static_cast<int>(candidates.size()) - 1;
            }
            else
                fuse(candidates[static_cast<size_t>(best)].phi, candidates[static_cast<size_t>(best)].d,
                     candidates[static_cast<size_t>(best)].information, sm[s].phi, sm[s].d, info_m);

            auto& c = candidates[static_cast<size_t>(best)];
            c.this_frame_seg = s;
            c.frames++;
            c.last_ms = timestamp_ms;
            fr.seg_to_candidate[static_cast<size_t>(s)] = best;
            const Eigen::Vector2f n = linefit::normal_of(c.phi);
            const Eigen::Vector2f tv = linefit::tangent_of(c.phi);
            const float extent = std::max(sg.s_max - sg.s_min, params.obs_sigma);
            for (int idx : sg.inliers)
            {
                const Eigen::Vector2f q = R * pts_robot[static_cast<size_t>(idx)] + t;
                const float r = n.dot(q) - c.d;
                c.gain += wallseg::point_gain_nats(r, sg.sigma2(), seg.clutter_area, extent);
                const float sc = tv.dot(q);
                if (c.npts == 0) { c.s_min = c.s_max = sc; }
                else { c.s_min = std::min(c.s_min, sc); c.s_max = std::max(c.s_max, sc); }
                c.npts++;
            }
        }

        // ── Jumps by model comparison ───────────────────────────────────────────────────────────
        const float var_phi_prior = kPi * kPi / 3.f;
        const float var_d_prior   = params.sensor_range * params.sensor_range / 12.f;
        const Eigen::Matrix2f L_prior = Eigen::Vector2f(1.f / var_phi_prior, 1.f / var_d_prior).asDiagonal();
        for (int ci = static_cast<int>(candidates.size()) - 1; ci >= 0; --ci)
        {
            auto& c = candidates[static_cast<size_t>(ci)];
            if (c.this_frame_seg < 0 or c.frames < params.birth_min_frames) continue;
            const Eigen::Matrix2f L_post = L_prior + c.information;
            const float occam = 0.5f * std::log(std::max(L_post.determinant(), 1e-30f) / L_prior.determinant());
            const auto cls = classify(c.phi);
            const float dF = c.gain - occam - cls.cost;
            if (not std::isfinite(dF) or dF <= params.birth_nats) continue;

            // "Two edges cannot occupy the same line": a candidate the gate cannot separate from an
            // existing edge IS that edge — fuse its evidence in rather than jumping.
            int nearest_idx = -1; float nearest_chi2 = -1.f; std::uint64_t nearest_id = 0;
            for (size_t wi = 0; wi < walls.size(); ++wi)
            {
                const auto& wl = walls[wi];
                const auto cov_w = cov_of(wl.information);
                const auto cov_c = cov_of(c.information);
                const Eigen::Matrix2f Sm = sys_cov(params) + (cov_w ? *cov_w : Eigen::Matrix2f::Zero())
                                         + (cov_c ? *cov_c : Eigen::Matrix2f::Zero());
                const Eigen::Vector2f r(wrap_pi(c.phi - wl.phi), c.d - wl.d);
                const float c2 = chi2_of(r, Sm);
                if (nearest_chi2 < 0.f or c2 < nearest_chi2)
                { nearest_chi2 = c2; nearest_idx = static_cast<int>(wi); nearest_id = wl.id; }
            }
            if (nearest_idx >= 0 and nearest_chi2 <= params.assoc_chi2)
            {
                auto& wl = walls[static_cast<size_t>(nearest_idx)];
                fuse(wl.phi, wl.d, wl.information, c.phi, c.d, c.information);
                if (c.npts > 0)
                {
                    if (not wl.has_extent) { wl.s_min = c.s_min; wl.s_max = c.s_max; wl.has_extent = true; }
                    else { wl.s_min = std::min(wl.s_min, c.s_min); wl.s_max = std::max(wl.s_max, c.s_max); }
                }
                candidates.erase(candidates.begin() + ci);
                fr.twins_fused++;
                continue;
            }

            const int seg_idx = c.this_frame_seg;
            const float dF_committed = dF;
            const int w = try_splice(c, fr, timestamp_ms);
            if (w < 0)
            {
                fr.splice_rejected++;   // kept: the polygon may grow a place for it later
                continue;
            }
            fr.births_info.back().nearest_chi2 = nearest_chi2;
            fr.births_info.back().dF = dF_committed;
            fr.births_info.back().seg = seg_idx;
            (void)nearest_id;
            // The completing segment is this frame's observation of the new edge.
            const auto& sg = seg.segments[static_cast<size_t>(seg_idx)];
            auto& wl = walls[static_cast<size_t>(w)];
            fr.seg_to_wall[static_cast<size_t>(seg_idx)] = w;
            fr.seg_pda[static_cast<size_t>(seg_idx)] = 1.f;
            fr.seg_to_candidate[static_cast<size_t>(seg_idx)] = -1;
            WallAssoc a;
            a.wall_id = wl.id; a.pda = 1.f;
            a.pts.resize(static_cast<long>(sg.inliers.size()), 2);
            a.weights.resize(static_cast<long>(sg.inliers.size()));
            for (size_t i = 0; i < sg.inliers.size(); ++i)
            {
                const Eigen::Vector2f& p = pts_robot[static_cast<size_t>(sg.inliers[i])];
                a.pts(static_cast<long>(i), 0) = p.x(); a.pts(static_cast<long>(i), 1) = p.y();
                a.weights(static_cast<long>(i)) = (weights.size() > sg.inliers[i]) ? weights(sg.inliers[i]) : 1.f;
            }
            fr.assoc.push_back(std::move(a));
            candidates.erase(candidates.begin() + ci);
        }

        while (static_cast<int>(candidates.size()) > params.max_candidates)
        {
            auto weakest = std::min_element(candidates.begin(), candidates.end(),
                [](const Candidate& a, const Candidate& b) { return a.npts < b.npts; });
            candidates.erase(weakest);
        }
        fr.candidates = static_cast<int>(candidates.size());
        fr.merged = merge_indistinguishable();
        repair_if_crossing();
        // Zero-evidence micro-edges: a polygon edge shorter than ~2 grid cells whose wall carries
        // essentially no observations asserts nothing — contour debris (a dilated stair-step)
        // kinking the boundary. Its neighbours re-intersect when it goes. Validated with the bias
        // compensation above (paired A/B).
        {
            const Polygon pnow = build_polygon();
            if (pnow.closed)
                for (size_t e = 0; e < pnow.verts.size(); ++e)
                {
                    const float elen = (pnow.verts[(e + 1) % pnow.verts.size()] - pnow.verts[e]).norm();
                    if (elen > 0.2f) continue;
                    const std::uint64_t id = pnow.wall_of_edge[e];
                    const auto* w = find(id);
                    if (w == nullptr or w->points_seen >= 50) continue;
                    if (std::count(order.begin(), order.end(), id) != 1) continue;
                    splice_out(id);
                    break;   // one per frame; the rebuilt polygon decides the next
                }
        }
        // Classes follow the walls: an edge that converged onto a Manhattan direction after being
        // born off it (the tilted-OBB transient) regains its class — and its room factor — here.
        reclassify_all();
        return fr;
    }

    int WallMap::merge_indistinguishable()
    {
        int merged = 0;
        bool again = true;
        while (again)
        {
            again = false;
            for (size_t a = 0; a < walls.size() and not again; ++a)
                for (size_t b = a + 1; b < walls.size() and not again; ++b)
                {
                    auto& A = walls[a]; auto& B = walls[b];
                    const auto ca = cov_of(A.information), cb = cov_of(B.information);
                    if (not ca or not cb) continue;
                    const Eigen::Vector2f r(wrap_pi(A.phi - B.phi), A.d - B.d);
                    const float c2 = chi2_of(r, *ca + *cb + sys_cov(params));
                    if (not (c2 <= params.merge_chi2)) continue;
                    fuse(A.phi, A.d, A.information, B.phi, B.d, B.information);
                    if (B.has_extent)
                    {
                        if (not A.has_extent) { A.s_min = B.s_min; A.s_max = B.s_max; A.has_extent = true; }
                        else { A.s_min = std::min(A.s_min, B.s_min); A.s_max = std::max(A.s_max, B.s_max); }
                    }
                    A.frames_seen += B.frames_seen;
                    A.points_seen += B.points_seen;
                    A.exist_lodds = std::max(A.exist_lodds, B.exist_lodds);
                    A.exist_bins.clear();
                    if (A.k < 0 and B.k >= 0) { A.k = B.k; A.manhattan_var = B.manhattan_var; }
                    const std::uint64_t dead = B.id, kept = A.id;
                    walls.erase(walls.begin() + static_cast<long>(b));
                    for (auto& id : order) if (id == dead) id = kept;
                    // Adjacent duplicates in the cycle collapse (non-adjacent repeats are a notch and legal).
                    for (int i = static_cast<int>(order.size()) - 1; i >= 0 and order.size() > 1; --i)
                        if (order[static_cast<size_t>(i)] == order[static_cast<size_t>((i + 1) % order.size())])
                            order.erase(order.begin() + i);
                    ++merged; again = true;
                }
        }
        if (merged > 0) heal_order();
        return merged;
    }


    void WallMap::FreeGrid::init(const Eigen::Vector2f& centre, float half_span)
    {
        cell = 0.08f;
        nx = ny = static_cast<int>(2.f * half_span / cell) + 1;
        x0 = centre.x() - half_span;
        y0 = centre.y() - half_span;
        lodds.assign(static_cast<size_t>(nx * ny), 0.f);
        hits.assign(static_cast<size_t>(nx * ny), 0);
    }

    void WallMap::FreeGrid::mark(const Eigen::Vector2f& origin, const std::vector<Eigen::Vector2f>& pts_map)
    {
        for (const auto& p : pts_map)
        {
            const Eigen::Vector2f d = p - origin;
            const float L = d.norm();
            if (L < 1e-3f) continue;
            // A hit localises matter to one cell precisely; a traversal of a PARTIALLY occupied cell
            // is weak counter-evidence (the beam may pass through the cell's free part). Hence the
            // asymmetric odds, and free-marching stops short of the endpoint so range noise cannot
            // erase the very cell the return supports.
            const int steps = std::max(0, static_cast<int>((L - 1.5f * cell) / cell));
            for (int k = 0; k < steps; ++k)
            {
                const Eigen::Vector2f q = origin + d * (static_cast<float>(k) * cell / L);
                const int i = static_cast<int>((q.x() - x0) / cell), j = static_cast<int>((q.y() - y0) / cell);
                if (not in(i, j)) continue;
                float& l = lodds[static_cast<size_t>(idx(i, j))];
                // The residual layer's grazing-beam lesson: matter established by RETURNS latches;
                // grazing passes cannot erase it (see FreeGrid::hits). Unestablished cells take full
                // free evidence, which keeps the free region dense enough to trace.
                l = std::max(-4.f, l - ((hits[static_cast<size_t>(idx(i, j))] >= 3 or l > 1.5f) ? 0.02f : 0.4f));
            }
            const int i = static_cast<int>((p.x() - x0) / cell), j = static_cast<int>((p.y() - y0) / cell);
            if (in(i, j))
            {
                lodds[static_cast<size_t>(idx(i, j))] = std::min(4.f, lodds[static_cast<size_t>(idx(i, j))] + 1.0f);
                if (hits[static_cast<size_t>(idx(i, j))] < 65535) ++hits[static_cast<size_t>(idx(i, j))];
            }
        }
    }

    std::vector<Eigen::Vector2f> WallMap::frontiers() const
    {
        std::vector<Eigen::Vector2f> out;
        if (not fgrid.ready()) return out;
        for (int i = 1; i + 1 < fgrid.nx; i += 2)
            for (int j = 1; j + 1 < fgrid.ny; j += 2)
            {
                if (not fgrid.is_free(i, j)) continue;
                const bool touches_unknown = fgrid.is_unknown(i + 1, j) or fgrid.is_unknown(i - 1, j)
                                          or fgrid.is_unknown(i, j + 1) or fgrid.is_unknown(i, j - 1);
                if (not touches_unknown) continue;
                // An unknown EXPLAINED by matter is not a question: the cells behind every wall are
                // for ever unknown, and counting them made the boundary one endless false frontier
                // (the explorer never terminated). Occupied anywhere in the 3×3 ⇒ the unknown
                // neighbour is the wall's shadow, not unexplored space.
                bool walled = false;
                for (int di = -1; di <= 1 and not walled; ++di)
                    for (int dj = -1; dj <= 1 and not walled; ++dj)
                        if (fgrid.is_occupied(i + di, j + dj))
                            walled = true;
                if (not walled) out.push_back(fgrid.at(i, j));
            }
        return out;
    }

    std::vector<Eigen::Vector2f> WallMap::weak_matter() const
    {
        std::vector<Eigen::Vector2f> out;
        if (not fgrid.ready()) return out;
        for (int i = 0; i < fgrid.nx; i += 2)
            for (int j = 0; j < fgrid.ny; j += 2)
            {
                const auto h = fgrid.hits[static_cast<size_t>(fgrid.idx(i, j))];
                const float l = fgrid.lodds[static_cast<size_t>(fgrid.idx(i, j))];
                const bool suspected = (h >= 1 and h <= 2);            // seen once or twice, unconfirmed
                const bool contested = (h >= 3 and l < 0.5f);          // returns vs grazing passes disagree
                if (not suspected and not contested) continue;
                // The THIN-WALL signature, and only that: free space on two OPPOSITE sides of the
                // cell. True for every unconfirmed gap along an interior wall (including the deep
                // tip), false for the fringe of a boundary wall (unknown behind it). The previous
                // filter — "no confirmed matter nearby" — self-limited: once a stretch confirmed,
                // its neighbourhood suppressed the remaining gaps of the SAME wall (31/42 held with
                // periodic holes, and the traced inlet stopped at the first leak).
                const auto free2 = [&](int di, int dj)
                { return fgrid.is_free(i + di, j + dj) or fgrid.is_free(i + 2 * di, j + 2 * dj); };
                const bool thin = (free2(1, 0) and free2(-1, 0)) or (free2(0, 1) and free2(0, -1))
                               or (free2(1, 1) and free2(-1, -1)) or (free2(1, -1) and free2(-1, 1));
                if (thin) out.push_back(fgrid.at(i, j));
            }
        return out;
    }

    bool WallMap::re_derive(const Eigen::Vector2f& robot_map)
    {
        if (not fgrid.ready()) return false;
        const int nx = fgrid.nx, ny = fgrid.ny;
        // Free-for-flood: free AND not touching matter (one cell of dilation). An 8 cm wall on an
        // 8 cm grid ALIASES across two cell columns — hits split, neither column confirms, and a
        // 4-connected flood zigzags through the alternating gaps, truncating the traced inlet at the
        // first leak. A genuinely traversable corridor is wider than one cell; a free cell whose
        // 8-neighbourhood holds matter is wall surface, not passage.
        const auto flood_free = [&](int i, int j) -> bool
        {
            if (not fgrid.is_free(i, j)) return false;
            for (int di = -1; di <= 1; ++di)
                for (int dj = -1; dj <= 1; ++dj)
                    if (fgrid.is_occupied(i + di, j + dj)) return false;
            return true;
        };
        // Connected free component containing the robot (4-connectivity flood fill).
        std::vector<char> comp(static_cast<size_t>(nx * ny), 0);
        {
            int ri = static_cast<int>((robot_map.x() - fgrid.x0) / fgrid.cell);
            int rj = static_cast<int>((robot_map.y() - fgrid.y0) / fgrid.cell);
            // The robot's own cell may sit within the dilation ring of a nearby wall: seed from the
            // nearest flood-free cell in a small window instead of giving up.
            if (not flood_free(ri, rj))
            {
                bool found = false;
                for (int r2 = 1; r2 <= 4 and not found; ++r2)
                    for (int di = -r2; di <= r2 and not found; ++di)
                        for (int dj = -r2; dj <= r2 and not found; ++dj)
                            if (flood_free(ri + di, rj + dj)) { ri += di; rj += dj; found = true; }
                if (not found) return false;
            }
            std::vector<int> stack = {fgrid.idx(ri, rj)};
            comp[static_cast<size_t>(fgrid.idx(ri, rj))] = 1;
            while (not stack.empty())
            {
                const int u = stack.back(); stack.pop_back();
                const int ui = u % nx, uj = u / nx;
                const int di[4] = {1, -1, 0, 0}, dj[4] = {0, 0, 1, -1};
                for (int k = 0; k < 4; ++k)
                {
                    const int vi = ui + di[k], vj = uj + dj[k];
                    if (flood_free(vi, vj) and comp[static_cast<size_t>(fgrid.idx(vi, vj))] == 0)
                    { comp[static_cast<size_t>(fgrid.idx(vi, vj))] = 1; stack.push_back(fgrid.idx(vi, vj)); }
                }
            }
        }
        const auto inc = [&](int i, int j) { return fgrid.in(i, j) and comp[static_cast<size_t>(fgrid.idx(i, j))] != 0; };
        // Moore boundary trace of the component, CCW.
        int si = -1, sj = -1;
        for (int j = 0; j < ny and si < 0; ++j)
            for (int i = 0; i < nx; ++i)
                if (inc(i, j)) { si = i; sj = j; break; }
        if (si < 0) return false;
        std::vector<Eigen::Vector2f> contour;
        {
            const int di8[8] = {1, 1, 0, -1, -1, -1, 0, 1};
            const int dj8[8] = {0, 1, 1, 1, 0, -1, -1, -1};
            int ci = si, cj = sj, dir = 6;   // came from below
            const int max_steps = 4 * nx * ny;
            for (int step = 0; step < max_steps; ++step)
            {
                contour.push_back(fgrid.at(ci, cj));
                int k = (dir + 6) % 8;   // start search from the right-rear (Moore tracing)
                bool moved = false;
                for (int t = 0; t < 8; ++t, k = (k + 1) % 8)
                {
                    const int niq = ci + di8[k], njq = cj + dj8[k];
                    if (inc(niq, njq)) { ci = niq; cj = njq; dir = k; moved = true; break; }
                }
                if (not moved) break;                       // isolated cell
                if (ci == si and cj == sj and contour.size() > 8) break;
            }
        }
        if (contour.size() < 12) return false;
        // Simplify: Douglas-Peucker with the grid's own resolution as tolerance.
        std::vector<Eigen::Vector2f> simp;
        {
            const float eps = 1.6f * fgrid.cell;
            std::vector<char> keep(contour.size(), 0);
            keep.front() = keep.back() = 1;
            std::vector<std::pair<int, int>> st = {{0, static_cast<int>(contour.size()) - 1}};
            while (not st.empty())
            {
                auto [a, b] = st.back(); st.pop_back();
                if (b - a < 2) continue;
                float worst = 0.f; int wi = -1;
                for (int m = a + 1; m < b; ++m)
                {
                    const float dcur = point_to_segment_local(contour[static_cast<size_t>(m)], contour[static_cast<size_t>(a)], contour[static_cast<size_t>(b)]);
                    if (dcur > worst) { worst = dcur; wi = m; }
                }
                if (worst > eps and wi > 0) { keep[static_cast<size_t>(wi)] = 1; st.push_back({a, wi}); st.push_back({wi, b}); }
            }
            for (size_t m = 0; m < contour.size(); ++m) if (keep[m]) simp.push_back(contour[m]);
            if (simp.size() > 1 and (simp.front() - simp.back()).norm() < 1e-3f) simp.pop_back();
        }
        if (simp.size() < 3) return false;
        // Ensure CCW.
        {
            float a2 = 0.f;
            for (size_t m = 0; m < simp.size(); ++m)
            { const auto& p = simp[m]; const auto& q = simp[(m + 1) % simp.size()]; a2 += p.x() * q.y() - q.x() * p.y(); }
            if (a2 < 0.f) std::reverse(simp.begin(), simp.end());
        }
        // Snap each contour edge to the best evidence line (walls first, then candidates; the
        // Manhattan prior enters through classify()). Unmatched long runs create new walls from the
        // contour itself, with weak info — the contour sits half a cell inside the true wall, which
        // the point factors then pull out.
        std::vector<std::uint64_t> new_order;
        std::vector<WallLandmark> created;
        const int M = static_cast<int>(simp.size());
        for (int m = 0; m < M; ++m)
        {
            const Eigen::Vector2f a = simp[static_cast<size_t>(m)], b = simp[static_cast<size_t>((m + 1) % M)];
            const float len = (b - a).norm();
            if (len < 2.f * fgrid.cell) continue;
            const Eigen::Vector2f e = (b - a) / len;
            const Eigen::Vector2f n(-e.y(), e.x());     // interior on the left of a CCW walk
            const float phi_c = std::atan2(n.y(), n.x());
            const float d_c = n.dot(a);
            std::uint64_t chosen = 0; float best = 1e9f;
            const auto consider = [&](float phi_w, float d_w, std::uint64_t id)
            {
                const float dphi = std::abs(wrap_pi(phi_c - phi_w));
                const float dd = std::abs(d_c - d_w);
                if (dphi > 0.25f or dd > 0.35f) return;
                const float sc = dphi * 2.f + dd;
                if (sc < best) { best = sc; chosen = id; }
            };
            for (const auto& w : walls) consider(w.phi, w.d, w.id);
            if (chosen == 0)
                for (size_t ciq = 0; ciq < candidates.size(); ++ciq)
                    if (candidates[ciq].npts >= 50)
                        consider(candidates[ciq].phi, candidates[ciq].d, 1000000000ULL + ciq);
            if (chosen == 0)
            {
                const Eigen::Matrix2f weak = Eigen::Vector2f(30.f, 15.f).asDiagonal();
                // The traced contour runs through DILATED free space, ~1.5 cells inside the true
                // wall; a wall created from it starts with that known bias removed (outward along
                // its inward normal), and the point factors refine from there. Validated 3-seed
                // paired A/B (2026-09-01): with this + micro-prune, IoU median 0.917 vs 0.774.
                WallLandmark w = make_wall(phi_c, d_c - 1.5f * fgrid.cell, weak, 0.5f * params.birth_nats, 0);
                const Eigen::Vector2f tv = w.tangent();
                w.s_min = std::min(tv.dot(a), tv.dot(b)); w.s_max = std::max(tv.dot(a), tv.dot(b));
                w.has_extent = true;
                created.push_back(w);
                chosen = w.id;
            }
            if (new_order.empty() or new_order.back() != chosen) new_order.push_back(chosen);
        }
        // Promote referenced candidates to walls.
        for (auto& id : new_order)
            if (id >= 1000000000ULL)
            {
                const auto& c = candidates[static_cast<size_t>(id - 1000000000ULL)];
                WallLandmark w = make_wall(c.phi, c.d, c.information, params.birth_nats, c.last_ms);
                w.s_min = c.s_min; w.s_max = c.s_max; w.has_extent = c.npts > 0;
                w.frames_seen = c.frames; w.points_seen = c.npts;
                created.push_back(w);
                id = w.id;
            }
        if (new_order.size() > 1 and new_order.front() == new_order.back()) new_order.pop_back();
        if (new_order.size() < 3) return false;

        // Adopt iff the new cycle explains the observed free space BETTER (grid IoU) — the global
        // free-energy comparison, evaluated on the evidence both cycles claim to explain.
        for (const auto& w : created) walls.push_back(w);
        const auto grid_iou = [&](const Polygon& poly) -> float
        {
            if (not poly.closed) return -1.f;
            long inter = 0, uni = 0;
            for (int i = 0; i < nx; i += 2)
                for (int j = 0; j < ny; j += 2)
                {
                    const bool fr_ = inc(i, j);
                    const bool in_ = corner_visibility::point_in_polygon(fgrid.at(i, j), poly.verts);
                    if (fr_ and in_) ++inter;
                    if (fr_ or in_) ++uni;
                }
            return (uni > 0) ? static_cast<float>(inter) / static_cast<float>(uni) : -1.f;
        };
        const float iou_new = grid_iou(build_from(new_order));
        const float iou_old = grid_iou(build_from(order));
        if (iou_new > iou_old + 0.02f)
        {
            order = new_order;
            // Erase promoted candidates (largest indices first) and orphaned walls.
            std::vector<size_t> promoted;
            for (const auto& w : created) (void)w;
            for (size_t ciq = candidates.size(); ciq-- > 0;)
            {
                bool used = false;
                // a promoted candidate's params now live in a wall on the order; drop near-duplicates
                for (const auto& id : order)
                    if (const auto* w = find(id); w != nullptr
                        and std::abs(wrap_pi(w->phi - candidates[ciq].phi)) < 0.05f
                        and std::abs(w->d - candidates[ciq].d) < 0.10f) { used = true; break; }
                if (used) candidates.erase(candidates.begin() + static_cast<long>(ciq));
            }
            for (int wi = static_cast<int>(walls.size()) - 1; wi >= 0; --wi)
                if (std::find(order.begin(), order.end(), walls[static_cast<size_t>(wi)].id) == order.end())
                    walls.erase(walls.begin() + wi);
            heal_order();
            reclassify_all();
            seed_extents_from_polygon();
            return true;
        }
        // Not better: discard the trial walls.
        for (const auto& w : created)
            if (const int wi = index_of(w.id); wi >= 0) walls.erase(walls.begin() + wi);
        return false;
    }

    Polygon WallMap::build_polygon() const { return build_from(order); }

    void WallMap::repair_if_crossing()
    {
        const Polygon poly = build_from(order);
        if (poly.crossing_edges.empty()) { crossing_frames_ = 0; return; }
        if (++crossing_frames_ < 5) return;      // persistence: one bad refinement frame must not amputate
        crossing_frames_ = 0;
        std::uint64_t weakest = 0; int weakest_pts = std::numeric_limits<int>::max();
        for (int e : poly.crossing_edges)
        {
            if (e < 0 or e >= static_cast<int>(poly.wall_of_edge.size())) continue;
            const auto* w = find(poly.wall_of_edge[static_cast<size_t>(e)]);
            if (w != nullptr and w->points_seen < weakest_pts) { weakest_pts = w->points_seen; weakest = w->id; }
        }
        if (weakest != 0) splice_out(weakest);
    }

    Polygon WallMap::build_from(const std::vector<std::uint64_t>& ord) const
    {
        Polygon poly;
        std::ostringstream st;
        const int n = static_cast<int>(ord.size());
        if (n < 3) { poly.status = "fewer than 3 edges in the cycle"; return poly; }

        std::vector<Corner> corners;
        bool closed = true;
        for (int i = 0; i < n; ++i)
        {
            const auto* A = find(ord[static_cast<size_t>((i + n - 1) % n)]);
            const auto* B = find(ord[static_cast<size_t>(i)]);
            if (A == nullptr or B == nullptr) { st << "edge " << i << " references a missing wall; "; closed = false; break; }
            Corner c = intersect_walls(*A, *B, false);
            if (not c.p.allFinite() or (c.p.x() == 0.f and c.p.y() == 0.f and not linefit::intersect(A->line(), B->line())))
            { st << "edges " << (i + n - 1) % n << " and " << i << " are parallel (no corner); "; closed = false; break; }
            corners.push_back(c);
        }
        if (closed)
        {
            for (int i = 0; i < n; ++i)
            {
                poly.verts.push_back(corners[static_cast<size_t>(i)].p);
                poly.wall_of_edge.push_back(ord[static_cast<size_t>(i)]);
            }
            poly.corners = corners;
            for (int i = 0; i < n and closed; ++i)
                for (int j = i + 2; j < n; ++j)
                {
                    if (i == 0 and j == n - 1) continue;
                    if (corner_visibility::segments_cross(poly.verts[static_cast<size_t>(i)], poly.verts[static_cast<size_t>((i + 1) % n)],
                                                          poly.verts[static_cast<size_t>(j)], poly.verts[static_cast<size_t>((j + 1) % n)]))
                    {
                        st << "self-crossing edges " << i << " and " << j << "; ";
                        closed = false;
                        poly.crossing_edges.push_back(i);
                        poly.crossing_edges.push_back(j);
                    }
                }
            float area2 = 0.f;
            for (int i = 0; i < n; ++i)
            {
                const auto& p = poly.verts[static_cast<size_t>(i)];
                const auto& q = poly.verts[static_cast<size_t>((i + 1) % n)];
                area2 += p.x() * q.y() - q.x() * p.y();
            }
            if (area2 <= 0.02f) { st << "polygon is not counter-clockwise or is degenerate; "; closed = false; }
            for (const auto& c : corners) poly.worst_corner_sigma = std::max(poly.worst_corner_sigma, c.sigma);
        }
        poly.closed = closed;
        poly.status = st.str();
        poly.publishable = closed and poly.status.empty()
                           and std::isfinite(poly.worst_corner_sigma)
                           and poly.worst_corner_sigma < params.publish_corner_sigma;
        if (closed and not poly.publishable and poly.status.empty())
        {
            std::ostringstream s2;
            s2 << "closed; worst corner sigma " << poly.worst_corner_sigma << " m >= " << params.publish_corner_sigma;
            poly.status = s2.str();
        }
        return poly;
    }

    void WallMap::reanchor(const Eigen::Vector2f& c, float rot)
    {
        const auto xf = [&](float& phi, float& d, Eigen::Matrix2f& info, float* s_min, float* s_max)
        {
            const Eigen::Vector2f n = linefit::normal_of(phi);
            const Eigen::Vector2f tv = linefit::tangent_of(phi);
            Eigen::Matrix2f J = Eigen::Matrix2f::Identity();
            J(1, 0) = -tv.dot(c);
            if (const auto cov = cov_of(info))
            {
                const Eigen::Matrix2f cov2 = J * (*cov) * J.transpose();
                if (cov2.determinant() > 0.f) info = cov2.inverse();
            }
            d = d - n.dot(c);
            phi = wrap_pi(phi - rot);
            if (s_min) *s_min -= tv.dot(c);
            if (s_max) *s_max -= tv.dot(c);
        };
        for (auto& w : walls)
        {
            const float shift = linefit::tangent_of(w.phi).dot(c);
            xf(w.phi, w.d, w.information, w.has_extent ? &w.s_min : nullptr, w.has_extent ? &w.s_max : nullptr);
            w.bins_s0 -= shift;
        }
        for (auto& cd : candidates) xf(cd.phi, cd.d, cd.information, &cd.s_min, &cd.s_max);
        if (theta0_born) theta0 = wrap_pi(theta0 - rot);
    }
} // namespace rc::wallmap
