#include "wall_map.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <set>
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

    // ─────────────────────────────────────────────────────────────────────────────────────────────
    //  The room's reference direction θ0 — a mixture, searched globally.
    // ─────────────────────────────────────────────────────────────────────────────────────────────
    // A Manhattan wall lies at θ0 + kπ/2. QUADRUPLING the angle folds the four classes onto one
    // circle, so the label k leaves the estimator instead of having to be guessed. What is left is
    // a two-component mixture on that circle, and BOTH components are needed:
    //   • a Manhattan component — von Mises about 4θ0, concentration κ = 1/(16 σ_M²), the same
    //     slack the room↔wall factor allows a real wall;
    //   • a UNIFORM component with the prior mass classify() already carries as manhattan_off_prior.
    // A real room is full of directions that are not walls: chamfers, furniture edges, the oblique
    // chord the segmenter fits across a corner. Without the uniform component they all vote, and on
    // the real apartamento they drag θ0 from 0.87° to 1.73° and cost 0.075 IoU (measured). With it
    // they land where they belong and stop voting.
    //
    // θ0 is then found by SEARCHING the whole quarter turn, not by descending from wherever the
    // estimate happens to sit. That is the difference that matters. The old estimator — the first
    // scan's oriented bounding box, refined by the mean residual INSIDE the current class
    // assignment — is a local method twice over: the mean cannot cross a half-class (the residuals
    // cancel there), and the Manhattan gate meanwhile refuses every wall that disagrees, so the
    // evidence that would correct it never becomes a wall. One of the fifty rooms finished 36.5°
    // out with 99% of itself observed. A global search over an outlier-tolerant likelihood has no
    // such trap: the wrong angle is simply not the maximum.
    void WallMap::update_theta0()
    {
        if (not params.theta0_posterior or not theta0_born) return;
        double tot = 0.0;
        for (const double w : orient_hist_) tot += w;
        if (not (tot > 0.0)) return;

        // Mixture terms, up to a common 1/2π: the von Mises is written relative to its own peak so
        // exp() never overflows, and the uniform is scaled by the same factor.
        const double kappa = 1.0 / (16.0 * static_cast<double>(params.manhattan_sigma_rad)
                                         * static_cast<double>(params.manhattan_sigma_rad));
        const double pi_m = 1.0 - std::clamp(static_cast<double>(params.manhattan_off_prior), 1e-6, 1.0 - 1e-6);
        // I0(κ) for κ ≫ 1: e^κ/√(2πκ). Written as vM/uniform = √(2πκ)·e^(κ(cos u − 1)).
        const double amp = std::sqrt(2.0 * kPi * kappa);
        const double bin = 2.0 * kPi / static_cast<double>(kOrientBins);
        const auto loglik = [&](double u0)      // u0 = 4·θ0 in the histogram's frame
        {
            double L = 0.0;
            for (int b = 0; b < kOrientBins; ++b)
            {
                const double w = orient_hist_[static_cast<size_t>(b)];
                if (w <= 0.0) continue;
                const double du = (static_cast<double>(b) + 0.5) * bin - u0;
                const double ratio = amp * std::exp(kappa * (std::cos(du) - 1.0));
                L += w * std::log(pi_m * ratio + (1.0 - pi_m));
            }
            return L;
        };
        // GLOBAL search over the quarter turn (one full period of the quadrupled angle), then one
        // responsibility-weighted M-step for sub-grid resolution.
        double best_u = 0.0, best_L = -std::numeric_limits<double>::infinity();
        for (int g = 0; g < kOrientBins; ++g)
        {
            const double u0 = (static_cast<double>(g) + 0.5) * bin;
            if (const double L = loglik(u0); L > best_L) { best_L = L; best_u = u0; }
        }
        double cr = 0.0, ci = 0.0, kk = 0.0;
        for (int b = 0; b < kOrientBins; ++b)
        {
            const double w = orient_hist_[static_cast<size_t>(b)];
            if (w <= 0.0) continue;
            const double uc = (static_cast<double>(b) + 0.5) * bin;
            const double ratio = amp * std::exp(kappa * (std::cos(uc - best_u) - 1.0));
            const double r = pi_m * ratio / (pi_m * ratio + (1.0 - pi_m));   // responsibility
            cr += w * r * std::cos(uc); ci += w * r * std::sin(uc); kk += w * r;
        }
        const double mag = std::hypot(cr, ci);
        const double u_hat = (mag > 0.0) ? std::atan2(ci, cr) : best_u;
        // Back to θ0 in the map frame, then lift to the branch nearest the incumbent so the class
        // LABELS stay continuous: branches sit π/2 apart, so a real rotation of up to a quarter
        // class is followed and a mere relabelling never happens.
        const float mode = static_cast<float>((u_hat - orient_off_) / 4.0);
        const float step = kPi * 0.5f;
        theta0 = mode + step * std::round((theta0 - mode) / step);
        // Reported spread: the resultant length of what the Manhattan component actually claimed.
        // It is a DISPERSION, not a standard error — it says how much the directions the room owns
        // disagree with each other, and does not shrink merely because the robot stood still and
        // re-observed them. (It is not the Manhattan gate's width; see theta0_sigma.)
        if (kk > 0.0)
        {
            const double R = std::clamp(mag / kk, 0.0, 1.0);
            const float sig = std::sqrt(static_cast<float>((1.0 - R) / 8.0));
            theta0_information = 1.f / (sig * sig + 1e-12f);
        }
    }

    WallMap::Precisions WallMap::precisions() const
    {
        Precisions pr;
        pr.frame = frames_observed_;
        const float r2d = 180.f / kPi;
        pr.theta0_deg = wrap_pi(theta0) * r2d;
        pr.theta0_disp_deg = (theta0_information > 0.f) ? r2d / std::sqrt(theta0_information) : 0.f;
        pr.theta0_gate_deg = (params.manhattan_gate_rad + 2.f * theta0_sigma()) * r2d;
        pr.n_walls = static_cast<int>(walls.size());
        pr.n_cand  = static_cast<int>(candidates.size());
        pr.n_order = static_cast<int>(order.size());
        std::vector<float> sp, sd, ex, ce;
        for (const auto id : order)
        {
            const auto* w = find(id);
            if (w == nullptr) continue;
            if (const auto c = cov_of(w->information); c)
            {
                if ((*c)(0, 0) > 0.f) sp.push_back(std::sqrt((*c)(0, 0)) * r2d);
                if ((*c)(1, 1) > 0.f) sd.push_back(std::sqrt((*c)(1, 1)));
            }
            ex.push_back(w->exist_lodds);
            if (w->k >= 0)
                ce.push_back(std::abs(wrap_pi(w->phi - theta0 - static_cast<float>(w->k) * kPi * 0.5f)) * r2d);
        }
        const auto med = [](std::vector<float>& v) -> float
        {
            if (v.empty()) return 0.f;
            std::nth_element(v.begin(), v.begin() + static_cast<long>(v.size() / 2), v.end());
            return v[v.size() / 2];
        };
        pr.sigma_phi_deg = med(sp);
        pr.sigma_d_m     = med(sd);
        pr.exist_nats    = med(ex);
        pr.class_err_deg = med(ce);
        std::vector<float> cs;
        for (const auto& c : build_polygon().corners)
        {
            pr.corner_sigma_m = std::max(pr.corner_sigma_m, c.sigma);
            if (std::isfinite(c.sigma)) cs.push_back(c.sigma);
        }
        pr.n_corners = static_cast<int>(cs.size());
        pr.corners_over_bar = static_cast<int>(std::count_if(cs.begin(), cs.end(),
                                  [&](float v){ return v > params.publish_corner_sigma; }));
        pr.corner_sigma_med_m = med(cs);
        return pr;
    }

    float WallMap::theta0_sigma() const
    {
        // Trust in θ0 comes from CONFIRMED walls agreeing with it: until a wall has real support,
        // θ0 is only as good as the OBB prior (a non-convex first scan tilts it 15-20°), and gating
        // structure against an unconfirmed θ0 re-creates the tilted-prior lock-out (measured:
        // births=0 from frame 0, the rectangle culled to 2 walls).
        // MEDIAN, not max: one confirmed off-axis wall must not hold the trust hostage — an
        // oblique 22k-point junk wall admitted while the gate was annealing-wide kept the gate at
        // 41° for ever through its own 15.6° error, re-admitting exactly its own kind (measured).
        // With the posterior in force none of that is needed: its curvature already IS the trust,
        // and it already shrinks when the directions disagree, over ALL of them and not just the
        // ones this very gate let through.
        // NOT the direction posterior's own width. The two answer different questions and only one
        // of them is the gate's. update_theta0 measures how much the OBSERVED directions disagree
        // with each other, which on an axis-aligned room is near zero from the first frame; the
        // gate needs to know how far the WALLS still sit from their classes, which during the
        // tilted-box transient is 15–20° and must stay tolerated until they anneal on. Substituting
        // the first for the second evicted the box before a single real wall could be born
        // (measured: the L room culled to 2 walls, no cycle at all).
        std::vector<float> eps_list;
        for (const auto id : order)
            if (const auto* w = find(id); w != nullptr and w->points_seen >= 500 and w->k >= 0)
                eps_list.push_back(std::abs(wrap_pi(w->phi - theta0 - static_cast<float>(w->k) * kPi * 0.5f)));
        if (eps_list.empty()) return params.rect_prior_sigma_phi_rad;
        std::nth_element(eps_list.begin(), eps_list.begin() + static_cast<long>(eps_list.size() / 2), eps_list.end());
        return std::max(eps_list[eps_list.size() / 2], params.manhattan_sigma_rad);
    }

    WallMap::ClassChoice WallMap::classify(float phi) const
    {
        ClassChoice c;
        if (not theta0_born) { c.k = 0; c.eps = 0.f; c.cost = 0.f; return c; }
        const float s0 = theta0_sigma();
        const float var = params.manhattan_sigma_rad * params.manhattan_sigma_rad + s0 * s0;
        float best = std::numeric_limits<float>::infinity();
        for (int k = 0; k < 4; ++k)
        {
            const float eps = wrap_pi(phi - theta0 - static_cast<float>(k) * kPi * 0.5f);
            const float cost = 0.5f * eps * eps / var;
            if (cost < best) { best = cost; c.k = k; c.eps = eps; c.cost = cost; }
        }
        const float off_cost = -std::log(std::clamp(params.manhattan_off_prior, 1e-6f, 1.f - 1e-6f));
        // Strict mode admits no class-less walls: the nearest class always stands, and structure
        // that cannot live with that is refused UPSTREAM (the splice gate), not exempted here.
        if (best > off_cost and not params.manhattan_strict) { c.k = -1; c.eps = 0.f; c.cost = off_cost; }
        return c;
    }

    void WallMap::reclassify_all()
    {
        update_theta0();
        for (auto& w : walls)
        {
            const auto cls = classify(w.phi);
            w.k = cls.k;
            // Strict mode has no class-less escape; the honesty moves into the VARIANCE — a wall
            // far from its class gets a factor only as strong as its current agreement
            // (max(σ², ε²)), so a tilted prior side ANNEALS onto its class as data rotates it
            // instead of being yanked (the yank collapsed the spur-room map when strictness
            // first landed: 15-20° OBB sides x a tight σ was a huge false residual).
            const float sig2 = params.manhattan_sigma_rad * params.manhattan_sigma_rad;
            w.manhattan_var = (w.k >= 0) ? std::max(sig2, cls.eps * cls.eps) : 0.f;
        }
    }

    void WallMap::enforce_manhattan()
    {
        // Projection EM at the MAP layer (the solver stays soft): four solver-level hard-pin
        // variants were measured worse — yank at birth, wholesale bin invalidation, a bistable
        // anneal gate. The projection is continuous (walls sit within the annealed factor's 1-2
        // degrees of their class), touches no solver state, and tilt is structurally impossible
        // in anything built from the walls afterwards.
        if (not params.manhattan_strict or not theta0_born) return;
        // The M-step below is the mean residual INSIDE the current class assignment; update_theta0
        // supersedes it (and is not conditioned on that assignment), so only one of the two runs.
        if (not params.theta0_posterior)
        {
            float num = 0.f, den = 0.f;
            for (const auto& w : walls)
                if (w.k >= 0 and w.points_seen > 0)
                {
                    const float wgt = static_cast<float>(w.points_seen);
                    num += wgt * wrap_pi(w.phi - theta0 - static_cast<float>(w.k) * kPi * 0.5f);
                    den += wgt;
                }
            if (den > 0.f) theta0 = wrap_pi(theta0 + num / den);
        }
        // A wall is projected onto its class only if it credibly BELONGS to that class — the same
        // two-component question update_theta0 asks of every segment, asked here of the wall. It
        // matters only during the transient, and only because θ0 now gets there first: the four
        // sides of the first scan's bounding box are 38° from their classes by construction, and
        // projecting them the moment θ0 becomes right swings each one 38° about its own centre.
        // The polygon that comes out crosses its own free space and the cull retires it — the spur
        // room lost every wall it had on frame 1 (measured). Once the walls converge the
        // responsibility is 1 to within float precision and the published polygon is exactly
        // Manhattan again, which is what level 2 relies on.
        const double kappa_m = 1.0 / (16.0 * static_cast<double>(params.manhattan_sigma_rad)
                                           * static_cast<double>(params.manhattan_sigma_rad));
        const double pi_m = 1.0 - std::clamp(static_cast<double>(params.manhattan_off_prior),
                                             1e-6, 1.0 - 1e-6);
        const double amp_m = std::sqrt(2.0 * kPi * kappa_m);
        for (auto& w : walls)
        {
            if (w.k < 0) continue;
            const float phi_new = wrap_pi(theta0 + static_cast<float>(w.k) * kPi * 0.5f);
            if (std::abs(wrap_pi(phi_new - w.phi)) < 1e-7f) continue;
            if (params.theta0_posterior)
            {
                const double du = 4.0 * static_cast<double>(wrap_pi(phi_new - w.phi));
                const double ratio = amp_m * std::exp(kappa_m * (std::cos(du) - 1.0));
                if (pi_m * ratio < (1.0 - pi_m)) continue;   // the uniform component explains it better
            }
            // Pivot about the wall's own extent centre: d and extents co-rotate, geometry moves
            // only at second order (the origin-pivot alternative swept far extents sideways).
            const Eigen::Vector2f n_old = w.normal(), t_old = w.tangent();
            const Eigen::Vector2f pc = n_old * w.d
                + t_old * (w.has_extent ? 0.5f * (w.s_min + w.s_max) : 0.f);
            w.phi = phi_new;
            const Eigen::Vector2f n_new = w.normal(), t_new = w.tangent();
            w.d = n_new.dot(pc);
            if (w.has_extent)
            {
                const float sc = t_new.dot(pc), half = 0.5f * (w.s_max - w.s_min);
                w.s_min = sc - half; w.s_max = sc + half;
            }
        }
    }

    Polygon WallMap::manhattan_polygon() const
    {
        WallMap proj(*this);            // project a copy; the live model is never touched
        proj.enforce_manhattan();
        // Projection makes same-class adjacent walls EXACTLY parallel (no corner) and can fold a
        // strongly tilted wall across a neighbour — both are the copy's problem, healed for free.
        proj.heal_order();
        Polygon p = proj.build_polygon();
        // The copy has no history, so the live 5-frame persistence would never fire on it: repair
        // immediately, and keep going while a crossing remains (uncrossing one pair can expose the
        // next). Bounded by the cycle length — every pass removes a wall.
        for (size_t guard = 0; not p.closed and not p.crossing_edges.empty() and guard < proj.order.size(); ++guard)
        {
            if (not proj.repair_if_crossing(/*immediate=*/true)) break;
            p = proj.build_polygon();
        }
        return decorate(p);
    }

    Polygon WallMap::decorate(const Polygon& pub) const
    {
        if (not params.enable_level2 or not pub.closed or pub.verts.size() < 4 or not fgrid.ready()) return pub;
        const float cell = fgrid.cell;
        // Residual cells: latched matter inside the polygon, free space outside it, each at least
        // one and a half cells from every edge (nearer is the wall's own surface and range noise),
        // outside cells within a margin of the polygon's box.
        Eigen::Vector2f lo = pub.verts.front(), hi = lo;
        for (const auto& v : pub.verts) { lo = lo.cwiseMin(v); hi = hi.cwiseMax(v); }
        const float margin = 1.5f;
        const int i0 = std::max(0, static_cast<int>((lo.x() - margin - fgrid.x0) / cell));
        const int i1 = std::min(fgrid.nx - 1, static_cast<int>((hi.x() + margin - fgrid.x0) / cell));
        const int j0 = std::max(0, static_cast<int>((lo.y() - margin - fgrid.y0) / cell));
        const int j1 = std::min(fgrid.ny - 1, static_cast<int>((hi.y() + margin - fgrid.y0) / cell));
        const auto edge_dist = [](const Eigen::Vector2f& p, const std::vector<Eigen::Vector2f>& V, int& emin)
        {
            float dmin = std::numeric_limits<float>::infinity(); emin = -1;
            for (size_t e = 0; e < V.size(); ++e)
            {
                const Eigen::Vector2f& a = V[e]; const Eigen::Vector2f ab = V[(e + 1) % V.size()] - a;
                const float l2 = ab.squaredNorm();
                const float tt = l2 > 1e-9f ? std::clamp((p - a).dot(ab) / l2, 0.f, 1.f) : 0.f;
                const float d = (p - (a + tt * ab)).norm();
                if (d < dmin) { dmin = d; emin = static_cast<int>(e); }
            }
            return dmin;
        };
        // FRESHNESS (the stub discriminator's rule): free space outside the polygon is evidence
        // only if it was seen free AFTER the edge it decorates came into existence — the grid never
        // forgets, so a room change leaves stale free cells where matter now stands (measured: the
        // notched L-room was extruded back into its old notch, 0.94 m off).
        const auto born_of_edge = [&](const std::vector<std::uint64_t>& woe, int e)
        {
            if (woe.empty()) return std::int64_t{0};
            const auto* w = find(woe[static_cast<size_t>(e) % woe.size()]);
            return w == nullptr ? std::int64_t{0} : w->born_ms;
        };
        std::vector<std::int8_t> cls(static_cast<size_t>(fgrid.nx * fgrid.ny), 0);   // 1 matter inside, 2 free outside
        for (int i = i0; i <= i1; ++i)
            for (int j = j0; j <= j1; ++j)
            {
                const Eigen::Vector2f p = fgrid.at(i, j);
                const bool inside = corner_visibility::point_in_polygon(p, pub.verts);
                std::int8_t c = 0;
                if (inside and fgrid.is_occupied(i, j)) c = 1;
                else if (not inside and fgrid.is_free(i, j)) c = 2;
                if (c == 0) continue;
                int e; if (edge_dist(p, pub.verts, e) < params.level2_clear_cells * cell) continue;
                if (c == 2 and fgrid.free_ms[static_cast<size_t>(fgrid.idx(i, j))] < born_of_edge(pub.wall_of_edge, e)) continue;
                cls[static_cast<size_t>(fgrid.idx(i, j))] = c;
            }
        // Connected components (4-connected), largest first.
        std::vector<std::vector<int>> comps;
        {
            std::vector<char> seen(cls.size(), 0);
            for (int i = i0; i <= i1; ++i)
                for (int j = j0; j <= j1; ++j)
                {
                    const int id = fgrid.idx(i, j);
                    if (cls[static_cast<size_t>(id)] == 0 or seen[static_cast<size_t>(id)]) continue;
                    const std::int8_t c = cls[static_cast<size_t>(id)];
                    std::vector<int> comp, stack{id}; seen[static_cast<size_t>(id)] = 1;
                    while (not stack.empty())
                    {
                        const int q = stack.back(); stack.pop_back(); comp.push_back(q);
                        const int qi = q % fgrid.nx, qj = q / fgrid.nx;
                        for (const auto& [di, dj] : std::initializer_list<std::pair<int, int>>{{1, 0}, {-1, 0}, {0, 1}, {0, -1}})
                        {
                            const int ni = qi + di, nj = qj + dj;
                            if (not fgrid.in(ni, nj)) continue;
                            const int nid = fgrid.idx(ni, nj);
                            if (seen[static_cast<size_t>(nid)] or cls[static_cast<size_t>(nid)] != c) continue;
                            seen[static_cast<size_t>(nid)] = 1; stack.push_back(nid);
                        }
                    }
                    if (comp.size() >= 4) comps.push_back(std::move(comp));
                }
            std::sort(comps.begin(), comps.end(), [](const auto& a, const auto& b) { return a.size() > b.size(); });
        }
        // Each component proposes a rectangular step of the edge it is attached to; the step is
        // accepted when the grid evidence it explains exceeds the code length of the edges it adds.
        Polygon out = pub;
        int accepted = 0;
        for (const auto& comp : comps)
        {
            if (accepted >= 8) break;
            const std::int8_t c = cls[static_cast<size_t>(comp.front())];
            // The attached edge of the CURRENT polygon: nearest to the component's centroid.
            Eigen::Vector2f cen = Eigen::Vector2f::Zero();
            for (const int id : comp) cen += fgrid.at(id % fgrid.nx, id / fgrid.nx);
            cen /= static_cast<float>(comp.size());
            int e; edge_dist(cen, out.verts, e);
            if (e < 0) continue;
            const size_t E = out.verts.size();
            const Eigen::Vector2f a = out.verts[static_cast<size_t>(e)], b = out.verts[(static_cast<size_t>(e) + 1) % E];
            const float len = (b - a).norm();
            if (len < 3.f * cell) continue;
            const Eigen::Vector2f t = (b - a) / len, n(-t.y(), t.x());   // n: interior side (CCW)
            // The component's footprint in the edge's frame; it must touch the edge (a decoration
            // of the boundary, not a free-standing blob) and lie on the side its class says.
            float s0 = std::numeric_limits<float>::infinity(), s1 = -s0, h = 0.f, near = s0;
            bool side_ok = true;
            for (const int id : comp)
            {
                const Eigen::Vector2f d = fgrid.at(id % fgrid.nx, id / fgrid.nx) - a;
                const float s = t.dot(d), z = n.dot(d) * (c == 1 ? 1.f : -1.f);
                if (z < 0.f) { side_ok = false; break; }
                s0 = std::min(s0, s); s1 = std::max(s1, s); h = std::max(h, z); near = std::min(near, z);
            }
            if (not side_ok or near > 3.f * cell) continue;
            const float raw_width = s1 - s0;   // the matter's own extent, before any padding
            // ── MID-EDGE STEP or CORNER STEP? A zone that reaches an end of its edge is a feature
            // AT THE CORNER, and the corner is where a mid-edge step cannot go: it would have a
            // zero-length side, and collapsing that vertex leaves a diagonal. Such a zone instead
            // replaces the corner VERTEX with three, which is +2 edges rather than +4 — a corner
            // feature is cheaper to describe because it reuses the corner. Only matter at a CONVEX
            // corner is offered: that is what a column standing in a corner is.
            const auto convex_at = [&](size_t v)
            {
                const size_t E2 = out.verts.size();
                const Eigen::Vector2f tp = (out.verts[v] - out.verts[(v + E2 - 1) % E2]).normalized();
                const Eigen::Vector2f tn = (out.verts[(v + 1) % E2] - out.verts[v]).normalized();
                return tp.x() * tn.y() - tp.y() * tn.x() > 0.f;
            };
            const size_t E0 = out.verts.size();
            // The reach test must be looser than the clearance that produced the footprint: a cell
            // may not come closer than `level2_clear_cells` to the perpendicular wall, so a genuine
            // corner column's cells stop that far short of the corner and a tighter test can never
            // fire (measured: one corner step in a whole run). Twice the clearance. A mid-edge
            // feature that happens to sit near a corner is not mis-read as one for free — the step
            // would claim the free gap between them as exterior, and the grid term charges for it.
            const float reach_tol = 2.f * params.level2_clear_cells * cell;
            const bool corner_hi = c == 1 and s1 > len - reach_tol and convex_at((static_cast<size_t>(e) + 1) % E0);
            const bool corner_lo = c == 1 and s0 < reach_tol and not corner_hi and convex_at(static_cast<size_t>(e));
            const bool corner = corner_hi or corner_lo;
            h += 0.5f * cell;
            if (corner_hi)      { s0 = std::max(0.5f * cell, s0 - 0.5f * cell); s1 = len; }
            else if (corner_lo) { s1 = std::min(len - 0.5f * cell, s1 + 0.5f * cell); s0 = 0.f; }
            else
            {
                s0 = std::max(0.5f * cell, s0 - 0.5f * cell);
                s1 = std::min(len - 0.5f * cell, s1 + 0.5f * cell);
            }
            if (s1 - s0 < params.level2_min_m or h < params.level2_min_m) continue;
            // ── FIT THE THREE DEGREES OF FREEDOM (Params doc). The cell box above is the
            // initialisation; each of the step's three faces now takes the median of the returns
            // that lie on it, which is the maximum-likelihood placement of that face under a
            // symmetric noise model and is robust to the returns of whatever stands beside it.
            if (params.level2_fit and not beams.empty())
            {
                const float sg = (c == 1) ? 1.f : -1.f;   // matter steps into the room, free steps out
                const auto median = [](std::vector<float>& v)
                { std::nth_element(v.begin(), v.begin() + static_cast<long>(v.size() / 2), v.end()); return v[v.size() / 2]; };
                // One pass of the fit. A face may only look at returns that could belong to IT: the
                // band never reaches back to the host wall (which would drag a shallow step's depth
                // toward zero — a 20 cm column read through a 24 cm band is measured against the
                // wall's own returns) nor across to the opposite side, and a beam that grazes a face
                // cannot place it, because its range error projects along the face instead of across
                // it. Called twice: once wide to find the faces, once narrow to measure them.
                const auto refit = [&](float band_cells)
                {
                    const float zlo = std::min(0.f, sg * h), zhi = std::max(0.f, sg * h);
                    const float bf = std::min(band_cells * cell, 0.4f * h);
                    const float bs = std::min(band_cells * cell, 0.4f * (s1 - s0));
                    const float zin = 0.2f * (zhi - zlo);
                    std::vector<float> front, lo_side, hi_side;
                    for (const auto& bm : beams)
                    {
                        const Eigen::Vector2f q = bm.o + bm.d * bm.r - a;
                        const float sq = t.dot(q), zq = n.dot(q);
                        if (std::abs(n.dot(bm.d)) > 0.2f
                            and sq > s0 + bs and sq < s1 - bs and std::abs(zq - sg * h) < bf)
                            front.push_back(zq);
                        if (std::abs(t.dot(bm.d)) > 0.2f and zq > zlo + zin and zq < zhi - zin)
                        {
                            if (std::abs(sq - s0) < bs) lo_side.push_back(sq);
                            if (std::abs(sq - s1) < bs) hi_side.push_back(sq);
                        }
                    }
                    if (front.size()   >= 20) h  = std::abs(median(front));
                    // A corner step has only ONE side face; the other end is the corner itself, and
                    // the returns there belong to the neighbouring wall.
                    const bool got_lo = lo_side.size() >= 20 and not corner_lo;
                    const bool got_hi = hi_side.size() >= 20 and not corner_hi;
                    if (got_lo) s0 = median(lo_side);
                    if (got_hi) s1 = median(hi_side);
                    // ONE-SIDED VISIBILITY. A thin protrusion is normally seen from one side only —
                    // measured on the generated rooms, 13 of 16 fits found returns on a single side.
                    // The unseen face then keeps the padded cell box, and the step claims the air
                    // beside the wall as if it were wall: the disagreement region came out half
                    // matter and half free and the move was refused at −7.6 nats. When exactly one
                    // side is seen and the box is wider than a wall, the unseen face is placed by
                    // the thin-wall prior instead of by the cell grid. Evidence still decides: the
                    // strip is then mostly matter and pays, or it is not and the move is refused.
                    // The discriminator is ASPECT, not width. An 8 cm grid cannot measure a 13 cm
                    // wall: its latched cluster comes out 0.32 m wide, so a width test either
                    // rejects every spur or also accepts the real flat's 0.33 m pillars and shrinks
                    // them (measured, −0.008 IoU). What separates them is that a spur is far deeper
                    // than it is broad. A protrusion more than twice as deep as it is wide is a
                    // piece of WALL, and the far face of a wall is one thickness away.
                    if (c == 1 and not corner and got_lo != got_hi
                        and h > 2.f * raw_width
                        and s1 - s0 > 1.5f * params.stub_thickness)
                    {
                        if (got_hi) s0 = s1 - params.stub_thickness;
                        else        s1 = s0 + params.stub_thickness;
                        if (params.debug_splice)
                            std::printf("[level2] one-sided: unseen face placed by the thin-wall prior, s=[%.3f,%.3f]\n", s0, s1);
                    }
                    if (params.debug_splice)
                        std::printf("[level2] fit(%.1f cells): s=[%.3f,%.3f] h=%.3f (front %zu, sides %zu/%zu)\n",
                                    band_cells, s0, s1, h, front.size(), lo_side.size(), hi_side.size());
                };
                refit(3.f);
                refit(1.f);
                if (not corner_lo) s0 = std::max(0.5f * cell, s0);
                if (not corner_hi) s1 = std::min(len - 0.5f * cell, s1);
                if (s1 - s0 < params.level2_min_m or h < params.level2_min_m) continue;
            }
            // A matter zone steps the boundary INTO the room around it; a free zone steps it OUT.
            const Eigen::Vector2f off = n * h * (c == 1 ? 1.f : -1.f);
            std::vector<Eigen::Vector2f> nv;
            nv.reserve(E + 4);
            if (corner)
            {
                // Replace the corner vertex with three: back along this edge to s0 (or forward to
                // s1), across by the depth, and out to where the next edge resumes.
                const size_t vrep = corner_hi ? (static_cast<size_t>(e) + 1) % E : static_cast<size_t>(e);
                const Eigen::Vector2f b_end = out.verts[(static_cast<size_t>(e) + 1) % E];
                const Eigen::Vector2f P1 = corner_hi ? Eigen::Vector2f(a + t * s0)       : Eigen::Vector2f(a + off);
                const Eigen::Vector2f P2 = corner_hi ? Eigen::Vector2f(a + t * s0 + off) : Eigen::Vector2f(a + off + t * s1);
                const Eigen::Vector2f P3 = corner_hi ? Eigen::Vector2f(b_end + off)      : Eigen::Vector2f(a + t * s1);
                for (size_t k = 0; k < E; ++k)
                {
                    if (k == vrep) { nv.push_back(P1); nv.push_back(P2); nv.push_back(P3); }
                    else nv.push_back(out.verts[k]);
                }
            }
            else
            {
                for (size_t k = 0; k <= static_cast<size_t>(e); ++k) nv.push_back(out.verts[k]);
                for (const Eigen::Vector2f& q : {Eigen::Vector2f(a + t * s0), Eigen::Vector2f(a + t * s0 + off),
                                                 Eigen::Vector2f(a + t * s1 + off), Eigen::Vector2f(a + t * s1)})
                    nv.push_back(q);
                for (size_t k = static_cast<size_t>(e) + 1; k < E; ++k) nv.push_back(out.verts[k]);
            }
            // INVARIANT: level 2 may not make the published polygon less rectilinear than it
            // found it. Every edge of the trial must be parallel or perpendicular to the host edge
            // (which the projection has already put on an axis); anything else is refused.
            {
                bool rectilinear = true;
                for (size_t k = 0; k < nv.size() and rectilinear; ++k)
                {
                    const Eigen::Vector2f e2 = nv[(k + 1) % nv.size()] - nv[k];
                    const float L2 = e2.norm();
                    if (L2 < 1e-4f) { rectilinear = false; break; }
                    rectilinear = std::min(std::abs(t.dot(e2)), std::abs(n.dot(e2))) < 1e-3f * L2;
                }
                if (not rectilinear)
                {
                    if (params.debug_splice)
                        std::printf("[level2] refused: the step would leave a non-rectilinear edge\n");
                    continue;
                }
            }
            Polygon trial = out;
            trial.verts = nv;
            const int added = static_cast<int>(nv.size()) - static_cast<int>(E);
            if (added <= 0) continue;
            const float cost = static_cast<float>(added) * edge_code_nats(true);
            const float gain = jump_delta_nats(out, trial, c == 2 ? born_of_edge(out.wall_of_edge, e) : 0);
            if (params.debug_splice)
                std::printf("[level2] %s %s zone %zu cells on edge %d: s=[%.2f,%.2f] h=%.2f +%d edges gain=%.1f cost=%.1f -> %s\n",
                            corner ? "CORNER" : "mid-edge", c == 1 ? "matter" : "free", comp.size(), e, s0, s1, h, added, gain, cost,
                            gain > cost ? "ACCEPT" : "refuse");
            referee("level2", gain > cost, gain, cost, cost, out.verts, trial.verts);
            if (gain <= cost) continue;
            // Bookkeeping the consumers expect: one wall id and one corner per edge.
            const std::uint64_t host = out.wall_of_edge.empty() ? 0 : out.wall_of_edge[static_cast<size_t>(e) % out.wall_of_edge.size()];
            const Corner hc = out.corners.empty() ? Corner{} : out.corners[static_cast<size_t>(e) % out.corners.size()];
            trial.wall_of_edge.assign(nv.size(), host);
            trial.corners.assign(nv.size(), hc);
            for (size_t k = 0; k < nv.size(); ++k)
            {
                // Original edges keep their ids where the vertex survived unchanged.
                for (size_t m = 0; m < E; ++m)
                    if ((nv[k] - out.verts[m]).norm() < 1e-6f and m < out.wall_of_edge.size())
                    { trial.wall_of_edge[k] = out.wall_of_edge[m]; if (m < out.corners.size()) trial.corners[k] = out.corners[m]; break; }
            }
            out = trial;
            ++accepted;
        }
        if (accepted > 0) out.status += " +L2:" + std::to_string(accepted);
        return out;
    }

    /// THE LINE-EVIDENCE RULE (Params doc, "ONE CURRENCY FOR LINE CLAIMS"): one frame's support
    /// and refutation of one extent bin → one increment in [−1, 1] nats. Shared by wall bins and
    /// candidate bins so their nats mean the same thing and one toll can price them all.
    static float frame_evidence_delta(float sup, float ref, float p_det)
    {
        const float denom = sup + p_det * ref;
        if (denom <= 0.f) return 0.f;
        return std::clamp((sup - p_det * ref) / std::max(denom, 1.f), -1.f, 1.f);
    }

    WallLandmark WallMap::make_wall(float phi, float d, const Eigen::Matrix2f& info, float exist_seed,
                                    std::int64_t ts)
    {
        WallLandmark w;
        w.id = next_id_++;
        w.phi = wrap_pi(phi);
        w.d = d;
        w.information = info;
        w.prior_mu = Eigen::Vector2f(w.phi, d);
        w.exist_lodds = exist_seed;
        w.last_seen_ms = ts;
        w.born_ms = ts;
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
        walls.clear(); candidates.clear(); order.clear(); corner_residue.clear();
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

        // A TIP CAP is not judged by beams on its own line: it closes a wrapped thin wall at the
        // opening where traffic passes, so beams cross its short window and fly on — refuting it
        // 600 times per run while its 12 cm face collects no supports (measured: every flap death
        // was a cap at lodds −6.9, frames=0). A beam beside the tip is CONSISTENT with the wrap;
        // only a beam through the 8 cm body would refute it, which is below beam discrimination.
        // A cap's evidence lives in its FACES: it dies when they do (heal), or when they stop
        // being its twins (it then loses this exemption and every cull applies again).
        std::set<std::uint64_t> cap_ids;
        {
            const int N = static_cast<int>(order.size());
            for (int i = 0; i < N; ++i)
            {
                const auto* wa = find(order[static_cast<size_t>((i + N - 1) % N)]);
                const auto* wb = find(order[static_cast<size_t>((i + 1) % N)]);
                if (wa != nullptr and wb != nullptr
                    and std::abs(wrap_pi(wrap_pi(wa->phi - wb->phi) - kPi)) < 0.35f
                    and std::abs(wa->d + wb->d) < 0.5f)
                    cap_ids.insert(order[static_cast<size_t>(i)]);
            }
        }

        for (auto& w : walls)
        {
            if (not w.has_extent or w.s_max - w.s_min < bw) continue;
            if (cap_ids.count(w.id) > 0) continue;
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
                const float delta = frame_evidence_delta(sup[static_cast<size_t>(bin)], ref[static_cast<size_t>(bin)], params.exist_refute_pdet);
                if (delta == 0.f) continue;
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

        // Deaths: refuted through, and spliced OUT so the cycle heals immediately. Two passes:
        // splice_out() runs heal_order(), which erases ANY wall the healed order no longer
        // references — at any index, not only the one that died — so indexing `walls` while
        // splicing walks off the end. Collect the ids first; a wall heal already removed is a
        // no-op for splice_out. Spliced LAST-BORN FIRST (descending index), the order the old
        // loop used: each heal depends on the neighbourhood the previous splice left, and the
        // bench is bit-exact under this order and shifts (0.942 -> 0.821 on seed 424242) under
        // the ascending one — the sweep is order-dependent, which is its own open item.
        std::vector<std::uint64_t> dead;
        for (const auto& w : walls)
        {
            if (w.exist_lodds > lo_clamp + 1e-3f) continue;
            if (params.debug_splice)
                std::printf("[death-exist] wall %llu phi=%.3f d=%.3f lodds=%.1f frames=%d pts=%d bins=%zu\n",
                            (unsigned long long)w.id, w.phi, w.d, w.exist_lodds,
                            w.frames_seen, w.points_seen, w.exist_bins.size());
            fr.deaths_info.push_back({w.id, w.exist_lodds, w.frames_seen, w.points_seen});
            fr.deaths++;
            dead.push_back(w.id);
        }
        for (auto it = dead.rbegin(); it != dead.rend(); ++it) splice_out(*it);
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
        // STRICT MANHATTAN: an off-axis candidate is not structural at this stage. It is kept in
        // the candidate bank (rejection preserves it) as the input of the chamfer postprocessing.
        if (params.manhattan_strict and theta0_born)
        {
            float best_eps = std::numeric_limits<float>::infinity();
            for (int k = 0; k < 4; ++k)
                best_eps = std::min(best_eps,
                                    std::abs(wrap_pi(c.phi - theta0 - static_cast<float>(k) * kPi * 0.5f)));
            // The gate widens by θ0's own honest uncertainty and anneals as walls confirm.
            if (best_eps > params.manhattan_gate_rad + 2.f * theta0_sigma())
            {
                if (params.debug_splice)
                    std::printf("[mh-gate] refused cand phi=%.3f d=%.3f npts=%d eps=%.1fdeg gate=%.1fdeg\n",
                                c.phi, c.d, c.npts, best_eps * 180.f / kPi,
                                (params.manhattan_gate_rad + 2.f * theta0_sigma()) * 180.f / kPi);
                return -1;
            }
        }
        const int N = static_cast<int>(order.size());
        // Chosen across ALL hosts and cases by SCORE, not first-valid: a valid polygon cannot tell
        // the right corner from a wrong one (the chamfer once replaced a dead ghost at the ghost's
        // position, wherever that was). The score is how much of C's OBSERVED extent lies on the
        // polygon edge C would get — evidence placing the jump, not geometry alone.
        struct Best { float score = -1.f; std::vector<WallLandmark> new_walls; std::vector<std::uint64_t> ord; std::uint64_t host = 0; bool is_stub = false; };
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

        // ── Stub-vs-boundary discriminator, measured on the free grid ────────────────────────────
        // Every boundary splice (notch, corner cut, replacement, direct) makes C a boundary edge:
        // interior in FRONT of C, exterior BEHIND it. The STUB claims the opposite — the room
        // CONTINUES behind the face and wraps around a free-standing interior wall. The two classes
        // differ over exactly one observable region (the strip behind C beyond the thin-wall
        // thickness), so the global ΔF selection the stub was gated on reduces to evidence there:
        // free cells count only if CONNECTED to the robot's own free component (a room change can
        // leave stale free log-odds behind a new wall — the grid never forgets — but that region is
        // sealed off, and unreachable free space supports no wrap); occupied cells refute; unknown
        // cells are silent.
        if (comp_cache_ts_ != ts)
        {
            comp_cache_ = free_component(last_pose_xy_);
            comp_cache_ts_ = ts;
        }
        const float wrap_evidence = [&]() -> float
        {
            if (not fgrid.ready() or comp_cache_.empty() or c.s_max <= c.s_min) return 0.f;
            int net = 0, ntot = 0;
            const float ds = std::max(fgrid.cell, (c.s_max - c.s_min) / 24.f);
            const float t_from = params.stub_thickness + fgrid.cell, t_to = params.stub_thickness + 0.5f;
            for (float s = c.s_min + 0.5f * ds; s < c.s_max; s += ds)
            {
                // No face, no question: a fresh connected-FREE cell ON the candidate line means no
                // wall exists at this stretch at all — a candidate's extent can spill past the real
                // wall's end into open room (measured: the notch candidate spilled 2 m, and the
                // spill's genuinely-free far side mis-classed the whole face as a stub). Such
                // columns are silent either way.
                {
                    const Eigen::Vector2f pl = n_c * c.d + t_c * s;
                    const int li = static_cast<int>((pl.x() - fgrid.x0) / fgrid.cell);
                    const int lj = static_cast<int>((pl.y() - fgrid.y0) / fgrid.cell);
                    if (fgrid.in(li, lj) and fgrid.is_free(li, lj)
                        and comp_cache_[static_cast<size_t>(fgrid.idx(li, lj))] != 0
                        and fgrid.free_ms[static_cast<size_t>(fgrid.idx(li, lj))] >= c.first_ms) continue;
                }
                for (float t = t_from + 0.5f * fgrid.cell; t < t_to; t += fgrid.cell)
                {
                    const Eigen::Vector2f p = n_c * (c.d - t) + t_c * s;
                    const int gi = static_cast<int>((p.x() - fgrid.x0) / fgrid.cell);
                    const int gj = static_cast<int>((p.y() - fgrid.y0) / fgrid.cell);
                    if (not fgrid.in(gi, gj)) continue;
                    ++ntot;
                    // A free cell asserts the wrap only when (a) currently CONNECTED to the robot's
                    // free component and (b) marked free at full weight AFTER this face first
                    // existed: passage while the face stood proves the room goes AROUND it. A room
                    // change leaves stale free log-odds behind a new wall (the grid never forgets),
                    // and a young wall's porous seal lets a bare connectivity flood leak through.
                    if (fgrid.is_occupied(gi, gj)) --net;
                    else if (fgrid.is_free(gi, gj) and comp_cache_[static_cast<size_t>(fgrid.idx(gi, gj))] != 0
                             and fgrid.free_ms[static_cast<size_t>(fgrid.idx(gi, gj))] >= c.first_ms) ++net;
                }
            }
            return (ntot > 0) ? static_cast<float>(net) / static_cast<float>(ntot) : 0.f;
        }();
        const bool wraps_behind = params.enable_stub_jumps and wrap_evidence > params.stub_free_behind_min;
        // If the mirror face ALREADY EXISTS as a mapped wall overlapping C, fabricating another can
        // only duplicate and wreck (measured: a 7 m fabricated twin of the space divider committed
        // beside the real far face). The wrap-completion family — which uses the existing wall as
        // the mirror — is then the only lawful stub-class move.
        bool mirror_exists = false;
        if (wraps_behind)
            for (const auto& w2 : walls)
                if (std::abs(wrap_pi(wrap_pi(w2.phi - c.phi) - kPi)) < 0.35f
                    and std::abs(w2.d + c.d) < 0.35f and w2.has_extent
                    and -w2.s_max < c.s_max and -w2.s_min > c.s_min)
                    { mirror_exists = true; break; }
        const Polygon poly_cur = build_polygon();
        if (params.debug_splice)
            std::printf("[splice] cand phi=%.3f d=%.3f s[%.2f,%.2f] npts=%d wrap=%.2f %s\n",
                        c.phi, c.d, c.s_min, c.s_max, c.npts, wrap_evidence,
                        wraps_behind ? "STUB-CLASS" : "boundary-class");

        for (int i = 0; i < N; ++i)
        {
            WallLandmark* E = find(order[static_cast<size_t>(i)]);
            if (E == nullptr or not E->has_extent) continue;
            // `E` points into `walls`; the trial loop below push_backs (may reallocate) and
            // resizes back (does not restore the old buffer). Nothing may read `E` after the first
            // trial push — the host id is the only thing needed there, so it is copied out here.
            const std::uint64_t host_id = E->id;
            const float dphi = wrap_pi(c.phi - E->phi);
            const bool parallel = std::abs(dphi) < kPi / 4.f;
            // Anti-parallel: a face looking the other way is not an edge of THIS boundary — except
            // as the FAR FACE of the host itself, when the room provably wraps around it (the
            // WRAP-COMPLETION family below). Everything else skips anti-parallel hosts.
            const bool antiparallel = std::abs(dphi) > 3.f * kPi / 4.f;
            if (antiparallel and params.debug_splice)
                std::printf("[splice]   antipar host %llu: dsum=%.2f ext=%d wrap_ok=%d\n",
                            (unsigned long long)E->id, c.d + E->d,
                            static_cast<int>(E->has_extent), static_cast<int>(wraps_behind));
            if (antiparallel and not (wraps_behind and std::abs(c.d + E->d) < 0.35f and E->has_extent))
                continue;

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
                WallLandmark J = make_wall(phi_j, d_j, prior_info, 0.5f * params.birth_nats, ts);
                // The jog physically spans from the host's line to the candidate's line — DECLARE
                // that, or the corner-evidence gate cannot see a jog edge amputate a room: an
                // extent-less trial wall is skipped by the gate, and a jog once extended 4 m to the
                // opposite wall, cutting off everything beyond it (synthetic spur harness).
                const Eigen::Vector2f t_j = linefit::tangent_of(phi_j);
                const Eigen::Vector2f p1 = linefit::normal_of(E->phi) * E->d + tE * s0;
                const Eigen::Vector2f p2 = p1 + n_c * (c.d - n_c.dot(p1));
                J.s_min = std::min(t_j.dot(p1), t_j.dot(p2)) - 0.10f;
                J.s_max = std::max(t_j.dot(p1), t_j.dot(p2)) + 0.10f;
                J.has_extent = true;
                return J;
            };

            // Enumerate order variants (walls listed by VALUE first; committed only on validation).
            struct Variant { std::vector<WallLandmark> new_walls; std::vector<std::uint64_t> ord; bool is_stub = false; };
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
            // The wall is born with the support its candidate observed, on top of the birth seed:
            // a bin at exactly the seed is untested, everything above it was seen (one currency).
            if (not c.bins.empty())
            {
                C.bins_s0 = c.bins_s0;
                C.exist_bins.resize(c.bins.size());
                for (size_t b = 0; b < c.bins.size(); ++b)
                    C.exist_bins[b] = std::min(2.f * params.birth_nats, params.birth_nats + c.bins[b]);
            }

            // RUN REPLACEMENT: C substitutes a consecutive RUN of edges starting at E. The single-host
            // moves cannot express "this 8-metre wall replaces the four junk edges currently standing
            // in for it" — every local variant still crosses the rest of the run and fails validation
            // (measured: a 25k-point real wall homeless through 792 rejections). Evidence-gated: the
            // run's combined support must be LESS than the candidate's own, so a supported stretch of
            // boundary can never be amputated by one loud line.
            if (not antiparallel)
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
            if (not antiparallel and (E->frames_seen == 0 or (at_lo and at_hi)))
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
            if (not parallel and not antiparallel and wraps_behind and not mirror_exists)
            {
                const float e0 = E->normal().dot(c0) - E->d;   // distances from the host line,
                const float e1 = E->normal().dot(c1) - E->d;   // positive = interior side
                const float near_e = std::min(e0, e1), far_e = std::max(e0, e1);
                const Eigen::Vector2f near_p = (e0 <= e1) ? c0 : c1;
                const Eigen::Vector2f far_p  = (e0 <= e1) ? c1 : c0;
                const float s_att = tE.dot(near_p);
                if (params.debug_splice)
                    std::printf("[splice]   stub geom @host %llu: near_e=%.2f far_e=%.2f s_att=%.2f span[%.2f,%.2f]\n",
                                (unsigned long long)E->id, near_e, far_e, s_att, E->s_min, E->s_max);
                if (near_e >= -tol and near_e <= tol
                    and far_e >= 4.f * params.exist_bin_m
                    and s_att > E->s_min + tol and s_att < E->s_max - tol)
                {
                    // The mirror goes WHERE THE MATTER IS: scan depth behind C for the boundary
                    // where free space begins — matter cannot be where free passage was observed.
                    // The fixed thickness prior put the mirror 4 cm into open space beside a real
                    // 8 cm wall, and the false face wrecked the cycle when the real one arrived.
                    // Unobserved depths fall back to the prior thickness.
                    float t0 = -1.f;
                    for (float tt = fgrid.cell; tt <= 0.4f + 1e-3f; tt += fgrid.cell)
                    {
                        int fre = 0, tot = 0;
                        const float ds2 = std::max(fgrid.cell, (c.s_max - c.s_min) / 16.f);
                        for (float s2 = c.s_min + 0.5f * ds2; s2 < c.s_max; s2 += ds2)
                        {
                            const Eigen::Vector2f p2 = n_c * (c.d - tt) + t_c * s2;
                            const int gi2 = static_cast<int>((p2.x() - fgrid.x0) / fgrid.cell);
                            const int gj2 = static_cast<int>((p2.y() - fgrid.y0) / fgrid.cell);
                            if (not fgrid.in(gi2, gj2)) continue;
                            ++tot;
                            if (fgrid.is_free(gi2, gj2)) ++fre;
                        }
                        if (tot > 0 and 2 * fre > tot) { t0 = std::max(tt - fgrid.cell, fgrid.cell); break; }
                    }
                    if (t0 < 0.f) t0 = params.stub_thickness;
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
                            v.is_stub = true;
                            WallLandmark Cw = C, Mw = M, Tw = T;
                            v.new_walls = c_first ? std::vector<WallLandmark>{Cw, Tw, Mw}
                                                  : std::vector<WallLandmark>{Mw, Tw, Cw};
                            v.ord = splice_at({v.new_walls[0].id, v.new_walls[1].id, v.new_walls[2].id}, true);
                            variants.push_back(std::move(v));
                        }
                    }
                }
            }

            // WRAP COMPLETION: C is the OTHER FACE of the host itself — an anti-parallel twin
            // within thin-wall separation, with connected free space observed behind C while it
            // existed. The fabricated-mirror stub above cannot serve here: the mirror ALREADY
            // EXISTS as E, and splicing a copy beside it can only self-cross (measured: ~10k stub
            // offers, ~10 survivors, every thin-wall far face homeless). Complete the wrap instead:
            // a tip cap T between the two faces — [.., E, T, C, ..] or [.., C, T, E, ..] — with the
            // neighbour wall optionally RESUMED after the return face (the spur springs from its
            // line and the boundary continues along it). Tip end, tip normal, insertion side and
            // resume are enumerated; validation + the corner-evidence gate + extent score choose,
            // as for every other jump.
            if (antiparallel and wraps_behind and std::abs(c.d + E->d) < 0.35f and E->has_extent)
            {
                const Eigen::Matrix2f prior_info =
                    Eigen::Vector2f(1.f / (params.rect_prior_sigma_phi_rad * params.rect_prior_sigma_phi_rad),
                                    1.f / (params.rect_prior_sigma_d * params.rect_prior_sigma_d)).asDiagonal();
                for (float s_end : {c.s_min, c.s_max})
                    for (int tsign : {+1, -1})
                    {
                        const float phi_t = wrap_pi(c.phi + static_cast<float>(tsign) * kPi * 0.5f);
                        const Eigen::Vector2f p_tip = n_c * c.d + t_c * s_end;
                        const WallLandmark T = make_wall(phi_t, linefit::normal_of(phi_t).dot(p_tip),
                                                         prior_info, 0.5f * params.birth_nats, ts);
                        for (bool c_after : {true, false})
                            for (bool resume : {false, true})
                            {
                                Variant v;
                                v.is_stub = true;
                                v.new_walls = {T, C};
                                std::vector<std::uint64_t> o;
                                for (int j = 0; j < N; ++j)
                                {
                                    if (j == i and not c_after)
                                    {
                                        if (resume) o.push_back(order[static_cast<size_t>((j + 1) % N)]);
                                        o.push_back(C.id); o.push_back(T.id);
                                    }
                                    o.push_back(order[static_cast<size_t>(j)]);
                                    if (j == i and c_after)
                                    {
                                        o.push_back(T.id); o.push_back(C.id);
                                        if (resume) o.push_back(order[static_cast<size_t>((j + N - 1) % N)]);
                                    }
                                }
                                v.ord = std::move(o);
                                variants.push_back(std::move(v));
                            }
                    }
            }

            // Direct connection, no jog — for ANY angle: a 45° chamfer sits exactly on the
            // parallel/oblique boundary, and when the candidate's endpoint lies ON the host line the
            // jog is zero-length by geometry. Validation + score decide against the jog variants.
            if (not antiparallel and at_lo)
            {
                Variant v; v.new_walls = {C};
                v.ord = splice_at({C.id}, false);
                variants.push_back(std::move(v));
            }
            if (not antiparallel and at_hi)
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
                // The discriminator's verdict is class-wide: connected free space behind C refutes
                // every boundary variant (they all claim exterior there); its absence refutes the stub.
                if (v.is_stub != wraps_behind) continue;
                // Trial: add the new walls, build, validate, score, roll back.
                const size_t walls_before = walls.size();
                for (const auto& w : v.new_walls) walls.push_back(w);
                Polygon poly = build_from(v.ord);
                bool ok = poly.closed;
                const char* why = ok ? "valid" : "not-closed";
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
                if (not ok and why[0] == 'v') why = "interior";
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
                if (not ok and why[0] == 'v') why = "corner";
                if (ok and not comp_cache_.empty())
                {
                    // A NEW wall's edge may not run through space the robot has seen FREE while
                    // this candidate existed: matter cannot be where fresh passage was observed.
                    // The amputating corner-cut passes every geometric gate — a jog spanning 4 m of
                    // toured room to reach a distant host is self-consistent — but the grid watched
                    // beams cross its claimed line (synthetic spur harness). Stale free history
                    // stays silent, so a genuine room change (the notch) still commits over its past.
                    for (size_t e2 = 0; e2 < poly.verts.size() and ok; ++e2)
                    {
                        const std::uint64_t eid = poly.wall_of_edge[e2];
                        bool is_new = false;
                        for (const auto& nwl : v.new_walls) if (nwl.id == eid) { is_new = true; break; }
                        if (not is_new) continue;
                        const Eigen::Vector2f a2 = poly.verts[e2];
                        const Eigen::Vector2f b2 = poly.verts[(e2 + 1) % poly.verts.size()];
                        int nfree = 0, ntot2 = 0;
                        const int nsamp = std::max(2, static_cast<int>((b2 - a2).norm() / fgrid.cell));
                        for (int k2 = 0; k2 <= nsamp; ++k2)
                        {
                            const Eigen::Vector2f p = a2 + (b2 - a2) * (static_cast<float>(k2) / static_cast<float>(nsamp));
                            const int gi = static_cast<int>((p.x() - fgrid.x0) / fgrid.cell);
                            const int gj = static_cast<int>((p.y() - fgrid.y0) / fgrid.cell);
                            if (not fgrid.in(gi, gj)) continue;
                            ++ntot2;
                            if (fgrid.is_free(gi, gj) and comp_cache_[static_cast<size_t>(fgrid.idx(gi, gj))] != 0
                                and fgrid.free_ms[static_cast<size_t>(fgrid.idx(gi, gj))] >= c.first_ms) ++nfree;
                        }
                        if (ntot2 > 0 and 2 * nfree > ntot2) ok = false;
                    }
                    if (not ok and why[0] == 'v') why = "edge-free";
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
                if (params.debug_splice)
                    std::printf("[splice]   %s-variant nw=%zu ord=%zu why=%s score=%.2f host=%llu\n",
                                v.is_stub ? "stub" : "bnd",
                                v.new_walls.size(), v.ord.size(), why, score, (unsigned long long)host_id);
                // Minimal overlap only — the old MAJORITY-of-extent bar froze a whole map: a
                // collinear-MERGED candidate spanning two spaces can never place a majority on one
                // edge (measured: 8261 valid variants refused, births 5). Wrong placements are now
                // refused by the ORDER-QUANTIZED global acceptance below, on evidence, not extent
                // fractions; the local score remains the RANKING among valid variants.
                if (ok and score > 0.5f * params.exist_bin_m and better(score, v.new_walls.size()))
                    best_v = Best{score, v.new_walls, v.ord, host_id, v.is_stub};
            }
        }
        if (best_v.score < 0.f)
        {
            if (params.debug_splice and wraps_behind)
                std::printf("[splice]   no stub variant survived for cand phi=%.3f d=%.3f\n", c.phi, c.d);
            return -1;
        }
        // ── ORDER-QUANTIZED GLOBAL ACCEPTANCE (user design 2026-09-01) ──────────────────────────
        // The winner must PAY for its order jump with evidence it newly explains where the two
        // polygons disagree. Constant cost per +2 edges (geometric prior on polygon order), a
        // parity surcharge for the odd (chamfer) jump — accumulated unmodelled evidence, not a
        // local extent fraction, decides.
        for (const auto& w : best_v.new_walls) walls.push_back(w);
        {
            const Polygon trial = build_from(best_v.ord);
            // CURRENCY: a spur is a NARROW, ELONGATED NOTCH of three walls (face, cap, face), and
            // its disagreement region is the thin wall body itself — which the grid cannot testify
            // about (grazing traffic carves it fresh-free, so the grid delta goes NEGATIVE on
            // exactly the real spurs). Matter claims pay in matter currency: the face candidate's
            // own accumulated point evidence, already in nats. Area claims (every boundary splice)
            // keep paying in grid nats. The discriminator and the geometry gates remain the guards.
            const float dnats = best_v.is_stub ? c.evidence() : jump_delta_nats(poly_cur, trial, c.first_ms);   // ONE CURRENCY: bins, not per-point gain
            const int dorder = static_cast<int>(best_v.ord.size()) - static_cast<int>(order.size());
            // A replacement (dorder == 0) changes no order, yet it still pays one pair: the free
            // version was TRIED and measured — with zero toll a noise-level replacement committed
            // against a transient baseline and amputated half a flat (IoU 0.865 → 0.460, seed
            // 1001). The toll is not only an order prior; it is the noise floor of the evidence
            // comparison itself.
            // Code length of the lines this move names (Params doc): |dorder| new edges, at least
            // the replacement's count; an odd jump's extra edge is off-class and names its angle.
            const int new_edges = std::max(params.replace_code_edges, std::abs(dorder));
            const bool off_class = std::abs(dorder) % 2 == 1;
            float cost = static_cast<float>(new_edges) * edge_code_nats(true)
                       + (off_class ? edge_code_nats(false) - edge_code_nats(true) : 0.f);
            // SURRENDER, the same term and the same currency the re-derivation judge uses: the
            // observed existence support this cycle would erase. Counted once per wall, only for
            // walls the new cycle does not hold anywhere along their LINE, and only ABOVE the birth
            // seed — a wall that has merely been born costs nothing to drop, a wall the sensor has
            // confirmed for a hundred frames costs what it earned.
            float surrender = 0.f;
            if (params.splice_surrender)
                for (auto it = order.begin(); it != order.end(); ++it)
                {
                    if (std::find(order.begin(), it, *it) != it) continue;   // once per wall
                    if (std::find(best_v.ord.begin(), best_v.ord.end(), *it) != best_v.ord.end()) continue;
                    const auto* w = find(*it);
                    if (w == nullptr) continue;
                    bool held = false;
                    for (const auto id2 : best_v.ord)
                        if (const auto* w2 = find(id2); w2 != nullptr
                            and std::abs(wrap_pi(w->phi - w2->phi)) < 0.2f
                            and std::abs(w->d - w2->d) < 0.3f) { held = true; break; }
                    if (held) continue;
                    for (const float b : w->exist_bins) surrender += std::max(0.f, b - params.birth_nats);
                }
            cost += surrender;
            if (params.debug_splice)
                std::printf("[splice]   jump dorder=%+d dnats=%.1f cost=%.1f (code %.1f + surrender %.1f) -> %s\n",
                            dorder, dnats, cost, cost - surrender, surrender,
                            dnats > cost ? "ACCEPT" : "refuse");
            referee("splice", dnats > cost, dnats, cost, cost, poly_cur.verts, trial.verts);
            if (dnats <= cost)
            {
                walls.resize(walls.size() - best_v.new_walls.size());
                return -1;
            }
        }
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

    int WallMap::try_spur_wraps(FrameResult& fr, std::int64_t ts)
    {
        if (order.size() < 3 or not params.enable_stub_jumps or not fgrid.ready()) return -1;
        const Polygon poly = build_polygon();
        if (not poly.closed) return -1;
        if (comp_cache_ts_ != ts) { comp_cache_ = free_component(last_pose_xy_); comp_cache_ts_ = ts; }
        if (comp_cache_.empty()) return -1;
        const int NE = static_cast<int>(poly.verts.size());
        if (NE != static_cast<int>(order.size())) return -1;
        for (int e = 0; e < NE; ++e)
        {
            if (poly.wall_of_edge[static_cast<size_t>(e)] != order[static_cast<size_t>(e)]) continue;
            const WallLandmark* Wp = find(order[static_cast<size_t>(e)]);
            if (Wp == nullptr or not Wp->has_extent or Wp->exist_bins.empty()) continue;
            // A VALUE copy: every (side, resume, tsign) trial below push_backs two walls (may
            // reallocate `walls`) and resizes back, and the next trial reads the host again — a
            // pointer into `walls` would be read after free. The host is never mutated here, and
            // the copy (a few hundred bins at most) is far cheaper than one trial polygon.
            const WallLandmark W = *Wp;
            const Eigen::Vector2f tW = W.tangent(), nW = W.normal();
            const float sa = tW.dot(poly.verts[static_cast<size_t>(e)]);
            const float sb = tW.dot(poly.verts[static_cast<size_t>((e + 1) % NE)]);
            const float e_lo = std::min(sa, sb), e_hi = std::max(sa, sb);
            for (int side : {+1, -1})
            {
                const float corner_s = (side > 0) ? e_hi : e_lo;
                const float reach    = (side > 0) ? W.s_max - corner_s : corner_s - W.s_min;
                if (reach < 2.f * params.exist_bin_m) continue;
                // Contiguous SOLID bins from the corner outward place the tip from evidence — the
                // existence layer already knows where the matter on this line ENDS.
                float tip_s = corner_s;
                float bin_nats = 0.f;   // the overshoot's accumulated existence evidence — the wrap's purse
                int bi = static_cast<int>((corner_s - W.bins_s0) / params.exist_bin_m) + side;
                // SOLID = observed support ABOVE the seed. Bins grown by extent are filled with the
                // wall's seed (birth_nats on a confirmed wall) and an untested bin keeps it exactly,
                // so `>= birth_nats` was true of 6.5 unobserved bins spilling into a doorway — enough
                // to pay the toll with zero returns. Strictly above ⇒ an untested bin ends the run
                // and pays nothing. The purse keeps each tested bin's FULL value on purpose: the
                // 30-nat toll was calibrated in that currency (a confirmed bin holds 2·birth_nats),
                // and netting the seed out (as surrender does at its own scale) left seed 1001's real
                // spur with ≤3.1 nats — refused, IoU 0.967 → 0.932. Re-deriving toll and purse in
                // one currency is review item #10, not this line.
                while (bi >= 0 and bi < static_cast<int>(W.exist_bins.size())
                       and W.exist_bins[static_cast<size_t>(bi)] > params.birth_nats)
                {
                    tip_s = W.bins_s0 + (static_cast<float>(bi) + (side > 0 ? 1.f : 0.f)) * params.exist_bin_m;
                    bin_nats += W.exist_bins[static_cast<size_t>(bi)] - params.birth_nats;
                    bi += side;
                }
                const float overshoot = (side > 0) ? tip_s - corner_s : corner_s - tip_s;
                if (overshoot < 2.f * params.exist_bin_m) continue;
                // The room must provably WRAP: connected free space observed behind the overshoot
                // stretch (beyond thin-wall thickness). Same measurement as the stub discriminator.
                const float s0 = std::min(corner_s, tip_s), s1 = std::max(corner_s, tip_s);
                int net = 0, ntot = 0;
                {
                    const float ds = std::max(fgrid.cell, (s1 - s0) / 24.f);
                    const float t_from = params.stub_thickness + fgrid.cell;
                    const float t_to   = params.stub_thickness + 0.5f;
                    for (float s = s0 + 0.5f * ds; s < s1; s += ds)
                        for (float t = t_from + 0.5f * fgrid.cell; t < t_to; t += fgrid.cell)
                        {
                            const Eigen::Vector2f p = nW * (W.d - t) + tW * s;
                            const int gi = static_cast<int>((p.x() - fgrid.x0) / fgrid.cell);
                            const int gj = static_cast<int>((p.y() - fgrid.y0) / fgrid.cell);
                            if (not fgrid.in(gi, gj)) continue;
                            ++ntot;
                            if (fgrid.is_occupied(gi, gj)) --net;
                            else if (fgrid.is_free(gi, gj)
                                     and comp_cache_[static_cast<size_t>(fgrid.idx(gi, gj))] != 0) ++net;
                        }
                }
                if (ntot == 0 or static_cast<float>(net) / static_cast<float>(ntot) <= params.stub_free_behind_min)
                    continue;
                // If the far face ALREADY EXISTS as a mapped wall overlapping this overshoot, the
                // wrap is not ours to fabricate — a twin mirror beside it can only duplicate and
                // churn (measured: 6 re-wraps of one wall in a run, each dying against the existing
                // face, each death leaving a near-parallel spike). That rearrangement belongs to
                // the global re-derivation.
                {
                    bool twin_exists = false;
                    for (const auto& w2 : walls)
                        if (std::abs(wrap_pi(wrap_pi(w2.phi - W.phi) - kPi)) < 0.35f
                            and std::abs(w2.d + W.d) < 0.35f and w2.has_extent
                            and -w2.s_max < s1 and -w2.s_min > s0)   // overlap in W's tangent coords
                            { twin_exists = true; break; }
                    if (twin_exists) continue;
                }
                if (params.debug_splice)
                    std::printf("[spur] wall %llu edge %d side %+d corner_s=%.2f tip_s=%.2f wrap=%.2f\n",
                                (unsigned long long)W.id, e, side, corner_s, tip_s,
                                static_cast<float>(net) / static_cast<float>(ntot));
                // Trial: mirror M (extent = the overshoot, mirrored) and tip cap T (both signs).
                const Eigen::Matrix2f prior_info =
                    Eigen::Vector2f(1.f / (params.rect_prior_sigma_phi_rad * params.rect_prior_sigma_phi_rad),
                                    1.f / (params.rect_prior_sigma_d * params.rect_prior_sigma_d)).asDiagonal();
                const std::uint64_t w_id = W.id;
                // RESUME first: without it the mirror connects to whatever followed the host, and
                // the wrap commits as an amputation — measured: the mirror ran 4.9 m to the bottom
                // wall and cut off the west half of the flat (IoU 0.503). The spur is a narrow
                // notch INTO the neighbouring boundary: that boundary resumes after the mirror.
                for (const bool resume : {true, false})
                for (int tsign : {+1, -1})
                {
                    const size_t walls_before = walls.size();
                    WallLandmark M = make_wall(wrap_pi(W.phi + kPi), params.stub_thickness - W.d,
                                               prior_info, 0.5f * params.birth_nats, ts);
                    M.s_min = -s1; M.s_max = -s0; M.has_extent = true;   // t_M = −t_W
                    const float phi_t = wrap_pi(W.phi + static_cast<float>(tsign) * kPi * 0.5f);
                    const Eigen::Vector2f p_tip = nW * W.d + tW * tip_s;
                    WallLandmark T = make_wall(phi_t, linefit::normal_of(phi_t).dot(p_tip),
                                               prior_info, 0.5f * params.birth_nats, ts);
                    // T carries a BOUNDED extent from the start — the cap is exactly the thin-wall
                    // thickness — so the corner-evidence gate applies to its corners at trial time
                    // (an extent-less T was skipped, and a near-parallel adjacency once threw its
                    // corner 8 m out of the room).
                    {
                        const float sT = linefit::tangent_of(phi_t).dot(p_tip);
                        T.s_min = sT - params.stub_thickness - 0.10f;
                        T.s_max = sT + params.stub_thickness + 0.10f;
                        T.has_extent = true;
                    }
                    std::vector<std::uint64_t> o;
                    const bool at_end_vertex = std::abs(sb - corner_s) < 1e-4f;
                    for (int j = 0; j < NE; ++j)
                    {
                        if (j == e and not at_end_vertex)
                        {
                            if (resume) o.push_back(order[static_cast<size_t>((j + 1) % NE)]);
                            o.push_back(M.id); o.push_back(T.id);
                        }
                        o.push_back(order[static_cast<size_t>(j)]);
                        if (j == e and at_end_vertex)
                        {
                            o.push_back(T.id); o.push_back(M.id);
                            if (resume) o.push_back(order[static_cast<size_t>((j + NE - 1) % NE)]);
                        }
                    }
                    walls.push_back(T); walls.push_back(M);
                    const Polygon p2 = build_from(o);
                    bool ok = p2.closed;
                    const char* why = ok ? "valid" : "not-closed";
                    for (size_t e2 = 0; e2 < p2.verts.size() and ok; ++e2)
                    {
                        const Eigen::Vector2f dirv = p2.verts[(e2 + 1) % p2.verts.size()] - p2.verts[e2];
                        if (dirv.norm() < 0.05f) { ok = false; break; }
                        const auto* We2 = find(p2.wall_of_edge[e2]);
                        if (We2 == nullptr) { ok = false; break; }
                        const Eigen::Vector2f left(-dirv.y(), dirv.x());
                        if (left.dot(We2->normal()) <= 0.f)
                        {
                            if (params.debug_splice)
                                std::printf("[spur]     interior-fail edge %zu wall %llu dir(%.2f,%.2f) n(%.2f,%.2f)\n",
                                            e2, (unsigned long long)p2.wall_of_edge[e2],
                                            dirv.x(), dirv.y(), We2->normal().x(), We2->normal().y());
                            ok = false; break;
                        }
                    }
                    if (not ok and why[0] == 'v') why = "interior";
                    for (size_t e2 = 0; e2 < p2.verts.size() and ok; ++e2)
                    {
                        const auto* W2 = find(p2.wall_of_edge[e2]);
                        if (W2 == nullptr) { ok = false; break; }
                        if (not W2->has_extent) continue;
                        const Eigen::Vector2f tv2 = W2->tangent();
                        for (const Eigen::Vector2f& vv : {p2.verts[e2], p2.verts[(e2 + 1) % p2.verts.size()]})
                        {
                            const float sc = tv2.dot(vv);
                            if (sc < W2->s_min - 1.0f or sc > W2->s_max + 1.0f) { ok = false; break; }
                        }
                    }
                    if (not ok and why[0] == 'v') why = "corner";
                    if (not ok)
                    {
                        if (params.debug_splice)
                            std::printf("[spur]   tsign %+d REFUSED: %s (%s)\n", tsign, why, p2.status.c_str());
                        walls.resize(walls_before);
                        continue;
                    }
                    // The wrap pays the same order-jump toll as every splice: +4 edges = 2 pairs.
                    // A marginal wrap (carved grid, few matter cells) no longer enters free, drifts
                    // and dies — the flap's entry fee.
                    {
                        // The wrap pays its +4 order toll in EXISTENCE-BIN nats — the very evidence
                        // that triggered it. The grid delta is the WRONG currency here: grazing
                        // aliasing carves a thin wall's own cells fresh-free, so releasing the body
                        // from the interior went NEGATIVE on exactly the seeds whose spur was real
                        // (measured: dnats −22..0 while 8 solid bins held ~72 nats). Bins are beam-
                        // endpoint evidence, immune to cell carving, and stable — no flap re-entry.
                        // ONE ENERGY: the overshoot's observed support (net of the seed) plus the
                        // grid's area term — which excludes the cells on the wrap's own lines, so the
                        // carved thin body no longer votes against a real spur.
                        const float e_grid = jump_delta_nats(poly, p2, 0);
                        const float dnats = bin_nats + e_grid;
                        const float cost = static_cast<float>(params.wrap_code_edges) * edge_code_nats(true);
                        if (params.debug_splice)
                            std::printf("[spur]   jump dnats=%.1f (bins %.1f + grid %.1f) cost=%.1f -> %s\n",
                                        dnats, bin_nats, e_grid, cost, dnats > cost ? "ACCEPT" : "refuse");
                        referee("wrap", dnats > cost, dnats, cost, cost, poly.verts, p2.verts);
                        if (dnats <= cost)
                        {
                            walls.resize(walls_before);
                            continue;
                        }
                    }
                    order = std::move(o);
                    heal_order();
                    reclassify_all();
                    seed_extents_from_polygon();
                    BirthInfo biM;
                    biM.id = M.id; biM.phi = M.phi; biM.d = M.d;
                    biM.nearest_wall = w_id;
                    fr.births_info.push_back(biM);
                    fr.births++;
                    if (params.debug_splice)
                        std::printf("[spur]   WRAPPED wall %llu: tip T=%llu mirror M=%llu (tsign %+d)\n",
                                    (unsigned long long)w_id, (unsigned long long)T.id,
                                    (unsigned long long)M.id, tsign);
                    return index_of(M.id);
                }
            }
        }
        return -1;
    }

    bool WallMap::mirror_backed(const WallLandmark& w) const
    {
        bool twin = false;
        for (const auto& w2 : walls)
            // SIGNED thin separation: a genuine mirror's twin lies BEHIND it (w2.d + w.d = +thickness,
            // by the mirror construction d_M = t0 − d_C); a phantom riding 0.2 m IN FRONT of a
            // boundary wall has the twin on its far-space side (sum negative) — and its "far space"
            // sample would be the toured room itself (measured: it shielded a frames=0 phantom).
            if (w2.points_seen > 0
                and std::abs(wrap_pi(wrap_pi(w2.phi - w.phi) - kPi)) < 0.35f
                and (w2.d + w.d) > 0.f and (w2.d + w.d) < 0.35f) { twin = true; break; }
        if (not twin or not w.has_extent or comp_cache_.empty() or not fgrid.ready()) return false;
        int nfree3 = 0, ntot3 = 0;
        const Eigen::Vector2f nw3 = w.normal(), tw3 = w.tangent();
        const float ds3 = std::max(fgrid.cell, (w.s_max - w.s_min) / 12.f);
        for (float s3 = w.s_min + 0.5f * ds3; s3 < w.s_max; s3 += ds3)
            for (float t3 = fgrid.cell; t3 <= 0.3f + 1e-3f; t3 += fgrid.cell)
            {
                const Eigen::Vector2f p3 = nw3 * (w.d + t3) + tw3 * s3;
                const int gi3 = static_cast<int>((p3.x() - fgrid.x0) / fgrid.cell);
                const int gj3 = static_cast<int>((p3.y() - fgrid.y0) / fgrid.cell);
                if (not fgrid.in(gi3, gj3)) continue;
                ++ntot3;
                if (fgrid.is_free(gi3, gj3)
                    and comp_cache_[static_cast<size_t>(fgrid.idx(gi3, gj3))] != 0) ++nfree3;
            }
        return ntot3 > 0 and 2 * nfree3 > ntot3;
    }

    int WallMap::try_down_jumps(FrameResult& fr, std::int64_t ts)
    {
        if (order.size() <= 4 or not fgrid.ready()) return -1;   // never below the rectangle
        const Polygon cur = build_polygon();
        if (not cur.closed) return -1;
        if (comp_cache_ts_ != ts) { comp_cache_ = free_component(last_pose_xy_); comp_cache_ts_ = ts; }
        const int N = static_cast<int>(order.size());
        // Only entries that can be surrendered cheaply are proposed: a zero-evidence wall, or a
        // DUPLICATE ride of a wall that stays in the cycle elsewhere.
        const auto qualifies = [&](int i) -> bool
        {
            const auto* w = find(order[static_cast<size_t>(i)]);
            if (w == nullptr) return false;
            if (w->points_seen > 0
                and std::count(order.begin(), order.end(), order[static_cast<size_t>(i)]) <= 1)
                return false;
            // Wrap members are RENT-EXEMPT: a cap or mirror carries no points BY NATURE and its
            // justification lives in the faces' existence bins — which a removal of cap or mirror
            // would not surrender (measured: the bench evicted every spur on the carved seeds
            // through exactly this hole, the carved body cells even PAYING for the eviction).
            const auto* wa = find(order[static_cast<size_t>((i + N - 1) % N)]);
            const auto* wb = find(order[static_cast<size_t>((i + 1) % N)]);
            if (w->points_seen == 0                       // a real cap has no points BY NATURE —
                and wa != nullptr and wb != nullptr       // a heavy wall between twins is no cap
                and std::abs(wrap_pi(wrap_pi(wa->phi - wb->phi) - kPi)) < 0.35f
                and std::abs(wa->d + wb->d) < 0.5f) return false;
            if (w->points_seen == 0 and mirror_backed(*w)) return false;
            return true;
        };
        std::vector<std::vector<int>> props;
        for (int i = 0; i < N; ++i)
            if (qualifies(i))
            {
                props.push_back({i});
                for (int j = i + 1; j < N; ++j)
                    if (qualifies(j)) props.push_back({i, j});
            }
        float best_margin = 0.f;
        float best_refund = 0.f;
        std::vector<Eigen::Vector2f> best_t;
        std::vector<std::uint64_t> best_o;
        int best_removed = 0;
        for (const auto& pr : props)
        {
            std::vector<std::uint64_t> o;
            for (int i = 0; i < N; ++i)
                if (std::find(pr.begin(), pr.end(), i) == pr.end())
                    o.push_back(order[static_cast<size_t>(i)]);
            // Collapse immediate duplicates the removal created.
            for (size_t k = o.size(); k-- > 1;)
                if (o[k] == o[k - 1]) o.erase(o.begin() + static_cast<long>(k));
            while (o.size() > 1 and o.front() == o.back()) o.pop_back();
            if (o.size() < 4) continue;
            const Polygon t = build_from(o);
            if (not t.closed) continue;
            bool ok = true;
            for (size_t e2 = 0; e2 < t.verts.size() and ok; ++e2)
            {
                const Eigen::Vector2f dirv = t.verts[(e2 + 1) % t.verts.size()] - t.verts[e2];
                if (dirv.norm() < 0.05f) { ok = false; break; }
                const auto* We2 = find(t.wall_of_edge[e2]);
                if (We2 == nullptr) { ok = false; break; }
                const Eigen::Vector2f left(-dirv.y(), dirv.x());
                if (left.dot(We2->normal()) <= 0.f) { ok = false; break; }
            }
            if (not ok) continue;
            // Evidence surrendered: the grid delta of the new claims, minus the existence-bin nats
            // of walls that leave the cycle entirely (bins are beam-endpoint evidence FOR them).
            float dnats = jump_delta_nats(cur, t, 0);
            for (const auto& w : walls)
                if (std::find(order.begin(), order.end(), w.id) != order.end()
                    and std::find(o.begin(), o.end(), w.id) == o.end())
                    for (const float b : w.exist_bins) dnats -= std::max(0.f, b - params.birth_nats);
            const int dorder = N - static_cast<int>(o.size());
            const float refund = params.order_keep_fraction * static_cast<float>(dorder) * edge_code_nats(true);
            const float margin = dnats + refund;
            if (margin > best_margin + 1e-3f)
            { best_margin = margin; best_o = o; best_removed = dorder; best_refund = refund; best_t = t.verts; }
        }
        if (best_o.empty()) return -1;
        referee("down", true, best_margin - best_refund, -best_refund, -best_refund, cur.verts, best_t);
        if (params.debug_splice)
            std::printf("[down] removing %d order entries, margin=%.1f nats\n", best_removed, best_margin);
        order = std::move(best_o);
        for (int wi = static_cast<int>(walls.size()) - 1; wi >= 0; --wi)
            if (std::find(order.begin(), order.end(), walls[static_cast<size_t>(wi)].id) == order.end()
                and walls[static_cast<size_t>(wi)].points_seen == 0)
                walls.erase(walls.begin() + wi);
        heal_order();
        reclassify_all();
        seed_extents_from_polygon();
        fr.deaths++;
        return best_removed;
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
            last_pose_xy_ = pose.head<2>();
            if (not fgrid.ready()) fgrid.init(pose.head<2>(), params.sensor_range);
            const Eigen::Matrix2f Rm = rot2(pose.z());
            std::vector<Eigen::Vector2f> pm;
            pm.reserve(pts_robot.size());
            for (const auto& pr : pts_robot)
                pm.push_back(Rm * pr + pose.head<2>());
            fgrid.mark(pose.head<2>(), pm, timestamp_ms);
            ++frames_observed_;
            if (params.forward_referee)
            {
                for (const auto& pr : pts_robot)
                {
                    const float r = pr.norm();
                    if (r < 1e-3f) continue;
                    const Beam b{pose.head<2>(), Rm * (pr / r), r};
                    if (beams.size() < params.beam_store_max) beams.push_back(b);
                    else { beams[beam_next_] = b; beam_next_ = (beam_next_ + 1) % beams.size(); }
                }
            }
        }
        // A trial adoption is measured in FRAMES of fresh evidence, so the window ticks here and
        // nowhere else — one tick per scan actually taken in, not per re-derivation attempted.
        if (trial_.open and --trial_.frames_left <= 0) resolve_trial();
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
            // DIRECTION EVIDENCE (update_theta0): every segment votes on the room's reference
            // direction, whatever the Manhattan gate later does with it. This is the only place
            // θ0 hears from anything, and segments are the only thing here that is measured.
            if (sm[s].ok and sm[s].Sigma(0, 0) > 0.f)
            {
                const double w = 1.0 / (16.0 * static_cast<double>(sm[s].Sigma(0, 0)));
                if (std::isfinite(w) and w > 0.0)
                {
                    double u = 4.0 * static_cast<double>(sm[s].phi) + orient_off_;
                    u -= 2.0 * kPi * std::floor(u / (2.0 * kPi));
                    const int b = std::clamp(static_cast<int>(u * kOrientBins / (2.0 * kPi)),
                                             0, kOrientBins - 1);
                    orient_hist_[static_cast<size_t>(b)] += w;
                }
            }
        }

        // ── CORNER EXPLANATION: the model predicts its own segmenter artifact ─────────────────────
        // Sequential RANSAC at a corner of the CURRENT polygon can fit one oblique chord through the
        // tails of the two meeting walls. Such a segment is not evidence of an unmodelled wall — it
        // is the predictable image of two MODELLED walls under the segmenter. It must neither
        // associate (it would drag both walls) nor feed candidates (it would accumulate the homeless
        // oblique mass whose housing was the inclined-wall failure). Signature: the modelled corner
        // lies within its span, its direction is OBLIQUE to both walls (a mixture direction — a
        // segment parallel to either wall is real evidence, e.g. a notch face), and the pair of
        // walls explains one endpoint each while neither explains both. A real CHAMFER shares this
        // signature and 3-30 points cannot separate them (measured: three likelihood-ratio variants
        // inert or leaky) — so every silenced chord is preserved in the CORNER RESIDUE ledger, where
        // a chamfer accumulates persistently for the postprocessing stage; artifacts stay weak.
        std::vector<char> corner_explained(static_cast<size_t>(S), 0);
        if (not order.empty())
        {
            const Polygon pcx = build_polygon();
            if (pcx.closed)
                for (int s = 0; s < S; ++s)
                {
                    if (not sm[s].ok) continue;
                    const auto& sgx = seg.segments[static_cast<size_t>(s)];
                    const Eigen::Vector2f e0 = R * sgx.p0 + t;
                    const Eigen::Vector2f e1 = R * sgx.p1 + t;
                    const float len = (e1 - e0).norm();
                    if (len < 1e-3f) continue;
                    const Eigen::Vector2f sd = (e1 - e0) / len;
                    const float band = 3.f * (params.map_sigma_d
                                              + std::sqrt(std::max(0.f, sm[s].Sigma(1, 1))));
                    const int NC = static_cast<int>(pcx.verts.size());
                    for (int c2 = 0; c2 < NC and corner_explained[static_cast<size_t>(s)] == 0; ++c2)
                    {
                        const Eigen::Vector2f& cv = pcx.verts[static_cast<size_t>(c2)];
                        const float sc = sd.dot(cv - e0);
                        if (sc < -0.2f or sc > len + 0.2f) continue;      // corner within the span
                        const auto* wA = find(pcx.wall_of_edge[static_cast<size_t>((c2 + NC - 1) % NC)]);
                        const auto* wB = find(pcx.wall_of_edge[static_cast<size_t>(c2)]);
                        if (wA == nullptr or wB == nullptr or wA->id == wB->id) continue;
                        // Oblique to BOTH walls: a mixture's direction lies strictly between theirs.
                        const float sgate = std::sin(params.manhattan_gate_rad + 2.f * theta0_sigma());
                        if (std::abs(sd.dot(wA->normal())) <= sgate
                            or std::abs(sd.dot(wB->normal())) <= sgate) continue;
                        const float dA0 = std::abs(wA->normal().dot(e0) - wA->d);
                        const float dB0 = std::abs(wB->normal().dot(e0) - wB->d);
                        const float dA1 = std::abs(wA->normal().dot(e1) - wA->d);
                        const float dB1 = std::abs(wB->normal().dot(e1) - wB->d);
                        // Neither wall alone explains it (else it associates normally)…
                        if ((dA0 < band and dA1 < band) or (dB0 < band and dB1 < band)) continue;
                        // …but the pair does, one end each.
                        if ((dA0 < band and dB1 < band) or (dB0 < band and dA1 < band))
                        {
                            corner_explained[static_cast<size_t>(s)] = 1;
                            if (params.debug_splice)
                                std::printf("[corner-explained] seg %d npts=%d phi=%.3f at corner %d\n",
                                            s, sgx.npts, sm[s].phi, c2);
                            bool fused_cr = false;
                            for (auto& cr : corner_residue)
                                if (std::abs(wrap_pi(sm[s].phi - cr.phi)) < 0.2f
                                    and std::abs(sm[s].d - cr.d) < 0.3f)   // ledger bucketing, not inference
                                {
                                    const float w = static_cast<float>(cr.npts)
                                                    / static_cast<float>(cr.npts + sgx.npts);
                                    cr.phi = wrap_pi(cr.phi + (1.f - w) * wrap_pi(sm[s].phi - cr.phi));
                                    cr.d   = w * cr.d + (1.f - w) * sm[s].d;
                                    cr.npts += sgx.npts; cr.frames += 1; fused_cr = true; break;
                                }
                            if (not fused_cr and corner_residue.size() < 64)
                            {
                                Candidate cr; cr.phi = sm[s].phi; cr.d = sm[s].d;
                                cr.npts = sgx.npts; cr.frames = 1;
                                corner_residue.push_back(cr);
                            }
                        }
                    }
                }
        }

        // ── Association: Mahalanobis under the innovation covariance, MANY-TO-ONE, PDA ───────────
        std::vector<std::vector<float>> chi2(static_cast<size_t>(S), std::vector<float>(static_cast<size_t>(W), std::numeric_limits<float>::infinity()));
        for (int s = 0; s < S; ++s)
        {
            if (not sm[s].ok) continue;
            if (corner_explained[static_cast<size_t>(s)] == 1) continue;   // predicted, not evidence
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
            if (fr.seg_to_wall[static_cast<size_t>(s)] >= 0 or not sm[s].ok
                or corner_explained[static_cast<size_t>(s)] == 1) continue;
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
            std::vector<float> scs;
            scs.reserve(sg.inliers.size());
            for (int idx : sg.inliers)
            {
                const Eigen::Vector2f q = R * pts_robot[static_cast<size_t>(idx)] + t;
                const float r = n.dot(q) - c.d;
                c.gain += wallseg::point_gain_nats(r, sg.sigma2(), seg.clutter_area, extent);
                const float sc = tv.dot(q);
                scs.push_back(sc);
                if (c.npts == 0) { c.s_min = c.s_max = sc; }
                else { c.s_min = std::min(c.s_min, sc); c.s_max = std::max(c.s_max, sc); }
                c.npts++;
            }
            // Line support in the ONE CURRENCY: each extent bin this frame's points touch gains at
            // most one nat, saturating at birth_nats — the same rule as a wall's existence bins.
            // c.gain (per point) decides BIRTH; this decides what a splice may SPEND.
            if (not scs.empty())
            {
                const float bw = params.exist_bin_m;
                if (c.bins.empty()) { c.bins_s0 = std::floor(c.s_min / bw) * bw; c.bins.assign(1, 0.f); }
                while (c.bins_s0 > c.s_min) { c.bins.insert(c.bins.begin(), 0.f); c.bins_s0 -= bw; }
                while (c.bins_s0 + bw * static_cast<float>(c.bins.size()) < c.s_max - 0.5f * bw) c.bins.push_back(0.f);
                const int nb = static_cast<int>(c.bins.size());
                std::vector<char> touched(static_cast<size_t>(nb), 0);
                for (const float sc : scs)
                    touched[static_cast<size_t>(std::clamp(static_cast<int>(std::floor((sc - c.bins_s0) / bw)), 0, nb - 1))] = 1;
                for (int b = 0; b < nb; ++b)
                    if (touched[static_cast<size_t>(b)])
                        c.bins[static_cast<size_t>(b)] = std::min(params.birth_nats, c.bins[static_cast<size_t>(b)] + 1.f);
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
        try_spur_wraps(fr, timestamp_ms);
        try_down_jumps(fr, timestamp_ms);
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
                    // A TIP CAP is not debris: a short edge whose two neighbours are anti-parallel
                    // twins at thin-wall separation is the structural cap of a spur wrap — the
                    // model's only assertion of where the wall ENDS. It is short BY NATURE and
                    // rarely collects returns (beams graze the tip edge-on); deleting it un-wraps
                    // the spur (measured: the wrap died 14 frames after commit, eaten here).
                    const auto* wa = find(pnow.wall_of_edge[(e + pnow.verts.size() - 1) % pnow.verts.size()]);
                    const auto* wb = find(pnow.wall_of_edge[(e + 1) % pnow.verts.size()]);
                    if (wa != nullptr and wb != nullptr
                        and std::abs(wrap_pi(wrap_pi(wa->phi - wb->phi) - kPi)) < 0.35f
                        and std::abs(wa->d + wb->d) < 0.5f) continue;
                    splice_out(id);
                    break;   // one per frame; the rebuilt polygon decides the next
                }
        }
        // A ZERO-EVIDENCE wall whose edge runs through observed free space is refuted structure:
        // it was never observed (points_seen == 0 ⇒ it has no existence bins for the step-back to
        // kill — immortal by omission), yet beams provably cross its claimed line. The trial-time
        // edge-free gate refuses NEW walls on this evidence; this is the same rule applied to
        // STANDING structure (measured: an oblique frames=0 leftover cut a room corner diagonally
        // for ever). One per frame; heal_order re-closes the cycle.
        if (fgrid.ready())
        {
            if (comp_cache_ts_ != timestamp_ms)
            { comp_cache_ = free_component(last_pose_xy_); comp_cache_ts_ = timestamp_ms; }
            const Polygon pz = build_polygon();
            if (pz.closed and not comp_cache_.empty())
                for (size_t e = 0; e < pz.verts.size(); ++e)
                {
                    const auto* w = find(pz.wall_of_edge[e]);
                    if (w == nullptr or w->points_seen > 0) continue;
                    const auto skip = [&](const char* why2)
                    {
                        if (params.debug_splice)
                            std::printf("[cull-skip] wall %llu: %s\n",
                                        (unsigned long long)w->id, why2);
                    };
                    if (std::count(order.begin(), order.end(), pz.wall_of_edge[e]) != 1) { skip("dup-in-order"); continue; }
                    // TIP-CAP exemption (as in the micro-edge cleanup): a cap between anti-parallel
                    // twins has 0 points BY NATURE — culling it un-wrapped the spur every frame
                    // (measured: 986 wrap/death flaps in one bench run).
                    {
                        const auto* wa = find(pz.wall_of_edge[(e + pz.verts.size() - 1) % pz.verts.size()]);
                        const auto* wb = find(pz.wall_of_edge[(e + 1) % pz.verts.size()]);
                        if (wa != nullptr and wb != nullptr
                            and std::abs(wrap_pi(wrap_pi(wa->phi - wb->phi) - kPi)) < 0.35f
                            and std::abs(wa->d + wb->d) < 0.5f) { skip("tip-cap"); continue; }
                    }
                    // MIRROR exemption: a points-0 wall whose ANTI-PARALLEL TWIN carries real
                    // observations, with the far space OBSERVED connected-free on its outward
                    // side, is the far face of a wrapped thin wall. Its line often READS free —
                    // grazing traffic carves an 8 cm wall on an 8 cm grid — but the observed near
                    // face plus proven free space behind it assert a boundary there; culling it
                    // re-opened the wrap each frame on exactly the carved seeds. The far-free
                    // verification keeps a phantom twin behind a boundary wall cullable.
                    if (mirror_backed(*w)) { skip("mirror-verified"); continue; }
                    const Eigen::Vector2f a2 = pz.verts[e];
                    const Eigen::Vector2f b2 = pz.verts[(e + 1) % pz.verts.size()];
                    int nfree = 0, ntot = 0;
                    const int nsamp = std::max(2, static_cast<int>((b2 - a2).norm() / fgrid.cell));
                    for (int k2 = 0; k2 <= nsamp; ++k2)
                    {
                        const Eigen::Vector2f p = a2 + (b2 - a2) * (static_cast<float>(k2) / static_cast<float>(nsamp));
                        const int gi = static_cast<int>((p.x() - fgrid.x0) / fgrid.cell);
                        const int gj = static_cast<int>((p.y() - fgrid.y0) / fgrid.cell);
                        if (not fgrid.in(gi, gj)) continue;
                        ++ntot;
                        if (fgrid.is_free(gi, gj)
                            and comp_cache_[static_cast<size_t>(fgrid.idx(gi, gj))] != 0) ++nfree;
                    }
                    if (ntot > 0 and 2 * nfree > ntot)
                    {
                        if (params.debug_splice)
                            std::printf("[cull] zero-evidence wall %llu phi=%.3f d=%.3f edge through free (%d/%d)\n",
                                        (unsigned long long)w->id, w->phi, w->d, nfree, ntot);
                        splice_out(pz.wall_of_edge[e]);
                        fr.deaths++;
                        break;
                    }
                    skip("edge-not-free");
                }
        }
        // STRICT MANHATTAN eviction: an order wall whose class error exceeds the annealed gate is
        // not structural at this stage, however many points it carries — an oblique junk line
        // admitted during the wide-gate transient must leave once θ0 is trusted, or it keeps the
        // polygon (and, through max-trust, once kept the gate itself) wrong. One per frame.
        if (params.manhattan_strict and theta0_born and order.size() > 4)
        {
            const float gate_eff = params.manhattan_gate_rad + 2.f * theta0_sigma();
            for (const auto id : order)
            {
                const auto* w = find(id);
                if (w == nullptr) continue;
                if (std::count(order.begin(), order.end(), id) != 1) continue;
                float best_eps = std::numeric_limits<float>::infinity();
                for (int k = 0; k < 4; ++k)
                    best_eps = std::min(best_eps,
                                        std::abs(wrap_pi(w->phi - theta0 - static_cast<float>(k) * kPi * 0.5f)));
                if (best_eps > gate_eff)
                {
                    if (params.debug_splice)
                        std::printf("[mh-evict] wall %llu phi=%.3f pts=%d eps=%.1fdeg > gate %.1fdeg\n",
                                    (unsigned long long)w->id, w->phi, w->points_seen,
                                    best_eps * 180.f / kPi, gate_eff * 180.f / kPi);
                    splice_out(id);
                    fr.deaths++;
                    break;
                }
            }
        }
        // Classes follow the walls: an edge that converged onto a Manhattan direction after being
        // born off it (the tilted-OBB transient) regains its class — and its room factor — here.
        reclassify_all();
        // GHOST SWEEP — the one order invariant that phi drift can break with no discrete event
        // following (measured: a 45 m spike corner from a 49-pt ghost 1 m behind the true wall,
        // whose absence evidence the boundary itself occludes for ever). Scan ONLY for that
        // signature — adjacent, same-facing, near-parallel, claiming the same span — and retire
        // the weaker from the cycle. Full heal_order() stays event-driven: running ALL its rules
        // per frame collapses the wrap/jog machinery's staged insertions (measured: 0.956→0.820).
        for (size_t oi = 0; oi < order.size() and order.size() >= 3; ++oi)
        {
            const auto* A = find(order[oi]);
            const auto* B = find(order[(oi + 1) % order.size()]);
            if (A == nullptr or B == nullptr or A->id == B->id) continue;
            if (not A->has_extent or not B->has_extent) continue;
            const float sin_ab = std::abs(A->normal().x() * B->normal().y()
                                          - A->normal().y() * B->normal().x());
            if (sin_ab >= 5e-2f or A->normal().dot(B->normal()) <= 0.f) continue;
            const float ov = std::min(A->s_max, B->s_max) - std::max(A->s_min, B->s_min);
            const float shorter = std::min(A->s_max - A->s_min, B->s_max - B->s_min);
            if (shorter <= 0.f or ov <= 0.5f * shorter) continue;
            const std::uint64_t victim = (A->points_seen < B->points_seen) ? A->id : B->id;
            if (params.debug_splice)
                std::printf("[ghost-sweep] wall %llu retired from the cycle (same-facing same-span with %llu)\n",
                            static_cast<unsigned long long>(victim),
                            static_cast<unsigned long long>((victim == A->id) ? B->id : A->id));
            order.erase(std::remove(order.begin(), order.end(), victim), order.end());
            heal_order();
            break;   // one repair per frame — bounded, gentle
        }
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
                    if (A.prior_info.trace() > 0.f or B.prior_info.trace() > 0.f)
                    {
                        float mphi = A.prior_mu.x(), md = A.prior_mu.y();
                        Eigen::Matrix2f lb = B.prior_info;
                        fuse(mphi, md, A.prior_info, B.prior_mu.x(), B.prior_mu.y(), lb);   // Λ_A += Λ_B, μ fused
                        A.prior_mu = Eigen::Vector2f(mphi, md);
                    }
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
        free_ms.assign(static_cast<size_t>(nx * ny), -1);
    }

    void WallMap::FreeGrid::mark(const Eigen::Vector2f& origin, const std::vector<Eigen::Vector2f>& pts_map,
                                 std::int64_t ts_ms)
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
                if (hits[static_cast<size_t>(idx(i, j))] >= 3 or l > 1.5f)
                    l = std::max(-4.f, l - 0.02f);
                else
                {
                    l = std::max(-4.f, l - 0.4f);
                    free_ms[static_cast<size_t>(idx(i, j))] = ts_ms;   // full-weight free passage NOW
                }
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

    std::vector<char> WallMap::free_component(const Eigen::Vector2f& seed_map) const
    {
        if (not fgrid.ready()) return {};
        const int nx = fgrid.nx, ny = fgrid.ny;
        // Free-for-flood: free AND not touching matter (one cell of dilation). An 8 cm wall on an
        // 8 cm grid ALIASES across two cell columns — hits split, neither column confirms, and a
        // 4-connected flood zigzags through the alternating gaps, truncating the traced inlet at the
        // first leak. A genuinely traversable corridor is wider than one cell; a free cell whose
        // 8-neighbourhood holds matter is wall surface, not passage.
        //
        // NOTE (bin-informed blocking, TRIED AND REVERTED 2026-09-01): letting walls' solid
        // existence bins block the flood was measured twice — masking at >= birth_nats let every
        // young wall's SEEDED bins wall off space wholesale (min IoU 0.931 -> 0.860); masking at
        // 1.5x birth flapped inside the adoption feedback loop (bins mature -> block -> contour
        // changes -> re-adoption resets them; 561 deaths in one run) and regressed the short spur
        // on every seed. A viable version needs a LATCH that survives bin remapping, plus the
        // downstream adoption trace — see the memory's short-spur campaign note.
        const auto flood_free = [&](int i, int j) -> bool
        {
            if (not fgrid.is_free(i, j)) return false;
            for (int di = -1; di <= 1; ++di)
                for (int dj = -1; dj <= 1; ++dj)
                    if (fgrid.is_occupied(i + di, j + dj)) return false;
            return true;
        };
        int ri = static_cast<int>((seed_map.x() - fgrid.x0) / fgrid.cell);
        int rj = static_cast<int>((seed_map.y() - fgrid.y0) / fgrid.cell);
        // The seed's own cell may sit within the dilation ring of a nearby wall: seed from the
        // nearest flood-free cell in a small window instead of giving up.
        if (not flood_free(ri, rj))
        {
            bool found = false;
            for (int r2 = 1; r2 <= 4 and not found; ++r2)
                for (int di = -r2; di <= r2 and not found; ++di)
                    for (int dj = -r2; dj <= r2 and not found; ++dj)
                        if (flood_free(ri + di, rj + dj)) { ri += di; rj += dj; found = true; }
            if (not found) return {};
        }
        std::vector<char> comp(static_cast<size_t>(nx * ny), 0);
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
        return comp;
    }

    float WallMap::jump_delta_nats(const Polygon& cur, const Polygon& trial, std::int64_t fresh_ref_ms,
                                   float* var_out) const
    {
        if (var_out != nullptr) *var_out = 0.f;
        if (not fgrid.ready() or not cur.closed or not trial.closed) return 0.f;
        // The two claims differ only near vertices one polygon has and the other lacks.
        float x0 = 1e9f, x1 = -1e9f, y0 = 1e9f, y1 = -1e9f;
        const auto unmatched = [&](const std::vector<Eigen::Vector2f>& A, const std::vector<Eigen::Vector2f>& B)
        {
            for (const auto& a : A)
            {
                bool m = false;
                for (const auto& b : B) if ((a - b).norm() < 1e-3f) { m = true; break; }
                if (not m)
                { x0 = std::min(x0, a.x()); x1 = std::max(x1, a.x()); y0 = std::min(y0, a.y()); y1 = std::max(y1, a.y()); }
            }
        };
        unmatched(cur.verts, trial.verts);
        unmatched(trial.verts, cur.verts);
        if (x0 > x1) return 0.f;
        const float M = 0.6f;
        const int i0 = std::max(0, static_cast<int>((x0 - M - fgrid.x0) / fgrid.cell));
        const int i1 = std::min(fgrid.nx - 1, static_cast<int>((x1 + M - fgrid.x0) / fgrid.cell));
        const int j0 = std::max(0, static_cast<int>((y0 - M - fgrid.y0) / fgrid.cell));
        const int j1 = std::min(fgrid.ny - 1, static_cast<int>((y1 + M - fgrid.y0) / fgrid.cell));
        float dn = 0.f;
        int n_mat_in = 0, n_mat_out = 0, n_free_in = 0, n_free_out = 0;
        float e_mat_in = 0.f, e_mat_out = 0.f, e_free_in = 0.f, e_free_out = 0.f;
        for (int i = i0; i <= i1; ++i)
            for (int j = j0; j <= j1; ++j)
            {
                const Eigen::Vector2f p = fgrid.at(i, j);
                const bool in_t = corner_visibility::point_in_polygon(p, trial.verts);
                const bool in_c = corner_visibility::point_in_polygon(p, cur.verts);
                if (in_t == in_c) continue;
                const size_t id = static_cast<size_t>(fgrid.idx(i, j));
                const float l = fgrid.lodds[id];
                const bool matter = fgrid.hits[id] >= 3 or l > 1.5f;
                const bool fresh_free = fgrid.is_free(i, j) and fgrid.free_ms[id] >= fresh_ref_ms;
                float w = 0.f;
                if (matter) w = -std::max(l, 2.f);      // claiming matter as interior costs
                else if (fresh_free) w = -l;            // claiming fresh free as interior gains (−l > 0)
                else continue;                          // stale free / unknown: silent
                const float e = in_t ? w : -w;
                dn += e;
                if (var_out != nullptr)
                {
                    // Two-valued outcome: this cell is really matter with p = sigmoid(l_eff) and
                    // really free otherwise, and the sign of its contribution follows. The spread
                    // between the two outcomes is 2|w|, so the variance is (2w)²·p(1−p).
                    const float l_eff = matter ? std::max(l, 2.f) : l;
                    const float p = 1.f / (1.f + std::exp(-l_eff));
                    *var_out += 4.f * w * w * p * (1.f - p);
                }
                if (matter and in_t) { ++n_mat_in; e_mat_in += e; } else if (matter) { ++n_mat_out; e_mat_out += e; }
                else if (in_t) { ++n_free_in; e_free_in += e; } else { ++n_free_out; e_free_out += e; }
            }
        if (params.debug_splice)
            std::printf("[gridterm] dn=%.1f | matter->in %d (%.1f) matter->out %d (%.1f) | free->in %d (%.1f) free->out %d (%.1f)\n",
                        dn, n_mat_in, e_mat_in, n_mat_out, e_mat_out, n_free_in, e_free_in, n_free_out, e_free_out);
        return dn;
    }

    // ─────────────────────────────────────────────────────────────────────────────────────────────
    //  Resolving a trial adoption.
    // ─────────────────────────────────────────────────────────────────────────────────────────────
    // Both outlines are scored against the grid AS IT IS NOW — which has had the whole trial window
    // of new beams added to it, including beams aimed at the region the two disagree about, because
    // the explorer goes where the map is uncertain. That is the entire mechanism: the proposal was
    // judged on a grid that could not settle it, and the verdict is passed on one that can.
    //
    // The comparison is not perfectly fair and it is worth being explicit about how. The challenger
    // has had the window to evolve — splices, births, deaths — while the snapshot is frozen at the
    // moment of the proposal, so a challenger is judged in its improved form against an incumbent
    // in its original one. Running both maps forward would fix that and cost twice the compute for
    // a difference that is second order over tens of frames: wall refinement in that window is
    // sub-centimetre against an 8 cm cell, and structural moves the challenger earned are mostly
    // ones the incumbent would have earned too.
    void WallMap::resolve_trial()
    {
        if (not trial_.open) return;
        const Polygon pnow = build_polygon();
        // RE-SCORE THE ORIGINAL QUESTION on the newer grid. Not "is the evolved map better than the
        // frozen one" — that comparison hands the challenger every splice, birth and death it
        // happened to earn during the window and the incumbent none, and it showed: the
        // apartamento's best seed fell 0.975 to 0.898 with deaths going 5 to 217, because a
        // challenger only has to out-evolve a corpse. The two outlines that were actually in
        // dispute are re-scored against the grid as it is now. Only the evidence has changed.
        Polygon pinc; pinc.verts = trial_.inc_verts; pinc.closed = trial_.inc_verts.size() >= 3;
        Polygon pcha; pcha.verts = trial_.cha_verts; pcha.closed = trial_.cha_verts.size() >= 3;
        float var = 0.f;
        const float dE_now = (pinc.closed and pcha.closed) ? jump_delta_nats(pinc, pcha, 0, &var) : 0.f;
        // A challenger whose cycle has since fallen open has failed outright, whatever the nats say.
        const bool survives = pnow.closed and pinc.closed and pcha.closed and dE_now > 0.f;
        if (params.debug_splice)
            std::printf("[trial] RESOLVE at frame %d (opened %d, dE was %.1f): now dE %.1f (sigma %.1f), closed %d -> %s\n",
                        frames_observed_, trial_.opened_at, trial_.dE_at_open, dE_now,
                        std::sqrt(std::max(var, 0.f)), static_cast<int>(pnow.closed),
                        survives ? "KEEP" : "REVERT");
        if (not survives)
        {
            walls = std::move(trial_.walls);
            order = std::move(trial_.order);
            candidates = std::move(trial_.candidates);
            heal_order();
            reclassify_all();
            seed_extents_from_polygon();
        }
        trial_ = Trial{};
    }

    bool WallMap::re_derive(const Eigen::Vector2f& robot_map)
    {
        if (not fgrid.ready()) return false;
        const int nx = fgrid.nx, ny = fgrid.ny;
        // Connected free component containing the robot.
        const std::vector<char> comp = free_component(robot_map);
        if (comp.empty())
        { if (params.debug_splice) std::printf("[rederive] bail: no free component\n"); return false; }
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
        if (contour.size() < 12)
        { if (params.debug_splice) std::printf("[rederive] bail: contour %zu cells\n", contour.size()); return false; }
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
        if (simp.size() < 3)
        { if (params.debug_splice) std::printf("[rederive] bail: simplified to %zu verts\n", simp.size()); return false; }
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
        std::vector<Eigen::Vector2f> starts;   // edge-start per new_order entry (junction anchors)
        std::vector<WallLandmark> created;
        float support_in = 0.f;   // line support the created walls bring in (candidate bins, one currency)
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
                // Strict Manhattan: a contour run is stair-stepped by the grid; snap its line to
                // the nearest Manhattan direction through the run's midpoint before creating the
                // wall, so re-derivation cannot introduce oblique edges either.
                float phi_w = phi_c, d_w = d_c;
                if (params.manhattan_strict and theta0_born)
                {
                    float best_eps = std::numeric_limits<float>::infinity();
                    for (int k2 = 0; k2 < 4; ++k2)
                    {
                        const float eps = wrap_pi(phi_c - theta0 - static_cast<float>(k2) * kPi * 0.5f);
                        if (std::abs(eps) < std::abs(best_eps))
                        { best_eps = eps; phi_w = wrap_pi(theta0 + static_cast<float>(k2) * kPi * 0.5f); }
                    }
                    d_w = linefit::normal_of(phi_w).dot(0.5f * (a + b));
                }
                WallLandmark w = make_wall(phi_w, d_w - 1.5f * fgrid.cell, weak, 0.5f * params.birth_nats, 0);
                const Eigen::Vector2f tv = w.tangent();
                w.s_min = std::min(tv.dot(a), tv.dot(b)); w.s_max = std::max(tv.dot(a), tv.dot(b));
                w.has_extent = true;
                created.push_back(w);
                chosen = w.id;
            }
            if (new_order.empty() or new_order.back() != chosen)
            { new_order.push_back(chosen); starts.push_back(a); }
        }
        // (rectilinearization happens AFTER candidate promotion below — a run snapped to a
        // candidate ref would otherwise be invisible to it and keep a parallel adjacency.)
        const auto rectilinearize = [&]()
        {
        if (params.manhattan_strict and new_order.size() >= 3 and starts.size() == new_order.size())
        {
            const auto dir_of = [&](std::uint64_t id) -> const WallLandmark*
            {
                if (id >= 1000000000ULL) return nullptr;   // candidate refs: promoted later, skip
                for (const auto& w : created) if (w.id == id) return &w;
                return find(id);
            };
            const Eigen::Matrix2f weak2 = Eigen::Vector2f(30.f, 15.f).asDiagonal();
            // Implied ⊥ connector through the junction; its normal is interior-LEFT of the walk,
            // which travels from wa2's line to wb2's line along wa2's normal. Fetch nothing from
            // wa2/wb2 after created.push_back — the pointers dangle on reallocation.
            const auto connect = [&](const WallLandmark* wa2, const WallLandmark* wb2, bool par,
                                     const Eigen::Vector2f& vj) -> std::uint64_t
            {
                const float gap = par ? std::abs(wa2->d - wb2->d) : std::abs(wa2->d + wb2->d);
                const Eigen::Vector2f na = wa2->normal();
                const float d_target = par ? wb2->d : -wb2->d;
                const Eigen::Vector2f wd = na * ((d_target - wa2->d) >= 0.f ? 1.f : -1.f);
                const Eigen::Vector2f nc(-wd.y(), wd.x());
                WallLandmark conn = make_wall(std::atan2(nc.y(), nc.x()), nc.dot(vj),
                                              weak2, 0.5f * params.birth_nats, 0);
                const Eigen::Vector2f tc2 = conn.tangent();
                conn.s_min = tc2.dot(vj) - gap - 0.15f;
                conn.s_max = tc2.dot(vj) + gap + 0.15f;
                conn.has_extent = true;
                created.push_back(conn);
                return conn.id;
            };
            // Single forward pass with true MERGING: two DISTINCT near-collinear walls adjacent in
            // the sequence previously survived (only identical ids were collapsed) and closure
            // failed on "edges parallel, no corner". Under 2.5 cells of separation — below any
            // main-line feature — they are one line at grid resolution: the better-supported wall
            // stands for both.
            std::vector<std::uint64_t> ro;
            const auto push_entry = [&](std::uint64_t id, const Eigen::Vector2f& vj)
            {
                while (true)
                {
                    if (ro.empty()) { ro.push_back(id); return; }
                    if (ro.back() == id) return;
                    const auto* wa2 = dir_of(ro.back());
                    const auto* wb2 = dir_of(id);
                    if (wa2 == nullptr or wb2 == nullptr) { ro.push_back(id); return; }
                    const float dphi2 = wrap_pi(wa2->phi - wb2->phi);
                    const bool par  = std::abs(dphi2) < 0.1f;
                    const bool anti = std::abs(wrap_pi(dphi2 - kPi)) < 0.1f;
                    if (not par and not anti) { ro.push_back(id); return; }
                    const float gap = par ? std::abs(wa2->d - wb2->d) : std::abs(wa2->d + wb2->d);
                    if (par and gap < 2.5f * fgrid.cell)
                    {
                        // Merge: keep the better-supported line and RE-CHECK against the new back.
                        if (wb2->points_seen > wa2->points_seen) { ro.pop_back(); continue; }
                        return;
                    }
                    ro.push_back(connect(wa2, wb2, par, vj));
                    ro.push_back(id);
                    return;
                }
            };
            const int M2 = static_cast<int>(new_order.size());
            for (int m = 0; m < M2; ++m)
                push_entry(new_order[static_cast<size_t>(m)], starts[static_cast<size_t>(m)]);
            // The seam (back ↔ front) obeys the same rules.
            while (ro.size() > 3)
            {
                if (ro.back() == ro.front()) { ro.pop_back(); continue; }
                const auto* wa2 = dir_of(ro.back());
                const auto* wb2 = dir_of(ro.front());
                if (wa2 == nullptr or wb2 == nullptr) break;
                const float dphi2 = wrap_pi(wa2->phi - wb2->phi);
                const bool par  = std::abs(dphi2) < 0.1f;
                const bool anti = std::abs(wrap_pi(dphi2 - kPi)) < 0.1f;
                if (not par and not anti) break;
                const float gap = par ? std::abs(wa2->d - wb2->d) : std::abs(wa2->d + wb2->d);
                if (par and gap < 2.5f * fgrid.cell) { ro.pop_back(); continue; }
                ro.push_back(connect(wa2, wb2, par, starts.front()));
                break;
            }
            new_order = std::move(ro);
        }
        };
        // Promote referenced candidates to walls.
        for (auto& id : new_order)
            if (id >= 1000000000ULL)
            {
                const auto& c = candidates[static_cast<size_t>(id - 1000000000ULL)];
                WallLandmark w = make_wall(c.phi, c.d, c.information, params.birth_nats, c.last_ms);
                w.s_min = c.s_min; w.s_max = c.s_max; w.has_extent = c.npts > 0;
                w.frames_seen = c.frames; w.points_seen = c.npts;
                // Born with the support its candidate observed (one currency), which is also what
                // the adoption may spend for it.
                if (not c.bins.empty())
                {
                    w.bins_s0 = c.bins_s0;
                    w.exist_bins.resize(c.bins.size());
                    for (size_t b = 0; b < c.bins.size(); ++b)
                        w.exist_bins[b] = std::min(2.f * params.birth_nats, params.birth_nats + c.bins[b]);
                }
                support_in += c.evidence();
                created.push_back(w);
                id = w.id;
            }
        // RECTILINEARIZE (strict Manhattan): snapping stair-stepped contour runs to the four
        // directions leaves ADJACENT PARALLEL runs — build_from finds no corner between parallel
        // lines and the cycle never closes (measured: closed=0 on 100% of re-derivations,
        // iou_new=-1 — the primary estimator silently dead since strictness landed). Same-line
        // neighbours merge; offset parallel neighbours get the implied perpendicular JOG through
        // their junction; ANTI-parallel neighbours get the implied CAP — the contour thereby
        // expresses notches and spurs natively, with no splice enumeration. Runs after candidate
        // promotion so every entry resolves to a real wall.
        rectilinearize();
        if (new_order.size() > 1 and new_order.front() == new_order.back()) new_order.pop_back();
        if (new_order.size() < 3)
        { if (params.debug_splice) std::printf("[rederive] bail: only %zu runs after snap\n", new_order.size()); return false; }

        // Adopt iff the new cycle explains the observed free space BETTER (grid IoU) — the global
        // free-energy comparison, evaluated on the evidence both cycles claim to explain.
        for (const auto& w : created) walls.push_back(w);
        // ── WRAP PRESERVATION: a bin-backed wrapped thin wall (an anti-parallel pair at thin
        // separation) must SURVIVE adoption. The trace caught a 2904-point, 26-bin face being
        // discarded by an adoption that improved grid IoU 0.757→0.843 — the area criterion is
        // blind to thin MATTER. Each lost evidence-backed pair is re-spliced into the contour
        // cycle as the narrow notch [host, face, cap, face, host] before the comparison: the
        // contour contributes topology, beam-confirmed walls are not erased for free.
        {
            const auto near3 = [&](const WallLandmark* x, const WallLandmark* y)
            { return std::abs(wrap_pi(x->phi - y->phi)) < 0.2f and std::abs(x->d - y->d) < 0.3f; };
            for (size_t a2 = 0; a2 < order.size(); ++a2)
                for (size_t b2 = a2 + 1; b2 < order.size(); ++b2)
                {
                    const auto* fa = find(order[a2]);
                    const auto* fb = find(order[b2]);
                    if (fa == nullptr or fb == nullptr or fa->id == fb->id) continue;
                    if (std::abs(wrap_pi(wrap_pi(fa->phi - fb->phi) - kPi)) >= 0.35f
                        or std::abs(fa->d + fb->d) >= 0.5f) continue;
                    const auto* fs = (fa->points_seen >= fb->points_seen) ? fa : fb;
                    if (fs->points_seen < 500) continue;                         // evidence-backed only
                    // Lost as a PAIR: the new cycle must hold an anti-parallel thin pair matching
                    // this one. Testing mere MEMBER presence was measured inert — the supported
                    // face survives as a plain boundary edge while the wrap is gone.
                    bool pair_present = false;
                    for (size_t k3 = 0; k3 < new_order.size() and not pair_present; ++k3)
                        for (size_t k4 = k3 + 1; k4 < new_order.size() and not pair_present; ++k4)
                        {
                            const auto* x2 = find(new_order[k3]);
                            const auto* y2 = find(new_order[k4]);
                            if (x2 == nullptr or y2 == nullptr) continue;
                            if (std::abs(wrap_pi(wrap_pi(x2->phi - y2->phi) - kPi)) >= 0.35f
                                or std::abs(x2->d + y2->d) >= 0.5f) continue;
                            if ((near3(x2, fa) and near3(y2, fb)) or (near3(x2, fb) and near3(y2, fa)))
                                pair_present = true;
                        }
                    if (pair_present) continue;
                    if (params.debug_splice)
                        std::printf("[wrap-keep?] lost pair (d=%.3f pts=%d / d=%.3f pts=%d) — searching anchor\n",
                                    fa->d, fa->points_seen, fb->d, fb->points_seen);
                    // Anchor at the SURVIVING member's entry (the measured case); insert the cap
                    // and the missing face beside it — tip end, cap sign, side and host-resume
                    // enumerated, validation choosing.
                    int surv_pos = -1;
                    const WallLandmark* miss = nullptr;
                    for (size_t k3 = 0; k3 < new_order.size() and surv_pos < 0; ++k3)
                        if (const auto* w2 = find(new_order[k3]); w2 != nullptr)
                        {
                            if (near3(w2, fa)) { surv_pos = static_cast<int>(k3); miss = fb; }
                            else if (near3(w2, fb)) { surv_pos = static_cast<int>(k3); miss = fa; }
                        }
                    if (surv_pos < 0)
                    {
                        if (params.debug_splice)
                            std::printf("[wrap-keep?] no anchor for the lost pair in the new cycle\n");
                        continue;
                    }
                    (void)miss;
                    // TRANSPLANT the old wrap's own sub-path — the pair's cyclic arc (faces plus
                    // the real cap between them) from the old, VALIDATED cycle — instead of
                    // fabricating a cap at the missing face's raw extent end (measured: tip -5.83,
                    // a 6 m slice, 118/128 variants self-crossing). Geometry is inherited;
                    // orientation and neighbour-resume are enumerated; validation chooses.
                    int pa = -1, pb = -1;
                    for (size_t k3 = 0; k3 < order.size(); ++k3)
                    {
                        if (order[k3] == fa->id and pa < 0) pa = static_cast<int>(k3);
                        if (order[k3] == fb->id and pb < 0) pb = static_cast<int>(k3);
                    }
                    if (pa < 0 or pb < 0) continue;
                    const int NO = static_cast<int>(order.size());
                    const auto arc = [&](int from, int to)
                    {
                        std::vector<std::uint64_t> s2;
                        for (int k3 = from; ; k3 = (k3 + 1) % NO)
                        {
                            s2.push_back(order[static_cast<size_t>(k3)]);
                            if (k3 == to) break;
                            if (static_cast<int>(s2.size()) > 5) { s2.clear(); break; }
                        }
                        return s2;
                    };
                    std::vector<std::uint64_t> sub = arc(pa, pb);
                    if (sub.empty()) sub = arc(pb, pa);
                    if (sub.empty())
                    {
                        if (params.debug_splice)
                            std::printf("[wrap-keep?] pair's old arc longer than 5 entries — skipped\n");
                        continue;
                    }
                    if (params.debug_splice)
                    {
                        std::printf("[wk-arc]");
                        for (const auto ida2 : sub)
                            if (const auto* wA = find(ida2); wA != nullptr)
                                std::printf(" (id=%llu phi=%.2f d=%.2f pts=%d)",
                                            (unsigned long long)wA->id, wA->phi, wA->d, wA->points_seen);
                        std::printf("\n");
                    }
                    bool injected = false;
                    for (const bool fwd : {true, false})
                    {
                        if (injected) break;
                        std::vector<std::uint64_t> path = sub;
                        if (not fwd) std::reverse(path.begin(), path.end());
                        for (const int resume3 : {0, 1, 2})   // none / resume-prev after / resume-next before
                        {
                            if (injected) break;
                            std::vector<std::uint64_t> o2;
                            const int NN = static_cast<int>(new_order.size());
                            for (int j = 0; j < NN; ++j)
                            {
                                if (j == surv_pos)
                                {
                                    if (resume3 == 2) o2.push_back(new_order[static_cast<size_t>((j + 1) % NN)]);
                                    for (const auto idp : path) o2.push_back(idp);
                                    if (resume3 == 1) o2.push_back(new_order[static_cast<size_t>((j + NN - 1) % NN)]);
                                    continue;                 // the near-duplicate entry is replaced
                                }
                                o2.push_back(new_order[static_cast<size_t>(j)]);
                            }
                            const Polygon t2 = build_from(o2);
                            bool ok2 = t2.closed;
                            const char* wv = ok2 ? "valid" : "not-closed";
                            for (size_t e2 = 0; e2 < t2.verts.size() and ok2; ++e2)
                            {
                                const Eigen::Vector2f dv = t2.verts[(e2 + 1) % t2.verts.size()] - t2.verts[e2];
                                if (dv.norm() < 0.04f) { ok2 = false; wv = "short-edge"; break; }
                                const auto* We3 = find(t2.wall_of_edge[e2]);
                                if (We3 == nullptr) { ok2 = false; wv = "missing-wall"; break; }
                                if (Eigen::Vector2f(-dv.y(), dv.x()).dot(We3->normal()) <= 0.f)
                                { ok2 = false; wv = "interior"; break; }
                            }
                            if (params.debug_splice and not ok2)
                                std::printf("[wk-var] arc=%zu fwd=%d res=%d -> %s [%.60s]\n",
                                            path.size(), static_cast<int>(fwd), resume3, wv, t2.status.c_str());
                            if (ok2)
                            {
                                new_order = std::move(o2);
                                injected = true;
                                if (params.debug_splice)
                                    std::printf("[wrap-keep] transplanted the pair's %zu-entry arc (pts=%d) into the contour cycle\n",
                                                path.size(), fs->points_seen);
                            }
                        }
                    }
                    if (not injected and params.debug_splice)
                        std::printf("[wrap-keep?] all transplant variants refused (surv_pos=%d)\n", surv_pos);
                }
        }
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
        // A re-derived cycle can self-cross (a stair-step run snapped onto two lines that meet the
        // wrong way round). Until 2026-09-03 that failed closure and the cycle was refused unjudged
        // — the referee found 118 of them on three seeds, some 0.5 IoU better than the incumbent.
        // Repair it the way the published copy repairs itself: splice out the weakest crossing edge
        // on a COPY, immediately, until it closes or nothing crosses.
        if (params.adopt_repair)
        {
            // The copy needs walls and grid, not the beam store or the decision log: move those
            // out for the duration of the copy (a const-correct copy without them is not available).
            auto saved_beams = std::move(beams);
            auto saved_decisions = std::move(decisions);
            WallMap fix(*this);
            beams = std::move(saved_beams);
            decisions = std::move(saved_decisions);
            fix.order = new_order;
            Polygon pf = fix.build_polygon();
            for (size_t guard = 0; not pf.closed and not pf.crossing_edges.empty() and guard < fix.order.size(); ++guard)
            {
                if (not fix.repair_if_crossing(/*immediate=*/true)) break;
                pf = fix.build_polygon();
            }
            if (fix.order.size() != new_order.size() and params.debug_splice)
                std::printf("[rederive] self-crossing repaired: %zu -> %zu entries\n", new_order.size(), fix.order.size());
            new_order = fix.order;
        }
        const Polygon pnew = build_from(new_order);
        const Polygon pold = build_from(order);
        const float iou_new = grid_iou(pnew);
        const float iou_old = grid_iou(pold);
        // SURRENDER: observed existence support the new cycle would erase. Counted once per wall
        // (duplicate rides skipped), only ABOVE the birth seed, and only for walls whose LINE the
        // new cycle does not hold anywhere. Until 2026-09-03 this was a VETO beside a grid-IoU
        // margin of 0.02; the forward-model referee caught that judge refusing 127 re-derived
        // cycles the truth preferred by 0.03-0.09 IoU. It is now a PRICE inside the one energy.
        float surrender = 0.f;
        for (auto it = order.begin(); it != order.end(); ++it)
        {
            if (std::find(order.begin(), it, *it) != it) continue;   // count each wall once
            const auto* w = find(*it);
            if (w == nullptr) continue;
            bool held = false;
            for (const auto id2 : new_order)
                if (const auto* w2 = find(id2); w2 != nullptr
                    and std::abs(wrap_pi(w->phi - w2->phi)) < 0.2f
                    and std::abs(w->d - w2->d) < 0.3f) { held = true; break; }
            if (held) continue;
            for (const float b : w->exist_bins) surrender += std::max(0.f, b - params.birth_nats);
        }
        // ONE ENERGY for the adoption, the same as a splice: the grid's area term over the region
        // where the two cycles disagree, plus the support the created walls bring in, minus the
        // support surrendered, minus the code length of the vertices the new cycle adds.
        float e_var = 0.f;
        const float e_grid = pnew.closed and pold.closed ? jump_delta_nats(pold, pnew, 0, &e_var) : 0.f;
        const float code   = (static_cast<float>(pnew.verts.size()) - static_cast<float>(pold.verts.size())) * edge_code_nats(true);
        const float dE     = e_grid + support_in - surrender - code;
        // The incumbent judge: a +0.02 grid-IoU margin protects a healthy incumbent from noise
        // flapping; an incumbent whose own corner uncertainty fails the publish bar gets no such
        // protection against a strictly healthier challenger no worse on IoU; and observed support
        // above the surrender bar may not be erased. MEASURED 2026-09-03: the one energy agrees
        // with the single-step truth far better (seed 7: 123/125 vs 42/150) and produces a WORSE
        // map (0.940/0.793/0.919 vs 0.951/0.969/0.943) — a cycle +0.01 better now that creates 28
        // thinly supported walls is adopted, they die, the cycle flaps. The margin and the veto are
        // hysteresis that a single-step criterion cannot see.
        // THE WAIVER, PRICED. What it buys is corner sharpness, and sharpness is a precision, so
        // the gain has a natural size in nats: a 2-D corner carries two degrees of freedom, and
        // tightening it from sigma_old to sigma_new is worth 2*ln(sigma_old/sigma_new). The edges
        // the challenger adds already cost `code` in the same currency. Adopt when the first beats
        // the second.
        //
        // Unpriced — "any improvement at all, however small" — this was the churn. On the
        // apartamento the waiver fired 30 times on a challenger that moved the IoU by less than a
        // thousandth and added four edges each time (29.3 nats of code for nothing); the new walls
        // then died of no support, the down-jump removed the leftovers, and the re-derivation
        // proposed the very same cycle again. 36 of 38 adoptions had NEGATIVE energy. A strict
        // inequality on a continuous quantity is not a criterion: two re-derivations always differ
        // in the fourth decimal, so one of them is always "healthier".
        float waiver_gain = 0.f;
        if (pnew.closed and pold.closed and pnew.worst_corner_sigma > 0.f
            and pold.worst_corner_sigma > 0.f)
            waiver_gain = 2.f * std::log(pold.worst_corner_sigma / pnew.worst_corner_sigma);
        const bool health_waiver = pnew.closed and pold.closed
            and iou_new >= iou_old
            and pold.worst_corner_sigma > params.publish_corner_sigma
            and pnew.worst_corner_sigma < pold.worst_corner_sigma
            and (not params.adopt_waiver_priced or waiver_gain > code);
        // THE MARGIN, IN NATS (judge 3). dE is a difference of grid evidence, and the grid is not
        // the truth — a beam grazing a wall carves it fresh-free — so dE carries an error of its
        // own, and jump_delta_nats now reports it. Adopt only when the improvement is larger than
        // that error can explain. Judge 0 asks for a fixed slice of an overlap that cannot see a
        // 0.13 m partition; judge 1 asks for nothing at all over a grid that lies. This asks the
        // question both were reaching for, and its threshold is measured rather than chosen.
        const float e_sigma = std::sqrt(std::max(e_var, 0.f));
        const bool adopt_ok = params.adopt_judge == 3
            ? (pnew.closed and dE > params.adopt_sigma_k * e_sigma)
            : params.adopt_judge == 2
                ? pnew.closed
                : params.adopt_judge == 1
                    ? (pnew.closed and dE > 0.f)
                    : (((iou_new > iou_old + 0.02f) or health_waiver) and surrender <= params.adopt_surrender_nats);
        if (params.debug_splice)
        {
            std::printf("[rederive] runs=%zu created=%zu closed=%d dE=%.1f (grid %.1f + in %.1f - out %.1f - code %.1f) iou %.3f->%.3f -> %s [%.70s]\n",
                        new_order.size(), created.size(), static_cast<int>(pnew.closed),
                        dE, e_grid, support_in, surrender, code, iou_old, iou_new,
                        adopt_ok ? "ADOPT" : "keep", pnew.status.c_str());
            std::printf("[rederive]   margin: dE %.1f vs %.1f x sigma %.1f = %.1f nats -> %s\n",
                        dE, params.adopt_sigma_k, e_sigma, params.adopt_sigma_k * e_sigma,
                        dE > params.adopt_sigma_k * e_sigma ? "significant" : "noise");
            std::printf("[rederive]   waiver: corner %.4f -> %.4f, gain %.1f nats vs code %.1f -> %s\n",
                        pold.worst_corner_sigma, pnew.worst_corner_sigma, waiver_gain, code,
                        health_waiver ? "waived" : "no");
        }
        // ── ADOPTION-LOSS TRACE: does the adopted cycle LOSE a wrapped thin wall — an
        // anti-parallel pair at thin separation — that the current cycle holds? Three fence
        // variants proved the short-spur loss lives HERE, downstream of a solid fence (41/42
        // latched cells, full frames, spur still gone): catch the discarding adoption in the act.
        if (params.debug_splice and adopt_ok)
        {
            const auto pairs_of = [&](const std::vector<std::uint64_t>& ord)
            {
                std::vector<std::pair<const WallLandmark*, const WallLandmark*>> out;
                for (size_t a2 = 0; a2 < ord.size(); ++a2)
                    for (size_t b2 = a2 + 1; b2 < ord.size(); ++b2)
                    {
                        const auto* wa2 = find(ord[a2]);
                        const auto* wb2 = find(ord[b2]);
                        if (wa2 == nullptr or wb2 == nullptr or wa2->id == wb2->id) continue;
                        if (std::abs(wrap_pi(wrap_pi(wa2->phi - wb2->phi) - kPi)) < 0.35f
                            and std::abs(wa2->d + wb2->d) < 0.5f)
                            out.push_back({wa2, wb2});
                    }
                return out;
            };
            const auto near2 = [&](const WallLandmark* x, const WallLandmark* y)
            { return std::abs(wrap_pi(x->phi - y->phi)) < 0.2f and std::abs(x->d - y->d) < 0.3f; };
            const auto new_pairs = pairs_of(new_order);
            for (const auto& [oa, ob] : pairs_of(order))
            {
                bool kept = false;
                for (const auto& [na2, nb2] : new_pairs)
                    if ((near2(oa, na2) and near2(ob, nb2)) or (near2(oa, nb2) and near2(ob, na2)))
                    { kept = true; break; }
                if (not kept)
                    std::printf("[adopt-loss] wrapped pair LOST by adoption: "
                                "(phi=%.3f d=%.3f pts=%d bins=%zu)/(phi=%.3f d=%.3f pts=%d bins=%zu) iou %.3f->%.3f\n",
                                oa->phi, oa->d, oa->points_seen, oa->exist_bins.size(),
                                ob->phi, ob->d, ob->points_seen, ob->exist_bins.size(),
                                iou_old, iou_new);
            }
        }
        referee("adopt", adopt_ok, dE, 0.f, code, pold.verts, pnew.verts);
        // ── THE SEQUENCE JUDGE ───────────────────────────────────────────────────────────────────
        // Every judge above asks one question of one step, and the measurement says no such judge
        // can work here: room 32 escapes a 36° error through a step worth 2.5 nats against a 33-nat
        // standard deviation — indistinguishable from noise, and right. The incumbent judge refuses
        // it, the energy judge takes it only because it refuses almost nothing, and a significance
        // margin refuses it precisely because it is calibrated. So stop asking the step to justify
        // itself and ask the SEQUENCE: adopt on trial, run the map on it, and keep it only if it
        // still explains the grid better once the robot has looked again. The verdict is then
        // passed on evidence that did not exist when the proposal was made, which is the one thing
        // a per-step test can never have.
        bool trial_open_now = false;
        if (not adopt_ok and params.trial_adoption and not trial_.open
            and pnew.closed and pold.closed and dE > 0.f)
        {
            trial_.walls = walls;
            // The trial's own new walls are already in `walls`; the snapshot is the map WITHOUT
            // them, so a revert leaves no orphans behind.
            for (const auto& w : created)
                for (int wi = static_cast<int>(trial_.walls.size()) - 1; wi >= 0; --wi)
                    if (trial_.walls[static_cast<size_t>(wi)].id == w.id)
                    { trial_.walls.erase(trial_.walls.begin() + wi); break; }
            trial_.order = order;
            trial_.candidates = candidates;
            trial_.inc_verts = pold.verts;
            trial_.cha_verts = pnew.verts;
            trial_.open = true;
            trial_.frames_left = std::max(1, params.trial_frames);
            trial_.opened_at = frames_observed_;
            trial_.dE_at_open = dE;
            trial_open_now = true;
            if (params.debug_splice)
                std::printf("[trial] OPEN at frame %d: dE %.1f, iou %.3f->%.3f, %zu -> %zu edges, %d frames to prove it\n",
                            frames_observed_, dE, iou_old, iou_new, pold.verts.size(), pnew.verts.size(),
                            trial_.frames_left);
        }
        if (adopt_ok or trial_open_now)
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
            // The conservative judge approving a further change while a trial is open is the map
            // moving on with its blessing: the trial has served its purpose and its snapshot is
            // now older than anything worth restoring.
            if (adopt_ok and trial_.open and not trial_open_now)
            {
                if (params.debug_splice)
                    std::printf("[trial] CLOSED by a judged adoption at frame %d\n", frames_observed_);
                trial_ = Trial{};
            }
            return true;
        }
        // Not better: discard the trial walls.
        for (const auto& w : created)
            if (const int wi = index_of(w.id); wi >= 0) walls.erase(walls.begin() + wi);
        return false;
    }

    Polygon WallMap::build_polygon() const { return build_from(order); }

    float WallMap::edge_code_nats(bool manhattan_class) const
    {
        const float res  = fgrid.ready() ? fgrid.cell : 0.08f;
        const float span = 2.f * params.sensor_range;
        const float offset = std::log(span / res);
        const float angle  = manhattan_class ? std::log(4.f) : std::log(2.f * kPi * params.sensor_range / res);
        return offset + angle;
    }

    bool WallMap::repair_if_crossing(bool immediate)
    {
        const Polygon poly = build_from(order);
        if (poly.crossing_edges.empty()) { crossing_frames_ = 0; return false; }
        if (not immediate and ++crossing_frames_ < 5) return false;   // persistence: one bad refinement frame must not amputate
        crossing_frames_ = 0;
        std::uint64_t weakest = 0; int weakest_pts = std::numeric_limits<int>::max();
        for (int e : poly.crossing_edges)
        {
            if (e < 0 or e >= static_cast<int>(poly.wall_of_edge.size())) continue;
            const auto* w = find(poly.wall_of_edge[static_cast<size_t>(e)]);
            if (w != nullptr and w->points_seen < weakest_pts) { weakest_pts = w->points_seen; weakest = w->id; }
        }
        if (weakest != 0)
        {
            if (params.debug_splice)
                std::printf("[death-cross] wall %llu pts=%d spliced out to uncross the cycle\n",
                            (unsigned long long)weakest, weakest_pts);
            splice_out(weakest);
            return true;
        }
        return false;
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
            if (w.prior_info.trace() > 0.f)
                xf(w.prior_mu.x(), w.prior_mu.y(), w.prior_info, nullptr, nullptr);
        }
        for (auto& cd : candidates) xf(cd.phi, cd.d, cd.information, &cd.s_min, &cd.s_max);
        // An open trial's snapshot lives in the same frame and must move with it, or reverting
        // would put the map back in the OLD frame — a silent (c, rot) error appearing tens of
        // frames after the re-anchor that caused it.
        if (trial_.open)
        {
            for (auto& w : trial_.walls)
            {
                const float shift = linefit::tangent_of(w.phi).dot(c);
                xf(w.phi, w.d, w.information, w.has_extent ? &w.s_min : nullptr,
                   w.has_extent ? &w.s_max : nullptr);
                w.bins_s0 -= shift;
                if (w.prior_info.trace() > 0.f)
                    xf(w.prior_mu.x(), w.prior_mu.y(), w.prior_info, nullptr, nullptr);
            }
            for (auto& cd : trial_.candidates) xf(cd.phi, cd.d, cd.information, &cd.s_min, &cd.s_max);
        }
        if (theta0_born) theta0 = wrap_pi(theta0 - rot);
        // The histogram lives in the map frame like everything else. Rotating the map by −rot
        // rotates the quadrupled angle by −4·rot; carrying that as an offset avoids re-binning
        // (and the interpolation error a re-bin would add at every re-anchor).
        orient_off_ -= 4.0 * static_cast<double>(rot);
        // Stored beams move with the frame: p' = R(−rot)(p − c).
        const Eigen::Matrix2f Rb = rot2(-rot);
        for (auto& b : beams) { b.o = Rb * (b.o - c); b.d = Rb * b.d; }
        // The free-space grid moves with the frame too — until 2026-09-03 it did not, and every
        // grid read after the agent's one-shot re-anchor (the stub discriminator, jump_delta_nats,
        // frontiers, the re-derivation, level 2) was off by (c, rot). The bench never re-anchors,
        // so it never showed. Resampled nearest-cell into a grid of the same size centred on the
        // new origin: old point of a new cell centre p' is R(rot) p' + c.
        if (fgrid.ready())
        {
            FreeGrid g2;
            g2.cell = fgrid.cell;
            g2.nx = fgrid.nx; g2.ny = fgrid.ny;
            g2.x0 = -0.5f * static_cast<float>(g2.nx) * g2.cell;
            g2.y0 = -0.5f * static_cast<float>(g2.ny) * g2.cell;
            g2.lodds.assign(static_cast<size_t>(g2.nx * g2.ny), 0.f);
            g2.hits.assign(static_cast<size_t>(g2.nx * g2.ny), 0);
            g2.free_ms.assign(static_cast<size_t>(g2.nx * g2.ny), -1);
            const Eigen::Matrix2f Rf = rot2(rot);
            for (int j = 0; j < g2.ny; ++j)
                for (int i = 0; i < g2.nx; ++i)
                {
                    const Eigen::Vector2f p = Rf * g2.at(i, j) + c;
                    const int oi = static_cast<int>(std::floor((p.x() - fgrid.x0) / fgrid.cell));
                    const int oj = static_cast<int>(std::floor((p.y() - fgrid.y0) / fgrid.cell));
                    if (not fgrid.in(oi, oj)) continue;
                    const size_t o = static_cast<size_t>(fgrid.idx(oi, oj)), n = static_cast<size_t>(g2.idx(i, j));
                    g2.lodds[n] = fgrid.lodds[o]; g2.hits[n] = fgrid.hits[o]; g2.free_ms[n] = fgrid.free_ms[o];
                }
            fgrid = std::move(g2);
            comp_cache_ts_ = -1;   // the cached free component is in the old frame
        }
    }

    float WallMap::beam_loglik(const Beam& b, const std::vector<Eigen::Vector2f>& poly) const
    {
        float r_pred = params.sensor_range;
        for (size_t e = 0; e < poly.size(); ++e)
            if (const auto t = corner_visibility::ray_segment_t(b.o, b.d, poly[e], poly[(e + 1) % poly.size()]); t and *t > 0.f)
                r_pred = std::min(r_pred, *t);
        const float sig = params.beam_sigma_m;
        const float z = b.r, zmax = params.sensor_range;
        const float w_rand = std::max(0.f, 1.f - params.beam_w_hit - params.beam_w_short);
        const float dz = (z - r_pred) / sig;
        const float p_hit   = params.beam_w_hit * std::exp(-0.5f * dz * dz) / (sig * 2.5066283f);
        const float p_short = (z < r_pred) ? params.beam_w_short / std::max(r_pred, sig) : 0.f;
        const float p_rand  = w_rand / zmax;
        return std::log(std::max(p_hit + p_short + p_rand, 1e-30f));
    }

    float WallMap::forward_delta(const std::vector<Eigen::Vector2f>& cur, const std::vector<Eigen::Vector2f>& trial) const
    {
        if (beams.empty() or cur.size() < 3 or trial.size() < 3) return 0.f;
        // Only beams whose ray enters the box of the vertices the two polygons do not share can
        // see a different first hit; everything else cancels exactly.
        Eigen::Vector2f lo(1e9f, 1e9f), hi(-1e9f, -1e9f);
        const auto unmatched = [&](const std::vector<Eigen::Vector2f>& A, const std::vector<Eigen::Vector2f>& B)
        {
            for (const auto& a : A)
            {
                bool m = false;
                for (const auto& q : B) if ((a - q).norm() < 1e-3f) { m = true; break; }
                if (not m) { lo = lo.cwiseMin(a); hi = hi.cwiseMax(a); }
            }
        };
        unmatched(cur, trial); unmatched(trial, cur);
        if (lo.x() > hi.x()) return 0.f;
        lo.array() -= 0.3f; hi.array() += 0.3f;
        const auto ray_hits_box = [&](const Beam& b)
        {
            float t0 = 0.f, t1 = b.r + 3.f * params.beam_sigma_m;
            for (int k = 0; k < 2; ++k)
            {
                const float o = b.o[k], d = b.d[k];
                if (std::abs(d) < 1e-9f) { if (o < lo[k] or o > hi[k]) return false; continue; }
                float ta = (lo[k] - o) / d, tb = (hi[k] - o) / d;
                if (ta > tb) std::swap(ta, tb);
                t0 = std::max(t0, ta); t1 = std::min(t1, tb);
                if (t0 > t1) return false;
            }
            return true;
        };
        float dll = 0.f;
        for (const auto& b : beams)
            if (ray_hits_box(b)) dll += beam_loglik(b, trial) - beam_loglik(b, cur);
        return dll;
    }

    void WallMap::referee(const char* site, bool accepted, float evidence, float cost, float fwd_cost,
                          const std::vector<Eigen::Vector2f>& cur, const std::vector<Eigen::Vector2f>& trial) const
    {
        if (not params.referee_log) return;
        Decision d;
        d.site = site; d.frame = frames_observed_; d.accepted = accepted;
        d.evidence = evidence; d.cost = cost; d.fwd_cost = fwd_cost;
        d.fwd_dll = forward_delta(cur, trial);
        d.cur = cur; d.trial = trial;
        if (params.debug_splice)
            std::printf("[referee] %-6s incumbent %s (%.1f vs %.1f) | forward %s (dll %.1f vs %.1f)\n",
                        site, accepted ? "ACCEPT" : "refuse", evidence, cost,
                        d.fwd_dll > fwd_cost ? "ACCEPT" : "refuse", d.fwd_dll, fwd_cost);
        decisions.push_back(std::move(d));
    }
} // namespace rc::wallmap
