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

        /// Covariance of a (φ, d) pair from its information, or nullopt when the information is not
        /// positive definite (a rank-1 segment: points all at the same tangent coordinate cannot say
        /// anything about φ). The caller decides what "unknown" means for its test.
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

        /// The systematic (model-error) covariance every innovation carries — see Params::map_sigma_d.
        Eigen::Matrix2f sys_cov(const Params& p)
        {
            return Eigen::Vector2f(p.map_sigma_phi_rad * p.map_sigma_phi_rad,
                                   p.map_sigma_d * p.map_sigma_d).asDiagonal();
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
            else
            {
                // Degenerate: neither side pins both parameters; take the better-supported one.
                if (info_b.trace() > info.trace()) { phi = wrap_pi(phib); d = d_b; }
            }
            info = L;
        }
    } // namespace

    std::optional<std::uint64_t> WallLandmark::neighbour(int end) const
    {
        const auto& v = votes[static_cast<size_t>(end)];
        if (v.empty()) return std::nullopt;
        auto best = std::max_element(v.begin(), v.end(),
                                     [](const auto& a, const auto& b) { return a.second < b.second; });
        return best->first;
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
        // n_m = R(θ) n_r ⇒ φ_m = φ_r + θ;  d_m = n_m·(R p + t) = d_r + n_m·t for any p on the line.
        phi_m = wrap_pi(phi_r + pose.z());
        const Eigen::Vector2f n = linefit::normal_of(phi_m);
        const Eigen::Vector2f tv = linefit::tangent_of(phi_m);
        const Eigen::Vector2f t = pose.head<2>();
        d_m = d_r + n.dot(t);
        // ∂φ_m/∂θ = 1;  ∂d_m/∂(x,y) = n;  ∂d_m/∂θ = (∂n/∂θ)·t = t(φ_m)·t
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
        // The alternative "this wall obeys no class" has prior mass p_off. If the best class costs more
        // than that alternative, the wall is a chamfer as far as the evidence goes: k = −1, no factor.
        const float off_cost = -std::log(std::clamp(params.manhattan_off_prior, 1e-6f, 1.f - 1e-6f));
        if (best > off_cost) { c.k = -1; c.eps = 0.f; c.cost = off_cost; }
        return c;
    }

    int WallMap::birth_from_candidate(int ci, std::int64_t /*ts*/)
    {
        const Candidate c = candidates[static_cast<size_t>(ci)];
        WallLandmark w;
        w.id = next_id_++;
        w.phi = c.phi; w.d = c.d;
        w.information = c.information;
        const auto cls = classify(c.phi);
        w.k = cls.k;
        w.manhattan_var = (w.k >= 0) ? params.manhattan_sigma_rad * params.manhattan_sigma_rad : 0.f;
        w.has_extent = c.npts > 0;
        w.s_min = c.s_min; w.s_max = c.s_max;
        w.frames_seen = c.frames;
        w.points_seen = c.npts;
        w.exist_lodds = params.birth_nats;   // what birth just proved; refutable from here
        walls.push_back(w);
        candidates.erase(candidates.begin() + ci);
        if (not theta0_born) try_birth_theta0();
        return static_cast<int>(walls.size()) - 1;
    }

    void WallMap::try_birth_theta0()
    {
        // Two walls agree modulo 90° when the best of the four class residuals between them is a
        // better explanation than "no class" — the same test classify() applies once θ₀ exists.
        const float var = params.manhattan_sigma_rad * params.manhattan_sigma_rad;
        const float off_cost = -std::log(std::clamp(params.manhattan_off_prior, 1e-6f, 1.f - 1e-6f));
        int best_a = -1, best_b = -1; float best_info = -1.f;
        for (size_t a = 0; a < walls.size(); ++a)
            for (size_t b = a + 1; b < walls.size(); ++b)
            {
                float best = std::numeric_limits<float>::infinity();
                for (int k = 0; k < 4; ++k)
                {
                    const float eps = wrap_pi(walls[b].phi - walls[a].phi - static_cast<float>(k) * kPi * 0.5f);
                    best = std::min(best, 0.5f * eps * eps / var);
                }
                if (best >= off_cost) continue;
                // Prefer the pair with the most angular information — the sharpest definition of the yaw.
                const float info = walls[a].information(0, 0) + walls[b].information(0, 0);
                if (info > best_info) { best_info = info; best_a = static_cast<int>(a); best_b = static_cast<int>(b); }
            }
        if (best_a < 0) return;
        const auto& A = walls[static_cast<size_t>(best_a)];
        const auto& B = walls[static_cast<size_t>(best_b)];
        // θ₀ := the sharper wall's angle (class 0); the other wall's residual is what σ_ε absorbs.
        const auto& ref = (A.information(0, 0) >= B.information(0, 0)) ? A : B;
        theta0 = ref.phi;
        theta0_born = true;
        theta0_information = std::max(0.f, ref.information(0, 0));
        reclassify_all();
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

    void WallMap::vote_adjacency(int wa, int wb, const Eigen::Vector2f& corner_map)
    {
        auto& A = walls[static_cast<size_t>(wa)];
        auto& B = walls[static_cast<size_t>(wb)];
        const auto end_of = [&](const WallLandmark& w) -> int
        {
            const float s = w.tangent().dot(corner_map);
            if (not w.has_extent) return (s >= 0.f) ? 1 : 0;
            return (std::abs(s - w.s_max) < std::abs(s - w.s_min)) ? 1 : 0;
        };
        A.votes[static_cast<size_t>(end_of(A))][B.id] += 1.f;
        B.votes[static_cast<size_t>(end_of(B))][A.id] += 1.f;
    }

    void WallMap::update_existence(const std::vector<Eigen::Vector2f>& pts_robot, const Eigen::Vector3f& pose,
                                   FrameResult& fr)
    {
        if (walls.empty() or pts_robot.empty()) return;
        const Eigen::Matrix2f Rm = rot2(pose.z());
        const Eigen::Matrix2f Rt = Rm.transpose();                  // map → robot rotation
        const Eigen::Vector2f t = pose.head<2>();
        const Eigen::Vector2f origin = Eigen::Vector2f::Zero();
        const float bw = std::max(0.05f, params.exist_bin_m);
        const float lo_clamp = -1.5f * params.birth_nats, hi_clamp = 2.f * params.birth_nats;

        for (auto& w : walls)
        {
            if (not w.has_extent or w.s_max - w.s_min < bw) continue;

            // Bins cover the CURRENT extent; extend (never here shrink) when association grew it.
            // New bins are seeded from the wall's summary odds clamped to [0, birth_nats] — extension
            // came from supported points, but a fresh stretch has not yet earned the full record.
            const float seed = std::clamp(w.exist_lodds, 0.f, params.birth_nats);
            if (w.exist_bins.empty())
            {
                // No bin may lie beyond the extent: a bin no beam can ever test would sit immortal at
                // its seed odds and keep a fully-refuted wall alive through max(bins).
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
                if (not hit) continue;                              // this beam never crossed the wall
                const float s_hit = tv.dot(Rm * (dir * *hit) + t);  // where along the wall, map frame
                const int bin = std::clamp(static_cast<int>(std::floor((s_hit - w.bins_s0) / bw)), 0, nb - 1);
                const float inc = std::abs(n_r.dot(dir));           // grazing beams testify weakly
                const float dr  = r - *hit;
                if (std::abs(dr) <= params.huber_delta)      sup[static_cast<size_t>(bin)] += inc;
                else if (dr > params.huber_delta)            ref[static_cast<size_t>(bin)] += inc;
                // dr < −huber_delta: return SHORT of the wall — occluded, no evidence for this bin.
            }
            for (int bin = 0; bin < nb; ++bin)
            {
                const float denom = sup[static_cast<size_t>(bin)] + params.exist_refute_pdet * ref[static_cast<size_t>(bin)];
                if (denom <= 0.f) continue;                         // no beam crossed this bin: holds
                // At most ±1 nat per bin per frame, and WEAK evidence scales down rather than being
                // skipped: a sliver crossed by one grazing beam per frame must still accumulate —
                // skipping it made a refuted wall's last untestable bin immortal at its seed odds.
                const float delta = std::clamp((sup[static_cast<size_t>(bin)] - params.exist_refute_pdet * ref[static_cast<size_t>(bin)]) / std::max(denom, 1.f), -1.f, 1.f);
                w.exist_bins[static_cast<size_t>(bin)] = std::clamp(w.exist_bins[static_cast<size_t>(bin)] + delta, lo_clamp, hi_clamp);
            }

            // Dead END bins shrink the extent (a doorway hole in the middle just stays a hole; it can
            // recover if the world changes back). The corners come from line intersections, so a
            // slightly conservative extent costs nothing there.
            int first = 0, last = nb - 1;
            while (first < nb and w.exist_bins[static_cast<size_t>(first)] < -params.birth_nats) ++first;
            while (last >= first and w.exist_bins[static_cast<size_t>(last)] < -params.birth_nats) --last;
            // Nothing left, or only a stub whose TESTABLE span (bins ∩ extent) is shorter than two
            // bins: a wall whose surviving extent is below what a line landmark can assert is not a
            // wall any more (the corner detector's admissibility rule, applied to what refutation
            // left standing). Testable span, not bin count — a sliver can straddle two bins.
            const float testable = (first > last) ? 0.f
                : std::min(w.s_max, w.bins_s0 + bw * static_cast<float>(last + 1))
                  - std::max(w.s_min, w.bins_s0 + bw * static_cast<float>(first));
            if (first > last or testable < 2.f * bw)
            { w.exist_lodds = lo_clamp; continue; }                 // marked for death below
            if (first > 0 or last < nb - 1)
            {
                w.exist_bins.assign(w.exist_bins.begin() + first, w.exist_bins.begin() + last + 1);
                w.bins_s0 += bw * static_cast<float>(first);
                w.s_min = std::max(w.s_min, w.bins_s0);
                w.s_max = std::min(w.s_max, w.bins_s0 + bw * static_cast<float>(w.exist_bins.size()));
            }
            w.exist_lodds = *std::max_element(w.exist_bins.begin(), w.exist_bins.end());
        }

        // Deaths — every bin refuted, at the SAME decisive bar as birth.
        for (int i = static_cast<int>(walls.size()) - 1; i >= 0; --i)
        {
            if (walls[static_cast<size_t>(i)].exist_lodds > lo_clamp + 1e-3f) continue;
            const auto& w = walls[static_cast<size_t>(i)];
            fr.deaths_info.push_back({w.id, w.exist_lodds, w.frames_seen, w.points_seen});
            fr.deaths++;
            const std::uint64_t dead = w.id;
            walls.erase(walls.begin() + i);
            for (auto& wl : walls)
                for (int e = 0; e < 2; ++e) wl.votes[static_cast<size_t>(e)].erase(dead);
        }
    }

    FrameResult WallMap::observe(const wallseg::Result& seg, const std::vector<Eigen::Vector2f>& pts_robot,
                                 const Eigen::VectorXf& weights, const Eigen::Vector3f& pose,
                                 const Eigen::Matrix3f& pose_cov, std::int64_t timestamp_ms)
    {
        FrameResult fr;
        // ── Existence first: the step-back operator. Runs on the PRE-association wall set so no
        // index computed below can dangle, and a just-born wall (seeded at +birth_nats) cannot die
        // on the very frame that bore it.
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
            // (φ_r, d_r) → (φ_m, d_m): φ_m = φ_r + θ, d_m = d_r + n(φ_m)·t ⇒ ∂d_m/∂φ_r = t(φ_m)·t.
            Eigen::Matrix2f J = Eigen::Matrix2f::Identity();
            J(1, 0) = linefit::tangent_of(sm[s].phi).dot(t);
            sm[s].Sigma = J * (*cov_r) * J.transpose() + H * pose_cov * H.transpose();
            sm[s].ok = sm[s].Sigma.allFinite();
        }

        // ── Association: Mahalanobis under the innovation covariance, MANY-TO-ONE, PDA ───────────
        // Deliberately NOT the corner detector's Hungarian: one-to-one is right for point landmarks
        // (one detection per corner) and wrong for extended ones — a single wall legitimately yields
        // SEVERAL segments per scan (furniture breaks it, doorways, occlusion). Under Hungarian the
        // extra segments of a wall were "unexplained", became candidates, and were born as TWINS
        // (measured live: BIRTH … nearest chi2=0.0). Each segment simply takes its best in-gate wall.
        std::vector<std::vector<float>> chi2(static_cast<size_t>(S), std::vector<float>(static_cast<size_t>(W), std::numeric_limits<float>::infinity()));
        for (int s = 0; s < S; ++s)
        {
            if (not sm[s].ok) continue;
            for (int w = 0; w < W; ++w)
            {
                const auto& wl = walls[static_cast<size_t>(w)];
                const auto cov_w = cov_of(wl.information);
                // A wall whose information is not yet positive definite is still associable on what it
                // does know: fall back to the segment's own covariance alone (the wall contributes no
                // extra spread, which is the conservative side: the gate gets TIGHTER, never looser).
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
            // PDA posterior over every wall inside the gate — an aliasing segment mutes itself.
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
            fr.assoc.push_back(std::move(a));
        }

        // ── Observed corners → adjacency votes (after this frame's leak) ─────────────────────────
        for (auto& w : walls)
            for (auto& v : w.votes)
                for (auto& [id, n] : v) n *= (1.f - params.vote_leak);
        for (const auto& oc : seg.corners)
        {
            const int wa = fr.seg_to_wall[static_cast<size_t>(oc.seg_a)];
            const int wb = fr.seg_to_wall[static_cast<size_t>(oc.seg_b)];
            if (wa < 0 or wb < 0 or wa == wb) continue;
            vote_adjacency(wa, wb, R * oc.point + t);
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
                if (c.this_frame_seg >= 0) continue;    // one segment per candidate per frame
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
            // What this frame's points say for the (fused) candidate line over clutter, and their extent.
            // The extent used is this SEGMENT's (the length the points were spread over when observed),
            // with the same clutter area the segmenter derived its band from.
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

        // ── Births by model comparison ──────────────────────────────────────────────────────────
        // ΔF = Σ gain − Occam − class. Occam charges the two parameters against a UNIFORM prior (φ on
        // the circle, d over the sensor's range) — the price of adding a wall at all.
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

            const int s = c.this_frame_seg;
            // Who is the nearest existing wall, and how decisively does the evidence separate them?
            BirthInfo bi;
            bi.phi = c.phi; bi.d = c.d; bi.npts = c.npts; bi.frames = c.frames; bi.dF = dF;
            int nearest_idx = -1;
            for (size_t wi = 0; wi < walls.size(); ++wi)
            {
                const auto& wl = walls[wi];
                const auto cov_w = cov_of(wl.information);
                const auto cov_c = cov_of(c.information);
                const Eigen::Matrix2f Sm = sys_cov(params) + (cov_w ? *cov_w : Eigen::Matrix2f::Zero())
                                         + (cov_c ? *cov_c : Eigen::Matrix2f::Zero());
                const Eigen::Vector2f r(wrap_pi(c.phi - wl.phi), c.d - wl.d);
                const float c2 = chi2_of(r, Sm);
                if (bi.nearest_chi2 < 0.f or c2 < bi.nearest_chi2)
                { bi.nearest_chi2 = c2; bi.nearest_wall = wl.id; nearest_idx = static_cast<int>(wi); }
            }
            // "Two walls cannot occupy the same line": a candidate the gate cannot separate from an
            // existing wall IS that wall — its points simply failed a per-segment association (a short
            // or oblique piece). Fuse its evidence in rather than bearing a twin.
            if (nearest_idx >= 0 and bi.nearest_chi2 <= params.assoc_chi2)
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
            const int w = birth_from_candidate(ci, timestamp_ms);
            bi.id = walls[static_cast<size_t>(w)].id;
            fr.births_info.push_back(bi);
            fr.births++;
            // The segment that completed the birth is this frame's observation of the new wall.
            const auto& sg = seg.segments[static_cast<size_t>(s)];
            auto& wl = walls[static_cast<size_t>(w)];
            fr.seg_to_wall[static_cast<size_t>(s)] = w;
            fr.seg_pda[static_cast<size_t>(s)] = 1.f;
            fr.seg_to_candidate[static_cast<size_t>(s)] = -1;
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
        }
        // Corners between a just-born wall and an already-associated one still count: re-run the
        // votes for pairs that became resolvable only now (cheap; corners are few).
        if (fr.births > 0)
            for (const auto& oc : seg.corners)
            {
                const int wa = fr.seg_to_wall[static_cast<size_t>(oc.seg_a)];
                const int wb = fr.seg_to_wall[static_cast<size_t>(oc.seg_b)];
                if (wa < 0 or wb < 0 or wa == wb) continue;
                // Only pairs not already voted above: those where at least one side was unassociated then.
                const bool a_was_new = (wa >= W), b_was_new = (wb >= W);
                if (a_was_new or b_was_new) vote_adjacency(wa, wb, R * oc.point + t);
            }

        // ── Bound the candidate list: drop the weakest by accumulated information trace ─────────
        while (static_cast<int>(candidates.size()) > params.max_candidates)
        {
            auto weakest = std::min_element(candidates.begin(), candidates.end(),
                [](const Candidate& a, const Candidate& b) { return a.npts < b.npts; });
            candidates.erase(weakest);
        }
        fr.candidates = static_cast<int>(candidates.size());
        fr.merged = merge_indistinguishable();
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
                    // Same physical wall: fuse information-weighted, union the extents, move B's votes to A.
                    fuse(A.phi, A.d, A.information, B.phi, B.d, B.information);
                    if (B.has_extent)
                    {
                        if (not A.has_extent) { A.s_min = B.s_min; A.s_max = B.s_max; A.has_extent = true; }
                        else { A.s_min = std::min(A.s_min, B.s_min); A.s_max = std::max(A.s_max, B.s_max); }
                    }
                    for (int e = 0; e < 2; ++e)
                        for (const auto& [id, n] : B.votes[static_cast<size_t>(e)]) A.votes[static_cast<size_t>(e)][id] += n;
                    A.frames_seen += B.frames_seen;
                    A.points_seen += B.points_seen;
                    A.exist_lodds = std::max(A.exist_lodds, B.exist_lodds);
                    A.exist_bins.clear();   // re-form over the union extent, seeded from the summary
                    if (A.k < 0 and B.k >= 0) { A.k = B.k; A.manhattan_var = B.manhattan_var; }
                    const std::uint64_t dead = B.id, kept = A.id;
                    walls.erase(walls.begin() + static_cast<long>(b));
                    for (auto& w : walls)
                        for (int e = 0; e < 2; ++e)
                        {
                            auto& v = w.votes[static_cast<size_t>(e)];
                            if (auto it = v.find(dead); it != v.end())
                            {
                                const float n = it->second; v.erase(it);
                                if (w.id != kept) v[kept] += n;
                            }
                        }
                    ++merged; again = true;
                }
        }
        return merged;
    }

    Corner WallMap::intersect_walls(const WallLandmark& a, const WallLandmark& b, bool inferred)
    {
        Corner c;
        c.wall_a = a.id; c.wall_b = b.id; c.inferred = inferred;
        c.sigma = std::numeric_limits<float>::infinity();
        const auto p = linefit::intersect(a.line(), b.line());
        if (not p) return c;
        c.p = *p;
        // p solves M p = d with M = [n_aᵀ; n_bᵀ]. dp = −M⁻¹ dM p + M⁻¹ dd:
        //   ∂p/∂d_a = M⁻¹ e₁,  ∂p/∂φ_a = −M⁻¹ e₁ (t_a·p);  likewise for b with e₂.
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

    Polygon WallMap::build_polygon() const
    {
        Polygon poly;
        std::ostringstream st;
        const int W = static_cast<int>(walls.size());
        if (W < 3) { poly.status = "fewer than 3 walls"; return poly; }

        // Neighbours: successor at the LOW end (CCW walk runs s decreasing), predecessor at the HIGH
        // end. A RECIPROCATED vote (B names A back at its own high end) beats a bare count, so a junk
        // wall that shares one corner with a real pair cannot capture the walk.
        std::vector<int> next(static_cast<size_t>(W), -1), prev(static_cast<size_t>(W), -1);
        const auto pick = [&](int i, int end) -> int
        {
            const auto& v = walls[static_cast<size_t>(i)].votes[static_cast<size_t>(end)];
            int best = -1; float best_n = 0.f; bool best_recip = false;
            for (const auto& [id, n] : v)
            {
                const int j = index_of(id);
                if (j < 0) continue;
                const auto& back = walls[static_cast<size_t>(j)].votes[static_cast<size_t>(1 - end)];
                const bool recip = back.count(walls[static_cast<size_t>(i)].id) > 0;
                if (best < 0 or (recip and not best_recip) or (recip == best_recip and n > best_n))
                { best = j; best_n = n; best_recip = recip; }
            }
            return best;
        };
        int voted = 0;
        for (int i = 0; i < W; ++i)
        {
            next[static_cast<size_t>(i)] = pick(i, 0);
            prev[static_cast<size_t>(i)] = pick(i, 1);
            if (next[static_cast<size_t>(i)] >= 0 or prev[static_cast<size_t>(i)] >= 0) ++voted;
        }
        if (voted < 3) { poly.status = "fewer than 3 walls with an observed corner"; return poly; }

        // Walk from EVERY wall; keep the longest closed cycle, else the longest open chain.
        std::vector<int> best_chain; bool best_closed = false;
        for (int s0 = 0; s0 < W; ++s0)
        {
            std::vector<int> chain;
            std::vector<char> visited(static_cast<size_t>(W), 0);
            int cur = s0; bool closed = false;
            while (cur >= 0 and not visited[static_cast<size_t>(cur)])
            {
                visited[static_cast<size_t>(cur)] = 1;
                chain.push_back(cur);
                cur = next[static_cast<size_t>(cur)];
                if (cur == s0) { closed = true; break; }
            }
            const bool better = (closed and not best_closed)
                             or (closed == best_closed and chain.size() > best_chain.size());
            if (better) { best_chain = chain; best_closed = closed; }
        }
        std::vector<int> chain = best_chain;
        bool closed = best_closed;
        if (chain.size() < 3) { poly.status = "no chain of 3 walls"; return poly; }

        // Stable start: the wall with the smallest (class, offset) in the cycle — a choice that does not
        // renumber the edges when the map is refined, which door_concept (keyed on the edge index) needs.
        if (closed)
        {
            size_t s = 0;
            for (size_t i = 1; i < chain.size(); ++i)
            {
                const auto& w = walls[static_cast<size_t>(chain[i])];
                const auto& b = walls[static_cast<size_t>(chain[s])];
                const int ke = (w.k < 0) ? 4 : w.k, ks = (b.k < 0) ? 4 : b.k;
                if (ke < ks or (ke == ks and w.d < b.d)) s = i;
            }
            std::rotate(chain.begin(), chain.begin() + static_cast<long>(s), chain.end());
        }

        bool closure_inferred = false;
        if (not closed)
        {
            // Exactly the two open ends left? Then the missing corner is the intersection of the two
            // end walls — determined by Manhattan even though nobody saw it — unless they are parallel.
            const auto& head = walls[static_cast<size_t>(chain.front())];
            const auto& tail = walls[static_cast<size_t>(chain.back())];
            const float sin_ab = std::abs(head.normal().x() * tail.normal().y() - head.normal().y() * tail.normal().x());
            if (sin_ab > 1e-3f) { closed = true; closure_inferred = true; }
            else st << "open chain of " << chain.size() << " walls (ends parallel); ";
        }
        const int orphans = voted - static_cast<int>(chain.size());
        const int unvoted = W - voted;
        if (orphans > 0) st << orphans << " voted wall(s) outside the loop; ";
        if (unvoted > 0) st << unvoted << " wall(s) with no observed corner (excluded); ";

        if (closed)
        {
            std::vector<Corner> corners;   // corners[i] = at the HIGH end of chain[i] (between chain[i-1] and chain[i])
            const int n = static_cast<int>(chain.size());
            for (int i = 0; i < n; ++i)
            {
                const int ip = chain[static_cast<size_t>((i + n - 1) % n)];
                const int ic = chain[static_cast<size_t>(i)];
                const bool inferred = (i == 0 and closure_inferred);
                corners.push_back(intersect_walls(walls[static_cast<size_t>(ip)], walls[static_cast<size_t>(ic)], inferred));
            }
            for (int i = 0; i < n; ++i)
            {
                poly.verts.push_back(corners[static_cast<size_t>(i)].p);
                poly.wall_of_edge.push_back(walls[static_cast<size_t>(chain[static_cast<size_t>(i)])].id);
            }
            poly.corners = corners;
            // Simple? No two non-adjacent edges may cross.
            for (int i = 0; i < n and closed; ++i)
                for (int j = i + 2; j < n; ++j)
                {
                    if (i == 0 and j == n - 1) continue;
                    if (corner_visibility::segments_cross(poly.verts[static_cast<size_t>(i)], poly.verts[static_cast<size_t>((i + 1) % n)],
                                                          poly.verts[static_cast<size_t>(j)], poly.verts[static_cast<size_t>((j + 1) % n)]))
                    { st << "self-crossing edges " << i << " and " << j << "; "; closed = false; }
                }
            // CCW by construction (normals into the room, s decreasing); a negative area means the
            // normals of at least one wall point the wrong way.
            float area2 = 0.f;
            for (int i = 0; i < n; ++i)
            {
                const auto& p = poly.verts[static_cast<size_t>(i)];
                const auto& q = poly.verts[static_cast<size_t>((i + 1) % n)];
                area2 += p.x() * q.y() - q.x() * p.y();
            }
            if (area2 <= 0.f) { st << "polygon is not counter-clockwise (a normal points out of the room); "; closed = false; }
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
            // d' = d − n·c has ∂d'/∂φ = −t·c: the information transforms with J = [[1,0],[−t·c,1]].
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
