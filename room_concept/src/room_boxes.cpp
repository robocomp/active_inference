#include "room_boxes.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <map>
#include <limits>

namespace rc::boxes
{
    namespace
    {
        constexpr float kPi = 3.14159265358979323846f;

        float wrap_pi(float a)
        {
            while (a >  kPi) a -= 2.f * kPi;
            while (a < -kPi) a += 2.f * kPi;
            return a;
        }

        /// Signed distance to one axis-aligned box, negative inside. The standard AABB form:
        /// d = max(lo - p, p - hi) componentwise; outside distance is |max(d,0)|, inside distance
        /// is the (negative) largest component. Exact everywhere including the corners.
        float sdf_box(const Box& b, const Eigen::Vector2f& p)
        {
            const Eigen::Vector2f d = (b.lo - p).cwiseMax(p - b.hi);
            const float outside = d.cwiseMax(0.f).norm();
            const float inside  = std::min(std::max(d.x(), d.y()), 0.f);
            return outside + inside;
        }

        /// Gradient of sdf_box. Outside it is the unit vector away from the nearest face/corner;
        /// inside it points along the axis of the nearest face.
        Eigen::Vector2f grad_box(const Box& b, const Eigen::Vector2f& p)
        {
            const Eigen::Vector2f d = (b.lo - p).cwiseMax(p - b.hi);
            if (d.x() > 0.f or d.y() > 0.f)
            {
                const Eigen::Vector2f q = d.cwiseMax(0.f);
                const float n = q.norm();
                if (n < 1e-9f) return {0.f, 0.f};
                // sign per axis: +1 if p is beyond hi, -1 if below lo
                const Eigen::Vector2f s((p.x() > b.hi.x()) ? 1.f : -1.f,
                                        (p.y() > b.hi.y()) ? 1.f : -1.f);
                return {s.x() * q.x() / n, s.y() * q.y() / n};
            }
            // inside: the nearest face decides
            if (d.x() > d.y()) return {(p.x() - b.lo.x() < b.hi.x() - p.x()) ? -1.f : 1.f, 0.f};
            return {0.f, (p.y() - b.lo.y() < b.hi.y() - p.y()) ? -1.f : 1.f};
        }
    }   // namespace

    namespace
    {
        /// Gaussian log-likelihood of the returns about the region's boundary. The only term
        /// structure has to beat, and it contains no tuned number: every sigma in it is physical.
        /// Each sample carries its OWN sigma, because a return placed by a pose that is uncertain
        /// to 20 cm cannot testify about a wall at 2 cm. The normaliser log(sigma) must be kept:
        /// without it a wide sigma would be free, and widening would always win.
        float log_likelihood(const Layout& L, const std::vector<CloudPoint>& pts)
        {
            double ll = 0.0;
            for (const auto& q : pts)
            {
                const float d = L.sdf(q.p);
                const float s = q.sigma_pose;
                ll -= static_cast<double>(d) * d / (2.0 * s * s) + std::log(s);
            }
            return static_cast<float>(ll);
        }

        /// One connected component, no holes. Checked on a PROPOSAL only — the continuous state
        /// cannot leave the valid set, so this never runs in the estimation loop.
        bool simply_connected(const Layout& L)
        {
            std::vector<float> xs, ys;
            for (const auto& b : L.boxes)
            { xs.push_back(b.lo.x()); xs.push_back(b.hi.x()); ys.push_back(b.lo.y()); ys.push_back(b.hi.y()); }
            std::sort(xs.begin(), xs.end()); xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
            std::sort(ys.begin(), ys.end()); ys.erase(std::unique(ys.begin(), ys.end()), ys.end());
            const int nx = static_cast<int>(xs.size()) - 1, ny = static_cast<int>(ys.size()) - 1;
            if (nx < 1 or ny < 1) return false;
            std::vector<char> in(static_cast<size_t>(nx * ny), 0);
            int total = 0, seed = -1;
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                {
                    const bool v = L.inside({0.5f * (xs[static_cast<size_t>(i)] + xs[static_cast<size_t>(i + 1)]),
                                             0.5f * (ys[static_cast<size_t>(j)] + ys[static_cast<size_t>(j + 1)])});
                    in[static_cast<size_t>(j * nx + i)] = v ? 1 : 0;
                    if (v) { ++total; if (seed < 0) seed = j * nx + i; }
                }
            if (total == 0) return false;
            std::vector<char> seen(in.size(), 0);
            std::vector<int> st{seed}; int reach = 0;
            while (not st.empty())
            {
                const int k = st.back(); st.pop_back();
                if (k < 0 or k >= nx * ny or seen[static_cast<size_t>(k)] or not in[static_cast<size_t>(k)]) continue;
                seen[static_cast<size_t>(k)] = 1; ++reach;
                const int i = k % nx, j = k / nx;
                if (i + 1 < nx) st.push_back(k + 1);
                if (i > 0)      st.push_back(k - 1);
                if (j + 1 < ny) st.push_back(k + nx);
                if (j > 0)      st.push_back(k - nx);
            }
            if (reach != total) return false;                 // disconnected
            // no holes: the OUTSIDE cells (padded) must also be one component
            const int px = nx + 2, py = ny + 2;
            std::vector<char> out(static_cast<size_t>(px * py), 1);
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i)
                    if (in[static_cast<size_t>(j * nx + i)]) out[static_cast<size_t>((j + 1) * px + i + 1)] = 0;
            std::vector<char> os(out.size(), 0);
            std::vector<int> st2{0}; int oreach = 0, ototal = 0;
            for (char c : out) ototal += c ? 1 : 0;
            while (not st2.empty())
            {
                const int k = st2.back(); st2.pop_back();
                if (k < 0 or k >= px * py or os[static_cast<size_t>(k)] or not out[static_cast<size_t>(k)]) continue;
                os[static_cast<size_t>(k)] = 1; ++oreach;
                const int i = k % px, j = k / px;
                if (i + 1 < px) st2.push_back(k + 1);
                if (i > 0)      st2.push_back(k - 1);
                if (j + 1 < py) st2.push_back(k + px);
                if (j > 0)      st2.push_back(k - px);
            }
            return oreach == ototal;                          // every outside cell reachable => no hole
        }
    }   // namespace

    bool Layout::inside(const Eigen::Vector2f& p) const
    {
        bool in = false;
        for (const auto& b : boxes)
            if (b.positive and b.contains(p)) { in = true; break; }
        if (not in) return false;
        for (const auto& b : boxes)
            if (not b.positive and b.contains(p)) return false;
        return true;
    }

    float Layout::sdf(const Eigen::Vector2f& p) const
    {
        if (boxes.empty()) return 0.f;
        float d = std::numeric_limits<float>::infinity();
        for (const auto& b : boxes)
            if (b.positive) d = std::min(d, sdf_box(b, p));          // union
        for (const auto& b : boxes)
            if (not b.positive) d = std::max(d, -sdf_box(b, p));     // subtraction
        return d;
    }

    std::vector<Eigen::Vector2f> Layout::polygon() const
    {
        // The arrangement of the boxes' own coordinates. Every vertex of the union is a crossing of
        // one x-offset with one y-offset, so the boundary is exact on this grid — there is no
        // intersection of near-parallel lines anywhere, which is what makes a runaway corner
        // arithmetically impossible.
        std::vector<Eigen::Vector2f> out;
        if (boxes.empty()) return out;
        std::vector<float> xs, ys;
        for (const auto& b : boxes)
        { xs.push_back(b.lo.x()); xs.push_back(b.hi.x()); ys.push_back(b.lo.y()); ys.push_back(b.hi.y()); }
        std::sort(xs.begin(), xs.end()); xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
        std::sort(ys.begin(), ys.end()); ys.erase(std::unique(ys.begin(), ys.end()), ys.end());
        const int nx = static_cast<int>(xs.size()) - 1, ny = static_cast<int>(ys.size()) - 1;
        if (nx < 1 or ny < 1) return out;

        auto cell_in = [&](int i, int j)
        {
            if (i < 0 or j < 0 or i >= nx or j >= ny) return false;
            return inside({0.5f * (xs[static_cast<size_t>(i)] + xs[static_cast<size_t>(i + 1)]),
                           0.5f * (ys[static_cast<size_t>(j)] + ys[static_cast<size_t>(j + 1)])});
        };

        // Walk the boundary anticlockwise with the interior on the left. Start at the lowest-left
        // inside cell's bottom-left corner heading +x — for a single box that is its corner, and the
        // walk returns exactly four vertices.
        int si = -1, sj = -1;
        for (int j = 0; j < ny and sj < 0; ++j)
            for (int i = 0; i < nx; ++i)
                if (cell_in(i, j)) { si = i; sj = j; break; }
        if (si < 0) return out;

        // Direction 0:+x 1:+y 2:-x 3:-y, on the grid graph of corner nodes.
        int ci = si, cj = sj, dir = 0;
        const int start_i = ci, start_j = cj;
        const int guard = 4 * (nx + 2) * (ny + 2);
        for (int step = 0; step < guard; ++step)
        {
            // Try to turn left first, then straight, then right, then back: the standard
            // left-hand wall follower on the cell complex, which traces one closed loop.
            for (int turn = 1; turn >= -2; --turn)
            {
                const int nd = ((dir + turn) % 4 + 4) % 4;
                const int di = (nd == 0) ? 1 : (nd == 2) ? -1 : 0;
                const int dj = (nd == 1) ? 1 : (nd == 3) ? -1 : 0;
                // the edge from corner (ci,cj) along nd is a boundary edge iff exactly one of the
                // two cells it separates is inside
                bool a = false, b = false;
                if (nd == 0) { a = cell_in(ci, cj);         b = cell_in(ci, cj - 1); }
                if (nd == 1) { a = cell_in(ci - 1, cj);     b = cell_in(ci, cj); }
                if (nd == 2) { a = cell_in(ci - 1, cj - 1); b = cell_in(ci - 1, cj); }
                if (nd == 3) { a = cell_in(ci, cj - 1);     b = cell_in(ci - 1, cj - 1); }
                if (a == b) continue;                       // not a boundary edge
                if (nd != dir or out.empty())
                    out.push_back({xs[static_cast<size_t>(ci)], ys[static_cast<size_t>(cj)]});
                ci += di; cj += dj; dir = nd;
                break;
            }
            if (ci == start_i and cj == start_j and step > 0) break;
        }
        return out;
    }

    std::vector<Eigen::Matrix2f> Layout::polygon_cov(float sigma_flat) const
    {
        const auto verts = polygon();
        std::vector<Eigen::Matrix2f> out(verts.size(), Eigen::Matrix2f::Identity() * sigma_flat * sigma_flat);
        if (cov.rows() != static_cast<long>(n_offsets())) return out;   // no joint covariance yet
        // A vertex IS two offsets — find which, and read the 2x2 sub-block. No line intersection,
        // so nothing can diverge when two walls are nearly parallel: they cannot be.
        for (size_t v = 0; v < verts.size(); ++v)
        {
            long ix = -1, iy = -1;
            for (size_t b = 0; b < boxes.size(); ++b)
            {
                const long o = static_cast<long>(b) * 4;
                if (std::abs(boxes[b].lo.x() - verts[v].x()) < 1e-6f) ix = o + 0;
                if (std::abs(boxes[b].hi.x() - verts[v].x()) < 1e-6f) ix = o + 2;
                if (std::abs(boxes[b].lo.y() - verts[v].y()) < 1e-6f) iy = o + 1;
                if (std::abs(boxes[b].hi.y() - verts[v].y()) < 1e-6f) iy = o + 3;
            }
            if (ix < 0 or iy < 0) continue;
            Eigen::Matrix2f S;
            S << cov(ix, ix), cov(ix, iy), cov(iy, ix), cov(iy, iy);
            out[v] = S + Eigen::Matrix2f::Identity() * sigma_flat * sigma_flat;
        }
        return out;
    }

    std::optional<std::pair<float, Layout>>
    init_from_scan(const std::vector<Eigen::Vector2f>& pts, const InitParams& p)
    {
        if (pts.size() < 20) return std::nullopt;

        // ── YAW: the quadrupled-angle vote ───────────────────────────────────────────────────
        // A Manhattan room's surface normals lie on two perpendicular axes, so 4*theta maps all
        // four classes onto one circle and the peak is the frame's orientation. Local direction
        // comes from each point's neighbour in scan order — no segmentation, no RANSAC, no OBB
        // (an OBB tilts 15-20 deg on a concave first scan; this cannot).
        std::vector<double> hist(static_cast<size_t>(p.yaw_bins), 0.0);
        for (size_t i = 1; i < pts.size(); ++i)
        {
            const Eigen::Vector2f d = pts[i] - pts[i - 1];
            const float len = d.norm();
            if (len < 1e-3f or len > 0.5f) continue;        // same surface, not a jump
            const float a = std::atan2(d.y(), d.x());
            double u = 4.0 * static_cast<double>(a);
            u -= 2.0 * kPi * std::floor(u / (2.0 * kPi));
            const int b = std::clamp(static_cast<int>(u * p.yaw_bins / (2.0 * kPi)), 0, p.yaw_bins - 1);
            hist[static_cast<size_t>(b)] += len;            // longer runs vote harder
        }
        const auto pk = std::max_element(hist.begin(), hist.end());
        if (pk == hist.end() or *pk <= 0.0) return std::nullopt;
        const double ub = (static_cast<double>(std::distance(hist.begin(), pk)) + 0.5)
                          * 2.0 * kPi / p.yaw_bins;
        // ⚠ THE FRAME IS ONLY DETERMINED MOD 90 DEGREES, AND THAT IS CORRECT.
        // A quadrupled-angle vote cannot tell the four Manhattan classes apart, and it does not
        // need to: the map frame is a GAUGE. The same room in a frame turned by 90 degrees is the
        // same room with its axes swapped, and the robot pose lives in that same frame, so every
        // consumer sees a consistent world. Do NOT try to pick a canonical quarter-turn from the
        // scan: measured 2026-09-19, a rectangle's residual is IDENTICAL under all four, so the
        // score cannot discriminate and picking one anyway broke the case that worked (rect IoU
        // 0.977 -> 0.074, pose error 5.05 m). Anything COMPARING this frame to another — a bench
        // scoring IoU against a world polygon, or a consumer holding an older map — must align the
        // two, which is a property of the comparison, not of the estimate.
        const float yaw = static_cast<float>(ub / 4.0);

        // ── THE BOX: interval hull along the recovered axes, at a quantile ───────────────────
        const float c = std::cos(-yaw), s = std::sin(-yaw);
        std::vector<float> xs, ys;
        xs.reserve(pts.size()); ys.reserve(pts.size());
        for (const auto& q : pts)
        { xs.push_back(c * q.x() - s * q.y()); ys.push_back(s * q.x() + c * q.y()); }
        auto quant = [](std::vector<float>& v, float f)
        {
            const size_t k = std::clamp<size_t>(static_cast<size_t>(f * static_cast<float>(v.size() - 1)),
                                                0, v.size() - 1);
            std::nth_element(v.begin(), v.begin() + static_cast<long>(k), v.end());
            return v[k];
        };
        std::vector<float> xs2 = xs, ys2 = ys;
        Box b;
        b.lo = {quant(xs, 1.f - p.edge_quantile), quant(ys, 1.f - p.edge_quantile)};
        b.hi = {quant(xs2, p.edge_quantile),      quant(ys2, p.edge_quantile)};
        b.positive = true; b.id = 1;
        if (not b.valid()) return std::nullopt;

        Layout L;
        L.boxes.push_back(b);
        // Offsets start at the sensor's own precision on the returns that defined them, saturated
        // by the flatness term — the Woodbury common mode, so a dense scan cannot claim sub-mm.
        const float var = p.sensor_sigma * p.sensor_sigma / std::max(1.f, 0.25f * static_cast<float>(pts.size()))
                        + p.sigma_flat * p.sigma_flat;
        L.cov = Eigen::MatrixXf::Identity(4, 4) * var;
        return std::make_pair(yaw, L);
    }

    namespace
    {
        /// Voxelise a cloud to one sample per cell, keeping the SMALLEST pose sigma seen there —
        /// the best look you ever had at that patch of surface. Shared by refit() and grow(), so
        /// they cannot disagree about what the evidence is.
        std::vector<CloudPoint> condense(const std::vector<CloudPoint>& cloud, const GrowParams& p)
        {
            const float s0sq = p.sensor_sigma * p.sensor_sigma + p.sigma_flat * p.sigma_flat;
            std::map<std::pair<int, int>, std::pair<Eigen::Vector2f, float>> vox;
            for (const auto& q : cloud)
            {
                const std::pair<int, int> k{static_cast<int>(std::floor(q.p.x() / p.cell)),
                                            static_cast<int>(std::floor(q.p.y() / p.cell))};
                auto it = vox.find(k);
                if (it == vox.end()) vox.emplace(k, std::make_pair(q.p, q.sigma_pose));
                else if (q.sigma_pose < it->second.second) it->second = {q.p, q.sigma_pose};
            }
            std::vector<CloudPoint> out;
            out.reserve(vox.size());
            for (const auto& [k, v] : vox) out.push_back({v.first, std::sqrt(s0sq + v.second * v.second)});
            return out;
        }
    }   // namespace

    float refit(Layout& L, const std::vector<CloudPoint>& cloud, const GrowParams& p, int iters)
    {
        if (L.empty() or cloud.empty()) return 0.f;
        const std::vector<CloudPoint> pts = condense(cloud, p);
        if (pts.empty()) return 0.f;

        const size_t n = L.n_offsets();
        const float eps = 1e-3f;
        double rms = 0.0;
        for (int it = 0; it < iters; ++it)
        {
            const Layout prev = L;
            std::vector<double> H(n, 0.0), g(n, 0.0);
            double ss = 0.0;
            for (const auto& q : pts)
            {
                const float d = L.sdf(q.p);
                const double w = 1.0 / (static_cast<double>(q.sigma_pose) * q.sigma_pose);
                ss += static_cast<double>(d) * d;
                for (size_t k = 0; k < n; ++k)
                {
                    Layout T = L;
                    Box& b = T.boxes[k / 4];
                    ((k % 4 == 0) ? b.lo.x() : (k % 4 == 1) ? b.lo.y() : (k % 4 == 2) ? b.hi.x() : b.hi.y()) += eps;
                    const float jk = (T.sdf(q.p) - d) / eps;
                    if (std::abs(jk) < 1e-4f) continue;          // this offset does not move this point
                    H[k] += w * jk * jk;
                    g[k] -= w * jk * static_cast<double>(d);
                }
            }
            rms = std::sqrt(ss / static_cast<double>(pts.size()));
            // Diagonal step. An offset no point is active on keeps its value — it is unobserved,
            // which is the honest outcome, not a reason to move it.
            for (size_t k = 0; k < n; ++k)
            {
                if (H[k] <= 0.0) continue;
                Box& b = L.boxes[k / 4];
                float& o = (k % 4 == 0) ? b.lo.x() : (k % 4 == 1) ? b.lo.y() : (k % 4 == 2) ? b.hi.x() : b.hi.y();
                o += static_cast<float>(g[k] / H[k]);
            }
            // Re-tie every attached face to its host. The step above treated all offsets as
            // free; an attached offset is not one, and letting it drift detaches the column.
            for (auto& b : L.boxes)
            {
                if (b.attach < 0 or b.host >= L.boxes.size()) continue;
                const Box& hb = L.boxes[b.host];
                switch (b.attach)
                {
                    case 0: b.lo.x() = hb.lo.x(); break;
                    case 1: b.lo.y() = hb.lo.y(); break;
                    case 2: b.hi.x() = hb.hi.x(); break;
                    case 3: b.hi.y() = hb.hi.y(); break;
                    // an alcove's inner face IS the wall it opens through — the host's far side
                    case 4: b.lo.x() = hb.hi.x(); break;
                    case 5: b.lo.y() = hb.hi.y(); break;
                    case 6: b.hi.x() = hb.lo.x(); break;
                    default: b.hi.y() = hb.lo.y(); break;
                }
            }
            // A box may not invert. `width -> 0` is the removal event and is priced elsewhere; here
            // it is simply held at the degenerate limit rather than allowed to turn inside out.
            for (auto& b : L.boxes)
            {
                if (b.hi.x() <= b.lo.x()) { const float m = 0.5f * (b.lo.x() + b.hi.x()); b.lo.x() = m - 1e-3f; b.hi.x() = m + 1e-3f; }
                if (b.hi.y() <= b.lo.y()) { const float m = 0.5f * (b.lo.y() + b.hi.y()); b.lo.y() = m - 1e-3f; b.hi.y() = m + 1e-3f; }
            }
            // A carve cannot be larger than the thing it is carved out of. An identity, not a
            // threshold: material removed outside the room was never there to remove.
            // A CARVE cannot be larger than the thing it is carved out of — an identity, not a
            // threshold. ⚠ It applies to negative boxes ONLY: an alcove lies outside its host by
            // construction, and clamping it there would collapse every outward proposal to zero.
            for (auto& b : L.boxes)
            {
                if (b.positive or b.host >= L.boxes.size()) continue;
                const Box& hb = L.boxes[b.host];
                b.lo = b.lo.cwiseMax(hb.lo); b.hi = b.hi.cwiseMin(hb.hi);
            }
            // ⚠ THE CONTINUOUS STATE *CAN* LEAVE THE VALID SET — the header's claim holds for the
            // positive part only. A negative box that grows until it touches the opposite wall
            // SPLITS the room in two, and nothing in the parametrisation forbids it: c1's carve
            // ran to 0.04 x 4.27 m, severed a perfect 6.00 x 4.00 room and took IoU 0.967 -> 0.137.
            // The invariant the design promises therefore has to be enforced here, on the step —
            // which is still a property of the REPRESENTATION, not a tuned rule about rooms.
            if (not simply_connected(L)) { L = prev; break; }
        }
        return static_cast<float>(rms);
    }

    GrowResult grow(Layout& L, const std::vector<CloudPoint>& cloud, const GrowParams& p)
    {
        GrowResult R;
        if (L.empty() or cloud.size() < static_cast<size_t>(p.min_cluster)) return R;

        // ── 0. VOXELISE. A surface seen a thousand times is one piece of geometric evidence. ──
        // Without this the likelihood is extensive in FRAME COUNT, so the description length — a
        // fixed ~24 nats — can never outvote it and every proposal is admitted.
        const float s0sq = p.sensor_sigma * p.sensor_sigma + p.sigma_flat * p.sigma_flat;
        const std::vector<CloudPoint> pts = condense(cloud, p);
        if (pts.size() < static_cast<size_t>(p.min_cluster)) return R;

        // ── 1. WHAT DOES THE REGION FAIL TO EXPLAIN? ─────────────────────────────────────────
        // A return lands on a surface, so |sdf| ~ 0 under a correct layout. Returns lying well
        // INSIDE are matter the region does not model — measured against their OWN sigma, so
        // odometry drift is not mistaken for a column.
        // Surprise is measured against the FITTED incumbent and the OVERDISPERSED sigma — the same
        // two corrections the scoring uses. Testing against the unfitted layout at sensor sigma is
        // what made a 14-cm wall misfit look like a room full of columns.
        Layout B0 = L; refit(B0, cloud, p, 3);
        float ov0 = 0.f;
        {
            std::vector<float> ad; ad.reserve(pts.size());
            double known = 0.0;
            for (const auto& q : pts)
            { ad.push_back(std::abs(B0.sdf(q.p))); known += static_cast<double>(q.sigma_pose) * q.sigma_pose; }
            std::nth_element(ad.begin(), ad.begin() + static_cast<long>(ad.size() / 2), ad.end());
            const float mad = 1.4826f * ad[ad.size() / 2];
            ov0 = std::max(0.f, mad * mad - static_cast<float>(known / static_cast<double>(pts.size())));
        }
        // Both directions. Inside = matter the region does not model (a column). Outside = room
        // the region does not cover (an alcove, a door recess). Same test, same sigma, same price.
        std::vector<CloudPoint> un, out;
        for (const auto& q : pts)
        {
            const float s = std::sqrt(q.sigma_pose * q.sigma_pose + ov0);
            const float dd = B0.sdf(q.p);
            if (dd < -3.f * s) un.push_back(q);
            else if (dd > 3.f * s) out.push_back(q);
        }
        if (un.size() < static_cast<size_t>(p.min_cluster)
            and out.size() < static_cast<size_t>(p.min_cluster)) return R;

        // ── 2. CLUSTER, on a grid, 4-connected. One routine, used for both directions. ───────
        const auto cluster = [&](const std::vector<CloudPoint>& src)
        {
            std::map<std::pair<int, int>, std::vector<int>> cells;
            for (size_t i = 0; i < src.size(); ++i)
                cells[{static_cast<int>(std::floor(src[i].p.x() / p.cell)),
                       static_cast<int>(std::floor(src[i].p.y() / p.cell))}].push_back(static_cast<int>(i));
            std::map<std::pair<int, int>, int> label;
            int nlab = 0;
            for (const auto& [cc, v] : cells)
            {
                if (label.count(cc)) continue;
                const int id = nlab++;
                std::vector<std::pair<int, int>> stack{cc};
                while (not stack.empty())
                {
                    const auto k = stack.back(); stack.pop_back();
                    if (label.count(k) or not cells.count(k)) continue;
                    label[k] = id;
                    stack.push_back({k.first + 1, k.second}); stack.push_back({k.first - 1, k.second});
                    stack.push_back({k.first, k.second + 1}); stack.push_back({k.first, k.second - 1});
                }
            }
            std::vector<std::vector<Eigen::Vector2f>> g(static_cast<size_t>(nlab));
            for (const auto& [cc, v] : cells)
                for (int i : v) g[static_cast<size_t>(label[cc])].push_back(src[static_cast<size_t>(i)].p);
            return g;
        };
        const auto groups     = cluster(un);
        const auto groups_out = cluster(out);

        // ── 2b. REMOVAL IS AN EDIT TOO, AND IT IS PRICED THE SAME WAY ───────────────────────
        // Without this, admission is a ratchet: a carve bought by early, badly-placed evidence can
        // never be given back, and c2 finished with eight negative boxes eating half the room at
        // IoU 0.490. Deleting a box REFUNDS its description length, so a carve that has stopped
        // paying for itself leaves on exactly the Bayes factor that let it in. Symmetric by
        // construction — there is no separate retirement rule and no hysteresis constant.
        {
            const float sigr = std::sqrt(s0sq);
            for (size_t bi = L.boxes.size(); bi-- > 0;)
            {
                if (L.boxes[bi].positive) continue;
                Layout T = L;
                T.boxes.erase(T.boxes.begin() + static_cast<long>(bi));
                for (auto& b : T.boxes) if (b.host > bi and b.host > 0) --b.host;
                if (not simply_connected(T)) continue;
                refit(T, cloud, p, 3);
                Layout K = L; refit(K, cloud, p, 3);
                const Box& hb = L.boxes[L.boxes[bi].host < L.boxes.size() ? L.boxes[bi].host : 0];
                const float span = std::max(1.f, hb.width() + hb.height());
                const float code = 3.f * std::log(span / sigr) + std::log(4.f);
                if (log_likelihood(T, pts) + code > log_likelihood(K, pts))
                { L = T; R.admitted = -1; return R; }     // one edit per call, removal included
            }
        }

        // ── 3. PROPOSE ONE EDIT PER CLUSTER, AND PRICE IT ────────────────────────────────────
        // Two kinds of edit, and the second is what stops the box count running away. A column is
        // first seen from ONE side, so its hull is too small on the far side; the next sweep finds
        // the rest of it still unexplained. Offering only "a new box" made that a SECOND box, and
        // c2 (two columns) ended with five. An edit that WIDENS the negative box already there
        // adds no parameters at all — code = 0 — so it wins whenever it explains anything, which
        // is exactly the Occam ordering we want and needs no rule to express it.
        // ⚠ THE INCUMBENT GETS TO MOVE TOO. Scoring a carve against a FROZEN layout asks the
        // wrong question: it compares "add a box" with "do nothing", when the real alternative is
        // "let the walls shift". With the offsets frozen, c2's positive box sat 14 cm too wide and
        // grow() answered that misfit with a 1.89 x 2.44 m slab — structure paying a debt the
        // continuous parameters owed. Both sides are refitted before either is scored.
        Layout B = L; refit(B, cloud, p, 3);

        // ── THE NOISE SCALE IS A PARAMETER OF THE MODEL, AND IT IS ESTIMATED, NOT ASSERTED ───
        // Everything above says how sharp a point COULD be. This says how sharp the points
        // actually are given the layout we have — the excess spread the model does not explain.
        // Without it the estimator read a 0.208 m misfit through a 0.022 m likelihood and bought
        // EIGHT carves in a single burst at frame 200, from the least reliable cloud of the whole
        // run, then never revised them: c2 IoU 0.503 with a positive box 14 cm too wide.
        //
        // Robust scale (MAD) of the incumbent's residuals, minus the variance already accounted
        // for. Computed ONCE from the incumbent and held fixed across every candidate in this
        // call — a proposal must not be able to win by inflating the noise it is judged against.
        // This is KISS-ICP's measured 3-sigma threshold as a variance term rather than a cutoff,
        // and it is self-limiting: while the room is badly explained nothing looks surprising, and
        // structure only becomes admissible as the continuous fit earns the precision to see it.
        float over_var = 0.f;
        {
            std::vector<float> ad;
            ad.reserve(pts.size());
            double known = 0.0;
            for (const auto& q : pts)
            { ad.push_back(std::abs(B.sdf(q.p))); known += static_cast<double>(q.sigma_pose) * q.sigma_pose; }
            std::nth_element(ad.begin(), ad.begin() + static_cast<long>(ad.size() / 2), ad.end());
            const float mad = 1.4826f * ad[ad.size() / 2];
            over_var = std::max(0.f, mad * mad - static_cast<float>(known / static_cast<double>(pts.size())));
        }
        std::vector<CloudPoint> spts = pts;
        for (auto& q : spts) q.sigma_pose = std::sqrt(q.sigma_pose * q.sigma_pose + over_var);

        const float base_ll = log_likelihood(B, spts);
        Box best; float best_dL = 0.f; bool have = false; int best_widen = -1;
        for (const auto& g : groups)
        {
            if (g.size() < static_cast<size_t>(p.min_cluster)) continue;
            ++R.proposed;
            Box nb;
            nb.lo = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
            nb.hi = {-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};
            for (const auto& q : g) { nb.lo = nb.lo.cwiseMin(q); nb.hi = nb.hi.cwiseMax(q); }
            nb.positive = false;
            // A column standing AGAINST a wall is bounded by that wall: extend the proposal to the
            // nearest face of the enclosing positive box so it is attached rather than floating.
            // A floating cluster stays floating and becomes a hole, which simple-connectivity
            // forbids — so it is not offered.
            const Box* host = nullptr;
            for (const auto& b : L.boxes)
                if (b.positive and b.contains(0.5f * (nb.lo + nb.hi))) { host = &b; break; }
            if (host == nullptr) continue;
            const float dl = nb.lo.x() - host->lo.x(), dr = host->hi.x() - nb.hi.x();
            const float db = nb.lo.y() - host->lo.y(), dt = host->hi.y() - nb.hi.y();
            const float m = std::min(std::min(dl, dr), std::min(db, dt));
            nb.host = static_cast<std::uint32_t>(host - L.boxes.data());
            if      (m == dl) { nb.lo.x() = host->lo.x(); nb.attach = 0; }
            else if (m == dr) { nb.hi.x() = host->hi.x(); nb.attach = 2; }
            else if (m == db) { nb.lo.y() = host->lo.y(); nb.attach = 1; }
            else              { nb.hi.y() = host->hi.y(); nb.attach = 3; }
            if (not nb.valid()) continue;

            const float sig = std::sqrt(s0sq);
            const float span = std::max(1.f, host->width() + host->height());

            // (a) WIDEN an existing negative box this cluster already touches. No new parameters,
            //     so the only price is re-coding the four offsets it moves: 4 log(span/sigma) is
            //     NOT charged again, because the box was already paid for. Code is zero.
            int widen = -1;
            for (size_t bi = 0; bi < L.boxes.size(); ++bi)
            {
                const Box& b = L.boxes[bi];
                if (b.positive) continue;
                const bool overlap = nb.lo.x() <= b.hi.x() and nb.hi.x() >= b.lo.x()
                                 and nb.lo.y() <= b.hi.y() and nb.hi.y() >= b.lo.y();
                if (overlap) { widen = static_cast<int>(bi); break; }
            }
            if (widen >= 0)
            {
                Layout T = L;
                T.boxes[static_cast<size_t>(widen)].lo = T.boxes[static_cast<size_t>(widen)].lo.cwiseMin(nb.lo);
                T.boxes[static_cast<size_t>(widen)].hi = T.boxes[static_cast<size_t>(widen)].hi.cwiseMax(nb.hi);
                if (simply_connected(T))
                {
                    refit(T, cloud, p, 3);
                    const float dL = log_likelihood(T, spts) - base_ll;
                    if (dL > best_dL)
                    { best_dL = dL; best = T.boxes[static_cast<size_t>(widen)]; have = true; best_widen = widen; }
                }
                continue;   // a cluster that touches an existing box is that box getting bigger
            }

            // (b) a genuinely NEW negative box: four new offsets, each coded at the Laplace/Occam
            //     precision log(span/sigma), plus where it attached. No tuned constant anywhere.
            Layout T = L; T.boxes.push_back(nb);
            if (not simply_connected(T)) continue;
            refit(T, cloud, p, 3);
            const float ll = log_likelihood(T, spts);
            // Three FREE offsets: the attached face is the host's, already paid for. Plus
            // log(4) for which face it attached to.
            const float code = 3.f * std::log(span / sig) + std::log(4.f);
            const float dL = (ll - base_ll) - code;
            if (dL > best_dL) { best_dL = dL; best = nb; have = true; best_widen = -1; }   // BF > 1
        }
        // ── 3b. OUTWARD: a cluster BEYOND a wall is room the region does not cover ───────────
        // An alcove, a door recess, a bay. The proposal is a POSITIVE box whose inner face IS the
        // wall it opens through — one shared offset, exactly as a carve shares the face it is cut
        // into, so it costs three free parameters and not four. Priced by the same Bayes factor:
        // there is no separate rule for outward growth, and no constant that distinguishes it.
        for (const auto& g : groups_out)
        {
            if (g.size() < static_cast<size_t>(p.min_cluster)) continue;
            ++R.proposed;
            Box nb;
            nb.lo = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
            nb.hi = {-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};
            for (const auto& q : g) { nb.lo = nb.lo.cwiseMin(q); nb.hi = nb.hi.cwiseMax(q); }
            nb.positive = true;
            const Eigen::Vector2f ctr = 0.5f * (nb.lo + nb.hi);

            // Which positive box does it lie beyond, and through which of that box's faces?
            const Box* host = nullptr; float bestd = std::numeric_limits<float>::max(); size_t hi_ = 0;
            for (size_t bi = 0; bi < L.boxes.size(); ++bi)
            {
                if (not L.boxes[bi].positive) continue;
                const float dd = sdf_box(L.boxes[bi], ctr);
                if (dd < bestd) { bestd = dd; host = &L.boxes[bi]; hi_ = bi; }
            }
            if (host == nullptr) continue;
            nb.host = static_cast<std::uint32_t>(hi_);
            // The face it opens through is the one it is furthest beyond.
            const float ox = ctr.x() - host->hi.x(), oxl = host->lo.x() - ctr.x();
            const float oy = ctr.y() - host->hi.y(), oyl = host->lo.y() - ctr.y();
            const float m = std::max(std::max(ox, oxl), std::max(oy, oyl));
            if (m <= 0.f) continue;                       // not actually outside this box
            if      (m == ox)  { nb.lo.x() = host->hi.x(); nb.attach = 4; }
            else if (m == oyl) { nb.hi.y() = host->lo.y(); nb.attach = 7; }
            else if (m == oxl) { nb.hi.x() = host->lo.x(); nb.attach = 6; }
            else               { nb.lo.y() = host->hi.y(); nb.attach = 5; }
            if (not nb.valid()) continue;

            Layout T = L; T.boxes.push_back(nb);
            if (not simply_connected(T)) continue;
            refit(T, cloud, p, 3);
            const float sig = std::sqrt(s0sq);
            const float span = std::max(1.f, host->width() + host->height());
            const float code = 3.f * std::log(span / sig) + std::log(4.f);
            const float dL = (log_likelihood(T, spts) - base_ll) - code;
            if (dL > best_dL) { best_dL = dL; best = nb; have = true; best_widen = -1; }
        }

        if (have)
        {
            L = B;                                       // keep the fit the comparison was made on
            if (best_widen >= 0) L.boxes[static_cast<size_t>(best_widen)] = best;
            else
            {
                best.id = static_cast<std::uint64_t>(L.boxes.size() + 1);
                L.boxes.push_back(best);
                L.cov.conservativeResize(static_cast<long>(L.n_offsets()), static_cast<long>(L.n_offsets()));
            }
            R.admitted = 1; R.best_dL = best_dL;
        }
        return R;
    }

    RegisterResult register_scan(const Layout& L, const std::vector<Eigen::Vector2f>& pts,
                                 const Eigen::Vector3f& odom, float sensor_sigma)
    {
        RegisterResult R;
        if (L.empty() or pts.size() < 10) return R;

        auto cost_at = [&](const Eigen::Vector3f& x)
        {
            const float c = std::cos(x.z()), s = std::sin(x.z());
            double e = 0.0;
            for (const auto& q : pts)
            {
                const Eigen::Vector2f m(c * q.x() - s * q.y() + x.x(), s * q.x() + c * q.y() + x.y());
                const float d = L.sdf(m);
                e += static_cast<double>(d) * d;
            }
            return static_cast<float>(e / static_cast<double>(pts.size()));
        };

        // ── KINEMATIC-ICP: the regulariser's weight is READ OFF THE DATA ─────────────────────
        // beta = the map cost at the odometry prediction. When odometry and the scan disagree —
        // which is exactly what a shared pose error looks like — beta is large and the prior
        // dissolves; when they agree it holds. No gate, no threshold, nothing tuned.
        // Translation only: heading is left to the LiDAR, the better channel and the
        // best-observed DOF in a rectangular room.
        R.beta = std::max(cost_at(odom), 1e-6f);
        const float w_prior = 1.f / R.beta;

        Eigen::Vector3f x = odom;
        // ⚠ THE MAP TERM IS WEIGHTED BY HOW WELL THE MAP EXPLAINS THIS SCAN, NOT BY SENSOR NOISE.
        // Kinematic-ICP's beta assumes the reference is always right — true for scan-to-scan,
        // FALSE for a partial layout that has not grown its columns yet. Weighting the map at
        // sensor sigma while it is wrong by 0.2 m lets it out-vote odometry and drag the pose
        // onto surfaces that do not exist: room c4, whose four corner columns are unmodelled,
        // registered to 0.161 m of error and then smeared the cloud so badly that the columns
        // could never be found — the structure and the pose deadlocking on each other.
        // The robust scale of the residuals at the odometry prediction is the map's own statement
        // of how much it can be trusted, and it dissolves as the structure is admitted. Floored
        // at the sensor, never below: a good map is still only as good as the LiDAR.
        float sig2 = std::max(1e-8f, sensor_sigma * sensor_sigma);
        {
            const float c0 = std::cos(odom.z()), s0 = std::sin(odom.z());
            std::vector<float> ad; ad.reserve(pts.size());
            for (const auto& q : pts)
                ad.push_back(std::abs(L.sdf({c0 * q.x() - s0 * q.y() + odom.x(),
                                             s0 * q.x() + c0 * q.y() + odom.y()})));
            std::nth_element(ad.begin(), ad.begin() + static_cast<long>(ad.size() / 2), ad.end());
            const float mad = 1.4826f * ad[ad.size() / 2];
            sig2 = std::max(sig2, mad * mad);
        }
        for (int it = 0; it < 25; ++it)
        {
            Eigen::Matrix3f H = Eigen::Matrix3f::Zero();
            Eigen::Vector3f g = Eigen::Vector3f::Zero();
            const float c = std::cos(x.z()), s = std::sin(x.z());
            double loss = 0.0; long nused = 0;
            for (const auto& q : pts)
            {
                const Eigen::Vector2f rp(c * q.x() - s * q.y(), s * q.x() + c * q.y());
                const Eigen::Vector2f m = rp + x.head<2>();
                // nearest box decides the gradient; for a union that is the active branch of min()
                float d = std::numeric_limits<float>::infinity();
                Eigen::Vector2f gr(0.f, 0.f);
                for (const auto& b : L.boxes)
                {
                    if (not b.positive) continue;
                    const float db = sdf_box(b, m);
                    if (db < d) { d = db; gr = grad_box(b, m); }
                }
                for (const auto& b : L.boxes)
                {
                    if (b.positive) continue;
                    const float db = -sdf_box(b, m);
                    if (db > d) { d = db; gr = -grad_box(b, m); }
                }
                if (not std::isfinite(d)) continue;
                Eigen::Vector3f J;
                J.head<2>() = gr;
                J(2) = gr.dot(Eigen::Vector2f(-rp.y(), rp.x()));   // d/dtheta of R(theta) q
                const float w = 1.f / sig2;
                H.noalias() += w * J * J.transpose();
                g.noalias() += w * d * J;
                loss += static_cast<double>(d) * d;
                ++nused;
            }
            // translation-only prior toward the odometry prediction
            H(0, 0) += w_prior; H(1, 1) += w_prior;
            g(0) += w_prior * (x.x() - odom.x());
            g(1) += w_prior * (x.y() - odom.y());

            const Eigen::Vector3f step = H.ldlt().solve(-g);
            if (not step.allFinite()) break;
            x.head<2>() += step.head<2>();
            x.z() = wrap_pi(x.z() + step.z());
            R.iterations = it + 1;
            // ⚠ SCALE THE COVARIANCE BY HOW BADLY THE SCAN ACTUALLY FIT. The raw Hessian inverse
            // answers "how sharp would this pose be if the map were exact and the residuals were
            // pure sensor noise" — and on room c4, whose four corner columns the layout does not
            // model yet, it answered 2 mm for a pose wrong by 165 mm. A pose is never sharper
            // than the map it was registered against, and the reduced chi-square is the map's own
            // report of that: sigma^2_hat = cost/(N-3), textbook, with no constant to choose.
            // The inflated sigma then flows into the fused cloud's weights, so a scan matched
            // against a wrong room contributes softly instead of confidently poisoning it.
            if (step.head<2>().norm() < 1e-5f and std::abs(step.z()) < 1e-6f or it == 24)
            {
                const double dof = std::max(1.0, static_cast<double>(nused) - 3.0);
                const float chi2r = static_cast<float>(std::max(1.0, loss / (sig2 * dof)));
                R.cov = H.inverse() * chi2r;
                if (step.head<2>().norm() < 1e-5f and std::abs(step.z()) < 1e-6f) break;
            }
        }
        R.pose = x;
        R.cost = cost_at(x);
        R.ok = R.pose.allFinite() and R.cov.allFinite();
        return R;
    }
}   // namespace rc::boxes
