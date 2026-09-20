#include "room_boxes_channel.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <cstdlib>
#include <limits>
#include <locale>
#include <set>

namespace rc::boxch
{
    static constexpr float kPiF = 3.14159265358979323846f;
    void Channel::observe(const std::vector<Eigen::Vector3f>& pts_robot,
                          const Eigen::Vector3f& pose, const Eigen::Matrix3f& cov,
                          const std::vector<float>& seg_phi, const std::vector<float>& seg_len)
    {
        if (not p_.enabled) return;
        ++frames_;

        // ── THE BAND. Not a tuning knob: a 2-D layout built from 3-D returns must exclude the
        // floor and the ceiling, and a ceiling return inside a wall SDF is a documented way to
        // stop a room stabilising while every pose diagnostic still looks healthy.
        std::vector<Eigen::Vector2f> band;
        band.reserve(pts_robot.size());
        for (const auto& q : pts_robot)
            if (q.z() >= p_.z_min and q.z() <= p_.z_max) band.emplace_back(q.x(), q.y());
        if (band.size() < 20) return;

        const float c = std::cos(pose.z()), s = std::sin(pose.z());

        // ⚠ NO init_from_scan() HERE. It seeds the box AND votes on the gauge from ONE scan,
        // using consecutive point differences — unusable on this input (see observe()'s header).
        // The layout is created in step() instead, once the segment vote has a frame to build in,
        // and from the FUSED cloud rather than a single view. Nothing is lost by waiting: returns
        // are still being accumulated, in the agent's own frame, from the first scan onwards.

        // ── FOLD THE SCAN IN, each return carrying the 1-sigma uncertainty of WHERE IT IS:
        // pose translation plus the lever arm of the heading error at that range. First-order
        // propagation of cov through p = t + R(theta) q. A point is no sharper than the pose that
        // placed it, and saying otherwise is what let drift be spent on structure.
        float smax = 0.f;
        for (const auto& q : band)
        {
            const float sg = point_sigma(cov, c, s, q);
            if (not std::isfinite(sg)) continue;
            // ⚠ The voxel map is kept in the AGENT'S MAP FRAME, not the layout's. The gauge is
            // still being estimated, and evidence stored in a frame that is itself moving cannot
            // be re-read once that frame corrects itself. fuse() applies the current yaw on the
            // way out instead, so every past frame benefits from every later vote.
            const Eigen::Vector2f g(c * q.x() - s * q.y() + pose.x(), s * q.x() + c * q.y() + pose.y());
            const float s0 = std::sqrt(sg * sg + p_.sensor_sigma * p_.sensor_sigma);
            const double w = 1.0 / (static_cast<double>(s0) * s0);
            auto& v = vmap_[{static_cast<int>(std::floor(g.x() / p_.cell)),
                             static_cast<int>(std::floor(g.y() / p_.cell))}];
            v.acc += w * g.cast<double>();
            v.w   += w;
            v.smin = std::min(v.smin, sg);
            smax = std::max(smax, sg);
        }
        last_sigma_ = smax;

        // ── A PROVISIONAL LAYOUT FROM THE VERY FIRST SCAN ───────────────────────────────────
        // ⚠ WITHOUT THIS THE OPENING OF EVERY RUN IS PURE DEAD RECKONING. Registration needs a
        // layout to register against, and the layout used to wait for the free-space cover — so
        // the first stretch was uncorrected odometry, and on a 36 m tour of a plain rectangle that
        // wrote the SAME TWO WALLS at several different places. Measured: the fused cloud spanned
        // 8.13 m in a 6.00 m room, with a ghost concentration of returns at x = -1.25 where no
        // wall exists and returns reaching x = +5.00 two metres past the right wall. Everything
        // downstream was then a faithful description of corrupted evidence — the layout fitted it
        // to 0.033 m, which is why no amount of re-projection or adoption judging could repair it.
        //
        // The seed does not need to be right, only PRESENT: it is the interval hull of one scan,
        // and the free-space cover replaces it through the same adoption judge as soon as there is
        // anything better (measured: hull costs 154053 nats, first real cover 10028 — adopted).
        // Its whole job is to give frame two something to register against.
        if (L_.empty() and yaw_votes_ > 0)
        {
            const double a4 = std::atan2(yaw4_sin_, yaw4_cos_);
            const float gy = static_cast<float>(a4 / 4.0);
            const float cg = std::cos(-gy), sg2 = std::sin(-gy);
            std::vector<float> xs, ys;
            xs.reserve(band.size()); ys.reserve(band.size());
            for (const auto& q : band)
            {
                const Eigen::Vector2f g(c * q.x() - s * q.y() + pose.x(), s * q.x() + c * q.y() + pose.y());
                xs.push_back(cg * g.x() - sg2 * g.y());
                ys.push_back(sg2 * g.x() + cg * g.y());
            }
            const auto quant = [](std::vector<float>& v, float f)
            {
                const size_t k = std::clamp<size_t>(static_cast<size_t>(f * static_cast<float>(v.size() - 1)),
                                                    0, v.size() - 1);
                std::nth_element(v.begin(), v.begin() + static_cast<long>(k), v.end());
                return v[k];
            };
            std::vector<float> xs2 = xs, ys2 = ys;
            rc::boxes::Box b;
            b.lo = {quant(xs, 0.01f), quant(ys, 0.01f)};
            b.hi = {quant(xs2, 0.99f), quant(ys2, 0.99f)};
            if (b.valid() and b.width() > 0.5f and b.height() > 0.5f)
            {
                yaw_ = gy;
                L_.boxes.push_back(b);
                L_.cov = Eigen::MatrixXf::Identity(4, 4) * (p_.sigma_flat * p_.sigma_flat);
            }
        }

        // Keep a bounded, subsampled record of the evidence in the frame it was measured in.
        if (frames_ - last_key_f_ >= 25 and keys_.size() < 200)
        {
            KeyFrame kf;
            kf.pose = pose; kf.sigma = smax; kf.cov = cov;
            kf.pts.reserve(band.size() / 3 + 1);
            for (size_t i = 0; i < band.size(); i += 3) kf.pts.push_back(band[i]);
            keys_.push_back(std::move(kf));
            last_key_f_ = frames_;
        }

        // ── MARK THE FREE SPACE EACH BEAM SWEPT ─────────────────────────────────────────────
        // From the sensor to just short of the return: those cells were seen through, so they are
        // room. The last cell is NOT marked — that is where the surface is.
        {
            const Eigen::Vector2f o = pose.head<2>();
            auto& fo = free_;
            const float step = 0.5f * p_.cell;
            for (const auto& q : band)
            {
                const Eigen::Vector2f g(c * q.x() - s * q.y() + pose.x(), s * q.x() + c * q.y() + pose.y());
                const Eigen::Vector2f d = g - o;
                const float len = d.norm();
                if (len < 1e-3f) continue;
                const Eigen::Vector2f u = d / len;
                for (float t = 0.f; t < len - p_.cell; t += step)
                {
                    const Eigen::Vector2f m = o + u * t;
                    ++fo[{static_cast<int>(std::floor(m.x() / p_.cell)),
                          static_cast<int>(std::floor(m.y() / p_.cell))}];
                }
            }
            last_cell_ = {static_cast<int>(std::floor(pose.x() / p_.cell)),
                          static_cast<int>(std::floor(pose.y() / p_.cell))};
            have_cell_ = true;
            ++free_[last_cell_];             // the robot is where it is
        }

        // ── THE GAUGE, VOTED FROM THE AGENT'S OWN WALL SEGMENTS ─────────────────────────────
        // A fitted segment's normal is an orientation estimate that does not care what order the
        // points arrived in, and its extent is the natural weight: a 4-metre wall should outvote a
        // 30-centimetre fragment. Quadrupled angle, so the four indistinguishable quarter-turns
        // coincide and average instead of cancelling.
        for (size_t k = 0; k < seg_phi.size() and k < seg_len.size(); ++k)
        {
            const float w = seg_len[k];
            if (not (w > 0.f) or not std::isfinite(seg_phi[k])) continue;
            const double a4 = 4.0 * (static_cast<double>(seg_phi[k]) + static_cast<double>(pose.z()));
            yaw4_cos_ += static_cast<double>(w) * std::cos(a4);
            yaw4_sin_ += static_cast<double>(w) * std::sin(a4);
            ++yaw_votes_;
        }
    }

    void Channel::reanchor(const Eigen::Vector2f& c, float rot)
    {
        if (not p_.enabled) return;
        const float cr = std::cos(-rot), sr = std::sin(-rot);
        // The voxel map is keyed by position, so it has to be rebuilt, not edited in place.
        std::map<std::pair<int, int>, Vox> nv;
        for (const auto& [k, v] : vmap_)
        {
            if (v.w <= 0.0) continue;
            const Eigen::Vector2f m(static_cast<float>(v.acc.x() / v.w) - c.x(),
                                    static_cast<float>(v.acc.y() / v.w) - c.y());
            const Eigen::Vector2f q(cr * m.x() - sr * m.y(), sr * m.x() + cr * m.y());
            auto& t = nv[{static_cast<int>(std::floor(q.x() / p_.cell)),
                          static_cast<int>(std::floor(q.y() / p_.cell))}];
            t.acc += v.w * q.cast<double>(); t.w += v.w; t.smin = std::min(t.smin, v.smin);
        }
        vmap_.swap(nv);

        // ⚠ AND THE BOXES MOVE TOO. The layout frame shares the map frame's ORIGIN — fuse() only
        // rotates — so when the map origin moves, the layout origin moves with it. A corner at
        // layout coords p has map coords R(yaw)p, and after map' = R(-rot)(map - c) with
        // yaw' = yaw - rot:
        //     p' = R(-yaw')R(-rot)(R(yaw)p - c) = p - R(-yaw)c
        // a PURE TRANSLATION, which is what keeps the boxes axis-aligned through a gauge change.
        // The first version of this rotated yaw_ and moved the voxels but left the boxes behind by
        // |c| — and the cloud still held two offset copies of the room: 33.8% of cells outside,
        // 32.8% inside, against 32.28%/31.89% measured live. Moving three of the four things that
        // live in a frame is not a re-anchor.
        const float cy = std::cos(-yaw_), sy = std::sin(-yaw_);      // OLD yaw, before the update
        const Eigen::Vector2f t(cy * c.x() - sy * c.y(), sy * c.x() + cy * c.y());
        for (auto& b : L_.boxes) { b.lo -= t; b.hi -= t; }
        yaw_ = yaw_ - rot;
        // The gauge's accumulated evidence is a direction, and directions rotate. Turning the
        // running quadrupled-angle sum by -4*rot keeps every past vote valid instead of throwing
        // the session's evidence away and re-converging from the next frame.
        const double a = -4.0 * static_cast<double>(rot);
        const double ca = std::cos(a), sa = std::sin(a);
        const double x = yaw4_cos_, y = yaw4_sin_;
        yaw4_cos_ = ca * x - sa * y;
        yaw4_sin_ = sa * x + ca * y;
    }

    rc::boxes::Box Channel::grow_free_box(const std::pair<int, int>& seed) const
    {
        const auto is_free = [&](int x, int y)
        {   // one sweep IS a measurement; requiring two deleted the evidence nearest the walls
            const auto it = free_.find({x, y});
            return it != free_.end() and it->second >= 1;
        };
        int x0 = seed.first, x1 = seed.first, y0 = seed.second, y1 = seed.second;
        bool moved = true;
        while (moved)
        {
            moved = false;
            // Each side advances only if the ENTIRE next line is free — the box stays a box, and
            // the stopping rule is "the data says no", not a tuned size.
            bool ok = true;
            for (int x = x0; x <= x1 and ok; ++x) ok = is_free(x, y1 + 1);
            if (ok) { ++y1; moved = true; }
            ok = true;
            for (int x = x0; x <= x1 and ok; ++x) ok = is_free(x, y0 - 1);
            if (ok) { --y0; moved = true; }
            ok = true;
            for (int y = y0; y <= y1 and ok; ++y) ok = is_free(x1 + 1, y);
            if (ok) { ++x1; moved = true; }
            ok = true;
            for (int y = y0; y <= y1 and ok; ++y) ok = is_free(x0 - 1, y);
            if (ok) { --x0; moved = true; }
        }
        rc::boxes::Box b;
        b.lo = {static_cast<float>(x0) * p_.cell, static_cast<float>(y0) * p_.cell};
        b.hi = {static_cast<float>(x1 + 1) * p_.cell, static_cast<float>(y1 + 1) * p_.cell};
        return b;
    }

    int Channel::simplify()
    {
        if (L_.boxes.size() < 2 or cloud_.empty()) return 0;
        rc::boxes::GrowParams gp;
        gp.sensor_sigma = p_.sensor_sigma; gp.sigma_flat = p_.sigma_flat;
        gp.cell = p_.cell; gp.min_cluster = p_.min_cluster;

        // The free space, in layout-frame cells, so the cost can see an empty claim.
        std::set<std::pair<int, int>> fl;
        {
            const float cy = std::cos(-yaw_), sy = std::sin(-yaw_);
            for (const auto& [k, n] : free_)
            {
                if (n < 1) continue;
                const Eigen::Vector2f m((static_cast<float>(k.first) + 0.5f) * p_.cell,
                                        (static_cast<float>(k.second) + 0.5f) * p_.cell);
                const Eigen::Vector2f q(cy * m.x() - sy * m.y(), sy * m.x() + cy * m.y());
                fl.insert({static_cast<int>(std::floor(q.x() / p_.cell)),
                           static_cast<int>(std::floor(q.y() / p_.cell))});
            }
        }
        int removed = 0;
        for (int round = 0; round < 40; ++round)
        {
            const float base = rc::boxes::mdl_cost(L_, cloud_, gp, &fl);
            float best = base; rc::boxes::Layout bestL; bool have = false;

            // (a) DELETE a box outright.
            for (size_t i = 0; i < L_.boxes.size(); ++i)
            {
                if (L_.boxes.size() < 2) break;
                rc::boxes::Layout T = L_;
                T.boxes.erase(T.boxes.begin() + static_cast<long>(i));
                if (T.polygon().size() < 4) continue;
                rc::boxes::refit(T, cloud_, gp, 3);
                const float cst = rc::boxes::mdl_cost(T, cloud_, gp, &fl);
                if (cst < best) { best = cst; bestL = T; have = true; }
            }
            // (b) MERGE a pair into their bounding box. Unlike the free-space merge this is
            //     PRICED rather than forbidden: swallowing a little non-free area is allowed when
            //     the description it saves is worth more than the returns it misplaces.
            for (size_t i = 0; i < L_.boxes.size(); ++i)
                for (size_t j = i + 1; j < L_.boxes.size(); ++j)
                {
                    if (not L_.boxes[i].positive or not L_.boxes[j].positive) continue;
                    rc::boxes::Layout T = L_;
                    T.boxes[i].lo = L_.boxes[i].lo.cwiseMin(L_.boxes[j].lo);
                    T.boxes[i].hi = L_.boxes[i].hi.cwiseMax(L_.boxes[j].hi);
                    T.boxes.erase(T.boxes.begin() + static_cast<long>(j));
                    if (T.polygon().size() < 4) continue;
                    rc::boxes::refit(T, cloud_, gp, 3);
                    const float cst = rc::boxes::mdl_cost(T, cloud_, gp, &fl);
                    if (cst < best) { best = cst; bestL = T; have = true; }
                }

            if (not have) break;
            L_ = bestL;
            ++removed;
        }
        n_removed_ += removed;
        return removed;
    }

    bool Channel::traversable(const Eigen::Vector2f& m) const
    {
        const std::pair<int, int> c{static_cast<int>(std::floor(m.x() / p_.cell)),
                                    static_cast<int>(std::floor(m.y() / p_.cell))};
        const auto it = free_.find(c);
        if (it == free_.end() or it->second < 1) return false;     // never swept: not floor
        const int r = std::max(0, static_cast<int>(std::ceil(p_.body_radius / p_.cell)));
        for (int dy = -r; dy <= r; ++dy)
            for (int dx = -r; dx <= r; ++dx)
            {
                if (dx * dx + dy * dy > r * r) continue;            // a disc, as in plan_path
                const auto v = vmap_.find({c.first + dx, c.second + dy});
                if (v != vmap_.end() and v->second.w > 0.0) return false;
            }
        return true;
    }

    std::vector<Eigen::Vector2f> Channel::plan_path(const Eigen::Vector2f& from_map, float horizon_m) const
    {
        last_gain_ = 0;
        std::vector<Eigen::Vector2f> out;
        if (free_.empty()) return out;

        // ⚠ DENSE GRID FOR THE DURATION OF THE PLAN. The free map is a std::map, and the gain
        // term probes ~1600 neighbours per frontier cell: at O(log n) against 20000 cells that is
        // hundreds of millions of lookups per plan, and the planner did not finish inside a
        // 200-second run. The same data as a flat array of bytes makes every probe O(1); the map
        // stays the authority, this is a scratch copy.
        int lox = std::numeric_limits<int>::max(), loy = lox;
        int hix = std::numeric_limits<int>::min(), hiy = hix;
        for (const auto& [k, n] : free_)
        {
            if (n < 1) continue;
            lox = std::min(lox, k.first);  hix = std::max(hix, k.first);
            loy = std::min(loy, k.second); hiy = std::max(hiy, k.second);
        }
        if (lox > hix) return out;
        const int W = hix - lox + 3, H = hiy - loy + 3;     // one cell of unknown margin all round
        // ⚠ THREE STATES, NOT TWO. FREE (a beam went through), OCCUPIED (a return landed here)
        // and UNKNOWN (neither). Treating "not free" as "unknown" makes every WALL look like
        // unexplored space: the frontier gain then peaks against the walls, the planner sends the
        // robot at them, and — with no collision in the bench — it drives straight through and
        // explores the outdoors. Measured: at frame 700 the robot was at (3.49, 7.77) in a room
        // roughly 6 x 4 m about the origin, with the gain still climbing because outside the room
        // everything is 'unknown' for ever.
        // The occupied set is free: it is the returns this channel already fuses.
        std::vector<char> gridf(static_cast<size_t>(W) * static_cast<size_t>(H), 0);
        std::vector<char> grido(static_cast<size_t>(W) * static_cast<size_t>(H), 0);
        const auto idx = [&](int x, int y) { return static_cast<size_t>((y - loy + 1) * W + (x - lox + 1)); };
        for (const auto& [k, n] : free_)
            if (n >= 1 and k.first >= lox and k.first <= hix and k.second >= loy and k.second <= hiy)
                gridf[idx(k.first, k.second)] = 1;
        for (const auto& [k, v] : vmap_)
        {
            if (v.w <= 0.0) continue;
            const Eigen::Vector2f m(static_cast<float>(v.acc.x() / v.w), static_cast<float>(v.acc.y() / v.w));
            const int ox = static_cast<int>(std::floor(m.x() / p_.cell)), oy = static_cast<int>(std::floor(m.y() / p_.cell));
            if (ox < lox or ox > hix or oy < loy or oy > hiy) continue;
            grido[idx(ox, oy)] = 1;
        }
        const auto is_occ = [&](int x, int y)
        {
            if (x < lox - 1 or x > hix + 1 or y < loy - 1 or y > hiy + 1) return false;
            return grido[idx(x, y)] != 0;
        };
        // UNKNOWN is the only thing worth travelling to see.
        const auto is_unknown = [&](int x, int y)
        {
            if (x < lox - 1 or x > hix + 1 or y < loy - 1 or y > hiy + 1) return false;  // beyond the map: not a target
            return gridf[idx(x, y)] == 0 and grido[idx(x, y)] == 0;
        };
        const auto is_free = [&](const std::pair<int, int>& k)
        {
            if (k.first < lox - 1 or k.first > hix + 1 or k.second < loy - 1 or k.second > hiy + 1) return false;
            return gridf[idx(k.first, k.second)] != 0;
        };

        // ── A CELL BEING EMPTY IS NOT THE SAME AS THE ROBOT FITTING IN IT ───────────────────
        // The planner used to route over bare free cells and hand the result to a pursuit
        // controller, so it would happily thread a body 0.46 m wide through a 0.12 m fin gap. The
        // bench has NO collision model, so nothing ever objected: measured on the apartamento hall,
        // the executed path crossed a wall on 19 of 217 legs. Plan in configuration space instead —
        // free space eroded by the robot's own radius — and a gap it cannot fit through simply
        // stops being a corridor.
        // Erosion is against KNOWN OCCUPANCY ONLY. Unknown space is not known to be an obstacle,
        // and a frontier cell is adjacent to unknown by definition, so eroding against unknown
        // would forbid approaching every frontier. This also RETIRES the ad-hoc "no occupied cell
        // within 2 cells" margin the frontier test used to carry: 2 cells was a stand-in for this.
        const int brad = std::max(0, static_cast<int>(std::ceil(p_.body_radius / p_.cell)));
        std::vector<char> blocked(gridf.size(), 0);
        for (int y = loy - 1; y <= hiy + 1; ++y)
            for (int x = lox - 1; x <= hix + 1; ++x)
            {
                if (grido[idx(x, y)] == 0) continue;
                for (int dy = -brad; dy <= brad; ++dy)
                    for (int dx = -brad; dx <= brad; ++dx)
                    {
                        if (dx * dx + dy * dy > brad * brad) continue;      // a disc, not a square
                        const int nx = x + dx, ny = y + dy;
                        if (nx < lox - 1 or nx > hix + 1 or ny < loy - 1 or ny > hiy + 1) continue;
                        blocked[idx(nx, ny)] = 1;
                    }
            }
        const auto is_trav = [&](const std::pair<int, int>& k)
        {
            if (not is_free(k)) return false;
            return blocked[idx(k.first, k.second)] == 0;
        };

        // ⚠ UNKNOWN SPACE OUTSIDE THE BELIEVED ROOM IS NOT AN EXPLORE TARGET. A 1-cell wall does
        // not seal the occupied grid: at grazing incidence 2 cm of range noise leaves gaps between
        // returns, and the free marking stops just short of the surface, so the 48 rays LEAK
        // THROUGH the walls and score the outdoors — which is unbounded. Measured on a plain
        // rectangle: gain RISING 930 -> 1212 -> 1304 over frames 500-800, so the frontier was
        // never exhausted, Explore never ended, and the refine phase, the frozen cover and the
        // Done rule have never run in any run of this planner.
        // The layout is the robot's own statement of where the room IS, so an unknown cell with
        // sdf >= 0 is outside it: reachable only through a face the layout does not support, which
        // is a REFINE question (is that face where I think it is?), not an Explore one. No
        // threshold — the sign of the SDF is the model's own boundary.
        const float cyl = std::cos(-yaw_), syl = std::sin(-yaw_);
        std::vector<char> gridi(gridf.size(), 1);
        if (not L_.empty())
            for (int y = loy - 1; y <= hiy + 1; ++y)
                for (int x = lox - 1; x <= hix + 1; ++x)
                {
                    const Eigen::Vector2f m((static_cast<float>(x) + 0.5f) * p_.cell,
                                            (static_cast<float>(y) + 0.5f) * p_.cell);
                    const Eigen::Vector2f ml(cyl * m.x() - syl * m.y(), syl * m.x() + cyl * m.y());
                    gridi[idx(x, y)] = (L_.sdf(ml) < 0.f) ? 1 : 0;
                }
        const auto is_inside = [&](int x, int y)
        {
            if (x < lox - 1 or x > hix + 1 or y < loy - 1 or y > hiy + 1) return false;
            return gridi[idx(x, y)] != 0;
        };

        const std::pair<int, int> start{static_cast<int>(std::floor(from_map.x() / p_.cell)),
                                        static_cast<int>(std::floor(from_map.y() / p_.cell))};
        if (not is_free(start)) return out;

        // Dijkstra-on-a-grid is unnecessary here: every step costs the same, so breadth-first over
        // free cells gives exact shortest paths and the parent map reconstructs them.
        std::vector<int> dist_a(gridf.size(), -1);
        std::vector<int> par_a(gridf.size(), -1);
        std::map<std::pair<int, int>, std::pair<int, int>> parent;
        std::map<std::pair<int, int>, int> dist;
        // ⚠ THE ROBOT IS WHERE IT IS. Seed the start unconditionally — it may already be inside the
        // eroded margin (it was driven there before the erosion existed, or a wall was discovered
        // beside it), and refusing to plan from an illegal cell would simply freeze it there.
        // If the eroded grid strands it, fall back to bare free space for this one plan so it can
        // extract itself, and say so in the phase log rather than standing still silently.
        bool eroded = true;
        const auto passable = [&](const std::pair<int, int>& k)
        { return eroded ? is_trav(k) : is_free(k); };
        std::deque<std::pair<int, int>> q{start};
        dist[start] = 0; dist_a[idx(start.first, start.second)] = 0;
        while (not q.empty())
        {
            const auto cur = q.front(); q.pop_front();
            const int d = dist[cur];
            const std::pair<int, int> nb[4] = {{cur.first + 1, cur.second}, {cur.first - 1, cur.second},
                                               {cur.first, cur.second + 1}, {cur.first, cur.second - 1}};
            for (const auto& n : nb)
            {
                if (not passable(n) or dist_a[idx(n.first, n.second)] >= 0) continue;
                dist_a[idx(n.first, n.second)] = d + 1;
                par_a[idx(n.first, n.second)] = static_cast<int>(idx(cur.first, cur.second));
                dist[n] = d + 1; parent[n] = cur; q.push_back(n);
            }
        }

        // The erosion can strand a robot that is already inside the margin: one cell reached means
        // every neighbour is blocked. Retry once on bare free space so it can drive out.
        if (dist.size() <= 1 and eroded)
        {
            eroded = false;
            dist.clear(); parent.clear();
            std::fill(dist_a.begin(), dist_a.end(), -1);
            std::fill(par_a.begin(), par_a.end(), -1);
            std::deque<std::pair<int, int>> q2{start};
            dist[start] = 0; dist_a[idx(start.first, start.second)] = 0;
            while (not q2.empty())
            {
                const auto cur = q2.front(); q2.pop_front();
                const int d = dist[cur];
                const std::pair<int, int> nb[4] = {{cur.first + 1, cur.second}, {cur.first - 1, cur.second},
                                                   {cur.first, cur.second + 1}, {cur.first, cur.second - 1}};
                for (const auto& n : nb)
                {
                    if (not is_free(n) or dist_a[idx(n.first, n.second)] >= 0) continue;
                    dist_a[idx(n.first, n.second)] = d + 1;
                    par_a[idx(n.first, n.second)] = static_cast<int>(idx(cur.first, cur.second));
                    dist[n] = d + 1; parent[n] = cur; q2.push_back(n);
                }
            }
        }

        // A frontier is a reachable free cell with an unknown neighbour. Its gain is the unknown
        // area within the sensor horizon — what standing there would actually reveal.
        const int R = std::max(1, static_cast<int>(horizon_m / p_.cell));
        const int R2 = R * R;
        (void)R2;

        // ── CLUSTER THE FRONTIER FIRST ──────────────────────────────────────────────────────
        // Ray-casting the visible gain costs 48 bearings x 80 steps, and there are THOUSANDS of
        // frontier cells: evaluating each one took the run past 400 seconds without finishing.
        // Adjacent frontier cells are the same opening anyway — ranking them individually is both
        // wasteful and worse, because it picks an arbitrary cell of a large opening instead of its
        // middle. Group them into connected clusters and evaluate one representative each, which
        // is what frontier-based exploration has always done.
        std::set<std::pair<int, int>> fr;
        for (const auto& [k, d] : dist)
        {
            const std::pair<int, int> nb4[4] = {{k.first + 1, k.second}, {k.first - 1, k.second},
                                                {k.first, k.second + 1}, {k.first, k.second - 1}};
            bool isf = false;
            for (const auto& n : nb4)
                if (is_unknown(n.first, n.second) and is_inside(n.first, n.second)) { isf = true; break; }
            if (not isf) continue;
            // the body-radius erosion has replaced the old hand-picked "no occupied cell within
            // 2 cells" margin; a cell the robot fits in is a cell it may stand in
            if (passable(k)) fr.insert(k);
        }
        std::vector<std::pair<int, int>> reps;
        {
            std::set<std::pair<int, int>> seen;
            for (const auto& s : fr)
            {
                if (seen.count(s)) continue;
                std::vector<std::pair<int, int>> stack{s}, cluster;
                seen.insert(s);
                while (not stack.empty())
                {
                    const auto cur = stack.back(); stack.pop_back();
                    cluster.push_back(cur);
                    for (int dx = -1; dx <= 1; ++dx)
                        for (int dy = -1; dy <= 1; ++dy)
                        {
                            const std::pair<int, int> n{cur.first + dx, cur.second + dy};
                            if (fr.count(n) and not seen.count(n)) { seen.insert(n); stack.push_back(n); }
                        }
                }
                // the cluster's middle, snapped back to a real frontier cell
                double mx = 0, my = 0;
                for (const auto& q : cluster) { mx += q.first; my += q.second; }
                mx /= static_cast<double>(cluster.size()); my /= static_cast<double>(cluster.size());
                std::pair<int, int> bestc = cluster.front(); double bd = 1e18;
                for (const auto& q : cluster)
                {
                    const double dd = (q.first - mx) * (q.first - mx) + (q.second - my) * (q.second - my);
                    if (dd < bd) { bd = dd; bestc = q; }
                }
                reps.push_back(bestc);
            }
        }

        std::pair<int, int> best{0, 0}; double best_score = 0.0; bool have = false;
        for (const auto& k : reps)
        {
            const int d = dist_a[idx(k.first, k.second)];
            if (d < 0) continue;
            // ⚠ COUNT WHAT A SCAN WOULD ACTUALLY SEE, NOT WHAT IS NEARBY. Counting every unknown
            // cell in a disc includes the INSIDES OF WALLS — sealed behind occupied cells, never
            // observable from anywhere. That leaves a permanent floor of unreachable reward: the
            // gain plateaued at ~227 cells and never fell to zero, so the frontier was never
            // "exhausted", Explore never ended, and the refine and consolidation phases never ran
            // at all. The robot kept exploring for information it could not obtain.
            // A ray stops at the first occupied cell, exactly as a beam does.
            int unknown = 0;
            const int NB = 48;                      // bearings; a coarse scan is enough to rank
            for (int a = 0; a < NB; ++a)
            {
                const float th = 2.f * kPiF * static_cast<float>(a) / static_cast<float>(NB);
                const float ux = std::cos(th), uy = std::sin(th);
                for (int s = 1; s <= R; ++s)
                {
                    const int cx = k.first  + static_cast<int>(std::lround(ux * static_cast<float>(s)));
                    const int cy = k.second + static_cast<int>(std::lround(uy * static_cast<float>(s)));
                    if (is_occ(cx, cy)) break;      // the beam stops here
                    if (is_unknown(cx, cy) and is_inside(cx, cy)) ++unknown;
                }
            }
            const double score = static_cast<double>(unknown)
                               / (1.0 + static_cast<double>(d) * p_.cell);
            if (score > best_score) { best_score = score; best = k; have = true; last_gain_ = unknown; phase_ = Phase::Explore; }
        }
        // ── PHASE 2: REFINE THE WORST-KNOWN WALL ────────────────────────────────────────────
        // No frontier means coverage is finished, NOT that the layout is finished. Find the face
        // whose offset the data constrains least, then stand square to it at a workable range.
        if (not have and not L_.empty())
        {
            phase_ = Phase::Refine;
            // ── THE WORST FACE IS THE ONE WITH THE WIDEST POSTERIOR, NOT THE LEAST FLOOR ───────
            // ⚠ This block used to sum cos^2(incidence)/range^2 over the free cells the robot
            // could STAND IN, and call that the face's information. That is the available VANTAGE,
            // a property of the floor plan: the "worst face" came out as the face with the least
            // open space in front of it, standing there could not change the number, and Refine
            // could therefore never converge — it re-elected the same face for ever.
            // refit() now KEEPS the posterior it always computed (the normal equations are
            // diagonal, so var(offset) = 1/H and it was being thrown away), so the estimand is
            // available directly: read it. The face to go and look at is the one whose OFFSET the
            // data constrains least, which is the definition of the quantity Refine drives down.
            struct FaceInfo { Eigen::Vector2f mid{0.f,0.f}, nrm{0.f,0.f}; double var = 0.0; };
            std::vector<FaceInfo> faces;
            // An offset with no posterior entry is UNKNOWN, not well known: score it at the room's
            // own size, the weakest proper statement there is (and what refit's own prior uses).
            double span = 0.0;
            for (const auto& b : L_.boxes) span = std::max(span, static_cast<double>(b.width() + b.height()));
            const double var_unknown = std::max(1e-6, span * span);
            const bool have_cov = L_.cov.rows() == static_cast<long>(L_.n_offsets())
                              and L_.cov.cols() == static_cast<long>(L_.n_offsets());
            const auto var_of = [&](size_t k)
            {
                if (not have_cov) return var_unknown;
                const float v = L_.cov(static_cast<long>(k), static_cast<long>(k));
                if (not std::isfinite(v) or v <= 0.f) return var_unknown;
                return static_cast<double>(v);
            };
            // ── THE CANDIDATES ARE THE OFFSETS THE EVIDENCE ACTUALLY LANDS ON ──────────────────
            // ⚠ A BOX FACE IS NOT A WALL, AND AN OFFSET IS NOT ALWAYS A PARAMETER.
            //   · Where two positive boxes abut, the shared face is an INTERNAL SEAM, strictly
            //     inside the union: no return can land on it.
            //   · Where two boxes share a wall, the two coincident offsets describe ONE surface;
            //     active_face() gives every return on it to one of them, so the other is redundant
            //     BY CONSTRUCTION.
            // Both cases leave H = 0, so refit honestly hands the offset its prior — sigma = the
            // room's own span — and a posterior-ranked Refine elects that phantom on every replan
            // and drives at a surface it can never improve. MEASURED before this: room idx 1 sat
            // in `refine` to the frame cap with wsig = 9.53 m (= span) on three of twelve offsets,
            // one of which was the coincident copy of the room's left wall — a wall known to 3 mm.
            // So do not enumerate box faces at all. Walk the region BOUNDARY, ask the estimator's
            // own attribution rule which offset each patch of it belongs to, and let the buckets
            // that come back be the candidates. A seam contributes no boundary samples and a
            // redundant offset is never attributed, so neither can be elected; every candidate is
            // a patch of real wall together with the parameter a look at it would sharpen.
            struct Bucket { Eigen::Vector2f acc{0.f, 0.f}; Eigen::Vector2f nrm{0.f, 0.f}; int n = 0; };
            std::map<int, Bucket> buckets;
            for (size_t bi = 0; bi < L_.boxes.size(); ++bi)
            {
                const auto& b = L_.boxes[bi];
                if (not b.positive) continue;
                const Eigen::Vector2f corner[4] = {{b.lo.x(), b.lo.y()}, {b.hi.x(), b.lo.y()},
                                                   {b.hi.x(), b.hi.y()}, {b.lo.x(), b.hi.y()}};
                const Eigen::Vector2f nrm[4] = {{0.f,-1.f}, {1.f, 0.f}, {0.f, 1.f}, {-1.f, 0.f}};
                for (int e = 0; e < 4; ++e)
                {
                    const Eigen::Vector2f a = corner[e], z = corner[(e + 1) % 4];
                    const int ns = std::max(2, static_cast<int>((z - a).norm() / p_.cell) + 1);
                    for (int i = 0; i < ns; ++i)
                    {
                        const float t = static_cast<float>(i) / static_cast<float>(ns - 1);
                        const Eigen::Vector2f s = a + t * (z - a);
                        if (L_.inside(s + p_.cell * nrm[e])) continue;   // seam: the region goes on
                        const int off = rc::boxes::active_face(L_, s + 0.5f * p_.cell * nrm[e]);
                        if (off < 0 or static_cast<size_t>(off) >= L_.n_offsets()) continue;
                        auto& bk = buckets[off];
                        bk.acc += s; bk.nrm = nrm[e]; ++bk.n;
                    }
                }
            }
            // ── AN OFFSET WITH NO SENSITIVITY CANNOT BE LEARNT FROM ANYWHERE ───────────────────
            // ⚠ ATTRIBUTION IS NOT ENOUGH, AND THIS IS THE ONE THAT COST THE MOST TO FIND.
            // Room idx 1, offset 0 = box 0's lo.x = the room's LEFT WALL: 214 condensed cloud
            // points attributed to it by active_face(), and a posterior still sitting exactly at
            // refit's prior (sigma = 9.53 m = the room's span). refit and the planner disagreeing
            // about the same layout and the same cloud IS the evidence: when two positive boxes
            // share a wall, `Layout::sdf` takes a MIN over boxes, so perturbing one of the two
            // coincident offsets leaves the min — and every residual — untouched. Its Jacobian is
            // identically zero, refit's own `abs(jk) < 1e-4` test drops every one of those 214
            // points, and no viewpoint in the world can inform it. The posterior is not broken; it
            // is telling the truth about a DEGENERATE PARAMETERISATION.
            // So ask the estimator's question: does moving this offset move the boundary where its
            // own returns are? If not, the wall is being carried by the other copy and there is
            // nothing here to refine. (The underlying degeneracy — two free offsets for one
            // surface, where the design already has `attach` to tie them — is the estimator's to
            // fix; the planner must not chase it either way.)
            std::vector<int> face_off;
            {
                rc::boxes::Layout T = L_;
                const float eps = 1e-3f;
                for (const auto& [off, bk] : buckets)
                {
                    if (bk.n <= 0) continue;
                    const Eigen::Vector2f mid = bk.acc / static_cast<float>(bk.n);
                    const Eigen::Vector2f probe = mid + 0.5f * p_.cell * bk.nrm;
                    const size_t k = static_cast<size_t>(off);
                    rc::boxes::Box& tb = T.boxes[k / 4];
                    float& o = (k % 4 == 0) ? tb.lo.x() : (k % 4 == 1) ? tb.lo.y()
                             : (k % 4 == 2) ? tb.hi.x() : tb.hi.y();
                    const float keep = o;
                    const float d0 = T.sdf(probe);
                    o += eps;
                    const float jk = (T.sdf(probe) - d0) / eps;
                    o = keep;
                    if (std::abs(jk) < 1e-4f) continue;          // the same test refit applies
                    faces.push_back({mid, bk.nrm, var_of(k)});
                    face_off.push_back(off);
                }
            }
            if (faces.empty()) { phase_ = Phase::Done; return out; }
            size_t worst = 0;
            for (size_t i = 1; i < faces.size(); ++i) if (faces[i].var > faces[worst].var) worst = i;
            worst_face_sigma_ = static_cast<float>(std::sqrt(faces[worst].var));
            static const bool probe = std::getenv("WS_IG_PROBE") != nullptr;
            if (probe)
            {
                std::fprintf(stderr, "[refine] %zu faces (of %zu offsets, cov %ld) worst=%zu var=%.4g "
                                     "mid=(%.2f,%.2f) nrm=(%+.0f,%+.0f) | vars:",
                             faces.size(), L_.n_offsets(), static_cast<long>(L_.cov.rows()), worst,
                             faces[worst].var, faces[worst].mid.x(), faces[worst].mid.y(),
                             faces[worst].nrm.x(), faces[worst].nrm.y());
                for (const auto& fc : faces) std::fprintf(stderr, " %.3g", std::sqrt(fc.var));
                std::fprintf(stderr, " | off:");
                for (const int o : face_off) std::fprintf(stderr, " %d", o);
                std::fprintf(stderr, " | cloudpts:");
                for (const int o : face_off)
                {
                    int n = 0;
                    for (const auto& q : cloud_) if (rc::boxes::active_face(L_, q.p) == o) ++n;
                    std::fprintf(stderr, " %d", n);
                }
                std::fprintf(stderr, "\n");
            }

            // ── WHEN IS THERE NOTHING LEFT TO LEARN ABOUT SHAPE? ────────────────────────────
            // Split the uncertainty on a wall's offset into what looking can change and what it
            // cannot. refit's 1/H is the REDUCIBLE part: it shrinks with every fresh view. The
            // wall's own flatness, sigma_flat, is COMMON-MODE — every return on that surface
            // shares it (the same correlated-evidence argument the proposal likelihood already
            // marginalises by Woodbury), so no number of visits reduces it. Refine stops when the
            // part it can still buy has fallen below the part it can never buy: at
            // 1/H <= sigma_flat^2 the next drive moves the belief about that wall by less than
            // the wall's own scatter. A physical constant of the generative model compared
            // against the posterior — not a stopping bar invented here.
            // ⚠ 1/H IS OPTIMISTIC ON ITS OWN: refit's normal equations are diagonal and carry no
            // common mode, so a well-seen face reports 1.6-2.1 mm on room idx 0, far under the
            // 10 mm flatness. That is why the comparison must be this way round — it asks whether
            // the REDUCIBLE part has become negligible, and never claims the wall is known to
            // 2 mm.
            // ⚠ Before this, Refine had NO exit: the viewpoint search below almost always
            // succeeds, so `Done` was reachable only on an empty layout and every run of this
            // planner ended at the frame cap.
            if (worst_face_sigma_ <= p_.sigma_flat) { phase_ = Phase::Done; return out; }

            // Stand square to it, at about 1.3 m — the standoff the fixed tour happens to use, and
            // the range where cos^2/range^2 is large without the wall filling the scan.
            // ⚠ THE VIEWPOINT MUST BE INSIDE THE ROOM. This loop ranges over every reachable free
            // cell, and free space is not confined to the layout: a beam that slips between two
            // returns sweeps the outdoors, and standing out there is square to the wall's OUTER
            // face at a perfectly good range. Measured (room idx 0, 6.8 x 4.0 m): the robot walked
            // monotonically out to 11.5 m from its start and spent the whole run outside, in BOTH
            // the old vantage-based refine and the posterior-based one. The layout is the robot's
            // own statement of where the room is; a viewpoint it does not contain is not a
            // viewpoint. Same SDF sign test as the frontier gain, same justification.
            std::pair<int, int> goal{0, 0}; double bestv = -1.0; bool gotv = false;
            for (const auto& [k, d] : dist)
            {
                if (not is_inside(k.first, k.second) or not passable(k)) continue;
                const Eigen::Vector2f m((static_cast<float>(k.first) + 0.5f) * p_.cell,
                                        (static_cast<float>(k.second) + 0.5f) * p_.cell);
                const Eigen::Vector2f ml(cyl * m.x() - syl * m.y(), syl * m.x() + cyl * m.y());
                const Eigen::Vector2f r = ml - faces[worst].mid;
                const float rng = r.norm();
                if (rng < 0.7f or rng > 2.6f) continue;
                const float cosi = std::abs(r.dot(faces[worst].nrm)) / std::max(1e-3f, rng);
                const double v = static_cast<double>(cosi) / (1.0 + 0.25 * static_cast<double>(d) * p_.cell);
                if (v > bestv) { bestv = v; goal = k; gotv = true; }
            }
            if (not gotv) { phase_ = Phase::Done; return out; }
            best = goal; have = true; last_gain_ = 0;
        }
        if (not have) { phase_ = Phase::Done; return out; }

        std::vector<std::pair<int, int>> rev;
        for (auto cur = best; cur != start; cur = parent[cur])
        {
            rev.push_back(cur);
            if (not parent.count(cur)) return out;      // unreachable; should not happen after BFS
        }
        // ── THIN THE PATH, BUT NEVER ACROSS SOMETHING THE PATH WENT AROUND ──────────────────
        // ⚠ THE OLD COMMENT HERE WAS FALSE. It claimed "the straight legs between waypoints stay
        // inside free space because the cells they join do" — true only for walls THICKER than the
        // stride. The BFS path is 4-connected over free cells, so it cannot cross a wall; decimating
        // it to a waypoint every 0.4 m then reconnects the survivors with chords that cut straight
        // through anything thinner. The apartamento hall has two 0.12 m fins and a 0.062 m edge:
        // measured, 19 of 217 executed legs crossed a wall, and the drawn trajectory ran through
        // both fins. It is not the controller and not the frame — it is this decimation.
        // So pull the string instead: extend a leg while every cell under it is FREE, and plant a
        // waypoint at the last cell for which that held. Legs come out longer than 0.4 m in open
        // space and as short as a cell around a fin, which is the right behaviour in both places,
        // and the constraint is a validity test rather than a tuned spacing.
        const auto seg_free = [&](const std::pair<int, int>& a, const std::pair<int, int>& b)
        {
            const float ax = static_cast<float>(a.first) + 0.5f, ay = static_cast<float>(a.second) + 0.5f;
            const float bx = static_cast<float>(b.first) + 0.5f, by = static_cast<float>(b.second) + 0.5f;
            const float dx = bx - ax, dy = by - ay;
            // sample at 0.4 of a cell: finer than the thinnest thing the grid can represent, so a
            // 2-cell fin cannot slip between two samples
            const int ns = std::max(1, static_cast<int>(std::ceil(std::max(std::abs(dx), std::abs(dy)) / 0.4f)));
            for (int k = 0; k <= ns; ++k)
            {
                const float t = static_cast<float>(k) / static_cast<float>(ns);
                const std::pair<int, int> c{static_cast<int>(std::floor(ax + t * dx)),
                                            static_cast<int>(std::floor(ay + t * dy))};
                if (not passable(c)) return false;
            }
            return true;
        };
        std::vector<std::pair<int, int>> fwd(rev.rbegin(), rev.rend());   // start-adjacent first
        std::pair<int, int> anchor = start;
        for (size_t i = 0; i < fwd.size(); ++i)
            if (i + 1 == fwd.size() or not seg_free(anchor, fwd[i + 1]))
            {
                out.emplace_back((static_cast<float>(fwd[i].first) + 0.5f) * p_.cell,
                                 (static_cast<float>(fwd[i].second) + 0.5f) * p_.cell);
                anchor = fwd[i];
            }
        const Eigen::Vector2f goal_m((static_cast<float>(best.first) + 0.5f) * p_.cell,
                                     (static_cast<float>(best.second) + 0.5f) * p_.cell);
        if (out.empty() or (out.back() - goal_m).norm() > 0.05f) out.emplace_back(goal_m);
        return out;
    }

    float Channel::reproject()
    {
        if (keys_.empty() or L_.empty()) return 0.f;
        const float gy = yaw_, cg = std::cos(-gy), sg = std::sin(-gy);
        const float cb = std::cos(gy), sb = std::sin(gy);
        double moved = 0.0; long n = 0;

        // 1. Re-register each keyframe against the CURRENT layout, starting from where it thought
        //    it was. The layout is the shared reference; this is what makes the poses mutually
        //    consistent instead of each one frozen at whatever the map looked like at the time.
        for (auto& kf : keys_)
        {
            const Eigen::Vector3f p_L(cg * kf.pose.x() - sg * kf.pose.y(),
                                      sg * kf.pose.x() + cg * kf.pose.y(),
                                      std::atan2(std::sin(kf.pose.z() - gy), std::cos(kf.pose.z() - gy)));
            const auto rr = rc::boxes::register_scan(L_, kf.pts, p_L, p_.sensor_sigma);
            if (not rr.ok) continue;
            const Eigen::Vector3f np(cb * rr.pose.x() - sb * rr.pose.y(),
                                     sb * rr.pose.x() + cb * rr.pose.y(),
                                     std::atan2(std::sin(rr.pose.z() + gy), std::cos(rr.pose.z() + gy)));
            moved += (np.head<2>() - kf.pose.head<2>()).norm(); ++n;
            kf.pose = np;
            // The re-registered pose comes with its own covariance, in the LAYOUT frame — rotate it
            // into the map frame the keyframe lives in and keep it. This is the one place a past
            // frame's evidence can get BETTER: a keyframe re-registered against a sharper layout
            // has a tighter pose, so its returns legitimately carry more information than when they
            // were captured. Without it a revisit can never pay and the explorer has no reason to
            // look at a wall twice.
            if (rr.cov.allFinite())
            {
                Eigen::Matrix3f R3 = Eigen::Matrix3f::Identity();
                R3.topLeftCorner<2, 2>() = Eigen::Rotation2Df(gy).toRotationMatrix();
                kf.cov = R3 * rr.cov * R3.transpose();
            }
        }

        // 2. Rebuild the occupancy from the corrected poses. THIS is the step a counter-based map
        //    can never do: cells swept in error are simply not swept again.
        vmap_.clear(); free_.clear();
        for (const auto& kf : keys_)
        {
            const float c = std::cos(kf.pose.z()), s = std::sin(kf.pose.z());
            const Eigen::Vector2f o = kf.pose.head<2>();
            for (const auto& q : kf.pts)
            {
                // ⚠ PER POINT, NOT PER SCAN. This used to weight every point of the keyframe by
                // kf.sigma — the scan's WORST point — and stamp that on `smin`, so the near returns
                // that carry the geometry were charged the far returns' lever arm. Same formula as
                // the live fold, same helper, so the rebuild cannot disagree with it.
                const float sg = point_sigma(kf.cov, c, s, q);
                if (not std::isfinite(sg)) continue;
                const float s0 = std::sqrt(sg * sg + p_.sensor_sigma * p_.sensor_sigma);
                const double w = 1.0 / (static_cast<double>(s0) * s0);
                const Eigen::Vector2f g(c * q.x() - s * q.y() + o.x(), s * q.x() + c * q.y() + o.y());
                auto& v = vmap_[{static_cast<int>(std::floor(g.x() / p_.cell)),
                                 static_cast<int>(std::floor(g.y() / p_.cell))}];
                v.acc += w * g.cast<double>(); v.w += w; v.smin = std::min(v.smin, sg);
                const Eigen::Vector2f d = g - o;
                const float len = d.norm();
                if (len < 1e-3f) continue;
                const Eigen::Vector2f u = d / len;
                for (float t = 0.f; t < len - p_.cell; t += 0.5f * p_.cell)
                {
                    const Eigen::Vector2f m = o + u * t;
                    ++free_[{static_cast<int>(std::floor(m.x() / p_.cell)),
                             static_cast<int>(std::floor(m.y() / p_.cell))}];
                }
            }
            ++free_[{static_cast<int>(std::floor(o.x() / p_.cell)),
                     static_cast<int>(std::floor(o.y() / p_.cell))}];
        }
        free_at_rebuild_ = free_.size();
        return n ? static_cast<float>(moved / static_cast<double>(n)) : 0.f;
    }

    void Channel::snap_coplanar()
    {
        // ── TWO FACES CLOSER THAN THE MEASUREMENT PRECISION ARE THE SAME WALL ───────────────
        // A cover assembled from many rectangles leaves their shared faces differing by fractions
        // of a cell, and Layout::polygon() faithfully traces every one of those steps: the region
        // scored IoU 0.926 while the published outline carried 196 vertices against a true 32.
        // Those are not 196 walls, they are ~32 walls and a staircase of quantisation.
        // Clustering the offsets at the precision the data actually supports is the same operation
        // wall_map calls merge_indistinguishable, and the tolerance is MEASURED (the current wall
        // residual, floored at the sensor) rather than chosen.
        if (L_.empty()) return;
        // ⚠ THE TOLERANCE IS THE GRID, NOT THE RESIDUAL. The staircase being removed here is a
        // QUANTISATION artefact of the cover, so the scale that defines it is the cell. Reading
        // the measured residual instead looked more principled and was not: that residual was
        // 0.946 m on this run, so "faces closer than the data can distinguish" became "faces
        // within a metre", and a 60-box layout collapsed to 3 boxes and 8 vertices with IoU
        // 0.609 -> 0.485. ★ A self-referential tolerance is only ever as good as the statistic it
        // reads, and this one was reading a number that was itself the symptom.
        const float tol = p_.cell;
        auto snap_axis = [&](bool xaxis)
        {
            std::vector<float> vals;
            for (const auto& b : L_.boxes)
            { vals.push_back(xaxis ? b.lo.x() : b.lo.y()); vals.push_back(xaxis ? b.hi.x() : b.hi.y()); }
            std::sort(vals.begin(), vals.end());
            std::vector<float> reps;                      // cluster means, in order
            size_t i = 0;
            while (i < vals.size())
            {
                size_t j = i; double sum = 0.0;
                while (j < vals.size() and vals[j] - vals[i] <= tol) { sum += vals[j]; ++j; }
                reps.push_back(static_cast<float>(sum / static_cast<double>(j - i)));
                i = j;
            }
            const auto nearest = [&](float v)
            {
                float best = v, bd = std::numeric_limits<float>::max();
                for (float r : reps) { const float d = std::abs(r - v); if (d < bd) { bd = d; best = r; } }
                return bd <= tol ? best : v;
            };
            for (auto& b : L_.boxes)
            {
                if (xaxis) { b.lo.x() = nearest(b.lo.x()); b.hi.x() = nearest(b.hi.x()); }
                else       { b.lo.y() = nearest(b.lo.y()); b.hi.y() = nearest(b.hi.y()); }
            }
        };
        snap_axis(true);
        snap_axis(false);
        // A box that collapsed to zero width is gone — `width -> 0` IS the removal event.
        for (size_t k = L_.boxes.size(); k-- > 0;)
            if (not L_.boxes[k].valid()) L_.boxes.erase(L_.boxes.begin() + static_cast<long>(k));
        // Snapping makes previously-misaligned neighbours exactly coincident, so absorb any box
        // now contained in another.
        for (size_t a = L_.boxes.size(); a-- > 0;)
            for (size_t b = 0; b < L_.boxes.size(); ++b)
            {
                if (a == b or a >= L_.boxes.size()) continue;
                if (L_.boxes[b].lo.x() <= L_.boxes[a].lo.x() + 1e-4f and
                    L_.boxes[b].hi.x() >= L_.boxes[a].hi.x() - 1e-4f and
                    L_.boxes[b].lo.y() <= L_.boxes[a].lo.y() + 1e-4f and
                    L_.boxes[b].hi.y() >= L_.boxes[a].hi.y() - 1e-4f)
                { L_.boxes.erase(L_.boxes.begin() + static_cast<long>(a)); break; }
            }
    }

    bool Channel::rebuild_from_free()
    {
        if (free_.empty()) return false;
        // Recompute only when the free set has grown by more than a twentieth since last time, or
        // when there is no layout yet. Nothing is lost: an unchanged cover would be recomputed to
        // the same answer.
        // ⚠ THE COVER IS THE ONLY THING TRACKING EXPLORATION NOW, so it must follow it closely.
        // At a twentieth it lagged badly: the layout stayed at whatever the robot had seen early,
        // registration ran against a region much smaller than the room, and the pose drifted to
        // 1.93 m with the gauge 9.9 degrees out. A hundredth costs more rebuilds and keeps the
        // reference honest.
        if (not L_.empty() and free_.size() < free_at_rebuild_ + free_at_rebuild_ / 100) return false;
        free_at_rebuild_ = free_.size();
        std::set<std::pair<int, int>> F;
        // ⚠ ONE SWEEP IS ALREADY A MEASUREMENT. Requiring two was my own invention, and in a
        // large apartment traversed once it removed most of the evidence near the walls: the cover
        // then sat ~1 m inside them (rms 1.157 m) and the room came out at IoU 0.224. A cell a
        // beam passed through is free; how many beams is a matter of confidence, not of fact, and
        // confidence belongs in the weighting, not in a count.
        for (const auto& [k, n] : free_) if (n >= 1) F.insert(k);
        if (F.size() < 12) return false;

        std::set<std::pair<int, int>> covered;
        std::vector<rc::boxes::Box> out;
        const auto is_free = [&](int x, int y) { return F.count({x, y}) > 0; };

        for (int iter = 0; iter < 24; ++iter)
        {
            // The largest free rectangle containing some UNCOVERED cell. Seeds are sampled rather
            // than exhaustive: the cover is greedy anyway, so an exact argmax buys nothing.
            long best_area = 0; rc::boxes::Box best; bool have = false;
            int tried = 0;
            for (const auto& s : F)
            {
                if (covered.count(s)) continue;
                if (++tried > 1200) break;
                int x0 = s.first, x1 = s.first, y0 = s.second, y1 = s.second;
                bool moved = true;
                while (moved)
                {
                    moved = false;
                    bool ok = true;
                    for (int x = x0; x <= x1 and ok; ++x) ok = is_free(x, y1 + 1);
                    if (ok) { ++y1; moved = true; }
                    ok = true;
                    for (int x = x0; x <= x1 and ok; ++x) ok = is_free(x, y0 - 1);
                    if (ok) { --y0; moved = true; }
                    ok = true;
                    for (int y = y0; y <= y1 and ok; ++y) ok = is_free(x1 + 1, y);
                    if (ok) { ++x1; moved = true; }
                    ok = true;
                    for (int y = y0; y <= y1 and ok; ++y) ok = is_free(x0 - 1, y);
                    if (ok) { --x0; moved = true; }
                }
                long gain = 0;
                for (int x = x0; x <= x1; ++x)
                    for (int y = y0; y <= y1; ++y) if (not covered.count({x, y})) ++gain;
                if (gain > best_area)
                {
                    best_area = gain; have = true;
                    best.lo = {static_cast<float>(x0) * p_.cell, static_cast<float>(y0) * p_.cell};
                    best.hi = {static_cast<float>(x1 + 1) * p_.cell, static_cast<float>(y1 + 1) * p_.cell};
                }
            }
            if (not have) break;
            // MDL: a box costs four offsets at the Occam precision log(span/sigma). It must explain
            // more cells than that many nats. Same currency as grow(); no new constant.
            const float sig = std::sqrt(p_.sensor_sigma * p_.sensor_sigma + p_.sigma_flat * p_.sigma_flat);
            const float span = std::max(1.f, best.width() + best.height());
            const float code = 4.f * std::log(span / sig);
            if (static_cast<float>(best_area) < code and not out.empty()) break;
            out.push_back(best);
            for (int x = static_cast<int>(std::floor(best.lo.x() / p_.cell));
                 x < static_cast<int>(std::floor(best.hi.x() / p_.cell)); ++x)
                for (int y = static_cast<int>(std::floor(best.lo.y() / p_.cell));
                     y < static_cast<int>(std::floor(best.hi.y() / p_.cell)); ++y)
                    covered.insert({x, y});
        }
        if (out.empty()) return false;

        // ── MERGE. A greedy cover is not a decomposition. ───────────────────────────────────
        // Picking maximal rectangles by area gain leaves the region correct but shredded: 45 boxes
        // and 196 published vertices for a 32-vertex hall. Two boxes may be replaced by their
        // bounding box IFF that bbox contains no non-free cell — the region is then IDENTICAL and
        // the description is shorter, so there is nothing to trade off and no constant to pick.
        // Run to a fixed point, then drop any box the others already cover.
        for (bool again = true; again; )
        {
            again = false;
            for (size_t i = 0; i < out.size() and not again; ++i)
                for (size_t j = i + 1; j < out.size() and not again; ++j)
                {
                    rc::boxes::Box u;
                    u.lo = out[i].lo.cwiseMin(out[j].lo);
                    u.hi = out[i].hi.cwiseMax(out[j].hi);
                    bool all_free = true;
                    for (int x = static_cast<int>(std::floor(u.lo.x() / p_.cell));
                         x < static_cast<int>(std::floor(u.hi.x() / p_.cell)) and all_free; ++x)
                        for (int y = static_cast<int>(std::floor(u.lo.y() / p_.cell));
                             y < static_cast<int>(std::floor(u.hi.y() / p_.cell)) and all_free; ++y)
                            if (not is_free(x, y)) all_free = false;
                    if (not all_free) continue;
                    out[i] = u;
                    out.erase(out.begin() + static_cast<long>(j));
                    again = true;
                }
        }
        // A box wholly inside another contributes nothing but vertices.
        for (size_t i = out.size(); i-- > 0;)
            for (size_t j = 0; j < out.size(); ++j)
            {
                if (i == j) continue;
                if (out[j].lo.x() <= out[i].lo.x() + 1e-4f and out[j].hi.x() >= out[i].hi.x() - 1e-4f and
                    out[j].lo.y() <= out[i].lo.y() + 1e-4f and out[j].hi.y() >= out[i].hi.y() - 1e-4f)
                { out.erase(out.begin() + static_cast<long>(i)); break; }
            }

        // Into the layout frame, and keep only boxes that touch the growing union so the region
        // stays connected — a detached box would be a second apartment.
        const float cy = std::cos(-yaw_), sy = std::sin(-yaw_);
        rc::boxes::Layout N;
        for (const auto& b : out)
        {
            const Eigen::Vector2f a(cy * b.lo.x() - sy * b.lo.y(), sy * b.lo.x() + cy * b.lo.y());
            const Eigen::Vector2f d(cy * b.hi.x() - sy * b.hi.y(), sy * b.hi.x() + cy * b.hi.y());
            rc::boxes::Box t; t.lo = a.cwiseMin(d); t.hi = a.cwiseMax(d);
            if (not t.valid()) continue;
            N.boxes.push_back(t);
        }
        if (N.boxes.empty()) return false;

        // ── ONE WRITER. THE COVER PROPOSES; mdl_cost DECIDES. ───────────────────────────────
        // The cover appended with a zero-tolerance geometric rule while simplify() deleted with a
        // priced MDL rule, so the two disagreed about every fringe cell and oscillated for ever:
        // 199 boxes removed on one run while the count still climbed. Two writers, two objectives,
        // one state. Now the cover builds a COMPLETE candidate — the boxes it would make plus the
        // ones already believed — and the candidate is adopted only if it costs fewer nats than
        // the incumbent. That is the trial-adoption judge that worked in the previous
        // representation (b83b4f3), and it makes oscillation impossible: nothing is written unless
        // the total goes down.
        rc::boxes::GrowParams gp;
        gp.sensor_sigma = p_.sensor_sigma; gp.sigma_flat = p_.sigma_flat;
        gp.cell = p_.cell; gp.min_cluster = p_.min_cluster;
        fuse();
        if (cloud_.empty()) return false;
        std::set<std::pair<int, int>> flc;
        {
            const float cyc = std::cos(-yaw_), syc = std::sin(-yaw_);
            for (const auto& [k, nn] : free_)
            {
                if (nn < 1) continue;
                const Eigen::Vector2f m((static_cast<float>(k.first) + 0.5f) * p_.cell,
                                        (static_cast<float>(k.second) + 0.5f) * p_.cell);
                const Eigen::Vector2f q(cyc * m.x() - syc * m.y(), syc * m.x() + cyc * m.y());
                flc.insert({static_cast<int>(std::floor(q.x() / p_.cell)),
                            static_cast<int>(std::floor(q.y() / p_.cell))});
            }
        }
        // ⚠ A COMPLETE RIVAL DESCRIPTION, NOT AN ADDENDUM. Building the candidate as
        // `incumbent + new boxes` makes it strictly larger and therefore strictly more expensive,
        // so after the first adoption every later proposal is rejected for ever. Measured on a
        // plain rectangle: adopt 8 boxes, simplify to 3, then reject 11/11/14/18/17-box candidates
        // in turn — leaving a 3-box layout over part of the room, a registration with almost
        // nothing to register against, and a pose 1.915 m out.
        // A trial-adoption judge needs two COMPLETE descriptions of the same evidence.
        rc::boxes::Layout cand = N;
        if (std::getenv("WS_ADOPT_PROBE") and not L_.empty())
        {
            rc::boxes::Layout k2 = L_, c2 = cand;
            rc::boxes::refit(k2, cloud_, gp, 3); rc::boxes::refit(c2, cloud_, gp, 3);
            std::fprintf(stderr, "[adopt] keep=%zu boxes cost=%.0f | cand=%zu boxes cost=%.0f | %s\n",
                         k2.boxes.size(), rc::boxes::mdl_cost(k2, cloud_, gp, &flc),
                         c2.boxes.size(), rc::boxes::mdl_cost(c2, cloud_, gp, &flc),
                         rc::boxes::mdl_cost(c2, cloud_, gp, &flc) < rc::boxes::mdl_cost(k2, cloud_, gp, &flc)
                            ? "ADOPT" : "reject");
        }
        if (not L_.empty())
        {
            rc::boxes::Layout keep = L_;
            rc::boxes::refit(keep, cloud_, gp, 3);
            rc::boxes::refit(cand, cloud_, gp, 3);
            if (rc::boxes::mdl_cost(cand, cloud_, gp, &flc)
                >= rc::boxes::mdl_cost(keep, cloud_, gp, &flc))
            { L_ = keep; return false; }          // the proposal does not pay — nothing is written
        }
        // ⚠ DISCARD THE GAUGE VOTES TAKEN AGAINST THE PROVISIONAL HULL. The seed exists so that
        // registration has SOMETHING from frame one, and it works — but it is one scan's interval
        // hull, so the poses it corrects are biased, and those poses feed the Manhattan vote
        // (seg_phi + pose.z()). Kept, they sit in the accumulator for ever: measured, the gauge
        // came out 2-3 degrees off across the featured rooms, which at 6 m misplaces the ends by
        // 0.3 m and costs ~0.03 of IoU while rms stays at 0.019 — a perfect fit in a tilted frame.
        // The first adopted cover is the first reference worth voting against, so the vote starts
        // there.
        if (not adopted_once_)
        {
            adopted_once_ = true;
            yaw4_cos_ = 0.0; yaw4_sin_ = 0.0; yaw_votes_ = 0;
        }
        L_ = cand;
        // The reference the evidence was registered against has just changed, so the evidence is
        // re-registered against it and rebuilt. Without this the adoption drags the pose instead
        // of the pose following the map.
        if (keys_.size() >= 4) reproject();
        // Merge across the whole layout, older boxes included: a new box often completes a
        // rectangle an older one only had part of.
        {
            const float cyb = std::cos(yaw_), syb = std::sin(yaw_);
            for (bool again = true; again; )
            {
                again = false;
                for (size_t i = 0; i < L_.boxes.size() and not again; ++i)
                    for (size_t j = i + 1; j < L_.boxes.size() and not again; ++j)
                    {
                        if (not L_.boxes[i].positive or not L_.boxes[j].positive) continue;
                        rc::boxes::Box u;
                        u.lo = L_.boxes[i].lo.cwiseMin(L_.boxes[j].lo);
                        u.hi = L_.boxes[i].hi.cwiseMax(L_.boxes[j].hi);
                        bool all_free = true;
                        for (float x = u.lo.x() + 0.5f * p_.cell; x < u.hi.x() and all_free; x += p_.cell)
                            for (float y = u.lo.y() + 0.5f * p_.cell; y < u.hi.y() and all_free; y += p_.cell)
                            {
                                const Eigen::Vector2f mp(cyb * x - syb * y, syb * x + cyb * y);
                                if (not is_free(static_cast<int>(std::floor(mp.x() / p_.cell)),
                                                static_cast<int>(std::floor(mp.y() / p_.cell)))) all_free = false;
                            }
                        if (not all_free) continue;
                        L_.boxes[i] = u;
                        L_.boxes.erase(L_.boxes.begin() + static_cast<long>(j));
                        again = true;
                    }
            }
        }
        // ⚠ DO NOT RESET THE POSTERIOR HERE. refit() computes it (1/H per offset) and this line
        // used to throw it away on every rebuild, every simplify and every seed — five sites. An
        // estimator that resets its own covariance has no memory of what it knows.
        return true;
    }

    std::vector<Eigen::Vector2f> Channel::polygon() const
    {
        if (L_.empty()) return {};
        const float cy = std::cos(yaw_), sy = std::sin(yaw_);
        std::vector<Eigen::Vector2f> out;
        for (const auto& v : L_.polygon())
            out.emplace_back(cy * v.x() - sy * v.y(), sy * v.x() + cy * v.y());
        return out;
    }

    void Channel::fuse()
    {
        // The evidence is FUSED, never appended. A capped accumulator stops taking data partway
        // through the run, so the map is decided by the earliest and worst-registered frames and no
        // later view can correct them. Fusing per cell by information weight keeps every frame and
        // bounds the set by the ROOM's size rather than the session's length.
        // Sigma kept is the best SINGLE look at the cell, never the fused one: repeated views of a
        // wall are correlated, and claiming sqrt(N) precision from them is the same error as
        // [[wall-factor-5000x-too-weak]].
        const float cy = std::cos(-yaw_), sy = std::sin(-yaw_);
        cloud_.clear();
        cloud_.reserve(vmap_.size());
        for (const auto& [k, v] : vmap_)
        {
            if (v.w <= 0.0) continue;
            const Eigen::Vector2f m(static_cast<float>(v.acc.x() / v.w),
                                    static_cast<float>(v.acc.y() / v.w));
            cloud_.push_back({Eigen::Vector2f(cy * m.x() - sy * m.y(), sy * m.x() + cy * m.y()), v.smin});
        }
    }

    void Channel::open_csv(const std::string& path)
    {
        csv_.open(path, std::ios::out | std::ios::trunc);
        if (not csv_.is_open()) return;
        // ⚠ imbue the CLASSIC locale. These machines run LANG=es_ES.UTF-8, where the decimal
        // separator is a COMMA; a data file written under it cannot be read back by from_chars
        // and — worse — is silently truncated by any strtof-based reader. See CLAUDE.md.
        csv_.imbue(std::locale::classic());
        // bw,bh are the FIRST BOX's own width and height, in the LAYOUT frame, where a box is
        // axis-aligned by construction and the two numbers therefore mean something. w,h are the
        // map-frame polygon's bounding box and are kept only for the viewer's sake.
        // ⚠ THESE DIFFER WHENEVER THE GAUGE IS NON-ZERO, and quoting the wrong pair is how a
        // 6.00 x 4.00 room tilted 17.56 deg got reported as 6.76 x 5.58 with an area of 37.71 m2.
        // A bounding box is not a measurement of the thing it bounds.
        csv_ << "frame,boxes,verts,w,h,bw,bh,rms,proposed,admitted,removed,last_dL,cells,sigma_pose,yaw_deg,votes,frac_out,frac_in,rms_core\n";
    }

    bool Channel::step()
    {
        if (not p_.enabled) return false;
        if (L_.empty() and yaw_votes_ <= 0) return false;   // no frame yet ⇒ nothing to build in
        if (p_.every_frames <= 0 or frames_ % static_cast<std::uint64_t>(p_.every_frames) != 0) return false;

        rc::boxes::GrowParams gp;
        gp.sensor_sigma = p_.sensor_sigma;
        gp.sigma_flat   = p_.sigma_flat;
        gp.cell         = p_.cell;
        gp.min_cluster  = p_.min_cluster;

        // ── ADOPT THE VOTED GAUGE ────────────────────────────────────────────────────────────
        // Circular mean on the quadrupled angle, then back to a quarter-turn. The representative
        // is chosen NEAREST THE INCUMBENT among the four, which is not a tie-break rule but the
        // statement that the four are the same frame: picking the near one keeps the published
        // polygon continuous instead of letting it snap 90 degrees between frames.
        if (yaw_votes_ > 0 and (yaw4_cos_ != 0.0 or yaw4_sin_ != 0.0))
        {
            const double a = std::atan2(yaw4_sin_, yaw4_cos_) / 4.0;
            const double q = M_PI / 2.0;
            const double k = std::round((static_cast<double>(yaw_) - a) / q);
            yaw_ = static_cast<float>(a + k * q);
        }

        // CONSTRUCT from free space. Cheap, deterministic, and it replaces both the hull seed and
        // most of what grow() was being asked to repair.
        // ⚠ THE COVER IS THE CONSTRUCTION; grow() MUST NOT SECOND-GUESS IT.
        // rebuild_from_free() is throttled to fire only when the free set has grown materially, and
        // the first version let step() fall through to refit+grow on every other call. grow() then
        // re-fragmented exactly what the cover had just built: the cover produced ~18 boxes and
        // grow() admitted 27 more on top, for 45 boxes and 196 published vertices on a 32-vertex
        // hall. Incremental refinement is the COVER being rebuilt as free space grows — that is
        // what makes this incremental — not a second mechanism editing its output.
        // ⚠ CALL IT, THEN TEST. `have_cover_ or rebuild_from_free()` SHORT-CIRCUITS: once the
        // cover existed the rebuild was never invoked again, so the layout froze at whatever the
        // robot had seen in its first few metres and three successive fixes to the rebuild changed
        // nothing at all. Bit-identical output across substantive edits is not a null result — it
        // says the edited code is not on the path.
        // ⚠ COVERAGE AND CONSOLIDATION ARE DIFFERENT ACTIVITIES, AND INTERLEAVING THEM COSTS THE
        // MAP. While the robot is still discovering space the cover is extended again and again,
        // and refit/simplify never work a stable layout: under exploration the room went from
        // 6 boxes and 0.082 m of residual to 15 boxes and 0.364 m, while the pose stayed healthy
        // at 0.19 m — the map fell apart, not the localisation.
        // The fixed tour gets this for free: free space plateaus after one lap and the SECOND lap
        // is pure consolidation. So once the planner stops exploring, the region stops growing and
        // the remaining effort goes into fitting and simplifying what is already there.
        const bool extended = (phase_ == Phase::Explore) ? rebuild_from_free() : false;
        if (have_cover_ or extended)
        {
            have_cover_ = true;
            // ⚠ REPAIR IS NOT CONDITIONAL ON ADOPTION. reproject() used to fire only when a new
            // cover was adopted — so the case that most needs repair could never reach it. A plain
            // rectangle drifts 2 m on odometry BEFORE any layout exists (registration needs a
            // layout to register against), so the first cover is adopted already 7.99 m wide in a
            // 6.00 m room, and from then on every fresh candidate is correctly rejected because
            // the incumbent fits the smeared evidence well. The map was wrong and self-consistent,
            // and nothing was allowed to touch it: y came out 4.01 against a true 4.00 while x came
            // out 7.99 against 6.00, which is the 1.99 m pose error written into the geometry.
            // Re-registering the keyframes against the layout and rebuilding the occupancy from
            // the corrected poses is what pulls the early, pre-layout frames back into agreement.
            if (keys_.size() >= 4 and (structure_steps_++ % 3) == 0) reproject();
            fuse();
            if (not cloud_.empty()) rms_ = rc::boxes::refit(L_, cloud_, gp);
            if (std::getenv("WS_CLOUD_PROBE") and not cloud_.empty())
            {
                // Histogram the fused returns along x. A 6 m room drives 6 m of wall; if the cloud
                // spans more, the extra is GHOST WALLS written by a drifting pose, and any layout
                // that fits it will be that wide too.
                std::map<int, int> hx;
                float lo = 1e9f, hi = -1e9f;
                for (const auto& q : cloud_)
                { hx[static_cast<int>(std::floor(q.p.x() / 0.25f))]++; lo = std::min(lo, q.p.x()); hi = std::max(hi, q.p.x()); }
                std::fprintf(stderr, "[cloud] x span %.2f m (%.2f..%.2f)  peaks:", hi - lo, lo, hi);
                for (const auto& [b, n] : hx) if (n > 30) std::fprintf(stderr, " %.2f(%d)", b * 0.25f, n);
                std::fprintf(stderr, "\n");
            }
            snap_coplanar();
            if (std::getenv("WS_BOXES_NOSIMP") == nullptr)
                simplify();      // measure the complexity, then force it down
            return true;
        }

        // ── BORN IN THE VOTED FRAME, FROM THE FUSED CLOUD ───────────────────────────────────
        // The smallest hypothesis: one box, the interval hull of everything seen so far, taken at
        // a quantile so a handful of stray returns cannot inflate it. Manhattan, closed and simply
        // connected by construction — there is no shape to validate.
        if (L_.empty())
        {
            fuse();
            if (cloud_.size() < static_cast<std::size_t>(gp.min_cluster)) return false;
            // ── SEED FROM FREE SPACE, NOT FROM THE HULL ─────────────────────────────────────
            if (have_cell_)
            {
                const rc::boxes::Box fb = grow_free_box(last_cell_);
                if (fb.valid() and fb.width() > 3.f * p_.cell and fb.height() > 3.f * p_.cell)
                {
                    const float cy = std::cos(-yaw_), sy = std::sin(-yaw_);
                    const Eigen::Vector2f c1(cy * fb.lo.x() - sy * fb.lo.y(), sy * fb.lo.x() + cy * fb.lo.y());
                    const Eigen::Vector2f c2(cy * fb.hi.x() - sy * fb.hi.y(), sy * fb.hi.x() + cy * fb.hi.y());
                    rc::boxes::Box b;
                    b.lo = c1.cwiseMin(c2); b.hi = c1.cwiseMax(c2);
                    L_.boxes.push_back(b);
                    L_.cov = Eigen::MatrixXf::Identity(4, 4) * (p_.sigma_flat * p_.sigma_flat);
                    return true;
                }
            }
            std::vector<float> xs, ys;
            xs.reserve(cloud_.size()); ys.reserve(cloud_.size());
            for (const auto& q : cloud_) { xs.push_back(q.p.x()); ys.push_back(q.p.y()); }
            const auto quant = [](std::vector<float>& v, float f)
            {
                const size_t k = std::clamp<size_t>(static_cast<size_t>(f * static_cast<float>(v.size() - 1)),
                                                    0, v.size() - 1);
                std::nth_element(v.begin(), v.begin() + static_cast<long>(k), v.end());
                return v[k];
            };
            std::vector<float> xs2 = xs, ys2 = ys;
            rc::boxes::Box b;
            b.lo = {quant(xs, 0.005f), quant(ys, 0.005f)};
            b.hi = {quant(xs2, 0.995f), quant(ys2, 0.995f)};
            if (not b.valid()) return false;
            L_.boxes.push_back(b);
            L_.cov = Eigen::MatrixXf::Identity(4, 4) * (p_.sigma_flat * p_.sigma_flat);
            qInfo_first_ = true;
        }

        bool changed = false;
        // OPTIMISE, THEN LOOK FOR WHAT IS LEFT. Structure is only ever asked to explain a residual
        // the continuous parameters have already been given every chance to remove. Reversing these
        // two is what produced four 0.12-m slivers lying along the walls of a room that had none.
        // The loop bound is the parameter budget, not a threshold.
        for (int it = 0; it < 4; ++it)
        {
            fuse();
            if (cloud_.size() < static_cast<std::size_t>(gp.min_cluster)) return changed;
            rms_ = rc::boxes::refit(L_, cloud_, gp);
            // Free space, in the LAYOUT frame and on grow()'s grid.
            std::set<std::pair<int, int>> fl;
            {
                const float cy = std::cos(-yaw_), sy = std::sin(-yaw_);
                for (const auto& [k, n] : free_)
                {
                    if (n < 2) continue;
                    const Eigen::Vector2f m((static_cast<float>(k.first) + 0.5f) * p_.cell,
                                            (static_cast<float>(k.second) + 0.5f) * p_.cell);
                    const Eigen::Vector2f q(cy * m.x() - sy * m.y(), sy * m.x() + cy * m.y());
                    fl.insert({static_cast<int>(std::floor(q.x() / p_.cell)),
                               static_cast<int>(std::floor(q.y() / p_.cell))});
                }
            }
            const auto gr = rc::boxes::grow(L_, cloud_, gp, &fl);
            n_proposed_ += gr.proposed;
            if (gr.admitted > 0) { ++n_admitted_; last_dL_ = gr.best_dL; changed = true; }
            else if (gr.admitted < 0) { ++n_removed_; changed = true; }
            else break;
        }
        // ── WHERE IS THE RESIDUAL? ───────────────────────────────────────────────────────────
        {
            long nout = 0, nin = 0, ncore = 0; double ss = 0.0;
            const float s0 = std::sqrt(p_.sensor_sigma * p_.sensor_sigma + p_.sigma_flat * p_.sigma_flat);
            for (const auto& q : cloud_)
            {
                const float dd = L_.sdf(q.p);
                const float t = 3.f * std::sqrt(s0 * s0 + q.sigma_pose * q.sigma_pose);
                if (dd >  t) { ++nout; continue; }     // beyond a wall: unreachable by a carve
                if (dd < -t) { ++nin;  continue; }     // deep inside: a column would explain it
                ss += static_cast<double>(dd) * dd; ++ncore;
            }
            const float n = static_cast<float>(std::max<size_t>(1, cloud_.size()));
            frac_out_ = static_cast<float>(nout) / n;
            frac_in_  = static_cast<float>(nin) / n;
            rms_core_ = ncore ? static_cast<float>(std::sqrt(ss / static_cast<double>(ncore))) : 0.f;
        }
        if (csv_.is_open())
        {
            // The polygon's own extent, so SIZE is graded from the thing that is published rather
            // than from the first box's width — with a carve, those are not the same number.
            const auto vs = polygon();
            Eigen::Vector2f lo(1e9f, 1e9f), hi(-1e9f, -1e9f);
            for (const auto& v : vs) { lo = lo.cwiseMin(v); hi = hi.cwiseMax(v); }
            const float bw = L_.boxes.empty() ? 0.f : L_.boxes.front().width();
            const float bh = L_.boxes.empty() ? 0.f : L_.boxes.front().height();
            csv_ << frames_ << ',' << L_.boxes.size() << ',' << vs.size() << ','
                 << (vs.empty() ? 0.f : hi.x() - lo.x()) << ',' << (vs.empty() ? 0.f : hi.y() - lo.y()) << ','
                 << bw << ',' << bh << ','
                 << rms_ << ',' << n_proposed_ << ',' << n_admitted_ << ',' << n_removed_ << ','
                 << last_dL_ << ',' << vmap_.size() << ',' << last_sigma_ << ','
                 << yaw_ * 180.f / static_cast<float>(M_PI) << ',' << yaw_votes_ << ','
                 << frac_out_ << ',' << frac_in_ << ',' << rms_core_ << '\n' << std::flush;
        }
        return changed;
    }
}   // namespace rc::boxch
