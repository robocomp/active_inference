#include "room_boxes_channel.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <locale>
#include <set>

namespace rc::boxch
{
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
            const Eigen::Vector2f jth(-s * q.x() - c * q.y(), c * q.x() - s * q.y());
            const Eigen::Vector2f pxth(cov(0, 2), cov(1, 2));
            const float tr = cov(0, 0) + cov(1, 1) + cov(2, 2) * jth.squaredNorm() + 2.f * pxth.dot(jth);
            const float sg = std::sqrt(std::max(0.f, tr) * 0.5f);
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
        if (removed > 0)
            L_.cov = Eigen::MatrixXf::Identity(static_cast<long>(L_.n_offsets()),
                                               static_cast<long>(L_.n_offsets()))
                   * (p_.sigma_flat * p_.sigma_flat);
        n_removed_ += removed;
        return removed;
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
        L_.cov = Eigen::MatrixXf::Identity(static_cast<long>(L_.n_offsets()),
                                           static_cast<long>(L_.n_offsets())) * (p_.sigma_flat * p_.sigma_flat);
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
        for (const auto& b : N.boxes) L_.boxes.push_back(b);      // APPEND, never replace
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
        L_.cov = Eigen::MatrixXf::Identity(static_cast<long>(L_.n_offsets()),
                                           static_cast<long>(L_.n_offsets())) * (p_.sigma_flat * p_.sigma_flat);
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
        const bool extended = rebuild_from_free();
        if (have_cover_ or extended)
        {
            have_cover_ = true;
            fuse();
            if (not cloud_.empty()) rms_ = rc::boxes::refit(L_, cloud_, gp);
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
