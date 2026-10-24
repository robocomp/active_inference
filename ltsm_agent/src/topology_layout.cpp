/*
 * topology_layout.cpp — see topology_layout.h.
 */
#include "topology_layout.h"
#include "room_eviction.h"   // ltsm::is_door, ltsm::seam_through_passage

#include <algorithm>
#include <chrono>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <ranges>

namespace ltsm::topo
{

namespace
{
/// The room a node hangs below (parent_att chain), or nullopt.
std::optional<DSR::Node> owning_room(DSR::DSRGraph &g, const DSR::Node &n)
{
    auto cur = std::optional<DSR::Node>(n);
    while (cur.has_value())
    {
        if (cur->type() == "room") return cur;
        const auto p = g.get_attrib_by_name<parent_att>(cur.value());
        if (not p.has_value() or p.value() == cur->id()) return std::nullopt;
        cur = g.get_node(p.value());
    }
    return std::nullopt;
}

std::vector<Eigen::Vector2d> local_outline(DSR::DSRGraph &g, const DSR::Node &room)
{
    std::vector<Eigen::Vector2d> out;
    const auto px = g.get_attrib_by_name<delimiting_polygon_x_att>(room);
    const auto py = g.get_attrib_by_name<delimiting_polygon_y_att>(room);
    if (not px.has_value() or not py.has_value() or px->get().size() != py->get().size()) return out;
    for (const auto &[x, y] : std::views::zip(px->get(), py->get())) out.emplace_back(x, y);
    return out;
}

/// Area centroid (shoelace); the vertex mean when the polygon has no area, the origin when it has no vertices.
Eigen::Vector2d centroid(const std::vector<Eigen::Vector2d> &poly)
{
    if (poly.empty()) return Eigen::Vector2d::Zero();
    double a2 = 0;
    Eigen::Vector2d c = Eigen::Vector2d::Zero();
    for (std::size_t i = 0; i < poly.size(); ++i)
    {
        const auto &p = poly[i];
        const auto &q = poly[(i + 1) % poly.size()];
        const double cr = p.x() * q.y() - q.x() * p.y();
        a2 += cr;
        c += cr * (p + q);
    }
    if (std::abs(a2) > std::numeric_limits<double>::epsilon()) return c / (3.0 * a2);
    Eigen::Vector2d m = Eigen::Vector2d::Zero();
    for (const auto &p : poly) m += p;
    return m / static_cast<double>(poly.size());
}

Eigen::Vector2d apply(const Mat::RTMat &t, const Eigen::Vector2d &p)
{
    return (t * Eigen::Vector3d(p.x(), p.y(), 0.0)).head<2>();
}

layout::Vec3 se2(const Mat::RTMat &T)
{
    return {T.translation().x(), T.translation().y(), std::atan2(T.rotation()(1, 0), T.rotation()(0, 0))};
}

Mat::RTMat rtmat(const layout::Vec3 &p)
{
    Mat::RTMat T = Mat::RTMat::Identity();
    T.translate(Mat::Vector3d(p.x(), p.y(), 0.0));
    T.rotate(Eigen::AngleAxisd(p.z(), Eigen::Vector3d::UnitZ()));
    return T;
}

struct LocalDoor
{
    std::string name;
    std::size_t room;
    Eigen::Vector2d a, b, centre;   ///< in the room's own frame
};
}   // namespace

std::optional<std::size_t> Map::room_index(const std::string &name) const
{
    const auto it = std::ranges::find(rooms, name, &Room::name);
    if (it == rooms.end()) return std::nullopt;
    return static_cast<std::size_t>(std::distance(rooms.begin(), it));
}

Map build(DSR::DSRGraph &mem, const std::string &anchor,
          const std::map<std::string, std::vector<Eigen::Vector2d>> &live_outlines,
          LayoutCache *cache, const layout::Model &model)
{
    Map map;

    // ── Rooms: the anchor first (its frame is the map's), then a stable order: room_id, then name ───
    const auto is_anchor = [&](const DSR::Node &n)
    { return n.name() == anchor or (n.name().starts_with('r') and n.name().ends_with("_" + anchor)); };
    auto room_nodes = mem.get_nodes_by_type("room");
    std::ranges::sort(room_nodes, [&](const DSR::Node &l, const DSR::Node &r)
    {
        if (is_anchor(l) != is_anchor(r)) return is_anchor(l);
        const auto il = mem.get_attrib_by_name<room_id_att>(l).value_or(0);
        const auto ir = mem.get_attrib_by_name<room_id_att>(r).value_or(0);
        return il != ir ? il < ir : l.name() < r.name();
    });
    if (not room_nodes.empty() and is_anchor(room_nodes.front())) map.anchor = 0;
    std::vector<std::vector<Eigen::Vector2d>> local(room_nodes.size());
    for (const auto &[i, r] : std::views::enumerate(room_nodes))
    {
        map.rooms.push_back({.name = r.name()});
        const auto live = live_outlines.find(r.name());
        local[i] = live != live_outlines.end() ? live->second : local_outline(mem, r);
    }

    // ── Doors, in their room's frame ───────────────────────────────────────────────────────────
    auto ie = mem.get_inner_eigen_api();
    std::vector<LocalDoor> doors;
    std::map<std::uint64_t, std::size_t> door_of_node;
    for (const auto &n : mem.get_nodes_by_type("object"))
    {
        if (not ltsm::is_door(mem, n)) continue;
        const auto room = owning_room(mem, n);
        if (not room.has_value()) continue;
        const auto ri = map.room_index(room->name());
        const auto t = ie->get_transformation_matrix(room->name(), n.name());
        if (not ri.has_value() or not t.has_value()) continue;
        const double half_w = 0.5 * static_cast<double>(mem.get_attrib_by_name<width_m_att>(n).value_or(0.f));
        const Eigen::Vector2d o = t->translation().head<2>();
        const Eigen::Vector2d x = t->linear().col(0).head<2>();
        door_of_node[n.id()] = doors.size();
        doors.push_back({.name = n.name(), .room = ri.value(), .a = o - half_w * x, .b = o + half_w * x, .centre = o});
    }

    // ── Passages: the two faces, their rooms, and the seam that relates them ───────────────────
    std::vector<std::optional<Mat::RTMat>> seam_ab;   ///< T(room_a <- room_b) per passage
    std::vector<std::optional<seam::Measured>> seam_m;   ///< the same with its covariance (only when solving)
    for (const auto &n : mem.get_nodes_by_type("metaconcept"))
    {
        if (mem.get_attrib_by_name<object_subtype_att>(n).value_or("") != "passage") continue;
        Passage p{.name = n.name()};
        p.alpha = mem.get_attrib_by_name<passage_pass_alpha_att>(n).value_or(1.f);
        p.beta = mem.get_attrib_by_name<passage_pass_beta_att>(n).value_or(1.f);
        p.crossings = mem.get_attrib_by_name<passage_crossings_att>(n).value_or(0);
        for (const auto &e : DSR::DSRGraph::get_node_edges_by_type(n, "match"))
        {
            const auto it = door_of_node.find(e.to());
            if (it == door_of_node.end()) continue;
            auto &slot_door = p.door_a.has_value() ? p.door_b : p.door_a;
            auto &slot_room = p.room_a.has_value() ? p.room_b : p.room_a;
            slot_door = it->second;
            slot_room = doors[it->second].room;
        }
        std::optional<Mat::RTMat> s;
        std::optional<seam::Measured> sm;
        if (p.room_a.has_value() and p.room_b.has_value() and p.room_a != p.room_b)
        {
            s = ltsm::seam_through_passage(mem, n.name(), map.rooms[*p.room_a].name, map.rooms[*p.room_b].name);
            if (s.has_value() and cache)
                sm = ltsm::seam_measured_through_passage(mem, n.name(), map.rooms[*p.room_a].name,
                                                         map.rooms[*p.room_b].name, p.stated_faces);
        }
        p.seam_ok = s.has_value();
        seam_ab.push_back(s);
        seam_m.push_back(sm);
        map.passages.push_back(std::move(p));
    }

    // ── Place: BFS over the passages, one component at a time. Room 0 (the anchor) starts the first
    //    component at the identity, so every room it reaches is expressed in the anchor's frame. `preset` overrides
    //    a room's pose as soon as the walk reaches it (the cached MAP poses): a newcomer is then placed from its
    //    neighbours' settled poses, not from where the bare tree would put them. ─────────────────────────────
    std::vector<int> start_of;                            ///< component → the room that starts it (its gauge)
    std::vector<bool> tree_edge(map.passages.size(), false);
    const auto place = [&](const std::function<std::optional<Mat::RTMat>(std::size_t)> &preset)
    {
        std::vector<Mat::RTMat> world(map.rooms.size(), Mat::RTMat::Identity());
        std::vector<bool> placed(map.rooms.size(), false);
        std::ranges::fill(tree_edge, false);
        start_of.clear();
        int component = 0;
        for (std::size_t start = 0; start < map.rooms.size(); ++start)
        {
            if (placed[start]) continue;
            placed[start] = true;
            if (const auto p = preset(start); p.has_value()) world[start] = p.value();
            map.rooms[start].component = component;
            start_of.push_back(static_cast<int>(start));
            std::deque<std::size_t> frontier{start};
            while (not frontier.empty())
            {
                const auto u = frontier.front();
                frontier.pop_front();
                for (const auto &[k, p] : std::views::enumerate(map.passages))
                {
                    if (not p.seam_ok) continue;
                    const bool from_a = p.room_a == u, from_b = p.room_b == u;
                    if (not from_a and not from_b) continue;
                    const auto v = from_a ? *p.room_b : *p.room_a;
                    if (placed[v]) continue;
                    world[v] = from_a ? Mat::RTMat(world[u] * seam_ab[k].value())
                                      : Mat::RTMat(world[u] * seam_ab[k]->inverse());
                    if (const auto pv = preset(v); pv.has_value()) world[v] = pv.value();
                    map.rooms[v].component = component;
                    placed[v] = true;
                    tree_edge[k] = true;
                    frontier.push_back(v);
                }
            }
            ++component;
        }
        return world;
    };
    const auto raw = place([](std::size_t) { return std::optional<Mat::RTMat>{}; });
    const int component = static_cast<int>(start_of.size());
    for (auto &&[r, w] : std::views::zip(map.rooms, raw)) { r.world = w; r.world_raw = w; }

    // ── MAP: solve when the structure changed, else reuse. The key is what the solve depends on structurally: which
    //    rooms exist and which passages relate them. Outlines are read at solve time and not re-solved for. ───────
    if (cache)
    {
        std::vector<std::string> names;
        for (const auto &r : map.rooms) names.push_back(r.name);
        std::ranges::sort(names);
        std::vector<std::string> pnames;
        for (const auto &p : map.passages)
            if (p.seam_ok) pnames.push_back(p.name);
        std::ranges::sort(pnames);
        std::string key;
        for (const auto &x : names) key += x + ",";
        key += "|";
        for (const auto &x : pnames) key += x + ",";

        if (key != cache->key)
        {
            const auto t0 = std::chrono::steady_clock::now();
            const auto init = place([&](std::size_t i) -> std::optional<Mat::RTMat>
            {
                const auto it = cache->pose.find(map.rooms[i].name);
                if (it == cache->pose.end()) return std::nullopt;
                return rtmat(it->second);
            });
            LayoutCache next;
            next.key = key;
            next.solves = cache->solves + 1;
            next.fresh = true;
            for (int c = 0; c < component; ++c)
            {
                std::vector<std::size_t> members;   // global room index, in local order
                std::map<std::size_t, std::size_t> local_of;
                for (std::size_t i = 0; i < map.rooms.size(); ++i)
                    if (map.rooms[i].component == c) { local_of[i] = members.size(); members.push_back(i); }
                std::vector<layout::Room> lr;
                for (const auto i : members)
                {
                    layout::Room room{.name = map.rooms[i].name, .init = se2(init[i]),
                                      .fixed = static_cast<int>(i) == start_of[c]};
                    for (const auto &q : local[i]) room.outline.push_back(q);
                    lr.push_back(std::move(room));
                }
                std::vector<layout::Seam> ls;
                std::vector<std::size_t> ls_passage;
                for (const auto &[k, p] : std::views::enumerate(map.passages))
                {
                    if (not p.seam_ok or map.rooms[*p.room_a].component != c) continue;
                    layout::Seam s{.name = p.name, .a = local_of.at(*p.room_a), .b = local_of.at(*p.room_b)};
                    if (seam_m[k].has_value()) { s.z = seam_m[k]->pose; s.cov = seam_m[k]->cov; }
                    else
                    {
                        // The faces composed but their chains did not (no parent_att path): the mean is still the
                        // graph's, the covariance the remembered-seam one (room_eviction.cpp kRemembered*).
                        s.z = se2(seam_ab[k].value());
                        s.cov = layout::Vec3(0.1 * 0.1, 0.1 * 0.1, 0.03 * 0.03).asDiagonal();
                    }
                    if (p.door_a.has_value()) s.aperture_a = std::array{doors[*p.door_a].a, doors[*p.door_a].b};
                    if (p.door_b.has_value()) s.aperture_b = std::array{doors[*p.door_b].a, doors[*p.door_b].b};
                    // The outlier component's support: b's origin is within the two rooms' reach of a's.
                    const auto reach = [&](std::size_t i)
                    {
                        double r = 0;
                        for (const auto &q : local[i]) r = std::max(r, q.norm());
                        return r;
                    };
                    s.extent = reach(*p.room_a) + reach(*p.room_b);
                    ls.push_back(std::move(s));
                    ls_passage.push_back(static_cast<std::size_t>(k));
                }
                const auto res = layout::optimise(lr, ls, model);
                for (const auto &[li, gi] : std::views::enumerate(members))
                {
                    next.pose[map.rooms[gi].name] = res.pose[li];
                    next.cov[map.rooms[gi].name] = res.cov[li];
                    next.p_manhattan[map.rooms[gi].name] = res.p_manhattan[li];
                }
                for (const auto &[si, k] : std::views::enumerate(ls_passage))
                {
                    next.seam_d2[map.passages[k].name] = res.seam_d2[si];
                    next.seam_p_inlier[map.passages[k].name] = res.seam_p_inlier[si];
                    next.stated_faces[map.passages[k].name] = map.passages[k].stated_faces;
                }
                if (c == 0) next.building_axis = res.building_axis;
                next.before.seam += res.before.seam;   next.after.seam += res.after.seam;
                next.before.manhattan += res.before.manhattan;   next.after.manhattan += res.after.manhattan;
                next.before.overlap += res.before.overlap;   next.after.overlap += res.after.overlap;
                next.before.wall += res.before.wall;   next.after.wall += res.after.wall;
                next.iterations = std::max(next.iterations, res.iterations);
                next.starts += res.starts;
                if (res.winning_start != 0) ++next.restarted;
                next.converged = next.converged and res.converged;
            }
            next.rooms = static_cast<int>(map.rooms.size());
            next.passages = static_cast<int>(pnames.size());
            next.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            *cache = std::move(next);
        }
        for (auto &r : map.rooms)
            if (const auto it = cache->pose.find(r.name); it != cache->pose.end())
            {
                r.world = rtmat(it->second);
                r.cov = cache->cov[r.name];
                r.p_manhattan = cache->p_manhattan[r.name];
            }
        for (auto &p : map.passages)
        {
            if (const auto it = cache->seam_d2.find(p.name); it != cache->seam_d2.end()) p.seam_d2 = it->second;
            if (const auto it = cache->seam_p_inlier.find(p.name); it != cache->seam_p_inlier.end()) p.p_inlier = it->second;
        }
        map.optimised = true;
    }

    // ── Pack components left to right, bottoms aligned; the gap is a median room size. Component 0
    //    (the anchor's) never moves: the map frame stays the anchor's own. ─────────────────────────
    if (component > 1)
    {
        struct Box { double x0, y0, x1, y1; };
        std::vector<Box> boxes(component, Box{std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
                                              std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest()});
        std::vector<double> sizes;
        for (const auto &[i, r] : std::views::enumerate(map.rooms))
        {
            auto &bx = boxes[r.component];
            const auto pts = local[i].empty() ? std::vector<Eigen::Vector2d>{Eigen::Vector2d::Zero()} : local[i];
            double rx0 = std::numeric_limits<double>::max(), rx1 = std::numeric_limits<double>::lowest();
            double ry0 = rx0, ry1 = rx1;
            for (const auto &q : pts)
            {
                const auto w = apply(r.world, q);
                rx0 = std::min(rx0, w.x()); rx1 = std::max(rx1, w.x());
                ry0 = std::min(ry0, w.y()); ry1 = std::max(ry1, w.y());
            }
            bx = {std::min(bx.x0, rx0), std::min(bx.y0, ry0), std::max(bx.x1, rx1), std::max(bx.y1, ry1)};
            if (not local[i].empty()) sizes.push_back(std::max(rx1 - rx0, ry1 - ry0));
        }
        double gap = 1.0;   // no outline anywhere: a metre
        if (not sizes.empty())
        {
            std::ranges::nth_element(sizes, sizes.begin() + sizes.size() / 2);
            gap = sizes[sizes.size() / 2];
        }
        std::vector<Eigen::Vector2d> shift(component, Eigen::Vector2d::Zero());
        double cursor = boxes[0].x1 + gap;
        for (int c = 1; c < component; ++c)
        {
            shift[c] = {cursor - boxes[c].x0, boxes[0].y0 - boxes[c].y0};
            cursor += (boxes[c].x1 - boxes[c].x0) + gap;
        }
        for (auto &r : map.rooms)
        {
            r.world.translation().head<2>() += shift[r.component];
            r.world_raw.translation().head<2>() += shift[r.component];
        }
    }

    // ── World geometry ─────────────────────────────────────────────────────────────────────────
    for (const auto &[i, r] : std::views::enumerate(map.rooms))
    {
        r.outline.clear();
        r.outline_raw.clear();
        for (const auto &q : local[i]) r.outline.push_back(apply(r.world, q));
        for (const auto &q : local[i]) r.outline_raw.push_back(apply(r.world_raw, q));
        r.centre = local[i].empty() ? apply(r.world, Eigen::Vector2d::Zero()) : centroid(r.outline);
    }
    for (const auto &d : doors)
    {
        const auto &w = map.rooms[d.room].world;
        map.doors.push_back({.name = d.name, .room = d.room, .a = apply(w, d.a), .b = apply(w, d.b),
                             .centre = apply(w, d.centre)});
    }

    // ── Loops: a passage the spanning tree did not use. Its two sides' door centres disagree by the
    //    accumulated seam error; reported, never averaged away. ──────────────────────────────────
    for (const auto &[k, p] : std::views::enumerate(map.passages))
    {
        if (not p.seam_ok or tree_edge[k]) continue;
        p.loop = true;
        if (p.door_a.has_value() and p.door_b.has_value())
            p.closure_m = (map.doors[*p.door_a].centre - map.doors[*p.door_b].centre).norm();
    }
    return map;
}

}   // namespace ltsm::topo
