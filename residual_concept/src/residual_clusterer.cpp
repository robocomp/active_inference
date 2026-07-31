/*
 * residual_clusterer.cpp  —  residual-concept perception front-end. See residual_clusterer.h.
 */

#include "residual_clusterer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <random>
#include <unordered_map>

namespace rc
{

// ─── 3D oriented-box SDF (utility for building SpecialistSdf from box nodes) ─────
float ResidualClusterer::box_sdf_3d(const Eigen::Vector3f& p, const Eigen::Vector3f& centre, float yaw,
                                    float hx, float hy, float hz)
{
    const float c = std::cos(yaw), s = std::sin(yaw);
    const float dx = p.x() - centre.x(), dy = p.y() - centre.y(), dz = p.z() - centre.z();
    const float lx =  c * dx + s * dy;   // R(−yaw)
    const float ly = -s * dx + c * dy;
    const float qx = std::abs(lx) - hx, qy = std::abs(ly) - hy, qz = std::abs(dz) - hz;
    const float ox = std::max(qx, 0.0f), oy = std::max(qy, 0.0f), oz = std::max(qz, 0.0f);
    const float outside = std::sqrt(ox * ox + oy * oy + oz * oz);
    const float inside  = std::min(std::max(qx, std::max(qy, qz)), 0.0f);
    return outside + inside;
}

// carve_box (occupancy/free-space evidence) moved to common/existence_belief/existence_belief.h

// ─── Point-in-polygon (ray cast) + distance to boundary ─────────────────────────
bool ResidualClusterer::point_in_polygon(const std::vector<Eigen::Vector2f>& poly, const Eigen::Vector2f& p)
{
    if (poly.size() < 3) return true;   // no polygon ⇒ don't reject
    bool inside = false;
    for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
    {
        const auto& a = poly[i]; const auto& b = poly[j];
        if (((a.y() > p.y()) != (b.y() > p.y())) and
            (p.x() < (b.x() - a.x()) * (p.y() - a.y()) / (b.y() - a.y() + 1e-12f) + a.x()))
            inside = not inside;
    }
    return inside;
}

float ResidualClusterer::dist_to_polygon_boundary(const std::vector<Eigen::Vector2f>& poly, const Eigen::Vector2f& p)
{
    if (poly.size() < 2) return std::numeric_limits<float>::max();
    float best = std::numeric_limits<float>::max();
    for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
    {
        const Eigen::Vector2f a = poly[j], b = poly[i], ab = b - a, ap = p - a;
        const float t = std::clamp(ap.dot(ab) / std::max(1e-12f, ab.squaredNorm()), 0.0f, 1.0f);
        best = std::min(best, (ap - t * ab).norm());
    }
    return best;
}

// ─── Robust infrastructure subtraction for a dense (ZED) cloud ──────────────────
std::vector<Eigen::Vector3f> ResidualClusterer::subtract_infrastructure(
    const std::vector<Eigen::Vector3f>& cloud_room, const Eigen::Vector3f& sensor_origin,
    const std::vector<Eigen::Vector2f>& room_polygon, const DepthInfraParams& p)
{
    std::vector<Eigen::Vector3f> keep;
    keep.reserve(cloud_room.size());
    const Eigen::Vector2f o = sensor_origin.head<2>();
    const bool have_poly = room_polygon.size() >= 3;
    for (const auto& q : cloud_room)
    {
        const float r = (q.head<2>() - o).norm();                 // horizontal range from the sensor
        const float sigma = p.sigma0_m + p.sigma_quad * r * r;    // stereo depth noise grows with range²
        const float band = p.k * sigma;
        const float floor_z = p.floor_a * q.x() + p.floor_b * q.y() + p.floor_c;   // fitted floor (0,0,0 ⇒ z=0)
        if (q.z() < floor_z + p.floor_z0 + band) continue;        // floor (range-growing band, on the real floor)
        if (q.z() > p.ceil_z - band)            continue;         // ceiling
        if (have_poly)
        {
            const Eigen::Vector2f xy = q.head<2>();
            if (not point_in_polygon(room_polygon, xy))           continue;   // outside the room ⇒ wall/beyond
            if (dist_to_polygon_boundary(room_polygon, xy) < p.wall_margin_m + band) continue;   // near a wall
        }
        keep.push_back(q);
    }
    return keep;
}

// ─── Residual filter ────────────────────────────────────────────────────────────
// The residual = every point NO explainer accounts for. All removal (floor/ceiling/walls/robot/objects) is
// now unified evidential model-explanation — no geometric-threshold masks. Each explainer owns its tolerance.
std::vector<Eigen::Vector3f> ResidualClusterer::residual_filter(
    const std::vector<Eigen::Vector3f>& room_points, const std::vector<SpecialistExplainer>& explainers) const
{
    std::vector<Eigen::Vector3f> out;
    out.reserve(room_points.size());
    for (const auto& p : room_points)
    {
        bool explained = false;
        for (const auto& ex : explainers)
            if (ex(p)) { explained = true; break; }
        if (not explained) out.push_back(p);
    }
    return out;
}

// ─── Grid-accelerated 3D DBSCAN ─────────────────────────────────────────────────
std::vector<std::vector<int>> ResidualClusterer::dbscan_3d(const std::vector<Eigen::Vector3f>& pts,
                                                           float eps, int min_pts)
{
    const int n = static_cast<int>(pts.size());
    std::vector<std::vector<int>> clusters;
    if (n == 0) return clusters;

    // Uniform grid (cell = eps) for O(1) neighbourhood queries → ~O(n) overall.
    const float inv = 1.0f / eps;
    const auto key = [&](const Eigen::Vector3f& p) -> std::int64_t {
        const std::int64_t ix = static_cast<std::int64_t>(std::floor(p.x() * inv));
        const std::int64_t iy = static_cast<std::int64_t>(std::floor(p.y() * inv));
        const std::int64_t iz = static_cast<std::int64_t>(std::floor(p.z() * inv));
        return (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791);   // spatial hash
    };
    std::unordered_map<std::int64_t, std::vector<int>> grid;
    grid.reserve(n * 2);
    for (int i = 0; i < n; ++i) grid[key(pts[i])].push_back(i);

    const float eps2 = eps * eps;
    const auto region = [&](int i) {
        std::vector<int> nb;
        const Eigen::Vector3f& p = pts[i];
        const std::int64_t bx = static_cast<std::int64_t>(std::floor(p.x() * inv));
        const std::int64_t by = static_cast<std::int64_t>(std::floor(p.y() * inv));
        const std::int64_t bz = static_cast<std::int64_t>(std::floor(p.z() * inv));
        for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy) for (int dz = -1; dz <= 1; ++dz)
        {
            const std::int64_t k = ((bx + dx) * 73856093) ^ ((by + dy) * 19349663) ^ ((bz + dz) * 83492791);
            const auto it = grid.find(k);
            if (it == grid.end()) continue;
            for (int j : it->second)
                if ((pts[j] - p).squaredNorm() <= eps2) nb.push_back(j);
        }
        return nb;
    };

    constexpr int UNVISITED = -1, NOISE = -2;
    std::vector<int> label(n, UNVISITED);
    for (int i = 0; i < n; ++i)
    {
        if (label[i] != UNVISITED) continue;
        std::vector<int> nb = region(i);
        if (static_cast<int>(nb.size()) < min_pts) { label[i] = NOISE; continue; }
        // start a new cluster; BFS the density-reachable set
        const int cid = static_cast<int>(clusters.size());
        clusters.emplace_back();
        label[i] = cid; clusters[cid].push_back(i);
        for (std::size_t q = 0; q < nb.size(); ++q)
        {
            const int j = nb[q];
            if (label[j] == NOISE) { label[j] = cid; clusters[cid].push_back(j); }   // border point
            if (label[j] != UNVISITED) continue;
            label[j] = cid; clusters[cid].push_back(j);
            std::vector<int> jnb = region(j);
            if (static_cast<int>(jnb.size()) >= min_pts)                             // j is a core point
                nb.insert(nb.end(), jnb.begin(), jnb.end());
        }
    }
    return clusters;
}

// ─── Convex hull (Andrew monotone chain) ────────────────────────────────────────
std::vector<Eigen::Vector2f> ResidualClusterer::convex_hull_2d(std::vector<Eigen::Vector2f> pts)
{
    const int n = static_cast<int>(pts.size());
    if (n < 3) return pts;
    std::sort(pts.begin(), pts.end(), [](const Eigen::Vector2f& a, const Eigen::Vector2f& b) {
        return a.x() < b.x() or (a.x() == b.x() and a.y() < b.y());
    });
    const auto cross = [](const Eigen::Vector2f& o, const Eigen::Vector2f& a, const Eigen::Vector2f& b) {
        return (a.x() - o.x()) * (b.y() - o.y()) - (a.y() - o.y()) * (b.x() - o.x());
    };
    std::vector<Eigen::Vector2f> h(2 * n);
    int k = 0;
    for (int i = 0; i < n; ++i) {                                   // lower hull
        while (k >= 2 and cross(h[k - 2], h[k - 1], pts[i]) <= 0) k--;
        h[k++] = pts[i];
    }
    for (int i = n - 2, t = k + 1; i >= 0; --i) {                   // upper hull
        while (k >= t and cross(h[k - 2], h[k - 1], pts[i]) <= 0) k--;
        h[k++] = pts[i];
    }
    h.resize(k - 1);
    return h;
}

// ─── PCA oriented-box seed + hull + z-band ──────────────────────────────────────
ResidualCluster ResidualClusterer::make_cluster(const std::vector<Eigen::Vector3f>& pts)
{
    ResidualCluster c;
    c.points = pts;
    if (pts.empty()) return c;

    Eigen::Vector2f mean = Eigen::Vector2f::Zero();
    float zmin = pts[0].z(), zmax = pts[0].z();
    std::vector<Eigen::Vector2f> xy; xy.reserve(pts.size());
    for (const auto& p : pts) { mean += p.head<2>(); xy.push_back(p.head<2>()); zmin = std::min(zmin, p.z()); zmax = std::max(zmax, p.z()); }
    mean /= static_cast<float>(pts.size());
    c.z_min = zmin; c.z_max = zmax;

    Eigen::Matrix2f cov = Eigen::Matrix2f::Zero();
    for (const auto& q : xy) { const Eigen::Vector2f e = q - mean; cov += e * e.transpose(); }
    cov /= static_cast<float>(xy.size());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> es(cov);
    Eigen::Vector2f u1 = es.eigenvectors().col(1);   // principal (largest eigenvalue)
    Eigen::Vector2f u2 = es.eigenvectors().col(0);
    c.yaw_seed = std::atan2(u1.y(), u1.x());

    // OBB extents = span of projections onto the principal axes; OBB centre = mean shifted to the span mid.
    float a1min = 1e9f, a1max = -1e9f, a2min = 1e9f, a2max = -1e9f;
    for (const auto& q : xy) {
        const Eigen::Vector2f e = q - mean;
        const float a1 = e.dot(u1), a2 = e.dot(u2);
        a1min = std::min(a1min, a1); a1max = std::max(a1max, a1);
        a2min = std::min(a2min, a2); a2max = std::max(a2max, a2);
    }
    c.w_seed = std::max(0.05f, a1max - a1min);
    c.d_seed = std::max(0.05f, a2max - a2min);
    c.centroid = mean + u1 * (0.5f * (a1max + a1min)) + u2 * (0.5f * (a2max + a2min));

    c.hull = convex_hull_2d(xy);
    return c;
}

// ─── Navigation fragment-merge ──────────────────────────────────────────────────
// Two obstacles closer than the robot can pass between are ONE obstacle for navigation. Union clusters whose
// closest footprint (xy) points are within gap_m (transitive, via union-find) and re-fit one box+hull to each
// group. Works on the cluster geometry alone (LiDAR-only), so it holds every cycle regardless of ZED.
std::vector<ResidualCluster> ResidualClusterer::merge_close_clusters(std::vector<ResidualCluster> clusters,
                                                                     float gap_m)
{
    const int n = static_cast<int>(clusters.size());
    if (gap_m <= 0.0f or n < 2) return clusters;

    // Per-cluster xy AABB for a cheap far-pair reject before the O(nA·nB) exact test.
    struct Aabb { float xmin, xmax, ymin, ymax; };
    std::vector<Aabb> bb(n);
    for (int i = 0; i < n; ++i)
    {
        Aabb a{1e9f, -1e9f, 1e9f, -1e9f};
        for (const auto& p : clusters[i].points)
        {
            a.xmin = std::min(a.xmin, p.x()); a.xmax = std::max(a.xmax, p.x());
            a.ymin = std::min(a.ymin, p.y()); a.ymax = std::max(a.ymax, p.y());
        }
        bb[i] = a;
    }
    const auto aabb_gap = [](const Aabb& a, const Aabb& b) {
        const float dx = std::max({0.0f, a.xmin - b.xmax, b.xmin - a.xmax});
        const float dy = std::max({0.0f, a.ymin - b.ymax, b.ymin - a.ymax});
        return std::hypot(dx, dy);
    };
    // Minimum footprint (xy) distance between two clusters, early-out once below gap.
    const float gap2 = gap_m * gap_m;
    const auto close = [&](int i, int j) {
        if (aabb_gap(bb[i], bb[j]) > gap_m) return false;
        for (const auto& pa : clusters[i].points)
            for (const auto& pb : clusters[j].points)
                if ((pa.head<2>() - pb.head<2>()).squaredNorm() <= gap2) return true;
        return false;
    };

    // Union-find over cluster indices.
    std::vector<int> parent(n);
    for (int i = 0; i < n; ++i) parent[i] = i;
    const std::function<int(int)> find = [&](int x) { while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (find(i) != find(j) and close(i, j))
                parent[find(i)] = find(j);

    // Regroup members by root and re-fit one cluster per group.
    std::unordered_map<int, std::vector<Eigen::Vector3f>> groups;
    for (int i = 0; i < n; ++i)
    {
        auto& g = groups[find(i)];
        g.insert(g.end(), clusters[i].points.begin(), clusters[i].points.end());
    }
    std::vector<ResidualCluster> out;
    out.reserve(groups.size());
    for (auto& [root, pts] : groups)
        out.push_back(make_cluster(pts));
    return out;
}

// ─── Full pipeline ──────────────────────────────────────────────────────────────
std::vector<ResidualCluster> ResidualClusterer::extract(
    const std::vector<Eigen::Vector3f>& room_points, const std::vector<SpecialistExplainer>& explainers) const
{
    const std::vector<Eigen::Vector3f> residual = residual_filter(room_points, explainers);
    // Anisotropic-z clustering: compress z so vertically-separated layers of ONE object connect (a pillar's
    // rings, a surface above its body) instead of splitting into per-layer sheets. Cluster on the scaled
    // copy; build the cluster geometry from the ORIGINAL points (true z-range/footprint).
    std::vector<Eigen::Vector3f> scaled = residual;
    for (auto& p : scaled) p.z() *= params_.dbscan_z_scale;
    const std::vector<std::vector<int>> idx = dbscan_3d(scaled, params_.dbscan_eps_m, params_.dbscan_min_pts);
    std::vector<ResidualCluster> out;
    for (const auto& members : idx)
    {
        if (static_cast<int>(members.size()) < params_.min_cluster_pts) continue;
        std::vector<Eigen::Vector3f> cp; cp.reserve(members.size());
        for (int i : members) cp.push_back(residual[i]);
        ResidualCluster c = make_cluster(cp);
        // Reject only FLAT + LOW sheets (floor smears). An elevated flat sheet (a tabletop/shelf) is a real
        // obstacle and is kept — a downward LiDAR sees a table mostly as its flat top.
        if ((c.z_max - c.z_min) < params_.min_vertical_extent_m and c.z_max < params_.min_top_z_m) continue;
        out.push_back(std::move(c));
    }
    return out;
}

// ─── Self-test: the ON-TABLE + FISSION scenario ─────────────────────────────────
bool ResidualClusterer::self_test()
{
    std::mt19937 rng(7);
    std::normal_distribution<float> nz(0.0f, 0.012f);   // ~1.2 cm LiDAR noise
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    bool ok = true;
    auto check = [&](bool c, const char* m) { if (!c) { ok = false; std::printf("  FAIL: %s\n", m); } };

    // Table: footprint centre (0,0), half (0.60, 0.40); tabletop slab z∈[0.70,0.74]; 4 legs at the corners.
    const float hx = 0.60f, hy = 0.40f, top = 0.74f, slab_h = 0.02f;   // half-thickness
    auto add_box = [&](std::vector<Eigen::Vector3f>& v, float cx, float cy, float x0, float x1,
                       float y0, float y1, float z0, float z1, int n) {
        for (int i = 0; i < n; ++i)
            v.push_back({cx + x0 + (x1 - x0) * U(rng) + nz(rng),
                         cy + y0 + (y1 - y0) * U(rng) + nz(rng),
                         z0 + (z1 - z0) * U(rng) + nz(rng)});
    };

    std::vector<Eigen::Vector3f> cloud;
    // tabletop slab surface (sampled as a thin slab)
    add_box(cloud, 0, 0, -hx, hx, -hy, hy, top - 2 * slab_h, top, 900);
    // legs
    for (float sx : {-1.f, 1.f}) for (float sy : {-1.f, 1.f})
        add_box(cloud, sx * (hx - 0.05f), sy * (hy - 0.05f), -0.03f, 0.03f, -0.03f, 0.03f, 0.0f, top, 120);
    // 4 chairs pushed against the 4 edges (each a dense blob spanning floor→~0.9, touching the table edge)
    add_box(cloud, 0.0f,  -0.55f, -0.20f, 0.20f, -0.13f, 0.13f, 0.0f, 0.90f, 350);   // −y
    add_box(cloud, 0.0f,   0.55f, -0.20f, 0.20f, -0.13f, 0.13f, 0.0f, 0.90f, 350);   // +y
    add_box(cloud, -0.75f, 0.0f,  -0.13f, 0.13f, -0.20f, 0.20f, 0.0f, 0.90f, 350);   // −x
    add_box(cloud,  0.75f, 0.0f,  -0.13f, 0.13f, -0.20f, 0.20f, 0.0f, 0.90f, 350);   // +x
    // an UNKNOWN OBJECT resting ON the tabletop (a ~16 cm bottle at (0.2,0.1), bottom flush on the slab)
    add_box(cloud, 0.20f, 0.10f, -0.04f, 0.04f, -0.04f, 0.04f, top, top + 0.16f, 140);
    // floor returns (rejected by the z-band)
    add_box(cloud, 0.0f, 0.0f, -1.5f, 1.5f, -1.5f, 1.5f, 0.0f, 0.03f, 400);

    ResidualClusterParams P;   // defaults: eps 0.12, explain 0.04, min_cluster_pts 8
    ResidualClusterer clus(P);
    const float em = P.explain_margin_m;
    // ROOM FLOOR explainer (fixed 10 cm band for the test): a floor return is explained by the room floor.
    const SpecialistExplainer floor = [](const Eigen::Vector3f& p) { return p.z() < 0.10f; };

    // ── (1) only the floor is modelled → table+chairs+cup are one connected residual (one big obstacle) ──
    {
        const auto cl = clus.extract(cloud, {floor});
        std::printf("  floor-only: %zu cluster(s)\n", cl.size());
        check(cl.size() == 1, "unmodelled scene should be ONE big residual cluster");
    }

    // ── (2) FLOOR + TABLE modelled → the table SDF (slab + legs) subtracts the connecting mass; the residual
    //        FISSIONS into the 4 now-disconnected chairs, and the on-table object SURVIVES (above the slab). ──
    {
        SpecialistSdf table_sdf = [=](const Eigen::Vector3f& p) {
            float d = box_sdf_3d(p, {0, 0, top - slab_h}, 0.0f, hx, hy, slab_h);   // slab
            for (float sx : {-1.f, 1.f}) for (float sy : {-1.f, 1.f})              // legs
                d = std::min(d, box_sdf_3d(p, {sx * (hx - 0.05f), sy * (hy - 0.05f), 0.5f * top},
                                           0.0f, 0.03f, 0.03f, 0.5f * top));
            return d;
        };
        const SpecialistExplainer table = [table_sdf, em](const Eigen::Vector3f& p) { return std::abs(table_sdf(p)) < em; };
        const auto cl = clus.extract(cloud, {floor, table});
        std::printf("  table-modelled: %zu cluster(s)\n", cl.size());

        int on_table = 0, chairs = 0;
        for (const auto& c : cl) {
            const bool inside_fp = std::abs(c.centroid.x()) < hx and std::abs(c.centroid.y()) < hy;
            if (inside_fp and c.z_min > top - 0.02f) ++on_table;    // above the slab, over the footprint
            else if (not inside_fp) ++chairs;
        }
        std::printf("  → chairs=%d on-table=%d\n", chairs, on_table);
        check(cl.size() >= 5, "table subtraction should FISSION into ≥5 clusters (4 chairs + on-table)");
        check(chairs == 4, "the four chairs should re-cluster as four separate residuals");
        check(on_table == 1, "the on-table object must SURVIVE (above the tabletop SDF, not swallowed)");
    }

    // ── (3) hull + seed sanity on a lone box cluster ──
    {
        std::vector<Eigen::Vector3f> box;
        add_box(box, 1.0f, 1.0f, -0.30f, 0.30f, -0.10f, 0.10f, 0.20f, 0.60f, 500);
        const auto c = make_cluster(box);
        std::printf("  lone box: centroid=(%.2f,%.2f) w=%.2f d=%.2f z=[%.2f,%.2f] hull=%zu\n",
                    c.centroid.x(), c.centroid.y(), c.w_seed, c.d_seed, c.z_min, c.z_max, c.hull.size());
        check(std::abs(c.centroid.x() - 1.0f) < 0.05f and std::abs(c.centroid.y() - 1.0f) < 0.05f, "OBB centre off");
        check(std::abs(c.w_seed - 0.60f) < 0.10f, "OBB principal extent (w) off");
        check(std::abs(c.d_seed - 0.20f) < 0.10f, "OBB secondary extent (d) off");
        check(c.hull.size() >= 4, "convex hull should have ≥4 vertices");
    }

    // ── (4) navigation fragment-merge: shattered tabletop arcs coalesce; a far object stays separate ──
    {
        std::vector<Eigen::Vector3f> arcs;
        // Three tabletop "ring-arc" fragments at z≈0.72, each ~0.30 m long, edge gaps ~0.35 m: ABOVE the
        // DBSCAN eps (0.20) so they cluster SEPARATELY, but BELOW the robot-width merge gap (0.60) so the
        // post-pass coalesces them. Centres 0.65 m apart → arcs [-0.80,-0.50],[-0.15,0.15],[0.50,0.80].
        add_box(arcs, -0.65f, 0.0f, -0.15f, 0.15f, -0.03f, 0.03f, 0.71f, 0.73f, 120);
        add_box(arcs,  0.00f, 0.0f, -0.15f, 0.15f, -0.03f, 0.03f, 0.71f, 0.73f, 120);
        add_box(arcs,  0.65f, 0.0f, -0.15f, 0.15f, -0.03f, 0.03f, 0.71f, 0.73f, 120);
        // A separate object 1.5 m away (gap ≫ robot width) — must NOT merge.
        add_box(arcs,  2.0f, 0.0f, -0.10f, 0.10f, -0.10f, 0.10f, 0.20f, 0.60f, 200);

        ResidualClusterParams Q; Q.min_top_z_m = 0.30f; Q.min_vertical_extent_m = 0.10f;
        ResidualClusterer cq(Q);
        const SpecialistExplainer none = [](const Eigen::Vector3f&) { return false; };
        auto raw = cq.extract(arcs, {none});
        std::printf("  frag-merge: %zu raw cluster(s)\n", raw.size());
        const auto merged = merge_close_clusters(raw, 0.60f);   // robot width + excess
        std::printf("  frag-merge: %zu after merge (gap 0.60)\n", merged.size());
        check(raw.size() >= 3, "the three arcs + far object should start as ≥4 raw clusters");
        check(merged.size() == 2, "the 3 arcs coalesce into 1 obstacle; the far object stays separate → 2");
        // The coalesced obstacle must span all three arcs (w ≳ 0.8 m).
        float wmax = 0.0f; for (const auto& c : merged) wmax = std::max(wmax, std::max(c.w_seed, c.d_seed));
        check(wmax > 0.75f, "the merged tabletop obstacle should span all three arcs");
    }

    // ── (5) free-space carve: occupancy (+), see-through (−), occluded (no evidence) ──
    {
        const Eigen::Vector3f origin(0.0f, 0.0f, 0.5f);
        const float bx = 2.0f, by = 0.0f, byaw = 0.0f, bw = 0.4f, bd = 0.6f, zmin = 0.2f, zmax = 0.8f;
        CarveParams cp;               // sensor defaults (detection/clutter rates)
        const float surf_sigma = 0.03f;

        // OCCUPANCY: returns on the near face (x≈1.8) → box still there → dL>0.
        std::vector<Eigen::Vector3f> occ;
        for (int i = 0; i < 60; ++i) occ.push_back({1.80f + nz(rng), -0.28f + 0.56f * U(rng), 0.25f + 0.5f * U(rng)});
        const auto e_occ = carve_box(origin, occ, bx, by, byaw, bw, bd, zmin, zmax, surf_sigma, cp);
        std::printf("  carve occ:  dL=%+.2f e_occ=%.1f e_free=%.1f reached=%d\n",
                    e_occ.log_odds_delta, e_occ.e_occ, e_occ.e_free, e_occ.n_reached);
        check(e_occ.log_odds_delta > 0.5f, "returns on the box face are OCCUPANCY evidence (dL>0)");

        // SEE-THROUGH: box empty, returns from the background (x≈4) along bearings that cross it → dL<0.
        std::vector<Eigen::Vector3f> thru;
        for (int i = 0; i < 60; ++i) thru.push_back({4.0f + nz(rng), -0.10f + 0.20f * U(rng), 0.40f + 0.2f * U(rng)});
        const auto e_thru = carve_box(origin, thru, bx, by, byaw, bw, bd, zmin, zmax, surf_sigma, cp);
        std::printf("  carve thru: dL=%+.2f e_occ=%.1f e_free=%.1f reached=%d\n",
                    e_thru.log_odds_delta, e_thru.e_occ, e_thru.e_free, e_thru.n_reached);
        check(e_thru.log_odds_delta < -0.5f, "beams passing THROUGH the box are see-through evidence (dL<0)");

        // OCCLUDED: returns stop SHORT (x≈1.0) → the box could still be behind → no evidence, HOLD.
        std::vector<Eigen::Vector3f> occl;
        for (int i = 0; i < 60; ++i) occl.push_back({1.0f + nz(rng), -0.05f + 0.10f * U(rng), 0.5f});
        const auto e_occl = carve_box(origin, occl, bx, by, byaw, bw, bd, zmin, zmax, surf_sigma, cp);
        std::printf("  carve occl: dL=%+.2f reached=%d\n", e_occl.log_odds_delta, e_occl.n_reached);
        check(e_occl.n_reached == 0 and e_occl.log_odds_delta == 0.0f, "occluded beams give NO evidence (HOLD)");
    }

    // ── (6) dense-cloud infrastructure subtraction: floor + wall removed (even with a z-offset), object kept ──
    {
        const Eigen::Vector3f sensor(0.0f, 0.0f, 0.5f);
        // Room = 6×6 square centred at origin.
        const std::vector<Eigen::Vector2f> poly = {{-3,-3},{3,-3},{3,3},{-3,3}};
        std::vector<Eigen::Vector3f> zed;
        // A dense FLOOR sheet, LIFTED by a 0.12 m calibration offset (z≈0.12) across the room out to 4 m range.
        for (int i = 0; i < 400; ++i) zed.push_back({-2.5f + 5.0f * U(rng), -2.5f + 5.0f * U(rng), 0.12f + nz(rng)});
        // A dense WALL sheet along x=+3 (the room's east wall), z up to 1.5.
        for (int i = 0; i < 200; ++i) zed.push_back({3.0f + nz(rng), -2.5f + 5.0f * U(rng), 1.5f * U(rng)});
        // A real OBJECT in the middle: a 0.3 m box at (1,0), z 0.15..0.5.
        int obj_in = 0;
        for (int i = 0; i < 150; ++i) { zed.push_back({1.0f - 0.15f + 0.3f * U(rng), -0.15f + 0.3f * U(rng), 0.15f + 0.35f * U(rng)}); ++obj_in; }

        DepthInfraParams P;                                  // σ0=0.03, quad=0.006, k=3 → floor band ≫ 0.12 offset
        const auto kept = subtract_infrastructure(zed, sensor, poly, P);
        int obj_kept = 0, infra_kept = 0;
        for (const auto& q : kept)
            (std::abs(q.x() - 1.0f) < 0.3f and std::abs(q.y()) < 0.3f) ? ++obj_kept : ++infra_kept;
        std::printf("  infra-subtract: %zu kept (object=%d, floor/wall leak=%d) of %zu\n",
                    kept.size(), obj_kept, infra_kept, zed.size());
        check(obj_kept > 100, "the real object must SURVIVE infrastructure subtraction");
        check(infra_kept < 20, "the lifted floor + wall must be REMOVED (no bridging sheet leaks)");
    }

    std::printf("ResidualClusterer::self_test %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

}  // namespace rc
