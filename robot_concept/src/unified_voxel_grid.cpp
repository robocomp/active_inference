#include "unified_voxel_grid.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>
#include <execution>
#include <numeric>
#include <queue>
#include <random>

// ═══════════════════════════════════════════════════════════════════════════
//  CategoryRegistry
// ═══════════════════════════════════════════════════════════════════════════

CategoryRegistry::CategoryRegistry(std::span<const char* const> cats)
{
    for (const char* c : cats)
        register_category(c);
}

int CategoryRegistry::register_category(const std::string& name)
{
    if (auto it = _n2i.find(name); it != _n2i.end())
        return it->second;
    const int i = static_cast<int>(_n2i.size());
    _n2i[name] = i;
    _i2n[i]    = name;
    return i;
}

int CategoryRegistry::idx(const std::string& name)
{
    return register_category(name);
}

int CategoryRegistry::idx_or(const std::string& name, int fallback) const
{
    auto it = _n2i.find(name);
    return (it != _n2i.end()) ? it->second : fallback;
}

std::string CategoryRegistry::name(int i) const
{
    if (auto it = _i2n.find(i); it != _i2n.end())
        return it->second;
    return "unknown";
}

std::vector<std::string> CategoryRegistry::names() const
{
    std::vector<std::string> out;
    out.reserve(_i2n.size());
    for (int i = 0; i < static_cast<int>(_i2n.size()); ++i)
        out.push_back(name(i));
    return out;
}

std::map<std::string, float>
CategoryRegistry::belief_to_dict(std::span<const float> alpha) const
{
    std::map<std::string, float> out;
    const float s = std::reduce(alpha.begin(), alpha.end(), 0.0f);
    const float inv_s = (s > 0.0f) ? 1.0f / s : 0.0f;
    for (int i = 0; i < std::min(static_cast<int>(alpha.size()), K()); ++i)
        out[name(i)] = alpha[i] * inv_s;
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════
//  UnifiedVoxelGrid — construction
// ═══════════════════════════════════════════════════════════════════════════

UnifiedVoxelGrid::UnifiedVoxelGrid(UnifiedGridConfig cfg, CategoryRegistry reg)
    : _cfg(std::move(cfg)), _reg(std::move(reg)), _inv(1.0f / _cfg.resolution)
{
    _grid.reserve(static_cast<std::size_t>(_cfg.max_voxels));
    _track_voxel_count.reserve(256);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Private helpers
// ═══════════════════════════════════════════════════════════════════════════

VoxelKey UnifiedVoxelGrid::make_key(const Eigen::Vector3f& pt) const noexcept
{
    return {static_cast<int32_t>(std::floor(pt.x() * _inv)),
            static_cast<int32_t>(std::floor(pt.y() * _inv)),
            static_cast<int32_t>(std::floor(pt.z() * _inv))};
}

bool UnifiedVoxelGrid::_is_categorized(const VoxelState& vs) const noexcept
{
    return vs.n_frames_seen >= _cfg.min_frames_to_confirm;
}

void UnifiedVoxelGrid::_track_inc(int track_id) noexcept
{
    ++_track_voxel_count[track_id];
}

int UnifiedVoxelGrid::_track_dec(int track_id) noexcept
{
    auto it = _track_voxel_count.find(track_id);
    if (it == _track_voxel_count.end()) return 0;
    it->second = std::max(0, it->second - 1);
    return it->second;
}

void UnifiedVoxelGrid::_reset_frame_bookkeeping()
{
    _touched_this_frame.clear();
    _observed_pts_this_frame.clear();
}

void UnifiedVoxelGrid::_enforce_max_voxels(int /*frame*/)
{
    if (static_cast<int>(_grid.size()) <= _cfg.max_voxels) return;

    // Evict up to 5 % of max_voxels starting from oldest observations.
    // Simple approach: collect all keys, sort by last_frame, delete oldest.
    const int to_evict = _cfg.max_voxels / 20;
    std::vector<std::pair<int,VoxelKey>> age_idx;
    age_idx.reserve(_grid.size());
    for (const auto& [k, vs] : _grid)
        age_idx.emplace_back(vs.last_frame, k);

    std::partial_sort(age_idx.begin(),
                      age_idx.begin() + to_evict,
                      age_idx.end(),
                      [](const auto& a, const auto& b){ return a.first < b.first; });

    for (int i = 0; i < to_evict; ++i)
    {
        auto it = _grid.find(age_idx[i].second);
        if (it != _grid.end())
        {
            _track_dec(it->second.track_id);
            _grid.erase(it);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  1. observe
// ═══════════════════════════════════════════════════════════════════════════

void UnifiedVoxelGrid::observe(int track_id,
                                std::span<const Eigen::Vector3f> pts,
                                const std::string& category,
                                int frame,
                                std::span<const std::string> per_point_labels,
                                std::span<const float>        confidences,
                                float detection_confidence)
{
    if (pts.empty()) return;

    const int  K       = _reg.K();
    const int  cat_idx = _reg.idx(category);
    const int  cap     = _cfg.max_voxels_per_track;

    for (std::size_t i = 0; i < pts.size(); ++i)
    {
        const VoxelKey key = make_key(pts[i]);
        const float    w   = confidences.empty() ? 1.0f : confidences[i];

        auto it = _grid.find(key);
        if (it == _grid.end())
        {
            // ── cap per track ───────────────────────────────────────────
            if (cap > 0 && _track_voxel_count[track_id] >= cap)
            {
                _touched_this_frame.insert(key);
                continue;
            }
            VoxelState vs;
            vs.centroid  = pts[i];
            vs.alpha.assign(static_cast<std::size_t>(K), _cfg.alpha_prior);
            vs.track_id  = track_id;
            _grid.emplace(key, std::move(vs));
            _track_inc(track_id);
            it = _grid.find(key);
        }
        else
        {
            VoxelState& vs = it->second;
            vs.centroid = (1.0f - _cfg.centroid_ema) * vs.centroid
                        +          _cfg.centroid_ema  * pts[i];

            if (vs.track_id != track_id)
            {
                // Ownership change
                _track_dec(vs.track_id);
                _track_inc(track_id);
                const int new_K = std::max(K, static_cast<int>(vs.alpha.size()));
                vs.alpha.assign(static_cast<std::size_t>(new_K), _cfg.alpha_prior);
                vs.n_obs = vs.n_frames_seen = 0;
                vs.best_confidence = 0.0f; vs.best_cat_idx = -1;
                vs.track_id = track_id;
            }
            else if (vs.best_cat_idx != -1 && vs.best_cat_idx != cat_idx)
            {
                if (detection_confidence > vs.best_confidence)
                {
                    // Higher-confidence category wins — reset Dirichlet
                    vs.alpha.assign(vs.alpha.size(), _cfg.alpha_prior);
                    vs.n_obs = vs.n_frames_seen = 0;
                }
                else
                {
                    // Lower confidence: protect existing category
                    _touched_this_frame.insert(key);
                    continue;
                }
            }
        }

        VoxelState& vs = it->second;

        if (vs.last_frame != frame)
            ++vs.n_frames_seen;

        // Dirichlet conjugate update
        const int c_idx = per_point_labels.empty()
                        ? cat_idx
                        : _reg.idx(per_point_labels[i]);

        if (c_idx >= static_cast<int>(vs.alpha.size()))
            vs.alpha.resize(static_cast<std::size_t>(c_idx + 1), _cfg.alpha_prior);

        vs.alpha[static_cast<std::size_t>(c_idx)] += w;

        if (detection_confidence > vs.best_confidence)
        {
            vs.best_confidence = detection_confidence;
            vs.best_cat_idx    = c_idx;
        }

        ++vs.n_obs;
        vs.last_frame       = frame;
        vs.n_ray_traversals = 0;
        _touched_this_frame.insert(key);
    }

    // Subsample for ray traversal
    const std::size_t step = std::max(std::size_t{1},
                                       pts.size() / std::max(1, MAX_RAYS / 4));
    for (std::size_t i = 0; i < pts.size(); i += step)
        _observed_pts_this_frame.push_back(pts[i]);

    _frame = frame;
    _enforce_max_voxels(frame);
}

// ═══════════════════════════════════════════════════════════════════════════
//  AABB slab test — SoA + par_unseq inner loop
// ═══════════════════════════════════════════════════════════════════════════

std::unordered_set<VoxelKey, VoxelKeyHash>
UnifiedVoxelGrid::_aabb_traversed(const Eigen::Vector3f& cam,
                                   std::span<const Eigen::Vector3f> targets) const
{
    if (_grid.empty() || targets.empty())
        return {};

    const float hs = _cfg.resolution * 0.5f;

    // ── Build SoA snapshot of voxel centers ──────────────────────────────
    // Pre-filter: only voxels inside the AABB of all rays.
    Eigen::Vector3f ray_min = cam, ray_max = cam;
    for (const auto& t : targets)
    {
        ray_min = ray_min.cwiseMin(t);
        ray_max = ray_max.cwiseMax(t);
    }
    ray_min.array() -= hs;
    ray_max.array() += hs;

    std::vector<VoxelKey>       keys;
    std::vector<float>          cx, cy, cz;   // voxel center SoA
    keys.reserve(_grid.size());
    cx.reserve(_grid.size()); cy.reserve(_grid.size()); cz.reserve(_grid.size());

    for (const auto& [k, _] : _grid)
    {
        const float ox = (static_cast<float>(k.x) + 0.5f) * _cfg.resolution;
        const float oy = (static_cast<float>(k.y) + 0.5f) * _cfg.resolution;
        const float oz = (static_cast<float>(k.z) + 0.5f) * _cfg.resolution;
        if (ox < ray_min.x() || ox > ray_max.x()) continue;
        if (oy < ray_min.y() || oy > ray_max.y()) continue;
        if (oz < ray_min.z() || oz > ray_max.z()) continue;
        keys.push_back(k);
        cx.push_back(ox); cy.push_back(oy); cz.push_back(oz);
    }

    const std::size_t M = keys.size();
    if (M == 0) return {};

    // ── Build ray SoA ─────────────────────────────────────────────────────
    const std::size_t N = targets.size();
    std::vector<float> ro_x(N), ro_y(N), ro_z(N); // origin (constant = cam)
    std::vector<float> inv_dx(N), inv_dy(N), inv_dz(N), ray_len(N);

    for (std::size_t j = 0; j < N; ++j)
    {
        Eigen::Vector3f d = targets[j] - cam;
        const float len   = d.norm();
        ray_len[j]        = len;
        if (len < 1e-6f) { inv_dx[j] = inv_dy[j] = inv_dz[j] = 0.0f; continue; }
        d /= len;
        constexpr float eps_d = 1e-10f;
        inv_dx[j] = 1.0f / (std::abs(d.x()) > eps_d ? d.x() : std::copysign(eps_d, d.x() + 1e-30f));
        inv_dy[j] = 1.0f / (std::abs(d.y()) > eps_d ? d.y() : std::copysign(eps_d, d.y() + 1e-30f));
        inv_dz[j] = 1.0f / (std::abs(d.z()) > eps_d ? d.z() : std::copysign(eps_d, d.z() + 1e-30f));
    }

    // ── Per-voxel hit flag: parallel over M voxels ────────────────────────
    // `any_hit[m]` = true if ANY ray traverses (not just endpoints) voxel m.
    std::vector<uint8_t> any_hit(M, 0u);

    // Index array for par_unseq
    std::vector<std::size_t> midx(M);
    std::iota(midx.begin(), midx.end(), std::size_t{0});

    std::for_each(std::execution::par_unseq, midx.begin(), midx.end(),
        [&](std::size_t m)
        {
            const float lo_x = cx[m] - hs, hi_x = cx[m] + hs;
            const float lo_y = cy[m] - hs, hi_y = cy[m] + hs;
            const float lo_z = cz[m] - hs, hi_z = cz[m] + hs;

            for (std::size_t j = 0; j < N; ++j)
            {
                const float t1x = (lo_x - cam.x()) * inv_dx[j];
                const float t2x = (hi_x - cam.x()) * inv_dx[j];
                const float t1y = (lo_y - cam.y()) * inv_dy[j];
                const float t2y = (hi_y - cam.y()) * inv_dy[j];
                const float t1z = (lo_z - cam.z()) * inv_dz[j];
                const float t2z = (hi_z - cam.z()) * inv_dz[j];

                const float t_near = std::max({std::min(t1x, t2x),
                                               std::min(t1y, t2y),
                                               std::min(t1z, t2z)});
                const float t_far  = std::min({std::max(t1x, t2x),
                                               std::max(t1y, t2y),
                                               std::max(t1z, t2z)});

                if (t_near < t_far && t_far > 0.0f && t_near < ray_len[j] * 0.97f)
                {
                    any_hit[m] = 1u;
                    return;  // early-out for this voxel
                }
            }
        });

    std::unordered_set<VoxelKey, VoxelKeyHash> result;
    result.reserve(M / 4);
    for (std::size_t m = 0; m < M; ++m)
        if (any_hit[m])
            result.insert(keys[m]);

    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
//  2. visibility_update
// ═══════════════════════════════════════════════════════════════════════════

std::expected<VisibilityStats, std::string>
UnifiedVoxelGrid::visibility_update(const Eigen::Vector3f& cam_pos,
                                     const Eigen::Matrix3f& /*cam_R*/,
                                     int frame,
                                     std::span<const Eigen::Vector3f> scene_pts)
{
    VisibilityStats stats;

    // ── Choose ray targets ────────────────────────────────────────────────
    std::vector<Eigen::Vector3f> ray_targets;
    if (!scene_pts.empty())
    {
        if (static_cast<int>(scene_pts.size()) > MAX_RAYS)
        {
            ray_targets.reserve(MAX_RAYS);
            std::mt19937 rng{static_cast<uint32_t>(frame)};
            std::vector<int> idx(scene_pts.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::shuffle(idx.begin(), idx.end(), rng);
            for (int i = 0; i < MAX_RAYS; ++i)
                ray_targets.push_back(scene_pts[idx[i]]);
        }
        else
        {
            ray_targets.assign(scene_pts.begin(), scene_pts.end());
        }
    }
    else
    {
        auto& obs = _observed_pts_this_frame;
        if (obs.empty()) { _reset_frame_bookkeeping(); return stats; }
        if (static_cast<int>(obs.size()) > MAX_RAYS)
        {
            std::mt19937 rng{static_cast<uint32_t>(frame)};
            std::shuffle(obs.begin(), obs.end(), rng);
            obs.resize(MAX_RAYS);
        }
        ray_targets = obs;
    }

    stats.n_rays_cast = static_cast<int>(ray_targets.size());

    // ── Endpoints ─────────────────────────────────────────────────────────
    std::unordered_set<VoxelKey, VoxelKeyHash> endpoints;
    endpoints.reserve(ray_targets.size());
    for (const auto& t : ray_targets)
    {
        VoxelKey k = make_key(t);
        if (_grid.contains(k))
            endpoints.insert(k);
    }

    // ── Traversed intermediates (AABB slab, par_unseq) ────────────────────
    auto intermediate = _aabb_traversed(cam_pos, ray_targets);

    const int thr = _cfg.traversal_delete_threshold;
    std::unordered_set<VoxelKey, VoxelKeyHash> to_delete;

    // ── Phase 1: endpoints ────────────────────────────────────────────────
    for (const auto& key : endpoints)
    {
        if (_touched_this_frame.contains(key)) continue;
        auto it = _grid.find(key);
        if (it == _grid.end()) continue;
        VoxelState& vs = it->second;

        if (!_is_categorized(vs))
        {
            to_delete.insert(key);
            ++stats.n_endpoint_no_cat_deleted;
        }
        else
        {
            vs.n_ray_traversals = 0;
            ++stats.n_endpoint_kept;
        }
    }

    // ── Phase 2: intermediates ────────────────────────────────────────────
    for (const auto& key : intermediate)
    {
        if (_touched_this_frame.contains(key)) continue;
        if (endpoints.contains(key)) continue;
        auto it = _grid.find(key);
        if (it == _grid.end()) continue;
        VoxelState& vs = it->second;

        if (!_is_categorized(vs))
        {
            to_delete.insert(key);
            ++stats.n_intermediate_deleted;
        }
        else
        {
            ++vs.n_ray_traversals;
            if (vs.n_ray_traversals >= thr)
            {
                to_delete.insert(key);
                ++stats.n_intermediate_deleted;
            }
            else
            {
                ++stats.n_intermediate_traversal_inc;
            }
        }
    }

    for (const auto& key : to_delete)
    {
        auto it = _grid.find(key);
        if (it != _grid.end())
        {
            _track_dec(it->second.track_id);
            _grid.erase(it);
        }
    }

    _reset_frame_bookkeeping();
    return stats;
}

// ═══════════════════════════════════════════════════════════════════════════
//  DBSCAN — brute-force SoA + bitmatrix adjacency + par_unseq
// ═══════════════════════════════════════════════════════════════════════════

std::vector<int> UnifiedVoxelGrid::_dbscan(std::span<const Eigen::Vector3f> pts,
                                             float eps, int min_samples) const
{
    const std::size_t N = pts.size();
    if (N == 0) return {};

    const float eps2 = eps * eps;

    // ── Build packed bit-adjacency matrix: adj[i*stride + j/8] bit(j%8) ──
    // Upper-triangle + diagonal. Row i's neighbours are columns where bit=1.
    const std::size_t stride = (N + 7u) / 8u;  // bytes per row
    std::vector<uint8_t> adj(N * stride, 0u);

    // Extract SoA for vectorizer
    std::vector<float> px(N), py(N), pz(N);
    for (std::size_t i = 0; i < N; ++i)
    {
        px[i] = pts[i].x(); py[i] = pts[i].y(); pz[i] = pts[i].z();
    }

    // Parallel over rows
    std::vector<std::size_t> rows(N);
    std::iota(rows.begin(), rows.end(), std::size_t{0});

    std::for_each(std::execution::par_unseq, rows.begin(), rows.end(),
        [&](std::size_t i)
        {
            uint8_t* row = adj.data() + i * stride;
            const float xi = px[i], yi = py[i], zi = pz[i];
            for (std::size_t j = 0; j < N; ++j)
            {
                const float dx = xi - px[j];
                const float dy = yi - py[j];
                const float dz = zi - pz[j];
                if (dx*dx + dy*dy + dz*dz <= eps2)
                    row[j / 8u] |= static_cast<uint8_t>(1u << (j % 8u));
            }
        });

    // ── Count neighbours (popcount over each row) ─────────────────────────
    std::vector<int> n_nbrs(N, 0);
    for (std::size_t i = 0; i < N; ++i)
    {
        const uint8_t* row = adj.data() + i * stride;
        int cnt = 0;
        for (std::size_t b = 0; b < stride; ++b)
            cnt += std::popcount(row[b]);
        n_nbrs[i] = cnt;
    }

    // ── BFS cluster assignment ────────────────────────────────────────────
    std::vector<int> labels(N, -1);
    int cluster_id = 0;
    std::queue<std::size_t> q;

    for (std::size_t i = 0; i < N; ++i)
    {
        if (labels[i] != -1) continue;
        if (n_nbrs[i] < min_samples) continue;  // not a core point

        labels[i] = cluster_id;
        q.push(i);

        while (!q.empty())
        {
            const std::size_t cur = q.front(); q.pop();
            const uint8_t* row = adj.data() + cur * stride;
            for (std::size_t j = 0; j < N; ++j)
            {
                if (!(row[j / 8u] & (1u << (j % 8u)))) continue;
                if (labels[j] != -1) continue;
                labels[j] = cluster_id;
                if (n_nbrs[j] >= min_samples)
                    q.push(j);
            }
        }
        ++cluster_id;
    }

    return labels;
}

// ═══════════════════════════════════════════════════════════════════════════
//  3. Point-cloud queries
// ═══════════════════════════════════════════════════════════════════════════

std::vector<Eigen::Vector3f> UnifiedVoxelGrid::get_points(int track_id) const
{
    std::vector<Eigen::Vector3f> out;
    for (const auto& [k, vs] : _grid)
        if (vs.track_id == track_id)
            out.push_back(vs.centroid);
    return out;
}

std::optional<Eigen::Vector3f> UnifiedVoxelGrid::get_extent(int track_id) const
{
    auto pts = get_points(track_id);
    if (pts.size() < 3) return std::nullopt;
    Eigen::Vector3f mn = pts[0], mx = pts[0];
    for (const auto& p : pts) { mn = mn.cwiseMin(p); mx = mx.cwiseMax(p); }
    return mx - mn;
}

std::optional<float> UnifiedVoxelGrid::get_z_centroid(int track_id) const
{
    auto pts = get_points(track_id);
    if (pts.size() < 3) return std::nullopt;
    std::vector<float> zs;
    zs.reserve(pts.size());
    for (const auto& p : pts) zs.push_back(p.z());
    std::nth_element(zs.begin(), zs.begin() + zs.size()/2, zs.end());
    return zs[zs.size() / 2];
}

int UnifiedVoxelGrid::get_n_voxels(int track_id) const noexcept
{
    auto it = _track_voxel_count.find(track_id);
    return (it != _track_voxel_count.end()) ? it->second : 0;
}

std::vector<Eigen::Vector3f>
UnifiedVoxelGrid::get_filtered_points(int track_id,
                                       const std::string& expected_cat,
                                       std::optional<float> threshold) const
{
    const float thr = threshold.value_or(_cfg.sdf_threshold);
    const int cat_idx = _reg.idx_or(expected_cat, -1);
    if (cat_idx < 0) return {};

    std::vector<Eigen::Vector3f> out;
    for (const auto& [k, vs] : _grid)
    {
        if (vs.track_id != track_id) continue;
        if (cat_idx >= static_cast<int>(vs.alpha.size())) continue;
        const float a_sum = std::reduce(vs.alpha.begin(), vs.alpha.end(), 0.0f);
        if (a_sum > 0.0f && vs.alpha[static_cast<std::size_t>(cat_idx)] / a_sum >= thr)
            out.push_back(vs.centroid);
    }
    return out;
}

std::vector<Eigen::Vector3f>
UnifiedVoxelGrid::get_points_clustered(int track_id,
                                        std::optional<std::string> expected_cat,
                                        std::optional<float> cat_threshold,
                                        std::optional<float> eps,
                                        std::optional<int>   min_samples,
                                        std::optional<int>   top_k,
                                        std::optional<float> min_ratio) const
{
    auto pts = expected_cat
             ? get_filtered_points(track_id, *expected_cat, cat_threshold)
             : get_points(track_id);

    const float _eps  = eps.value_or(_cfg.dbscan_eps_factor * _cfg.resolution);
    const int   _ms   = min_samples.value_or(_cfg.dbscan_min_samples);
    const int   _tk   = top_k.value_or(_cfg.dbscan_top_k);
    const float _mr   = min_ratio.value_or(_cfg.dbscan_min_ratio);

    if (static_cast<int>(pts.size()) < _ms * 2) return pts;

    const auto labels = _dbscan(pts, _eps, _ms);

    // Count cluster sizes
    std::unordered_map<int,int> cnt;
    for (int l : labels) if (l >= 0) ++cnt[l];
    if (cnt.empty()) return pts;

    // Sort clusters by size
    std::vector<std::pair<int,int>> sorted_cnt(cnt.begin(), cnt.end());
    std::sort(sorted_cnt.begin(), sorted_cnt.end(),
              [](const auto& a, const auto& b){ return a.second > b.second; });
    sorted_cnt.resize(std::min(static_cast<int>(sorted_cnt.size()), _tk));

    const int max_count = sorted_cnt.front().second;
    std::unordered_set<int> keep;
    for (const auto& [lbl, c] : sorted_cnt)
        if (keep.empty() || static_cast<float>(c) / max_count >= _mr)
            keep.insert(lbl);

    std::vector<Eigen::Vector3f> filtered;
    filtered.reserve(pts.size());
    for (std::size_t i = 0; i < pts.size(); ++i)
        if (keep.contains(labels[i]))
            filtered.push_back(pts[i]);

    // Safety: if <30% of original survived, skip filtering
    if (static_cast<int>(filtered.size()) < std::max(3, static_cast<int>(pts.size() * 0.30f)))
        return pts;

    return filtered;
}

// ═══════════════════════════════════════════════════════════════════════════
//  4. Ownership / category
// ═══════════════════════════════════════════════════════════════════════════

float UnifiedVoxelGrid::ownership_prob(VoxelKey key, int track_id) const noexcept
{
    auto it = _grid.find(key);
    if (it == _grid.end()) return 0.0f;
    return (it->second.track_id == track_id) ? 1.0f : 0.0f;
}

std::pair<int, float> UnifiedVoxelGrid::dominant_owner(VoxelKey key) const noexcept
{
    auto it = _grid.find(key);
    if (it == _grid.end()) return {0, 1.0f};
    return {it->second.track_id, 1.0f};
}

std::vector<float> UnifiedVoxelGrid::object_category_belief(int track_id) const
{
    const int K = _reg.K();
    std::vector<double> total(static_cast<std::size_t>(K), 0.0);

    for (const auto& [k, vs] : _grid)
    {
        if (vs.track_id != track_id) continue;
        const std::size_t n = std::min(static_cast<std::size_t>(K), vs.alpha.size());
        for (std::size_t i = 0; i < n; ++i)
            total[i] += vs.alpha[i];
    }

    const double s = std::reduce(total.begin(), total.end(), 0.0);
    std::vector<float> out(static_cast<std::size_t>(K));
    const double inv_s = (s > 0.0) ? 1.0 / s : 1.0 / std::max(K, 1);
    for (int i = 0; i < K; ++i)
        out[static_cast<std::size_t>(i)] = static_cast<float>(total[static_cast<std::size_t>(i)] * inv_s);
    return out;
}

std::map<std::string, float>
UnifiedVoxelGrid::object_category_dict(int track_id) const
{
    auto belief = object_category_belief(track_id);
    return _reg.belief_to_dict(belief);
}

std::pair<std::string, float>
UnifiedVoxelGrid::object_dominant_category(int track_id) const
{
    auto belief = object_category_belief(track_id);
    if (belief.empty()) return {"unknown", 0.0f};
    const auto it = std::max_element(belief.begin(), belief.end());
    const int idx = static_cast<int>(std::distance(belief.begin(), it));
    return {_reg.name(idx), *it};
}

// ═══════════════════════════════════════════════════════════════════════════
//  5. Track management
// ═══════════════════════════════════════════════════════════════════════════

void UnifiedVoxelGrid::remove(int track_id)
{
    std::erase_if(_grid, [&](const auto& kv){
        if (kv.second.track_id == track_id) { _track_dec(track_id); return true; }
        return false;
    });
    _track_voxel_count.erase(track_id);
}

int UnifiedVoxelGrid::cleanup_voxels(int track_id,
                                      std::optional<float> eps,
                                      std::optional<int>   min_samples,
                                      std::optional<float> min_ratio)
{
    std::vector<VoxelKey> keys;
    for (const auto& [k, vs] : _grid)
        if (vs.track_id == track_id)
            keys.push_back(k);

    if (static_cast<int>(keys.size()) < 4) return 0;

    const float _eps = eps.value_or(_cfg.dbscan_eps_factor * _cfg.resolution);
    const int   _ms  = min_samples.value_or(_cfg.dbscan_min_samples);
    const float _mr  = min_ratio.value_or(_cfg.dbscan_min_ratio);

    std::vector<Eigen::Vector3f> pts;
    pts.reserve(keys.size());
    for (const auto& k : keys)
        pts.push_back(_grid.at(k).centroid);

    if (static_cast<int>(pts.size()) < _ms * 2) return 0;

    const auto labels = _dbscan(pts, _eps, _ms);

    std::unordered_map<int,int> cnt;
    for (int l : labels) if (l >= 0) ++cnt[l];
    if (cnt.empty()) return 0;

    const int max_cnt = std::max_element(cnt.begin(), cnt.end(),
        [](const auto& a, const auto& b){ return a.second < b.second; })->second;

    std::unordered_set<int> keep;
    for (const auto& [lbl, c] : cnt)
        if (static_cast<float>(c) / max_cnt >= _mr)
            keep.insert(lbl);

    int deleted = 0;
    for (std::size_t i = 0; i < keys.size(); ++i)
    {
        if (labels[i] == -1 || !keep.contains(labels[i]))
        {
            _grid.erase(keys[i]);
            _track_dec(track_id);
            ++deleted;
        }
    }
    return deleted;
}

int UnifiedVoxelGrid::prune_to_sdf(
    int track_id,
    std::function<std::vector<float>(std::span<const Eigen::Vector3f>)> sdf_fn,
    float sdf_threshold)
{
    std::vector<VoxelKey>         keys;
    std::vector<Eigen::Vector3f>  centroids;
    for (const auto& [k, vs] : _grid)
        if (vs.track_id == track_id)
        {
            keys.push_back(k);
            centroids.push_back(vs.centroid);
        }
    if (keys.empty()) return 0;

    const auto sdf_vals = sdf_fn(centroids);
    int deleted = 0;
    for (std::size_t i = 0; i < keys.size(); ++i)
    {
        if (std::abs(sdf_vals[i]) > sdf_threshold)
        {
            _grid.erase(keys[i]);
            _track_dec(track_id);
            ++deleted;
        }
    }
    return deleted;
}

int UnifiedVoxelGrid::reassign_ownership(int from_id, int to_id)
{
    int n = 0;
    for (auto& [k, vs] : _grid)
        if (vs.track_id == from_id) { vs.track_id = to_id; ++n; }
    if (n)
    {
        _track_voxel_count[to_id] = get_n_voxels(to_id) + n;
        _track_voxel_count.erase(from_id);
    }
    return n;
}

std::unordered_set<int> UnifiedVoxelGrid::get_all_track_ids() const
{
    std::unordered_set<int> out;
    for (const auto& [k, vs] : _grid)
        if (vs.track_id != 0)
            out.insert(vs.track_id);
    return out;
}

std::unordered_map<int, int> UnifiedVoxelGrid::summary() const
{
    std::unordered_map<int, int> out;
    for (const auto& [k, vs] : _grid)
        if (vs.track_id != 0)
            ++out[vs.track_id];
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════
//  6. Export & diagnostics
// ═══════════════════════════════════════════════════════════════════════════

SemanticVoxelExport
UnifiedVoxelGrid::export_semantic_voxels(const std::unordered_set<int>* track_ids,
                                          const std::unordered_map<int,std::string>* hints,
                                          float min_prob) const
{
    SemanticVoxelExport out;
    for (const auto& [k, vs] : _grid)
    {
        if (track_ids && !track_ids->contains(vs.track_id)) continue;

        const float a_sum = std::reduce(vs.alpha.begin(), vs.alpha.end(), 0.0f);
        std::string cat_name = "unknown";
        float p_cat = 0.0f;
        if (a_sum > 0.0f)
        {
            const auto it = std::max_element(vs.alpha.begin(), vs.alpha.end());
            const int idx = static_cast<int>(std::distance(vs.alpha.begin(), it));
            cat_name = (idx < _reg.K()) ? _reg.name(idx) : "unknown";
            p_cat = *it / a_sum;
        }
        else if (hints)
        {
            auto hit = hints->find(vs.track_id);
            if (hit != hints->end()) cat_name = hit->second;
        }

        if (p_cat < min_prob) cat_name = "unknown";

        out.points.push_back(vs.centroid);
        out.categories.push_back(std::move(cat_name));
        out.probs.push_back(p_cat);
        out.track_ids.push_back(vs.track_id);
    }
    return out;
}

std::map<std::string, std::variant<int, std::map<std::string,int>>>
UnifiedVoxelGrid::global_debug_stats() const
{
    std::map<std::string,int> by_cat;
    int n_owned = 0;
    for (const auto& [k, vs] : _grid)
    {
        if (vs.track_id == 0) continue;
        ++n_owned;
        if (!vs.alpha.empty())
        {
            const auto it = std::max_element(vs.alpha.begin(), vs.alpha.end());
            const int idx = static_cast<int>(std::distance(vs.alpha.begin(), it));
            const std::string cat = (idx < _reg.K()) ? _reg.name(idx) : "unknown";
            ++by_cat[cat];
        }
    }
    return {
        {"n_total",     static_cast<int>(_grid.size())},
        {"n_owned",     n_owned},
        {"n_background",std::max(0, static_cast<int>(_grid.size()) - n_owned)},
        {"by_category", by_cat},
    };
}
