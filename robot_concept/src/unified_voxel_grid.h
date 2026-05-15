#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <execution>
#include <expected>
#include <flat_set>
#include <functional>
#include <map>
#include <optional>
#include <print>
#include <ranges>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════════
//  VoxelKey — 3-int key with fast hasher
// ═══════════════════════════════════════════════════════════════════════════

struct VoxelKey
{
    int32_t x, y, z;
    bool operator==(const VoxelKey&) const noexcept = default;
    auto operator<=>(const VoxelKey&) const noexcept = default;
};

struct VoxelKeyHash
{
    std::size_t operator()(const VoxelKey& k) const noexcept
    {
        // Murmur-inspired mixing
        auto h = [](uint64_t v) noexcept -> uint64_t {
            v ^= v >> 33; v *= 0xff51afd7ed558ccdULL;
            v ^= v >> 33; v *= 0xc4ceb9fe1a85ec53ULL;
            v ^= v >> 33; return v;
        };
        return h(static_cast<uint64_t>(k.x) * 2654435761ULL
               ^ static_cast<uint64_t>(k.y) * 805459861ULL
               ^ static_cast<uint64_t>(k.z) * 3674653429ULL);
    }
};

// ═══════════════════════════════════════════════════════════════════════════
//  CategoryRegistry
// ═══════════════════════════════════════════════════════════════════════════

class CategoryRegistry
{
    public:
        static constexpr std::array DEFAULT_CATEGORIES{
            "chair", "table", "desk", "sofa", "bed", "pot", "plant", "tv", "laptop", "keyboard",
            "monitor", "shelf", "door", "window"
        };

        explicit CategoryRegistry(std::span<const char* const> cats = DEFAULT_CATEGORIES);

        int  register_category(const std::string& name);
        int  idx(const std::string& name);         // registers if missing
        [[nodiscard]] int idx_or(const std::string& name, int fallback = -1) const;  // const, no register
        [[nodiscard]] std::string name(int i) const;
        [[nodiscard]] int K() const noexcept { return static_cast<int>(_n2i.size()); }
        [[nodiscard]] std::vector<std::string> names() const;
        [[nodiscard]] std::map<std::string, float>
            belief_to_dict(std::span<const float> alpha) const;

    private:
        std::unordered_map<std::string, int> _n2i;
        std::unordered_map<int, std::string> _i2n;
};

// ═══════════════════════════════════════════════════════════════════════════
//  Config
// ═══════════════════════════════════════════════════════════════════════════

struct UnifiedGridConfig
{
    float resolution            = 0.05f;
    int   max_voxels            = 250'000;
    int   max_voxels_per_track  = 3'000;

    float alpha_prior           = 0.1f;
    float centroid_ema          = 0.3f;

    int   min_frames_to_confirm = 15;
    bool  release_categorized_endpoint_on_unknown_hit = true;
    int   traversal_delete_threshold = 5;

    float own_threshold         = 0.5f;
    float sdf_threshold         = 0.5f;
    float min_concentration     = 2.0f;

    float fov_h                 = 120.0f;
    float fov_v                 = 60.0f;
    float max_range             = 5.0f;
    float min_range             = 0.3f;
    float cam_forward_sign      = 1.0f;

    float dbscan_eps_factor     = 3.2f;
    int   dbscan_min_samples    = 6;
    int   dbscan_top_k          = 1;
    float dbscan_min_ratio      = 0.50f;
};

// ═══════════════════════════════════════════════════════════════════════════
//  VoxelState (internal)
// ═══════════════════════════════════════════════════════════════════════════

struct VoxelState
{
    Eigen::Vector3f    centroid;
    std::vector<float> alpha;         // Dirichlet counts, length == K at creation
    int   track_id        = 0;
    int   n_obs           = 0;
    int   n_frames_seen   = 0;
    int   last_frame      = -1;
    int   n_ray_traversals= 0;
    float best_confidence = 0.0f;
    int   best_cat_idx    = -1;
};

// ═══════════════════════════════════════════════════════════════════════════
//  Return types
// ═══════════════════════════════════════════════════════════════════════════

struct VisibilityStats
{
    int n_rays_cast                     = 0;
    int n_intermediate_deleted          = 0;
    int n_intermediate_traversal_inc    = 0;
    int n_endpoint_no_cat_deleted       = 0;
    int n_endpoint_categorized_deleted  = 0;
    int n_endpoint_kept                 = 0;
};

struct SemanticVoxelExport
{
    std::vector<Eigen::Vector3f> points;
    std::vector<std::string>     categories;
    std::vector<float>           probs;
    std::vector<int>             track_ids;
};

// ═══════════════════════════════════════════════════════════════════════════
//  UnifiedVoxelGrid
// ═══════════════════════════════════════════════════════════════════════════

class UnifiedVoxelGrid
{
    public:
        // ── Construction ──────────────────────────────────────────────────────
        explicit UnifiedVoxelGrid(UnifiedGridConfig cfg = {},
                                CategoryRegistry  reg = CategoryRegistry{});

        // ── 1. Positive observation ───────────────────────────────────────────
        void observe(int track_id,
                    std::span<const Eigen::Vector3f> pts,
                    const std::string& category,
                    int frame,
                    std::span<const std::string> per_point_labels = {},
                    std::span<const float>        confidences     = {},
                    float detection_confidence = 1.0f);

        // ── 2. Ray-based visibility update ───────────────────────────────────
        [[nodiscard]] std::expected<VisibilityStats, std::string>
        visibility_update(const Eigen::Vector3f& cam_pos,
                        const Eigen::Matrix3f& cam_R,
                        int frame,
                        std::span<const Eigen::Vector3f> scene_pts = {});

        // ── 3. Point-cloud queries ────────────────────────────────────────────
        [[nodiscard]] std::vector<Eigen::Vector3f>
            get_points(int track_id) const;

        [[nodiscard]] std::optional<Eigen::Vector3f>
            get_extent(int track_id) const;

        [[nodiscard]] std::optional<float>
            get_z_centroid(int track_id) const;

        [[nodiscard]] int get_n_voxels(int track_id) const noexcept;

        [[nodiscard]] std::vector<Eigen::Vector3f>
            get_filtered_points(int track_id,
                                const std::string& expected_cat,
                                std::optional<float> threshold = std::nullopt) const;

        [[nodiscard]] std::vector<Eigen::Vector3f>
            get_points_clustered(int track_id,
                                std::optional<std::string> expected_cat = std::nullopt,
                                std::optional<float> cat_threshold      = std::nullopt,
                                std::optional<float> eps                = std::nullopt,
                                std::optional<int>   min_samples        = std::nullopt,
                                std::optional<int>   top_k              = std::nullopt,
                                std::optional<float> min_ratio          = std::nullopt) const;

        // ── 4. Ownership / category ───────────────────────────────────────────
        [[nodiscard]] float ownership_prob(VoxelKey key, int track_id) const noexcept;
        [[nodiscard]] std::pair<int, float> dominant_owner(VoxelKey key) const noexcept;

        [[nodiscard]] std::vector<float>
            object_category_belief(int track_id) const;

        [[nodiscard]] std::map<std::string, float>
            object_category_dict(int track_id) const;

        [[nodiscard]] std::pair<std::string, float>
            object_dominant_category(int track_id) const;

        // ── 5. Track management ───────────────────────────────────────────────
        void remove(int track_id);
        int  cleanup_voxels(int track_id,
                            std::optional<float> eps         = std::nullopt,
                            std::optional<int>   min_samples = std::nullopt,
                            std::optional<float> min_ratio   = std::nullopt);
        int  prune_to_sdf(int track_id,
                        std::function<std::vector<float>(std::span<const Eigen::Vector3f>)> sdf_fn,
                        float sdf_threshold = 0.10f);
        int  reassign_ownership(int from_id, int to_id);

        [[nodiscard]] std::unordered_set<int> get_all_track_ids() const;
        [[nodiscard]] std::unordered_map<int, int> summary() const;

        // ── 6. Export & diagnostics ───────────────────────────────────────────
        [[nodiscard]] SemanticVoxelExport
            export_semantic_voxels(const std::unordered_set<int>* track_ids   = nullptr,
                                const std::unordered_map<int,std::string>* hints = nullptr,
                                float min_prob = 0.0f) const;

        [[nodiscard]] std::map<std::string,
                            std::variant<int, std::map<std::string,int>>>
            global_debug_stats() const;

    private:
        // ── Helpers ───────────────────────────────────────────────────────────
        [[nodiscard]] VoxelKey make_key(const Eigen::Vector3f& pt) const noexcept;
        [[nodiscard]] bool _is_categorized(const VoxelState& vs) const noexcept;
        void _enforce_max_voxels(int frame);
        void _reset_frame_bookkeeping();

        // AABB slab test: returns keys of voxels traversed (not endpoints).
        // SoA layout fed to par_unseq inner loop.
        [[nodiscard]] std::unordered_set<VoxelKey, VoxelKeyHash>
            _aabb_traversed(const Eigen::Vector3f& cam,
                            std::span<const Eigen::Vector3f> targets) const;

        // DBSCAN — brute-force SoA, bitmatrix adjacency, par_unseq.
        // Returns per-point label (-1 = noise).
        [[nodiscard]] std::vector<int>
            _dbscan(std::span<const Eigen::Vector3f> pts,
                    float eps, int min_samples) const;

        int  _track_dec(int track_id) noexcept;   // decrement count, clamp ≥ 0
        void _track_inc(int track_id) noexcept;

        // ── State ─────────────────────────────────────────────────────────────
        UnifiedGridConfig _cfg;
        CategoryRegistry  _reg;
        float             _inv;   // 1.0f / resolution

        std::unordered_map<VoxelKey, VoxelState, VoxelKeyHash> _grid;
        std::unordered_map<int, int>  _track_voxel_count;

        // Per-frame bookkeeping — flat_set for cache-friendly membership test
        std::flat_set<VoxelKey> _touched_this_frame;
        std::vector<Eigen::Vector3f> _observed_pts_this_frame;

        static constexpr int MAX_RAYS = 500;
        int _frame = 0;
};
