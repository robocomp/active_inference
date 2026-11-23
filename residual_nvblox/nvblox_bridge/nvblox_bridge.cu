#include "nvblox_bridge.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "nvblox/map/internal/cuda/impl/layer_impl.cuh"
#include "nvblox/core/indexing.h"
#include "nvblox/mapper/mapper.h"
#include "nvblox/sensors/lidar.h"
#include "nvblox/sensors/pointcloud.h"

namespace nvb
{

struct Mapper::Impl
{
    explicit Impl(const Params& p)
        : mapper(p.voxel_m, nvblox::BlockMemoryPoolParams(), nvblox::ProjectiveLayerType::kOccupancy), prm(p)
    {
        nvblox::MapperParams mp;
        mp.occupancy_integrator_params.free_region_occupancy_probability = p.p_free;
        mp.occupancy_integrator_params.occupied_region_occupancy_probability = p.p_occ;
        mp.occupancy_integrator_params.occupied_region_half_width_m = p.half_width_m;
        mp.projective_integrator_params.lidar_projective_integrator_max_integration_distance_m = p.max_range_m;
        mp.esdf_integrator_params.esdf_slice_min_height = p.esdf_z0;
        mp.esdf_integrator_params.esdf_slice_max_height = p.esdf_z1;
        mp.esdf_integrator_params.esdf_slice_height = p.esdf_z_out;
        mp.esdf_integrator_params.esdf_integrator_max_distance_m = p.esdf_max_m;
        if (p.decay)
        {
            mp.occupancy_decay_integrator_params.occupied_region_decay_probability = p.decay_occ_p;
            mp.occupancy_decay_integrator_params.free_region_decay_probability = p.decay_free_p;
        }
        mapper.setMapperParams(mp);
    }

    nvblox::Mapper mapper;
    Params prm;
    nvblox::CudaStreamOwning stream;
    nvblox::Pointcloud cloud{nvblox::MemoryType::kDevice};
    std::vector<nvblox::Vector3f> pts;
    std::vector<unsigned char> hit;
    struct SelfMask
    {
        SelfMaskParams p;
        std::string file;
        int left = 0, n = 0, A = 0, E = 0;
        std::vector<unsigned short> self_n, empty_n, empty_far_n;
        std::vector<unsigned char> mask;
    };
    std::map<int, SelfMask> self;
    std::vector<nvblox::Index3D> guard_idx;
    std::vector<nvblox::VoxelBlock<nvblox::OccupancyVoxel>> guard_old;
    nvblox::device_vector<nvblox::Vector3f> positions;
    nvblox::device_vector<nvblox::OccupancyVoxel> voxels;
    nvblox::device_vector<bool> flags;
    nvblox::device_vector<nvblox::Vector3f> positions2d;
    nvblox::device_vector<nvblox::EsdfVoxel> esdf_vox;
    std::vector<nvblox::EsdfVoxel> h_esdf;
    std::size_t n_cells = 0;
    std::vector<nvblox::OccupancyVoxel> h_vox;
    std::unique_ptr<bool[]> h_flags;
    std::size_t n_samples = 0;
};

Mapper::Mapper(const Params& p) : d_(std::make_unique<Impl>(p)) {}
Mapper::~Mapper() = default;

int Mapper::integrate_lidar(const float* T, const float* xyz, int n, const LidarModel& m, float rx, float ry)
{
    if (n <= 0) return 0;
    nvblox::Transform T_L_S = nvblox::Transform::Identity();
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c) T_L_S.linear()(r, c) = T[r * 4 + c];
        T_L_S.translation()(r) = T[r * 4 + 3];
    }
    d_->pts.resize(n);
    for (int i = 0; i < n; ++i) d_->pts[i] = nvblox::Vector3f(xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]);
    static int cloud_calls = 0;
    const bool dump_cloud = d_->prm.dump_clouds and (cloud_calls++ % 40) < 2;
    const auto write_cloud = [&](const char* tag)
    {
        const std::string f = std::string("/dev/shm/nvblox_cloud_") + tag + "_" + std::to_string(m.sensor_id) + ".bin";
        if (FILE* fp = std::fopen(f.c_str(), "wb"))
        {
            const int np = static_cast<int>(d_->pts.size());
            std::fwrite(T, 4, 12, fp); std::fwrite(&np, 4, 1, fp);
            std::fwrite(d_->pts.data(), sizeof(float) * 3, d_->pts.size(), fp);
            std::fclose(fp);
        }
    };
    if (dump_cloud) write_cloud("pre");
    nvblox::Lidar lidar(m.azimuth_divisions, m.rows, m.min_range_m, m.below_rad, m.above_rad);
    lidar.nearest_interpolation_max_allowable_dist_to_ray_vox(d_->prm.nearest_ray_dist_vox);
    lidar.nearest_interpolation_ring_fraction(d_->prm.ring_fraction);
    if (not m.ring_elev_deg.empty())
    {
        std::vector<float> rad(m.ring_elev_deg.size());
        for (std::size_t i = 0; i < rad.size(); ++i) rad[i] = m.ring_elev_deg[i] * 0.0174532925f;
        lidar.set_ring_elevations_rad(rad.data(), static_cast<int>(rad.size()));
    }
    int filled = 0;
    const int A = lidar.num_azimuth_divisions(), E = lidar.num_elevation_divisions();
    if (auto it = d_->self.find(m.sensor_id); it != d_->self.end())
    {
        auto& sm = it->second;
        const nvblox::Vector3f o = T_L_S.translation();
        const nvblox::Matrix3f R = T_L_S.linear();
        if (sm.left > 0)
        {
            const std::size_t N = static_cast<std::size_t>(A) * E;
            if (sm.self_n.size() != N)
            { sm.self_n.assign(N, 0); sm.empty_n.assign(N, 0); sm.empty_far_n.assign(N, 0); sm.A = A; sm.E = E; }
            std::vector<unsigned char> seen(N, 0), own(N, 0);
            for (const auto& p : d_->pts)
            {
                nvblox::Index2D u;
                if (not lidar.project(p, &u) or u.x() < 0 or u.x() >= A or u.y() < 0 or u.y() >= E) continue;
                const std::size_t k = static_cast<std::size_t>(u.y()) * A + u.x();
                seen[k] = 1;
                const nvblox::Vector3f q = R * p;
                if (std::hypot(q.x(), q.y()) < sm.p.clear_r_m and o.z() + q.z() > d_->prm.floor_z + sm.p.floor_tol_m) own[k] = 1;
            }
            for (int v = 0; v < E; ++v)
                for (int u = 0; u < A; ++u)
                {
                    const std::size_t k = static_cast<std::size_t>(v) * A + u;
                    sm.self_n[k] += own[k];
                    if (seen[k]) continue;
                    const nvblox::Vector3f dir = R * lidar.vectorFromPixelIndices(nvblox::Index2D(u, v));
                    if (dir.z() > -1e-3f) continue;
                    const float t = (d_->prm.floor_z - o.z()) / dir.z();
                    if (t <= m.min_range_m or t > d_->prm.max_range_m) continue;
                    if (t * std::hypot(dir.x(), dir.y()) < 3.0f) ++sm.empty_n[k]; else ++sm.empty_far_n[k];
                }
            ++sm.n;
            if (--sm.left == 0)
            {
                // 1 = SELF (points dropped, never filled): own returns + rays the body empties (floor < 3 m), dilated
                //     over the mixed-return edge. 2 = NO-FILL only: always empty although the floor is in range further
                //     out (body-blocked grazing rays, or a far floor that never returns) — a point there still counts.
                std::vector<unsigned char> base(N, 0);
                int n_own = 0, n_empty = 0, n_far = 0;
                for (std::size_t k = 0; k < N; ++k)
                {
                    const bool own_k = sm.self_n[k] >= sm.p.min_frac * sm.n, empty_k = sm.empty_n[k] >= 0.9f * sm.n;
                    base[k] = own_k or empty_k;
                    n_own += own_k; n_empty += (empty_k and not own_k);
                }
                sm.mask.assign(N, 0);
                for (std::size_t k = 0; k < N; ++k)
                    if (not base[k] and sm.empty_far_n[k] >= 0.9f * sm.n) { sm.mask[k] = 2; ++n_far; }
                for (int v = 0; v < E; ++v)
                    for (int u = 0; u < A; ++u)
                        if (base[static_cast<std::size_t>(v) * A + u])
                            for (int dv = -sm.p.dilate_ring; dv <= sm.p.dilate_ring; ++dv)
                                for (int du = -sm.p.dilate_az; du <= sm.p.dilate_az; ++du)
                                    if (v + dv >= 0 and v + dv < E)
                                        sm.mask[static_cast<std::size_t>(v + dv) * A + (u + du + A) % A] = 1;
                int total = 0;
                for (const auto b : sm.mask) total += (b == 1);
                if (FILE* fp = std::fopen(sm.file.c_str(), "wb"))
                {
                    std::fwrite(&A, 4, 1, fp); std::fwrite(&E, 4, 1, fp);
                    std::fwrite(sm.mask.data(), 1, N, fp);
                    std::fclose(fp);
                }
                std::printf("[self-mask] sensor %d: %d sweeps, %d own-return + %d body-emptied pixels -> %d SELF of %zu "
                            "after dilation, + %d no-fill (far, always empty), saved %s\n", m.sensor_id, sm.n, n_own,
                            n_empty, total, N, n_far, sm.file.c_str());
                std::fflush(stdout);
            }
            return 0;
        }
        if (sm.mask.size() == static_cast<std::size_t>(A) * E)
        {
            // Inside the self radius a masked pixel keeps only what lies ON the floor plane (FloorPush's certain-floor
            // band), pushed under the floor right here so it clears and never reaches the wall-foot test: the body's
            // own skirt and wheels stand vertically on the floor and would pass as a wall foot. Beyond the radius the
            // beam got past the body: whatever it struck is real.
            const float z_under = d_->prm.floor_z - d_->prm.floor_fill_below_m;
            std::size_t w = 0;
            for (std::size_t k = 0; k < d_->pts.size(); ++k)
            {
                nvblox::Index2D u;
                nvblox::Vector3f p = d_->pts[k];
                const nvblox::Vector3f q = R * p;
                const float hr = std::hypot(q.x(), q.y());
                if (hr < sm.p.clear_r_m and lidar.project(p, &u) and u.x() >= 0 and u.x() < A and u.y() >= 0 and
                    u.y() < E and sm.mask[static_cast<std::size_t>(u.y()) * A + u.x()] == 1)
                {
                    const float floor_band = d_->prm.floor_z + d_->prm.floor_clear_z0 + d_->prm.floor_clear_slope * hr;
                    if (q.z() > -1e-3f or o.z() + q.z() > floor_band) continue;
                    p *= (z_under - o.z()) / q.z();
                }
                d_->pts[w++] = p;
            }
            d_->pts.resize(w);
        }
    }
    if (d_->prm.floor_fill)   // every pixel holding a real return, BEFORE any is pushed or dropped
    {
        d_->hit.assign(static_cast<std::size_t>(A) * E, 0);
        for (const auto& p : d_->pts)
        {
            nvblox::Index2D u;
            if (lidar.project(p, &u) and u.x() >= 0 and u.x() < A and u.y() >= 0 and u.y() < E)
                d_->hit[static_cast<std::size_t>(u.y()) * A + u.x()] = 1;
        }
    }
    if (d_->prm.floor_push)
    {
        const nvblox::Vector3f o = T_L_S.translation();
        const nvblox::Matrix3f R = T_L_S.linear();
        const float z_under = d_->prm.floor_z - d_->prm.floor_fill_below_m;
        // VERTICAL CONTINUITY: an AMBIGUOUS return whose ring above or below (same column) lands at the same horizontal
        // range (±5 cm) and higher is the foot of a vertical surface (wall, box side), not floor: it keeps its return.
        // Certain floor is tested first — the floor at the foot of the robot's own skirt has that neighbour too.
        // On the floor, neighbouring rings land at clearly different ranges.
        const std::size_t N = static_cast<std::size_t>(A) * E;
        std::vector<int> pix_pt(N, -1), pt_pix(d_->pts.size(), -1);
        for (std::size_t k = 0; k < d_->pts.size(); ++k)
        {
            nvblox::Index2D u;
            if (not lidar.project(d_->pts[k], &u) or u.x() < 0 or u.x() >= A or u.y() < 0 or u.y() >= E) continue;
            const int j = u.y() * A + u.x();
            pt_pix[k] = j;
            if (pix_pt[j] < 0 or d_->pts[k].squaredNorm() < d_->pts[pix_pt[j]].squaredNorm()) pix_pt[j] = static_cast<int>(k);
        }
        const auto wall_foot = [&](std::size_t k, const nvblox::Vector3f& q)
        {
            if (pt_pix[k] < 0) return false;
            const int v = pt_pix[k] / A, u = pt_pix[k] % A;
            for (const int dv : {-1, 1})
            {
                if (v + dv < 0 or v + dv >= E) continue;
                const int j = pix_pt[(v + dv) * A + u];
                if (j < 0) continue;
                const nvblox::Vector3f qn = R * d_->pts[j];
                if (std::abs(std::hypot(qn.x(), qn.y()) - std::hypot(q.x(), q.y())) < 0.05f and qn.z() > q.z() + 0.02f)
                    return true;
            }
            return false;
        };
        std::vector<unsigned char> foot(d_->pts.size(), 0);
        for (std::size_t k = 0; k < d_->pts.size(); ++k) foot[k] = wall_foot(k, R * d_->pts[k]);
        std::size_t w = 0;
        for (std::size_t k = 0; k < d_->pts.size(); ++k)
        {
            nvblox::Vector3f p = d_->pts[k];
            const nvblox::Vector3f q = R * p;   // sensor → point, room axes
            const float hr = std::hypot(q.x(), q.y()), z = o.z() + q.z();
            const float clear_band = d_->prm.floor_z + d_->prm.floor_clear_z0 + d_->prm.floor_clear_slope * hr;
            const float ambiguous = d_->prm.floor_z + d_->prm.floor_push_z0 + d_->prm.floor_push_slope * hr;
            if (q.z() < -1e-3f and z < clear_band) p *= (z_under - o.z()) / q.z();        // certainly floor: clear through
            else if (foot[k]) {}                                                        // wall foot: a real return
            else if (q.z() < -1e-3f and z < ambiguous) continue;                     // floor or low obstacle: no evidence
            d_->pts[w++] = p;
        }
        d_->pts.resize(w);
    }
    if (d_->prm.floor_fill)
    {
        const nvblox::Vector3f o = T_L_S.translation();
        const nvblox::Matrix3f R = T_L_S.linear();
        for (int v = 0; v < E; ++v)
            for (int u = 0; u < A; ++u)
            {
                if (d_->hit[static_cast<std::size_t>(v) * A + u]) continue;
                if (const auto it = d_->self.find(m.sensor_id); it != d_->self.end() and
                    it->second.mask.size() == d_->hit.size() and it->second.mask[static_cast<std::size_t>(v) * A + u]) continue;
                // A hole with a return just above or below is a ring that fell into the neighbouring row of the
                // uniform model (helios rings straddle row edges, swapping rows with azimuth), not a missing return:
                // filling it would fire a free ray to the floor straight through whatever that ring hit.
                if ((v > 0 and d_->hit[static_cast<std::size_t>(v - 1) * A + u]) or
                    (v + 1 < E and d_->hit[static_cast<std::size_t>(v + 1) * A + u])) continue;
                const nvblox::Vector3f dir = lidar.vectorFromPixelIndices(nvblox::Index2D(u, v));
                const float dz = (R * dir).z();
                if (dz > -1e-3f) continue;
                const float t_floor = (d_->prm.floor_z - o.z()) / dz;
                if (t_floor < m.min_range_m or t_floor > d_->prm.max_range_m) continue;
                d_->pts.push_back(dir * ((d_->prm.floor_z - d_->prm.floor_fill_below_m - o.z()) / dz));
                ++filled;
            }
    }
    // Snapshot the blocks covering the blind volume, integrate, then let no voxel inside it go DOWN.
    const float R = std::max(m.blind_r, m.skirt_r);
    auto& layer = d_->mapper.occupancy_layer();
    const float bs = layer.block_size();
    constexpr int V = nvblox::VoxelBlock<nvblox::OccupancyVoxel>::kVoxelsPerSide;
    d_->guard_idx.clear();
    if (R > 0.0f)
    {
        const float z0 = std::min(m.blind_z0, m.skirt_z0), z1 = std::max(m.blind_z1, m.skirt_z1);
        const auto lo = nvblox::getBlockIndexFromPositionInLayer(bs, nvblox::Vector3f(rx - R, ry - R, z0));
        const auto hi = nvblox::getBlockIndexFromPositionInLayer(bs, nvblox::Vector3f(rx + R, ry + R, z1));
        for (int x = lo.x(); x <= hi.x(); ++x)
            for (int y = lo.y(); y <= hi.y(); ++y)
                for (int z = lo.z(); z <= hi.z(); ++z) d_->guard_idx.emplace_back(x, y, z);
        d_->guard_old.resize(d_->guard_idx.size());
        for (std::size_t k = 0; k < d_->guard_idx.size(); ++k)
        {
            auto blk = layer.allocateBlockAtIndex(d_->guard_idx[k]);   // unallocated = unknown (0): keep it so
            cudaMemcpy(&d_->guard_old[k], blk.get(), sizeof(nvblox::VoxelBlock<nvblox::OccupancyVoxel>), cudaMemcpyDefault);
        }
    }
    if (dump_cloud) write_cloud("post");
    d_->cloud.copyPointsFromAsync(d_->pts, d_->stream);
    d_->stream.synchronize();
    d_->mapper.integrateDepth(d_->cloud, T_L_S, lidar, false);
    if (R > 0.0f)
    {
        cudaDeviceSynchronize();
        nvblox::VoxelBlock<nvblox::OccupancyVoxel> now;
        for (std::size_t k = 0; k < d_->guard_idx.size(); ++k)
        {
            auto blk = layer.getBlockAtIndex(d_->guard_idx[k]);
            if (not blk) continue;
            cudaMemcpy(&now, blk.get(), sizeof(now), cudaMemcpyDefault);
            bool changed = false;
            for (int x = 0; x < V; ++x)
                for (int y = 0; y < V; ++y)
                    for (int z = 0; z < V; ++z)
                    {
                        const auto c = nvblox::getCenterPositionFromBlockIndexAndVoxelIndex(bs, d_->guard_idx[k],
                                                                                           nvblox::Index3D(x, y, z));
                        const float r = std::hypot(c.x() - rx, c.y() - ry);
                        const bool blind = (r < m.blind_r and c.z() >= m.blind_z0 and c.z() <= m.blind_z1) or
                                           (r < m.skirt_r and c.z() >= m.skirt_z0 and c.z() <= m.skirt_z1);
                        float& v = now.voxels[x][y][z].log_odds;
                        const float old = d_->guard_old[k].voxels[x][y][z].log_odds;
                        if (blind and v < old) { v = old; changed = true; }
                    }
            if (changed) cudaMemcpy(blk.get(), &now, sizeof(now), cudaMemcpyDefault);
        }
    }
    return filled;
}

void Mapper::decay()
{
    if (d_->prm.decay) d_->mapper.decayOccupancyAllVoxels();
}

void Mapper::set_sampling(float xmin, float ymin, float cell, int w, int h, int n_bins, float bin_h)
{
    std::vector<nvblox::Vector3f> p;
    p.reserve(static_cast<std::size_t>(w) * h * n_bins);
    for (int iy = 0; iy < h; ++iy)
        for (int ix = 0; ix < w; ++ix)
            for (int b = 0; b < n_bins; ++b)
                p.emplace_back(xmin + (ix + 0.5f) * cell, ymin + (iy + 0.5f) * cell, (b + 0.5f) * bin_h);
    d_->n_samples = p.size();
    d_->positions.copyFromAsync(p, d_->stream);
    std::vector<nvblox::Vector3f> p2;
    p2.reserve(static_cast<std::size_t>(w) * h);
    for (int iy = 0; iy < h; ++iy)
        for (int ix = 0; ix < w; ++ix) p2.emplace_back(xmin + (ix + 0.5f) * cell, ymin + (iy + 0.5f) * cell, d_->prm.esdf_z_out);
    d_->n_cells = p2.size();
    d_->positions2d.copyFromAsync(p2, d_->stream);
    d_->h_esdf.resize(d_->n_cells);
    d_->h_vox.resize(d_->n_samples);
    d_->h_flags = std::make_unique<bool[]>(d_->n_samples);
    d_->stream.synchronize();
}

void Mapper::sample(float* out)
{
    if (d_->n_samples == 0) return;
    d_->mapper.occupancy_layer().getVoxelsGPU(d_->positions, &d_->voxels, &d_->flags, &d_->stream);
    d_->voxels.copyToAsync(d_->h_vox.data(), d_->stream);
    d_->flags.copyToAsync(d_->h_flags.get(), d_->stream);
    d_->stream.synchronize();
    for (std::size_t k = 0; k < d_->n_samples; ++k) out[k] = d_->h_flags[k] ? d_->h_vox[k].log_odds : 0.0f;
}

void Mapper::esdf(float* out)
{
    if (d_->n_cells == 0) return;
    d_->mapper.updateEsdfSlice();
    d_->mapper.esdf_layer().getVoxelsGPU(d_->positions2d, &d_->esdf_vox, &d_->flags, &d_->stream);
    d_->esdf_vox.copyToAsync(d_->h_esdf.data(), d_->stream);
    d_->flags.copyToAsync(d_->h_flags.get(), d_->stream);
    d_->stream.synchronize();
    const float vs = d_->prm.voxel_m;
    for (std::size_t k = 0; k < d_->n_cells; ++k)
    {
        const auto& v = d_->h_esdf[k];
        if (not d_->h_flags[k] or not v.observed) { out[k] = std::nanf(""); continue; }
        const float d = std::sqrt(v.squared_distance_vox) * vs;
        out[k] = v.is_inside ? -d : d;
    }
}

void Mapper::self_mask_calibrate(int id, const SelfMaskParams& p, const std::string& file)
{
    auto& sm = d_->self[id];
    sm = Impl::SelfMask{};
    sm.p = p;
    sm.file = file;
    sm.left = std::max(1, p.sweeps);
}

bool Mapper::self_mask_load(int id, const SelfMaskParams& p, const std::string& file)
{
    FILE* fp = std::fopen(file.c_str(), "rb");
    if (not fp) return false;
    int A = 0, E = 0;
    bool ok = std::fread(&A, 4, 1, fp) == 1 and std::fread(&E, 4, 1, fp) == 1 and A > 0 and E > 0;
    std::vector<unsigned char> mask(ok ? static_cast<std::size_t>(A) * E : 0);
    ok = ok and std::fread(mask.data(), 1, mask.size(), fp) == mask.size();
    std::fclose(fp);
    if (not ok) return false;
    auto& sm = d_->self[id];
    sm = Impl::SelfMask{};
    sm.p = p;
    sm.A = A; sm.E = E;
    sm.mask = std::move(mask);
    return true;
}

int Mapper::self_mask_pixels(int id) const
{
    const auto it = d_->self.find(id);
    if (it == d_->self.end() or it->second.left > 0 or it->second.mask.empty()) return -1;
    int n = 0;
    for (const auto b : it->second.mask) n += (b == 1);
    return n;
}

std::size_t Mapper::allocated_blocks() const
{
    return d_->mapper.occupancy_layer().numAllocatedBlocks();
}

}  // namespace nvb
