/*
 * residual_nvblox_map.cpp  —  see residual_nvblox_map.h
 */

#include "residual_nvblox_map.h"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <print>
#include <utility>

namespace rc
{

NvbloxMap::NvbloxMap(const ResidualConfig& cfg, const OccupancyGrid& grid) : cfg_(cfg)
{
    nvb::Params np;
    np.voxel_m = cfg_.nvblox_voxel_m;
    np.p_free = cfg_.nvblox_p_free;
    np.p_occ = cfg_.nvblox_p_occ;
    np.half_width_m = cfg_.nvblox_half_width_m;
    np.max_range_m = cfg_.nvblox_max_range_m;
    np.decay = cfg_.nvblox_decay;
    np.decay_occ_p = cfg_.nvblox_decay_occ_p;
    np.decay_free_p = cfg_.nvblox_decay_free_p;
    np.floor_fill = cfg_.nvblox_floor_fill;
    np.floor_fill_below_m = cfg_.nvblox_floor_fill_below_m;
    np.floor_push = cfg_.nvblox_floor_push;
    np.floor_push_z0 = cfg_.nvblox_floor_push_z0;
    np.floor_push_slope = cfg_.nvblox_floor_push_slope;
    np.floor_clear_z0 = cfg_.nvblox_floor_clear_z0;
    np.floor_clear_slope = cfg_.nvblox_floor_clear_slope;
    np.nearest_ray_dist_vox = cfg_.nvblox_nearest_ray_dist_vox;
    np.ring_fraction = cfg_.nvblox_ring_fraction;
    np.esdf_z0 = cfg_.nvblox_esdf_z0;
    np.esdf_z1 = cfg_.nvblox_esdf_z1;
    np.esdf_max_m = cfg_.nvblox_esdf_max_m;
    np.dump_clouds = cfg_.nvblox_dump_clouds;
    mapper_ = std::make_unique<nvb::Mapper>(np);

    for (const auto& [id, name] : {std::pair{0, "helios"}, std::pair{1, "bpearl"}})
    {
        if (cfg_.self_mask_sensors.find(name) == std::string::npos) continue;
        const std::string file = cfg_.self_mask_prefix + name + ".bin";
        const nvb::SelfMaskParams smp{cfg_.self_mask_sweeps, cfg_.self_mask_clear_r, cfg_.self_mask_floor_tol,
                                      cfg_.self_mask_min_frac, cfg_.self_mask_dilate_az, cfg_.self_mask_dilate_ring};
        if (cfg_.self_mask_calibrate)
        {
            mapper_->self_mask_calibrate(id, smp, file);
            std::println("[self-mask] {}: CALIBRATING over {} sweeps — keep >= {:.1f} m of free floor around "
                         "the robot; nothing is mapped until it ends", name, cfg_.self_mask_sweeps, cfg_.self_mask_clear_r);
        }
        else if (cfg_.self_mask_enabled)
            std::println("[self-mask] {}: {}", name, mapper_->self_mask_load(id, smp, file)
                         ? std::format("{} pixels masked ({})", mapper_->self_mask_pixels(id), file)
                         : std::format("no mask at {} — run once with SelfMask.Calibrate = true", file));
    }

    constexpr float d2r = 0.0174532925f;
    helios_ = {cfg_.nvblox_helios_az, cfg_.nvblox_helios_rows, cfg_.helios_min_range_m,
               cfg_.nvblox_helios_below_deg * d2r, cfg_.nvblox_helios_above_deg * d2r};
    helios_.sensor_id = 0;
    if (not cfg_.nvblox_helios_ring_elev_deg.empty())
    {
        helios_.ring_elev_deg = cfg_.nvblox_helios_ring_elev_deg;
        helios_.rows = static_cast<int>(helios_.ring_elev_deg.size());
    }
    bpearl_ = {cfg_.nvblox_bpearl_az, cfg_.nvblox_bpearl_rows, cfg_.bpearl_min_range_m,
               cfg_.nvblox_bpearl_below_deg * d2r, cfg_.nvblox_bpearl_above_deg * d2r};
    bpearl_.blind_r = cfg_.nvblox_bpearl_blind_r; bpearl_.blind_z0 = cfg_.nvblox_bpearl_blind_z0;
    bpearl_.blind_z1 = cfg_.nvblox_bpearl_blind_z1;
    bpearl_.skirt_r = cfg_.nvblox_bpearl_skirt_r; bpearl_.skirt_z0 = cfg_.nvblox_bpearl_skirt_z0;
    bpearl_.skirt_z1 = cfg_.nvblox_bpearl_skirt_z1;
    bpearl_.sensor_id = 1;
    if (cfg_.use_helios) names_.emplace_back("helios");
    if (cfg_.use_bpearl) names_.emplace_back("bpearl");
    last_stamp_.assign(names_.size(), 0);
    sweeps_.assign(names_.size(), 0);

    if (cfg_.nvblox_display)
        view_ = std::make_unique<NvbloxView>(cfg_.nvblox_display_scale);
    if (cfg_.nvblox_display or cfg_.nvblox_dump_3d)
        esdf_.assign(static_cast<std::size_t>(grid.width()) * grid.height(), 0.0f);
    constexpr int NB = OccGridParams::Z_BINS;
    mapper_->set_sampling(grid.xmin(), grid.ymin(), grid.cell_size(), grid.width(), grid.height(), NB,
                          grid.params().bin_span_m / NB);
    cols_.assign(static_cast<std::size_t>(grid.width()) * grid.height() * NB, 0.0f);
    std::println("[nvblox] voxel {:.3f} m, p_free {:.2f} p_occ {:.2f} half-width {:.2f} m, decay {}, "
                 "floor fill {} ({:.2f} m under), nearest-ray {:.1f} vox", np.voxel_m, np.p_free, np.p_occ,
                 np.half_width_m, np.decay, np.floor_fill, np.floor_fill_below_m, np.nearest_ray_dist_vox);
}

void NvbloxMap::integrate(const ResidualLidarIngestor& lidar, const SensorPose& room_T_sensor, long occupied_cells)
{
    const auto& pts = lidar.sweep_room();
    const auto& pid = lidar.plane_id();
    const auto& stamps = lidar.plane_stamp_ms();
    const Eigen::Vector3f axis = lidar.origin_room();   // robot axis, centre of the bpearl blind cylinders
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t p = 0; p < names_.size() and p < stamps.size(); ++p)
    {
        if (stamps[p] <= last_stamp_[p] or pid.size() != pts.size()) continue;   // integrate each sweep once
        // Same RT query LidarPlaneReader used to put the plane in the room, so T⁻¹ recovers the sensor-frame points.
        const auto M = room_T_sensor(names_[p], stamps[p]);
        if (not M.has_value()) continue;
        const Eigen::Matrix4d Minv = M->inverse();
        xyz_.clear();
        for (std::size_t k = 0; k < pts.size(); ++k)
        {
            if (pid[k] != p) continue;
            const Eigen::Vector4d q = Minv * Eigen::Vector4d(pts[k].x(), pts[k].y(), pts[k].z(), 1.0);
            xyz_.insert(xyz_.end(), {static_cast<float>(q.x()), static_cast<float>(q.y()), static_cast<float>(q.z())});
        }
        float T34[12];
        for (int r = 0; r < 3; ++r) for (int c = 0; c < 4; ++c) T34[r * 4 + c] = static_cast<float>((*M)(r, c));
        filled_ = mapper_->integrate_lidar(T34, xyz_.data(), static_cast<int>(xyz_.size() / 3),
                                           names_[p] == "helios" ? helios_ : bpearl_, axis.x(), axis.y());
        last_stamp_[p] = stamps[p];
        ++sweeps_[p];
    }
    mapper_->decay();
    const auto t1 = std::chrono::steady_clock::now();
    mapper_->sample(cols_.data());
    const auto t2 = std::chrono::steady_clock::now();
    ms_integrate_ = std::chrono::duration<float, std::milli>(t1 - t0).count();
    ms_sample_ = std::chrono::duration<float, std::milli>(t2 - t1).count();
    if ((n_log_++ % 50) == 0)
    {
        std::string per;
        for (std::size_t p = 0; p < names_.size(); ++p)
            per += std::format(" {}={}", names_[p], sweeps_[p]);
        std::println("[nvblox] integrate {:.2f} ms, sample {:.2f} ms, {} blocks, {} occupied cells | sweeps{} | last floor fill {}",
                     ms_integrate_, ms_sample_, mapper_->allocated_blocks(), occupied_cells, per, filled_);
    }
}

void NvbloxMap::log_map_time(double total_ms, double ingest_ms, long occupied_cells)
{
    sum_total_ += total_ms; sum_ingest_ += ingest_ms; sum_integrate_ += ms_integrate_; sum_sample_ += ms_sample_;
    if (++n_time_ < 100) return;
    const double n = n_time_;
    std::println("[map-time] nvblox: map update {:.2f} ms/cycle = integrate {:.2f} + sample {:.2f} + ingest {:.2f} "
                 "(+ transform/fill on CPU) | {} occupied", sum_total_ / n, sum_integrate_ / n, sum_sample_ / n,
                 sum_ingest_ / n, occupied_cells);
    sum_total_ = sum_ingest_ = sum_integrate_ = sum_sample_ = 0.0;
    n_time_ = 0;
}

void NvbloxMap::show(const OccupancyGrid& grid, const OccupancyGrid::CellExplained& explained, const Eigen::Vector3f& robot)
{
    if ((not view_ and not cfg_.nvblox_dump_3d) or (n_show_++ % std::max(1, cfg_.nvblox_display_every_n)) != 0) return;
    const auto t0 = std::chrono::steady_clock::now();
    mapper_->esdf(esdf_.data());
    const float esdf_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    NvbloxView::Frame f;
    f.w = grid.width(); f.h = grid.height();
    f.xmin = grid.xmin(); f.ymin = grid.ymin(); f.cell = grid.cell_size();
    f.cols = &cols_;
    f.nbins = OccGridParams::Z_BINS;
    f.bin_h = grid.params().bin_span_m / OccGridParams::Z_BINS;
    f.zlo = cfg_.nvblox_read_min_z; f.zhi = grid.params().ceil_z;
    f.esdf = &esdf_; f.esdf_max = cfg_.nvblox_esdf_max_m;
    f.state.assign(static_cast<std::size_t>(f.w) * f.h, 0);
    long n_occ = 0, n_res = 0;
    for (int iy = 0; iy < f.h; ++iy)
        for (int ix = 0; ix < f.w; ++ix)
            if (grid.occupied(ix, iy)) { f.state[static_cast<std::size_t>(iy) * f.w + ix] = 1; ++n_occ; }
    const auto res = grid.residual_cell_centres(explained);
    for (std::size_t k = 0; k + 1 < res.size(); k += 2)
    {
        int ix, iy;
        if (grid.world_to_cell(res[k], res[k + 1], ix, iy)) { f.state[static_cast<std::size_t>(iy) * f.w + ix] = 2; ++n_res; }
    }
    f.robot_x = robot.x(); f.robot_y = robot.y();
    f.info = std::format("robot ({:.2f}, {:.2f}) · ocupadas {} · residuales {} · nvblox integra {:.1f} ms, muestrea {:.1f} ms, "
                         "ESDF {:.1f} ms · {} bloques · relleno suelo {}", robot.x(), robot.y(), n_occ, n_res, ms_integrate_,
                         ms_sample_, esdf_ms, mapper_->allocated_blocks(), filled_);
    if (view_) view_->show_frame(f);
    if (cfg_.nvblox_dump_3d) dump_3d(f);
}

// Snapshot for tools/nvblox_rerun_view.py: header, ESDF w·h, state w·h, then occupied voxels (x, y, z, log-odds).
void NvbloxMap::dump_3d(const NvbloxView::Frame& f) const
{
    std::vector<float> vox;
    for (int c = 0; c < f.w * f.h; ++c)
    {
        const float* col = &cols_[static_cast<std::size_t>(c) * f.nbins];
        for (int b = 0; b < f.nbins; ++b)
            if (col[b] > 0.0f)
                vox.insert(vox.end(), {f.xmin + (c % f.w + 0.5f) * f.cell, f.ymin + (c / f.w + 0.5f) * f.cell, (b + 0.5f) * f.bin_h, col[b]});
    }
    const std::int32_t hdr[3] = {f.w, f.h, static_cast<std::int32_t>(vox.size() / 4)};
    const float geo[7] = {f.xmin, f.ymin, f.cell, f.bin_h, f.robot_x, f.robot_y, f.esdf_max};
    {
        std::ofstream o("/dev/shm/residual_nvblox_3d.tmp", std::ios::binary);
        o.write(reinterpret_cast<const char*>(hdr), sizeof hdr);
        o.write(reinterpret_cast<const char*>(geo), sizeof geo);
        o.write(reinterpret_cast<const char*>(f.esdf->data()), f.esdf->size() * sizeof(float));
        o.write(reinterpret_cast<const char*>(f.state.data()), f.state.size());
        o.write(reinterpret_cast<const char*>(vox.data()), vox.size() * sizeof(float));
    }
    std::filesystem::rename("/dev/shm/residual_nvblox_3d.tmp", "/dev/shm/residual_nvblox_3d.bin");
}

}  // namespace rc
