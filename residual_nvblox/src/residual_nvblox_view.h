/*
 * residual_nvblox_view.h  —  small live window (Nvblox.Display): what nvblox holds vs what the residual publishes.
 *
 * Four panels on the grid's cells (north up, robot = white cross):
 *   P(occupied)  — nvblox column max over [read_min_z, ceil_z]; grey = never observed
 *   top height   — z of the highest occupied voxel (low obstacles vs tall ones)
 *   ESDF 2D      — nvblox distance to the nearest obstacle in the slice band, 0.5 m isolines
 *   residual     — latched cells explained by a model (blue) vs residual cells published to the DSR (magenta)
 */
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <QWidget>

class QLabel;

namespace rc
{

class NvbloxView : public QWidget
{
public:
    struct Frame
    {
        int w = 0, h = 0;
        float xmin = 0, ymin = 0, cell = 0.05f;
        const std::vector<float>* cols = nullptr;   // w·h·nbins voxel log-odds, 0 = unobserved
        int nbins = 64;
        float bin_h = 2.0f / 64, zlo = 0.11f, zhi = 1.8f;
        const std::vector<float>* esdf = nullptr;   // w·h signed distance (m), NaN = unobserved
        float esdf_max = 2.0f;
        std::vector<std::uint8_t> state;            // w·h: 0 free, 1 latched ∧ explained, 2 residual (published)
        float robot_x = 0, robot_y = 0;
        std::string info;
    };

    explicit NvbloxView(int scale);
    void show_frame(const Frame& f);

private:
    std::array<QLabel*, 4> img_{};
    QLabel* info_ = nullptr;
    int scale_ = 2;
};

}  // namespace rc
