// reloc_rival_score.h -- scores a pose with the relocaliser's own per-point likelihood (spec sec 3, 4.3).
// No likelihood code lives here: score() calls rc::reloc::log_likelihood on the SAME point subset search() polishes
// on (polish_points), with the SAME Params (robust outlier mixture, apertures, listed-door cells, BeyondLines), so
// ell_0 and ell_r are comparable with Mode::log_lik and the door/mismatch terms cancel in dll.
#pragma once
#include "reloc_search.h"
#include <vector>

namespace rc::rivals
{
// ell(pose) on the scan. `raster` supplies the extent the search used; the polygon is the room outline.
float score(const rc::reloc::LikelihoodRaster& raster, const std::vector<Eigen::Vector2f>& polygon,
            const std::vector<Eigen::Vector2f>& pts_robot, const Eigen::Vector3f& pose, const rc::reloc::Params& p);

// dll[i] = ell(rivals[i]) - ell(incumbent) on one scan, in the order of `rivals` (what Belief::update consumes).
// Subsampling and BeyondLines are computed once for all poses.
std::vector<float> dll(const rc::reloc::LikelihoodRaster& raster, const std::vector<Eigen::Vector2f>& polygon,
                       const std::vector<Eigen::Vector2f>& pts_robot, const Eigen::Vector3f& incumbent,
                       const std::vector<Eigen::Vector3f>& rivals, const rc::reloc::Params& p);
}
