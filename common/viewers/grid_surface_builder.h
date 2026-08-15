/*
 *  grid_surface_builder.h — shared builder for the residual occupancy "surprise landscape" surface.
 *
 *  Turns the residual grid's occupied cells (each carrying a real top height in z) into a coarse,
 *  smooth heightfield mesh: a planar lattice deformed FROM BELOW by a SUM OF GAUSSIANS (one bump per
 *  occupied cell, amplitude = its height, MAX-blended so a cluster becomes a plateau at obstacle
 *  height, never additive spires). Cheap scatter-splat (±3σ window per cell) — NOT neural splatting.
 *
 *  Output is frame-agnostic: positions/normals are in a right-handed ROOM frame (X=room x, Y=room y,
 *  Z=height up) and colours run blue→orange→red with height. Each consumer maps to its own axis
 *  convention and lights/bakes shading as it likes, so the standalone lit viewer and the retina's
 *  unlit scene render the SAME geometry from one code path.
 */
#ifndef RC_COMMON_GRID_SURFACE_BUILDER_H
#define RC_COMMON_GRID_SURFACE_BUILDER_H

#include <QVector3D>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace rc::viewers
{

struct GridSurfaceVertex { QVector3D pos; QVector3D nrm; QVector3D col; };   // ROOM frame (Z up); col rgb 0..1

// Tuning shared by every consumer so the views match.
inline constexpr float kGridBaseZ     = 0.02f;   // lift the surface just off the floor
inline constexpr float kGridColorRef  = 1.20f;   // height (m) mapped to the top (red) of the ramp
inline constexpr float kGridHeightCap = 1.60f;   // clamp displayed obstacle height (guards a runaway z-band)
inline constexpr float kGridFloorEps  = 0.03f;   // quads with all corners below this are floor → not emitted

// Height (m, above the floor) → blue→orange→red ramp (dark floor at 0).
inline std::array<float, 3> grid_height_ramp(float h_m)
{
	static constexpr float S[4][4] = {{0.00f, 0.10f, 0.12f, 0.20f},   // floor: dark slate
	                                  {0.18f, 0.15f, 0.40f, 0.95f},   // low:   blue
	                                  {0.55f, 1.00f, 0.60f, 0.10f},   // mid:   orange
	                                  {1.00f, 0.95f, 0.15f, 0.12f}};  // high:  red
	float h = std::clamp(h_m / kGridColorRef, 0.f, 1.f);
	for(int i = 0; i < 3; ++i)
		if(h <= S[i + 1][0])
		{
			const float u = (h - S[i][0]) / (S[i + 1][0] - S[i][0]);
			return {S[i][1] + u * (S[i + 1][1] - S[i][1]),
			        S[i][2] + u * (S[i + 1][2] - S[i][2]),
			        S[i][3] + u * (S[i + 1][3] - S[i][3])};
		}
	return {S[3][1], S[3][2], S[3][3]};
}

// Build the surprise-landscape surface as a flat triangle soup (3 verts per triangle) in the ROOM
// frame. `occupied` = residual cell centres (x, y, z = real top height, metres). The grid extent is
// [xmin, ymin] + cell·(w, h). Empty result if there is no valid grid.
inline std::vector<GridSurfaceVertex> build_residual_surface(std::span<const QVector3D> occupied,
                                                             float xmin, float ymin, float cell, int w, int h)
{
	std::vector<GridSurfaceVertex> out;
	if(w < 2 or h < 2 or cell <= 1e-6f)
		return out;

	const float x0 = xmin, y0 = ymin;
	const float x1 = xmin + w * cell, y1 = ymin + h * cell;
	const float sigma  = 2.0f * cell;                             // bump half-width (metres)
	const float span   = std::max(x1 - x0, y1 - y0);
	const float lspace = std::max(sigma * 0.6f, span / 160.f);    // lattice spacing (coarse), capped
	const int   LX = std::clamp(static_cast<int>(std::ceil((x1 - x0) / lspace)) + 1, 2, 200);
	const int   LY = std::clamp(static_cast<int>(std::ceil((y1 - y0) / lspace)) + 1, 2, 200);
	const float sx = (x1 - x0) / (LX - 1), sy = (y1 - y0) / (LY - 1);
	std::vector<float> LH(static_cast<std::size_t>(LX) * static_cast<std::size_t>(LY), 0.f);
	const float inv2s2 = 1.0f / (2.0f * sigma * sigma);
	const float radius = 3.0f * sigma;

	for(const QVector3D &oc : occupied)
	{
		const float amp = std::min(oc.z(), kGridHeightCap);      // real obstacle top height (m), capped
		if(amp <= kGridBaseZ) continue;
		const float cx = oc.x(), cy = oc.y();
		const int li0 = std::max(0,      static_cast<int>(std::floor((cx - radius - x0) / sx)));
		const int li1 = std::min(LX - 1, static_cast<int>(std::ceil ((cx + radius - x0) / sx)));
		const int lj0 = std::max(0,      static_cast<int>(std::floor((cy - radius - y0) / sy)));
		const int lj1 = std::min(LY - 1, static_cast<int>(std::ceil ((cy + radius - y0) / sy)));
		for(int lj = lj0; lj <= lj1; ++lj)
			for(int li = li0; li <= li1; ++li)
			{
				const float vx = x0 + li * sx, vy = y0 + lj * sy;
				const float d2 = (vx - cx) * (vx - cx) + (vy - cy) * (vy - cy);
				if(d2 > radius * radius) continue;
				const std::size_t k = static_cast<std::size_t>(lj) * LX + li;
				LH[k] = std::max(LH[k], amp * std::exp(-d2 * inv2s2));   // MAX-blend → plateau
			}
	}

	const auto lheight = [&](int li, int lj) { return LH[static_cast<std::size_t>(lj) * LX + li]; };
	const auto lvtx = [&](int li, int lj) -> GridSurfaceVertex
	{
		const float ht  = lheight(li, lj);
		const auto  col = grid_height_ramp(ht);
		const int il = std::max(0, li - 1), ir = std::min(LX - 1, li + 1);
		const int jl = std::max(0, lj - 1), jr = std::min(LY - 1, lj + 1);
		const float dHdx = (lheight(ir, lj) - lheight(il, lj)) / ((ir - il) * sx);
		const float dHdy = (lheight(li, jr) - lheight(li, jl)) / ((jr - jl) * sy);
		// ROOM frame with Z up → gradient normal (−dH/dx, −dH/dy, 1).
		QVector3D n(-dHdx, -dHdy, 1.0f);
		if(n.lengthSquared() > 1e-12f) n.normalize();
		return {QVector3D(x0 + li * sx, y0 + lj * sy, kGridBaseZ + ht), n, QVector3D(col[0], col[1], col[2])};
	};

	out.reserve(static_cast<std::size_t>(LX - 1) * (LY - 1) * 6);
	for(int lj = 0; lj + 1 < LY; ++lj)
		for(int li = 0; li + 1 < LX; ++li)
		{
			// Skip floor quads (all corners below eps) so the surface is ONLY the obstacle bumps.
			if(lheight(li, lj) < kGridFloorEps and lheight(li + 1, lj) < kGridFloorEps
			   and lheight(li, lj + 1) < kGridFloorEps and lheight(li + 1, lj + 1) < kGridFloorEps)
				continue;
			const GridSurfaceVertex v00 = lvtx(li, lj), v10 = lvtx(li + 1, lj),
			                        v01 = lvtx(li, lj + 1), v11 = lvtx(li + 1, lj + 1);
			out.push_back(v00); out.push_back(v10); out.push_back(v11);
			out.push_back(v00); out.push_back(v11); out.push_back(v01);
		}
	return out;
}

}   // namespace rc::viewers

#endif   // RC_COMMON_GRID_SURFACE_BUILDER_H
