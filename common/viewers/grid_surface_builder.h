/*
 *  grid_surface_builder.h — shared builder for the residual occupancy display: one COLUMN per cell.
 *
 *  Each occupied cell becomes an axis-aligned box on its OWN cell footprint, spanning the z-band the
 *  cell actually holds. The grid is a quantised thing and the columns say so: a cell is exactly one
 *  box wide, so what you read off the screen is what the planner reads out of the grid.
 *
 *  ★WHY NOT THE GAUSSIAN SPLAT IT REPLACES. That drew a smooth heightfield — a lattice deformed from
 *  below by a sum of MAX-blended bumps — which was pretty and lied twice. It spread every cell over
 *  ±3σ ≈ 6 cells, so a single-cell obstacle looked like a soft hill several times its footprint (the
 *  free space around it looked occupied, and the picket-fence failure mode — many cells each smaller
 *  than the robot — was invisible under one continuous plateau). And a heightfield has ONE surface per
 *  (x, y): it can only hang things from the floor, so it could not draw a cell whose evidence starts
 *  at 0.75 m and ends at 0.9 m. Columns carry a base, so they can float.
 *
 *  ★THE BASE IS PER CELL, NOT THE FLOOR. An unknown object ON A TABLE occupies a band well above z=0
 *  and nothing below it; drawing it from the floor would claim the volume under the table is blocked,
 *  which is both wrong and exactly the case a residual field exists to surface. `base_z` supplies each
 *  cell's band bottom; when it is empty every column is floor-standing, which is the honest fallback
 *  for a producer that does not publish a bottom yet.
 *
 *  Output is frame-agnostic: positions/normals are in a right-handed ROOM frame (X=room x, Y=room y,
 *  Z=height up) and colours run blue→orange→red with height. Each consumer maps to its own axis
 *  convention and lights/bakes shading as it likes, so every view renders the SAME geometry from one
 *  code path.
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
inline constexpr float kGridBaseZ      = 0.02f;   // lift a floor-standing column just off the floor
inline constexpr float kGridColorRef   = 1.20f;   // height (m) mapped to the top (red) of the ramp
inline constexpr float kGridHeightCap  = 1.60f;   // clamp displayed obstacle height (guards a runaway z-band)
inline constexpr float kGridFloorEps   = 0.03f;   // a floor-standing cell no taller than this IS the floor
inline constexpr float kGridMinColumnH = 0.04f;   // shortest column drawn: a cell reporting no thickness is
                                                 // still evidence, and a zero-height box is invisible

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

// Build the residual display as a flat triangle soup (3 verts per triangle) in the ROOM frame.
//   occupied : residual cell centres, z = the cell's CURRENT top height (m).
//   base_z   : the cell's band BOTTOM (m), one per entry of `occupied`, SAME order. Empty ⇒ every
//              column stands on the floor. A shorter span than `occupied` is treated as absent for
//              the entries it does not cover — never as a zero, which would silently plant a
//              floating column on the ground.
//   [xmin, ymin] + cell·(w, h) frames the lattice; it is used ONLY to find each cell's neighbours
//              for face culling, so a missing/invalid meta costs interior faces, not correctness.
inline std::vector<GridSurfaceVertex> build_residual_columns(std::span<const QVector3D> occupied,
                                                             std::span<const float> base_z,
                                                             float xmin, float ymin, float cell,
                                                             int w, int h)
{
	std::vector<GridSurfaceVertex> out;
	if(cell <= 1e-6f or occupied.empty())
		return out;

	const float half = 0.5f * cell;

	// The band a cell occupies, after clamping. Kept in one place because the culling pass below has
	// to agree with the emitting pass EXACTLY — a neighbour judged "covering" on a different top than
	// the one it is drawn with punches holes in the blob.
	const auto band_of = [&](std::size_t i, float& lo, float& hi)
	{
		hi = std::min(occupied[i].z(), kGridHeightCap);
		lo = (i < base_z.size()) ? std::max(base_z[i], 0.f) : kGridBaseZ;
		if(lo > hi) lo = hi;                       // a bottom above the top is not a band; collapse it
		if(hi - lo < kGridMinColumnH)              // ...and a collapsed band still has to be SEEN: a cell
			hi = lo + kGridMinColumnH;             // that reports no thickness is evidence, not nothing
	};

	// Neighbour index, so two adjacent columns do not draw the wall between them. Beyond the vertex
	// saving this is what makes a cluster read as one solid block: coincident interior faces z-fight
	// and speckle, and every one of them is a surface that does not exist.
	const bool have_meta = (w > 0 and h > 0);
	std::vector<int> at(have_meta ? static_cast<std::size_t>(w) * static_cast<std::size_t>(h) : 0u, -1);
	const auto cell_ix = [&](const QVector3D& c, int& ix, int& iy)
	{
		ix = static_cast<int>(std::floor((c.x() - xmin) / cell));
		iy = static_cast<int>(std::floor((c.y() - ymin) / cell));
		return ix >= 0 and iy >= 0 and ix < w and iy < h;
	};
	if(have_meta)
		for(std::size_t i = 0; i < occupied.size(); ++i)
		{
			int ix = 0, iy = 0;
			if(cell_ix(occupied[i], ix, iy))
				at[static_cast<std::size_t>(iy) * w + ix] = static_cast<int>(i);
		}
	// True when the neighbour at (ix+dx, iy+dy) spans at least [lo, hi] — i.e. it hides this whole face.
	const auto covered_by_neighbour = [&](int ix, int iy, int dx, int dy, float lo, float hi)
	{
		if(not have_meta) return false;
		const int nx = ix + dx, ny = iy + dy;
		if(nx < 0 or ny < 0 or nx >= w or ny >= h) return false;
		const int k = at[static_cast<std::size_t>(ny) * w + nx];
		if(k < 0) return false;
		float nlo = 0.f, nhi = 0.f;
		band_of(static_cast<std::size_t>(k), nlo, nhi);
		return nlo <= lo + 1e-4f and nhi >= hi - 1e-4f;
	};

	// One quad → two triangles, coloured PER VERTEX by that vertex's own height. A tall column then
	// carries the ramp up its side, and a floating one starts part-way up it — so the colour says how
	// high the evidence sits, not merely how thick it is.
	const auto quad = [&](const QVector3D& a, const QVector3D& b, const QVector3D& c, const QVector3D& d,
	                      const QVector3D& n)
	{
		const auto vtx = [&](const QVector3D& p)
		{
			const auto col = grid_height_ramp(p.z());
			return GridSurfaceVertex{p, n, QVector3D(col[0], col[1], col[2])};
		};
		const GridSurfaceVertex va = vtx(a), vb = vtx(b), vc = vtx(c), vd = vtx(d);
		out.push_back(va); out.push_back(vb); out.push_back(vc);
		out.push_back(va); out.push_back(vc); out.push_back(vd);
	};

	out.reserve(occupied.size() * 30);   // ~5 faces × 6 verts for a typical floor-standing column
	for(std::size_t i = 0; i < occupied.size(); ++i)
	{
		// ★TESTED ON THE RAW BAND, BEFORE band_of() CLAMPS IT. band_of applies the minimum column
		// height, so a cell reporting top=0 comes back 4 cm tall and would sail through a test written
		// against the drawn value — the floor would be re-drawn as a carpet of chips.
		const float raw_top  = occupied[i].z();
		const float raw_base = (i < base_z.size()) ? std::max(base_z[i], 0.f) : kGridBaseZ;
		// A floor-standing cell with nothing above the floor IS the floor. A FLOATING one is never
		// dropped on height alone — a thin band at 0.8 m is the whole point of this display.
		if(raw_base <= kGridBaseZ + 1e-4f and raw_top <= kGridFloorEps) continue;

		float lo = 0.f, hi = 0.f;
		band_of(i, lo, hi);

		const float cx = occupied[i].x(), cy = occupied[i].y();
		const float x0 = cx - half, x1 = cx + half, y0 = cy - half, y1 = cy + half;
		int ix = 0, iy = 0;
		const bool indexed = have_meta and cell_ix(occupied[i], ix, iy);

		// Sides. Skipped where the neighbouring column already spans the whole face.
		if(not(indexed and covered_by_neighbour(ix, iy, +1, 0, lo, hi)))
			quad({x1, y0, lo}, {x1, y1, lo}, {x1, y1, hi}, {x1, y0, hi}, {1.f, 0.f, 0.f});
		if(not(indexed and covered_by_neighbour(ix, iy, -1, 0, lo, hi)))
			quad({x0, y1, lo}, {x0, y0, lo}, {x0, y0, hi}, {x0, y1, hi}, {-1.f, 0.f, 0.f});
		if(not(indexed and covered_by_neighbour(ix, iy, 0, +1, lo, hi)))
			quad({x1, y1, lo}, {x0, y1, lo}, {x0, y1, hi}, {x1, y1, hi}, {0.f, 1.f, 0.f});
		if(not(indexed and covered_by_neighbour(ix, iy, 0, -1, lo, hi)))
			quad({x0, y0, lo}, {x1, y0, lo}, {x1, y0, hi}, {x0, y0, hi}, {0.f, -1.f, 0.f});

		quad({x0, y0, hi}, {x1, y0, hi}, {x1, y1, hi}, {x0, y1, hi}, {0.f, 0.f, 1.f});   // top cap
		// Underside ONLY when the column floats. On a floor-standing one it is a face nobody can see
		// that still costs a z-fight against the floor plane.
		if(lo > kGridBaseZ + 1e-3f)
			quad({x0, y1, lo}, {x1, y1, lo}, {x1, y0, lo}, {x0, y0, lo}, {0.f, 0.f, -1.f});
	}
	return out;
}

}   // namespace rc::viewers

#endif   // RC_COMMON_GRID_SURFACE_BUILDER_H
