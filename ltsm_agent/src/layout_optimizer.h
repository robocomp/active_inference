/*
 * layout_optimizer.h — the metric layout of long-term memory as a MAP estimate, not a spanning tree.
 *
 * topology_layout.cpp places rooms by a breadth-first walk over the passage seams: the first seam that reaches a
 * room decides where it is, a second passage into it is only reported (loop closure), and nothing knows that rooms
 * are disjoint, that buildings are mostly square, or that the two sides of a doorway are one wall. This module
 * puts all of that into ONE generative model over the rooms' SE(2) poses in the map frame and returns the mode of
 * the posterior and its Laplace covariance. The negative log-joint it minimises is the free energy of that model
 * under a Gaussian (Laplace) posterior; nothing is gated, every effect is a likelihood term with a precision:
 *
 *   SEAM      every passage whose two faces compose: T(a<-b) observed with the covariance propagated from the two
 *             door faces' RT covariances. Tree AND loop passages alike, so a loop's closure error is shared out by
 *             precision instead of being dumped on the last edge. Each seam is an INLIER/OUTLIER MIXTURE, as in
 *             seam_fusion.h: N(z, Σ) with prior ½, or uniform over where b's frame can be relative to a's (a disc of
 *             radius = the two rooms' extents, any yaw). A seam that contradicts the rest by many σ stops pulling and
 *             reports P(inlier) ≈ 0 — gold2 holds two passages between the same rooms that disagree by 6.7 m.
 *   MANHATTAN one latent BUILDING AXIS ψ per component. Each room's dominant wall direction (quadrupled-angle vote of
 *             its outline) is either square to ψ (von Mises on 4·(room axis − ψ), concentration from how clearly the
 *             outline votes and how square buildings are) or unrelated to it (uniform) — a two-component mixture, so a
 *             30° room is explained as "not Manhattan" rather than forced square. P(Manhattan | pose) is reported.
 *   OVERLAP   rooms are disjoint free space. Each sample of room b — on its outline AND on a grid over its interior —
 *             is outside room a with probability Φ(d/σ), d the signed distance to a's outline and σ the two outlines'
 *             combined vertex uncertainty; energy −log Φ. Smooth and one-sided: ~0 a few σ outside, quadratic inside.
 *             Samples are weighted by the length/area they stand for over a correlation length, so a densely sampled
 *             wall does not count as many walls. (Boundary-only sampling let two rooms sit exactly on top of each other
 *             at almost no cost — their outlines coincide.)
 *   WALL      the two faces of a passage sit in ONE wall: the outline edge each door lies on (the edge whose line
 *             best contains both aperture ends — not the nearest edge, which on a notched outline is a door JAMB) is
 *             parallel to the other side's and separated from it by the wall thickness, a prior. It is evidence from
 *             the same door faces as the seam, so a passage is right or wrong AS A WHOLE: seam and wall share one
 *             inlier/outlier mixture.
 *   ADJACENCY a passage joins rooms that touch: b's origin is within b's own reach of the doorway in a. Always true, so
 *             not in the mixture; a broad Gaussian (σ = that reach) that is negligible beside a good seam and is what
 *             keeps a room whose every seam is disowned from drifting off (it went 288 m in a real memory without it).
 *
 * Every residual is whitened, so the cost is ½Σr² and Levenberg–Marquardt applies; the non-Gaussian terms enter as
 * r = ±√(2(E − E_min)), which leaves their energy exact. Jacobians are numeric per factor (each factor touches at
 * most two rooms and ψ), so a solve costs milliseconds for tens of rooms. Pure Eigen: no DSR, no Qt.
 */
#pragma once

#include <Eigen/Dense>

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace ltsm::layout
{
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;   ///< (x, y, yaw)
using Mat3 = Eigen::Matrix3d;
using Polygon = std::vector<Vec2>;

struct Room
{
    std::string name;
    Polygon outline;          ///< in the room's own frame; empty ⇒ the room enters only through its seams
    Vec3 init = Vec3::Zero(); ///< T(map <- room) to start from
    bool fixed = false;       ///< the gauge: this room's pose IS the map frame (at least one per call, else room 0)
};

struct Seam
{
    std::string name;
    std::size_t a = 0, b = 0;           ///< indices into the rooms
    Vec3 z = Vec3::Zero();              ///< observed T(a <- b), additive (x, y, yaw) as cortex stores RT covariance
    Mat3 cov = Mat3::Identity();        ///< its covariance, in a's frame
    /// The aperture's two ends in a's / b's own frame (the door face, width_m along its x): which outline edge is the
    /// door's wall (WALL), and where b must be relative to a (ADJACENCY).
    std::optional<std::array<Vec2, 2>> aperture_a, aperture_b;
    double extent = 0;                  ///< m, how far b's origin can be from a's (the rooms' radii summed): the
                                        ///< outlier component's support. 0 ⇒ no outlier component (pure Gaussian)
};

/// The model's constants. Each one is a statement about buildings or sensors, FLAGGED here, not a tuning knob.
struct Model
{
    double sigma_outline = 0.05;    ///< m, one outline vertex (room_concept's published corner σ is 2–6 cm)
    double sample_step = 0.25;      ///< m, outline sampling for OVERLAP (a discretisation, not a model constant)
    double interior_step = 0.5;     ///< m, interior grid for OVERLAP (a discretisation, not a model constant)
    double corr_length = 0.5;       ///< m, outline error correlation length: samples closer than this are not independent
    double wall_t_mean = 0.12;      ///< m, interior wall thickness (inner surface to inner surface)
    double wall_t_sigma = 0.08;     ///< m, spread of it across buildings
    double sigma_square = 0.0175;   ///< rad (1°), how square a square building is
    double p_manhattan = 0.5;       ///< prior P(a room is square to the building): ignorance
    double p_seam_inlier = 0.5;     ///< prior P(a seam is right): ignorance, as seam_fusion.h
    bool seams = true, manhattan = true, overlap = true, walls = true, adjacency = true;
    int max_iterations = 100;
    bool verbose = false;           ///< per-iteration F on stderr (diagnosis only)
};

struct Terms
{
    double seam = 0, manhattan = 0, overlap = 0, wall = 0, adjacency = 0;
    [[nodiscard]] double total() const { return seam + manhattan + overlap + wall + adjacency; }
};

struct Result
{
    std::vector<Vec3> pose;                 ///< T(map <- room), per room
    std::vector<Mat3> cov;                  ///< Laplace covariance per room (zero for the fixed room)
    std::vector<double> p_manhattan;        ///< P(room square to ψ | optimum); −1 where the room has no outline
    std::vector<double> seam_d2;            ///< Mahalanobis² of each seam at the optimum
    std::vector<double> seam_p_inlier;      ///< P(seam right | optimum); 1 for a seam with no outlier component
    std::optional<double> building_axis;    ///< ψ (rad, mod π/2), when any room votes one
    Terms before, after;                    ///< energy per family at the start and at the optimum
    int iterations = 0;
    bool converged = false;
    int starts = 1;          ///< initial guesses tried: the given poses + one spanning tree per seam
    int winning_start = 0;   ///< 0 = the given poses won
};

/// One connected component's rooms and the seams among them. Rooms with `fixed` are not moved. Solved from the given
/// poses AND from one spanning tree per seam (that seam walked first); the start reaching the lowest energy wins.
Result optimise(const std::vector<Room> &rooms, const std::vector<Seam> &seams, const Model &model = {});

/// A room's dominant wall direction from its outline: axis φ (rad, in [−π/4, π/4)) and the standard error of 4φ.
/// nullopt for an outline with no edges. Exposed for the test.
struct Axis { double phi = 0; double se4 = 0; };
std::optional<Axis> dominant_axis(const Polygon &outline);

/// Signed distance from p to the polygon's boundary: positive outside, negative inside. Exposed for the test.
double signed_distance(const Polygon &poly, const Vec2 &p);

}   // namespace ltsm::layout
