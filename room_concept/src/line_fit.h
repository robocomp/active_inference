/*
 *  line_fit.h — 2-D line primitives shared by the corner detector and the wall segmenter.
 *
 *  These three (Line2D, fit_line_pca, intersect) were private statics of CornerDetector. The wall
 *  segmenter needs the same fit and the same intersection, and two copies of a PCA line fit would
 *  disagree the first time one of them was touched. CornerDetector keeps forwarding aliases so its
 *  behaviour is byte-identical.
 *
 *  A line is stored in Hesse normal form  n·p = d  with |n| = 1. Its angle φ = atan2(n_y, n_x) and
 *  offset d are the two parameters a wall landmark carries (see wall_map.h); the tangent
 *  t(φ) = (−sin φ, cos φ) is the direction along the wall.
 */
#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <optional>
#include <vector>

namespace rc::linefit
{
    /// 2D line: normal · p = d   (normal is unit length)
    struct Line2D
    {
        Eigen::Vector2f normal;
        float d = 0.f;
        float resid_var = 0.f;   // λ_min / N — mean squared perpendicular scatter of the fitted points
                                 // about the line (≈ sensor noise for a clean wall; large for a cluttered
                                 // gather). Inflates σ_L so a poorly-fit wall is trusted less.
        int   npts = 0;          // number of points the line was fit to
        Eigen::Vector2f direction() const { return Eigen::Vector2f(-normal.y(), normal.x()); }
        float phi() const { return std::atan2(normal.y(), normal.x()); }
    };

    inline Eigen::Vector2f normal_of(float phi) { return {std::cos(phi), std::sin(phi)}; }
    inline Eigen::Vector2f tangent_of(float phi) { return {-std::sin(phi), std::cos(phi)}; }
    inline float wrap_pi(float a) { return std::atan2(std::sin(a), std::cos(a)); }

    /// PCA line fit — returns nullopt if fewer than min_points.
    inline std::optional<Line2D> fit_line_pca(const std::vector<Eigen::Vector2f>& pts, int min_points)
    {
        if (static_cast<int>(pts.size()) < min_points)
            return std::nullopt;

        Eigen::Vector2f centroid = Eigen::Vector2f::Zero();
        for (const auto& p : pts)
            centroid += p;
        centroid /= static_cast<float>(pts.size());

        Eigen::Matrix2f scatter = Eigen::Matrix2f::Zero();
        for (const auto& p : pts)
        {
            Eigen::Vector2f dp = p - centroid;
            scatter += dp * dp.transpose();
        }

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eig(scatter);
        Line2D line;
        line.normal = eig.eigenvectors().col(0);  // smallest eigenvalue = line normal
        line.d = line.normal.dot(centroid);
        // Smallest eigenvalue = Σ perpendicular² of the points about the fitted line.
        // Per-point mean square scatter → a graded quality signal for σ_L (clean wall ≈ sensor noise).
        line.npts = static_cast<int>(pts.size());
        line.resid_var = std::max(0.f, eig.eigenvalues()(0)) / static_cast<float>(line.npts);
        return line;
    }

    /// Intersect two lines.  Returns nullopt if (nearly) parallel.
    inline std::optional<Eigen::Vector2f> intersect(const Line2D& a, const Line2D& b,
                                                    float* angle_deg = nullptr)
    {
        const float det = a.normal.x() * b.normal.y() - a.normal.y() * b.normal.x();
        if (std::abs(det) < 1e-6f)
            return std::nullopt;

        const float x = (b.normal.y() * a.d - a.normal.y() * b.d) / det;
        const float y = (a.normal.x() * b.d - b.normal.x() * a.d) / det;

        if (angle_deg)
        {
            // Angle between the two wall directions (= 180° - angle between normals)
            const float ndot = std::abs(a.normal.dot(b.normal));
            const float clamped = std::min(1.0f, ndot);
            *angle_deg = 180.0f - std::acos(clamped) * 180.0f / static_cast<float>(M_PI);
        }

        return Eigen::Vector2f(x, y);
    }

    /// Information matrix of the (φ, d) line parameters given the points it was fitted to, under an
    /// isotropic per-point perpendicular noise of variance σ². For the residual r_i = n(φ)·p_i − d,
    ///   ∂r_i/∂φ = t(φ)·p_i ,   ∂r_i/∂d = −1
    /// so  Λ = σ⁻² Σ_i [ (t·p_i)²  −(t·p_i) ;  −(t·p_i)  1 ].
    /// The φ row is dominated by how far the points spread ALONG the wall (a short segment says
    /// little about its angle), which is exactly what a corner intersection needs to know.
    inline Eigen::Matrix2f info_phi_d(const std::vector<Eigen::Vector2f>& pts, const Line2D& line,
                                      float sigma2)
    {
        Eigen::Matrix2f L = Eigen::Matrix2f::Zero();
        if (sigma2 <= 0.f) return L;
        const Eigen::Vector2f t = line.direction();
        for (const auto& p : pts)
        {
            const float s = t.dot(p);
            L(0, 0) += s * s;
            L(0, 1) -= s;
            L(1, 1) += 1.f;
        }
        L(1, 0) = L(0, 1);
        return L / sigma2;
    }
} // namespace rc::linefit
