/*
 * common/motion_filter/cv_filter.h
 *
 * Decoupled constant-velocity (CV) Kalman filter over N independent 1-D position axes (each a 2-state
 * [pos, vel]). For MOVABLE concept objects (a bottle) whose pose TRACKS rather than hardens — unlike
 * furniture (table/chair), whose centre is static and is handled by the harden info-filter.
 *
 *   predict(dt): advance the belief by the estimated velocity; inflate the covariance by an
 *                acceleration process noise → the association gate WIDENS during motion.
 *   correct(z,R): fold a per-frame position measurement → the covariance tightens.
 *
 * When the object is still the estimated velocity decays to ~0 and the position tightens, so the filter
 * is "static-when-static, tracking-when-moving" (it subsumes the plastic-position fix). Header-only,
 * one filter per instance. The grasp case (known control input) is a later mode — see CONCEPT notes.
 */

#pragma once

#include <vector>
#include <Eigen/Dense>

namespace rc {

class CVFilter
{
public:
    // accel_std: how fast velocity can change (m/s² std) → process noise. init_vel_std: 1-σ on the
    // initial (unknown) velocity (m/s).
    void configure(double accel_std, double init_vel_std)
    {
        accel_var_    = accel_std * accel_std;
        init_vel_var_ = init_vel_std * init_vel_std;
    }

    bool   initialized() const { return init_; }
    int    size()        const { return static_cast<int>(pos_.size()); }
    double position(int a) const { return pos_[a]; }
    double velocity(int a) const { return vel_[a]; }
    double pos_var(int a)  const { return P_[a](0, 0); }   // for the association gate

    void init(const std::vector<double>& z, const std::vector<double>& meas_var)
    {
        const std::size_t n = z.size();
        pos_ = z;
        vel_.assign(n, 0.0);
        P_.assign(n, Eigen::Matrix2d::Zero());
        for (std::size_t a = 0; a < n; ++a)
        {
            P_[a](0, 0) = meas_var[a];
            P_[a](1, 1) = init_vel_var_;
        }
        init_ = true;
    }

    // x = F x ;  P = F P Fᵀ + Q   with F = [[1,dt],[0,1]], Q = white-acceleration discretisation.
    void predict(double dt)
    {
        if (not init_ or dt <= 0.0)
            return;
        const double q11 = accel_var_ * dt * dt * dt * dt / 4.0;
        const double q12 = accel_var_ * dt * dt * dt / 2.0;
        const double q22 = accel_var_ * dt * dt;
        for (std::size_t a = 0; a < pos_.size(); ++a)
        {
            pos_[a] += vel_[a] * dt;
            const double p00 = P_[a](0, 0), p01 = P_[a](0, 1), p10 = P_[a](1, 0), p11 = P_[a](1, 1);
            P_[a](0, 0) = p00 + dt * (p01 + p10) + dt * dt * p11 + q11;
            P_[a](0, 1) = p01 + dt * p11 + q12;
            P_[a](1, 0) = p10 + dt * p11 + q12;
            P_[a](1, 1) = p11 + q22;
        }
    }

    // Standard 1-D KF update per axis, H = [1, 0].
    void correct(const std::vector<double>& z, const std::vector<double>& meas_var)
    {
        if (not init_)
        {
            init(z, meas_var);
            return;
        }
        for (std::size_t a = 0; a < pos_.size(); ++a)
        {
            const double S = P_[a](0, 0) + meas_var[a];
            if (S <= 0.0)
                continue;
            const double k0 = P_[a](0, 0) / S, k1 = P_[a](1, 0) / S;
            const double y = z[a] - pos_[a];
            pos_[a] += k0 * y;
            vel_[a] += k1 * y;
            const double p00 = P_[a](0, 0), p01 = P_[a](0, 1), p10 = P_[a](1, 0), p11 = P_[a](1, 1);
            P_[a](0, 0) = (1.0 - k0) * p00;
            P_[a](0, 1) = (1.0 - k0) * p01;
            P_[a](1, 0) = p10 - k1 * p00;
            P_[a](1, 1) = p11 - k1 * p01;
        }
    }

private:
    std::vector<double>          pos_, vel_;
    std::vector<Eigen::Matrix2d> P_;
    double accel_var_    = 0.25;   // (0.5 m/s²)²
    double init_vel_var_ = 1.0;    // (1 m/s)²
    bool   init_ = false;
};

}  // namespace rc
