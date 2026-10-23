// Offline check of rc::masks::rotate_azimuth_windows (no DSR, no Qt).
#include "../mask_azimuth_windows.h"
#include <cstdio>
#include <numbers>

static int fails = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

static float wrap(float a) { return std::atan2(std::sin(a), std::cos(a)); }
static bool near(float a, float b) { return std::abs(wrap(a - b)) < 1e-5f; }

int main()
{
    using std::numbers::pi_v;
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    const auto Rz = [](double t) { return Eigen::AngleAxisd(t, Eigen::Vector3d::UnitZ()).toRotationMatrix(); };

    // absent / empty attribute => empty (unknown)
    CHECK(rc::masks::rotate_azimuth_windows({}, Rz(1.0), true).empty());

    // identity (no transform needed): values pass through
    {
        const auto w = rc::masks::rotate_azimuth_windows({0.1f, 0.5f, 1.0f, 2.0f}, I, false);
        CHECK(w.size() == 2 and near(w[0].x(), 0.1f) and near(w[1].y(), 2.0f));
    }
    // yaw +0.3: both ends shift by 0.3
    {
        const auto w = rc::masks::rotate_azimuth_windows({-1.0f, 0.0f}, Rz(0.3), true);
        CHECK(w.size() == 1 and near(w[0].x(), -0.7f) and near(w[0].y(), 0.3f));
    }
    // wrap across +-pi: window (3.0, -3.0 [= 3.28]) shifted +0.5 -> from 3.5 -> -2.783, to -2.5
    {
        const auto w = rc::masks::rotate_azimuth_windows({3.0f, -3.0f}, Rz(0.5), true);
        CHECK(near(w[0].x(), 3.5f) and near(w[0].y(), -2.5f));
        CHECK(w[0].x() <= pi_v<float> and w[0].x() > -pi_v<float>);
        // CCW width preserved (0.2832 rad) after the seam crossing
        CHECK(near(wrap(w[0].y() - w[0].x()), wrap(-3.0f - 3.0f)));
    }
    // three adjacent strips stay adjacent after rotation
    {
        const float t = 2.0f * pi_v<float> / 3.0f;
        const auto w = rc::masks::rotate_azimuth_windows({-pi_v<float>, -pi_v<float> + t, -pi_v<float> + t, -pi_v<float> + 2 * t,
                                                          -pi_v<float> + 2 * t, pi_v<float>}, Rz(-1.2), true);
        CHECK(w.size() == 3 and near(w[0].y(), w[1].x()) and near(w[1].y(), w[2].x()) and near(w[2].y(), w[0].x()));
    }
    // odd trailing float dropped
    CHECK(rc::masks::rotate_azimuth_windows({0.f, 1.f, 2.f}, I, false).size() == 1);

    std::printf(fails ? "mask_azimuth_windows_test: %d FAILED\n" : "mask_azimuth_windows_test: all passed\n", fails);
    return fails ? 1 : 0;
}
