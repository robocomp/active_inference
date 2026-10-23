// Offline test of the pure mount-overlay functions (src/mount_overlay.h): parse, world check, compose
// nominal∘correction, inverse, format round trip. No DSR, no Qt.
//   make -C build mount_overlay_test && ../bin/mount_overlay_test
#include "mount_overlay.h"
#include <clocale>
#include <cmath>
#include <cstdio>
#include <numbers>

using namespace rc::mount;
static int fails = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static constexpr double kPi  = std::numbers::pi;
static constexpr double kDeg = kPi / 180.0;

static const char* kFile =
    "# robot: shadow\n# world: webots\n"
    "# sensor x y z roll pitch yaw version iso_date source\n"
    "helios 0 0 0 0 0 0.0321140 3 2026-10-23T10:00:00 calib one\n"
    "bpearl 0 0 -0.005 0 0 0 2 2026-10-23T10:00:00 hand\n";

// Helios and bpearl sit with rot y=pi in shadow.json
static const Vec3 kNomT{0.0, -0.155, 1.075}, kNomR{0.0, kPi, 0.0};

static void test_parse_and_world()
{
    const Parsed p = parse_overlay(kFile);
    CHECK(p.robot == "shadow" and p.world == "webots");
    CHECK(p.rows.size() == 2 and p.warnings.empty());
    CHECK(p.rows.at("helios").version == 3 and p.rows.at("helios").source == "calib one");
    CHECK(std::abs(p.rows.at("helios").rpy[2] - 0.032114) < 1e-12);
    CHECK(std::abs(p.rows.at("bpearl").t[2] + 0.005) < 1e-12);
    CHECK(not check_world(p, "shadow", "webots").has_value());
    CHECK(check_world(p, "shadow", "real").has_value());     // mismatched world refused
    CHECK(check_world(p, "p3bot", "webots").has_value());    // mismatched robot refused
    CHECK(check_world(p, "shadow", "").has_value());         // unconfigured world refused
    CHECK(check_world(parse_overlay("helios 0 0 0 0 0 0 1 d s\n"), "shadow", "webots").has_value());  // no tag
}

static void test_bad_rows()
{
    const Parsed p = parse_overlay("# world: real\nzed 0 0 0 0 0 abc 1 d s\nzed 1 2\nricoh 0.5 0 0 0 0 0 1 d s\n");
    CHECK(p.rows.size() == 1 and p.rows.count("ricoh") == 1);   // partial rows are not mounts
    CHECK(p.warnings.size() == 2);
    CHECK(std::abs(p.rows.at("ricoh").t[0] - 0.5) < 1e-12);
}

static void test_compose_helios_yaw_bpearl_z()
{
    const Parsed p = parse_overlay(kFile);
    const Mount h = compose(kNomT, kNomR, p.rows.at("helios"));
    // yaw +1.84 deg about the BODY z axis: R = Rz(1.84deg) * R_nominal, translation untouched
    const Mat3 want = Eigen::AngleAxisd(0.0321140, Eigen::Vector3d::UnitZ()).toRotationMatrix() * euler_to_matrix(kNomR);
    CHECK((euler_to_matrix(h.rpy) - want).cwiseAbs().maxCoeff() < 1e-9);
    CHECK(std::abs(0.0321140 - 1.84 * kDeg) < 1e-6);           // the brief's number
    for (int i = 0; i < 3; ++i) CHECK(std::abs(h.t[i] - kNomT[i]) < 1e-12);
    // the body-frame heading of the sensor's forward axis moved by exactly +1.84 deg
    const Eigen::Vector3d f0 = euler_to_matrix(kNomR) * Eigen::Vector3d::UnitX();
    const Eigen::Vector3d f1 = euler_to_matrix(h.rpy) * Eigen::Vector3d::UnitX();
    CHECK(std::abs(std::remainder(std::atan2(f1.y(), f1.x()) - std::atan2(f0.y(), f0.x()), 2 * kPi) - 0.0321140) < 1e-9);

    const Mount b = compose(kNomT, kNomR, p.rows.at("bpearl"));
    CHECK(std::abs(b.t[2] - (kNomT[2] - 0.005)) < 1e-12);       // z equals nominal - 0.005 (1e-9)
    CHECK(b.rpy == kNomR);                                      // zero correction keeps nominal verbatim
}

static void test_inverse_and_format()
{
    const Parsed p = parse_overlay(kFile);
    const Mount h = compose(kNomT, kNomR, p.rows.at("helios"));
    const Correction c = correction_from(kNomT, kNomR, h.t, h.rpy);
    CHECK(std::abs(c.rpy[2] - 0.032114) < 1e-9 and std::abs(c.rpy[0]) < 1e-9 and std::abs(c.rpy[1]) < 1e-9);
    CHECK(c.t[0] == 0.0 or std::abs(c.t[0]) < 1e-12);

    std::setlocale(LC_ALL, "es_ES.UTF-8");                      // comma-decimal locale if installed
    std::map<std::string, Correction> rows = p.rows;
    const std::string txt = format_overlay("shadow", "webots", rows);
    CHECK(txt.find(',') == std::string::npos); if (txt.find(',') != std::string::npos) std::puts(txt.c_str());
    const Parsed q = parse_overlay(txt);
    CHECK(q.rows.size() == 2 and q.world == "webots" and q.robot == "shadow");
    CHECK(std::abs(q.rows.at("helios").rpy[2] - 0.032114) < 1e-12 and q.rows.at("helios").version == 3);
    CHECK(next_version(q.rows, "helios") == 4 and next_version(q.rows, "zed") == 1);
}

int main()
{
    test_parse_and_world(); test_bad_rows(); test_compose_helios_yaw_bpearl_z(); test_inverse_and_format();
    std::printf(fails ? "mount_overlay_test: %d FAILED\n" : "mount_overlay_test: ALL PASSED\n", fails);
    return fails != 0;
}
