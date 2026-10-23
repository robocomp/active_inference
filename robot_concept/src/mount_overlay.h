#pragma once
// Pure parse / check / compose / format functions of the sensor-mount overlay. No DSR, no Qt, so they are
// unit-testable (tests/mount_overlay_test.cpp). SpecificWorker::apply_mount_overlay does only the graph I/O.
//
// FILE  etc/mount_calib_<robot>_<world>.txt   (<world> = webots | real), one CORRECTION per sensor:
//     # robot: shadow
//     # world: webots
//     <sensor> x y z roll pitch yaw version iso_date source...
//   x y z in metres, roll pitch yaw in radians. `source` is the rest of the line.
//
// CONVENTION  The file holds a correction, never an absolute mount:  mount = correction ∘ nominal, where the
//   correction is expressed in BODY-ALIGNED axes at the sensor origin:
//       t = t_nominal + t_corr            R = Rx(roll) Ry(pitch) Rz(yaw) * R_nominal
//   Left-multiplying (not nominal * corr) is deliberate: helios and bpearl are mounted with rot y=pi in
//   shadow.json, so a yaw about the SENSOR's own z would turn the opposite way in the robot frame and a
//   z offset would flip sign. Calibration speaks the robot's language ("helios is 1.84 deg off in yaw,
//   bpearl 5 mm low"), and in body axes that reads the same for all four sensors. For zed/ricoh
//   (nominal rotation 0) the two conventions coincide.
//   A zero rotation correction returns the nominal Euler triple verbatim (no re-extraction).
//
// WORLD  A file whose `# world:` / `# robot:` tag differs from the configured one is REFUSED (a real-robot
//   calibration must never move a Webots mount, or the reverse). A missing tag is a mismatch.
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace rc::mount
{
using Vec3 = std::array<double, 3>;
using Mat3 = Eigen::Matrix3d;

struct Correction { Vec3 t{}, rpy{}; int version = 0; std::string date, source; };
struct Mount      { Vec3 t{}, rpy{}; };
struct Parsed
{
	std::string robot, world;
	std::map<std::string, Correction> rows;
	std::vector<std::string> warnings;          // rows that did not parse: IGNORED, never half-applied
};

/// The four head sensors, Helios first: a Helios commit rotates every map/pose, so it precedes the cameras.
inline constexpr std::array<const char*, 4> kSensors{"helios", "bpearl", "zed", "ricoh"};

/// cortex rt_rotation_euler_xyz convention: R = Rx * Ry * Rz.
[[nodiscard]] inline Mat3 euler_to_matrix(const Vec3& e)
{
	return (Eigen::AngleAxisd(e[0], Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(e[1], Eigen::Vector3d::UnitY())
	        * Eigen::AngleAxisd(e[2], Eigen::Vector3d::UnitZ())).toRotationMatrix();
}
[[nodiscard]] inline Vec3 matrix_to_euler(const Mat3& R)       // inverse of euler_to_matrix, pitch in [-pi/2, pi/2]
{
	const double b = std::asin(std::clamp(R(0, 2), -1.0, 1.0));
	return {std::atan2(-R(1, 2), R(2, 2)), b, std::atan2(-R(0, 1), R(0, 0))};
}
[[nodiscard]] inline bool is_zero(const Vec3& v) { return v[0] == 0.0 and v[1] == 0.0 and v[2] == 0.0; }

namespace detail
{
inline std::vector<std::string> split_ws(const std::string& s)
{
	std::vector<std::string> out; std::istringstream is(s); is.imbue(std::locale::classic());
	for (std::string w; is >> w;) out.push_back(w);
	return out;
}
[[nodiscard]] inline bool num(const std::string& s, double& out)
{
	const auto r = std::from_chars(s.data(), s.data() + s.size(), out);
	return r.ec == std::errc() and r.ptr == s.data() + s.size();
}
[[nodiscard]] inline bool inum(const std::string& s, int& out)
{
	const auto r = std::from_chars(s.data(), s.data() + s.size(), out);
	return r.ec == std::errc() and r.ptr == s.data() + s.size();
}
} // namespace detail

[[nodiscard]] inline Parsed parse_overlay(const std::string& text)
{
	Parsed p;
	std::istringstream in(text);
	for (std::string line; std::getline(in, line);)
	{
		const auto tok = detail::split_ws(line);
		if (tok.empty()) continue;
		if (tok[0][0] == '#')
		{
			if (tok.size() >= 3 and tok[1] == "robot:") p.robot = tok[2];
			if (tok.size() >= 3 and tok[1] == "world:") p.world = tok[2];
			continue;
		}
		Correction c; bool ok = tok.size() >= 10;
		for (int i = 0; ok and i < 3; ++i) ok = detail::num(tok[1 + i], c.t[i]) and detail::num(tok[4 + i], c.rpy[i]);
		ok = ok and detail::inum(tok[7], c.version);
		if (not ok) { p.warnings.push_back("row did not parse, ignored: " + line); continue; }
		c.date = tok[8];
		for (std::size_t i = 9; i < tok.size(); ++i) c.source += (i > 9 ? " " : "") + tok[i];
		p.rows[tok[0]] = c;
	}
	return p;
}

/// nullopt = the file may be applied; otherwise the one warning to log.
[[nodiscard]] inline std::optional<std::string> check_world(const Parsed& p, const std::string& robot, const std::string& world)
{
	if (world != "webots" and world != "real")
		return "Transforms.world is '" + world + "' (must be webots|real): mount overlay refused";
	if (p.world != world)
		return "mount overlay is tagged world '" + p.world + "' but this run is '" + world + "': refused, nominal mounts stand";
	if (p.robot != robot)
		return "mount overlay is tagged robot '" + p.robot + "' but this run is '" + robot + "': refused, nominal mounts stand";
	return std::nullopt;
}

/// Convention in the header comment.
[[nodiscard]] inline Mount compose(const Vec3& nom_t, const Vec3& nom_rpy, const Correction& c)
{
	Mount m{nom_t, nom_rpy};
	for (int i = 0; i < 3; ++i) m.t[i] += c.t[i];
	if (not is_zero(c.rpy)) m.rpy = matrix_to_euler(euler_to_matrix(c.rpy) * euler_to_matrix(nom_rpy));
	return m;
}

/// Inverse of compose: the correction that turns nominal into live.
[[nodiscard]] inline Correction correction_from(const Vec3& nom_t, const Vec3& nom_rpy, const Vec3& live_t, const Vec3& live_rpy)
{
	Correction c;
	for (int i = 0; i < 3; ++i) c.t[i] = live_t[i] - nom_t[i];
	c.rpy = matrix_to_euler(euler_to_matrix(live_rpy) * euler_to_matrix(nom_rpy).transpose());
	return c;
}

/// Monotonic per-sensor version: the next one to write.
[[nodiscard]] inline int next_version(const std::map<std::string, Correction>& rows, const std::string& sensor)
{
	const auto it = rows.find(sensor);
	return it == rows.end() ? 1 : it->second.version + 1;
}

/// Locale-proof output (classic locale, round-trip precision).
[[nodiscard]] inline std::string format_overlay(const std::string& robot, const std::string& world,
                                                const std::map<std::string, Correction>& rows)
{
	std::ostringstream o; o.imbue(std::locale::classic());
	o << std::setprecision(std::numeric_limits<double>::max_digits10);
	o << "# robot: " << robot << "\n# world: " << world << '\n'
	  << "# CORRECTIONS in body-aligned axes: applied over the nominal mount. Delete the file to return to nominal.\n"
	  << "# sensor x y z roll pitch yaw version iso_date source\n";
	for (const auto& [s, c] : rows)
		o << s << ' ' << c.t[0] << ' ' << c.t[1] << ' ' << c.t[2] << ' ' << c.rpy[0] << ' ' << c.rpy[1] << ' '
		  << c.rpy[2] << ' ' << c.version << ' ' << (c.date.empty() ? "unknown" : c.date) << ' '
		  << (c.source.empty() ? "unknown" : c.source) << '\n';
	return o.str();
}
} // namespace rc::mount
