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
// EULER  The edge stores a Euler triple, but compose() works on matrices and re-extracts the triple
//   (pitch in [-pi/2, pi/2]). For a Ry(pi) mount that is an EQUIVALENT triple, not the nominal one with
//   the yaw added: helios (0, pi, 0) with yaw +a is stored as (+-pi, 0, pi - a) (the sign of pi is whatever atan2 returns). Consumers that read the
//   raw Euler angles of a corrected edge see this; consumers that build the rotation are unaffected.
//
// WORLD  A file whose `# world:` / `# robot:` tag differs from the configured one is REFUSED (a real-robot
//   calibration must never move a Webots mount, or the reverse). A missing tag is a mismatch.
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <charconv>
#include <functional>
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
// ── GRAPH SEAM ───────────────────────────────────────────────────────────────────────────────────
// apply/persist decisions, with the edge read/write behind callables so a test can fake the graph.
struct EdgeIO
{
	std::function<std::optional<Mount>(const std::string&)> read;          // body->sensor edge, nullopt if absent
	std::function<void(const std::string&, const Mount&)>   write;         // static RT write
};
struct State
{
	std::map<std::string, Mount>      nominal;       // edges as seeded, before the overlay
	std::map<std::string, Correction> rows;          // corrections the file held at start
	bool refused = false;                            // file existed but was refused: never overwrite it
};

/// `text` = file contents, nullopt if the file does not exist. Returns warnings (one per refusal/bad row).
inline std::vector<std::string> apply_overlay(const std::optional<std::string>& text, const std::string& robot,
                                              const std::string& world, const EdgeIO& io, State& st)
{
	std::vector<std::string> warn;
	for (const char* s : kSensors)
		if (const auto m = io.read(s)) st.nominal[s] = *m;
	if (not text) return warn;
	const Parsed p = parse_overlay(*text);
	if (const auto why = check_world(p, robot, world)) { st.refused = true; warn.push_back(*why); return warn; }
	warn = p.warnings;
	st.rows = p.rows;
	for (const char* s : kSensors)                                         // Helios before the cameras
	{
		const auto it = p.rows.find(s); const auto nom = st.nominal.find(s);
		if (it != p.rows.end() and nom != st.nominal.end())
			io.write(s, compose(nom->second.t, nom->second.rpy, it->second));
	}
	return warn;
}

/// New file contents to write, or nullopt (nothing to write, or the overlay was refused: warning added).
inline std::optional<std::string> persist_overlay(const std::string& robot, const std::string& world, const EdgeIO& io,
                                                  const State& st, const std::string& now, std::vector<std::string>& warn)
{
	if (st.refused) { warn.push_back("mount overlay was refused at start: persist skipped, file not overwritten"); return std::nullopt; }
	std::map<std::string, Correction> out = st.rows;
	bool changed = false;
	for (const char* s : kSensors)
	{
		const auto nom = st.nominal.find(s); const auto live = io.read(s);
		if (nom == st.nominal.end() or not live) continue;
		Correction c = correction_from(nom->second.t, nom->second.rpy, live->t, live->rpy);
		const auto old = out.find(s);
		const Correction prev = old == out.end() ? Correction{} : old->second;
		double d = 0;                                  // float edge storage: compare at float resolution
		for (int i = 0; i < 3; ++i) d = std::max({d, std::abs(c.t[i] - prev.t[i]), std::abs(c.rpy[i] - prev.rpy[i])});
		if (d < 1e-6) continue;
		c.version = next_version(out, s); c.date = now; c.source = "robot_concept:persist";
		out[s] = c; changed = true;
	}
	if (not changed) return std::nullopt;
	return format_overlay(robot, world, out);
}
} // namespace rc::mount
