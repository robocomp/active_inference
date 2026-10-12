// Offline harness for door_frame_evidence (the doorway-FRAME existence channel), on REAL ZED frames saved by
// retina's [ZedDump] during the live try32 crossing (retina/etc/zed_frames/*.jpg + poses.csv, room frame of
// the room the robot was in at the time), plus a synthetic depth scene for the depth half.
//
//   (a) a SHUT door in view (room_1, door_1, green leaf)            → frame + leaf support, verdict > 0
//   (b) the try32 failure: door_3 seen from INSIDE room_2, leaf ~95° OPEN OUT of room_2 (not visible)
//       → the frame alone confirms; L does not fall
//   (c) PHANTOM apertures slid along the same walls onto blank wall → verdict ≤ ~0, never confirmed
//   (d) depth half on a synthetic wall: open hole confirms, flush/blank abstains, a leaf swung toward the
//       camera in front of the hole is weighted OUT rather than refuting
//   (e) existence integration: try32's leaf-silhouette numbers alone vs + the frame channel
//   (f) the WALL BAND (door_room_band.h): containment prior + leaf-beyond weight on try34's geometry
//   (g) SYNTHETIC rendered RGB + depth scene (room_2 frame, 0.11 m wall with reveals, leaf swung OUT
//       into room_1): the frame channel from inside the room the door opens out of — frontal, oblique
//       from both sides, far; a shut door; phantoms on the blank wall; a phantom mid-room
//   (h) REPLAY, 100 s: the carried open-out door while the robot backs away and turns (vis falling) holds
//       L high; a phantom on the blank wall is not confirmed by the frame and is removed by the leaf
//       silhouette; a phantom born mid-room is removed by the band prior even unobserved
// (a)-(c) need try32's saved ZED frames (retina/etc/zed_frames, [ZedDump]); they were deleted 2026-10-12
// to free 13 GB, so those sections now report SKIPPED instead of failing on missing files. (g)/(h) are
// self-contained.
//
// Build + run (from door_concept/):
//   g++ -std=c++23 -O2 -I/usr/include/eigen3 -I/usr/include/opencv4 tests/test_door_frame_evidence.cpp \
//       -o /tmp/test_door_frame_evidence -lopencv_core -lopencv_imgproc -lopencv_imgcodecs && /tmp/test_door_frame_evidence
// Optional: DUMP_OVERLAYS=<dir> writes each scored frame with the frame probe drawn on it.
#include <charconv>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <print>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "../src/door_frame_evidence.cpp"
#include "../../common/contour_edge/contour_edge_check.cpp"
#include "../../common/contour_edge/contour_edge_project.cpp"
#include "../../common/contour_edge/contour_depth_check.cpp"
#include "../../common/existence_belief/existence_belief.h"
#include "../../common/occlusion/occlusion.h"
#include "../src/door_room_band.h"

using namespace rc::door;

static int failures = 0;
static void check(bool ok, const std::string& what)
{
    std::println("  [{}] {}", ok ? "PASS" : "FAIL", what);
    if (not ok) ++failures;
}

static const std::string kFrames = "../retina/etc/zed_frames/";

struct CamPose { float x = 0, y = 0, z = 0, yaw = 0; };

// poses.csv: stamp_ms,file,cam_x,cam_y,cam_z,cam_yaw,...   (std::from_chars: the locale rule in CLAUDE.md)
static std::map<std::string, CamPose> load_poses()
{
    std::map<std::string, CamPose> out;
    std::ifstream f(kFrames + "poses.csv");
    std::string line;
    std::getline(f, line);
    while (std::getline(f, line))
    {
        std::vector<std::string> c;
        std::size_t a = 0;
        for (std::size_t b; (b = line.find(',', a)) != std::string::npos; a = b + 1) c.push_back(line.substr(a, b - a));
        c.push_back(line.substr(a));
        if (c.size() < 6) continue;
        CamPose p;
        const auto num = [](const std::string& s, float& v) { std::from_chars(s.data(), s.data() + s.size(), v); };
        num(c[2], p.x); num(c[3], p.y); num(c[4], p.z); num(c[5], p.yaw);
        out[c[0]] = p;
    }
    return out;
}

// ZED in this codebase's convention: x right, y FORWARD, z up; upright mount; fx = fy = 448, 1280x720
// (robot_concept/shadow.json). yaw_of(room_T_zed) = atan2(R10, R00) ⇒ camera +x = (cos, sin).
static rc::edges::PointProjector make_projector(const CamPose& c)
{
    return [c](const Eigen::Vector3d& P) -> std::optional<rc::edges::ProjectedVertex>
    {
        const double dx = P.x() - c.x, dy = P.y() - c.y, dz = P.z() - c.z;
        const double X = dx * std::cos(c.yaw) + dy * std::sin(c.yaw);
        const double Y = -dx * std::sin(c.yaw) + dy * std::cos(c.yaw);
        if (Y <= 0.20) return std::nullopt;
        rc::edges::ProjectedVertex v;
        v.px = cv::Point2f(static_cast<float>(640.0 + 448.0 * X / Y), static_cast<float>(360.0 - 448.0 * dz / Y));
        v.depth_m = static_cast<float>(Y);
        return v;
    };
}

static Aperture make_ap(Eigen::Vector2f O, Eigen::Vector2f u, float w, float h)
{
    Aperture a; a.wall_O = O; a.wall_u = u.normalized(); a.s = 0.0f; a.w = w; a.h = h; a.floor_z = 0.0f;
    a.thickness = 0.05f; return a;
}

static bool frame_exists(const std::string& stamp)
{
    return std::ifstream(kFrames + stamp + ".jpg").good();
}

static FrameScore score_on(const std::string& stamp, const std::map<std::string, CamPose>& poses, const Aperture& ap,
                           float* vis_out = nullptr, const char* tag = nullptr)
{
    if (not frame_exists(stamp) or not poses.contains(stamp)) return {};
    const cv::Mat img = cv::imread(kFrames + stamp + ".jpg", cv::IMREAD_COLOR);
    if (img.empty()) return {};
    const auto prep = rc::edges::prepare_frame(img);
    const auto probe = build_frame_probe(ap, make_projector(poses.at(stamp)), nullptr, nullptr, img.cols, img.rows);
    if (vis_out) *vis_out = probe.vis_frac();
    if (const char* dir = std::getenv("DUMP_OVERLAYS"); dir != nullptr and tag != nullptr)
    {
        cv::Mat o = img.clone();
        for (const auto& s : probe.samples)
        {
            cv::circle(o, s.px, 2, {0, 0, 255}, -1);
            for (const auto& c : s.ctl) if (c.w > 0.0f) cv::circle(o, c.px, 1, {255, 0, 0}, -1);
        }
        cv::imwrite(std::string(dir) + "/" + tag + "_" + stamp + ".png", o);
    }
    return score_frame(probe, &prep, nullptr);
}

static void print_score(const std::string& what, const FrameScore& s)
{
    std::println("    {:<44} exc={:+.2f} (t={:.1f} c={:.1f}) n={} ctl={} vis={:.2f} verdict={:+.2f}",
                 what, s.excess, s.s_true, s.s_control, s.n_edge, s.n_ctl, s.vis_frac, s.verdict());
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
// SYNTHETIC SCENE — room_2's frame as try34 saw it. The robot's room is y > 0; its south wall's ROOM-SIDE face
// is y = 0 (where room_concept draws the polygon), 0.11 m thick (the Webots walls), so the far face — room_1's
// side — is y = −0.11. The doorway is a hole x ∈ [hx0, hx0 + 1.0], z ∈ [0, 2.05] with REVEALS (the jamb side
// faces and the soffit). The leaf is hinged on the far (+x) jamb at the room_1 face and swung OUT into room_1.
// Forward depth + an 8-bit image are ray-cast per pixel with the same pinhole the projector uses (fx = 448,
// 1280x720). Shades are a plain bright room_2 (the "bright, plain walls"), a dimmer room_1 behind the hole, a
// wooden leaf, with a deterministic ±3 texture so the frame's own gradient reference is not degenerate.
// ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
namespace syn
{
constexpr float T = 0.11f;            // wall thickness
constexpr float HX0 = -0.60f, HW = 1.00f, HH = 2.05f;
constexpr float CEIL = 2.6f;

// Webots' Door.proto (tworooms-piso.wbt DOOR_1: size 0.1 x 1 x 2.1, frameSize 0.05): 0.05 m frame boards
// line the 1 m gap and stand 0.05 m PROUD of both wall faces; the leaf is surface-mounted on the room_1 side
// against the boards' outer face, 0.95 m wide.
constexpr float CASE_W = 0.05f, CASE_P = 0.05f;

struct Scene
{
    bool  hole = true;                // false ⇒ a blank wall (no doorway at all)
    int   leaf = 2;                   // 0 none · 1 shut · 2 swung OUT into room_1
    float phi  = 1.75f;               // opening angle for leaf == 2 (rad, ~100°)
    bool  casing = true;              // the Door.proto frame boards; false ⇒ a knife-edge opening
};

struct Hit { float t = 1e9f; float shade = 0.0f; };

inline float tex(float a, float b)    // deterministic ±3 texture
{
    const float v = std::sin(a * 37.1f + b * 11.7f) * 43758.5453f;
    return 6.0f * (v - std::floor(v)) - 3.0f;
}

// Ray o + t·d, d with unit FORWARD component (so t is the forward depth).
inline Hit trace(const Scene& sc, const Eigen::Vector3f& o, const Eigen::Vector3f& d)
{
    Hit h;
    const auto take = [&](float t, float shade) { if (t > 1e-4f and t < h.t) { h.t = t; h.shade = shade; } };
    const auto at = [&](float t) -> Eigen::Vector3f { return o + t * d; };
    const auto in_hole = [&](float x, float z) { return sc.hole and x > HX0 and x < HX0 + HW and z > 0.0f and z < HH; };
    // planes y = c
    const auto yplane = [&](float c, auto&& accept, float shade)
    {
        if (std::abs(d.y()) < 1e-7f) return;
        const float t = (c - o.y()) / d.y(); const auto p = at(t);
        if (accept(p)) take(t, shade + tex(p.x(), p.z()));
    };
    const auto xplane = [&](float c, auto&& accept, float shade)
    {
        if (std::abs(d.x()) < 1e-7f) return;
        const float t = (c - o.x()) / d.x(); const auto p = at(t);
        if (accept(p)) take(t, shade + tex(p.y(), p.z()));
    };
    const auto zplane = [&](float c, auto&& accept, float shade)
    {
        if (std::abs(d.z()) < 1e-7f) return;
        const float t = (c - o.z()) / d.z(); const auto p = at(t);
        if (accept(p)) take(t, shade + tex(p.x(), p.y()));
    };
    // room_2 side of the south wall, and room_1 side of it (visible only through the hole region's surround)
    yplane(0.0f, [&](const Eigen::Vector3f& p) { return std::abs(p.x()) < 3.0f and p.z() > 0 and p.z() < CEIL and not in_hole(p.x(), p.z()); }, 200.0f);
    yplane(-T,   [&](const Eigen::Vector3f& p) { return std::abs(p.x()) < 5.0f and p.z() > 0 and p.z() < CEIL and not in_hole(p.x(), p.z()); }, 140.0f);
    if (sc.hole)
    {   // reveals: the jamb side faces and the soffit, at the wall's own shade but lit differently
        xplane(HX0,      [&](const Eigen::Vector3f& p) { return p.y() > -T and p.y() < 0 and p.z() > 0 and p.z() < HH; }, 172.0f);
        xplane(HX0 + HW, [&](const Eigen::Vector3f& p) { return p.y() > -T and p.y() < 0 and p.z() > 0 and p.z() < HH; }, 178.0f);
        zplane(HH,       [&](const Eigen::Vector3f& p) { return p.y() > -T and p.y() < 0 and p.x() > HX0 and p.x() < HX0 + HW; }, 160.0f);
    }
    // room_2: other walls, floor, ceiling
    yplane(4.0f,  [&](const Eigen::Vector3f& p) { return std::abs(p.x()) < 3.0f and p.z() > 0 and p.z() < CEIL; }, 196.0f);
    xplane(-3.0f, [&](const Eigen::Vector3f& p) { return p.y() > 0 and p.y() < 4 and p.z() > 0 and p.z() < CEIL; }, 205.0f);
    xplane( 3.0f, [&](const Eigen::Vector3f& p) { return p.y() > 0 and p.y() < 4 and p.z() > 0 and p.z() < CEIL; }, 205.0f);
    // room_1 (beyond): back wall, side walls
    yplane(-4.0f, [&](const Eigen::Vector3f& p) { return std::abs(p.x()) < 5.0f and p.z() > 0 and p.z() < CEIL; }, 120.0f);
    xplane(-5.0f, [&](const Eigen::Vector3f& p) { return p.y() < -T and p.y() > -4 and p.z() > 0 and p.z() < CEIL; }, 128.0f);
    xplane( 5.0f, [&](const Eigen::Vector3f& p) { return p.y() < -T and p.y() > -4 and p.z() > 0 and p.z() < CEIL; }, 128.0f);
    // floor / ceiling everywhere
    zplane(0.0f, [&](const Eigen::Vector3f& p) { return p.y() > 0 ? (std::abs(p.x()) < 3 and p.y() < 4) : (std::abs(p.x()) < 5 and p.y() > -4); }, 80.0f);
    zplane(CEIL, [&](const Eigen::Vector3f& p) { return std::abs(p.x()) < 5 and std::abs(p.y()) < 4.5f; }, 235.0f);
    // the casing: three boards, axis-aligned boxes (slab test), lit by face
    const auto box = [&](Eigen::Vector3f lo, Eigen::Vector3f hi, float shade)
    {
        float t0 = -1e9f, t1 = 1e9f; int axis = -1;
        for (int i = 0; i < 3; ++i)
        {
            if (std::abs(d[i]) < 1e-9f) { if (o[i] < lo[i] or o[i] > hi[i]) return; continue; }
            float a = (lo[i] - o[i]) / d[i], b = (hi[i] - o[i]) / d[i];
            if (a > b) std::swap(a, b);
            if (a > t0) { t0 = a; axis = i; }
            t1 = std::min(t1, b);
        }
        if (t0 <= t1 and t0 > 1e-4f) { const auto p = at(t0); take(t0, shade - 12.0f * axis + tex(p.x() + p.y(), p.z())); }
    };
    if (sc.hole and sc.casing)
    {
        const float y0 = -T - CASE_P, y1 = CASE_P;
        box({HX0, y0, 0.0f},                 {HX0 + CASE_W, y1, HH},        150.0f);
        box({HX0 + HW - CASE_W, y0, 0.0f},   {HX0 + HW, y1, HH},            150.0f);
        box({HX0, y0, HH - CASE_W},          {HX0 + HW, y1, HH},            150.0f);
    }
    // leaf (surface-mounted on the room_1 side when there is a casing; in the reveal otherwise)
    const float leaf_y = sc.casing ? -T - CASE_P - 0.025f : -T + 0.02f;
    if (sc.hole and sc.leaf == 1)
        yplane(leaf_y, [&](const Eigen::Vector3f& p) { return p.x() > HX0 and p.x() < HX0 + HW and p.z() > 0 and p.z() < HH; }, 110.0f);
    if (sc.hole and sc.leaf == 2)
    {   // hinge at the far (+x) jamb on the room_1 side; at phi = 0 it would span toward −x; opening OUT ⇒ −y
        const Eigen::Vector2f hinge(HX0 + HW, leaf_y), dir(-std::cos(sc.phi), -std::sin(sc.phi));
        const Eigen::Vector2f nrm(-dir.y(), dir.x());
        const float den = nrm.dot(Eigen::Vector2f(d.x(), d.y()));
        if (std::abs(den) > 1e-7f)
        {
            const float t = nrm.dot(hinge - Eigen::Vector2f(o.x(), o.y())) / den;
            const auto p = at(t);
            const float a = (Eigen::Vector2f(p.x(), p.y()) - hinge).dot(dir);
            if (a > 0.0f and a < HW - 0.02f and p.z() > 0.0f and p.z() < HH) take(t, 110.0f + tex(a, p.z()));
        }
    }
    return h;
}

struct Frame { cv::Mat gray, depth; };

inline Frame render(const Scene& sc, const CamPose& c)
{
    Frame f{cv::Mat(720, 1280, CV_8UC1), cv::Mat(720, 1280, CV_32F)};
    const float cs = std::cos(c.yaw), sn = std::sin(c.yaw);
    const Eigen::Vector3f o(c.x, c.y, c.z), right(cs, sn, 0.0f), fwd(-sn, cs, 0.0f), up(0, 0, 1);
    for (int r = 0; r < 720; ++r)
        for (int col = 0; col < 1280; ++col)
        {
            const float X = (col - 640.0f) / 448.0f, Z = (360.0f - r) / 448.0f;
            const Hit h = trace(sc, o, fwd + X * right + Z * up);
            f.gray.at<std::uint8_t>(r, col) = static_cast<std::uint8_t>(std::clamp(h.shade, 0.0f, 255.0f));
            f.depth.at<float>(r, col) = (h.t < 1e8f) ? h.t : 0.0f;
        }
    return f;
}

// The room polygon room_concept would publish for room_2: the ROOM-SIDE faces.
inline std::vector<Eigen::Vector2f> room_poly() { return {{-3.0f, 0.0f}, {3.0f, 0.0f}, {3.0f, 4.0f}, {-3.0f, 4.0f}}; }

// The agent's compute_frame_probe, minus the DSR: visibility =
// frustum (build_frame_probe) × walls_block(room polygon, own-wall skip 0.30) × own leaf (an open-OUT leaf is
// behind the wall and never in front of a frame point; nullptr here).
inline FrameProbe probe(const Aperture& belief, const CamPose& c, const rc::door::RoomBand& band,
                        float casing_m = 0.06f)
{
    const auto poly = room_poly();
    const Aperture& ap = belief;
    const Eigen::Vector2f O(c.x, c.y);
    const PointVisibility vis = [&](const Eigen::Vector3f& P)
    { return rc::occlusion::walls_block(O, {P.x(), P.y()}, poly, 0.30f) ? 0.0f : 1.0f; };
    return build_frame_probe(ap, make_projector(c), vis, nullptr, 1280, 720, casing_m, band.sigma_m);
}

inline FrameScore score(const Frame& f, const FrameProbe& pr)
{
    const auto prep = rc::edges::prepare_frame(f.gray);
    return score_frame(pr, &prep, &f.depth);
}

// A camera at (x, y) looking at the point (tx, ty) (yaw in the projector's convention: forward = (−sin, cos)).
inline CamPose look_at(float x, float y, float tx, float ty, float extra_yaw = 0.0f)
{
    return {x, y, 1.1f, std::atan2(-(tx - x), ty - y) + extra_yaw};
}

// The aperture as the AGENT believed it in try34: 1.0 m wide, its line 6 cm IN FRONT of the room-side face and
// 4 cm off along the wall (wall_u = +x ⇒ across_u = +y, into the room).
inline Aperture belief_open() { return make_ap({HX0 + 0.04f, 0.06f}, {1.0f, 0.0f}, HW, HH); }
}   // namespace syn

static void print_syn(const std::string& what, const FrameScore& s)
{
    std::println("    {:<50} exc={:+.2f} (t={:.1f} c={:.1f}) dep={:+.2f}/{} vis={:.2f} verdict={:+.2f}",
                 what, s.excess, s.s_true, s.s_control, s.depth_verdict, s.n_depth, s.vis_frac, s.verdict());
}

// (g) — returns the mean verdict on the open-out door (feeds (e) when the real frames are gone).
static float synthetic_frames(const rc::door::RoomBand& band)
{
    std::println("(g) SYNTHETIC room_2: 0.11 m wall + reveals, leaf ~100° OUT into room_1 (the try34 door), belief 6 cm in front of the face");
    const syn::Scene open{true, 2, 1.75f}, shut{true, 1, 0.0f}, blank{false, 0, 0.0f};
    const Aperture bel = syn::belief_open();
    const float cx = syn::HX0 + 0.5f * syn::HW;
    struct View { const char* name; CamPose c; };
    const std::vector<View> views = {
        {"frontal 2.0 m",                 syn::look_at(cx, 2.0f, cx, 0.0f)},
        {"frontal 3.6 m",                 syn::look_at(cx, 3.6f, cx, 0.0f)},
        {"oblique from -x (hinge-opposite)", syn::look_at(-1.9f, 1.4f, cx, 0.0f)},
        {"oblique from +x (hinge side)",  syn::look_at(1.5f, 1.3f, cx, 0.0f)},
        {"near, door at the image edge",  syn::look_at(cx, 1.6f, cx, 0.0f, 0.75f)},
    };
    float mean_open = 0.0f, min_open = 1e9f, min_dep = 1e9f;
    for (const auto& v : views)
    {
        const auto f = syn::render(open, v.c);
        const auto s = syn::score(f, syn::probe(bel, v.c, band));
        print_syn(std::string("open-out, ") + v.name, s);
        mean_open += s.verdict() / views.size();
        min_open = std::min(min_open, s.verdict());
        if (s.n_depth > 0) min_dep = std::min(min_dep, s.depth_verdict);
    }
    // THE DEPTH-HALF DIAGNOSIS: the same frames read as a knife-edge opening (FrameCasingM = 0) vs with the casing.
    std::println("    depth half, 3-6 px reads (as live) vs casing+registration window (live try34: dep -0.45 / -0.32 from inside room_2):");
    float worst_knife = 1e9f;
    for (const auto& v : views)
    {
        const auto f = syn::render(open, v.c);
        rc::door::RoomBand knife = band; knife.sigma_m = 0.0f;   // bare 3-6 px reads, as before
        const auto k0 = syn::score(f, syn::probe(bel, v.c, knife, 0.0f));
        const auto k1 = syn::score(f, syn::probe(bel, v.c, band, 0.06f));
        std::println("      {:<34} dep {:+.2f} → {:+.2f}", v.name, k0.depth_verdict, k1.depth_verdict);
        worst_knife = std::min(worst_knife, k0.depth_verdict);
    }
    check(worst_knife < -0.2f, std::format("reproduces the live NEGATIVE depth read when the casing is ignored ({:+.2f})", worst_knife));
    check(mean_open > 0.5f, std::format("open-out door from inside: mean frame verdict {:+.2f} > +0.5", mean_open));
    check(min_open > 0.0f, std::format("open-out door: no view refutes it (min verdict {:+.2f} > 0)", min_open));
    check(min_dep > -0.1f, std::format("open-out door: the depth half never reads the doorway as NEARER (min {:+.2f})", min_dep));

    float mean_shut = 0.0f;
    for (const auto& v : views)
    {
        const auto s = syn::score(syn::render(shut, v.c), syn::probe(bel, v.c, band));
        print_syn(std::string("shut, ") + v.name, s);
        mean_shut += s.verdict() / views.size();
    }
    check(mean_shut > 0.5f, std::format("shut door: mean frame verdict {:+.2f} > +0.5", mean_shut));

    // Phantoms: the same aperture slid along the wall onto blank wall (both scenes: with and without the real
    // doorway elsewhere), and one floating mid-room parallel to the wall.
    std::vector<float> ph;
    for (const float off : {1.4f, 1.9f, -1.9f})
        for (const auto& v : views)
        {
            Aperture a = bel; a.wall_O.x() += off;
            for (const auto* sc : {&open, &blank})
            {
                const auto pr = syn::probe(a, v.c, band);
                if (pr.vis_frac() < 0.05f) continue;
                const auto s = syn::score(syn::render(*sc, v.c), pr);
                if (s.n_samples() == 0) continue;
                ph.push_back(s.verdict());
            }
        }
    float pm = 0.0f, pmax = -1e9f;
    for (const float v : ph) { pm += v / ph.size(); pmax = std::max(pmax, v); }
    std::println("    phantom placements on the blank wall: {}  mean {:+.3f}  max {:+.3f}", ph.size(), pm, pmax);
    check(not ph.empty() and pm <= 0.05f, std::format("blank-wall phantoms: mean frame verdict {:+.3f} ≤ ~0", pm));
    check(pmax < 0.5f * mean_open, std::format("worst blank-wall phantom {:+.2f} well below the real door {:+.2f}", pmax, mean_open));

    {
        const Aperture mid = make_ap({-0.6f, 2.0f}, {1.0f, 0.0f}, 1.0f, 2.05f);
        const auto c = syn::look_at(-0.1f, 3.7f, -0.1f, 2.0f);
        const auto s = syn::score(syn::render(open, c), syn::probe(mid, c, band));
        print_syn("phantom mid-room (y = 2.0)", s);
        check(s.verdict() < 0.5f * mean_open, std::format("mid-room phantom frame verdict {:+.2f} well below the real door", s.verdict()));
    }
    return mean_open;
}

// (h) The replay. Mirrors update_existence_beliefs' arithmetic for one door, per evidence cycle:
//   wall-band prior (static term, replaced) → leaf silhouette absence (worst case: the detector NEVER fires on
//   it, p_detect = obliquity × visible fraction with envelope/resolvability/centrality all 1) weighted by
//   1 − P(beyond) → the doorway frame (one tanh) → decide_removal (look = p_detect ∨ (1 − π)).
struct ReplayOut { float L_end = 0.0f, L_min = 1e9f; int removed_at = -1; float vis_last = 0.0f; };

static ReplayOut replay(const syn::Scene& sc, const Aperture& bel, float phi, bool leaf_open_out,
                        const std::vector<CamPose>& path, const rc::door::RoomBand& band, float L0,
                        bool use_beyond = true, bool use_frame = true, bool use_sil = true, bool verbose = false)
{
    rc::exist::SensorModel sm;
    rc::exist::RemovalPolicy pol;   // door_concept: MaxLogodds 4, RemovalProb 0.12, RemoveFrames 15
    rc::exist::ExistenceBelief B(L0, pol.logodds_max);
    rc::exist::RemovalDebounce deb;
    const auto poly = syn::room_poly();
    float prior_term = 0.0f;
    ReplayOut out;
    LeafState ls; ls.phi = phi; ls.hinge = HingeSide::Far; ls.swing = leaf_open_out ? -1.0f : 1.0f;   // −across = out of room_2
    const LeafPose L = leaf_pose(bel, ls);
    for (std::size_t k = 0; k < path.size(); ++k)
    {
        const CamPose& c = path[k];
        // prior
        const float delta = rc::door::signed_boundary_distance(poly, bel.centre_xy());
        const float term = rc::door::band_log_prior(delta, band);
        B.set(B.logodds() - prior_term + term); prior_term = term;
        const float prior_look = 1.0f - std::exp(term);
        // leaf silhouette (absence only)
        const auto proj = make_projector(c);
        const Eigen::Vector2f O(c.x, c.y);
        const Eigen::Vector2f n_out = rc::door::outward_normal(poly, bel.centre_xy(), bel.across_u(), band);
        const float r0 = rc::door::face_offset_from_aperture(delta);
        int n_total = 0, n_det = 0; float det_w = 0.0f, e_free = 0.0f;
        for (int ix = 0; ix < 14; ++ix)
            for (int iz = 0; iz < 30; ++iz)
            {
                ++n_total;
                const Eigen::Vector3f P = leaf_point(L, (-1.0f + 2.0f * (ix + 0.5f) / 14) * L.half_w, 0.0f,
                                                     bel.floor_z + bel.h * (iz + 0.5f) / 30);
                const auto v = proj(P.cast<double>());
                if (not v or v->px.x < 0 or v->px.y < 0 or v->px.x >= 1280 or v->px.y >= 720) continue;
                if (rc::occlusion::walls_block(O, {P.x(), P.y()}, poly, 0.30f)) continue;
                const float a = (Eigen::Vector2f(P.x(), P.y()) - bel.wall_O).dot(n_out);
                const float w = use_beyond ? 1.0f - rc::door::p_beyond_far_face(a, r0, band.sigma_m) : 1.0f;
                if (not (w > 0.0f)) continue;
                ++n_det; det_w += w; e_free += w;
            }
        const Eigen::Vector2f ray = (L.centre_xy - O).normalized();
        const float oblq = std::abs(ray.dot(L.ey));
        const float p_detect = n_total > 0 ? oblq * det_w / n_total : 0.0f;
        // frame
        FrameScore fs{};
        if (use_frame)
        {
            const auto pr = syn::probe(bel, c, band);
            if (pr.vis_frac() > 0.0f)
                fs = syn::score(syn::render(sc, c), pr);
        }
        out.vis_last = fs.vis_frac;
        const bool probed = det_w > 0.0f and use_sil;
        if (fs.n_samples() > 0)
            B.integrate(rc::exist::contour_evidence(fs.verdict(), fs.n_samples(), sm), 1.0f);
        float look = prior_look;
        if (probed)
        {
            rc::exist::Evidence ev;
            ev.n_reached = n_det;
            ev.log_odds_delta = p_detect * rc::exist::mask_evidence(0.0f, e_free, n_det, sm).log_odds_delta;
            B.integrate(ev, p_detect);
            look = 1.0f - (1.0f - p_detect) * (1.0f - prior_look);
        }
        out.L_min = std::min(out.L_min, B.logodds());
        if (verbose and k % 20 == 0)
            std::println("      t={:5.1f}s L={:+.2f} | sil det_w={:.0f}/{} free={:.0f} pdet={:.2f} | frame vis={:.2f} exc={:+.2f} dep={:+.2f} v={:+.2f} | pi={:.2f}",
                         0.5f * k, B.logodds(), det_w, n_total, e_free, p_detect, fs.vis_frac, fs.excess,
                         fs.depth_verdict, fs.verdict(), std::exp(term));
        if (rc::exist::decide_removal(B, deb, pol, look).remove) { out.removed_at = static_cast<int>(k); break; }
    }
    out.L_end = B.logodds();
    return out;
}

static void replays(const rc::door::RoomBand& band)
{
    std::println("(h) REPLAY, 2 Hz evidence for 100 s: the robot backs away from the carried open-out door and turns (vis falls)");
    const float cx = syn::HX0 + 0.5f * syn::HW;
    std::vector<CamPose> path;
    for (int k = 0; k < 200; ++k)
    {
        const float u = k / 199.0f;
        const float x = cx + 0.6f * std::sin(3.0f * u), y = 1.2f + 2.5f * u;
        // facing the door for the first half, then turning steadily away (up to ~110° by the end)
        const float turn = (u < 0.5f) ? 0.0f : 1.9f * (u - 0.5f) * 2.0f * (k % 2 ? 1.0f : 0.97f);
        path.push_back(syn::look_at(x, y, cx, 0.0f, turn));
    }
    const syn::Scene open{true, 2, 1.75f}, blank{false, 0, 0.0f};
    const Aperture bel = syn::belief_open();

    const auto real = replay(open, bel, 1.75f, true, path, band, 2.0f, true, true, true, true);
    std::println("    carried open-out door: L_end {:+.2f}, L_min {:+.2f}, removed at {} (vis at the end {:.2f})",
                 real.L_end, real.L_min, real.removed_at, real.vis_last);
    check(real.removed_at < 0 and real.L_min > 1.0f,
          std::format("the carried open-out door holds L high for 100 s (min {:+.2f} > +1, never removed)", real.L_min));
    const auto no_beyond = replay(open, bel, 1.75f, true, path, band, 2.0f, false, true, true);
    std::println("    same, WITHOUT the leaf-beyond weight: L_end {:+.2f}, L_min {:+.2f}, removed at {}",
                 no_beyond.L_end, no_beyond.L_min, no_beyond.removed_at);
    const auto sil_only = replay(open, bel, 1.75f, true, path, band, 2.0f, true, false, true);
    std::println("    leaf-beyond weight, NO frame channel:   L_end {:+.2f}, L_min {:+.2f}, removed at {}",
                 sil_only.L_end, sil_only.L_min, sil_only.removed_at);

    // A phantom on the blank wall, its (shut) leaf facing the camera: the frame must not confirm it, and the
    // silhouette (no mask ever) removes it.
    Aperture ph = bel; ph.wall_O.x() += 1.9f;
    std::vector<CamPose> ph_path;
    for (int k = 0; k < 200; ++k) { const float u = k / 199.0f; ph_path.push_back(syn::look_at(ph.centre_xy().x() + 0.8f * std::sin(4 * u), 1.6f + 1.5f * u, ph.centre_xy().x(), 0.0f)); }
    const auto ph_frame = replay(open, ph, 0.0f, true, ph_path, band, 1.0f, true, true, false);
    std::println("    blank-wall phantom, frame channel ONLY: L {:+.2f} → {:+.2f}", 1.0f, ph_frame.L_end);
    check(ph_frame.L_end < 1.0f + 1.0f, std::format("the frame does not confirm a blank-wall phantom (L {:+.2f} from +1.00)", ph_frame.L_end));
    const auto ph_all = replay(open, ph, 0.0f, true, ph_path, band, 1.0f);
    std::println("    blank-wall phantom, all channels: removed at cycle {} (L {:+.2f})", ph_all.removed_at, ph_all.L_end);
    check(ph_all.removed_at >= 0, "a blank-wall phantom is still removed (by the leaf silhouette)");

    // A phantom born MID-ROOM: off the band by ~2 m. Removed by the band prior, even never looked at.
    const Aperture mid = make_ap({-0.6f, 2.0f}, {1.0f, 0.0f}, 1.0f, 2.05f);
    std::vector<CamPose> away(200, syn::look_at(-0.1f, 3.7f, -0.1f, 8.0f));   // looking the other way
    const auto mid_r = replay(open, mid, 0.0f, true, away, band, 4.0f);
    std::println("    mid-room phantom, never looked at, from L=+4: removed at cycle {}", mid_r.removed_at);
    check(mid_r.removed_at >= 0 and mid_r.removed_at < 40, "a phantom born mid-room is removed by the band prior");
    // …and the old per-frame prior's failure, on the carried door: aperture 6 cm INSIDE the polygon is π≈0.84
    // under a per-frame form, which would accumulate; the static term costs it log π once.
    const float d_try34 = rc::door::signed_boundary_distance(syn::room_poly(), bel.centre_xy());
    std::println("    try34 carried aperture: δ = {:+.3f} m ⇒ static log-prior {:+.3f} (applied once, not per frame)",
                 d_try34, rc::door::band_log_prior(d_try34, band));
}

// (f) The wall band on try34's numbers: room_2's polygon y ∈ [0.03, 4.16], the carried aperture at y = 0.091.
static void band_checks(const rc::door::RoomBand& band)
{
    std::println("(f) WALL BAND (T = {:.2f} m, σ = {:.2f} m)", band.wall_thickness_m, band.sigma_m);
    const std::vector<Eigen::Vector2f> poly = {{-2.90f, 0.03f}, {3.17f, 0.03f}, {3.17f, 4.16f}, {-2.90f, 4.16f}};
    const auto lp = [&](float x, float y) { return rc::door::band_log_prior(rc::door::signed_boundary_distance(poly, {x, y}), band); };
    const float carried = lp(-0.11f, 0.091f), centre = lp(-0.11f, -0.025f), far_face = lp(-0.11f, -0.08f);
    const float out40 = lp(-0.11f, -0.40f), out80 = lp(-0.11f, -0.80f), mid = lp(0.6f, 2.2f);
    std::println("    log π: carried(6 cm inside) {:+.2f}  wall centre {:+.2f}  far face {:+.2f}  0.43 m out {:+.2f}  "
                 "0.83 m out {:+.2f}  mid-room {:+.1f}", carried, centre, far_face, out40, out80, mid);
    check(centre == 0.0f and far_face == 0.0f, "an aperture anywhere IN the wall pays exactly nothing");
    check(carried > -0.25f, "try34's carried aperture (6 cm in front of the face) pays < 0.25 nats, ONCE");
    check(out80 < -8.0f and mid < -8.0f, "an aperture far outside, or well inside the room, is pushed to the floor");
    check(rc::door::p_beyond_far_face(0.0f, 0.1f, 0.1f) == 0.0f and rc::door::p_beyond_far_face(0.8f, 0.1f, 0.1f) > 0.99f,
          "a leaf sample ON the aperture line is never 'beyond'; one 0.8 m out is");
}

int main()
{
    std::setlocale(LC_ALL, "");   // the agent runs under Qt's locale; the harness must too (CLAUDE.md)
    const auto poses = load_poses();

    // ── geometry fitted to the images (checked by overlay in two views of each door) ──────────────────────
    // door_3 in room_2's frame: the wall at y = 0.09, hole x ∈ [-0.32, 0.52], h 2.05 (inner casing edges).
    const Aperture door3 = make_ap({0.52f, 0.09f}, {-1.0f, 0.0f}, 0.84f, 2.05f);
    // What the agent BELIEVED for door_3 in try32 (mirrored from door_1: w = 1.00, centre x = 0.33): the
    // live channel is scored on this, so it must survive a ~10-15 cm belief error.
    const Aperture door3_belief = make_ap({0.83f, 0.09f}, {-1.0f, 0.0f}, 1.00f, 2.11f);
    // door_1 in room_1's frame (shut, green leaf): wall y = 4.61, hole x ∈ [-0.99, 0.00], h 2.10.
    const Aperture door1 = make_ap({-0.99f, 4.61f}, {1.0f, 0.0f}, 0.99f, 2.10f);
    const Aperture door1_belief = make_ap({-1.115f, 4.61f}, {1.0f, 0.0f}, 1.05f, 2.10f);

    const std::vector<std::string> open_out = {"1791834211973", "1791834212310", "1791834212570", "1791834212911",
                                               "1791834213310", "1791834213844", "1791834215054"};
    const std::vector<std::string> shut = {"1791834129174", "1791834129303", "1791834129434", "1791834129593",
                                           "1791834129715", "1791834129875"};

    bool have_real = not poses.empty();
    for (const auto& f : open_out) have_real = have_real and frame_exists(f);
    for (const auto& f : shut)     have_real = have_real and frame_exists(f);
    float mean_a = 0.0f, mean_ab = 0.0f, mean_b = 0.0f, mean_bb = 0.0f;
    if (not have_real)
        std::println("(a)-(c) SKIPPED — try32's ZED frames are no longer under {} (deleted 2026-10-12); "
                     "the synthetic scene (g)/(h) covers the same cases", kFrames);
    else
    {
    std::println("(a) SHUT door_1 from room_1 — frame on the green leaf/wall boundary");
    for (const auto& f : shut)
    {
        const auto s = score_on(f, poses, door1, nullptr, "a");
        const auto sb = score_on(f, poses, door1_belief);
        print_score(f + " fitted", s); print_score(f + " agent belief", sb);
        mean_a += s.verdict() / shut.size(); mean_ab += sb.verdict() / shut.size();
    }
    check(mean_a > 0.5f, std::format("shut door, fitted aperture: mean frame verdict {:+.2f} > +0.5", mean_a));
    check(mean_ab > 0.3f, std::format("shut door, agent's belief: mean frame verdict {:+.2f} > +0.3", mean_ab));

    std::println("(b) OPEN door_3 seen from INSIDE room_2, leaf swung OUT (the try32 death)");
    float min_b = 1e9f;
    for (const auto& f : open_out)
    {
        const auto s = score_on(f, poses, door3, nullptr, "b");
        const auto sb = score_on(f, poses, door3_belief);
        print_score(f + " fitted", s); print_score(f + " agent belief", sb);
        mean_b += s.verdict() / open_out.size(); mean_bb += sb.verdict() / open_out.size();
        min_b = std::min(min_b, sb.verdict());
    }
    check(mean_b > 0.5f, std::format("open-out door, fitted aperture: mean frame verdict {:+.2f} > +0.5", mean_b));
    check(mean_bb > 0.3f, std::format("open-out door, agent's belief: mean frame verdict {:+.2f} > +0.3", mean_bb));

    std::println("(c) PHANTOM apertures slid along the same walls (|offset| ≥ w + 0.3 m from the real hole)");
    // A sweep, not one hand-picked spot: every placement along the wall that is clear of the real frame.
    std::vector<float> phantom;
    const auto sweep = [&](const Aperture& real, const std::vector<std::string>& frames, const char* name)
    {
        std::print("    {} offsets→verdict:", name);
        for (float off = -3.0f; off <= 3.01f; off += 0.25f)
        {
            if (std::abs(off) < real.w + 0.3f) continue;
            Aperture a = real; a.s += off;
            float acc = 0.0f; int n = 0;
            for (const auto& f : frames)
            {
                float vis = 0.0f;
                const auto s = score_on(f, poses, a, &vis);
                if (s.n_edge == 0 or vis < 1e-3f) continue;
                acc += s.verdict(); ++n;
            }
            if (n == 0) continue;
            std::print(" {:+.2f}:{:+.2f}", off, acc / n);
            phantom.push_back(acc / n);
        }
        std::println("");
    };
    sweep(door3, open_out, "room_2 wall");
    sweep(door1, shut, "room_1 wall");
    float pm = 0.0f, pmax = -1e9f;
    for (const float v : phantom) { pm += v / phantom.size(); pmax = std::max(pmax, v); }
    std::println("    phantom placements {}: mean {:+.3f}, max {:+.3f}", phantom.size(), pm, pmax);
    check(not phantom.empty() and pm <= 0.05f, std::format("phantom mean verdict {:+.3f} ≤ ~0 (not confirmed)", pm));
    check(pmax < 0.5f * std::min(mean_b, mean_a),
          std::format("worst phantom {:+.2f} well below the real doors ({:+.2f}, {:+.2f})", pmax, mean_a, mean_b));
    }   // have_real

    // ── (d) depth half, synthetic: camera at the origin looking +y, wall at y = 3, hole x∈[-.45,.45] ─────────
    std::println("(d) depth half (synthetic wall at 3 m, hole 0.9 x 2.0 m)");
    {
        const Aperture ap = make_ap({-0.45f, 3.0f}, {1.0f, 0.0f}, 0.9f, 2.0f);
        // Ray-cast render of the forward depth. hole_fill < 0 ⇒ see through the hole to 8 m; else a leaf
        // plane at that y (flush = 3.0). leaf_near: the leaf swung 90° TOWARD the camera on the x = -0.45
        // hinge, i.e. a panel in the plane x = -0.45 spanning y ∈ [2.1, 3.0], z ∈ [0, 2].
        const auto render = [&](const CamPose& c, bool hole, float hole_fill, bool leaf_near)
        {
            cv::Mat d(720, 1280, CV_32F);
            const double cs = std::cos(c.yaw), sn = std::sin(c.yaw);
            for (int r = 0; r < 720; ++r)
                for (int col = 0; col < 1280; ++col)
                {
                    const double X = (col - 640.0) / 448.0, Z = (360.0 - r) / 448.0;
                    // world direction with unit FORWARD component, so the ray parameter IS the forward depth
                    const double dx = X * cs - sn, dy = X * sn + cs;
                    double best = 8.0;
                    if (dy > 1e-6)
                    {
                        const double t = (3.0 - c.y) / dy;
                        const double xw = c.x + t * dx, zw = c.z + t * Z;
                        const bool in_hole = hole and xw > -0.45 and xw < 0.45 and zw > 0.0 and zw < 2.0;
                        if (not in_hole) best = t;
                        else if (hole_fill > 0.0f) best = (hole_fill - c.y) / dy;
                        else best = std::min(8.0, (11.0 - c.y) / dy);
                    }
                    if (leaf_near and std::abs(dx) > 1e-6)
                    {
                        const double t = (-0.45 - c.x) / dx;
                        const double yw = c.y + t * dy, zw = c.z + t * Z;
                        if (t > 0.0 and yw > 2.1 and yw < 3.0 and zw > 0.0 and zw < 2.0) best = std::min(best, t);
                    }
                    d.at<float>(r, col) = static_cast<float>(best);
                }
            return d;
        };
        LeafState toward_st; toward_st.phi = 1.5708f; toward_st.swing = -1.0f; toward_st.hinge = HingeSide::Near;  // −across = camera side
        const LeafPose leaf_toward = leaf_pose(ap, toward_st);
        const auto run = [&](const char* what, const CamPose& c, const cv::Mat& d, bool model_leaf)
        {
            const Eigen::Vector3f O(c.x, c.y, c.z);
            const PointVisibility clear = [&](const Eigen::Vector3f& P)
            { return leaf_blocks_sightline(leaf_toward, O, P) ? 0.0f : 1.0f; };
            const auto probe = build_frame_probe(ap, make_projector(c), model_leaf ? clear : nullptr,
                                                 model_leaf ? clear : nullptr, 1280, 720);
            const auto s = score_frame(probe, nullptr, &d);
            std::println("    {:<56} depth={:+.2f} n={} vis={:.2f}", what, s.depth_verdict, s.n_depth, s.vis_frac);
            return s;
        };
        const CamPose front{0.0f, 0.0f, 0.95f, 0.0f};
        const CamPose side{-1.6f, 0.4f, 0.95f, static_cast<float>(-std::atan2(1.6, 2.6))};   // from the HINGE side, facing the hole
        const auto open = run("open hole, frontal", front, render(front, true, -1.0f, false), false);
        const auto recess = run("shut leaf recessed 5 cm in the reveal", front, render(front, true, 3.05f, false), false);
        const auto flush = run("shut leaf flush with the wall", front, render(front, true, 3.0f, false), false);
        const auto blank = run("phantom: blank wall", front, render(front, false, -1.0f, false), false);
        const auto side_open = run("open hole, oblique from the hinge side", side, render(side, true, -1.0f, false), false);
        const auto toward_raw = run("oblique, leaf swung TOWARD camera, leaf NOT modelled", side, render(side, true, -1.0f, true), false);
        const auto toward = run("oblique, leaf swung TOWARD camera, leaf modelled", side, render(side, true, -1.0f, true), true);
        check(open.depth_verdict > 0.8f and side_open.depth_verdict > 0.8f, "open hole confirms (depth verdict > 0.8)");
        check(recess.depth_verdict > 0.0f, "recessed shut leaf leans positive");
        check(std::abs(flush.depth_verdict) < 0.1f, "flush shut leaf abstains (|v| < 0.1)");
        check(std::abs(blank.depth_verdict) < 0.1f, "blank-wall phantom abstains (|v| < 0.1)");
        check(toward.depth_verdict > toward_raw.depth_verdict + 0.3f,
              std::format("modelling the own leaf removes its false refutation ({:+.2f} → {:+.2f})",
                          toward_raw.depth_verdict, toward.depth_verdict));
    }

    const rc::door::RoomBand band{0.20f, 0.10f};   // etc/config.toml Existence.WallThicknessM / BoundarySigmaM
    band_checks(band);
    const float syn_open = synthetic_frames(band);
    if (not have_real) mean_bb = syn_open;

    // ── (e) existence integration, try32 numbers ────────────────────────────────────────────────────────────
    std::println("(e) existence over 60 evidence cycles from L = +3.6 (try32 door_3: occ 0, free 90/90, pdet 0.07)");
    {
        rc::exist::SensorModel sm;   // door_concept defaults (pd 0.85, pc 0.05) — see etc/config.toml
        const float pdet = 0.07f;
        const auto run = [&](bool leaf_probed, bool frame_on, float frame_verdict)
        {
            rc::exist::ExistenceBelief L(3.6f, 4.0f);
            for (int k = 0; k < 60; ++k)
            {
                if (frame_on)
                {
                    const auto ev = rc::exist::contour_evidence(frame_verdict, 60, sm);
                    L.integrate(ev, 1.0f);
                }
                if (leaf_probed)
                {
                    rc::exist::Evidence ev;
                    ev.n_reached = 90;
                    ev.log_odds_delta = pdet * rc::exist::mask_evidence(0.0f, 90.0f, 90, sm).log_odds_delta;
                    L.integrate(ev, pdet);
                }
            }
            return L.logodds();
        };
        const float L_before   = run(true, false, 0.0f);
        const float L_frame    = run(true, true, mean_bb);    // the frame verdict the AGENT's belief scores
        const float L_stopgap  = run(false, true, mean_bb);   // stopgap on: the leaf is not probed at all
        std::println("    leaf only (as live)            L60 = {:+.2f}", L_before);
        std::println("    + frame (belief verdict {:+.2f}) L60 = {:+.2f}", mean_bb, L_frame);
        std::println("    + frame, stopgap on            L60 = {:+.2f}", L_stopgap);
        check(L_before < -2.0f, "reproduces the live death without the frame channel");
        check(L_frame > 2.0f, "with the frame channel the open-out door stays alive WITHOUT the stopgap");
    }

    replays(band);

    std::println("\n{} failure(s)", failures);
    return failures == 0 ? 0 : 1;
}
