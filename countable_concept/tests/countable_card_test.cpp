// countable_card_test <agent_dir> [--json]: make_card(config as <agent_dir>/etc/config.toml sets it) carries the
// agent's real envelope, channels and geometry, and the card survives a JSON round trip. One binary, run once per class
// folder (nightstand_concept, bed_concept, toilet_concept). The class is the folder name; the expected geometry is the
// class MANIFEST's prior (common/concept_manifest/<class>.concept.toml [prior.footprint] mean_m / depth_mean_m,
// [prior.height] mean_m) — an independent source from the config keys the card is built from.
// (Adapted from bed_concept/tests/bed_card_test.cpp.)  Run under the agent locale:
//   LANG=es_ES.UTF-8 ./countable_card_test ../../bed_concept
#include "countable_card.h"
#include "perception_card/evaluator.h"
#include "perception_card/tests/toml_peek.h"   // via -I common
#include <charconv>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)
static bool near(float a, float b) { return std::fabs(a - b) < 1e-5f; }
using namespace rc;

namespace {
std::string trim(std::string_view s)
{
    const auto b = s.find_first_not_of(" \t\r");
    if (b == std::string_view::npos) return {};
    return std::string(s.substr(b, s.find_last_not_of(" \t\r") - b + 1));
}

// The quoted string of the first `key = "value"` line (comments skipped); "" when absent.
std::string toml_peek_str(const std::string& path, const std::string& key)
{
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line))
    {
        const std::string l = trim(line);
        if (l.empty() or l.front() == '#') continue;
        const auto eq = l.find('=');
        if (eq == std::string::npos or trim(std::string_view(l).substr(0, eq)) != key) continue;
        const auto a = l.find('"', eq), b = a == std::string::npos ? a : l.find('"', a + 1);
        if (b != std::string::npos) return l.substr(a + 1, b - a - 1);
    }
    return {};
}

// A number `key = v` inside block [block] of a manifest, through from_chars (locale-independent).
std::optional<float> manifest_peek(const std::string& path, const std::string& block, const std::string& key)
{
    std::ifstream f(path);
    std::string line, cur;
    while (std::getline(f, line))
    {
        const std::string l = trim(line);
        if (l.empty() or l.front() == '#') continue;
        if (l.front() == '[') { cur = trim(std::string_view(l).substr(1, l.find(']') - 1)); continue; }
        const auto eq = l.find('=');
        if (cur != block or eq == std::string::npos or trim(std::string_view(l).substr(0, eq)) != key) continue;
        const std::string v = trim(std::string_view(l).substr(eq + 1));
        float x = 0.0f;
        if (std::from_chars(v.data(), v.data() + v.size(), x).ec == std::errc{}) return x;
    }
    return std::nullopt;
}
}  // namespace

int main(int argc, char** argv)
{
    CHECK(std::setlocale(LC_ALL, "") != nullptr);
    if (argc < 2) { std::puts("usage: countable_card_test <agent_dir> [--json]"); return 2; }
    const std::filesystem::path dir = std::filesystem::path(argv[1]).lexically_normal();
    std::string folder = (dir.has_filename() ? dir : dir.parent_path()).filename().string();
    CHECK(folder.ends_with("_concept"));
    const std::string cls = folder.substr(0, folder.size() - std::string_view("_concept").size());
    const std::string toml = (dir / "etc/config.toml").string();
    const std::string manifest = (dir / ".." / "common/concept_manifest" / (cls + ".concept.toml")).lexically_normal().string();

    CountableConfig cfg;   // struct defaults, then what etc/config.toml sets on top (the loader's behaviour)
    cfg.concept_class = toml_peek_str(toml, "Class");
    cfg.detect_min_fill = toml_peek(toml, "DetectMinFill", cfg.detect_min_fill);
    cfg.detect_max_fill = toml_peek(toml, "DetectMaxFill", cfg.detect_max_fill);
    cfg.detect_soft = toml_peek(toml, "DetectSoft", cfg.detect_soft);
    cfg.existence_detection_prob = toml_peek(toml, "ExistenceDetectionProb", cfg.existence_detection_prob);
    cfg.ai2_prior_footprint_m = toml_peek(toml, "AI2PriorFootprintM", cfg.ai2_prior_footprint_m);
    cfg.ai2_prior_height_m = toml_peek(toml, "AI2PriorHeightM", cfg.ai2_prior_height_m);
    cfg.ai2_prior_depth_m = toml_peek(toml, "AI2PriorDepthM", cfg.ai2_prior_depth_m);

    // This class's own geometry, from its manifest (a square-plan class declares no depth_mean_m: depth = width).
    const auto mw = manifest_peek(manifest, "prior.footprint", "mean_m");
    const auto md = manifest_peek(manifest, "prior.footprint", "depth_mean_m");
    const auto mh = manifest_peek(manifest, "prior.height", "mean_m");
    CHECK(mw.has_value() and mh.has_value());
    const float ew = mw.value_or(-1.0f), ed = md.value_or(ew), eh = mh.value_or(-1.0f);

    // Expected, literally: the fleet-prior envelope 0.10 / 0.60 / 0.06 (never fitted on any of the three).
    const card::Card c = make_card(cfg);
    if (argc > 2 and std::string_view(argv[2]) == "--json") std::puts(card::to_json(c).c_str());
    CHECK(c.concept_ == cls and c.level == card::Level::Countable);
    CHECK(c.channels.size() == 3 and c.channels[0].sensor == "zed" and c.channels[0].role == card::Role::Identity);
    CHECK(c.channels[1].sensor == "helios" and c.channels[1].role == card::Role::Occupancy);
    CHECK(c.channels[2].sensor == "ricoh" and not c.channels[2].modelled);
    const auto& p = c.subtypes.at("");
    CHECK(near(p.env.min_fill, 0.10f) and near(p.env.max_fill, 0.60f) and near(p.env.soft, 0.06f));
    CHECK(near(p.q_return, 0.85f));
    CHECK(p.terms.size() == 2 and p.terms[0].name == "envelope" and p.terms[1].name == "central");
    CHECK(near(p.terms[0].params.at("axes"), 1.0f));
    CHECK(near(c.geometry.w, ew) and near(c.geometry.d, ed) and near(c.geometry.h, eh));
    CHECK(near(c.geometry.z0, 0.0f) and near(c.geometry.z1, eh));
    CHECK(c.calibration == "prior" and c.calibration_date.empty());
    // Level 1 = countable_projection.cpp's own-wall skip (rc::kCountableOwnWallSkipM) against LINE walls.
    CHECK(c.walls and near(c.walls->own_wall_skip_m, 0.70f) and near(c.walls->own_wall_skip_m, kCountableOwnWallSkipM));
    CHECK(c.walls and c.walls->own_wall_plane_m < 0.0f and near(c.walls->wall_thickness_m, 0.0f));
    { // I4 end to end with THIS card: back 0.35 m from the x = 6 wall, 3.15 m stand-off, ZED + helios. The same P with
      // that wall and with it 3 m further away, and nothing reads BehindWall. Then the wall line through the carcass
      // (the back 0.15 m beyond it): still the same P, because it is the object's own wall.
        card::World w;
        w.polygon = {{0, 0}, {6, 0}, {6, 4}, {0, 4}};
        rc::nbv::Sensor z;
        z.hfov_rad = 2.0f * std::atan(640.0f / 448.0f); z.vfov_rad = 2.0f * std::atan(360.0f / 448.0f); z.height_m = 0.945f;
        w.zed = z; w.zed_mount_z = 0.945f;
        card::LidarRig hel;
        hel.model.version = 1; hel.model.rings = 32; hel.model.fov_start_deg = -55.0f; hel.model.fov_end_deg = 15.0f;
        hel.model.azimuth_step_deg = 0.4f; hel.model.range_min_m = 0.1f; hel.model.range_max_m = 80.0f;
        hel.mount = Eigen::Isometry3f::Identity();
        hel.mount.linear() = Eigen::AngleAxisf(std::numbers::pi_v<float>, Eigen::Vector3f::UnitX()).toRotationMatrix();
        hel.mount.translation() = Eigen::Vector3f(0.0f, -0.155f, 1.075f);
        w.helios = hel;
        const float half = 0.5f * c.geometry.d;
        card::Hypothesis h; h.concept_ = cls; h.x = 6.0f - 0.35f - half; h.y = 2.0f;
        const card::Viewpoint vp{{h.x - 3.15f, 2.0f}, std::nullopt};
        card::World far = w; far.polygon = {{0, 0}, {9, 0}, {9, 4}, {0, 4}};
        card::World cut = w; cut.polygon = {{0, 0}, {h.x + half - 0.15f, 0}, {h.x + half - 0.15f, 4}, {0, 4}};
        const auto rw = card::evaluate(c, w, h, vp), rf = card::evaluate(c, far, h, vp), rc_ = card::evaluate(c, cut, h, vp);
        std::printf("%s I4: wall rec=%.4f notice=%.4f limit=%d | far rec=%.4f | cut rec=%.4f limit=%d\n", cls.c_str(),
                    rw.p_recognise, rw.p_notice, int(rw.limit), rf.p_recognise, rc_.p_recognise, int(rc_.limit));
        CHECK(std::fabs(rw.p_recognise - rf.p_recognise) < 1e-6f and std::fabs(rw.p_notice - rf.p_notice) < 1e-6f);
        CHECK(std::fabs(rc_.p_recognise - rf.p_recognise) < 1e-6f and std::fabs(rc_.p_notice - rf.p_notice) < 1e-6f);
        CHECK(rw.limit != card::Limit::BehindWall and rc_.limit != card::Limit::BehindWall);
        for (const auto& ch : rc_.channels) CHECK(ch.limit != card::Limit::BehindWall);
        CHECK(rw.p_notice >= rw.p_recognise and rw.p_recognise > 0.5f);
    }
    const auto back = card::card_from_json(card::to_json(c));
    CHECK(back.has_value() and card::to_json(*back) == card::to_json(c));
    std::printf(bad ? "countable_card_test %s: %d FAILED\n" : "countable_card_test %s: OK\n", cls.c_str(), bad);
    return bad != 0;
}
