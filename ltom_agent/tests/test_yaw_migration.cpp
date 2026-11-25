// Offline test of src/yaw_migration.h and tools/chair_yaw_migrate (shared-engine K2b fix round 1) on a SYNTHETIC repertory —
// never on real ltom data. Runs under the es_ES locale the agents use.
//   1. a legacy chair (every keyframe before the cut): keyframe and move yaws turned by π, thumbnail bins by kViewBins/2, marked;
//   2. a chair seen by both binaries: only the keyframes before the cut turned, its (mixed) thumbnails dropped, marked;
//   3. another class and an already-marked chair: untouched;
//   4. idempotent: a second pass changes nothing; the marker survives a save + load, and a third pass after it changes nothing;
//   5. the tool end to end on a temp file: the first run writes, the second reports "already" and leaves the file byte-identical.
#include "../src/repertory_io.h"
#include "../src/yaw_migration.h"

#include <clocale>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <print>
#include <sstream>
#include <unistd.h>

using namespace ltom;
namespace fs = std::filesystem;

static int failures = 0;
#define CHECK(c) do { if (not (c)) { std::println(stderr, "FAIL {}:{}: {}", __FILE__, __LINE__, #c); ++failures; } } while (0)
static bool near(double a, double b) { return std::abs(a - b) < 1e-12; }

static constexpr std::int64_t kCut = 1'800'000'000'000;   // the countable chair took over here (synthetic)
static Keyframe kf(std::int64_t t, double yaw) { Keyframe k; k.t_ms = t; k.room_key = 1; k.pose.x = 1.0; k.pose.y = 2.0; k.pose.yaw = yaw; return k; }
static Thumbnail th(float s) { return {s, {0xff, 0xd8, static_cast<std::uint8_t>(s)}}; }

static Repertory synthetic()
{
    Repertory r(3000);
    Entry a; a.uid = 101; a.cls = "chair"; a.name = "chair_1";
    a.keyframes = {kf(kCut - 5000, 0.3), kf(kCut - 1000, 3.0)};
    Move m; m.t_ms = kCut - 900; m.live_uid = 205; m.from = kf(kCut - 1000, 3.0); m.to = kf(kCut - 900, -1.0);
    a.moves = {m};
    a.thumbs = {{0, th(10)}, {3, th(20)}, {7, th(30)}};
    r.insert_loaded(a);
    Entry b; b.uid = 102; b.cls = "chair"; b.name = "chair_2";
    b.keyframes = {kf(kCut - 1000, 1.0), kf(kCut + 1000, -2.0)};
    b.thumbs = {{2, th(5)}};
    r.insert_loaded(b);
    Entry c; c.uid = 103; c.cls = "bed"; c.name = "bed_1";
    c.keyframes = {kf(kCut - 1000, 0.5)};
    c.thumbs = {{1, th(7)}};
    r.insert_loaded(c);
    Entry d; d.uid = 104; d.cls = "chair"; d.name = "chair_3"; d.yaw_convention = std::string(kYawBackPlusY);
    d.keyframes = {kf(kCut - 1000, 0.7)};
    d.thumbs = {{4, th(9)}};
    r.insert_loaded(d);
    return r;
}

int main()
{
    std::setlocale(LC_ALL, "es_ES.UTF-8");
    const double pi = std::numbers::pi;
    Repertory r = synthetic();
    const auto rep = migrate_yaw_convention(r, "chair", kCut);
    CHECK(rep.migrated == 2 and rep.already == 1 and rep.other_class == 1);
    CHECK(rep.keyframes_turned == 3 and rep.moves_turned == 1 and rep.thumbs_turned == 3 and rep.thumbs_dropped == 1);
    const auto& E = r.entries();
    // 1. the legacy chair
    const Entry& a = E.at(101);
    CHECK(near(a.keyframes[0].pose.yaw, 0.3 + pi - 2.0 * pi) and near(a.keyframes[1].pose.yaw, 3.0 + pi - 2.0 * pi));
    CHECK(a.keyframes[0].pose.x == 1.0 and a.keyframes[0].pose.y == 2.0);   // only the yaw
    CHECK(near(a.moves[0].from.pose.yaw, 3.0 - pi) and near(a.moves[0].to.pose.yaw, -1.0 + pi));
    CHECK(a.thumbs.size() == 3 and a.thumbs.contains(6) and a.thumbs.contains(9) and a.thumbs.contains(1));
    CHECK(a.thumbs.at(6).score == 10.f and a.thumbs.at(9).score == 20.f and a.thumbs.at(1).score == 30.f);
    CHECK(a.yaw_convention == kYawBackPlusY);
    // 2. the chair seen by both binaries
    const Entry& b = E.at(102);
    CHECK(near(b.keyframes[0].pose.yaw, 1.0 - pi) and b.keyframes[1].pose.yaw == -2.0);
    CHECK(b.thumbs.empty() and b.yaw_convention == kYawBackPlusY);
    // 3. untouched
    CHECK(E.at(103).keyframes[0].pose.yaw == 0.5 and E.at(103).thumbs.contains(1) and E.at(103).yaw_convention.empty());
    CHECK(E.at(104).keyframes[0].pose.yaw == 0.7 and E.at(104).thumbs.contains(4));
    // wrap: (−π, π]
    CHECK(near(turned_half(0.0), pi) and near(turned_half(pi), 0.0) and near(turned_half(-pi / 2), pi / 2));

    // 4. idempotent, in memory and through the store
    const auto before = r.entries();
    const auto rep2 = migrate_yaw_convention(r, "chair", kCut);
    CHECK(rep2.migrated == 0 and rep2.already == 3 and rep2.keyframes_turned == 0 and rep2.thumbs_turned == 0);
    CHECK(r.entries().at(101).keyframes[0].pose.yaw == before.at(101).keyframes[0].pose.yaw);
    const fs::path dir = fs::temp_directory_path() / ("ltom_yaw_mig_" + std::to_string(::getpid()));
    fs::remove_all(dir);
    fs::create_directories(dir);
    const fs::path file = dir / "ltom_repertory.json";
    CHECK(save_repertory(file, r));
    Repertory back(3000);
    CHECK(load_repertory(file, back) == LoadResult::loaded);
    CHECK(back.entries().at(101).yaw_convention == kYawBackPlusY and back.entries().at(103).yaw_convention.empty());
    const auto rep3 = migrate_yaw_convention(back, "chair", kCut);
    CHECK(rep3.migrated == 0 and rep3.already == 3);

    // 5. the tool, twice, on a fresh synthetic file
    {
        const fs::path f2 = dir / "fresh.json";
        CHECK(save_repertory(f2, synthetic()));
        const std::string cmd = std::string(TOOL_PATH) + " " + f2.string() + " " + std::to_string(kCut) + " > " + (dir / "out.txt").string();
        CHECK(std::system(cmd.c_str()) == 0);
        Repertory t(3000);
        CHECK(load_repertory(f2, t) == LoadResult::loaded);
        CHECK(t.entries().at(101).yaw_convention == kYawBackPlusY and near(t.entries().at(101).keyframes[0].pose.yaw, 0.3 - pi));
        const auto slurp = [](const fs::path& p) { std::ifstream in(p, std::ios::binary); std::stringstream ss; ss << in.rdbuf(); return ss.str(); };
        const std::string once = slurp(f2);
        CHECK(std::system(cmd.c_str()) == 0);
        CHECK(slurp(f2) == once);   // the second run wrote nothing
        const std::string out = slurp(dir / "out.txt");
        CHECK(out.find("migrated 0, already 3") != std::string::npos);
        std::print("{}", out);
    }
    fs::remove_all(dir);
    if (failures == 0) std::println("test_yaw_migration: OK");
    return failures == 0 ? 0 : 1;
}
