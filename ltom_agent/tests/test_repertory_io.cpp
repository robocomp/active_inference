// Offline test of src/repertory_io.h (JSON store of record + CSV view). Runs under the es_ES locale the agents use.
#include "../src/repertory_io.h"

#include <clocale>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <print>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

using namespace ltom;
namespace fs = std::filesystem;

static int failures = 0;
#define CHECK(c) do { if (not (c)) { std::println(stderr, "FAIL {}:{}: {}", __FILE__, __LINE__, #c); ++failures; } } while (0)

static fs::path fresh_dir()
{
    static int k = 0;
    const fs::path d = fs::temp_directory_path() / ("ltom_io_test_" + std::to_string(::getpid()) + "_" + std::to_string(k++));
    fs::remove_all(d);
    fs::create_directories(d);
    return d;
}
static std::string read_file(const fs::path& p)
{
    std::ifstream f(p);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
static void write_file(const fs::path& p, const std::string& t) { std::ofstream f(p); f << t; }

constexpr std::uint64_t kBig = (20ull << 40) | 123456789ull;     // > 2^53
constexpr std::uint64_t kSmall = (21ull << 40) | 2;

static Repertory make_repertory()
{
    Repertory r(3000);
    Entry a;
    a.uid = kBig; a.cls = "chair"; a.name = "chair_1"; a.w = 0.5f; a.d = 0.45f; a.h = 1.25f; a.n_sightings = 42;
    a.lives = {Life{100, 200, 300, 400, End::death}, Life{500, 600, 700, 800, End::owner_exit},
               Life{900, 1000, 1100, 1100, End::gap}, Life{1200, 1300, 1400, -1, End::alive}};
    a.keyframes = {Keyframe{250, 1, Pose2{1.25, 2.5, 0.3, 0.0009, 0.0016, 0.01}},
                   Keyframe{1350, 2, Pose2{-0.75, 3.125, -1.5, 0.0025, 0.0036, 0.02}}};
    Entry b;
    b.uid = kSmall; b.cls = "table"; b.name = "table_3"; b.w = 1.5f; b.d = 0.75f; b.h = 0.75f; b.n_sightings = 3;
    b.lives = {Life{10, 20, 30, -1, End::alive}};
    b.keyframes = {Keyframe{25, 7, Pose2{0.5, 0.5, 0.0, 0.01, 0.01, 0.01}}};
    r.insert_loaded(a);
    r.insert_loaded(b);
    return r;
}

static void test_round_trip_preserves_entries()
{
    const auto f = fresh_dir() / "rep.json";
    const Repertory src = make_repertory();
    CHECK(save_repertory(f, src));
    Repertory got(3000);
    CHECK(load_repertory(f, got) == LoadResult::loaded);
    CHECK(got.entries().size() == 2);
    const Entry& a = got.entries().at(kBig);
    CHECK(a.cls == "chair" and a.name == "chair_1" and a.w == 0.5f and a.d == 0.45f and a.h == 1.25f);
    CHECK(a.n_sightings == 42 and a.lives.size() == 4 and a.keyframes.size() == 2);
    CHECK(a.lives[0].end == End::death and a.lives[1].end == End::owner_exit
          and a.lives[2].end == End::gap and a.lives[3].end == End::alive);
    CHECK(a.lives[0].born_ms == 100 and a.lives[0].first_seen_ms == 200 and a.lives[0].last_seen_ms == 300
          and a.lives[0].end_ms == 400 and a.lives[3].end_ms == -1);
    CHECK(a.keyframes[0].t_ms == 250 and a.keyframes[0].room_key == 1);
    CHECK(a.keyframes[0].pose.x == 1.25 and a.keyframes[0].pose.y == 2.5 and a.keyframes[0].pose.yaw == 0.3);
    CHECK(a.keyframes[0].pose.vx == 0.0009 and a.keyframes[0].pose.vy == 0.0016 and a.keyframes[0].pose.vyaw == 0.01);
    CHECK(a.keyframes[1].pose.x == -0.75 and a.keyframes[1].room_key == 2);
    CHECK(got.entries().at(kSmall).cls == "table");
    CHECK(not got.dirty());
}

static void test_load_closes_open_lives_as_gap()
{
    const auto f = fresh_dir() / "rep.json";
    CHECK(save_repertory(f, make_repertory()));
    Repertory got(3000);
    CHECK(load_repertory(f, got) == LoadResult::loaded);
    got.close_open_lives_as_gap();
    const Life& l = got.entries().at(kSmall).lives.back();
    CHECK(l.end == End::gap and l.end_ms == l.last_seen_ms and l.end_ms == 30);
}

static void test_corrupt_main_falls_back_to_prev()
{
    const auto f = fresh_dir() / "rep.json";
    CHECK(save_repertory(f, make_repertory()));
    CHECK(save_repertory(f, make_repertory()));       // second save rotates the first to .prev
    CHECK(fs::exists(fs::path(f).concat(".prev")));
    write_file(f, "{ this is not json");
    Repertory got(3000);
    CHECK(load_repertory(f, got) == LoadResult::loaded_prev);
    CHECK(got.entries().size() == 2);
    // both bad: unreadable, bad main moved aside
    write_file(fs::path(f).concat(".prev"), "garbage");
    Repertory none(3000);
    CHECK(load_repertory(f, none) == LoadResult::unreadable);
    CHECK(none.entries().empty());
    CHECK(not fs::exists(f));
    int bad = 0;
    for (const auto& e : fs::directory_iterator(f.parent_path()))
        if (e.path().filename().string().starts_with("rep.json.bad.")) ++bad;
    CHECK(bad == 1);
}

static void test_missing_is_missing()
{
    Repertory r(3000);
    CHECK(load_repertory(fresh_dir() / "nope.json", r) == LoadResult::missing);
    CHECK(r.entries().empty());
}

static void test_csv_has_points_not_commas()
{
    const auto d = fresh_dir();
    CHECK(write_csv(d / "rep.csv", make_repertory()));
    const std::string t = read_file(d / "rep.csv");
    CHECK(t.find("1.25") != std::string::npos);
    CHECK(t.find("1,25") == std::string::npos);
}

static void test_csv_row_count_and_header()
{
    const auto d = fresh_dir();
    CHECK(write_csv(d / "rep.csv", make_repertory()));
    std::istringstream in(read_file(d / "rep.csv"));
    std::string line, header;
    std::getline(in, header);
    CHECK(header == "uid,owner,class,name,lives,state,born_ms,first_seen_ms,last_seen_ms,end_ms,room_key,x,y,yaw,sx,sy,w,d,h,n_sightings,n_keyframes");
    int rows = 0;
    while (std::getline(in, line)) if (not line.empty()) ++rows;
    CHECK(rows == 2);
}

static int count_prefix(const fs::path& dir, const std::string& pre)
{
    int n = 0;
    for (const auto& e : fs::directory_iterator(dir))
        if (e.path().filename().string().starts_with(pre)) ++n;
    return n;
}

static void test_fallback_moves_bad_main_aside()
{
    const auto f = fresh_dir() / "rep.json";
    const fs::path prev = fs::path(f).concat(".prev");
    CHECK(save_repertory(f, make_repertory()));
    CHECK(save_repertory(f, make_repertory()));
    write_file(f, "{ broken");
    const std::string prev_before = read_file(prev);
    Repertory got(3000);
    CHECK(load_repertory(f, got) == LoadResult::loaded_prev);
    CHECK(not fs::exists(f));
    CHECK(count_prefix(f.parent_path(), "rep.json.bad.") == 1);
    CHECK(read_file(prev) == prev_before);
    CHECK(save_repertory(f, got));
    Repertory a(3000), b(3000);
    CHECK(load_repertory(f, a) == LoadResult::loaded and a.entries().size() == 2);
    write_file(f, "x");
    CHECK(load_repertory(f, b) == LoadResult::loaded_prev and b.entries().size() == 2);
}

static void test_missing_main_bad_prev_moved_aside()
{
    const auto f = fresh_dir() / "rep.json";
    const fs::path prev = fs::path(f).concat(".prev");
    write_file(prev, "garbage");
    Repertory r(3000);
    CHECK(load_repertory(f, r) == LoadResult::unreadable);
    CHECK(not fs::exists(prev));
    CHECK(count_prefix(f.parent_path(), "rep.json.prev.bad.") == 1);
}

static void test_failed_move_aside_blocks_saving()
{
    if (::geteuid() == 0) return;   // root ignores the permission bits
    const auto d = fresh_dir();
    const auto f = d / "rep.json";
    const auto r0 = make_repertory();
    CHECK(save_repertory(f, r0));
    CHECK(save_repertory(f, r0));                     // main + .prev both good
    write_file(f, "garbage");                         // main torn
    ::chmod(d.c_str(), 0500);                         // rename out of d now fails
    Repertory r(3000);
    CHECK(load_repertory(f, r) == LoadResult::blocked);
    CHECK(r.entries().size() == 2);                   // the good .prev is still read
    ::chmod(d.c_str(), 0700);
    CHECK(fs::exists(f) and fs::exists(fs::path(f).concat(".prev")));   // nothing was lost or rotated
}

static void test_thumbnails_survive_a_round_trip()
{
    const fs::path f = fresh_dir() / "r.json";
    Repertory src(3000);
    Entry e; e.uid = kBig; e.cls = "chair"; e.name = "chair_1";
    e.thumbs[3] = Thumbnail{12.5f, {0, 255, 7, 128}};
    src.insert_loaded(e);
    CHECK(save_repertory(f, src));
    Repertory got(3000);
    CHECK(load_repertory(f, got) == LoadResult::loaded);
    const auto& t = got.entries().at(kBig).thumbs;
    CHECK(t.size() == 1 and t.at(3).score == 12.5f and t.at(3).jpeg == (std::vector<std::uint8_t>{0, 255, 7, 128}));
}

static void test_a_file_without_thumbnails_still_loads()
{
    const fs::path f = fresh_dir() / "r.json";
    Repertory src(3000);
    Entry e; e.uid = kSmall; e.cls = "chair"; e.name = "chair_2";
    src.insert_loaded(e);
    CHECK(save_repertory(f, src));
    std::string text = read_file(f);
    const auto at = text.find("\"thumbs\"");
    CHECK(at != std::string::npos);
    if (at != std::string::npos) { const auto end = text.find(']', at); text.erase(at, end - at + 1); const auto comma = text.rfind(',', at); if (text.find_first_not_of(" \n", at) != std::string::npos and text[text.find_first_not_of(" \n", at)] == '}') text.erase(comma, 1); else { const auto c2 = text.find(',', at); text.erase(c2, 1); } write_file(f, text); }
    Repertory got(3000);
    CHECK(load_repertory(f, got) == LoadResult::loaded);
    CHECK(got.entries().at(kSmall).thumbs.empty());
}

int main()
{
    test_thumbnails_survive_a_round_trip();
    test_a_file_without_thumbnails_still_loads();
    if (not std::setlocale(LC_ALL, "es_ES.UTF-8")) std::setlocale(LC_ALL, "");
    test_round_trip_preserves_entries();
    test_load_closes_open_lives_as_gap();
    test_corrupt_main_falls_back_to_prev();
    test_missing_is_missing();
    test_fallback_moves_bad_main_aside();
    test_missing_main_bad_prev_moved_aside();
    test_failed_move_aside_blocks_saving();
    test_csv_has_points_not_commas();
    test_csv_row_count_and_header();
    if (failures == 0) std::println("test_repertory_io: ok");
    return failures == 0 ? 0 : 1;
}
