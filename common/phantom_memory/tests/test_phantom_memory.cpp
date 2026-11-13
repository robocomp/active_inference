// phantom_memory.h + door_concept/src/phantom_bridge.h (2026-11-13, door memory through ltsm, Task 5).
//   g++ -std=c++23 -O1 -w -I/usr/include/eigen3 common/phantom_memory/tests/test_phantom_memory.cpp -o /tmp/tpm && /tmp/tpm
// The round trip runs under the machine's real locale (es_ES.UTF-8: comma decimals), which is what breaks strtof.
#include "../phantom_memory.h"
#include "../../../door_concept/src/phantom_bridge.h"
#include "../../../door_concept/src/core/place_prior.h"

#include <clocale>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <fstream>
#include <print>

namespace
{
int failures = 0, checks = 0;
void check(bool ok, const char* what)
{
    ++checks;
    if (not ok) { ++failures; std::println("FAIL: {}", what); }
}
const std::filesystem::path kDir = std::filesystem::temp_directory_path() / "test_phantom_memory";

using rc::phantom::Memory;
using rc::phantom::Row;

void round_trip_under_comma_locale()
{
    const char* loc = std::setlocale(LC_ALL, "es_ES.UTF-8");
    if (loc == nullptr) loc = std::setlocale(LC_ALL, "");
    char probe[16]; std::snprintf(probe, sizeof probe, "%.1f", 0.5);
    std::println("locale '{}' prints 0.5 as '{}'", loc ? loc : "(none)", probe);
    check(probe[1] == ',', "the test runs under a COMMA locale (else it proves nothing about strtof)");
    std::filesystem::create_directories(kDir);
    const auto p = kDir / "rt.csv";
    Memory m;
    m.rows = {{1, -4.8f, 0.93f, 1.5707963f, 1.00258f, 3.0f, 0.0237f, 1'794'526'950'123},
              {3, 0.260417f, -9.2e-05f, -0.125f, 0.9f, 2.1f, 0.08f, 1'794'526'951'000},
              {3, 1e7f, -3.4e-3f, 0.f, 0.f, 0.f, 0.f, 0}};
    m.rows[0].exist_logodds = -3.25f; m.rows[0].exist_physical = -0.125f; m.rows[0].seen_ms = 1'794'526'940'000;
    m.rows[0].ledger = "VL3 1 0.5 2.25 3 -0.0001 1.5e-05";   // spaces and decimal points, no comma
    m.rows[1].exist_logodds = 0.260417f;                       // only L present: the other cells are empty
    check(m.save(p), "save ok");
    check(not std::filesystem::exists(p.string() + ".tmp"), "no .tmp left behind (atomic rename)");
    {   // the file itself has points, never commas
        std::ifstream f(p); std::string all((std::istreambuf_iterator<char>(f)), {});
        check(all.find("0.260417") != std::string::npos, "file written with '.' decimals");
        std::println("{}", all);
    }
    const auto l = Memory::load(p);
    check(l.has_value() and l->rows.size() == 3, "3 rows load back");
    if (l and l->rows.size() == 3)
        for (std::size_t i = 0; i < 3; ++i)
        {
            const auto &a = m.rows[i], &b = l->rows[i];
            check(a.room_id == b.room_id and a.removed_ms == b.removed_ms, "ids and ms exact");
            check(a.exist_logodds == b.exist_logodds and a.exist_physical == b.exist_physical and a.seen_ms == b.seen_ms and a.ledger == b.ledger,
                  "evidence columns round trip exactly (-3.25, 0.260417, absent stays absent, ledger cell)");
            check(std::abs(a.x - b.x) < 1e-6f and std::abs(a.y - b.y) < 1e-6f and std::abs(a.yaw - b.yaw) < 1e-6f
                  and std::abs(a.width - b.width) < 1e-6f and std::abs(a.height - b.height) < 1e-6f
                  and std::abs(a.height_sd - b.height_sd) < 1e-6f, "floats equal to 1e-6 (0.260417 must not read as 0)");
        }
    {   // a bad row is skipped, the rest survive; a truncated one too
        std::ofstream f(kDir / "bad.csv");
        f << "room_id,x,y,yaw,width,height,height_sd,removed_ms\n1,0.5,0.5,0,1,2,0.1,5\nnot,a,row\n2,1,1,0,1,2,0.1,7\n3,1,1,0\n";
    }
    {   // the 8-column form (a file written before the evidence columns): read as "absent"
        std::ofstream f(kDir / "old8.csv");
        f << "room_id,x,y,yaw,width,height,height_sd,removed_ms\n1,0.5,0.25,0,1,2,0.1,5\n";
    }
    const auto o8 = Memory::load(kDir / "old8.csv");
    check(o8.has_value() and o8->rows.size() == 1 and not o8->rows[0].exist_logodds and not o8->rows[0].seen_ms and o8->rows[0].ledger.empty()
          and o8->rows[0].x == 0.5f and o8->rows[0].y == 0.25f, "8-column file: evidence absent");
    const auto b = Memory::load(kDir / "bad.csv");
    check(b.has_value() and b->rows.size() == 2 and b->rows[1].room_id == 2, "bad rows skipped");
}

void forgotten_room_rows_dropped()
{
    Memory m;
    m.rows = {{1, 0, 0, 0, 1, 2, .1f, 1}, {2, 0, 0, 0, 1, 2, .1f, 1}, {3, 0, 0, 0, 1, 2, .1f, 1}, {3, 1, 1, 0, 1, 2, .1f, 1}};
    const std::uint64_t keep[] = {1, 3};
    check(m.forget_except(keep) == 1, "one row dropped");
    check(m.rows.size() == 3 and m.of_room(2).empty() and m.of_room(3).size() == 2, "room 2 gone, rooms 1 and 3 kept");
    check(m.forget_except(std::span<const std::uint64_t>{}) == 3 and m.rows.empty(), "forget_except with an empty list forgets all (the primitive)");
}

// An EMPTY room_ids array is LTSM's memory "not loaded yet" (a transient), never a ruling: the bridge's forget() keeps everything.
void empty_room_list_forgets_nothing()
{
    Memory m;
    m.rows = {{1, 0, 0, 0, 1, 2, .1f, 1}, {2, 0, 0, 0, 1, 2, .1f, 1}, {3, 0, 0, 0, 1, 2, .1f, 1}};
    const std::map<std::string, std::uint64_t> entered{{"room_1", 1}};
    check(rc::door::phantom_bridge::forget(m, std::span<const std::uint64_t>{}, entered) == 0 and m.rows.size() == 3,
          "empty room_ids: nothing forgotten (rooms 2 and 3 not yet entered)");
    const std::uint64_t ids[] = {3};
    check(rc::door::phantom_bridge::forget(m, ids, entered) == 1 and m.of_room(2).empty() and m.of_room(1).size() == 1 and m.of_room(3).size() == 1,
          "a real list forgets room 2 only; the room entered this run (1) stays though LTSM has not archived it");
}

// A removed place refuted by VISION (a normal-height door, L = -3 when removed) must seed a new candidate exactly as the same row loaded from
// the identity CSV does: the phantom file carries the place's own evidence (L, Sp, last sighting, ledger).
void vision_refuted_phantom_seeds_like_the_csv_row()
{
    using namespace rc::door;
    using namespace rc::door::core;
    constexpr std::int64_t kRemoved = 1'794'526'950'000, kNow = kRemoved + 120'000;
    IdentityRow ir;
    ir.room = "room_1"; ir.name = "door_4"; ir.xy = {2.5f, -1.25f};
    ir.pose = RestorePose{0.3f, 0.95f, 2.1f, 0.12f};
    ir.exist_logodds = -3.0f; ir.exist_physical = -0.5f; ir.seen_wall_ms = kRemoved - 5'000; ir.removed_ms = kRemoved;
    ParsedIdentityTable parsed; parsed.rows = {ir};
    PlaceTable csv(false);
    csv.load(parsed, 0.1f);
    const PlaceRow* a = csv.match("room_1", Vec2f(2.5f, -1.25f), Eigen::Matrix2f::Identity() * 0.04f, 9.0f);
    check(a != nullptr and a->removed_wall_ms.has_value(), "CSV path: the removed row matches");
    // the phantom path: the table's removed rows -> file (round trip under the comma locale) -> a fresh table
    Memory m;
    m.set_room(7, phantom_bridge::removed_rows(csv, 7, "room_1"));
    check(m.save(kDir / "vr.csv"), "save");
    const auto l = Memory::load(kDir / "vr.csv");
    PlaceTable ph(false);
    check(l.has_value() and phantom_bridge::inject(ph, *l, 7, "room_1", 0.1f) == 1, "injected");
    const PlaceRow* b = ph.match("room_1", Vec2f(2.5f, -1.25f), Eigen::Matrix2f::Identity() * 0.04f, 9.0f);
    check(b != nullptr and b->removed_wall_ms.has_value(), "phantom path: the removed row matches");
    if (a and b)
    {
        const auto ca = csv.restored_belief(*a, kNow), cb = ph.restored_belief(*b, kNow);
        std::println("birth prior, CSV path L_evidence {:+.5f}, phantom path {:+.5f}", ca.L_evidence(), cb.L_evidence());
        check(ca.L_evidence() < -1.0f, "the vision-refuted row pushes the candidate down (not neutral)");
        check(std::abs(ca.L_evidence() - cb.L_evidence()) < 1e-5f and std::abs(ca.Sp() - cb.Sp()) < 1e-5, "same birth prior as the CSV row");
        check(std::abs(csv.aged_L(*a, kNow) - ph.aged_L(*b, kNow)) < 1e-5f, "same aged L");
    }
}

void missing_file_is_empty()
{
    const auto m = Memory::load(kDir / "does_not_exist.csv");
    check(m.has_value() and m->rows.empty(), "missing file -> empty Memory");
}

// clean60's GlassBay (see door_concept/tests/test_door_remembered_height.cpp): a 3 m glass partition at (-4.8, 0.93), refuted for its height.
void glass_partition_not_reborn()
{
    using namespace rc::door;
    using namespace rc::door::core;
    constexpr std::int64_t kRemovedMs = 1'794'526'950'000;
    const SizePrior prior{1.00f, 0.06f, 2.10f, 0.08f};
    const float base = -0.41f, delete_line = -1.08f, vision = 1.5f;
    const auto size_term = [&](float w, float w_sd, float h, float h_sd)
    {
        const std::vector<Vec2f> none;
        PlacePriorInput in{none, true, rc::door::RoomBand{}, Vec2f::Zero(), Vec2f::UnitX(), w, h, w_sd, h_sd, true, 0, prior, 0.f};
        return place_prior(in).size;
    };
    // the file, written and read back (a later run), then injected for room_id 7 under the room's live name
    Memory m;
    m.rows = {{7, -4.8f, 0.93f, 0.f, 1.00f, 3.0f, 0.024f, kRemovedMs}, {9, 0.f, 0.f, 0.f, 1.f, 2.1f, 0.05f, kRemovedMs}};
    check(m.save(kDir / "gb.csv"), "save");
    const auto l = Memory::load(kDir / "gb.csv");
    check(l.has_value(), "load");
    PlaceTable t(false);
    check(phantom_bridge::inject(t, *l, 7, "room_1", 0.1f) == 1, "only room 7's row injected under room_1");
    check(phantom_bridge::inject(t, *l, 7, "room_1", 0.1f) == 0, "re-entry does not duplicate");
    // a weak detection at the place 0.3 s after the removal
    const PlaceRow* row = t.match("room_1", Vec2f(-4.8f, 0.93f), Eigen::Matrix2f::Identity() * 0.04f, 9.0f);
    check(row != nullptr and row->removed_wall_ms.has_value(), "the detection matches the injected removed place");
    if (row)
    {
        const auto mh = t.remembered_height(*row, kRemovedMs + 300, prior.h_sd);
        check(mh.has_value(), "the place's height is remembered");
        const float s = mh ? size_term(1.00f, 0.20f, mh->height, mh->sd) : 0.f;
        const float L = base + s + t.aged_L(*row, kRemovedMs + 300) + vision;
        std::println("glass bay from the phantom file: size {:+.2f}, L {:+.2f} (delete line {:+.2f})", s, L, delete_line);
        check(L < delete_line, "not born again: below the delete line");
        check(base + size_term(1.00f, 0.20f, prior.h_m, prior.h_sd) + vision > 0.f, "(without the memory it WOULD be born)");
    }
    // the save side: removed rows of the entered room come back as file rows; a room never entered keeps its file rows
    const auto out = phantom_bridge::merged(t, *l, {{"room_1", 7}});
    check(out.of_room(7).size() == 1 and out.of_room(9).size() == 1, "merged keeps the other room's row, rewrites room 7's");
    check(out.of_room(7)[0].height == 3.0f and out.of_room(7)[0].removed_ms == kRemovedMs, "room 7's row round-trips through the table");
    // a swap drops a provisional room: nothing entered for it, so its table rows never reach the file
    const auto dropped = phantom_bridge::merged(t, Memory{}, {});
    check(dropped.rows.empty(), "an un-entered (provisional) room contributes nothing");
}
}   // namespace

int main()
{
    std::filesystem::remove_all(kDir);
    round_trip_under_comma_locale();
    forgotten_room_rows_dropped();
    empty_room_list_forgets_nothing();
    vision_refuted_phantom_seeds_like_the_csv_row();
    missing_file_is_empty();
    glass_partition_not_reborn();
    std::println("{} checks, {} failure(s)", checks, failures);
    return failures == 0 ? 0 : 1;
}
