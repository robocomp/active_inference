// test_place_table.cpp — Task 7: the place table (the memory row is the belief).
// Build + run (from door_concept/):
//   g++ -std=c++23 -O1 -w -I/usr/include/eigen3 -I/usr/include/opencv4 -I../common tests/test_place_table.cpp -o /tmp/test_place_table && /tmp/test_place_table
#include <clocale>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include "../src/core/place_table.h"
using namespace rc::door;
using namespace rc::door::core;
static int fails = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static PlaceRow row(const char* name, float x, float y, float L, float sd = 0.05f)
{
    PlaceRow r; r.room = "room_1"; r.name = name; r.xy = {x, y}; r.cov = Eigen::Matrix2f::Identity() * sd * sd;
    r.yaw = 0.f; r.width = 0.85f; r.depth = 0.12f; r.height = 2.05f; r.L_evidence = L; r.visit = 1; r.last_seen_wall_ms = 1'000'000;
    return r;
}

int main()
{
    std::setlocale(LC_ALL, "");
    const auto tmp = std::filesystem::temp_directory_path() / ("test_place_table_" + std::to_string(::getpid()) + ".csv");
    { // restored_continues_L: write a row with L 2.1, load, match; the candidate starts at 2.1 aged by hazard_per_s()
      PlaceTable A(false);
      PlaceRow r = row("door_3", 1.f, 2.f, 2.1f);
      ViewLedger led; const auto v = led.observe({3.f, 2.f}, {1.f, 2.f}, {1.f, 0.f}, 0.5f, 0.f, 1); led.add_actuation(v, 0.7f); led.recompute(v, false);
      r.ledger = led;
      A.upsert(r);
      { std::ofstream f(tmp); write_identity_table(f, A.to_identity_rows()); }
      PlaceTable B(false);
      { std::ifstream f(tmp); B.load(parse_identity_table(f), 0.05f); }
      CHECK(B.rows().size() == 1);
      Eigen::Matrix2f c = Eigen::Matrix2f::Identity() * 0.0025f;
      PlaceRow* m = B.match("room_1", {1.01f, 2.f}, c, 3.f);
      CHECK(m != nullptr);
      if (m)
      {
          CHECK(std::fabs(m->L_evidence - 2.1f) < 1e-5f);
          CHECK(m->ledger.views().size() == 1);
          CHECK(std::fabs(m->ledger.sum() - 0.7f) < 1e-5f);
          CHECK(m->visit == 1);
          const std::int64_t now = m->last_seen_wall_ms + 300'000;
          const float expect = 2.1f * std::exp(-float(B.hazard_per_s()) * 300.f);
          CHECK(std::fabs(B.aged_L(*m, now) - expect) < 1e-5f);
          CHECK(B.aged_L(*m, now) < 2.1f and B.aged_L(*m, now) > 0.f);
          CHECK(B.aged_L(*m, m->last_seen_wall_ms) == 2.1f);
      }
      CHECK(B.match("room_2", {1.f, 2.f}, c, 3.f) == nullptr);
    }
    { // phantom_row_is_prior: a removed row matched by a new candidate starts it at -3 (kept for ever, not aged)
      PlaceTable T(false);
      PlaceRow r = row("door_9", 4.f, 4.f, -3.f); r.removed_wall_ms = 5'000'000;
      T.upsert(r);
      PlaceRow* m = T.match("room_1", {4.02f, 3.99f}, Eigen::Matrix2f::Identity() * 0.01f, 3.f);
      CHECK(m != nullptr);
      if (m) { CHECK(m->removed_wall_ms.has_value()); CHECK(T.aged_L(*m, m->last_seen_wall_ms) == -3.f);
        const float e = -3.f * std::exp(-float(T.hazard_per_s()) * 600.f);
        CHECK(std::fabs(T.aged_L(*m, m->last_seen_wall_ms + 600'000) - e) < 1e-5f); CHECK(e > -3.f); }
      std::ofstream f(tmp); write_identity_table(f, T.to_identity_rows()); f.close();
      PlaceTable U(false); std::ifstream g(tmp); U.load(parse_identity_table(g), 0.05f);
      CHECK(U.rows().size() == 1 and U.rows()[0].removed_wall_ms == std::optional<std::int64_t>(5'000'000));
    }
    { // rows_never_forgotten: 1000 upserts at 3 places give 3 rows
      PlaceTable T(false);
      for (int i = 0; i < 1000; ++i)
      {
          const int k = i % 3;
          const std::string nm = "door_" + std::to_string(k + 1);
          T.upsert(row(nm.c_str(), float(k), 0.f, float(i % 7) - 3.f));
      }
      CHECK(T.rows().size() == 3);
    }
    { // gate_grows_with_cov: 0.8 m off matches with sigma 0.5, not with sigma 0.1
      PlaceTable T(false); T.upsert(row("door_1", 0.f, 0.f, 1.f, 0.01f));
      CHECK(T.match("room_1", {0.8f, 0.f}, Eigen::Matrix2f::Identity() * 0.25f, 9.f) != nullptr);
      CHECK(T.match("room_1", {0.8f, 0.f}, Eigen::Matrix2f::Identity() * 0.01f, 9.f) == nullptr);
      { PlaceTable Z(false); Z.upsert(row("door_1", 0.f, 0.f, 1.f, 0.f)); const Eigen::Matrix2f c = Eigen::Matrix2f::Identity() * 0.04f;   // sigma 0.2
        CHECK(Z.match("room_1", {0.5f, 0.f}, c, 9.f) != nullptr);    // 2.5 sigma: m2 = 6.25
        CHECK(Z.match("room_1", {0.7f, 0.f}, c, 9.f) == nullptr); }  // 3.5 sigma: m2 = 12.25
      // nearest of two wins
      T.upsert(row("door_2", 0.5f, 0.f, 1.f, 0.01f));
      PlaceRow* m = T.match("room_1", {0.45f, 0.f}, Eigen::Matrix2f::Identity() * 0.25f, 9.f);
      CHECK(m != nullptr and m->name == "door_2");
    }
    { // old_row_loads_with_empty_ledger: a 19-column row from today's format
      std::stringstream s(std::string("room,name,ap_x,ap_y,yaw,width,height,depth,exist_logodds,hinge_x,hinge_y,open_x,open_y,hinge_llr,swing_llr,phi,phi_sigma,seen_ms,place_sd\n")
                          + "room_1,door_5,1.5,2.5,0.1,0.9,2.0,0.2,1.8,,,,,,,,,123456,0.04\n");
      const auto P = parse_identity_table(s);
      CHECK(P.malformed == 0 and P.rows.size() == 1);
      CHECK(P.rows[0].place_sd.has_value());
      CHECK(not P.rows[0].ledger.has_value() and not P.rows[0].removed_ms.has_value());
      PlaceTable T(false); T.load(P, 0.3f);
      CHECK(T.rows().size() == 1);
      if (T.rows().size() == 1)
      {
          const auto& r = T.rows()[0];
          CHECK(r.ledger.views().empty()); CHECK(std::fabs(r.L_evidence - float(saturated_vision(1.8, 0.05))) < 1e-6f); CHECK(r.visit == 0);   // a pre-look-alike row: its L is the old raw sum, saturated once (spec §3.1b)
          CHECK(r.last_seen_wall_ms == 123456); CHECK(not r.removed_wall_ms.has_value());
          CHECK(std::fabs(r.width - 0.9f) < 1e-6f);
      }
      // a 4-column row too
      std::stringstream s4("room,name,ap_x,ap_y\nroom_1,door_6,0.5,0.5\n");
      PlaceTable T4(false); T4.load(parse_identity_table(s4), 0.3f); CHECK(T4.rows().size() == 1);
    }
    { // a corrupt ledger cell is dropped to an empty ledger; the row and its L stay
      std::stringstream s(std::string("room,name,ap_x,ap_y,yaw,width,height,depth,exist_logodds,hinge_x,hinge_y,open_x,open_y,hinge_llr,swing_llr,phi,phi_sigma,seen_ms,place_sd,ledger,visit,removed_ms\n")
                          + "room_1,door_5,1.5,2.5,0.1,0.9,2.0,0.2,1.8,,,,,,,,,123456,0.04,VL1 zz,2,\n");
      PlaceTable T(false); T.load(parse_identity_table(s), 0.3f);
      CHECK(T.rows().size() == 1 and T.rows()[0].ledger.views().empty() and std::fabs(T.rows()[0].L_evidence - float(saturated_vision(1.8, 0.05))) < 1e-6f and T.rows()[0].visit == 2);
    }
    { // hazard_learns: 10 revisits, all found after 100 s, give a smaller lambda than the prior mean
      PlaceTable T(false); PlaceRow& r = T.upsert(row("door_1", 0.f, 0.f, 1.f));
      const double prior = T.hazard_per_s();
      CHECK(std::fabs(prior - 1.0 / 600.0) < 1e-12);
      for (int i = 0; i < 10; ++i) T.note_revisit(true, 100.0);
      CHECK(T.hazard_per_s() < prior);
      CHECK(std::fabs(T.hazard_per_s() - 1.0 / 1600.0) < 1e-12);
      PlaceTable U(false); PlaceRow& q = U.upsert(row("door_1", 0.f, 0.f, 1.f));
      for (int i = 0; i < 10; ++i) U.note_revisit(false, 100.0);   // not found counts as a change
      CHECK(U.hazard_per_s() > prior);
    }
    { // the learned hazard PERSISTS: write (#hazard line + sighted_ms column), parse, load into a fresh table -> the same lambda
      PlaceTable T(false); T.upsert(row("door_1", 0.f, 0.f, 1.f));
      for (int i = 0; i < 7; ++i) T.note_revisit(true, 1234.5);
      for (int i = 0; i < 3; ++i) T.note_revisit(false, 321.25);
      auto rows = T.to_identity_rows(); rows[0].sighted_ms = 987654321;
      std::stringstream ss; write_identity_table(ss, rows, 5, T.hazard_state());
      const auto parsed = parse_identity_table(ss);
      CHECK(parsed.hazard.has_value() and parsed.malformed == 0 and parsed.rows.size() == 1);
      CHECK(parsed.rows[0].sighted_ms.has_value() and *parsed.rows[0].sighted_ms == 987654321);
      PlaceTable U(false); U.load(parsed, 0.3f);
      CHECK(U.hazard_per_s() == T.hazard_per_s());                                   // bit-identical
      CHECK(U.rows()[0].sighted_wall_ms == 987654321);
      U.load(parsed, 0.3f); CHECK(U.hazard_per_s() == T.hazard_per_s());              // loading twice is the same table (SET, not add)
      // an older file (no #hazard line) -> the Gamma prior alone
      std::stringstream old("room,name,ap_x,ap_y\nroom_1,door_6,0.5,0.5\n");
      PlaceTable V(false); V.load(parse_identity_table(old), 0.3f);
      CHECK(std::fabs(V.hazard_per_s() - 1.0 / 600.0) < 1e-12);
    }
    { // one sample per place per visit (sample_revisit); the exposure runs from the last SIGHTING
      PlaceTable T(false);
      const auto at = [](const char* name, int visit, std::int64_t sighted)
      { PlaceRow r; r.room = "room_1"; r.name = name; r.visit = visit; r.sighted_wall_ms = sighted; return r; };
      CHECK(T.sample_revisit(at("door_1", 2, 1000), 3, true, 61'000));
      CHECK(not T.sample_revisit(at("door_1", 2, 1000), 3, false, 70'000));      // same visit: no second sample, found or not
      CHECK(T.sample_revisit(at("door_1", 2, 1000), 4, false, 121'000));          // a new visit: one more
      CHECK(T.sample_revisit(at("door_2", 1, 1000), 3, true, 61'000));            // another place: its own sample
      CHECK(not T.sample_revisit(at("door_3", 1, 0), 3, true, 61'000));           // never sighted: no exposure, no sample
      CHECK(not T.sample_revisit(at("door_4", 3, 1000), 3, false, 61'000));       // I2: a row written THIS visit is no revisit
      CHECK(not T.sample_revisit(at("door_5", 5, 1000), 3, false, 61'000));       // nor one from a LATER visit id
      const auto h = T.hazard_state();
      CHECK(h.changes == 1.0 and std::fabs(h.sum_elapsed_s - (60.0 + 120.0 + 60.0)) < 1e-9);
    }
    { // session-scoped visits: max_visit() covers the rows and their ledger views
      PlaceTable T(false);
      PlaceRow a; a.room = "room_1"; a.name = "door_1"; a.visit = 4; T.upsert(a);
      PlaceRow b; b.room = "room_1"; b.name = "door_2"; b.xy = {5.f, 5.f}; b.visit = 2; T.upsert(b);
      CHECK(T.max_visit() == 4);
    }
    { // depth: 0 round-trips with no pose; real depth 0.12 round-trips as 0.12
      PlaceTable T(false); PlaceRow a = row("door_1", 0.f, 0.f, 1.f); a.depth = 0.f; PlaceRow b = row("door_2", 3.f, 0.f, 1.f); b.depth = 0.12f;
      T.upsert(a); T.upsert(b);
      const auto ids = T.to_identity_rows();
      CHECK(not ids[0].pose.has_value()); CHECK(ids[1].pose.has_value() and ids[1].pose->depth == 0.12f);
      { std::ofstream f(tmp); write_identity_table(f, ids); }
      PlaceTable U(false); { std::ifstream f(tmp); U.load(parse_identity_table(f), 0.05f); }
      CHECK(U.rows().size() == 2 and U.rows()[0].depth == 0.f and U.rows()[1].depth == 0.12f);
    }
    std::filesystem::remove(tmp);
    std::printf(fails ? "test_place_table: %d FAILED\n" : "test_place_table: ALL PASS\n", fails);
    return fails ? 1 : 0;
}
