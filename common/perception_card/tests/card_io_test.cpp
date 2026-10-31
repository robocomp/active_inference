// Card file I/O: atomic write, read, corrupt -> nullopt, tree scan.   LANG=es_ES.UTF-8 ./card_io_test
#include "perception_card/card_io.h"
#include <clocale>
#include <cstdio>
#include <fstream>

static int bad = 0;
#define CHECK(c) do { if (not (c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)
using namespace rc::card;
namespace fs = std::filesystem;

static Card make(const std::string& name)
{
    Card c;
    c.concept_ = name; c.graph_type = name; c.level = Level::Countable;
    c.channels = {{"zed", Role::Both, true}};
    c.geometry = {0.7f, 0.05f, 2.0f, 0.0f, 2.05f};
    SubtypeParams d; d.env = {0.12f, 0.55f, 0.07f}; d.q_return = 0.85f;
    d.terms = {{"range_conf", {{"r0", 4.5f}, {"k", 0.123456f}}}};
    c.subtypes[""] = d;
    c.calibration = "fitted";
    return c;
}

int main()
{
    CHECK(std::setlocale(LC_ALL, "") != nullptr);
    const fs::path root = fs::temp_directory_path() / "card_io_test_tree";
    fs::remove_all(root);
    fs::create_directories(root);

    const Card a = make("door");
    CHECK(write_card_file(root / "door_concept", a));      // creates etc/
    CHECK(fs::exists(root / "door_concept/etc/perception_card.json"));
    CHECK(not fs::exists(root / "door_concept/etc/perception_card.json.tmp"));
    const auto r = read_card_file(root / "door_concept");
    CHECK(r.has_value());
    CHECK(r and to_json(*r) == to_json(a));
    CHECK(r and r->subtypes.at("").terms[0].params.at("k") == 0.123456f);

    CHECK(write_card_file(root / "bottle_concept", make("bottle")));
    CHECK(write_card_file(root / "not_an_agent", make("zzz")));      // ignored: no _concept suffix
    fs::create_directories(root / "empty_concept");                  // no card: skipped silently
    fs::create_directories(root / "bad_concept/etc");
    { std::ofstream(root / "bad_concept/etc/perception_card.json") << "{ not json"; }
    CHECK(not read_card_file(root / "bad_concept").has_value());
    CHECK(not read_card_file(root / "empty_concept").has_value());

    const auto all = read_all_cards(root);
    CHECK(all.size() == 2);
    CHECK(all.size() == 2 and all[0].concept_ == "bottle" and all[1].concept_ == "door");

    fs::remove_all(root);
    std::printf(bad ? "FAILED %d\n" : "card_io_test OK\n", bad);
    return bad ? 1 : 0;
}
