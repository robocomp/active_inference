// skills (src/skills.h): the loader, the index line, and — the point of the missions skill — every example mission
// in etc/skills/missions.md must parse and validate with the real checker.
#include "../src/skills.h"

#include <clocale>
#include <fstream>
#include <print>

#include "../../common/mission/mission_text.h"

namespace
{
int failures = 0, checks = 0;
void check(bool ok, int line, std::string_view what)
{
    ++checks;
    if (not ok)
    {
        ++failures;
        std::println("FAIL line {}: {}", line, what);
    }
}
#define CHECK(x) check((x), __LINE__, #x)

void write(const std::filesystem::path& p, std::string_view s) { std::ofstream(p) << s; }

using namespace rc::chat;
}   // namespace

int main()
{
    std::setlocale(LC_ALL, "");
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "test_skills_dir";
    fs::remove_all(dir);
    fs::create_directories(dir);
    write(dir / "a.md", "---\ndescription: does A\n---\nbody of A\nline 2\n");
    write(dir / "b.md", "no front matter here\n");
    write(dir / "c.txt", "---\ndescription: not markdown\n---\nx\n");

    const auto skills = load_skills(dir);
    CHECK(skills.size() == 1);
    if (skills.size() == 1)
    {
        CHECK(skills[0].name == "a");
        CHECK(skills[0].description == "does A");
        CHECK(skills[0].body == "body of A\nline 2\n");
    }
    CHECK(skills_index(skills) == "- a: does A\n");
    CHECK(load_skills(dir / "nope").empty());
    fs::remove_all(dir);

    // ── the shipped skills: all three load; every ```-block in missions.md parses + validates ───────────────────
    const auto shipped = load_skills(SKILLS_DIR);
    CHECK(shipped.size() == 3);
    int examples = 0;
    for (const auto& s : shipped)
    {
        CHECK(not s.description.empty());
        if (s.name != "missions") continue;
        std::size_t pos = 0;
        while ((pos = s.body.find("```", pos)) != std::string::npos)
        {
            const std::size_t start = s.body.find('\n', pos) + 1, end = s.body.find("```", start);
            if (end == std::string::npos) break;
            const std::string block = s.body.substr(start, end - start);
            pos = end + 3;
            if (not block.starts_with("mission")) continue;
            ++examples;
            const auto m = rc::mission::parse_text(block);
            if (not m.has_value()) std::println("  example {}: {}\n{}", examples, m.error().text, block);
            CHECK(m.has_value());
        }
    }
    CHECK(examples >= 5);

    // the prose claims in missions.md / doors.md are pinned here, not just the examples
    using rc::mission::parse_text;
    CHECK(not parse_text("mission \"x\"\n  go_to door_1\n").has_value());
    CHECK(not parse_text("mission \"x\"\n  go_to passage:p@room_2\n").has_value());
    CHECK(parse_text("mission \"x\"\n  go_to pose:room_1(1, 2)\n").has_value());
    CHECK(parse_text("mission \"x\"\n  anchor a\n  go_to anchor:a\n").has_value());
    CHECK(not parse_text("mission \"x\"\n  open door_1 bind=n\n").has_value());
    CHECK(not parse_text("mission \"x\"\n  repeat until=in_room(room_1)\n    stop\n").has_value());
    CHECK(parse_text("mission \"x\"\n  repeat until=in_room(room_1) budget_s=30s\n    stop\n").has_value());
    CHECK(parse_text("mission \"x\"\n  wait until=door_open(door_1)>=0.8 s=30s\n").has_value());
    CHECK(parse_text("mission \"x\"\n  cross passage:passage_0@room_2 expect=room_1\n").has_value());

    std::println("test_skills: {} checks, {} failed", checks, failures);
    return failures == 0 ? 0 : 1;
}
