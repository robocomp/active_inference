// chair_yaw_migrate — turn a repertory's stored chair yaws and thumbnail bins into the countable frame, ONCE (shared-engine K2b).
//
//   chair_yaw_migrate <ltom_repertory.json> <legacy_until_ms> [class] [--dry-run]
//
// legacy_until_ms: the wall-clock epoch ms from which the chair ran on the countable binary (keyframes and moves before it are
// the legacy chair's, see src/yaw_migration.h). class defaults to "chair". The file is loaded and saved through ltom's own
// repertory_io (the old file is kept as <file>.prev by the save); run it with ltom_agent STOPPED, or ltom's next save
// overwrites the result. Idempotent: a second run reports every entry as already migrated and writes nothing.
// Refuses a file that did not load as itself (only its .prev, or blocked): migrating a backup would hide the bad main.
// ⚠ The owner decides when it runs on real data (K2b fix round 1); the test runs it on a synthetic repertory only.
#include "../src/repertory_io.h"
#include "../src/yaw_migration.h"

#include <charconv>
#include <clocale>
#include <cstdio>
#include <print>
#include <string>
#include <string_view>

int main(int argc, char** argv)
{
    std::setlocale(LC_ALL, "");
    if (argc < 3)
    {
        std::println(stderr, "usage: {} <ltom_repertory.json> <legacy_until_ms> [class=chair] [--dry-run]", argv[0]);
        return 2;
    }
    const std::filesystem::path file = argv[1];
    std::int64_t until = 0;
    {
        const std::string_view s = argv[2];
        const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), until);
        if (ec != std::errc{} or p != s.data() + s.size() or until <= 0)
        {
            std::println(stderr, "legacy_until_ms '{}' is not a positive integer (epoch ms)", s);
            return 2;
        }
    }
    std::string cls = "chair";
    bool dry = false;
    for (int i = 3; i < argc; ++i)
        if (std::string_view(argv[i]) == "--dry-run") dry = true;
        else cls = argv[i];

    ltom::Repertory r(0);
    ltom::HumanBook human;
    if (const auto res = ltom::load_repertory(file, r, &human); res != ltom::LoadResult::loaded)
    {
        std::println(stderr, "{}: did not load as itself (result {}) — refusing to migrate", file.string(), static_cast<int>(res));
        return 1;
    }
    const auto rep = ltom::migrate_yaw_convention(r, cls, until);
    std::println("{}: class '{}' before {} ms — migrated {}, already {}, other classes {}; keyframes turned {}, moves turned {}, "
                 "thumbnails turned {}, dropped (mixed) {}{}", file.string(), cls, until, rep.migrated, rep.already, rep.other_class,
                 rep.keyframes_turned, rep.moves_turned, rep.thumbs_turned, rep.thumbs_dropped, dry ? " [dry run: nothing written]" : "");
    if (dry or rep.migrated == 0) return 0;
    if (not ltom::save_repertory(file, r, &human))
    {
        std::println(stderr, "{}: save FAILED", file.string());
        return 1;
    }
    return 0;
}
