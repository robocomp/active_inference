// Persistence of the repertory: JSON is the store of record, CSV is a read-only view for humans.
#pragma once
#include "repertory.h"
#include "human.h"

#include <filesystem>

namespace ltom
{
// blocked: a bad file could not be moved aside (rename failed). Whatever was readable is loaded, but the caller
// must NOT save: the next save would rotate the bad main over the good .prev.
enum class LoadResult { loaded, loaded_prev, missing, unreadable, blocked };

// Loads `file`, else `file.prev`. If neither parses, the bad main file is moved to `file.bad.<epoch_ms>`
// and `unreadable` is returned (the repertory is left empty). The caller then runs close_open_lives_as_gap().
// `human` (optional) receives the file's "humans" section (the one human at the chat); a file without the section
// leaves it empty ("no human known yet").
LoadResult load_repertory(const std::filesystem::path& file, Repertory& into, HumanBook* human);
LoadResult load_repertory(const std::filesystem::path& file, Repertory& into);

// Writes `file.tmp`, then rotates the old `file` to `file.prev` and renames tmp into place.
// With `human`, the file also carries a "humans" list (one entry today); nullptr writes no section.
bool save_repertory(const std::filesystem::path& file, const Repertory& r, const HumanBook* human);
bool save_repertory(const std::filesystem::path& file, const Repertory& r);

// One row per entry (its last life and last keyframe).
bool write_csv(const std::filesystem::path& file, const Repertory& r);
}
