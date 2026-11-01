// Persistence of the repertory: JSON is the store of record, CSV is a read-only view for humans.
#pragma once
#include "repertory.h"

#include <filesystem>

namespace ltom
{
enum class LoadResult { loaded, loaded_prev, missing, unreadable };

// Loads `file`, else `file.prev`. If neither parses, the bad main file is moved to `file.bad.<epoch_ms>`
// and `unreadable` is returned (the repertory is left empty). The caller then runs close_open_lives_as_gap().
LoadResult load_repertory(const std::filesystem::path& file, Repertory& into);

// Writes `file.tmp`, then rotates the old `file` to `file.prev` and renames tmp into place.
bool save_repertory(const std::filesystem::path& file, const Repertory& r);

// One row per entry (its last life and last keyframe).
bool write_csv(const std::filesystem::path& file, const Repertory& r);
}
