/*
 * manifest_toml.h — parse a <concept>.concept.toml as TOML (toml++). SHARED, header-only, Qt-free.
 *
 * The one parser for whole-manifest reads (load_geometry, layer_spread, the agents' cross-checks). Its own header so
 * standalone tests and Qt-free readers can use it without concept_manifest.h's genericworker.h.
 */
#pragma once

#include <format>
#include <stdexcept>
#include <string>

#include <toml++/toml.h>

namespace rc::manifest {

// ─── PARSING A MANIFEST: toml++ directly, never ConfigLoader ──────────────────────────────────────
//
// ★A manifest is TOML, and ConfigLoader is not a TOML reader. ConfigLoader (RoboComp's agent-config loader) flattens a
// file into scalars and homogeneous scalar arrays and THROWS on anything else: an array of tables ("Unsupported array
// type encountered." — [[shape.part]], on bed / nightstand / toilet since 2026-11-15) or an empty array ("Empty arrays
// are not supported." — hood). A whole-file load then fails for a valid manifest, and every live countable run printed
// MANIFEST NOT LOADED while the textual readers (back panel, shape parts, class prior) read the same file fine. So the
// whole-file reads parse with toml++ (the parser ConfigLoader itself wraps) and look keys up by dotted path. Throws
// with the parser's reason (file, line) on an unreadable or malformed file — callers print it, never swallow it.
inline toml::table parse_manifest(const std::string& path)
{
    try { return toml::parse_file(path); }
    catch (const toml::parse_error& e)
    {
        const auto& b = e.source().begin;
        throw std::runtime_error(std::format("{}:{}:{}: {}", path, b.line, b.column, e.description()));
    }
}

}  // namespace rc::manifest
