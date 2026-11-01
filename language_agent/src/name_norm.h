#pragma once
/*
 * name_norm.h — ONE rule for the loose names a person (or the model) uses for doors, rooms, objects and places:
 * one leading article dropped (EN/ES), case folded, '-' / ' ' / '_' equivalent. "the door 2", "Door-2" and "door_2" are
 * one name; "Room_2" and "room 2" too. Header-only, stdlib only (topology.h and can_see-side lookups both use it).
 */

#include <cctype>
#include <string>
#include <string_view>

namespace rc::chat
{

[[nodiscard]] inline std::string norm_name(std::string_view s)
{
    while (not s.empty() and std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
    while (not s.empty() and std::isspace(static_cast<unsigned char>(s.back()))) s.remove_suffix(1);
    std::string t(s);
    for (auto& c : t) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    for (const std::string_view art : {"the ", "an ", "a ", "el ", "la ", "los ", "las ", "unas ", "unos ", "una ", "un "})
        if (t.starts_with(art)) { t.erase(0, art.size()); break; }
    for (auto& c : t) if (c == '-' or c == ' ') c = '_';
    return t;
}

}   // namespace rc::chat
