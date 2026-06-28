/*
 * prior_store.cpp — TOML I/O for optional [[humans]] priors (toml++ v3).
 */

#include "prior_store.h"

#include <print>

#include <toml++/toml.hpp>

namespace rc {

PriorStore::PriorStore(std::string priors_path)
    : priors_path_(std::move(priors_path))
{}

std::vector<HumanPrior> PriorStore::load_priors() const
{
    std::vector<HumanPrior> result;

    toml::table tbl;
    try
    {
        tbl = toml::parse_file(priors_path_);
    }
    catch (const toml::parse_error& e)
    {
        std::print("PriorStore: failed to parse '{}': {}\n", priors_path_, e.what());
        return result;
    }

    const auto* arr = tbl.get_as<toml::array>("humans");
    if (not arr)
    {
        std::print("PriorStore: no [[humans]] array in '{}' (live-track scaffolding only)\n", priors_path_);
        return result;
    }

    for (const auto& elem : *arr)
    {
        const auto* t = elem.as_table();
        if (not t) continue;

        HumanPrior p;
        if (auto v = t->get_as<std::string>("node_name")) p.node_name = **v;
        if (auto v = t->get_as<std::int64_t>("track_id"))  p.track_id  = static_cast<int>(**v);
        if (auto v = t->get_as<double>("room_x_m"))        p.room_x_m  = static_cast<float>(**v);
        if (auto v = t->get_as<double>("room_y_m"))        p.room_y_m  = static_cast<float>(**v);
        if (auto v = t->get_as<double>("room_z_m"))        p.room_z_m  = static_cast<float>(**v);

        if (p.node_name.empty())
        {
            std::print("PriorStore: skipping human entry with no node_name\n");
            continue;
        }
        result.push_back(std::move(p));
    }

    std::print("PriorStore: loaded {} human prior(s) from '{}'\n", result.size(), priors_path_);
    return result;
}

}  // namespace rc
