/*
 * prior_store.cpp
 *
 * TOML I/O for bottle priors using toml++ v3.
 * Ported from table_concept (tables→bottles, width/depth→radius, +room_z_m).
 */

#include "prior_store.h"

#include <print>

#include <toml++/toml.hpp>

// ─── Constructor ─────────────────────────────────────────────────────────────

PriorStore::PriorStore(std::string priors_path)
    : priors_path_(std::move(priors_path))
{}

// ─── load_priors ─────────────────────────────────────────────────────────────

std::vector<BottlePrior> PriorStore::load_priors() const
{
    std::vector<BottlePrior> result;

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

    const auto* bottles_arr = tbl.get_as<toml::array>("bottles");
    if (not bottles_arr)
    {
        std::print("PriorStore: no [[bottles]] array found in '{}'\n", priors_path_);
        return result;
    }

    for (const auto& elem : *bottles_arr)
    {
        const auto* t = elem.as_table();
        if (not t) continue;

        BottlePrior p;
        if (auto v = t->get_as<std::string>("node_name")) p.node_name  = **v;
        if (auto v = t->get_as<double>("radius_m"))        p.radius_m   = static_cast<float>(**v);
        if (auto v = t->get_as<double>("height_m"))        p.height_m   = static_cast<float>(**v);
        if (auto v = t->get_as<double>("room_x_m"))        p.room_x_m   = static_cast<float>(**v);
        if (auto v = t->get_as<double>("room_y_m"))        p.room_y_m   = static_cast<float>(**v);
        if (auto v = t->get_as<double>("room_z_m"))        p.room_z_m   = static_cast<float>(**v);
        if (auto v = t->get_as<double>("sigma_pose"))      p.sigma_pose = static_cast<float>(**v);
        if (auto v = t->get_as<double>("sigma_size"))      p.sigma_size = static_cast<float>(**v);

        if (p.node_name.empty())
        {
            std::print("PriorStore: skipping bottle entry with no node_name\n");
            continue;
        }
        result.push_back(std::move(p));
    }

    std::print("PriorStore: loaded {} prior(s) from '{}'\n", result.size(), priors_path_);
    return result;
}
