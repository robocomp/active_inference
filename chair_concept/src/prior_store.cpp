/*
 * prior_store.cpp
 *
 * TOML I/O for chair priors using toml++ v3.
 */

#include "prior_store.h"

#include <print>
#include <utility>

#include <toml++/toml.hpp>

// ─── Constructor ─────────────────────────────────────────────────────────────


namespace rc {
PriorStore::PriorStore(std::string priors_path)
    : priors_path_(std::move(priors_path))
{}

// ─── load_priors ─────────────────────────────────────────────────────────────

std::vector<ChairPrior> PriorStore::load_priors() const
{
    std::vector<ChairPrior> result;

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

    const auto* tables_arr = tbl.get_as<toml::array>("chairs");
    if (not tables_arr)
    {
        std::print("PriorStore: no [[chairs]] array found in '{}'\n", priors_path_);
        return result;
    }

    for (const auto& elem : *tables_arr)
    {
        const auto* t = elem.as_table();
        if (not t) continue;

        ChairPrior p;
        if (auto v = t->get_as<std::string>("node_name")) p.node_name  = **v;
        if (auto v = t->get_as<double>("seat_w_m"))        p.seat_w     = static_cast<float>(**v);
        if (auto v = t->get_as<double>("seat_d_m"))        p.seat_d     = static_cast<float>(**v);
        if (auto v = t->get_as<double>("seat_h_m"))        p.seat_h     = static_cast<float>(**v);
        if (auto v = t->get_as<double>("back_h_m"))        p.back_h     = static_cast<float>(**v);
        if (auto v = t->get_as<double>("room_x_m"))        p.room_x_m   = static_cast<float>(**v);
        if (auto v = t->get_as<double>("room_y_m"))        p.room_y_m   = static_cast<float>(**v);
        if (auto v = t->get_as<double>("yaw_rad"))         p.yaw_rad    = static_cast<float>(**v);
        if (auto v = t->get_as<double>("sigma_pose"))      p.sigma_pose = static_cast<float>(**v);
        if (auto v = t->get_as<double>("sigma_size"))      p.sigma_size = static_cast<float>(**v);

        if (p.node_name.empty())
        {
            std::print("PriorStore: skipping chair entry with no node_name\n");
            continue;
        }
        result.push_back(std::move(p));
    }

    std::print("PriorStore: loaded {} prior(s) from '{}'\n", result.size(), priors_path_);
    return result;
}

}  // namespace rc
