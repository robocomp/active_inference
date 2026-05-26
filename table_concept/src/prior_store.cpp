/*
 * prior_store.cpp
 *
 * TOML I/O for table priors and convergence checkpoints using toml++ v3.
 */

#include "prior_store.h"

#include <fstream>
#include <print>
#include <stdexcept>

#include <toml++/toml.hpp>

// ─── Constructor ─────────────────────────────────────────────────────────────

PriorStore::PriorStore(std::string priors_path, std::string checkpoint_path)
    : priors_path_(std::move(priors_path)),
      checkpoint_path_(std::move(checkpoint_path))
{}

// ─── load_priors ─────────────────────────────────────────────────────────────

std::vector<TablePrior> PriorStore::load_priors() const
{
    std::vector<TablePrior> result;

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

    const auto* tables_arr = tbl.get_as<toml::array>("tables");
    if (not tables_arr)
    {
        std::print("PriorStore: no [[tables]] array found in '{}'\n", priors_path_);
        return result;
    }

    for (const auto& elem : *tables_arr)
    {
        const auto* t = elem.as_table();
        if (not t) continue;

        TablePrior p;
        if (auto v = t->get_as<std::string>("node_name")) p.node_name  = **v;
        if (auto v = t->get_as<double>("width_m"))         p.width_m    = static_cast<float>(**v);
        if (auto v = t->get_as<double>("depth_m"))         p.depth_m    = static_cast<float>(**v);
        if (auto v = t->get_as<double>("height_m"))        p.height_m   = static_cast<float>(**v);
        if (auto v = t->get_as<double>("room_x_m"))        p.room_x_m   = static_cast<float>(**v);
        if (auto v = t->get_as<double>("room_y_m"))        p.room_y_m   = static_cast<float>(**v);
        if (auto v = t->get_as<double>("yaw_rad"))         p.yaw_rad    = static_cast<float>(**v);
        if (auto v = t->get_as<double>("sigma_pose"))      p.sigma_pose = static_cast<float>(**v);
        if (auto v = t->get_as<double>("sigma_size"))      p.sigma_size = static_cast<float>(**v);

        if (p.node_name.empty())
        {
            std::print("PriorStore: skipping table entry with no node_name\n");
            continue;
        }
        result.push_back(std::move(p));
    }

    std::print("PriorStore: loaded {} prior(s) from '{}'\n", result.size(), priors_path_);
    return result;
}

// ─── load_checkpoint ─────────────────────────────────────────────────────────

std::optional<TableCheckpoint> PriorStore::load_checkpoint(const std::string& node_name) const
{
    toml::table tbl;
    try
    {
        tbl = toml::parse_file(checkpoint_path_);
    }
    catch (...)
    {
        return std::nullopt;   // file does not exist or is malformed — not an error
    }

    const auto* ckpt_tbl = tbl.get_as<toml::table>("checkpoint");
    if (not ckpt_tbl)
        return std::nullopt;

    const auto* entry = ckpt_tbl->get_as<toml::table>(node_name);
    if (not entry)
        return std::nullopt;

    TableCheckpoint c;
    c.node_name = node_name;
    if (auto v = entry->get_as<double>("width_m"))     c.width_m     = static_cast<float>(**v);
    if (auto v = entry->get_as<double>("depth_m"))     c.depth_m     = static_cast<float>(**v);
    if (auto v = entry->get_as<double>("height_m"))    c.height_m    = static_cast<float>(**v);
    if (auto v = entry->get_as<double>("room_x_m"))    c.room_x_m    = static_cast<float>(**v);
    if (auto v = entry->get_as<double>("room_y_m"))    c.room_y_m    = static_cast<float>(**v);
    if (auto v = entry->get_as<double>("yaw_rad"))     c.yaw_rad     = static_cast<float>(**v);
    if (auto v = entry->get_as<double>("free_energy")) c.free_energy = static_cast<float>(**v);
    if (auto v = entry->get_as<bool>("model_stable"))  c.model_stable = **v;

    return c;
}

// ─── save_checkpoint ─────────────────────────────────────────────────────────

void PriorStore::save_checkpoint(const TableCheckpoint& ckpt) const
{
    // Load existing checkpoint file (if any)
    toml::table tbl;
    try
    {
        tbl = toml::parse_file(checkpoint_path_);
    }
    catch (...)
    {
        // File absent or corrupt — start fresh
    }

    // Get or create [checkpoint] table
    if (not tbl.contains("checkpoint"))
        tbl.insert("checkpoint", toml::table{});

    auto* ckpt_tbl = tbl["checkpoint"].as_table();
    if (not ckpt_tbl)
        return;

    toml::table entry;
    entry.insert_or_assign("width_m",      static_cast<double>(ckpt.width_m));
    entry.insert_or_assign("depth_m",      static_cast<double>(ckpt.depth_m));
    entry.insert_or_assign("height_m",     static_cast<double>(ckpt.height_m));
    entry.insert_or_assign("room_x_m",     static_cast<double>(ckpt.room_x_m));
    entry.insert_or_assign("room_y_m",     static_cast<double>(ckpt.room_y_m));
    entry.insert_or_assign("yaw_rad",      static_cast<double>(ckpt.yaw_rad));
    entry.insert_or_assign("free_energy",  static_cast<double>(ckpt.free_energy));
    entry.insert_or_assign("model_stable", ckpt.model_stable);

    ckpt_tbl->insert_or_assign(ckpt.node_name, std::move(entry));

    std::ofstream ofs(checkpoint_path_);
    if (not ofs.is_open())
    {
        std::print("PriorStore: cannot write checkpoint to '{}'\n", checkpoint_path_);
        return;
    }
    ofs << tbl;
    std::print("PriorStore: saved checkpoint for '{}' to '{}'\n",
               ckpt.node_name, checkpoint_path_);
}
