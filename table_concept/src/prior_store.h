/*
 * prior_store.h
 *
 * Load table priors from etc/object_priors.toml and save/load
 * convergence checkpoints alongside the priors file.
 *
 * Prior file schema (TABLE_CONCEPT.md §9.1):
 *
 *   [[tables]]
 *   node_name  = "bootstrap_table_1"
 *   width_m    = 2.0
 *   depth_m    = 0.8
 *   height_m   = 0.75
 *   room_x_m   = 3.5
 *   room_y_m   = 1.2
 *   yaw_rad    = 0.0
 *   sigma_pose = 0.1
 *   sigma_size = 0.15
 *
 * Checkpoint schema (TABLE_CONCEPT.md §9.2):
 *
 *   [checkpoint.bootstrap_table_1]
 *   width_m      = 2.03
 *   depth_m      = 0.81
 *   height_m     = 0.74
 *   room_x_m     = 3.52
 *   room_y_m     = 1.18
 *   yaw_rad      = 0.02
 *   free_energy  = 0.014
 *   model_stable = true
 */

#pragma once

#include <optional>
#include <string>
#include <vector>

// ─── Data structs ─────────────────────────────────────────────────────────────

struct TablePrior
{
    std::string node_name;
    float width_m    = 1.0f;
    float depth_m    = 0.6f;
    float height_m   = 0.75f;
    float room_x_m   = 0.0f;
    float room_y_m   = 0.0f;
    float yaw_rad    = 0.0f;
    float sigma_pose = 0.1f;
    float sigma_size = 0.15f;
};

struct TableCheckpoint
{
    std::string node_name;
    float width_m    = 0.0f;
    float depth_m    = 0.0f;
    float height_m   = 0.0f;
    float room_x_m   = 0.0f;
    float room_y_m   = 0.0f;
    float yaw_rad    = 0.0f;
    float free_energy = -1.0f;
    bool  model_stable = false;
};

// ─── PriorStore ──────────────────────────────────────────────────────────────

class PriorStore
{
public:
    /**
     * @param priors_path     Path to object_priors.toml (contains [[tables]] entries).
     * @param checkpoint_path Path to the checkpoint TOML file (may not exist yet).
     */
    PriorStore(std::string priors_path, std::string checkpoint_path);

    /** Load all [[tables]] entries from the priors file. */
    std::vector<TablePrior> load_priors() const;

    /** Return the checkpoint for a specific node, if it exists. */
    std::optional<TableCheckpoint> load_checkpoint(const std::string& node_name) const;

    /**
     * Write (or overwrite) the [checkpoint.<node_name>] section in the
     * checkpoint file.  Creates the file if it does not exist.
     */
    void save_checkpoint(const TableCheckpoint& ckpt) const;

private:
    std::string priors_path_;
    std::string checkpoint_path_;
};
