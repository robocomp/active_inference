/*
 * prior_store.h
 *
 * Load bottle priors from etc/object_priors.toml and save/load convergence
 * checkpoints alongside the priors file. Ported from table_concept; geometry
 * is a vertical cylinder (radius + height, no yaw).
 *
 * Prior file schema:
 *
 *   [[bottles]]
 *   node_name  = "bottle_1"
 *   radius_m   = 0.035
 *   height_m   = 0.20
 *   room_x_m   = 0.40
 *   room_y_m   = 0.00
 *   sigma_pose = 0.10
 *   sigma_size = 0.03
 *
 * Checkpoint schema:
 *
 *   [checkpoint.bottle_1]
 *   radius_m     = 0.034
 *   height_m     = 0.205
 *   room_x_m     = 0.41
 *   room_y_m     = 0.01
 *   room_z_m     = 0.85
 *   free_energy  = 0.014
 *   model_stable = true
 */

#pragma once

#include <optional>
#include <string>
#include <vector>

// ─── Data structs ─────────────────────────────────────────────────────────────

struct BottlePrior
{
    std::string node_name;
    float radius_m   = 0.035f;
    float height_m   = 0.20f;
    float room_x_m   = 0.0f;
    float room_y_m   = 0.0f;
    float room_z_m   = 0.85f;   // cylinder centre Z (table_top + height/2)
    float sigma_pose = 0.10f;
    float sigma_size = 0.03f;
};

struct BottleCheckpoint
{
    std::string node_name;
    float radius_m    = 0.0f;
    float height_m    = 0.0f;
    float room_x_m    = 0.0f;
    float room_y_m    = 0.0f;
    float room_z_m    = 0.0f;
    float free_energy = -1.0f;
    bool  model_stable = false;
};

// ─── PriorStore ──────────────────────────────────────────────────────────────

class PriorStore
{
public:
    /**
     * @param priors_path     Path to object_priors.toml (contains [[bottles]] entries).
     * @param checkpoint_path Path to the checkpoint TOML file (may not exist yet).
     */
    PriorStore(std::string priors_path, std::string checkpoint_path);

    /** Load all [[bottles]] entries from the priors file. */
    std::vector<BottlePrior> load_priors() const;

    /** Return the checkpoint for a specific node, if it exists. */
    std::optional<BottleCheckpoint> load_checkpoint(const std::string& node_name) const;

    /**
     * Write (or overwrite) the [checkpoint.<node_name>] section in the
     * checkpoint file. Creates the file if it does not exist.
     */
    void save_checkpoint(const BottleCheckpoint& ckpt) const;

private:
    std::string priors_path_;
    std::string checkpoint_path_;
};
