/*
 * prior_store.h
 *
 * Load bottle priors from etc/object_priors.toml. Ported from table_concept;
 * geometry is a vertical cylinder (radius + height, no yaw).
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
 */

#pragma once

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

// ─── PriorStore ──────────────────────────────────────────────────────────────

class PriorStore
{
public:
    /** @param priors_path Path to object_priors.toml (contains [[bottles]] entries). */
    explicit PriorStore(std::string priors_path);

    /** Load all [[bottles]] entries from the priors file. */
    std::vector<BottlePrior> load_priors() const;

private:
    std::string priors_path_;
};
