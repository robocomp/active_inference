/*
 * prior_store.h
 *
 * Load table priors from etc/object_priors.toml.
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
 */

#pragma once

#include <string>
#include <vector>

// ─── Data structs ─────────────────────────────────────────────────────────────


namespace rc {
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

// ─── PriorStore ──────────────────────────────────────────────────────────────

class PriorStore
{
public:
    /**
     * @param priors_path Path to object_priors.toml (contains [[tables]] entries).
     */
    explicit PriorStore(std::string priors_path);

    /** Load all [[tables]] entries from the priors file. */
    std::vector<TablePrior> load_priors() const;

private:
    std::string priors_path_;
};

}  // namespace rc
