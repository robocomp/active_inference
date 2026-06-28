/*
 * prior_store.h
 *
 * Load chair priors from etc/object_priors.toml.
 *
 * Prior file schema (TABLE_CONCEPT.md §9.1):
 *
 *   [[chairs]]
 *   node_name  = "chair_1"
 *   seat_w_m   = 0.45
 *   seat_d_m   = 0.45
 *   seat_h_m   = 0.45
 *   back_h_m   = 0.45
 *   room_x_m   = 2.0
 *   room_y_m   = 1.0
 *   yaw_rad    = 0.0
 *   sigma_pose = 0.10
 *   sigma_size = 0.08
 */

#pragma once

#include <string>
#include <vector>

// ─── Data structs ─────────────────────────────────────────────────────────────


namespace rc {
struct ChairPrior
{
    std::string node_name;
    float seat_w     = 0.45f;
    float seat_d     = 0.45f;
    float seat_h     = 0.45f;
    float back_h     = 0.45f;
    float room_x_m   = 0.0f;
    float room_y_m   = 0.0f;
    float yaw_rad    = 0.0f;
    float sigma_pose = 0.1f;
    float sigma_size = 0.08f;
};

// ─── PriorStore ──────────────────────────────────────────────────────────────

class PriorStore
{
public:
    /**
     * @param priors_path Path to object_priors.toml (contains [[chairs]] entries).
     */
    explicit PriorStore(std::string priors_path);

    /** Load all [[chairs]] entries from the priors file. */
    std::vector<ChairPrior> load_priors() const;

private:
    std::string priors_path_;
};

}  // namespace rc
