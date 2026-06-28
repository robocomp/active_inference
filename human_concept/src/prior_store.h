/*
 * prior_store.h
 *
 * Optional static person priors from etc/object_priors.toml. Unlike bottle/table,
 * human_concept primarily scaffolds "person_N" nodes on demand from live skeleton
 * track ids; priors are an optional seed (e.g. an expected greeter location).
 *
 *   [[humans]]
 *   node_name = "person_1"
 *   track_id  = 0
 *   room_x_m  = 0.0
 *   room_y_m  = 0.0
 *   room_z_m  = 0.0
 */

#pragma once

#include <string>
#include <vector>

namespace rc {

struct HumanPrior
{
    std::string node_name;
    int   track_id = 0;
    float room_x_m = 0.0f;
    float room_y_m = 0.0f;
    float room_z_m = 0.0f;   // pelvis/root height (≈0; the model fixes pelvis y=0 in its own frame)
};

class PriorStore
{
public:
    explicit PriorStore(std::string priors_path);
    std::vector<HumanPrior> load_priors() const;

private:
    std::string priors_path_;
};

}  // namespace rc
