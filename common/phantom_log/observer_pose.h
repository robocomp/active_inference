#pragma once
/*
 * common/phantom_log/observer_pose.h — stamp a PhantomEvent with WHERE THE ROBOT WAS LOOKING FROM.
 *
 * Kept out of phantom_log.h on purpose: that header is DSR-free (a plain CSV writer, usable from a test or an
 * offline tool), and this one needs InnerEigenAPI. Same split as birth_surprise/residual_field_reader.h.
 *
 * WHY THE OBSERVER POSE IS PART OF A PHANTOM RECORD AT ALL. A phantom is a CLASSIFIER failure, and classifier
 * failures are VIEWPOINT-dependent — a radiator reads as a chair from some angles and from no others. So the
 * false-alarm field the log is being collected for is keyed on (world cell × bearing). Keyed on place alone it
 * would learn "nothing may be born here", suppressing a genuine object later placed at that spot from EVERY
 * direction. All seven agents had worked this out and all seven had written the same ten lines to record it.
 *
 * MAIN-THREAD ONLY: the lookup asks for ts == 0, and the InnerEigenAPI ts==0 cache is unlocked (CLAUDE.md).
 */

#include <cmath>

#include <dsr/api/dsr_inner_eigen_api.h>

#include "phantom_log.h"   // rc::history::PhantomEvent

namespace rc::history
{

// Fill e.robot_* / e.view_bearing / e.range_m from the robot's current room-frame pose. `x, y` is the
// instance centre, already in e. A missing transform leaves the fields at zero and is NOT an error — the
// room node can legitimately be absent (a death recorded while localisation is down still wants recording).
inline void note_observer(PhantomEvent& e, DSR::InnerEigenAPI* inner_eigen, float x, float y)
{
    if (inner_eigen == nullptr)
        return;
    const auto rtb = inner_eigen->get_transformation_matrix("room", "body", 0);
    if (not rtb.has_value())
        return;                          // ALWAYS check the optional — see CLAUDE.md (bad_optional_access)
    const auto& Tm = rtb.value();
    e.robot_x      = static_cast<float>(Tm(0, 3));
    e.robot_y      = static_cast<float>(Tm(1, 3));
    e.robot_yaw    = std::atan2(static_cast<float>(Tm(1, 0)), static_cast<float>(Tm(0, 0)));
    e.view_bearing = std::atan2(e.robot_y - y, e.robot_x - x);   // instance → camera, room frame
    e.range_m      = std::hypot(e.robot_x - x, e.robot_y - y);
}

}  // namespace rc::history
