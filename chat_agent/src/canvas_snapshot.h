#pragma once
/*
 * canvas_snapshot.h — what a canvas script may know: the robot's world as snapshot.json in the run dir.
 *
 * A canvas script (canvas/rcanvas.py documents the schema it reads) never touches the graph: before every run the
 * session writes this snapshot, on the MAIN thread, with the main thread's InnerEigenAPI (CLAUDE.md ts==0 ownership).
 *
 *   room     the current room's name + delimiting_polygon_x/y (room frame)           null when unknown
 *   robot    `body` in the room frame: x, y, yaw (of the body frame's +x), heading = the direction it FACES
 *            (body +y, ROBOT_GEOMETRY.md ⇒ heading = yaw + 90°), radius                null when unresolved
 *   motion   robot_current_{advance,side,angular}_speed from the robot node           null when absent
 *   doors    each door of the current room: centre, yaw, width_m, ends a/b (the aperture's local +x runs along the
 *            wall — door_belief.h), door_open_prob
 *   objects  every object of the current room (Digest::objects): centre + yaw in the room frame, sizes, class
 *   target   the executing affordance's goal (room frame), as the STATE's target line reads it
 *   mission  the controller mission's title / status / progress line
 *   canvas   the pixel size of the panel the drawing will fill
 *
 * Every get_transformation_matrix optional is checked; what does not resolve is written as null, never guessed.
 * QJsonDocument formats numbers itself (locale-free), so the CLAUDE.md comma-decimal trap does not apply here.
 */

#include <memory>

#include <QJsonObject>
#include <QString>

#include <dsr/api/dsr_api.h>
#include <dsr/api/dsr_inner_eigen_api.h>

#include "../../common/mission/mission_progress.h"
#include "world_digest.h"

namespace rc::chat
{

struct SnapshotSize { int w = 800, h = 600; };

[[nodiscard]] QJsonObject build_canvas_snapshot(DSR::DSRGraph& G, DSR::InnerEigenAPI& inner, const Digest& digest,
                                                const rc::mission::Progress& mission, double robot_radius_m,
                                                SnapshotSize canvas);

/// Writes `snapshot` as <dir>/snapshot.json (atomically: tmp + rename). False on an I/O error.
bool write_canvas_snapshot(const QString& dir, const QJsonObject& snapshot);

}   // namespace rc::chat
