#pragma once

// Which strips of the 360 panorama this frame is looking at.
//
// ★WHY IT IS SHARED STATE AND NOT A COUNTER INSIDE EACH STAGE. Two stages now look at the panorama —
// YOLO-seg and (next) the ADE20K semantic pass — and they MUST look at the same strip on the same
// frame. Give each its own round-robin counter and they start in phase and stay in phase only until
// one of them is skipped, disabled, or decimated once; from then on the semantic labels describe a
// different 60° of the world than the instance masks do, every frame, silently. Nothing would crash
// and nothing would look wrong in either stage's own output.
//
// So the schedule is decided ONCE per frame, by whoever runs first, and read by everyone else.
//
// Single-threaded by construction: all panorama stages run in sequence on the one ricoh worker thread.

#include <algorithm>
#include <vector>

namespace rc
{

class StripSchedule
{
public:
    // Advance to the next window and return it. Called by the FIRST stage of the frame, once.
    // per_frame <= 0 or >= n_strips ⇒ every strip (the original behaviour), and the rotation is idle.
    const std::vector<int>& advance(int n_strips, int per_frame)
    {
        current_.clear();
        if (n_strips <= 0)
            return current_;
        if (per_frame <= 0 or per_frame >= n_strips)
            return current_;   // empty == all, the convention Detection360Config already uses
        current_.reserve(static_cast<std::size_t>(per_frame));
        for (int k = 0; k < per_frame; ++k)
            current_.push_back((next_ + k) % n_strips);
        next_ = (next_ + per_frame) % n_strips;
        return current_;
    }

    // What this frame is looking at. Empty = all strips.
    [[nodiscard]] const std::vector<int>& current() const { return current_; }

private:
    std::vector<int> current_;
    int              next_ = 0;
};

}   // namespace rc
