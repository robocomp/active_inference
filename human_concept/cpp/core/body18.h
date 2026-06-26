// body18.h
// ZED BODY_18 keypoint conventions: index map, names, and skeleton edges.
// Ported from human_kinematic_model.py (KP, KP18_NAMES) and
// visualization_compare.py (BODY18_EDGES).
#pragma once

#include <array>
#include <string_view>
#include <utility>

namespace rc::human
{
// ---------------- keypoint indices ----------------
namespace KP
{
inline constexpr int NOSE       = 0;
inline constexpr int NECK       = 1;
inline constexpr int R_SHOULDER = 2;
inline constexpr int R_ELBOW    = 3;
inline constexpr int R_WRIST    = 4;
inline constexpr int L_SHOULDER = 5;
inline constexpr int L_ELBOW    = 6;
inline constexpr int L_WRIST    = 7;
inline constexpr int R_HIP      = 8;
inline constexpr int R_KNEE     = 9;
inline constexpr int R_ANKLE    = 10;
inline constexpr int L_HIP      = 11;
inline constexpr int L_KNEE     = 12;
inline constexpr int L_ANKLE    = 13;
inline constexpr int R_EYE      = 14;
inline constexpr int L_EYE      = 15;
inline constexpr int R_EAR      = 16;
inline constexpr int L_EAR      = 17;
}  // namespace KP

inline constexpr int NUM_KP = 18;

// ---------------- keypoint names ----------------
inline constexpr std::array<std::string_view, NUM_KP> KP18_NAMES = {
    "nose", "neck",
    "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "r_eye", "l_eye", "r_ear", "l_ear"
};

// ---------------- skeleton edges (for visualization) ----------------
inline constexpr std::array<std::pair<int, int>, 17> BODY18_EDGES = {{
    {1, 0},
    {0, 14}, {14, 16},
    {0, 15}, {15, 17},
    {1, 2}, {2, 3}, {3, 4},
    {1, 5}, {5, 6}, {6, 7},
    {2, 8}, {8, 9}, {9, 10},
    {5, 11}, {11, 12}, {12, 13},
}};

}  // namespace rc::human
