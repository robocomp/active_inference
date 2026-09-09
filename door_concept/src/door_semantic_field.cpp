#include "door_semantic_field.h"

#include <algorithm>
#include <cmath>

namespace rc
{

bool SemanticProbField::refresh(DSR::DSRGraph* G, int class_id)
{
    valid_ = false;
    if (G == nullptr)
        return false;

    const auto node = G->get_node("semantic");
    if (not node.has_value())
        return false;   // retina not publishing (or not up) — leave every caller unchanged

    // Type-attributed access throughout (CLAUDE.md): a typo or a type mismatch is a compile error here,
    // not a runtime throw on a live agent.
    const auto probs_o = G->get_attrib_by_name<semantic_class_probs_att>(node.value());
    const auto ids_o   = G->get_attrib_by_name<semantic_prob_class_ids_att>(node.value());
    const auto pw_o    = G->get_attrib_by_name<semantic_prob_width_att>(node.value());
    const auto ph_o    = G->get_attrib_by_name<semantic_prob_height_att>(node.value());
    const auto fw_o    = G->get_attrib_by_name<semantic_width_att>(node.value());
    const auto fh_o    = G->get_attrib_by_name<semantic_height_att>(node.value());
    if (not (probs_o.has_value() and ids_o.has_value() and pw_o.has_value() and ph_o.has_value()
             and fw_o.has_value() and fh_o.has_value()))
        return false;   // ungraded model: the plain export publishes no posterior at all

    const std::vector<float>& probs = probs_o.value().get();
    const std::vector<float>& ids   = ids_o.value().get();
    const int pw = pw_o.value(), ph = ph_o.value();
    const int fw = fw_o.value(), fh = fh_o.value();
    if (pw <= 0 or ph <= 0 or fw <= 0 or fh <= 0 or ids.empty())
        return false;

    // Which plane is ours. ★The ids ride WITH the data rather than being configured: a re-exported model
    // with a different channel order would otherwise be read silently out of the wrong plane, and the
    // resulting numbers would look perfectly reasonable.
    int channel = -1;
    for (std::size_t i = 0; i < ids.size(); ++i)
        if (static_cast<int>(std::lround(ids[i])) == class_id)
        {
            channel = static_cast<int>(i);
            break;
        }
    if (channel < 0)
        return false;   // this model exposes no posterior for our class

    const std::size_t plane_n = static_cast<std::size_t>(pw) * static_cast<std::size_t>(ph);
    const std::size_t offset  = static_cast<std::size_t>(channel) * plane_n;
    // A short buffer means the producer and this reader disagree about the layout. Refuse rather than
    // read past the end or, worse, read a neighbouring class's plane as our own.
    if (probs.size() < offset + plane_n)
        return false;

    plane_.assign(probs.begin() + static_cast<std::ptrdiff_t>(offset),
                  probs.begin() + static_cast<std::ptrdiff_t>(offset + plane_n));

    // Background = the whole frame's mean door-ness. NaN cells are "not looked at" (a 360 strip the
    // scheduler skipped) and are excluded rather than counted as zero, which would depress the background
    // and make every contour look falsely contrasty.
    double sum = 0.0;
    int    n   = 0;
    for (const float v : plane_)
        if (std::isfinite(v)) { sum += v; ++n; }
    if (n == 0)
        return false;
    bg_mean_ = static_cast<float>(sum / n);

    pw_ = pw; ph_ = ph; frame_w_ = fw; frame_h_ = fh;
    if (const auto ts = G->get_attrib_by_name<semantic_timestamp_ms_att>(node.value()); ts.has_value())
        stamp_ms_ = ts.value();
    valid_ = true;
    return true;
}

float SemanticProbField::at(int u, int v) const noexcept
{
    if (not valid_ or u < 0 or v < 0 or u >= frame_w_ or v >= frame_h_)
        return -1.0f;
    // Frame pixel → native cell, cell centres, the inverse of the producer's mapping. Nearest rather than
    // bilinear on purpose: the model has no opinion between its own cells, and interpolating invents one.
    const int cx = std::clamp(static_cast<int>((static_cast<float>(u) + 0.5f) * pw_
                                               / static_cast<float>(frame_w_)), 0, pw_ - 1);
    const int cy = std::clamp(static_cast<int>((static_cast<float>(v) + 0.5f) * ph_
                                               / static_cast<float>(frame_h_)), 0, ph_ - 1);
    return plane_[static_cast<std::size_t>(cy) * static_cast<std::size_t>(pw_)
                  + static_cast<std::size_t>(cx)];
}

}   // namespace rc
