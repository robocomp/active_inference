/*
 * specificworker_dashboard.cpp -- the sensor_calibration dashboard window (separate TU, as bottle_concept).
 * DISPLAY ONLY: copies of data the agent already holds go into sc::dash::DashboardModel (pure, tested) and a thin
 * QPainter view (dashboard_view.h) draws it. Nothing here reads back into the harvester or any estimate.
 * Main thread only; the view is repainted at <= 5 Hz from compute(), never per pair. With [Dashboard] show = false
 * dash_model_ stays null and every feed returns at once.
 */
#include "specificworker.h"

#include "../../common/dashboard/window_geometry.h"   // rc::dashboard:: (SHARED)
#include "dashboard_view.h"


namespace
{
std::int64_t wall_ms()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}
}

void SpecificWorker::build_dashboard()
{
    if (not dashboard_show_)
        return;
    const sc::dash::Priors pr{harvester_cfg_.prior_sigma_xy * harvester_cfg_.prior_sigma_xy,
                              harvester_cfg_.prior_sigma_yaw * harvester_cfg_.prior_sigma_yaw};
    dash_model_ = std::make_unique<sc::dash::DashboardModel>(pr);
    auto* v = new DashboardView(*dash_model_);
    v->setWindowTitle(QStringLiteral("sensor_calibration - trajectory fragments"));
    dash_view_ = v;
    dashboard_window_ = v;
    rc::dashboard::restore_window_geometry(dashboard_window_, QStringLiteral("sensor_calibration"), QStringLiteral("DashboardWindow"), QSize(760, 470));
    dashboard_window_->show();
}

void SpecificWorker::save_dashboard_geometry() const
{
    rc::dashboard::save_window_geometry(dashboard_window_, QStringLiteral("sensor_calibration"), QStringLiteral("DashboardWindow"));
}

void SpecificWorker::feed_dashboard_keyframe(bool captured, bool stalled)
{
    if (dash_model_)
        dash_model_->note_keyframe(captured, stalled);
}

void SpecificWorker::feed_dashboard_pairs(const std::vector<sc::calib::Pair>& pairs)
{
    if (dash_model_)
        for (const auto& p : pairs)
            dash_model_->add_pair(p);
}

void SpecificWorker::refresh_dashboard()
{
    if (not dash_model_ or not inputs_)
        return;
    const auto now = std::chrono::steady_clock::now();
    if (dash_last_tp_.time_since_epoch().count() != 0 and std::chrono::duration<double>(now - dash_last_tp_).count() < 0.2)
        return;
    dash_last_tp_ = now;
    dash_model_->sample_stall(wall_ms(), inputs_->any_stalled());

    // Rates from the cumulative arrival counts over >= 1 s windows (status_line()'s own windows are left alone).
    const auto states = inputs_->stream_states();
    const double dt = dash_rate_tp_.time_since_epoch().count() == 0 ? 0.0 : std::chrono::duration<double>(now - dash_rate_tp_).count();
    if (dash_rate_tp_.time_since_epoch().count() == 0 or dt >= 1.0)
    {
        for (const auto& s : states)
        {
            const auto it = dash_rate_totals_.find(s.key);
            dash_rates_[s.key] = (it != dash_rate_totals_.end() and dt > 0.0) ? double(s.total - it->second) / dt : 0.0;
            dash_rate_totals_[s.key] = s.total;
        }
        dash_rate_tp_ = now;
    }
    std::vector<DashStreamRow> rows;
    for (const auto& s : states)
        rows.push_back({s.key, dash_rates_[s.key], s.stalled, s.ever_seen});
    dash_view_->set_streams(std::move(rows));
    dash_view_->update();
}
