/*  calibration_viewer.h — a window showing how the self-calibration is going.
 *
 *  One row per parameter: its current value with 1-sigma, whether this window actually TAUGHT it,
 *  and a trace of how it has moved.
 *
 *  ★ THE "informed" LAMP IS THE POINT, not decoration. A parameter's VALUE cannot distinguish
 *  "converged" from "never asked": both look like a number that has stopped moving. Only the
 *  precision separates them, and both cases have been seen live — eps_yaw sat near zero with a
 *  WIDENING sigma through a turn-heavy run, and k_omega did the same through a straight-heavy one.
 *  Showing the value alone would have made each look like a settled measurement.
 */
#pragma once

#include "calibration_estimator.h"
#include "camera_calibration.h"
#include "timeseries_plot.h"

#include <QDialog>
#include <QLabel>
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <functional>

class QHBoxLayout;

namespace rc
{
    class CalibrationViewer : public QDialog
    {
        Q_OBJECT
    public:
        explicit CalibrationViewer(QWidget* parent = nullptr);

        /// Called once per localiser cycle. Cheap when the window is hidden: the traces are only
        /// appended while visible, so an unopened viewer costs nothing but the call.
        /// What "Reset" does. Set by the owner; the window itself knows nothing about the estimator.
        void set_reset_handler(std::function<void()> h) { on_reset_ = std::move(h); }

        /// Declare which cameras get a column, in display order, and which one drives the pose.
        /// Called once by the owner right after construction.
        ///
        /// ★ A CAMERA DECLARED HERE BUT NEVER UPDATED RENDERS AS AN EXPLICIT "no evidence" COLUMN.
        ///   That is the whole reason the list is declared instead of columns appearing when data
        ///   first arrives: a column that only exists once it has numbers cannot say "nothing has
        ///   been measured for this camera", and silence would then look exactly like a reading.
        void set_cameras(const std::vector<std::string>& cameras, const std::string& driving = "");

        /// The CAMERA block for ONE camera: a separate estimator with its own evidence, its own
        /// `informed` flags and its own conditioning. Separate from the motion block because the two
        /// are fed by different streams and share no covariate — see camera_calibration.h. `pairs` is
        /// corner pairings, not driving episodes, and the label says so: two counts called
        /// "episodes" would be a quiet lie.
        ///
        /// ★ `camera` SELECTS A COLUMN, it does not relabel a shared one. These parameters describe
        ///   ONE mount (body→zed at 1.08 m, body→ricoh at 1.42 m), so one block whose caption
        ///   followed whichever camera reported last showed one camera's numbers under the other's
        ///   name on every alternate update. An unknown name gets its own column rather than being
        ///   dropped.
        void update_camera(const Eigen::Matrix<float, rc::camcal::P_COUNT, 1>& value,
                           const Eigen::Matrix<float, rc::camcal::P_COUNT, 1>& sigma,
                           int informed_mask, float condition, long pairs,
                           const std::string& camera = "");

        /// The sensor triangle: camera-vs-camera disagreement with the LiDAR's own corner error
        /// CANCELLED. It is the only line in this window that needs no ground truth, which is why
        /// it is the one that still works on the real robot.
        void update_loop_closure(double du_deg, double dv_deg, double sd_du, double sd_dv, long n);


        void update_values(const Eigen::Matrix<float, rc::calib::P_COUNT, 1>& value,
                           const Eigen::Matrix<float, rc::calib::P_COUNT, 1>& sigma,
                           int informed_mask, float condition, int episodes);

    private:
        struct Row
        {
            QLabel* name = nullptr;
            QLabel* value = nullptr;
            QLabel* lamp = nullptr;
            TimeSeriesPlot* plot = nullptr;
            float scale = 1.f;          // display scaling (rad -> deg, fraction -> %)
            const char* unit = "";
            // What this parameter is and what motion identifies it — shown on hover over the name,
            // the value and the lamp. Static string literals from the spec table; not owned.
            const char* why = "";
        };
        /// One camera = one column. Held by pointer so the Rows' addresses survive the vector
        /// growing when a camera nobody declared turns up.
        struct CameraBlock
        {
            std::string name;
            bool        driving = false;      ///< this is the camera the localiser actually uses
            bool        has_evidence = false; ///< has update_camera() ever been called for it
            QLabel*     title  = nullptr;
            QLabel*     status = nullptr;     ///< pairs + conditioning, or the "no evidence" state
            std::array<Row, rc::camcal::P_COUNT> rows{};
        };

        std::array<Row, rc::calib::P_COUNT>  rows_{};
        std::vector<std::unique_ptr<CameraBlock>> cam_blocks_;
        QHBoxLayout* cam_columns_ = nullptr;   ///< the side-by-side strip the columns live in
        QLabel* summary_ = nullptr;
        QLabel* loop_label_ = nullptr;
        std::function<void()> on_reset_;

        /// The column for `camera`, creating one if the name is new. Empty name = the first column.
        CameraBlock* block_for(const std::string& camera);
        /// Build one column's widgets and append it to `cam_columns_`.
        void build_camera_column(CameraBlock& b);
    };
}
