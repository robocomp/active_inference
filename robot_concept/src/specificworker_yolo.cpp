#include "specificworker.h"

#include "scene_processor.h"
#include "yolo_processor.h"

#include <QImage>
#include <QPixmap>

#include <chrono>

void SpecificWorker::update_yolo_tab_display(const RoboCompCameraRGBDSimple::TRGBD& rgbd,
	                                          const std::vector<SegDetection>& detections)
{
	static auto last_display_update = std::chrono::steady_clock::time_point{};
	static float display_fps = 0.f;

	if (yolo_image_label_ == nullptr || rgbd.image.width == 0 || rgbd.image.height == 0
		|| !custom_widget_yolo.isVisible() || !yolo_processor)
		return;

	const cv::Mat rgb_frame(rgbd.image.height, rgbd.image.width, CV_8UC3,
		const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(rgbd.image.image.data())));
	const cv::Mat masked_rgb_frame = yolo_processor->apply_tray_mask(rgb_frame);
	cv::Mat yolo_canvas = yolo_processor->compose_detection_canvas(masked_rgb_frame, detections);
	if (scene_processor)
		scene_processor->overlay_room_polygon_on_canvas(yolo_canvas, rgbd);
	cv::Mat yolo_canvas_rgb;
	cv::cvtColor(yolo_canvas, yolo_canvas_rgb, cv::COLOR_BGR2RGB);
	QImage yolo_qimg(yolo_canvas_rgb.data,
		yolo_canvas_rgb.cols,
		yolo_canvas_rgb.rows,
		static_cast<int>(yolo_canvas_rgb.step),
		QImage::Format_RGB888);
	QPixmap yolo_pix = QPixmap::fromImage(yolo_qimg, Qt::NoFormatConversion);
	yolo_image_label_->setPixmap(yolo_pix.scaled(yolo_image_label_->size(), Qt::KeepAspectRatio, Qt::FastTransformation));

	if (yolo_fps_label_ != nullptr)
	{
		const auto now = std::chrono::steady_clock::now();
		if (last_display_update != std::chrono::steady_clock::time_point{})
		{
			const auto dt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_display_update).count();
			if (dt_ms > 0)
			{
				const float inst_fps = 1000.0f / static_cast<float>(dt_ms);
				display_fps = (display_fps > 0.f) ? (0.85f * display_fps + 0.15f * inst_fps) : inst_fps;
			}
		}
		last_display_update = now;
		yolo_fps_label_->setText(QString("YOLO display FPS: %1").arg(display_fps, 0, 'f', 1));
	}
}