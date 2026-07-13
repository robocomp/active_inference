/*
 *  Media-plane "View data" viewers for robot_concept.
 *
 *  Since raw sensor streams left the DSR graph for the zero-copy DDS media plane, the old dsr_gui
 *  node widgets (which read blobs off the graph node) show nothing. dsr_gui now forwards a
 *  right-click "View data" as GraphViewer::view_data_signal; robot_concept catches it and opens one
 *  of these instead:
 *    - zed   -> ImageStreamViewer   (RGB image window)
 *    - ricoh -> Image360Viewer      (360 panorama window)
 *    - laser -> LidarStreamViewer   (3D OpenGL point cloud, rc::viewers::GLPointCloudViewer)
 *
 *  DRIVEN BY DATA ARRIVAL, NOT A TIMER. Each viewer runs a per-viewer worker thread that BLOCKS on
 *  the subscriber's wait_and_poll() and wakes the instant a frame arrives; it deep-copies the
 *  payload (image -> QImage, cloud -> QVector3D vector) OFF the GUI thread, then posts it to the GUI
 *  thread with QMetaObject::invokeMethod(QueuedConnection). So the window refreshes at exactly the
 *  stream's arrival rate — no over-polling, no dropped frames — which matters for an occasionally
 *  opened debugging tool. No cv::Mat/QImage buffer is shared across the boundary (deep copy first).
 *
 *  Non-Q_OBJECT (invokeMethod uses lambdas), so no MOC pass is needed.
 */
#ifndef ROBOT_CONCEPT_MEDIA_STREAM_VIEWERS_H
#define ROBOT_CONCEPT_MEDIA_STREAM_VIEWERS_H

#include <QWidget>
#include <QLabel>
#include <QVBoxLayout>
#include <QImage>
#include <QPixmap>
#include <QPainter>
#include <QString>
#include <QVector3D>
#include <QMetaObject>
#include <QDebug>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stop_token>
#include <thread>
#include <vector>

#include "../../common/media_transport/media_transport.h"
#include "../../common/viewers/gl_point_cloud_viewer.h"
#include "../../common/viewers/gl_imu_viewer.h"

namespace rc::viewers
{

// Deep-copy a raw media image buffer into a QImage. Format enum values coincide for ImageFrame and
// Image360Frame (BGR8=0, RGB8=1, GRAY8=2); depth formats are normalised to an 8-bit grey ramp.
inline QImage qimage_from_media(const std::uint8_t *data, int w, int h, int step, std::uint32_t fmt)
{
	if(data == nullptr or w <= 0 or h <= 0)
		return {};
	switch(fmt)
	{
		case rc::media::FORMAT_RGB8:  return QImage(data, w, h, step, QImage::Format_RGB888).copy();
		case rc::media::FORMAT_BGR8:  return QImage(data, w, h, step, QImage::Format_RGB888).rgbSwapped();
		case rc::media::FORMAT_GRAY8: return QImage(data, w, h, step, QImage::Format_Grayscale8).copy();
		case rc::media::FORMAT_Z16:
		{
			const int n = w * h;
			const auto *d16 = reinterpret_cast<const std::uint16_t *>(data);
			std::uint16_t lo = std::numeric_limits<std::uint16_t>::max(), hi = 0;
			for(int i = 0; i < n; ++i)
				if(const std::uint16_t v = d16[i]; v != 0) { lo = std::min(lo, v); hi = std::max(hi, v); }
			QImage img(w, h, QImage::Format_Grayscale8);
			const float scale = (hi > lo) ? 255.0f / static_cast<float>(hi - lo) : 0.0f;
			for(int y = 0; y < h; ++y)
			{
				auto *row = img.scanLine(y);
				for(int x = 0; x < w; ++x)
				{
					const std::uint16_t v = d16[y * w + x];
					row[x] = (v == 0) ? 0 : static_cast<std::uint8_t>(std::clamp((v - lo) * scale, 0.0f, 255.0f));
				}
			}
			return img;
		}
		default: return {};
	}
}

// ── Image / 360-panorama viewer ──────────────────────────────────────────────
// Templated so ZED (MediaSubscriber/ImageFrame) and Ricoh-360 (Image360Subscriber/Image360Frame)
// share one implementation. The worker thread blocks on wait_and_poll and posts each decoded frame.
template <class Subscriber, class Frame>
class ImageViewerT : public QWidget
{
public:
	ImageViewerT(std::unique_ptr<Subscriber> sub, QString title, QWidget *parent = nullptr)
		: QWidget(parent), sub_(std::move(sub))
	{
		setWindowTitle(std::move(title));
		setAttribute(Qt::WA_DeleteOnClose);
		auto *layout = new QVBoxLayout(this);
		layout->setContentsMargins(0, 0, 0, 0);
		layout->addWidget(&label_);
		label_.setAlignment(Qt::AlignCenter);
		label_.setMinimumSize(320, 240);
		label_.setText("waiting for frames…");
		resize(720, 480);

		// Arrival-driven: block for a frame, decode off-thread, post to GUI.
		poller_ = std::jthread([this](std::stop_token st) { this->run(st); });
	}

	~ImageViewerT() override
	{
		poller_.request_stop();
		if(poller_.joinable())
			poller_.join();   // stop the worker before sub_/this are destroyed
	}

private:
	void run(std::stop_token st)
	{
		bool logged = false;
		while(not st.stop_requested())
		{
			QImage frame; int w = 0, h = 0; std::uint32_t fmt = 0;
			// Blocks up to 200 ms; returns the instant a frame is available (drains to newest).
			sub_->wait_and_poll([&](const Frame &f, std::int64_t)
			{
				const auto &buf = f.data();
				const int step = (f.step() > 0) ? static_cast<int>(f.step()) : static_cast<int>(f.width()) * 3;
				w = static_cast<int>(f.width()); h = static_cast<int>(f.height()); fmt = f.format();
				frame = qimage_from_media(buf.data(), w, h, step, fmt);   // deep copy, off GUI thread
			}, 200);
			if(frame.isNull())
				continue;
			if(not logged) { qInfo() << "[view-data] image viewer first frame" << w << "x" << h << "fmt" << fmt; logged = true; }
			// Rolling display-rate estimate from arrival deltas (worker thread).
			const auto now = std::chrono::steady_clock::now();
			if(have_last_)
			{
				const double dt_ms = std::chrono::duration<double, std::milli>(now - last_).count();
				if(dt_ms > 0.0 and dt_ms < 10000.0)
				{
					const float inst = 1000.0f / static_cast<float>(dt_ms);
					fps_ = (fps_ > 0.0f) ? (0.85f * fps_ + 0.15f * inst) : inst;
				}
			}
			last_ = now; have_last_ = true;
			// One arrival → one repaint on the GUI thread.
			const float fps = fps_;
			QMetaObject::invokeMethod(this, [this, frame, fps]() { render(frame, fps); }, Qt::QueuedConnection);
		}
	}

	void render(const QImage &frame, float fps)   // GUI thread
	{
		QImage shown = frame.scaled(label_.size(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
		QPainter p(&shown);
		p.fillRect(QRect(6, 6, 190, 22), QColor(0, 0, 0, 160));
		p.setPen(Qt::white);
		p.drawText(QRect(12, 6, 180, 22), Qt::AlignLeft | Qt::AlignVCenter,
		           QString::asprintf("FPS: %.1f   %dx%d", fps, frame.width(), frame.height()));
		label_.setPixmap(QPixmap::fromImage(shown));
	}

	std::unique_ptr<Subscriber> sub_;
	QLabel label_;
	float fps_ = 0.0f;
	std::chrono::steady_clock::time_point last_{};
	bool have_last_ = false;
	std::jthread poller_;   // declared last → stops/joins first on destruction
};

using ImageStreamViewer = ImageViewerT<rc::media::MediaSubscriber,    rc::media::ImageFrame>;
using Image360Viewer    = ImageViewerT<rc::media::Image360Subscriber, rc::media::Image360Frame>;

// ── IMU scrolling-plot viewer ────────────────────────────────────────────────
// robot_concept receives IMU over Ice and republishes it on the media plane (ImuFrame). The visible
// window IS the reusable rc::viewers::GLImuViewer (OpenGL time-series of accel/gyro/euler). The
// worker thread blocks on each arrival, copies the scalars off the GUI thread and push_sample()s
// them on the GUI thread.
class ImuStreamViewer : public GLImuViewer
{
public:
	ImuStreamViewer(std::unique_ptr<rc::media::ImuSubscriber> sub, QString title, QWidget *parent = nullptr)
		: GLImuViewer(parent), sub_(std::move(sub))
	{
		setWindowTitle(std::move(title));
		setAttribute(Qt::WA_DeleteOnClose);
		poller_ = std::jthread([this](std::stop_token st) { this->run(st); });
	}

	~ImuStreamViewer() override
	{
		poller_.request_stop();
		if(poller_.joinable())
			poller_.join();
	}

private:
	struct Sample { float ax, ay, az, gx, gy, gz, roll, pitch, yaw; std::uint64_t ts; };

	void run(std::stop_token st)
	{
		bool logged = false;
		while(not st.stop_requested())
		{
			Sample s{}; bool got = false;
			sub_->wait_and_poll([&](const rc::media::ImuFrame &f, std::int64_t)
			{
				s = {f.acc_x(), f.acc_y(), f.acc_z(), f.gyro_x(), f.gyro_y(), f.gyro_z(),
				     f.roll(), f.pitch(), f.yaw(), f.stamp_ms()};
				got = true;
			}, 200);
			if(not got)
				continue;
			if(not logged) { qInfo() << "[view-data] imu viewer first sample ts" << s.ts; logged = true; }
			QMetaObject::invokeMethod(this, [this, s]()
			{
				push_sample(s.ax, s.ay, s.az, s.gx, s.gy, s.gz, s.roll, s.pitch, s.yaw, s.ts);
			}, Qt::QueuedConnection);
		}
	}

	std::unique_ptr<rc::media::ImuSubscriber> sub_;
	std::jthread poller_;
};

// ── LiDAR 3D point viewer ────────────────────────────────────────────────────
// The visible window IS the reusable rc::viewers::GLPointCloudViewer (OpenGL, orbit camera). This
// wrapper owns up to two LiDAR subscribers (helios + bpearl) and a worker thread that blocks on
// their arrivals, converts the latest cloud of each ring to QVector3D (off the GUI thread), and
// pushes both groups into the GL viewer via set_points() on the GUI thread.
class LidarStreamViewer : public GLPointCloudViewer
{
public:
	LidarStreamViewer(std::unique_ptr<rc::media::LidarSubscriber> helios,
	                  std::unique_ptr<rc::media::LidarSubscriber> bpearl,
	                  QString title, QWidget *parent = nullptr)
		: GLPointCloudViewer(parent), helios_(std::move(helios)), bpearl_(std::move(bpearl))
	{
		setWindowTitle(std::move(title));
		setAttribute(Qt::WA_DeleteOnClose);
		poller_ = std::jthread([this](std::stop_token st) { this->run(st); });
	}

	~LidarStreamViewer() override
	{
		poller_.request_stop();
		if(poller_.joinable())
			poller_.join();
	}

private:
	static bool drain_latest(rc::media::LidarSubscriber *sub, std::vector<QVector3D> &dst, int timeout_ms)
	{
		if(sub == nullptr)
			return false;
		bool got = false;
		const auto cb = [&](const rc::media::LidarFrame &f, std::int64_t)
		{
			const std::size_t stride = (f.stride() > 0) ? f.stride() : 3;
			const std::size_t npts = std::min<std::size_t>(f.count(), f.points().size() / stride);
			dst.clear();
			dst.reserve(npts);
			const auto &pts = f.points();
			for(std::size_t i = 0; i < npts; ++i)
				dst.emplace_back(pts[i * stride + 0], pts[i * stride + 1], pts[i * stride + 2]);
			got = true;
		};
		// Wait on the primary ring (helios); non-blocking drain for the secondary (bpearl).
		if(timeout_ms > 0) sub->wait_and_poll(cb, timeout_ms);
		else               sub->poll(cb);
		return got;
	}

	void run(std::stop_token st)
	{
		bool logged = false;
		while(not st.stop_requested())
		{
			// Block on whichever ring exists; refresh the other non-blockingly. Keep the last known
			// cloud of each so a repaint always shows both rings.
			const bool a = drain_latest(helios_.get(), helios_pts_, helios_ ? 200 : 0);
			const bool b = drain_latest(bpearl_.get(), bpearl_pts_, helios_ ? 0 : 200);
			if(not a and not b)
				continue;
			if(not logged) { qInfo() << "[view-data] lidar viewer first frame — helios" << helios_pts_.size()
			                         << "bpearl" << bpearl_pts_.size(); logged = true; }
			// Copy the vectors for the GUI thread; the worker keeps its own persistent copies.
			auto ha = helios_pts_; auto bp = bpearl_pts_;
			QMetaObject::invokeMethod(this, [this, ha = std::move(ha), bp = std::move(bp)]()
			{
				set_points(std::span<const QVector3D>(ha), std::span<const QVector3D>(bp));
			}, Qt::QueuedConnection);
		}
	}

	std::unique_ptr<rc::media::LidarSubscriber> helios_, bpearl_;
	std::vector<QVector3D> helios_pts_, bpearl_pts_;   // worker-thread persistent latest per ring
	std::jthread poller_;
};

}   // namespace rc::viewers

#endif   // ROBOT_CONCEPT_MEDIA_STREAM_VIEWERS_H
