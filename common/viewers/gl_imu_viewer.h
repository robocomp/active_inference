/*
 *  Reusable IMU scrolling-plot viewer (active_inference/common).
 *
 *  OpenGL rendering (three stacked time-series plots for acceleration / angular velocity / euler
 *  pose, with grid + legend) copied from cortex dsr_gui's GraphNodeIMUWidget, with the graph
 *  coupling removed: instead of reading imu_* attributes off a DSR node, samples are pushed in via
 *  push_sample(). That makes it a generic widget any agent can drive from whatever source it owns —
 *  in particular a media-plane ImuSubscriber (robot_concept). Living in common (next to
 *  media_transport) avoids the cortex→active_inference dependency inversion.
 *
 *  Non-Q_OBJECT: the owner drives it via push_sample(), so no MOC pass is needed.
 */
#ifndef RC_COMMON_GL_IMU_VIEWER_H
#define RC_COMMON_GL_IMU_VIEWER_H

#include <QColor>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLWidget>
#include <QPainter>
#include <QRect>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace rc::viewers
{

class GLImuViewer : public QOpenGLWidget, protected QOpenGLFunctions
{
public:
	explicit GLImuViewer(QWidget *parent = nullptr) : QOpenGLWidget(parent)
	{
		resize(920, 620);
		setWindowTitle("IMU data");
		setFocusPolicy(Qt::StrongFocus);
	}

	~GLImuViewer() override
	{
		if(context())
		{
			makeCurrent();
			if(vbo_.isCreated()) vbo_.destroy();
			if(vao_.isCreated()) vao_.destroy();
			doneCurrent();
		}
	}

	// Push one IMU sample (GUI thread). Duplicate timestamps are ignored. ts==0 means "no stamp" and
	// is always accepted.
	void push_sample(float ax, float ay, float az, float gx, float gy, float gz,
	                  float roll, float pitch, float yaw, std::uint64_t ts)
	{
		if(ts != 0 and ts == last_timestamp_ms_)
			return;
		appendSample(plots_[0], ax, ay, az);
		appendSample(plots_[1], gx, gy, gz);
		appendSample(plots_[2], roll, pitch, yaw);
		last_timestamp_ms_ = ts;
		sample_count_ = std::min<std::size_t>(history_size, sample_count_ + 1);
		rebuildVertices();
		updateGpuBuffers();
		update();
	}

protected:
	void initializeGL() override
	{
		initializeOpenGLFunctions();
		glClearColor(0.06f, 0.06f, 0.07f, 1.0f);

		program_.addShaderFromSourceCode(QOpenGLShader::Vertex, R"(
			#version 330 core
			layout(location = 0) in vec2 position;
			layout(location = 1) in vec3 color;
			out vec3 v_color;
			void main() { gl_Position = vec4(position, 0.0, 1.0); v_color = color; }
		)");
		program_.addShaderFromSourceCode(QOpenGLShader::Fragment, R"(
			#version 330 core
			in vec3 v_color;
			out vec4 fragColor;
			void main() { fragColor = vec4(v_color, 1.0); }
		)");
		program_.link();

		vao_.create();
		vao_.bind();
		vbo_.create();
		vbo_.bind();
		vbo_.setUsagePattern(QOpenGLBuffer::DynamicDraw);
		vbo_.release();
		vao_.release();
		updateGpuBuffers();
	}

	void resizeGL(int w, int h) override { glViewport(0, 0, w, h); }

	void paintGL() override
	{
		glClear(GL_COLOR_BUFFER_BIT);
		if(not program_.isLinked())
			return;

		program_.bind();
		vao_.bind();
		if(not vertices_.empty())
		{
			vbo_.bind();
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), nullptr);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<const void *>(offsetof(Vertex, r)));
			glEnableVertexAttribArray(0);
			glEnableVertexAttribArray(1);

			if(grid_vertex_count_ > 0)
				glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(grid_vertex_count_));

			std::size_t offset = grid_vertex_count_;
			for(const auto &plot : plots_)
				for(const auto &channel : plot.channels)
				{
					const auto n = static_cast<GLsizei>(channel.values.size());
					if(n >= 2)
						glDrawArrays(GL_LINE_STRIP, static_cast<GLint>(offset), n);
					offset += channel.values.size();
				}

			glDisableVertexAttribArray(0);
			glDisableVertexAttribArray(1);
			vbo_.release();
		}
		vao_.release();
		program_.release();

		QPainter painter(this);
		painter.setRenderHint(QPainter::TextAntialiasing, true);
		painter.setPen(QColor(240, 240, 240));
		painter.drawText(QRect(10, 10, width() - 20, 24), Qt::AlignLeft | Qt::AlignTop,
		                 QString("IMU samples: %1    Last ts: %2")
		                     .arg(static_cast<int>(sample_count_))
		                     .arg(last_timestamp_ms_ > 0 ? QString::number(last_timestamp_ms_) : QStringLiteral("--")));

		const auto layout = computePlotLayout();
		for(std::size_t pi = 0; pi < plots_.size(); ++pi)
		{
			const auto &plot = plots_[pi];
			const auto &pl = layout[pi];
			painter.setPen(QColor(210, 210, 210));
			painter.drawText(pl.title_rect, Qt::AlignLeft | Qt::AlignVCenter, QString::fromStdString(plot.title));
			for(std::size_t ci = 0; ci < plot.channels.size(); ++ci)
			{
				const auto &ch = plot.channels[ci];
				painter.setPen(ch.color);
				painter.drawText(pl.value_rects[ci], Qt::AlignLeft | Qt::AlignVCenter,
				                 QString("%1: %2").arg(QString::fromStdString(ch.label))
				                     .arg(ch.values.empty() ? QStringLiteral("--") : QString::number(ch.values.back(), 'f', 4)));
			}
		}
	}

private:
	struct Vertex { float x, y, r, g, b; };
	struct Channel { std::string label; QColor color; std::deque<float> values; };
	struct Plot { std::string title; std::array<Channel, 3> channels; };
	struct PlotLayout
	{
		QRect title_rect;
		std::array<QRect, 3> value_rects;
		float plot_top = 0.0f, plot_bottom = 0.0f, plot_mid = 0.0f, data_left = 0.0f, data_right = 0.0f;
	};

	static constexpr std::size_t history_size = 360;
	std::uint64_t last_timestamp_ms_ = 0;
	std::size_t sample_count_ = 0;

	std::array<Plot, 3> plots_{{
		Plot{"Acceleration",     {Channel{"ax", QColor(255, 159, 28), {}}, Channel{"ay", QColor(46, 204, 113), {}}, Channel{"az", QColor(52, 152, 219), {}}}},
		Plot{"Angular velocity", {Channel{"gx", QColor(231, 76, 60), {}},  Channel{"gy", QColor(155, 89, 182), {}}, Channel{"gz", QColor(241, 196, 15), {}}}},
		Plot{"Euler xyz pose",   {Channel{"roll", QColor(26, 188, 156), {}}, Channel{"pitch", QColor(230, 126, 34), {}}, Channel{"yaw", QColor(236, 240, 241), {}}}},
	}};

	std::vector<Vertex> vertices_;
	std::size_t grid_vertex_count_ = 0;
	QOpenGLShaderProgram program_;
	QOpenGLVertexArrayObject vao_;
	QOpenGLBuffer vbo_{QOpenGLBuffer::VertexBuffer};

	static void pushValue(std::deque<float> &h, float v)
	{
		if(h.size() == history_size) h.pop_front();
		h.push_back(v);
	}
	void appendSample(Plot &p, float a, float b, float c)
	{
		pushValue(p.channels[0].values, a);
		pushValue(p.channels[1].values, b);
		pushValue(p.channels[2].values, c);
	}

	static float pixelToNdcX(float px, float w) { return -1.0f + 2.0f * px / std::max(1.0f, w); }
	static float pixelToNdcY(float py, float h) { return 1.0f - 2.0f * py / std::max(1.0f, h); }

	std::array<PlotLayout, 3> computePlotLayout() const
	{
		std::array<PlotLayout, 3> layout;
		const int ww = std::max(1, width()), wh = std::max(1, height());
		const int outer = 10, header = 34, gap = 18;
		const int legend = std::clamp(ww / 5, 180, 240);
		const int plots_top = outer + header, plots_bottom = wh - outer;
		const int total_gap = gap * static_cast<int>(plots_.size() - 1);
		const int avail = std::max(180, plots_bottom - plots_top - total_gap);
		const int ph = std::max(96, avail / static_cast<int>(plots_.size()));
		const int data_left_px = std::min(ww - outer - 40, outer + legend + 12);
		const int data_right_px = ww - outer;

		for(std::size_t i = 0; i < plots_.size(); ++i)
		{
			auto &pl = layout[i];
			const int top_px = plots_top + static_cast<int>(i) * (ph + gap);
			const int bottom_px = std::min(wh - outer, top_px + ph);
			pl.title_rect = QRect(outer + 4, top_px + 4, legend - 8, 20);
			for(std::size_t ci = 0; ci < pl.value_rects.size(); ++ci)
				pl.value_rects[ci] = QRect(outer + 12, top_px + 30 + static_cast<int>(ci) * 18, legend - 16, 18);
			pl.plot_top = pixelToNdcY(static_cast<float>(top_px), static_cast<float>(wh));
			pl.plot_bottom = pixelToNdcY(static_cast<float>(bottom_px), static_cast<float>(wh));
			pl.plot_mid = 0.5f * (pl.plot_top + pl.plot_bottom);
			pl.data_left = pixelToNdcX(static_cast<float>(data_left_px), static_cast<float>(ww));
			pl.data_right = pixelToNdcX(static_cast<float>(data_right_px), static_cast<float>(ww));
		}
		return layout;
	}

	float computeAmplitude(const Plot &p) const
	{
		float amp = 0.1f;
		for(const auto &ch : p.channels)
			for(const float v : ch.values)
				amp = std::max(amp, std::abs(v));
		return amp;
	}

	void pushLine(float x0, float y0, float x1, float y1, const QColor &c)
	{
		vertices_.push_back(Vertex{x0, y0, static_cast<float>(c.redF()), static_cast<float>(c.greenF()), static_cast<float>(c.blueF())});
		vertices_.push_back(Vertex{x1, y1, static_cast<float>(c.redF()), static_cast<float>(c.greenF()), static_cast<float>(c.blueF())});
	}

	void rebuildVertices()
	{
		vertices_.clear();
		grid_vertex_count_ = 0;
		vertices_.reserve(plots_.size() * (12 + history_size * 3));
		const auto layout = computePlotLayout();

		for(const auto &pl : layout)
		{
			pushLine(pl.data_left, pl.plot_bottom, pl.data_right, pl.plot_bottom, QColor(68, 72, 78));
			pushLine(pl.data_left, pl.plot_top, pl.data_right, pl.plot_top, QColor(68, 72, 78));
			pushLine(pl.data_left, pl.plot_mid, pl.data_right, pl.plot_mid, QColor(90, 95, 100));
			pushLine(pl.data_left, pl.plot_bottom, pl.data_left, pl.plot_top, QColor(68, 72, 78));
			pushLine(pl.data_right, pl.plot_bottom, pl.data_right, pl.plot_top, QColor(68, 72, 78));
		}
		grid_vertex_count_ = vertices_.size();

		for(std::size_t pi = 0; pi < plots_.size(); ++pi)
		{
			const auto &plot = plots_[pi];
			const auto &pl = layout[pi];
			const float amp = std::max(1e-3f, computeAmplitude(plot));
			const float plot_h = pl.plot_top - pl.plot_bottom;
			const float scale = (plot_h * 0.42f) / amp;
			const float x_pad = std::max(0.006f, (pl.data_right - pl.data_left) * 0.02f);
			const float y_pad = std::max(0.008f, plot_h * 0.05f);

			for(const auto &ch : plot.channels)
			{
				const auto count = ch.values.size();
				if(count < 2)
					continue;
				for(std::size_t i = 0; i < count; ++i)
				{
					const float x = (pl.data_left + x_pad)
					              + (pl.data_right - pl.data_left - 2.0f * x_pad)
					                    * static_cast<float>(i) / static_cast<float>(history_size - 1);
					const float y = std::clamp(pl.plot_mid + ch.values[i] * scale,
					                           pl.plot_bottom + y_pad, pl.plot_top - y_pad);
					vertices_.push_back(Vertex{x, y, static_cast<float>(ch.color.redF()),
					                           static_cast<float>(ch.color.greenF()), static_cast<float>(ch.color.blueF())});
				}
			}
		}
	}

	void updateGpuBuffers()
	{
		if(not context() or not vbo_.isCreated())
			return;
		makeCurrent();
		vbo_.bind();
		vbo_.allocate(vertices_.data(), static_cast<int>(vertices_.size() * sizeof(Vertex)));
		vbo_.release();
		doneCurrent();
	}
};

}   // namespace rc::viewers

#endif   // RC_COMMON_GL_IMU_VIEWER_H
