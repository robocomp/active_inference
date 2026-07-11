/*
 *  Reusable 2D polygon viewer (active_inference/common).
 *
 *  Draws a closed polygon given as parallel x/y coordinate arrays, auto-scaled (equal aspect, Y up)
 *  to the window with a margin. Pure renderer: points come in through set_polygon(), it knows
 *  nothing about the graph or the media plane, so any agent can drive it. Non-Q_OBJECT (the owner
 *  drives it), so no MOC pass is needed.
 */
#ifndef RC_COMMON_POLYGON_VIEWER_H
#define RC_COMMON_POLYGON_VIEWER_H

#include <QWidget>
#include <QPainter>
#include <QPointF>

#include <algorithm>
#include <cmath>
#include <span>
#include <vector>

namespace rc::viewers
{

class PolygonViewer : public QWidget
{
public:
	explicit PolygonViewer(QWidget *parent = nullptr) : QWidget(parent)
	{
		resize(560, 560);
		setWindowTitle("polygon");
	}

	// Replace the polygon. xs/ys are parallel coordinate arrays (same length; extra tail ignored).
	void set_polygon(std::span<const float> xs, std::span<const float> ys)
	{
		const std::size_t n = std::min(xs.size(), ys.size());
		xs_.assign(xs.begin(), xs.begin() + n);
		ys_.assign(ys.begin(), ys.begin() + n);
		update();
	}

protected:
	void paintEvent(QPaintEvent *) override
	{
		QPainter p(this);
		p.setRenderHint(QPainter::Antialiasing, true);
		p.fillRect(rect(), QColor(18, 18, 22));
		p.setPen(QColor(200, 200, 200));
		p.drawText(8, 18, QString("delimiting polygon — %1 vertices").arg(xs_.size()));

		if(xs_.size() < 2)
			return;

		const auto [minx_it, maxx_it] = std::minmax_element(xs_.begin(), xs_.end());
		const auto [miny_it, maxy_it] = std::minmax_element(ys_.begin(), ys_.end());
		const float minx = *minx_it, maxx = *maxx_it, miny = *miny_it, maxy = *maxy_it;
		if(maxx <= minx or maxy <= miny)
			return;

		// Equal-aspect fit (rooms shouldn't be stretched), Y up.
		const float margin = 34.0f;
		const float s = std::min((width()  - 2 * margin) / (maxx - minx),
		                         (height() - 2 * margin) / (maxy - miny));
		const float cx = width() / 2.0f, cy = height() / 2.0f;
		const float mx = (minx + maxx) / 2.0f, my = (miny + maxy) / 2.0f;
		const auto to_screen = [&](float x, float y)
		{ return QPointF(cx + (x - mx) * s, cy - (y - my) * s); };   // screen Y is down → flip

		// World origin marker (helps read the room pose).
		p.setPen(QPen(QColor(120, 60, 60), 1));
		const QPointF o = to_screen(0.0f, 0.0f);
		p.drawLine(QPointF(o.x() - 6, o.y()), QPointF(o.x() + 6, o.y()));
		p.drawLine(QPointF(o.x(), o.y() - 6), QPointF(o.x(), o.y() + 6));

		// Closed polygon outline.
		std::vector<QPointF> pts;
		pts.reserve(xs_.size() + 1);
		for(std::size_t i = 0; i < xs_.size(); ++i)
			pts.push_back(to_screen(xs_[i], ys_[i]));
		pts.push_back(pts.front());   // close it
		p.setPen(QPen(QColor(90, 200, 250), 2));
		p.drawPolyline(pts.data(), static_cast<int>(pts.size()));

		// Vertices.
		p.setBrush(QColor(255, 200, 90));
		p.setPen(Qt::NoPen);
		for(std::size_t i = 0; i < xs_.size(); ++i)
			p.drawEllipse(pts[i], 3.0, 3.0);
	}

private:
	std::vector<float> xs_, ys_;
};

}   // namespace rc::viewers

#endif   // RC_COMMON_POLYGON_VIEWER_H
