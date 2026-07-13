#pragma once

/*
 *  SemanticMapViewer — a pure (non-Q_OBJECT) QLabel that renders a dense ADE20K-150
 *  per-pixel class-id map as a colour image and reports the class name under the cursor.
 *
 *  This is the renderer half of robot_concept's "View data" viewer for the voxelizer's
 *  'semantic' (type semantic_grid) node. It mirrors what the voxelizer's YoloViewer shows
 *  in-process: the SAME canonical ADE20K SceneParse150 colormap for the colour image and
 *  the SAME class-name table for the hover readout, so the graph viewer and the producer
 *  window agree pixel-for-pixel. It is self-contained (no OpenCV, no cortex) so it can live
 *  in common/ and be reused by any agent — the caller feeds a raw row-major byte buffer.
 *
 *  Contract: set_label_map(ids, w, h) where ids[y*w + x] is the class id in [0,150) or the
 *  ADE20K ignore label 255 (rendered black / reported as "(unlabelled)").
 */

#include <QImage>
#include <QLabel>
#include <QMouseEvent>
#include <QPixmap>
#include <QResizeEvent>
#include <QSizePolicy>
#include <QString>
#include <QToolTip>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace rc::viewers
{

// Canonical ADE20K (MIT SceneParse150 / mmsegmentation) colormap: one DISTINCT colour per
// class id, in [R,G,B] order. Indexed by class id (0=wall … 149=flag). Copied verbatim from
// the voxelizer's compose_semantic_canvas so the two views match.
inline const std::array<std::array<std::uint8_t, 3>, 150> &ade20k_palette()
{
	static const std::array<std::array<std::uint8_t, 3>, 150> palette = {{
		{120,120,120}, {180,120,120}, {  6,230,230}, { 80, 50, 50}, {  4,200,  3}, {120,120, 80},
		{140,140,140}, {204,  5,255}, {230,230,230}, {  4,250,  7}, {224,  5,255}, {235,255,  7},
		{150,  5, 61}, {120,120, 70}, {  8,255, 51}, {255,  6, 82}, {143,255,140}, {204,255,  4},
		{255, 51,  7}, {204, 70,  3}, {  0,102,200}, { 61,230,250}, {255,  6, 51}, { 11,102,255},
		{255,  7, 71}, {255,  9,224}, {  9,  7,230}, {220,220,220}, {255,  9, 92}, {112,  9,255},
		{  8,255,214}, {  7,255,224}, {255,184,  6}, { 10,255, 71}, {255, 41, 10}, {  7,255,255},
		{224,255,  8}, {102,  8,255}, {255, 61,  6}, {255,194,  7}, {255,122,  8}, {  0,255, 20},
		{255,  8, 41}, {255,  5,153}, {  6, 51,255}, {235, 12,255}, {160,150, 20}, {  0,163,255},
		{140,140,140}, {250, 10, 15}, { 20,255,  0}, { 31,255,  0}, {255, 31,  0}, {255,224,  0},
		{153,255,  0}, {  0,  0,255}, {255, 71,  0}, {  0,235,255}, {  0,173,255}, { 31,  0,255},
		{ 11,200,200}, {255, 82,  0}, {  0,255,245}, {  0, 61,255}, {  0,255,112}, {  0,255,133},
		{255,  0,  0}, {255,163,  0}, {255,102,  0}, {194,255,  0}, {  0,143,255}, { 51,255,  0},
		{  0, 82,255}, {  0,255, 41}, {  0,255,173}, { 10,  0,255}, {173,255,  0}, {  0,255,153},
		{255, 92,  0}, {255,  0,255}, {255,  0,245}, {255,  0,102}, {255,173,  0}, {255,  0, 20},
		{255,184,184}, {  0, 31,255}, {  0,255, 61}, {  0, 71,255}, {255,  0,204}, {  0,255,194},
		{  0,255, 82}, {  0, 10,255}, {  0,112,255}, { 51,  0,255}, {  0,194,255}, {  0,122,255},
		{  0,255,163}, {255,153,  0}, {  0,255, 10}, {255,112,  0}, {143,255,  0}, { 82,  0,255},
		{163,255,  0}, {255,235,  0}, {  8,184,170}, {133,  0,255}, {  0,255, 92}, {184,  0,255},
		{255,  0, 31}, {  0,184,255}, {  0,214,255}, {255,  0,112}, { 92,255,  0}, {  0,224,255},
		{112,224,255}, { 70,184,160}, {163,  0,255}, {153,  0,255}, { 71,255,  0}, {255,  0,163},
		{255,204,  0}, {255,  0,143}, {  0,255,235}, {133,255,  0}, {255,  0,235}, {245,  0,255},
		{255,  0,122}, {255,245,  0}, { 10,190,212}, {214,255,  0}, {  0,204,255}, { 20,  0,255},
		{255,255,  0}, {  0,153,255}, {  0, 41,255}, {  0,255,204}, { 41,  0,255}, { 41,255,  0},
		{173,  0,255}, {  0,245,255}, { 71,  0,255}, {122,  0,255}, {  0,255,184}, {  0, 92,255},
		{184,255,  0}, {  0,133,255}, {255,214,  0}, { 25,194,194}, {102,255,  0}, { 92,  0,255},
	}};
	return palette;
}

// ADE20K SceneParse150 label set in model class-id order (0 = wall … 149 = flag). Copied
// verbatim from the voxelizer's YoloSemanticSegmenter::default_class_names().
inline const std::vector<std::string> &ade20k_class_names()
{
	static const std::vector<std::string> names = {
		"wall","building","sky","floor","tree","ceiling","road","bed","windowpane","grass",
		"cabinet","sidewalk","person","earth","door","table","mountain","plant","curtain","chair",
		"car","water","painting","sofa","shelf","house","sea","mirror","rug","field",
		"armchair","seat","fence","desk","rock","wardrobe","lamp","bathtub","railing","cushion",
		"base","box","column","signboard","chest of drawers","counter","sand","sink","skyscraper","fireplace",
		"refrigerator","grandstand","path","stairs","runway","case","pool table","pillow","screen door","stairway",
		"river","bridge","bookcase","blind","coffee table","toilet","flower","book","hill","bench",
		"countertop","stove","palm","kitchen island","computer","swivel chair","boat","bar","arcade machine","hovel",
		"bus","towel","light","truck","tower","chandelier","awning","streetlight","booth","television receiver",
		"airplane","dirt track","apparel","pole","land","bannister","escalator","ottoman","bottle","buffet",
		"poster","stage","van","ship","fountain","conveyer belt","canopy","washer","plaything","swimming pool",
		"stool","barrel","basket","waterfall","tent","bag","minibike","cradle","oven","ball",
		"food","step","tank","trade name","microwave","pot","animal","bicycle","lake","dishwasher",
		"screen","blanket","sculpture","hood","sconce","vase","traffic light","tray","ashcan","fan",
		"pier","crt screen","plate","monitor","bulletin board","shower","radiator","glass","clock","flag"
	};
	return names;
}

class SemanticMapViewer : public QLabel
{
public:
	explicit SemanticMapViewer(QWidget *parent = nullptr) : QLabel(parent)
	{
		setAlignment(Qt::AlignCenter);
		setMinimumSize(320, 240);
		setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
		setText("Waiting for semantic frame…");
		setMouseTracking(true);   // fire mouseMoveEvent without a pressed button (hover readout)
	}

	// ids = row-major CV_8UC1 class-id buffer (w*h bytes); values in [0,150) or 255 (ignore).
	void set_label_map(const std::vector<std::uint8_t> &ids, int w, int h)
	{
		if (w <= 0 or h <= 0 or ids.size() < static_cast<std::size_t>(w) * static_cast<std::size_t>(h))
			return;
		labels_ = ids;
		lw_ = w;
		lh_ = h;

		const auto &palette = ade20k_palette();
		QImage img(w, h, QImage::Format_RGB888);
		for (int y = 0; y < h; ++y)
		{
			uchar *dst = img.scanLine(y);
			const std::uint8_t *src = ids.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(w);
			for (int x = 0; x < w; ++x)
			{
				const std::uint8_t id = src[x];
				if (id < palette.size())   // < 150: paint; 255/ignore → black
				{
					const auto &c = palette[id];
					dst[3 * x + 0] = c[0];
					dst[3 * x + 1] = c[1];
					dst[3 * x + 2] = c[2];
				}
				else
				{
					dst[3 * x + 0] = 0;
					dst[3 * x + 1] = 0;
					dst[3 * x + 2] = 0;
				}
			}
		}
		last_pixmap_ = QPixmap::fromImage(img);
		// Nearest-neighbour scaling: a class-id colormap must NOT be smoothed (blending two ids
		// invents a third colour at every boundary).
		setPixmap(last_pixmap_.scaled(size(), Qt::KeepAspectRatio, Qt::FastTransformation));
	}

protected:
	void resizeEvent(QResizeEvent *event) override
	{
		QLabel::resizeEvent(event);
		if (not last_pixmap_.isNull())
			setPixmap(last_pixmap_.scaled(size(), Qt::KeepAspectRatio, Qt::FastTransformation));
	}

	void mouseMoveEvent(QMouseEvent *event) override
	{
		QLabel::mouseMoveEvent(event);

		if (labels_.empty() or last_pixmap_.isNull())
		{
			QToolTip::hideText();
			return;
		}

		// The pixmap is shown scaled-to-fit (KeepAspectRatio) and centred (AlignCenter), so undo that
		// letterbox transform to recover the image pixel under the cursor.
		const QSize shown = last_pixmap_.size().scaled(size(), Qt::KeepAspectRatio);
		const int off_x = (width() - shown.width()) / 2;
		const int off_y = (height() - shown.height()) / 2;

		const QPoint p = event->position().toPoint();
		const double rx = static_cast<double>(p.x() - off_x) / shown.width();
		const double ry = static_cast<double>(p.y() - off_y) / shown.height();
		if (rx < 0.0 or rx >= 1.0 or ry < 0.0 or ry >= 1.0)   // cursor in the letterbox margin
		{
			QToolTip::hideText();
			return;
		}

		const int ix = std::clamp(static_cast<int>(rx * lw_), 0, lw_ - 1);
		const int iy = std::clamp(static_cast<int>(ry * lh_), 0, lh_ - 1);
		const int id = labels_[static_cast<std::size_t>(iy) * static_cast<std::size_t>(lw_) + static_cast<std::size_t>(ix)];

		const auto &names = ade20k_class_names();
		QString text;
		if (id >= 0 and id < static_cast<int>(names.size()))
			text = QString::fromStdString(names[static_cast<std::size_t>(id)]);
		else
			text = QStringLiteral("(unlabelled)");   // ignore label (255) / out of range

		QToolTip::showText(event->globalPosition().toPoint(), text, this);
	}

private:
	QPixmap last_pixmap_;
	std::vector<std::uint8_t> labels_;   // last colourised label buffer (for the hover lookup)
	int lw_ = 0;
	int lh_ = 0;
};

}   // namespace rc::viewers
