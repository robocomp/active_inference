#include "residual_nvblox_view.h"

#include <algorithm>
#include <cmath>

#include <QGridLayout>
#include <QImage>
#include <QLabel>
#include <QPixmap>
#include <QVBoxLayout>

namespace rc
{
namespace
{
QRgb ramp(float t)   // blue → cyan → yellow → red
{
    t = std::clamp(t, 0.0f, 1.0f);
    const float k[4][3] = {{40, 60, 220}, {40, 210, 230}, {250, 230, 40}, {230, 40, 30}};
    const float s = t * 3.0f;
    const int i = std::min(2, static_cast<int>(s));
    const float u = s - i;
    return qRgb(static_cast<int>(k[i][0] + u * (k[i + 1][0] - k[i][0])),
                static_cast<int>(k[i][1] + u * (k[i + 1][1] - k[i][1])),
                static_cast<int>(k[i][2] + u * (k[i + 1][2] - k[i][2])));
}
constexpr QRgb kUnknown = qRgb(70, 70, 70);
constexpr QRgb kFree = qRgb(15, 15, 20);
}  // namespace

NvbloxView::NvbloxView(int scale) : scale_(std::max(1, scale))
{
    setWindowTitle("residual_nvblox");
    auto* grid = new QGridLayout;
    const char* titles[4] = {"nvblox P(ocupado)", "altura del vóxel ocupado más alto", "ESDF 2D (m, isolíneas 0,5 m)",
                             "residual: explicado (azul) · publicado (magenta)"};
    for (int k = 0; k < 4; ++k)
    {
        auto* box = new QVBoxLayout;
        box->addWidget(new QLabel(titles[k]));
        img_[k] = new QLabel;
        box->addWidget(img_[k]);
        grid->addLayout(box, k / 2, k % 2);
    }
    info_ = new QLabel;
    grid->addWidget(info_, 2, 0, 1, 2);
    setLayout(grid);
}

void NvbloxView::show_frame(const Frame& f)
{
    if (f.w <= 0 or f.h <= 0 or not f.cols or not f.esdf) return;
    std::array<QImage, 4> im;
    for (auto& i : im) i = QImage(f.w, f.h, QImage::Format_RGB32);
    const int b_lo = std::clamp(static_cast<int>(f.zlo / f.bin_h), 0, f.nbins - 1);
    const int b_hi = std::clamp(static_cast<int>(f.zhi / f.bin_h), 0, f.nbins - 1);
    for (int iy = 0; iy < f.h; ++iy)
        for (int ix = 0; ix < f.w; ++ix)
        {
            const int c = iy * f.w + ix, row = f.h - 1 - iy;   // north up
            const float* col = &(*f.cols)[static_cast<std::size_t>(c) * f.nbins];
            float mx = 0.0f; bool seen = false; int top = -1;
            for (int b = b_lo; b <= b_hi; ++b)
            {
                if (col[b] == 0.0f) continue;
                if (not seen or col[b] > mx) mx = col[b];
                seen = true;
                if (col[b] > 0.0f) top = b;
            }
            const float p = 1.0f / (1.0f + std::exp(-mx));
            im[0].setPixel(ix, row, not seen ? kUnknown : p < 0.5f ? qRgb(0, 0, static_cast<int>(40 + 120 * p)) : ramp((p - 0.5f) * 2));
            im[1].setPixel(ix, row, not seen ? kUnknown : top < 0 ? kFree : ramp((top + 0.5f) * f.bin_h / f.zhi));
            const float d = (*f.esdf)[c];
            QRgb e = kUnknown;
            if (not std::isnan(d))
            {
                if (d <= 0.0f) e = qRgb(230, 40, 30);
                else if (std::fmod(d, 0.5f) < f.cell * 0.7f) e = qRgb(230, 230, 230);
                else { const int g = static_cast<int>(30 + 200 * std::min(1.0f, d / f.esdf_max)); e = qRgb(0, g / 2, g); }
            }
            im[2].setPixel(ix, row, e);
            const std::uint8_t s = f.state.empty() ? 0 : f.state[c];
            im[3].setPixel(ix, row, s == 2 ? qRgb(255, 0, 255) : s == 1 ? qRgb(70, 110, 200) : (seen ? kFree : kUnknown));
        }
    const int rx = static_cast<int>((f.robot_x - f.xmin) / f.cell), ry = f.h - 1 - static_cast<int>((f.robot_y - f.ymin) / f.cell);
    for (auto& i : im)
        for (int k = -4; k <= 4; ++k)
        {
            if (rx + k >= 0 and rx + k < f.w and ry >= 0 and ry < f.h) i.setPixel(rx + k, ry, qRgb(255, 255, 255));
            if (ry + k >= 0 and ry + k < f.h and rx >= 0 and rx < f.w) i.setPixel(rx, ry + k, qRgb(255, 255, 255));
        }
    for (int k = 0; k < 4; ++k)
        img_[k]->setPixmap(QPixmap::fromImage(im[k].scaled(f.w * scale_, f.h * scale_)));
    info_->setText(QString::fromStdString(f.info));
    if (not isVisible()) show();
}

}  // namespace rc
