#include "thumbnail_view.h"
#include "../../common/thumbs/thumb_select.h"

#include <QGridLayout>
#include <QLabel>
#include <QPixmap>
#include <QVBoxLayout>

namespace ltom
{
QWidget* make_thumbnail_view(const Entry& e, QWidget* parent)
{
    static constexpr int kPerRow = 4;
    auto* view = new QWidget(parent);
    auto* column = new QVBoxLayout(view);
    column->addWidget(new QLabel(QString("<b>%1</b>   uid %2").arg(QString::fromStdString(e.name)).arg(e.uid), view));
    auto* grid = new QGridLayout();
    column->addLayout(grid);
    int shown = 0;
    for (const auto& [bin, thumb] : e.thumbs)
    {
        QPixmap image;
        if (not image.loadFromData(thumb.jpeg.data(), static_cast<uint>(thumb.jpeg.size()), "JPG")) continue;
        auto* picture = new QLabel(view);
        picture->setPixmap(image);
        auto* caption = new QLabel(QString("from %1°").arg(bin * 360 / rc::thumbs::kViewBins), view);
        caption->setAlignment(Qt::AlignHCenter);
        const int row = 2 * (shown / kPerRow), col = shown % kPerRow;
        grid->addWidget(picture, row, col);
        grid->addWidget(caption, row + 1, col);
        ++shown;
    }
    if (shown == 0) column->addWidget(new QLabel("No thumbnails retained for this object yet.", view));
    return view;
}
}   // namespace ltom
