// The body of the "View data" popup in the memory window: a retained object's views. Qt widgets only.
#pragma once
#include "repertory.h"

#include <QWidget>

namespace ltom
{
QWidget* make_thumbnail_view(const Entry& e, QWidget* parent);
}
