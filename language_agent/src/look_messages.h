#pragma once
/*
 * look_messages.h — the chat-history side of the look() tool (chat_session.h), header-only on Qt Core so the
 * request JSON can be checked offline (tests/test_look.cpp).
 *
 * THE FLOW. The model calls look(camera?) → the session grabs + shrinks the frame (camera_grabber.h, image_prep.h) →
 * the role:"tool" result ("OK: image attached — front camera, taken 120 ms ago, 1280x720 pixels") carries the JPEG
 * as base64 in its own `images` field (ollama /api/chat). A follow-up request lets the model answer with the picture
 * in front of it.
 *   ★The image rides on the TOOL message, not on an extra user message. Measured with qwen3-vl:8b-instruct
 *   (2026-10-12): both are seen, but an extra English "user" caption pulled a Spanish conversation into English and
 *   once turned a closed door into "appears open"; on the tool message the reply kept the person's language and
 *   read the door right.
 *
 * IMAGES LIVE FOR ONE TURN. After the request that carried it, every image is replaced by a short text placeholder
 * (strip_images), so a few look() calls cannot fill the 8k context with pictures nobody refers to any more; the
 * model keeps its own words about the image in the history. Logs never receive the base64 (redact_images).
 */

#include <deque>
#include <optional>

#include <QByteArray>
#include <QJsonArray>
#include <QJsonObject>
#include <QString>

namespace rc::chat
{

/// look(camera): "" / "front" / "zed" ⇒ front; "360" / "ricoh" / "pano…" / "around" ⇒ 360; anything else ⇒ nullopt.
/// Returns true for the front camera.
[[nodiscard]] inline std::optional<bool> look_camera_is_front(const QString& arg)
{
    const QString a = arg.trimmed().toLower();
    if (a.isEmpty() or a == "front" or a == "zed" or a == "forward" or a == "ahead") return true;
    if (a == "360" or a == "ricoh" or a.startsWith("pano") or a == "around" or a == "all around") return false;
    return std::nullopt;
}

[[nodiscard]] inline QString look_tool_result(const QString& label, qint64 age_ms, int w, int h)
{
    const QString age = age_ms < 0 ? QStringLiteral("of unknown age") : QStringLiteral("taken %1 ms ago").arg(age_ms);
    return QStringLiteral("OK: image attached — %1, %2, %3x%4 pixels.").arg(label, age).arg(w).arg(h);
}

/// The look() tool result: the text above plus the picture.
[[nodiscard]] inline QJsonObject make_look_tool_message(const QString& result_text, const QByteArray& jpeg)
{
    return QJsonObject{{"role", "tool"}, {"tool_name", "look"}, {"content", result_text},
                       {"images", QJsonArray{QString::fromLatin1(jpeg.toBase64())}}};
}

[[nodiscard]] inline bool has_image(const QJsonObject& m) { return m.contains("images"); }

/// Replace every image in the history by a text placeholder. Returns how many were dropped.
inline int strip_images(std::deque<QJsonObject>& history)
{
    int n = 0;
    for (auto& m : history)
        if (has_image(m))
        {
            m.remove("images");
            m.insert("content", m.value("content").toString() + QStringLiteral(" (the image is no longer attached)"));
            ++n;
        }
    return n;
}

/// A copy of `msgs` fit for a log: base64 payloads replaced by their size.
[[nodiscard]] inline QJsonArray redact_images(const QJsonArray& msgs)
{
    QJsonArray out;
    for (const auto& v : msgs)
    {
        QJsonObject m = v.toObject();
        if (has_image(m))
        {
            QJsonArray sizes;
            for (const auto& img : m.value("images").toArray()) sizes.append(QStringLiteral("<base64, %1 chars>").arg(img.toString().size()));
            m.insert("images", sizes);
        }
        out.append(m);
    }
    return out;
}

}   // namespace rc::chat
