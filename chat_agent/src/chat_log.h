#pragma once
/*
 * chat_log.h — one JSON object per line, per run: logs/chat_<YYYYmmdd-HHMMSS>.jsonl.
 *
 * Every event a chat session produces goes here — the user's text, the STATE the model was given, the request,
 * tool calls, gate verdicts, confirmations, intention writes and the controller's state transitions, errors, and
 * latency/eval counts — so a conversation can be replayed and graded from the file rather than from the terminal.
 * Each record carries `t` (epoch ms) and `ev` (event kind). The stream is imbued with the classic locale (CLAUDE.md:
 * a data file must never acquire a comma decimal separator); QJsonDocument formats numbers itself anyway.
 * Flushed per record: the interesting records are the ones written just before something went wrong.
 */

#include <filesystem>
#include <fstream>
#include <locale>
#include <string>

#include <QDateTime>
#include <QJsonDocument>
#include <QJsonObject>

namespace rc::chat
{

class ChatLog
{
public:
    bool open(const std::string& dir)
    {
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        path_ = dir + "/chat_" + QDateTime::currentDateTime().toString("yyyyMMdd-HHmmss").toStdString() + ".jsonl";
        f_.open(path_, std::ios::out | std::ios::app);
        f_.imbue(std::locale::classic());
        return f_.is_open();
    }
    void write(const char* ev, QJsonObject o = {})
    {
        if (not f_.is_open()) return;
        o.insert("t", QDateTime::currentMSecsSinceEpoch());
        o.insert("ev", QString::fromUtf8(ev));
        f_ << QJsonDocument(o).toJson(QJsonDocument::Compact).toStdString() << '\n';
        f_.flush();
    }
    [[nodiscard]] const std::string& path() const { return path_; }

private:
    std::ofstream f_;
    std::string path_;
};

}   // namespace rc::chat
