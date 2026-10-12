#pragma once
/*
 * ollama_client.h — a minimal, streaming ollama /api/chat client on QNetworkAccessManager (no libcurl, no
 * nlohmann: neither is installed; Qt6 Network + QJsonDocument are).
 *
 *   ping()     GET /api/version, then /api/tags — is the server up, and is the configured model pulled?
 *   warmup()   load the model (empty chat, keep_alive) so the first real question does not pay the load.
 *   chat()     POST /api/chat {model, messages, tools, stream:true, think:false, keep_alive, options}.
 *              The body is NDJSON: one JSON object per line, cut into chunks by the network — NdjsonSplitter
 *              re-assembles lines. Tokens stream to on_token; tool calls are collected and delivered once, just
 *              before on_done; an {"error":...} line or a transport error goes to on_error.
 *   cancel()   aborts the request in flight; no callback of that request fires afterwards.
 *
 * ONE request in flight: chat() returns false while another is running. Two timeouts:
 *   · QNetworkRequest::setTransferTimeout — no bytes at all for transfer_timeout_ms ⇒ the transport gives up.
 *   · a FIRST-TOKEN watchdog — nothing useful (content or tool call) within first_token_timeout_ms ⇒ abort. The
 *     model load on a cold server is the usual reason, which is why warmup() exists.
 *
 * No Q_OBJECT (no moc): connections are made to lambdas with ctx_ as the context object, so destroying the client
 * disconnects everything. Main (GUI) thread only.
 */

#include <cstdint>
#include <functional>
#include <memory>

#include <QJsonArray>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QObject>
#include <QPointer>
#include <QString>
#include <QTimer>

#include "ndjson_splitter.h"

namespace rc::chat
{

struct OllamaConfig
{
    QString url = "http://localhost:11434";
    QString model = "qwen3:8b";
    int num_ctx = 8192;
    double temperature = 0.2;
    int first_token_timeout_ms = 60000;
    int transfer_timeout_ms = 120000;
    QString keep_alive = "30m";
};

struct ChatStats
{
    std::int64_t eval_count = 0;
    std::int64_t prompt_eval_count = 0;
    double total_ms = 0.0;         ///< server-side total_duration
    double first_token_ms = -1.0;  ///< client-side, request sent → first content/tool call
    double wall_ms = 0.0;          ///< client-side, request sent → done
    QString done_reason;
};

class OllamaClient
{
public:
    struct Callbacks
    {
        std::function<void(const QString&)> on_token;
        std::function<void(const QJsonArray&)> on_tool_calls;   ///< message.tool_calls, all of them, once
        std::function<void(const ChatStats&)> on_done;
        std::function<void(const QString&)> on_error;
    };
    struct PingResult
    {
        bool server_up = false;
        bool model_present = false;
        QString version;
        QString message;   ///< human-readable summary (warning when the model is missing)
    };

    explicit OllamaClient(OllamaConfig cfg);
    ~OllamaClient();

    void ping(std::function<void(const PingResult&)> done);
    void warmup();
    bool chat(const QJsonArray& messages, const QJsonArray& tools, Callbacks cb);
    void cancel();
    [[nodiscard]] bool busy() const { return not reply_.isNull(); }
    [[nodiscard]] const OllamaConfig& config() const { return cfg_; }

private:
    void handle_line(const std::string& line);
    void finish_with_error(const QString& msg);
    void finish_ok();
    void teardown();

    OllamaConfig cfg_;
    QObject ctx_;                         ///< connection context; destroyed with the client
    std::unique_ptr<QNetworkAccessManager> nam_;
    QPointer<QNetworkReply> reply_;
    QTimer first_token_timer_;
    NdjsonSplitter splitter_;
    Callbacks cb_;
    QJsonArray tool_calls_;
    ChatStats stats_;
    std::int64_t t0_ms_ = 0;
    bool got_first_ = false;
    bool done_seen_ = false;
    bool finished_ = false;               ///< a terminal callback has fired for this request
    std::uint64_t generation_ = 0;        ///< bumps on every new request / cancel; stale signals are ignored
};

}   // namespace rc::chat
