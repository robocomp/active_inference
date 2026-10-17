#include "ollama_client.h"

#include <QDateTime>
#include <QJsonDocument>
#include <QNetworkRequest>
#include <QUrl>

namespace rc::chat
{

namespace
{
std::int64_t now_ms() { return QDateTime::currentMSecsSinceEpoch(); }

QNetworkRequest json_request(const QString& url, int transfer_timeout_ms)
{
    QNetworkRequest req{QUrl(url)};
    req.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));
    if (transfer_timeout_ms > 0) req.setTransferTimeout(transfer_timeout_ms);
    return req;
}
}   // namespace

OllamaClient::OllamaClient(OllamaConfig cfg) : cfg_(std::move(cfg)), nam_(std::make_unique<QNetworkAccessManager>())
{
    first_token_timer_.setSingleShot(true);
    QObject::connect(&first_token_timer_, &QTimer::timeout, &ctx_, [this]()
    {
        if (reply_.isNull() or got_first_) return;
        finish_with_error(QStringLiteral("no answer from the model after %1 s (is it still loading? run `ollama ps`)")
                              .arg(cfg_.first_token_timeout_ms / 1000));
    });
}

OllamaClient::~OllamaClient()
{
    ++generation_;
    teardown();
}

void OllamaClient::ping(std::function<void(const PingResult&)> done)
{
    auto* r = nam_->get(json_request(cfg_.url + "/api/version", 3000));
    QObject::connect(r, &QNetworkReply::finished, &ctx_, [this, r, done]()
    {
        r->deleteLater();
        PingResult res;
        if (r->error() != QNetworkReply::NoError)
        {
            res.message = QStringLiteral("ollama not reachable at %1 (%2)").arg(cfg_.url, r->errorString());
            if (done) done(res);
            return;
        }
        res.server_up = true;
        res.version = QJsonDocument::fromJson(r->readAll()).object().value("version").toString();
        auto* t = nam_->get(json_request(cfg_.url + "/api/tags", 3000));
        QObject::connect(t, &QNetworkReply::finished, &ctx_, [this, t, done, res]() mutable
        {
            t->deleteLater();
            const auto models = QJsonDocument::fromJson(t->readAll()).object().value("models").toArray();
            for (const auto& m : models)
            {
                const QString name = m.toObject().value("name").toString();
                const QString model = m.toObject().value("model").toString();
                if (name == cfg_.model or model == cfg_.model or name == cfg_.model + ":latest") res.model_present = true;
            }
            res.message = res.model_present
                ? QStringLiteral("ollama %1, model %2 ready").arg(res.version, cfg_.model)
                : QStringLiteral("ollama %1 up, but model %2 is missing — run `ollama pull %2`").arg(res.version, cfg_.model);
            if (done) done(res);
        });
    });
}

void OllamaClient::warmup()
{
    QJsonObject body{{"model", cfg_.model}, {"messages", QJsonArray{}}, {"keep_alive", cfg_.keep_alive}};
    auto* r = nam_->post(json_request(cfg_.url + "/api/chat", cfg_.transfer_timeout_ms),
                         QJsonDocument(body).toJson(QJsonDocument::Compact));
    QObject::connect(r, &QNetworkReply::finished, &ctx_, [r]() { r->deleteLater(); });
}

bool OllamaClient::chat(const QJsonArray& messages, const QJsonArray& tools, Callbacks cb)
{
    if (busy()) return false;
    const std::uint64_t gen = ++generation_;
    cb_ = std::move(cb);
    splitter_.reset();
    tool_calls_ = QJsonArray{};
    stats_ = ChatStats{};
    got_first_ = false;
    done_seen_ = false;
    finished_ = false;
    t0_ms_ = now_ms();

    QJsonObject body{{"model", cfg_.model},
                     {"messages", messages},
                     {"stream", true},
                     {"think", false},
                     {"keep_alive", cfg_.keep_alive},
                     {"options", QJsonObject{{"temperature", cfg_.temperature}, {"num_ctx", cfg_.num_ctx}}}};
    if (cfg_.num_predict > 0)
    {
        QJsonObject opts = body.value("options").toObject();
        opts.insert("num_predict", cfg_.num_predict);
        body.insert("options", opts);
    }
    if (not tools.isEmpty()) body.insert("tools", tools);

    reply_ = nam_->post(json_request(cfg_.url + "/api/chat", cfg_.transfer_timeout_ms),
                        QJsonDocument(body).toJson(QJsonDocument::Compact));
    QNetworkReply* r = reply_.data();
    QObject::connect(r, &QNetworkReply::readyRead, &ctx_, [this, r, gen]()
    {
        if (gen != generation_ or finished_) return;
        const QByteArray chunk = r->readAll();
        for (const auto& line : splitter_.feed(std::string_view(chunk.constData(), static_cast<std::size_t>(chunk.size()))))
        {
            handle_line(line);
            if (finished_ or gen != generation_) return;
        }
    });
    QObject::connect(r, &QNetworkReply::finished, &ctx_, [this, r, gen]()
    {
        r->deleteLater();
        if (gen != generation_ or finished_) return;
        if (const QByteArray rest = r->readAll(); not rest.isEmpty())
            for (const auto& line : splitter_.feed(std::string_view(rest.constData(), static_cast<std::size_t>(rest.size()))))
            {
                handle_line(line);
                if (finished_) return;
            }
        if (const auto last = splitter_.flush(); last.has_value())
        {
            handle_line(*last);
            if (finished_) return;
        }
        if (r->error() != QNetworkReply::NoError)
        {
            const int http = r->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
            finish_with_error(QStringLiteral("HTTP %1: %2").arg(http).arg(r->errorString()));
            return;
        }
        if (done_seen_) finish_ok();
        else finish_with_error(QStringLiteral("the stream ended without a final 'done' message"));
    });
    first_token_timer_.start(cfg_.first_token_timeout_ms);
    return true;
}

void OllamaClient::handle_line(const std::string& line)
{
    QJsonParseError perr{};
    const auto doc = QJsonDocument::fromJson(QByteArray::fromStdString(line), &perr);
    if (perr.error != QJsonParseError::NoError or not doc.isObject())
    {
        finish_with_error(QStringLiteral("unparseable stream line: %1").arg(QString::fromStdString(line).left(200)));
        return;
    }
    const QJsonObject o = doc.object();
    if (o.contains("error"))
    {
        finish_with_error(o.value("error").toString());
        return;
    }
    const QJsonObject msg = o.value("message").toObject();
    const QString content = msg.value("content").toString();
    const QJsonArray calls = msg.value("tool_calls").toArray();
    if ((not content.isEmpty() or not calls.isEmpty()) and not got_first_)
    {
        got_first_ = true;
        first_token_timer_.stop();
        stats_.first_token_ms = static_cast<double>(now_ms() - t0_ms_);
    }
    for (const auto& c : calls) tool_calls_.append(c);
    if (not content.isEmpty() and cb_.on_token) cb_.on_token(content);
    if (o.value("done").toBool())
    {
        done_seen_ = true;
        stats_.eval_count = o.value("eval_count").toInteger();
        stats_.prompt_eval_count = o.value("prompt_eval_count").toInteger();
        stats_.total_ms = o.value("total_duration").toDouble() / 1e6;
        stats_.done_reason = o.value("done_reason").toString();
        finish_ok();
    }
}

void OllamaClient::finish_ok()
{
    if (finished_) return;
    finished_ = true;
    first_token_timer_.stop();
    stats_.wall_ms = static_cast<double>(now_ms() - t0_ms_);
    auto cb = std::move(cb_);
    cb_ = {};
    const QJsonArray calls = tool_calls_;
    ++generation_;   // nothing more from this reply
    teardown();
    if (not calls.isEmpty() and cb.on_tool_calls) cb.on_tool_calls(calls);
    if (cb.on_done) cb.on_done(stats_);
}

void OllamaClient::finish_with_error(const QString& msg)
{
    if (finished_) return;
    finished_ = true;
    first_token_timer_.stop();
    auto cb = std::move(cb_);
    cb_ = {};
    ++generation_;
    teardown();
    if (cb.on_error) cb.on_error(msg);
}

void OllamaClient::cancel()
{
    if (reply_.isNull()) return;
    finish_with_error(QStringLiteral("cancelled"));
}

void OllamaClient::teardown()
{
    first_token_timer_.stop();
    if (QNetworkReply* r = reply_.data(); r != nullptr)
    {
        reply_.clear();
        if (r->isRunning()) r->abort();   // emits finished(); its lambda sees a stale generation and only deleteLater()s
    }
}

}   // namespace rc::chat
