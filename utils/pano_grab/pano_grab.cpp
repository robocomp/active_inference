// pano_grab — grab ONE frame off the 360 media plane and write it to a PNG.
//
// Diagnostic tool. A panorama defect (a mirrored half, a seam in the wrong column, elevation
// crushed by the wrong projection) is a claim about PIXELS, and eyeballing a scaled screenshot
// cannot settle it. This puts the actual frame on disk so it can be measured.
//
//   pano_grab [out.png] [topic=rc/ricoh/rgb] [domain=7]

#include "../../common/media_transport/media_transport.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char** argv)
{
    const std::string out    = argc > 1 ? argv[1] : "pano.png";
    const std::string topic  = argc > 2 ? argv[2] : "rc/ricoh/rgb";
    const std::uint32_t domain = argc > 3 ? static_cast<std::uint32_t>(std::stoul(argv[3])) : 7u;

    rc::media::SubscriberConfig cfg;
    cfg.domain_id         = domain;
    cfg.topic_name        = topic;
    cfg.history_depth     = 2;
    cfg.shared_memory_only = true;
    cfg.data_sharing      = true;   // must match the producer's advertised setting

    rc::media::Image360Subscriber sub;
    if (not sub.init(cfg))
    {
        std::fprintf(stderr, "init failed (domain %u topic '%s')\n", domain, topic.c_str());
        return 2;
    }
    std::printf("subscribed: domain=%u topic='%s' data_sharing=%s\n",
                domain, topic.c_str(), sub.data_sharing_active() ? "ON" : "off");

    bool got = false;
    for (int i = 0; i < 100 and not got; ++i)
    {
        sub.wait_and_poll([&](const rc::media::Image360Frame& f, std::int64_t)
        {
            if (got or f.width() == 0 or f.height() == 0)
                return;
            const int w = static_cast<int>(f.width()), h = static_cast<int>(f.height());
            const std::uint32_t need = f.width() * f.height() * 3u;
            if (f.size() < need)
            {
                std::fprintf(stderr, "short frame: size=%u need=%u\n", f.size(), need);
                return;
            }
            // Honour the tagged channel order exactly as retina does (media_plane_source.cpp:254);
            // guessing here would put a colour bug on top of whatever we are trying to measure.
            cv::Mat view(h, w, CV_8UC3, const_cast<unsigned char*>(f.data().data()));
            cv::Mat bgr;
            if (f.format() == rc::media::IMG360_FORMAT_RGB8)
                cv::cvtColor(view, bgr, cv::COLOR_RGB2BGR);
            else
                bgr = view.clone();
            cv::imwrite(out, bgr);
            std::printf("wrote %s  %dx%d  frame_id=%llu stamp_ms=%llu format=%u size=%u\n",
                        out.c_str(), w, h,
                        static_cast<unsigned long long>(f.frame_id()),
                        static_cast<unsigned long long>(f.stamp_ms()),
                        f.format(), f.size());
            got = true;
        }, 200);
    }
    if (not got)
        std::fprintf(stderr, "no frame in ~20 s\n");
    return got ? 0 : 1;
}
