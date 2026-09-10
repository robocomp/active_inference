#include <gtest/gtest.h>
#include "../src/scene_processor.h"

TEST(ProcessorTest, GetFrameTimestampMsWithRGBValid) {
    SceneProcessor processor(nullptr);
    processor.media_rgb_.valid = true;
    processor.media_rgb_.stamp = 123456789;

    EXPECT_EQ(processor.get_frame_timestamp_ms(), 123456789);
}

TEST(ProcessorTest, GetFrameTimestampMsWithDepthValid) {
    SceneProcessor processor(nullptr);
    processor.media_depth_.valid = true;
    processor.media_depth_.stamp = 987654321;

    EXPECT_EQ(processor.get_frame_timestamp_ms(), 987654321);
}

TEST(ProcessorTest, GetFrameTimestampMsWithNoValidSources) {
    SceneProcessor processor(nullptr);

    EXPECT_GT(processor.get_frame_timestamp_ms(), 0);
}