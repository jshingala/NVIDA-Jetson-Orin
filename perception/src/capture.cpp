// Loci perception, step 1: aligned colour + depth capture from the D415.
//
// Streams 640x480 colour and depth at 30 FPS, aligns depth to the colour
// image, and once a second prints the capture rate, the time spent aligning
// and the distance at the image centre.
//
//   ./loci_capture [seconds]      (default 10)

#include <librealsense2/rs.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>

namespace {

constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr int kFps = 30;
constexpr int kWarmupFrames = 30;  // first frames often have no depth while auto-exposure settles

using Clock = std::chrono::steady_clock;

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

}  // namespace

int main(int argc, char** argv) try {
    const double run_seconds = argc > 1 ? std::atof(argv[1]) : 10.0;

    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, kWidth, kHeight, RS2_FORMAT_BGR8, kFps);
    cfg.enable_stream(RS2_STREAM_DEPTH, kWidth, kHeight, RS2_FORMAT_Z16, kFps);

    rs2::pipeline pipe;
    rs2::pipeline_profile profile = pipe.start(cfg);
    rs2::device dev = profile.get_device();
    const float depth_scale = dev.first<rs2::depth_sensor>().get_depth_scale();
    std::printf("%s (serial %s, USB %s), depth scale %.6f m/unit\n",
                dev.get_info(RS2_CAMERA_INFO_NAME),
                dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER),
                dev.supports(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR)
                    ? dev.get_info(RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR) : "?",
                depth_scale);

    rs2::align align_to_color(RS2_STREAM_COLOR);
    for (int i = 0; i < kWarmupFrames; ++i) pipe.wait_for_frames();

    long total_frames = 0;
    double total_align_ms = 0.0;
    int window_frames = 0;
    double window_align_ms = 0.0;

    const Clock::time_point start = Clock::now();
    Clock::time_point window_start = start;

    while (ms_between(start, Clock::now()) < run_seconds * 1000.0) {
        rs2::frameset frames = pipe.wait_for_frames();

        const Clock::time_point t0 = Clock::now();
        rs2::frameset aligned = align_to_color.process(frames);
        const Clock::time_point t1 = Clock::now();

        rs2::depth_frame depth = aligned.get_depth_frame();
        rs2::video_frame color = aligned.get_color_frame();
        if (!depth || !color) continue;

        const double align_ms = ms_between(t0, t1);
        total_align_ms += align_ms;
        window_align_ms += align_ms;
        ++total_frames;
        ++window_frames;

        const double window_ms = ms_between(window_start, Clock::now());
        if (window_ms >= 1000.0) {
            const float center_m = depth.get_distance(depth.get_width() / 2, depth.get_height() / 2);
            if (center_m > 0.0f) {
                std::printf("%5.1f FPS | align %5.2f ms | centre %.3f m\n",
                            window_frames * 1000.0 / window_ms, window_align_ms / window_frames, center_m);
            } else {
                std::printf("%5.1f FPS | align %5.2f ms | centre: no depth\n",
                            window_frames * 1000.0 / window_ms, window_align_ms / window_frames);
            }
            window_frames = 0;
            window_align_ms = 0.0;
            window_start = Clock::now();
        }
    }

    pipe.stop();
    const double elapsed_s = ms_between(start, Clock::now()) / 1000.0;
    if (total_frames > 0) {
        std::printf("\nDone: %ld frames in %.1f s (%.1f FPS), mean align %.2f ms\n",
                    total_frames, elapsed_s, total_frames / elapsed_s, total_align_ms / total_frames);
    }
    return EXIT_SUCCESS;
} catch (const rs2::error& e) {
    std::fprintf(stderr, "RealSense error in %s(%s): %s\n",
                 e.get_failed_function().c_str(), e.get_failed_args().c_str(), e.what());
    return EXIT_FAILURE;
} catch (const std::exception& e) {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return EXIT_FAILURE;
}
