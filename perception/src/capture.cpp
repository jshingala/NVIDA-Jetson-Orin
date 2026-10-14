// Loci perception, step 1: aligned colour + depth capture from the D415.
//
// Streams 640x480 colour and depth at 30 FPS, aligns depth to the colour
// image, and once a second prints the capture rate, the time spent aligning
// and the distance at the image centre.
//
//   ./loci_capture [seconds] [--save DIR] [--every N]
//
// --save writes every Nth colour frame (default every 30th, one per second)
// to DIR as 24-bit BMP, e.g. for detector checks and INT8 calibration.

#include <librealsense2/rs.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr int kFps = 30;
constexpr int kWarmupFrames = 30;  // first frames often have no depth while auto-exposure settles

using Clock = std::chrono::steady_clock;

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// 24-bit BMP stores BGR rows bottom-up, each padded to a multiple of 4 bytes.
void save_bmp(const rs2::video_frame& color, const std::string& path) {
    const int w = color.get_width();
    const int h = color.get_height();
    const int stride = color.get_stride_in_bytes();
    const int row_bytes = (w * 3 + 3) & ~3;
    const uint32_t image_bytes = static_cast<uint32_t>(row_bytes) * h;

    unsigned char header[54] = {'B', 'M'};
    auto put32 = [&header](int at, uint32_t v) {
        for (int i = 0; i < 4; ++i) header[at + i] = static_cast<unsigned char>(v >> (8 * i));
    };
    put32(2, 54 + image_bytes);  // file size
    put32(10, 54);               // pixel data offset
    put32(14, 40);               // DIB header size
    put32(18, static_cast<uint32_t>(w));
    put32(22, static_cast<uint32_t>(h));
    header[26] = 1;              // planes
    header[28] = 24;             // bits per pixel
    put32(34, image_bytes);

    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(header), sizeof(header));
    const auto* data = static_cast<const char*>(color.get_data());
    const std::vector<char> pad(static_cast<size_t>(row_bytes - w * 3), 0);
    for (int y = h - 1; y >= 0; --y) {
        out.write(data + static_cast<size_t>(y) * stride, w * 3);
        out.write(pad.data(), static_cast<std::streamsize>(pad.size()));
    }
}

}  // namespace

int main(int argc, char** argv) try {
    double run_seconds = 10.0;
    std::string save_dir;
    int save_every = 30;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            save_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--every") == 0 && i + 1 < argc) {
            save_every = std::max(1, std::atoi(argv[++i]));
        } else {
            run_seconds = std::atof(argv[i]);
        }
    }
    if (!save_dir.empty()) std::filesystem::create_directories(save_dir);

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
    int saved = 0;
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

        if (!save_dir.empty() && total_frames % save_every == 0) {
            char name[32];
            std::snprintf(name, sizeof(name), "frame_%05d.bmp", saved++);
            save_bmp(color, save_dir + "/" + name);
        }

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

    const double elapsed_s = ms_between(start, Clock::now()) / 1000.0;  // before stop(), which takes ~2 s
    pipe.stop();
    if (total_frames > 0) {
        std::printf("\nDone: %ld frames in %.1f s (%.1f FPS), mean align %.2f ms\n",
                    total_frames, elapsed_s, total_frames / elapsed_s, total_align_ms / total_frames);
    }
    if (!save_dir.empty()) std::printf("Saved %d colour frames to %s\n", saved, save_dir.c_str());
    return EXIT_SUCCESS;
} catch (const rs2::error& e) {
    std::fprintf(stderr, "RealSense error in %s(%s): %s\n",
                 e.get_failed_function().c_str(), e.get_failed_args().c_str(), e.what());
    return EXIT_FAILURE;
} catch (const std::exception& e) {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return EXIT_FAILURE;
}
