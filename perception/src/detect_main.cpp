// Loci perception, step 3: live YOLOE detections with a 3D position in meters.
//
// Each frame: capture colour + depth, align depth to colour, run the TensorRT
// detector, then take the median depth of the middle of each box and
// deproject the box centre to (x, y, z) in the camera frame
// (x right, y down, z forward). Prints a summary once a second.
//
//   ./loci_detect ENGINE CLASSES [seconds] [--conf 0.15] [--cpu-pre] [--check]
//
// --cpu-pre  preprocess on the CPU instead of the CUDA kernel (baseline)
// --check    on the first frame, compare CPU and GPU preprocessing and print the largest difference

#include "detector.h"

#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace {

constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr int kFps = 30;
constexpr int kWarmupFrames = 30;
constexpr float kIou = 0.7f;

using Clock = std::chrono::steady_clock;

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Median depth (m) of the central half of the box, ignoring pixels with no reading.
float box_depth(const rs2::depth_frame& depth, float depth_scale, const loci::Detection& d) {
    const int w = depth.get_width();
    const int h = depth.get_height();
    const int x1 = std::clamp(static_cast<int>(d.x1 + (d.x2 - d.x1) / 4), 0, w - 1);
    const int x2 = std::clamp(static_cast<int>(d.x2 - (d.x2 - d.x1) / 4), x1 + 1, w);
    const int y1 = std::clamp(static_cast<int>(d.y1 + (d.y2 - d.y1) / 4), 0, h - 1);
    const int y2 = std::clamp(static_cast<int>(d.y2 - (d.y2 - d.y1) / 4), y1 + 1, h);

    const auto* z16 = static_cast<const uint16_t*>(depth.get_data());
    std::vector<uint16_t> values;
    values.reserve(static_cast<size_t>(x2 - x1) * (y2 - y1));
    for (int y = y1; y < y2; ++y) {
        for (int x = x1; x < x2; ++x) {
            if (const uint16_t v = z16[y * w + x]; v > 0) values.push_back(v);
        }
    }
    if (values.empty()) return 0.0f;
    auto mid = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), mid, values.end());
    return *mid * depth_scale;
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s ENGINE CLASSES [seconds] [--conf 0.15] [--cpu-pre] [--check]\n", argv[0]);
        return EXIT_FAILURE;
    }
    double run_seconds = 10.0;
    float conf = 0.15f;
    bool cpu_pre = false;
    bool check = false;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--conf") == 0 && i + 1 < argc) {
            conf = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(argv[i], "--cpu-pre") == 0) {
            cpu_pre = true;
        } else if (std::strcmp(argv[i], "--check") == 0) {
            check = true;
        } else {
            run_seconds = std::atof(argv[i]);
        }
    }

    // Load the engine before starting the camera, so a model problem shows up on its own.
    std::printf("Loading %s\n", argv[1]);
    loci::Detector detector(argv[1], argv[2]);
    detector.set_gpu_preprocess(!cpu_pre);
    const std::vector<std::string>& names = detector.class_names();
    std::printf("%zu classes, input %dx%d, conf %.2f, preprocess on %s\n", names.size(), detector.input_size(),
                detector.input_size(), conf, cpu_pre ? "CPU" : "GPU (CUDA kernel)");

    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, kWidth, kHeight, RS2_FORMAT_BGR8, kFps);
    cfg.enable_stream(RS2_STREAM_DEPTH, kWidth, kHeight, RS2_FORMAT_Z16, kFps);
    rs2::pipeline pipe;
    rs2::pipeline_profile profile = pipe.start(cfg);
    const float depth_scale = profile.get_device().first<rs2::depth_sensor>().get_depth_scale();
    // Depth is aligned to colour, so colour intrinsics deproject both.
    const rs2_intrinsics intr =
        profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();
    rs2::align align_to_color(RS2_STREAM_COLOR);
    for (int i = 0; i < kWarmupFrames; ++i) pipe.wait_for_frames();

    if (check) {
        rs2::video_frame c = align_to_color.process(pipe.wait_for_frames()).get_color_frame();
        const float diff = detector.check_preprocess(static_cast<const uint8_t*>(c.get_data()), c.get_width(),
                                                     c.get_height(), c.get_stride_in_bytes());
        std::printf("Preprocess check: max |GPU - CPU| = %.3g over %d values\n", diff,
                    3 * detector.input_size() * detector.input_size());
    }

    long total_frames = 0;
    double sum_pre = 0, sum_infer = 0, sum_post = 0, sum_depth = 0, sum_loop = 0;
    int window_frames = 0;
    const Clock::time_point start = Clock::now();
    Clock::time_point window_start = start;

    while (ms_between(start, Clock::now()) < run_seconds * 1000.0) {
        rs2::frameset aligned = align_to_color.process(pipe.wait_for_frames());
        const Clock::time_point loop_start = Clock::now();
        rs2::depth_frame depth = aligned.get_depth_frame();
        rs2::video_frame color = aligned.get_color_frame();
        if (!depth || !color) continue;

        loci::DetectTimings t;
        const std::vector<loci::Detection> dets =
            detector.detect(static_cast<const uint8_t*>(color.get_data()), color.get_width(),
                            color.get_height(), color.get_stride_in_bytes(), conf, kIou, &t);

        const Clock::time_point depth_start = Clock::now();
        struct Placed { const loci::Detection* det; float xyz[3]; };
        std::vector<Placed> placed;
        for (const loci::Detection& d : dets) {
            Placed p{&d, {0, 0, 0}};
            const float z = box_depth(depth, depth_scale, d);
            if (z > 0.0f) {
                const float pixel[2] = {(d.x1 + d.x2) / 2, (d.y1 + d.y2) / 2};
                rs2_deproject_pixel_to_point(p.xyz, &intr, pixel, z);
            }
            placed.push_back(p);
        }
        const double depth_ms = ms_between(depth_start, Clock::now());

        sum_pre += t.pre_ms;
        sum_infer += t.infer_ms;
        sum_post += t.post_ms;
        sum_depth += depth_ms;
        sum_loop += ms_between(loop_start, Clock::now());
        ++total_frames;
        ++window_frames;

        const double window_ms = ms_between(window_start, Clock::now());
        if (window_ms >= 1000.0) {
            std::printf("%5.1f FPS | pre %.1f  infer %.1f  post %.1f  depth %.2f ms | %zu objects\n",
                        window_frames * 1000.0 / window_ms, t.pre_ms, t.infer_ms, t.post_ms, depth_ms, dets.size());
            for (const Placed& p : placed) {
                if (p.xyz[2] > 0.0f) {
                    std::printf("    %-15s %.2f  x %+.2f  y %+.2f  z %.2f m\n", names[p.det->cls].c_str(),
                                p.det->score, p.xyz[0], p.xyz[1], p.xyz[2]);
                } else {
                    std::printf("    %-15s %.2f  (no depth)\n", names[p.det->cls].c_str(), p.det->score);
                }
            }
            window_frames = 0;
            window_start = Clock::now();
        }
    }

    const double elapsed_s = ms_between(start, Clock::now()) / 1000.0;
    pipe.stop();
    if (total_frames > 0) {
        const double n = static_cast<double>(total_frames);
        std::printf("\nDone: %ld frames in %.1f s (%.1f FPS)\n", total_frames, elapsed_s, n / elapsed_s);
        std::printf("Mean per frame: pre %.2f  infer %.2f  post %.2f  depth %.2f ms, total %.2f ms\n",
                    sum_pre / n, sum_infer / n, sum_post / n, sum_depth / n, sum_loop / n);
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
