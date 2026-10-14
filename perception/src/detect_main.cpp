// Loci perception, step 3: live YOLOE detections with a 3D position in meters.
//
// Each frame: capture colour + depth, align depth to colour, run the TensorRT
// detector, then take the median depth of the middle of each box and
// deproject the box centre to (x, y, z) in the camera frame
// (x right, y down, z forward). Prints a summary once a second.
//
//   ./loci_detect ENGINE CLASSES [seconds] [--conf 0.15] [--cpu-pre] [--gpu-depth] [--check]
//
// --cpu-pre    preprocess on the CPU instead of the CUDA kernel (baseline)
// --gpu-depth  box depth + 3D position in CUDA kernel 2 instead of the CPU
// --check      compare CPU and GPU preprocessing on the first frame, and with
//              --gpu-depth compare every GPU 3D position against librealsense

#include "depth3d.h"
#include "detector.h"
#include "tracker.h"

#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>

#include <algorithm>
#include <chrono>
#include <cmath>
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

// CPU baseline for kernel 2: exact median of the non-zero depth values in each
// region, deprojected with librealsense's own rs2_deproject_pixel_to_point.
std::vector<loci::Point3> locate_cpu(const uint16_t* z16, int width, float depth_scale,
                                     const rs2_intrinsics& intr, const std::vector<loci::BoxRegion>& regions) {
    std::vector<loci::Point3> points;
    std::vector<uint16_t> values;
    for (const loci::BoxRegion& r : regions) {
        values.clear();
        for (int y = r.y1; y < r.y2; ++y) {
            for (int x = r.x1; x < r.x2; ++x) {
                if (const uint16_t v = z16[y * width + x]; v > 0) values.push_back(v);
            }
        }
        loci::Point3 p{0.0f, 0.0f, 0.0f};
        if (!values.empty()) {
            auto mid = values.begin() + values.size() / 2;
            std::nth_element(values.begin(), mid, values.end());
            const float pixel[2] = {r.u, r.v};
            float xyz[3];
            rs2_deproject_pixel_to_point(xyz, &intr, pixel, *mid * depth_scale);
            p = {xyz[0], xyz[1], xyz[2]};
        }
        points.push_back(p);
    }
    return points;
}

}  // namespace

int main(int argc, char** argv) try {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s ENGINE CLASSES [seconds] [--conf 0.15] [--cpu-pre] [--gpu-depth] [--check]\n", argv[0]);
        return EXIT_FAILURE;
    }
    double run_seconds = 10.0;
    float conf = 0.15f;
    bool cpu_pre = false;
    bool check = false;
    bool gpu_depth = false;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--conf") == 0 && i + 1 < argc) {
            conf = static_cast<float>(std::atof(argv[++i]));
        } else if (std::strcmp(argv[i], "--cpu-pre") == 0) {
            cpu_pre = true;
        } else if (std::strcmp(argv[i], "--gpu-depth") == 0) {
            gpu_depth = true;
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
    std::printf("%zu classes, input %dx%d, conf %.2f, preprocess on %s, box depth on %s\n", names.size(),
                detector.input_size(), detector.input_size(), conf, cpu_pre ? "CPU" : "GPU (CUDA kernel)",
                gpu_depth ? "GPU (CUDA kernel)" : "CPU");
    loci::BoxLocator locator;
    loci::Tracker tracker;

    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, kWidth, kHeight, RS2_FORMAT_BGR8, kFps);
    cfg.enable_stream(RS2_STREAM_DEPTH, kWidth, kHeight, RS2_FORMAT_Z16, kFps);
    rs2::pipeline pipe;
    rs2::pipeline_profile profile = pipe.start(cfg);
    const float depth_scale = profile.get_device().first<rs2::depth_sensor>().get_depth_scale();
    // Depth is aligned to colour, so colour intrinsics deproject both.
    const rs2_intrinsics intr =
        profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>().get_intrinsics();
    const loci::CameraIntrinsics cam{intr.ppx, intr.ppy, intr.fx, intr.fy, static_cast<int>(intr.model),
                                     {intr.coeffs[0], intr.coeffs[1], intr.coeffs[2], intr.coeffs[3], intr.coeffs[4]}};
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
    double sum_track = 0;
    double sum_depth_busy = 0;  // depth time over frames that had at least one detection
    long busy_frames = 0, checked_boxes = 0;
    float worst_xyz_diff = 0.0f;
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

        const int dw = depth.get_width();
        const auto* z16 = static_cast<const uint16_t*>(depth.get_data());  // aligned depth is tightly packed
        std::vector<loci::BoxRegion> regions;
        for (const loci::Detection& d : dets) {
            regions.push_back(loci::central_region(d.x1, d.y1, d.x2, d.y2, dw, depth.get_height()));
        }
        const Clock::time_point depth_start = Clock::now();
        const std::vector<loci::Point3> points =
            gpu_depth ? locator.locate(z16, dw, depth.get_height(), depth_scale, cam, regions)
                      : locate_cpu(z16, dw, depth_scale, intr, regions);
        const double depth_ms = ms_between(depth_start, Clock::now());
        if (!dets.empty()) {
            sum_depth_busy += depth_ms;
            ++busy_frames;
        }
        if (check && gpu_depth && !dets.empty()) {
            const std::vector<loci::Point3> ref = locate_cpu(z16, dw, depth_scale, intr, regions);
            for (size_t i = 0; i < ref.size(); ++i) {
                worst_xyz_diff = std::max({worst_xyz_diff, std::fabs(points[i].x - ref[i].x),
                                           std::fabs(points[i].y - ref[i].y), std::fabs(points[i].z - ref[i].z)});
                ++checked_boxes;
            }
        }

        const Clock::time_point track_start = Clock::now();
        std::vector<loci::TrackerInput> inputs;
        for (size_t i = 0; i < dets.size(); ++i) {
            const loci::Detection& d = dets[i];
            inputs.push_back({{d.x1, d.y1, d.x2, d.y2}, d.score, d.cls, {points[i].x, points[i].y, points[i].z}});
        }
        const std::vector<loci::TrackOutput> tracks = tracker.update(inputs);
        sum_track += ms_between(track_start, Clock::now());

        sum_pre += t.pre_ms;
        sum_infer += t.infer_ms;
        sum_post += t.post_ms;
        sum_depth += depth_ms;
        sum_loop += ms_between(loop_start, Clock::now());
        ++total_frames;
        ++window_frames;

        const double window_ms = ms_between(window_start, Clock::now());
        if (window_ms >= 1000.0) {
            std::printf("%5.1f FPS | pre %.1f  infer %.1f  post %.1f  depth %.2f ms | %zu detections, %zu tracks\n",
                        window_frames * 1000.0 / window_ms, t.pre_ms, t.infer_ms, t.post_ms, depth_ms, dets.size(),
                        tracks.size());
            for (const loci::TrackOutput& tr : tracks) {
                std::printf("    #%-3d %-15s", tr.id, names[tr.cls].c_str());
                if (tr.lost_frames > 0) {
                    std::printf(" lost %2d frames", tr.lost_frames);
                } else {
                    std::printf(" %.2f          ", tr.score);
                }
                if (tr.pos.z > 0.0f) {
                    std::printf("  x %+.2f  y %+.2f  z %.2f m\n", tr.pos.x, tr.pos.y, tr.pos.z);
                } else {
                    std::printf("  (no depth yet)\n");
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
        std::printf("Tracker: %d track IDs started, %.3f ms per frame\n", tracker.tracks_started(), sum_track / n);
        if (busy_frames > 0) {
            std::printf("Box depth on %s: %.3f ms per frame with detections (%ld frames)\n",
                        gpu_depth ? "GPU" : "CPU", sum_depth_busy / busy_frames, busy_frames);
        }
        if (check && gpu_depth) {
            std::printf("Box depth check: max |GPU - librealsense| = %.3g m over %ld boxes\n", worst_xyz_diff,
                        checked_boxes);
        }
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
