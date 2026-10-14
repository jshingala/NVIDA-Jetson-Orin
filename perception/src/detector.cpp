#include "detector.h"

#include <NvInfer.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <numeric>
#include <stdexcept>

namespace loci {
namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) std::fprintf(stderr, "[TensorRT] %s\n", msg);
    }
};

Logger g_logger;

std::vector<char> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// Ultralytics writes [int32 little-endian length][JSON metadata] before the plan.
size_t plan_offset(const std::vector<char>& file) {
    if (file.size() < 5) return 0;
    int32_t len = 0;
    std::memcpy(&len, file.data(), 4);
    if (len > 0 && static_cast<size_t>(len) + 4 < file.size() && file[4] == '{') return 4 + len;
    return 0;
}

float iou(const Detection& a, const Detection& b) {
    const float ix = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    const float iy = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    const float inter = ix * iy;
    const float uni = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

}  // namespace

Detector::Detector(const std::string& engine_path, const std::string& classes_path) {
    std::ifstream classes(classes_path);
    if (!classes) throw std::runtime_error("cannot open " + classes_path);
    for (std::string line; std::getline(classes, line);) {
        if (!line.empty()) names_.push_back(line);
    }

    const std::vector<char> file = read_file(engine_path);
    const size_t offset = plan_offset(file);
    runtime_.reset(nvinfer1::createInferRuntime(g_logger));
    engine_.reset(runtime_->deserializeCudaEngine(file.data() + offset, file.size() - offset));
    if (!engine_) throw std::runtime_error("failed to load TensorRT engine " + engine_path);
    context_.reset(engine_->createExecutionContext());
    check_cuda(cudaStreamCreate(&stream_), "cudaStreamCreate");

    // Bind every I/O tensor; the mask outputs must have memory even though Loci ignores them.
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        const nvinfer1::Dims dims = engine_->getTensorShape(name);
        size_t count = 1;
        for (int d = 0; d < dims.nbDims; ++d) count *= static_cast<size_t>(dims.d[d]);

        void* buffer = nullptr;
        check_cuda(cudaMalloc(&buffer, count * sizeof(float)), "cudaMalloc");
        device_buffers_.push_back(buffer);
        context_->setTensorAddress(name, buffer);

        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            input_h_ = static_cast<int>(dims.d[2]);
            input_w_ = static_cast<int>(dims.d[3]);
            device_input_ = buffer;
        } else if (dims.nbDims == 3) {  // [1, 4 + classes + mask coefs, anchors]
            pred_channels_ = static_cast<int>(dims.d[1]);
            pred_anchors_ = static_cast<int>(dims.d[2]);
            device_pred_ = buffer;
        }
        std::printf("  tensor %-8s [", name);
        for (int d = 0; d < dims.nbDims; ++d) std::printf(d ? "x%ld" : "%ld", static_cast<long>(dims.d[d]));
        std::printf("]\n");
    }
    if (!device_input_ || !device_pred_) throw std::runtime_error("unexpected engine inputs/outputs");
    if (pred_channels_ < 4 + static_cast<int>(names_.size())) {
        throw std::runtime_error("classes.txt has more names than the engine has class scores");
    }
    host_input_.resize(static_cast<size_t>(3) * input_w_ * input_h_);
    host_pred_.resize(static_cast<size_t>(pred_channels_) * pred_anchors_);
}

Detector::~Detector() {
    for (void* buffer : device_buffers_) cudaFree(buffer);
    if (device_image_) cudaFree(device_image_);
    if (stream_) cudaStreamDestroy(stream_);
}

// CPU baseline: same output as the kernel, but nearest-neighbour when resizing
// (the camera gives 640x480, so at 640 input there is no resize and the two match exactly).
void Detector::preprocess_cpu(const uint8_t* bgr, int width, int height, int stride) {
    lb_ = make_letterbox(width, height, input_w_, input_h_);
    const size_t plane = static_cast<size_t>(input_w_) * input_h_;
    std::fill(host_input_.begin(), host_input_.end(), 114.0f / 255.0f);
    for (int y = 0; y < lb_.new_h; ++y) {
        const int sy = std::min(height - 1, static_cast<int>(y / lb_.scale));
        const uint8_t* row = bgr + static_cast<size_t>(sy) * stride;
        const size_t out = static_cast<size_t>(y + lb_.pad_y) * input_w_ + lb_.pad_x;
        for (int x = 0; x < lb_.new_w; ++x) {
            const uint8_t* px = row + 3 * std::min(width - 1, static_cast<int>(x / lb_.scale));
            host_input_[out + x] = px[2] / 255.0f;              // R
            host_input_[plane + out + x] = px[1] / 255.0f;      // G
            host_input_[2 * plane + out + x] = px[0] / 255.0f;  // B
        }
    }
    check_cuda(cudaMemcpyAsync(device_input_, host_input_.data(), host_input_.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream_), "copy input");
}

// Upload the raw 8-bit frame (0.9 MB at 640x480 instead of 4.9 MB of floats)
// and let the kernel write the TensorRT input directly.
void Detector::preprocess_gpu(const uint8_t* bgr, int width, int height, int stride) {
    lb_ = make_letterbox(width, height, input_w_, input_h_);
    const size_t bytes = static_cast<size_t>(stride) * height;
    if (bytes > device_image_bytes_) {
        if (device_image_) cudaFree(device_image_);
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_image_), bytes), "cudaMalloc image");
        device_image_bytes_ = bytes;
    }
    check_cuda(cudaMemcpyAsync(device_image_, bgr, bytes, cudaMemcpyHostToDevice, stream_), "copy image");
    launch_preprocess(device_image_, width, height, stride, static_cast<float*>(device_input_),
                      input_w_, input_h_, lb_, stream_);
    check_cuda(cudaGetLastError(), "preprocess kernel");
}

float Detector::check_preprocess(const uint8_t* bgr, int width, int height, int stride) {
    preprocess_gpu(bgr, width, height, stride);
    std::vector<float> gpu(host_input_.size());
    check_cuda(cudaMemcpyAsync(gpu.data(), device_input_, gpu.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_), "copy back");
    check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    preprocess_cpu(bgr, width, height, stride);
    check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    float worst = 0.0f;
    for (size_t i = 0; i < gpu.size(); ++i) worst = std::max(worst, std::fabs(gpu[i] - host_input_[i]));
    return worst;
}

std::vector<Detection> Detector::decode(int width, int height, float conf, float iou_thresh) const {
    const int num_classes = static_cast<int>(names_.size());
    const float* p = host_pred_.data();
    const int n = pred_anchors_;

    std::vector<Detection> candidates;
    for (int a = 0; a < n; ++a) {
        int best = 0;
        float best_score = p[4 * n + a];
        for (int c = 1; c < num_classes; ++c) {
            const float s = p[(4 + c) * n + a];
            if (s > best_score) {
                best_score = s;
                best = c;
            }
        }
        if (best_score < conf) continue;

        const float cx = p[a], cy = p[n + a], w = p[2 * n + a], h = p[3 * n + a];
        auto to_image_x = [&](float v) { return std::clamp((v - lb_.pad_x) / lb_.scale, 0.0f, width - 1.0f); };
        auto to_image_y = [&](float v) { return std::clamp((v - lb_.pad_y) / lb_.scale, 0.0f, height - 1.0f); };
        candidates.push_back({to_image_x(cx - w / 2), to_image_y(cy - h / 2),
                              to_image_x(cx + w / 2), to_image_y(cy + h / 2), best_score, best});
    }

    // Per-class NMS, highest score first.
    std::sort(candidates.begin(), candidates.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<Detection> kept;
    std::vector<bool> removed(candidates.size(), false);
    for (size_t i = 0; i < candidates.size() && kept.size() < 300; ++i) {
        if (removed[i]) continue;
        kept.push_back(candidates[i]);
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (!removed[j] && candidates[j].cls == candidates[i].cls && iou(candidates[i], candidates[j]) > iou_thresh) {
                removed[j] = true;
            }
        }
    }
    return kept;
}

std::vector<Detection> Detector::detect(const uint8_t* bgr, int width, int height, int stride,
                                        float conf, float iou_thresh, DetectTimings* timings) {
    Clock::time_point t = Clock::now();
    if (gpu_pre_) {
        preprocess_gpu(bgr, width, height, stride);
    } else {
        preprocess_cpu(bgr, width, height, stride);
    }
    check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");  // so pre and infer are timed apart
    const double pre_ms = ms_since(t);

    t = Clock::now();
    if (!context_->enqueueV3(stream_)) throw std::runtime_error("TensorRT enqueueV3 failed");
    check_cuda(cudaMemcpyAsync(host_pred_.data(), device_pred_, host_pred_.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_), "copy output");
    check_cuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    const double infer_ms = ms_since(t);

    t = Clock::now();
    std::vector<Detection> result = decode(width, height, conf, iou_thresh);
    if (timings) *timings = {pre_ms, infer_ms, ms_since(t)};
    return result;
}

}  // namespace loci
