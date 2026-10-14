// TensorRT detector for the YOLOE engine exported by tools/yoloe/export_yoloe.py.
//
// Preprocessing runs in a fused CUDA kernel by default (preprocess.cu); the
// CPU version is kept as the baseline it is measured and checked against.

#pragma once

#include "preprocess.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
}  // namespace nvinfer1

namespace loci {

struct Detection {
    float x1, y1, x2, y2;  // pixels in the original image
    float score;
    int cls;
};

struct DetectTimings {
    double pre_ms = 0.0;    // letterbox + normalise (GPU kernel incl. upload, or CPU)
    double infer_ms = 0.0;  // copy in, TensorRT, copy out
    double post_ms = 0.0;   // decode + NMS (CPU)
};

class Detector {
public:
    Detector(const std::string& engine_path, const std::string& classes_path);
    ~Detector();
    Detector(const Detector&) = delete;
    Detector& operator=(const Detector&) = delete;

    // bgr: packed 8-bit BGR rows, `stride` bytes apart.
    std::vector<Detection> detect(const uint8_t* bgr, int width, int height, int stride,
                                  float conf, float iou, DetectTimings* timings = nullptr);

    // Largest absolute difference between the CPU and GPU preprocessed inputs for one frame.
    float check_preprocess(const uint8_t* bgr, int width, int height, int stride);

    void set_gpu_preprocess(bool on) { gpu_pre_ = on; }
    bool gpu_preprocess() const { return gpu_pre_; }
    const std::vector<std::string>& class_names() const { return names_; }
    int input_size() const { return input_w_; }

private:
    void preprocess_cpu(const uint8_t* bgr, int width, int height, int stride);
    void preprocess_gpu(const uint8_t* bgr, int width, int height, int stride);
    std::vector<Detection> decode(int width, int height, float conf, float iou) const;

    std::vector<std::string> names_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_ = nullptr;

    int input_w_ = 0, input_h_ = 0;
    int pred_channels_ = 0, pred_anchors_ = 0;  // output0 is [1, channels, anchors]

    std::vector<float> host_input_;
    std::vector<float> host_pred_;
    std::vector<void*> device_buffers_;
    void* device_input_ = nullptr;
    void* device_pred_ = nullptr;
    uint8_t* device_image_ = nullptr;  // raw camera frame for the GPU kernel
    size_t device_image_bytes_ = 0;

    bool gpu_pre_ = true;
    Letterbox lb_;  // geometry of the last frame
};

}  // namespace loci
