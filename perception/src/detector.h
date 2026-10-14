// TensorRT detector for the YOLOE engine exported by tools/yoloe/export_yoloe.py.
//
// Step 3 baseline: letterbox and box decoding run on the CPU so the CUDA
// kernels in step 4 have something correct to be measured against.

#pragma once

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
    double pre_ms = 0.0;    // letterbox + normalise (CPU)
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

    const std::vector<std::string>& class_names() const { return names_; }
    int input_size() const { return input_w_; }

private:
    void letterbox(const uint8_t* bgr, int width, int height, int stride);
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

    // letterbox parameters of the last frame
    float scale_ = 1.0f;
    int pad_x_ = 0, pad_y_ = 0;
};

}  // namespace loci
