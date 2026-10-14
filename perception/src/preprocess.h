// Fused preprocessing kernel (Week 2 step 4, kernel 1).

#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>

namespace loci {

struct Letterbox {
    float scale = 1.0f;     // resize factor applied to the source image
    int pad_x = 0, pad_y = 0;  // left/top padding in the network input
    int new_w = 0, new_h = 0;  // size of the resized image inside the input
};

// Ultralytics-style letterbox geometry: keep aspect ratio, centre the image.
Letterbox make_letterbox(int src_w, int src_h, int dst_w, int dst_h);

// One pass on the GPU: bilinear letterbox with grey (114) padding, BGR -> RGB,
// scale to [0, 1] and HWC uint8 -> CHW float, written straight into `dst`
// (the TensorRT input buffer, 3 x dst_h x dst_w).
void launch_preprocess(const uint8_t* src_bgr, int src_w, int src_h, int src_pitch,
                       float* dst, int dst_w, int dst_h, const Letterbox& lb, cudaStream_t stream);

}  // namespace loci
