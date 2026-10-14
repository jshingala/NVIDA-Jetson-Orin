#include "preprocess.h"

#include <cmath>

namespace loci {
namespace {

__global__ void preprocess_kernel(const uint8_t* __restrict__ src, int src_w, int src_h, int pitch,
                                  float* __restrict__ dst, int dst_w, int dst_h,
                                  float inv_scale, int pad_x, int pad_y, int new_w, int new_h) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    const int plane = dst_w * dst_h;
    const int out = y * dst_w + x;
    const int cx = x - pad_x;
    const int cy = y - pad_y;
    if (cx < 0 || cy < 0 || cx >= new_w || cy >= new_h) {
        const float grey = 114.0f / 255.0f;
        dst[out] = grey;
        dst[plane + out] = grey;
        dst[2 * plane + out] = grey;
        return;
    }

    // Pixel-centre mapping, as in cv2.resize(INTER_LINEAR). At scale 1 it lands exactly on a pixel.
    const float sx = fminf(fmaxf((cx + 0.5f) * inv_scale - 0.5f, 0.0f), src_w - 1.0f);
    const float sy = fminf(fmaxf((cy + 0.5f) * inv_scale - 0.5f, 0.0f), src_h - 1.0f);
    const int x0 = static_cast<int>(sx);
    const int y0 = static_cast<int>(sy);
    const int x1 = min(x0 + 1, src_w - 1);
    const int y1 = min(y0 + 1, src_h - 1);
    const float fx = sx - x0;
    const float fy = sy - y0;

    const uint8_t* p00 = src + y0 * pitch + 3 * x0;
    const uint8_t* p01 = src + y0 * pitch + 3 * x1;
    const uint8_t* p10 = src + y1 * pitch + 3 * x0;
    const uint8_t* p11 = src + y1 * pitch + 3 * x1;
    float v[3];
    for (int c = 0; c < 3; ++c) {
        v[c] = (1.0f - fy) * ((1.0f - fx) * p00[c] + fx * p01[c]) + fy * ((1.0f - fx) * p10[c] + fx * p11[c]);
    }
    dst[out] = v[2] / 255.0f;              // R
    dst[plane + out] = v[1] / 255.0f;      // G
    dst[2 * plane + out] = v[0] / 255.0f;  // B
}

}  // namespace

Letterbox make_letterbox(int src_w, int src_h, int dst_w, int dst_h) {
    Letterbox lb;
    lb.scale = fminf(static_cast<float>(dst_w) / src_w, static_cast<float>(dst_h) / src_h);
    lb.new_w = static_cast<int>(std::round(src_w * lb.scale));
    lb.new_h = static_cast<int>(std::round(src_h * lb.scale));
    lb.pad_x = (dst_w - lb.new_w) / 2;
    lb.pad_y = (dst_h - lb.new_h) / 2;
    return lb;
}

void launch_preprocess(const uint8_t* src_bgr, int src_w, int src_h, int src_pitch,
                       float* dst, int dst_w, int dst_h, const Letterbox& lb, cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
    preprocess_kernel<<<grid, block, 0, stream>>>(src_bgr, src_w, src_h, src_pitch, dst, dst_w, dst_h,
                                                  1.0f / lb.scale, lb.pad_x, lb.pad_y, lb.new_w, lb.new_h);
}

}  // namespace loci
