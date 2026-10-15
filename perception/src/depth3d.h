// CUDA kernel 2 (Week 2 step 5): per-box robust depth + deprojection to 3D.
//
// For every detection: median of the non-zero depth values in the middle half
// of its box (a shared-memory histogram, exact for depths below 8.191 m at
// 1 mm units), then the box centre deprojected through the colour intrinsics,
// mirroring rs2_deproject_pixel_to_point.

#pragma once

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace loci {

// Subset of rs2_intrinsics; `model` uses the rs2_distortion values.
struct CameraIntrinsics {
    float ppx, ppy, fx, fy;
    int model;
    float coeffs[5];
};

// Pixels whose depth is sampled (half-open), plus the box centre to deproject.
struct BoxRegion {
    int x1, y1, x2, y2;
    float u, v;
};

struct Point3 {
    float x, y, z;  // metres, camera frame (x right, y down, z forward); z == 0 means no depth
};

// Middle half of a box, clamped to the image; used by both the CPU and GPU paths.
inline BoxRegion central_region(float x1, float y1, float x2, float y2, int width, int height) {
    BoxRegion r;
    r.x1 = std::clamp(static_cast<int>(x1 + (x2 - x1) / 4), 0, width - 1);
    r.x2 = std::clamp(static_cast<int>(x2 - (x2 - x1) / 4), r.x1 + 1, width);
    r.y1 = std::clamp(static_cast<int>(y1 + (y2 - y1) / 4), 0, height - 1);
    r.y2 = std::clamp(static_cast<int>(y2 - (y2 - y1) / 4), r.y1 + 1, height);
    r.u = (x1 + x2) / 2;
    r.v = (y1 + y2) / 2;
    return r;
}

class BoxLocator {
public:
    explicit BoxLocator(int max_boxes = 300);
    ~BoxLocator();
    BoxLocator(const BoxLocator&) = delete;
    BoxLocator& operator=(const BoxLocator&) = delete;

    // depth: host Z16 image (width x height, tightly packed). One point per region;
    // throws if there are more regions than `max_boxes` rather than dropping boxes.
    std::vector<Point3> locate(const uint16_t* depth, int width, int height, float depth_scale,
                               const CameraIntrinsics& intr, const std::vector<BoxRegion>& regions);

private:
    int max_boxes_;
    cudaStream_t stream_ = nullptr;
    uint16_t* d_depth_ = nullptr;
    size_t depth_bytes_ = 0;
    BoxRegion* d_regions_ = nullptr;
    Point3* d_points_ = nullptr;
};

}  // namespace loci
