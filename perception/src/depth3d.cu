#include "depth3d.h"

#include <stdexcept>
#include <string>

namespace loci {
namespace {

constexpr int kBins = 8192;  // one bin per raw depth unit (1 mm on the D415); larger values land in the last bin
constexpr int kThreads = 256;
constexpr int kBinsPerThread = kBins / kThreads;

// Same maths as rs2_deproject_pixel_to_point for the models a D415 colour stream uses.
__device__ void deproject(const CameraIntrinsics& in, float u, float v, float depth, Point3* out) {
    float x = (u - in.ppx) / in.fx;
    float y = (v - in.ppy) / in.fy;
    const float xo = x, yo = y;
    const bool distorted = in.coeffs[0] != 0 || in.coeffs[1] != 0 || in.coeffs[2] != 0 ||
                           in.coeffs[3] != 0 || in.coeffs[4] != 0;
    if (distorted && (in.model == 2 || in.model == 4)) {  // inverse / plain Brown-Conrady
        for (int i = 0; i < 10; ++i) {
            const float r2 = x * x + y * y;
            const float icdist = 1.0f / (1.0f + ((in.coeffs[4] * r2 + in.coeffs[1]) * r2 + in.coeffs[0]) * r2);
            const float xq = in.model == 2 ? x / icdist : x;
            const float yq = in.model == 2 ? y / icdist : y;
            const float dx = 2 * in.coeffs[2] * xq * yq + in.coeffs[3] * (r2 + 2 * xq * xq);
            const float dy = 2 * in.coeffs[3] * xq * yq + in.coeffs[2] * (r2 + 2 * yq * yq);
            x = (xo - dx) * icdist;
            y = (yo - dy) * icdist;
        }
    }
    out->x = depth * x;
    out->y = depth * y;
    out->z = depth;
}

// One block per box: shared-memory histogram of the box's depth values, then the median bin.
__global__ void box_depth_kernel(const uint16_t* __restrict__ depth, int width, float depth_scale,
                                 CameraIntrinsics intr, const BoxRegion* __restrict__ regions,
                                 Point3* __restrict__ points) {
    __shared__ unsigned int hist[kBins];
    __shared__ unsigned int partial[kThreads];
    const BoxRegion r = regions[blockIdx.x];

    for (int i = threadIdx.x; i < kBins; i += kThreads) hist[i] = 0;
    __syncthreads();

    const int w = r.x2 - r.x1;
    const int n = w * (r.y2 - r.y1);
    for (int i = threadIdx.x; i < n; i += kThreads) {
        const uint16_t v = depth[(r.y1 + i / w) * width + r.x1 + i % w];
        if (v > 0) atomicAdd(&hist[min(static_cast<int>(v), kBins - 1)], 1u);
    }
    __syncthreads();

    unsigned int sum = 0;
    for (int i = 0; i < kBinsPerThread; ++i) sum += hist[threadIdx.x * kBinsPerThread + i];
    partial[threadIdx.x] = sum;
    __syncthreads();

    if (threadIdx.x != 0) return;
    unsigned int total = 0;
    for (int t = 0; t < kThreads; ++t) total += partial[t];
    Point3 p{0.0f, 0.0f, 0.0f};
    if (total > 0) {
        // Element at sorted index total/2, the same one std::nth_element picks on the CPU.
        const unsigned int target = total / 2;
        unsigned int seen = 0;
        int group = 0;
        while (seen + partial[group] <= target) seen += partial[group++];
        int bin = group * kBinsPerThread;
        while (seen + hist[bin] <= target) seen += hist[bin++];
        deproject(intr, r.u, r.v, bin * depth_scale, &p);
    }
    points[blockIdx.x] = p;
}

void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

}  // namespace

BoxLocator::BoxLocator(int max_boxes) : max_boxes_(max_boxes) {
    check(cudaStreamCreate(&stream_), "cudaStreamCreate");
    check(cudaMalloc(reinterpret_cast<void**>(&d_regions_), max_boxes_ * sizeof(BoxRegion)), "cudaMalloc regions");
    check(cudaMalloc(reinterpret_cast<void**>(&d_points_), max_boxes_ * sizeof(Point3)), "cudaMalloc points");
}

BoxLocator::~BoxLocator() {
    cudaFree(d_depth_);
    cudaFree(d_regions_);
    cudaFree(d_points_);
    if (stream_) cudaStreamDestroy(stream_);
}

std::vector<Point3> BoxLocator::locate(const uint16_t* depth, int width, int height, float depth_scale,
                                       const CameraIntrinsics& intr, const std::vector<BoxRegion>& regions) {
    const int n = std::min(static_cast<int>(regions.size()), max_boxes_);
    std::vector<Point3> points(n);
    if (n == 0) return points;

    const size_t bytes = static_cast<size_t>(width) * height * sizeof(uint16_t);
    if (bytes > depth_bytes_) {
        cudaFree(d_depth_);
        check(cudaMalloc(reinterpret_cast<void**>(&d_depth_), bytes), "cudaMalloc depth");
        depth_bytes_ = bytes;
    }
    check(cudaMemcpyAsync(d_depth_, depth, bytes, cudaMemcpyHostToDevice, stream_), "copy depth");
    check(cudaMemcpyAsync(d_regions_, regions.data(), n * sizeof(BoxRegion), cudaMemcpyHostToDevice, stream_),
          "copy regions");
    box_depth_kernel<<<n, kThreads, 0, stream_>>>(d_depth_, width, depth_scale, intr, d_regions_, d_points_);
    check(cudaGetLastError(), "box_depth_kernel");
    check(cudaMemcpyAsync(points.data(), d_points_, n * sizeof(Point3), cudaMemcpyDeviceToHost, stream_),
          "copy points");
    check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize");
    return points;
}

}  // namespace loci
