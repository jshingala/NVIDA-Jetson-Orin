// Duplicate removal for detector output (non-maximum suppression), CPU only.

#pragma once

#include <vector>

namespace loci {

struct Detection {
    float x1, y1, x2, y2;  // pixels in the original image
    float score;
    int cls;
};

// Per-class NMS, highest score first. A lower-scored box of the same class is
// dropped when its IoU with a kept box is above `iou_thresh`, or when it lies
// mostly inside / around it (intersection over the smaller box above
// `contain_thresh`). The second rule removes part-boxes, e.g. one lens of a
// pair of glasses inside the box of the whole pair, which plain IoU keeps
// because their sizes differ.
std::vector<Detection> nms(std::vector<Detection> boxes, float iou_thresh, float contain_thresh = 0.8f,
                           size_t max_keep = 300);

}  // namespace loci
