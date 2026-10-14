#include "nms.h"

#include <algorithm>

namespace loci {
namespace {

float area(const Detection& d) { return std::max(0.0f, d.x2 - d.x1) * std::max(0.0f, d.y2 - d.y1); }

float intersection(const Detection& a, const Detection& b) {
    const float ix = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    const float iy = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    return ix * iy;
}

}  // namespace

std::vector<Detection> nms(std::vector<Detection> boxes, float iou_thresh, float contain_thresh, size_t max_keep) {
    std::sort(boxes.begin(), boxes.end(), [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<Detection> kept;
    std::vector<bool> removed(boxes.size(), false);
    for (size_t i = 0; i < boxes.size() && kept.size() < max_keep; ++i) {
        if (removed[i]) continue;
        kept.push_back(boxes[i]);
        const float area_i = area(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (removed[j] || boxes[j].cls != boxes[i].cls) continue;
            const float inter = intersection(boxes[i], boxes[j]);
            const float area_j = area(boxes[j]);
            const float uni = area_i + area_j - inter;
            const float smaller = std::min(area_i, area_j);
            if ((uni > 0 && inter / uni > iou_thresh) || (smaller > 0 && inter / smaller > contain_thresh)) {
                removed[j] = true;
            }
        }
    }
    return kept;
}

}  // namespace loci
