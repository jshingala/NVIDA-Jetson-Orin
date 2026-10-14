// DeepStream nvinfer box parser for the Loci YOLOE-26s-seg engine.
//
// output0 is [4 + classes + 32 mask coefs, 8400] after nvinfer drops the batch
// dimension: rows cx, cy, w, h (network-input pixels), then one sigmoid score
// per class. Mask outputs are ignored. NMS is left to nvinfer (cluster-mode=2).
// LOCI_DS_DEBUG=1 prints the output layers and the best score once a second.

#include "nvdsinfer_custom_impl.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" bool NvDsInferParseYoloE(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
                                    NvDsInferNetworkInfo const& networkInfo,
                                    NvDsInferParseDetectionParams const& detectionParams,
                                    std::vector<NvDsInferObjectDetectionInfo>& objectList) {
    static const bool debug = std::getenv("LOCI_DS_DEBUG") != nullptr;
    static long calls = 0;
    const bool report = debug && calls++ % 30 == 0;
    if (report) {
        std::fprintf(stderr, "YoloE parser: call %ld, net %ux%u, %zu output layers\n", calls, networkInfo.width,
                     networkInfo.height, outputLayersInfo.size());
        for (const NvDsInferLayerInfo& l : outputLayersInfo) {
            std::fprintf(stderr, "  %s type %d dims", l.layerName ? l.layerName : "?", static_cast<int>(l.dataType));
            for (unsigned i = 0; i < l.inferDims.numDims; ++i) std::fprintf(stderr, " %u", l.inferDims.d[i]);
            std::fprintf(stderr, "\n");
        }
    }
    const NvDsInferLayerInfo* pred = nullptr;
    for (const NvDsInferLayerInfo& layer : outputLayersInfo) {
        if (layer.inferDims.numDims != 2) continue;
        if (!pred || (layer.layerName && std::strcmp(layer.layerName, "output0") == 0)) pred = &layer;
    }
    if (!pred) {
        std::fprintf(stderr, "YoloE parser: no [channels, anchors] output layer\n");
        return false;
    }
    if (pred->dataType != FLOAT) {
        std::fprintf(stderr, "YoloE parser: output %s is not FP32\n", pred->layerName);
        return false;
    }

    const int channels = static_cast<int>(pred->inferDims.d[0]);
    const int n = static_cast<int>(pred->inferDims.d[1]);
    const int num_classes = static_cast<int>(detectionParams.numClassesConfigured);
    if (channels < 4 + num_classes) {
        std::fprintf(stderr, "YoloE parser: %d channels but %d classes configured\n", channels, num_classes);
        return false;
    }

    const float* p = static_cast<const float*>(pred->buffer);
    const float net_w = static_cast<float>(networkInfo.width);
    const float net_h = static_cast<float>(networkInfo.height);
    const std::vector<float>& thresholds = detectionParams.perClassPreclusterThreshold;

    float top_score = 0.0f;
    int top_anchor = 0;
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
        if (best_score > top_score) {
            top_score = best_score;
            top_anchor = a;
        }
        const float thresh = best < static_cast<int>(thresholds.size()) ? thresholds[best] : 0.25f;
        if (best_score < thresh) continue;

        const float cx = p[a], cy = p[n + a], w = p[2 * n + a], h = p[3 * n + a];
        const float x1 = std::clamp(cx - w / 2, 0.0f, net_w - 1.0f);
        const float y1 = std::clamp(cy - h / 2, 0.0f, net_h - 1.0f);
        const float x2 = std::clamp(cx + w / 2, 0.0f, net_w - 1.0f);
        const float y2 = std::clamp(cy + h / 2, 0.0f, net_h - 1.0f);
        if (x2 <= x1 || y2 <= y1) continue;

        NvDsInferObjectDetectionInfo obj{};
        obj.classId = static_cast<unsigned int>(best);
        obj.left = x1;
        obj.top = y1;
        obj.width = x2 - x1;
        obj.height = y2 - y1;
        obj.detectionConfidence = best_score;
        objectList.push_back(obj);
    }
    if (report) {
        const int a = top_anchor;
        std::fprintf(stderr, "  best score %.3f at anchor %d (box %.0f %.0f %.0f %.0f), %zu objects >= threshold\n",
                     top_score, a, p[a], p[n + a], p[2 * n + a], p[3 * n + a], objectList.size());
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloE);
