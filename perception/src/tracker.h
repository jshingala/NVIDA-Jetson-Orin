// ByteTrack-style multi-object tracker (Week 2 step 6).
//
// Gives each object a stable ID across frames. Two association passes per
// frame, as in ByteTrack: confident detections first, then the leftover
// low-confidence ones are used only to keep existing tracks alive (which
// stops borderline objects from flickering in and out). Matching is
// class-aware and greedy on IoU against a constant-velocity prediction.
// No camera, CUDA or TensorRT dependency, so it is unit-tested on its own.

#pragma once

#include <vector>

namespace loci {

struct Box {
    float x1, y1, x2, y2;
};

struct TrackerInput {
    Box box;
    float score;
    int cls;
};

struct TrackerParams {
    float high_score = 0.35f;  // detections at or above this can start tracks (pass 1)
    float match_iou = 0.3f;    // pass 1 minimum IoU
    float low_match_iou = 0.5f;  // pass 2 (low-confidence detections) minimum IoU
    int min_hits = 3;          // consecutive hits before a new track is reported
    int max_lost = 30;         // frames a track survives without a match (1 s at 30 FPS)
};

struct TrackOutput {
    int id;
    int cls;
    Box box;          // latest matched box, or the prediction while lost
    float score;      // score of the matched detection, 0 while lost
    int detection;    // index into this frame's detections, -1 if not matched this frame
    int lost_frames;  // 0 when matched this frame
};

class Tracker {
public:
    explicit Tracker(TrackerParams params = {});

    // One call per frame. Returns every confirmed track (matched or briefly lost).
    std::vector<TrackOutput> update(const std::vector<TrackerInput>& detections);

    int tracks_started() const { return next_id_ - 1; }  // confirmed track IDs handed out so far

private:
    struct Track {
        int id = 0;  // 0 until confirmed
        int cls = 0;
        Box box{};
        float vx = 0, vy = 0;  // centre velocity, pixels per frame
        float score = 0;
        int hits = 0;
        int lost = 0;
        int detection = -1;
    };

    void associate(const std::vector<TrackerInput>& dets, std::vector<int>& track_ids_pool,
                   std::vector<bool>& det_used, bool high_pass);
    static Box predict(const Track& t);
    static void correct(Track& t, const Box& measured);

    TrackerParams p_;
    std::vector<Track> tracks_;
    int next_id_ = 1;
};

float iou(const Box& a, const Box& b);

}  // namespace loci
