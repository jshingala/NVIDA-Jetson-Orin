// ByteTrack-style multi-object tracker (Week 2 step 6).
//
// Gives each object a stable ID across frames. As in ByteTrack, confident
// detections are matched first and the leftover low-confidence ones are used
// only to keep existing tracks alive (which stops borderline objects from
// flickering in and out). Matching is class-aware and greedy: first on IoU
// against a constant-velocity prediction, then, for what is still unmatched,
// on 3D distance in metres. The 3D pass keeps an ID when the detector redraws
// the same object with a very different box (e.g. one lens instead of the
// whole pair of glasses), and a confident detection within `dup_dist_m` of an
// existing track of its class never starts a second track.
// No camera, CUDA or TensorRT dependency, so it is unit-tested on its own.

#pragma once

#include <vector>

namespace loci {

struct Box {
    float x1, y1, x2, y2;
};

struct Vec3 {
    float x = 0, y = 0, z = 0;  // camera frame, metres; z == 0 means unknown
};

struct TrackerInput {
    Box box;
    float score;
    int cls;
    Vec3 pos{};  // optional 3D position from depth
};

struct TrackerParams {
    float high_score = 0.35f;  // detections at or above this can start tracks (pass 1)
    float match_iou = 0.3f;    // pass 1 minimum IoU
    float low_match_iou = 0.5f;  // pass 2 (low-confidence detections) minimum IoU
    int min_hits = 3;          // consecutive hits before a new track is reported
    int max_lost = 30;         // frames a track survives without a match (1 s at 30 FPS)
    float max_dist_m = 0.10f;  // 3D pass: same object if the positions are this close
    float dup_dist_m = 0.10f;  // no new track this close to an existing one of the same class
};

struct TrackOutput {
    int id;
    int cls;
    Box box;          // latest matched box, or the prediction while lost
    Vec3 pos;         // latest known 3D position (z == 0 if never measured)
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
        Vec3 pos{};
        float vx = 0, vy = 0;  // centre velocity, pixels per frame
        float score = 0;
        int hits = 0;
        int lost = 0;
        int detection = -1;
    };

    enum class Match { kIou, kDistance };
    void associate(const std::vector<TrackerInput>& dets, std::vector<int>& pool, std::vector<bool>& det_used,
                   bool high_pass, Match by);
    static Box predict(const Track& t);
    static void correct(Track& t, const Box& measured, bool update_velocity);

    TrackerParams p_;
    std::vector<Track> tracks_;
    int next_id_ = 1;
};

float iou(const Box& a, const Box& b);
float distance(const Vec3& a, const Vec3& b);  // metres; a large value if either position is unknown

}  // namespace loci
