#include "tracker.h"

#include <algorithm>
#include <tuple>

namespace loci {

float iou(const Box& a, const Box& b) {
    const float ix = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    const float iy = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    const float inter = ix * iy;
    const float uni = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

Tracker::Tracker(TrackerParams params) : p_(params) {}

Box Tracker::predict(const Track& t) {
    return {t.box.x1 + t.vx, t.box.y1 + t.vy, t.box.x2 + t.vx, t.box.y2 + t.vy};
}

// Smoothed centre velocity; the box itself snaps to the measurement.
void Tracker::correct(Track& t, const Box& m) {
    constexpr float kAlpha = 0.5f;
    const float dx = (m.x1 + m.x2 - t.box.x1 - t.box.x2) / 2;  // box already coasted while lost, so this is the residual
    const float dy = (m.y1 + m.y2 - t.box.y1 - t.box.y2) / 2;
    t.vx = kAlpha * dx + (1 - kAlpha) * t.vx;
    t.vy = kAlpha * dy + (1 - kAlpha) * t.vy;
    t.box = m;
}

// Greedy, class-aware matching on IoU between predicted track boxes and detections.
void Tracker::associate(const std::vector<TrackerInput>& dets, std::vector<int>& pool,
                        std::vector<bool>& det_used, bool high_pass) {
    const float min_iou = high_pass ? p_.match_iou : p_.low_match_iou;
    std::vector<std::tuple<float, int, int>> pairs;  // (iou, track index, detection index)
    for (int ti : pool) {
        const Box pred = predict(tracks_[ti]);
        for (int di = 0; di < static_cast<int>(dets.size()); ++di) {
            if (det_used[di] || dets[di].cls != tracks_[ti].cls) continue;
            if ((dets[di].score >= p_.high_score) != high_pass) continue;
            const float o = iou(pred, dets[di].box);
            if (o >= min_iou) pairs.emplace_back(o, ti, di);
        }
    }
    std::sort(pairs.begin(), pairs.end(), [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });

    std::vector<bool> track_done(tracks_.size(), false);
    for (const auto& [o, ti, di] : pairs) {
        if (track_done[ti] || det_used[di]) continue;
        track_done[ti] = true;
        det_used[di] = true;
        Track& t = tracks_[ti];
        correct(t, dets[di].box);
        t.score = dets[di].score;
        t.detection = di;
        t.lost = 0;
        ++t.hits;
    }
    pool.erase(std::remove_if(pool.begin(), pool.end(), [&](int ti) { return track_done[ti]; }), pool.end());
}

std::vector<TrackOutput> Tracker::update(const std::vector<TrackerInput>& dets) {
    for (Track& t : tracks_) t.detection = -1;
    std::vector<bool> det_used(dets.size(), false);

    // Pass 1: every live track against confident detections.
    std::vector<int> pool;
    for (int i = 0; i < static_cast<int>(tracks_.size()); ++i) pool.push_back(i);
    associate(dets, pool, det_used, true);

    // Pass 2: still-unmatched confirmed tracks against the low-confidence leftovers.
    std::vector<int> confirmed;
    for (int ti : pool) {
        if (tracks_[ti].id != 0) confirmed.push_back(ti);
    }
    associate(dets, confirmed, det_used, false);

    // Unmatched tracks: tentative ones die at once, confirmed ones coast on their prediction.
    for (Track& t : tracks_) {
        if (t.detection >= 0) continue;
        if (t.id == 0) {
            t.lost = p_.max_lost + 1;
        } else {
            t.box = predict(t);
            t.score = 0;
            ++t.lost;
        }
    }
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [&](const Track& t) { return t.lost > p_.max_lost; }),
                  tracks_.end());

    // New tentative tracks from confident detections nobody claimed.
    for (int di = 0; di < static_cast<int>(dets.size()); ++di) {
        if (det_used[di] || dets[di].score < p_.high_score) continue;
        Track t;
        t.cls = dets[di].cls;
        t.box = dets[di].box;
        t.score = dets[di].score;
        t.hits = 1;
        t.detection = di;
        tracks_.push_back(t);
    }

    std::vector<TrackOutput> out;
    for (Track& t : tracks_) {
        if (t.id == 0 && t.hits >= p_.min_hits) t.id = next_id_++;
        if (t.id != 0) out.push_back({t.id, t.cls, t.box, t.score, t.detection, t.lost});
    }
    return out;
}

}  // namespace loci
