// Unit tests for the tracker. No framework needed: build and run `tracker_test`, or `ctest`.

#include "tracker.h"

#include <cstdio>
#include <set>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

using loci::Tracker;
using loci::TrackerInput;
using loci::TrackOutput;

TrackerInput det(float x, float y, float score = 0.8f, int cls = 0, float w = 40, float h = 30) {
    return {{x, y, x + w, y + h}, score, cls};
}

const TrackOutput* find_cls(const std::vector<TrackOutput>& out, int cls) {
    for (const TrackOutput& t : out) {
        if (t.cls == cls) return &t;
    }
    return nullptr;
}

void new_track_needs_min_hits() {
    Tracker tr;
    CHECK(tr.update({det(100, 100)}).empty());
    CHECK(tr.update({det(100, 100)}).empty());
    const auto out = tr.update({det(100, 100)});
    CHECK(out.size() == 1);
    CHECK(out.size() == 1 && out[0].id == 1 && out[0].detection == 0);
}

void static_object_keeps_id() {
    Tracker tr;
    std::set<int> ids;
    for (int f = 0; f < 100; ++f) {
        for (const TrackOutput& t : tr.update({det(100 + (f % 3), 100 - (f % 2))})) ids.insert(t.id);
    }
    CHECK(ids.size() == 1);
    CHECK(tr.tracks_started() == 1);
}

void short_gap_keeps_id() {
    Tracker tr;
    for (int f = 0; f < 5; ++f) tr.update({det(200, 150)});
    for (int f = 0; f < 10; ++f) {
        const auto out = tr.update({});
        CHECK(out.size() == 1 && out[0].detection == -1 && out[0].lost_frames == f + 1);
    }
    const auto out = tr.update({det(201, 151)});
    CHECK(out.size() == 1 && out[0].id == 1 && out[0].lost_frames == 0);
}

void long_gap_gives_new_id() {
    Tracker tr;
    for (int f = 0; f < 5; ++f) tr.update({det(200, 150)});
    for (int f = 0; f < 31; ++f) tr.update({});
    CHECK(tr.update({}).empty());
    for (int f = 0; f < 3; ++f) tr.update({det(200, 150)});
    const auto out = tr.update({det(200, 150)});
    CHECK(out.size() == 1 && out[0].id == 2);
}

void moving_object_keeps_id() {
    Tracker tr;
    std::set<int> ids;
    for (int f = 0; f < 60; ++f) {
        for (const TrackOutput& t : tr.update({det(50 + 12.0f * f, 100)})) ids.insert(t.id);  // 12 px/frame, box 40 wide
    }
    CHECK(ids.size() == 1);
}

void classes_do_not_swap() {
    Tracker tr;
    for (int f = 0; f < 5; ++f) tr.update({det(100, 100, 0.8f, 3), det(110, 105, 0.8f, 4)});
    const auto before = tr.update({det(100, 100, 0.8f, 3), det(110, 105, 0.8f, 4)});
    const auto after = tr.update({det(110, 105, 0.8f, 4), det(100, 100, 0.8f, 3)});  // order swapped
    const TrackOutput* g0 = find_cls(before, 3);
    const TrackOutput* g1 = find_cls(after, 3);
    CHECK(g0 && g1 && g0->id == g1->id);
    CHECK(g1 && g1->detection == 1);
}

void low_score_keeps_track_but_cannot_start_one() {
    Tracker tr;
    for (int f = 0; f < 5; ++f) tr.update({det(300, 200, 0.8f)});
    const auto out = tr.update({det(301, 200, 0.2f), det(10, 10, 0.2f)});  // weak detections
    CHECK(out.size() == 1 && out[0].id == 1 && out[0].detection == 0 && out[0].lost_frames == 0);
    for (int f = 0; f < 10; ++f) tr.update({det(10, 10, 0.2f)});
    CHECK(tr.tracks_started() == 1);  // the weak far-away box never became a track
}

void two_same_class_objects_stay_apart() {
    Tracker tr;
    std::set<int> ids;
    for (int f = 0; f < 50; ++f) {
        for (const TrackOutput& t : tr.update({det(100, 100), det(400, 100)})) ids.insert(t.id);
    }
    CHECK(ids.size() == 2);
}

}  // namespace

int main() {
    struct Test { const char* name; void (*fn)(); };
    const Test tests[] = {
        {"new_track_needs_min_hits", new_track_needs_min_hits},
        {"static_object_keeps_id", static_object_keeps_id},
        {"short_gap_keeps_id", short_gap_keeps_id},
        {"long_gap_gives_new_id", long_gap_gives_new_id},
        {"moving_object_keeps_id", moving_object_keeps_id},
        {"classes_do_not_swap", classes_do_not_swap},
        {"low_score_keeps_track_but_cannot_start_one", low_score_keeps_track_but_cannot_start_one},
        {"two_same_class_objects_stay_apart", two_same_class_objects_stay_apart},
    };
    for (const Test& t : tests) {
        const int before = g_failures;
        t.fn();
        std::printf("%s %s\n", g_failures == before ? "PASS" : "FAIL", t.name);
    }
    std::printf("%d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
