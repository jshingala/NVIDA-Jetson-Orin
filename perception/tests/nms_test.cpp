// Unit tests for duplicate removal. Build and run `nms_test`, or `ctest`.

#include "nms.h"

#include <cstdio>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            ++g_failures;                                                    \
        }                                                                    \
    } while (0)

using loci::Detection;
using loci::nms;

void overlapping_duplicate_removed() {
    const auto out = nms({{100, 100, 200, 160, 0.9f, 0}, {104, 102, 204, 162, 0.6f, 0}}, 0.7f);
    CHECK(out.size() == 1 && out[0].score == 0.9f);
}

void part_box_inside_whole_removed() {
    // one lens (small, inside) and the whole pair: IoU only ~0.19, but the lens is fully contained
    const auto out = nms({{100, 100, 260, 160, 0.5f, 3}, {110, 105, 160, 155, 0.8f, 3}}, 0.7f);
    CHECK(out.size() == 1 && out[0].score == 0.8f);
}

void separate_objects_kept() {
    const auto out = nms({{100, 100, 200, 160, 0.9f, 0}, {300, 100, 400, 160, 0.8f, 0}}, 0.7f);
    CHECK(out.size() == 2);
}

void different_classes_kept() {
    const auto out = nms({{100, 100, 200, 160, 0.9f, 0}, {110, 105, 160, 155, 0.8f, 1}}, 0.7f);
    CHECK(out.size() == 2);
}

void sorted_by_score() {
    const auto out = nms({{0, 0, 10, 10, 0.2f, 0}, {50, 50, 60, 60, 0.9f, 0}, {100, 100, 110, 110, 0.5f, 0}}, 0.7f);
    CHECK(out.size() == 3 && out[0].score == 0.9f && out[1].score == 0.5f && out[2].score == 0.2f);
}

}  // namespace

int main() {
    struct Test { const char* name; void (*fn)(); };
    const Test tests[] = {
        {"overlapping_duplicate_removed", overlapping_duplicate_removed},
        {"part_box_inside_whole_removed", part_box_inside_whole_removed},
        {"separate_objects_kept", separate_objects_kept},
        {"different_classes_kept", different_classes_kept},
        {"sorted_by_score", sorted_by_score},
    };
    for (const Test& t : tests) {
        const int before = g_failures;
        t.fn();
        std::printf("%s %s\n", g_failures == before ? "PASS" : "FAIL", t.name);
    }
    std::printf("%d failure(s)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
