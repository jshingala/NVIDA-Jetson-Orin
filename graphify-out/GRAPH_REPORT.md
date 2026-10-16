# Graph Report - NVIDA-Jetson-Orin  (2026-10-16)

## Corpus Check
- Corpus is ~19,522 words - fits in a single context window. You may not need a graph.

## Summary
- 370 nodes · 589 edges · 21 communities (15 shown, 6 thin omitted)
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 49 edges (avg confidence: 0.83)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- CUDA Box Depth Kernel
- Working Rules and Download Lessons
- TensorRT Detector Runtime
- Architecture Decisions and Status
- Python Tooling Scripts
- Image Preprocessing and TensorRT Logger
- C++ Standard Library Includes
- NMS Duplicate Removal
- Tracker Unit Tests
- Track State
- Tracker Input Types
- Tracker Parameters
- Tracker Class
- Tracker Output
- Tracker Matching Functions
- Box Geometry
- Own-Object Dataset Plan
- DeepStream Setup Script
- Parcel3D Training Script
- Branch Naming Rule
- Commit Authorship Rule

## God Nodes (most connected - your core abstractions)
1. `Detector` - 36 edges
2. `Track` - 15 edges
3. `BoxRegion` - 13 edges
4. `Detection` - 13 edges
5. `TrackOutput` - 13 edges
6. `Tracker` - 13 edges
7. `BoxLocator` - 12 edges
8. `nms()` - 12 edges
9. `TrackerInput` - 12 edges
10. `main()` - 11 edges

## Surprising Connections (you probably didn't know these)
- `Parcel3D is synthetic; real-image accuracy unmeasured` --semantically_similar_to--> `Zero-shot YOLOE misnames classes`  [INFERRED] [semantically similar]
  README.md → docs/PROGRESS.md
- `One resident VLM within 8 GB shared memory` --semantically_similar_to--> `workers=1 batch=2 PIN_MEMORY=False to stay within 8 GB`  [INFERRED] [semantically similar]
  docs/CLAUDE.md → tools/parcel3d/README.md
- `Say upfront whether a speed-up trick will help` --semantically_similar_to--> `GPU depth is opt-in; CPU path is the default`  [INFERRED] [semantically similar]
  docs/CLAUDE.md → perception/README.md
- `PyTorch to ONNX to TensorRT speed-up table` --references--> `Box/Parcel YOLO26n warm-up detector`  [EXTRACTED]
  docs/PROGRESS.md → README.md
- `Planned 256 GB NVMe for keyframes and swap` --conceptually_related_to--> `Install JetPack on microSD (no NVMe fitted)`  [AMBIGUOUS]
  docs/LOCI_SPEC.md → README.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Parcel3D convert, train, validate and export pipeline** — tools_parcel3d_readme_convert, tools_parcel3d_readme_train, tools_parcel3d_readme_val, tools_parcel3d_readme_export_engine, readme_parcel3d_detector [EXTRACTED 1.00]
- **Week 2 fast perception loop stages** — perception_readme_step1_capture, perception_readme_step3_live_detection, perception_readme_step4_preprocess_kernel, perception_readme_step5_depth_kernel, perception_readme_step6_tracker [INFERRED 0.85]
- **Loci gate sequence** — docs_loci_spec_gate_1, docs_loci_spec_gate_2, docs_loci_spec_gate_3, docs_loci_spec_gate_4 [EXTRACTED 1.00]

## Communities (21 total, 6 thin omitted)

### Community 0 - "CUDA Box Depth Kernel"
Cohesion: 0.06
Nodes (34): box_depth_kernel(), BoxLocator, d_depth_, d_points_, d_regions_, depth_bytes_, max_boxes_, stream_ (+26 more)

### Community 1 - "Working Rules and Download Lessons"
Cohesion: 0.06
Nodes (31): Parcel3D download incident (Oct 12 2026), ByteTrack-style tracker in C++, YOLOE open-vocabulary detector (fallback YOLO11n), Pipeline is camera-bound at about 29.8 FPS, DeepStream 9.1 pipeline running the YOLOE engine, Parcel3D YOLO26n training complete (Oct 13 2026), loci_capture app (capture.cpp), loci_core library (nms.cpp, tracker.cpp) (+23 more)

### Community 2 - "TensorRT Detector Runtime"
Cohesion: 0.05
Nodes (32): Detector, context_, device_buffers_, device_image_, device_image_bytes_, device_input_, device_pred_, engine_ (+24 more)

### Community 3 - "Architecture Decisions and Status"
Cohesion: 0.07
Nodes (24): Project status log (Gate 1 passed Oct 13, Week 2 started Oct 14), Agent and API layer: Python, FastAPI, MCP Python SDK, Loci architecture: C++/CUDA loop, memory store, VLM worker, agent, Loci design spec (LOCI_SPEC), 8-week build plan, Gate 1 (after week 1): platform spike, Gate 2 (after week 3): keys question answered from memory without VLM, Gate 3 (after week 6): measurement table filled by script (+16 more)

### Community 4 - "Python Tooling Scripts"
Cohesion: 0.10
Nodes (8): main(), suppress_duplicates(), yolo_lines(), box_depth(), main(), parse_args(), export_int8(), main()

### Community 5 - "Image Preprocessing and TensorRT Logger"
Cohesion: 0.11
Nodes (8): check_cuda(), Logger, ms_since(), plan_offset(), read_file(), launch_preprocess(), make_letterbox(), preprocess_kernel()

### Community 6 - "C++ Standard Library Includes"
Cohesion: 0.11
Nodes (4): main(), ms_between(), save_bmp(), NvDsInferParseYoloE()

### Community 7 - "NMS Duplicate Removal"
Cohesion: 0.15
Nodes (16): area(), Detection, cls, score, x1, x2, y1, y2 (+8 more)

### Community 8 - "Tracker Unit Tests"
Cohesion: 0.21
Nodes (14): box_shape_jump_keeps_id_in_3d(), classes_do_not_swap(), det(), det3(), duplicate_box_does_not_start_second_track(), far_apart_in_3d_stay_separate(), find_cls(), long_gap_gives_new_id() (+6 more)

### Community 9 - "Track State"
Cohesion: 0.18
Nodes (11): Track, box, cls, detection, hits, id, lost, pos (+3 more)

### Community 10 - "Tracker Input Types"
Cohesion: 0.22
Nodes (9): TrackerInput, box, cls, pos, score, Vec3, x, y (+1 more)

### Community 11 - "Tracker Parameters"
Cohesion: 0.22
Nodes (9): Tracker::Tracker(), TrackerParams, dup_dist_m, high_score, low_match_iou, match_iou, max_dist_m, max_lost (+1 more)

### Community 12 - "Tracker Class"
Cohesion: 0.25
Nodes (7): Match, kDistance, kIou, Tracker, next_id_, p_, tracks_

### Community 13 - "Tracker Output"
Cohesion: 0.25
Nodes (8): TrackOutput, box, cls, detection, id, lost_frames, pos, score

### Community 15 - "Box Geometry"
Cohesion: 0.40
Nodes (5): Box, x1, x2, y1, y2

## Ambiguous Edges - Review These
- `Install JetPack on microSD (no NVMe fitted)` → `Planned 256 GB NVMe for keyframes and swap`  [AMBIGUOUS]
  docs/LOCI_SPEC.md · relation: conceptually_related_to

## Knowledge Gaps
- **101 isolated node(s):** `ppx`, `ppy`, `fx`, `fy`, `model` (+96 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 170 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Detector` connect `TensorRT Detector Runtime` to `Image Preprocessing and TensorRT Logger`, `C++ Standard Library Includes`?**
  _High betweenness centrality (0.108) - this node is a cross-community bridge._
- **What connects `ppx`, `ppy`, `fx` to the rest of the system?**
  _101 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `CUDA Box Depth Kernel` be split into smaller, more focused modules?**
  _Cohesion score 0.05603864734299517 - nodes in this community are weakly interconnected._
- **Why does `main()` connect `CUDA Box Depth Kernel` to `C++ Standard Library Includes`, `NMS Duplicate Removal`, `Tracker Unit Tests`, `Tracker Input Types`, `Tracker Output`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **Should `Working Rules and Download Lessons` be split into smaller, more focused modules?**
  _Cohesion score 0.05893719806763285 - nodes in this community are weakly interconnected._
- **Why does `Detection` connect `NMS Duplicate Removal` to `CUDA Box Depth Kernel`, `Image Preprocessing and TensorRT Logger`?**
  _High betweenness centrality (0.041) - this node is a cross-community bridge._

### Low-confidence Hints
_AMBIGUOUS edges — the extractor was unsure. Verify before acting on these._

- **What is the exact relationship between `Install JetPack on microSD (no NVMe fitted)` and `Planned 256 GB NVMe for keyframes and swap`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._