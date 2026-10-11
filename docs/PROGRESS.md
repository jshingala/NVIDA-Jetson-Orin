# Progress

What has been done on the Jetson so far, and what comes next. Numbers are measured on this board.

```
 WHAT WE DID SO FAR (Oct 10-11, 2026)
 ════════════════════════════════════════════════════════════════

 1. SETUP — Jetson Orin Nano (JetPack 6.2.1, SD card)
    ├─ Installed JetPack packages ── CUDA 12.6, TensorRT 10.3
    ├─ Power mode 7W ──► MAXN SUPER
    ├─ Docker reinstalled + GPU access (test printed True)
    ├─ SSH from Mac, monitor removed
    └─ D415 camera works (depth 0.55 m, 15-30 FPS)
                │
                ▼
 2. DATASET — Roboflow "Parcels and Boxes" (2 classes: Box, Parcel)
    ├─ train 4,194 images ── the model learns from these
    ├─ valid 1,190 images ── scored after every epoch
    └─ test    604 images ── final check, never seen in training
                │
                ▼
 3. TRAINING — YOLO26n, 50 epochs, batch 2 (~8 h)
    ├─ Fixed: out-of-memory errors, data.yaml paths
    └─ Result (test): mAP50 0.946 | mAP50-95 0.808
                │
                ▼
 4. SPEED-UP — PyTorch ──► ONNX ──► TensorRT
    ┌────────────┬──────────┬───────────┐
    │ Version    │ mAP50-95 │ Inference │
    ├────────────┼──────────┼───────────┤
    │ PyTorch    │  0.808   │  30.1 ms  │
    │ TRT FP16   │  0.800   │  12.1 ms  │
    │ TRT INT8   │  0.780   │   9.6 ms  │ ◄── 3.1x faster
    └────────────┴──────────┴───────────┘
                │
                ▼
 5. LIVE DEMO — D415 + INT8 engine (tools/parcel_demo/d415_detect.py)
    ├─ 9.6 ms inference, ~30 FPS (limited by the camera)
    ├─ Distance + size label on every box
    └─ Saved demo.mp4
                │
                ▼
 ════════════════ YOU ARE HERE ═════════════════
                │
                ▼
 6. BACKUP ~/yolo ──► Mac (~/JetsonBackup)
                │
                ▼
 7. GATE 1 — JetPack 7.2.1 on NVMe (USB installer)
    └─ then: camera check, detector, VLM, memory baseline
                │
                ▼
 8. LOCI — weeks 2-8 (C++/CUDA loop, memory, VLM, agent)
```

Benchmark details: test split (604 images), batch 1, Ultralytics validation timings, JetPack 6.2.1, MAXN SUPER, TensorRT 10.7 inside the `ultralytics/ultralytics:latest-jetson-jetpack6` container. Full plan: [LOCI_SPEC.md](LOCI_SPEC.md).
