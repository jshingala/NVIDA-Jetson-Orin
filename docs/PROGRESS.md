# Progress

What has been done on the Jetson so far, and what comes next. Numbers are measured on this board.

```
 WHAT WE DID SO FAR (Oct 10-12, 2026)
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
 6. BACKUP ~/yolo ──► Mac Desktop (JetsonBackup, 1 GB)
                │
                ▼
 7. JETPACK 7.2.1 — fresh install (Oct 12, 2026)
    ├─ Jetson ISO from USB stick ──► 233 GB microSD (no NVMe in board)
    │    └─ 1st try failed: picked "Install on NVMe"; microSD option worked
    ├─ L4T R39.2.1, kernel 6.8.12-tegra, Ubuntu 24.04
    ├─ nvidia-jetpack 7.2.1-b49 ── CUDA 13.2.86, TensorRT 10.16.2.10
    ├─ Power mode ──► MAXN SUPER
    └─ ~/yolo copied back from Mac (scp + rsync); old .engine files
       need rebuilding for TensorRT 10.16 (best.onnx is reusable)
                │
                ▼
 ════════════════ YOU ARE HERE ═════════════════
                │
                ▼
 8. GATE 1 — platform spike on JetPack 7.2.1
    ├─ [~ ] Camera: librealsense v2.59.1 built; D415 seen, FW 5.17.0.10,
    │        but USB 2.1 link (USB 3 cable arriving Oct 13)
    ├─ [x] Detector: TensorRT 10.16 FP16 engine from best.onnx
    │        4.79 ms GPU compute, 208 inferences/s (trtexec)
    ├─ [  ] VLM: Cosmos-Reason2-2B INT4 runs on its own
    └─ [  ] Headless memory baseline (tegrastats)
                │
                ▼
 9. LOCI — weeks 2-8 (C++/CUDA loop, memory, VLM, agent)
```

Benchmark details: test split (604 images), batch 1, Ultralytics validation timings, JetPack 6.2.1, MAXN SUPER, TensorRT 10.7 inside the `ultralytics/ultralytics:latest-jetson-jetpack6` container. Full plan: [LOCI_SPEC.md](LOCI_SPEC.md).
