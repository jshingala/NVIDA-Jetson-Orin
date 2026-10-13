# Loci: a spatial memory agent on Jetson Orin Nano

Loci gives AI agents a memory of the physical world. It runs entirely on a Jetson Orin Nano Super with a RealSense D415 depth camera, remembers what it saw and where (in meters), and answers plain-English questions such as "Where did I leave my keys?". No video leaves the board.

Full design and 8-week plan: [docs/LOCI_SPEC.md](docs/LOCI_SPEC.md). Step-by-step progress: [docs/PROGRESS.md](docs/PROGRESS.md).

## Status

Week 1, **Gate 1 (platform spike)**: in progress.

| Gate 1 item | Status |
|---|---|
| JetPack 7.2.1 (CUDA 13.2, TensorRT 10.16) | ✅ Installed from the Jetson ISO onto the microSD |
| Camera (librealsense from source) | 🟡 D415 detected, firmware 5.17.0.10, depth read from Python works; link is still USB 2.1 |
| Detector engine (TensorRT) | ✅ FP16 engine built on TensorRT 10.16: 4.79 ms GPU time per 640 px frame |
| VLM (Cosmos-Reason2-2B, 4-bit) | ✅ llama.cpp on the GPU: 37.8 tokens/s generation, 1,552 tokens/s prompt reading, 4.2 GB peak board RAM; describes live D415 photos |
| Headless memory baseline (tegrastats) | ✅ 583 MB used at idle without the desktop (1,340 MB with it); 6.9 GB available, about 3 W |

## Hardware and software

| Part | Version |
|---|---|
| Board | Jetson Orin Nano Super Developer Kit, 8 GB, MAXN SUPER power mode, headless over SSH |
| Storage | 233 GB microSD (no NVMe installed) |
| OS | JetPack 7.2.1 (Jetson Linux R39.2.1, kernel 6.8, Ubuntu 24.04) |
| CUDA / TensorRT | 13.2.86 / 10.16.2.10 |
| Camera | Intel RealSense D415, librealsense v2.59.1 built from source (RSUSB backend, CUDA on) |

## First results: Box/Parcel detector

A warm-up model before Loci's own detector: YOLO26n fine-tuned on the Roboflow [Parcels and Boxes](https://universe.roboflow.com/abl-okoo2/parcels-and-boxes-dv7sg/dataset/1) dataset (2 classes; 4,194 train / 1,190 val / 604 test images), 50 epochs at 640 px, trained on the Jetson itself.

**Accuracy and speed** (test split, 604 images, batch 1; JetPack 6.2.1, MAXN SUPER; Ultralytics timings, which include Python pre/post-processing):

| Model | mAP50 | mAP50-95 | Pre + inference + post (ms) |
|---|---|---|---|
| PyTorch FP32 | 0.946 | 0.808 | 1.9 + 30.1 + 2.7 |
| TensorRT FP16 | 0.946 | 0.800 | 3.4 + 12.1 + 5.3 |
| TensorRT INT8 (MinMax calibration) | 0.940 | 0.780 | 3.2 + 9.6 + 4.9 |

- INT8 inference is **3.1x faster** than PyTorch and 1.26x faster than FP16, for -2.8 points of mAP50-95.
- Python pre/post-processing is about **46%** of the INT8 end-to-end time. That's the cost Loci's fused CUDA preprocessing is meant to remove.
- On **JetPack 7.2.1** (TensorRT 10.16), `trtexec` measures the FP16 engine at **4.79 ms mean GPU compute**, 5.18 ms including copies, 208 inferences/s. This is GPU time only, so it isn't comparable with the Ultralytics numbers above.

**Live demo** ([tools/parcel_demo/d415_detect.py](tools/parcel_demo/d415_detect.py)): D415 + INT8 engine, headless, with distance and rough size on every box. It ran at **29.8 FPS** (limited by the camera's 30 FPS), with 9.6 ms inference.

## Damaged-parcel detector (Parcel3D)

YOLO26n trained on the Jetson itself (JetPack 7.2.1, PyTorch 2.14.1 + CUDA 13.0) on [Parcel3D](https://zenodo.org/records/8032204), a synthetic dataset of intact and damaged parcels (2 classes; 9,193 train / 2,000 val / 2,200 test images). 50 epochs at 640 px, batch 2, about 15.5 hours.

**Test split** (2,200 held-out images, batch 1; Ultralytics timings, which include Python pre/post-processing):

| Model | mAP50 | mAP50-95 (all) | mAP50-95 normal / damaged | Pre + inference + post (ms) |
|---|---|---|---|---|
| PyTorch FP32 | 0.995 | 0.980 | 0.986 / 0.973 | 1.2 + 26.8 + 2.8 |
| TensorRT 10.16 FP16 | 0.995 | 0.983 | 0.988 / 0.979 | 1.4 + 3.5 + 2.7 |

- TensorRT FP16 runs inference **7.7x faster** (3.5 ms vs 26.8 ms) and the whole pipeline about 4x faster, with no accuracy loss.
- FP16 scores slightly higher than PyTorch. The likely reason: the exported engine uses the standard detection head with NMS, while the PyTorch model uses YOLO26's NMS-free head. This isn't verified.
- Pre/post-processing in Python is now more than half of the 7.6 ms total.
- Test scores match validation (0.981), so the model isn't overfit to the validation set.
- These are synthetic images. Accuracy on real camera images hasn't been measured yet.

## Running the demo

Measured on JetPack 6.2.1 inside the `ultralytics/ultralytics:latest-jetson-jetpack6` container; the JetPack 7.2.1 path isn't verified yet.

```bash
sudo docker run -it --rm --runtime=nvidia --ipc=host --privileged \
  -v /dev:/dev -v ~/yolo:/yolo -w /yolo ultralytics/ultralytics:latest-jetson-jetpack6
pip install pyrealsense2==2.59.1
python3 d415_detect.py --model /yolo/mine/weights/best_int8.engine --seconds 30
```

It writes an annotated `demo.mp4` and prints FPS and latency every second. Load the TensorRT model **before** starting the RealSense pipeline; otherwise TensorRT's CUDA graph capture fails.

## Setup notes (JetPack 7.2.1 on Orin Nano)

- **ISO install:** in the GRUB menu, choose **"Install on microSD (for AGX Orin, Orin Nano/NX)"**. The default "Install on NVMe" fails at partitioning when no NVMe is fitted.
- **librealsense v2.59.1:** with `-DBUILD_GRAPHICAL_EXAMPLES=false`, cmake fails because `examples/CMakeLists.txt` always links `OpenGL::GL`. Build with `-DBUILD_EXAMPLES=false`; command-line tools such as `rs-enumerate-devices` still build. `scripts/setup_udev_rules.sh` needs `v4l-utils` installed first.
- **USB:** the dev kit's USB-C port is device-only (flashing), so the camera goes in a USB-A port. Check the link with `rs-enumerate-devices | grep -i "usb type"`; a USB 2 cable limits it to 2.1.

## Repository layout

```
docs/LOCI_SPEC.md        Design, tech stack, 8-week plan and gates
docs/PROGRESS.md         Progress diagram
tools/parcel_demo/       Live D415 + TensorRT detection demo
tools/parcel3d/          Parcel3D conversion, training and test commands
```
