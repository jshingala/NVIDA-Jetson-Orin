# Perception (C++)

The fast loop from `docs/LOCI_SPEC.md`: D415 capture, detector, tracker and depth-to-3D, built up step by step in Week 2.

## Step 1: aligned colour + depth capture

`loci_capture` streams 640x480 colour and depth at 30 FPS, aligns depth to colour and prints the frame rate, the align time and the distance at the image centre once a second.

Needs librealsense v2.59.1 built from source and installed (see the main README).

```bash
cmake -S perception -B perception/build
cmake --build perception/build -j4
./perception/build/loci_capture 10
```

To keep colour frames (24-bit BMP, one per second by default), e.g. for detector checks or INT8 calibration:

```bash
./perception/build/loci_capture 20 --save ~/frames --every 30
```

If the camera stops sending frames ("Frame didn't arrive"), reset it: `python3 -c "import pyrealsense2 as rs; rs.context().query_devices()[0].hardware_reset()"`.

## Step 3: live detections with 3D positions

`loci_detect` loads the YOLOE TensorRT engine (from `tools/yoloe/export_yoloe.py`), runs it on every frame and gives each object a position in meters: the median depth of the middle half of its box, deprojected through the colour intrinsics (x right, y down, z forward from the camera). Decoding runs on the CPU.

```bash
cmake -S perception -B perception/build && cmake --build perception/build -j4
./perception/build/loci_detect ~/yolo/yoloe/yoloe-26s-seg.engine ~/yolo/yoloe/classes.txt 15 --conf 0.15
```

It prints per-stage times (pre, infer, post, depth) and every detection once a second, then the means over the run.

## Step 4: CUDA kernel 1, fused preprocessing

`src/preprocess.cu` uploads the raw 8-bit frame (0.9 MB at 640x480 instead of 4.9 MB of floats) and does the letterbox (bilinear, grey 114 padding), BGR to RGB, scaling to [0, 1] and HWC to CHW in one pass, writing straight into the TensorRT input buffer. It is the default; `--cpu-pre` switches back to the CPU baseline, and `--check` compares the two on one frame.

```bash
./perception/build/loci_detect ~/yolo/yoloe/yoloe-26s-seg.engine ~/yolo/yoloe/classes.txt 15 --check
./perception/build/loci_detect ~/yolo/yoloe/yoloe-26s-seg.engine ~/yolo/yoloe/classes.txt 15 --cpu-pre
```

In both modes `pre` includes copying the input to the GPU, and `infer` is TensorRT plus copying the predictions back.
