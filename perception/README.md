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
