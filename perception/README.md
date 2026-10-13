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
