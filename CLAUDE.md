# CLAUDE.md

## North star

The end goal of this repo is **Loci**, a spatial memory agent on a Jetson Orin Nano Super + RealSense D415.
The full spec is `docs/LOCI_SPEC.md` (original: `docs/Loci_Proposal.pdf`). Read it before any work here.

## Working rules

- **Do not write project code until the user explicitly says to start.** The user gives the first steps and requirements one at a time.
- Follow the 8-week plan and its gates in order (Gate 1 → 4). Don't skip ahead.
- Every layer has a primary choice and a fallback (see the tech-stack table). Pick the fallback only when the primary is shown not to work.
- Memory is the constraint: 8 GB shared CPU/GPU. One resident VLM only.
- Build from source for sm_87 on JetPack 7.2; don't trust JetPack 6 wheels or containers.
- Verify exact versions and specs before claiming compatibility. Say so when unsure.

## Status

- Current phase: Week 1 / Gate 1 (platform spike) — not yet passed.
- Board confirmed on L4T 36.4.7 with nvidia-jetpack 6.2.1+b38 installed (CUDA 12.6.68, TensorRT 10.3.0.30) as of Oct 10, 2026 — JetPack 7.2.1 not installed yet. Power mode set to MAXN_SUPER (nvpmodel ID 2) on Oct 10, 2026; IDs: 0=15W, 1=25W, 2=MAXN_SUPER, 3=7W.
- D415 confirmed working on JetPack 6: `rs-hello-realsense` reads depth, `rs-capture` streams depth/IR/color at 15 FPS.
- Gate 1 still needs: JetPack 7.2.1 on NVMe, then re-check camera, detector engine, VLM and headless memory baseline.
- YOLO training uses `ultralytics/ultralytics:latest-jetson-jetpack6` (matches JetPack 6). First attempt hit an out-of-memory error with the desktop + Firefox open. Docker went missing after the nvidia-jetpack install; reinstalled docker.io + nvidia-ctk runtime config on Oct 10, 2026 — `torch.cuda.is_available()` returns True in the container.
- Jetson reachable over SSH at 10.0.0.19 (user jenil, hostname Jnvidia). Dataset at ~/yolo/mydata (classes: Box, Parcel; data.yaml paths fixed).
- Side work (not part of Loci): a YOLO26n parcel detector trained from a Roboflow dataset.
