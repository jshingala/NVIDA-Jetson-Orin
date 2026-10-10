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
- Gate 1 evidence still needed: `cat /etc/nv_tegra_release`, an idle `tegrastats` line, and whether `realsense-viewer` sees the D415.
- Side work (not part of Loci): a YOLO26n parcel detector trained from a Roboflow dataset.
