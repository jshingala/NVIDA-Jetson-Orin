# Loci: A Spatial Memory Agent on Jetson Orin Nano

> **Main project file.** This is the end goal for this repo. Source: `docs/Loci_Proposal.pdf` (Oct 9, 2026).
> Hyperlinks from the original PDF are not reproduced here; open the PDF for the source URLs.

## The pitch

Loci gives AI agents a memory of the physical world, running entirely on a Jetson Orin Nano Super and a RealSense D415. It watches a room or workbench, remembers what it saw and where (in meters), and answers plain-English questions or enforces watch rules. No video ever leaves the board.

The name comes from the method of loci, the memory-palace technique of storing memories by place.

The 90-second demo (outputs are examples, to be replaced with real ones):

1. **Ask:** "Where did I leave my keys?" returns "On the desk, left of the monitor, about 1.4 m from the camera, last seen 3:42 pm" plus the keyframe.
2. **Watch:** "Alert me if someone stays within 1 m of the 3D printer for over 10 seconds" compiles into an on-device rule; it fires live, and the VLM confirms before the alert goes out.
3. **Measure:** "How big is this box?" returns width, depth and height from the depth camera.
4. **Integrate:** an outside agent (Claude, or a NeMo Agent Toolkit agent) connects over MCP and asks "What changed on the bench since 9 am?"
5. **Prove it:** a jtop overlay shows everything running locally, with live power draw.

It is a system, not a model demo: a real-time C++/CUDA loop, metric 3D grounding, a memory store, a scheduler, an agent and an open protocol, all fitted into 8 GB with measured tradeoffs.

## Hardware reality check

Memory, not TOPS, is the constraint: 8 GB shared by CPU and GPU, and the same 2B VLM peaks at 4.3 GB or 6.9 GB depending on the runtime.

**Board.** Orin Nano Super Developer Kit, 8 GB, on JetPack 6.2 from the SD card (L4T 36.4.7) per July setup notes, with a 256 GB NVMe installed but not yet set up, plus a RealSense D415.

**Verified specs (NVIDIA):** 67 INT8 TOPS; Ampere GPU with 1024 CUDA cores and 32 Tensor Cores; 6-core Cortex-A78AE up to 1.7 GHz; 8 GB 128-bit LPDDR5 at 102 GB/s; 7–25 W.

**Software to build on (October 2026):**

- **JetPack 7.2**: Jetson Linux 39.2, kernel 6.8, Ubuntu 24.04, CUDA 13.2.1, TensorRT 10.16.2, DeepStream 9.1, Nsight Systems 2026.3. The Orin Nano dev kit no longer has an SD card image; it installs from an ISO on a USB stick, which suits the unused NVMe. **JetPack 7.2.1** flashes the dev kit in Super mode by default.
- **TensorRT Edge-LLM**: NVIDIA's C++ runtime for LLMs and VLMs on Jetson, no Python in the inference path. On Orin it requires JetPack 7.2 and runs FP16 or INT4 AWQ (FP8 and NVFP4 are Thor-only). Its export step runs on an x86 NVIDIA GPU or a Thor, not on the Orin.
- **Cosmos Reason 2 2B**: NVIDIA's compact VLM with chain-of-thought and spatial reasoning. Jetson AI Lab recommends llama.cpp on the Orin Nano; its vLLM recipe for the Nano needs tight limits (768-token context, one image per prompt).
- **DeepStream 9.1**: first release targeting JetPack 7.2 on Orin, but its VLM plugin is documented for x86 GPUs only. On-device VLM work goes through Edge-LLM or llama.cpp.
- **librealsense**: documents an RSUSB build that skips kernel patching and covers Ubuntu 24.04 / kernel 6.8. Orin Nano on JetPack 7.2 is not listed as verified — test it first.

**Measured by others** (one developer's benchmark repo, JetPack 6.2.2, MAXN SUPER). A baseline to reproduce, not a promise.

| Workload | Runtime | Measured |
|---|---|---|
| YOLOv8n, 640x640 | TensorRT | 4.43 ms FP16, 3.49 ms INT8 (GPU time only) |
| Cosmos-Reason2-2B, 4-bit | llama.cpp | 38 tok/s; 114 ms to first token with an image |
| Cosmos-Reason2-2B, 4-bit | vLLM | 56 tok/s; 75 ms TTFT with an image; 6.9 GB peak |
| Cosmos-Reason2-2B, 4-bit | TensorRT Edge-LLM | 60 tok/s; 420 ms TTFT with an image; 4.3 GB peak |

On JetPack 6.2.2 the Edge-LLM path needed two workarounds (a CMA kernel parameter and an ONNX graph rewrite). Whether JetPack 7.2 still needs them is an open question to answer and publish.

GPU-only timings hide pipeline cost: one forum report shows 3.8 ms preprocess and 8.0 ms postprocess around a 13.5 ms inference in Python — exactly what a C++/CUDA path removes.

**Memory budget** (targets; measure each with tegrastats in week 1):

| Component | Plan | Budget |
|---|---|---|
| OS, headless | Jetson AI Lab RAM-optimization steps, no desktop | measure; target ≤ 1.5 GB |
| VLM (eyes and agent brain) | Cosmos-Reason2-2B INT4 on Edge-LLM | 4.3 GB peak (measured by others) |
| Detector, tracker, frame buffers | TensorRT INT8 engine, pinned zero-copy buffers | est. 0.3–0.5 GB |
| Embeddings and vector index | small text-embedding model over captions | est. 0.2–0.5 GB |
| Agent, API, MCP server | one Python process | est. 0.2–0.3 GB |
| Headroom | KV-cache growth, spikes | keep ≥ 0.5 GB |

The budget only closes because **one model does both the seeing and the reasoning**. A second resident LLM would not fit — the first design decision to defend.

## Architecture

Every frame runs the fast C++ loop; the VLM only sees events. Video never leaves the board; agents reach it over MCP.

```
RealSense D415 ──► Perception (C++/CUDA) ──► Events + rules (C++)
 RGB + aligned depth   fused preprocess kernel    appear, move, dwell, vanish
 USB3, librealsense    TensorRT INT8 detector     compiled watch rules
 RSUSB, 30 FPS         tracker + depth-to-3D       pickup and dropout timers
                       kernel                          │            │
                                              all events│            │salient events
                                                       ▼            ▼
Agent + MCP server ◄──► Memory store ◄────────── VLM worker (Edge-LLM)
 Python, FastAPI,       SQLite events +          Cosmos Reason 2 2B, INT4
 MCP SDK                3D positions             queue: ask, verify, caption
 find, query,           vector index over        the one resident model
 measure, watch         captions                        │
 rule compiler +        keyframes on NVMe               │ if unsure
 guardrails  ──── planning and live questions ──►       ▼
     ▲                                           Cloud escalation (optional)
     │ requests                                  NVIDIA NIM call, hard
Clients over MCP and REST                        questions only; text out,
 web dashboard, Claude, NeMo Agent Toolkit,      never video
 Agentforce; alerts to phone, Slack or MQTT;
 token auth on every call
```

Every event lands in memory, only salient ones reach the VLM, and the agent answers from memory while borrowing the VLM for planning and live questions. Only text leaves the device; the cloud call is optional and text-only.

The camera loop, the VLM worker and the agent run as **separate processes under systemd**, so a VLM crash never stops perception, and each can be restarted, profiled and tested on its own.

## Five ideas that make it non-trivial

Each comes with a number to measure.

1. **Two-speed perception.** A cheap C++ loop (detector, tracker, depth) runs on every frame; the VLM runs only on salient events, user questions and alert checks. *Proof:* VLM calls per hour with and without gating, at the same event recall.
2. **The LLM is a compiler, not an interpreter.** "Tell me if the garage door stays open after 10 pm" is compiled once into a small JSON rule (subject, zone in meters, condition, pickup delay, dropout delay) that a C++ engine checks every frame. No language model in the hot loop; rules are unit-testable; the user confirms each compiled rule before it goes live. *Proof:* rule-check time per frame, and alert precision with and without VLM confirmation.
3. **One resident model, budgeted to the megabyte.** A single 2B VLM is both the eyes (captions, alert checks) and the brain (agent planning through schema-checked JSON tool calls). A priority scheduler serves user questions first, then alert checks, then background captions, and merges stale captions under load. *Proof:* per-process memory high-water marks and question latency under load.
4. **Memories anchored in meters.** Depth from the D415 turns each detection into a 3D point, so zones are defined in meters and answers can say "1.4 m left of the door". Two custom CUDA kernels: fused preprocessing straight into the TensorRT input buffer, and per-box robust depth (histogram median) plus deprojection. *Proof:* position error in cm against tape-measured ground truth, and kernel time against a CPU baseline.
5. **MCP-native, so any agent can use it.** Tools (find object, query events, describe now, measure, create watch rule) served over MCP's Streamable HTTP transport with token auth. NeMo Agent Toolkit works as an MCP client and Agentforce ships a native one.

## Tech stack, with a fallback for every layer

Rule on JetPack 7.2: **build from source for the Orin's CUDA architecture (sm_87)**. A June 2026 setup guide found containers and wheels built for JetPack 6 crashed, fell back to CPU or produced bad output, while source builds worked.

| Layer | Primary choice | Why | Fallback |
|---|---|---|---|
| OS | JetPack 7.2.1 on the NVMe, installed from the ISO on a USB stick | Newest stack, official Edge-LLM path for Orin | JetPack 6.2.2 with llama.cpp for the VLM |
| Camera | librealsense, RSUSB build with CUDA on | No kernel patching | Native backend with the kernel patch script |
| Detector | YOLOE with own object vocabulary (keys are not a COCO class); ONNX exported on the Mac, INT8 engine built on the board | Open vocabulary; DeepStream 9.1 documents YOLOE | YOLO11n fine-tuned on own objects |
| Tracker | ByteTrack-style tracker in C++ | Stable IDs for dwell, move, vanish events | DeepStream NvDCF tracker |
| Kernels | CUDA C++, profiled in Nsight Systems | Removes CPU pre/post-processing cost | OpenCV CUDA or VPI 4.1 |
| VLM | Cosmos-Reason2-2B INT4 AWQ on TensorRT Edge-LLM's C++ runtime | 4.3 GB peak in community tests | Same model as GGUF on llama.cpp |
| Rules | C++20 engine over a JSON rule schema, GoogleTest | Deterministic and testable | Python engine at a lower check rate |
| Memory | SQLite for events, 3D positions and rules; vector index over captions; keyframes on NVMe | Embedded, zero-ops, easy to inspect | SQLite FTS5 keyword search |
| Agent and API | Python, FastAPI, MCP Python SDK over Streamable HTTP | Builds on prior RAG and MCP work | Plain REST tools |
| UI | Small React dashboard: live view, timeline, chat, rules | Already ship React | Server-rendered HTML |
| Ops | Docker Compose, systemd watchdogs, Prometheus metrics, GitHub Actions CI | What SDE1 interviews probe | Shell scripts and log files |

## 8-week build plan

A demo-able MVP at the end of week 3 and frozen benchmark numbers by week 6. Each gate must pass before the next phase starts.

| Week | Phase | Work |
|---|---|---|
| 1 | Platform spike | JetPack 7.2.1 on NVMe, camera, TensorRT, VLM memory baseline |
| 2 | Fast loop in C++ | Capture to 3D tracks, two CUDA kernels, INT8 detector engine |
| 3 | Events + memory | Event engine, SQLite, keyframes, search; MVP demo, no VLM |
| 4 | VLM worker | Edge-LLM, INT4 model, priority scheduler, captions, alert checks |
| 5 | Agent and MCP | MCP tools, rules, React dashboard, token auth |
| 6 | Benchmarks | Loci-QA, 3D error, power and latency, Nsight traces |
| 7 | Hardening | Watchdogs, retention, Docker, CI, tests, 1-hour soak test |
| 8 | Ship and publish | README, demo video, blog, forum post, upstream notes |

**Gates:**

- **Gate 1 (after week 1):** JetPack 7.2.1 boots from the NVMe, realsense-viewer streams depth, and the detector engine and the VLM each run on their own, with a headless memory baseline recorded.
- **Gate 2 (after week 3):** "Where did I leave my keys?" is answered from memory with a 3D position and a keyframe, no VLM involved, and the event engine's unit tests pass in CI. Start applying here with the MVP demo.
- **Gate 3 (after week 6):** every row of the measurement table is filled by a script; those numbers become the README and resume figures.
- **Gate 4 (after week 8):** the repo, demo video and write-up are public, and the Edge-LLM notes are posted.

Assumes ~15 hours a week. With less time, stretch the weeks but keep the gates in order.

## What to measure and publish

Every claim comes from this table, measured on this board and published with scripts that reproduce it.

| Question | Metric | How |
|---|---|---|
| Is perception real-time? | End-to-end FPS; p50 and p99 latency per stage | CUDA events plus an Nsight Systems trace, 10-minute runs |
| Did INT8 cost accuracy? | mAP on ~300 labeled frames of the room, FP16 vs INT8 | Own labeled set |
| Is 3D grounding accurate? | Position error (cm) at 0.5–3 m; box-dimension error vs a ruler | Tape-measured ground truth |
| Does the memory answer correctly? | Accuracy on a 100-question "Loci-QA" set over an 8-hour recording | Questions written before seeing answers, graded by script |
| Does gating pay off? | VLM calls per hour and event recall, gated vs ungated | Same recording replayed both ways |
| Does VLM confirmation help alerts? | Alert precision and recall, with vs without the VLM check | 50 staged events with known labels |
| Does it fit? | Peak and steady memory per process; swap use | tegrastats during a 1-hour soak test |
| Does the detector survive VLM load? | Detector p99 latency while the VLM decodes, with and without CUDA stream priorities | Two runs, same workload |
| What does it cost to run? | Watts, and joules per answered question, at 15 W, 25 W and MAXN SUPER | jtop and tegrastats power readings |
| Which runtime wins on JetPack 7.2? | TTFT, tokens/s, peak memory: Edge-LLM vs llama.cpp | Same prompts and image, 5 runs each |

The last row doubles as a community contribution: published Orin Nano VLM numbers are from JetPack 6.2.2, so JetPack 7.2 results are new data.

## Risks and how to de-risk them early

| Risk | Early test (week 1) | Fallback |
|---|---|---|
| Edge-LLM on the Nano still needs the JetPack 6.2.2 workarounds, or the engine build runs out of memory | Run the official Qwen3-4B INT4 tutorial first, then the VLM, with the low-memory export flag | Cosmos-Reason2-2B as GGUF on llama.cpp |
| No x86 NVIDIA GPU for the Edge-LLM export step (Mac cannot run it) | Rent an Ampere-or-newer cloud GPU with 24 GB for an afternoon | llama.cpp path, which needs no export |
| Prebuilt JetPack 6 containers and wheels misbehave on 7.2 | Build everything from source for sm_87 | Stay on JetPack 6.2.2 |
| RealSense not detected on JetPack 7.2 | RSUSB build and realsense-viewer on day 1 | Native backend kernel patch, or JetPack 6.2.2 |
| ISO install fails from current firmware | Try the ISO first | SDK Manager (needs an Ubuntu x86 machine) |
| Memory pressure kills the VLM | Headless boot plus a 1-hour tegrastats soak | Shorter context, text-only embeddings, NVMe swap |
| Boot failures after power cycles on some Orin Nano modules | Install the PCIe overlay fix listed on the JetPack 7.2 page | Shut down cleanly before unplugging |
| Scope creep | MVP gate at week 3: the keys question works without the VLM | Cut stretch goals, never the benchmarks |

## Career framing

Hiring map, per-company pitch, alternatives considered and resume lines are in the original PDF (`docs/Loci_Proposal.pdf`, pages 1–2 and 8–12). Kept out of this file so the spec can go public at Gate 4 without personal job-search notes.
