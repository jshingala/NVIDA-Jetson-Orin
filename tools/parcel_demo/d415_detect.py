"""Live Box/Parcel detection on the RealSense D415 with distance and rough size.

Runs headless (no monitor needed): writes an annotated MP4 and prints a
per-second summary. Meant to run inside the Ultralytics jetpack6 container
with the camera passed through (--privileged -v /dev:/dev).

    python3 d415_detect.py --seconds 30
"""

import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO  # load CUDA/TensorRT before librealsense
import pyrealsense2 as rs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/yolo/mine/weights/best_int8.engine")
    p.add_argument("--out", default="/yolo/demo.mp4")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--conf", type=float, default=0.5)
    return p.parse_args()


def box_depth(depth_m, x1, y1, x2, y2):
    """Median depth of the central half of the box, ignoring pixels with no reading."""
    w, h = x2 - x1, y2 - y1
    cx1, cy1 = x1 + w // 4, y1 + h // 4
    cx2, cy2 = x2 - w // 4, y2 - h // 4
    patch = depth_m[max(cy1, 0):max(cy2, cy1 + 1), max(cx1, 0):max(cx2, cx1 + 1)]
    patch = patch[patch > 0]
    return float(np.median(patch)) if patch.size else None


def main():
    args = parse_args()
    model = YOLO(args.model, task="detect")
    # build the TensorRT context before the camera starts, so a model problem shows up on its own
    model(np.zeros((480, 640, 3), dtype=np.uint8), imgsz=640, verbose=False)

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipe.start(cfg)
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    align = rs.align(rs.stream.color)

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 15, (640, 480))
    frames, infer_ms, loop_ms = 0, [], []
    start = last_report = time.time()

    try:
        while time.time() - start < args.seconds:
            t0 = time.time()
            fs = align.process(pipe.wait_for_frames())
            color, depth = fs.get_color_frame(), fs.get_depth_frame()
            if not color or not depth:
                continue
            img = np.asanyarray(color.get_data())
            depth_m = np.asanyarray(depth.get_data()) * scale

            res = model(img, imgsz=640, conf=args.conf, verbose=False)[0]
            infer_ms.append(res.speed["inference"])

            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                z = box_depth(depth_m, x1, y1, x2, y2)
                name = model.names[int(box.cls)]
                label = f"{name} {float(box.conf):.2f}"
                if z:
                    # pinhole model: pixel extent * depth / focal length = metres
                    w_cm = (x2 - x1) * z / intr.fx * 100
                    h_cm = (y2 - y1) * z / intr.fy * 100
                    label += f" {z:.2f}m ~{w_cm:.0f}x{h_cm:.0f}cm"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            writer.write(img)
            frames += 1
            loop_ms.append((time.time() - t0) * 1000)

            if time.time() - last_report >= 1.0:
                print(f"{frames:5d} frames | {len(res.boxes)} objects | "
                      f"infer {np.mean(infer_ms[-30:]):.1f} ms | loop {np.mean(loop_ms[-30:]):.1f} ms "
                      f"({1000 / np.mean(loop_ms[-30:]):.1f} FPS)")
                last_report = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()
        writer.release()

    if frames:
        print(f"\nDone: {frames} frames, inference {np.mean(infer_ms):.1f} ms, "
              f"full loop {np.mean(loop_ms):.1f} ms ({1000 / np.mean(loop_ms):.1f} FPS). Video: {args.out}")


if __name__ == "__main__":
    main()
