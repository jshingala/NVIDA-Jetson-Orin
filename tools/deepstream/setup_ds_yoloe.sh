#!/usr/bin/env bash
# Loci: YOLOE-26s-seg inside DeepStream 9.1.
#   bash tools/deepstream/setup_ds_yoloe.sh
# Builds the box parser from this folder and writes everything to ~/deepstream/yoloe:
#   libnvds_yoloe_parser.so, yoloe-26s-seg.plan, config_infer_yoloe.txt, ds_yoloe_cam.txt
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
DS=/opt/nvidia/deepstream/deepstream-9.1
W=$HOME/deepstream/yoloe
ENGINE=$HOME/yolo/yoloe/yoloe-26s-seg.engine
ONNX=$HOME/yolo/yoloe/yoloe-26s-seg.onnx
CLASSES=$HOME/yolo/yoloe/classes.txt
SAMPLE=$HOME/deepstream/headless_src4_tracker.txt
mkdir -p "$W"
cd "$W"

echo "== 1/4 parser library"
g++ -shared -fPIC -std=c++17 -O2 -w -I$DS/sources/includes -I/usr/local/cuda/include \
    "$HERE/nvdsparsebbox_yoloe.cpp" -o libnvds_yoloe_parser.so
nm -D --defined-only libnvds_yoloe_parser.so | grep -q NvDsInferParseYoloE
echo "   ok: $W/libnvds_yoloe_parser.so"

echo "== 2/4 engine without the Ultralytics metadata header"
python3 - "$ENGINE" yoloe-26s-seg.plan <<'EOF'
import struct, sys
data = open(sys.argv[1], "rb").read()
n = struct.unpack("<i", data[:4])[0]
off = 4 + n if 0 < n < len(data) - 4 and data[4:5] == b"{" else 0
open(sys.argv[2], "wb").write(data[off:])
print(f"   ok: skipped {off} header bytes, plan {len(data) - off} bytes")
EOF

echo "== 3/4 camera node (D415 RGB = YUYV 640x480)"
command -v v4l2-ctl >/dev/null || { echo "   missing v4l2-ctl: sudo apt install -y v4l-utils, then rerun"; exit 1; }
NODE=""
for d in /dev/video*; do
    if v4l2-ctl -d "$d" --list-formats-ext 2>/dev/null | sed -n "/'YUYV'/,/\]: '/p" | grep -q "640x480"; then
        NODE=${d#/dev/video}; break
    fi
done
[ -n "$NODE" ] || { echo "   no /dev/video* offers YUYV 640x480 (is the D415 plugged in and free?)"; exit 1; }
echo "   ok: /dev/video$NODE"

echo "== 4/4 configs"
cat > config_infer_yoloe.txt <<EOF
[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-color-format=0
model-engine-file=$W/yoloe-26s-seg.plan
onnx-file=$ONNX
labelfile-path=$CLASSES
batch-size=1
network-mode=2
num-detected-classes=10
interval=0
gie-unique-id=1
process-mode=1
network-type=0
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=1
parse-bbox-func-name=NvDsInferParseYoloE
custom-lib-path=$W/libnvds_yoloe_parser.so

[class-attrs-all]
pre-cluster-threshold=0.15
nms-iou-threshold=0.5
topk=100
EOF

# Reuse the NvDCF tracker settings from the sample that already ran.
LL_LIB=$(grep -m1 '^ll-lib-file' "$SAMPLE" 2>/dev/null | cut -d= -f2- || true)
LL_CFG=$(grep -m1 '^ll-config-file' "$SAMPLE" 2>/dev/null | cut -d= -f2- || true)
: "${LL_LIB:=$DS/lib/libnvds_nvmultiobjecttracker.so}"
: "${LL_CFG:=$DS/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml}"

cat > ds_yoloe_cam.txt <<EOF
[application]
enable-perf-measurement=1
perf-measurement-interval-sec=2

[tiled-display]
enable=0

[source0]
enable=1
type=1
camera-width=640
camera-height=480
camera-fps-n=30
camera-fps-d=1
camera-v4l2-dev-node=$NODE

[streammux]
gpu-id=0
live-source=1
batch-size=1
batched-push-timeout=40000
width=640
height=480
enable-padding=0

[primary-gie]
enable=1
gpu-id=0
gie-unique-id=1
config-file=$W/config_infer_yoloe.txt

[tracker]
enable=1
tracker-width=640
tracker-height=384
ll-lib-file=$LL_LIB
ll-config-file=$LL_CFG
gpu-id=0
display-tracking-id=1

[osd]
enable=1
gpu-id=0
border-width=2
text-size=12
text-color=1;1;1;1
text-bg-color=0.3;0.3;0.3;1
font=Serif
show-clock=0

# Orin Nano has no NVENC: software x264, MKV so Ctrl+C still leaves a playable file.
[sink0]
enable=1
type=3
container=2
codec=1
enc-type=1
sync=0
bitrate=4000000
output-file=$W/yoloe_ds.mkv
source-id=0
EOF

gst-inspect-1.0 x264enc >/dev/null 2>&1 || echo "   note: x264enc missing: sudo apt install -y gstreamer1.0-plugins-ugly"
echo "   ok: $W/config_infer_yoloe.txt, $W/ds_yoloe_cam.txt"
echo
echo "Run:  cd $W && timeout -s INT 25 deepstream-app -c ds_yoloe_cam.txt"
