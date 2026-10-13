#!/usr/bin/env bash
# Per-epoch validation metrics from an Ultralytics results.csv (columns found by name).
# Usage: bash epochs.sh [results.csv]
csv="${1:-$HOME/yolo/runs/p3d_full/results.csv}"
awk -F, 'NR==1{for(i=1;i<=NF;i++){gsub(/ /,"",$i);c[$i]=i}; print "epoch  P      R      mAP50  mAP50-95"; next} {printf "%5d  %.3f  %.3f  %.3f  %.3f\n",$c["epoch"],$c["metrics/precision(B)"],$c["metrics/recall(B)"],$c["metrics/mAP50(B)"],$c["metrics/mAP50-95(B)"]}' "$csv"
