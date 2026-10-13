# Parcel3D: damaged-parcel detector

Commands used to train and benchmark YOLO26n on [Parcel3D](https://zenodo.org/records/8032204) on the Jetson (JetPack 7.2.1). Results are in the main [README](../../README.md).

Environment: venv `~/yolo-venv` (`--system-site-packages`, for JetPack's TensorRT), torch 2.14.1+cu130 and torchvision 0.29.1 from `download.pytorch.org/whl/cu130`, ultralytics 8.4.176. The dataset lives on an external exFAT drive at `/media/jenil/D D/parcel3d`.

1. Convert COCO to YOLO format (dry run first):
   ```bash
   python3 convert_parcel3d.py && python3 convert_parcel3d.py --go
   ```
2. Train. `workers=1 batch=2 PIN_MEMORY=False` keeps it inside 8 GB; larger settings froze the board.
   ```bash
   PIN_MEMORY=False yolo train model=yolo26n.pt data="/media/jenil/D D/parcel3d/yolo/data.yaml" \
     epochs=50 patience=15 imgsz=640 batch=2 workers=1 amp=True deterministic=False \
     project=$HOME/yolo/runs name=p3d_full
   ```
3. Watch per-epoch metrics: `bash epochs.sh`
4. Test (PyTorch):
   ```bash
   PIN_MEMORY=False yolo val model=$HOME/yolo/runs/p3d_full/weights/best.pt \
     data="/media/jenil/D D/parcel3d/yolo/data.yaml" split=test imgsz=640 batch=1 workers=1
   ```
5. Export a TensorRT FP16 engine and test it:
   ```bash
   yolo export model=$HOME/yolo/runs/p3d_full/weights/best.pt format=engine half=True imgsz=640 batch=1 device=0
   PIN_MEMORY=False yolo val model=$HOME/yolo/runs/p3d_full/weights/best.engine \
     data="/media/jenil/D D/parcel3d/yolo/data.yaml" split=test imgsz=640 batch=1 workers=1
   ```

Run long jobs inside tmux, started in one command, e.g. `tmux new -d -s train '...; bash'`.
