Steel Surface Defect Project

Overview
- This repository contains scripts for training and evaluating multiple models for steel surface defect analysis:
  - Detection: YOLO series, Faster/Cascade R-CNN, DetectoRS, DINO (via MMDetection)
  - Classification (multi-label): VGG19, ResNet101, ConvNeXt, EfficientNet variants
  - Ensemble: threshold/weight tuning for late fusion

Privacy & paths
- All hard-coded personal absolute paths have been removed. Scripts now read paths from environment variables with safe placeholders. See each script header for the env vars it respects.
- You can also modify them to use relative paths under data/.

Structure
- yolo_series/: Ultralytics YOLO training/eval examples
- faster_series/: Notes and command templates for MMDetection models (see faster_series/README.md)
- each_cla_expert/: Classification experts (seed and sequential variants)
- all_expert_ensemble/: Late-fusion threshold/weight tuning utilities
- data/: Dataset notes and expected structure (see data/README.md)

Setup
- Python: 3.8.20
- Install: pip install -r requirements.txt
- Datasets: see data/README.md for download links (Severstal, NEU) and layout.
- MMDetection: use official repo/configs; this repo does not duplicate training code. See faster_series/README.md for command templates.

Running
- YOLO evaluation: set YOLO_DATA_CONFIG and YOLO_WEIGHTS, then run yolo_series/yolov8.py or yolo11.py
- Ensemble tuning: set PATH_* env vars to the prediction JSONs from each model and TRUE_LABEL_JSON; run all_expert_ensemble/7_tune_threshold.py or 7_tune_weight.py
- Classification experts: set TRAIN/VAL/TEST dirs and LABEL_FILE via env vars used at the top of each script.

What you may still need to add
- Exact dataset conversion/code to produce COCO-format jsons (val.json/test.json) for your data.
- Exact YOLO data.yaml under data/yolo/data.yaml tailored to your paths/classes.
- MMDetection configs (customized) under your mmdetection checkout and any training wrappers you use.
- Any model weights or checkpoints (not included here).
- Detailed hyperparameter choices per experiment beyond what is embedded in scripts.

License and credits
- Respect licenses for third-party datasets and frameworks (Ultralytics, MMDetection). This repository contains only glue code and documentation.
