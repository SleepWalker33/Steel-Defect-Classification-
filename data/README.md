Datasets

- Severstal Steel Defect Detection (classification/detection)
  - Download: https://www.kaggle.com/competitions/severstal-steel-defect-detection/overview
  - Note: requires a Kaggle account and accepting competition terms.
  - Organize images and annotations into your preferred format (e.g., COCO-style for detection, CSV/TXT for classification labels).

- NEU Surface Defect Database
  - Classification set (6 classes): often referred to as NEU-SD.
  - Official page: http://faculty.neu.edu.cn/songkechen/zh_CN/zhym/263269/list/index.htm
  - Verify license/terms before use.

Suggested structure
- data/
  - images/
    - train/
    - val/
    - test/
  - labels/
    - train/
    - val/
    - test/
  - train.json            # COCO-style for train
  - val.json              # COCO-style for validation
  - test.json             # COCO-style for test
  - yolo/
    - data.yaml            # Ultralytics YOLO dataset config

Notes
- Adjust environment variables used in scripts to point to your local paths:
  - VAL_IMAGE_DIR, TEST_IMAGE_DIR, TRAIN_IMAGE_DIR, LABEL_FILE, TRUE_LABEL_JSON
  - YOLO_DATA_CONFIG, YOLO_WEIGHTS, and PATH_* variables for ensemble inputs
  - Alternatively, edit scripts to use relative paths under this data/ folder.
