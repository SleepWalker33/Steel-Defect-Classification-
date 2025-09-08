Faster-Series (MMDetection)

- Environment: CUDA 12.x, Python 3.8+, PyTorch 2.4.x, MMDetection 3.3.0, MMCV 2.1.0, MMEngine 0.10.x
- Models: Faster R-CNN, Cascade R-CNN, DetectoRS, DINO (DETR with Improved deNoising anchOr boxes)

What this folder provides
- High-level description of models and training setup.
- Example commands to train/evaluate with MMDetection.
- No copy of MMDetection training code is included here; please use the official repo and configs.

Why not include MMDetection training code?
- MMDetection is a separate, actively maintained framework. The recommended practice is to use its official configs and scripts rather than duplicating them here.
- This README documents the exact configs and command templates to reproduce results.

Recommended setup
1) Install dependencies
   - pip install -r requirements.txt (in the project root)
   - Or follow official MMDetection installation guide.
2) Prepare datasets (COCO-style or your own; see data/README.md).
3) Use official MMDetection configs as a base and customize.

Example commands (MMDetection)
- Faster R-CNN (example config):
  - Training: mmdet/tools/train.py configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py
  - Evaluation: mmdet/tools/test.py  work_dirs/…/latest.pth  

- Cascade R-CNN:
  - Training: mmdet/tools/train.py configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py
  - Evaluation: mmdet/tools/test.py  work_dirs/…/latest.pth  
- DetectoRS:
  - Training: mmdet/tools/train.py configs/detectors/detectors_r50_1x_coco.py
  - Evaluation: mmdet/tools/test.py  work_dirs/…/latest.pth  

- DINO:
  - Training: mmdet/tools/train.py configs/dino/dino-4scale_r50_8xb2-12e_coco.py
  - Evaluation: mmdet/tools/test.py  work_dirs/…/latest.pth  

Notes and tips
- Set random seeds for reproducibility (e.g., --seed 42, deterministic=True in configs).
- Use data augmentation as needed (e.g., random flip, crop, brightness). Keep configs under version control.
- Export predictions as COCO json for downstream fusion (see ensemble scripts). Use jsonfile_prefix to control output file path.

Previously in details.rtf
- The prior RTF summary is replaced by this Markdown README for clarity and version control friendliness.
