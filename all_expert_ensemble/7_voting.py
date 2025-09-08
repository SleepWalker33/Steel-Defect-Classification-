
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_fscore_support
)
from sklearn.metrics import multilabel_confusion_matrix

# ---------------- Helper functions ----------------
def load_bbox_format(json_path, is_yolo, score_thres, NUM_CLASSES=5):
    with open(json_path) as f:
        data = json.load(f)
    out = {}
    for d in data:
        img = os.path.splitext(str(d['image_id']))[0]
        sc  = d['score']
        cls = d['category_id']
        thr = score_thres[cls-1] if is_yolo else score_thres[cls]
        if sc < thr:
            continue
        vec = out.setdefault(img, [0] * NUM_CLASSES)
        vec[cls if is_yolo else cls+1] = 1
        vec[0] = 0
    return out

def load_score_vector_format(json_path, score_thres, NUM_CLASSES=5):
    with open(json_path) as f:
        data = json.load(f)
    scores = {}
    for d in data:
        img = str(d['image_id'])
        cls = d['category_id']
        sc  = d['score']
        scores.setdefault(img, [0.0] * NUM_CLASSES)[cls] = sc
    out = {}
    for img, sc_list in scores.items():
        bg_sc     = sc_list[0]
        defect_sc = sc_list[1:]
        if bg_sc > max(defect_sc):
            vec = [1, 0, 0, 0, 0]
        else:
            vec = [0] * NUM_CLASSES
            for cls in range(1, NUM_CLASSES):
                if sc_list[cls] >= score_thres[cls]:
                    vec[cls] = 1
            if sum(vec[1:]) == 0:
                vec = [1, 0, 0, 0, 0]
        out[img] = vec
    return out

def load_true_labels(json_path, NUM_CLASSES=5):
    with open(json_path) as f:
        data = json.load(f)
    anns = data.get('annotations', data)
    tmp = {}
    for ann in anns:
        img = str(ann.get('image_id') or ann.get('image_id_str'))
        cls = ann.get('category_id')
        tmp.setdefault(img, []).append(int(cls))
    out = {}
    for img, lst in tmp.items():
        vec = [0] * NUM_CLASSES
        for c in lst:
            idx = c + 1
            if idx < NUM_CLASSES:
                vec[idx] = 1
        out[img] = vec
    return out

def apply_voting(vecs_per_sample, method_weights, NUM_CLASSES=5):
    """Weighted voting policy for [bg, d0..d3].

    1) vote_sum = sum(w_i * vec_i)
    2) If vote_sum[0] >= 6 => pure background
    3) Else defect j is 1 if vote_sum[j] >= 3.5
    4) If 1 <= vote_sum[0] < 6 and any defects => unset background
    5) If all defects are 0 => set background
    """
    results = []
    for vecs in vecs_per_sample:
        weighted = [np.array(vecs[i]) * method_weights[i] for i in range(len(vecs))]
        vote_sum = np.sum(weighted, axis=0).tolist()

        # 1) At least 6 background votes => pure background
        if vote_sum[0] >= 6:
            results.append([1, 0, 0, 0, 0])
            continue

        # 2) Majority voting for defects
        defect_bits = [1 if vote_sum[i] >= 3.5 else 0 for i in range(1, NUM_CLASSES)]

        # 3) Conflict: 1~5 background votes and defects present => remove background
        bg_bit = 1 if sum(defect_bits) == 0 else 0
        if 1 <= vote_sum[0] < 6 and any(defect_bits):
            bg_bit = 0

        # 4) If all defects 0 => set background
        if sum(defect_bits) == 0:
            bg_bit = 1

        results.append([bg_bit] + defect_bits)
    return results

# ---------------- Main ----------------
if __name__ == "__main__":
    # 7 models to ensemble
    selected_models = ['yolo11', 'yolo8', 'faster', 'vgg', 'resnet', 'convnext', 'effient']

    # Prediction path mapping (override with env vars)
    model_paths = {
        'detectors' : os.getenv('PATH_DETECTORS_TEST', 'path/to/detectors_test.json'),
        'yolo11'    : os.getenv('PATH_YOLO11_TEST',    'path/to/yolo11_test_predictions.json'),
        'yolo8'     : os.getenv('PATH_YOLO8_TEST',     'path/to/yolov8_test_predictions.json'),
        'faster'    : os.getenv('PATH_FASTER_TEST',    'path/to/faster_test.json'),
        'cascade'   : os.getenv('PATH_CASCADE_TEST',   'path/to/cascade_test.json'),
        'vgg'       : os.getenv('PATH_VGG_TEST',       'path/to/vgg_test_probabilities.json'),
        'resnet'    : os.getenv('PATH_RESNET_TEST',    'path/to/resnet_test_probabilities.json'),
        'convnext'  : os.getenv('PATH_CONVNEXT_TEST',  'path/to/convnext_test_probabilities.json'),
        'effient'   : os.getenv('PATH_EFFIENT_TEST',   'path/to/effient_test_probabilities.json'),
    }

    # Model format mapping
    model_format = {
        'detectors': 'bbox_detector',
        'faster': 'bbox_detector',
        'cascade': 'bbox_detector',
        'yolo11': 'bbox_yolo',
        'yolo8': 'bbox_yolo',
        'vgg': 'score_vector',
        'resnet': 'score_vector',
        'convnext': 'score_vector',
        'effient': 'score_vector'
    }

    # Ensure model names are consistent across selected_models and model_paths

    val_dir           = os.getenv('TEST_IMAGE_DIR',  'data/images/test')
    true_label_path   = os.getenv('TRUE_LABEL_JSON', 'data/labels/test.json')
    NUM_CLASSES = 5

    # Fixed weights for 7 models (sum can be normalized as needed)
    method_weights = [1.0, 1.5, 0.5, 2.0, 1.0, 1.0, 0.0]

    default = [1, 0, 0, 0, 0]
    val_ids = [os.path.splitext(n)[0] for n in os.listdir(val_dir)]
    gt_labels = load_true_labels(true_label_path, NUM_CLASSES)

    # Fixed thresholds (one per model) matching selected_models order
    fixed_thresholds = [0.4, 0.35, 0.35, 0.8, 0.55, 0.15, 0.15]

    # Load predictions from each model (fixed thresholds)
    preds_all = []
    for i, model in enumerate(selected_models):
        path = model_paths[model]
        fmt  = model_format[model]
        th_val = fixed_thresholds[i]
        th     = [th_val] * NUM_CLASSES
        if fmt == 'bbox_detector':
            th = th[:NUM_CLASSES - 1]
            preds = load_bbox_format(path, is_yolo=False, score_thres=th, NUM_CLASSES=NUM_CLASSES)
        elif fmt == 'bbox_yolo':
            th = th[:NUM_CLASSES - 1]
            preds = load_bbox_format(path, is_yolo=True, score_thres=th, NUM_CLASSES=NUM_CLASSES)
        elif fmt == 'score_vector':
            preds = load_score_vector_format(path, score_thres=th, NUM_CLASSES=NUM_CLASSES)
        else:
            raise ValueError(f"Unknown model format: {fmt}")
        preds_all.append(preds)

    # Voting fusion across models
    y_true_list, y_pred_list = [], []
    for img in val_ids:
        y_true_list.append(gt_labels.get(img, default))
        vecs = [preds_all[i].get(img, default) for i in range(len(selected_models))]
        vote = apply_voting([vecs], method_weights, NUM_CLASSES)[0]
        y_pred_list.append(vote)

    y_true_np = np.array(y_true_list)
    y_pred_np = np.array(y_pred_list)

    # Compute evaluation metrics
    p   = precision_score(y_true_np, y_pred_np, average=None, zero_division=0)
    r   = recall_score(y_true_np, y_pred_np, average=None, zero_division=0)
    f1  = 2 * p * r / (p + r + 1e-8)
    acc = np.mean(np.all(y_true_np == y_pred_np, axis=1))  # exact-match sample accuracy

    print("\n=== Voting Results (Fixed Thresholds & Weights, 7 Models) ===")
    print(f"Models: {selected_models}")
    print(f"Thresholds: {fixed_thresholds}")
    print(f"Weights: {method_weights}")
    print(f"Average Precision : {np.mean(p):.4f}")
    print(f"Average Recall    : {np.mean(r):.4f}")
    print(f"Average F1-score  : {np.mean(f1):.4f}")
    print(f"Overall Accuracy  : {acc:.4f}")

    # Per-class metrics
    print("\n--- Per-Class Metrics ---")
    mcm = multilabel_confusion_matrix(y_true_np, y_pred_np)
    per_class_acc = []

    for i in range(NUM_CLASSES):
        tn, fp, fn, tp = mcm[i].ravel()
        acc_i = (tp + tn) / (tp + tn + fp + fn)
        per_class_acc.append(acc_i)
        print(f"Class {i}: Precision={p[i]:.4f}, Recall={r[i]:.4f}, F1-score={f1[i]:.4f}, Accuracy={acc_i:.4f}")

    print(f"\nMean Per-Class Accuracy (Diagonal Avg): {np.mean(per_class_acc):.4f}")
