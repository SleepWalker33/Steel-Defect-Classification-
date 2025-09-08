# unified_threshold_eval.py

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# ---------------- Path configuration (via env vars) ----------------
VAL_IMAGE_DIR    = os.getenv("VAL_IMAGE_DIR", "data/images/val")
TEST_IMAGE_DIR   = os.getenv("TEST_IMAGE_DIR", "data/images/test")
LABEL_FILE       = os.getenv("LABEL_FILE", "data/labels/labels1.txt")
VAL_PRED_JSON    = os.getenv("VAL_PRED_JSON", "runs/val/predictions.json")
TEST_PRED_JSON   = os.getenv("TEST_PRED_JSON", "runs/test/predictions.json")

LOG_FILE         = "unified_threshold_eval.log"
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")
    print(msg)

def load_labels(label_file, image_ids):
    """Load ground-truth multi-label vectors from a label text file.

    Format: `img_id lb1 lb2 ...`; if raw == ['-1'], it means background.
    Otherwise, each entry is a defect index in [0..3].
    Returns a dict: img_id -> one-hot [bg, d0, d1, d2, d3].
    """
    gt = {img: np.zeros(5, dtype=int) for img in image_ids}
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            img, raw = parts[0], parts[1:]
            if img not in gt:
                continue
            if raw == ['-1']:
                gt[img][0] = 1
            else:
                for lb in raw:
                    gt[img][int(lb)+1] = 1
    return gt

def load_pred(pred_json, image_ids, threshold):
    """Load predictions from a list of {image_id, category_id, score}.

    category_id in {1,2,3,4} maps to defect {0..3}.
    Returns dict: img_id -> one-hot [bg, d0..d3] using the threshold.
    """
    pred = {img: np.zeros(5, dtype=int) for img in image_ids}
    with open(pred_json) as f:
        data = json.load(f)
    for ann in data:
        img = str(ann["image_id"])
        if img not in pred:
            continue
        cid = ann["category_id"] - 1  # map 1-4 → 0-3
        if ann["score"] >= threshold:
            pred[img][cid+1] = 1
    # If no defects predicted, mark as background
    for img in image_ids:
        if pred[img][1:].sum() == 0:
            pred[img][0] = 1
    return pred

def confusion_stats(y_true, y_pred):
    """Compute confusion matrix, normalized variants, and class metrics.

    Returns:
      - counts 5x5 confusion matrix
      - column-normalized and row-normalized percentages
      - per-class precision/recall/F1
      - per-class 2x2 normalized matrices
    """
    # Raw 5x5 confusion matrix
    cm = np.zeros((5,5), int)
    for tvec, pvec in zip(y_true, y_pred):
        tcls = list(np.where(tvec==1)[0])
        pcls = list(np.where(pvec==1)[0])
        if len(tcls)==1 and len(pcls)==1:
            cm[pcls[0], tcls[0]] += 1
        else:
            if len(pcls)==0:
                cm[0, tcls[0] if tcls else 0] += 1
            else:
                cm[pcls[0], tcls[0] if tcls else 0] += 1

    df_cm = pd.DataFrame(cm,
                         index=[f'Pred{i}' for i in range(5)],
                         columns=[f'True{j}' for j in range(5)])
    # Column-normalized percentages
    df_pct_col = df_cm.div(df_cm.sum(axis=0).replace(0,1), axis=1)*100
    # Row-normalized percentages
    df_pct_row = df_cm.div(df_cm.sum(axis=1).replace(0,1), axis=0)*100

    # P/R/F1
    p = precision_score(y_true, y_pred, average=None, zero_division=0)
    r = recall_score(   y_true, y_pred, average=None, zero_division=0)
    f = f1_score(       y_true, y_pred, average=None, zero_division=0)

    # per-class 2x2
    twobytwo = {}
    for i in range(5):
        tp = np.sum((y_pred[:,i]==1)&(y_true[:,i]==1))
        fp = np.sum((y_pred[:,i]==1)&(y_true[:,i]==0))
        fn = np.sum((y_pred[:,i]==0)&(y_true[:,i]==1))
        tn = np.sum((y_pred[:,i]==0)&(y_true[:,i]==0))
        total = tp+fp+fn+tn or 1
        twobytwo[i] = np.array([[tp/total, fn/total],
                                [fp/total, tn/total]])
    return df_cm, df_pct_col, df_pct_row, p, r, f, twobytwo

if __name__=="__main__":
    # Collect image IDs
    val_ids  = sorted(os.path.splitext(n)[0] for n in os.listdir(VAL_IMAGE_DIR))
    test_ids = sorted(os.path.splitext(n)[0] for n in os.listdir(TEST_IMAGE_DIR))

    # Load ground truth
    gt_val  = load_labels(LABEL_FILE, val_ids)
    gt_test = load_labels(LABEL_FILE, test_ids)

    # Tune threshold on validation set
    best_th, best_score = None, -1
    log("=== Tuning on VAL ===")
    for th in np.arange(0.1, 0.9, 0.1):
        pred_val = load_pred(VAL_PRED_JSON, val_ids, th)
        y_true = np.array([gt_val[i]   for i in val_ids])
        y_pred = np.array([pred_val[i] for i in val_ids])
        p = precision_score(y_true, y_pred, average=None, zero_division=0)
        r = recall_score(   y_true, y_pred, average=None, zero_division=0)
        score = p.mean() + r.mean()
        log(f"th={th:.2f} avg_P={p.mean():.4f} avg_R={r.mean():.4f} sum={score:.4f}")
        if score > best_score:
            best_score, best_th = score, th
    log(f"Best threshold = {best_th:.2f}\n")

    # Evaluate on test set using best threshold
    log("=== Evaluating TEST ===")
    pred_test = load_pred(TEST_PRED_JSON, test_ids, best_th)
    y_true = np.array([gt_test[i]    for i in test_ids])
    y_pred = np.array([pred_test[i]  for i in test_ids])

    df_cm, df_pct_col, df_pct_row, p, r, f, twobytwo = confusion_stats(y_true, y_pred)

    log("\nConfusion Matrix (counts):")
    log(df_cm.to_string())

    log("\nConfusion Matrix (% per True):")
    log(df_pct_col.round(2).to_string())

    log("\nConfusion Matrix (% per Pred):")
    log(df_pct_row.round(2).to_string())

    log("\nPer-class Precision/Recall/F1:")
    for i in range(5):
        log(f" Class{i}: P={p[i]:.4f} R={r[i]:.4f} F1={f[i]:.4f}")

    log("\nPer-class 2x2 (normalized sum=1):")
    for i in range(5):
        m = twobytwo[i]
        log(f"\nClass{i}:")
        log(f"[{m[0,0]:.4f}, {m[0,1]:.4f}]")
        log(f"[{m[1,0]:.4f}, {m[1,1]:.4f}]")
