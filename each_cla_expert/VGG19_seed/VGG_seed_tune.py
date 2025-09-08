# vgg_eval_with_tuning.py

import os
import json
import time
import random
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image

from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

from VGG_seed_train import (
    MultiLabelDataset, load_samples,
    build_model, get_image_ids_from_dir,
    build_confusion_and_trace
)

# ---------------- Configuration ----------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(SEED)

# Number of classes
NUM_CLASSES = 5  # [bg, d0, d1, d2, d3]

# Path configuration
VAL_IMAGE_DIR   = os.getenv("VAL_IMAGE_DIR",  "data/images/val")
TEST_IMAGE_DIR  = os.getenv("TEST_IMAGE_DIR", "data/images/test")
LABEL_FILE      = os.getenv("LABEL_FILE",     "data/labels/labels1.txt")
BEST_MODEL_PATH = "base_seed42_valtest_sep/best.pth"
LOG_FILE        = "log_eval.txt"
MISCLASS_DIR    = "misclassified_val"

BATCH_SIZE = 16
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Grid search thresholds: 0.1 to 0.8
THRESHOLDS = np.round(np.arange(0.1, 0.9, 0.1), 2)

def write_log(msg):
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')
    print(msg)

def eval_dir(image_dir, thresholds, model, device):
    """Evaluate on image_dir using thresholds vector; returns y_true, y_pred."""
    img_ids = get_image_ids_from_dir(image_dir)
    samples = load_samples(LABEL_FILE, img_ids)  # Multi-label samples including background
    loader  = DataLoader(
        MultiLabelDataset(samples, image_dir, TRANSFORM),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, worker_init_fn=seed_worker, generator=g
    )

    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for ids, imgs, labels in loader:
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs)).cpu().numpy()  # shape (B,5)
            preds = (probs > thresholds).astype(int)
            # If any defect -> background=0; if all zero -> background=1
            for i in range(preds.shape[0]):
                if preds[i,1:].sum() > 0:
                    preds[i,0] = 0
                elif preds[i].sum() == 0:
                    preds[i,0] = 1
            y_true_list.extend(labels.numpy())
            y_pred_list.extend(preds)

    return np.array(y_true_list), np.array(y_pred_list), img_ids

if __name__ == "__main__":
    # Clean old logs
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    write_log("=== Start Threshold Tuning ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))

    # Validation threshold tuning
    best_th, best_score = None, -1
    for th in THRESHOLDS:
        thr_vec = np.array([th] * NUM_CLASSES, dtype=float)
        y_true, y_pred, _ = eval_dir(VAL_IMAGE_DIR, thr_vec, model, device)
        p = precision_score(y_true, y_pred, average=None, zero_division=0)
        r = recall_score(y_true, y_pred, average=None, zero_division=0)
        avg_p, avg_r = p.mean(), r.mean()
        score = avg_p + avg_r
        write_log(f"Threshold={th:.2f} -> avg_P={avg_p:.4f}, avg_R={avg_r:.4f}, sum={score:.4f}")
        if score > best_score:
            best_score, best_th = score, th

    write_log(f"Best threshold: {best_th:.2f}")
    write_log("=== End Threshold Tuning ===\n")

    # Evaluate test set with best threshold
    write_log("=== Start Test Evaluation ===")
    thr_vec = np.array([best_th] * NUM_CLASSES, dtype=float)
    y_true, y_pred, ids = eval_dir(TEST_IMAGE_DIR, thr_vec, model, device)
    
    # Save test-set raw probabilities as JSON
    write_log("Saving raw sigmoid probabilities to test_probabilities.json ...")

    samples = load_samples(LABEL_FILE, ids)
    loader  = DataLoader(MultiLabelDataset(samples, TEST_IMAGE_DIR, TRANSFORM),
                         batch_size=BATCH_SIZE, shuffle=False,
                         worker_init_fn=seed_worker, generator=g)

    model.eval()
    json_out = []
    with torch.no_grad():
        for img_ids, imgs, _ in loader:
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs)).cpu().numpy()  # shape (B, 5)
            for i, img_id in enumerate(img_ids):
                for cls in range(NUM_CLASSES):
                    json_out.append({
                        "image_id": img_id,
                        "category_id": cls,
                        "score": float(probs[i, cls])
                    })

    with open("test_probabilities.json", "w") as f:
        json.dump(json_out, f, indent=2)
    write_log("Saved raw probability predictions to test_probabilities.json.")

    # Save validation-set raw probabilities as JSON
    write_log("Saving raw sigmoid probabilities to val_probabilities.json ...")

    val_ids = get_image_ids_from_dir(VAL_IMAGE_DIR)
    val_samples = load_samples(LABEL_FILE, val_ids)
    val_loader  = DataLoader(MultiLabelDataset(val_samples, VAL_IMAGE_DIR, TRANSFORM),
                             batch_size=BATCH_SIZE, shuffle=False,
                             worker_init_fn=seed_worker, generator=g)

    model.eval()
    val_json_out = []
    with torch.no_grad():
        for img_ids, imgs, _ in val_loader:
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs)).cpu().numpy()  # shape (B, 5)
            for i, img_id in enumerate(img_ids):
                for cls in range(NUM_CLASSES):
                    val_json_out.append({
                        "image_id": img_id,
                        "category_id": cls,
                        "score": float(probs[i, cls])
                    })

    with open("val_probabilities.json", "w") as f:
        json.dump(val_json_out, f, indent=2)
    write_log("Saved raw probability predictions to val_probabilities.json.")
    
    # 1) Confusion matrix (counts)
    cm = build_confusion_and_trace(y_true, y_pred, ids, NUM_CLASSES)[0]
    df_cm = pd.DataFrame(cm)
    write_log("\nConfusion Matrix (counts):")
    write_log(df_cm.to_string())

    # 2) Column-normalized by True class
    df_pct_col = df_cm.div(df_cm.sum(axis=0).replace(0,1), axis=1) * 100
    write_log("\nConfusion Matrix (% per True class):")
    write_log(df_pct_col.round(2).to_string())

    # 3) Row-normalized by Pred class
    df_pct_row = df_cm.div(df_cm.sum(axis=1).replace(0,1), axis=0) * 100
    write_log("\nConfusion Matrix (% per Predicted class):")
    write_log(df_pct_row.round(2).to_string())

    # 4) Precision/Recall/F1
    p = precision_score(y_true, y_pred, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, average=None, zero_division=0)
    f = f1_score(y_true, y_pred, average=None, zero_division=0)
    write_log("\nPer-class Precision/Recall/F1:")
    for i in range(NUM_CLASSES):
        write_log(f" Class {i}: P={p[i]:.4f}, R={r[i]:.4f}, F1={f[i]:.4f}")

    # 5) Per-class 2x2 confusion matrices (normalized to 1)
    write_log("\nPer-class 2x2 confusion matrices (normalized):")
    for c in range(NUM_CLASSES):
        tp = np.sum((y_pred[:,c]==1) & (y_true[:,c]==1))
        fp = np.sum((y_pred[:,c]==1) & (y_true[:,c]==0))
        fn = np.sum((y_pred[:,c]==0) & (y_true[:,c]==1))
        tn = np.sum((y_pred[:,c]==0) & (y_true[:,c]==0))
        total = tp + fp + fn + tn or 1
        write_log(f"\nClass{c}:")
        write_log(f"[{tp/total:.4f}, {fn/total:.4f}]")
        write_log(f"[{fp/total:.4f}, {tn/total:.4f}]")

    # 6) Per-class PPV / NPV (macro/micro); vectorized TP/FP/FN/TN
    tp = np.sum((y_pred == 1) & (y_true == 1), axis=0)
    fp = np.sum((y_pred == 1) & (y_true == 0), axis=0)
    fn = np.sum((y_pred == 0) & (y_true == 1), axis=0)
    tn = np.sum((y_pred == 0) & (y_true == 0), axis=0)

    # Per-class PPV/NPV with zero_division protection
    ppv = np.divide(tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0)
    npv = np.divide(tn, tn + fn, out=np.zeros_like(tn, dtype=float), where=(tn + fn) > 0)

    # Macro average: mean per class
    macro_ppv = float(np.mean(ppv))
    macro_npv = float(np.mean(npv))

    # Micro average: aggregate then compute
    TP, FP, FN, TN = tp.sum(), fp.sum(), fn.sum(), tn.sum()
    micro_ppv = float(TP / (TP + FP)) if (TP + FP) > 0 else 0.0
    micro_npv = float(TN / (TN + FN)) if (TN + FN) > 0 else 0.0

    write_log("\nPer-class PPV (Precision) and NPV:")
    for i in range(NUM_CLASSES):
        write_log(f" Class {i}: PPV={ppv[i]:.4f}, NPV={npv[i]:.4f}")

    write_log("\nAveraged PPV/NPV:")
    write_log(f" Macro-Avg PPV={macro_ppv:.4f}, NPV={macro_npv:.4f}")
    write_log(f" Micro-Avg PPV={micro_ppv:.4f}, NPV={micro_npv:.4f}")
    
    write_log("\n=== Test Evaluation Complete ===")
