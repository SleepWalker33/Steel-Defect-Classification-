# threshold_tuning_and_evaluation.py

import os
import json
import time
import random
import shutil
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
import torchvision.transforms as transforms
from PIL import Image

from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

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

NUM_CLASSES_FULL   = 5  # [bg, d0, d1, d2, d3]
NUM_CLASSES_STAGE1 = 2
NUM_CLASSES_STAGE2 = 4

# Path configuration (override via environment variables)
VAL_DIR     = os.getenv("VAL_IMAGE_DIR",  "data/images/val")
TEST_DIR    = os.getenv("TEST_IMAGE_DIR", "data/images/test")
LABEL_FILE  = os.getenv("LABEL_FILE",     "data/labels/labels1.txt")
BEST_STAGE1 = "best_stage1.epoch"
BEST_STAGE2 = "best_stage2.epoch"
LOG_FILE    = "log_tune.txt"

BATCH_SIZE = 16
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Threshold search range: 0.1 to 0.8, step 0.1
THRESHOLDS = np.round(np.arange(0.1, 0.9, 0.1), 2)

def write_log(msg):
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')
    print(msg)

class MultiLabelDataset(Dataset):
    def __init__(self, samples, image_dir, transform=None):
        self.samples = samples
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, label = self.samples[idx]
        for ext in ('.jpg', '.png', '.jpeg'):
            path = os.path.join(self.image_dir, img_id + ext)
            if os.path.exists(path):
                img = Image.open(path).convert("RGB")
                break
        else:
            raise FileNotFoundError(f"{img_id} not found in {self.image_dir}")
        if self.transform:
            img = self.transform(img)
        return img_id, img, torch.tensor(label, dtype=torch.float32)

def get_image_ids_from_dir(img_dir):
    return sorted({os.path.splitext(n)[0]
                   for n in os.listdir(img_dir)
                   if n.lower().endswith(('.jpg','.png','.jpeg'))})

def build_model(num_classes):
    weights = ConvNeXt_Small_Weights.DEFAULT
    model = convnext_small(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model

def load_stage1_samples(label_file, image_ids):
    samples = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            img, raw = parts[0], parts[1:]
            if img not in image_ids: continue
            vec = [0, 0]
            if raw == ['-1']:
                vec[0] = 1
            else:
                vec[1] = 1
            samples.append((img, vec))
    return samples

def load_stage2_samples(label_file, image_ids):
    samples = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            img, raw = parts[0], parts[1:]
            if img not in image_ids or raw == ['-1']: continue
            vec = [0] * NUM_CLASSES_STAGE2
            for lb in raw:
                vec[int(lb)] = 1
            samples.append((img, vec))
    return samples

def get_true_labels(label_file, image_ids):
    gt = {}
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            img, raw = parts[0], parts[1:]
            if img not in image_ids: continue
            vec = [0] * NUM_CLASSES_FULL
            if raw == ['-1']:
                vec[0] = 1
            else:
                for lb in raw:
                    vec[int(lb)+1] = 1
            gt[img] = vec
    return gt

def run_pipeline(image_dir, threshold, model_s1, model_s2, device):
    img_ids = get_image_ids_from_dir(image_dir)
    gt_dict = get_true_labels(LABEL_FILE, img_ids)
    default = [1, 0, 0, 0, 0]

    # Stage 1: binary classification
    samples1 = load_stage1_samples(LABEL_FILE, set(img_ids))
    loader1 = DataLoader(MultiLabelDataset(samples1, image_dir, TRANSFORM),
                         batch_size=BATCH_SIZE, shuffle=False,
                         worker_init_fn=seed_worker, generator=g)
    model_s1.eval()
    stage1_cache, pos_ids = [], []
    with torch.no_grad():
        for ids, imgs, _ in loader1:
            imgs = imgs.to(device)
            probs = torch.sigmoid(model_s1(imgs)).cpu().numpy()
            preds = (probs > threshold).astype(int)
            for i, img_id in enumerate(ids):
                stage1_cache.append((img_id, preds[i]))
                if preds[i][1] == 1:
                    pos_ids.append(img_id)

    # Stage 2: multi-label
    samples2 = load_stage2_samples(LABEL_FILE, set(pos_ids))
    loader2 = DataLoader(MultiLabelDataset(samples2, image_dir, TRANSFORM),
                         batch_size=BATCH_SIZE, shuffle=False,
                         worker_init_fn=seed_worker, generator=g)
    model_s2.eval()
    stage2_dict = {}
    with torch.no_grad():
        for ids, imgs, _ in loader2:
            imgs = imgs.to(device)
            probs = torch.sigmoid(model_s2(imgs)).cpu().numpy()
            preds = (probs > threshold).astype(int)
            for i, img_id in enumerate(ids):
                stage2_dict[img_id] = preds[i]

    # Compose y_true and y_pred
    y_true_list, y_pred_list = [], []
    for img_id, pred1 in stage1_cache:
        y_true_list.append(gt_dict.get(img_id, default))
        if pred1[1] == 0:
            y_pred_list.append([1, 0, 0, 0, 0])
        else:
            y_pred_list.append([0] + list(stage2_dict.get(img_id, [0]*NUM_CLASSES_STAGE2)))

    return np.array(y_true_list), np.array(y_pred_list), img_ids

if __name__ == "__main__":
    # Clean old logs
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    model_s1 = build_model(NUM_CLASSES_STAGE1).to(device)
    model_s1.load_state_dict(torch.load(BEST_STAGE1))
    model_s2 = build_model(NUM_CLASSES_STAGE2).to(device)
    model_s2.load_state_dict(torch.load(BEST_STAGE2))

    # Validation threshold tuning
    best_th, best_score = None, -1
    write_log("Tuning thresholds on validation set...")
    for th in THRESHOLDS:
        y_true, y_pred, _ = run_pipeline(VAL_DIR, th, model_s1, model_s2, device)
        p = precision_score(y_true, y_pred, average=None, zero_division=0)
        r = recall_score(y_true, y_pred, average=None, zero_division=0)
        score = p.mean() + r.mean()
        write_log(f"Threshold={th:.2f} -> avg_P={p.mean():.4f}, avg_R={r.mean():.4f}, sum={score:.4f}")
        if score > best_score:
            best_score, best_th = score, th

    write_log(f"Best threshold found: {best_th:.2f}")

    # ========== Save validation-set prediction probabilities as JSON ==========
    write_log("\nSaving full validation set raw probabilities...")

    val_img_ids = get_image_ids_from_dir(VAL_DIR)

    # Stage 1 probabilities
    samples1_val = load_stage1_samples(LABEL_FILE, set(val_img_ids))
    loader1_val = DataLoader(MultiLabelDataset(samples1_val, VAL_DIR, TRANSFORM),
                             batch_size=BATCH_SIZE, shuffle=False,
                             worker_init_fn=seed_worker, generator=g)
    s1_probs_val = {}
    model_s1.eval()
    with torch.no_grad():
        for ids, imgs, _ in loader1_val:
            probs = torch.sigmoid(model_s1(imgs.to(device))).cpu().numpy()
            for i, img_id in enumerate(ids):
                s1_probs_val[img_id] = probs[i]

    # Stage 2 probabilities
    samples2_val = load_stage2_samples(LABEL_FILE, set(val_img_ids))
    loader2_val = DataLoader(MultiLabelDataset(samples2_val, VAL_DIR, TRANSFORM),
                             batch_size=BATCH_SIZE, shuffle=False,
                             worker_init_fn=seed_worker, generator=g)
    s2_probs_val = {}
    model_s2.eval()
    with torch.no_grad():
        for ids, imgs, _ in loader2_val:
            probs = torch.sigmoid(model_s2(imgs.to(device))).cpu().numpy()
            for i, img_id in enumerate(ids):
                s2_probs_val[img_id] = probs[i]

    # Concatenate and write JSON file
    val_json_out = []
    for img_id in val_img_ids:
        vec = np.zeros(NUM_CLASSES_FULL)
        if img_id in s1_probs_val:
            vec[0] = s1_probs_val[img_id][0]
            if img_id in s2_probs_val:
                vec[1:] = s2_probs_val[img_id]
            else:
                vec[1:] = 0  # not in stage 2 => zeros

        for cid in range(NUM_CLASSES_FULL):
            val_json_out.append({
                "image_id": img_id,
                "category_id": cid,
                "score": float(vec[cid])
            })

    with open("val_probabilities.json", "w") as f:
        json.dump(val_json_out, f, indent=2)
    write_log("Saved full probability results to val_probabilities.json")
    
    # Test-set evaluation
    write_log("\nEvaluating on test set with best threshold...")
    y_true, y_pred, _ = run_pipeline(TEST_DIR, best_th, model_s1, model_s2, device)

    # ========== Save test-set prediction probabilities as JSON ==========
    write_log("\nSaving full test set raw probabilities...")

    img_ids = get_image_ids_from_dir(TEST_DIR)

    # Stage 1 probabilities
    samples1 = load_stage1_samples(LABEL_FILE, set(img_ids))
    loader1 = DataLoader(MultiLabelDataset(samples1, TEST_DIR, TRANSFORM),
                         batch_size=BATCH_SIZE, shuffle=False,
                         worker_init_fn=seed_worker, generator=g)
    s1_probs = {}
    model_s1.eval()
    with torch.no_grad():
        for ids, imgs, _ in loader1:
            probs = torch.sigmoid(model_s1(imgs.to(device))).cpu().numpy()
            for i, img_id in enumerate(ids):
                s1_probs[img_id] = probs[i]

    # Stage 2 probabilities
    samples2 = load_stage2_samples(LABEL_FILE, set(img_ids))
    loader2 = DataLoader(MultiLabelDataset(samples2, TEST_DIR, TRANSFORM),
                         batch_size=BATCH_SIZE, shuffle=False,
                         worker_init_fn=seed_worker, generator=g)
    s2_probs = {}
    model_s2.eval()
    with torch.no_grad():
        for ids, imgs, _ in loader2:
            probs = torch.sigmoid(model_s2(imgs.to(device))).cpu().numpy()
            for i, img_id in enumerate(ids):
                s2_probs[img_id] = probs[i]

    # Concatenate and write JSON (set defects to 0 if not in Stage 2)
    json_out = []
    for img_id in img_ids:
        vec = np.zeros(NUM_CLASSES_FULL)
        if img_id in s1_probs:
            vec[0] = s1_probs[img_id][0]  # background probability
            if img_id in s2_probs:
                vec[1:] = s2_probs[img_id]  # defects from stage 2
            else:
                vec[1:] = 0  # not in stage 2 => zeros

        for cid in range(NUM_CLASSES_FULL):
            json_out.append({
                "image_id": img_id,
                "category_id": cid,
                "score": float(vec[cid])
            })

    with open("test_probabilities.json", "w") as f:
        json.dump(json_out, f, indent=2)
    write_log("Saved full probability results to test_probabilities.json")


    # Build confusion matrix
    cm = np.zeros((NUM_CLASSES_FULL, NUM_CLASSES_FULL), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        tc = list(np.where(yt == 1)[0])
        pc = list(np.where(yp == 1)[0])
        if len(tc) == 1 and len(pc) == 1:
            cm[pc[0], tc[0]] += 1
        else:
            max_len = max(len(tc), len(pc))
            tc_pad = tc + [0]*(max_len-len(tc))
            pc_pad = pc + [0]*(max_len-len(pc))
            for t in tc_pad:
                if t in pc_pad:
                    cm[t, t] += 1
                    pc_pad.remove(t)
            for t, p in zip([t for t in tc_pad if t not in pc_pad], pc_pad):
                cm[p, t] += 1

    df_cm = pd.DataFrame(cm)
    write_log("\nConfusion Matrix (counts):")
    write_log(df_cm.to_string())

    # Column-normalized percentages
    col_tot = df_cm.sum(axis=0).replace(0, 1)
    df_pct_col = df_cm.div(col_tot, axis=1) * 100
    write_log("\nConfusion Matrix (% per True class):")
    write_log(df_pct_col.round(2).to_string())

    # Row-normalized percentages
    row_tot = df_cm.sum(axis=1).replace(0, 1)
    df_pct_row = df_cm.div(row_tot, axis=0) * 100
    write_log("\nConfusion Matrix (% per Predicted class):")
    write_log(df_pct_row.round(2).to_string())

    # Precision/Recall/F1
    write_log("\nPer-class Precision/Recall/F1:")
    p = precision_score(y_true, y_pred, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, average=None, zero_division=0)
    f = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i in range(NUM_CLASSES_FULL):
        write_log(f" Class{i}: P={p[i]:.4f} R={r[i]:.4f} F1={f[i]:.4f}")

    # Per-class 2x2 confusion matrices (normalized to sum=1)
    write_log("\nPer-class 2x2 confusion matrices (normalized):")
    for c in range(NUM_CLASSES_FULL):
        tp = np.sum((y_pred[:,c]==1)&(y_true[:,c]==1))
        fp = np.sum((y_pred[:,c]==1)&(y_true[:,c]==0))
        fn = np.sum((y_pred[:,c]==0)&(y_true[:,c]==1))
        tn = np.sum((y_pred[:,c]==0)&(y_true[:,c]==0))
        total = tp+fp+fn+tn or 1
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
    for i in range(NUM_CLASSES_FULL):
        write_log(f" Class {i}: PPV={ppv[i]:.4f}, NPV={npv[i]:.4f}")

    write_log("\nAveraged PPV/NPV:")
    write_log(f" Macro-Avg PPV={macro_ppv:.4f}, NPV={macro_npv:.4f}")
    write_log(f" Micro-Avg PPV={micro_ppv:.4f}, NPV={micro_npv:.4f}")

    write_log(f"\nEvaluation completed in {(time.time())/3600:.2f} hours")
