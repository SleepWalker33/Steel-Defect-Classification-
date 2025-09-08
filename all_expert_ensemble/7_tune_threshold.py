import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score
from itertools import product
from tqdm import tqdm
from itertools import product
import time

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
    """Weighted voting logic for [bg, d0..d3].

    1) vote_sum = sum(w_i * vec_i)
    2) If vote_sum[0] >= 6 => pure background
    3) Else defect j is 1 if vote_sum[j] >= 3.5
    4) If 1 <= vote_sum[0] < 6 and any defect_bits => remove background
    5) If all defects are 0 => set background
    """
    results = []
    for vecs in vecs_per_sample:
        weighted = [np.array(vecs[i]) * method_weights[i] for i in range(len(vecs))]
        vote_sum = np.sum(weighted, axis=0).tolist()

        # 1) At least 6 votes for background => pure background
        if vote_sum[0] >= 6:
            results.append([1, 0, 0, 0, 0])
            continue

        # 2) Majority voting for defects
        defect_bits = [1 if vote_sum[i] >= 3.5 else 0 for i in range(1, NUM_CLASSES)]

        # 3) Conflict: 1~5 votes background and defects present => drop background
        bg_bit = 1 if sum(defect_bits) == 0 else 0
        if 1 <= vote_sum[0] < 6 and any(defect_bits):
            bg_bit = 0

        # 4) If all defects 0 => set background
        if sum(defect_bits) == 0:
            bg_bit = 1

        results.append([bg_bit] + defect_bits)
    return results

# ---------------- Main tuning pipeline ----------------
if __name__ == "__main__":
    selected_models = ['yolo11','yolo8', 'faster', 'vgg', 'resnet', 'convnext','effient']  # specify models to ensemble

    # Model prediction paths (override via env vars to keep private paths out of code)
    model_paths = {
        'detectors' : os.getenv('PATH_DETECTORS_VAL', 'path/to/detectors_val.json'),
        'yolo11'    : os.getenv('PATH_YOLO11_VAL', 'path/to/yolo11_val_predictions.json'),
        'yolo8'     : os.getenv('PATH_YOLO8_VAL',  'path/to/yolov8_val_predictions.json'),
        'faster'    : os.getenv('PATH_FASTER_VAL',  'path/to/faster_val.json'),
        'cascade'   : os.getenv('PATH_CASCADE_VAL', 'path/to/cascade_val.json'),
        'vgg'       : os.getenv('PATH_VGG_VAL',     'path/to/vgg_val_probabilities.json'),
        'resnet'    : os.getenv('PATH_RESNET_VAL',  'path/to/resnet_val_probabilities.json'),
        'convnext'  : os.getenv('PATH_CONVNEXT_VAL','path/to/convnext_val_probabilities.json'),
        'effient'   : os.getenv('PATH_EFFIENT_VAL', 'path/to/effient_val_probabilities.json')
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
    val_dir           = os.getenv('VAL_IMAGE_DIR',  'data/images/val')
    true_label_path   = os.getenv('TRUE_LABEL_JSON','data/labels/val.json')
    NUM_CLASSES = 5
    method_weights = [1.0, 1.5, 0.5, 2.0, 1.0 ,1.0, 0.0]  # adjust weights per model as needed
    default = [1, 0, 0, 0, 0]
    val_ids = [os.path.splitext(n)[0] for n in os.listdir(val_dir)]
    gt_labels = load_true_labels(true_label_path, NUM_CLASSES)

    # ========= Objective: compute score for a set of thresholds =========
    def evaluate_once(selected_models, preds_all, val_ids, y_true_np, method_weights, NUM_CLASSES=5):
        default = [1,0,0,0,0]
        y_pred_list = []
        for img in val_ids:
            vecs = [preds_all[i].get(img, default) for i in range(len(selected_models))]
            vote = apply_voting([vecs], method_weights, NUM_CLASSES)[0]
            y_pred_list.append(vote)
        y_pred_np = np.array(y_pred_list)
        p = precision_score(y_true_np, y_pred_np, average=None, zero_division=0)
        r = recall_score(y_true_np, y_pred_np, average=None, zero_division=0)
        score = np.mean(p) + np.mean(r)
        return score, p, r

    # ========= Preload ground truth to avoid recomputation =========
    default = [1,0,0,0,0]
    val_ids = [os.path.splitext(n)[0] for n in os.listdir(val_dir)]
    gt_labels = load_true_labels(true_label_path, NUM_CLASSES)
    y_true_list = [gt_labels.get(img, default) for img in val_ids]
    y_true_np = np.array(y_true_list)

    # ========= 1) Coarse grid =========
    COARSE = np.round(np.arange(0.2, 0.9, 0.2), 2)  # {0.2, 0.4, 0.6, 0.8}
    n_models = len(selected_models)

    # Cache predictions per model for each candidate threshold to avoid repeated I/O
    def build_preds_for_model(model, th_list):
        path = model_paths[model]
        fmt  = model_format[model]
        preds_by_th = {}
        for th in th_list:
            th_vec = [th]*NUM_CLASSES
            if fmt == 'bbox_detector':
                preds = load_bbox_format(path, is_yolo=False, score_thres=th_vec[:NUM_CLASSES-1], NUM_CLASSES=NUM_CLASSES)
            elif fmt == 'bbox_yolo':
                preds = load_bbox_format(path, is_yolo=True, score_thres=th_vec[:NUM_CLASSES-1], NUM_CLASSES=NUM_CLASSES)
            elif fmt == 'score_vector':
                preds = load_score_vector_format(path, score_thres=th_vec, NUM_CLASSES=NUM_CLASSES)
            else:
                raise ValueError(f"Unknown model format: {fmt}")
            preds_by_th[th] = preds
        return preds_by_th

    # Build cache for coarse grid
    coarse_cache = [build_preds_for_model(m, COARSE) for m in selected_models]

    from itertools import product
    coarse_results = []
    for th_tuple in tqdm(product(COARSE, repeat=n_models), total=(len(COARSE)**n_models), desc="Coarse grid", dynamic_ncols=True):
        preds_all = [coarse_cache[i][th_tuple[i]] for i in range(n_models)]
        score, p, r = evaluate_once(selected_models, preds_all, val_ids, y_true_np, method_weights, NUM_CLASSES)
        coarse_results.append({
            "ths": th_tuple,
            "score": score,
            "p": p, "r": r
        })

    # Take Top-K seeds for fine grid
    K = 20
    coarse_results.sort(key=lambda x: x["score"], reverse=True)
    seeds = coarse_results[:K]

    # ========= 2) Fine grid (refine around each seed) =========
    def neighbors(center, step=0.05, lower=0.05, upper=0.95):
        # Generate neighborhood candidates for a single threshold (center±step)
        cand = [center]
        a = round(center - step, 2)
        b = round(center + step, 2)
        if a >= lower: cand.append(a)
        if b <= upper: cand.append(b)
        # Deduplicate and sort
        cand = sorted(set(cand))
        return cand

    # Build a dynamic cache for fine grid (only what is needed)
    fine_cache = [dict() for _ in range(n_models)]  # fine_cache[i][th] = preds

    def get_preds(i_model, th):
        if th in coarse_cache[i_model]:
            return coarse_cache[i_model][th]
        if th in fine_cache[i_model]:
            return fine_cache[i_model][th]
        # Build and cache
        model = selected_models[i_model]
        path  = model_paths[model]
        fmt   = model_format[model]
        th_vec = [th]*NUM_CLASSES
        if fmt == 'bbox_detector':
            preds = load_bbox_format(path, is_yolo=False, score_thres=th_vec[:NUM_CLASSES-1], NUM_CLASSES=NUM_CLASSES)
        elif fmt == 'bbox_yolo':
            preds = load_bbox_format(path, is_yolo=True, score_thres=th_vec[:NUM_CLASSES-1], NUM_CLASSES=NUM_CLASSES)
        elif fmt == 'score_vector':
            preds = load_score_vector_format(path, score_thres=th_vec, NUM_CLASSES=NUM_CLASSES)
        else:
            raise ValueError(f"Unknown model format: {fmt}")
        fine_cache[i_model][th] = preds
        return preds

    all_fine_results = []   # store all fine-grid evaluations
    best_overall = None

    for sidx, seed in enumerate(seeds, 1):
        per_dim_cands = [neighbors(seed["ths"][i], step=0.05, lower=0.05, upper=0.95) for i in range(n_models)]
        total = np.prod([len(x) for x in per_dim_cands])
        for idxs in tqdm(product(*per_dim_cands), total=total,
                        desc=f"Fine grid {sidx}/{len(seeds)}", dynamic_ncols=True, leave=False):
            preds_all = [get_preds(i, idxs[i]) for i in range(n_models)]
            score, p, r = evaluate_once(selected_models, preds_all, val_ids, y_true_np, method_weights, NUM_CLASSES)
            rec = {"ths": idxs, "score": score, "p": p, "r": r}
            all_fine_results.append(rec)
            if (best_overall is None) or (score > best_overall["score"]):
                best_overall = rec
            
    # ========= Summary output =========
    print("\n[Best by coarse-to-fine]")
    print("thresholds:", best_overall["ths"])
    print("score (avgP+avgR):", best_overall["score"])
    print("P:", best_overall["p"])
    print("R:", best_overall["r"])

    # ========= Print Top-10 =========
    all_fine_results.sort(key=lambda x: x["score"], reverse=True)
    top10 = all_fine_results[:10]

    rows = []
    for rec in top10:
        row = {"score": rec["score"]}
        # Per-model thresholds
        for i, th in enumerate(rec["ths"]):
            row[f"th{i+1}"] = th
        # Per-class precision/recall
        for i in range(NUM_CLASSES):
            row[f"P{i}"] = rec["p"][i]
            row[f"R{i}"] = rec["r"][i]
        rows.append(row)

    df_top10 = pd.DataFrame(rows)
    print("\n=== Top 10 Threshold Combinations (coarse-to-fine) ===")
    print(df_top10.to_string(index=False))
