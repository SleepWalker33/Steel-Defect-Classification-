import os
import json
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score

# ===================== Configuration =====================
# Fixed thresholds (same per model/class); can be tuned
FIXED_TH_LIST = [0.4,0.35,0.35,0.8,0.55,0.15,0.15]

# Weight search min/max and initial step; sum(weights)=#models
WEIGHT_MIN = 0.1
WEIGHT_MAX = 3.7
STEP_INIT  = 0.1

# Auto coarsen step if combinations explode
AUTO_COARSEN = True
MAX_COMB     = 500_000
COARSEN_CAND = [0.2, 0.25, 0.5]

TOP_K = 10  # output top-K weight combinations

# ===================== Data loaders =====================
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
    """Weighted voting for [bg, d0..d3] with background conflict handling.

    1) vote_sum = sum(w_i * vec_i)
    2) If vote_sum[0] >= 6 => pure background
    3) Else defect j is 1 if vote_sum[j] >= 3.5
    4) If 1 <= vote_sum[0] < 6 and any defect_bits => drop background
    5) If all defects are 0 => set background
    """
    results = []
    for vecs in vecs_per_sample:
        weighted = [np.array(vecs[i]) * method_weights[i] for i in range(len(vecs))]
        vote_sum = np.sum(weighted, axis=0).tolist()

        # 1) At least 6 votes background => pure background
        if vote_sum[0] >= 6:
            results.append([1, 0, 0, 0, 0])
            continue

        # 2) Majority voting for defects
        defect_bits = [1 if vote_sum[i] >= 3.5 else 0 for i in range(1, NUM_CLASSES)]

        # 3) Conflict: 1~5 background votes and defects present => drop background
        bg_bit = 1 if sum(defect_bits) == 0 else 0
        if 1 <= vote_sum[0] < 6 and any(defect_bits):
            bg_bit = 0

        # 4) If all defects 0 => set background
        if sum(defect_bits) == 0:
            bg_bit = 1

        results.append([bg_bit] + defect_bits)
    return results

# ===================== DP combination estimation & generator =====================
def build_weight_grid(step):
    """Return integerized bounds/scale for weight grid (unit=step)."""
    # Integer grid to avoid fp accumulation; unit=step
    scale = int(round(1.0/step))
    L = int(round(WEIGHT_MIN * scale))
    U = int(round(WEIGHT_MAX * scale))
    return scale, L, U

def count_bounded_compositions(M, S_units, L_units, U_units):
    """Count combinations: x_i in [L,U]∩Z with sum x_i = S. O(M*S*(U-L))."""
    # Shift to non-negative: y_i = x_i - L; sum y_i = S - M*L; and 0 <= y_i <= (U-L)
    S2 = S_units - M*L_units
    if S2 < 0:
        return 0
    cap = U_units - L_units
    dp = np.zeros(S2+1, dtype=np.int64)
    dp[0] = 1
    for _ in range(M):
        ndp = np.zeros_like(dp)
        # Bounded knapsack with prefix sums optimization
        cumsum = np.cumsum(dp)
        for s in range(S2+1):
            left = s - cap - 1
            ndp[s] = cumsum[s] - (cumsum[left] if left >= 0 else 0)
        dp = ndp
    return int(dp[S2])

def gen_bounded_compositions(M, S_units, L_units, U_units):
    """Generate all combinations (integer units). DFS with pruning."""
    remain = S_units
    chosen = []

    def dfs(i, rem):
        if i == M - 1:
            x = rem
            if L_units <= x <= U_units:
                yield chosen + [x]
            return
        # Pruning ranges
        min_sum = (M - 1 - i) * L_units
        max_sum = (M - 1 - i) * U_units
        lo = max(L_units, rem - max_sum)
        hi = min(U_units, rem - min_sum)
        for x in range(lo, hi + 1):
            chosen.append(x)
            yield from dfs(i + 1, rem - x)
            chosen.pop()

    yield from dfs(0, remain)

# ===================== Main =====================
if __name__ == "__main__":
    # ------ Model configuration ------
    selected_models = ['yolo11','yolo8','faster','vgg','resnet','convnext','effient']  # 7 models
    n_models = len(selected_models)
    assert len(FIXED_TH_LIST) == n_models, "FIXED_TH_LIST length must equal selected_models"
    
    # Provide paths via env vars to remove any personal absolute paths from code
    model_paths = {
        'detectors' : os.getenv('PATH_DETECTORS_VAL', 'path/to/detectors_val.json'),
        'yolo11'    : os.getenv('PATH_YOLO11_VAL',    'path/to/yolo11_val_predictions.json'),
        'yolo8'     : os.getenv('PATH_YOLO8_VAL',     'path/to/yolov8_val_predictions.json'),
        'faster'    : os.getenv('PATH_FASTER_VAL',    'path/to/faster_val.json'),
        'cascade'   : os.getenv('PATH_CASCADE_VAL',   'path/to/cascade_val.json'),
        'vgg'       : os.getenv('PATH_VGG_VAL',       'path/to/vgg_val_probabilities.json'),
        'resnet'    : os.getenv('PATH_RESNET_VAL',    'path/to/resnet_val_probabilities.json'),
        'convnext'  : os.getenv('PATH_CONVNEXT_VAL',  'path/to/convnext_val_probabilities.json'),
        'effient'   : os.getenv('PATH_EFFIENT_VAL',   'path/to/effient_val_probabilities.json')
    }
    model_format = {
        'detectors':'bbox_detector',
        'faster'  :'bbox_detector',
        'cascade' :'bbox_detector',
        'yolo11'  :'bbox_yolo',
        'yolo8'   :'bbox_yolo',
        'vgg'     :'score_vector',
        'resnet'  :'score_vector',
        'convnext':'score_vector',
        'effient' :'score_vector'
    }

    val_dir         = os.getenv('VAL_IMAGE_DIR', 'data/images/val')
    true_label_path = os.getenv('TRUE_LABEL_JSON', 'data/labels/val.json')
    NUM_CLASSES = 5

    # ------ Preload ground truth ------
    default = [1,0,0,0,0]
    val_ids = [os.path.splitext(n)[0] for n in os.listdir(val_dir)]
    gt_labels = load_true_labels(true_label_path, NUM_CLASSES)
    y_true_list = [gt_labels.get(img, default) for img in val_ids]
    y_true_np = np.array(y_true_list)

    # ------ Build predictions per model with fixed threshold ------
    def build_preds_fixed_th(model, fixed_th):
        path = model_paths[model]
        fmt  = model_format[model]
        th_vec = [fixed_th]*NUM_CLASSES
        if fmt == 'bbox_detector':
            return load_bbox_format(path, is_yolo=False, score_thres=th_vec[:NUM_CLASSES-1], NUM_CLASSES=NUM_CLASSES)
        elif fmt == 'bbox_yolo':
            return load_bbox_format(path, is_yolo=True,  score_thres=th_vec[:NUM_CLASSES-1], NUM_CLASSES=NUM_CLASSES)
        elif fmt == 'score_vector':
            return load_score_vector_format(path, score_thres=th_vec, NUM_CLASSES=NUM_CLASSES)
        else:
            raise ValueError(f"Unknown model format: {fmt}")

    preds_per_model = [
        build_preds_fixed_th(m, FIXED_TH_LIST[i]) 
        for i, m in enumerate(selected_models)
    ]

    # ------ Estimate combinations & optional step coarsening ------
    step = STEP_INIT
    while True:
        scale, L_units, U_units = build_weight_grid(step)
        S_units = int(round(n_models * 1.0 * scale))  # sum w_i = n_models
        total_comb = count_bounded_compositions(n_models, S_units, L_units, U_units)

        print(f"\n[Estimate] models={n_models}, step={step}, "
              f"candidate per dim=[{WEIGHT_MIN},{WEIGHT_MAX}] -> "
              f"combinations ≈ {total_comb:,}")

        if (not AUTO_COARSEN) or (total_comb <= MAX_COMB):
            break
        # Auto-coarsen
        next_step = None
        for s in COARSEN_CAND:
            if s > step:
                next_step = s
                break
        if next_step is None:
            print("Step cannot be coarsened further; continuing (may be slow).")
            break
        print(f"Too many combinations (>{MAX_COMB:,}); coarsen step from {step} to {next_step} ...")
        step = next_step

    # ------ Weight search: generate all integer-unit combinations summing to S ------
    # Keep Top-K
    topK = []
    def maybe_push_top(rec, K=TOP_K):
        topK.append(rec)
        topK.sort(key=lambda x: x["score"], reverse=True)
        if len(topK) > K:
            topK.pop()

    # Speed up evaluation: capture preds_per_model for closure use
    def evaluate_weights(weight_units_vec):
        # Map integer units -> floating weights
        method_weights = np.array([wu/scale for wu in weight_units_vec], dtype=float)
        # Voting
        y_pred_list = []
        for img in val_ids:
            vecs = [preds_per_model[i].get(img, default) for i in range(n_models)]
            vote = apply_voting([vecs], method_weights, NUM_CLASSES)[0]
            y_pred_list.append(vote)
        y_pred_np = np.array(y_pred_list)
        p = precision_score(y_true_np, y_pred_np, average=None, zero_division=0)
        r = recall_score(y_true_np, y_pred_np, average=None, zero_division=0)
        score = float(np.mean(p) + np.mean(r))
        return score, p, r, method_weights

    best = None
    # Iterate over the generator with a progress display
    with tqdm(total=total_comb, desc="Weight search", dynamic_ncols=True) as pbar:
        for w_units in gen_bounded_compositions(n_models, S_units, L_units, U_units):
            s, p, r, w_real = evaluate_weights(w_units)
            rec = {"weights": w_real, "score": s, "p": p, "r": r}
            if (best is None) or (s > best["score"]):
                best = rec
            maybe_push_top(rec, TOP_K)
            pbar.update(1)

    # ------ Print best and Top-K ------
    print("\n[Best weights]")
    print("weights:", np.round(best["weights"], 4).tolist())
    print("score (avgP+avgR):", best["score"])
    print("avg_precision:", float(np.mean(best["p"])), "avg_recall:", float(np.mean(best["r"])))
    print("P:", best["p"])
    print("R:", best["r"])

    # TopK DataFrame
    rows = []
    for rec in topK:
        row = {
            "score": rec["score"],
            "avg_precision": float(np.mean(rec["p"])),
            "avg_recall": float(np.mean(rec["r"]))
        }
        for i, w in enumerate(rec["weights"]):
            row[f"w{i+1}"] = float(w)
        for i in range(NUM_CLASSES):
            row[f"P{i}"] = float(rec["p"][i])
            row[f"R{i}"] = float(rec["r"][i])
        rows.append(row)
    df_top = pd.DataFrame(rows)
    print("\n=== Top 10 Weight Combinations (fixed threshold) ===")
    print(df_top.to_string(index=False))
