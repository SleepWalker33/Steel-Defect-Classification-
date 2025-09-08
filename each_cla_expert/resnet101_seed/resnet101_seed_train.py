# vgg_train.py
import os, random, shutil, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet101, ResNet101_Weights
import torchvision.transforms as transforms
from PIL import Image

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# dataloader worker seeding
def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Generator for reproducible shuffling
g = torch.Generator()
g.manual_seed(SEED)

threshold = 0.5
NUM_CLASSES = 5
NUM_EPOCHS = 60
BATCH_SIZE = 16
LR = 0.001
PATIENCE = 5
MIN_DELTA = 1e-4

TRAIN_IMAGE_DIR = os.getenv("TRAIN_IMAGE_DIR", "data/images/train")
VAL_IMAGE_DIR   = os.getenv("VAL_IMAGE_DIR",   "data/images/val")
LABEL_FILE      = os.getenv("LABEL_FILE",     "data/labels/labels1.txt")

BEST_MODEL_PATH = "best.epoch"
LOG_FILE = "log_train.txt"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# if os.path.exists(LOG_FILE):
#     os.remove(LOG_FILE)
# Ensure creating a new empty file (clear old content)
# with open(LOG_FILE, 'w') as f:
#     f.write(f"=== Training Log Start ===\n")
def init_train_log():
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    with open(LOG_FILE, 'w') as f:
        f.write("=== Training Log Start ===\n")
    
class MultiLabelDataset(Dataset):
    def __init__(self, samples, image_dir, transform=None):
        self.samples = samples
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, label = self.samples[idx]
        for ext in ('.jpg', '.png', '.jpeg'):
            path = os.path.join(self.image_dir, image_id + ext)
            if os.path.exists(path):
                img = Image.open(path).convert("RGB")
                break
        else:
            raise FileNotFoundError(f"{image_id} not found in {self.image_dir}")
        if self.transform:
            img = self.transform(img)
        return image_id, img, torch.tensor(label, dtype=torch.float32)

def load_samples(label_file, image_ids):
    samples = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            img_id, raw = parts[0], parts[1:]
            if img_id not in image_ids:
                continue
            vec = np.zeros(NUM_CLASSES, dtype=np.float32)
            if raw == ['-1']:
                vec[0] = 1.0
            else:
                for lb in raw:
                    vec[int(lb) + 1] = 1.0
            samples.append((img_id, vec))
    return samples

def build_model():
    m = resnet101(weights=ResNet101_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m

def write_log(msg):
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')
    print(msg)

# def get_image_ids_from_dir(img_dir):
#     ids = set()
#     for name in os.listdir(img_dir):
#         if name.endswith(('.jpg', '.png', '.jpeg')):
#             ids.add(os.path.splitext(name)[0])
#     return ids

def get_image_ids_from_dir(img_dir):
    names = [os.path.splitext(n)[0]
             for n in os.listdir(img_dir)
             if n.lower().endswith(('.jpg','.png','.jpeg'))]
    return sorted(names)


def build_confusion_and_trace(y_true, y_pred, img_ids, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    misclassified_by_pair = dict()

    for idx in range(len(y_true)):
        true_vec = y_true[idx]
        pred_vec = y_pred[idx]
        true_classes = list(np.where(true_vec == 1)[0])
        pred_classes = list(np.where(pred_vec == 1)[0])

        # If both have only one label, count directly
        if len(true_classes) == 1 and len(pred_classes) == 1:
            t = true_classes[0]
            p = pred_classes[0]
            cm[p][t] += 1
            if p != t:
                misclassified_by_pair.setdefault((p, t), []).append((img_ids[idx], [t], [p]))
            continue

        # Align number of labels
        max_len = max(len(true_classes), len(pred_classes))
        pad_true = true_classes + [0] * (max_len - len(true_classes))
        pad_pred = pred_classes + [0] * (max_len - len(pred_classes))

        # Set intersection for correct predictions
        matched = []
        for t in pad_true:
            if t in pad_pred:
                cm[t][t] += 1
                matched.append(t)
                pad_pred.remove(t)

        # Remove matched labels from pad_true
        unmatched_true = [t for t in pad_true if t not in matched]

        # Pair remaining unmatched labels in order
        for t, p in zip(unmatched_true, pad_pred):
            cm[p][t] += 1
            if p != t:
                misclassified_by_pair.setdefault((p, t), []).append(
                    (img_ids[idx], sorted(true_classes), sorted(pred_classes))
                )

    return cm, misclassified_by_pair


def train():
    init_train_log() 
    train_ids = get_image_ids_from_dir(TRAIN_IMAGE_DIR)
    val_ids = get_image_ids_from_dir(VAL_IMAGE_DIR)

    train_samples = load_samples(LABEL_FILE, train_ids)
    val_samples = load_samples(LABEL_FILE, val_ids)

    train_loader = DataLoader(MultiLabelDataset(train_samples, TRAIN_IMAGE_DIR, transform),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                              worker_init_fn=seed_worker,  # per-worker reproducibility
                              generator=g                  # reproducible shuffle
                              )
    val_loader = DataLoader(MultiLabelDataset(val_samples, VAL_IMAGE_DIR, transform),
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss, total, correct = 0.0, 0, 0
        for _, imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(outputs) > threshold ).float()
            correct += (preds == labels).all(dim=1).sum().item()
            total += imgs.size(0)

        train_loss = epoch_loss / total
        acc = correct / total
        write_log(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Acc: {acc:.4f}")

        if train_loss + MIN_DELTA < best_loss:
            best_loss = train_loss
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            write_log(f"[Epoch {epoch}] Best model saved.")
        else:
            patience_counter += 1
            write_log(f"[Epoch {epoch}] No improvement. Patience {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                write_log(f"[Early Stop] Training stopped at epoch {epoch}.")
                break

if __name__ == '__main__':
    start = time.time()
    train()
    end = time.time() 
    write_log(f"Training completed in {(end - start)/3600} hours")
