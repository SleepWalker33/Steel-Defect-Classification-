
import os, random, time, json, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import convnext_small, ConvNeXt_Small_Weights
import torchvision.transforms as transforms
from PIL import Image

# ---------------- Global settings ----------------
SEED = 2
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

# Stage 1: binary classification (defect vs. background)
NUM_CLASSES_STAGE1 = 2
# Stage 2: four-class multi-label (defects 0/1/2/3)
NUM_CLASSES_STAGE2 = 4

BATCH_SIZE = 16
NUM_EPOCHS = 60
LR = 0.001
PATIENCE = 5
MIN_DELTA = 1e-4

TRAIN_IMAGE_DIR = os.getenv("TRAIN_IMAGE_DIR", "data/images/train")
VAL_IMAGE_DIR   = os.getenv("VAL_IMAGE_DIR",   "data/images/val")
LABEL_FILE      = os.getenv("LABEL_FILE",     "data/labels/labels1.txt")

LOG_STAGE1 = "log_train_stage1.txt"
LOG_STAGE2 = "log_train_stage2.txt"
BEST_STAGE1 = "best_stage1.epoch"
BEST_STAGE2 = "best_stage2.epoch"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def write_log(msg, path):
    with open(path, 'a') as f:
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
        image_id, label = self.samples[idx]
        for ext in ('.jpg', '.png', '.jpeg'):
            path = os.path.join(self.image_dir, image_id + ext)
            if os.path.exists(path):
                img = Image.open(path).convert("RGB")
                break
        else:
            raise FileNotFoundError(f"{image_id} not found")
        if self.transform:
            img = self.transform(img)
        return image_id, img, torch.tensor(label, dtype=torch.float32)

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

def build_model(num_classes):
    weights = ConvNeXt_Small_Weights.DEFAULT
    model = convnext_small(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model

def load_stage1_binary_samples(label_file, image_ids):
    samples = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            img_id, raw = parts[0], parts[1:]
            if img_id not in image_ids:
                continue
            vec = np.zeros(2, dtype=np.float32)
            if raw == ['-1']:
                vec[0] = 1.0  # no defect
            else:
                vec[1] = 1.0  # has defect
            samples.append((img_id, vec))
    return samples

def load_stage2_defect_samples(label_file, image_ids):
    samples = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            img_id, raw = parts[0], parts[1:]
            if img_id not in image_ids or raw == ['-1']:
                continue
            vec = np.zeros(4, dtype=np.float32)
            for lb in raw:
                vec[int(lb)] = 1.0
            samples.append((img_id, vec))
    return samples

def train_stage(stage_id, samples, num_classes, log_file, model_path):
    if os.path.exists(log_file):
        os.remove(log_file)
    write_log(f"=== Stage {stage_id} Training Start ===", log_file)

    loader = DataLoader(MultiLabelDataset(samples, TRAIN_IMAGE_DIR, transform),
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                        worker_init_fn=seed_worker,  # per-worker reproducibility
                        generator=g                  # reproducible shuffle
                        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_loss, total, correct = 0.0, 0, 0
        for _, imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).all(dim=1).sum().item()
            total += imgs.size(0)

        train_loss = epoch_loss / total
        acc = correct / total
        write_log(f"[Epoch {epoch}] Loss: {train_loss:.4f}, Acc: {acc:.4f}", log_file)

        if train_loss + MIN_DELTA < best_loss:
            best_loss = train_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            write_log(f"[Epoch {epoch}] Best model saved.", log_file)
        else:
            patience_counter += 1
            write_log(f"[Epoch {epoch}] No improvement. Patience {patience_counter}/{PATIENCE}", log_file)
            if patience_counter >= PATIENCE:
                write_log(f"[Early Stop] Training stopped at epoch {epoch}.", log_file)
                break

if __name__ == '__main__':
    start = time.time()
    train_ids = get_image_ids_from_dir(TRAIN_IMAGE_DIR)

    # Stage 1: train binary classifier (with/without defect)
    samples1 = load_stage1_binary_samples(LABEL_FILE, train_ids)
    train_stage(stage_id=1, samples=samples1, num_classes=2,
                log_file=LOG_STAGE1, model_path=BEST_STAGE1)

    # Stage 2: train multi-label classifier for defects 0-3
    samples2 = load_stage2_defect_samples(LABEL_FILE, train_ids)
    train_stage(stage_id=2, samples=samples2, num_classes=4,
                log_file=LOG_STAGE2, model_path=BEST_STAGE2)

    end = time.time()
    print(f"Total training time: {(end - start)/3600:.2f} hours")
