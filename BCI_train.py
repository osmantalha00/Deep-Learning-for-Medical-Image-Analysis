import os
import time
import random
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import io

# -------------------- Config --------------------
CONFIG = dict(
    IMAGE_SIZE=(224, 224),
    BATCH_SIZE=32,
    EPOCHS=200,
    LEARNING_RATE=0.0005,
    SEED=42,
    TRAIN_DIR="BCI_dataset/train",
    TEST_DIR="BCI_dataset/test",
    SAVE_MODEL_PATH="best_model.pth",
    LOG_DIR="runs/bci_exp",
    RESULTS_DIR="results",
    DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
os.makedirs(CONFIG['RESULTS_DIR'], exist_ok=True)

# -------------------- Seed --------------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed(CONFIG['SEED'])

# -------------------- Dataset --------------------
def get_transform():
    return transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomCrop(CONFIG['IMAGE_SIZE']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

transforms_by_label = {
    0: get_transform(),
    1: get_transform(),
    2: get_transform(),
    3: get_transform()
}

basic_transform = transforms.Compose([
    transforms.Resize(CONFIG['IMAGE_SIZE']),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

class BCIDataset(Dataset):
    label_map = {"0": 0, "1+": 1, "2+": 2, "3+": 3}

    def __init__(self, image_paths, mode='train'):
        self.image_paths = image_paths
        self.mode = mode

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')
        label_str = os.path.basename(path).split('_')[-1].replace('.png', '')
        label = self.label_map[label_str]
        transform = transforms_by_label[label] if self.mode == 'train' else basic_transform
        return transform(image), label

# -------------------- Model --------------------
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        feature_layers = [
            nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        ]
        self.features = nn.Sequential(*feature_layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# -------------------- Helper --------------------
def plot_confusion_matrix(cm, path):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=["0", "1+", "2+", "3+"], yticklabels=["0", "1+", "2+", "3+"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path)

# -------------------- Train --------------------
def train():
    writer = SummaryWriter(CONFIG['LOG_DIR'])
    all_images = glob.glob(os.path.join(CONFIG['TRAIN_DIR'], '*.png'))
    labels = [os.path.basename(p).split('_')[-1].replace('.png', '') for p in all_images]
    numeric_labels = [BCIDataset.label_map[l] for l in labels]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=CONFIG['SEED'])
    train_idx, val_idx = next(sss.split(all_images, numeric_labels))
    train_paths = [all_images[i] for i in train_idx]
    val_paths = [all_images[i] for i in val_idx]

    train_loader = DataLoader(BCIDataset(train_paths, mode='train'), batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(BCIDataset(val_paths, mode='val'), batch_size=CONFIG['BATCH_SIZE'])

    model = CNNModel().to(CONFIG['DEVICE'])
    class_counts = Counter([os.path.basename(p).split('_')[-1].replace('.png', '') for p in train_paths])
    total = sum(class_counts.values())
    weights = torch.tensor([total / class_counts[k] for k in ["0", "1+", "2+", "3+"]]).to(CONFIG['DEVICE'])

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

    best_val_loss = float('inf')
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(CONFIG['DEVICE']), y.to(CONFIG['DEVICE'])
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
        train_acc = 100 * correct / total

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(CONFIG['DEVICE']), y.to(CONFIG['DEVICE'])
                preds = model(x)
                loss = criterion(preds, y)
                val_loss += loss.item()
                val_correct += (preds.argmax(1) == y).sum().item()
                val_total += y.size(0)
        val_acc = 100 * val_correct / val_total
        scheduler.step(val_loss)

        writer.add_scalars("Loss", {"Train": train_loss/len(train_loader), "Val": val_loss/len(val_loader)}, epoch)
        writer.add_scalars("Accuracy", {"Train": train_acc, "Val": val_acc}, epoch)

        print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CONFIG['SAVE_MODEL_PATH'])
            print("[INFO] En iyi model kaydedildi.")

    writer.close()
    test_model(model, writer)

# -------------------- Test --------------------
def test_model(model, writer):
    model.load_state_dict(torch.load(CONFIG['SAVE_MODEL_PATH']))
    model.eval()
    test_loader = DataLoader(BCIDataset(glob.glob(os.path.join(CONFIG['TEST_DIR'], '*.png')), mode='val'), batch_size=CONFIG['BATCH_SIZE'])

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(CONFIG['DEVICE'])
            outputs = model(x)
            preds = outputs.argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())

    acc = 100 * np.mean(np.array(y_true) == np.array(y_pred))
    print(f"\n[TEST] Accuracy: {acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["0", "1+", "2+", "3+"]))

    cm = confusion_matrix(y_true, y_pred, normalize='true')
    cm_path = os.path.join(CONFIG['RESULTS_DIR'], "confusion_matrix.png")
    plot_confusion_matrix(cm, cm_path)
    print(f"Confusion matrix saved as '{cm_path}'")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image = transforms.ToTensor()(image)
    writer.add_image("Confusion_Matrix", image)
    writer.close()

if __name__ == '__main__':
    train()
