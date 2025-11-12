import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from torch.utils.tensorboard import SummaryWriter
from medsegbench import TnbcnucleiMSBench
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision

# ============================ CONFIG =============================
CONFIG = {
    'BATCH_SIZE': 8,
    'EPOCHS': 150,
    'LEARNING_RATE': 0.0005,
    'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'LOG_DIR': 'runs/tnbc_segmentation_original_unet',
    'MODEL_SAVE_PATH': 'best_model_original_unet.pth'
}

os.makedirs(CONFIG['LOG_DIR'], exist_ok=True)

# ============================ TRANSFORMS =============================
transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = TnbcnucleiMSBench(split="train", download=True, transform=transform, target_transform=transform, size=256)
val_dataset = TnbcnucleiMSBench(split="val", download=True, transform=transform, target_transform=transform, size=256)
test_dataset = TnbcnucleiMSBench(split="test", download=True, transform=transform, target_transform=transform, size=256)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ============================ U-NET MODEL =============================
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(3, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = conv_block(512, 1024)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = conv_block(128, 64)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return torch.sigmoid(self.final(d1))

# ============================ LOSS FUNCTIONS =============================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss()

    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)
        return self.alpha * (1 - pt) ** self.gamma * BCE_loss

criterion_bce = nn.BCELoss()
criterion_focal = FocalLoss()

def dice_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3))
    dice = (2. * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
    return 1 - dice.mean()

# ============================ TRAINING =============================
model = UNet().to(CONFIG['DEVICE'])
optimizer = optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'])
writer = SummaryWriter(CONFIG['LOG_DIR'])

# Model mimarisi TensorBoard'a yazdırılıyor:
writer.add_graph(model, torch.randn(1, 3, 256, 256).to(CONFIG['DEVICE']))

best_val_loss = float('inf')
for epoch in range(CONFIG['EPOCHS']):
    model.train()
    epoch_loss, epoch_acc = 0, 0
    for images, masks in train_loader:
        images, masks = images.to(CONFIG['DEVICE']), masks.to(CONFIG['DEVICE'])
        preds = model(images)
        loss = criterion_bce(preds, masks) + dice_loss(preds, masks) + criterion_focal(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        preds_bin = (preds > 0.5).float()
        epoch_acc += (preds_bin == masks).float().mean().item()

    avg_train_loss = epoch_loss / len(train_loader)
    avg_train_acc = epoch_acc / len(train_loader)

    model.eval()
    val_loss, val_acc = 0, []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(CONFIG['DEVICE']), masks.to(CONFIG['DEVICE'])
            preds = model(images)
            val_batch_loss = criterion_bce(preds, masks) + dice_loss(preds, masks) + criterion_focal(preds, masks)
            val_loss += val_batch_loss.item()
            preds_bin = (preds > 0.5).float()
            val_acc.append((preds_bin == masks).float().mean().item())

    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = np.mean(val_acc)

    writer.add_scalar('Loss/Train', avg_train_loss, epoch)
    writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
    writer.add_scalar('Accuracy/Train', avg_train_acc, epoch)
    writer.add_scalar('Accuracy/Validation', avg_val_acc, epoch)

    if epoch in [9, 64, 119,149]:
        model.eval()
        with torch.no_grad():
            sample_image, sample_mask = next(iter(val_loader))
            sample_image = sample_image.to(CONFIG['DEVICE'])
            sample_pred = model(sample_image)
            sample_pred_bin = (sample_pred > 0.5).float()

        writer.add_images(f"Validation/Input_Epoch_{epoch}", sample_image.cpu(), epoch)
        writer.add_images(f"Validation/Predicted_Epoch_{epoch}", sample_pred_bin.cpu(), epoch)
        writer.add_images(f"Validation/Target_Epoch_{epoch}", sample_mask.cpu(), epoch)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), CONFIG['MODEL_SAVE_PATH'])
        print(f"✅ Best model saved at epoch {epoch+1} with Val Loss: {avg_val_loss:.4f}")

    print(f"Epoch {epoch+1}/{CONFIG['EPOCHS']} - Train Loss: {avg_train_loss:.4f} - Train Acc: {avg_train_acc:.4f} - Val Loss: {avg_val_loss:.4f} - Val Acc: {avg_val_acc:.4f}")

# ============================ TESTING =============================
model.load_state_dict(torch.load(CONFIG['MODEL_SAVE_PATH']))
model.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(CONFIG['DEVICE'])
        preds = model(images)
        preds_bin = (preds > 0.5).cpu().numpy().flatten()
        masks = masks.numpy().flatten()
        all_preds.extend(preds_bin)
        all_targets.extend(masks)

acc = accuracy_score(all_targets, all_preds)
prec = precision_score(all_targets, all_preds, zero_division=0)
rec = recall_score(all_targets, all_preds, zero_division=0)
f1 = f1_score(all_targets, all_preds, zero_division=0)
iou = jaccard_score(all_targets, all_preds, zero_division=0)

print(f"Test Accuracy: {acc:.4f}")
print(f"Test Precision: {prec:.4f}")
print(f"Test Recall: {rec:.4f}")
print(f"Test F1-Score: {f1:.4f}")
print(f"Test IoU: {iou:.4f}")

# ============================ CONFUSION MATRIX =============================
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()
conf_matrix_img = Image.open("confusion_matrix.png")
writer.add_image("Confusion_Matrix", torchvision.transforms.ToTensor()(conf_matrix_img))

# ============================ TEST GÖRSELİ =============================
example_image, example_mask = next(iter(test_loader))
example_image = example_image.to(CONFIG['DEVICE'])
with torch.no_grad():
    pred = model(example_image)
    pred_bin = (pred > 0.5).float()

plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.title("Input Image")
plt.imshow(example_image.cpu().squeeze().permute(1, 2, 0), cmap="gray")
plt.subplot(1, 3, 2)
plt.title("Predicted Mask")
plt.imshow(pred_bin.cpu().squeeze(), cmap="gray")
plt.subplot(1, 3, 3)
plt.title("Ground Truth")
plt.imshow(example_mask.squeeze(), cmap="gray")
plt.tight_layout()
plt.savefig("test_visualization.png")
plt.show()

writer.close()
