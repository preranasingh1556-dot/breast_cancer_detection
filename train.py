"""
Breast Cancer Detection - Training Script
Dataset : IDC Histopathology (download from Kaggle - see README)
Model   : ResNet18 with Transfer Learning
"""

import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# ─── Config ────────────────────────────────────────────────────────────────────
DATA_DIR        = "data/processed"          # run prepare_data.py first
MODEL_SAVE_PATH = "app/model/breast_cancer_model.pth"
EPOCHS          = 10
BATCH_SIZE      = 32
LEARNING_RATE   = 0.001
IMG_SIZE        = 50
# ───────────────────────────────────────────────────────────────────────────────


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def build_model(num_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Freeze all layers except the final classifier
    for param in model.parameters():
        param.requires_grad = False
    # Replace final fully-connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total  += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return total_loss / len(loader), 100 * correct / total


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(DATA_DIR):
        print(f"\nData folder '{DATA_DIR}' not found.")
        print("Please run:  python prepare_data.py  first.\n")
        return

    train_tf, val_tf = get_transforms()

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tf)
    print(f"Total images: {len(full_dataset)}  |  Classes: {full_dataset.classes}")

    train_size = int(0.8 * len(full_dataset))
    val_size   = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # Apply separate transforms to validation set
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, correct, total = 0, 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted  = outputs.max(1)
            total         += labels.size(0)
            correct       += predicted.eq(labels).sum().item()

        train_acc = 100 * correct / total
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train Acc: {train_acc:.2f}%  |  Val Loss: {val_loss:.4f}  |  Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  Saved best model (val_acc={val_acc:.2f}%)")

    print(f"\nTraining complete! Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
