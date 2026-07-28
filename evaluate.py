"""
evaluate.py
Evaluates the trained model on the validation set and saves metrics.
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

# ---- Config (same as train.py) ----
DATA_DIR = "data/processed"
MODEL_SAVE_PATH = "app/model/breast_cancer_model.pth"
METRICS_SAVE_PATH = "app/model/metrics.json"
BATCH_SIZE = 32
IMG_SIZE = 50

def build_model(num_classes=2):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_ds = random_split(full_dataset, [train_size, val_size])

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model().to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds) * 100
    recall = recall_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds) * 100
    cm = confusion_matrix(all_labels, all_preds).tolist()

    metrics = {
        "accuracy": round(accuracy, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1, 2),
        "confusion_matrix": cm,
        "class_names": full_dataset.classes,
        "val_samples": len(val_ds)
    }

    print("\n--- Evaluation Results ---")
    print(f"Accuracy:  {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall:    {recall:.2f}%")
    print(f"F1-Score:  {f1:.2f}%")
    print(f"Confusion Matrix: {cm}")

    os.makedirs(os.path.dirname(METRICS_SAVE_PATH), exist_ok=True)
    with open(METRICS_SAVE_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {METRICS_SAVE_PATH}")

if __name__ == "__main__":
    main()