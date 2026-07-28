"""
Demo Model Creator
------------------
Creates a quick demo model using ResNet18 pretrained on ImageNet.
This is just to test the app UI + Grad-CAM visualization.
For real predictions, run train.py after downloading the Kaggle dataset.
"""

import os
import torch
import torch.nn as nn
from torchvision import models

MODEL_SAVE_PATH = "app/model/breast_cancer_model.pth"

def create_demo_model():
    print("Creating demo model (ResNet18 with pretrained ImageNet weights)...")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)   # 2 classes: 0=non-cancerous, 1=cancerous

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"Demo model saved to: {MODEL_SAVE_PATH}")
    print("\nNOTE: This is a DEMO model — predictions won't be accurate.")
    print("To train a real model, download the Kaggle dataset and run: python train.py")
    print("\nNow go to the Streamlit app and upload any histopathology image!")

if __name__ == "__main__":
    create_demo_model()
