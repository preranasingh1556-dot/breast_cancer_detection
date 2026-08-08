# 🩺 Breast Cancer Detection with Grad-CAM Explainability

An AI-powered web application that analyzes microscopic breast tissue images
(histopathology patches) and predicts whether they show signs of **Invasive
Ductal Carcinoma (IDC)** — the most common subtype of breast cancer,
responsible for roughly 80% of all cases. Unlike a typical "black box" model,
this tool **visually explains** which parts of the image influenced its
decision using **Grad-CAM (Gradient-weighted Class Activation Mapping)**, and
generates a plain-language explanation of the result using the Gemini API.

---

## ✨ Features

- **Image Upload Interface** — drag-and-drop histopathology patch upload (Streamlit)
- **Real-time Prediction** — instant IDC classification with confidence score
- **Class Probability Breakdown** — visual bars for both classes, not just a binary label
- **Grad-CAM Heatmap Visualization** — three-panel view (original image, raw heatmap,
  overlay) showing exactly where the model "looked"
- **AI-Generated Plain-Language Explanation** — Gemini API translates the technical
  prediction into a human-readable explanation
- **Model Performance Dashboard** — Accuracy, Precision, Recall, and F1-Score on a
  held-out validation set, so users can judge how trustworthy the model is
- **Medical Disclaimer** — clearly marked as educational-use-only

---

## 🧠 Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch, torchvision |
| Model | ResNet18 (ImageNet-pretrained, fine-tuned via transfer learning) |
| Explainability | Grad-CAM (custom implementation) |
| Web App | Streamlit |
| AI Explanation | Google Gemini API (`gemini-3.5-flash`) |
| Evaluation | scikit-learn |
| Image Processing | OpenCV, PIL, NumPy |

---

## 📊 Model Performance

Evaluated on a held-out, patient-wise validation split (no patient overlap
between train/val/test to avoid data leakage):

| Metric | Score |
|---|---|
| Accuracy | 83.85% |
| Precision | 84.34% |
| Recall | 83.16% |
| F1 Score | 83.74% |
| Validation samples | 4,000 |

---

## 📁 Dataset

[**Breast Histopathology Images**](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)
(Kaggle) — 277,524 patches (50×50 px) extracted from 162 whole-slide breast
cancer specimens, labeled IDC-positive or IDC-negative.

> The dataset is **not included** in this repository due to its size (~5.8 GB).
> See [Setup](#-setup--installation) below to download it yourself.

---

## 📂 Project Structure

```
breast_cancer_detection/
├── app.py                   # Streamlit web app (main entry point)
├── gradcam.py                # Grad-CAM implementation
├── train.py                  # Model training script
├── evaluate.py                # Model evaluation script
├── prepare_data.py            # Dataset scanning + train/val/test split
├── check_models.py            # Lists available Gemini models for your API key
├── create_demo_model.py
├── generate_sample_image.py
├── model/
│   ├── breast_cancer_model.pth   # Trained model weights
│   └── metrics.json               # Saved evaluation metrics
├── data/                       # Dataset (gitignored — see Setup)
├── requirements.txt
├── .env.example                 # Template for API key config
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```powershell
git clone <your-repo-url>
cd breast_cancer_detection
```

### 2. Create and activate a virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```

### 4. Download the dataset (for retraining — optional)
Download [Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)
from Kaggle and extract it into `data/IDC_regular_ps50_idx5/`.

### 5. Add your Gemini API key
Create a `.env` file in the project root (use `.env.example` as a template):
```
GOOGLE_API_KEY=your_api_key_here
```
Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

### 6. Run the app
```powershell
streamlit run app.py
```
Open the URL shown in the terminal (typically `http://localhost:8501`).

---

## 🔍 How It Works

1. User uploads a histopathology patch image (`.jpg` / `.png`)
2. Image is resized, normalized, and converted to a tensor
3. A fine-tuned ResNet18 classifies it as Cancerous or Non-Cancerous
4. Grad-CAM analyzes gradients in the model's convolutional layers to
   generate a heatmap of influential regions
5. The heatmap is overlaid on the original image (warm regions = high influence)
6. The Gemini API generates a plain-language explanation of the result
7. Validation metrics are displayed for transparency

---

## ⚠️ Limitations

- Trained on a specific dataset (IDC histopathology patches at 40x
  magnification); may not generalize to other cancer types or imaging conditions
- A research/educational prototype — **not validated for clinical use or
  regulatory approval**
- Grad-CAM highlights *correlated* regions, not necessarily causally correct
  medical reasoning — it should support, not replace, expert judgment

## 🩹 Medical Disclaimer

This tool is intended for **educational and research purposes only**. It is
not a diagnostic device and should never be used as a substitute for
professional medical evaluation.

---

## 🚀 Future Enhancements

- Downloadable PDF report combining image, prediction, and heatmap
- Out-of-distribution detection to flag invalid uploads
- Similar-case retrieval from the training set
- Multi-layer Grad-CAM comparison for deeper interpretability

---

## 📄 License

This project is open source. Add a license of your choice (e.g., MIT) if you
plan to share or accept contributions.
