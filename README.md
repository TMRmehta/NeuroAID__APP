# NeuroAID — Explainable Brain Tumor Detection (MRI)

NeuroAID is a lightweight, end-to-end app for MRI-based brain tumor detection with
built-in explainability. It lets you upload MR images, runs a trained CNN/ViT model,
and overlays visual attributions (Grad-CAM/SHAP) to make predictions interpretable.

> ✨ Why this repo?
> - One-command local app (Streamlit UI)
> - Reproducible training & evaluation scripts
> - Explainability: class activation maps and feature attributions
> - Clear, clinic-friendly outputs (prediction + confidence + heatmap)

---

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Installing PyTorch](#installing-pytorch)
- [Run the App](#run-the-app)
- [Training & Evaluation](#training--evaluation)
- [Explainability](#explainability)
- [Data Notes](#data-notes)
- [Configuration](#configuration)
- [Testing](#testing)
- [Docker (optional)](#docker-optional)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features
- **Inference UI**: Drag-and-drop DICOM/NIfTI/PNG, instant prediction + heatmaps
- **Models**: Plug-in architecture (e.g., ResNet50 / ViT / custom)
- **Explainability**: Grad-CAM for CNNs; SHAP for tabular/combined inputs
- **Metrics**: AUROC, F1, PR-AUC, confusion matrix; saves per-case CSV
- **Reproducible**: Seeded runs, config-driven training, checkpoints

---

## Project Structure

```
NeuroAID__APP/
├─ app/
│  ├─ ui.py                 # Streamlit app entrypoint
│  ├─ inference.py          # Loads model, pre/post-processing
│  ├─ explainability.py     # Grad-CAM / SHAP helpers
│  ├─ preprocess.py         # DICOM/NIfTI reading, normalization, resizing
│  └─ assets/               # Icons, sample images
├─ models/
│  ├─ build_model.py        # Model factory (ResNet/ViT/etc.)
│  └─ weights/              # Put .pt/.pth here (gitignored)
├─ train/
│  ├─ train.py              # Training loop
│  ├─ eval.py               # Evaluation & metrics
│  └─ datamodule.py         # Dataset/Dataloader definitions
├─ configs/
│  ├─ default.yaml          # Global config (paths, hyperparams)
│  └─ app.yaml              # UI-specific settings
├─ tests/
│  └─ test_inference.py
├─ requirements.txt
├─ README.md
└─ LICENSE
```

> If your repo layout differs, keep this README and adjust the paths/commands.

---

## Quickstart

```bash
# 1) Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2) Install dependencies (PyTorch installed separately below)
pip install -r requirements.txt

# 3) Add a trained model
# Place your checkpoint at: models/weights/model.pt

# 4) Launch the UI
streamlit run app/ui.py
```

---

## Installing PyTorch

Install the Torch build that matches your CUDA/OS. Examples:

```bash
# CPU-only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.1 (example)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> See: https://pytorch.org/get-started/locally/  
> (Torch is intentionally **not** pinned in `requirements.txt` since wheels differ by GPU/OS.)

---

## Run the App

```bash
# Default run (uses configs/app.yaml)
streamlit run app/ui.py

# With explicit model path and config
streamlit run app/ui.py -- --config configs/app.yaml --weights models/weights/model.pt
```

**UI flow**
1. Upload an MRI file (`.dcm`, `.nii`, `.nii.gz`) or a pre-extracted slice (`.png/.jpg`).
2. Choose model + preprocessing (e.g., modality, slice selection).
3. Click **Predict** → get **class**, **probability**, and **Grad-CAM** heatmap.
4. Export results as image or CSV.

---

## Training & Evaluation

```bash
# Training
python -m train.train   --config configs/default.yaml   --data_root /path/to/dataset   --out_dir runs/experiment_01

# Evaluation
python -m train.eval   --weights models/weights/model.pt   --data_root /path/to/valset   --out_dir runs/eval_01
```

Key config knobs in `configs/default.yaml`:
```yaml
seed: 42
img_size: 224
batch_size: 16
max_epochs: 30
optimizer: adamw
lr: 3e-4
weight_decay: 0.01
model:
  name: resnet50
  pretrained: true
data:
  modalities: ["T1", "T2", "FLAIR"]   # adjust to your dataset
  slice_mode: "center"                # center | index | montage
```

---

## Explainability

- **Grad-CAM** (default): saliency heatmaps overlaid on the MRI slice  
  ```bash
  python -m app.explainability --weights models/weights/model.pt --image path/to/img.png
  ```
- **SHAP** (optional): if you use clinical + imaging features, SHAP summary plots and per-case force plots are supported.

---

## Data Notes

- Supported inputs: **DICOM**, **NIfTI**, **PNG/JPG** (single-slice pipelines).
- Preprocessing: z-score or min-max normalization, optional skull-strip (if using pre-processed data).
- **De-identification**: ensure all DICOMs are PHI-free before use.

---

## Configuration

- App defaults: `configs/app.yaml`
- Training defaults: `configs/default.yaml`
- Environment variables (optional):
  - `NEUROAID_DEVICE` = `cpu` / `cuda`
  - `NEUROAID_MODEL_PATH` = path to weights

---

## Testing

```bash
pytest -q
```

Unit tests cover:
- Model load & inference shape
- Preprocessing for DICOM/NIfTI
- Grad-CAM generation

---

## Docker (optional)

```bash
docker build -t neuroaid:latest .
docker run --rm -p 8501:8501 -v $PWD:/app neuroaid:latest
```

Add a minimal `Dockerfile` if desired; I can provide one.

---

## Troubleshooting

- **Torch import fails** → reinstall Torch for your exact CUDA/OS (see section above).
- **OpenCV headless errors** → ensure `opencv-python-headless` is installed on servers.
- **NIfTI reading issues** → verify file paths and install `nibabel`.

---

## License
Specify your license (e.g., MIT). Add a `LICENSE` file at repo root.
