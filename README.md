# Dual Degree Project: Enhancing Nucleus Segmentation and Classification in Histopathology

This repository presents the codebase for my dual degree project, which focuses on improving performance and efficiency in histopathological nucleus segmentation and classification. The project is structured into two main components:

---

## 📁 `CellViT_Enhancements/`  
### 🧪 Enhancing CellViT with Specialized Loss Functions

In this part, we improve the performance of the [CellViT](https://github.com/MASILab/CellViT) model by integrating domain-informed and geometry-aware loss functions:

- **SAMS Loss**: A stain-aware loss function that encourages consistency under stain variations.
- **Bending Energy Loss**: Targets accurate contour and shape prediction for nuclei.
- **Huber Loss**: Replaces MSE loss in regression branches for robustness to outliers.
- **Augmentation Enhancements**: Introduced additional augmentations to improve generalizability.

These modifications result in improved segmentation performance across datasets such as PanNuke.

➡️ See `CellViT_Enhancements/README.md` for training instructions, results, and baseline comparisons.

---

## 📁 `Cascaded_Pipeline/`  
### 🧠 A Novel Efficient Two-Stage Pipeline with Context-Aware Classification

This section proposes a new lightweight framework designed for real-time, scalable analysis of histology images. It includes:

- **Stage 1**: Nuclei detection using a fast object detector (e.g., YOLO).
- **Stage 2**: Nucleus segmentation using a compact UNet++ variant.
- **Stage 3 (New)**: **Classification of nuclei using context and tight patch features**:
  - **Context-Aware Spatial Residual Attention**: Learns to focus on informative regions in the surrounding tissue context.
  - **Feature Fusion**: Attention-masked context features are concatenated with tight (nucleus) features.
  - **Generalizable Hierarchical Loss Function**: Allows training on datasets with different but related label sets.

This modular pipeline achieves state-of-the-art performance with a fraction of the compute cost.

➡️ See `Cascaded_Pipeline/README.md` for model architecture details, training scripts, and evaluation.

---

## 📦 Datasets

We evaluate our models on the following datasets:

- [PanNuke](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke)
- [MoNuSeg](https://monuseg.grand-challenge.org/)
- [PUMA](https://puma.grand-challenge.org/) — for real-world validation

---

## 🔧 Setup Instructions

```bash
# Create environment
conda create -n nucleus_analysis python=3.9
conda activate nucleus_analysis

# Install dependencies
pip install -r requirements.txt
