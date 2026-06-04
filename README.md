# Viskores-Guided CNN Brain Tumor Classification

This project investigates whether Viskores-guided geometric and intensity feature discovery can improve CNN-based brain tumor classification.

## Pipeline 1: Baseline CNN

2D MRI Images
→ CNN
→ Classification

Accuracy: 88%

---

## Pipeline 2: Viskores-Guided CNN

2D MRI Images
→ Simulated 3D Volume
→ VTK Conversion
→ Viskores Isosurface Analysis
→ Geometric + Intensity Feature Discovery
→ Feature Enhancement
→ CNN
→ Classification

Accuracy: 92%

---

## Workflow

![Workflow](docs/workflow_overview.png)

---

## Dataset

Brain Tumor MRI Dataset

Classes:

- Glioma
- Meningioma
- Pituitary
- No Tumor

---

## Technologies

- Python
- TensorFlow / Keras
- VTK
- Viskores
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib

---

## Author

Dr. Suzan Anwar
Philander Smith University
