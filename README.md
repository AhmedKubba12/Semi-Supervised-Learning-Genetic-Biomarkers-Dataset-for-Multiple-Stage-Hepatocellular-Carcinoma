# Semi-Supervised Learning-Based Genetic Biomarkers Dataset for Multiple-Stage Hepatocellular Carcinoma Prediction

> Published at the **2025 18th International Conference on Development in eSystem Engineering (DeSE)**  
> DOI: [10.1109/DESE68208.2025.11368219](https://doi.org/10.1109/DESE68208.2025.11368219)

---

## Authors

| Name | Affiliation |
|------|-------------|
| Ahmed Ammar Kubba | Department of Computer Science, University of Sharjah, UAE |
| Manar Abu Talib | Department of Computer Science, University of Sharjah, UAE |
| Jibran Sualeh Muhammad | Department of Biomedical Sciences, University of Birmingham, UK |
| Ali Bou Nassif | Department of Computer Engineering, University of Sharjah, UAE |
| Abdalla Sayed Mohamed | Department of Computer Science, University of Sharjah, UAE |
| Darko Castven | Department of Medicine I, University Medical Center Schleswig-Holstein, Germany |
| Jens U. Marquardt | Department of Medicine I, University Medical Center Schleswig-Holstein, Germany |

---

## Overview

Hepatocellular Carcinoma (HCC) accounts for over 90% of liver cancer cases — a disease responsible for 830,180 deaths globally in 2023. A critical bottleneck in automated HCC diagnosis is the scarcity of large, well-labeled genomic datasets.

This work addresses that gap by constructing a **multi-stage HCC genomic dataset** using a semi-supervised learning pipeline built on top of three source datasets. The final dataset enables AI models to distinguish between five clinically meaningful stages of HCC development.

---

## Dataset

The constructed HCC dataset has the following properties:

- **770 patient samples** total
- **11,150 gene expression features** per sample
- **5 class labels** representing distinct HCC stages

| Class | Label | Description |
|-------|-------|-------------|
| Surrounding Liver | `sl` | Normal liver tissue; no HCC gene expressions |
| Low-Grade Dysplastic Nodules | `lgdn` | Early precancerous lesions; low risk of HCC progression |
| High-Grade Dysplastic Nodules | `hgdn` | Advanced precancerous lesions; high risk of HCC progression |
| Early HCC | `ehcc` | Early-stage hepatocellular carcinoma |
| Progressed HCC | `phcc` | Advanced-stage hepatocellular carcinoma |

> **Class Imbalance Note:** The `phcc` and `ehcc` classes are the most represented, while `sl`, `lgdn`, and `hgdn` are underrepresented. Techniques such as class weighting or SMOTE are recommended when training downstream models.

---

## Methodology

The dataset was constructed through a three-phase pipeline:

### 1. Data Collection

Three source datasets were used:

| Dataset | Type | Samples | Gene Features | Labels |
|---------|------|---------|---------------|--------|
| **Lübeck Dataset** | Private | 29 | 16,381 | 5 (sl, ehcc, phcc, lgdn, hgdn) |
| **TCGA** | Public | 782 | 16,382 | 3 (Normal, HCC, Transition) |
| **GSE89377** | Public | 107 | ~48,000 | 9 (Normal, Chronic Hepatitis, Cirrhosis, Dysplastic Nodules, Early HCC, TG1/TG2/TG3) |

### 2. Pre-Processing

- Gene features were aligned across all three datasets using the **Illumina HumanHT-12 V3.0 Expression BeadChip (GPL6947)** as a reference platform.
- Only gene features common to all three datasets were retained.
- GSE89377 class labels were mapped to the five target classes with guidance from medical domain experts:
  - Cirrhosis → `Normal` (added to TCGA)
  - Chronic Hepatitis → discarded (unrelated to HCC)
  - Normal → `sl`; Dysplastic Nodules → `lgdn`/`hgdn`; Early HCC → `ehcc`
  - TG1 → HCC (added to TCGA); TG2/TG3 → `phcc`

### 3. Semi-Supervised Learning

```
Train XGBoost on Lübeck dataset
        ↓
Predict labels on TCGA dataset
        ↓
Filter high-confidence predictions (≥ 40% confidence)
that align with TCGA ground-truth labels
        ↓
Add accepted predictions to Lübeck dataset
        ↓
Retrain XGBoost on expanded dataset
        ↓
Repeat until convergence
```

Label matching rules during filtering:
- TCGA `HCC` → must predict `ehcc` or `phcc`
- TCGA `Transition` → must predict `lgdn` or `hgdn`
- TCGA `Normal` → must predict `sl`

---

## Results

### XGBoost Model Performance (on final HCC dataset)

| Metric | Score |
|--------|-------|
| Accuracy | **96.5%** |
| Macro Precision | 90.6% |
| Macro Recall | 88.9% |
| Macro F1 Score | 89.3% |
| Cross-Entropy Loss | 0.1015 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| `ehcc` | 98.9% | 96.7% | 97.8% | 92 |
| `hgdn` | 66.7% | 72.7% | 69.6% | 11 |
| `lgdn` | 87.5% | 100.0% | 93.3% | 21 |
| `phcc` | 100.0% | 100.0% | 100.0% | 99 |
| `sl` | 100.0% | 75.0% | 85.7% | 8 |

> The model was evaluated on a stratified 70/30 train-test split. The XGBoost model was primarily used for dataset construction, not as the final clinical classifier.

---

## Repository Structure

```
├── data/
│   └── hcc_dataset.csv          # Final constructed HCC dataset (770 × 11,150)
├── preprocessing/
│   ├── align_features.py        # Gene feature alignment across datasets
│   └── label_mapping.py         # GSE89377 label mapping to target classes
├── semi_supervised/
│   ├── train_xgboost.py         # XGBoost training on Lübeck dataset
│   ├── predict_tcga.py          # Confidence-filtered label transfer
│   └── iterative_expansion.py   # Full semi-supervised loop
├── evaluation/
│   ├── metrics.py               # Accuracy, precision, recall, F1
│   └── confusion_matrix.py      # Confusion matrix visualization
├── figures/
│   ├── pca_projection.png       # PCA of dataset by class
│   ├── class_distribution.png   # Class frequency bar chart
│   └── confusion_matrix.png     # XGBoost confusion matrix
└── README.md
```

---

## Requirements

Key dependencies:

- `xgboost`
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`

---

## Limitations

- **Class imbalance:** `sl`, `lgdn`, and `hgdn` are underrepresented. Future work should apply SMOTE or class weighting.
- **Single model:** XGBoost was used solely for dataset construction. Downstream deep learning models require separate evaluation.
- **External validation:** Further validation on independent cohorts is needed to assess generalizability.

---

## Citation

If you use this dataset or methodology in your research, please cite:

```bibtex
@inproceedings{kubba2025hcc,
  title     = {Semi-Supervised Learning-Based Genetic Biomarkers Dataset for Multiple-Stage Hepatocellular Carcinoma Prediction},
  author    = {Kubba, Ahmed Ammar and Abu Talib, Manar and Muhammad, Jibran Sualeh and Bou Nassif, Ali and Sayed Mohamed, Abdalla and Castven, Darko and Marquardt, Jens U.},
  booktitle = {2025 18th International Conference on Development in eSystem Engineering (DeSE)},
  pages     = {555--560},
  year      = {2025},
  doi       = {10.1109/DESE68208.2025.11368219},
  publisher = {IEEE}
}
```

---

## License

This repository is intended for academic and research use. The Lübeck dataset is private and not redistributed here. The TCGA and GSE89377 datasets are publicly available through their respective repositories:

- [TCGA via GDC Portal](https://portal.gdc.cancer.gov/)
- [GSE89377 via NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE89377)

---

## Acknowledgements

This work was supported by the University of Sharjah and the University Medical Center Schleswig-Holstein, Campus Lübeck. The authors declare no competing financial interests.
