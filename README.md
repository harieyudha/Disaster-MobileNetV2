# Disaster Image Classification — MobileNetV2

Klasifikasi citra bencana alam menggunakan **Transfer Learning MobileNetV2** dengan dua fase pelatihan: *Feature Extraction* → *Fine-Tuning*.

Model mencapai akurasi **94.31%** pada data uji dari dataset Comprehensive Disaster Dataset (CDD).

---

## Kelas yang Diklasifikasikan

| Kelas | Precision | Recall | F1-Score |
|---|---|---|---|
| Damaged Infrastructure | 0.7519 | 0.9269 | 0.8303 |
| Fire Disaster | 0.9769 | 0.9007 | 0.9373 |
| Human Damage | 0.9355 | 0.7838 | 0.8529 |
| Land Disaster | 0.7763 | 0.5900 | 0.6705 |
| Non Damage | 0.9935 | 0.9942 | 0.9939 |
| Water Disaster | 0.8759 | 0.8141 | 0.8439 |
| **Weighted Avg** | **0.9457** | **0.9431** | **0.9425** |

**AUC Macro:** 0.9924

---

## Struktur Proyek

```
Disaster-MobileNetV2/
├── checkpoints/            # Model checkpoint terbaik per fase training
│   ├── best_freeze.keras
│   └── best_finetune.keras
├── dataset/
│   └── processed/          # Dataset setelah di-split (tidak di-track Git)
│       ├── train/
│       ├── val/
│       └── test/
├── docs/
│   └── PRD.md              # Product Requirements Document
├── models/                 # Model final & artefak training (tidak di-track Git)
│   ├── final_model.keras
│   ├── class_indices.pkl
│   └── history*.pkl
├── notebooks/              # Jupyter notebooks alur kerja
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Model_Training.ipynb
│   ├── 04_Model_Evaluation.ipynb
│   ├── 05_GradCAM.ipynb
│   └── 06_Testing.ipynb
├── results/
│   ├── eda/                # Grafik eksplorasi data
│   ├── evaluation/         # Confusion matrix, ROC curve, classification report
│   ├── gradcam/            # Visualisasi GradCAM
│   ├── reports/            # Laporan ringkas
│   └── training/           # Log training CSV & kurva
├── src/                    # Source code utama
│   ├── config.py           # Konfigurasi global (path, hyperparameter)
│   ├── augmentation.py     # Pipeline augmentasi data
│   ├── dataset.py          # Utility loading dataset
│   ├── model.py            # Definisi arsitektur model
│   ├── train.py            # Script training (2 fase)
│   ├── evaluate.py         # Evaluasi & visualisasi metrik
│   ├── predict.py          # Inferensi single image
│   ├── gradcam.py          # Visualisasi Grad-CAM
│   ├── preprocessing.py    # Preprocessing & split dataset
│   ├── utils.py            # Fungsi utilitas
│   └── visualization.py   # Helper plot
├── generate_paper.py       # Generator laporan otomatis
├── requirements.txt
└── README.md
```

---

## Arsitektur Model

- **Base Model:** MobileNetV2 (pretrained ImageNet, `include_top=False`)
- **Custom Head:** GlobalAveragePooling2D → BatchNorm → Dense(256) → Dropout → Dense(128) → Dropout → Softmax(6)
- **Input Size:** 224 × 224 × 3
- **Fase 1 — Feature Extraction:** Base model di-freeze, hanya head yang dilatih (10 epoch, LR=0.001)
- **Fase 2 — Fine-Tuning:** Layer atas base (ab layer ke-100) di-unfreeze (10 epoch, LR=0.0001)

---

## Cara Instalasi

```bash
# Clone repo
git clone https://github.com/<username>/Disaster-MobileNetV2.git
cd Disaster-MobileNetV2

# Install dependensi
pip install -r requirements.txt
```

> Disarankan menggunakan virtual environment.

---

## Persiapan Dataset

Dataset yang digunakan adalah **Comprehensive Disaster Dataset (CDD)**. Setelah dataset disiapkan, jalankan preprocessing untuk split data:

```bash
python -c "from src.preprocessing import run; run()"
```

Dataset akan di-split dengan rasio **70% train / 15% val / 15% test** ke folder `dataset/processed/`.

---

## Cara Menjalankan

### Training
```bash
python src/train.py
```

### Evaluasi
```bash
python src/evaluate.py
```

### Prediksi Single Image
```bash
python src/predict.py --image path/to/image.jpg
```

### Notebook
Buka dan jalankan notebook secara berurutan di folder `notebooks/`:
1. `01_EDA.ipynb` — Eksplorasi data
2. `02_Preprocessing.ipynb` — Preprocessing & split
3. `03_Model_Training.ipynb` — Training model
4. `04_Model_Evaluation.ipynb` — Evaluasi model
5. `05_GradCAM.ipynb` — Visualisasi GradCAM
6. `06_Testing.ipynb` — Testing akhir

---

## Hasil

| Metrik | Nilai |
|---|---|
| Test Accuracy | **94.31%** |
| Test Loss | 0.2236 |
| Precision (weighted) | 0.9457 |
| Recall (weighted) | 0.9431 |
| F1-Score (weighted) | 0.9425 |
| AUC Macro | **0.9924** |

Hasil lengkap tersedia di folder `results/evaluation/`:
- `confusion_matrix.png` — Confusion matrix
- `confusion_matrix_norm.png` — Confusion matrix ternormalisasi
- `roc_curves.png` — ROC curve per kelas
- `classification_report.txt` — Laporan lengkap
- `training_curves.png` — Kurva training & validasi

---

## Dependensi Utama

| Package | Versi |
|---|---|
| TensorFlow | ≥ 2.15.0 |
| NumPy | ≥ 1.24.0 |
| scikit-learn | ≥ 1.3.0 |
| OpenCV | ≥ 4.8.0 |
| Matplotlib | ≥ 3.7.0 |

Lihat `requirements.txt` untuk daftar lengkap.

---

## Lisensi

MIT License — bebas digunakan untuk keperluan penelitian dan edukasi.
