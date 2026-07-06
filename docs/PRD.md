# PRD — Klasifikasi Citra Bencana Alam Menggunakan MobileNetV2

## Latar Belakang
Bencana alam menyebabkan kerugian besar. Deteksi otomatis dari citra satelit/drone
dapat mempercepat respons darurat. Deep Learning, khususnya CNN berbasis transfer
learning, terbukti efektif untuk klasifikasi citra dalam domain bencana.

## Tujuan Penelitian
1. Membangun model klasifikasi citra bencana menggunakan MobileNetV2
2. Menerapkan transfer learning + fine-tuning untuk meningkatkan akurasi
3. Mengevaluasi model dengan metrik Accuracy, Precision, Recall, F1-Score, AUC
4. Menginterpretasi prediksi model menggunakan Grad-CAM

## Dataset
- **Nama**: Comprehensive Disaster Dataset (CDD)
- **Sumber**: Kaggle
- **Total gambar**: ~13.561
- **Struktur original**: 6 kategori utama, 12 subfolder

## Pemetaan Kelas
| Subfolder CDD | Label Final |
|---------------|-------------|
| Water_Disaster | Flood |
| Damaged_Infrastructure/Earthquake | Earthquake |
| Damaged_Infrastructure/Infrastructure | Earthquake |
| Fire_Disaster/Wild_Fire | Wildfire |
| Fire_Disaster/Urban_Fire | Wildfire |
| Land_Disaster/Land_Slide | Landslide |
| Land_Disaster/Drought | Landslide |
| Human_Damage | Cyclone |
| Non_Damage/* | Non_Damage |

## Roadmap Penelitian
```
Dataset → EDA → Preprocessing → Augmentation → Split Dataset (70/15/15)
→ Transfer Learning MobileNetV2 → Training (Feature Extraction)
→ Fine-Tuning → Evaluation → Confusion Matrix → Classification Report
→ Grad-CAM → Prediction → Analisis → Kesimpulan
```

## Metrik Evaluasi
- **Accuracy**: (TP+TN)/(TP+TN+FP+FN)
- **Precision**: TP/(TP+FP)
- **Recall**: TP/(TP+FN)
- **F1-Score**: 2×(P×R)/(P+R)
- **AUC-ROC**: Area Under Curve (One-vs-Rest)

## Teknologi
- Python 3.10
- TensorFlow 2.13 / Keras
- MobileNetV2 (pretrained ImageNet)
- Grad-CAM untuk interpretabilitas
- Scikit-learn untuk metrik evaluasi
