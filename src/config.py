"""
Konfigurasi Proyek Disaster Detection - MobileNetV2
"""

import os

# ============================================================
# BASE PATH
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# DATASET PATH
# ============================================================
RAW_DIR = r"C:\Users\ACER\Documents\ANALIS CITRA.py\dataset\raw\Comprehensive Disaster Dataset(CDD)"

# 🔥 PERBAIKAN: Gunakan folder processed
PROCESSED_DIR = os.path.join(BASE_DIR, 'dataset', 'processed')

# Folder hasil split (LANGSUNG ke processed)
TRAIN_DIR = os.path.join(PROCESSED_DIR, 'train')
VAL_DIR = os.path.join(PROCESSED_DIR, 'val')
TEST_DIR = os.path.join(PROCESSED_DIR, 'test')

# ============================================================
# CLASSES
# ============================================================
CLASSES = [
    'Damaged_Infrastructure',
    'Fire_Disaster',
    'Human_Damage',
    'Land_Disaster',
    'Non_Damage',
    'Water_Disaster'
]

CLASS_MAPPING = {
    'Damaged_Infrastructure': 'Damaged_Infrastructure',
    'Fire_Disaster': 'Fire_Disaster',
    'Human_Damage': 'Human_Damage',
    'Land_Disaster': 'Land_Disaster',
    'Non_Damage': 'Non_Damage',
    'Water_Disaster': 'Water_Disaster'
}

NUM_CLASSES = len(CLASSES)

# ============================================================
# DATA SPLIT
# ============================================================
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ============================================================
# MODEL PARAMETERS
# ============================================================
IMAGE_SIZE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
EPOCHS = 20
FREEZE_EPOCHS = 10
FINETUNE_EPOCHS = 10
LEARNING_RATE = 0.001
FINETUNE_LR = 0.0001
DROPOUT_RATE = 0.3

# ============================================================
# MOBILENETV2 PARAMETERS
# ============================================================
MOBILENET_WEIGHTS = 'imagenet'
FINE_TUNE_AT = 100

# ============================================================
# RESULTS PATH
# ============================================================
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
LOGS_DIR = os.path.join(RESULTS_DIR, 'training')

# Buat folder
for path in [RESULTS_DIR, MODEL_DIR, CHECKPOINT_DIR, LOGS_DIR]:
    os.makedirs(path, exist_ok=True)

# ============================================================
# DATA AUGMENTATION
# ============================================================
ROTATION_RANGE = 20
WIDTH_SHIFT_RANGE = 0.2
HEIGHT_SHIFT_RANGE = 0.2
SHEAR_RANGE = 0.15
ZOOM_RANGE = 0.2
HORIZONTAL_FLIP = True
VERTICAL_FLIP = False
BRIGHTNESS_RANGE = (0.8, 1.2)

# ============================================================
# CALLBACKS
# ============================================================
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-7

print("✅ Config loaded successfully!")
print(f"📁 RAW_DIR: {RAW_DIR}")
print(f"📁 TRAIN_DIR: {TRAIN_DIR}")
print(f"📁 VAL_DIR: {VAL_DIR}")
print(f"📁 TEST_DIR: {TEST_DIR}")
print(f"📊 Classes: {CLASSES}")
print(f"📐 Input Shape: {INPUT_SHAPE}")
print(f"🔄 FREEZE_EPOCHS: {FREEZE_EPOCHS}")
print(f"🔄 FINETUNE_EPOCHS: {FINETUNE_EPOCHS}")