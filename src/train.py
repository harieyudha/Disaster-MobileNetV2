"""
Training Model: Klasifikasi Citra Bencana Alam - MobileNetV2
Dua fase: Feature Extraction → Fine-Tuning
"""

import os
import sys
import shutil
import pickle
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    TensorBoard, CSVLogger
)

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from src.config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    CHECKPOINT_DIR, MODEL_DIR, RESULTS_DIR, LOGS_DIR,
    CLASSES, NUM_CLASSES, IMAGE_SIZE, BATCH_SIZE,
    FREEZE_EPOCHS, FINETUNE_EPOCHS,
    LEARNING_RATE, FINETUNE_LR, DROPOUT_RATE,
    FINE_TUNE_AT,
    ROTATION_RANGE, WIDTH_SHIFT_RANGE, HEIGHT_SHIFT_RANGE,
    SHEAR_RANGE, ZOOM_RANGE, HORIZONTAL_FLIP, BRIGHTNESS_RANGE,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR,
    RANDOM_SEED
)


# ============================================================
# UTILITAS
# ============================================================

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"[saved] {path}")


def remove_broken_images(directory: str) -> int:
    """
    Scan rekursif dan hapus gambar rusak/truncated.
    Pakai img.load() (full decode) — lebih ketat dari img.verify()
    sehingga file truncated juga terdeteksi.
    """
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = False  # strict: error jika truncated

    broken = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                fpath = os.path.join(root, fname)
                try:
                    with Image.open(fpath) as img:
                        img.load()   # force full pixel decode
                except Exception:
                    broken.append(fpath)

    for p in broken:
        os.remove(p)
        print(f"  [removed] {os.path.basename(p)}")

    if not broken:
        print(f"  ✓ Semua valid di {os.path.basename(directory)}")
    else:
        print(f"  {len(broken)} file dihapus dari {os.path.basename(directory)}")
    return len(broken)


# ============================================================
# CALLBACKS
# ============================================================

def get_callbacks(phase: str = "freeze") -> list:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    return [
        ModelCheckpoint(
            filepath       = os.path.join(CHECKPOINT_DIR, f'best_{phase}.keras'),
            monitor        = 'val_accuracy',
            save_best_only = True,
            verbose        = 1,
        ),
        EarlyStopping(
            monitor              = 'val_accuracy',
            patience             = EARLY_STOPPING_PATIENCE,
            restore_best_weights = True,
            verbose              = 1,
        ),
        ReduceLROnPlateau(
            monitor  = 'val_loss',
            factor   = REDUCE_LR_FACTOR,
            patience = REDUCE_LR_PATIENCE,
            min_lr   = 1e-7,
            verbose  = 1,
        ),
        CSVLogger(
            os.path.join(LOGS_DIR, f'log_{phase}.csv'),
            append=True
        ),
    ]


# ============================================================
# DATA GENERATORS
# ============================================================

def get_generators():
    """
    Buat generator train (dengan augmentasi) dan val/test (tanpa augmentasi).
    Pakai preprocess_input MobileNetV2 — skala ke [-1, 1], BUKAN rescale=1/255.
    """
    # Safety net: izinkan PIL load file truncated daripada crash training
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    train_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input,   # ← WAJIB untuk MobileNetV2
        rotation_range         = ROTATION_RANGE,
        width_shift_range      = WIDTH_SHIFT_RANGE,
        height_shift_range     = HEIGHT_SHIFT_RANGE,
        shear_range            = SHEAR_RANGE,
        zoom_range             = ZOOM_RANGE,
        horizontal_flip        = HORIZONTAL_FLIP,
        brightness_range       = BRIGHTNESS_RANGE,
        fill_mode              = 'nearest',
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size = IMAGE_SIZE,
        batch_size  = BATCH_SIZE,
        class_mode  = 'categorical',
        shuffle     = True,
        seed        = RANDOM_SEED,
    )
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size = IMAGE_SIZE,
        batch_size  = BATCH_SIZE,
        class_mode  = 'categorical',
        shuffle     = False,
    )
    test_gen = val_datagen.flow_from_directory(
        TEST_DIR,
        target_size = IMAGE_SIZE,
        batch_size  = BATCH_SIZE,
        class_mode  = 'categorical',
        shuffle     = False,
    )
    return train_gen, val_gen, test_gen


# ============================================================
# BUILD MODEL
# ============================================================

def build_model():
    """
    MobileNetV2 (pretrained ImageNet) + custom head.
    Base frozen untuk Phase 1.
    """
    base = MobileNetV2(
        weights     = 'imagenet',
        include_top = False,
        input_shape = IMAGE_SIZE + (3,),
    )
    base.trainable = False   # Frozen — Phase 1

    x = base.output
    x = GlobalAveragePooling2D(name='gap')(x)
    x = BatchNormalization(name='bn1')(x)
    x = Dense(256, activation='relu', name='dense_256')(x)
    x = Dropout(DROPOUT_RATE, name='dropout1')(x)
    x = Dense(128, activation='relu', name='dense_128')(x)
    x = Dropout(DROPOUT_RATE, name='dropout2')(x)
    out = Dense(NUM_CLASSES, activation='softmax', name='predictions')(x)

    model = Model(inputs=base.input, outputs=out, name='DisasterMobileNetV2')
    model.compile(
        optimizer = Adam(learning_rate=LEARNING_RATE),
        loss      = 'categorical_crossentropy',
        metrics   = ['accuracy'],
    )
    return model, base


# ============================================================
# FINE-TUNING
# ============================================================

def enable_fine_tuning(model, base_model):
    """Unfreeze layer atas base mulai dari FINE_TUNE_AT, LR kecil."""
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"[Fine-Tune] Trainable layers di base: {trainable_count} (dari layer ke-{FINE_TUNE_AT})")

    model.compile(
        optimizer = Adam(learning_rate=FINETUNE_LR),
        loss      = 'categorical_crossentropy',
        metrics   = ['accuracy'],
    )
    return model


# ============================================================
# MAIN TRAIN
# ============================================================

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # --- Validasi gambar ---
    print("\n[0/5] Validasi gambar (hapus file rusak) ...")
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        remove_broken_images(d)

    # --- Generator ---
    print("\n[1/5] Memuat data generator ...")
    train_gen, val_gen, test_gen = get_generators()
    print(f"  Train: {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}")
    print(f"  Kelas: {train_gen.class_indices}")
    save_pickle(train_gen.class_indices, os.path.join(MODEL_DIR, 'class_indices.pkl'))

    # --- Build ---
    print("\n[2/5] Membangun model ...")
    model, base_model = build_model()
    total_p     = model.count_params()
    trainable_p = sum(tf.size(w).numpy() for w in model.trainable_weights)
    print(f"  Total params    : {total_p:,}")
    print(f"  Trainable params: {trainable_p:,} (head only)")

    # ==========================================
    # PHASE 1 — Feature Extraction (base frozen)
    # ==========================================
    print(f"\n[3/5] Phase 1 — Feature Extraction ({FREEZE_EPOCHS} epochs) ...")
    hist1 = model.fit(
        train_gen,
        validation_data = val_gen,
        epochs          = FREEZE_EPOCHS,
        callbacks       = get_callbacks('freeze'),
        verbose         = 1,
    )
    save_pickle(hist1.history, os.path.join(MODEL_DIR, 'history_freeze.pkl'))
    print(f"  Best val_accuracy (Phase 1): {max(hist1.history['val_accuracy']):.4f}")

    # ==========================================
    # PHASE 2 — Fine-Tuning
    # ==========================================
    print(f"\n[4/5] Phase 2 — Fine-Tuning ({FINETUNE_EPOCHS} epochs) ...")
    model = enable_fine_tuning(model, base_model)
    hist2 = model.fit(
        train_gen,
        validation_data = val_gen,
        epochs          = FINETUNE_EPOCHS,
        callbacks       = get_callbacks('finetune'),
        verbose         = 1,
    )
    save_pickle(hist2.history, os.path.join(MODEL_DIR, 'history_finetune.pkl'))
    print(f"  Best val_accuracy (Phase 2): {max(hist2.history['val_accuracy']):.4f}")

    # Gabung history
    combined = {}
    for key in hist1.history:
        combined[key] = hist1.history[key] + hist2.history.get(key, [])
    save_pickle(combined, os.path.join(MODEL_DIR, 'history.pkl'))

    # --- Simpan model final ---
    print("\n[5/5] Menyimpan model ...")
    model.save(os.path.join(MODEL_DIR, 'final_model.keras'))
    print(f"  [saved] {os.path.join(MODEL_DIR, 'final_model.keras')}")

    print("\n✅ Training selesai!")
    return model, combined, test_gen


if __name__ == "__main__":
    train()
