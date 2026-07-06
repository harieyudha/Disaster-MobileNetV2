"""
Preprocessing:
- Normalisasi (MobileNetV2 preprocess_input)
- Resize ke 224×224
- Validasi integritas gambar
"""
import os
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.config import IMAGE_SIZE


# ------------------------------------------------------------------ #
#  VALIDASI GAMBAR                                                     #
# ------------------------------------------------------------------ #

def validate_images(directory: str) -> list[str]:
    """
    Cek seluruh gambar di directory (rekursif).
    Return list path gambar rusak/tidak bisa dibaca.
    """
    broken = []
    exts   = {".png", ".jpg", ".jpeg"}
    for root, _, files in os.walk(directory):
        for fname in files:
            if Path(fname).suffix.lower() in exts:
                fpath = os.path.join(root, fname)
                try:
                    with Image.open(fpath) as img:
                        img.verify()
                except (UnidentifiedImageError, Exception):
                    broken.append(fpath)
    return broken


def remove_broken_images(directory: str):
    """Hapus gambar rusak dari directory."""
    broken = validate_images(directory)
    if not broken:
        print(f"[OK] Tidak ada gambar rusak di {directory}")
        return
    for p in broken:
        os.remove(p)
        print(f"[REMOVED] {p}")
    print(f"[INFO] Total dihapus: {len(broken)} gambar")


# ------------------------------------------------------------------ #
#  PREPROCESS SINGLE IMAGE                                             #
# ------------------------------------------------------------------ #

def load_and_preprocess(
    image_path: str,
    target_size: tuple = IMAGE_SIZE,
) -> np.ndarray:
    """
    Load 1 gambar → resize → MobileNetV2 preprocess_input.
    Output: array float32 shape (224, 224, 3), nilai [-1, 1].
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)          # skala ke [-1, 1]
    return arr


def preprocess_batch(image_paths: list[str]) -> np.ndarray:
    """Preprocess sekumpulan gambar → array (N, H, W, C)."""
    return np.stack([load_and_preprocess(p) for p in image_paths])


# ------------------------------------------------------------------ #
#  TENSORFLOW DATA PIPELINE                                            #
# ------------------------------------------------------------------ #

def _parse_image(path: str, label: int, target_size: tuple = IMAGE_SIZE):
    """Parse satu file untuk tf.data pipeline."""
    raw   = tf.io.read_file(path)
    img   = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img   = tf.image.resize(img, target_size)
    img   = tf.cast(img, tf.float32)
    img   = preprocess_input(img)
    return img, label


def build_tf_dataset(
    image_paths: list[str],
    labels:      list[int],
    batch_size:  int,
    shuffle:     bool = False,
    cache:       bool = True,
) -> tf.data.Dataset:
    """
    Buat tf.data.Dataset dari list path & label.
    Lebih efisien dari ImageDataGenerator untuk dataset besar.
    """
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths), seed=42)
    ds = ds.map(
        lambda p, l: _parse_image(p, l),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
