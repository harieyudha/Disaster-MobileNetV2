"""
Utility functions: path, seed, logging, folder creation.
"""
import os
import random
import pickle
import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42):
    """Pastikan hasil reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dirs(*dirs):
    """Buat direktori jika belum ada."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[saved] {path}")


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def count_files(directory: str, ext: str = ".png") -> int:
    """Hitung jumlah file dengan ekstensi tertentu secara rekursif."""
    count = 0
    for _, _, files in os.walk(directory):
        count += sum(1 for f in files if f.lower().endswith(ext))
    return count


def class_distribution(directory: str) -> dict:
    """Hitung distribusi kelas dari folder yang berisi subfolder kelas."""
    dist = {}
    for cls in sorted(os.listdir(directory)):
        cls_path = os.path.join(directory, cls)
        if os.path.isdir(cls_path):
            dist[cls] = len([
                f for f in os.listdir(cls_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ])
    return dist


def print_class_distribution(dist: dict, title: str = "Class Distribution"):
    print(f"\n{'='*40}")
    print(f"  {title}")
    print(f"{'='*40}")
    total = sum(dist.values())
    for cls, n in dist.items():
        bar = "█" * int(30 * n / total)
        print(f"  {cls:<20} {n:>5}  {bar}")
    print(f"  {'TOTAL':<20} {total:>5}")
    print(f"{'='*40}\n")
