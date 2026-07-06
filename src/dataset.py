"""
Modul dataset:
- Scan & mapping CDD → kelas (sesuai CLASS_MAPPING di config)
- Split train/val/test dengan stratifikasi
- Copy file ke folder split
"""
import os
import shutil
import random
from collections import defaultdict

from src.config import (
    RAW_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    CLASS_MAPPING, CLASSES,
    TRAIN_SPLIT, VAL_SPLIT, RANDOM_SEED
)
from src.utils import ensure_dirs, set_seed, print_class_distribution


# ------------------------------------------------------------------ #
#  1. SCAN RAW DATASET                                                 #
# ------------------------------------------------------------------ #

def scan_raw_dataset(raw_dir: str = RAW_DIR) -> dict:
    """
    Scan semua gambar di CDD secara rekursif dan petakan ke label kelas.
    Mendukung subfolder berlapis (misal: Damaged_Infrastructure/Earthquake).

    Returns
    -------
    dict {label: [list_of_full_paths]}
    """
    label_to_paths = defaultdict(list)

    for folder_name, label in CLASS_MAPPING.items():
        folder = os.path.join(raw_dir, folder_name)
        if not os.path.isdir(folder):
            print(f"[WARNING] Folder tidak ditemukan: {folder}")
            continue

        # os.walk → rekursif masuk ke subfolder (Earthquake, Infrastructure, dll.)
        found = 0
        for root, _, files in os.walk(folder):
            for fname in files:
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    label_to_paths[label].append(os.path.join(root, fname))
                    found += 1

        print(f"[SCAN] {folder_name:<35} → {label:<25} ({found} gambar)")

    return dict(label_to_paths)


# ------------------------------------------------------------------ #
#  2. SPLIT DATASET                                                    #
# ------------------------------------------------------------------ #

def split_dataset(
    label_to_paths: dict,
    train_ratio: float = TRAIN_SPLIT,
    val_ratio:   float = VAL_SPLIT,
    seed:        int   = RANDOM_SEED,
) -> tuple:
    """
    Bagi setiap kelas secara stratifikasi (per-kelas acak).

    Returns
    -------
    train_dict, val_dict, test_dict  → {label: [paths]}
    """
    set_seed(seed)
    train_d, val_d, test_d = {}, {}, {}

    for label, paths in label_to_paths.items():
        paths = paths.copy()
        random.shuffle(paths)

        n     = len(paths)
        n_tr  = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_d[label] = paths[:n_tr]
        val_d[label]   = paths[n_tr : n_tr + n_val]
        test_d[label]  = paths[n_tr + n_val:]

    return train_d, val_d, test_d


# ------------------------------------------------------------------ #
#  3. COPY FILE KE FOLDER SPLIT                                        #
# ------------------------------------------------------------------ #

def _copy_split(split_dict: dict, dest_dir: str):
    """Copy gambar ke dest_dir/label/ dengan nama file unik."""
    total = 0
    for label, paths in split_dict.items():
        cls_dir = os.path.join(dest_dir, label)
        ensure_dirs(cls_dir)

        seen = {}
        for src in paths:
            stem, ext = os.path.splitext(os.path.basename(src))
            key = stem + ext

            # Beri suffix counter jika nama sudah ada (konflik antar subfolder)
            if key in seen:
                seen[key] += 1
                fname = f"{stem}_{seen[key]:04d}{ext}"
            else:
                seen[key] = 0
                fname = key

            shutil.copy2(src, os.path.join(cls_dir, fname))
            total += 1

    print(f"  [copied] {total} file → {dest_dir}")


# ------------------------------------------------------------------ #
#  4. PIPELINE LENGKAP                                                 #
# ------------------------------------------------------------------ #

def prepare_splits(
    raw_dir:   str  = RAW_DIR,
    train_dir: str  = TRAIN_DIR,
    val_dir:   str  = VAL_DIR,
    test_dir:  str  = TEST_DIR,
    overwrite: bool = False,
):
    """
    Pipeline: scan → split → copy.
    Skip jika folder sudah ada (kecuali overwrite=True).
    """
    # Cek apakah sudah pernah diproses
    if (not overwrite
            and os.path.isdir(train_dir)
            and any(os.scandir(train_dir))):
        print("[INFO] Folder split sudah ada. Gunakan overwrite=True untuk reset.")
        return

    print("\n[1/3] Scanning raw dataset ...")
    label_paths = scan_raw_dataset(raw_dir)

    dist_raw = {k: len(v) for k, v in label_paths.items()}
    print_class_distribution(dist_raw, "Distribusi Raw Dataset")

    print("[2/3] Splitting dataset (70/15/15) ...")
    train_d, val_d, test_d = split_dataset(label_paths)

    print("[3/3] Copying files ...")
    for name, split_d, dest in [
        ("Train",      train_d, train_dir),
        ("Validation", val_d,   val_dir),
        ("Test",       test_d,  test_dir),
    ]:
        ensure_dirs(dest)
        print(f"  → {name} ...")
        _copy_split(split_d, dest)

    # Ringkasan split
    print()
    for name, d in [("Train", train_d), ("Validation", val_d), ("Test", test_d)]:
        dist = {k: len(v) for k, v in d.items()}
        print_class_distribution(dist, f"Split: {name}")

    print("[DONE] Dataset siap digunakan!\n")


if __name__ == "__main__":
    prepare_splits()
