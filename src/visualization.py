"""
Visualisasi EDA dan augmentasi:
  - Sample gambar per kelas
  - Distribusi kelas (bar chart)
  - Contoh augmentasi
  - Pixel intensity distribution
"""
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from src.config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    CLASSES, RESULTS_DIR, IMAGE_SIZE, BATCH_SIZE
)
from src.utils import ensure_dirs, class_distribution

EDA_DIR = os.path.join(RESULTS_DIR, "eda")


# ------------------------------------------------------------------ #
#  SAMPLE IMAGES PER KELAS                                            #
# ------------------------------------------------------------------ #

def plot_sample_images(
    directory:   str,
    n_per_class: int  = 4,
    save_path:   str  = None,
):
    """Tampilkan n sampel gambar untuk tiap kelas."""
    classes = sorted([
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ])
    n_cls = len(classes)
    fig, axes = plt.subplots(n_cls, n_per_class,
                              figsize=(n_per_class * 3, n_cls * 3))

    for r, cls in enumerate(classes):
        cls_dir = os.path.join(directory, cls)
        imgs    = [
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        samples = random.sample(imgs, min(n_per_class, len(imgs)))

        for c, fname in enumerate(samples):
            img = Image.open(os.path.join(cls_dir, fname)).convert("RGB")
            img = img.resize(IMAGE_SIZE)
            axes[r, c].imshow(img)
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_ylabel(cls, fontsize=11,
                                      fontweight="bold", rotation=0,
                                      labelpad=70, va="center")

    plt.suptitle("Sample Gambar per Kelas", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[saved] {save_path}")
    plt.show()


# ------------------------------------------------------------------ #
#  CLASS DISTRIBUTION                                                  #
# ------------------------------------------------------------------ #

def plot_class_distribution(
    directories: dict,          # {"Train": path, "Val": path, "Test": path}
    save_path:   str = None,
):
    """Bar chart distribusi kelas untuk setiap split."""
    fig, axes = plt.subplots(1, len(directories),
                              figsize=(6 * len(directories), 5),
                              sharey=False)

    colors = sns.color_palette("Set2", len(CLASSES))

    for ax, (split_name, directory) in zip(axes, directories.items()):
        if not os.path.isdir(directory):
            ax.set_title(f"{split_name}\n(tidak ada)")
            continue

        dist   = class_distribution(directory)
        labels = list(dist.keys())
        counts = list(dist.values())

        bars = ax.bar(labels, counts, color=colors[:len(labels)],
                      edgecolor="white", linewidth=0.8)
        ax.set_title(f"{split_name}\n(n={sum(counts):,})",
                     fontweight="bold")
        ax.set_xlabel("Kelas")
        ax.set_ylabel("Jumlah Gambar")
        ax.tick_params(axis="x", rotation=35)

        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(counts)*0.01,
                    str(cnt), ha="center", fontsize=9)

    plt.suptitle("Distribusi Kelas per Split", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[saved] {save_path}")
    plt.show()


# ------------------------------------------------------------------ #
#  CONTOH AUGMENTASI                                                   #
# ------------------------------------------------------------------ #

def plot_augmentation_examples(
    image_path: str,
    save_path:  str = None,
    n_aug:      int = 8,
):
    """Tampilkan gambar asli vs hasil augmentasi."""
    from tensorflow.keras.preprocessing.image import (
        ImageDataGenerator, img_to_array, array_to_img
    )
    from src.config import (
        ROTATION_RANGE, WIDTH_SHIFT_RANGE, HEIGHT_SHIFT_RANGE,
        SHEAR_RANGE, ZOOM_RANGE, HORIZONTAL_FLIP, BRIGHTNESS_RANGE
    )

    datagen = ImageDataGenerator(
        rotation_range    = ROTATION_RANGE,
        width_shift_range = WIDTH_SHIFT_RANGE,
        height_shift_range = HEIGHT_SHIFT_RANGE,
        shear_range       = SHEAR_RANGE,
        zoom_range        = ZOOM_RANGE,
        horizontal_flip   = HORIZONTAL_FLIP,
        brightness_range  = BRIGHTNESS_RANGE,
        fill_mode         = "nearest",
    )

    img  = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    arr  = img_to_array(img)[np.newaxis, ...]

    fig, axes = plt.subplots(2, (n_aug + 2) // 2, figsize=(16, 6))
    axes      = axes.flatten()

    axes[0].imshow(img)
    axes[0].set_title("Original", fontweight="bold")
    axes[0].axis("off")

    gen = datagen.flow(arr, batch_size=1, seed=42)
    for i in range(1, n_aug + 1):
        aug = array_to_img(next(gen)[0])
        axes[i].imshow(aug)
        axes[i].set_title(f"Aug {i}")
        axes[i].axis("off")

    for ax in axes[n_aug + 1:]:
        ax.axis("off")

    plt.suptitle("Contoh Augmentasi Data", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[saved] {save_path}")
    plt.show()


# ------------------------------------------------------------------ #
#  RUN EDA LENGKAP                                                    #
# ------------------------------------------------------------------ #

def run_eda(train_dir=TRAIN_DIR, val_dir=VAL_DIR, test_dir=TEST_DIR):
    ensure_dirs(EDA_DIR)

    # 1. Sample gambar
    if os.path.isdir(train_dir) and any(os.scandir(train_dir)):
        plot_sample_images(
            train_dir,
            save_path=os.path.join(EDA_DIR, "sample_images.png")
        )

    # 2. Distribusi kelas
    plot_class_distribution(
        {"Train": train_dir, "Validation": val_dir, "Test": test_dir},
        save_path=os.path.join(EDA_DIR, "class_distribution.png"),
    )

    print("[DONE] EDA selesai. Hasil tersimpan di:", EDA_DIR)
