"""
Prediksi gambar baru (single image atau folder).
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import tensorflow as tf

from src.config import (
    IMAGE_SIZE, CLASSES, MODEL_DIR, RESULTS_DIR
)
from src.preprocessing import load_and_preprocess
from src.utils import ensure_dirs, load_pickle

PRED_DIR = os.path.join(RESULTS_DIR, "prediction")


# ------------------------------------------------------------------ #
#  LOAD MODEL & CLASS INDEX                                           #
# ------------------------------------------------------------------ #

def load_assets(model_path: str = None):
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "final_model.keras")
    model = tf.keras.models.load_model(model_path)

    idx_path = os.path.join(MODEL_DIR, "class_indices.pkl")
    if os.path.exists(idx_path):
        class_indices = load_pickle(idx_path)
        idx_to_class  = {v: k for k, v in class_indices.items()}
    else:
        idx_to_class = {i: c for i, c in enumerate(CLASSES)}

    return model, idx_to_class


# ------------------------------------------------------------------ #
#  PREDIKSI SINGLE IMAGE                                              #
# ------------------------------------------------------------------ #

def predict_image(
    image_path:  str,
    model:       tf.keras.Model,
    idx_to_class: dict,
    top_k:       int  = 3,
    show_plot:   bool = True,
    save_path:   str  = None,
) -> dict:
    """
    Prediksi satu gambar.

    Returns
    -------
    dict dengan keys: predicted_class, confidence, top_k_results
    """
    img_array = load_and_preprocess(image_path)[np.newaxis, ...]
    probs     = model.predict(img_array, verbose=0)[0]

    pred_idx  = np.argmax(probs)
    pred_cls  = idx_to_class[pred_idx]
    confidence = float(probs[pred_idx]) * 100

    # Top-K
    top_k_idx    = np.argsort(probs)[::-1][:top_k]
    top_k_results = [
        {"class": idx_to_class[i], "probability": float(probs[i]) * 100}
        for i in top_k_idx
    ]

    result = {
        "image"          : os.path.basename(image_path),
        "predicted_class": pred_cls,
        "confidence"     : round(confidence, 2),
        "top_k"          : top_k_results,
    }

    if show_plot or save_path:
        _plot_prediction(image_path, probs, idx_to_class, save_path)

    return result


def _plot_prediction(
    image_path:  str,
    probs:       np.ndarray,
    idx_to_class: dict,
    save_path:   str = None,
):
    orig = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    pred_idx = np.argmax(probs)
    classes  = [idx_to_class[i] for i in range(len(probs))]

    colors = ["#4CAF50" if i == pred_idx else "#90CAF9"
              for i in range(len(probs))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.imshow(orig)
    ax1.set_title(
        f"Predicted: {idx_to_class[pred_idx]}\n"
        f"Confidence: {probs[pred_idx]*100:.1f}%",
        fontweight="bold"
    )
    ax1.axis("off")

    bars = ax2.barh(classes, probs * 100, color=colors)
    ax2.set_xlabel("Probability (%)")
    ax2.set_title("Class Probabilities", fontweight="bold")
    ax2.set_xlim(0, 105)
    for bar, p in zip(bars, probs * 100):
        ax2.text(p + 1, bar.get_y() + bar.get_height()/2,
                 f"{p:.1f}%", va="center", fontsize=9)
    ax2.grid(axis="x", alpha=0.3)

    plt.suptitle(os.path.basename(image_path), fontsize=10, color="gray")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[saved] {save_path}")
    plt.show()


# ------------------------------------------------------------------ #
#  PREDIKSI FOLDER                                                    #
# ------------------------------------------------------------------ #

def predict_folder(
    folder_path: str,
    model:       tf.keras.Model,
    idx_to_class: dict,
    save_dir:    str  = PRED_DIR,
) -> list[dict]:
    ensure_dirs(save_dir)
    results = []
    exts = {".png", ".jpg", ".jpeg"}

    images = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in exts
    ]

    print(f"[INFO] Memprediksi {len(images)} gambar …")
    for img_path in images:
        fname    = os.path.splitext(os.path.basename(img_path))[0]
        save_p   = os.path.join(save_dir, f"{fname}_pred.png")
        result   = predict_image(img_path, model, idx_to_class,
                                 show_plot=False, save_path=save_p)
        results.append(result)
        print(f"  {result['image']} → {result['predicted_class']} "
              f"({result['confidence']:.1f}%)")

    # Simpan hasil JSON
    json_path = os.path.join(save_dir, "predictions.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {json_path}")
    return results


# ------------------------------------------------------------------ #
#  CLI                                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys
    model, idx_to_class = load_assets()

    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <image_path_or_folder>")
        sys.exit(1)

    target = sys.argv[1]
    if os.path.isdir(target):
        predict_folder(target, model, idx_to_class)
    else:
        res = predict_image(target, model, idx_to_class)
        print(json.dumps(res, indent=2))
