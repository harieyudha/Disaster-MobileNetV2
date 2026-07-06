"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
Visualisasi area penting yang dilihat model saat prediksi.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from PIL import Image

from src.config import (
    IMAGE_SIZE, CLASSES, RESULTS_DIR, MODEL_DIR
)
from src.preprocessing import load_and_preprocess
from src.utils import ensure_dirs

GRADCAM_DIR = os.path.join(RESULTS_DIR, "gradcam")

# Layer terakhir MobileNetV2 sebelum GAP
LAST_CONV_LAYER = "Conv_1_bn"       # atau "out_relu"


# ------------------------------------------------------------------ #
#  COMPUTE GRAD-CAM HEATMAP                                           #
# ------------------------------------------------------------------ #

def make_gradcam_heatmap(
    img_array:       np.ndarray,
    model:           tf.keras.Model,
    last_conv_layer: str   = LAST_CONV_LAYER,
    pred_index:      int   = None,
) -> np.ndarray:
    """
    Hitung heatmap Grad-CAM.

    Parameters
    ----------
    img_array       : preprocessed image (1, H, W, C)
    model           : model Keras
    last_conv_layer : nama layer konvolusi terakhir
    pred_index      : indeks kelas target (None = kelas prediksi)

    Returns
    -------
    heatmap: np.ndarray shape (h, w), nilai [0, 1]
    """
    # Buat model yang output feature map + prediksi
    grad_model = tf.keras.Model(
        inputs  = model.inputs,
        outputs = [
            model.get_layer(last_conv_layer).output,
            model.output,
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling pada gradien
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


# ------------------------------------------------------------------ #
#  OVERLAY HEATMAP KE GAMBAR ASLI                                     #
# ------------------------------------------------------------------ #

def overlay_gradcam(
    original_img:  np.ndarray,      # uint8 (H, W, 3) unnormalized
    heatmap:       np.ndarray,
    alpha:         float = 0.4,
    colormap:      str   = "jet",
) -> np.ndarray:
    """
    Overlay heatmap ke gambar asli.
    Returns uint8 array (H, W, 3).
    """
    h, w = original_img.shape[:2]
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(255 * heatmap)).resize((w, h))
    )

    cmap       = cm.get_cmap(colormap)
    heatmap_c  = cmap(heatmap_resized / 255.0)[:, :, :3]   # (H, W, 3) float
    heatmap_c  = np.uint8(255 * heatmap_c)

    superimposed = np.uint8(
        (1 - alpha) * original_img + alpha * heatmap_c
    )
    return superimposed


# ------------------------------------------------------------------ #
#  VISUALISASI SATU GAMBAR                                            #
# ------------------------------------------------------------------ #

def visualize_gradcam(
    image_path:  str,
    model:       tf.keras.Model,
    class_names: list  = CLASSES,
    save_path:   str   = None,
    alpha:       float = 0.4,
):
    """
    Tampilkan: gambar asli | heatmap | overlay.
    """
    # Load & preprocess
    img_array = load_and_preprocess(image_path)[np.newaxis, ...]
    orig_img  = np.array(Image.open(image_path).convert("RGB").resize(IMAGE_SIZE))

    # Prediksi
    preds     = model.predict(img_array, verbose=0)
    pred_idx  = np.argmax(preds[0])
    pred_cls  = class_names[pred_idx]
    confidence = preds[0][pred_idx] * 100

    # Heatmap
    heatmap   = make_gradcam_heatmap(img_array, model, pred_index=pred_idx)
    overlay   = overlay_gradcam(orig_img, heatmap, alpha=alpha)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    titles    = ["Original", "Grad-CAM Heatmap", "Overlay"]
    imgs      = [
        orig_img,
        plt.cm.jet(heatmap)[:, :, :3],
        overlay,
    ]

    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.set_title(title, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
        f"Pred: {pred_cls}  ({confidence:.1f}%)\n{os.path.basename(image_path)}",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[saved] {save_path}")
    plt.show()
    return pred_cls, confidence, heatmap


# ------------------------------------------------------------------ #
#  BATCH GRAD-CAM (satu contoh per kelas)                             #
# ------------------------------------------------------------------ #

def batch_gradcam_by_class(
    test_dir:    str,
    model:       tf.keras.Model,
    class_names: list = CLASSES,
    n_per_class: int  = 3,
    save_dir:    str  = GRADCAM_DIR,
):
    """
    Untuk setiap kelas, ambil n_per_class gambar dan simpan Grad-CAM.
    """
    ensure_dirs(save_dir)

    for cls in class_names:
        cls_dir = os.path.join(test_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"[SKIP] {cls_dir} tidak ditemukan")
            continue

        images = [
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ][:n_per_class]

        for i, img_path in enumerate(images):
            save_path = os.path.join(save_dir, f"{cls}_{i+1:02d}_gradcam.png")
            visualize_gradcam(img_path, model, class_names,
                              save_path=save_path)


if __name__ == "__main__":
    import sys
    from src.evaluate import load_model
    from src.config import TEST_DIR

    model = load_model()
    batch_gradcam_by_class(TEST_DIR, model)
