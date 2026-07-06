"""
Evaluasi model:
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1, Accuracy)
  - Kurva Training (Loss & Accuracy)
  - ROC-AUC per kelas
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc
)
import tensorflow as tf

from src.config import (
    TEST_DIR, MODEL_DIR, RESULTS_DIR,
    CLASSES, NUM_CLASSES, BATCH_SIZE
)
from src.augmentation import get_test_generator
from src.utils import ensure_dirs, load_pickle


EVAL_DIR = os.path.join(RESULTS_DIR, "evaluation")


# ------------------------------------------------------------------ #
#  LOAD MODEL & PREDIKSI                                              #
# ------------------------------------------------------------------ #

def load_model(model_path: str = None) -> tf.keras.Model:
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, "final_model.keras")
    model = tf.keras.models.load_model(model_path)
    print(f"[loaded] {model_path}")
    return model


def get_predictions(model, test_gen) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    y_true    : int  (N,)
    y_pred    : int  (N,)
    y_proba   : float (N, C) — probabilitas tiap kelas
    """
    test_gen.reset()
    y_proba = model.predict(test_gen, verbose=1)
    y_pred  = np.argmax(y_proba, axis=1)
    y_true  = test_gen.classes
    return y_true, y_pred, y_proba


# ------------------------------------------------------------------ #
#  CONFUSION MATRIX                                                    #
# ------------------------------------------------------------------ #

def plot_confusion_matrix(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    class_names: list = CLASSES,
    save_path: str   = None,
    normalize: bool  = False,
):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt, title = ".2f", "Normalized Confusion Matrix"
    else:
        fmt, title = "d", "Confusion Matrix"

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot       = True,
        fmt         = fmt,
        cmap        = "Blues",
        xticklabels = class_names,
        yticklabels = class_names,
        linewidths  = 0.5,
    )
    plt.title(title, fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[saved] {save_path}")
    plt.show()
    return cm


# ------------------------------------------------------------------ #
#  CLASSIFICATION REPORT                                               #
# ------------------------------------------------------------------ #

def print_classification_report(
    y_true:      np.ndarray,
    y_pred:      np.ndarray,
    class_names: list = CLASSES,
    save_path:   str  = None,
):
    report = classification_report(
        y_true, y_pred,
        target_names = class_names,
        digits       = 4,
    )
    print("\n" + "="*60)
    print("  CLASSIFICATION REPORT")
    print("="*60)
    print(report)

    if save_path:
        with open(save_path, "w") as f:
            f.write(report)
        print(f"[saved] {save_path}")
    return report


# ------------------------------------------------------------------ #
#  TRAINING CURVES                                                     #
# ------------------------------------------------------------------ #

def plot_training_curves(
    history:   dict,
    save_dir:  str = None,
):
    save_dir = save_dir or EVAL_DIR
    ensure_dirs(save_dir)

    epochs = range(1, len(history["accuracy"]) + 1)
    freeze_end = None
    # Deteksi epoch akhir phase freeze dari history (opsional marker)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Accuracy ---
    axes[0].plot(epochs, history["accuracy"],     label="Train Acc",  color="#2196F3")
    axes[0].plot(epochs, history["val_accuracy"], label="Val Acc",    color="#FF5722", linestyle="--")
    axes[0].set_title("Accuracy", fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # --- Loss ---
    axes[1].plot(epochs, history["loss"],     label="Train Loss", color="#2196F3")
    axes[1].plot(epochs, history["val_loss"], label="Val Loss",   color="#FF5722", linestyle="--")
    axes[1].set_title("Loss", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Training & Validation Curves — MobileNetV2",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    print(f"[saved] {path}")
    plt.show()


# ------------------------------------------------------------------ #
#  ROC CURVE (multi-class OvR)                                        #
# ------------------------------------------------------------------ #

def plot_roc_curves(
    y_true:    np.ndarray,
    y_proba:   np.ndarray,
    class_names: list = CLASSES,
    save_path: str   = None,
):
    from sklearn.preprocessing import label_binarize

    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))
    plt.figure(figsize=(9, 7))

    colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
    for i, (cls, col) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc     = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=col, lw=2,
                 label=f"{cls} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0, 1]); plt.ylim([0, 1.02])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("ROC Curve — One-vs-Rest", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[saved] {save_path}")
    plt.show()


# ------------------------------------------------------------------ #
#  PIPELINE EVALUASI LENGKAP                                          #
# ------------------------------------------------------------------ #

def run_evaluation(model_path: str = None):
    ensure_dirs(EVAL_DIR)

    print("\n[1/5] Load model …")
    model = load_model(model_path)

    print("\n[2/5] Load test generator …")
    test_gen = get_test_generator(TEST_DIR, batch_size=BATCH_SIZE)
    class_names = list(test_gen.class_indices.keys())

    print("\n[3/5] Prediksi …")
    y_true, y_pred, y_proba = get_predictions(model, test_gen)

    print("\n[4/5] Confusion Matrix …")
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path = os.path.join(EVAL_DIR, "confusion_matrix.png"),
    )
    plot_confusion_matrix(
        y_true, y_pred, class_names, normalize=True,
        save_path = os.path.join(EVAL_DIR, "confusion_matrix_norm.png"),
    )

    print("\n[5/5] Classification Report …")
    print_classification_report(
        y_true, y_pred, class_names,
        save_path = os.path.join(EVAL_DIR, "classification_report.txt"),
    )

    # Training curves
    hist_path = os.path.join(MODEL_DIR, "history.pkl")
    if os.path.exists(hist_path):
        history = load_pickle(hist_path)
        plot_training_curves(history, save_dir=EVAL_DIR)

    # ROC
    plot_roc_curves(
        y_true, y_proba, class_names,
        save_path = os.path.join(EVAL_DIR, "roc_curves.png"),
    )

    # Simpan metrik ringkas
    test_loss, test_acc, *_ = model.evaluate(test_gen, verbose=0)
    print(f"\n{'='*40}")
    print(f"  Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"{'='*40}\n")

    return y_true, y_pred, y_proba


if __name__ == "__main__":
    run_evaluation()
