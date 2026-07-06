"""
Arsitektur model:
  MobileNetV2 (ImageNet pretrained) + Custom Classification Head

Dua fase:
  1. Feature Extraction  — base frozen, hanya head yang dilatih
  2. Fine-Tuning         — unfreeze layer atas base, LR sangat kecil
"""
import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

from src.config import (
    INPUT_SHAPE, NUM_CLASSES, CLASSES,
    MOBILENET_WEIGHTS, FINE_TUNE_AT,
    LEARNING_RATE, FINETUNE_LR, DROPOUT_RATE
)


# ------------------------------------------------------------------ #
#  BUILD MODEL                                                         #
# ------------------------------------------------------------------ #

def build_model(
    input_shape:   tuple = INPUT_SHAPE,
    num_classes:   int   = NUM_CLASSES,
    dropout_rate:  float = DROPOUT_RATE,
    weights:       str   = MOBILENET_WEIGHTS,
) -> Model:
    """
    Buat model MobileNetV2 + classification head.

    Arsitektur head:
      GlobalAveragePooling2D
      → BatchNormalization
      → Dense(256, relu, L2)
      → Dropout
      → Dense(num_classes, softmax)
    """
    # --- Base model ---
    base = MobileNetV2(
        input_shape = input_shape,
        include_top = False,
        weights     = weights,
    )
    base.trainable = False          # Frozen saat feature extraction

    # --- Classification head ---
    inputs = tf.keras.Input(shape=input_shape, name="input_image")
    x      = base(inputs, training=False)
    x      = layers.GlobalAveragePooling2D(name="gap")(x)
    x      = layers.BatchNormalization(name="bn")(x)
    x      = layers.Dense(
                 256,
                 activation    = "relu",
                 kernel_regularizer = regularizers.l2(1e-4),
                 name          = "dense_256"
             )(x)
    x      = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(
                 num_classes,
                 activation = "softmax",
                 name       = "predictions"
             )(x)

    model = Model(inputs, outputs, name="DisasterMobileNetV2")
    return model, base


def compile_model(model: Model, learning_rate: float = LEARNING_RATE) -> Model:
    model.compile(
        optimizer = Adam(learning_rate=learning_rate),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy",
                     tf.keras.metrics.Precision(name="precision"),
                     tf.keras.metrics.Recall(name="recall"),
                     tf.keras.metrics.AUC(name="auc")],
    )
    return model


# ------------------------------------------------------------------ #
#  FINE-TUNING: UNFREEZE LAYER ATAS                                   #
# ------------------------------------------------------------------ #

def enable_fine_tuning(
    model:       Model,
    base_model:  Model,
    fine_tune_at: int   = FINE_TUNE_AT,
    lr:          float  = FINETUNE_LR,
) -> Model:
    """
    Unfreeze layer base_model mulai dari indeks fine_tune_at.
    Recompile dengan LR lebih kecil.
    """
    base_model.trainable = True

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    n_trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"[Fine-Tune] Trainable layers di base: {n_trainable} "
          f"(dari layer ke-{fine_tune_at})")

    model = compile_model(model, learning_rate=lr)
    return model


# ------------------------------------------------------------------ #
#  SUMMARY HELPER                                                      #
# ------------------------------------------------------------------ #

def model_summary(model: Model):
    model.summary(line_length=90)
    total      = model.count_params()
    trainable  = sum(tf.size(w).numpy() for w in model.trainable_weights)
    frozen     = total - trainable
    print(f"\nTotal params     : {total:,}")
    print(f"Trainable params : {trainable:,}")
    print(f"Frozen params    : {frozen:,}\n")


if __name__ == "__main__":
    model, base = build_model()
    model = compile_model(model)
    model_summary(model)
