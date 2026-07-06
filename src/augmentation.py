"""
Augmentasi data training menggunakan ImageDataGenerator Keras.
Augmentasi hanya diterapkan pada set TRAINING, bukan val/test.
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from src.config import (
    IMAGE_SIZE, BATCH_SIZE,
    ROTATION_RANGE, WIDTH_SHIFT_RANGE, HEIGHT_SHIFT_RANGE,
    SHEAR_RANGE, ZOOM_RANGE, HORIZONTAL_FLIP, VERTICAL_FLIP,
    BRIGHTNESS_RANGE, TRAIN_DIR, VAL_DIR, TEST_DIR
)


def get_train_generator(
    train_dir:  str   = TRAIN_DIR,
    image_size: tuple = IMAGE_SIZE,
    batch_size: int   = BATCH_SIZE,
    seed:       int   = 42,
):
    """
    Generator training WITH augmentasi.
    preprocess_input MobileNetV2 skala pixel ke [-1, 1].
    """
    datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        rotation_range         = ROTATION_RANGE,
        width_shift_range      = WIDTH_SHIFT_RANGE,
        height_shift_range     = HEIGHT_SHIFT_RANGE,
        shear_range            = SHEAR_RANGE,
        zoom_range             = ZOOM_RANGE,
        horizontal_flip        = HORIZONTAL_FLIP,
        vertical_flip          = VERTICAL_FLIP,
        brightness_range       = BRIGHTNESS_RANGE,
        fill_mode              = "nearest",
    )
    gen = datagen.flow_from_directory(
        train_dir,
        target_size  = image_size,
        batch_size   = batch_size,
        class_mode   = "categorical",
        shuffle      = True,
        seed         = seed,
    )
    return gen


def get_val_generator(
    val_dir:    str   = VAL_DIR,
    image_size: tuple = IMAGE_SIZE,
    batch_size: int   = BATCH_SIZE,
):
    """Generator validasi — HANYA preprocess, tanpa augmentasi."""
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    gen = datagen.flow_from_directory(
        val_dir,
        target_size = image_size,
        batch_size  = batch_size,
        class_mode  = "categorical",
        shuffle     = False,
    )
    return gen


def get_test_generator(
    test_dir:   str   = TEST_DIR,
    image_size: tuple = IMAGE_SIZE,
    batch_size: int   = BATCH_SIZE,
):
    """Generator test — HANYA preprocess, tanpa augmentasi."""
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    gen = datagen.flow_from_directory(
        test_dir,
        target_size = image_size,
        batch_size  = batch_size,
        class_mode  = "categorical",
        shuffle     = False,
    )
    return gen
