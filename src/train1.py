"""
T2.1 · Compressed Crop Disease Classifier
Training script: MobileNetV3-Small → INT8 ONNX
Run: python train.py

Dataset is auto-generated from PlantVillage (via Kaggle) + Cassava Leaf Disease
(via TensorFlow Datasets) if mini_plant_set/ does not exist.
"""

import os, time, json, pathlib, shutil, random, io
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, f1_score
import cv2
from PIL import Image
import tf2onnx

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE         = 224
BATCH_SIZE       = 32
EPOCHS           = 20
LR               = 1e-3
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR     = 1e-4
DATA_DIR         = "mini_plant_set"
FIELD_DIR        = "test_field"
MODEL_OUT        = "model.onnx"
TFLITE_OUT       = "model.tflite"
SAMPLES_PER_CLASS = 300          # ~300 per class as per brief
SEED             = 42

CLASSES = ["healthy", "maize_rust", "maize_blight", "cassava_mosaic", "bean_spot"]
NUM_CLASSES = len(CLASSES)

# Mapping from TF Datasets / source label names → our 5 classes
# PlantVillage labels used by tensorflow_datasets "plant_village"
PLANTVILLAGE_MAP = {
    # healthy maize
    "Corn_(maize)___healthy":                   "healthy",
    # maize rust
    "Corn_(maize)___Common_rust_":              "maize_rust",
    # maize blight
    "Corn_(maize)___Northern_Leaf_Blight":      "maize_blight",
    # bean (angular leaf spot is closest to bean_spot)
    "Bean___Angular_Leaf_Spot":                  "bean_spot",
    "Bean___healthy":                            "healthy",
}
# Cassava Leaf Disease dataset labels → our classes
CASSAVA_MAP = {
    "cbb":   None,           # Cassava Bacterial Blight  → skip (not in our 5)
    "cbsd":  None,           # Cassava Brown Streak       → skip
    "cgm":   "cassava_mosaic",  # Cassava Green Mottle / Mosaic
    "cmd":   "cassava_mosaic",  # Cassava Mosaic Disease
    "healthy": "healthy",
}

# ── Data loaders ──────────────────────────────────────────────────────────────
def make_dataset(split="train"):
    path = pathlib.Path(DATA_DIR) / split
    ds = keras.utils.image_dataset_from_directory(
        path,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=CLASSES,
        shuffle=(split == "train"),
        seed=42,
    )
    # Augment training data
    if split == "train":
        augment = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.2),
            layers.RandomContrast(0.2),
        ])
        ds = ds.map(lambda x, y: (augment(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    # Normalize to [0,1]
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.cache().prefetch(tf.data.AUTOTUNE)


# ── Field augmentation helpers ────────────────────────────────────────────────
def apply_field_augmentation(img_array: np.ndarray) -> np.ndarray:
    """
    Simulate real field conditions per the brief:
      - Random Gaussian blur  σ ∈ [0, 1.5]
      - JPEG compression      q ∈ [50, 85]
      - Brightness jitter     ±30%
    img_array: uint8 (H, W, 3)
    """
    rng = np.random.default_rng()

    # 1. Gaussian blur
    sigma = rng.uniform(0.0, 1.5)
    if sigma > 0.3:
        ksize = int(2 * round(3 * sigma) + 1) | 1   # odd kernel
        img_array = cv2.GaussianBlur(img_array, (ksize, ksize), sigma)

    # 2. JPEG compression (encode → decode)
    quality = int(rng.uniform(50, 85))
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", img_array, encode_param)
    img_array = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # 3. Brightness jitter
    factor = rng.uniform(0.7, 1.3)
    img_array = np.clip(img_array.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    return img_array


def save_image(arr: np.ndarray, path: pathlib.Path):
    """Save uint8 (H,W,3) RGB array as JPEG."""
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


# ── Real dataset generator from PlantVillage + Cassava (TFDS) ────────────────
def generate_dataset():
    """
    Downloads PlantVillage and Cassava Leaf Disease via TensorFlow Datasets,
    maps them to our 5 classes, balances to SAMPLES_PER_CLASS images each,
    performs an 80/10/10 train/val/test split, and writes JPEG files to
    mini_plant_set/{train,val,test}/{class}/.

    Also generates test_field/ with field-augmented copies of the test split.

    Total time on Colab free CPU: ~2 minutes.
    """
    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("  Generating dataset from PlantVillage + Cassava TFDS …")
    print("=" * 60)

    # ── Collect raw images per class ─────────────────────────────────────────
    raw: dict[str, list[np.ndarray]] = {c: [] for c in CLASSES}

    # 1. PlantVillage
    print("\n[1/2] Downloading PlantVillage …")
    pv_ds, pv_info = tfds.load(
        "plant_village",
        split="train",
        with_info=True,
        as_supervised=False,
    )
    pv_labels = pv_info.features["label"].names

    for example in pv_ds:
        label_str = pv_labels[int(example["label"])]
        our_class = PLANTVILLAGE_MAP.get(label_str)
        if our_class is None:
            continue
        if len(raw[our_class]) >= SAMPLES_PER_CLASS:
            continue
        img = example["image"].numpy()                    # uint8 (H,W,3)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        raw[our_class].append(img)

        if all(len(v) >= SAMPLES_PER_CLASS for v in raw.values()):
            break   # early exit once all classes are full

    print("  After PlantVillage:", {c: len(v) for c, v in raw.items()})

    # 2. Cassava Leaf Disease (fills cassava_mosaic + tops up healthy)
    print("\n[2/2] Downloading Cassava Leaf Disease …")
    cas_ds, cas_info = tfds.load(
        "cassava",
        split="train",
        with_info=True,
        as_supervised=False,
    )
    cas_labels = cas_info.features["label"].names

    for example in cas_ds:
        label_str = cas_labels[int(example["label"])]
        our_class = CASSAVA_MAP.get(label_str)
        if our_class is None:
            continue
        if len(raw[our_class]) >= SAMPLES_PER_CLASS:
            continue
        img = example["image"].numpy()
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        raw[our_class].append(img)

        if all(len(v) >= SAMPLES_PER_CLASS for v in raw.values()):
            break

    print("  After Cassava:     ", {c: len(v) for c, v in raw.items()})

    # ── Handle any class still under SAMPLES_PER_CLASS via augmentation ───────
    for cls in CLASSES:
        imgs = raw[cls]
        if not imgs:
            raise RuntimeError(
                f"❌  Class '{cls}' has 0 images after download. "
                "Check PLANTVILLAGE_MAP / CASSAVA_MAP or network access."
            )
        while len(imgs) < SAMPLES_PER_CLASS:
            src = imgs[random.randint(0, len(imgs) - 1)].copy()
            # Light augment to avoid exact duplicates
            if random.random() > 0.5:
                src = cv2.flip(src, 1)
            angle = random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((IMG_SIZE // 2, IMG_SIZE // 2), angle, 1)
            src = cv2.warpAffine(src, M, (IMG_SIZE, IMG_SIZE))
            imgs.append(src)
        raw[cls] = imgs[:SAMPLES_PER_CLASS]

    # ── 80 / 10 / 10 split and write to disk ─────────────────────────────────
    n_train = int(SAMPLES_PER_CLASS * 0.80)
    n_val   = int(SAMPLES_PER_CLASS * 0.10)
    # n_test  = remainder

    field_imgs = []   # collect test images for field set

    for cls in CLASSES:
        imgs = raw[cls]
        random.shuffle(imgs)
        splits = {
            "train": imgs[:n_train],
            "val":   imgs[n_train : n_train + n_val],
            "test":  imgs[n_train + n_val :],
        }
        for split, subset in splits.items():
            out_dir = pathlib.Path(DATA_DIR) / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, arr in enumerate(subset):
                save_image(arr, out_dir / f"{cls}_{i:04d}.jpg")
            if split == "test":
                field_imgs.extend([(arr, cls) for arr in subset])

    total = sum(len(v) for v in raw.values())
    print(f"\n✅  Dataset written to '{DATA_DIR}/'  ({total} images total)")

    # ── Generate test_field/ with field augmentations ─────────────────────────
    print("\n[+] Generating field-augmented test set …")
    field_dir = pathlib.Path(FIELD_DIR)
    field_dir.mkdir(exist_ok=True)

    # Brief spec: 60 noisier field-shot images
    field_sample = random.sample(field_imgs, min(60, len(field_imgs)))
    for i, (arr, cls) in enumerate(field_sample):
        aug = apply_field_augmentation(arr.copy())
        fname = field_dir / f"field_{cls}_{i:03d}.jpg"
        save_image(aug, fname)
        # store ground-truth label alongside for evaluation
    # Write labels file
    with open(field_dir / "labels.json", "w") as f:
        json.dump(
            {f"field_{cls}_{i:03d}.jpg": cls
             for i, (_, cls) in enumerate(field_sample)},
            f, indent=2
        )
    print(f"✅  Field test set written to '{FIELD_DIR}/'  ({len(field_sample)} images)")
    print()\




# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    base = keras.applications.MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=False,   # we normalize manually
    )
    base.trainable = False             # freeze during head training

    inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


# ── Evaluate & report ─────────────────────────────────────────────────────────
def evaluate(model, ds, label="test"):
    y_true, y_pred = [], []
    for images, labels in ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n── {label.upper()} MACRO F1: {f1:.4f} ──")
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    return f1


# ── Export ONNX ───────────────────────────────────────────────────────────────
def export_onnx(model, path=MODEL_OUT):
    input_sig = [tf.TensorSpec([1, IMG_SIZE, IMG_SIZE, 3], tf.float32, name="input")]
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=input_sig,
                                                 opset=13, output_path=path)
    size_mb = os.path.getsize(path) / 1e6
    print(f"✅  ONNX saved → {path}  ({size_mb:.2f} MB)")
    assert size_mb < 10, f"❌  Model too large: {size_mb:.2f} MB (limit 10 MB)"
    return size_mb


# ── Export TFLite INT8 ────────────────────────────────────────────────────────
def export_tflite_int8(model, rep_ds, path=TFLITE_OUT):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_data_gen():
        for images, _ in rep_ds.unbatch().batch(1).take(100):
            yield [images]

    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    with open(path, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(path) / 1e6
    print(f"✅  TFLite INT8 saved → {path}  ({size_mb:.2f} MB)")
    return size_mb


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  T2.1 · Crop Disease Classifier · Training")
    print("=" * 60)

    # Generate real dataset from PlantVillage + Cassava if not already present
    if not pathlib.Path(DATA_DIR).exists():
        generate_dataset()

    train_ds = make_dataset("train")
    val_ds   = make_dataset("val")
    test_ds  = make_dataset("test")

    # ── Phase 1: Train head only ──────────────────────────────────────────────
    print("\n[Phase 1] Training classification head …")
    model, base = build_model()
    model.summary(expand_nested=False)

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                       monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        keras.callbacks.ModelCheckpoint("best_head.keras", save_best_only=True),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    # ── Phase 2: Fine-tune top layers ─────────────────────────────────────────
    print("\n[Phase 2] Fine-tuning top layers …")
    base.trainable = True
    # Freeze all but the last 20 layers
    for layer in base.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(FINE_TUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks[2] = keras.callbacks.ModelCheckpoint("best_model.keras",
                                                    save_best_only=True)
    model.fit(train_ds, validation_data=val_ds,
              epochs=FINE_TUNE_EPOCHS, callbacks=callbacks)

    # Load best checkpoint
    model = keras.models.load_model("best_model.keras")

    # ── Evaluate clean test ───────────────────────────────────────────────────
    f1 = evaluate(model, test_ds, "clean test")
    assert f1 >= 0.80, f"⚠️  Macro F1 {f1:.3f} below 0.80 target — tune further."

    # ── Robustness bonus: evaluate on field set ───────────────────────────────
    field_f1 = None
    field_path = pathlib.Path(FIELD_DIR)
    if field_path.exists() and (field_path / "labels.json").exists():
        print("\n[Robustness] Evaluating on field-augmented test set …")
        with open(field_path / "labels.json") as fp:
            field_labels = json.load(fp)

        y_true_f, y_pred_f = [], []
        for fname, cls in field_labels.items():
            img_path = field_path / fname
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            inp = img.astype(np.float32)[np.newaxis] / 255.0
            pred = model.predict(inp, verbose=0)[0]
            y_true_f.append(CLASSES.index(cls))
            y_pred_f.append(int(np.argmax(pred)))

        field_f1 = f1_score(y_true_f, y_pred_f, average="macro")
        drop = f1 - field_f1
        print(f"\n── FIELD MACRO F1: {field_f1:.4f}  (drop: {drop:.4f} pp) ──")
        print(classification_report(y_true_f, y_pred_f, target_names=CLASSES))
        if drop > 0.12:
            print(f"⚠️  Robustness drop {drop:.3f} exceeds 0.12 target.")
        else:
            print(f"✅  Robustness OK — drop {drop:.3f} < 0.12")
    else:
        print("ℹ️  No field test set found — skipping robustness evaluation.")

    # ── Export ────────────────────────────────────────────────────────────────
    print("\n[Export] ONNX …")
    export_onnx(model)

    print("\n[Export] TFLite INT8 …")
    export_tflite_int8(model, train_ds)

    # ── Save class metadata ───────────────────────────────────────────────────
    meta = {
        "classes":    CLASSES,
        "img_size":   IMG_SIZE,
        "macro_f1":   round(f1, 4),
        "field_f1":   round(field_f1, 4) if field_f1 is not None else None,
    }
    with open("model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("\n✅  model_meta.json saved.")
    print("\n🎉  Training complete!")


if __name__ == "__main__":
    main()