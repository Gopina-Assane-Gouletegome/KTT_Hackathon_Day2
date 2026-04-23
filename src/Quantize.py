"""
T2.1 · ONNX INT8 Quantization
Run from project root: python src/quantize.py
"""

import os, pathlib
import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

# ── Config ────────────────────────────────────────────────────────────────────
_HERE        = pathlib.Path(__file__).resolve().parent
_ROOT        = _HERE.parent
FP32_MODEL   = _ROOT / "model.onnx"
INT8_MODEL   = _ROOT / "model_int8.onnx"
DATA_DIR     = _ROOT / "mini_plant_set"
IMG_SIZE     = 224
NUM_CAL_IMGS = 100


# ── Calibration data reader ───────────────────────────────────────────────────
class LeafCalibrationReader(CalibrationDataReader):
    def __init__(self, input_name: str):
        self._input_name = input_name
        self._images     = self._load()
        self._it         = iter(self._images)

    def _load(self):
        images = []
        paths  = sorted(DATA_DIR.rglob("*.jpg"))[:NUM_CAL_IMGS]
        for p in paths:
            img = cv2.imread(str(p))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img.astype(np.float32)[np.newaxis] / 255.0)
        print(f"  Loaded {len(images)} calibration images from {DATA_DIR}")
        return images

    def get_next(self):
        batch = next(self._it, None)
        return {self._input_name: batch} if batch is not None else None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  T2.1 · ONNX INT8 Quantization")
    print("=" * 60)

    # Validate inputs
    if not FP32_MODEL.exists():
        raise FileNotFoundError(
            f"❌  {FP32_MODEL} not found. Run train.py first."
        )
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"❌  {DATA_DIR} not found. Run train.py first to generate the dataset."
        )

    fp32_mb = FP32_MODEL.stat().st_size / 1e6
    print(f"\n  FP32 model : {FP32_MODEL}  ({fp32_mb:.2f} MB)")
    print(f"  Output     : {INT8_MODEL}")

    # Get input name from FP32 model
    sess       = ort.InferenceSession(str(FP32_MODEL), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    print(f"  Input name : {input_name}")

    # Run static INT8 quantization
    print("\n[Quantizing] Running calibration + INT8 conversion …")
    quantize_static(
        model_input=str(FP32_MODEL),
        model_output=str(INT8_MODEL),
        calibration_data_reader=LeafCalibrationReader(input_name),
        weight_type=QuantType.QInt8,
    )

    int8_mb = INT8_MODEL.stat().st_size / 1e6
    print(f"\n✅  model_int8.onnx saved → {INT8_MODEL}  ({int8_mb:.2f} MB)")

    if int8_mb < 10:
        print(f"✅  Size check passed ({int8_mb:.2f} MB < 10 MB)")
    else:
        print(f"❌  Size check FAILED ({int8_mb:.2f} MB ≥ 10 MB)")

    print(f"    Compression: {fp32_mb:.2f} MB → {int8_mb:.2f} MB  "
          f"({(1 - int8_mb/fp32_mb)*100:.1f}% reduction)")
    print("\n🎉  Quantization complete!")


if __name__ == "__main__":
    main()