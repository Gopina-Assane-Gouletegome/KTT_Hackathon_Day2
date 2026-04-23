"""
T2.1 · Crop Disease Classifier · FastAPI Inference Service
Run (from anywhere):  uvicorn src.app:app --host 0.0.0.0 --port 8000
         or:          python src/app.py
"""

import io, time, json, sys
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

# ── Paths — always resolved relative to THIS file, not cwd ───────────────────
_HERE      = Path(__file__).resolve().parent   # .../src/  (or .../service/)
_ROOT      = _HERE.parent                      # project root  (contains model.onnx)
MODEL_PATH = _ROOT / "model.onnx"
META_PATH  = _ROOT / "model_meta.json"
IMG_SIZE   = 224

# ── Graceful startup check ────────────────────────────────────────────────────
def _check_files():
    missing = [p for p in (MODEL_PATH, META_PATH) if not p.exists()]
    if missing:
        print("\n❌  Missing files (train first with: python src/train.py):")
        for p in missing:
            print(f"     {p}")
        sys.exit(1)

_check_files()

# ── Load model once at startup ────────────────────────────────────────────────
with open(META_PATH) as f:
    META = json.load(f)
CLASSES = META["classes"]

SESSION     = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
INPUT_NAME  = SESSION.get_inputs()[0].name
OUTPUT_NAME = SESSION.get_outputs()[0].name


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Crop Disease Classifier",
    description="Edge-AI crop disease detection for offline diagnostics (T2.1)",
    version="1.0.0",
)


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Decode JPEG/PNG, resize to 224x224, normalize to [0,1], add batch dim."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]   # (1, 224, 224, 3)


def generate_rationale(label: str, confidence: float) -> str:
    rationales = {
        "healthy": (
            "Leaf coloration and texture appear normal. "
            "No visible lesions, discoloration, or abnormal spots detected."
        ),
        "maize_rust": (
            "Orange-brown pustules detected on leaf surface — consistent with "
            "Puccinia sorghi (common maize rust). High humidity favors spread."
        ),
        "maize_blight": (
            "Large tan/brown necrotic lesions with wavy margins detected — "
            "consistent with Northern Corn Leaf Blight (Exserohilum turcicum)."
        ),
        "cassava_mosaic": (
            "Mosaic yellowing and leaf distortion pattern detected — consistent "
            "with Cassava Mosaic Virus (CMV), spread by whitefly vectors."
        ),
        "bean_spot": (
            "Angular water-soaked lesions detected — consistent with Bean "
            "Angular Leaf Spot (Phaeoisariopsis griseola)."
        ),
    }
    conf_note = (
        "High confidence." if confidence > 0.85
        else "Moderate confidence — consider a second photo from a different angle."
        if confidence > 0.65
        else "Low confidence — please retake photo in good lighting."
    )
    return f"{rationales.get(label, 'Unknown condition.')} {conf_note}"


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    top3: list
    latency_ms: float
    rationale: str
    low_confidence_alert: bool


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Accept a JPEG/PNG and return crop disease diagnosis."""
    if file.content_type not in ("image/jpeg", "image/jpg", "image/png"):
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are accepted.")

    image_bytes = await file.read()
    if len(image_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 5 MB).")

    try:
        inp = preprocess(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image decode error: {e}")

    t0 = time.perf_counter()
    outputs    = SESSION.run([OUTPUT_NAME], {INPUT_NAME: inp})[0][0]   # (5,)
    latency_ms = (time.perf_counter() - t0) * 1000

    top3_idx   = np.argsort(outputs)[::-1][:3]
    label      = CLASSES[int(top3_idx[0])]
    confidence = float(outputs[top3_idx[0]])

    top3 = [
        {"rank": i + 1, "label": CLASSES[int(idx)], "confidence": round(float(outputs[idx]), 4)}
        for i, idx in enumerate(top3_idx)
    ]

    return PredictionResponse(
        label=label,
        confidence=round(confidence, 4),
        top3=top3,
        latency_ms=round(latency_ms, 2),
        rationale=generate_rationale(label, confidence),
        low_confidence_alert=(confidence < 0.65),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": str(MODEL_PATH), "classes": CLASSES}


@app.get("/")
def root():
    return {"message": "Crop Disease Classifier API — POST /predict with a JPEG image."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)

