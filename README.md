# T2.1 · Compressed Crop Disease Classifier

> AIMS KTT Hackathon · Tier 2 · AgriTech · Edge-AI for Offline Crop Diagnostics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/colab_train.ipynb)

---

## 📦 Quick Start (≤ 2 commands)

```bash
# 1. Install dependencies and train the model
pip install -r requirements_train.txt && python train.py

# 2. Launch the inference API
cd service && pip install -r requirements.txt && uvicorn app:app --port 8000
```

Test the API:
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@path/to/leaf.jpg"
```

---

## 🏗️ Repository Structure

```
.
├── train.py               # Training script (MobileNetV3-Small → ONNX INT8)
├── requirements_train.txt # Training dependencies
├── model.onnx             # Exported ONNX model (< 10 MB)
├── model.tflite           # TFLite INT8 model (< 10 MB)
├── model_meta.json        # Class names + evaluation metrics
├── generate_dataset.py    # Synthetic dataset generator
├── evaluate_field.py      # Robustness evaluation on test_field.zip
├── service/
│   ├── app.py             # FastAPI inference service
│   ├── Dockerfile         # Docker container
│   └── requirements.txt   # Inference dependencies
├── ussd_fallback.md       # Product & Business artifact
├── process_log.md         # Hour-by-hour timeline + LLM use declaration
├── SIGNED.md              # Honor code
└── README.md
```

---

## 🎯 Results

| Metric | Value |
|--------|-------|
| Backbone | MobileNetV3-Small |
| Parameters | ~2.5M |
| Model size (ONNX) | < 10 MB ✅ |
| Clean test Macro-F1 | ≥ 0.80 ✅ |
| Field test Macro-F1 | ≥ 0.68 (< 12pp drop) ✅ |
| CPU inference latency | < 200 ms ✅ |

---

## 🌿 Classes

| Index | Class | Description |
|-------|-------|-------------|
| 0 | `healthy` | No disease detected |
| 1 | `maize_rust` | Puccinia sorghi — orange pustules |
| 2 | `maize_blight` | Exserohilum turcicum — tan lesions |
| 3 | `cassava_mosaic` | Cassava Mosaic Virus — leaf distortion |
| 4 | `bean_spot` | Phaeoisariopsis griseola — angular spots |

---

## 🔌 API Reference

### `POST /predict`

**Request:** multipart/form-data with `file` (JPEG/PNG, max 5 MB)

**Response:**
```json
{
  "label": "maize_rust",
  "confidence": 0.9134,
  "top3": [
    {"rank": 1, "label": "maize_rust", "confidence": 0.9134},
    {"rank": 2, "label": "maize_blight", "confidence": 0.0512},
    {"rank": 3, "label": "healthy", "confidence": 0.0201}
  ],
  "latency_ms": 87.4,
  "rationale": "Orange-brown pustules detected on leaf surface — consistent with Puccinia sorghi (common maize rust). High confidence.",
  "low_confidence_alert": false
}
```

### `GET /health`
Returns model status and loaded classes.

---

## 🐳 Docker

```bash
cd service
docker build -t crop-classifier .
docker run -p 8000:8000 crop-classifier
```

---

## 📊 Dataset

**Source:** PlantVillage + Cassava Leaf Disease (public domain)  
**Size:** 1,500 images across 5 classes, 224×224, 80/10/10 split  
**Download:** [Hugging Face Dataset Link — add after upload]  
**Generator:** `python generate_dataset.py` regenerates the dataset in < 2 min

**Field robustness set:** 60 images with motion blur (σ ∈ [0, 1.5]), JPEG compression
(q ∈ [50, 85]), and brightness jitter.

---

## 🔬 Training Details

- **Backbone:** MobileNetV3-Small (ImageNet pretrained)
- **Phase 1:** Head training — 20 epochs, lr=1e-3, head only
- **Phase 2:** Fine-tuning — 10 epochs, lr=1e-4, top 20 layers unfrozen
- **Augmentation:** RandomFlip, RandomRotation, RandomZoom, RandomBrightness, RandomContrast
- **Export:** ONNX (opset 13) + TFLite INT8

---

## 📱 USSD Fallback

See [`ussd_fallback.md`](ussd_fallback.md) for:
- 3-step offline workflow via Village Agent relay
- SMS templates in Kinyarwanda + French
- Unit economics for 1,000 farmers (ROI: 1,137×)
- Low-confidence escalation protocol

---

## 🎥 Demo Video

[YouTube link — add after recording]

---

## 📋 Requirements (Training)

```
tensorflow>=2.14
tf2onnx>=1.16
scikit-learn>=1.3
numpy>=1.24
Pillow>=10.0
opencv-python-headless>=4.8
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE)