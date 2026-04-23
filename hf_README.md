---
language:
- en
- rw
- fr
license: mit
tags:
- image-classification
- plant-disease
- agriculture
- onnx
- mobilenetv3
- edge-ai
datasets:
- plant_village
- cassava
metrics:
- f1
model-index:
- name: crop-disease-classifier
  results:
  - task:
      type: image-classification
    dataset:
      name: PlantVillage + Cassava Leaf Disease
      type: plant_village
    metrics:
    - type: f1
      value: 1.0
      name: Macro F1 (clean test)
    - type: f1
      value: 1.0
      name: Macro F1 (field robustness)
---

# 🌿 Crop Disease Classifier — T2.1

Edge-AI crop disease detection for offline diagnostics in rural Africa.  
Trained for the **AIMS KTT Hackathon · Tier 2 · Challenge 1**.

## Model Description

A fine-tuned **MobileNetV3-Small** backbone that classifies maize, cassava, and bean leaf images into 5 classes. Exported to ONNX (FP32 + INT8) for CPU-only edge deployment — no GPU, no internet required at inference.

| File | Size | Format |
|------|------|--------|
| `model.onnx` | 3.78 MB | ONNX FP32 |
| `model_int8.onnx` | ~1.2 MB | ONNX INT8 (static quantization) |

## Classes

| Index | Label | Disease |
|-------|-------|---------|
| 0 | `healthy` | No disease |
| 1 | `maize_rust` | Puccinia sorghi — common maize rust |
| 2 | `maize_blight` | Exserohilum turcicum — northern leaf blight |
| 3 | `cassava_mosaic` | Cassava Mosaic Virus (CMV) |
| 4 | `bean_spot` | Phaeoisariopsis griseola — angular leaf spot |

## Performance

| Split | Macro F1 |
|-------|----------|
| Clean test set | **1.00** |
| Field robustness (blur + JPEG + brightness jitter) | **1.00** |
| Robustness drop | **0.00 pp** (target: < 12 pp) ✅ |

## Usage

```python
import cv2, numpy as np, onnxruntime as ort

session    = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
CLASSES    = ["healthy", "maize_rust", "maize_blight", "cassava_mosaic", "bean_spot"]

img = cv2.imread("leaf.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224)).astype(np.float32)[np.newaxis] / 255.0

probs = session.run(None, {input_name: img})[0][0]
print(CLASSES[probs.argmax()], f"{probs.max():.2%}")
```

## Training

- **Backbone:** MobileNetV3-Small (ImageNet pretrained)
- **Phase 1:** Head training — 20 epochs, lr=1e-3
- **Phase 2:** Fine-tuning top 20 layers — 10 epochs, lr=1e-4
- **Data:** 300 images/class from PlantVillage + Cassava Leaf Disease (TFDS)
- **Augmentation:** RandomFlip, RandomRotation, RandomZoom, RandomBrightness, RandomContrast

## Intended Use

Designed for deployment via Village Agent relay networks in Rwanda and DRC. Supports offline diagnosis with USSD/SMS fallback for feature-phone users. See the [GitHub repo](https://github.com/YOUR_USERNAME/YOUR_REPO) for the full system design including USSD templates in Kinyarwanda + French.

## Limitations

- `bean_spot` class uses augmented synthetic data (no real Bean Angular Leaf Spot images found in TFDS) — real-world performance on this class may be lower.
- Trained on 224×224 lab-condition images; field performance depends on photo quality.
- Not a substitute for agronomist advice in ambiguous cases.