"""
T2.1 · Crop Disease Classifier · Gradio UI
3-step Village Agent workflow for feature-phone farmers

Run: python src/gradio_app.py
"""

import json
import pathlib
import time

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE      = pathlib.Path(__file__).resolve().parent
_ROOT      = _HERE.parent
MODEL_PATH = _ROOT / "model.onnx"
META_PATH  = _ROOT / "model_meta.json"

# ── Load model ────────────────────────────────────────────────────────────────
with open(META_PATH) as f:
    META = json.load(f)

CLASSES = META["classes"]
SESSION = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
INPUT_NAME  = SESSION.get_inputs()[0].name

# ── Disease metadata ──────────────────────────────────────────────────────────
DISEASE_INFO = {
    "healthy": {
        "emoji": "✅",
        "kiny":  "Umutsima — Nta ndwara",
        "fr":    "Sain — Aucune maladie",
        "action_en": "No treatment needed. Continue normal care.",
        "action_kiny": "Nta muti usabwa. Komeza isarura nk'isanzwe.",
        "action_fr":  "Aucun traitement nécessaire. Continuez les soins normaux.",
        "treatment": None,
    },
    "maize_rust": {
        "emoji": "🟠",
        "kiny":  "Rustu y'ibigori",
        "fr":    "Rouille du Maïs",
        "action_en": "Apply Mancozeb (2g/L) or Propiconazole (1mL/L). Spray 2× per week for 3 weeks.",
        "action_kiny": "Koresha Mancozeb (2g/L) cyangwa Propiconazole (1mL/L). Fifikira inshuro 2 mu cyumweru.",
        "action_fr":  "Appliquer Mancozeb (2g/L) ou Propiconazole (1mL/L). Traiter 2×/semaine pendant 3 semaines.",
        "treatment": "Mancozeb 2g/L  |  Propiconazole 1mL/L",
    },
    "maize_blight": {
        "emoji": "🟤",
        "kiny":  "Ubwandu bw'ibigori",
        "fr":    "Brûlure du Maïs",
        "action_en": "Apply Azoxystrobin (1mL/L) or Chlorothalonil (2g/L). Remove and burn infected leaves.",
        "action_kiny": "Koresha Azoxystrobin (1mL/L). Kuramo amababi yaranzwe hanyuma uyashe.",
        "action_fr":  "Appliquer Azoxystrobin (1mL/L). Retirer et brûler les feuilles infectées.",
        "treatment": "Azoxystrobin 1mL/L  |  Chlorothalonil 2g/L",
    },
    "cassava_mosaic": {
        "emoji": "🟡",
        "kiny":  "Indwara ya Mosaique (Manioke)",
        "fr":    "Mosaïque du Manioc",
        "action_en": "No chemical cure. Uproot and burn infected plants. Use whitefly control. Plant resistant varieties next season.",
        "action_kiny": "Nta muti wa chimique. Kuramo imizi irwaye uyishe. Fata inzuki zibi. Biba indyo zikingira.",
        "action_fr":  "Aucun remède chimique. Arracher et brûler les plantes infectées. Contrôler les aleurodes.",
        "treatment": "No chemical cure — rogue & burn plants",
    },
    "bean_spot": {
        "emoji": "🔴",
        "kiny":  "Indwara y'ibishyimbo",
        "fr":    "Tache Angulaire du Haricot",
        "action_en": "Apply Copper Oxychloride (2g/L) weekly for 3 weeks. Avoid overhead irrigation.",
        "action_kiny": "Koresha Copper Oxychloride (2g/L) buri cyumweru mu byumweru 3.",
        "action_fr":  "Appliquer Oxychlorure de cuivre (2g/L) hebdomadairement pendant 3 semaines.",
        "treatment": "Copper Oxychloride 2g/L",
    },
}

# ── Inference ─────────────────────────────────────────────────────────────────
def run_inference(pil_image: Image.Image):
    img = pil_image.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)[np.newaxis] / 255.0   # (1,224,224,3)
    t0  = time.perf_counter()
    out = SESSION.run(None, {INPUT_NAME: arr})[0][0]             # (5,)
    latency_ms = (time.perf_counter() - t0) * 1000
    return out, latency_ms


# ── SMS template builder ──────────────────────────────────────────────────────
def build_sms(label: str, confidence: float, lang: str) -> str:
    info = DISEASE_INFO[label]
    pct  = int(confidence * 100)

    if lang == "Kinyarwanda":
        return (
            f"INDWARA Y'IKIMERA | Ubutumwa bw'ubuganga\n"
            f"─────────────────────────────\n"
            f"Icyerekezo : {info['kiny']}\n"
            f"Ikizere    : {pct}%\n\n"
            f"IBIKORWA BISABWA:\n{info['action_kiny']}\n\n"
            f"Baza inzobere: 0800-AGR-123 (Ubuntu)"
        )
    else:
        return (
            f"DIAGNOSTIC PLANTE | Service IA Agricole\n"
            f"─────────────────────────────\n"
            f"Diagnostic : {info['fr']}\n"
            f"Confiance  : {pct}%\n\n"
            f"ACTIONS RECOMMANDÉES:\n{info['action_fr']}\n\n"
            f"Assistance: 0800-AGR-123 (gratuit)"
        )


# ── Main prediction function ──────────────────────────────────────────────────
def predict(image, language):
    if image is None:
        return (
            "## ⚠️ No image uploaded",
            "", "", "", "", ""
        )

    pil = Image.fromarray(image) if not isinstance(image, Image.Image) else image
    probs, latency_ms = run_inference(pil)

    top3_idx   = np.argsort(probs)[::-1][:3]
    label      = CLASSES[int(top3_idx[0])]
    confidence = float(probs[top3_idx[0]])
    low_conf   = confidence < 0.65
    info       = DISEASE_INFO[label]

    # ── Step 2 output: Diagnosis panel ───────────────────────────────────────
    conf_pct = int(confidence * 100)
    conf_bar = "█" * (conf_pct // 5) + "░" * (20 - conf_pct // 5)

    diagnosis_md = f"""
## {info['emoji']} Diagnosis: {label.replace('_', ' ').title()}

**{info['kiny']}**  ·  *{info['fr']}*

**Confidence:** `{conf_bar}` {conf_pct}%
{"⚠️ **Low confidence** — ask farmer to retake photo from a different angle." if low_conf else ""}

---

### 🌍 Recommended Actions

**English:** {info['action_en']}

**Kinyarwanda:** {info['action_kiny']}

**Français:** {info['action_fr']}

{"**Treatment:** `" + info['treatment'] + "`" if info['treatment'] else ""}

---

*Latency: {latency_ms:.1f} ms · Model: MobileNetV3-Small ONNX*
"""

    # ── Top 3 breakdown ───────────────────────────────────────────────────────
    top3_md = "### Top 3 Predictions\n\n"
    for i, idx in enumerate(top3_idx):
        cls = CLASSES[int(idx)]
        pct = int(probs[idx] * 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        medal = ["🥇", "🥈", "🥉"][i]
        top3_md += f"{medal} **{cls.replace('_',' ')}** `{bar}` {pct}%\n\n"

    # ── Step 3: SMS template ──────────────────────────────────────────────────
    sms = build_sms(label, confidence, language)

    # ── Low confidence alert ──────────────────────────────────────────────────
    alert = ""
    if low_conf:
        alert = (
            "⚠️ **Low Confidence Alert**\n\n"
            "The model is uncertain. The Village Agent should:\n"
            "1. Ask the farmer for a second photo (different angle, better light)\n"
            "2. If still uncertain, escalate to Extension Officer via ticket\n"
            "3. Extension Officer will review and send confirmed SMS within 24h"
        )

    return diagnosis_md, top3_md, sms, alert


# ── Gradio UI ─────────────────────────────────────────────────────────────────
WORKFLOW_MD = """
## How this works — 3-Step Village Agent Workflow

| Step | Who | Action |
|------|-----|--------|
| **1️⃣ Photo Capture** | Village Agent | Farmer walks to VA (≤3 km). VA photographs the diseased leaf on their Android phone. |
| **2️⃣ Upload & Diagnose** | Village Agent | VA uploads photo here via 2G/3G. AI returns diagnosis in <2 seconds. |
| **3️⃣ SMS Delivery** | Farmer | VA reads result aloud OR farmer receives SMS/USSD in Kinyarwanda or French. |

> **Why a Village Agent?** Rwanda has ~4,500 trained VAs (1 per ~200 households). Each already owns a subsidised Android phone.  
> **Offline fallback:** VA describes symptoms via structured USSD menu → rule-based triage with no internet needed.
"""

with gr.Blocks(title="🌿 Crop Disease Classifier", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🌿 Crop Disease Classifier")
    gr.Markdown("**Edge-AI for Offline Crop Diagnostics · AIMS KTT Hackathon T2.1**")
    gr.Markdown(WORKFLOW_MD)

    gr.Markdown("---")

    # ── Step 1 ────────────────────────────────────────────────────────────────
    gr.Markdown("## 1️⃣  Step 1 — Village Agent uploads the leaf photo")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="📷 Upload leaf photo (JPG/PNG)",
                type="numpy",
                height=300,
            )
            language = gr.Radio(
                choices=["Kinyarwanda", "Français"],
                value="Kinyarwanda",
                label="SMS language for farmer",
            )
            submit_btn = gr.Button("🔍 Diagnose", variant="primary", size="lg")

        # ── Step 2 ────────────────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("## 2️⃣  Step 2 — AI Diagnosis")
            diagnosis_out = gr.Markdown(label="Diagnosis")
            top3_out      = gr.Markdown(label="Top 3")
            alert_out     = gr.Markdown(label="Alert")

    gr.Markdown("---")

    # ── Step 3 ────────────────────────────────────────────────────────────────
    gr.Markdown("## 3️⃣  Step 3 — SMS/USSD message sent to farmer")
    sms_out = gr.Textbox(
        label="📱 SMS template (copy to send via Africa's Talking / MTN API)",
        lines=10,
    )

    gr.Markdown(
        "> **Unit economics:** Cost per diagnosis ≈ $0.03 · "
        "Value of a saved maize crop ≈ $120 · ROI for 1,000 farmers ≈ **1,137×**"
    )

    # ── Wire up ───────────────────────────────────────────────────────────────
    submit_btn.click(
        fn=predict,
        inputs=[image_input, language],
        outputs=[diagnosis_out, top3_out, sms_out, alert_out],
    )

    gr.Examples(
        examples=[
            [str(p), "Kinyarwanda"]
            for p in sorted((_ROOT / "mini_plant_set" / "test").rglob("*.jpg"))[:6]
            if p.exists()
        ],
        inputs=[image_input, language],
        label="Try example images from the test set",
    )

if __name__ == "__main__":
    demo.launch(share=True)