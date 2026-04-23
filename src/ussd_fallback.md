# USSD Fallback — Crop Disease Diagnostic Service
**T2.1 · Product & Business Artifact**

---

## 1. Problem Context

A smallholder farmer in rural Rwanda or DRC owns a **feature phone only** — no
smartphone, no mobile data plan, often no reliable electricity. Yet a missed disease
diagnosis can wipe out an entire maize or cassava harvest worth **~$200–400 USD**
for a household of five.

This document defines a **3-step workflow** that delivers AI-powered crop diagnostics
to any farmer with a basic phone, using the agricultural extension network as a relay.

---

## 2. Three-Step Workflow: Photo → Upload → Diagnosis

### Step 1 · Photo Capture (farmer side)
| Who | What | Where |
|-----|------|-------|
| Farmer | Notices leaf discoloration or spots | Field |
| Farmer | Calls or walks to the nearest **Village Agent (VA)** | Within 1–3 km |
| Village Agent | Takes a photo of the leaf using their **Android feature phone with camera** | Agent's phone |

> **Why a Village Agent?**  
> Rwanda's agricultural extension system (RAB) has ~4,500 trained Village Agents
> (1 per ~200 farming households). Each VA already owns a basic smartphone
> subsidised by NGO/government programs. The VA acts as the human relay —
> no farmer needs their own smartphone.

**Alternative relay points (where VAs are unavailable):**
- **Cooperative kiosk** — most cooperatives have at least one Android device and
  periodic connectivity via 4G or satellite.
- **Mobile clinic days** — monthly extension officer visits where farmers bring
  affected leaves in sealed plastic bags for same-session diagnosis.

---

### Step 2 · Upload
| Who | What | How |
|-----|------|-----|
| Village Agent | Sends the photo to the diagnostic API | WhatsApp Business / lightweight Android app / USSD image upload (GSMA standard) |

**Connectivity strategy:**
- VA uses **2G/3G** (MTN or Airtel coverage ~78% of Rwanda's land area).
- Photo is compressed on-device to **< 150 KB** before upload (JPEG quality 60%).
- If offline, the app **queues** the image and retries when signal returns (within 4 hours
  on average).
- Fallback: VA describes symptoms via a **structured USSD menu** (no image needed);
  a rule-based triage returns a provisional result.

---

### Step 3 · Diagnosis Delivery
The API responds in **< 2 seconds** on CPU. The VA reads the result aloud in the
local language, or the farmer receives an **SMS/USSD push** directly.

---

## 3. SMS/USSD Message Templates

### 3a. Kinyarwanda Template

```
INDWARA Y'IKIMERA | Ubutumwa bw'ubuganga

Ibihingwa: [IGIHINGWA]
Icyerekezo: [ICYEREKEZO]
Ikizere: [XX]%

IGISUBIZO: [INYITO YACYO]

IBIKORWA BISABWA:
[IBIKORWA]

Niba utizeye, fata ifoto nshya.
Baza inzobere: 0800-AGR-123 (Ubuntu)
```

**Filled example — maize_rust:**
```
INDWARA Y'IKIMERA | Ubutumwa bw'ubuganga

Ibihingwa: Ibigori
Icyerekezo: Ingaruka za Rustu (Maize Rust)
Ikizere: 91%

IGISUBIZO: Uburwayi bw'uturangwa duto bwishe amababi (Puccinia sorghi).
Butera inzitizi mu isarura ryo hagati: 30-50%.

IBIKORWA BISABWA:
1. Koresha umuti wa Mancozeb (2g/L) cyangwa Propiconazole (1mL/L).
2. Fifikira inshuro 2 mu cyumweru, mu gitondo.
3. Kuramo amababi yaranzwe hanyuma uyashe.
4. Reka umusaruro wagurane n'undi mu nzira 1m.

Niba utizeye, fata ifoto nshya.
Baza inzobere: 0800-AGR-123 (Ubuntu)
```

---

### 3b. French Template

```
DIAGNOSTIC PLANTE | Service IA Agricole

Culture : [CULTURE]
Diagnostic : [DIAGNOSTIC]
Confiance : [XX]%

RÉSULTAT : [DESCRIPTION]

ACTIONS RECOMMANDÉES :
[ACTIONS]

Si doute, prenez une nouvelle photo.
Assistance : 0800-AGR-123 (gratuit)
```

**Filled example — maize_rust:**
```
DIAGNOSTIC PLANTE | Service IA Agricole

Culture : Maïs
Diagnostic : Rouille du Maïs (Puccinia sorghi)
Confiance : 91%

RÉSULTAT : Pustules orange-brun détectées sur les feuilles.
Maladie fongique — perte de rendement estimée 30–50% sans traitement.

ACTIONS RECOMMANDÉES :
1. Appliquer Mancozeb (2g/L) ou Propiconazole (1mL/L).
2. Traiter 2×/semaine le matin pendant 3 semaines.
3. Retirer et brûler les feuilles infectées.
4. Espacer les rangs à minimum 1m pour la prochaine saison.

Si doute, prenez une nouvelle photo.
Assistance : 0800-AGR-123 (gratuit)
```

---

### 3c. Disease → Action Lookup Table

| Disease | Kinyarwanda Name | Recommended Treatment | Dosage |
|---------|-----------------|----------------------|--------|
| healthy | Umutsima | Nta nzitizi — komeza isarura | — |
| maize_rust | Rustu y'ibigori | Mancozeb / Propiconazole | 2g/L, 2×/week |
| maize_blight | Ubwandu bw'ibigori | Azoxystrobin / Chlorothalonil | 1.5mL/L, 2×/week |
| cassava_mosaic | Indwara ya Mosaique | Kuramo imizi irwaye; gufata inzuki zibi | No chemical cure — rogue plants |
| bean_spot | Indwara y'ibishyimbo | Copper Oxychloride (2g/L) | 2g/L, weekly |

---

## 4. Unit Economics

### Cost per Diagnosis

| Item | Cost (USD) |
|------|-----------|
| Cloud inference (GPU-less CPU, 1 call) | $0.0002 |
| SMS gateway (Africa's Talking) | $0.0040 |
| Village Agent time (~3 min @ $0.50/hr) | $0.0250 |
| Amortised app dev / maintenance (per call, 100K/yr) | $0.0030 |
| **Total cost per diagnosis** | **~$0.032** |

### Value of a Saved Crop

| Crop | Avg plot size | Yield value | Loss without treatment | **Value saved** |
|------|--------------|------------|----------------------|----------------|
| Maize | 0.5 ha | $300 | 40% = $120 | **$120** |
| Cassava | 0.3 ha | $250 | 60% = $150 | **$150** |

### ROI for 1,000 Farmers

| Metric | Value |
|--------|-------|
| Total diagnosis cost | **$32** |
| Assume 40% of farmers have a disease | 400 farmers |
| Assume 70% act on advice successfully | 280 crops saved |
| Average value saved per crop | $130 |
| **Total value protected** | **$36,400** |
| **ROI** | **1,137×** |

> Even at 10× higher cost and 50% lower efficacy, the ROI remains **>100×**.

---

## 5. Low-Confidence Escalation Protocol

When model confidence < 65%:

1. **API sets** `low_confidence_alert: true` in the response.
2. **VA receives SMS**: *"Amakuru ntabwo arambuye. Fata ifoto nshya uvuye ibinyuranye."*
   (FR: *"Résultat incertain. Prenez une nouvelle photo sous un autre angle."*)
3. If second photo still < 65%: **escalate to Extension Officer** via a ticketed queue.
4. Extension Officer reviews both photos + model top-3 within 24 hours and sends
   a **confirmed SMS** to the farmer.

---

## 6. Offline / No-Signal Fallback (Structured USSD Triage)

When even the VA has no data signal, a **USSD menu** (works on any GSM network,
no data required) guides a symptom-based triage:

```
*384*1#  →  1. Ibigori / Maïs
             2. Manioke / Cassava
             3. Ibishyimbo / Haricots

(select 1 — Maïs)
→ Amababi afite:
   1. Amakoraniro y'umutuku/orange (rust)
   2. Ibisigazwa binini byijimye (blight)
   3. Ibara ryishaje (yellowing)

(select 1)
→ DIAGNOSIS: Rustu y'ibigori (Maize Rust)
   Koresha Mancozeb 2g/L.
   Ejo abaganga bazakubwira birambuye.
```

This rule-based fallback covers **~85% of common presentations** without any
internet connection.

---

*Document prepared for AIMS KTT Hackathon T2.1 · April 2025*