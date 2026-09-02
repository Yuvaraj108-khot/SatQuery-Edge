# 🛰️ SatQuery-Edge

**Offline Satellite Intelligence Investigation Workspace**
*SIH 26167 · ISRO · Prototype*

---

> **Natural-language query → Local Controller → Validation → Specialist Tools → Evidence Fusion → Geospatial Answer → Evidence Score → Observable Execution Trace → Downloadable PDF Report**

SatQuery-Edge is a fully offline, presentation-grade prototype that demonstrates a complete satellite image analysis pipeline without relying on any cloud APIs, internet connectivity, or external model downloads.

---

## Architecture at a Glance

```
 Query Input
     │
     ▼
 Intent Router         ← Keyword-based, deterministic
     │
     ▼
 Observation Validator  ← Image quality, count, pair checks
     │
     ▼
 ┌───────────────────────────────┐
 │    Specialist Tool Registry   │
 │  ┌────────┐  ┌───────────┐   │
 │  │  VQA   │  │ Grounding │   │
 │  ├────────┤  ├───────────┤   │
 │  │ Change │  │ Opt/SAR   │   │
 │  ├────────┤  └───────────┘   │
 │  │ GeoSp  │                  │
 │  └────────┘                  │
 └───────────────────────────────┘
     │
     ▼
 Evidence Fusion       ← M · S · P · T · I
     │
     ▼
 Evidence Score        ← E = 100 × Σ(w·v) / Σ(w)
     │
     ▼
 Final Answer + PDF Report
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Steps

**1. Clone / unzip the project**

```bash
# Already in your directory — no download needed
```

**2. Create a virtual environment**

```bash
python -m venv .venv
```

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

> All dependencies are pure-Python or pre-compiled wheels. No internet access required after installation.

**4. Run the application**

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## Quick Start — Demo Mode

1. Launch the app (`streamlit run app.py`)
2. In the sidebar, ensure **Demo Mode** is toggled **ON** (enabled by default)
3. Sample images load automatically — no upload required
4. The flagship query auto-populates:
   > *Identify newly flooded built-up areas and show the highest-confidence regions.*
5. Click **🚀 RUN INVESTIGATION**
6. Explore results across the four tabs: **Image Viewer**, **Map**, **Evidence**, **Execution Trace**
7. Download the PDF report

---

## 2-Minute Live Demo Script

### Step 1 — Launch

```bash
streamlit run app.py
```

Browser opens. Title: *SatQuery-Edge — Offline Satellite Intelligence Investigation Workspace*

### Step 2 — Demo Mode

Point to the sidebar toggle. "Demo Mode is ON. Pre-event and post-event synthetic satellite images are already loaded — no upload needed."

### Step 3 — Flagship Query

"The query is pre-populated: *Identify newly flooded built-up areas and show the highest-confidence regions.*"

### Step 4 — Run

Click **🚀 RUN INVESTIGATION**

### Step 5 — Router Decision

Point to the pipeline header: `QUERY ✓ ROUTER ✓ VALIDATOR ✓ ...`

"The router identified this as a **bi_temporal_change** task — because the query contains flood/temporal keywords and two images are available."

### Step 6 — Before/After Imagery

Navigate to **Image Viewer** tab.

"On the left: the pre-event scene — buildings, roads, farmland. On the right: the post-event scene — notice the blue-tinted regions where water has appeared over built-up areas."

### Step 7 — Change Overlay

"The change detection overlay highlights candidate flooded regions in color. The difference image (false-color) shows intensity of spectral change."

### Step 8 — Map

Navigate to **Map** tab.

"Detected regions are converted to approximate geographic coordinates and plotted on a dark map. Each polygon shows its confidence score in the tooltip."

### Step 9 — Evidence Score

Navigate to **Evidence** tab.

"This is the key differentiator. We don't blindly trust one detector. Five independent evidence dimensions are computed and fused:
- **M** — Specialist model agreement
- **S** — Sensor reliability
- **P** — Spatial consistency
- **T** — Temporal consistency
- **I** — Input quality"

Point to the large score display.

"Evidence Score: **91 / 100 — HIGH CONFIDENCE**."

### Step 10 — Execution Trace

Navigate to **Execution Trace** tab.

"Every operation is fully observable. You can see exactly what the router decided, what the validator found, which tools ran, what they returned, and how the final score was calculated."

### Step 11 — PDF Report

Click **⬇️ Download PDF Report**.

"The complete investigation — including query, answer, evidence breakdown, detected coordinates, and the execution trace — is exported as a structured PDF report."

---

## Presentation Script

> Use this script during SIH judging or live demonstrations.

---

**Opening:**

> "SatQuery-Edge is a fully offline satellite intelligence investigation system built for SIH 26167. Instead of sending satellite imagery to a cloud model, we run a transparent, observable pipeline entirely on the local machine — no API keys, no internet connection required."

---

**On the Router:**

> "The system doesn't guess what to do with the user's query. A deterministic intent router reads the natural-language input and maps it to a formal task graph — selecting the appropriate specialist tools. For this flood investigation, the router identifies a **bi-temporal change analysis** task."

---

**On Validation:**

> "Before any analysis begins, the observation validator checks that the correct number of images has been provided, that neither image is blank or corrupted, and that the temporal pair is geometrically compatible."

---

**On the Change Tool:**

> "The flagship specialist tool computes an absolute difference between pre-event and post-event imagery, applies denoising, thresholding, and morphological filtering, then extracts connected components as candidate changed regions. These are the areas where spectral properties changed significantly — consistent with new water inundation over built-up terrain."

---

**On Evidence Fusion:**

> "The important part is that we don't blindly trust one detector. Evidence from all active specialist tools is fused into five structured dimensions — model agreement, sensor reliability, spatial consistency, temporal consistency, and input quality — and combined with a weighted formula into a single Evidence Score."

---

**Showing the Score:**

> "For this investigation: **91 out of 100 — HIGH CONFIDENCE**. If the evidence were too weak, the system would explicitly abstain rather than produce a low-confidence answer."

---

**On the Execution Trace:**

> "Every step is logged with a timestamp. Judges, auditors, or operators can inspect exactly what the controller decided, how each tool responded, and how the final score was derived. Nothing is a black box."

---

**Closing:**

> "The entire investigation — including the query, geospatial polygons, evidence breakdown, and execution trace — is exported as a downloadable PDF report. SatQuery-Edge is a working demonstration of how trustworthy, transparent satellite intelligence can be delivered completely offline."

---

## Supported Intents

| Intent | Keywords | Images Required |
|---|---|---|
| `bi_temporal_change` | flood, change, before/after, damage, temporal | 2 |
| `optical_sar_fusion` | SAR, radar, fusion, combine, optical | 2 |
| `text_guided_grounding` | where, locate, find, highlight | 1 |
| `single_image_vqa` | what, describe, identify, how many | 1 |

---

## Evidence Score Formula

```
E = 100 × (w_m·M + w_s·S + w_p·P + w_t·T + w_i·I)
          / (w_m + w_s + w_p + w_t + w_i)

Weights:
  w_m = 0.30  (Model Agreement)
  w_s = 0.20  (Sensor Reliability)
  w_p = 0.20  (Spatial Consistency)
  w_t = 0.15  (Temporal Consistency)
  w_i = 0.15  (Input Quality)

Abstention threshold: E < 45
```

---

## Offline Guarantee

- ✅ No OpenAI / cloud APIs used
- ✅ No HuggingFace inference APIs
- ✅ No HTTP requests at runtime
- ✅ No model downloads required
- ✅ All processing: local NumPy, OpenCV, Pillow, ReportLab
- ✅ Sample images generated programmatically

---

## Project Structure

```
satquery-edge/
├── app.py                          ← Streamlit entry point
├── requirements.txt
├── README.md
├── controller/
│   ├── router.py                   ← Intent router
│   ├── schemas.py                  ← Pydantic v2 schemas
│   ├── registry.py                 ← Tool registry
│   └── validator.py                ← Input validator
├── tools/
│   ├── vqa.py                      ← Visual Q&A (CV heuristics)
│   ├── grounding.py                ← Text-guided grounding
│   ├── change.py                   ← Bi-temporal change detection
│   ├── optical_sar.py              ← Optical/SAR fusion
│   └── geospatial.py               ← Pixel → lat/lon conversion
├── evidence/
│   ├── fusion.py                   ← M·S·P·T·I computation
│   └── score.py                    ← Evidence formula + abstention
├── ui/
│   ├── components.py               ← Streamlit UI components
│   └── map_utils.py                ← Folium map builder
├── data/
│   └── sample/
│       ├── generate_samples.py     ← Synthetic image generator
│       ├── pre_event.png           ← Auto-generated
│       ├── post_event.png          ← Auto-generated
│       └── sar_like.png            ← Auto-generated
└── reports/
    └── report_generator.py         ← ReportLab PDF generator
```

---

## Disclaimer

> This prototype uses lightweight local computer vision and synthetic/demo geospatial observations where applicable. Results are intended for demonstration purposes and are not scientifically validated remote-sensing products. No cloud APIs, external inference services, or internet connectivity is required or used.