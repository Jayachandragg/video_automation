
#  EAF → Event-Level JSON Segmentation Pipeline

An end-to-end pipeline that converts **ELAN (.eaf) annotation files** into clean, **event-level JSON segments** using a combination of **rule-based merging** and **LLM-based semantic merging (CPU)**.

---

##  Overview

ELAN annotation files often contain **fine-grained, fragmented segments** that are difficult to use directly for downstream tasks.

This project solves that by:

* Extracting structured segments from `.eaf`
* Merging fragmented segments using **deterministic rules**
* Refining boundaries using a **lightweight LLM on CPU**
* Producing a clean, structured **event-level JSON output**

---

##  Pipeline Architecture

```
EAF File
  ↓
Stage 0: Segment Extraction
  ↓
Stage 1: Rule-Based Merge
  ↓
Stage 2: Consecutive Topic Merge
  ↓
LLM-Based Adjacency Merge (CPU)
  ↓
final.json
```

---

##  Key Features

*  Works directly with **ELAN (.eaf)** files
*  Combines **rule-based + LLM-based merging**
*  Handles **interruptions** (noise, silence, filler topics)
*  CPU-friendly (no GPU required)
*  Deployable via **Hugging Face Spaces**
*  Robust to noisy annotations

---

##  Input Format

### Required ELAN Tiers:

* **Topic Tier** → Contains activity/topic labels
* **Speaker Tier** → Contains transcript (e.g., mother/child speech)

---

##  Output Format (`final.json`)

```json
{
  "event_01": {
    "topic": "feeding",
    "start": 12.4,
    "end": 25.8,
    "transcript": "M: take the spoon C: okay..."
  }
}
```

### Output Characteristics:

* Event-level segmentation
* Continuous transcript per event
* Clean timestamps (in seconds)

---

##  Pipeline Stages

###  Stage 0: EAF → Segments

* Parses `.eaf` using `pympi`
* Extracts:

  * Topic label
  * Time boundaries
  * Overlapping speaker transcripts

---

###  Stage 1: Rule-Based Merge

* Merges segments if:

  * Same topic
  * Small time gap (≤ 0.4s)
* Handles:

  * Filler topics (`other`, `noise`)
  * Short interruptions via **tolerant merging**

---

###  Stage 2: Consecutive Merge

* Merges consecutive segments with the same topic
* Further reduces fragmentation

---

###  LLM Merge (CPU)

* Uses a lightweight model:

  * `Qwen/Qwen2.5-0.5B-Instruct`
* Decides:

  > Should two adjacent segments be merged into the same event?

#### Guardrails:

* Prevents over-merging
* Uses:

  * Gap constraints
  * Minimum transcript length
  * Fallback logic if parsing fails

---

##  Example Use Cases

* Behavioral analysis from annotated videos
* Child-parent interaction studies
* Video segmentation for ML pipelines
* Preprocessing for multimodal AI systems

---

##  Live Demo

 Try the app here:
**https://huggingface.co/spaces/chandu1083/Video_segmentation?logs=build**

Upload an `.eaf` file and download `final.json`.

---

## 🛠️ Installation (Local)

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo

pip install -r requirements.txt
```

Run locally:

```bash
python app.py
```

---

##  Dependencies

* `pympi-ling`
* `transformers`
* `torch`
* `gradio`

---

##  Model Details

* Model: `Qwen2.5-0.5B-Instruct`
* Runs entirely on CPU
* Deterministic output (temperature = 0)

---

##  Limitations

* Depends on annotation quality
* LLM decisions are local (adjacent segments only)
* CPU inference may be slower for large files

---



## Author

**Jayachandra Galda**
M.S. in Artificial Intelligence
University at Buffalo

