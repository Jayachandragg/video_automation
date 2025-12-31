import os
import re
import json
import tempfile
from typing import Tuple

import gradio as gr
import pympi
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# =========================
# CONFIG (CPU-Friendly)
# =========================
SKIP_TOPICS = {"other", "others"}

# Stage-1 merge rules
MAX_GAP_SEC_STAGE1 = 0.40
ENABLE_TOLERANT_MERGE = True
LOOKAHEAD_MAX_SEGMENTS = 4
LOOKAHEAD_MAX_SECONDS = 8.0
MAX_SINGLE_INTERRUPT_SEC = 2.0
MAX_TOTAL_INTERRUPT_SEC = 3.0

RESTRICT_INTERRUPT_TOPICS = False
ALLOWED_INTERRUPT_TOPICS = {"noise", "silence", "uh", "um", "laugh"}

USE_SEMICOLON_TOPIC_SETS = True
TRANSCRIPTION_FIELD = "mother_transcription"

# Stage-2 consecutive merge
MAX_GAP_SEC_STAGE2 = None  # None = ignore gap

# ✅ CPU LLM (small)
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"  # fast CPU. Optional: "Qwen/Qwen2.5-1.5B-Instruct"
LLM_MAX_NEW_TOKENS = 96
LLM_TEMPERATURE = 0.0

# Guardrails to prevent over-merging
LLM_MAX_GAP_SEC = 2.0
LLM_MIN_TEXT_CHARS = 12


# =========================
# Helpers
# =========================
def get_uploaded_path(f):
    """
    Gradio can pass:
    - a tempfile object with .name
    - a dict like {"name": "..."}
    - or a string path
    """
    if f is None:
        return None
    if isinstance(f, str):
        return f
    if isinstance(f, dict) and "name" in f:
        return f["name"]
    if hasattr(f, "name"):
        return f.name
    raise ValueError(f"Unsupported upload type: {type(f)}")

def norm(s: str) -> str:
    return (s or "").strip()

def norm_lower(s: str) -> str:
    return norm(s).lower()

def is_skip_topic(topic: str) -> bool:
    return norm_lower(topic) in {t.lower() for t in SKIP_TOPICS}

def ms_to_timestamp(ms: float) -> str:
    sec = ms / 1000.0
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:05.2f}"

def timestamp_to_sec(t: str) -> float:
    t = norm(t)
    parts = t.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return float(m) * 60 + float(s)
    return float(t)

def overlap(a_start, a_end, b_start, b_end) -> bool:
    return a_start < b_end and a_end > b_start

def normalize_prefix(text: str) -> str:
    if not text:
        return ""
    t = text.strip().replace("：", ":")
    t = re.sub(r"^(mother|mom|m)\s*:\s*", "M: ", t, flags=re.IGNORECASE)
    t = re.sub(r"^(child|kid|c)\s*:\s*", "C: ", t, flags=re.IGNORECASE)
    return t.strip()

def topic_set(topic: str) -> set:
    if not USE_SEMICOLON_TOPIC_SETS:
        return {norm_lower(topic)}
    return {p.strip().lower() for p in (topic or "").split(";") if p.strip()}

def same_topic(a: str, b: str) -> bool:
    if USE_SEMICOLON_TOPIC_SETS:
        return topic_set(a) == topic_set(b)
    return norm_lower(a) == norm_lower(b)

def seg_duration(seg_info: dict) -> float:
    return timestamp_to_sec(seg_info["time"]["end"]) - timestamp_to_sec(seg_info["time"]["start"])

def gap_between(seg_a: dict, seg_b: dict) -> float:
    return timestamp_to_sec(seg_b["time"]["start"]) - timestamp_to_sec(seg_a["time"]["end"])

def is_allowed_interrupt(topic: str) -> bool:
    if is_skip_topic(topic):
        return True
    if not RESTRICT_INTERRUPT_TOPICS:
        return True
    return norm_lower(topic) in {t.lower() for t in ALLOWED_INTERRUPT_TOPICS}


# =========================
# Stage 0: EAF -> segments JSON
# =========================
def eaf_to_segments_json(eaf_file: str, topic_tier: str, speaker_tier: str) -> dict:
    eaf = pympi.Elan.Eaf(eaf_file)
    video_id = os.path.splitext(os.path.basename(eaf_file))[0]

    topic_annotations = eaf.get_annotation_data_for_tier(topic_tier)
    speaker_annotations = eaf.get_annotation_data_for_tier(speaker_tier)

    segments = {}
    seg_count = 0

    for (start_ms, end_ms, label) in topic_annotations:
        topic = norm(label)

        texts = []
        for s_start, s_end, s_text in speaker_annotations:
            if overlap(s_start, s_end, start_ms, end_ms):
                t = normalize_prefix(s_text or "")
                if t:
                    texts.append(t)

        combined = " ".join(texts).strip()

        seg_count += 1
        seg_id = f"segment_{seg_count:02d}"
        segments[seg_id] = {
            "topic": topic,
            TRANSCRIPTION_FIELD: combined,
            "time": {"start": ms_to_timestamp(start_ms), "end": ms_to_timestamp(end_ms)},
            "with_object": "",
            "strategy_usage": {}
        }

    return {"video_id": video_id, "segments": segments}


# =========================
# Stage 1: rule merge
# =========================
def stage1_rule_merge(stage0: dict) -> dict:
    segments = stage0["segments"]
    seg_items = sorted(segments.items(), key=lambda x: int(x[0].split("_")[1]))

    merged_blocks = []
    i = 0

    while i < len(seg_items):
        seg_id, seg_info = seg_items[i]
        base_topic = seg_info["topic"]

        if is_skip_topic(base_topic) or not norm(base_topic):
            i += 1
            continue

        block_start = timestamp_to_sec(seg_info["time"]["start"])
        block_end = timestamp_to_sec(seg_info["time"]["end"])
        merged_from = [seg_id]

        text_pieces = []
        t0 = norm(seg_info.get(TRANSCRIPTION_FIELD, ""))
        if t0:
            text_pieces.append(t0)

        j = i + 1
        while j < len(seg_items):
            next_id, next_info = seg_items[j]
            next_topic = next_info["topic"]

            last_info = segments[merged_from[-1]]
            if gap_between(last_info, next_info) > MAX_GAP_SEC_STAGE1:
                break

            if is_skip_topic(next_topic):
                merged_from.append(next_id)
                block_end = max(block_end, timestamp_to_sec(next_info["time"]["end"]))
                tt = norm(next_info.get(TRANSCRIPTION_FIELD, ""))
                if tt:
                    text_pieces.append(tt)
                j += 1
                continue

            if same_topic(next_topic, base_topic):
                merged_from.append(next_id)
                block_end = max(block_end, timestamp_to_sec(next_info["time"]["end"]))
                tt = norm(next_info.get(TRANSCRIPTION_FIELD, ""))
                if tt:
                    text_pieces.append(tt)
                j += 1
                continue

            if not ENABLE_TOLERANT_MERGE:
                break

            # tolerant bridge
            interrupt_ids = []
            interrupt_total = 0.0
            interrupt_texts = []
            found_return = False

            lookahead_limit = min(len(seg_items), j + LOOKAHEAD_MAX_SEGMENTS + 1)
            anchor_time = timestamp_to_sec(next_info["time"]["start"])

            k = j
            prev_chain = segments[merged_from[-1]]

            while k < lookahead_limit:
                cand_id, cand_info = seg_items[k]
                cand_topic = cand_info["topic"]

                if (timestamp_to_sec(cand_info["time"]["start"]) - anchor_time) > LOOKAHEAD_MAX_SECONDS:
                    break
                if gap_between(prev_chain, cand_info) > MAX_GAP_SEC_STAGE1:
                    break

                if is_skip_topic(cand_topic):
                    interrupt_ids.append(cand_id)
                    tt = norm(cand_info.get(TRANSCRIPTION_FIELD, ""))
                    if tt:
                        interrupt_texts.append(tt)
                    prev_chain = cand_info
                    k += 1
                    continue

                if same_topic(cand_topic, base_topic):
                    found_return = True
                    return_id, return_info = cand_id, cand_info
                    break

                if not is_allowed_interrupt(cand_topic):
                    found_return = False
                    break

                d = seg_duration(cand_info)
                if d > MAX_SINGLE_INTERRUPT_SEC:
                    found_return = False
                    break

                interrupt_total += d
                if interrupt_total > MAX_TOTAL_INTERRUPT_SEC:
                    found_return = False
                    break

                interrupt_ids.append(cand_id)
                tt = norm(cand_info.get(TRANSCRIPTION_FIELD, ""))
                if tt:
                    interrupt_texts.append(tt)
                prev_chain = cand_info
                k += 1

            if found_return:
                merged_from.extend(interrupt_ids)
                text_pieces.extend(interrupt_texts)

                merged_from.append(return_id)
                block_end = max(block_end, timestamp_to_sec(return_info["time"]["end"]))
                tt = norm(return_info.get(TRANSCRIPTION_FIELD, ""))
                if tt:
                    text_pieces.append(tt)

                j = k + 1
                continue

            break

        merged_blocks.append({
            "topic": base_topic,
            TRANSCRIPTION_FIELD: " ".join(text_pieces).strip(),
            "time": {"start": ms_to_timestamp(block_start * 1000), "end": ms_to_timestamp(block_end * 1000)},
            "merged_from": merged_from,
            "with_object": "",
            "strategy_usage": {}
        })

        i = j

    out = {"video_id": stage0["video_id"], "merged_segments": {}}
    for idx, seg in enumerate(merged_blocks, start=1):
        out["merged_segments"][f"merged_segment_{idx:02d}"] = seg
    return out


# =========================
# Stage 2: consecutive merge
# =========================
def stage2_consecutive_merge(stage1: dict) -> dict:
    ms = stage1["merged_segments"]
    items = sorted(ms.items(), key=lambda x: int(x[0].split("_")[-1]))

    out_blocks = []
    cur = None

    for _, seg in items:
        topic = seg.get("topic", "")
        if not topic or is_skip_topic(topic):
            continue

        if cur is None:
            cur = dict(seg)
            cur[TRANSCRIPTION_FIELD] = norm(cur.get(TRANSCRIPTION_FIELD, ""))
            cur["merged_from"] = list(cur.get("merged_from", []))
            continue

        can_merge = same_topic(cur["topic"], topic)
        if can_merge and (MAX_GAP_SEC_STAGE2 is not None):
            if gap_between(cur, seg) > MAX_GAP_SEC_STAGE2:
                can_merge = False

        if can_merge:
            cur["time"]["end"] = seg["time"]["end"]
            tt = norm(seg.get(TRANSCRIPTION_FIELD, ""))
            if tt:
                cur[TRANSCRIPTION_FIELD] = (cur[TRANSCRIPTION_FIELD] + " " + tt).strip() if cur[TRANSCRIPTION_FIELD] else tt
            cur["merged_from"].extend(seg.get("merged_from", []))
        else:
            out_blocks.append(cur)
            cur = dict(seg)
            cur[TRANSCRIPTION_FIELD] = norm(cur.get(TRANSCRIPTION_FIELD, ""))
            cur["merged_from"] = list(cur.get("merged_from", []))

    if cur is not None:
        out_blocks.append(cur)

    out = {"video_id": stage1["video_id"], "merged_segments": {}}
    for idx, seg in enumerate(out_blocks, start=1):
        out["merged_segments"][f"merged_segment_{idx:02d}"] = seg
    return out


# =========================
# CPU LLM (cached)
# =========================
_PIPE = None

_PIPE = None

def load_llm_cpu_cached():
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    # ✅ No device_map => no accelerate needed
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32
    )
    model.to("cpu")
    model.eval()

    # ✅ device=-1 forces CPU in pipeline
    _PIPE = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
        temperature=LLM_TEMPERATURE,
        device=-1
    )
    return _PIPE



def safe_json_from_text(text: str):
    """
    Tries hard to extract a valid JSON object from model output.
    Returns dict or None.
    """
    if not text:
        return None

    # 1) Prefer fenced json blocks if present
    fence = re.findall(r"```json\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    candidates = fence if fence else []

    # 2) Otherwise collect brace objects (non-greedy)
    candidates += re.findall(r"\{[\s\S]*?\}", text)

    # Try from the last candidate backward (most likely the final answer)
    for cand in reversed(candidates):
        c = cand.strip()

        # Common cleanup
        c = c.replace("“", '"').replace("”", '"').replace("’", "'")
        c = re.sub(r",\s*}", "}", c)   # remove trailing comma
        c = re.sub(r",\s*]", "]", c)

        # keep only first {...} span if it contains extra text
        m = re.search(r"^\{[\s\S]*\}$", c)
        if not m:
            continue

        try:
            return json.loads(c)
        except:
            continue

    return None


def ask_llm_merge(pipe, prev_seg, curr_seg):
    prev_topic = norm_lower(prev_seg.get("topic", ""))
    curr_topic = norm_lower(curr_seg.get("topic", ""))
    gap = float(curr_seg["start"] - prev_seg["end"])

    a_txt = norm(prev_seg.get("transcript", ""))
    b_txt = norm(curr_seg.get("transcript", ""))

    # Guard 1: short transcripts => conservative
    if (len(a_txt) < LLM_MIN_TEXT_CHARS) or (len(b_txt) < LLM_MIN_TEXT_CHARS):
        if prev_topic == curr_topic and prev_topic:
            return True, "Short transcript but same topic"
        return False, "Transcript too short"

    # Guard 2: different topic + large gap => NO
    if prev_topic != curr_topic and gap > LLM_MAX_GAP_SEC:
        return False, "Different topic + large gap"

    prompt = f"""
You decide whether two consecutive segments should be merged into ONE event.

Be strict:
- merge=true ONLY if segment B continues the SAME toy/activity as segment A.
- merge=false if it changes toy/activity or starts a new event.
- If unsure, choose merge=false.

Return ONLY JSON, one object, nothing else:
{{"merge": false, "reason": "..."}}

Segment A
Topic: {prev_seg['topic']}
Transcript: {a_txt}

Segment B
Topic: {curr_seg['topic']}
Transcript: {b_txt}

Time gap (seconds): {gap:.2f}
"""

    out = pipe(prompt, do_sample=False, return_full_text=False)[0]["generated_text"].strip()

    res = safe_json_from_text(out)

    # ✅ Fallback if parsing fails (prevents crashing your Space)
    if res is None:
        # conservative fallback: only merge if exact topic match
        if prev_topic == curr_topic and prev_topic:
            return True, "Fallback: same topic (LLM JSON parse failed)"
        return False, "Fallback: LLM JSON parse failed"

    return bool(res.get("merge", False)), str(res.get("reason", "no reason"))

def llm_merge_stage(stage2: dict) -> dict:
    ms = stage2["merged_segments"]
    segs = []

    for _, v in ms.items():
        segs.append({
            "topic": v["topic"],
            "start": timestamp_to_sec(v["time"]["start"]),
            "end": timestamp_to_sec(v["time"]["end"]),
            "transcript": (v.get(TRANSCRIPTION_FIELD, "") or "").strip()
        })

    segs.sort(key=lambda x: x["start"])

    pipe = load_llm_cpu_cached()

    final = []
    if segs:
        cur = dict(segs[0])
        for i in range(1, len(segs)):
            nxt = segs[i]

            # Fast path: same topic
            if norm_lower(cur["topic"]) == norm_lower(nxt["topic"]):
                merge = True
            else:
                merge, _ = ask_llm_merge(pipe, cur, nxt)

            if merge:
                cur["end"] = max(cur["end"], nxt["end"])
                if nxt["transcript"]:
                    cur["transcript"] = (cur["transcript"] + " " + nxt["transcript"]).strip()
            else:
                final.append(cur)
                cur = dict(nxt)

        final.append(cur)

    # Final JSON format
    out = {}
    for i, s in enumerate(final, 1):
        out[f"event_{i:02d}"] = {
            "topic": s["topic"],
            "start": s["start"],
            "end": s["end"],
            "transcript": s["transcript"]
        }

    return out

# =========================
# Gradio function (JSON only)
# =========================
def run(eaf_file, topic_tier, speaker_tier):
    try:
        eaf_path = get_uploaded_path(eaf_file)
        if not eaf_path or not os.path.exists(eaf_path):
            raise gr.Error("Uploaded .eaf file path not found.")

        stage0 = eaf_to_segments_json(eaf_path, topic_tier, speaker_tier)
        stage1 = stage1_rule_merge(stage0)
        stage2 = stage2_consecutive_merge(stage1)

        final = llm_merge_stage(stage2)


        # Save final.json in a stable temp folder (not deleted before download)
        

        out_dir = tempfile.mkdtemp()
        final_path = os.path.join(out_dir, "final.json")
        
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2, ensure_ascii=False)
        
        return final_path


        

        return final_path

    except Exception as e:
        # Show the real error in the UI instead of a generic "Error"
        raise gr.Error(f"Failed: {e}")



demo = gr.Interface(
    fn=run,
    inputs=[
        gr.File(label="Upload ELAN (.eaf)", file_types=[".eaf"]),
        gr.Textbox(value="Topic", label="Topic Tier Name (must match EAF)"),
        gr.Textbox(value="Speaker", label="Speaker Tier Name (must match EAF)"),
    ],
    outputs=[gr.File(label="Download final.json")],
    title="EAF → Stage Merges → LLM Merge (CPU) → final.json",
    description=(
        "Uploads an ELAN .eaf file, extracts transcript from the Speaker tier, "
        "runs stage merges, then does LLM adjacency merge on CPU and outputs final.json."
    ),
)

if __name__ == "__main__":
    demo.launch()
