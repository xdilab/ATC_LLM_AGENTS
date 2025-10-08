# ===============================
# XAION_CONTROL_ST_V56.py — Part 1/3
# Foundations: config, models, helpers, parsing, scenarios
# ===============================

import os, gc, re, ast, math, time, socket, tempfile, uuid, wave, random
from collections import deque

import numpy as np
import torch
import pandas as pd
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------
# CONFIG (prefer XAION_config; fallback to env)
# -------------------------------
OPENAI_API_KEY = None
USE_WHISPER_API_RAW = None
ELEVEN_API_KEY = None
ELEVEN_VOICE_ID_DEFAULT = None
PILOT_VOICE_ID = None
AUDIO_GAP_SEC = 1.0

try:
    from XAION_config import (
        OPENAI_API_KEY as _C_OPENAI_API_KEY,
        USE_WHISPER_API as _C_USE_WHISPER_API,
        ELEVEN_API_KEY as _C_ELEVEN_API_KEY,
        ELEVEN_VOICE_ID as _C_ELEVEN_VOICE_ID,
        PILOT_VOICE_ID as _C_PILOT_VOICE_ID,
        AUDIO_GAP_SEC as _C_AUDIO_GAP_SEC,
    )
    OPENAI_API_KEY = _C_OPENAI_API_KEY
    USE_WHISPER_API_RAW = _C_USE_WHISPER_API
    ELEVEN_API_KEY = _C_ELEVEN_API_KEY
    ELEVEN_VOICE_ID_DEFAULT = _C_ELEVEN_VOICE_ID
    PILOT_VOICE_ID = _C_PILOT_VOICE_ID
    AUDIO_GAP_SEC = _C_AUDIO_GAP_SEC
except Exception:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    USE_WHISPER_API_RAW = os.getenv("USE_WHISPER_API", "1")
    ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY", "").strip()
    ELEVEN_VOICE_ID_DEFAULT = os.getenv("ELEVEN_VOICE_ID", "cNYrMw9glwJZXR8RwbuR").strip()
    PILOT_VOICE_ID = os.getenv("PILOT_VOICE_ID", "s5TajbcxRcHxifBxVr3H").strip()
    AUDIO_GAP_SEC = float(os.getenv("AUDIO_GAP_SEC", "1.0"))

USE_WHISPER_API = str(USE_WHISPER_API_RAW).strip().lower() not in ("0", "false", "no", "")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# -------------------------------
# CONSTANTS / PATHS
# -------------------------------
PHI4_MODEL_PATH    = "./phi4_atc_response_model_v1"
SNAPSHOT_PATH      = "snapshot_final_24hr(in).csv"
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base.en")

ATC_INITIAL_PROMPT = (
    "Aviation radio phraseology, U.S. English. Use these terms:\n"
    "Greensboro Ground, Greensboro Approach, Tower, runway, Runway 05, Runway 23, "
    "taxi, via, hold short, cross, request taxi, for departure, "
    "ILS, ILS runway, localizer, glideslope, intercept, established, on final, mile final, "
    "cleared ILS runway 05, cleared to land, maintain, descend and maintain, "
    "heading, turn left heading, speed, knots, kts, feet, degrees, nm.\n"
    "Prefer numerals and standard ATC call signs."
)

KGSO_RUNWAYS = {"05": (30, 60), "23": (210, 240), "14": (140, 170), "32": (320, 350)}
DEFAULT_RWY_PRIMARY = "05"
KGSO_TOWER_FREQ = "119.1"

# -------------------------------
# Gates (diagram-based, simplified groupings)
# -------------------------------
KGSO_GATES_WEST  = list(range(1, 40))   # 1–39
KGSO_GATES_EAST  = list(range(40, 50))  # 40–49
ALL_GATES = set(KGSO_GATES_WEST + KGSO_GATES_EAST)

# Taxiway macros for plausible routes by side/runway
TAXI_ROUTES = {
    ("to_runway", "05", "east_concourse"):  ["via A, A2, hold short Runway 05"],
    ("to_runway", "05", "west_concourse"):  ["via B, B3, cross A at A4, hold short Runway 05"],
    ("to_runway", "23", "east_concourse"):  ["via A, A5, hold short Runway 23"],
    ("to_runway", "23", "west_concourse"):  ["via B, B2, hold short Runway 23"],
    ("to_runway", "14", "east_concourse"):  ["via A, A1, hold short Runway 14"],
    ("to_runway", "14", "west_concourse"):  ["via B, B1, hold short Runway 14"],
    ("to_runway", "32", "east_concourse"):  ["via A, A3, hold short Runway 32"],
    ("to_runway", "32", "west_concourse"):  ["via B, B5, hold short Runway 32"],
    ("to_gate", "east_concourse"): ["exit via A, then A eastbound"],
    ("to_gate", "west_concourse"): ["exit via B, then B westbound"],
}

# Role emojis
EMOJI_PILOT = "🧑‍✈️"
EMOJI_ATC   = "🗼"

# Acknowledgement variants
PILOT_ACK_VARIANTS = [
    "Roger, {cs}.", "Wilco, {cs}.", "Copy, {cs}.", "Affirmative, {cs}.",
    "{cs}, copy.", "{cs}, roger.", "Understood, {cs}."
]

# -------------------------------
# DEBUG BUFFER (UI + terminal)
# -------------------------------
DEBUG_LOG = []
def debug_line(s: str):
    print(s, flush=True)
    DEBUG_LOG.append(s)
    if len(DEBUG_LOG) > 1600:
        del DEBUG_LOG[:800]

def debug_block(title: str):
    line = f"===== {title} ====="
    debug_line(line)
    return line

def snapshot_debug_dump():
    return "\n".join(DEBUG_LOG[-600:])

# Rich monitor-style blocks (exactly like your screenshot)
def monitor_line(s: str):
    debug_line(s)

def monitor_header(title: str):
    monitor_line(f"======== {title} ========")

def monitor_block(role: str, phase: str, runway: str):
    monitor_line("----- GENERATION CALL -----")
    monitor_line(f"[ROLE] {role} | [PHASE] {phase} | [RWY] {runway}")

def _ctx_dump_lines(ctx_line_now: str, n_recent: int = 3, dialogue_text: str | None = None):
    """
    Write a monitor-style block of (context frame + recent frames + dialogue).
    If dialogue_text is provided, use it; otherwise use global VOICE_DIALOGUE.
    """
    monitor_line("[Context Frame] " + ctx_line_now)
    monitor_line("[Recent Context Frames]")
    rc = recent_context_block(n_recent)
    for ln in (rc.split("\n") if rc else ["(none)"]):
        monitor_line(ln)
    monitor_line("[Dialogue]")
    d = ((dialogue_text if dialogue_text is not None else (VOICE_DIALOGUE or ""))).strip() or "(empty)"
    for ln in d.split("\n"):
        monitor_line(ln)
    monitor_line("-" * 28)

# -------------------------------
# SETUP
# -------------------------------
os.environ["TORCH_ENABLE_FALLBACK_MULTI_TENSOR_REDUCE"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gc.collect()
try: torch.cuda.empty_cache()
except Exception: pass

monitor_header("XAION INIT")

# Snapshot
try:
    snapshot_df = pd.read_csv(SNAPSHOT_PATH)
    debug_line(f"[INIT] Loaded snapshot: {SNAPSHOT_PATH} ({len(snapshot_df)} rows)")
except Exception as e:
    debug_line(f"[INIT] Snapshot load failed: {e}")
    snapshot_df = pd.DataFrame()

if "DT Time" in snapshot_df.columns:
    snapshot_df["__dt"] = pd.to_datetime(
        snapshot_df["DT Time"].astype(str).str.strip(),
        format="%H%M%SZ",
        errors="coerce"
    )
else:
    if not snapshot_df.empty:
        snapshot_df["DT Time"] = ""
        snapshot_df["__dt"] = pd.NaT

# -------------------------------
# MODELS
# -------------------------------
phi4_tokenizer = AutoTokenizer.from_pretrained(PHI4_MODEL_PATH)
phi4_model = AutoModelForCausalLM.from_pretrained(
    PHI4_MODEL_PATH,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    low_cpu_mem_usage=True,
).eval()
debug_line("[INIT] Phi-4 response model loaded.")

# Whisper setup
whisper_model = None
openai_client = None
if USE_WHISPER_API:
    try:
        from openai import OpenAI
        openai_client = OpenAI()
        debug_line(f"[Whisper API] Using OpenAI Whisper API (key present: {'yes' if OPENAI_API_KEY else 'no'})")
    except Exception as e:
        debug_line(f"[Whisper API] init error: {e} — falling back to local Whisper.")
        USE_WHISPER_API = False

if not USE_WHISPER_API:
    try:
        import whisper as _w
        whisper_model = _w.load_model(WHISPER_MODEL_SIZE, device=str(device))
        debug_line("[Whisper] Local model loaded.")
    except Exception as e:
        debug_line(f"[Whisper] Local model load failed: {e}")

# TTS (ElevenLabs)
_has_eleven = False
try:
    if ELEVEN_API_KEY:
        from elevenlabs import ElevenLabs
        eleven_client = ElevenLabs(api_key=ELEVEN_API_KEY)
        _has_eleven = True
        debug_line("[TTS] ElevenLabs ready.")
    else:
        debug_line("[TTS] ELEVEN_API_KEY not set; voice output disabled.")
except Exception as _e:
    debug_line(f"[TTS] ElevenLabs unavailable: {_e}")

# Mutagen duration helper
def _audio_duration_sec(path: str) -> float | None:
    try:
        from mutagen import File as MFile
        mf = MFile(path)
        if mf and getattr(mf, "info", None):
            return float(getattr(mf.info, "length", 0.0)) or None
    except Exception:
        return None
    return None

# -------------------------------
# Helpers / parsing / gates
# -------------------------------
# ===== NO-PROMPT-BLEED GENERATION & SANITIZERS =====
def _hf_generate_new(prompt: str,
                     *,
                     max_new_tokens: int = 64,
                     temperature: float = 0.6,
                     top_p: float = 0.9) -> str:
    """Return ONLY newly generated text (no prompt echo)."""
    tok = phi4_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)
    gen = phi4_model.generate(
        tok.input_ids,
        attention_mask=tok.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.08,
        no_repeat_ngram_size=3,
        pad_token_id=phi4_tokenizer.eos_token_id,
        eos_token_id=phi4_tokenizer.eos_token_id,
    )
    # Slice off the prompt; decode ONLY the continuation
    new_ids = gen[0, tok.input_ids.shape[1]:]
    return phi4_tokenizer.decode(new_ids, skip_special_tokens=True)

def _strip_prompt_bleed(text: str) -> str:
    """Remove any leaked instruction/prompt scaffolding."""
    t = (text or "").strip()

    # Nuke common scaffold chunks if they leaked
    t = re.sub(r"(?is)\bYou\s+are\s+the\s+Pilot\b.*$", "", t)                      # instruction preamble
    t = re.sub(r"(?is)\bOutput\s+format\s*\(strict\)\s*:\s*.*$", "", t)            # "Output format (strict):"
    t = re.sub(r"(?is)\bATC_CORE\b.*?\]", "", t)                                   # bracketed core
    t = re.sub(r"(?is)===.*?===", "", t)                                           # divider lines
    t = re.sub(r"(?is)\[.*?\]", "", t)                                            # any stray [blocks]
    t = re.sub(r"\s{2,}", " ", t).strip(" \n\r\t-•:")                              # whitespace/punct

    # Keep to first sentence for radio brevity
    t = _first_sentence(t)
    return t

def _dedup_leading_callsigns(cs: str, s: str) -> str:
    """Collapse 'CS, CS, CS,' → 'CS,' once at the start of a line."""
    if not cs: return s
    return re.sub(rf"^\s*(?:{re.escape(cs)}\s*,\s*)+", f"{cs}, ", s, flags=re.I)


def canonicalize_pilot_ui(text: str, cs: str) -> str:
    """
    Force the pilot line to: 'Pilot: CS, <content>.'
    Removes any leading CS or stray 'Pilot:' tokens to prevent
    artifacts like 'CS, Pilot: CS, ...'.
    """
    t = (text or "").strip()

    # Remove leading role tag and emoji
    t = re.sub(r"^\s*(?:🧑‍✈️\s*)?Pilot:\s*", "", t, flags=re.I)

    # Remove 'CS, Pilot:' and duplicate leading callsigns
    t = re.sub(rf"^\s*(?:{re.escape(cs)}\s*,\s*)+Pilot:\s*", "", t, flags=re.I)
    t = re.sub(rf"^\s*(?:{re.escape(cs)}\s*,\s*)+", "", t, flags=re.I)

    # Kill any remaining 'Pilot:' anywhere inside
    t = re.sub(r"\bPilot:\s*", "", t, flags=re.I)

    # Clean punctuation / spacing
    t = t.strip(" ,.;")

    # Ensure canonical header + final punctuation
    return f"Pilot: {cs}, {t}."



def _gate_audio_playback(audio_path: str | None, spoken_text: str | None):
    """Sleep just long enough that subsequent autoplay clips don't overlap."""
    dur = _estimate_audio_length_sec(spoken_text or "", audio_path)
    time.sleep(dur + max(0.0, AUDIO_GAP_SEC))


def strip_role_prefix(line: str) -> str:
    """Remove any leading 'Pilot:' or 'ATC:' (with or without emoji) from a single-line string."""
    return re.sub(r"^\s*(?:🧑‍✈️\s*)?(?:Pilot|ATC)\b[:,-]?\s*", "", line or "", flags=re.I).strip()

def _estimate_audio_length_sec(text: str, path: str | None) -> float:
    """Prefer real file duration; otherwise estimate by words/sec."""
    d = _audio_duration_sec(path) if path else None
    if d and d > 0.2:
        return d
    # ~2.4 words/sec conservative, min 1.2s
    w = max(1, len(re.findall(r"\w+", text or "")))
    return max(1.2, w / 2.4)


def strip_leading_callsign(text: str, cs: str) -> str:
    t = text or ""
    # drop any leading “CSB421,” or duplicates “CSB421, CSB421,”
    t = re.sub(rf'^\s*(?:{re.escape(cs)}\s*,\s*)+', '', t, flags=re.I)
    return t.strip()

def dedup_callsign_runs(line: str, cs: str, role: str) -> str:
    """Collapse 'CS, CS, ...' to a single 'CS,' and ensure proper role preface."""
    if not line:
        return line
    # remove role tag when cleaning
    core = re.sub(r'^\s*(?:🗼\s*)?ATC:\s*|\s*(?:🧑‍✈️\s*)?Pilot:\s*', '', line, flags=re.I)
    # one run only
    core = re.sub(rf'^\s*(?:{re.escape(cs)}\s*,\s*)+', f'{cs}, ', core, flags=re.I)
    # avoid 'This is XAION CONTROL' duplication in the core
    core = re.sub(r'(?i)\bThis\s+is\s+XAION\s+CONTROL\.?\s*', '', core).strip()
    # rebuild with role label if needed
    if role.upper() == "ATC":
        return f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}{core}"
    else:
        # pilot lines always start 'Pilot: CS, ...'
        if not re.match(rf'^\s*Pilot\s*:', line, flags=re.I):
            core = f"Pilot: {cs}, {core.lstrip(', ').strip()}"
        else:
            # make sure it has a leading callsign exactly once
            core = re.sub(rf'^\s*Pilot\s*:\s*(?:{re.escape(cs)}\s*,\s*)+', f"Pilot: {cs}, ", core, flags=re.I)
        return f"{EMOJI_PILOT} {core}"

def _sleep_for_audio(path: str | None):
    """
    Blocks until this clip would have finished + a small gap.
    If we can't read the length, assume ~2.2s.
    """
    if not path:
        time.sleep(max(0.0, AUDIO_GAP_SEC))
        return
    dur = _audio_duration_sec(path) or 2.2
    time.sleep(dur + max(0.0, AUDIO_GAP_SEC))


# === Globals for Vocal Input readback tracking ===
VOICE_LAST_ATC_STRUCT = None  # populated after each ATC line in voice mode

# === Repeat-detection (covers "say again", "repeat", etc.) ===
REPEAT_RX = re.compile(r'\b(say\s+again|repeat|last\s+transmission|please\s+repeat|confirm)\b', re.I)
# === Voice flow regex (module-scope so static analyzers see them) ===
READY_RX               = re.compile(r"\b(holding\s+short\s+runway\s*(\d{2}[LRC]?)\b.*\bready\b|ready\s+(?:for\s+)?(?:departure|takeoff))", re.I)
CONTACT_TWR_RX         = re.compile(r"\b(?:contact(?:ing)?|switch(?:ing)?)\s*(?:tower|\b119\.1\b)\b", re.I)
REQ_TAXI_RX            = re.compile(r"\brequest\s+taxi\s+to\s+runway\s*(\d{1,2}[LRC]?)\b", re.I)
LINEUP_RX              = re.compile(r"\b(?:line\s*up(?:\s*and\s*wait)?|luaw)\b.*\brunway\s*(\d{2}[LRC]?)\b", re.I)
PILOT_ASSERT_TKOF_RX   = re.compile(r"\bcleared\s+for\s+takeoff\b", re.I)
# NEW: detect when the pilot is talking to tower and says ready / holding short
TOWER_READY_RX = re.compile(r"\btower\b.*\b(ready|holding\s+short)\b", re.I)

INBOUND_RX             = re.compile(r"\binbound(?:\s+runway\s*(\d{2}[LRC]?))?\b", re.I)
ESTABLISHED_RX         = re.compile(r"\b(?:established\b|on\s+(?:localizer|glideslope|final)|short\s+final)\b", re.I)
CLEARED_LAND_ASSERT_RX = re.compile(r"\bcleared\s+to\s+land\b", re.I)


# === Canonical ATC-line helpers used everywhere (single definitions) ===
def strip_atc_ui_to_core(atc_ui_line: str, cs: str) -> str:
    """From a UI ATC line, return ONLY the instruction core: no role tag, no brand, no leading callsign(s)."""
    t = atc_ui_line or ""
    t = re.sub(r'^\s*(?:🗼\s*)?ATC:\s*', '', t)  # remove "ATC: "
    t = re.sub(r'(?i)\bThis\s+is\s+XAION\s+CONTROL\.?\s*', '', t)  # remove brand line
    # Remove any *run* of leading callsigns like "CS, CS, CS,"
    t = re.sub(rf'^\s*(?:{re.escape(cs)}\s*,\s*)+', '', t, flags=re.I)
    return t.strip()

def atc_prefix_and_dedup(cs: str, core: str) -> str:
    """Prefix brand once and ensure NO duplicate callsigns at the start."""
    core = (core or "").strip()
    core = re.sub(r'(?i)\bThis\s+is\s+XAION\s+CONTROL\.?\s*', '', core)
    core = re.sub(rf'^\s*(?:{re.escape(cs)}\s*,\s*)+', '', core, flags=re.I)  # collapse "CS, CS," to none
    core = core.strip()
    if re.match(rf'^{re.escape(cs)}\s*,', core, flags=re.I):
        core = re.sub(rf'^{re.escape(cs)}\s*,\s*', '', core, flags=re.I)
    return f"{cs}, This is XAION CONTROL. {core}"

# --- Acknowledgement helper (single definition; used by ATC “readback correct” path) ---
try:
    pilot_ack  # type: ignore[name-defined]
except NameError:
    def pilot_ack(cs: str) -> str:
        return f"Roger, {cs}."

def _ack_line(cs: str) -> str:
    try:
        return pilot_ack(cs)
    except Exception:
        return f"Roger, {cs}."

_BAD_CS = {"", "NONE", "UNKNOWN", "NULL", "nan", "None"}

def _safe_callsign_from_row(r) -> str:
    for k in ("display_id", "ident", "Flight Number", "Callsign", "Call Sign", "Aircraft Registration", "Hex", "hex"):
        v = (r.get(k) if isinstance(r, dict) else getattr(r, k, None))
        if v is None: continue
        s = str(v).strip()
        if s and s.upper() not in _BAD_CS:
            return s
    return ""

def safe_literal_eval(obj_str):
    try:
        txt = str(obj_str).replace("nan", "None")
        return ast.literal_eval(txt)
    except Exception:
        return None

def num(x, default=None):
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return default
        return float(str(x).strip().split()[0])
    except Exception:
        return default

def runway_from_heading(track):
    if track is None or (isinstance(track, float) and math.isnan(track)):
        return None
    try:
        h = float(track) % 360
    except Exception:
        return None
    for rw, (lo, hi) in KGSO_RUNWAYS.items():
        if lo <= h <= hi:
            return rw
    return None

def col_series(df, col):
    return df[col].astype(str).str.strip() if col in df.columns else pd.Series([""] * len(df), index=df.index)

# degree heading default for runway
def _hdg_to_runway(rw: str) -> int:
    if not rw: return 50
    base = {"05": 50, "23": 230, "14": 140, "32": 320}
    return base.get(rw.replace("L","").replace("R","").replace("C",""), 50)

# --- Gate parsing from transcript ---
GATE_RX = re.compile(r"\b(?:gate|g)\s*(\d{1,3})\b", re.I)

def _gate_from_text(text: str) -> str | None:
    m = GATE_RX.search(text or "")
    if not m:
        return None
    gate_num = int(m.group(1))
    if gate_num in ALL_GATES:
        return f"Gate {gate_num}"
    return None

def _side_for_gate(gate: str | None) -> str:
    if not gate:
        return "east_concourse"
    try:
        n = int(re.search(r"\d{1,3}", gate).group(0))
    except Exception:
        return "east_concourse"
    return "east_concourse" if n >= 40 else "west_concourse"

def _side_for_callsign(cs: str) -> str:
    try:
        h = sum(ord(c) for c in (cs or ""))
        return "east_concourse" if (h % 2 == 0) else "west_concourse"
    except Exception:
        return "east_concourse"

def pick_gate_for_callsign(cs: str) -> str:
    side = _side_for_callsign(cs)
    pool = KGSO_GATES_EAST if side == "east_concourse" else KGSO_GATES_WEST
    if not pool: return "Gate 41"
    idx = (sum(ord(c) for c in (cs or "")) % len(pool))
    return f"Gate {pool[idx]}"

def _compose_taxi_to_runway(callsign: str, runway: str, gate: str | None) -> str:
    side = _side_for_gate(gate) if gate else _side_for_callsign(callsign)
    segs = TAXI_ROUTES.get(("to_runway", runway, side))
    if not segs:
        return f"{callsign}, taxi to Runway {runway}, hold short Runway {runway}."
    body = ", ".join(segs)
    return f"{callsign}, taxi to Runway {runway} {body}."

def _compose_taxi_to_gate(callsign: str, gate: str | None) -> str:
    side = _side_for_gate(gate) if gate else _side_for_callsign(callsign)
    segs = TAXI_ROUTES.get(("to_gate", side))
    if not segs:
        return f"{callsign}, taxi to {gate or 'the gate area'}."
    body = ", ".join(segs)
    if gate:
        return f"{callsign}, {body}, proceed to {gate}."
    return f"{callsign}, {body}, proceed to gate area."

# -------------------------------
# Flatten snapshot
# -------------------------------
def build_flat_df(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    if snapshot_df.empty:
        return pd.DataFrame([{"DT Time": "", "parent_dt": "", "__dt": pd.NaT}])

    flat_records = []
    for _, row in snapshot_df.iterrows():
        dt_str = row.get("DT Time", "")
        dt_parsed = row.get("__dt", pd.NaT)

        obj = safe_literal_eval(row.get("Aircraft(Obj)", None))
        if obj is None:
            flat_records.append({
                "DT Time": dt_str,
                "parent_dt": dt_str,
                "__dt": dt_parsed,
                "Distance to GSO": row.get("Distance to GSO", None),
                "Latitude": row.get("Latitude", None),
                "Longitude": row.get("Longitude", None),
                "Airspace Status": row.get("Airspace Status", None),
            })
            continue

        objs = obj if isinstance(obj, list) else [obj]
        for a in objs:
            rec = dict(a) if isinstance(a, dict) else {}
            rec["DT Time"] = dt_str
            rec["parent_dt"] = dt_str
            rec["__dt"] = dt_parsed
            for k in ("Distance to GSO","Latitude","Longitude","Airspace Status"):
                if k not in rec and k in row:
                    rec[k] = row[k]
            flat_records.append(rec)

    flat_df = pd.DataFrame(flat_records)
    if flat_df.empty:
        flat_df = pd.DataFrame([{"DT Time": "", "parent_dt": "", "__dt": pd.NaT}])

    flat_df["dist_nm"]     = flat_df.get("Distance to GSO", pd.Series([None]*len(flat_df))).apply(lambda v: num(v, None))
    flat_df["alt_geom_ft"] = flat_df.get("Altitude Geometric", pd.Series([None]*len(flat_df))).apply(lambda v: num(v, 0.0))
    flat_df["gs_kts"]      = flat_df.get("Ground Speed", pd.Series([None]*len(flat_df))).apply(lambda v: num(v, 0.0))
    flat_df["track_deg"]   = flat_df.get("Track", pd.Series([None]*len(flat_df))).apply(lambda v: num(v, None))
    flat_df["on_ground"]   = flat_df.apply(
        lambda r: ("ground" in str(r.get("Airspace Status", "")).lower())
                  or (str(r.get("Altitude Barometric", "")).lower() == "ground"),
        axis=1
    )

    ident = col_series(flat_df, "Flight Number")
    ident = ident.where(ident != "", col_series(flat_df, "Callsign"))
    ident = ident.where(ident != "", col_series(flat_df, "Call Sign"))
    ident = ident.where(ident != "", col_series(flat_df, "Aircraft Registration"))
    ident = ident.where(ident != "", col_series(flat_df, "Hex"))
    ident = ident.where(ident != "", col_series(flat_df, "hex"))
    flat_df["ident"] = ident.where(ident != "", "UNKNOWN")

    disp = col_series(flat_df, "Flight Number")
    disp = disp.where(disp != "", col_series(flat_df, "Hex"))
    flat_df["display_id"] = disp.where(disp != "", "UNKNOWN")

    flat_df = flat_df.sort_values(by="__dt", kind="stable").reset_index(drop=True)
    debug_line(f"[INIT] Flattened snapshot records: {len(flat_df)}")
    return flat_df

flat_df = build_flat_df(snapshot_df)

# -------------------------------
# Scenario options built from flat_df
# -------------------------------
def build_scenarios_from_flat(flat_df: pd.DataFrame):
    def _good_callsign(cs: str) -> bool:
        bad = {"", "NONE", "UNKNOWN", "NULL"}
        return bool(cs) and cs.upper() not in bad

    takeoff_rows, landing_rows = [], []
    for ident_key, grp in flat_df.groupby("ident", sort=False):
        if not _good_callsign(ident_key):
            continue
        g = grp.copy()
        if g["__dt"].isna().all():
            continue
        g = g.sort_values("__dt")
        ground_idx = g[g["on_ground"]].index
        air_idx    = g[~g["on_ground"]].index
        if len(ground_idx) and len(air_idx):
            g0 = ground_idx.min()
            if (g.loc[air_idx, "__dt"] > g.loc[g0, "__dt"]).any():
                takeoff_rows.append(int(g0))
        if len(air_idx) and len(ground_idx):
            a0 = air_idx.min()
            if (g.loc[ground_idx, "__dt"] > g.loc[a0, "__dt"]).any():
                landing_rows.append(int(a0))

    def _row_ok_for_takeoff(r):
        cs = (r.get("display_id") or r.get("ident") or "").strip()
        if not _good_callsign(cs): return False
        spd_ok = pd.notna(r.get("gs_kts")) and float(r.get("gs_kts")) > 0
        dist_ok = pd.notna(r.get("dist_nm")) and float(r.get("dist_nm")) > 0
        return bool(r.get("on_ground", False)) and spd_ok and dist_ok

    def _row_ok_for_landing(r):
        cs = (r.get("display_id") or r.get("ident") or "").strip()
        if not _good_callsign(cs): return False
        dist_ok = pd.notna(r.get("dist_nm")) and float(r.get("dist_nm")) > 0
        return (not r.get("on_ground", False)) and dist_ok

    takeoff_rows = [i for i in takeoff_rows if _row_ok_for_takeoff(flat_df.loc[i])]
    landing_rows = [i for i in landing_rows if _row_ok_for_landing(flat_df.loc[i])]

    def fmt_takeoff_row(r):
        cs = (r.get("display_id") or r.get("ident")).strip()
        spd = f"{int(r['gs_kts'])} knots"
        dist = f"{r['dist_nm']:.1f} nautical miles"
        rw = runway_from_heading(r.get("track_deg")) or DEFAULT_RWY_PRIMARY
        gate = pick_gate_for_callsign(cs)
        return f"Greensboro Ground, {cs}, at {gate}, {spd}, {dist} from field, request taxi to Runway {rw} for departure."

    def fmt_landing_row(r):
        cs = (r.get("display_id") or r.get("ident")).strip()
        alt = f"{int(r['alt_geom_ft'])} feet" if r.get("alt_geom_ft") else "altitude unknown"
        dist = f"{r['dist_nm']:.1f} nautical miles"
        hdg = f"{int(r['track_deg'])} degrees" if pd.notna(r["track_deg"]) else "heading unknown"
        rw = runway_from_heading(r.get("track_deg"))
        rw_txt = f", inbound Runway {rw}" if rw else ""
        return f"Greensboro Approach, {cs}, {alt}, {dist} from GSO, {hdg}{rw_txt}."

    TAKEOFF_SCENARIOS = [fmt_takeoff_row(flat_df.loc[i]) for i in takeoff_rows]
    LANDING_SCENARIOS = [fmt_landing_row(flat_df.loc[i]) for i in landing_rows]
    DT_INDEX_MAP = {"Takeoff": takeoff_rows, "Landing": landing_rows}
    DT_SCENARIOS = {"Takeoff": TAKEOFF_SCENARIOS, "Landing": LANDING_SCENARIOS}
    debug_line(f"[INIT] Scenarios — Takeoff: {len(TAKEOFF_SCENARIOS)}, Landing: {len(LANDING_SCENARIOS)}")
    return DT_SCENARIOS, DT_INDEX_MAP

DT_SCENARIOS, DT_INDEX_MAP = build_scenarios_from_flat(flat_df)

# -------------------------------
# Dialogue state & utilities
# -------------------------------
CONTEXT_HISTORY = deque(maxlen=24)
VOICE_DIALOGUE = ""
LAST_PILOT_TEXT = ""
REQUIRE_PILOT_FIRST = True  # enforce seed order

def reset_voice_dialogue():
    global VOICE_DIALOGUE, LAST_PILOT_TEXT
    VOICE_DIALOGUE = ""
    LAST_PILOT_TEXT = ""

ACK_WORDS = ("roger", "wilco", "copy", "affirmative")
REQ_RX = re.compile(
    r"\b(request|taxi|runway|ready|inbound|final|approach|clearance|push|start|depart|"
    r"take\s*off|takeoff|land|with\s+information|holding\s+short|line\s+up|squawk|contact|"
    r"heading|altitude|descend|climb|maintain|vector|handoff)\b",
    re.I
)

def is_ack_only(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    has_ack = any(w in t for w in ACK_WORDS)
    has_intent = bool(REQ_RX.search(t))
    return has_ack and not has_intent

def is_pure_ack(text: str) -> bool:
    t = (text or "").strip().lower()
    return bool(re.fullmatch(r"(roger|wilco|copy|affirmative)[\.\!\s]*", t))

def context_frame_from_ctx(ctx: dict, push=True):
    dt_show  = ctx.get("dt", "Unknown")
    cs  = ctx.get("callsign", "UNKNOWN")
    alt = f"{int(ctx['alt_ft'])} ft" if (ctx.get("alt_ft") is not None and not (isinstance(ctx.get('alt_ft'), float) and math.isnan(ctx.get('alt_ft')))) else "Unknown ft"
    spd = f"{int(ctx['spd_kts'])} kts" if (ctx.get("spd_kts") is not None and not (isinstance(ctx.get('spd_kts'), float) and math.isnan(ctx.get('spd_kts')))) else "Unknown kts"
    dst = f"{ctx['dist_nm']:.2f} nm" if (ctx.get("dist_nm") is not None and not (isinstance(ctx.get('dist_nm'), float) and math.isnan(ctx.get('dist_nm')))) else "Unknown nm"
    hdg = "Unknown°"
    if ctx.get("hdg_deg") is not None and not (isinstance(ctx.get("hdg_deg"), float) and math.isnan(ctx.get("hdg_deg"))):
        try: hdg = f"{int(ctx['hdg_deg'])}°"
        except Exception: hdg = "Unknown°"
    gate = ctx.get("gate", "Unknown")
    lat = ctx.get("lat", "Unknown")
    lon = ctx.get("lon", "Unknown")
    air = ctx.get("airspace", "Unknown")
    og  = bool(ctx.get("on_ground", False))
    line = (f"[DT Time: {dt_show}] Flight: {cs}, Alt: {alt}, Spd: {spd}, "
            f"Dist: {dst}, Hdg: {hdg}, Gate: {gate}, Lat: {lat}, Lon: {lon}, "
            f"Airspace: {air}, On-Ground: {og}")
    if push:
        CONTEXT_HISTORY.append(line)
    return line

def context_frame_from_idx(idx: int, push=True):
    r = flat_df.loc[idx]
    dt_show  = r.get("parent_dt", r.get("DT Time", "Unknown"))
    cs  = (r.get("display_id") or r.get("ident") or "UNKNOWN").strip()
    alt = f"{int(r['alt_geom_ft'])} ft" if r.get("alt_geom_ft") else "Unknown ft"
    spd = f"{int(r['gs_kts'])} kts" if r.get("gs_kts") else "Unknown kts"
    dst = f"{r['dist_nm']:.2f} nm" if pd.notna(r.get("dist_nm")) else "Unknown nm"
    hdg = f"{int(r['track_deg'])}°" if pd.notna(r.get("track_deg")) else "Unknown°"
    gate = pick_gate_for_callsign(cs)
    lat = r.get("Latitude", "Unknown")
    lon = r.get("Longitude", "Unknown")
    air = r.get("Airspace Status", "Unknown")
    og  = bool(r.get("on_ground", False))
    line = (f"[DT Time: {dt_show}] Flight: {cs}, Alt: {alt}, Spd: {spd}, "
            f"Dist: {dst}, Hdg: {hdg}, Gate: {gate}, Lat: {lat}, Lon: {lon}, "
            f"Airspace: {air}, On-Ground: {og}")
    if push:
        CONTEXT_HISTORY.append(line)
    return line

def recent_context_block(n=3):
    frames = list(CONTEXT_HISTORY)[-n:]
    return "\n".join(frames[::-1])

# -------------------------------
# Transcript parsing (single definitions; reused everywhere)
# -------------------------------
CALLSIGN_RX = re.compile(r"\b(?:[A-Z]{2,4}[\s\.\,\-]*\d{2,4}|N\s*\d{1,5}\s*[A-Z]{0,2})\b", re.I)
CALLSIGN_CHUNK_JOINERS = re.compile(r"\b([A-Z]{2,3})\s*(?:,|\s+(?:flight|number|no\.?|n|a))?\s*(\d{2,4})\b", re.I)
RUNWAY_RX   = re.compile(r"\b(?:runway|rwy)\s*(\d{1,2}[LRC]?)\b", re.I)
ALT_RX      = re.compile(r"\b(\d{2,5})\s*(?:feet|ft)\b", re.I)
SPD_RX      = re.compile(r"\b(\d{1,3})\s*(?:knots|kts|kt)\b", re.I)
DIST_RX     = re.compile(r"\b(\d+(?:\.\d+)?)\s*(?:nm|nautical\s*miles?|miles?|mile|dme)\b", re.I)
HDG_RX      = re.compile(r"\b(?:heading|hdg)\s*(\d{1,3})\b", re.I)

def _normalize_callsign(s: str) -> str: return re.sub(r"[\s\.\,\-]+", "", s.upper())
def _normalize_ils(text: str) -> str:
    text = re.sub(r"\bI[\s\.\-]*L[\s\.\-]*S\b", "ILS", text, flags=re.I)
    text = re.sub(r"(?:\s*\.\s*){2,}", " ", text)
    text = re.sub(r"(,\s*){2,}", ", ", text)
    return text
def _normalize_ack(text: str) -> str:
    t = re.sub(r"\broj+er\b", "roger", text, flags=re.I)
    t = re.sub(r"\bcopy that\b", "copy", t, flags=re.I)
    t = re.sub(r"^(roger|wilco|copy|affirmative)\b", lambda m: m.group(1).capitalize(), t, flags=re.I)
    return t
def _join_callsign_chunks(text: str) -> str: return CALLSIGN_CHUNK_JOINERS.sub(lambda m: f"{m.group(1).upper()}{m.group(2)}", text)

def normalize_transcript(text: str) -> str:
    if not text: return ""
    t = text.strip()
    t = _normalize_ils(t); t = _normalize_ack(t)
    t = t.replace(",", " "); t = _join_callsign_chunks(t)
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\b(uh|um|erm|hmm)\b", "", t, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t).strip(" ,.-")
    return t

# -------------------------------
# ASR (Whisper API or local)
# -------------------------------
def _write_wav_float32(path: str, sr: int, data: np.ndarray):
    x = np.asarray(data)
    if x.ndim == 1: x = x[:, None]
    x = np.clip(x, -1.0, 1.0)
    x_int16 = (x * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(x_int16.shape[1]); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(x_int16.tobytes())

def _resample_to_16k(wave_np: np.ndarray, sr: int) -> np.ndarray:
    if wave_np.ndim == 2: wave_np = wave_np.mean(axis=1)
    if sr == 16000: return wave_np.astype(np.float32)
    x_old = np.linspace(0, 1, num=len(wave_np), endpoint=False)
    x_new = np.linspace(0, 1, num=int(len(wave_np) * 16000 / max(sr, 1)), endpoint=False)
    return np.interp(x_new, x_old, wave_np).astype(np.float32)

def transcribe_audio(audio_numpy_tuple) -> str:
    global whisper_model
    if audio_numpy_tuple is None: return ""
    sr, data = audio_numpy_tuple
    if data is None or len(data) == 0: return ""

    if USE_WHISPER_API and openai_client is not None:
        try:
            tmp = os.path.join(tempfile.gettempdir(), f"xaion_{uuid.uuid4().hex}.wav")
            _write_wav_float32(tmp, int(sr), np.array(data, dtype=np.float32))
            with open(tmp, "rb") as f:
                result = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language="en",
                    prompt=ATC_INITIAL_PROMPT,
                    response_format="verbose_json",
                    temperature=0.0,
                )
            try: os.remove(tmp)
            except Exception: pass
            txt = getattr(result, "text", "") or ""
            return normalize_transcript(txt)
        except Exception as e:
            debug_line(f"[Whisper API] error: {e} — falling back to local…")

    if whisper_model is None:
        import whisper as _wh
        whisper_model = _wh.load_model(WHISPER_MODEL_SIZE, device=str(device))
    wav16 = _resample_to_16k(np.array(data, dtype=np.float32), int(sr))
    wav16 = np.clip(wav16, -1.0, 1.0).astype(np.float32)

    use_fp16 = torch.cuda.is_available()
    result = whisper_model.transcribe(
        wav16,
        language="en",
        fp16=use_fp16,
        condition_on_previous_text=False,
        temperature=0.0,
        beam_size=5,
        best_of=5,
        initial_prompt=ATC_INITIAL_PROMPT,
        without_timestamps=True,
        no_speech_threshold=0.3,
        logprob_threshold=-1.2,
        compression_ratio_threshold=2.2,
    )
    return normalize_transcript(result.get("text") or "")

# -------------------------------
# Cleaners / guards
# -------------------------------
WEATHER_RX = re.compile(r"(?i)\b(weather|metar|atis|notam|altimeter|qnh|qfe|wind|gust|visibility|vis|rvr|runway\s+visual\s+range|ceiling|cloud|overcast|broken|few|scattered|temperature|temp|dew\s*point|icing|snow|precip|hail|thunder|storm|braking)\b")

def strip_weather_clauses(text: str) -> str:
    if not text: return text
    parts = re.split(r"([.;\n])", text)
    kept = []
    for i in range(0, len(parts), 2):
        clause = parts[i].strip()
        sep = parts[i+1] if i+1 < len(parts) else ""
        if clause and not WEATHER_RX.search(clause):
            kept.append(clause + sep)
    out = " ".join(p.strip() for p in kept).strip()
    return out

def _strip_role_leaks(text: str) -> str:
    text = re.sub(r"(?i)\b(?:ATC|PILOT)\s*:\s*", "", text)
    text = text.replace("LAST_ATC", "")
    text = re.sub(r"={2,}.*?={2,}", "", text, flags=re.DOTALL)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,")
    return text

def _first_sentence(text: str) -> str:
    parts = re.split(r"[.;\n]", text)
    for p in parts:
        p = p.strip()
        if p: return p + "."
    return text.strip()

def _strip_control_artifacts(text: str, callsign: str) -> str:
    t = text.strip()
    if t.lower().startswith("this is xaion control"):
        t = re.sub(r"^This is XAION CONTROL\.?\s*", "", t, flags=re.I)
    t = re.sub(r"\bNone\b", callsign or "None", t)
    return t

def clean_response(text, *, phase="approach", runway_hint=None, callsign=""):
    if not text: return ""
    for s in ("<|end_of_document|>", "<|endoftext|>", "<|im_end|>", "</s>", "<s>"):
        text = text.replace(s, "")
    text = re.sub(r"\[.*?\]", "", text)
    text = _strip_role_leaks(text)
    text = strip_weather_clauses(text).strip()
    text = _first_sentence(text)

    words = text.split()
    if len(words) > 28:
        text = " ".join(words[:28]).rstrip(",.;:") + "."
    return text

# -------------------------------
# UI/Audio phrasing helpers
# -------------------------------
def _example_from_scenario(comm_type: str, scen: str) -> str:
    """
    Builds a step-by-step 'What to say' helper for the selected scenario.
    Uses callsign/runway parsed from the scenario and your KGSO_TOWER_FREQ.
    """
    import re
    if not scen:
        return ""

    # Try to extract callsign + runway from the scenario text
    m_cs = re.search(r",\s*([A-Z]{2,4}\d{2,6})\b", scen)
    cs = (m_cs.group(1) if m_cs else "N123AB").upper()

    m_rw = re.search(r"Runway\s+(\d{1,2}[LRC]?)", scen, flags=re.I)
    rw = (m_rw.group(1).upper() if m_rw else DEFAULT_RWY_PRIMARY)

    twr = KGSO_TOWER_FREQ if 'KGSO_TOWER_FREQ' in globals() else "119.1"

    if comm_type == "Takeoff":
        return (
f"""**What to say — Takeoff (step-by-step)**

1) **Request taxi (Ground)**
   - “Greensboro Ground, {cs}, at gate, request taxi to Runway {rw} for departure.”

2) **Read back taxi (repeat the route + HOLD SHORT)**
   - “{cs}, taxiing via *[route]*, holding short Runway {rw}, {cs}.”
   - If told to cross: “{cs}, crossing Runway *[ID]*, then via *[route]*, holding short Runway {rw}, {cs}.”

3) **If instructed: “Contact Tower {twr}”**
   - “{cs}, switching to Tower {twr}, {cs}.”

4) **Call Tower when at the hold short line**
   - “Greensboro Tower, {cs}, holding short Runway {rw}, ready.”

5) **If told ‘Line up and wait’**
   - “{cs}, lining up and waiting Runway {rw}, {cs}.”

6) **When cleared for takeoff**
   - “{cs}, cleared for takeoff Runway {rw}, {cs}.”

7) **(Optional) Initial departure check-in after liftoff (if given)**
   - “Greensboro Departure, {cs}, passing *[altitude]*, runway heading.”


**Readback checklist (takeoff):**
- Always include: **route segments**, **HOLD SHORT**, any **CROSS Runway** instruction.
- For clearances: **say the runway** (“cleared for takeoff Runway {rw}”).
- Include numbers exactly as given: **heading/altitude/speed/frequency**.
"""
        )

    # Landing
    return (
f"""**What to say — Landing (step-by-step)**

1) **Initial check-in (Approach)**
   - “Greensboro Approach, {cs}, *[altitude]* feet, *[distance]* miles from GSO, *[heading]* degrees, inbound Runway {rw}.”

2) **Vectors / altitude / speed assignments — read back numbers**
   - “{cs}, turning heading *[hdg]*, maintaining *[alt]*, {cs}.”

3) **Approach clearance (if issued)**
   - “{cs}, descending and maintaining *[alt]* until established, cleared ILS Runway {rw}, {cs}.”
   - If visual: “{cs}, cleared visual Runway {rw}, {cs}.”

4) **If instructed: “Contact Tower {twr}”**
   - “{cs}, switching to Tower {twr}, {cs}.”

5) **Tower check-in on final**
   - “Greensboro Tower, {cs}, *[x]*-mile final Runway {rw}.”

6) **When cleared to land**
   - “{cs}, cleared to land Runway {rw}, {cs}.”

7) **After landing / exit / taxi-in**
   - “{cs}, clear of Runway {rw} at *[taxiway]*.”
   - Read back taxi-in route to gate: “{cs}, via *[route]*, {cs}.”

**Readback checklist (landing):**
- For clearances: **say the runway** (“cleared to land Runway {rw}”).
- Echo **ALT/HDG/SPD** numbers and any **frequency**.
- Read back **CROSS** or **HOLD SHORT** if assigned.
"""
    )

def xaion_prefix(callsign: str) -> str:
    tag = (callsign if callsign and callsign.upper() != "UNKNOWN" else "").strip()
    if tag:
        return f"{tag}, This is XAION CONTROL. "
    return "This is XAION CONTROL. "

def pilot_ack(cs: str) -> str:
    return random.choice(PILOT_ACK_VARIANTS).format(cs=cs)

def _audio_text_for_tts(ui_text: str) -> str:
    t = (ui_text or "").strip()

    # Strip UI role tags so TTS never says the word "Pilot" or "ATC"
    t = re.sub(r"^\s*(🧑‍✈️\s*)?Pilot:\s*", "", t, flags=re.I)  # leading "Pilot:"
    t = re.sub(r"\bPilot:\s*", "", t, flags=re.I)               # stray "Pilot:" anywhere
    t = re.sub(r"^\s*(🗼\s*)?ATC:\s*", "", t, flags=re.I)       # leading "ATC:"

    # Brand normalization for TTS
    t = t.replace("XAION CONTROL", "Zion Control")

    # -------- Helpers for pronunciations --------
    digit_map = {"0":"zero","1":"one","2":"two","3":"tree","4":"four","5":"fife","6":"six","7":"seven","8":"eight","9":"niner"}

    def speak_callsign(cs: str) -> str:
        cs = cs.strip().upper()
        m = re.match(r"^([A-Z]{2,4})(\d{2,6})$", cs)
        if not m:
            return re.sub(r"(\d)", lambda mm: " " + digit_map[mm.group(1)], cs)
        letters, digits = m.group(1), m.group(2)
        return f'{" ".join(list(letters))} {" ".join(digit_map[d] for d in digits)}'

    # If the line begins with "<CS>, Zion Control." re-speak callsign nicely
    m = re.match(r"^\s*([A-Z0-9]{3,8})\s*,\s*Zion Control\.\s*(.*)$", t, flags=re.I)
    if m:
        cs_raw, rest = m.group(1).upper(), m.group(2)
        rest = re.sub(rf"^\s*{re.escape(cs_raw)}\s*,\s*", "", rest, flags=re.I)
        t = f"{speak_callsign(cs_raw)}, Zion Control. {rest}"

    # “readback” -> “reedback”
    t = re.sub(r"\breadback\b", "reedback", t, flags=re.I)

    # Units -> words
    t = re.sub(r"(\d+)\s*°", r"\1 degrees", t)
    t = t.replace(" kts", " knots").replace(" nm", " nautical miles").replace(" ft", " feet")

    # headings: "heading 230" -> "heading two three zero"
    t = re.sub(r"\bheading\s+(\d{3})\b",
               lambda m: "heading " + " ".join(digit_map[d] for d in m.group(1)),
               t, flags=re.I)

    # runway numbers
    def _rw(m):
        raw = m.group(1).upper()
        num = re.match(r"\d{1,2}", raw).group(0)
        side = raw[len(num):] if len(raw) > len(num) else ""
        if len(num) == 2 and num.startswith("0"): spoken = digit_map[num[1]]
        elif len(num) == 1: spoken = digit_map[num]
        else: spoken = " ".join(digit_map[d] for d in num)
        side_spoken = {"L": "left", "R": "right", "C": "center"}.get(side, "")
        return "Runway " + (spoken + (" " + side_spoken if side_spoken else ""))
    t = re.sub(r"\bRunway\s+(\d{1,2}[LRC]?)\b", _rw, t, flags=re.I)

    # frequencies: "119.1"
    t = re.sub(r"\b(\d{3}\.\d)\b",
               lambda m: " ".join(digit_map.get(ch, "point") for ch in m.group(1)),
               t)

    # callsigns elsewhere
    def _cs_everywhere(m):
        letters, digits = m.group(1), m.group(2)
        return f'{" ".join(list(letters))} {" ".join(digit_map[d] for d in digits)}'
    t = re.sub(r"\b([A-Z]{2,4})(\d{2,6})\b", _cs_everywhere, t)

    return t

# -------------------------------
# Scenario seeding helpers
# -------------------------------
def fmt_takeoff_from_idx(idx: int) -> str:
    r = flat_df.loc[idx]
    cs = _safe_callsign_from_row(r)
    if not cs: return ""
    spd  = f"{int(r['gs_kts'])} knots" if r.get("gs_kts") else "speed unknown"
    dist = f"{r['dist_nm']:.1f} nautical miles" if pd.notna(r.get("dist_nm")) else "distance unknown"
    rw = runway_from_heading(r.get("track_deg")) or DEFAULT_RWY_PRIMARY
    gate = pick_gate_for_callsign(cs)
    return f"Greensboro Ground, {cs}, at {gate}, {spd}, {dist} from field, request taxi to Runway {rw} for departure."

def fmt_landing_from_idx(idx: int) -> str:
    r = flat_df.loc[idx]
    cs = _safe_callsign_from_row(r)
    if not cs: return ""
    alt = f"{int(r['alt_geom_ft'])} feet" if r.get("alt_geom_ft") else "altitude unknown"
    dist = f"{r['dist_nm']:.1f} nautical miles" if pd.notna(r.get("dist_nm")) else "distance unknown"
    hdg = f"{int(r['track_deg'])} degrees" if pd.notna(r["track_deg"]) else "heading unknown"
    rw = runway_from_heading(r.get("track_deg"))
    rw_txt = f", inbound Runway {rw}" if rw else ""
    return f"Greensboro Approach, {cs}, {alt}, {dist} from GSO, {hdg}{rw_txt}."

# -------------------------------
# Networking helper (used by UI)
# -------------------------------
def _first_open_port(preferred: int, tries: int = 6):
    for i in range(tries):
        p = preferred + i
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
    return preferred
# ===============================
# XAION_CONTROL_ST_V56.py — Part 2/3
# State machine, validators, generators, voice/full-sim runners
# ===============================

# -------------------------------
# State machine utilities
# -------------------------------
CALLSTATE = {}  # { cs: {"phase": "ground"/"approach", "gate": str|None, "runway": str, "stage": str|None, "last_dt": ts} }

def _row_phase(r) -> str:
    return "ground" if bool(r.get("on_ground", False)) else "approach"

def _row_dist_nm(r):
    v = r.get("dist_nm")
    try:
        return float(v) if pd.notna(v) else None
    except Exception:
        return None

# -------------------------------
# Readback extraction / validator
# -------------------------------
_NUM = r"(?:\d+(?:\.\d+)?)"

def _extract_items(atc_text):
    """Return a dict of pertinent items from an ATC line."""
    t = (atc_text or "").lower()
    return {
        "runway_takeoff": bool(re.search(r"cleared (?:for )?takeoff .*runway\s*(\d{2}[lrc]?)", t)),
        "runway_land":   bool(re.search(r"cleared to land .*runway\s*(\d{2}[lrc]?)", t)),
        "cross":         re.findall(r"\bcross\s+runway\s*(\d{2}[lrc]?)", t),
        "hold_short":    re.findall(r"\bhold\s+short\s+runway\s*(\d{2}[lrc]?)", t),
        "taxi":          "taxi" in t,
        "alt":           re.findall(rf"\b(?:maintain|climb|descend).*?({_NUM})\s*(?:feet|ft)", t),
        "hdg":           re.findall(rf"\bheading\s*({_NUM})", t),
        "spd":           re.findall(rf"\b(?:speed|maintain\s*speed)\s*({_NUM})\s*(?:kts|kt|knots)", t),
        "freq":          re.findall(rf"\b(\d{{3}}\.\d)\b", t),  # e.g., 119.1
        "approach":      re.findall(r"\b(?:ils|rnnav?|visual)\b.*?(?:runway\s*\d{2}[lrc]?)?", t),
    }

def _items_nonempty(items: dict | None) -> bool:
    if not items:
        return False
    for v in items.values():
        if isinstance(v, bool) and v:
            return True
        if isinstance(v, (list, tuple)) and len(v) > 0:
            return True
    return False

def _pilot_covers_items(pilot_text, items):
    """
    Lenient readback validator.
    Returns False if ATC line had no extractable tokens (prevents 'readback correct'
    after a line with no actionable content).
    """
    if not _items_nonempty(items):
        return False

    p = normalize_transcript(pilot_text or "").lower()

    # Accept taxi/taxiing
    has_taxi = bool(re.search(r"\btaxi(?:ing)?\b", p))
    # Accept hold short / holding short
    has_hold_short = bool(re.search(r"\bhold(?:ing)?\s+short\b", p))
    # Runway mentioned?
    has_runway = bool(re.search(r"\brunway\s*\d{1,2}[lrc]?\b", p))

    # Taxi clearances must include HOLD SHORT; runway mention strongly preferred
    if items.get("taxi") and items.get("hold_short"):
        if not has_hold_short:
            return False
        # If ATC named a specific runway to hold short, prefer that the pilot echoed a runway
        if items["hold_short"] and not has_runway:
            return False

    # Takeoff
    if items.get("runway_takeoff"):
        if not re.search(r"\b(takeoff|line\s*up\s*and\s*wait)\b", p): return False
        if not has_runway:                                             return False

    # Landing
    if items.get("runway_land"):
        if not re.search(r"\bcleared\s+to\s+land\b", p):              return False
        if not has_runway:                                            return False

    # Altitude / Heading / Speed
    if items.get("alt"):
        if not re.search(r"\b\d{2,5}\s*(?:feet|ft)\b", p):            return False
    if items.get("hdg"):
        if not re.search(r"\bheading\s*\d{1,3}\b", p):                return False
    if items.get("spd"):
        if not re.search(r"\b\d{1,3}\s*(?:kts?|knots?)\b", p):        return False

    # Frequency (either the number or explicit switching/contacting)
    if items.get("freq"):
        has_num = any(re.search(fr"\b{re.escape(f)}\b", p) for f in items["freq"])
        has_words = bool(re.search(r"\b(switching|contact(?:ing)?)\b", p))
        if not (has_num or has_words):                                return False

    # Approach type
    if items.get("approach"):
        if not re.search(r"\b(ils|rnnav?|visual)\b", p):              return False

    return True


def _pilot_readback_from_atc_text(cs: str, atc_core: str) -> str:
    """
    Convert a one-sentence ATC instruction into a natural pilot readback.
    """
    t = atc_core.strip()
    t = re.sub(rf"^\s*{re.escape(cs)}\s*,?\s*", "", t, flags=re.I)  # strip leading CS

    # progressive verbs for readback
    t = re.sub(r"\bturn heading\b", "turning heading", t, flags=re.I)
    t = re.sub(r"\bdescend and maintain\b", "descending and maintaining", t, flags=re.I)
    t = re.sub(r"\bmaintain\b", "maintaining", t, flags=re.I)
    t = re.sub(r"\btaxi to\b", "taxiing to", t, flags=re.I)
    t = re.sub(r"\bhold short\b", "holding short", t, flags=re.I)
    t = re.sub(r"\bcross\b", "crossing", t, flags=re.I)
    t = re.sub(r"\bcontact\b", "switching to", t, flags=re.I)

    core = t.rstrip(".")
    return f"Pilot: {cs}, {core}, {cs}."

# -------------------------------
# Simple rule ATC for fallback
# -------------------------------
def _rule_based_atc_for_row(r, cs, rw) -> str:
    phase = "ground" if r.get("on_ground") else "approach"
    if phase == "ground":
        return f"{cs}, taxi to Runway {rw}, hold short Runway {rw}."
    final_hdg = _hdg_to_runway(rw)
    return f"{cs}, turn heading {int(final_hdg)}, maintain 2000 until established."

# -------------------------------
# LLM Monitor (decide next role)
# -------------------------------
def _last_atc(dialogue: str) -> str:
    if not dialogue: return ""
    lines = [ln for ln in dialogue.split("\n") if ln.strip()]
    atc_lines = [ln for ln in lines if re.search(r"\bATC\s*:", ln)]
    return atc_lines[-1] if atc_lines else ""

def monitor_decide_next_role(dialogue: str, phase: str, callsign: str) -> str:
    prompt = (
        "You are the LLM Monitor for an ATC radio simulation.\n"
        "Given the running dialogue and current phase, output exactly one word: Pilot or ATC — who speaks next.\n\n"
        f"Phase: {phase}\nCall sign: {callsign or 'UNKNOWN'}\n\nDialogue so far:\n{(dialogue or '').strip()}\n\nDecision:"
    )
    try:
        tok = phi4_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        out = phi4_model.generate(
            tok.input_ids,
            attention_mask=tok.attention_mask,
            max_new_tokens=4, do_sample=False, num_beams=1,
            pad_token_id=phi4_tokenizer.eos_token_id, eos_token_id=phi4_tokenizer.eos_token_id
        )
        text = phi4_tokenizer.decode(out[0], skip_special_tokens=True)
        decision = "ATC" if "ATC" in text.upper() else "Pilot"
    except Exception as e:
        debug_line(f"[MONITOR] fallback due to: {e}; defaulting to ATC")
        decision = "ATC"

    debug_line(f"[MONITOR] Next role → {decision}")
    return decision

# -------------------------------
# Guard rails (callsign/runway -> Tower)
# -------------------------------
def guard_or_tower(cs: str | None, runway: str | None):
    bad_cs = (not cs) or (cs.strip().upper() in ("", "UNKNOWN", "NONE"))
    bad_rw = runway not in KGSO_RUNWAYS
    if bad_cs:
        return True, f"None, This is XAION CONTROL. Unable to proceed (callsign unknown). Contact Tower {KGSO_TOWER_FREQ}."
    if bad_rw:
        corr = DEFAULT_RWY_PRIMARY
        tag = cs or "None"
        return True, f"{tag}, This is XAION CONTROL. Runway not recognized. Proceed to Runway {corr}, then contact Tower {KGSO_TOWER_FREQ}."
    return False, ""

# -------------------------------
# Role generators (ATC/Pilot with validators)
# -------------------------------
def gen_once(role, dialogue, ctx):
    """
    One LLM turn (ATC or Pilot).
    Adds:
      - Anti-template echo: never outputs '<instruction here>' or angle-bracketed placeholders.
      - Safe fallback synthesis when ATC model under-outputs.
    """
    phase = ctx.get("phase", "approach")
    runway_hint = ctx.get("runway", DEFAULT_RWY_PRIMARY)
    callsign = ctx.get("callsign", "UNKNOWN")

    def _safe_atc(callsign: str, phase: str, rw: str) -> str:
        if phase == "ground":
            return f"{callsign}, taxi to Runway {rw}, hold short Runway {rw}."
        else:
            return f"{callsign}, turn heading {int(_hdg_to_runway(rw))}, maintain 2000 until established."

    def _anti_template(s: str) -> str:
        # Strip any angle-bracketed placeholders and their wording
        s2 = re.sub(r"<[^>]*>", "", s)
        s2 = re.sub(r"(?i)\binstruction here\b", "", s2)
        return re.sub(r"\s{2,}", " ", s2).strip(" ,.;")

    monitor_header("XAION SIM START (VOICE CTX)")
    monitor_block(role, phase, runway_hint)
    ctx_current = context_frame_from_ctx(ctx, push=True)
    _ctx_dump_lines(ctx_current, dialogue_text=dialogue)

    if role == "ATC":
        if is_ack_only(LAST_PILOT_TEXT):
            atc_struct = ctx.get("_last_atc_struct")
            if atc_struct and _pilot_covers_items(LAST_PILOT_TEXT, atc_struct):
                out_line = f"{callsign}, readback correct."
            else:
                out_line = _ack_line(callsign)
            out_line = atc_prefix_and_dedup(callsign, out_line)
            monitor_header("XAION SIM END (VOICE CTX)")
            return out_line

        if phase == "ground":
            phase_goal = ("Issue ONE specific taxi instruction with explicit route segments and HOLD SHORT. "
                          "Authorize runway crossings only with 'CROSS Runway <id>'.")
        elif phase == "approach":
            phase_goal = ("Issue ONE concise vector/altitude/speed or approach clearance. "
                          "If short final and runway clear, you may issue 'cleared to land'.")
        else:
            phase_goal = "Issue ONE concise instruction or respond to the request."

        instruction = f"""
You are ATC. U.S. phraseology. Rules:
- Start with CALLSIGN once (no brand text), then the clearance/Instruction.
- ONE sentence, <= 25 words. No meta/explanations.
- Keep safety tokens explicit: RUNWAY for takeoff/landing; HOLD SHORT; CROSS Runway <id>; numeric ALT/HDG/SPD; frequencies.
- If the prior Pilot readback was incomplete, restate the full instruction clearly.
- DO NOT write angle brackets or the words 'instruction here'.

{phase_goal}

Output format (example—do NOT copy literally):
ATC: {callsign}, <real instruction>.
""".strip()
    else:
        last_atc = _last_atc(dialogue)
        instruction = f"""
You are the Pilot. Provide ONE short readback.
- Start with "Pilot: {callsign}, ..."
- Read back ALL pertinent items from LAST ATC: taxi route + HOLD SHORT; any CROSS + runway; takeoff/landing + runway; ALT/HDG/SPD numbers; frequency; approach type/runway.
- If ambiguous, ask for clarification concisely.
- <= 25 words. No meta.

LAST_ATC: {last_atc}

Output format (example—do NOT copy literally):
Pilot: {callsign}, <readback>.
""".strip()

    base_prompt = (
        f"{instruction}\n\n=== CONTEXT FRAME (CURRENT) ===\n{ctx_current}\n\n"
        f"=== RECENT CONTEXT FRAMES (NEWEST FIRST) ===\n{recent_context_block(3)}\n\n"
        f"=== DIALOGUE SO FAR ===\n{dialogue.strip()}\n\n{role}:"
    )

    try:
        tok = phi4_tokenizer(base_prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)
        out = phi4_model.generate(
            tok.input_ids, attention_mask=tok.attention_mask,
            max_new_tokens=80, do_sample=True, temperature=0.7, top_p=0.9,
            repetition_penalty=1.08, no_repeat_ngram_size=3,
            pad_token_id=phi4_tokenizer.eos_token_id, eos_token_id=phi4_tokenizer.eos_token_id,
        )
        text = phi4_tokenizer.decode(out[0], skip_special_tokens=True)
        if f"{role}:" in text:
            text = text.split(f"{role}:", 1)[-1]
        text_raw = text.strip()
    except Exception as e:
        debug_line(f"[{role}] LLM error -> fallback: {e}")
        text_raw = ""

    text_nowx = strip_weather_clauses(text_raw)
    resp = clean_response(text_nowx, phase=phase, runway_hint=runway_hint, callsign=callsign)
    resp = _anti_template(_strip_role_leaks(resp))
    resp = _strip_control_artifacts(resp, callsign)

    # Guard/repair ATC under-output or template echo
    if role == "ATC":
        # if too short after removing the callsign or still has placeholders -> synthesize safe line
        body = re.sub(rf"^\s*{re.escape(callsign)}\s*,\s*", "", resp, flags=re.I).strip()
        if len(body) < 6 or re.search(r"[<>]|(?i)instruction here", resp):
            resp = _safe_atc(callsign, phase, runway_hint)

    # Final basic guard
    if len(resp) < 8:
        resp = (_safe_atc(callsign, phase, runway_hint)
                if role == "ATC"
                else f"Pilot: {callsign}, say again the full instruction.")

    if role == "ATC":
        # Repeat handling
        if REPEAT_RX.search(LAST_PILOT_TEXT or ""):
            last_ui = _last_atc(dialogue)
            if last_ui:
                last_core = strip_atc_ui_to_core(last_ui, callsign)
                if last_core:
                    resp = last_core
        atc_struct = _extract_items(resp)
        ctx["_last_atc_struct"] = atc_struct
        resp = atc_prefix_and_dedup(callsign, resp)
    else:
        atc_struct = ctx.get("_last_atc_struct", None)
        if not _pilot_covers_items(resp, atc_struct):
            resp = f"Pilot: {callsign}, say again the full instruction."
        elif callsign.upper() not in resp.upper():
            resp = f"{resp.rstrip('.').rstrip(',')}, {callsign}."

    monitor_header("XAION SIM END (VOICE CTX)")
    return resp


def gen_once_idx(role, dialogue, idx):
    r = flat_df.loc[idx]
    phase = "ground" if r["on_ground"] else "approach"
    runway_hint = runway_from_heading(r["track_deg"]) or DEFAULT_RWY_PRIMARY
    callsign = (r.get("display_id") or r.get("ident") or "UNKNOWN").strip()

    monitor_header("GENERATION CALL (TYPED CTX)")
    monitor_block(role, phase, runway_hint)
    ctx_current = context_frame_from_idx(idx, push=True)
    _ctx_dump_lines(ctx_current, dialogue_text=dialogue)

    if role == "ATC":
        if phase == "ground":
            phase_goal = (
                "Issue ONE specific taxi instruction to a destination (e.g., a runway). "
                "Include the exact route segments and explicit HOLD SHORT where applicable. "
                "Never authorize crossing an active runway without a specific 'cross' clearance."
            )
        elif phase == "approach":
            phase_goal = (
                "Issue ONE concise control or approach instruction (heading/altitude/speed or approach clearance). "
                "If on final and runway is clear, you may issue 'cleared to land'."
            )
        else:
            phase_goal = "Issue ONE concise instruction (altitude, heading, speed) or respond to a pilot request."

        instruction = f"""
You are ATC. Follow U.S. FAA/ICAO phraseology. Hard rules:
- Address the aircraft by CALLSIGN at the start of the line.
- Do NOT include meta text. No explanations, no notes.
- Keep it ONE sentence, <= 25 words.
- Never clear an unsafe action (e.g., takeoff/land/cross if runway is occupied).
- If prior Pilot readback was incomplete/incorrect, RESTATE the exact instruction clearly.

Pertinent items that MUST be explicit in any clearance:
- Taxi route segments and HOLD SHORT.
- 'CROSS' when authorizing a runway crossing (name the runway).
- Takeoff/Landing clearance MUST include runway.
- ALT/HDG/SPD assignments MUST include numeric values.
- Frequency change MUST include the frequency.

{phase_goal}

Output format (strict):
ATC: {callsign}, <instruction here>.
""".strip()
    else:
        last_atc = _last_atc(dialogue)
        instruction = f"""
You are the Pilot. ONE short readback line. Hard rules:
- Start with 'Pilot: ' then CALLSIGN.
- Read back ALL pertinent items from the LAST ATC instruction: taxi route + HOLD SHORT; 'CROSS' and runway; takeoff/landing clearance + runway; altitude/heading/speed values; assigned frequency; approach type + runway.
- If the ATC line was ambiguous, request clarification concisely.
- No meta text. <= 25 words.

LAST_ATC: {last_atc}

Output format (strict):
Pilot: {callsign}, <exact readback or clarification>.
""".strip()

    base_prompt = (
        f"{instruction}\n\n=== CONTEXT FRAME (CURRENT) ===\n{ctx_current}\n\n"
        f"=== RECENT CONTEXT FRAMES (NEWEST FIRST) ===\n{recent_context_block(3)}\n\n"
        f"=== DIALOGUE SO FAR ===\n{dialogue.strip()}\n\n{role}:"
    )
    inputs = phi4_tokenizer(base_prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)
    out = phi4_model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.8,
        top_p=0.92,
        repetition_penalty=1.08,
        no_repeat_ngram_size=3,
        pad_token_id=phi4_tokenizer.eos_token_id,
        eos_token_id=phi4_tokenizer.eos_token_id,
    )
    text = phi4_tokenizer.decode(out[0], skip_special_tokens=True)
    if f"{role}:" in text: text = text.split(f"{role}:", 1)[-1]
    text_raw = text.strip()
    text_nowx = strip_weather_clauses(text_raw)
    debug_line(f"[ATC RAW REDACTED] {text_nowx}")

    resp = clean_response(text_nowx, phase=phase, runway_hint=runway_hint, callsign=callsign).strip()
    resp = _strip_role_leaks(resp)
    resp = _strip_control_artifacts(resp, callsign)
    if len(resp) < 8:
        resp = _rule_based_atc_for_row(r, callsign, runway_hint) if role == "ATC" else f"Pilot: {callsign}, say again the full instruction."

    if role == "ATC":
        resp = xaion_prefix(callsign) + resp
    else:
        if callsign.upper() not in resp.upper():
            resp = f"{resp.rstrip('.').rstrip(',')}, {callsign}."

    debug_line(f"[ATC FINAL] {resp}")
    return resp

# -------------------------------
# Voice step (single-turn ATC reply path)
# -------------------------------
def parse_transcript_ctx(text: str) -> dict:
    t = normalize_transcript(text)
    cs = None; m = CALLSIGN_RX.search(t.upper())
    if m: cs = _normalize_callsign(m.group(0))
    rwy = None; m = RUNWAY_RX.search(t);  rwy = m.group(1).upper() if m else None
    alt = None; m = ALT_RX.search(t);  alt = float(m.group(1)) if m else None
    spd = None; m = SPD_RX.search(t);  spd = float(m.group(1)) if m else None
    dist = None; m = DIST_RX.search(t); dist = float(m.group(1)) if m else None
    hdg = None; m = HDG_RX.search(t);  hdg = float(m.group(1)) if m else None
    gate = _gate_from_text(t)
    low_t = t.lower()
    on_ground = any(kw in low_t for kw in ["ground", "ramp", "taxi", "gate"])
    approach_cues = ("approach","final","established","localizer","glideslope","cleared to land","inbound")
    is_approach = any(kw in low_t for kw in approach_cues)
    phase = "ground" if on_ground and not is_approach else "approach"
    if not rwy and hdg is not None: rwy = runway_from_heading(hdg)
    if not rwy: rwy = DEFAULT_RWY_PRIMARY
    if not gate and cs:
        gate = pick_gate_for_callsign(cs)
    return {
        "dt": "Unknown",
        "callsign": cs or "UNKNOWN",
        "alt_ft": alt if alt is not None else (0.0 if on_ground else None),
        "spd_kts": spd,
        "dist_nm": dist,
        "hdg_deg": hdg,
        "gate": gate,
        "lat": "Unknown",
        "lon": "Unknown",
        "airspace": "Grounded/Parked" if on_ground else "Unknown",
        "on_ground": on_ground,
        "phase": phase,
        "runway": rwy,
    }

def tts_elevenlabs(text: str, *, voice_id: str) -> str | None:
    if not text:
        return None
    if not _has_eleven:
        return None
    try:
        os.makedirs("tts_cache", exist_ok=True)

        # Robust role-tag stripping for speech
        t = text
        # Remove leading role labels even if the LLM varied punctuation/case
        t = re.sub(r"(?im)^(?:🧑‍✈️\s*)?(?:Pilot|ATC)\b[:,-]?\s*", "", t).strip()
        # Also strip any stray inline role labels that might sneak in
        t = re.sub(r"(?i)(?:🧑‍✈️\s*)?(?:Pilot|ATC)\b[:,-]?\s*", "", t).strip()

        # Brand normalization & pronunciation tweaks
        t = t.replace("XAION CONTROL", "Zion Control")
        t = _audio_text_for_tts(t)  # keep your existing numeral/runway/freq normalizations

        path = os.path.join("tts_cache", f"{voice_id}_{int(time.time()*1000)}.mp3")
        audio_gen = eleven_client.text_to_speech.convert(
            voice_id=voice_id,
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=t,
            model_id="eleven_turbo_v2_5",
        )
        with open(path, "wb") as f:
            for chunk in audio_gen:
                if chunk:
                    f.write(chunk)
        return path
    except Exception as e:
        debug_line(f"[TTS] ElevenLabs error: {e}")
        return None

# --- Vocal Input progression helpers (ground) ---
# --- Vocal Input progression helpers (ground) ---
READY_RX        = re.compile(r"\b(holding\s+short\s+runway\s*(\d{2}[LRC]?)\b.*\bready\b|ready\s+(?:for\s+)?(?:departure|takeoff))", re.I)
CONTACT_TWR_RX  = re.compile(r"\b(contact(?:ing)?|switching\s+to)\s*(?:tower|\b119\.1\b)\b", re.I)
REQ_TAXI_RX     = re.compile(r"\brequest\s+taxi\s+to\s+runway\s*(\d{1,2}[LRC]?)\b", re.I)
LINEUP_RX       = re.compile(r"\b(line\s*up(?:\s*and\s*wait)?|luaw)\b.*\brunway\s*(\d{2}[LRC]?)\b", re.I)
PILOT_ASSERT_TKOF_RX = re.compile(r"\bcleared\s+for\s+takeoff\b", re.I)

def _vocal_ground_flow(cs: str, ctx: dict, pilot_text: str, dialogue: str) -> str | None:
    """
    For Vocal Input only: detect ground milestones and emit ONE ATC line,
    phrased by the LLM, that advances toward takeoff.
    Returns a fully-branded ATC UI line, or None to fall back.
    """
    rw = ctx.get("runway") or DEFAULT_RWY_PRIMARY
    st = CALLSTATE.setdefault(cs, {"phase":"ground","runway":rw,"stage":None, "gate":ctx.get("gate")})

    # 1) Pilot requests taxi -> taxi + HOLD SHORT
    if REQ_TAXI_RX.search(pilot_text):
        atc_core = _compose_taxi_to_runway(cs, rw, st.get("gate"))
        core = llm_style_atc_from_core(cs, atc_core, dialogue, {"phase":"ground","runway":rw,"callsign":cs})
        st.update(stage="taxi_issued")
        return f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}{core}"

    # 2) Holding short & ready, or LUAW -> hand off to Tower (if not already handed off)
    if (READY_RX.search(pilot_text) or LINEUP_RX.search(pilot_text)) and st.get("stage") in (None, "taxi_issued"):
        atc_core = f"{cs}, contact Tower {KGSO_TOWER_FREQ}."
        core = llm_style_atc_from_core(cs, atc_core, dialogue, {"phase":"ground","runway":rw,"callsign":cs})
        st.update(stage="handoff_twr")
        return f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}{core}"

    # 3) Pilot says 'switching/contacting Tower' -> cleared for takeoff
    if CONTACT_TWR_RX.search(pilot_text) and st.get("stage") in ("handoff_twr", "taxi_issued"):
        atc_core = f"{cs}, cleared for takeoff Runway {rw}."
        core = llm_style_atc_from_core(cs, atc_core, dialogue, {"phase":"ground","runway":rw,"callsign":cs})
        st.update(stage="cleared_takeoff")
        return f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}{core}"

    # 4) Pilot prematurely says "cleared for takeoff" -> either clear now (if at Tower) or push to Tower
    if PILOT_ASSERT_TKOF_RX.search(pilot_text):
        if st.get("stage") == "handoff_twr":
            atc_core = f"{cs}, cleared for takeoff Runway {rw}."
            core = llm_style_atc_from_core(cs, atc_core, dialogue, {"phase":"ground","runway":rw,"callsign":cs})
            st.update(stage="cleared_takeoff")
            return f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}{core}"
        else:
            atc_core = f"{cs}, contact Tower {KGSO_TOWER_FREQ}."
            core = llm_style_atc_from_core(cs, atc_core, dialogue, {"phase":"ground","runway":rw,"callsign":cs})
            st.update(stage="handoff_twr")
            return f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}{core}"

    return None



def voice_step(pilot_text: str):
    """
    Vocal Input step with robust landing handling.

    Key behaviors:
      • Phase detection prioritizes APPROACH cues (inbound/final/established/ILS/visual) over generic 'runway'.
      • ILS/visual clearance readback ⇒ 'readback correct.' (stage -> ils_cleared).
      • Tower '[x]-mile final' ⇒ 'cleared to land Runway <rw>.' (stage -> cleared_land).
      • 'clear of Runway <rw>' ⇒ taxi-in to gate (not a taxi-to-takeoff route).
      • Readback coverage still uses VOICE_LAST_ATC_STRUCT.
      • Ground flow (taxi → contact Tower → cleared for takeoff) unaffected.
    """
    import re
    global VOICE_DIALOGUE, LAST_PILOT_TEXT, VOICE_LAST_ATC_STRUCT, CALLSTATE

    # -------- Regex (local, no external refs) --------
    repeat_rx        = re.compile(r'\b(say\s+again|repeat|last\s+transmission|please\s+repeat|confirm)\b', re.I)

    # Ground cues: DO NOT include the word 'runway' alone (approach also says runway).
    ground_cues_rx   = re.compile(r'\b(ground|holding\s+short|hold\s+short|line\s*up|gate|taxi|ramp)\b', re.I)
    tower_ready_rx   = re.compile(r'\btower\b.*\b(ready|holding\s+short|line\s*up)\b', re.I)

    # Approach cues & readbacks
    inbound_rx       = re.compile(r'\binbound\b', re.I)
    established_rx   = re.compile(r'\bestablished\b', re.I)
    final_rx         = re.compile(r'\b(\d+(?:\.\d+)?)\s*-\s*mile\s+final\b|\b(\d+(?:\.\d+)?)\s*mile\s+final\b', re.I)
    cleared_ils_rx   = re.compile(r'\bcleared\s+(?:ils|localizer|rnnav?|visual)\s+runway\s*\d{1,2}[lrc]?\b', re.I)
    desc_est_rx      = re.compile(r'\bdescending\s+and\s+maintaining\b.*\buntil\s+established\b', re.I)
    cleared_land_rx  = re.compile(r'\bcleared\s+to\s+land\b', re.I)
    contact_twr_rx   = re.compile(r'\b(switch(?:ing)?\s*to|contact(?:ing)?)\s*tower\b', re.I)
    clear_of_rw_rx   = re.compile(r'\bclear\s+of\s+runway\s*(\d{1,2}[lrc]?)\b', re.I)

    # -------- Input ----------
    pt = (pilot_text or "").strip()
    if not pt:
        debug_line("[VOICE] Empty transcript.")
        return "❌", "None", "❌ No transcript. Please type or record a request.", None, "Awaiting pilot reply…", snapshot_debug_dump(), None
    LAST_PILOT_TEXT = pt

    # Parse + keep last good callsign
    ctx = parse_transcript_ctx(pt)
    cs  = ctx.get("callsign", "UNKNOWN")
    last_cs = globals().setdefault("LAST_ACTIVE_CS", None)
    if (not cs) or (str(cs).upper() in {"", "UNKNOWN", "NONE"}):
        if last_cs:
            cs = last_cs
            ctx["callsign"] = cs
    else:
        globals()["LAST_ACTIVE_CS"] = cs

    rw    = ctx.get("runway", DEFAULT_RWY_PRIMARY)
    phase = ctx.get("phase", "approach")  # provisional; we refine below

    # ---- Phase decision: approach cues override ground cues ----
    low = pt.lower()
    has_approach = bool(
        inbound_rx.search(low) or established_rx.search(low) or final_rx.search(low) or
        cleared_ils_rx.search(low) or desc_est_rx.search(low) or cleared_land_rx.search(low)
    )
    has_ground = bool(ground_cues_rx.search(low) or tower_ready_rx.search(low))

    if has_approach:
        phase = "approach"
    elif has_ground:
        phase = "ground"

    # Honor existing flow if already known
    st = CALLSTATE.setdefault(cs, {"phase": phase, "runway": rw, "gate": ctx.get("gate"), "stage": None, "last_dt": None})
    if st.get("phase") == "ground" and not has_approach:
        phase = "ground"
    st["phase"] = phase
    st["runway"] = rw

    # ---- Monitor / context ----
    monitor_header("XAION SIM START (VOICE CTX)")
    monitor_block("Pilot", phase, rw)
    ctx_line = context_frame_from_ctx(ctx, push=True)
    pilot_ui = f"{EMOJI_PILOT} Pilot: {pt}"
    VOICE_DIALOGUE = f"{VOICE_DIALOGUE}\n{pilot_ui}".strip()
    _ctx_dump_lines(ctx_line, dialogue_text=VOICE_DIALOGUE)

    # ---- Guards ----
    guarded, tower_line = guard_or_tower(cs if cs and cs.upper() != "UNKNOWN" else None, rw)
    if guarded:
        atc_ui = f"{EMOJI_ATC} ATC: {tower_line}"
        VOICE_DIALOGUE += f"\n{atc_ui}"
        atc_audio = tts_elevenlabs(strip_role_prefix(atc_ui), voice_id=ELEVEN_VOICE_ID_DEFAULT)
        status = "" if atc_audio else ("ATC audio unavailable (check ELEVEN_API_KEY)" if _has_eleven else "Text OK — TTS not configured.")
        monitor_header("XAION SIM END (VOICE CTX)")
        return pt, "Invalid callsign/runway in pilot transmission.", VOICE_DIALOGUE, atc_audio, status, snapshot_debug_dump(), None

    # ---- 'Say again' ----
    if repeat_rx.search(pt):
        last_ui = _last_atc(VOICE_DIALOGUE)
        if last_ui:
            core = strip_atc_ui_to_core(last_ui, cs)
            atc_line = atc_prefix_and_dedup(cs, core)
            atc_ui = f"{EMOJI_ATC} ATC: {atc_line}"
            VOICE_DIALOGUE += f"\n{atc_ui}"
            atc_audio = tts_elevenlabs(atc_line, voice_id=ELEVEN_VOICE_ID_DEFAULT)
            status = "" if atc_audio else ("ATC audio unavailable (check ELEVEN_API_KEY)" if _has_eleven else "Text OK — TTS not configured.")
            monitor_header("XAION SIM END (VOICE CTX)")
            return pt, "None", VOICE_DIALOGUE, (atc_audio or None), status, snapshot_debug_dump(), None

    # -------------------------
    # FORCED FLOW (GROUND)
    # -------------------------
    if phase == "ground":
        # minimal ground driver as before (request taxi -> handoff -> cleared for TO)
        if re.search(r'\b(request|ready\s*to)?\s*taxi\b', low):
            forced_core = _compose_taxi_to_runway(cs, rw, st.get("gate")); st["stage"] = "taxi_issued"
        elif tower_ready_rx.search(low) and st.get("stage") not in ("handoff_twr", "cleared_takeoff"):
            forced_core = f"{cs}, contact Tower {KGSO_TOWER_FREQ}."; st["stage"] = "handoff_twr"
        elif st.get("stage") == "handoff_twr":
            forced_core = f"{cs}, cleared for takeoff Runway {rw}."; st["stage"] = "cleared_takeoff"
        else:
            forced_core = None

        if forced_core:
            core = re.sub(rf'^\s*{re.escape(cs)}\s*,\s*', '', forced_core, flags=re.I)
            atc_line_full = atc_prefix_and_dedup(cs, core)
            atc_ui = f"{EMOJI_ATC} ATC: {atc_line_full}"
            VOICE_DIALOGUE += f"\n{atc_ui}"
            VOICE_LAST_ATC_STRUCT = _extract_items(strip_atc_ui_to_core(atc_ui, cs))
            atc_audio = tts_elevenlabs(atc_line_full, voice_id=ELEVEN_VOICE_ID_DEFAULT)
            status = "" if atc_audio else ("ATC audio unavailable (check ELEVEN_API_KEY)" if _has_eleven else "Text OK — TTS not configured.")
            monitor_header("XAION SIM END (VOICE CTX)")
            return pt, "None", VOICE_DIALOGUE, (atc_audio or None), status, snapshot_debug_dump(), None

    # -------------------------
    # FORCED FLOW (APPROACH / LANDING)
    # -------------------------
    forced_core = None
    if phase == "approach":
        # Pilot read back an approach clearance (ILS/visual) → readback correct; remember stage
        if cleared_ils_rx.search(low) or desc_est_rx.search(low):
            forced_core = f"{cs}, readback correct."
            st["stage"] = "ils_cleared"

        # Tower call on final → cleared to land
        elif final_rx.search(low):
            forced_core = f"{cs}, cleared to land Runway {rw}."
            st["stage"] = "cleared_land"

        # Pilot says 'cleared to land …' → readback correct
        elif cleared_land_rx.search(low):
            forced_core = f"{cs}, readback correct."
            st["stage"] = "cleared_land"

        # After landing (clear of runway) → taxi-in
        elif clear_of_rw_rx.search(low):
            gate_use = st.get("gate") or pick_gate_for_callsign(cs)
            forced_core = _compose_taxi_to_gate(cs, gate_use)
            st["stage"] = "taxi_in_issued"

        # Early inbound → give one safe vector if nothing issued yet
        elif inbound_rx.search(low) and st.get("stage") is None:
            forced_core = f"{cs}, turn heading {int(_hdg_to_runway(rw))}, maintain 2000 until established."
            st["stage"] = "vector_issued"

    if forced_core:
        # Keep instruction concise but let the LLM restyle if available
        try:
            core = llm_style_atc_from_core(cs, forced_core, VOICE_DIALOGUE, {"phase": phase, "runway": rw, "callsign": cs})
        except Exception:
            core = re.sub(rf'^\s*{re.escape(cs)}\s*,\s*', '', forced_core, flags=re.I)

        atc_line_full = atc_prefix_and_dedup(cs, core)
        atc_ui = f"{EMOJI_ATC} ATC: {atc_line_full}"
        VOICE_DIALOGUE += f"\n{atc_ui}"
        VOICE_LAST_ATC_STRUCT = _extract_items(strip_atc_ui_to_core(atc_ui, cs))
        atc_audio = tts_elevenlabs(atc_line_full, voice_id=ELEVEN_VOICE_ID_DEFAULT)
        status = "" if atc_audio else ("ATC audio unavailable (check ELEVEN_API_KEY)" if _has_eleven else "Text OK — TTS not configured.")
        monitor_header("XAION SIM END (VOICE CTX)")
        return pt, "None", VOICE_DIALOGUE, (atc_audio or None), status, snapshot_debug_dump(), None

    # ---------- Generic readback-correct when it matches our last struct ----------
    if VOICE_LAST_ATC_STRUCT and _pilot_covers_items(pt, VOICE_LAST_ATC_STRUCT):
        atc_line = atc_prefix_and_dedup(cs, f"{cs}, readback correct.")
        atc_ui = f"{EMOJI_ATC} ATC: {atc_line}"
        VOICE_DIALOGUE += f"\n{atc_ui}"
        atc_audio = tts_elevenlabs(atc_line, voice_id=ELEVEN_VOICE_ID_DEFAULT)
        status = "" if atc_audio else ("ATC audio unavailable (check ELEVEN_API_KEY)" if _has_eleven else "Text OK — TTS not configured.")
        monitor_header("XAION SIM END (VOICE CTX)")
        return pt, "None", VOICE_DIALOGUE, (atc_audio or None), status, snapshot_debug_dump(), None

    # ---------- Fallback: ATC LLM (Approach) / deterministic (Ground) ----------
    if phase == "approach":
        _ = monitor_decide_next_role(VOICE_DIALOGUE, phase, cs)
        monitor_block("ATC", phase, rw)
        atc_line_full = gen_once("ATC", VOICE_DIALOGUE, ctx)  # branded inside
        atc_ui = f"{EMOJI_ATC} ATC: {atc_line_full}"
        VOICE_DIALOGUE += f"\n{atc_ui}"
        _ctx_dump_lines(ctx_line, dialogue_text=VOICE_DIALOGUE)
        VOICE_LAST_ATC_STRUCT = _extract_items(strip_atc_ui_to_core(atc_ui, cs))
        atc_audio = tts_elevenlabs(atc_line_full, voice_id=ELEVEN_VOICE_ID_DEFAULT)
        status = "" if atc_audio else ("ATC audio unavailable (check ELEVEN_API_KEY)" if _has_eleven else "Text OK — TTS not configured.")
        monitor_header("XAION SIM END (VOICE CTX)")
        return pt, "None", VOICE_DIALOGUE, (atc_audio or None), status, snapshot_debug_dump(), None
    else:
        # ground deterministic advance (never vectors on ground)
        if st.get("stage") is None:
            core = _compose_taxi_to_runway(cs, rw, st.get("gate")); st["stage"]="taxi_issued"
        elif st.get("stage") == "taxi_issued":
            core = f"{cs}, contact Tower {KGSO_TOWER_FREQ}."; st["stage"]="handoff_twr"
        else:
            core = f"{cs}, cleared for takeoff Runway {rw}."; st["stage"]="cleared_takeoff"

        core = re.sub(rf'^\s*{re.escape(cs)}\s*,\s*', '', core, flags=re.I)
        atc_line_full = atc_prefix_and_dedup(cs, core)
        atc_ui = f"{EMOJI_ATC} ATC: {atc_line_full}"
        VOICE_DIALOGUE += f"\n{atc_ui}"
        VOICE_LAST_ATC_STRUCT = _extract_items(strip_atc_ui_to_core(atc_ui, cs))
        atc_audio = tts_elevenlabs(atc_line_full, voice_id=ELEVEN_VOICE_ID_DEFAULT)
        status = "" if atc_audio else ("ATC audio unavailable (check ELEVEN_API_KEY)" if _has_eleven else "Text OK — TTS not configured.")
        monitor_header("XAION SIM END (VOICE CTX)")
        return pt, "None", VOICE_DIALOGUE, (atc_audio or None), status, snapshot_debug_dump(), None


# -------------------------------
# Triplet helpers (ATC LLM → Pilot LLM → ATC confirm)
# -------------------------------
def llm_style_atc_from_core(cs: str, atc_core: str, dialogue: str, ctx: dict) -> str:
    """
    Rephrase a deterministic ATC instruction with the ATC LLM while preserving
    key tokens. RETURNS *CORE WITHOUT CALLSIGN/BRAND* so the caller can prepend
    'CS, This is XAION CONTROL. ' exactly once.
    """
    try:
        # Seed with a plain ATC line that includes the CS, but we'll strip it later.
        forced_dialogue = f"{(dialogue or '').strip()}\n{EMOJI_ATC} ATC: {cs}, {atc_core.strip()}"
        llm_out = gen_once("ATC", forced_dialogue, ctx.copy())  # branded + may include CS
        # 1) remove brand
        llm_out = re.sub(r"(?i)\bThis\s+is\s+XAION\s+CONTROL\.?\s*", "", llm_out or "")
        # 2) remove any role label & emoji
        llm_out = re.sub(r"^\s*(?:🗼\s*)?ATC:\s*", "", llm_out).strip()
        # 3) strip *leading* callsign if present
        core = re.sub(rf"^\s*{re.escape(cs)}\s*,\s*", "", llm_out, flags=re.I).strip()

        # Normalize & trim
        core = clean_response(core, callsign=cs)

        # Validate must-keep tokens from original core
        musts = []
        musts += [t.lower() for t in re.findall(r"\brunway\s*\d{1,2}[lrc]?\b", atc_core, flags=re.I)]
        if re.search(r"(?i)\bhold\s+short\b", atc_core):
            musts.append("hold short")
        musts += [t.lower() for t in re.findall(r"\bcross\s+runway\s*\d{1,2}[lrc]?\b", atc_core, flags=re.I)]
        musts += re.findall(r"\b\d{3}\.\d\b", atc_core)  # frequencies

        low = core.lower()
        if all(tok in low for tok in musts) and core:
            return core
        # Fallback: just return the deterministic core (without CS)
        return re.sub(rf"^\s*{re.escape(cs)}\s*,\s*", "", atc_core.strip(), flags=re.I)
    except Exception:
        return re.sub(rf"^\s*{re.escape(cs)}\s*,\s*", "", atc_core.strip(), flags=re.I)


def llm_atc_confirm_from_core(cs: str, atc_line_core: str, pilot_rb: str, dialogue: str, ctx: dict) -> str:
    """
    Use the ATC LLM to confirm/correct the pilot's readback.
    Returns a *branded* UI line: "CS, This is XAION CONTROL. <confirm/correction>."
    """
    try:
        items = _extract_items(atc_line_core)
        rb_clean = re.sub(r"^\s*(?:🧑‍✈️\s*)?Pilot:\s*", "", (pilot_rb or ""), flags=re.I).strip()

        if not rb_clean or is_ack_only(rb_clean) or REPEAT_RX.search(rb_clean):
            decision = "SAY_AGAIN"
        elif _pilot_covers_items(rb_clean, items):
            decision = "READBACK_CORRECT"
        else:
            decision = "RESTATE"

        prompt = f"""
You are ATC. Confirm or correct the pilot's readback in ONE sentence (<= 18 words).
Hard rules:
- Start with CALLSIGN once (no brand text).
- If DECISION=READBACK_CORRECT → say exactly "readback correct."
- If DECISION=SAY_AGAIN     → say exactly "say again the full instruction."
- If DECISION=RESTATE       → restate the correct instruction in ATC form (include HOLD SHORT/CROSS/RUNWAY and numbers).
- No angle brackets. No meta/explanations.

DECISION: {decision}
ATC_CORE: {cs}, {atc_line_core.strip()}
PILOT_READBACK: {rb_clean}

Output format (strict):
ATC: {cs}, <confirmation/correction>.
""".strip()

        tok = phi4_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=768).to(device)
        out = phi4_model.generate(
            tok.input_ids, attention_mask=tok.attention_mask,
            max_new_tokens=64, do_sample=True, temperature=0.6, top_p=0.9,
            repetition_penalty=1.08, no_repeat_ngram_size=3,
            pad_token_id=phi4_tokenizer.eos_token_id, eos_token_id=phi4_tokenizer.eos_token_id,
        )
        text = phi4_tokenizer.decode(out[0], skip_special_tokens=True)

        # keep only after last "ATC:" if present, remove role tags/brand, clean, and strip placeholders
        if "ATC:" in text:
            text = text.split("ATC:", 1)[-1]
        text = _strip_role_leaks(text)
        text = clean_response(text, callsign=cs)
        text = re.sub(r"<[^>]*>", "", text)              # no angle brackets
        text = re.sub(r"(?i)\binstruction here\b", "", text).strip(" ,.;")

        # Final safety: if too short or empty, fall back
        if len(text) < 6:
            raise ValueError("LLM confirm under-output")

        # Brand once, dedupe any leading callsign
        return atc_prefix_and_dedup(cs, text)

    except Exception:
        # Deterministic fallback
        if _pilot_covers_items(pilot_rb or "", _extract_items(atc_line_core)):
            core = "readback correct."
        elif not (pilot_rb or "").strip():
            core = "say again the full instruction."
        else:
            # restate original authoritative instruction
            core = re.sub(rf"^\s*{re.escape(cs)}\s*,\s*", "", atc_line_core.strip(), flags=re.I)
        return atc_prefix_and_dedup(cs, core)


def _dedup_leading_callsigns(cs: str, txt: str) -> str:
    return re.sub(rf"^\s*(?:{re.escape(cs)}\s*,\s*)+", f"{cs}, ", txt, flags=re.I)

def llm_pilot_readback_from_atc_core(cs: str, atc_line_core: str, dialogue: str, ctx: dict) -> str:
    """
    Pilot LLM readback from a clean ATC core. Bleed-proof; exactly one 'Pilot:'.
    """
    try:
        prompt = f"""
You are the Pilot. Provide ONE short readback (<= 25 words).
- Start exactly with: "Pilot: {cs}, ..."
- Echo ALL pertinent items from the ATC line: taxi route + HOLD SHORT; any CROSS + runway; takeoff/landing + runway; numeric ALT/HDG/SPD; frequency; approach type/runway.
- No meta text. Do NOT include angle brackets or repeat the word "Pilot:" inside the body.

ATC_CORE (authoritative):
[{cs}, {atc_line_core.strip()}]

Output (example—do NOT copy literally):
Pilot: {cs}, <readback>.
""".strip()

        tok = phi4_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=768).to(device)
        out = phi4_model.generate(
            tok.input_ids, attention_mask=tok.attention_mask,
            max_new_tokens=64, do_sample=True, temperature=0.6, top_p=0.9,
            repetition_penalty=1.1, no_repeat_ngram_size=3,
            pad_token_id=phi4_tokenizer.eos_token_id, eos_token_id=phi4_tokenizer.eos_token_id,
        )
        text = phi4_tokenizer.decode(out[0], skip_special_tokens=True).strip()

        # De-scaffold: keep only content after the LAST 'Pilot:' if present
        if "Pilot:" in text:
            text = "Pilot:" + text.split("Pilot:", maxsplit=1)[-1]
        # Remove ANY number of leading 'Pilot:' tokens and rebuild once
        text = re.sub(r"^(?:\s*Pilot:\s*)+", "", text, flags=re.I).strip()
        text = re.sub(r"<[^>]*>", "", text)  # strip any angle-bracket placeholders

        # Ensure exactly one header + one callsign
        if re.match(rf"^\s*{re.escape(cs)}\s*,", text, flags=re.I):
            body = text
        else:
            body = f"{cs}, {text}"
        body = _dedup_leading_callsigns(cs, body)
        text = f"Pilot: {body}"

        # Final clean + coverage validation
        text = clean_response(text, callsign=cs)
        items = _extract_items(atc_line_core)
        if not _pilot_covers_items(text, items):
            return _pilot_readback_from_atc_text(cs, f"{cs}, {atc_line_core.strip()}")

        if cs.upper() not in text.upper():
            text = f"{text.rstrip('.').rstrip(',')}, {cs}."
        return text

    except Exception:
        return _pilot_readback_from_atc_text(cs, f"{cs}, {atc_line_core.strip()}")





def emit_llm_triplet(cs: str, atc_core: str, phase: str, rw: str, dialogue: str):
    """
    ATC LLM -> Pilot LLM -> ATC confirm
    Returns list of (role, ui_text, tts_text).
    """
    ctx = {"phase": phase, "runway": rw, "callsign": cs}

    _ = monitor_decide_next_role(dialogue, phase, cs)

    # ATC
    atc_line_core = llm_style_atc_from_core(cs, atc_core, dialogue, ctx)  # "CS, …"
    atc_body = re.sub(rf"^\s*{re.escape(cs)}\s*,\s*", "", atc_line_core, flags=re.I)
    atc_ui_line = xaion_prefix(cs) + atc_body
    atc_ui = f"{EMOJI_ATC} ATC: {atc_ui_line}"
    out = [("atc", atc_ui, atc_ui_line)]

    _ = monitor_decide_next_role(dialogue + "\n" + atc_ui, phase, cs)

    # Pilot
    pilot_rb = llm_pilot_readback_from_atc_core(cs, atc_line_core, dialogue + "\n" + atc_ui, ctx)
    pilot_ui = f"{EMOJI_PILOT} {pilot_rb}"
    pilot_tts = re.sub(r"^\s*Pilot:\s*", "", pilot_rb, flags=re.I)
    pilot_tts = _dedup_leading_callsigns(cs, pilot_tts)
    out.append(("pilot", pilot_ui, pilot_tts))

    # Confirm / restate
    # items = _extract_items(atc_line_core)
    # ok = _pilot_covers_items(pilot_rb, items)
    # conf_core = f"{cs}, readback correct." if ok else f"{cs}, say again the full instruction."
    # conf_body = re.sub(rf"^\s*{re.escape(cs)}\s*,\s*", "", conf_core, flags=re.I)
    # conf_ui_line = xaion_prefix(cs) + conf_body
    # conf_ui = f"{EMOJI_ATC} ATC: {conf_ui_line}"
    # out.append(("atc", conf_ui, conf_ui_line))


    # NEW (LLM-confirm with fallback + brand/dedupe inside)
    conf_ui_line = llm_atc_confirm_from_core(cs, atc_line_core, pilot_rb, dialogue + "\n" + atc_ui + "\n" + pilot_ui, {"phase": phase, "runway": rw, "callsign": cs})
    conf_ui = f"{EMOJI_ATC} ATC: {conf_ui_line}"


    return out



def _emit_triplet(cs: str, atc_core_maybe_cs: str, phase: str, rw: str):
    core_no_cs = strip_leading_callsign(atc_core_maybe_cs, cs)
    atc_ui = f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}{core_no_cs}"
    atc_ui = dedup_callsign_runs(atc_ui, cs, "ATC")
    pilot_rb = _pilot_readback_from_atc_text(cs, f"{cs}, {core_no_cs}")
    pilot_ui = dedup_callsign_runs(pilot_rb, cs, "PILOT")
    conf_core = f"{cs}, readback correct."
    conf_ui = f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}readback correct."
    conf_ui = dedup_callsign_runs(conf_ui, cs, "ATC")
    return [("atc", atc_ui, f"{cs}, {core_no_cs}"),
            ("pilot", pilot_ui, pilot_rb),
            ("atc", conf_ui, conf_core)]


def _advance_takeoff(cs, r, gate, rw, dialogue_for_llm: str):
    st = CALLSTATE.setdefault(cs, {"phase": "ground", "gate": gate, "runway": rw, "stage": None, "last_dt": r.get("__dt")})
    out = []

    now_dt  = r.get("__dt")
    prev_dt = st.get("last_dt")
    st["last_dt"] = now_dt

    if st["stage"] is None:
        atc_core = _compose_taxi_to_runway(cs, rw, gate)
        out.extend(emit_llm_triplet(cs, atc_core, "ground", rw, dialogue_for_llm))
        st["stage"] = "taxi_issued"
        return out

    if st["stage"] == "taxi_issued":
        atc_core = f"{cs}, contact Tower {KGSO_TOWER_FREQ}."
        out.extend(emit_llm_triplet(cs, atc_core, "ground", rw, dialogue_for_llm))
        st["stage"] = "handoff_twr"
        return out

    if st["stage"] == "handoff_twr":
        # If ~30s or more passed since handoff, clear for takeoff
        elapsed = (now_dt - prev_dt).total_seconds() if (now_dt is not None and prev_dt is not None) else 30.0
        if elapsed >= 30.0:
            atc_core = f"{cs}, cleared for takeoff Runway {rw}."
            out.extend(emit_llm_triplet(cs, atc_core, "ground", rw, dialogue_for_llm))
            st["stage"] = "cleared_takeoff"
            return out

    return out


def _advance_landing(cs, r, gate_hint, rw, dialogue_for_llm: str):
    st = CALLSTATE.setdefault(cs, {"phase": "approach", "gate": gate_hint, "runway": rw, "stage": None, "last_dt": r.get("__dt")})
    out = []

    d = _row_dist_nm(r)            # nm
    a = r.get("alt_geom_ft")       # ft
    og = bool(r.get("on_ground", False))

    # Early landing clearance
    if st["stage"] is None and (d is not None and d <= 2.5) and (a is None or a <= 3000):
        atc_core = f"{cs}, cleared to land Runway {rw}."
        out.extend(emit_llm_triplet(cs, atc_core, "approach", rw, dialogue_for_llm))
        st["stage"] = "cleared_land"
        return out

    if st["stage"] is None:
        final_hdg = _hdg_to_runway(rw)
        atc_core = f"{cs}, turn heading {int(final_hdg)}, maintain 2000 until established."
        out.extend(emit_llm_triplet(cs, atc_core, "approach", rw, dialogue_for_llm))
        st["stage"] = "vector_issued"
        return out

    if st["stage"] == "vector_issued" and (d is not None and d <= 7.0):
        atc_core = f"{cs}, descend and maintain 1300 until established, cleared ILS Runway {rw}."
        out.extend(emit_llm_triplet(cs, atc_core, "approach", rw, dialogue_for_llm))
        st["stage"] = "ils_cleared"
        return out

    if st["stage"] in ("vector_issued","ils_cleared") and (d is not None and d <= 2.5):
        atc_core = f"{cs}, cleared to land Runway {rw}."
        out.extend(emit_llm_triplet(cs, atc_core, "approach", rw, dialogue_for_llm))
        st["stage"] = "cleared_land"
        return out

    if st["stage"] == "cleared_land" and og:
        gate_use = st.get("gate") or gate_hint or pick_gate_for_callsign(cs)
        atc_core = _compose_taxi_to_gate(cs, gate_use)
        out.extend(emit_llm_triplet(cs, atc_core, "approach", rw, dialogue_for_llm))
        st["stage"] = "taxi_in_issued"
        return out

    return out


# -------------------------------
# Full Simulation (state-driven, Pilot-first per event)
# -------------------------------
def build_events_in_window_seconds(start_idx: int, seconds: float, callsign_only: str | None = None):
    start_dt = flat_df.loc[start_idx, "__dt"]
    if pd.isna(start_dt): return []
    end_dt = start_dt + pd.Timedelta(seconds=seconds)
    mask = (flat_df["__dt"].notna()) & (flat_df["__dt"] >= start_dt) & (flat_df["__dt"] <= end_dt)

    w = flat_df[mask].copy().sort_values("__dt")
    w["__safe_cs"] = w.apply(_safe_callsign_from_row, axis=1)
    if callsign_only:
        mask_cs = w["__safe_cs"].str.upper() == str(callsign_only).upper()
        w = w[mask_cs]

    events = []
    for i in w.index:
        r = flat_df.loc[i]
        cs = _safe_callsign_from_row(r)
        if not cs:
            continue
        dt_show = r.get("parent_dt", r.get("DT Time", "Unknown"))
        pilot_line = fmt_takeoff_from_idx(i) if r.get("on_ground", False) else fmt_landing_from_idx(i)
        if not pilot_line:
            continue
        events.append((dt_show, i, pilot_line))
    return events

def _pilot_seed_for_state(cs: str, phase: str, rw: str, idx: int) -> str:
    st = CALLSTATE.get(cs, {}) or {}
    stage = st.get("stage")

    if phase == "ground":
        if stage in (None,):  # first contact with Ground
            return fmt_takeoff_from_idx(idx)
        if stage in ("taxi_issued", "handoff_twr"):  # we should be at/near the hold short, talking to Tower
            return f"Greensboro Tower, {cs}, holding short Runway {rw}, ready."
        if stage == "cleared_takeoff":
            return f"{cs}, rolling Runway {rw}, {cs}."
        return fmt_takeoff_from_idx(idx)

    # approach
    if stage in (None,):  # initial approach call
        return fmt_landing_from_idx(idx)
    if stage == "vector_issued":
        return f"{cs}, turning heading {int(_hdg_to_runway(rw))}, maintaining 2000, {cs}."
    if stage == "ils_cleared":
        return f"{cs}, established, {cs}."
    if stage == "cleared_land":
        return f"{cs}, cleared to land Runway {rw}, {cs}."
    return fmt_landing_from_idx(idx)


def run_full_simulation(comm_type, scenario_text, seconds: float, stop_flag: dict):
    monitor_header("XAION SIM START (FULL)")
    CONTEXT_HISTORY.clear()
    CALLSTATE.clear()

    options = DT_SCENARIOS.get(comm_type, [])
    if not options or scenario_text not in options:
        debug_line("[FULLSIM] Invalid scenario selection.")
        yield "❌", "None", "❌ Select a scenario from the list.", None, "", snapshot_debug_dump(), None
        monitor_header("XAION SIM END (FULL)")
        return

    i = options.index(scenario_text)
    start_idx = DT_INDEX_MAP[comm_type][i]
    start_row = flat_df.loc[start_idx]
    locked_cs = _safe_callsign_from_row(start_row)

    # one call sign or multiple

    # events = build_events_in_window_seconds(start_idx, seconds, callsign_only=locked_cs)
    events = build_events_in_window_seconds(start_idx, seconds, callsign_only=None)

    dialogue_accum = ""
    for dt_show, idx, pilot_seed in events:
        if stop_flag.get("stop"):
            debug_line("[FULLSIM] Stop requested.")
            break

        r = flat_df.loc[idx]
        cs = _safe_callsign_from_row(r) or "UNKNOWN"
        gate_hint = _gate_from_text(pilot_seed) or pick_gate_for_callsign(cs)
        rw = runway_from_heading(r.get("track_deg")) or DEFAULT_RWY_PRIMARY
        phase = "ground" if r.get("on_ground", False) else "approach"

        # Simulated Pilot seed
        monitor_block("Pilot", phase, rw)
        ctx_line = context_frame_from_idx(idx, push=True)
        pilot_seed_default = fmt_takeoff_from_idx(idx) if r.get("on_ground", False) else fmt_landing_from_idx(idx)
        pilot_seed = _pilot_seed_for_state(cs, phase, rw, idx) or pilot_seed_default
        pilot_ui = f"{EMOJI_PILOT} Pilot: {pilot_seed}"

        dialogue_accum += (("\n\n🕒 [" + dt_show + "] ") if dt_show else "\n\n") + pilot_ui
        _ctx_dump_lines(ctx_line, dialogue_text=dialogue_accum)

        pilot_audio = tts_elevenlabs(pilot_seed, voice_id=PILOT_VOICE_ID) if _has_eleven else None
        yield pilot_seed, "None", dialogue_accum.strip(), None, "", snapshot_debug_dump(), pilot_audio
        _sleep_for_audio(pilot_audio)  # <<< prevent overlap

        # Keep monitor active
        _ = monitor_decide_next_role(dialogue_accum, phase, cs)

        # Advance with LLM triplets
        emitted = _advance_takeoff(cs, r, gate_hint, rw, dialogue_accum) if phase == "ground" else _advance_landing(cs, r, gate_hint, rw, dialogue_accum)
        if not emitted:
            core = _rule_based_atc_for_row(r, cs, rw)
            emitted = emit_llm_triplet(cs, core, phase, rw, dialogue_accum)

        if emitted:
            first_atc = next((e for e in emitted if e[0] == "atc"), None)
            pilot_e = next((e for e in emitted if e[0] == "pilot"), None)
            atc_core = strip_atc_ui_to_core(first_atc[1], cs) if first_atc else ""
            items = _extract_items(atc_core)
            ok = _pilot_covers_items(pilot_e[1] if pilot_e else "", items)

            conf_core = f"{cs}, {'readback correct.' if ok else 'say again the full instruction.'}"
            conf_body = re.sub(rf"^\s*{re.escape(cs)}\s*,\s*", "", conf_core, flags=re.I)
            conf_ui_line = xaion_prefix(cs) + conf_body
            conf_tuple = ("atc", f"{EMOJI_ATC} ATC: {conf_ui_line}", conf_core)

            if emitted[-1][0] != "atc":  # if the last line isn't the confirm, add it
                emitted.append(conf_tuple)


        for role, ui_text, tts_text in emitted:
            monitor_block("Pilot" if role.startswith("pilot") else "ATC", phase, rw)
            _ctx_dump_lines(ctx_line, dialogue_text=dialogue_accum)

            dialogue_accum += (("\n\n🕒 [" + dt_show + "] ") if dt_show else "\n\n") + ui_text
            if role.startswith("pilot"):
                audio = tts_elevenlabs(tts_text, voice_id=PILOT_VOICE_ID) if (_has_eleven and tts_text) else None
                yield pilot_seed, "None", dialogue_accum.strip(), None, "", snapshot_debug_dump(), audio
                _sleep_for_audio(audio)  # <<< prevent overlap
            else:
                audio = tts_elevenlabs(tts_text, voice_id=ELEVEN_VOICE_ID_DEFAULT) if (_has_eleven and tts_text) else None
                status = "" if audio else ("ATC audio unavailable (check ELEVEN_API_KEY)" if _has_eleven else "Text OK — TTS not configured.")
                yield pilot_seed, "None", dialogue_accum.strip(), audio, status, snapshot_debug_dump(), None
                _sleep_for_audio(audio)  # <<< prevent overlap

            # Keep Monitor engaged between lines
            _ = monitor_decide_next_role(dialogue_accum, phase, cs)

    monitor_header("XAION SIM END (FULL)")

# -------------------------------
# Single Handoff Simulation
# -------------------------------
def single_handoff(comm_type, scenario_text):
    # Fresh run state (ok to clear here; keep state across steps within this function)
    CONTEXT_HISTORY.clear()
    CALLSTATE.clear()

    options = DT_SCENARIOS.get(comm_type, [])
    if not options or scenario_text not in options:
        debug_line("[SINGLE] Invalid scenario selection.")
        yield "❌", "None", "❌ Select a scenario from the list.", None, "", snapshot_debug_dump(), None
        return

    i = options.index(scenario_text)
    flat_idx = DT_INDEX_MAP[comm_type][i]
    r = flat_df.loc[flat_idx]

    cs = (_safe_callsign_from_row(r) or "UNKNOWN").strip()
    rw = runway_from_heading(r.get("track_deg")) or DEFAULT_RWY_PRIMARY
    gate = _gate_from_text(scenario_text) or pick_gate_for_callsign(cs)
    dt_show = r.get("parent_dt", r.get("DT Time", "Unknown"))
    phase = "ground" if r.get("on_ground", False) else "approach"
    ground = bool(r.get("on_ground", False))

    # Seed pilot transmission
    dialogue = ""
    monitor_header("XAION SIM START (SINGLE HANDOFF)")
    monitor_block("Pilot", phase, rw)
    ctx_line = context_frame_from_idx(flat_idx, push=True)

    pilot_line = scenario_text
    pilot_ui = f"{EMOJI_PILOT} Pilot: {pilot_line}"
    dialogue += pilot_ui
    _ctx_dump_lines(ctx_line, dialogue_text=dialogue)

    pilot_audio = tts_elevenlabs(pilot_line, voice_id=PILOT_VOICE_ID) if _has_eleven else None
    yield pilot_line, "None", dialogue, None, "", snapshot_debug_dump(), pilot_audio
    _sleep_for_audio(pilot_audio)

    # Keep Monitor active
    _ = monitor_decide_next_role(dialogue, phase, cs)

    # Aim for: ground -> taxi + confirm, handoff + confirm, takeoff + confirm
    #          approach -> vector/ILS + confirm, land + confirm
    max_steps = 3 if ground else 2

    for _ in range(max_steps):
        # Advance scenario; each call should return a triplet (ATC -> Pilot -> ATC confirm)
        stage_emits = _advance_takeoff(cs, r, gate, rw, dialogue) if ground \
                      else _advance_landing(cs, r, gate, rw, dialogue)

        # Fallback if nothing emitted this step
        if not stage_emits:
            core = _rule_based_atc_for_row(r, cs, rw)
            stage_emits = emit_llm_triplet(cs, core, phase, rw, dialogue)

        # --- Safety confirm injection bookkeeping (in case a confirm is missing) ---
        saw_pilot = False
        saw_confirm = False
        last_pilot_text = ""
        last_atc_core = ""  # ATC core (no role/brand/leading CS) for this stage

        # Yield the stage items
        for role, ui_text, tts_text in stage_emits:
            monitor_block("Pilot" if role.startswith("pilot") else "ATC", phase, rw)
            _ctx_dump_lines(ctx_line, dialogue_text=dialogue)

            dialogue = f"{dialogue}\n\n🕒 [{dt_show}] {ui_text}"

            if role.startswith("pilot"):
                # Track pilot readback content for validator
                saw_pilot = True
                # Strip leading role for cleaner validation (numbers/keywords still match either way)
                last_pilot_text = strip_role_prefix(ui_text)
                audio = tts_elevenlabs(tts_text, voice_id=PILOT_VOICE_ID) if (_has_eleven and tts_text) else None
                yield pilot_line, "None", dialogue, None, "", snapshot_debug_dump(), audio
            else:
                # ATC lines: capture core, detect if this line is already a confirm
                atc_core = strip_atc_ui_to_core(ui_text, cs)
                if atc_core:
                    last_atc_core = atc_core
                if re.search(r"(?i)\breadback\s+correct\b", ui_text) or re.search(r"(?i)say\s+again", ui_text):
                    saw_confirm = True

                audio = tts_elevenlabs(tts_text, voice_id=ELEVEN_VOICE_ID_DEFAULT) if (_has_eleven and tts_text) else None
                yield pilot_line, "None", dialogue, audio, "", snapshot_debug_dump(), None

            _sleep_for_audio(audio)
            _ = monitor_decide_next_role(dialogue, phase, cs)

        # If the triplet didn't include a confirm but we have a pilot readback + an ATC core, inject confirm now
        if saw_pilot and not saw_confirm and last_atc_core:
            items = _extract_items(last_atc_core)
            if _pilot_covers_items(last_pilot_text, items):
                conf_ui_line = f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}readback correct."
                dialogue = f"{dialogue}\n\n🕒 [{dt_show}] {conf_ui_line}"
                conf_tts = f"{cs}, readback correct."
                audio = tts_elevenlabs(conf_tts, voice_id=ELEVEN_VOICE_ID_DEFAULT) if _has_eleven else None
                yield pilot_line, "None", dialogue, audio, "", snapshot_debug_dump(), None
                _sleep_for_audio(audio)
            elif _items_nonempty(items):
                # If items existed but weren't covered, explicitly ask for a full readback (keeps the flow honest)
                conf_ui_line = f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}say again the full instruction."
                dialogue = f"{dialogue}\n\n🕒 [{dt_show}] {conf_ui_line}"
                conf_tts = f"{cs}, say again the full instruction."
                audio = tts_elevenlabs(conf_tts, voice_id=ELEVEN_VOICE_ID_DEFAULT) if _has_eleven else None
                yield pilot_line, "None", dialogue, audio, "", snapshot_debug_dump(), None
                _sleep_for_audio(audio)

        # Stop once we’ve reached the natural end of the exchange
        st = CALLSTATE.get(cs, {})
        if st.get("stage") in ("cleared_takeoff", "cleared_land", "taxi_in_issued"):
            break

    monitor_header("XAION SIM END (SINGLE HANDOFF)")



# ===============================
# XAION_CONTROL_ST_V56.py — Part 3/3
# Unified run handler + Gradio UI (Pilot transcript visible only in Vocal Input)
# ===============================

# -------------------------------
# Unified run handler
# -------------------------------
def run_simulation(mode, comm_type, scenario_text, vocal_text, fullsim_window, stop_state):
    """
    Central dispatcher for all three modes.
    Ensures pilot transcript only appears in Vocal Input:
      - In non-vocal modes we never echo pilot freeform input to the "Pilot Input" box.
      - The helper Markdown is hidden by the UI toggle below.
    """
    stop_state["stop"] = False
    debug_line(f"[RUN] mode={mode}")
    # Initial 'pulse' so UI clears/status shows
    yield (vocal_text or ""), "None", "", None, "Starting…", snapshot_debug_dump(), None

    if mode == "Vocal Input":
        pt, anomaly, dialogue, atc_audio, status, dbg, _ = voice_step((vocal_text or "").strip())
        # First frame: show text responses; defer audio next yield
        yield pt, anomaly, dialogue, None, status, dbg, None
        time.sleep(max(0.0, AUDIO_GAP_SEC))
        # Second frame: deliver audio (if any)
        yield pt, anomaly, dialogue, (atc_audio or None), status, snapshot_debug_dump(), None
        return

    # Non-vocal modes: do NOT surface any free-typed pilot transcript in the "Pilot Input" output box.
    blank_pilot = ""

    if mode == "Simulated Dialogue (Single Handoff)":
        reset_voice_dialogue()
        for step in single_handoff(comm_type, scenario_text):
            # step = (pilot_line, anomaly, dialogue, atc_audio, status, dbg, pilot_audio)
            pilot_line, anomaly, dialogue, atc_audio, status, dbg, pilot_audio = step
            yield blank_pilot, anomaly, dialogue, atc_audio, status, dbg, pilot_audio
        return

    # Full Simulation
    dur_map = {"30 sec": 30.0, "1 min": 60.0, "2 min": 120.0}
    seconds = dur_map.get(fullsim_window or "1 min", 60.0)
    for step in run_full_simulation(comm_type, scenario_text, seconds, stop_state):
        pilot_seed, anomaly, dialogue, atc_audio, status, dbg, pilot_audio = step
        yield blank_pilot, anomaly, dialogue, atc_audio, status, dbg, pilot_audio


def stop_simulation(stop_state):
    stop_state["stop"] = True
    debug_line("[CTRL] Stop requested.")
    return "Stopping…"


def _first_open_port(preferred: int, tries: int = 6):
    for i in range(tries):
        p = preferred + i
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
    return preferred


# -------------------------------
# Gradio UI — Pilot transcript helper is ONLY visible in 'Vocal Input'
# -------------------------------
preferred_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
server_port = _first_open_port(preferred_port, tries=6)

with gr.Blocks() as demo:
    gr.Markdown("## 🛫 XAION: eXplainable Aviation Intelligence and Operations Navigator")

    mode_selector = gr.Radio(
        ["Simulated Dialogue (Single Handoff)", "Vocal Input", "Full Simulation"],
        label="Select Operation Mode",
        value="Simulated Dialogue (Single Handoff)"
    )

    comm_type_dropdown = gr.Dropdown(choices=["Takeoff","Landing"], label="Scenario Type", value="Takeoff")
    scenario_dropdown = gr.Dropdown(choices=[], label="Scenario", value=None)

    # Pilot transcript helper (only for Vocal Input)
    helper_title = gr.Markdown("**PILOT TRANSCRIPT (what to say next)**", visible=False)
    example_md = gr.Markdown("", visible=False)

    fullsim_window = gr.Dropdown(choices=["30 sec", "1 min", "2 min"], value="1 min",
                                 label="Full Simulation Window", visible=False)

    with gr.Row(visible=False) as vocal_row:
        mic = gr.Audio(sources=["microphone"], type="numpy", label="🎙️ Speak your pilot call/reply")
        transcript_box = gr.Textbox(label=f"📝 Transcript ({'Whisper API' if USE_WHISPER_API else 'Whisper API-'})",
                                    interactive=True)

    atc_voice_out = gr.Audio(label="🎧 ATC (voice)", type="filepath", autoplay=True)
    pilot_voice_out = gr.Audio(label="🎧 Pilot (voice)", type="filepath", autoplay=True)

    atc_status = gr.Textbox(label="Status", interactive=False, value="", visible=False)
    stop_msg = gr.Textbox(label="Full Simulation Control", interactive=False, visible=False)

    run_btn = gr.Button("Run Simulation / Continue", variant="primary")
    stop_btn = gr.Button("Stop", variant="stop", visible=False)

    out_pilot = gr.Textbox(label=f"{EMOJI_PILOT} Pilot Input")  # kept but blank in non-vocal modes
    out_anomaly = gr.Textbox(label="⚠️ Detected Anomaly")
    out_response = gr.Textbox(label="🧠 ATC / Dialogue", lines=24, interactive=False, show_copy_button=True)
    out_debug = gr.Textbox(label="🪛 LLM Monitor / Debug", lines=28, interactive=False, show_copy_button=True,
                           value=snapshot_debug_dump())

    stop_state = gr.State({"stop": False})

    # --- UI helpers ---
    # Acknowledgement helper (fallback if not defined earlier)
    try:
        pilot_ack  # type: ignore[name-defined]
    except NameError:
        def pilot_ack(cs: str) -> str:
            return f"Roger, {cs}."

    def _ack_line(cs: str) -> str:
        try:
            return pilot_ack(cs)
        except Exception:
            return f"Roger, {cs}."

    def _example_block(comm_type: str, scenario: str | None):
        """Builds the helper Markdown from the selected scenario (only when visible)."""
        if not scenario:
            return ""
        try:
            return _example_from_scenario(comm_type, scenario)
        except Exception:
            return ""

    def on_load():
        opts = DT_SCENARIOS.get("Takeoff", [])
        debug_line("[UI] App loaded.")
        # Default example text (hidden until mode toggle shows it)
        ex = _example_block("Takeoff", opts[0] if opts else "")
        return (gr.update(choices=opts, value=(opts[0] if opts else None)),
                gr.update(value=ex))

    # Set scenario choices and compute example text (hidden by default until 'Vocal Input')
    out_init = demo.load(fn=on_load, inputs=None, outputs=[scenario_dropdown, example_md])

    def on_type_change(comm_type):
        opts = DT_SCENARIOS.get(comm_type, [])
        debug_line(f"[UI] Scenario type change -> {comm_type} (options={len(opts)})")
        ex = _example_block(comm_type, opts[0] if opts else "")
        return (gr.update(choices=opts, value=(opts[0] if opts else None)),
                gr.update(value=ex))

    comm_type_dropdown.change(fn=on_type_change,
                              inputs=comm_type_dropdown,
                              outputs=[scenario_dropdown, example_md])

    def on_scenario_change(comm_type, scenario_sel):
        debug_line("[UI] Scenario selected.")
        ex = _example_block(comm_type, scenario_sel)
        # Update the helper markdown and clear the transcript box
        return gr.update(value=ex), gr.update(value="")

    scenario_dropdown.change(
        fn=on_scenario_change,
        inputs=[comm_type_dropdown, scenario_dropdown],
        outputs=[example_md, transcript_box],
    )

    # Because Gradio needs the component object, wire the transcript reset correctly:
    scenario_dropdown.change(fn=lambda _: gr.update(value=""),
                             inputs=scenario_dropdown,
                             outputs=transcript_box)

    # Mic -> transcript live update
    def _on_audio(audio, current_text):
        text = transcribe_audio(audio)
        debug_line(f"[VOICE] Mic updated; transcript len={len(text or '')}.")
        return text or current_text or ""
    mic.change(fn=_on_audio, inputs=[mic, transcript_box], outputs=transcript_box)

    # Toggle visibility depending on mode
    def _toggle_rows(mode):
        vocal_vis = (mode == "Vocal Input")
        fs_vis = (mode == "Full Simulation")
        pilot_voice_vis = (mode != "Vocal Input")
        debug_line(f"[UI] Mode -> {mode}")
        return (
            gr.update(visible=vocal_vis),   # vocal_row
            gr.update(visible=vocal_vis),   # helper_title
            gr.update(visible=vocal_vis),   # example_md
            gr.update(visible=fs_vis),      # fullsim_window
            gr.update(visible=fs_vis),      # atc_status (re-used for FS status)
            gr.update(visible=fs_vis),      # stop_btn
            gr.update(visible=pilot_voice_vis),  # pilot_voice_out (only sim modes)
        )

    mode_selector.change(
        fn=_toggle_rows,
        inputs=mode_selector,
        outputs=[vocal_row, helper_title, example_md, fullsim_window, atc_status, stop_btn, pilot_voice_out],
    )

    # Changing mode should not carry the transcript over
    mode_selector.change(fn=lambda _: gr.update(value=""), inputs=mode_selector, outputs=transcript_box)

    # Wrapper around run_simulation that guarantees the pilot box is blank in non-vocal modes
    def _run_and_debug(mode, comm_type, scenario, transcript, window, state):
        for (a, b, c, d, e, f, g) in run_simulation(mode, comm_type, scenario, transcript, window, state):
            # a == pilot text; replace with "" unless Vocal Input mode
            a_out = a if mode == "Vocal Input" else ""
            yield a_out, b, c, d, e, snapshot_debug_dump(), g

    run_btn.click(
        fn=_run_and_debug,
        inputs=[mode_selector, comm_type_dropdown, scenario_dropdown, transcript_box, fullsim_window, stop_state],
        outputs=[out_pilot, out_anomaly, out_response, atc_voice_out, atc_status, out_debug, pilot_voice_out],
        api_name="run_simulation"
    )

    stop_btn.click(fn=stop_simulation, inputs=[stop_state], outputs=stop_msg)

# Final init messages
debug_line(f"* Whisper mode: {'API' if USE_WHISPER_API else 'Local'}")
debug_line(f"* ElevenLabs: {'ON' if _has_eleven else 'OFF'}")

server_port = int(os.getenv("GRADIO_SERVER_PORT", server_port))
demo.launch(server_name="0.0.0.0", server_port=server_port, share=True)
