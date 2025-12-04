# ===============================
# XAION_CONTROL_ST_V78.py 
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
DEFAULT_RWY_PRIMARY = "05R"   # 05R/23L is the 10,001' runway

KGSO_TOWER_FREQ = "119.1"

# -------------------------------
# Gates (diagram-based, simplified groupings)
# -------------------------------
KGSO_GATES_WEST  = list(range(1, 40))   # 1–39
KGSO_GATES_EAST  = list(range(40, 50))  # 40–49
ALL_GATES = set(KGSO_GATES_WEST + KGSO_GATES_EAST)

# Taxiway macros for plausible routes by side/runway
TAXI_ROUTES = {
    # To 05R / 23L (east side)
    ("to_runway", "05R", "east_concourse"): ["via A, A2, hold short Runway 05R"],
    ("to_runway", "23L", "east_concourse"): ["via A, A5, hold short Runway 23L"],

    # To 05L / 23R (west side)
    ("to_runway", "05L", "west_concourse"): ["via B, B3, cross A at A4, hold short Runway 05L"],
    ("to_runway", "23R", "west_concourse"): ["via B, B2, hold short Runway 23R"],

    # 14/32 from BOTH sides (add these two lines)
    ("to_runway", "14", "west_concourse"): ["via B, B1, hold short Runway 14"],
    ("to_runway", "32", "east_concourse"): ["via A, A3, hold short Runway 32"],

    # Existing examples
    ("to_runway", "14", "east_concourse"): ["via A, A1, hold short Runway 14"],
    ("to_runway", "32", "west_concourse"): ["via B, B5, hold short Runway 32"],

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

KGSO_DEP_FREQ_A = "124.35"  # sectors 250–049 (wraps across 360→0)
KGSO_DEP_FREQ_B = "126.6"   # sectors 050–249
# -------------------------------
# Helpers / parsing / gates / lanes
# -------------------------------
import json
import os
import math
import time
from typing import List, Dict, Optional

# ---------- 3) Runway config helpers ----------

_DEFAULT_XAION_RUNWAYS = [
    {'id': '05', 'lat': 36.100300, 'lon': -79.937800, 'true_heading': 52.0},
    {'id': '23', 'lat': 36.100300, 'lon': -79.937800, 'true_heading': 232.0},
    {'id': '14', 'lat': 36.087000, 'lon': -79.974000, 'true_heading': 140.0},
    {'id': '32', 'lat': 36.087000, 'lon': -79.974000, 'true_heading': 320.0},
]

XAION_RUNWAYS = list(_DEFAULT_XAION_RUNWAYS)  # global runtime list

def set_xaion_runways(runways: List[Dict]):
    """
    """
    global XAION_RUNWAYS
    safe = []
    for r in runways:
        try:
            safe.append({
                'id': str(r['id']),
                'lat': float(r['lat']),
                'lon': float(r['lon']),
                'true_heading': float(r['true_heading'])
            })
        except Exception:
            # skip malformed runway entries
            continue
    if safe:
        XAION_RUNWAYS = safe
    else:
        # keep previous if input invalid
        print("set_xaion_runways: provided runway list invalid, keeping previous runways.")

def load_runways_from_json(path: str):
    """
    Load runway definitions from a JSON file.
    JSON format: list of {id, lat, lon, true_heading}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Runway JSON not found: {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    set_xaion_runways(data)
    return XAION_RUNWAYS

def load_runways_from_csv(path: str, id_col='id', lat_col='lat', lon_col='lon', heading_col='true_heading'):
    """
    Load runway definitions from a CSV (pandas will be required).
    Expected columns: id_col, lat_col, lon_col, heading_col
    """
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("pandas required to load CSV runways") from e
    df = pd.read_csv(path)
    runways = []
    for _, r in df.iterrows():
        try:
            runways.append({
                'id': str(r[id_col]),
                'lat': float(r[lat_col]),
                'lon': float(r[lon_col]),
                'true_heading': float(r[heading_col])
            })
        except Exception:
            continue
    set_xaion_runways(runways)
    return XAION_RUNWAYS

def get_xaion_runways():

    """Return current runways (runtime copy)."""
    return list(XAION_RUNWAYS)


import math
import time
import json
import ast
from datetime import datetime

# -----------------------
# Utility: safe row accessor (works for dict or pandas Series)
# -----------------------
def _get(row, key, default=None):
    try:
        if hasattr(row, "get"):
            return row.get(key, default)
        else:
            return getattr(row, key, default)
    except Exception:
        try:
            return row[key]
        except Exception:
            return default


# ---------- 4) LLM monitor: prompt builder + multi-backend sender ----------


MONITOR_CONFIG = {
    'mode': 'stub',            # 'stub' | 'openai' | 'hf'
    'openai_api_key': None,    # set if using openai mode
    'openai_model': 'gpt-4o-mini',  # example
    'hf_model_name': None,     # e.g., "phi-2/your-monitor"
    'hf_device': 'cpu',        # 'cpu' or 'cuda'
    'max_tokens': 250,
    'temperature': 0.0,
}

def build_monitor_prompt(payload: dict) -> str:
    """
    Build a concise monitor prompt summarizing payload.
    The LLM monitor uses this to reason about anomalous behavior and recommended actions.
    """
    callsign = payload.get('callsign') or payload.get('flight') or 'UNKNOWN'
    runway = payload.get('inferred_runway') or 'UNKNOWN'
    phase = payload.get('inferred_phase') or 'unknown'
    anomalies = payload.get('anomalies') or []
    raw_summary_lines = []
    # include key fields
    key_fields = ['timestamp','hex','registration','aircraft_type','lat','lon','alt_baro','ground_speed','track','baro_rate','distance_to_gso','airspace_status','seen','sil','nac_p','nac_v','rssi']
    for k in key_fields:
        v = payload.get(k, None)
        raw_summary_lines.append(f"{k}: {v}")
    prompt = (
        f"You are an ATC safety monitor. Evaluate the following aircraft snapshot and produce:\n"
        f"  1) a single-line 'issue' summary (or 'none')\n"
        f"  2) list of anomalous indicators found (JSON array)\n"
        f"  3) recommended quick action(s) for ATC (short, bullet points)\n\n"
        f"Snapshot metadata: callsign={callsign}, inferred_phase={phase}, inferred_runway={runway}\n\n"
        f"Snapshot details:\n" + "\n".join(raw_summary_lines) + "\n\n"
        f"Anomalies detected by heuristics: {json.dumps(anomalies)}\n\n"
        f"Return a JSON object with keys: issue (string), anomalies (array), recommended_actions (array).\n"
        f"Be concise and prefer operationally-safe, conservative actions (e.g., 'confirm readback', 'stop taxi', 'issue go-around').\n"
    )
    return prompt



# Lazy-loaded HF model cache
_HF_MONITOR = {'tokenizer': None, 'model': None}

def _ensure_hf_model_loaded():
    """
    Internal loader for HF model (called when MONITOR_CONFIG['mode']=='hf')
    """
    if MONITOR_CONFIG.get('hf_model_name') is None:
        raise RuntimeError("hf_model_name not configured in MONITOR_CONFIG")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except Exception as e:
        raise RuntimeError("transformers required for hf mode") from e

    if _HF_MONITOR['model'] is None:
        print(f"Loading HF monitor model {MONITOR_CONFIG['hf_model_name']} on {MONITOR_CONFIG['hf_device']}...")
        tok = AutoTokenizer.from_pretrained(MONITOR_CONFIG['hf_model_name'], use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(MONITOR_CONFIG['hf_model_name'], device_map='auto', torch_dtype=None)
        _HF_MONITOR['tokenizer'] = tok
        _HF_MONITOR['model'] = model

def send_to_llm_monitor(payload: dict, llm_client=None, timeout_s: int=10) -> dict:
    """
    Primary monitor call. Mode controlled by MONITOR_CONFIG['mode'].
    Returns a dict with at least keys: {'status': 'ok'|'error', 'parsed': {...}, 'raw': raw_text_or_response}
    """
    mode = MONITOR_CONFIG.get('mode', 'stub')
    prompt = build_monitor_prompt(payload)
    start = time.time()

    if mode == 'stub':
        # simple deterministic response built from anomalies
        anomalies = payload.get('anomalies', []) or []
        issue = 'none' if not anomalies else ' / '.join(anomalies[:3])
        recs = []
        if 'low_sil' in anomalies:
            recs.append("Verify position via secondary sensor; request pilot readback.")
        if 'weak_rssi' in anomalies:
            recs.append("Flag as weak-signal; cross-check with MLAT or radar if available.")
        if 'sudden_large_position_jump' in anomalies:
            recs.append("Treat as possible beacon glitch; instruct pilot to confirm position.")
        if 'unrealistic_groundspeed' in anomalies:
            recs.append("Hold any runway clearances; verify with pilot.")
        if not recs:
            recs = ["No immediate action required."]
        parsed = {'issue': issue, 'anomalies': anomalies, 'recommended_actions': recs}
        return {'status': 'ok', 'parsed': parsed, 'raw': 'stub', 'latency_s': time.time()-start}

    elif mode == 'openai':
        # Attempt to call OpenAI ChatCompletion (or compatible). Requires openai package and API key.
        try:
            import openai
            key = MONITOR_CONFIG.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
            if not key:
                raise RuntimeError("OpenAI API key not provided")
            openai.api_key = key
            model = MONITOR_CONFIG.get('openai_model', 'gpt-4o-mini')
            # Use ChatCompletion if available; fallback to Completion
            try:
                resp = openai.ChatCompletion.create(
                    model=model,
                    messages=[{'role':'system','content':'You are an ATC safety monitor.'},
                              {'role':'user','content': prompt}],
                    temperature=MONITOR_CONFIG.get('temperature', 0.0),
                    max_tokens=MONITOR_CONFIG.get('max_tokens', 250),
                )
                text = resp['choices'][0]['message']['content']
            except Exception:
                resp = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=MONITOR_CONFIG.get('max_tokens', 250),
                    temperature=MONITOR_CONFIG.get('temperature', 0.0),
                )
                text = resp['choices'][0]['text']
            # try to parse JSON out of model output
            parsed = _try_parse_json_from_text(text)
            return {'status': 'ok', 'parsed': parsed, 'raw': text, 'latency_s': time.time()-start}
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'raw': None, 'latency_s': time.time()-start}

    elif mode == 'hf':
        try:
            _ensure_hf_model_loaded()
            tok = _HF_MONITOR['tokenizer']
            model = _HF_MONITOR['model']
            # build simple generation prompt
            prompt_text = prompt + "\n\nAnswer (JSON):"
            inputs = tok(prompt_text, return_tensors="pt")
            # move inputs to model device if needed (transformers handles with device_map='auto')
            gen = model.generate(**inputs, max_new_tokens=MONITOR_CONFIG.get('max_tokens', 250), do_sample=False)
            out = tok.batch_decode(gen, skip_special_tokens=True)[0]
            parsed = _try_parse_json_from_text(out)
            return {'status': 'ok', 'parsed': parsed, 'raw': out, 'latency_s': time.time()-start}
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'raw': None, 'latency_s': time.time()-start}

    else:
        return {'status': 'error', 'error': f"Unknown monitor mode: {mode}", 'latency_s': time.time()-start}

def _try_parse_json_from_text(text: Optional[str]):
    """
    Attempt to extract a single JSON object from text. If found, return the parsed object.
    Otherwise return {'_raw_text': text}
    """
    if not text:
        return {'_raw_text': ''}
    # quick heuristic: find first { ... } block
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            # try fix common issues (single quotes -> double)
            try:
                fixed = candidate.replace("'", '"')
                return json.loads(fixed)
            except Exception:
                pass
    # fallback: return raw text
    return {'_raw_text': text}


# -----------------------
# Haversine - nautical miles
# -----------------------
def haversine_nm(lat1, lon1, lat2, lon2):
    # returns distance in nautical miles
    R = 6371.0  # km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2.0) ** 2)
    km = 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    nm = km * 0.539957
    return nm

# -----------------------
# Each runway: {'id':'05/23', 'lat':..., 'lon':..., 'true_heading':...}
# -----------------------
XAION_RUNWAYS = [
    # KGSO runways 
    {'id': '05', 'lat': 36.1003, 'lon': -79.9378, 'true_heading': 052.0},
    {'id': '23', 'lat': 36.1003, 'lon': -79.9378, 'true_heading': 232.0},
    {'id': '14', 'lat': 36.0870, 'lon': -79.9740, 'true_heading': 140.0},
    {'id': '32', 'lat': 36.0870, 'lon': -79.9740, 'true_heading': 320.0},
]

def infer_nearest_runway(lat, lon, track=None):
    """
    Return nearest runway dict and distance_nm and whether track aligned (bool).
    """
    if lat is None or lon is None:
        return None, None, False
    best = None
    best_dist = 1e9
    for r in XAION_RUNWAYS:
        d = haversine_nm(lat, lon, r['lat'], r['lon'])
        if d < best_dist:
            best_dist = d
            best = r
    aligned = False
    if best and track is not None:
        # difference between track and runway heading (wrap)
        dh = abs((track - best['true_heading'] + 180) % 360 - 180)
        aligned = dh <= 30  # within 30 deg considered aligned (tunable)
    return best, best_dist, aligned

# -----------------------
# Phase inference
# -----------------------
def infer_phase(row, prev_row=None):
    """
    Heuristic inference of phase:
      at_gate, pushback, taxi, takeoff_roll, departure_climb, enroute,
      approach, landing_roll, on_ground (generic), unknown
    Returns: inferred_phase (str), confidence (0-1)
    """
    # robustly extract fields
    lat = _get(row, 'Latitude') or _get(row, 'lat') or _get(row, 'latitude')
    lon = _get(row, 'Longitude') or _get(row, 'lon') or _get(row, 'longitude')
    gs = _get(row, 'Ground Speed') or _get(row, 'ground_speed') or _get(row, 'GS')
    alt_baro = _get(row, 'Altitude Barometric') or _get(row, 'Altitude Geometric') or _get(row, 'Altitude')
    seen = _get(row, 'Seen') or _get(row, 'Seen Position') or _get(row, 'SeenPos') or 0
    distance_to_gso = _get(row, 'Distance to GSO') or _get(row, 'distance_to_gso')
    airspace_status = (_get(row, 'Airspace Status') or '').lower() if _get(row, 'Airspace Status') else ''

    # normalize numeric fields
    try:
        gs = float(gs) if gs is not None else 0.0
    except Exception:
        gs = 0.0
    try:
        alt_val = float(alt_baro)
    except Exception:
        try:
            alt_val = float(_get(row, 'Altitude Geometric') or 0.0)
        except Exception:
            alt_val = None

    # default phase
    phase = 'unknown'
    conf = 0.3

    # basic ground vs airborne
    if alt_val is not None and alt_val < 1500 and gs < 60:
        # likely ground/taxi or at gate
        # use distance to nearest runway and RSSI/seen to infer gate vs taxi
        runway, run_dist_nm, aligned = infer_nearest_runway(lat, lon, _get(row, 'Track'))
        if run_dist_nm is not None:
            if run_dist_nm > 1.5 and gs < 10:
                phase = 'at_gate'
                conf = 0.9
            elif run_dist_nm <= 1.5 and gs >= 5:
                phase = 'taxi'
                conf = 0.85
            else:
                phase = 'on_ground'
                conf = 0.6
        else:
            phase = 'on_ground'
            conf = 0.5
    else:
        # airborne heuristics
        if gs is not None and gs > 100:
            # approaching or enroute or climb
            # if descending (baro rate negative) and close-to-airport -> approach
            baro_rate = _get(row, 'Barometric Rate') or _get(row, 'Barometric_Rate') or 0
            try:
                br = float(baro_rate)
            except Exception:
                br = 0
            if br < -300 and (distance_to_gso is not None and float(distance_to_gso) < 50):
                phase = 'approach'
                conf = 0.85
            elif prev_row:
                # look for rapid climb from near ground -> takeoff
                prev_alt = _get(prev_row, 'Altitude Barometric') or _get(prev_row, 'Altitude Geometric') or None
                try:
                    prev_alt = float(prev_alt) if prev_alt is not None else None
                except Exception:
                    prev_alt = None
                if prev_alt is not None and alt_val is not None and alt_val - prev_alt > 500 and gs > 150:
                    phase = 'departure_climb'
                    conf = 0.9
                else:
                    phase = 'enroute'
                    conf = 0.6
            else:
                phase = 'enroute'
                conf = 0.5
        elif gs is not None and 60 <= gs <= 100:
            phase = 'landing_roll' if (alt_val is not None and alt_val < 2000) else 'approach'
            conf = 0.6
        else:
            phase = 'unknown'
            conf = 0.3

    return phase, conf

# -----------------------
# Simple anomaly detectors
# -----------------------
def detect_anomalies(row, prev_row=None):
    """
    Returns list of anomaly strings. Add domain specific rules as needed.
    """
    anomalies = []
    try:
        sil = _get(row, 'SIL Source Integrity Level') or _get(row, 'SIL')
        if sil is not None:
            try:
                if float(sil) < 1.0:
                    anomalies.append('low_sil')
            except Exception:
                pass
    except Exception:
        pass

    try:
        gs = _get(row, 'Ground Speed') or 0.0
        if gs is not None:
            try:
                if float(gs) > 600:
                    anomalies.append('unrealistic_groundspeed')
            except Exception:
                pass
    except Exception:
        pass

    try:
        seen = _get(row, 'Seen') or _get(row, 'Seen Position') or 0.0
        if seen is not None:
            try:
                if float(seen) > 60:
                    anomalies.append('stale_position_reading')
            except Exception:
                pass
    except Exception:
        pass

    # sudden large deviations vs previous row
    if prev_row is not None:
        try:
            lat = float(_get(row, 'Latitude') or 0.0)
            lon = float(_get(row, 'Longitude') or 0.0)
            plat = float(_get(prev_row, 'Latitude') or 0.0)
            plon = float(_get(prev_row, 'Longitude') or 0.0)
            nm = haversine_nm(lat, lon, plat, plon)
            if nm > 5.0:  # moved >5nm between ticks (tunable)
                anomalies.append('sudden_large_position_jump')
        except Exception:
            pass

        try:
            alt = float(_get(row, 'Altitude Barometric') or _get(row, 'Altitude Geometric') or 0.0)
            palt = float(_get(prev_row, 'Altitude Barometric') or _get(prev_row, 'Altitude Geometric') or 0.0)
            if abs(alt - palt) > 5000:
                anomalies.append('sudden_large_alt_change')
        except Exception:
            pass

    return anomalies

# -----------------------
# Build monitor payload to send to LLM monitor
# -----------------------
def build_monitor_payload(row, prev_row=None, inferred_phase=None, inferred_runway=None, anomalies=None):
    """
    Collects the key fields and returns a JSON-serializable dict
    that will be passed to the LLM monitor.
    """
    # normalize row to dict
    try:
        if hasattr(row, "to_dict"):
            r = row.to_dict()
        elif hasattr(row, 'items'):
            r = dict(row)
        else:
            r = dict(row)
    except Exception:
        r = {}

    payload = {
        'timestamp': _get(row, 'Time') or datetime.utcnow().isoformat(),
        'hex': _get(row, 'Hex') or _get(row, 'ICAO'),
        'callsign': (_get(row, 'Flight Number') or _get(row, 'Flight')) and str((_get(row, 'Flight Number') or _get(row, 'Flight'))).strip(),
        'registration': _get(row, 'Aircraft Registration'),
        'aircraft_type': _get(row, 'Aircraft Type'),
        'lat': _get(row, 'Latitude'),
        'lon': _get(row, 'Longitude'),
        'alt_baro': _get(row, 'Altitude Barometric') or _get(row, 'Altitude Geometric'),
        'ground_speed': _get(row, 'Ground Speed'),
        'track': _get(row, 'Track'),
        'baro_rate': _get(row, 'Barometric Rate'),
        'distance_to_gso': _get(row, 'Distance to GSO'),
        'airspace_status': _get(row, 'Airspace Status'),
        'seen': _get(row, 'Seen'),
        'sil': _get(row, 'SIL Source Integrity Level') or _get(row, 'SIL'),
        'nac_p': _get(row, 'NAC P'),
        'nac_v': _get(row, 'NAC V'),
        'rssi': _get(row, 'RSSI'),
        'inferred_phase': inferred_phase,
        'inferred_runway': inferred_runway if inferred_runway else None,
        'anomalies': anomalies or [],
        'raw_row': r,  # small convenience to debug
    }
    return payload

# -----------------------
# Stub: send to LLM monitor 
# ------------------------------------------------------------
def send_to_llm_monitor(payload, llm_client=None):


    try:
        s = json.dumps(payload, default=str)
    except Exception:
        s = str(payload)
    # DEBUG: print or push to UI debug/log
    print("XAION_MONITOR_PAYLOAD:", s)
    # Optionally: return a placeholder decision dict
    return {'status': 'ok', 'issues': payload.get('anomalies', [])}

# -----------------------
# Integration helper that ties everything for a single row
# -----------------------
def process_row_for_monitor(row, prev_row=None, llm_client=None):
    try:
   

        if isinstance(row, dict):
            pass
        else:
            # keep as-is for pd.Series
            pass

        lat = _get(row, 'Latitude') or _get(row, 'lat')
        lon = _get(row, 'Longitude') or _get(row, 'lon')
        track = _get(row, 'Track') or _get(row, 'track')

        # infer nearest runway
        runway, run_dist_nm, aligned = infer_nearest_runway(lat, lon, track)
        inferred_runway = runway['id'] if runway else None

        # infer phase
        inferred_phase, conf = infer_phase(row, prev_row)

        # detect anomalies
        anomalies = detect_anomalies(row, prev_row)

        # build payload
        payload = build_monitor_payload(row, prev_row, inferred_phase=inferred_phase, inferred_runway=inferred_runway, anomalies=anomalies)

        # push to LLM monitor (or print for now)
        monitor_result = send_to_llm_monitor(payload, llm_client=llm_client)

        return payload, monitor_result
    except Exception as e:
        print("process_row_for_monitor error:", str(e))
        return None, None

# End of helper block





def on_run_full_sim_click(window_label: str | int | None = "2 min"):
    """
    Gradio-click handler.
    - window_label is your dropdown value (e.g., '2 min', '60 sec', etc.)
    Returns a tuple shaped like your UI expects:
      (atc_audio_path, pilot_audio_path, pilot_input_text, anomaly, dialogue_text, context_block, debug_dump)
    If your UI wiring differs, adjust the return arity/order accordingly.
    """
    sim_seconds = _parse_fullsim_window_to_seconds(window_label, default_sec=120)

    # Run it (text-only here; you can layer audio playback if desired)
    dialogue = run_full_simulation(sim_seconds=sim_seconds)

    anomaly = "None"  # placeholder — hook in your anomaly detector if you have one
    pilot_input_text = ""  # nothing typed during automated full sim
    atc_audio_path = None  # if you want TTS here, generate and set these
    pilot_audio_path = None

    return atc_audio_path, pilot_audio_path, pilot_input_text, anomaly, dialogue, recent_context_block(3), snapshot_debug_dump()


def build_fullsim_sequence(sim_seconds: int) -> list[int]:
    """
    Use the chosen scenario (SIM_START/LAST_SELECTED_IDX) and produce an ordered list
    of flat_df indices to simulate, guaranteeing the selected aircraft appears first.
    """
    start_idx = SIM_START.get("idx") or globals().get("LAST_SELECTED_IDX")
    if start_idx is None:
        # very defensive fallback: first valid row
        try:
            start_idx = int(flat_df.index[0])
        except Exception:
            return []

    return order_rows_for_full_sim(
        start_idx,
        window_sec=sim_seconds,
        suppress_same_dt_before_selected=True
    )


def _parse_fullsim_window_to_seconds(window_label: str | int | None, default_sec: int = 120) -> int:
    """
    Accepts '30 sec', '1 min', '2 min', an int, or None.
    Returns a positive number of seconds (default 120).
    """
    if window_label is None:
        return default_sec
    if isinstance(window_label, (int, float)) and window_label > 0:
        return int(window_label)
    s = str(window_label).strip().lower()
    if "sec" in s:
        try:
            return max(5, int(re.findall(r"\d+", s)[0]))
        except Exception:
            return default_sec
    if "min" in s:
        try:
            return max(5, int(re.findall(r"\d+", s)[0]) * 60)
        except Exception:
            return default_sec
    try:
        v = int(s)
        return max(5, v)
    except Exception:
        return default_sec



# -------- Full Sim start ordering helpers --------
SIM_START = {"idx": None, "dt": None, "ident": None}

def _selected_ident_at(idx: int) -> str:
    r = flat_df.loc[idx]
    return (_safe_callsign_from_row(r) or str(r.get("ident") or r.get("display_id") or "UNKNOWN")).strip()

def order_rows_for_full_sim(start_idx: int, window_sec: int = 120,
                            suppress_same_dt_before_selected: bool = True) -> list[int]:
    """
    Returns a list of flat_df indices for the simulation window, ordered so that
    the selected row is ALWAYS first. Optionally suppress any other traffic that
    shared the exact same DT and would have appeared before the selected row.

    Usage in your runner:
        seq = order_rows_for_full_sim(LAST_SELECTED_IDX, window_sec)
        for i in seq: ...  # drive sim
    """
    if start_idx is None:
        return []

    t0 = flat_df.loc[start_idx, "__dt"]
    if pd.isna(t0):
        # fallback to whatever is present; still guarantee the chosen row is first
        t0 = flat_df.loc[start_idx, "__dt"] = pd.Timestamp.utcnow()

    t1 = t0 + pd.Timedelta(seconds=window_sec)

    # Window rows
    in_win = flat_df[(flat_df["__dt"] >= t0) & (flat_df["__dt"] <= t1)].copy()
    if in_win.empty:
        return [start_idx]

    # Primary key: selected first; then by dt; then by original index (stable)
    in_win["__pri"] = 1
    in_win.loc[start_idx, "__pri"] = 0
    in_win["__i"] = in_win.index

    if suppress_same_dt_before_selected:
        same_dt = in_win["__dt"] == t0
        # Drop any rows at the same DT whose original index sorts before the selected index
        in_win = in_win[~(same_dt & (in_win["__i"] < start_idx))]

    in_win = in_win.sort_values(by=["__pri", "__dt", "__i"], kind="stable")
    # Ensure selected is absolutely first even after any pandas quirks
    ordered = [start_idx] + [i for i in in_win.index.tolist() if i != start_idx]
    return ordered


def _stamp_block(ui_block: str, ctx: dict) -> str:
    """Stamp each non-empty line in a multi-line UI block."""
    lines = [ln for ln in (ui_block or "").splitlines() if ln.strip()]
    return "\n".join(_stamp_ui(ln, ctx) for ln in lines)


# --- UI → speech cleaning ---
_UI_LEAD_EMOJI_RX = re.compile(r"^\s*(?:🧑‍✈️|🗼)\s*")
def _strip_ui_wrappers_for_tts(s: str) -> str:
    """Remove clock-stamp + role/emoji so TTS never says them."""
    t = _strip_leading_stamps(s)              # removes 🕒 [HHMMSSZ]
    t = _UI_LEAD_EMOJI_RX.sub("", t)          # drops leading role emoji
    t = strip_role_prefix(t)                  # drops 'Pilot:' / 'ATC:'
    t = t.strip()
    return t



# --- Frequency normalizers (Tower + Departure) ---
_DEP_RX       = re.compile(r'(?i)\b(Departure)\s+(?:freq\s*)?(\d{3})(?:\.(\d))?\b')
_TWR_BAD_RX   = re.compile(r'(?i)\b(Tower)\s+119(?:\.1+)?\b')   # Tower 119, Tower 119.1.1, etc.
_TWR_DUP_RX   = re.compile(r'(?i)\b119\.1(?:\.\d+)+\b')         # bare 119.1.1 → 119.1

def _normalize_comm_freqs(text: str, *, hdg: float | None = None, runway: str | None = None) -> str:
    """
    - Forces 'Tower 119' and '119.1.1' → 'Tower 119.1'
    - Fills/repairs 'Departure 126'/'Departure 124' → sector-correct 126.6 or 124.35
      based on heading if available, else runway nominal heading.
    """
    if not text:
        return text

    s = text

    # Tower fixes
    s = _TWR_BAD_RX.sub(lambda m: f"{m.group(1)} {KGSO_TOWER_FREQ}", s)
    s = _TWR_DUP_RX.sub(KGSO_TOWER_FREQ, s)

    # Departure fix (choose best freq by heading, else runway)
    dep = kgso_departure_freq_for_heading(hdg) if hdg is not None \
          else kgso_departure_freq_for_runway(runway or DEFAULT_RWY_PRIMARY)

    s = _DEP_RX.sub(lambda m: f"{m.group(1)} {dep}", s)
    return s

def _normalize_tower_and_runway(text: str) -> str:
    if not text: return text
    text = _TWR_119_RX.sub(lambda m: f"{m.group(1)} Tower {KGSO_TOWER_FREQ}", text)
    text = re.sub(r'(?i)\bTower\s+119\b', f"Tower {KGSO_TOWER_FREQ}", text)
    text = re.sub(r'(?i)\b119\.1(?:\.\d+)+\b', KGSO_TOWER_FREQ, text)  # new: 119.1.1 → 119.1
    text = _RWY_DUPE_RX.sub(r"Runway \1\2", text)  # 05LL → 05L
    return text




# --- Parallel runway helpers (KGSO has 05L/23R and 05R/23L) ---
# --- ATC phrasing normalizers (use AFTER any LLM styling) ---
_TWR_119_RX = re.compile(r'(?i)\b(contact(?:ing)?|switch(?:ing)?)\s+(?:the\s+)?tower\s+119\b')
_MUST_CONTAIN = {
    "contact_tower": ("contact", "tower"),
    "cleared_tkof": ("cleared", "takeoff"),
}

# --- Normalizers---
_TWR_119_CMD_RX  = re.compile(r'(?i)\b(contact(?:ing)?|switch(?:ing)?)\s+(?:the\s+)?tower\s+119\b')
_TWR_119_ANY_RX  = re.compile(r'(?i)\bTower\s+119(?:\.1(?:\.1+)*)?\b')  # collapses 119.1.1 → 119.1
_RWY_DUPE_RX     = re.compile(r'(?i)\bRunway\s*(\d{2})([LRC])\2\b')

def _normalize_tower_and_runway(text: str) -> str:
    if not text:
        return text
    # "contact/switching Tower 119" → "... Tower 119.1"
    text = _TWR_119_CMD_RX.sub(lambda m: f"{m.group(1)} Tower {KGSO_TOWER_FREQ}", text)
    # Any "Tower 119", "Tower 119.1", "Tower 119.1.1..." → "Tower 119.1"
    text = _TWR_119_ANY_RX.sub(f"Tower {KGSO_TOWER_FREQ}", text)
    # Collapse 05RR / 05LL → 05R / 05L
    text = _RWY_DUPE_RX.sub(r"Runway \1\2", text)
    return text


def _require_tokens_or_fallback(styled: str, required: tuple[str,...], fallback_core: str) -> str:
    low = (styled or "").lower()
    return styled if all(tok in low for tok in required) else fallback_core





PARALLEL_RWY = {
    "05": ("05L", "05R"),
    "23": ("23L", "23R"),
}

def _rw_base(rw: str | None) -> str:
    """Return the numeric runway (e.g., '05' for '05R')."""
    if not rw: return ""
    m = re.match(r"^\s*(\d{1,2})", str(rw).upper().strip())
    return (m.group(1) if m else "").zfill(2)

def _rw_with_side(base: str, side_hint: str) -> str:
    """
    For base '05' or '23', pick L/R by side hint:
      - east_concourse → 05R / 23L (A-side closer)
      - west_concourse → 05L / 23R (B-side closer)
    """
    b = _rw_base(base)
    if b not in PARALLEL_BASES:
        return b
    if (side_hint or "").startswith("east"):
        return "05R" if b == "05" else "23L"
    else:
        return "05L" if b == "05" else "23R"


def normalize_lr(rw: str | None, gate: str | None = None, callsign: str | None = None) -> str:
    """
    If runway already has L/R, keep it. If it's 05/23 without side, choose L/R from gate/callsign.
    Otherwise return as-is (e.g., 14/32). Falls back to DEFAULT_RWY_PRIMARY.
    """
    if not rw:
        return DEFAULT_RWY_PRIMARY
    s = str(rw).upper().strip()
    if re.search(r"\d{1,2}[LRC]$", s):  # has side already
        return s
    base = _rw_base(s)
    if base in PARALLEL_BASES:
        side = _side_for_gate(gate) if gate else _side_for_callsign(callsign or "")
        return _rw_with_side(base, side)
    return s

def runway_from_heading_lr(track: float | None, gate: str | None = None, callsign: str | None = None) -> str:
    """
    Pick a base runway from magnetic track, then add L/R if needed.
    """
    base = runway_from_heading(track) or DEFAULT_RWY_PRIMARY
    return normalize_lr(base, gate, callsign)






ROLE_HDR_RX = re.compile(r"^\s*(?:🧑‍✈️\s*)?Pilot:|^\s*(?:🗼\s*)?ATC:", re.I)

def _ui_with_role(role: str, text: str) -> str:
    """Force consistent UI headers + emojis, stripping any existing role tag first."""
    core = ROLE_HDR_RX.sub("", text or "").strip()
    if role.lower().startswith("pilot"):
        return f"{EMOJI_PILOT} Pilot: {core}"
    else:
        return f"{EMOJI_ATC} ATC: {core}"


# --- Time-stamp helpers (place near other utils) ---
import re
from datetime import datetime, timedelta

STAMP_STEP_SEC = 8  # per-line tick within a single event

# strip ANY number of leading stamps like "🕒 [HHMMSSZ]" (even if duplicated)
_LEADING_STAMPS_RX = re.compile(r'^(?:\s*🕒\s*\[\d{6}Z\]\s*)+', re.UNICODE)

def _strip_leading_stamps(s: str) -> str:
    return _LEADING_STAMPS_RX.sub("", s or "").strip()

def _bump_z(base_z: str, step: int) -> str:
    try:
        t0 = datetime.strptime((base_z or "").strip(), "%H%M%SZ")
        t1 = t0 + timedelta(seconds=STAMP_STEP_SEC * int(step))
        return t1.strftime("%H%M%SZ")
    except Exception:
        return base_z or _now_dt_tag()




_STAMP_RX = re.compile(r"^\s*🕒\s*\[\d{6}Z\]\s")

def _ensure_stamped(ui_text: str, dt_show: str) -> str:
    """If a line isn't already 🕒-stamped, add one; otherwise return as-is."""
    return ui_text if _STAMP_RX.match(ui_text or "") else f"🕒 [{dt_show}] {ui_text}"



_DT_RX = re.compile(r"^\s*(\d{6})Z\s*$")

def _now_dt_tag() -> str:
    return time.strftime("%H%M%SZ", time.gmtime())

def _dt_tag_from_ctx(ctx: dict | None) -> str:
    if not ctx: return _now_dt_tag()
    raw = str(ctx.get("dt", "")).strip()
    m = _DT_RX.match(raw)
    return m.group(1) + "Z" if m else _now_dt_tag()

def _stamp_ui(ui_line: str, ctx: dict | None) -> str:
    return f"🕒 [{_dt_tag_from_ctx(ctx)}] {ui_line}"



SAY_AGAIN_RX = re.compile(r"\b(say\s+again|repeat)\b", re.I)

def _synth_pilot_from_last_atc(dialogue: str, cs: str) -> str | None:
    """Build a deterministic pilot readback from the last ATC UI line."""
    last_ui = _last_atc(dialogue)
    if not last_ui:
        return None
    core = strip_atc_ui_to_core(last_ui, cs)
    if not core:
        return None
    return _pilot_readback_from_atc_text(cs, core)


import time as _time
RUNWAY_STATE = {}  # e.g., {"05": {"busy_until": 0.0, "by": "CS"}}

def _runway_is_free(rw: str) -> bool:
    st = RUNWAY_STATE.get(rw, {})
    return _time.time() >= float(st.get("busy_until", 0.0))

def _reserve_runway(rw: str, cs: str, hold_sec: float = 45.0):
    RUNWAY_STATE[rw] = {"busy_until": _time.time() + hold_sec, "by": cs}

def gate_runway_clearance(action: str, rw: str, cs: str) -> tuple[bool, str]:
    """
    action: "takeoff" or "land"
    Returns (ok, alt_core_if_blocked)
    """
    if _runway_is_free(rw):
        # Reserve the runway for a little while when we *do* clear
        _reserve_runway(rw, cs, 55.0 if action == "takeoff" else 85.0)
        return True, ""
    # Block with a safe alternative
    if action == "takeoff":
        return False, f"{cs}, line up and wait Runway {rw}."
    # landing
    return False, f"{cs}, continue, expect late landing clearance."


def kgso_departure_freq_for_runway(rw: str) -> str:
    """
    Use runway's nominal heading if no explicit departure heading is known.
    RWY 05 ~050, 23 ~230, 14 ~140, 32 ~320.
    """
    base = {"05": 50, "23": 230, "14": 140, "32": 320}
    return kgso_departure_freq_for_heading(base.get((rw or "").upper(), 50))



def _maybe_departure_handoff(cs: str, st: dict) -> str | None:
    """
    After we issue takeoff clearance once, hand the pilot to the correct TRIAD DP
    sector frequency based on heading/sectorization. Only do this once.
    """
    if st.get("stage") == "cleared_takeoff" and not st.get("handoff_dep_done"):
        st["handoff_dep_done"] = True
        # Prefer an assigned heading if you have one; else use runway nominal heading
        hdg = st.get("assigned_hdg") or st.get("hdg")
        freq = kgso_departure_freq_for_heading(hdg) if hdg is not None \
               else kgso_departure_freq_for_runway(st.get("runway"))
        return f"{cs}, contact Departure {freq}."
    return None



# --- Departure (KGSO/TRIAD) sectorized frequencies ---


def kgso_departure_freq_for_heading(hdg_deg: float | None) -> str:
    """
    Return the correct KGSO Departure freq by sector:
      - 250–359 or 0–49  -> 124.35
      - 50–249           -> 126.6
    If hdg_deg is None, caller should pass a runway-based heading instead.
    """
    if hdg_deg is None:
        return KGSO_DEP_FREQ_B  # conservative default (most runway headings at KGSO fall in 050–249)
    h = int(hdg_deg) % 360
    if (250 <= h <= 359) or (0 <= h <= 49):
        return KGSO_DEP_FREQ_A
    return KGSO_DEP_FREQ_B


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
    If the content is only the callsign (even repeated), synthesize/ask again.
    """
    t = (text or "").strip()

    # Strip leading role tags anywhere
    t = re.sub(r"^\s*(?:🧑‍✈️\s*)?Pilot:\s*", "", t, flags=re.I)
    t = re.sub(r"\bPilot:\s*", "", t, flags=re.I)

    # Strip prompt artifacts
    t = re.sub(r"<[^>]*>", "", t)                              # <placeholders>
    t = re.sub(r"(?i)\binstruction here\b", "", t)
    t = re.sub(r"(?i)\bstart with\b.*$", "", t)
    t = re.sub(r"(?i)\boutput\s*format.*$", "", t)
    t = re.sub(r"[`\"“”‘’]+", "", t)
    t = re.sub(r"\s{2,}", " ", t).strip(" ,.;-")

    # Remove any *run* of leading callsigns with or without separators
    # e.g., "CSB421", "CSB421, ", "CSB421 - ", "CSB421 : " (repeated allowed)
    lead_pat = rf"(?i)^\s*(?:{re.escape(cs)}\b[,\s:\-;]*)+"
    t = re.sub(lead_pat, "", t).strip(" ,.;-")

    # If what's left is *still* just the callsign (possibly repeated), treat as empty
    cs_only_rx = rf"(?i)^(?:{re.escape(cs)}\b[,\s]*)+\.?$"
    if (not t) or re.fullmatch(cs_only_rx, (text or "").strip()):
        return f"Pilot: {cs}, say again the full instruction."

    # Ensure final punctuation and canonical header
    t = t.rstrip(".")
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

CONTACT_TWR_RX = re.compile(
    r"\b(?:contact(?:ing)?|switch(?:ing)?(?:\s*to)?)\s*(?:the\s*)?(?:tower|119\.1)\b",
    re.I
)

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
    """
    Prefix brand once and ensure NO duplicate callsigns at the start — even if the
    model output is just the bare callsign (e.g., "SKW5503").
    """
    core = (core or "").strip()

    # remove any brand the model might have spoken
    core = re.sub(r'(?i)\bThis\s+is\s+XAION\s+CONTROL\.?\s*', '', core)

    # remove any *run* of leading callsigns with or without punctuation
    # e.g., "CSB421", "CSB421, ", "CSB421 - ", "CSB421 : " (repeated allowed)
    lead_pat = rf'(?i)^\s*(?:{re.escape(cs)}\b[,\s:\-;]*)+'
    core = re.sub(lead_pat, '', core).strip()

    # if the remaining core still equals the callsign (bare), drop it
    if re.fullmatch(rf'(?i){re.escape(cs)}', core):
        core = ""

    # return brand + core (trim any trailing space if core is empty)
    return f"{cs}, This is XAION CONTROL. {core}".rstrip()

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
    base = _rw_base(rw)
    base_map = {"05": 50, "23": 230, "14": 140, "32": 320}
    return base_map.get(base, 50)


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

# Use a concrete side by default
DEFAULT_RWY_PRIMARY = "05R"   # 05R/23L is the 10,001' runway

PARALLEL_BASES = set(PARALLEL_RWY.keys())
VALID_RUNWAYS  = {"05L","05R","23L","23R","14","32"}

def _rw_with_side(base: str, side_hint: str) -> str:
    """
    For base '05' or '23', pick L/R based on side_hint ('east_concourse'/'west_concourse').
    - East concourse → 05R / 23L (closer to A taxiway & terminal)
    - West concourse → 05L / 23R (closer to B taxiway)
    """
    b = (base or "").upper()
    if b not in PARALLEL_BASES:
        return b
    if (side_hint or "").startswith("east"):
        return "05R" if b == "05" else "23L"
    else:
        return "05L" if b == "05" else "23R"

TAXI_ROUTES = {
    # To 05R / 23L (east side)
    ("to_runway", "05R", "east_concourse"): ["via A, A2, hold short Runway 05R"],
    ("to_runway", "23L", "east_concourse"): ["via A, A5, hold short Runway 23L"],

    # To 05L / 23R (west side)
    ("to_runway", "05L", "west_concourse"): ["via B, B3, cross A at A4, hold short Runway 05L"],
    ("to_runway", "23R", "west_concourse"): ["via B, B2, hold short Runway 23R"],

    # 14/32 from BOTH sides (add these two lines)
    ("to_runway", "14", "west_concourse"): ["via B, B1, hold short Runway 14"],
    ("to_runway", "32", "east_concourse"): ["via A, A3, hold short Runway 32"],

    # Existing examples
    ("to_runway", "14", "east_concourse"): ["via A, A1, hold short Runway 14"],
    ("to_runway", "32", "west_concourse"): ["via B, B5, hold short Runway 32"],

    ("to_gate", "east_concourse"): ["exit via A, then A eastbound"],
    ("to_gate", "west_concourse"): ["exit via B, then B westbound"],
}




def _compose_taxi_to_runway(callsign: str, runway: str, gate: str | None) -> str:
    """
    Compose a realistic taxi instruction. Ensures runway is resolved to L/R
    and updates any route text to match the chosen side.
    """
    side  = _side_for_gate(gate) if gate else _side_for_callsign(callsign)
    rw_lr = normalize_lr(runway, gate, callsign)
    base  = _rw_base(rw_lr)

    segs = (
        TAXI_ROUTES.get(("to_runway", rw_lr, side)) or
        TAXI_ROUTES.get(("to_runway", base,  side))
    )

    if not segs:
        return f"{callsign}, taxi to Runway {rw_lr}, hold short Runway {rw_lr}."

    body = ", ".join(segs)
    # If a base-only placeholder was used in the route text, rewrite with the resolved L/R
    body = re.sub(rf"\bRunway\s+{re.escape(base)}\b", f"Runway {rw_lr}", body, flags=re.I)

    return f"{callsign}, taxi to Runway {rw_lr}, {body}"





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
def build_scenarios_from_flat(snapshot_df: pd.DataFrame):
    """
    Build Takeoff and Landing scenario strings from the flattened snapshot.

    TAKEOFF: pilot prompt shows ONLY callsign + Gate + runway request.
    LANDING: IFR-style initial check-in: callsign + (### feet if known) + inbound ILS Runway <XX>.
             No distance/DME or heading on the initial call.
    """
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
        return _good_callsign(cs) and bool(r.get("on_ground", False))

    def _row_ok_for_landing(r):
        cs = (r.get("display_id") or r.get("ident") or "").strip()
        airborne = not r.get("on_ground", False)
        has_any_nav = (
            pd.notna(r.get("dist_nm")) or
            pd.notna(r.get("track_deg")) or
            (r.get("alt_geom_ft") is not None and not (isinstance(r.get("alt_geom_ft"), float) and math.isnan(r.get("alt_geom_ft"))))
        )
        return _good_callsign(cs) and airborne and has_any_nav

    takeoff_rows = [i for i in takeoff_rows if _row_ok_for_takeoff(flat_df.loc[i])]
    landing_rows = [i for i in landing_rows if _row_ok_for_landing(flat_df.loc[i])]

    def fmt_takeoff_row(r):
        cs = (r.get("display_id") or r.get("ident")).strip()
        rw = runway_from_heading(r.get("track_deg")) or DEFAULT_RWY_PRIMARY
        rw = normalize_lr(rw, pick_gate_for_callsign(cs), cs)
        gate = pick_gate_for_callsign(cs)
        return f"Greensboro Ground, {cs}, at {gate}, request taxi to Runway {rw} for departure."

    def _round_alt_100(ft_val):
        try:
            return int(round(float(ft_val) / 100.0) * 100)
        except Exception:
            return None

    def fmt_landing_row(r):
        cs = (r.get("display_id") or r.get("ident")).strip()
        rw = runway_from_heading(r.get("track_deg")) or DEFAULT_RWY_PRIMARY
        rw = normalize_lr(rw, pick_gate_for_callsign(cs), cs)
        alt_ft = r.get("alt_geom_ft")
        alt_rounded = _round_alt_100(alt_ft) if alt_ft else None

        # 🔧 Say "#### feet" instead of "level ####"
        if alt_rounded and alt_rounded > 0:
            return f"Greensboro Approach, {cs}, {alt_rounded} feet, inbound ILS Runway {rw}."
        else:
            return f"Greensboro Approach, {cs}, inbound ILS Runway {rw}."

    TAKEOFF_SCENARIOS = [fmt_takeoff_row(flat_df.loc[i]) for i in takeoff_rows]
    LANDING_SCENARIOS = [fmt_landing_row(flat_df.loc[i]) for i in landing_rows]
    DT_INDEX_MAP = {"Takeoff": takeoff_rows, "Landing": landing_rows}
    DT_SCENARIOS = {"Takeoff": TAKEOFF_SCENARIOS, "Landing": LANDING_SCENARIOS}

    debug_line(f"[INIT] Scenarios — Takeoff: {len(TAKEOFF_SCENARIOS)}, Landing: {len(LANDING_SCENARIOS)}")
    return DT_SCENARIOS, DT_INDEX_MAP

# --- Build scenarios at import time (module globals) ---
DT_SCENARIOS: dict[str, list[str]] = {"Takeoff": [], "Landing": []}
DT_INDEX_MAP: dict[str, list[int]] = {"Takeoff": [], "Landing": []}

DT_SCENARIOS, DT_INDEX_MAP = build_scenarios_from_flat(flat_df)

def pilot_ack(cs: str) -> str:
    return random.choice(PILOT_ACK_VARIANTS).format(cs=cs)


# -------------------------------
# Dialogue state & utilities
# -------------------------------
CONTEXT_HISTORY = deque(maxlen=24)
VOICE_DIALOGUE = ""
LAST_PILOT_TEXT = ""
REQUIRE_PILOT_FIRST = True  # enforce seed order


def reset_for_new_run(reason: str = "NEW RUN / SCENARIO CHANGE"):
    """
    Hard reset of conversational state, runway state, and the rolling debug buffer.
    Call this when the user selects a new scenario or clicks 'Restart Simulation'.
    """
    global DEBUG_LOG, CONTEXT_HISTORY, VOICE_DIALOGUE, LAST_PILOT_TEXT
    global CALLSTATE, RUNWAY_STATE, VOICE_LAST_ATC_STRUCT, LAST_ACTIVE_CS

    # Clear dialogue + per-flight state
    VOICE_DIALOGUE = ""
    LAST_PILOT_TEXT = ""
    VOICE_LAST_ATC_STRUCT = None
    LAST_ACTIVE_CS = None
    CALLSTATE.clear()
    RUNWAY_STATE.clear()

    # Refresh monitor/debug panes
    try:
        CONTEXT_HISTORY.clear()
    except Exception:
        CONTEXT_HISTORY = deque(maxlen=24)

    try:
        DEBUG_LOG.clear()
    except Exception:
        # Fall back if list was replaced
        DEBUG_LOG[:] = []

    monitor_header(reason)


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

# === Handlers: scenario change / restart ===
# === Handlers: scenario change / restart ===
def on_select_scenario(kind: str, scen_idx: int):
    """
    kind: "Takeoff" or "Landing"
    scen_idx: index into DT_SCENARIOS[kind]
    """
    reset_for_new_run(f"NEW RUN — {kind} scenario selected (idx={scen_idx})")

    try:
        idx = DT_INDEX_MAP[kind][scen_idx]
        globals()["LAST_SELECTED_KIND"] = kind
        globals()["LAST_SELECTED_IDX"]  = idx

        # Remember start for Full Sim ordering
        SIM_START["idx"]   = idx
        SIM_START["dt"]    = flat_df.loc[idx, "__dt"]
        SIM_START["ident"] = _selected_ident_at(idx)

        # Seed the monitor with the *selected* aircraft first
        context_frame_from_idx(idx, push=True)
    except Exception:
        pass

    return recent_context_block(3), snapshot_debug_dump()


def on_restart_click():
    reset_for_new_run("RESTART SIMULATION")
    # Keep the output arity identical to on_select_scenario (context, debug)
    return recent_context_block(3), snapshot_debug_dump()




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

def _strip_leading_cs(text: str, cs: str) -> str:
    """
    Remove one or more leading callsign runs with or without punctuation:
    'CSB421', 'CSB421, ', 'CSB421 :', 'CSB421 - ' → ''
    """
    if not text: return ""
    # e.g. ^ (CSB421\b [,:;\-\s]*)+
    pat = rf"(?i)^\s*(?:{re.escape(cs)}\b[,\s:\-;]*)+"
    return re.sub(pat, "", text).strip()


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
    import re
    if not scen:
        return ""
    m_cs = re.search(r",\s*([A-Z]{2,4}\d{2,6})\b", scen)
    cs = (m_cs.group(1) if m_cs else "N123AB").upper()
    m_rw = re.search(r"Runway\s+(\d{1,2}[LRC]?)", scen, flags=re.I)
    rw = (m_rw.group(1).upper() if m_rw else DEFAULT_RWY_PRIMARY)
    twr = KGSO_TOWER_FREQ if 'KGSO_TOWER_FREQ' in globals() else "119.1"

    if comm_type == "Takeoff":
        return (
f"""**What to say — Takeoff (step-by-step)**

1) **Request taxi (Ground)**
   - “Greensboro Ground, {cs}, at Gate <XX>, request taxi to Runway {rw} for departure.”

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
"""
        )

    # IFR-style landing examples (no DME/heading on the first call)
    return (
f"""**What to say — Landing (step-by-step)**

1) **Initial check-in (Approach, IFR)**
   - “Greensboro Approach, {cs}, 3000 feet, inbound ILS Runway {rw}.”

2) **Vectors / altitude / speed assignments — read back numbers**
   - “{cs}, turning heading *[hdg]*, maintaining *[alt]*, {cs}.”

3) **Approach clearance (if issued)**
   - “{cs}, descending and maintaining *[alt]* until established, cleared ILS Runway {rw}, {cs}.”

4) **When established (optional)**
   - “Greensboro Approach, {cs}, established localizer Runway {rw}.”

5) **To Tower on final**
   - “Greensboro Tower, {cs}, five-mile final Runway {rw}.”
"""
    )

def xaion_prefix(callsign: str) -> str:
    tag = (callsign if callsign and callsign.upper() != "UNKNOWN" else "").strip()
    if tag:
        return f"{tag}, This is XAION CONTROL. "
    return "This is XAION CONTROL. "



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
    gate = pick_gate_for_callsign(cs)
    rw = runway_from_heading_lr(r.get("track_deg"), gate, cs)
    return f"Greensboro Ground, {cs}, at {gate}, request taxi to Runway {rw} for departure."


def fmt_landing_from_idx(idx: int) -> str:
    r = flat_df.loc[idx]
    cs = _safe_callsign_from_row(r)
    if not cs:
        return ""
    gate = pick_gate_for_callsign(cs)
    rw = runway_from_heading_lr(r.get("track_deg"), gate, cs)

    def _round_alt_100(ft_val):
        try:
            return int(round(float(ft_val) / 100.0) * 100)
        except Exception:
            return None

    alt_rounded = _round_alt_100(r.get("alt_geom_ft")) if r.get("alt_geom_ft") else None
    # 🔧 Say "#### feet" instead of "level ####"
    if alt_rounded and alt_rounded > 0:
        return f"Greensboro Approach, {cs}, {alt_rounded} feet, inbound ILS Runway {rw}."
    else:
        return f"Greensboro Approach, {cs}, inbound ILS Runway {rw}."



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

def _extract_items(atc_text: str):
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
        "continue":      bool(re.search(r"\bcontinue\b", t)),
        "expect_late":   bool(re.search(r"\bexpect\b.*\blanding\b.*\bclearance\b", t)),
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

def _pilot_covers_items(pilot_text: str, items: dict) -> bool:
    """Lenient readback validator (now accepts 'continue' readbacks)."""
    if not _items_nonempty(items):
        return False

    p = normalize_transcript(pilot_text or "").lower()

    has_taxi       = bool(re.search(r"\btaxi(?:ing)?\b", p))
    has_hold_short = bool(re.search(r"\bhold(?:ing)?\s+short\b", p))
    has_runway     = bool(re.search(r"\brunway\s*\d{1,2}[lrc]?\b", p))
    has_takeoff    = bool(re.search(r"\b(takeoff|line\s*up\s*and\s*wait|luaw)\b", p))
    has_ldg_rb     = bool(re.search(r"\bcleared\s+to\s+land\b", p))
    has_continue   = bool(re.search(r"\bcontinu(?:e|ing)\b", p))

    if items.get("taxi") and items.get("hold_short") is not None:
        if not (has_taxi and has_hold_short):
            return False
        if not has_runway and len(items.get("hold_short", [])) != 1:
            return False

    if items.get("runway_takeoff"):
        if not (has_takeoff and has_runway):
            return False
    if items.get("runway_land"):
        if not (has_ldg_rb and has_runway):
            return False

    if items.get("alt"):
        has_alt_units = bool(re.search(r"\b\d{2,5}\s*(?:feet|ft)\b", p))
        has_alt_bare  = bool(re.search(r"\b(?:maintain(?:ing)?|descend(?:ing)?|climb(?:ing)?)\s*\d{3,5}\b", p))
        if not (has_alt_units or has_alt_bare):
            return False

    if items.get("hdg") and not re.search(r"\bheading\s*\d{1,3}\b", p):
        return False
    if items.get("spd") and not re.search(r"\b\d{1,3}\s*(?:kts?|knots?)\b", p):
        return False
    if items.get("freq"):
        has_num   = any(re.search(fr"\b{re.escape(f)}\b", p) for f in items["freq"])
        has_words = bool(re.search(r"\b(switch(?:ing)?|contact(?:ing)?)\b", p))
        if not (has_num or has_words):
            return False

    # NEW: If ATC said “continue…”, accept a simple “continue” readback.
    if items.get("continue") and not has_continue:
        return False

    return True

def _sanitize_atc_line(core_no_brand: str, callsign: str, dialogue: str) -> str:
    """
    Prevent ATC from saying 'say again / repeat'. If it happens, restate the previous
    instruction core instead of asking the pilot to repeat.
    """
    low = (core_no_brand or "").lower()
    if re.search(r"\b(say\s+again|repeat)\b", low):
        prev = strip_atc_ui_to_core(_last_atc(dialogue), callsign) or "stand by."
        return prev if prev else "stand by."
    return core_no_brand




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
        gen = phi4_model.generate(
            tok.input_ids, attention_mask=tok.attention_mask,
            max_new_tokens=4, do_sample=False, num_beams=1,
            pad_token_id=phi4_tokenizer.eos_token_id, eos_token_id=phi4_tokenizer.eos_token_id
        )
        new_ids = gen[0, tok.input_ids.shape[1]:]
        text = phi4_tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        decision = "ATC" if re.search(r"\bATC\b", text, re.I) else "Pilot"
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
    base_map = {"05", "23", "14", "32"}
    base = _rw_base(runway or "")
    bad_rw = base not in base_map

    if bad_cs:
        return True, "This is XAION CONTROL. Unable to proceed (callsign unknown). Contact Tower 119.1."

    if bad_rw:
        corr = DEFAULT_RWY_PRIMARY
        tag = cs.strip()
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
      - Pilot fallback: synthesize a full readback from the last ATC when the LLM under-performs
        or tries to ask 'say again'.
      - Normalizes Tower frequency, runway side dupes, and Departure frequencies.
    """
    phase = ctx.get("phase", "approach")
    runway_hint = ctx.get("runway", DEFAULT_RWY_PRIMARY)
    callsign = ctx.get("callsign", "UNKNOWN")

    # local no-op wrapper so we don't crash if _normalize_comm_freqs isn't present yet
    def _norm_comm_line(s, *, hdg=None, runway=None):
        try:
            return _normalize_comm_freqs(s, hdg=hdg, runway=runway)
        except NameError:
            return s

    def _safe_atc(callsign: str, phase: str, rw: str) -> str:
        if phase == "ground":
            return f"{callsign}, taxi to Runway {rw}, hold short Runway {rw}."
        else:
            return f"{callsign}, turn heading {int(_hdg_to_runway(rw))}, maintain 2000 until established."

    def _anti_template(s: str) -> str:
        # Strip any angle-bracketed placeholders and their wording
        s2 = re.sub(r"<[^>]*>", "", s or "")
        s2 = re.sub(r"(?i)\binstruction here\b", "", s2)
        return re.sub(r"\s{2,}", " ", s2).strip(" ,.;")

    monitor_header("XAION SIM START (VOICE CTX)")
    monitor_block(role, phase, runway_hint)
    ctx_current = context_frame_from_ctx(ctx, push=True)
    _ctx_dump_lines(ctx_current, dialogue_text=dialogue)

    # ---------------- Prompt ----------------
    if role == "ATC":
        if is_ack_only(LAST_PILOT_TEXT):
            atc_struct = ctx.get("_last_atc_struct")
            if atc_struct and _pilot_covers_items(LAST_PILOT_TEXT, atc_struct):
                out_line = f"{callsign}, readback correct."
            else:
                out_line = _ack_line(callsign)
            out_line = atc_prefix_and_dedup(callsign, out_line)
            out_line = _normalize_tower_and_runway(out_line)
            out_line = _norm_comm_line(
                out_line,
                hdg=ctx.get("assigned_hdg") or ctx.get("hdg_deg"),
                runway=runway_hint
            )
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

    # ---------------- LLM call ----------------
    try:
        tok = phi4_tokenizer(base_prompt, return_tensors="pt",
                             truncation=True, padding=True, max_length=1024).to(device)
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

    # ---------------- Sanitize ----------------
    text_nowx = strip_weather_clauses(text_raw)
    resp = clean_response(text_nowx, phase=phase, runway_hint=runway_hint, callsign=callsign)
    resp = _anti_template(_strip_role_leaks(resp))
    resp = _strip_control_artifacts(resp, callsign)
    resp = _normalize_tower_and_runway(resp)

    # ---------------- Role-specific repairs ----------------
    if role == "ATC":
        # Under-output / template echo guard
        body = re.sub(rf"^\s*{re.escape(callsign)}\s*,\s*", "", resp, flags=re.I).strip()
        if len(body) < 6 or re.search(r"[<>]|(?i)instruction here", resp):
            resp = _safe_atc(callsign, phase, runway_hint)

        # “Say again” from pilot → restate previous ATC core
        if REPEAT_RX.search(LAST_PILOT_TEXT or ""):
            last_ui = _last_atc(dialogue)
            if last_ui:
                last_core = strip_atc_ui_to_core(last_ui, callsign)
                if last_core:
                    resp = last_core

        # Track the extracted structure for the *next* pilot validation
        atc_struct = _extract_items(resp)
        ctx["_last_atc_struct"] = atc_struct

        # Brand once + dedup any leading callsign runs
        resp = atc_prefix_and_dedup(callsign, resp)

        # Final minimal guard
        if len(resp) < 8:
            resp = atc_prefix_and_dedup(callsign, _safe_atc(callsign, phase, runway_hint))

        # Final normalize (Tower, Runway, and Departure frequencies)
        resp = _normalize_tower_and_runway(resp)
        resp = _norm_comm_line(
            resp,
            hdg=ctx.get("assigned_hdg") or ctx.get("hdg_deg"),
            runway=runway_hint
        )

        monitor_header("XAION SIM END (VOICE CTX)")
        return resp

    # ---------- PILOT path ----------
    # Canonicalize first (ensures 'Pilot: CS, ...' and trims placeholders)
    resp = canonicalize_pilot_ui(resp, callsign)

    # Validate against the last ATC structure if we have it
    atc_struct = ctx.get("_last_atc_struct", None)

    needs_synth = False
    if len(strip_role_prefix(resp)) < 10:
        needs_synth = True
    if SAY_AGAIN_RX.search(resp or ""):
        needs_synth = True
    if atc_struct and not _pilot_covers_items(resp, atc_struct):
        needs_synth = True

    if needs_synth:
        syn = _synth_pilot_from_last_atc(dialogue, callsign)
        if syn:
            resp = syn
        else:
            resp = f"Pilot: {callsign}, say again the full instruction."

    # Ensure callsign present once; tidy punctuation
    if callsign.upper() not in resp.upper():
        resp = f"{resp.rstrip('.').rstrip(',')}, {callsign}."

    # Normalize Tower/Runway + Departure frequencies on Pilot readbacks too
    resp = _normalize_tower_and_runway(resp)
    resp = _norm_comm_line(
        resp,
        hdg=ctx.get("assigned_hdg") or ctx.get("hdg_deg"),
        runway=runway_hint
    )

    # Final minimal guard for pilot (prefer synthesis over a looped 'say again')
    if len(strip_role_prefix(resp)) < 8:
        syn = _synth_pilot_from_last_atc(dialogue, callsign)
        resp = syn or f"Pilot: {callsign}, say again the full instruction."

    monitor_header("XAION SIM END (VOICE CTX)")
    return resp

    # ---------- PILOT path ----------
    # Canonicalize first (ensures 'Pilot: CS, ...' and trims placeholders)
    resp = canonicalize_pilot_ui(resp, callsign)

    # Validate against the last ATC structure if we have it
    atc_struct = ctx.get("_last_atc_struct", None)

    needs_synth = False
    if len(strip_role_prefix(resp)) < 10:
        needs_synth = True
    if SAY_AGAIN_RX.search(resp or ""):
        needs_synth = True
    if atc_struct and not _pilot_covers_items(resp, atc_struct):
        needs_synth = True

    if needs_synth:
        syn = _synth_pilot_from_last_atc(dialogue, callsign)
        if syn:
            resp = syn
        else:
            resp = f"Pilot: {callsign}, say again the full instruction."

    # Ensure callsign present once; tidy punctuation
    if callsign.upper() not in resp.upper():
        resp = f"{resp.rstrip('.').rstrip(',')}, {callsign}."

    # Normalize Tower/Runway + Departure frequencies on Pilot readbacks too
    resp = _normalize_tower_and_runway(resp)
    resp = _normalize_comm_freqs(resp, hdg=ctx.get("assigned_hdg") or ctx.get("hdg_deg"),
                                 runway=runway_hint)

    # Final minimal guard for pilot (prefer synthesis over a looped 'say again')
    if len(strip_role_prefix(resp)) < 8:
        syn = _synth_pilot_from_last_atc(dialogue, callsign)
        resp = syn or f"Pilot: {callsign}, say again the full instruction."

    monitor_header("XAION SIM END (VOICE CTX)")
    return resp



def gen_once_idx(role, dialogue, idx):
    r = flat_df.loc[idx]
    phase = "ground" if r["on_ground"] else "approach"
    callsign = (r.get("display_id") or r.get("ident") or "UNKNOWN").strip()
    gate_hint = pick_gate_for_callsign(callsign)
    runway_hint = normalize_lr(runway_from_heading(r["track_deg"]) or DEFAULT_RWY_PRIMARY, gate_hint, callsign)

    # Heading hint for frequency normalization
    try:
        hdg_hint = float(r.get("track_deg")) if pd.notna(r.get("track_deg")) else None
    except Exception:
        hdg_hint = None

    monitor_header("GENERATION CALL (TYPED CTX)")
    monitor_block(role, phase, runway_hint)
    ctx_current = context_frame_from_idx(idx, push=True)
    _ctx_dump_lines(ctx_current, dialogue_text=dialogue)

    # ---- Prompt scaffold ----
    if role == "ATC":
        if phase == "ground":
            phase_goal = (
                "Issue ONE specific taxi instruction to a destination (e.g., a runway). "
                "Include exact route segments and explicit HOLD SHORT. "
                "Authorize runway crossings only with 'CROSS Runway <id>'."
            )
        elif phase == "approach":
            phase_goal = (
                "Issue ONE concise control or approach instruction (heading/altitude/speed or approach clearance). "
                "If on final and runway is clear, you may issue 'cleared to land'."
            )
        else:
            phase_goal = "Issue ONE concise instruction or respond to the request."

        instruction = f"""
You are ATC. Follow U.S. phraseology. Hard rules:
- Address the aircraft by CALLSIGN at the start of the line.
- No meta. ONE sentence, <= 25 words.
- Include HOLD SHORT/CROSS/RUNWAY and numeric ALT/HDG/SPD/frequency as applicable.

{phase_goal}

Output format (strict):
ATC: {callsign}, <instruction here>.
""".strip()
    else:
        last_atc = _last_atc(dialogue)
        instruction = f"""
You are the Pilot. ONE short readback line.
- Start with 'Pilot: ' then CALLSIGN.
- Read back ALL pertinent items (route + HOLD SHORT; CROSS + runway; takeoff/landing + runway; ALT/HDG/SPD; frequency; approach type/runway).
- No meta. <= 25 words.

LAST_ATC: {last_atc}

Output format (strict):
Pilot: {callsign}, <exact readback or clarification>.
""".strip()

    base_prompt = (
        f"{instruction}\n\n=== CONTEXT FRAME (CURRENT) ===\n{ctx_current}\n\n"
        f"=== RECENT CONTEXT FRAMES (NEWEST FIRST) ===\n{recent_context_block(3)}\n\n"
        f"=== DIALOGUE SO FAR ===\n{dialogue.strip()}\n\n{role}:"
    )

    # ---- LLM call ----
    try:
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
        if f"{role}:" in text:
            text = text.split(f"{role}:", 1)[-1]
        text_raw = text.strip()
    except Exception as e:
        debug_line(f"[{role}] LLM error -> fallback: {e}")
        text_raw = ""

    # ---- sanitize / trim ----
    def _anti_template(s: str) -> str:
        s = s or ""
        s = re.sub(r"<[^>]*>", "", s)
        s = re.sub(r"(?i)\binstruction here\b", "", s)
        s = re.sub(r"(?i)\bstart with\b.*$", "", s)
        s = re.sub(r"(?i)\boutput\s*format.*$", "", s)
        s = re.sub(r"(?i)\bthen\s*CALLSIGN\b\.?", "", s)
        s = re.sub(r"[`\"“”‘’]+", "", s)
        return re.sub(r"\s{2,}", " ", s).strip(" ,.;-")

    text_nowx = strip_weather_clauses(text_raw)
    resp = clean_response(text_nowx, phase=phase, runway_hint=runway_hint, callsign=callsign).strip()
    resp = _anti_template(_strip_role_leaks(resp))
    resp = _strip_control_artifacts(resp, callsign)
    resp = _normalize_tower_and_runway(resp)

    if role == "ATC":
        # Fallback if the model produced nothing but the callsign or nearly nothing
        body = _strip_leading_cs(resp, callsign)
        if (not body) or (len(body) < 6) or (body.upper() == callsign.upper()):
            resp = _rule_based_atc_for_row(r, callsign, runway_hint)

        # Brand once
        resp = atc_prefix_and_dedup(callsign, resp)

        # Final normalize (Tower/Runway + Departure frequencies)
        resp = _normalize_tower_and_runway(resp)
        resp = _normalize_comm_freqs(resp, hdg=hdg_hint, runway=runway_hint)

        debug_line(f"[ATC FINAL] {resp}")
        return resp

    # ---------- PILOT path: validate against last ATC; synthesize if weak ----------
    resp = canonicalize_pilot_ui(resp, callsign)

    bare = strip_role_prefix(resp)
    cs_only_rx = rf"(?i)^(?:{re.escape(callsign)}\b[,\s]*)+\.?$"
    if re.fullmatch(cs_only_rx, bare):
        needs_synth = True
    else:
        needs_synth = False

    last_ui = _last_atc(dialogue)
    last_core = strip_atc_ui_to_core(last_ui, callsign) if last_ui else ""
    items = _extract_items(last_core) if last_core else None

    if (not needs_synth):
        if len(bare) < 10:
            needs_synth = True
        if SAY_AGAIN_RX.search(resp):
            needs_synth = True
        if items is not None and not _pilot_covers_items(resp, items):
            needs_synth = True

    if needs_synth and last_core:
        resp = _pilot_readback_from_atc_text(callsign, last_core)

    if callsign.upper() not in resp.upper():
        resp = f"{resp.rstrip('.').rstrip(',')}, {callsign}."

    # Normalize Tower/Runway + Departure freqs on Pilot readbacks
    resp = _normalize_tower_and_runway(resp)
    resp = _normalize_comm_freqs(resp, hdg=hdg_hint, runway=runway_hint)

    debug_line(f"[PILOT FINAL] {resp}")
    return resp


# -------------------------------
# Voice step (single-turn ATC reply path)
# -------------------------------
def parse_transcript_ctx(text: str) -> dict:
    t = normalize_transcript(text)
    cs = None; m = CALLSIGN_RX.search(t.upper());  cs = _normalize_callsign(m.group(0)) if m else None
    rwy = None; m = RUNWAY_RX.search(t);           rwy = m.group(1).upper() if m else None
    alt = None; m = ALT_RX.search(t);              alt = float(m.group(1)) if m else None
    spd = None; m = SPD_RX.search(t);              spd = float(m.group(1)) if m else None
    dist = None; m = DIST_RX.search(t);            dist = float(m.group(1)) if m else None
    hdg = None; m = HDG_RX.search(t);              hdg = float(m.group(1)) if m else None
    gate = _gate_from_text(t)

    low_t = t.lower()
    on_ground = any(kw in low_t for kw in ["ground", "ramp", "taxi", "gate"])
    approach_cues = ("approach","final","established","localizer","glideslope","cleared to land","inbound")
    is_approach = any(kw in low_t for kw in approach_cues)
    phase = "ground" if on_ground and not is_approach else "approach"

    # Pick a gate if missing so we can choose L/R side deterministically
    if not gate and cs:
        gate = pick_gate_for_callsign(cs)

    # Pick runway: prefer explicit; else from heading; else default — then add L/R if needed
    if not rwy and hdg is not None:
        rwy = runway_from_heading(hdg)
    if not rwy:
        rwy = DEFAULT_RWY_PRIMARY
    rwy = normalize_lr(rwy, gate, cs)

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

        # ✅ NEW: strip stamps and role/emoji before anything else
        t = _strip_ui_wrappers_for_tts(text)

        # Brand normalization & pronunciation tweaks (unchanged)
        t = t.replace("XAION CONTROL", "Zion Control")
        t = _audio_text_for_tts(t)  # numerals/runway/freq normalizations

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
REQ_TAXI_RX     = re.compile(r"\brequest\s+taxi\s+to\s+runway\s*(\d{1,2}[LRC]?)\b", re.I)
LINEUP_RX       = re.compile(r"\b(line\s*up(?:\s*and\s*wait)?|luaw)\b.*\brunway\s*(\d{2}[LRC]?)\b", re.I)
PILOT_ASSERT_TKOF_RX = re.compile(r"\bcleared\s+for\s+takeoff\b", re.I)

# --- Vocal Input progression helpers (ground) ---
# --- Vocal Input progression helpers (ground) ---
def _vocal_ground_flow(cs: str, ctx: dict, pilot_text: str, dialogue: str) -> str | None:
    """
    For Vocal Input only: detect ground milestones and emit ONE ATC line,
    phrased by the LLM, that advances toward takeoff.
    Returns a fully-branded ATC UI block (1–2 lines), or None to fall back.
    """
    rw = ctx.get("runway") or DEFAULT_RWY_PRIMARY
    st = CALLSTATE.setdefault(cs, {"phase": "ground", "runway": rw, "stage": None, "gate": ctx.get("gate")})

    # Regex (module-level too, but keep here for clarity if imported standalone)
    READY_RX        = re.compile(r"\b(holding\s+short\s+runway\s*(\d{2}[LRC]?)\b.*\bready\b|ready\s+(?:for\s+)?(?:departure|takeoff))", re.I)
    CONTACT_TWR_RX  = re.compile(r"\b(contact(?:ing)?|switch(?:ing)?(?:\s*to)?)\s*(?:tower|\b119\.1\b)\b", re.I)
    LINEUP_RX       = re.compile(r"\b(?:line\s*up(?:\s*and\s*wait)?|luaw)\b.*\brunway\s*(\d{2}[LRC]?)\b", re.I)
    PILOT_ASSERT_TKOF_RX = re.compile(r"\bcleared\s+for\s+takeoff\b", re.I)

    # 1) Pilot requests taxi -> taxi + HOLD SHORT
    if REQ_TAXI_RX.search(pilot_text):
        atc_core = _compose_taxi_to_runway(cs, rw, st.get("gate"))
        core = llm_style_atc_from_core(cs, atc_core, dialogue, {"phase": "ground", "runway": rw, "callsign": cs})
        core = _normalize_tower_and_runway(core)
        st.update(stage="taxi_issued")
        return f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}{core}"

    # 2) Holding short & ready OR LUAW BEFORE handoff -> hand off to Tower
    if (READY_RX.search(pilot_text) or LINEUP_RX.search(pilot_text)) and st.get("stage") in (None, "taxi_issued"):
        atc_core = f"{cs}, contact Tower {KGSO_TOWER_FREQ}."
        core = llm_style_atc_from_core(cs, atc_core, dialogue, {"phase": "ground", "runway": rw, "callsign": cs})
        core = _normalize_tower_and_runway(core)
        core = _require_tokens_or_fallback(core, _MUST_CONTAIN["contact_tower"], atc_core)
        st.update(stage="handoff_twr")
        return f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}{core}"

    # 2b) AFTER handoff, any new READY / LUAW -> issue TAKEOFF CLEARANCE
    if (READY_RX.search(pilot_text) or LINEUP_RX.search(pilot_text)) and st.get("stage") == "handoff_twr":
        atc_core = f"{cs}, cleared for takeoff Runway {rw}."
        core = llm_style_atc_from_core(cs, atc_core, dialogue, {"phase": "ground", "runway": rw, "callsign": cs})
        core = _normalize_tower_and_runway(core)
        core = _require_tokens_or_fallback(core, _MUST_CONTAIN["cleared_tkof"], atc_core)
        st.update(stage="cleared_takeoff")
        # Optional immediate Departure handoff:
        handoff_core = _maybe_departure_handoff(cs, st)
        if handoff_core:
            hand_core = llm_style_atc_from_core(cs, handoff_core, dialogue, {"phase": "ground", "runway": rw, "callsign": cs})
            hand_core = _normalize_tower_and_runway(hand_core)
            hand_ui = f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}{hand_core}"
            return f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}{core}\n{hand_ui}"
        return f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}{core}"

    # 3) Pilot says 'switching/contacting Tower' -> cleared for takeoff
    if CONTACT_TWR_RX.search(pilot_text) and st.get("stage") in ("handoff_twr", "taxi_issued"):
        atc_core = f"{cs}, cleared for takeoff Runway {rw}."
        core = llm_style_atc_from_core(cs, atc_core, dialogue, {"phase": "ground", "runway": rw, "callsign": cs})
        core = _normalize_tower_and_runway(core)
        core = _require_tokens_or_fallback(core, _MUST_CONTAIN["cleared_tkof"], atc_core)
        st.update(stage="cleared_takeoff")
        handoff_core = _maybe_departure_handoff(cs, st)
        if handoff_core:
            hand_core = llm_style_atc_from_core(cs, handoff_core, dialogue, {"phase": "ground", "runway": rw, "callsign": cs})
            hand_core = _normalize_tower_and_runway(hand_core)
            hand_ui = f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}{hand_core}"
            return f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}{core}\n{hand_ui}"
        return f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}{core}"

    # 4) Pilot prematurely says "cleared for takeoff"
    if PILOT_ASSERT_TKOF_RX.search(pilot_text):
        if st.get("stage") == "handoff_twr":
            atc_core = f"{cs}, cleared for takeoff Runway {rw}."
            core = llm_style_atc_from_core(cs, atc_core, dialogue, {"phase": "ground", "runway": rw, "callsign": cs})
            core = _normalize_tower_and_runway(core)
            core = _require_tokens_or_fallback(core, _MUST_CONTAIN["cleared_tkof"], atc_core)
            st.update(stage="cleared_takeoff")
            handoff_core = _maybe_departure_handoff(cs, st)
            if handoff_core:
                hand_core = llm_style_atc_from_core(cs, handoff_core, dialogue, {"phase": "ground", "runway": rw, "callsign": cs})
                hand_core = _normalize_tower_and_runway(hand_core)
                hand_ui = f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}{hand_core}"
                return f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}{core}\n{hand_ui}"
            return f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}{core}"
        else:
            atc_core = f"{cs}, contact Tower {KGSO_TOWER_FREQ}."
            core = llm_style_atc_from_core(cs, atc_core, dialogue, {"phase": "ground", "runway": rw, "callsign": cs})
            core = _normalize_tower_and_runway(core)
            core = _require_tokens_or_fallback(core, _MUST_CONTAIN["contact_tower"], atc_core)
            st.update(stage="handoff_twr")
            return f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}{core}"

    return None

def _vocal_approach_flow(cs: str, ctx: dict, pilot_text: str, dialogue: str) -> str | None:
    """
    Vocal Input — Approach ladder:
      1) Initial inbound/check-in -> issue approach clearance.
      2) Established / (n-mile) final on Approach -> hand off to Tower 119.1.
      3) Tower check-in on final -> clear to land if runway free; else
         'continue approach, expect late landing clearance.'
    Returns a fully-branded ATC UI line (or multi-line) or None to let caller fall back.
    """
    rw = ctx.get("runway") or DEFAULT_RWY_PRIMARY
    st = CALLSTATE.setdefault(cs, {"phase": "approach", "runway": rw, "stage": None, "gate": ctx.get("gate")})

    t   = normalize_transcript(pilot_text or "")
    low = t.lower()

    INBOUND_RX       = re.compile(r"\binbound\b", re.I)
    ILS_RX           = re.compile(r"\b(ils|localizer|visual|rnav)\b", re.I)
    ESTABLISHED_RX   = re.compile(r"\b(established\b|on\s+(?:final|localizer|glideslope)|short\s+final)\b", re.I)
    FINAL_NM_RX      = re.compile(r"\b(\d+(?:\.\d+)?)\s*-?\s*mile\s+final\b", re.I)
    TOWER_RX         = re.compile(r"\btower\b", re.I)
    TOWER_FINAL_RX   = re.compile(r"\btower\b.*\b(final|short\s*final|on\s*final|\d+(?:\.\d+)?\s*mile\s*final|established)\b", re.I)

    def _brand(ui_core: str) -> str:
        return f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}{ui_core}"

    # 1) Initial check-in → approach clearance (once)
    if st.get("stage") is None and (INBOUND_RX.search(low) or ILS_RX.search(low)):
        atc_core = f"descend and maintain 3000 until established, cleared ILS Runway {rw}."
        core = llm_style_atc_from_core(cs, atc_core, dialogue, ctx, must_keep=("cleared","ils"))
        core = _normalize_tower_and_runway(_sanitize_atc_line(core, cs, dialogue))
        st["stage"] = "apch_cleared"
        # track structure for readback checking
        ctx["_last_atc_struct"] = _extract_items(core)
        return _brand(core)

    # 2) Established/on final while still with Approach → hand off to Tower
    if (st.get("stage") in (None, "apch_cleared")) and (ESTABLISHED_RX.search(low) or FINAL_NM_RX.search(low)):
        atc_core = f"contact Tower {KGSO_TOWER_FREQ}."
        core = llm_style_atc_from_core(cs, atc_core, dialogue, ctx, must_keep=("contact","tower"))
        core = _normalize_tower_and_runway(_sanitize_atc_line(core, cs, dialogue))
        st["stage"] = "handoff_twr"
        ctx["_last_atc_struct"] = _extract_items(core)
        return _brand(core)

    # 3) Tower check-in on final → attempt landing clearance
    if st.get("stage") == "handoff_twr" and TOWER_FINAL_RX.search(low):
        ok, alt_core = gate_runway_clearance("land", rw, cs)
        if ok:
            atc_core = f"cleared to land Runway {rw}."
            core = llm_style_atc_from_core(cs, atc_core, dialogue, ctx, must_keep=("cleared","land"))
            st["stage"] = "cleared_land"
        else:
            # NEVER bare 'continue' — always include expectation
            atc_core = "continue approach, expect late landing clearance."
            core = llm_style_atc_from_core(cs, atc_core, dialogue, ctx, must_keep=("continue","expect","landing"))
            st["stage"] = "continue_expect"

        core = _normalize_tower_and_runway(_sanitize_atc_line(core, cs, dialogue))
        ctx["_last_atc_struct"] = _extract_items(core)
        return _brand(core)

    # If they just say “Tower, <cs> …” but without a final cue, still give handoff/standby wording
    if st.get("stage") == "handoff_twr" and TOWER_RX.search(low):
        atc_core = f"continue approach, expect late landing clearance."
        core = llm_style_atc_from_core(cs, atc_core, dialogue, ctx, must_keep=("continue","expect","landing"))
        core = _normalize_tower_and_runway(_sanitize_atc_line(core, cs, dialogue))
        ctx["_last_atc_struct"] = _extract_items(core)
        return _brand(core)

    return None


def voice_step(pilot_text: str):
    """
    Vocal Input step with:
      • Early 'readback correct' when pilot covers last ATC items.
      • Ground ladder (taxi → handoff → takeoff + optional Departure handoff).
      • Approach ladder (check-in → approach clr → Tower handoff → landing clr).
      • Per-line 🕒 stamping (multi-line safe) and TTS without speaking timestamps.
    """
    import re
    global VOICE_DIALOGUE, LAST_PILOT_TEXT, VOICE_LAST_ATC_STRUCT, CALLSTATE

    # ---------- small helpers local to this function ----------
    def __post_fix_freq_punct(s: str) -> str:
        # Collapse accidental duplicated tenths (e.g., "119.1.1" → "119.1")
        s = re.sub(r'\b(119\.1)\.1\b', r'\1', s)
        s = re.sub(r'\b(126\.6)\.6\b', r'\1', s)
        # If a frequency is immediately followed by an extra '.', drop the extra
        s = re.sub(r'\b(\d{3}\.\d)\.(?=\s|$)', r'\1', s)
        return s

    def __stamp_block(block: str, ctx: dict) -> str:
        """
        Stamp each non-empty line with 🕒 [HHMMSSZ], bumping +STAMP_STEP_SEC per line.
        """
        if not block:
            return block
        lines = [ln for ln in block.split("\n") if ln.strip()]
        base_dt = _dt_tag_from_ctx(ctx)
        stamped = []
        for i, ln in enumerate(lines):
            # normalize Tower freq + runway dupes, then fix freq punctuation
            ln = _normalize_tower_and_runway(ln)
            ln = __post_fix_freq_punct(ln)
            dt = base_dt if i == 0 else _bump_z(base_dt, i)
            if not _STAMP_RX.match(ln or ""):
                ln = f"🕒 [{dt}] {ln}"
            stamped.append(ln)
        return "\n".join(stamped)

    def __speak_block(block_unstamped: str) -> str | None:
        """
        Speak each ATC line in order (without the 🕒 stamp).
        Returns the *last* audio path so the UI can autoplay it.
        """
        last_audio = None
        for raw_ln in (block_unstamped or "").split("\n"):
            ln = raw_ln.strip()
            if not ln:
                continue
            # Only voice ATC lines
            if not re.search(r'^\s*🗼\s*ATC:', ln):
                continue
            say = strip_role_prefix(ln)  # strips "ATC:" and any emoji
            say = say.strip()
            if not say:
                continue
            path = tts_elevenlabs(say, voice_id=ELEVEN_VOICE_ID_DEFAULT)
            last_audio = path or last_audio
            _sleep_for_audio(path)
        return last_audio

    # ---------- validate pilot input ----------
    pt = (pilot_text or "").strip()
    if not pt:
        debug_line("[VOICE] Empty transcript.")
        return "❌", "None", "❌ No transcript. Please type or record a request.", None, "Awaiting pilot reply…", snapshot_debug_dump(), None

    LAST_PILOT_TEXT = pt

    # Parse pilot → ctx (phase/runway/cs/etc.)
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
    phase = ctx.get("phase", "approach")

    # Phase cues (stronger than parse heuristics)
    ground_cues_rx = re.compile(r'\b(ground|holding\s+short|hold\s+short|line\s*up|gate|taxi|ramp)\b', re.I)
    apch_cues_rx   = re.compile(r'\b(approach|inbound|final|established|glideslope|localizer|cleared\s+to\s+land)\b', re.I)
    low_pt = pt.lower()
    if apch_cues_rx.search(low_pt):
        phase = "approach"
    elif ground_cues_rx.search(low_pt):
        phase = "ground"

    # Seed/refresh call state
    st = CALLSTATE.setdefault(cs, {"phase": phase, "runway": rw, "gate": ctx.get("gate"), "stage": None, "last_dt": None})
    st["phase"] = phase
    st["runway"] = rw

    # ---------- Monitor & stamp pilot line ----------
    monitor_header("XAION SIM START (VOICE CTX)")
    monitor_block("Pilot", phase, rw)
    ctx_line = context_frame_from_ctx(ctx, push=True)
    pilot_ui = _stamp_ui(f"{EMOJI_PILOT} Pilot: {pt}", ctx)
    VOICE_DIALOGUE = f"{VOICE_DIALOGUE}\n{pilot_ui}".strip()
    _ctx_dump_lines(ctx_line, dialogue_text=VOICE_DIALOGUE)

    # ---------- Guards (bad callsign/runway) ----------
    guarded, tower_line = guard_or_tower(cs if cs and cs.upper() != "UNKNOWN" else None, rw)
    if guarded:
        # tower_line is already branded & normalized by guard_or_tower()
        atc_unstamped = f"{EMOJI_ATC} ATC: {tower_line}"
        atc_block     = __stamp_block(atc_unstamped, ctx)
        VOICE_DIALOGUE += f"\n{atc_block}"
        audio = __speak_block(atc_unstamped)
        status = "" if audio else ("ATC audio unavailable (check ELEVEN_API_KEY)" if _has_eleven else "Text OK — TTS not configured.")
        monitor_header("XAION SIM END (VOICE CTX)")
        return pt, "Invalid callsign/runway in pilot transmission.", VOICE_DIALOGUE, (audio or None), status, snapshot_debug_dump(), None

    # ---------- “Say again” → restate last ATC core ----------
    if REPEAT_RX.search(pt):
        last_ui = _last_atc(VOICE_DIALOGUE)
        if last_ui:
            last_core = strip_atc_ui_to_core(last_ui, cs)
            atc_line  = atc_prefix_and_dedup(cs, last_core)
            atc_unstamped = f"{EMOJI_ATC} ATC: {atc_line}"
            atc_block     = __stamp_block(atc_unstamped, ctx)
            VOICE_DIALOGUE += f"\n{atc_block}"
            audio = __speak_block(atc_unstamped)
            status = "" if audio else ("ATC audio unavailable (check ELEVEN_API_KEY)" if _has_eleven else "Text OK — TTS not configured.")
            # Update struct for next early-RB
            last_ui2 = _last_atc(VOICE_DIALOGUE)
            last_core2 = strip_atc_ui_to_core(last_ui2, cs) if last_ui2 else ""
            VOICE_LAST_ATC_STRUCT = _extract_items(last_core2) if last_core2 else None
            monitor_header("XAION SIM END (VOICE CTX)")
            return pt, "None", VOICE_DIALOGUE, (audio or None), status, snapshot_debug_dump(), None

    # ---------- Early readback-correct (before any ladder) ----------
    if VOICE_LAST_ATC_STRUCT and _pilot_covers_items(pt, VOICE_LAST_ATC_STRUCT):
        debug_line("[VOICE] EARLY-RB ✓ → readback correct")
        if st.get("stage") is None:
            st["stage"] = "taxi_issued"
        atc_line = atc_prefix_and_dedup(cs, f"{cs}, readback correct.")
        atc_unstamped = f"{EMOJI_ATC} ATC: {atc_line}"
        atc_block     = __stamp_block(atc_unstamped, ctx)
        VOICE_DIALOGUE += f"\n{atc_block}"
        audio = __speak_block(atc_unstamped)
        status = "" if audio else ("ATC audio unavailable (check ELEVEN_API_KEY)" if _has_eleven else "Text OK — TTS not configured.")
        # Refresh structure after the new ATC
        last_ui2 = _last_atc(VOICE_DIALOGUE); last_core2 = strip_atc_ui_to_core(last_ui2, cs) if last_ui2 else ""
        VOICE_LAST_ATC_STRUCT = _extract_items(last_core2) if last_core2 else None
        monitor_header("XAION SIM END (VOICE CTX)")
        return pt, "None", VOICE_DIALOGUE, (audio or None), status, snapshot_debug_dump(), None

    # ---------- Forced flows ----------
    # GROUND ladder (taxi → handoff Tower → takeoff (+ Departure handoff))
    if phase == "ground":
        forced_unstamped = _vocal_ground_flow(cs, ctx, pt, VOICE_DIALOGUE)
        if forced_unstamped:
            atc_block = __stamp_block(forced_unstamped, ctx)
            VOICE_DIALOGUE += "\n" + atc_block
            audio = __speak_block(forced_unstamped)
            status = "" if audio else ("ATC audio unavailable (check ELEVEN_API_KEY)" if _has_eleven else "Text OK — TTS not configured.")
            # Update struct using the last ATC line now in dialogue
            last_ui2 = _last_atc(VOICE_DIALOGUE); last_core2 = strip_atc_ui_to_core(last_ui2, cs) if last_ui2 else ""
            VOICE_LAST_ATC_STRUCT = _extract_items(last_core2) if last_core2 else None
            monitor_header("XAION SIM END (VOICE CTX)")
            return pt, "None", VOICE_DIALOGUE, (audio or None), status, snapshot_debug_dump(), None

    # APPROACH ladder (check-in → approach clr → Tower handoff → landing clr)
    if phase == "approach":
        forced_unstamped = _vocal_approach_flow(cs, ctx, pt, VOICE_DIALOGUE)
        if forced_unstamped:
            atc_block = __stamp_block(forced_unstamped, ctx)
            VOICE_DIALOGUE += "\n" + atc_block
            audio = __speak_block(forced_unstamped)
            status = "" if audio else ("ATC audio unavailable (check ELEVEN_API_KEY)" if _has_eleven else "Text OK — TTS not configured.")
            # Update struct for early-RB on the next turn
            last_ui2 = _last_atc(VOICE_DIALOGUE); last_core2 = strip_atc_ui_to_core(last_ui2, cs) if last_ui2 else ""
            VOICE_LAST_ATC_STRUCT = _extract_items(last_core2) if last_core2 else None
            monitor_header("XAION SIM END (VOICE CTX)")
            return pt, "None", VOICE_DIALOGUE, (audio or None), status, snapshot_debug_dump(), None

    # ---------- Fallback: 1 LLM turn (ATC) ----------
    # Use your gen_once() which already styles, normalizes, and brand-prefixes output.
    try:
        atc_unstamped_line = gen_once("ATC", VOICE_DIALOGUE, {
            "phase": phase, "runway": rw, "callsign": cs
        })
        # Ensure string + single/multi-line safe
        atc_unstamped = (atc_unstamped_line or "").strip()
        if not atc_unstamped:
            # deterministic ultra-short fallback
            core = f"{cs}, say again."
            atc_unstamped = f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}{core}"
        elif not atc_unstamped.lower().startswith("🗼 atc:"):
            atc_unstamped = f"{EMOJI_ATC} ATC: {atc_unstamped}"
    except Exception as e:
        debug_line(f"[VOICE Fallback] gen_once error: {e}")
        core = f"{cs}, say again."
        atc_unstamped = f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}{core}"

    atc_block = __stamp_block(atc_unstamped, ctx)
    VOICE_DIALOGUE += "\n" + atc_block
    audio = __speak_block(atc_unstamped)
    status = "" if audio else ("ATC audio unavailable (check ELEVEN_API_KEY)" if _has_eleven else "Text OK — TTS not configured.")

    # Update structure for next-turn early-RB
    last_ui2 = _last_atc(VOICE_DIALOGUE); last_core2 = strip_atc_ui_to_core(last_ui2, cs) if last_ui2 else ""
    VOICE_LAST_ATC_STRUCT = _extract_items(last_core2) if last_core2 else None

    monitor_header("XAION SIM END (VOICE CTX)")
    return pt, "None", VOICE_DIALOGUE, (audio or None), status, snapshot_debug_dump(), None

# -------------------------------
# Triplet helpers (ATC LLM → Pilot LLM → ATC confirm)
# -------------------------------
def llm_style_atc_from_core(callsign: str,
                            atc_core: str,
                            dialogue: str,
                            ctx: dict,
                            must_keep: tuple[str, ...] = ()) -> str:
    """
    Style 'atc_core' into a single, natural US-ATC sentence WITHOUT brand and WITHOUT a leading callsign.
    Guarantees that any tokens in 'must_keep' survive; otherwise returns the original core.
    """
    raw_core = (atc_core or "").strip()
    # Always build the core WITH callsign so the model has context...
    seed = f"ATC: {callsign}, {raw_core}"
    prompt = (
        "Rewrite the ATC line below as ONE concise US ATC transmission.\n"
        "- Keep all numbers/frequencies/runways exactly.\n"
        "- No meta, no brackets, no brand text, no extra sentences.\n"
        f"- Address the aircraft once by its callsign at the start.\n\n"
        f"INPUT:\n{seed}\n\nOUTPUT (one sentence): ATC: {callsign}, ..."
    )

    try:
        tok = phi4_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=768).to(device)
        out = phi4_model.generate(
            tok.input_ids, attention_mask=tok.attention_mask,
            max_new_tokens=64, do_sample=True, temperature=0.6, top_p=0.9,
            repetition_penalty=1.08, no_repeat_ngram_size=3,
            pad_token_id=phi4_tokenizer.eos_token_id, eos_token_id=phi4_tokenizer.eos_token_id,
        )
        txt = phi4_tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception:
        txt = seed

    # Keep to one sentence, strip any roles/brand
    txt = re.sub(r"(?is)^.*?ATC\s*:\s*", "", txt).strip()
    txt = re.sub(r"(?i)\bThis\s+is\s+XAION\s+CONTROL\.?\s*", "", txt)
    txt = re.split(r"[.\n]", txt, 1)[0].strip()

    # Remove any *run* of leading callsigns; return the body only
    txt = re.sub(rf"(?i)^\s*(?:{re.escape(callsign)}\b[,\s:\-;]*)+", "", txt).strip(" ,;")

    # Token guarantee: if any required token is missing, fall back to original core (body only)
    low = txt.lower()
    if must_keep and not all(tok in low for tok in must_keep):
        txt = re.sub(rf"(?i)^\s*(?:{re.escape(callsign)}\b[,\s:\-;]*)+", "", raw_core).strip(" ,;")

    # Final cleanup
    txt = _normalize_tower_and_runway(txt)
    return txt.rstrip(".") + "."


def llm_atc_confirm_from_atc_core_decision(cs: str, decision: str, atc_line_core: str, pilot_rb: str) -> str:
    """Deterministic confirm text (no brand/callsign added here)."""
    if decision == "READBACK_CORRECT":
        return "readback correct."
    if decision == "SAY_AGAIN":
        return "say again the full instruction."
    # decision == RESTATE → restate authoritative instruction core
    # Keep the original ATC core as the safe default.
    return re.sub(rf'^\s*{re.escape(cs)}\s*,\s*', '', atc_line_core.strip(), flags=re.I)

def llm_atc_confirm_from_core(cs: str, atc_line_core: str, pilot_rb: str, dialogue: str, ctx: dict) -> str:
    """
    Confirm/correct the pilot's readback.
    Deterministic for CORRECT/SAY-AGAIN; LLM only for RESTATE;
    Validates the LLM output; always returns a branded + deduped UI line.
    """
    items    = _extract_items(atc_line_core)
    rb_clean = re.sub(r"^\s*(?:🧑‍✈️\s*)?Pilot:\s*", "", (pilot_rb or ""), flags=re.I).strip()

    if not rb_clean or is_ack_only(rb_clean) or REPEAT_RX.search(rb_clean):
        decision = "SAY_AGAIN"
    elif _pilot_covers_items(rb_clean, items):
        decision = "READBACK_CORRECT"
    else:
        decision = "RESTATE"

    # Deterministic paths first
    if decision in ("READBACK_CORRECT", "SAY_AGAIN"):
        return atc_prefix_and_dedup(cs, llm_atc_confirm_from_atc_core_decision(cs, decision, atc_line_core, rb_clean))

    # RESTATE → try LLM, but validate; otherwise fall back to deterministic restate
    try:
        prompt = f"""
You are ATC. Restate the correct instruction in ONE sentence (<= 18 words).
Start with CALLSIGN once (no brand). Include RUNWAY/HOLD SHORT/CROSS and numbers (ALT/HDG/SPD/freq) as applicable.
No meta, no angle brackets.

ATC_CORE: {cs}, {atc_line_core.strip()}
PILOT_READBACK: {rb_clean}

Output (strict):
ATC: {cs}, <restate>.
""".strip()

        tok = phi4_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=768).to(device)
        out = phi4_model.generate(
            tok.input_ids, attention_mask=tok.attention_mask,
            max_new_tokens=64, do_sample=True, temperature=0.6, top_p=0.9,
            repetition_penalty=1.08, no_repeat_ngram_size=3,
            pad_token_id=phi4_tokenizer.eos_token_id, eos_token_id=phi4_tokenizer.eos_token_id,
        )
        text = phi4_tokenizer.decode(out[0], skip_special_tokens=True)
        if "ATC:" in text:
            text = text.split("ATC:", 1)[-1]
        text = _strip_role_leaks(clean_response(text, callsign=cs))
        text = re.sub(r"<[^>]*>", "", text).strip(" ,.;")

        # Throw away useless outputs (e.g., callsign only)
        core_no_cs = re.sub(rf"^\s*{re.escape(cs)}\s*,\s*", "", text, flags=re.I).strip()
        has_keyword = bool(re.search(r"\b(readback correct|say again|hold short|cross|cleared|heading|maintain|descend|climb|contact|speed|runway)\b", core_no_cs, re.I))
        if len(core_no_cs) < 6 or not has_keyword:
            raise ValueError("Confirm LLM under-output")

        return atc_prefix_and_dedup(cs, text)

    except Exception:
        # Deterministic restate fallback
        return atc_prefix_and_dedup(cs, llm_atc_confirm_from_atc_core_decision(cs, "RESTATE", atc_line_core, rb_clean))


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
    Build a single ATC → Pilot pair using the LLMs but with strict guards.
    Returns a list of tuples: [(role, ui_text, tts_text), ...]
    - role: "atc" | "pilot"
    - ui_text: with role tag/emoji; no stamps (runner stamps)
    - tts_text: clean speech text (brand is okay; TTS normalizer handles it)
    """
    # 1) ATC line (core rephrased safely)
    core = llm_style_atc_from_core(cs, atc_core, dialogue, {"phase": phase, "runway": rw, "callsign": cs})
    atc_ui  = f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}{core}"
    atc_tts = f"{cs}, {core}"

    # 2) Pilot readback (canonicalize and auto-synthesize if weak)
    #    - We pass the dialogue that already includes the ATC line so the Pilot model can read back correctly.
    forced_dialogue = f"{dialogue.strip()}\n{atc_ui}"
    pilot_line_raw = gen_once("Pilot", forced_dialogue, {"phase": phase, "runway": rw, "callsign": cs})
    pilot_ui = canonicalize_pilot_ui(pilot_line_raw, cs)

    # If the canonicalizer still ends up skinny, synthesize from the ATC core
    if len(strip_role_prefix(pilot_ui)) < 10 or SAY_AGAIN_RX.search(pilot_ui) or cs.upper() not in pilot_ui.upper():
        synth = _pilot_readback_from_atc_text(cs, core) or f"Pilot: {cs}, say again the full instruction."
        pilot_ui = canonicalize_pilot_ui(synth, cs)

    return [
        ("atc",   atc_ui,   atc_tts),
        ("pilot", pilot_ui, pilot_ui),  # tts_text: pilot line as-is; TTS cleaner strips "Pilot:" label
    ]




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

    handoff = _maybe_departure_handoff(cs, st)
    if handoff:
        out.extend(emit_llm_triplet(cs, handoff, "ground", rw, dialogue_for_llm))

    return out

def _advance_landing(cs: str, r, gate_hint: str | None, rw: str, dialogue: str):
    """
    Landing ladder driven by simple stage state + last pilot content:
      • Initial: vector & 2000
      • Next: descend 1300 until established, cleared ILS
      • When pilot says 'established': hand off to Tower 119.1
      • If pilot says '[x]-mile final': clear to land (runway availability checked)
    Returns list of (role, ui_text, tts_text).
    """
    st = CALLSTATE.setdefault(cs, {"phase": "approach", "runway": rw, "stage": None})
    st["phase"] = "approach"; st["runway"] = rw

    # Find the most recent pilot line for this callsign
    last_pilot = ""
    for ln in (dialogue or "").splitlines()[::-1]:
        if "Pilot:" in ln and re.search(rf"\b{re.escape(cs)}\b", ln, re.I):
            last_pilot = ln
            break
    low = last_pilot.lower()

    final_rx      = re.compile(r'\b(\d+(?:\.\d+)?)\s*mile\s*final\b', re.I)
    established_rx= re.compile(r'\bestablished\b', re.I)

    emitted = []

    # 1) If we detect short/final now → try to clear to land (runway gate)
    if final_rx.search(low) or re.search(r"\bshort\s+final\b", low):
        ok, alt_if_blocked = gate_runway_clearance("land", rw, cs)
        if ok:
            core = f"cleared to land Runway {rw}."
        else:
            core = alt_if_blocked  # e.g., "continue, expect late landing clearance."
        emitted += emit_llm_triplet(cs, core, "approach", rw, dialogue)
        st["stage"] = "cleared_land" if ok else "continue"
        return emitted

    # 2) Normal ladder
    if st.get("stage") is None:
        core = f"turn heading {int(_hdg_to_runway(rw))}, maintain 2000 until established."
        emitted += emit_llm_triplet(cs, core, "approach", rw, dialogue)
        st["stage"] = "vector_issued"
        return emitted

    if st.get("stage") == "vector_issued":
        core = f"descend and maintain 1300 until established, cleared ILS Runway {rw}."
        emitted += emit_llm_triplet(cs, core, "approach", rw, dialogue)
        st["stage"] = "ils_cleared"
        return emitted

    # 3) Pilot reports "established" → hand off to Tower
    if established_rx.search(low) and st.get("stage") in ("vector_issued", "ils_cleared"):
        core = f"contact Tower {KGSO_TOWER_FREQ}."
        emitted += emit_llm_triplet(cs, core, "approach", rw, dialogue)
        st["stage"] = "handoff_twr_app"
        return emitted

    # 4) Otherwise keep it short and safe: "continue"
    core = "continue."
    emitted += emit_llm_triplet(cs, core, "approach", rw, dialogue)
    return emitted


# -------------------------------
# Full Simulation (state-driven, Pilot-first per event)
# -------------------------------
def build_events_in_window_seconds(start_idx: int, seconds: float, callsign_only: str | None = None):
    """
    Returns a list of (dt_show, idx, pilot_seed_hint) tuples for the full simulation window.
    - Uses order_rows_for_full_sim(...) so the *selected* row is always first.
    - Drops any earlier-sorted traffic that shares the exact same DT as the selected row.
    - If callsign_only is provided, limits rows to that callsign.
    """
    if start_idx is None or not len(flat_df):
        return []

    # Core ordering (selected first; suppress earlier same-DT)
    seq = order_rows_for_full_sim(
        start_idx,
        window_sec=int(seconds),
        suppress_same_dt_before_selected=True
    )

    # Optional: restrict to a single callsign if requested
    if callsign_only:
        def _match_cs(i):
            r = flat_df.loc[i]
            return (_safe_callsign_from_row(r) or "").strip().upper() == callsign_only.strip().upper()
        seq = [i for i in seq if _match_cs(i)]

    # Use each row's own DT if available; otherwise fall back to the selected row's DT (or now)
    def _row_dt(i):
        dt = str(flat_df.loc[i].get("DT Time") or "").strip()
        if re.fullmatch(r"\d{6}Z", dt):
            return dt
        base = str(flat_df.loc[start_idx].get("DT Time") or "").strip()
        return base if re.fullmatch(r"\d{6}Z", base) else _now_dt_tag()

    events = []
    for i in seq:
        events.append((_row_dt(i), i, ""))  # pilot_seed_hint not required; kept for API compatibility
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

    # Selected scenario -> start index
    scen_idx   = options.index(scenario_text)
    start_idx  = DT_INDEX_MAP[comm_type][scen_idx]
    start_row  = flat_df.loc[start_idx]
    locked_cs  = _safe_callsign_from_row(start_row)

    # Make sure global SIM_START reflects this choice (helps other parts of the app)
    SIM_START["idx"]   = start_idx
    SIM_START["dt"]    = flat_df.loc[start_idx, "__dt"]
    SIM_START["ident"] = locked_cs

    # Build the timeline using the new ordering policy
    events = build_events_in_window_seconds(start_idx, seconds, callsign_only=None)

    dialogue_accum = ""
    for dt_show, idx, pilot_seed_in in events:
        if stop_flag.get("stop"):
            debug_line("[FULLSIM] Stop requested.")
            break

        r  = flat_df.loc[idx]
        cs = _safe_callsign_from_row(r) or "UNKNOWN"
        gate_hint = _gate_from_text(pilot_seed_in) or pick_gate_for_callsign(cs)
        rw_base   = runway_from_heading(r.get("track_deg")) or DEFAULT_RWY_PRIMARY
        rw        = normalize_lr(rw_base, gate_hint, cs)
        phase     = "ground" if r.get("on_ground", False) else "approach"

        # --- Simulated Pilot seed (from row, with your existing fallback) ---
        monitor_block("Pilot", phase, rw)
        ctx_line          = context_frame_from_idx(idx, push=True)
        pilot_seed_default = fmt_takeoff_from_idx(idx) if r.get("on_ground", False) else fmt_landing_from_idx(idx)
        pilot_seed         = _pilot_seed_for_state(cs, phase, rw, idx) or pilot_seed_default

        pilot_ui_safe = _ui_with_role("pilot", f"Pilot: {pilot_seed}")
        dialogue_accum = f"{dialogue_accum}\n\n{_ensure_stamped(pilot_ui_safe, dt_show)}"
        _ctx_dump_lines(ctx_line, dialogue_text=dialogue_accum)

        pilot_audio = tts_elevenlabs(pilot_seed, voice_id=PILOT_VOICE_ID) if _has_eleven else None
        yield pilot_seed, "None", dialogue_accum.strip(), None, "", snapshot_debug_dump(), pilot_audio
        _sleep_for_audio(pilot_audio)

        # Keep monitor active
        _ = monitor_decide_next_role(dialogue_accum, phase, cs)

        # --- LLM triplet (ATC -> Pilot) or rule-based fallback ---
        emitted = (
            _advance_takeoff(cs, r, gate_hint, rw, dialogue_accum)
            if phase == "ground"
            else _advance_landing(cs, r, gate_hint, rw, dialogue_accum)
        )
        if not emitted:
            core = _rule_based_atc_for_row(r, cs, rw)
            emitted = emit_llm_triplet(cs, core, phase, rw, dialogue_accum)

        # Optional “readback correct / say again” closer
        if emitted:
            first_atc = next((e for e in emitted if e[0] == "atc"), None)
            pilot_e   = next((e for e in emitted if e[0] == "pilot"), None)
            atc_core  = strip_atc_ui_to_core(first_atc[1], cs) if first_atc else ""
            items     = _extract_items(atc_core)
            ok        = _pilot_covers_items(pilot_e[1] if pilot_e else "", items)
            conf_core = f"{cs}, {'readback correct.' if ok else 'say again the full instruction.'}"
            conf_body = re.sub(rf"^\s*{re.escape(cs)}\s*,\s*", "", conf_core, flags=re.I)
            conf_ui   = xaion_prefix(cs) + conf_body
            conf_tuple = ("atc", f"{EMOJI_ATC} ATC: {conf_ui}", conf_core)
            if emitted[-1][0] != "atc":
                emitted.append(conf_tuple)

        # --- Emit stamped lines + audio ---
        for role, ui_text, tts_text in emitted:
            monitor_block("Pilot" if role.startswith("pilot") else "ATC", phase, rw)
            _ctx_dump_lines(ctx_line, dialogue_text=dialogue_accum)

            safe_ui = _ui_with_role(role, ui_text)
            dialogue_accum = f"{dialogue_accum}\n\n{_ensure_stamped(safe_ui, dt_show)}"

            if role.startswith("pilot"):
                audio = tts_elevenlabs(tts_text, voice_id=PILOT_VOICE_ID) if (_has_eleven and tts_text) else None
                yield pilot_seed, "None", dialogue_accum.strip(), None, "", snapshot_debug_dump(), audio
                _sleep_for_audio(audio)
            else:
                audio = tts_elevenlabs(tts_text, voice_id=ELEVEN_VOICE_ID_DEFAULT) if (_has_eleven and tts_text) else None
                status = "" if audio else ("ATC audio unavailable (check ELEVEN_API_KEY)" if _has_eleven else "Text OK — TTS not configured.")
                yield pilot_seed, "None", dialogue_accum.strip(), audio, status, snapshot_debug_dump(), None
                _sleep_for_audio(audio)

            _ = monitor_decide_next_role(dialogue_accum, phase, cs)

    monitor_header("XAION SIM END (FULL)")

# -------------------------------
# Single Handoff Simulation
# -------------------------------
def single_handoff(comm_type: str, scenario_text: str):
    """
    State-driven, DT-stamped single handoff.
    Yields two or more frames: Pilot seed → ATC/Pilot/ATC confirm triplet(s),
    with audio split so only one side plays each frame.
    """
    # Fresh run state
    CONTEXT_HISTORY.clear()
    CALLSTATE.clear()

    options = DT_SCENARIOS.get(comm_type, [])
    if not options or scenario_text not in options:
        debug_line("[SINGLE] Invalid scenario selection.")
        yield "❌", "None", "❌ Select a scenario from the list.", None, "", snapshot_debug_dump(), None
        return

    scen_idx = options.index(scenario_text)
    flat_idx = DT_INDEX_MAP[comm_type][scen_idx]
    r        = flat_df.loc[flat_idx]

    cs   = (_safe_callsign_from_row(r) or "UNKNOWN").strip()
    rw   = runway_from_heading(r.get("track_deg")) or DEFAULT_RWY_PRIMARY
    gate = _gate_from_text(scenario_text) or pick_gate_for_callsign(cs)
    dt_show = r.get("parent_dt", r.get("DT Time", "Unknown"))
    phase   = "ground" if r.get("on_ground", False) else "approach"

    # Seed Pilot transmission (DT stamped)
    monitor_header("XAION SIM START (SINGLE HANDOFF)")
    monitor_block("Pilot", phase, rw)
    ctx_line = context_frame_from_idx(flat_idx, push=True)
    pilot_line = scenario_text.strip()
    pilot_ui   = f"{EMOJI_PILOT} Pilot: {pilot_line}"
    dialogue   = f"🕒 [{dt_show}] {pilot_ui}"
    _ctx_dump_lines(ctx_line, dialogue_text=dialogue)

    pilot_audio = tts_elevenlabs(pilot_line, voice_id=PILOT_VOICE_ID) if _has_eleven else None
    yield pilot_line, "None", dialogue, None, "", snapshot_debug_dump(), pilot_audio
    _sleep_for_audio(pilot_audio)

    # Keep monitor active
    _ = monitor_decide_next_role(dialogue, phase, cs)

    # Advance one stage using the proper state machine (prevents repeats)
    stage_emits = _advance_takeoff(cs, r, gate, rw, dialogue) if phase == "ground" \
                  else _advance_landing(cs, r, gate, rw, dialogue)

    # Fallback: deterministic ATC core if nothing emitted
    if not stage_emits:
        core = _rule_based_atc_for_row(r, cs, rw)
        stage_emits = emit_llm_triplet(cs, core, phase, rw, dialogue)

    # Ensure there is an explicit confirm at the end (readback correct / say again)
    saw_pilot = any(role == "pilot" for role, *_ in stage_emits)
    if saw_pilot and stage_emits[-1][0] != "atc":
        # Extract items for validator
        first_atc = next((e for e in stage_emits if e[0] == "atc"), None)
        atc_core  = strip_atc_ui_to_core(first_atc[1], cs) if first_atc else ""
        items     = _extract_items(atc_core)
        pilot_ui_line = next((ui for role, ui, _ in stage_emits if role == "pilot"), "")
        ok = _pilot_covers_items(strip_role_prefix(pilot_ui_line), items)
        conf_core = f"{cs}, {'readback correct.' if ok else 'say again the full instruction.'}"
        conf_ui   = f"{EMOJI_ATC} ATC: {xaion_prefix(cs)}{re.sub(rf'^s*{re.escape(cs)}s*,s*', '', conf_core, flags=re.I)}"
        stage_emits.append(("atc", _ensure_stamped(conf_ui, dt_show), conf_core))

    # Emit triplet with DT stamps and split audio (one side at a time)
    for role, ui_text, tts_text in stage_emits:
        dialogue = f"{dialogue}\n{_ensure_stamped(ui_text, dt_show)}"
        if role == "pilot":
            audio = tts_elevenlabs(tts_text, voice_id=PILOT_VOICE_ID) if (_has_eleven and tts_text) else None
            yield pilot_line, "None", dialogue, None, "", snapshot_debug_dump(), audio
        else:
            audio = tts_elevenlabs(tts_text, voice_id=ELEVEN_VOICE_ID_DEFAULT) if (_has_eleven and tts_text) else None
            yield pilot_line, "None", dialogue, audio, "", snapshot_debug_dump(), None
        _sleep_for_audio(audio)

    monitor_header("XAION SIM END (SINGLE HANDOFF)")



# Unified run handler + Gradio UI (Pilot transcript visible only in Vocal Input)
# ===============================

# -------------------------------
# Minimal shims for scenario → df index and simple sim loops
# -------------------------------

def _df_idx_from_scenario(kind: str, scenario_text: str) -> int:
    """Map scenario text → flat_df row index using DT_SCENARIOS/DT_INDEX_MAP."""
    try:
        scen_list = DT_SCENARIOS.get(kind, []) or []
        scen_idx = scen_list.index(scenario_text)
        return DT_INDEX_MAP[kind][scen_idx]
    except Exception:
        return 0

def single_handoff(comm_type: str, scenario_text: str):
    """
    Yield two turns: ATC answer then Pilot readback.
    Step shape matches run_simulation’s expectations:
      (pilot_line, anomaly, dialogue, atc_audio, status, dbg, pilot_audio)
    """
    df_idx = _df_idx_from_scenario(comm_type, scenario_text or "")
    dialogue = f"{EMOJI_PILOT} Pilot: {scenario_text.strip()}"

    # ATC turn
    atc_line = gen_once_idx("ATC", dialogue, df_idx)
    atc_ui = f"{EMOJI_ATC} ATC: {atc_line}"
    dialogue = f"{dialogue}\n{atc_ui}"
    atc_audio = tts_elevenlabs(atc_line, voice_id=ELEVEN_VOICE_ID_DEFAULT)
    status = "" if atc_audio else ("Text OK — TTS not configured." if not _has_eleven else "")
    yield scenario_text, "None", dialogue, atc_audio, status, snapshot_debug_dump(), None

    # Pilot readback
    pilot_line = gen_once_idx("Pilot", dialogue, df_idx)
    pilot_ui = f"{EMOJI_PILOT} {pilot_line}"
    dialogue = f"{dialogue}\n{pilot_ui}"
    pilot_audio = tts_elevenlabs(pilot_line, voice_id=PILOT_VOICE_ID) if _has_eleven else None
    yield "", "None", dialogue, None, "", snapshot_debug_dump(), pilot_audio


# -------------------------------
# Unified run handler
# -------------------------------
def run_simulation(mode, comm_type, scenario_text, vocal_text, fullsim_window, stop_state):
    """
    Unified runner that:
      • uses voice_step() (already stamped) for Vocal Input,
      • uses the stateful single_handoff()/run_full_simulation() above for sim modes,
      • preserves split-audio behavior.
    """
    stop_state["stop"] = False
    debug_line(f"[RUN] mode={mode}")
    yield (vocal_text or ""), "None", "", None, "Starting…", snapshot_debug_dump(), None

    if mode == "Vocal Input":
        pt, anomaly, dialogue, atc_audio, status, dbg, _ = voice_step((vocal_text or "").strip())

        # frame A: show text, no audio yet
        yield pt, anomaly, dialogue, None, status, dbg, None

        # frame B: play ATC audio only (if any)
        if atc_audio:
            time.sleep(max(0.0, AUDIO_GAP_SEC))
            yield pt, anomaly, dialogue, atc_audio, status, snapshot_debug_dump(), None
            _sleep_for_audio(atc_audio)
        return

    # --- Sim modes ---
    def _seconds(label: str) -> float:
        return {"30 sec": 30.0, "1 min": 60.0, "2 min": 120.0}.get(label or "1 min", 60.0)

    if mode == "Simulated Dialogue (Single Handoff)":
        # Pilot audio frame → ATC audio frame (split)
        for step in single_handoff(comm_type, scenario_text):
            pilot_line, anomaly, dialogue, atc_audio, status, dbg, pilot_audio = step
            yield "", anomaly, dialogue, None, status, dbg, pilot_audio
            _sleep_for_audio(pilot_audio)
            yield "", anomaly, dialogue, atc_audio, status, snapshot_debug_dump(), None
            _sleep_for_audio(atc_audio)
        return

    # Full Simulation
    for step in run_full_simulation(comm_type, scenario_text, _seconds(fullsim_window), stop_state):
        pilot_seed, anomaly, dialogue, atc_audio, status, dbg, pilot_audio = step
        yield "", anomaly, dialogue, None, status, dbg, pilot_audio      # pilot-only frame
        _sleep_for_audio(pilot_audio)
        yield "", anomaly, dialogue, atc_audio, status, snapshot_debug_dump(), None  # ATC-only frame
        _sleep_for_audio(atc_audio)


def stop_simulation(stop_state):
    stop_state["stop"] = True
    debug_line("[CTRL] Stop requested.")
    return "Stopping…"


# Only define _first_open_port here if Part 1/2 didn’t.
try:
    _first_open_port  # type: ignore[name-defined]
except NameError:
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
            # a == pilot text; 
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
