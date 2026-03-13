'''
MultiAnomalyDetector — Detecta anomalias e gera alertas com timestamps em vídeos em tempo real.

Categorias detectadas:
1. Acidente de Trânsito
2. Violação de Linha de Parada
3. Violação de Semáforo
4. Ambulância Presa
5. Furto
6. Atividade Suspeita
+ Detectores adicionais: fogo, queda, contra-mão, etc.

Design otimizado para CPU:
- Usa BLIP base (2x mais rápido que o modelo grande)
- Perguntas VQA só em quadros marcados como anômalos
- Amostragem de quadros esparsa (configurável, padrão 0.5 fps)
- Inferência Torch com `no_grad()` e modo `eval()`

Estrutura de saída:
OUTPUT_ROOT/
  <tipo-de-anomalia>/
    <timestamp>_frame<N>.jpg — captura anotada
    <timestamp>_frame<N>.txt — log de legenda
'''

from __future__ import annotations

import base64
import math
import os
import tempfile
import threading
import warnings
from datetime import datetime
from io import BytesIO
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from flask_socketio import SocketIO, join_room
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim
from werkzeug.utils import secure_filename

try:
    from transformers import BlipForConditionalGeneration, BlipProcessor
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False

warnings.filterwarnings("ignore")

#  CONFIGURATION

# frames extracted per second of source video (0.5 -> 1 frame every 2 s)
EXTRACT_FPS: float = 0.5
# hard cap — prevents memory blow-up on very long clips
MAX_FRAMES: int = 60
# Resolution used for frame-difference computation only (keep small for speed)
DIFF_SIZE: tuple[int, int] = (96, 96)
# root output directory — one sub-folder is created per anomaly type
OUTPUT_ROOT: str = r"C:\Users\Gustavo\Desktop\DrApurbasTasks\IncidentAnomaliesAlerts\output_video"
# JPEG quality for saved screenshots (lower → faster disk write, still legible)
JPEG_QUALITY: int = 85
# BLIP checkpoint — "base" is ~2× faster than "large" on CPU with minimal
# accuracy loss for the short, constrained captions we generate.
BLIP_CHECKPOINT: str = "Salesforce/blip-image-captioning-base"

#  CAPTION & VQA PROMPTS

# Conditional captioning prompts 
PROMPT_NORMAL     = "a traffic camera photo of"
PROMPT_ANOMALY    = "a traffic camera showing an incident where"
PROMPT_CROSSWALK  = "a traffic camera showing a crosswalk where"

# VQA question bank 
# core — always evaluated on anomalous frames
VQA_ACCIDENT   = "is there a vehicle collision or traffic accident in this image?"
VQA_FIRE       = "is there fire or smoke visible in this image?"

# gated — only queried when prior evidence justifies the cost
VQA_FALL       = "has a person fallen down or been knocked over in this image?"
VQA_WRONG_WAY  = "is a vehicle driving on the wrong side of the road in this image?"
VQA_SEVERITY   = "how severe is the incident? answer only: minor, moderate, or severe"

# crosswalk — dual-question consensus prevents false positives
VQA_PED_CROSSWALK      = "is a pedestrian walking across a zebra crossing or crosswalk in this image?"
VQA_VEHICLE_CROSS_1    = "is a car, truck, bus, or motorcycle on top of or blocking a zebra crossing?"
VQA_VEHICLE_CROSS_2    = "is any vehicle stopped, parked, or driving over a striped pedestrian crossing?"
VQA_IS_MOTOR_VEHICLE   = "is there a car, truck, bus, van, or motorcycle visible in this image?"

# positive-answer normalisation
_YES_TOKENS = ("yes", "yeah", "true", "correct", "yep")
_VALID_SEVERITIES = {"minor", "moderate", "severe"}


def is_yes(answer: str) -> bool:
    """Return True if the VQA answer begins with any recognised affirmative."""
    token = (answer or "").strip().lower()
    return any(token.startswith(y) for y in _YES_TOKENS)


def parse_severity(raw: str) -> str:
    """Extract the first word from a severity answer and validate it."""
    if not raw or raw == "n/a":
        return "n/a"
    first = raw.strip().lower().split()[0]
    return first if first in _VALID_SEVERITIES else "n/a"

#  ALERT RULE TABLE  (first match wins — ordered by priority)

# Each entry:  ( [keyword_list], "ALERT LABEL", "SEVERITY" )
# Severity levels:  CRITICAL → HIGH → MEDIUM → YELLOW → INFO → NORMAL
ALERT_RULES: list[tuple[list[str], str, str]] = [
    # CRITICAL 
    (["crash", "collision", "collide", "accident", "wreck", "smash", "impact",
      "overturned", "flipped", "rolled over", "pile-up", "pileup",
      "rear-end", "head-on", "sideswipe", "t-bone", "spun out", "skidded",
      "slammed", "rammed", "struck", "hit by", "ran into", "drove into",
      "traffic accident", "vehicle crash", "car crash", "car accident"],
     "COLLISION / CRASH", "CRITICAL"),

    (["fire", "smoke", "burning", "flame", "blaze", "engulfed", "on fire"],
     "FIRE / SMOKE", "CRITICAL"),

    (["stop line violation", "crossed stop line", "crossing stop line",
      "stop line breached", "ran stop line"],
     "STOP LINE VIOLATION", "CRITICAL"),

    (["red light violation", "ran red light", "crossed red light",
      "red light breached", "signal violation"],
     "RED LIGHT VIOLATION", "CRITICAL"),

    (["ambulance stuck", "ambulance blocked", "ambulance in traffic",
      "ambulance unable to move", "ambulance delayed"],
     "AMBULANCE STUCK IN TRAFFIC", "CRITICAL"),

    (["stealing", "theft", "robbery", "shoplifting",
      "people stealing", "person stealing", "item stolen"],
     "THEFT / STEALING", "CRITICAL"),

    (["suspicious activity", "suspicious person", "unusual activity",
      "unusual behavior", "strange activity"],
     "SUSPICIOUS ACTIVITY", "CRITICAL"),

    (["vehicle on crosswalk", "car on crosswalk", "truck on crosswalk",
      "motorcycle on crosswalk", "blocking crosswalk",
      "blocking pedestrian crossing", "stopped on crosswalk",
      "parked on crosswalk"],
     "VEHICLE BLOCKING CROSSWALK", "CRITICAL"),

    (["pedestrian on car", "person on top of car", "person climbing car"],
     "PEDESTRIAN ON VEHICLE", "CRITICAL"),

    # HIGH
    (["fall", "fallen", "falling", "knocked down", "lying on road",
      "pedestrian down", "person down", "run over"],
     "PERSON DOWN / FALL", "HIGH"),

    (["fight", "fighting", "violence", "attack", "assault", "brawl"],
     "VIOLENCE / ASSAULT", "HIGH"),

    (["debris", "obstacle", "object on road", "blocking the road"],
     "ROAD DEBRIS / HAZARD", "HIGH"),

    # MEDIUM
    (["speeding", "racing", "high speed", "reckless"],
     "RECKLESS DRIVING", "MEDIUM"),

    (["skid", "swerve", "lost control", "hydroplane"],
     "LOSS OF CONTROL", "MEDIUM"),

    (["stalled", "broken down", "stopped in lane", "disabled vehicle"],
     "STALLED VEHICLE", "MEDIUM"),

    #(informational warning) 
    (["pedestrian crossing", "person crossing", "pedestrian on crosswalk",
      "crossing the street", "crossing the road", "zebra crossing",
      "using crosswalk", "crossing at crosswalk"],
     "PEDESTRIAN ON CROSSWALK", "YELLOW"),

    # INFO (normal traffic scene) 
    (["traffic", "road", "street", "highway", "car", "vehicle", "truck",
      "bus", "motorcycle", "intersection", "driving", "lane", "signal"],
     "TRAFFIC SCENE", "INFO"),
]


def map_caption_to_alert(caption: str) -> tuple[str, str]:
    """Scan the caption against ALERT_RULES and return (label, severity)."""
    lower = caption.lower()
    for keywords, label, severity in ALERT_RULES:
        if any(kw in lower for kw in keywords):
            return label, severity
    return "NO ALERT", "NORMAL"


#  FLASK APPLICATION

app = Flask(__name__)
app.config["SECRET_KEY"]         = "multianomaly-secret-key"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB upload limit
app.config["UPLOAD_FOLDER"]      = tempfile.gettempdir()

CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# global model references — loaded once at startup
_processor: "BlipProcessor | None"                   = None
_model: "BlipForConditionalGeneration | None"        = None
_device: torch.device                                = torch.device("cpu")

# maps session_id -> background thread
active_sessions: dict[str, threading.Thread] = {}


#  MODEL INITIALISATION

def load_blip_model() -> None:
    """
    Load the BLIP base checkpoint into global variables.

    Uses the *base* variant rather than *large* for CPU throughput.
    Model is pinned to eval() mode and inference always runs inside
    torch.no_grad() to suppress gradient bookkeeping overhead.
    """
    global _processor, _model, _device

    if not BLIP_AVAILABLE:
        print("[WARN] transformers not installed — BLIP unavailable.")
        return

    print(f"[BLIP] Loading {BLIP_CHECKPOINT} …")
    _processor = BlipProcessor.from_pretrained(BLIP_CHECKPOINT, use_fast=True)
    _model     = BlipForConditionalGeneration.from_pretrained(
        BLIP_CHECKPOINT, ignore_mismatched_sizes=True
    )
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(_device).eval()
    print(f"[BLIP] Model ready on {_device}.")


#  BLIP INFERENCE HELPERS

def blip_caption(img: Image.Image, prompt: str | None = None) -> str:
    """
    Generate a caption for *img*.

    Parameters
    ----------
    img:    PIL RGB image
    prompt: Optional conditional prefix (e.g. PROMPT_ANOMALY)

    Returns
    -------
    Caption string with the prompt prefix stripped.
    """
    if not _processor or not _model:
        return "(BLIP unavailable)"

    with torch.no_grad():
        if prompt:
            inputs = _processor(images=img, text=prompt, return_tensors="pt").to(_device)
        else:
            inputs = _processor(images=img, return_tensors="pt").to(_device)

        # max_new_tokens=48 is sufficient for short traffic captions;
        # fewer tokens → faster generation on CPU.
        out = _model.generate(
            **inputs,
            max_new_tokens=48,
            num_beams=3,          # reduced from 5 — good quality / speed trade-off
            length_penalty=1.0,
        )

    text = _processor.decode(out[0], skip_special_tokens=True)
    if prompt and text.lower().startswith(prompt.lower()):
        text = text[len(prompt):].strip()
    return text


def blip_vqa(img: Image.Image, question: str) -> str:
    """
    Run Visual Question Answering.  Returns a short lowercase answer string.
    Always returns "n/a" on any error so callers can treat it safely.
    """
    if not _processor or not _model:
        return "n/a"
    try:
        with torch.no_grad():
            inputs = _processor(images=img, text=question, return_tensors="pt").to(_device)
            out    = _model.generate(**inputs, max_new_tokens=6)   # answers are 1–3 words
        return _processor.decode(out[0], skip_special_tokens=True).strip().lower()
    except Exception:
        return "n/a"

#  PER-FRAME ANALYSIS  (caption + VQA + alert classification)

def analyse_frame(
    img: Image.Image,
    is_anomaly: bool,
) -> tuple[str, str, str, dict[str, str]]:
    """
    Produce a caption, alert label, severity, and VQA evidence dict for *img*.

    VQA questions are gated on *is_anomaly* to avoid running 5+ expensive
    inference passes on every normal frame — the biggest single CPU saving.

    Returns
    -------
    caption      : descriptive sentence
    alert_label  : short alert type (e.g. "COLLISION / CRASH")
    severity     : "CRITICAL" | "HIGH" | "MEDIUM" | "YELLOW" | "INFO" | "NORMAL"
    vqa          : dict mapping question keys → raw model answers
    """
    vqa: dict[str, str] = {}

    # step 1: core VQA — always run on anomalous frames 
    if is_anomaly:
        vqa["accident"] = blip_vqa(img, VQA_ACCIDENT)
        vqa["fire"]     = blip_vqa(img, VQA_FIRE)
    else:
        vqa["accident"] = "n/a"
        vqa["fire"]     = "n/a"

    # step 2: crosswalk — dual-consensus check (run on all frames)
    vqa["ped_crosswalk"]   = blip_vqa(img, VQA_PED_CROSSWALK)
    vqa["vehicle_cross_1"] = blip_vqa(img, VQA_VEHICLE_CROSS_1)
    vqa["vehicle_cross_2"] = blip_vqa(img, VQA_VEHICLE_CROSS_2)

    vehicle_cross_votes         = (int(is_yes(vqa["vehicle_cross_1"]))
                                   + int(is_yes(vqa["vehicle_cross_2"])))
    vehicle_crosswalk_confirmed = vehicle_cross_votes >= 2
    accident_confirmed          = is_yes(vqa["accident"])
    fire_confirmed              = is_yes(vqa["fire"])
    ped_crosswalk_confirmed     = is_yes(vqa["ped_crosswalk"])

    # step 3: anti-false-positive gate for crosswalk vehicle
    # BLIP can confuse cyclists/skateboarders with motor vehicles.
    # only accept the vehicle-on-crosswalk finding if a motor vehicle is present.
    if vehicle_crosswalk_confirmed:
        vqa["is_motor_vehicle"] = blip_vqa(img, VQA_IS_MOTOR_VEHICLE)
        if not is_yes(vqa["is_motor_vehicle"]):
            vehicle_crosswalk_confirmed = False

    # step 4: choose caption prompt based on detected context 
    if vehicle_crosswalk_confirmed:
        caption = blip_caption(img, prompt=PROMPT_CROSSWALK)

    elif is_anomaly or accident_confirmed or fire_confirmed:
        caption = blip_caption(img, prompt=PROMPT_ANOMALY)
        # additional gated questions — only worthwhile on confirmed anomalies
        vqa["fall"]     = blip_vqa(img, VQA_FALL)
        vqa["wrong_way"] = blip_vqa(img, VQA_WRONG_WAY)
        vqa["severity"] = blip_vqa(img, VQA_SEVERITY)

    else:
        caption = blip_caption(img, prompt=PROMPT_NORMAL)
        vqa["fall"]     = "n/a"
        vqa["wrong_way"] = "n/a"
        vqa["severity"] = "n/a"

    # step 5: keyword-based alert classification
    alert_label, severity = map_caption_to_alert(caption)

    # step 6: VQA overrides (priority order — most critical first) 
    if vehicle_crosswalk_confirmed:
        alert_label, severity = "VEHICLE BLOCKING CROSSWALK", "CRITICAL"

    elif accident_confirmed and severity != "CRITICAL":
        alert_label, severity = "COLLISION / CRASH", "CRITICAL"

    elif fire_confirmed and severity != "CRITICAL":
        alert_label, severity = "FIRE / SMOKE", "CRITICAL"

    elif is_yes(vqa.get("fall", "no")) and severity not in ("CRITICAL", "HIGH"):
        alert_label, severity = "PERSON DOWN / FALL", "HIGH"

    elif is_yes(vqa.get("wrong_way", "no")) and severity != "CRITICAL":
        alert_label, severity = "WRONG-WAY VEHICLE", "CRITICAL"

    elif ped_crosswalk_confirmed and severity not in ("CRITICAL", "HIGH", "MEDIUM"):
        alert_label, severity = "PEDESTRIAN ON CROSSWALK", "YELLOW"

    # step 7: append human-readable severity tag to caption
    if is_anomaly or severity in ("CRITICAL", "HIGH"):
        sev_str = parse_severity(vqa.get("severity", "n/a"))
        if sev_str != "n/a":
            caption = f"{caption}  [severity: {sev_str}]"

    return caption, alert_label, severity, vqa

#  VIDEO FRAME EXTRACTION

def extract_frames(
    video_path: str,
    session_dir: str,
) -> list[dict]:
    """
    Decode the source video and sample one frame every 1/EXTRACT_FPS seconds.

    Returns a list of dicts:
        frame_idx  : sequential integer index
        time_sec   : timestamp in the source video (seconds)
        image      : PIL RGB Image
        path       : saved JPEG path on disk
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frames_dir = os.path.join(session_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    frames: list[dict] = []
    idx = 0
    sec = 0.0

    while len(frames) < MAX_FRAMES:
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, bgr = cap.read()
        if not ret:
            break

        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        fp  = os.path.join(frames_dir, f"frame_{idx:05d}_{sec:.2f}s.jpg")
        img.save(fp, quality=JPEG_QUALITY)

        frames.append({"frame_idx": idx, "time_sec": sec, "image": img, "path": fp})
        idx += 1
        sec += 1.0 / EXTRACT_FPS

    cap.release()
    return frames

#  ANOMALY SCORING  (frame-difference + SSIM)

def compute_anomaly_scores(
    frames: list[dict],
) -> tuple[list[float], list[float], float, float]:
    """
    Compute pixel-difference and SSIM between consecutive frames.

    Thresholds are set at mean ± 1.5 std, making them adaptive to the
    overall motion level of the input clip (fast motorway vs. quiet car park).

    Returns
    -------
    diffs   : per-frame L1 difference vs. previous frame
    ssims   : per-frame SSIM vs. previous frame
    thr_d   : anomaly threshold for differences (upper bound)
    thr_s   : anomaly threshold for SSIM (lower bound)
    """
    diffs: list[float] = [0.0]
    ssims: list[float] = [1.0]

    prev_arr   = np.array(frames[0]["image"].resize(DIFF_SIZE)).astype(np.float32)
    prev_gray  = cv2.cvtColor(np.uint8(prev_arr), cv2.COLOR_RGB2GRAY)

    for i in range(1, len(frames)):
        curr_arr  = np.array(frames[i]["image"].resize(DIFF_SIZE)).astype(np.float32)
        curr_gray = cv2.cvtColor(np.uint8(curr_arr), cv2.COLOR_RGB2GRAY)

        diffs.append(float(np.mean(np.abs(curr_arr - prev_arr))))
        ssims.append(float(ssim(curr_gray, prev_gray, data_range=255)))

        prev_arr, prev_gray = curr_arr, curr_gray

    arr_d = np.array(diffs)
    arr_s = np.array(ssims)

    thr_d = float(arr_d.mean() + 1.5 * arr_d.std())
    thr_s = float(arr_s.mean() - 1.5 * arr_s.std())

    return diffs, ssims, thr_d, thr_s


def label_frames(
    frames: list[dict],
    diffs: list[float],
    ssims: list[float],
    thr_d: float,
    thr_s: float,
) -> list[dict]:
    """
    Assign an anomaly flag to each frame using a hysteresis rule:
    - Enter anomaly state when difference exceeds threshold OR SSIM drops
    - Exit only when both metrics return to clearly normal levels

    Returns a list of result dicts (one per frame).
    """
    low_d       = thr_d * 0.5    # hysteresis lower edge for difference
    high_s      = thr_s * 1.5   # hysteresis upper edge for SSIM
    anom_active = False
    results: list[dict] = []

    for i, fr in enumerate(frames):
        d, s = diffs[i], ssims[i]

        if (d > thr_d) or (d > 30.0) or (s < thr_s):
            anom_active = True
        elif d < low_d and s > high_s:
            anom_active = False

        results.append({
            "frame_idx":  fr["frame_idx"],
            "time_sec":   fr["time_sec"],
            "difference": float(d),
            "ssim":       float(s),
            "is_anomaly": bool(anom_active),
        })

    return results

#  SCREENSHOT PERSISTENCE  (timestamped, organised by anomaly type)
def _sanitise_folder_name(name: str) -> str:
    """Replace characters that are illegal in directory names on Windows/Linux."""
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, "_")
    return name.strip().upper().replace(" ", "_")


def save_alert_screenshot(
    frame_data:  dict,
    result:      dict,
    caption:     str,
    alert_label: str,
    severity:    str,
    session_dir: str,
    color:       str = "red",
) -> tuple[str, str]:
    """
    Annotate the frame image and write it to:
        OUTPUT_ROOT/<ALERT_TYPE>/<timestamp>_frame<N>.jpg
        OUTPUT_ROOT/<ALERT_TYPE>/<timestamp>_frame<N>.txt

    Also saves into the session temp dir for web delivery.

    Returns
    -------
    b64_img : base64-encoded JPEG string for SocketIO transmission
    saved_path : absolute path of the saved JPEG
    """
    # draw banner on a copy 
    img_copy = frame_data["image"].copy()
    draw     = ImageDraw.Draw(img_copy)
    w, h     = img_copy.size

    # colour palette per severity
    palette = {
        "yellow": ((140, 120,   0), (  0,   0,   0), ( 20,  20,  20)),
        "orange": ((150,  55,   0), (255, 255, 255), (255, 210, 160)),
        "red":    ((120,   8,   8), (255, 210,  50), (255, 255, 255)),
    }
    fill, txt_col, cap_col = palette.get(color, palette["red"])

    draw.rectangle([(0, h - 52), (w, h)], fill=fill)
    draw.text((6, h - 50), alert_label,     fill=txt_col)
    draw.text((6, h - 30), caption[:90],    fill=cap_col)

    # Build output paths 
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    fnum      = result["frame_idx"]
    base_name = f"{ts}_frame{fnum:05d}_{result['time_sec']:.2f}s"

    # persistent output folder — grouped by alert type
    alert_dir = os.path.join(
        OUTPUT_ROOT,
        _sanitise_folder_name(alert_label),
    )
    os.makedirs(alert_dir, exist_ok=True)

    # session temp folder — for web delivery
    session_anom_dir = os.path.join(session_dir, "anomalies")
    os.makedirs(session_anom_dir, exist_ok=True)

    jpg_path   = os.path.join(alert_dir,         f"{base_name}.jpg")
    txt_path   = os.path.join(alert_dir,         f"{base_name}.txt")
    temp_path  = os.path.join(session_anom_dir,  f"{base_name}.jpg")

    img_copy.save(jpg_path,  quality=JPEG_QUALITY)
    img_copy.save(temp_path, quality=JPEG_QUALITY)

    # Write text log alongside the screenshot 
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(
            f"ALERT     : {alert_label}\n"
            f"SEVERITY  : {severity}\n"
            f"FRAME     : {fnum}\n"
            f"TIMESTAMP : {result['time_sec']:.2f}s  "
            f"({int(result['time_sec']//60):02d}:{int(result['time_sec']%60):02d})\n"
            f"SAVED AT  : {datetime.now().isoformat()}\n"
            f"CAPTION   : {caption}\n"
        )

    # Encode for web transmission
    buf = BytesIO()
    img_copy.save(buf, format="JPEG", quality=JPEG_QUALITY)
    b64_img = base64.b64encode(buf.getvalue()).decode()

    return b64_img, jpg_path

#  VISUALISATION HELPERS

def create_frame_grid(
    frames:        list[dict],
    frame_results: list[dict],
    captions:      list[str],
    session_dir:   str,
) -> str:
    """
    Render a contact-sheet style grid of all frames with status overlays.
    Returns the path to the saved JPEG.
    """
    N_COLS = 5
    TW, TH, LH = 220, 135, 85    # tile width, tile height, label height

    n_rows = math.ceil(len(frames) / N_COLS)
    grid   = Image.new("RGB", (N_COLS * TW, n_rows * (TH + LH)), (10, 12, 20))
    draw   = ImageDraw.Draw(grid)

    for pos, fr in enumerate(frames):
        res   = frame_results[pos]
        cap   = captions[pos] if pos < len(captions) else ""
        col   = pos % N_COLS
        row   = pos // N_COLS
        x, y  = col * TW, row * (TH + LH)

        # paste thumbnail
        grid.paste(fr["image"].resize((TW, TH), Image.LANCZOS), (x, y))

        sev   = res.get("severity", "NORMAL")
        alert = res.get("alert", "")
        anom  = res["is_anomaly"]

        # status strip colour coding
        if alert == "VEHICLE BLOCKING CROSSWALK":
            fc, oc, lc, label = (100, 40, 0), (220, 90, 0),  (255, 130, 30), "VEH ON XWALK"
        elif sev == "YELLOW":
            fc, oc, lc, label = ( 80, 70, 0), (200, 180, 0), (240, 215,  0), "PED XWALK"
        elif anom:
            fc, oc, lc, label = (100,  8, 8), (200,  25, 25),(255,  70, 70), "ANOMALY"
        else:
            fc, oc, lc, label = (  8, 50,18), ( 20, 140, 50),( 60, 240, 90), "Normal"

        draw.rectangle([(x, y + TH), (x + TW - 1, y + TH + LH - 1)], fill=fc)
        draw.text((x + 4, y + TH +  3), label,                              fill=lc)
        draw.text((x + 4, y + TH + 18), f"t={res['time_sec']:.1f}s  Δ={res['difference']:.1f}", fill=(180, 180, 180))
        draw.text((x + 4, y + TH + 34), alert[:30],                         fill=(240, 190, 50) if (anom or sev == "YELLOW") else (130, 130, 180))
        draw.text((x + 4, y + TH + 52), (cap[:50] + "…") if len(cap) > 50 else cap, fill=(160, 160, 230))
        draw.rectangle([(x, y), (x + TW - 1, y + TH + LH - 1)], outline=oc, width=3)

    path = os.path.join(session_dir, "grid.jpg")
    grid.save(path, quality=JPEG_QUALITY)
    return path


def create_anomaly_timeline(
    frame_results: list[dict],
    thr_d:         float,
    session_dir:   str,
) -> str:
    """
    Plot frame-difference over time with anomaly markers.
    Returns the path to the saved PNG.
    """
    times = [r["time_sec"]   for r in frame_results]
    diffs = [r["difference"] for r in frame_results]

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("#080c14")
    ax.set_facecolor("#0d1220")

    ax.plot(times, diffs, color="#38bdf8", lw=1.6, zorder=2)
    ax.fill_between(times, diffs, alpha=0.12, color="#38bdf8")
    ax.axhline(thr_d, color="#fbbf24", linestyle="--", lw=1.8, zorder=4, label="Threshold")

    for r in frame_results:
        sev = r.get("severity", "NORMAL")
        if r["is_anomaly"]:
            ax.scatter(r["time_sec"], r["difference"], color="#ef4444", s=65, zorder=5)
            ax.annotate(
                r.get("alert", "")[:18],
                (r["time_sec"], r["difference"]),
                textcoords="offset points",
                xytext=(4, 5),
                fontsize=5.5,
                color="#fca5a5",
            )
        elif sev == "YELLOW":
            ax.scatter(r["time_sec"], r["difference"], color="#fbbf24", s=55, zorder=5, marker="^")

    ax.set_title("Anomaly Score — Frame Difference Over Time",
                 color="#e2e8f0", fontsize=12, pad=10, fontfamily="monospace")
    ax.set_xlabel("Time (s)", color="#94a3b8")
    ax.set_ylabel("Frame Δ",  color="#94a3b8")
    ax.tick_params(colors="#64748b")
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e293b")
    ax.grid(axis="y", color="#1e293b", linestyle="--", lw=0.8)

    handles = [
        mpatches.Patch(color="#38bdf8", label="Normal"),
        mpatches.Patch(color="#ef4444", label="Anomaly"),
        mpatches.Patch(color="#fbbf24", label="Pedestrian crosswalk"),
        plt.Line2D([0], [0], color="#fbbf24", linestyle="--", label="Threshold"),
    ]
    ax.legend(handles=handles, facecolor="#0d1220", labelcolor="#cbd5e1",
              edgecolor="#1e293b", fontsize=8)

    plt.tight_layout()
    path = os.path.join(session_dir, "timeline.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return path

#MAIN STREAMING PIPELINE
# alert types that should trigger a sequential (state-machine) alert event
# in addition to the per-frame event.
_SEQUENTIAL_ALERT_TYPES: list[tuple[str, str]] = [
    ("traffic accident",          "TRAFFIC ACCIDENT"),
    ("stop line violation",       "STOP LINE VIOLATION"),
    ("red light violation",       "RED LIGHT VIOLATION"),
    ("ambulance stuck",           "AMBULANCE STUCK IN TRAFFIC"),
    ("people stealing",           "THEFT / STEALING"),
    ("suspicious activity",       "SUSPICIOUS ACTIVITY"),
    ("vehicle blocking crosswalk","VEHICLE BLOCKING CROSSWALK"),
]


def process_video_streaming(
    video_path: str,
    session_id: str,
    sio:        SocketIO,
) -> None:
    """
    Background thread: extract → score → label → analyse → emit.

    All SocketIO events are emitted to the *session_id* room so multiple
    concurrent sessions never cross-contaminate.
    """
    session_dir = os.path.join(
        app.config["UPLOAD_FOLDER"],
        f"multianomaly_{session_id}",
    )
    os.makedirs(session_dir, exist_ok=True)

    try:
        # 1. Extract frames
        frames = extract_frames(video_path, session_dir)
        if not frames:
            sio.emit("error", {"message": "No frames could be extracted from video."}, room=session_id)
            return

        # 2. Score & label anomalies based on frame differences
        diffs, ssims, thr_d, thr_s = compute_anomaly_scores(frames)
        frame_results              = label_frames(frames, diffs, ssims, thr_d, thr_s)
        per_frame_captions: list[str] = []

        # 3. Analyse each frame and stream results 
        for i, res in enumerate(frame_results):
            img      = frames[i]["image"]
            is_anom  = res["is_anomaly"]

            caption, alert_label, severity, vqa = analyse_frame(img, is_anom)
            per_frame_captions.append(caption)

            # VQA evidence can promote a frame to anomaly status
            if (is_yes(vqa.get("accident", "no"))
                    or is_yes(vqa.get("fire", "no"))
                    or alert_label == "VEHICLE BLOCKING CROSSWALK"):
                is_anom = True
                frame_results[i]["is_anomaly"] = True

            # convenience flags used by the frontend
            is_ped_crosswalk     = (severity == "YELLOW" or alert_label == "PEDESTRIAN ON CROSSWALK")
            is_vehicle_crosswalk = (alert_label == "VEHICLE BLOCKING CROSSWALK")

            # sequential alert detection
            seq_alert = seq_type = None
            for typ, label in _SEQUENTIAL_ALERT_TYPES:
                if typ in caption.lower() or typ in alert_label.lower():
                    seq_alert, seq_type = label, typ
                    break

            # save screenshot to disk & encode for web
            screenshot = None

            if is_anom or is_ped_crosswalk:
                if is_vehicle_crosswalk:
                    color = "orange"
                elif is_ped_crosswalk:
                    color = "yellow"
                else:
                    color = "red"

                screenshot, saved_path = save_alert_screenshot(
                    frames[i], res, caption, alert_label, severity, session_dir, color
                )

            # build frame payload
            timestamp_str = f"{int(res['time_sec']//60):02d}:{int(res['time_sec']%60):02d}"
            frame_payload = {
                "frame_idx":           res["frame_idx"],
                "time_sec":            res["time_sec"],
                "timestamp":           timestamp_str,
                "caption":             caption,
                "is_anomaly":          is_anom,
                "alert":               alert_label,
                "severity":            severity,
                "difference":          res["difference"],
                "is_ped_crosswalk":    is_ped_crosswalk,
                "is_vehicle_crosswalk": is_vehicle_crosswalk,
                "sequential_type":     seq_type,
                "sequential_alert":    seq_alert,
            }
            if screenshot:
                frame_payload["screenshot"] = screenshot

            # emit specialised events
            if is_ped_crosswalk:
                sio.emit("yellow_anomaly", {
                    "type":       "PEDESTRIAN ON CROSSWALK",
                    "timestamp":  timestamp_str,
                    "caption":    caption,
                    "severity":   "YELLOW",
                    "frame_idx":  res["frame_idx"],
                    "screenshot": screenshot or "",
                }, room=session_id)

            if is_vehicle_crosswalk:
                sio.emit("vehicle_crosswalk", {
                    "type":       "VEHICLE BLOCKING CROSSWALK",
                    "timestamp":  timestamp_str,
                    "caption":    caption,
                    "severity":   "CRITICAL",
                    "frame_idx":  res["frame_idx"],
                    "screenshot": screenshot or "",
                }, room=session_id)

            if seq_alert and is_anom:
                sio.emit("sequential_alert", {
                    "type":      seq_alert,
                    "timestamp": timestamp_str,
                    "caption":   caption,
                    "severity":  severity,
                }, room=session_id)

            # emit the main frame event (always)
            sio.emit("frame", frame_payload, room=session_id)

            # persist enriched result for visualisation step
            res.update({
                "caption":   caption,
                "alert":     alert_label,
                "severity":  severity,
                "vqa":       vqa,
            })

        # 4. Generate summary visualisations 
        grid_path     = create_frame_grid(frames, frame_results, per_frame_captions, session_dir)
        timeline_path = create_anomaly_timeline(frame_results, thr_d, session_dir)

        with open(grid_path,     "rb") as f:
            grid_b64     = base64.b64encode(f.read()).decode()
        with open(timeline_path, "rb") as f:
            timeline_b64 = base64.b64encode(f.read()).decode()

        sio.emit("done", {
            "total_frames":     len(frame_results),
            "anomalous_frames": sum(1 for r in frame_results if r["is_anomaly"]),
            "threshold":        thr_d,
            "grid":             grid_b64,
            "timeline":         timeline_b64,
        }, room=session_id)

    except Exception as exc:
        sio.emit("error", {"message": str(exc)}, room=session_id)
        raise

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
        active_sessions.pop(session_id, None)

#  ROUTES
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided."}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    filename    = secure_filename(file.filename)
    session_id  = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + str(abs(hash(file.filename)))[:8]
    )
    session_dir = os.path.join(app.config["UPLOAD_FOLDER"], f"multianomaly_{session_id}")
    os.makedirs(session_dir, exist_ok=True)

    video_path = os.path.join(session_dir, filename)
    file.save(video_path)

    thread = threading.Thread(
        target=process_video_streaming,
        args=(video_path, session_id, socketio),
        daemon=True,
    )
    thread.start()
    active_sessions[session_id] = thread

    return jsonify({"session_id": session_id, "message": "Processing started."})

#  SOCKETIO EVENT HANDLERS
@socketio.on("connect")
def handle_connect():
    print("[SocketIO] Client connected.")


@socketio.on("disconnect")
def handle_disconnect():
    print("[SocketIO] Client disconnected.")


@socketio.on("join")
def handle_join(data: dict):
    sid = data.get("session_id")
    if sid:
        join_room(sid)
        print(f"[SocketIO] Client joined room {sid}.")

if __name__ == "__main__":
    load_blip_model()
    socketio.run(app, debug=False, host="0.0.0.0", port=5000)