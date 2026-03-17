"""
MultiAnomalyDetector  -  TERMINAL-ONLY VERSION

Detects 6 uncorrelated anomaly categories with timestamped alerts
-------------------------------------------------------------------
1. TRAFFIC ACCIDENT          4. AMBULANCE STUCK IN TRAFFIC
2. STOP LINE VIOLATION       5. THEFT / STEALING
3. RED LIGHT VIOLATION       6. SUSPICIOUS ACTIVITY

Also detects:
   - Pedestrian crossing (YELLOW alert with warning)

Detection layers
-------------------------------------------------------------------
Layer 1  Motion scoring  (L1 pixel diff  +  SSIM)
Layer 2  BLIP VQA        (yes/no questions per category)
Layer 3  Caption keywords (400 + keyword rule table)

USAGE
  python anomaly_detector_terminal.py                         (menu)
  python anomaly_detector_terminal.py --video clip.mp4
  python anomaly_detector_terminal.py --video clip.mp4 --chunk 1.5
  python anomaly_detector_terminal.py --folder ./videos --output ./out

DEPENDENCIES
  pip install opencv-python numpy Pillow scikit-image
  pip install torch torchvision transformers matplotlib
"""

from __future__ import annotations

# stdlib
import argparse
import math
import os
import sys
import textwrap
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

# third-party (graceful import errors)
try:
    import cv2
except ImportError:
    sys.exit("MISSING  ->  pip install opencv-python")

try:
    import numpy as np
except ImportError:
    sys.exit("MISSING  ->  pip install numpy")

try:
    from PIL import Image, ImageDraw
except ImportError:
    sys.exit("MISSING  ->  pip install Pillow")

try:
    from skimage.metrics import structural_similarity as ssim_fn
except ImportError:
    sys.exit("MISSING  ->  pip install scikit-image")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False
    print("WARNING  matplotlib not found - timeline chart will be skipped.\n")

try:
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor
    BLIP_OK = True
except ImportError:
    BLIP_OK = False
    print("WARNING  torch/transformers not found - VQA/captions disabled.\n")


# SECTION 1  -  GLOBAL CONFIGURATION

VIDEOS_FOLDER : str   = "./videosFolder"          # default folder for --folder flag
OUTPUT_ROOT   : str   = "./output_anomalies"
CHUNK_SECONDS : float = 2.0                 # one frame sampled every N seconds
MAX_FRAMES    : int   = 600                 # hard cap per video

BLIP_CHECKPOINT : str = "Salesforce/blip-image-captioning-base"
BLIP_INPUT_SIZE : int = 384
CAPTION_TOKENS  : int = 80
VQA_TOKENS      : int = 8
NUM_BEAMS       : int = 3

DIFF_SIZE        : tuple[int, int] = (96, 96)
SIGMA_FACTOR     : float           = 0.60
DIFF_HARD_FLOOR  : float           = 12.0

STALL_DIFF_MAX   : float = 7.0
STALL_MIN_FRAMES : int   = 4

JPEG_QUALITY : int = 90

VIDEO_EXTENSIONS : tuple[str, ...] = (
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"
)

# SECTION 2  -  VQA QUESTION BANKS  (one bank per anomaly category)

# Category 1: Traffic Accident
VQA_ACCIDENT = [
    "is there a car crash, vehicle collision, or traffic accident in this image?",
    "are two or more vehicles making contact, colliding, or involved in an accident?",
    "has a vehicle been struck, rammed, hit, or impacted by another vehicle?",
    "is there evidence of a traffic accident such as damage, debris, or displaced vehicles?",
    "are any vehicles overturned, flipped, skidding out of control, or off the road?",
]

# Category 2: Stop Line Violation
VQA_STOP_LINE = [
    "is a vehicle past or crossing the white stop line at a red signal?",
    "has a vehicle gone over the stop line at an intersection on red?",
    "is a car or truck violating the stop line at a traffic signal?",
]

# Category 3: Red Light Violation
VQA_RED_LIGHT = [
    "is a vehicle crossing an intersection while the traffic light is red?",
    "is a car running a red light or ignoring a red traffic signal?",
    "is a vehicle passing through an intersection against a red light?",
    "is a police car stopped at a red light?",
]

# Category 4: Ambulance Stuck
VQA_AMBULANCE = [
    "is an ambulance, fire truck, or police vehicle blocked and unable to move?",
    "is an emergency vehicle stuck in traffic or surrounded by other vehicles?",
    "is there an ambulance or emergency service vehicle that cannot pass through traffic?",
]

# Category 5: Theft / Stealing
VQA_THEFT = [
    "is someone stealing, shoplifting, or taking items without paying?",
    "is a person concealing, hiding, or pocketing merchandise in a store?",
    "is someone grabbing goods and running without permission?",
    "is there a robbery, bag snatching, or theft happening in this scene?",
]

# Category 6: Suspicious Activity
VQA_SUSPICIOUS = [
    "is there a person acting suspiciously, loitering, or behaving strangely?",
    "is someone behaving in an unusual, erratic, or threatening manner?",
    "is a person lurking, trespassing, or casing a building or vehicle?",
    "does any person appear to be planning or conducting criminal activity?",
]

# Supplementary (secondary, gated on motion candidates)
VQA_WRONG_WAY = [
    "is a vehicle driving on the wrong side of the road toward oncoming traffic?",
    "is any vehicle facing the opposite direction from normal traffic flow?",
]

VQA_FALL = [
    "has a person collapsed, fallen, or been knocked to the ground?",
    "is there a person lying motionless on the road or floor?",
]

VQA_FIRE = [
    "is there visible fire or flames in this image?",
    "is there heavy smoke or a burning vehicle or building visible?",
]

VQA_STALL = [
    "is there a bus, truck, van, or large vehicle stopped and blocking the road?",
    "is a vehicle parked or stopped illegally in the middle of a lane?",
    "is any vehicle broken down or stalled and obstructing traffic flow?",
]

# Pedestrian crossing (improved with multiple questions)
VQA_PEDESTRIAN = [
    "is a pedestrian using or crossing a marked crosswalk or zebra crossing?",
    "is a person crossing the street?",
    "are there pedestrians on the road?",
    "is someone walking across the road?",
    "is a person crossing at an intersection?",
]

VQA_SEVERITY = "how severe is this incident? answer only: minor, moderate, or severe"

_YES_TOKENS = ("yes", "yeah", "yep", "true", "correct", "certainly",
               "definitely", "absolutely", "affirmative", "indeed")
_SEV_TOKENS = frozenset({"minor", "moderate", "severe"})


def is_yes(answer: str) -> bool:
    t = (answer or "").strip().lower()
    return any(t.startswith(w) for w in _YES_TOKENS)


def any_yes(answers: list[str]) -> bool:
    return any(is_yes(a) for a in answers)


def parse_sev(raw: str) -> str:
    if not raw:
        return "n/a"
    w = raw.strip().lower().split()[0]
    return w if w in _SEV_TOKENS else "n/a"


# SECTION 3  -  KEYWORD ALERT TABLE  (400 + terms, 6 primary categories)
#
# Format:  ( [keyword_list], "LABEL", "SEVERITY" )
# First matching rule wins - order is intentional (most critical first).

ALERT_RULES: list[tuple[list[str], str, str]] = [

    # 1.  TRAFFIC ACCIDENT
    ([
        "crash","crashed","crashing","crashes",
        "collision","collide","collided","colliding","collisions",
        "accident","traffic accident","car accident","road accident",
        "vehicle accident","fatal accident","serious accident","major accident",
        "wreck","wrecked","wreckage","totaled","totalled",
        "smash","smashed","smashing","smashes",
        "crumpled","crushed","mangled","destroyed vehicle","damaged vehicle",
        "impact","impacted","high-speed impact","violent impact",
        "overturned","overturning","flipped","flipped over","upside down",
        "rolled over","roll over","rolling over","on its side",
        "spun out","spinning out","skidded","skidding","skids","skid marks",
        "hydroplaned","lost control","out of control","slid into",
        "pile-up","pileup","pile up","chain reaction",
        "multi-vehicle","multi-car","multiple vehicle",
        "rear-end","rear ended","rear-ended","rear collision","tailgated",
        "head-on","head on","head-on collision",
        "side collision","broadside","sideswipe","sideswiped",
        "t-bone","t-boned",
        "rammed","ramming","struck by","struck a","struck the",
        "hit by a car","hit by vehicle","hit by truck",
        "ran into","drove into","slammed into","knocked into",
        "plowed into","ploughed into",
        "run over","ran over","running over","knocked down by car",
        "pedestrian struck","pedestrian hit","person struck",
        "person hit by car","cyclist hit","motorcycle collision",
        "motorcycle crash","bike crash","bicycle accident",
        "airbag deployed","airbags deployed",
        "car off road","vehicle off road","hit a barrier","hit a wall",
        "hit a pole","hit a tree","into a ditch","into a median",
        "accident scene","crash scene","flipped vehicle","crashed vehicle",
    ], "TRAFFIC ACCIDENT", "CRITICAL"),

    # 2.  STOP LINE VIOLATION
    ([
        "stop line violation","stop-line violation",
        "crossed stop line","crossing stop line",
        "ran the stop line","ran stop line",
        "passed the stop line","crossed the stop line",
        "over the stop line","beyond the stop line","past the stop line",
        "crossed the white line","over the white line",
        "stop line breached","ignored stop line",
        "failed to stop at line","did not stop at line",
        "violation of stop line",
    ], "STOP LINE VIOLATION", "CRITICAL"),

    # 3.  RED LIGHT VIOLATION
    ([
        "red light violation","red-light violation",
        "ran red light","ran a red light","ran the red light",
        "crossed on red","crossing on red",
        "running a red","running the red","running red light",
        "drove through red","drove through a red",
        "went through red","passed a red light",
        "ran through red","against a red light",
        "ignoring red light","ignored red light",
        "traffic signal violation","signal violation",
        "disregarded red light","beat the red light",
        "through a red signal","disobeyed signal",
        "ran a stop light","blew a red",
    ], "RED LIGHT VIOLATION", "CRITICAL"),

    # 4.  AMBULANCE STUCK IN TRAFFIC
    ([
        "ambulance stuck","ambulance blocked",
        "ambulance is stuck","ambulance is blocked",
        "ambulance cannot move","ambulance unable to move",
        "ambulance not moving","ambulance delayed",
        "ambulance in traffic","ambulance trapped",
        "ambulance surrounded","ambulance gridlocked", "emergency vehicle stuck", "a bus crashed into a pedestrian bridge", "a car crashed into the intersection of fifth and madison streets",
        "emergency vehicle blocked",
        "emergency vehicle cannot pass","emergency vehicle unable",
        "fire truck blocked","fire truck stuck",
        "police car blocked","police vehicle stuck",
        "paramedic vehicle blocked","ambulance cannot pass",
        "blocked ambulance","emergency services delayed",
        "emergency response blocked",
        "not yielding to ambulance","failure to yield to ambulance",
        "blocking ambulance","obstructing ambulance",
        "vehicles surrounding ambulance","cars blocking ambulance",
    ], "AMBULANCE STUCK IN TRAFFIC", "CRITICAL"),

    # 5.  THEFT / STEALING
    ([
        "steal","stole","stealing","steals","stolen",
        "theft","thief","thieves",
        "shoplifting","shoplifter","shoplifts","shoplifted",
        "robbery","robbing","robbed","rob","robs",
        "burglary","burglar","burglarize",
        "mugging","mugged","mugger",
        "pickpocket","pickpocketing","pickpocketed",
        "snatch","snatching","snatched","bag snatch","purse snatch","phone snatch",
        "grabs","grabbed","grabbing",
        "takes item","taking item","took item",
        "takes merchandise","taking merchandise",
        "concealing item","concealing merchandise","hides item","hiding item",
        "pockets item","pocketing item","stuffs into bag","stuffs item",
        "puts item in bag","slips item",
        "without paying","without permission","without scanning","unpaid item",
        "flees store","runs from store","leaves without paying",
        "store theft","retail theft","shoplifting incident",
        "item stolen","goods stolen","merchandise stolen",
        "people stealing","person stealing",
    ], "THEFT / STEALING", "CRITICAL"),

    # 6.  SUSPICIOUS ACTIVITY
    ([
        "suspicious activity","suspicious person",
        "suspicious individual","suspicious behavior","suspicious behaviour",
        "suspicious movement","acting suspiciously","behaving suspiciously",
        "behaving strangely","behaving erratically",
        "unusual activity","unusual behaviour","unusual behavior",
        "unusual movement","strange activity","strange behavior",
        "strange behaviour",
        "loitering","loiter","loiters",
        "trespassing","trespass","trespasses",
        "lurking","lurk","lurks",
        "casing the","casing a building","casing a store",
        "watching nervously","looking around nervously","looking suspicious",
        "erratic behavior","erratic behaviour","erratic movement",
        "irrational behavior","threatening behavior","threatening behaviour",
        "intimidating","menacing",
        "vandalism","vandalizing","spray painting",
        "prowling","prowl","prowls",
    ], "SUSPICIOUS ACTIVITY", "HIGH"),

    # Secondary categories (kept for caption context, not primary detections)
    (["wrong way","wrong-way vehicle","wrong side of the road",
      "driving against traffic","oncoming lane","against traffic flow",
      "wrong direction","against the flow"],
     "WRONG-WAY VEHICLE", "CRITICAL"),

    (["fire","on fire","in flames","smoke","thick smoke","burning",
      "ablaze","flame","flames","engulfed in fire","vehicle on fire"],
     "FIRE / SMOKE", "CRITICAL"),

    (["person down","fallen","collapsed","lying on the road",
      "lying on the ground","knocked down","unconscious","motionless"],
     "PERSON DOWN / FALL", "HIGH"),

    (["fight","fighting","brawl","violence","assault","punching",
      "kicking","beating","hitting","altercation","physical fight"],
     "VIOLENCE / ASSAULT", "HIGH"),

    (["vehicle on crosswalk","car on crosswalk","bus on crosswalk",
      "blocking the crosswalk","blocking pedestrian crossing",
      "stopped on crosswalk","bus stopped at crosswalk"],
     "VEHICLE BLOCKING CROSSWALK", "CRITICAL"),

    (["pedestrian crossing","person crossing","pedestrian on crosswalk",
      "crossing the street","zebra crossing"],
     "PEDESTRIAN ON CROSSWALK", "YELLOW"),

    (["traffic","road","street","car","vehicle","truck","bus","intersection",
      "driving","lane","signal","parking","store","shop"],
     "TRAFFIC SCENE", "INFO"),
]


def classify_caption(caption: str) -> tuple[str, str]:
    """Returns (alert_label, severity) for the first keyword match."""
    lower = caption.lower()
    for keywords, label, severity in ALERT_RULES:
        if any(kw in lower for kw in keywords):
            return label, severity
    return "NO ALERT", "NORMAL"


# SECTION 4  -  BLIP MODEL  (loaded once, reused for all frames)

_processor : Optional["BlipProcessor"]                 = None
_model     : Optional["BlipForConditionalGeneration"]  = None
_device    : Optional["torch.device"]                  = None


def load_model() -> None:
    global _processor, _model, _device
    if not BLIP_OK:
        _print_warn("BLIP unavailable - install torch + transformers for VQA.")
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _print_section("LOADING BLIP MODEL")
    _print_kv("Checkpoint", BLIP_CHECKPOINT)
    _print_kv("Device",     str(_device))
    t0 = time.time()
    _processor = BlipProcessor.from_pretrained(BLIP_CHECKPOINT, use_fast=True)
    _model     = BlipForConditionalGeneration.from_pretrained(
        BLIP_CHECKPOINT, ignore_mismatched_sizes=True
    )
    _model.to(_device).eval()
    _print_kv("Load time", f"{time.time() - t0:.1f}s")
    print()


def _resize_blip(img: Image.Image) -> Image.Image:
    w, h  = img.size
    scale = BLIP_INPUT_SIZE / max(w, h)
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def _clean_caption(text: str) -> str:
    """Remove BLIP word-repetition artefacts (e.g. 'signal signal signal')."""
    words, out, run, prev = text.split(), [], 0, ""
    for w in words:
        if w == prev:
            run += 1
            if run <= 1:
                out.append(w)
        else:
            run, prev = 0, w
            out.append(w)
    return " ".join(out)


def blip_caption(img: Image.Image, prompt: str | None = None) -> str:
    if not _processor or not _model:
        return "(BLIP unavailable)"
    small = _resize_blip(img)
    with torch.no_grad():
        inputs = (
            _processor(images=small, text=prompt, return_tensors="pt").to(_device)
            if prompt else
            _processor(images=small, return_tensors="pt").to(_device)
        )
        out = _model.generate(
            **inputs,
            max_new_tokens=CAPTION_TOKENS,
            num_beams=NUM_BEAMS,
            length_penalty=1.2,
        )
    text = _processor.decode(out[0], skip_special_tokens=True)
    if prompt and text.lower().startswith(prompt.lower()):
        text = text[len(prompt):].strip()
    return _clean_caption(text)


def blip_vqa_batch(img: Image.Image, questions: list[str]) -> list[str]:
    """Run multiple VQA questions on the same image in one Python call."""
    if not _processor or not _model:
        return ["n/a"] * len(questions)
    small   = _resize_blip(img)
    answers = []
    for q in questions:
        try:
            with torch.no_grad():
                inputs = _processor(images=small, text=q,
                                    return_tensors="pt").to(_device)
                out = _model.generate(**inputs, max_new_tokens=VQA_TOKENS)
            answers.append(
                _processor.decode(out[0], skip_special_tokens=True).strip().lower()
            )
        except Exception:
            answers.append("n/a")
    return answers


def blip_vqa_single(img: Image.Image, question: str) -> str:
    return blip_vqa_batch(img, [question])[0]


# SECTION 5  -  FRAME EXTRACTION  +  MOTION SCORING  (Layer 1)

def extract_frames(video_path: str) -> list[dict]:
    """Sample one frame every CHUNK_SECONDS using MSEC seek."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _print_error(f"Cannot open: {video_path}")
        return []
    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_s  = n_frames / fps if fps > 0 and n_frames > 0 else float("inf")
    frames   : list[dict] = []
    idx, sec = 0, 0.0
    while len(frames) < MAX_FRAMES:
        if sec > total_s:
            break
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, bgr = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append({
            "idx":       idx,
            "time_sec":  sec,
            "timestamp": f"{int(sec // 60):02d}:{int(sec % 60):02d}",
            "image":     Image.fromarray(rgb),
        })
        idx += 1
        sec += CHUNK_SECONDS
    cap.release()
    return frames


def compute_motion(frames: list[dict]) -> tuple[list[float], list[float]]:
    """Return (diffs, ssims) - frame 0 gets diff=0.0, ssim=1.0."""
    diffs = [0.0]; ssims = [1.0]
    prev_rgb  = np.array(frames[0]["image"].resize(DIFF_SIZE)).astype(np.float32)
    prev_gray = cv2.cvtColor(np.uint8(prev_rgb), cv2.COLOR_RGB2GRAY)
    for i in range(1, len(frames)):
        curr_rgb  = np.array(frames[i]["image"].resize(DIFF_SIZE)).astype(np.float32)
        curr_gray = cv2.cvtColor(np.uint8(curr_rgb), cv2.COLOR_RGB2GRAY)
        diffs.append(float(np.mean(np.abs(curr_rgb - prev_rgb))))
        ssims.append(float(ssim_fn(curr_gray, prev_gray, data_range=255)))
        prev_rgb, prev_gray = curr_rgb, curr_gray
    return diffs, ssims


def flag_motion_candidates(diffs: list[float], ssims: list[float]) -> list[bool]:
    """Adaptive hysteresis threshold - marks frames with significant motion."""
    arr_d  = np.array(diffs)
    arr_s  = np.array(ssims)
    thr_d  = float(arr_d.mean() + SIGMA_FACTOR * arr_d.std())
    thr_s  = float(arr_s.mean() - SIGMA_FACTOR * arr_s.std())
    low_d  = thr_d * 0.5
    high_s = min(thr_s * 1.5, 0.98)
    flags, active = [], False
    for d, s in zip(diffs, ssims):
        if d > thr_d or d > DIFF_HARD_FLOOR or s < thr_s:
            active = True
        elif d < low_d and s > high_s:
            active = False
        flags.append(active)
    return flags


def flag_stall_candidates(diffs: list[float]) -> list[bool]:
    """Mark runs of STALL_MIN_FRAMES+ consecutive low-diff frames."""
    n     = len(diffs)
    stall = [False] * n
    run   = 0; start = 0
    for i in range(n):
        if diffs[i] < STALL_DIFF_MAX:
            if run == 0:
                start = i
            run += 1
        else:
            if run >= STALL_MIN_FRAMES:
                for j in range(start, i):
                    stall[j] = True
            run = 0
    if run >= STALL_MIN_FRAMES:
        for j in range(start, n):
            stall[j] = True
    return stall


# SECTION 6  -  PER-FRAME ANALYSIS  (Layers 2 + 3 combined)

PROMPT_NORMAL     = "a traffic or surveillance camera photo showing"
PROMPT_ANOMALY    = "a surveillance camera clearly showing an incident where"
PROMPT_ACCIDENT   = "a traffic camera showing a road collision or accident where"
PROMPT_THEFT      = "a store surveillance camera showing a person stealing where"
PROMPT_SUSPICIOUS = "a surveillance camera showing a person behaving suspiciously"
PROMPT_AMBULANCE  = "a traffic camera showing an ambulance or emergency vehicle where"
PROMPT_RED_LIGHT  = "a traffic camera at an intersection showing a vehicle that"
PROMPT_STALL      = "a traffic camera showing a vehicle stopped or blocking the road"
PROMPT_PEDESTRIAN = "a traffic camera showing pedestrians crossing the street where"


def analyse_frame(
    img:      Image.Image,
    is_cand:  bool,
    is_stall: bool,
) -> tuple[str, str, str]:
    """
    Runs all three layers on a single frame.

    Layer 2A  (critical VQA - every frame, no motion gate):
        VQA_ACCIDENT, VQA_RED_LIGHT, VQA_STOP_LINE,
        VQA_AMBULANCE, VQA_THEFT, VQA_SUSPICIOUS

    Layer 2B  (secondary VQA - motion candidates only):
        VQA_STALL, VQA_WRONG_WAY, VQA_FIRE, VQA_FALL, VQA_PEDESTRIAN (improved)

    Layer 3  (caption + keyword matching):
        Prompt chosen from best-fitting Layer-2 signal.

    Fusion: VQA overrides caption; CRITICAL is never downgraded.

    Returns (caption, alert_label, severity).
    """

    # Layer 2A
    acc_ans  = blip_vqa_batch(img, VQA_ACCIDENT)
    red_ans  = blip_vqa_batch(img, VQA_RED_LIGHT)
    stop_ans = blip_vqa_batch(img, VQA_STOP_LINE)
    amb_ans  = blip_vqa_batch(img, VQA_AMBULANCE)
    tft_ans  = blip_vqa_batch(img, VQA_THEFT)
    sus_ans  = blip_vqa_batch(img, VQA_SUSPICIOUS)

    # Layer 2B  (motion / stall gated)
    if is_cand or is_stall:
        stl_ans  = blip_vqa_batch(img, VQA_STALL)      if is_stall else ["n/a"]
        ww_ans   = blip_vqa_batch(img, VQA_WRONG_WAY)
        fire_ans = blip_vqa_batch(img, VQA_FIRE)
        fall_ans = blip_vqa_batch(img, VQA_FALL)
        ped_ans  = blip_vqa_batch(img, VQA_PEDESTRIAN)   # now multiple questions
    else:
        stl_ans = ww_ans = fire_ans = fall_ans = ped_ans = ["n/a"]

    # Boolean flags
    accident_vqa  = any_yes(acc_ans)
    red_vqa       = any_yes(red_ans)
    stop_vqa      = any_yes(stop_ans)
    ambulance_vqa = any_yes(amb_ans)
    theft_vqa     = any_yes(tft_ans)
    suspicious_vqa= any_yes(sus_ans)
    stall_vqa     = any_yes(stl_ans)
    wrong_way_vqa = any_yes(ww_ans)
    fire_vqa      = any_yes(fire_ans)
    fall_vqa      = any_yes(fall_ans)
    ped_vqa       = any_yes(ped_ans)

    # Layer 3: pick most specific caption prompt
    if accident_vqa:
        prompt = PROMPT_ACCIDENT
    elif theft_vqa:
        prompt = PROMPT_THEFT
    elif suspicious_vqa:
        prompt = PROMPT_SUSPICIOUS
    elif ambulance_vqa:
        prompt = PROMPT_AMBULANCE
    elif red_vqa or stop_vqa:
        prompt = PROMPT_RED_LIGHT
    elif stall_vqa:
        prompt = PROMPT_STALL
    elif ped_vqa:
        prompt = PROMPT_PEDESTRIAN
    elif is_cand or wrong_way_vqa or fire_vqa or fall_vqa:
        prompt = PROMPT_ANOMALY
    else:
        prompt = PROMPT_NORMAL

    caption = blip_caption(img, prompt=prompt)
    alert_label, severity = classify_caption(caption)

    # Fusion: VQA overrides (highest severity wins)
    def _upgrade(lbl: str, sev: str) -> None:
        nonlocal alert_label, severity
        _SEV = {"NORMAL": 0, "INFO": 0, "YELLOW": 1,
                "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        if _SEV.get(sev, 0) > _SEV.get(severity, 0):
            alert_label, severity = lbl, sev

    if accident_vqa:   _upgrade("TRAFFIC ACCIDENT",           "CRITICAL")
    if red_vqa:        _upgrade("RED LIGHT VIOLATION",        "CRITICAL")
    if stop_vqa:       _upgrade("STOP LINE VIOLATION",        "CRITICAL")
    if ambulance_vqa:  _upgrade("AMBULANCE STUCK IN TRAFFIC", "CRITICAL")
    if theft_vqa:      _upgrade("THEFT / STEALING",           "CRITICAL")
    if wrong_way_vqa:  _upgrade("WRONG-WAY VEHICLE",          "CRITICAL")
    if fire_vqa:       _upgrade("FIRE / SMOKE",               "CRITICAL")
    if suspicious_vqa: _upgrade("SUSPICIOUS ACTIVITY",        "HIGH")
    if stall_vqa:      _upgrade("STALLED / BLOCKING VEHICLE", "HIGH")
    if fall_vqa:       _upgrade("PERSON DOWN / FALL",         "HIGH")
    if ped_vqa and severity in ("NORMAL", "INFO"):
        _upgrade("PEDESTRIAN ON CROSSWALK", "YELLOW")
        # Ensure a clear warning in caption
        caption = f"WARNING: Pedestrian crossing. {caption}"

    # Append severity qualifier
    any_critical = (accident_vqa or red_vqa or stop_vqa or ambulance_vqa
                    or theft_vqa or suspicious_vqa or wrong_way_vqa
                    or fire_vqa or fall_vqa)
    if any_critical or is_cand:
        sv = parse_sev(blip_vqa_single(img, VQA_SEVERITY))
        if sv != "n/a":
            caption = f"{caption}  [{sv}]"

    return caption, alert_label, severity


# SECTION 7  -  ANNOTATED SCREENSHOT SAVING

_BANNER_PALETTE = {
    "CRITICAL": ((110, 10, 10),  (255, 55, 55),   (225, 175, 175)),
    "HIGH":     ((90,  45,  0),  (255, 140, 30),  (215, 180, 140)),
    "YELLOW":   ((70,  60,  0),  (230, 200,  0),  (210, 195, 130)),
    "NORMAL":   ((14,  60, 22),  ( 50, 200, 70),  (160, 190, 160)),
    "INFO":     ((14,  60, 22),  ( 50, 200, 70),  (160, 190, 160)),
}


def save_screenshot(
    frame:       dict,
    caption:     str,
    alert_label: str,
    severity:    str,
    video_name:  str,
) -> str:
    """Draw banner, save annotated JPEG; return saved path."""
    img  = frame["image"].copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)

    pal          = _BANNER_PALETTE.get(severity, _BANNER_PALETTE["NORMAL"])
    fill, lbl_c, cap_c = pal
    banner_h     = max(60, h // 9)
    y0           = h - banner_h

    draw.rectangle([(0, y0), (w, h)], fill=fill)

    # Status pill
    pill_text = alert_label
    pill_w    = len(pill_text) * 7 + 20
    draw.rounded_rectangle([(4, y0 + 4), (pill_w, y0 + 26)],
                            radius=5, fill=lbl_c)
    draw.text((10, y0 + 8), pill_text, fill=fill)

    # Timestamp + severity
    ts_str = (f"  {frame['timestamp']}  (t={frame['time_sec']:.1f}s)"
              f"  [{severity}]  {datetime.now().strftime('%H:%M:%S')}")
    draw.text((pill_w + 8, y0 + 8), ts_str, fill=lbl_c)

    # Caption line
    cap_line = caption[:115] + ("..." if len(caption) > 115 else "")
    draw.text((6, y0 + banner_h // 2 + 6), cap_line, fill=cap_c)

    # Save
    safe_name  = _safe_name(video_name)
    ts_stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name  = f"{ts_stamp}_f{frame['idx']:04d}_{frame['time_sec']:.1f}s.jpg"
    sub_folder = os.path.join(OUTPUT_ROOT, safe_name,
                              _safe_name(alert_label))
    os.makedirs(sub_folder, exist_ok=True)
    path = os.path.join(sub_folder, file_name)
    img.save(path, quality=JPEG_QUALITY)
    return path


def _safe_name(name: str) -> str:
    import unicodedata
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    for ch in r'\/:*?"<>|':
        name = name.replace(ch, "_")
    return name.strip().replace(" ", "_").upper()


# SECTION 8  -  SUMMARY VISUALS  (contact sheet + timeline)

def create_contact_sheet(
    frames:  list[dict],
    results: list[dict],
    out_dir: str,
) -> str:
    N_COLS       = 5
    TW, TH, LH  = 220, 135, 80
    n_rows       = math.ceil(len(frames) / N_COLS)
    grid         = Image.new("RGB", (N_COLS * TW, n_rows * (TH + LH)), (10, 12, 20))
    draw         = ImageDraw.Draw(grid)

    for pos, fr in enumerate(frames):
        res   = results[pos]
        cap   = res.get("caption", "")
        col   = pos % N_COLS
        row   = pos // N_COLS
        x, y  = col * TW, row * (TH + LH)
        alert = res.get("alert", "")
        sev   = res.get("severity", "NORMAL")
        anom  = res.get("is_anomaly", False)

        grid.paste(fr["image"].resize((TW, TH), Image.LANCZOS), (x, y))

        if sev == "CRITICAL":
            fc, oc, lc = (100, 8, 8),  (200, 25, 25), (255, 70, 70)
            lbl = "CRITICAL"
        elif sev == "HIGH":
            fc, oc, lc = (90, 50, 0),  (200, 100, 0), (255, 140, 30)
            lbl = "HIGH"
        elif sev == "YELLOW":
            fc, oc, lc = (80, 70, 0),  (200, 180, 0), (240, 215, 0)
            lbl = "YELLOW"
        else:
            fc, oc, lc = (8, 50, 18),  (20, 140, 50), (60, 240, 90)
            lbl = "NORMAL"

        draw.rectangle([(x, y + TH), (x + TW - 1, y + TH + LH - 1)], fill=fc)
        draw.text((x + 4, y + TH + 3),  lbl,    fill=lc)
        draw.text((x + 4, y + TH + 18),
                  f"t={res['time_sec']:.1f}s  Δ={res['diff']:.1f}",
                  fill=(180, 180, 180))
        draw.text((x + 4, y + TH + 34),
                  alert[:30],
                  fill=(240, 190, 50) if anom else (130, 130, 180))
        draw.text((x + 4, y + TH + 52),
                  (cap[:48] + "...") if len(cap) > 50 else cap,
                  fill=(160, 160, 230))
        draw.rectangle([(x, y), (x + TW - 1, y + TH + LH - 1)],
                       outline=oc, width=3)

    path = os.path.join(out_dir, "contact_sheet.jpg")
    grid.save(path, quality=JPEG_QUALITY)
    return path


def create_timeline(
    results: list[dict],
    thr_d:   float,
    out_dir: str,
) -> str:
    if not MATPLOTLIB_OK:
        return ""
    times = [r["time_sec"] for r in results]
    diffs = [r["diff"]     for r in results]

    fig, ax = plt.subplots(figsize=(16, 4))
    fig.patch.set_facecolor("#080c14")
    ax.set_facecolor("#0d1220")

    ax.plot(times, diffs, color="#38bdf8", lw=1.6, zorder=2)
    ax.fill_between(times, diffs, alpha=0.12, color="#38bdf8")
    ax.axhline(thr_d, color="#fbbf24", linestyle="--", lw=1.8, zorder=4,
               label="Threshold")

    for r in results:
        sev = r.get("severity", "NORMAL")
        if r.get("is_anomaly"):
            colour = {"CRITICAL": "#ef4444", "HIGH": "#fb923c"}.get(sev, "#fbbf24")
            ax.scatter(r["time_sec"], r["diff"], color=colour, s=70, zorder=5)
            label_txt = r.get("alert", "")[:20]
            ax.annotate(label_txt, (r["time_sec"], r["diff"]),
                        textcoords="offset points", xytext=(4, 5),
                        fontsize=5.2, color="#fca5a5")

    ax.set_title("Anomaly Score - Frame Difference Over Time",
                 color="#e2e8f0", fontsize=12, pad=10, fontfamily="monospace")
    ax.set_xlabel("Time (s)",   color="#94a3b8")
    ax.set_ylabel("Frame Δ",   color="#94a3b8")
    ax.tick_params(colors="#64748b")
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e293b")
    ax.grid(axis="y", color="#1e293b", linestyle="--", lw=0.8)

    handles = [
        mpatches.Patch(color="#38bdf8", label="Frame diff"),
        mpatches.Patch(color="#ef4444", label="CRITICAL"),
        mpatches.Patch(color="#fb923c", label="HIGH"),
        mpatches.Patch(color="#fbbf24", label="YELLOW"),
        plt.Line2D([0], [0], color="#fbbf24", linestyle="--", label="Threshold"),
    ]
    ax.legend(handles=handles, facecolor="#0d1220", labelcolor="#cbd5e1",
              edgecolor="#1e293b", fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "anomaly_timeline.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return path


# SECTION 9  -  PLAIN ASCII TERMINAL OUTPUT (no emojis, no box drawing)

W = 78  # print width

_C = {
    "reset":  "\033[0m",  "bold":   "\033[1m",  "dim":    "\033[2m",
    "red":    "\033[91m", "orange": "\033[93m",  "green":  "\033[92m",
    "cyan":   "\033[96m", "blue":   "\033[94m",
    "grey":   "\033[90m", "white":  "\033[97m",  "yellow": "\033[33m",
    "purple": "\033[95m", "teal":   "\033[36m",
}

_SEV_COLOR = {
    "CRITICAL": "red",
    "HIGH":     "orange",
    "YELLOW":   "yellow",
    "MEDIUM":   "teal",
    "INFO":     "grey",
    "NORMAL":   "green",
}

# No icons, just plain text
_CATEGORY_ICONS = {
    "TRAFFIC ACCIDENT":           "",
    "STOP LINE VIOLATION":        "",
    "RED LIGHT VIOLATION":        "",
    "AMBULANCE STUCK IN TRAFFIC": "",
    "THEFT / STEALING":           "",
    "SUSPICIOUS ACTIVITY":        "",
    "PEDESTRIAN ON CROSSWALK":    "",
}


def _c(text: str, *codes: str) -> str:
    return "".join(_C.get(k, "") for k in codes) + str(text) + _C["reset"]


def _rule(ch: str = "-", color: str = "grey") -> None:
    # Print a simple line of the given character (default '-')
    print(_c(ch * W, color))


def _print_section(title: str) -> None:
    print()
    _rule("=", "blue")   # using '=' for section lines
    print(_c(f"  {title}", "white", "bold"))
    _rule("=", "blue")


def _print_kv(key: str, val: str, w: int = 18) -> None:
    print(f"  {_c(key + ':', 'grey'):<{w+10}}  {val}")


def _print_warn(msg: str) -> None:
    print(_c(f"  [WARN]  {msg}", "orange"))


def _print_error(msg: str) -> None:
    print(_c(f"  [ERROR]  {msg}", "red", "bold"))


def print_banner() -> None:
    """Plain ASCII header."""
    print()
    _rule("=", "blue")
    lines = [
        "",
        "  MultiAnomalyDetector  -  Terminal Edition",
        "  Uncorrelated Anomaly Detection with Timestamped Alerts",
        "",
        "  Detecting 6 Categories:",
        "    1. TRAFFIC ACCIDENT            4. AMBULANCE STUCK IN TRAFFIC",
        "    2. STOP LINE VIOLATION         5. THEFT / STEALING",
        "    3. RED LIGHT VIOLATION         6. SUSPICIOUS ACTIVITY",
        "",
        "  Also detects:",
        "    - Pedestrian crossing (YELLOW alert with warning)",
        "",
        "  3 Detection Layers:  Motion (L1 + SSIM)  |  VQA (BLIP)  |  Keywords",
        "",
    ]
    for line in lines:
        print(_c(line, "cyan"))
    _rule("=", "blue")
    print()


def print_video_header(
    video_path: str,
    n_frames:   int,
    n_cands:    int,
    n_stalls:   int,
) -> None:
    _rule("-", "cyan")
    print(_c(f"  VIDEO  ->  {Path(video_path).name}", "white", "bold"))
    print(_c(
        f"  {n_frames} frames sampled @ {CHUNK_SECONDS}s intervals  |  "
        f"{n_cands} motion candidates  |  "
        f"{n_stalls} stall suspects",
        "grey",
    ))
    _rule("-", "cyan")
    hdr = (
        f"  {_c('[ Fr ]', 'grey', 'dim')}  "
        f"{_c('TIME ', 'grey', 'dim')}  "
        f"{_c(' t(s)  ', 'grey', 'dim')}  "
        f"{_c('[STATUS  ]', 'grey', 'dim')}  "
        f"{_c('LABEL / CAPTION', 'grey', 'dim')}"
    )
    print(hdr)
    print(_c("  " + "-" * (W - 2), "grey", "dim"))


def print_frame_line(
    idx:         int,
    timestamp:   str,
    time_sec:    float,
    alert_label: str,
    severity:    str,
    caption:     str,
    is_anomaly:  bool,
) -> None:
    fr_col = _c(f"[{idx:>4}]", "grey")
    tm_col = _c(f"{timestamp}  ({time_sec:>5.1f}s)", "grey")

    if not is_anomaly:
        status = _c("[ NORMAL ]", "green", "bold")
        cap_s  = caption[:72] + ("..." if len(caption) > 72 else "")
        print(f"  {fr_col}  {tm_col}  {status}  {_c(cap_s, 'grey')}")
    else:
        sc     = _SEV_COLOR.get(severity, "white")
        badge  = _c(f"[{severity:<8}]", sc, "bold")
        icon   = _CATEGORY_ICONS.get(alert_label, "")   # empty string
        lbl_s  = f"{icon} {alert_label[:32]}"
        cap_s  = caption[:50] + ("..." if len(caption) > 50 else "")
        print(
            f"  {fr_col}  {tm_col}  {badge}  "
            f"{_c(lbl_s, sc, 'bold')}  "
            f"{_c('|', 'grey')}  "                       # plain pipe instead of │
            f"{_c(cap_s, 'yellow')}"
        )


def print_alert_flash(
    alert_label: str,
    severity:    str,
    timestamp:   str,
    caption:     str,
    saved_path:  str,
) -> None:
    """Large alert block printed once per new anomaly type detected."""
    print("\nALERT DETECTED")
    print(f"Category  :  {alert_label}")
    print(f"Severity  :  {severity}")
    print(f"Timestamp :  {timestamp}  -  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    cap_wrapped = textwrap.fill(f"Caption   :  {caption}", width=W - 4,
                                subsequent_indent="              ")
    print(f"{cap_wrapped}")
    if saved_path:
        print(f"Saved     :  {saved_path}")


def print_summary(
    video_name: str,
    results:    list[dict],
    elapsed:    float,
    output_dir: str,
) -> None:
    total  = len(results)
    n_norm = sum(1 for r in results if not r["is_anomaly"])
    n_anom = total - n_norm

    # Per-category counts (6 primary + others)
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if r["is_anomaly"]:
            by_cat[r["alert"]].append(r)

    print()
    _rule("=", "cyan")
    print(_c(f"  ANALYSIS SUMMARY  -  {video_name}", "cyan", "bold"))
    _rule("-", "cyan")
    _print_kv("Total frames",     str(total))
    _print_kv("Normal frames",    _c(str(n_norm), "green", "bold"))
    _print_kv("Anomaly frames",
              _c(str(n_anom), "red" if n_anom else "green", "bold"))
    _print_kv("Processing time",  f"{elapsed:.1f}s  ({elapsed / max(total, 1):.2f}s/frame)")
    _print_kv("Output directory", output_dir)

    if by_cat:
        print()
        print("Anomalies by Category:")
        for cat, recs in sorted(by_cat.items(), key=lambda x: -len(x[1])):
            count  = len(recs)
            ts_list= "  ".join(r["timestamp"] for r in recs[:6])
            ts_str = ts_list + ("  ..." if count > 6 else "")
            print(f"  {cat:<36}  {count}")
            print(f"       timestamps: {ts_str}")
    else:
        print()
        print("No anomalies detected in this video.")

    _rule("=", "cyan")


def print_all_alerts_log(results: list[dict]) -> None:
    """Print a compact chronological log of every alert at the end."""
    alerts = [r for r in results if r["is_anomaly"]]
    if not alerts:
        return

    print()
    print("FULL ALERT LOG - Chronological")
    print(f"{'#':>3}  {'Time':>5}  {'t(s)':>7}  {'Category':<38}  {'Severity':<10}  Caption")
    for i, r in enumerate(alerts, 1):
        cap_s = r["caption"][:46] + ("..." if len(r["caption"]) > 46 else "")
        time_sec_str = '{:>6.1f}s'.format(r['time_sec'])
        alert_str = f'{r["alert"][:36]}'
        severity_str = f"{r['severity']:<10}"
        print(f"{str(i).rjust(3)}  {r['timestamp']}  {time_sec_str}  {alert_str:<38}  {severity_str}  {cap_s}")


# SECTION 10  -  MAIN PROCESSING PIPELINE

def process_video(video_path: str) -> list[dict]:
    """
    Full pipeline for one video file:
      1  Extract frames
      2  Compute motion scores (L1 diff + SSIM)
      3  Flag motion / stall candidates
      4  Analyse each frame (VQA + caption + keyword fusion)
      5  Save annotated screenshots for anomalous frames
      6  Generate contact sheet + timeline chart
      7  Print summary
    """
    video_name = Path(video_path).stem
    t_start    = time.time()

    # Step 1: extract
    print(_c(f"\n  Extracting frames (1 per {CHUNK_SECONDS}s) ...", "cyan"))
    frames = extract_frames(video_path)
    if not frames:
        _print_error(f"No frames extracted from: {video_path}")
        return []

    # Step 2: motion scoring
    print(_c("  Computing motion scores (L1 diff + SSIM) ...", "dim"))
    diffs, ssims = compute_motion(frames)
    arr_d  = np.array(diffs)
    thr_d  = float(arr_d.mean() + SIGMA_FACTOR * arr_d.std())

    motion_flags = flag_motion_candidates(diffs, ssims)
    stall_flags  = flag_stall_candidates(diffs)
    n_cands      = sum(motion_flags)
    n_stalls     = sum(stall_flags)

    print_video_header(video_path, len(frames), n_cands, n_stalls)
    print(_c("  Running 3-layer anomaly analysis ...\n", "dim"))

    results     : list[dict] = []
    seen_alerts : set[str]   = set()

    # Step 4: per-frame analysis
    for i, frame in enumerate(frames):
        caption, alert_label, severity = analyse_frame(
            frame["image"],
            is_cand  = motion_flags[i],
            is_stall = stall_flags[i],
        )

        # A frame is an anomaly if it's not merely informational.
        # Pedestrian crossing (YELLOW) is now considered an anomaly.
        is_anomaly = (
            severity not in ("INFO", "NORMAL")
            and alert_label not in (
                "NO ALERT", "TRAFFIC SCENE"   # PEDESTRIAN ON CROSSWALK removed
            )
        )

        # Step 5: save screenshot
        saved_path = ""
        if is_anomaly:
            try:
                saved_path = save_screenshot(
                    frame, caption, alert_label, severity, video_name
                )
            except Exception as exc:
                _print_warn(f"Screenshot save failed: {exc}")

        # Print frame line
        print_frame_line(frame["idx"], frame["timestamp"], frame["time_sec"],
                         alert_label, severity, caption, is_anomaly)

        # Print alert flash (first time per category)
        if is_anomaly and alert_label not in seen_alerts:
            seen_alerts.add(alert_label)
            print_alert_flash(alert_label, severity,
                              frame["timestamp"], caption, saved_path)

        results.append({
            "idx":        frame["idx"],
            "time_sec":   frame["time_sec"],
            "timestamp":  frame["timestamp"],
            "caption":    caption,
            "alert":      alert_label,
            "severity":   severity,
            "is_anomaly": is_anomaly,
            "diff":       diffs[i],
            "saved":      saved_path,
        })

    # Step 6: summary visuals
    safe_vid    = _safe_name(video_name)
    summary_dir = os.path.join(OUTPUT_ROOT, safe_vid, "_SUMMARY")
    os.makedirs(summary_dir, exist_ok=True)

    print(_c("\n  Generating contact sheet ...", "dim"))
    sheet_path = create_contact_sheet(frames, results, summary_dir)
    print(_c(f"  Contact sheet -> {sheet_path}", "grey"))

    if MATPLOTLIB_OK:
        print(_c("  Generating anomaly timeline chart ...", "dim"))
        tl_path = create_timeline(results, thr_d, summary_dir)
        if tl_path:
            print(_c(f"  Timeline      -> {tl_path}", "grey"))

    elapsed = time.time() - t_start

    # Step 7: summary + alert log
    print_summary(video_name, results, elapsed, os.path.join(OUTPUT_ROOT, safe_vid))
    print_all_alerts_log(results)

    return results


# SECTION 11  -  INTERACTIVE MENU  +  ENTRY POINT

def _video_duration(path: str) -> str:
    try:
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frm = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps > 0 and frm > 0:
            s = int(frm / fps)
            return f"{s // 60:02d}:{s % 60:02d}"
    except Exception:
        pass
    return "?:??"


def _collect_videos(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(
        str(p) for p in Path(folder).iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS
    )


def _select_video(video_list: list[str]) -> str | None:
    print(_c("  Available videos:", "white", "bold"))
    print()
    for i, path in enumerate(video_list, 1):
        size_mb = os.path.getsize(path) / 1_048_576
        print(
            f"    {_c(str(i).rjust(3), 'cyan', 'bold')}  "
            f"{_c(Path(path).name, 'white'):<52}  "
            f"{_c(_video_duration(path), 'grey')}  "
            f"{_c(f'{size_mb:.1f} MB', 'grey', 'dim')}"
        )
    print()
    print(_c("  Enter a number to select a video, or 0 to exit.", "grey"))
    print()
    while True:
        try:
            raw = input(_c("  Choice: ", "cyan", "bold")).strip()
        except (EOFError, KeyboardInterrupt):
            print(); return None
        if raw in ("", "0"):
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(video_list):
            chosen = video_list[int(raw) - 1]
            print()
            print(_c(f"  Selected: {Path(chosen).name}", "cyan"))
            print()
            return chosen
        print(_c(f"  Please enter a number between 1 and {len(video_list)}.", "orange"))


def _ask_another(folder: str) -> str | None:
    print()
    _rule("-", "cyan")
    try:
        ans = input(_c(
            "  Analyse another video? [y / n]: ", "cyan", "bold"
        )).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print(); return None
    if ans not in ("y", "yes"):
        print(_c(f"\n  All outputs saved in: {OUTPUT_ROOT}", "grey"))
        return None
    video_list = _collect_videos(folder)
    if not video_list:
        _print_error(f"No videos found in: {folder}")
        return None
    return _select_video(video_list)


def main() -> None:
    global CHUNK_SECONDS, OUTPUT_ROOT, VIDEOS_FOLDER

    parser = argparse.ArgumentParser(
        prog="anomaly_detector_terminal.py",
        description="MultiAnomalyDetector - Terminal-only 3-layer anomaly detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples
            --------
              Interactive menu (videos from ./videos/):
                python anomaly_detector_terminal.py

              Analyse a single file:
                python anomaly_detector_terminal.py --video clip.mp4

              Finer sampling (one frame every 1.5 s):
                python anomaly_detector_terminal.py --video clip.mp4 --chunk 1.5

              Custom folders:
                python anomaly_detector_terminal.py \\
                    --folder /data/videos --output /data/results

            Detected anomaly categories
            ---------------------------
              1. TRAFFIC ACCIDENT
              2. STOP LINE VIOLATION
              3. RED LIGHT VIOLATION
              4. AMBULANCE STUCK IN TRAFFIC
              5. THEFT / STEALING
              6. SUSPICIOUS ACTIVITY
              (plus pedestrian crossing - YELLOW alert)

            Required packages
            -----------------
              pip install opencv-python numpy Pillow scikit-image
              pip install torch torchvision transformers matplotlib
        """),
    )
    parser.add_argument("--video",  metavar="PATH",    help="Analyse one video file directly.")
    parser.add_argument("--folder", metavar="DIR",     default=VIDEOS_FOLDER,
                        help=f"Folder to scan for videos (default: {VIDEOS_FOLDER})")
    parser.add_argument("--output", metavar="DIR",     default=OUTPUT_ROOT,
                        help=f"Output directory (default: {OUTPUT_ROOT})")
    parser.add_argument("--chunk",  metavar="SECONDS", type=float, default=CHUNK_SECONDS,
                        help=f"Seconds between sampled frames (default: {CHUNK_SECONDS})")
    args = parser.parse_args()

    # Apply CLI overrides
    CHUNK_SECONDS  = args.chunk
    OUTPUT_ROOT    = args.output
    VIDEOS_FOLDER  = args.folder
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print_banner()
    load_model()

    # Determine first video
    if args.video:
        if not os.path.isfile(args.video):
            _print_error(f"File not found: {args.video}")
            sys.exit(1)
        current_video: str | None = args.video
    else:
        video_list = _collect_videos(args.folder)
        if not video_list:
            _print_error(f"No videos found in: {args.folder}")
            print(_c(f"  Tip: use --video <path> or place videos in: {args.folder}", "grey"))
            sys.exit(1)
        current_video = _select_video(video_list)
        if current_video is None:
            print(_c("  No video selected. Exiting.", "grey"))
            sys.exit(0)

    # Main loop
    while current_video is not None:
        try:
            results = process_video(current_video)
        except KeyboardInterrupt:
            print(_c("\n  Interrupted by user.", "yellow"))
            break
        except Exception as exc:
            _print_error(f"Processing failed for {Path(current_video).name}: {exc}")
            raise

        n_anom = sum(1 for r in results if r["is_anomaly"])
        print(_c(
            f"\n  [DONE]  {len(results)} frames analysed,  "
            f"{n_anom} anomalous frame(s) flagged.",
            "white", "bold",
        ))
        print(_c(f"  Output directory: {OUTPUT_ROOT}\n", "grey"))

        if args.video:
            break  # single-file mode: no loop
        current_video = _ask_another(args.folder)

    print(_c("\n  All done. Goodbye!\n", "cyan", "bold"))


if __name__ == "__main__":
    main()