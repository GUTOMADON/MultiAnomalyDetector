"""
MultiAnomalyDetector  v2.0  -  TERMINAL-ONLY VERSION
-----
Detects 6 uncorrelated anomaly categories with timestamped alerts,
confidence scoring, and temporal consistency tracking.

CATEGORIES
----------
1. TRAFFIC ACCIDENT

2. STOP LINE VIOLATION

3. RED LIGHT VIOLATION

4. AMBULANCE STUCK IN TRAFFIC

5. THEFT / STEALING

6. SUSPICIOUS ACTIVITY

DEPENDENCIES
----------
  pip install opencv-python numpy Pillow scikit-image
  pip install torch torchvision transformers matplotlib
"""

from __future__ import annotations
import argparse
import math
import os
import sys
import textwrap
import time
import warnings
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Optional
warnings.filterwarnings("ignore")

#  graceful imports 
try:
    import cv2
except ImportError:
    sys.exit("MISSING  ->  pip install opencv-python")

try:
    import numpy as np
except ImportError:
    sys.exit("MISSING  ->  pip install numpy")

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
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

#  SECTION 1  -  GLOBAL CONFIGURATION


VIDEOS_FOLDER      : str   = "./videosFolder"
OUTPUT_ROOT        : str   = "./output_anomalies"
CHUNK_SECONDS      : float = 2.0
MAX_FRAMES         : int   = 600

BLIP_CHECKPOINT    : str   = "Salesforce/blip-image-captioning-base"
BLIP_INPUT_SIZE    : int   = 384
CAPTION_TOKENS     : int   = 80
VQA_TOKENS         : int   = 10
NUM_BEAMS          : int   = 4

DIFF_SIZE          : tuple[int, int] = (96, 96)
SIGMA_FACTOR       : float = 0.60
DIFF_HARD_FLOOR    : float = 12.0

STALL_DIFF_MAX     : float = 7.0
STALL_MIN_FRAMES   : int   = 4

JPEG_QUALITY       : int   = 90

# Confidence thresholds (percentage of VQA questions answering YES)
MIN_CONFIDENCE_PCT : int   = 35     # below this -> no alert (overrideable via CLI)
HIGH_CONF_PCT      : int   = 65     # above this -> HIGH severity boost
CRITICAL_CONF_PCT  : int   = 80     # above this -> CRITICAL severity boost

# Temporal consistency: how many consecutive frames trigger a severity upgrade
TEMPORAL_WINDOW    : int   = 3      # frames in rolling window
TEMPORAL_BOOST_THR : int   = 2      # min frames in window to apply boost

# Alert cooldown: same category won't print a flash more often than this
ALERT_COOLDOWN_S   : float = 10.0

VIDEO_EXTENSIONS   : tuple[str, ...] = (
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"
)


#  SECTION 2  -  VQA QUESTION BANKS  (expanded, confidence-weighted)
#
#  Each bank is a list of questions.  The confidence score for a category is:
#      confidence = (number_of_YES_answers / len(bank)) * 100
#  Questions are ordered from most discriminative (first) to supporting (last).


# 1. TRAFFIC ACCIDENT
VQA_ACCIDENT = [
    # Primary discriminators
    "is there a car crash, vehicle collision, or traffic accident visible in this image?",
    "are two or more vehicles making physical contact, colliding, or crashing?",
    "has a vehicle been struck, rammed, or violently impacted by another vehicle?",
    # Secondary evidence
    "are there visible signs of a crash such as crumpled metal, broken glass, or debris on the road?",
    "is any vehicle overturned, flipped upside down, or lying on its side?",
    "are vehicles positioned at an unnatural angle suggesting a collision?",
    "is there a skid mark, black tyre mark, or debris trail on the road?",
    # Motion/dynamics cues
    "does any vehicle appear to have spun out, lost control, or gone off the road?",
    "are emergency responders, police, or ambulance attending a road incident?",
    "is there visible damage to road infrastructure such as barriers, poles, or signs?",
]

# 2. STOP LINE VIOLATION
VQA_STOP_LINE = [
    "is a vehicle positioned past or beyond the white stop line at an intersection?",
    "has a vehicle crossed the stop line while the traffic light is red?",
    "is the front of a car or truck clearly past the painted stop line at a signal?",
    "is a vehicle violating the stop line rule at a traffic signal junction?",
    "does a vehicle appear to have advanced past where it should have stopped?",
]

# Category 3: Red Light Violation
VQA_RED_LIGHT = [
    "is the traffic light in this image showing red?",
    "is a vehicle crossing an intersection while the traffic signal is red?",
    "is a car, truck, or motorcycle running a red light or ignoring a red traffic signal?",
    "is a vehicle entering or passing through an intersection against a red signal?",
    "is there a traffic light showing red with a vehicle clearly moving through?",
    "does it appear that a driver has disobeyed or ignored a red stop signal?",
    "is a vehicle caught mid-intersection when the light appears to be red?",
]

#  Red-light auxiliary: check light colour separately 
VQA_TRAFFIC_LIGHT_RED = [
    "is there a red traffic light visible and illuminated in this image?",
    "is the traffic signal showing a red light?",
    "can you see a red stop signal or red traffic light that is currently on?",
]

VQA_VEHICLE_CROSSING = [
    "is a vehicle moving through or crossing the intersection in this image?",
    "is a car or truck currently passing through an intersection or junction?",
    "is there a moving vehicle inside the marked intersection area?",
]

#  4. AMBULANCE STUCK IN TRAFFIC 
VQA_AMBULANCE_PRESENCE = [
    "is there an ambulance visible in this image?",
    "is there a fire truck, paramedic vehicle, or emergency service vehicle in this image?",
    "is there any vehicle with flashing red and blue emergency lights?",
    "can you see a white vehicle with red crosses or emergency markings?",
    "is there a vehicle that appears to be an emergency response vehicle?",
]

VQA_AMBULANCE_BLOCKED = [
    "is the ambulance or emergency vehicle surrounded by other vehicles and unable to move?",
    "is the emergency vehicle stuck in heavy traffic or gridlock?",
    "are other vehicles blocking the path of an ambulance or emergency vehicle?",
    "is an emergency vehicle unable to pass through or navigate the traffic?",
    "does it appear that the emergency vehicle's movement is being obstructed?",
    "are drivers failing to give way or yield to an emergency vehicle?",
    "is the ambulance or emergency vehicle stationary when it should be moving urgently?",
]

# Combined ambulance bank used for quick screening
VQA_AMBULANCE = VQA_AMBULANCE_PRESENCE[:3] + VQA_AMBULANCE_BLOCKED[:3]

#  5. THEFT / STEALING 
VQA_THEFT = [
    # Direct theft actions
    "is someone stealing, shoplifting, or taking items without paying in this image?",
    "is a person concealing or hiding merchandise, products, or items under clothing or in a bag?",
    "is someone grabbing items and fleeing or running away from a store or person?",
    "is there a robbery, bag snatching, phone snatching, or theft occurring?",
    # Behavioural cues
    "is a person pocketing, stuffing, or secretly placing items into their bag without scanning them?",
    "does someone appear to be taking an item that does not belong to them?",
    "is a person quickly hiding or concealing an object they just picked up?",
    "is someone running or hurrying away after picking up items?",
    # Context cues
    "are there unpaid goods, merchandise, or products being carried away without going through a checkout?",
    "is there a confrontation, chase, or altercation related to someone taking property?",
    "is a person leaving a store or area without paying for items visible in their hands or bag?",
]

#  6. SUSPICIOUS ACTIVITY 
VQA_SUSPICIOUS = [
    # Core suspicion indicators
    "is a person acting suspiciously, loitering without apparent purpose, or behaving strangely?",
    "is someone behaving in an unusual, erratic, or threatening manner in this scene?",
    "is a person lurking, trespassing, or watching a building, vehicle, or person nervously?",
    "does any person appear to be casing, surveilling, or planning criminal activity?",
    # Behavioural cues
    "is someone looking over their shoulder, acting nervous, or checking if they are being watched?",
    "is a person wearing a mask, hood, or disguise while behaving strangely?",
    "is there a person standing in an unusual location for an extended time without reason?",
    "is someone attempting to tamper with, break into, or interfere with a vehicle or property?",
    # Environmental cues
    "does the scene show signs of vandalism, graffiti, or property damage in progress?",
    "is there something about the body language or actions of a person that seems threatening?",
    "is a person making furtive or secretive movements near a restricted or private area?",
]

# Secondary / gated banks
VQA_WRONG_WAY = [
    "is a vehicle driving on the wrong side of the road directly toward oncoming traffic?",
    "is any vehicle facing the opposite direction from normal traffic flow on this road?",
    "is a car driving against the direction of traffic in this lane?",
    "does a vehicle appear to have entered a one-way road going the wrong direction?",
]

VQA_FALL = [
    "has a person collapsed, fallen, or been knocked to the ground?",
    "is there a person lying motionless or flat on the road or pavement?",
    "does it appear that someone has had a fall or medical emergency and is on the ground?",
    "is a person in a position that suggests they have fallen or been knocked down?",
]

VQA_FIRE = [
    "is there visible fire, flames, or burning in this image?",
    "is there heavy smoke, thick black smoke, or signs of burning?",
    "is a vehicle, building, or object on fire or visibly burning?",
    "does the scene show evidence of a fire or explosion?",
]

VQA_STALL = [
    "is there a bus, truck, van, or large vehicle stopped in the middle of a lane and blocking traffic?",
    "is a vehicle parked illegally, broken down, or stalled in an active traffic lane?",
    "is any vehicle obstructing traffic flow because it is not moving when it should be?",
    "does a stationary vehicle appear to be causing a traffic obstruction or blockage?",
]

VQA_PEDESTRIAN = [
    "is a pedestrian using or crossing a marked crosswalk or zebra crossing?",
    "is a person crossing the street at an intersection or crosswalk?",
    "are there pedestrians walking across the road in this image?",
    "is someone crossing the street, possibly in front of traffic?",
    "is a person on the roadway or in a crosswalk area?",
]

VQA_VIOLENCE = [
    "is there a physical fight, brawl, or violent altercation in this image?",
    "is someone being assaulted, hit, punched, or attacked?",
    "are two or more people engaged in a violent confrontation?",
    "is there evidence of a physical attack or violence against a person?",
]

VQA_SEVERITY = (
    "how severe is this incident on the road or in this scene? "
    "answer only: minor, moderate, or severe"
)

_YES_TOKENS = (
    "yes", "yeah", "yep", "true", "correct", "certainly",
    "definitely", "absolutely", "affirmative", "indeed", "it is",
    "there is", "there are", "i can see", "visible",
)
_SEV_TOKENS = frozenset({"minor", "moderate", "severe"})


def is_yes(answer: str) -> bool:
    t = (answer or "").strip().lower()
    return any(t.startswith(w) for w in _YES_TOKENS)


def any_yes(answers: list[str]) -> bool:
    return any(is_yes(a) for a in answers)


def confidence_pct(answers: list[str]) -> float:
    """Return percentage of YES answers (0.0 – 100.0)."""
    if not answers:
        return 0.0
    yes_count = sum(1 for a in answers if is_yes(a))
    return (yes_count / len(answers)) * 100.0


def parse_sev(raw: str) -> str:
    if not raw:
        return "n/a"
    w = raw.strip().lower().split()[0]
    return w if w in _SEV_TOKENS else "n/a"


#  SECTION 3  -  KEYWORD ALERT TABLE  (500+ terms, 6 primary categories)

ALERT_RULES: list[tuple[list[str], str, str]] = [

    # 1. TRAFFIC ACCIDENT 
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
        "pile-up","pileup","pile up","chain reaction","chain-reaction",
        "multi-vehicle","multi-car","multiple vehicle","multiple cars",
        "rear-end","rear ended","rear-ended","rear collision","tailgated",
        "head-on","head on","head-on collision","frontal collision",
        "side collision","broadside","sideswipe","sideswiped",
        "t-bone","t-boned","side impact","lateral collision",
        "rammed","ramming","struck by","struck a","struck the",
        "hit by a car","hit by vehicle","hit by truck","hit by bus",
        "ran into","drove into","slammed into","knocked into",
        "plowed into","ploughed into","veered into",
        "run over","ran over","running over","knocked down by car",
        "pedestrian struck","pedestrian hit","person struck",
        "person hit by car","cyclist hit","motorcycle collision",
        "motorcycle crash","bike crash","bicycle accident",
        "airbag deployed","airbags deployed","airbag visible",
        "car off road","vehicle off road","hit a barrier","hit a wall",
        "hit a pole","hit a tree","into a ditch","into a median",
        "accident scene","crash scene","flipped vehicle","crashed vehicle",
        "traffic incident","road incident","vehicle damage","road damage",
        "broken windshield","shattered glass","glass on road",
        "debris on road","wreckage on road","vehicle debris",
    ], "TRAFFIC ACCIDENT", "CRITICAL"),

    # 2. STOP LINE VIOLATION
    ([
        "stop line violation","stop-line violation",
        "crossed stop line","crossing stop line",
        "ran the stop line","ran stop line",
        "passed the stop line","crossed the stop line",
        "over the stop line","beyond the stop line","past the stop line",
        "crossed the white line","over the white line","past the white line",
        "stop line breached","ignored stop line","violated stop line",
        "failed to stop at line","did not stop at line",
        "violation of stop line","encroached stop line",
        "advanced past stop","stopped beyond line","beyond the stop",
    ], "STOP LINE VIOLATION", "CRITICAL"),

    #  3. RED LIGHT VIOLATION 
    ([
        "red light violation","red-light violation",
        "ran red light","ran a red light","ran the red light",
        "crossed on red","crossing on red","crossed on a red",
        "running a red","running the red","running red light",
        "drove through red","drove through a red","drove through the red",
        "went through red","passed a red light","passed through red",
        "ran through red","against a red light","against a red signal",
        "ignoring red light","ignored red light","disregarding red light",
        "traffic signal violation","signal violation","signal infraction",
        "disregarded red light","beat the red light","through a red signal",
        "disobeyed signal","ran a stop light","blew a red","blew a light",
        "moving on red","vehicle on red","car on red",
        "intersection on red","entering on red",
        "failed to stop at red","did not stop at red",
        "vehicle through intersection on red","red signal ignored",
    ], "RED LIGHT VIOLATION", "CRITICAL"),

    #  4. AMBULANCE STUCK IN TRAFFIC 
    ([
        "ambulance stuck","ambulance blocked","ambulance is stuck",
        "ambulance is blocked","ambulance cannot move","ambulance unable to move",
        "ambulance not moving","ambulance delayed","ambulance in traffic",
        "ambulance trapped","ambulance surrounded","ambulance gridlocked",
        "emergency vehicle stuck","emergency vehicle blocked",
        "emergency vehicle cannot pass","emergency vehicle unable",
        "fire truck blocked","fire truck stuck","fire engine blocked",
        "police car blocked","police vehicle stuck","police blocked",
        "paramedic vehicle blocked","ambulance cannot pass",
        "blocked ambulance","emergency services delayed",
        "emergency response blocked","emergency access blocked",
        "not yielding to ambulance","failure to yield to ambulance",
        "blocking ambulance","obstructing ambulance",
        "vehicles surrounding ambulance","cars blocking ambulance",
        "no path for ambulance","ambulance has no path",
        "sirens blocked","siren unable","lights and sirens stuck",
        "lights flashing blocked","flashing lights blocked",
        "emergency vehicle obstructed","rescue vehicle blocked",
        "rescue blocked","paramedics blocked","ems blocked",
    ], "AMBULANCE STUCK IN TRAFFIC", "CRITICAL"),

    #  5. THEFT / STEALING 
    ([
        "steal","stole","stealing","steals","stolen",
        "theft","thief","thieves","thieving",
        "shoplifting","shoplifter","shoplifts","shoplifted",
        "robbery","robbing","robbed","rob","robs","armed robbery",
        "burglary","burglar","burglarize","break in","breaking in",
        "mugging","mugged","mugger","mugs","street robbery",
        "pickpocket","pickpocketing","pickpocketed","pocket theft",
        "snatch","snatching","snatched","bag snatch","purse snatch",
        "phone snatch","chain snatch","jewellery snatch",
        "grabs","grabbed","grabbing","snatches","grabs and runs",
        "takes item","taking item","took item","takes product",
        "takes merchandise","taking merchandise","merchandise theft",
        "concealing item","concealing merchandise","hides item","hiding item",
        "hides goods","hides product","conceals goods",
        "pockets item","pocketing item","stuffs into bag","stuffs item",
        "puts item in bag","slips item","slips product",
        "without paying","without scanning","unpaid item","unpaid goods",
        "flees store","runs from store","leaves without paying",
        "store theft","retail theft","shoplifting incident",
        "item stolen","goods stolen","merchandise stolen",
        "people stealing","person stealing","person taking",
        "taking without permission","walking out without paying",
        "evading payment","skipping checkout","bypassing payment",
        "concealing under jacket","hiding under clothes",
        "theft in progress","robbery in progress","stealing in progress",
    ], "THEFT / STEALING", "CRITICAL"),

    #  6. SUSPICIOUS ACTIVITY 
    ([
        "suspicious activity","suspicious person","suspicious individual",
        "suspicious behavior","suspicious behaviour","suspicious movement",
        "acting suspiciously","behaving suspiciously","looks suspicious",
        "behaving strangely","behaving erratically","acting strangely",
        "unusual activity","unusual behaviour","unusual behavior",
        "unusual movement","strange activity","strange behavior",
        "strange behaviour","unusual person",
        "loitering","loiter","loiters","loitered",
        "trespassing","trespass","trespasses","trespassed",
        "lurking","lurk","lurks","lurked",
        "casing the","casing a building","casing a store","casing a vehicle",
        "watching nervously","looking around nervously","looking suspicious",
        "erratic behavior","erratic behaviour","erratic movement",
        "irrational behavior","threatening behavior","threatening behaviour",
        "intimidating","menacing","making threats",
        "vandalism","vandalizing","spray painting","graffiti","tagging",
        "prowling","prowl","prowls","prowled",
        "tampering with","interfering with","attempting to enter",
        "trying to break in","peering inside","peering through",
        "checking locks","checking doors","trying handles",
        "nervous behavior","anxious behavior","furtive movements",
        "suspicious package","unattended package","abandoned bag",
        "checking surroundings repeatedly","looking over shoulder",
        "wearing disguise","wearing mask suspiciously",
    ], "SUSPICIOUS ACTIVITY", "HIGH"),

    # Secondary / environmental categories
    (["wrong way","wrong-way vehicle","wrong side of the road",
      "driving against traffic","oncoming lane","against traffic flow",
      "wrong direction","against the flow","driving into oncoming"],
     "WRONG-WAY VEHICLE", "CRITICAL"),

    (["fire","on fire","in flames","smoke","thick smoke","burning",
      "ablaze","flame","flames","engulfed in fire","vehicle on fire",
      "building on fire","structure fire","car fire","bus fire"],
     "FIRE / SMOKE", "CRITICAL"),

    (["person down","fallen","collapsed","lying on the road",
      "lying on the ground","knocked down","unconscious","motionless",
      "unresponsive","person on ground","body on ground","fell down"],
     "PERSON DOWN / FALL", "HIGH"),

    (["fight","fighting","brawl","violence","assault","punching",
      "kicking","beating","hitting","altercation","physical fight",
      "violent confrontation","attacking","gang fight","mob attack"],
     "VIOLENCE / ASSAULT", "HIGH"),

    (["vehicle on crosswalk","car on crosswalk","bus on crosswalk",
      "blocking the crosswalk","blocking pedestrian crossing",
      "stopped on crosswalk","bus stopped at crosswalk",
      "obstructing crosswalk","blocking zebra crossing"],
     "VEHICLE BLOCKING CROSSWALK", "CRITICAL"),

    (["pedestrian crossing","person crossing","pedestrian on crosswalk",
      "crossing the street","zebra crossing","crosswalk crossing",
      "crossing at light","pedestrian at intersection"],
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


#  SECTION 4  -  BLIP MODEL  (loaded once, reused)


_processor : Optional["BlipProcessor"]                = None
_model     : Optional["BlipForConditionalGeneration"] = None
_device    : Optional["torch.device"]                 = None


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
    """Remove BLIP word-repetition artefacts."""
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
    """Run multiple VQA questions; return list of lowercase answers."""
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


#  SECTION 5  -  ROI CROPS  (region-of-interest for specialised checks)


def crop_top_half(img: Image.Image) -> Image.Image:
    """Upper half: useful for traffic lights, overhead signals."""
    w, h = img.size
    return img.crop((0, 0, w, h // 2))


def crop_bottom_half(img: Image.Image) -> Image.Image:
    """Lower half: useful for pedestrians, road markings."""
    w, h = img.size
    return img.crop((0, h // 2, w, h))


def crop_centre(img: Image.Image, ratio: float = 0.6) -> Image.Image:
    """Central crop: useful for close-up action analysis."""
    w, h = img.size
    mw = int(w * (1 - ratio) / 2)
    mh = int(h * (1 - ratio) / 2)
    return img.crop((mw, mh, w - mw, h - mh))


def enhance_contrast(img: Image.Image) -> Image.Image:
    """Boost contrast to improve BLIP performance on dark scenes."""
    return ImageEnhance.Contrast(img).enhance(1.4)


#  SECTION 6  -  FRAME EXTRACTION  +  MOTION SCORING  (Layer 1)


def extract_frames(video_path: str) -> list[dict]:
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


#  SECTION 7  -  TEMPORAL CONSISTENCY TRACKER

class TemporalTracker:
    """
    Keeps a rolling window of recent alert labels.
    If the same category appears >= TEMPORAL_BOOST_THR times in the window,
    the severity can be boosted by one level.
    """

    _SEV_ORDER = ["NORMAL", "INFO", "YELLOW", "MEDIUM", "HIGH", "CRITICAL"]

    def __init__(self, window: int = TEMPORAL_WINDOW, threshold: int = TEMPORAL_BOOST_THR):
        self.window    = window
        self.threshold = threshold
        self._history  : deque[str] = deque(maxlen=window)

    def push(self, label: str) -> None:
        self._history.append(label)

    def boost_severity(self, label: str, current_sev: str) -> str:
        """Return potentially upgraded severity if label is persistent."""
        count = sum(1 for l in self._history if l == label)
        if count < self.threshold:
            return current_sev
        idx = self._SEV_ORDER.index(current_sev) if current_sev in self._SEV_ORDER else 0
        new_idx = min(idx + 1, len(self._SEV_ORDER) - 1)
        boosted = self._SEV_ORDER[new_idx]
        if boosted != current_sev:
            _print_warn(
                f"Temporal boost: {label} appeared {count}x in last {self.window} frames "
                f"-> {current_sev} upgraded to {boosted}"
            )
        return boosted

    def recent_labels(self) -> list[str]:
        return list(self._history)



#  SECTION 8  -  PER-FRAME ANALYSIS  (Layers 2 + 3 + 4)

PROMPT_NORMAL        = "a traffic or surveillance camera photo showing"
PROMPT_ANOMALY       = "a surveillance camera clearly showing an incident where"
PROMPT_ACCIDENT      = "a traffic camera showing a road collision or accident where"
PROMPT_THEFT         = "a store surveillance camera showing a person stealing where"
PROMPT_SUSPICIOUS    = "a surveillance camera showing a person behaving suspiciously"
PROMPT_AMBULANCE     = "a traffic camera showing an ambulance or emergency vehicle where"
PROMPT_RED_LIGHT     = "a traffic camera at an intersection showing a vehicle that ran a red light"
PROMPT_STALL         = "a traffic camera showing a vehicle stopped or blocking the road"
PROMPT_PEDESTRIAN    = "a traffic camera showing pedestrians crossing the street"
PROMPT_VIOLENCE      = "a surveillance camera showing a violent confrontation where"
PROMPT_FIRE          = "a camera showing fire or smoke where"


def _severity_rank(sev: str) -> int:
    return {"NORMAL": 0, "INFO": 0, "YELLOW": 1,
            "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}.get(sev, 0)


def analyse_frame(
    img:       Image.Image,
    is_cand:   bool,
    is_stall:  bool,
    min_conf:  int = MIN_CONFIDENCE_PCT,
) -> tuple[str, str, str, dict[str, float]]:
    """
    3-layer analysis returning (caption, alert_label, severity, confidences).

    Layer 2A  – All 6 primary category banks run on EVERY frame.
    Layer 2B  – Secondary banks gated on motion / stall candidates.
    Layer 2C  – Two-step red-light check using ROI crops.
    Layer 2D  – Ambulance two-step: presence THEN blocked.
    Layer 3   – Caption + keyword.
    Layer 4   – Confidence gating: only emit alert if conf >= min_conf.

    Returns
    -------
    caption        : str
    alert_label    : str
    severity       : str
    confidences    : dict mapping category name -> confidence pct (0-100)
    """

    # Prepare ROI crops
    img_top    = crop_top_half(img)
    img_bottom = crop_bottom_half(img)
    img_enh    = enhance_contrast(img)

    # Layer 2A: primary banks (all frames)
    acc_ans  = blip_vqa_batch(img,     VQA_ACCIDENT)
    red_ans  = blip_vqa_batch(img_top, VQA_RED_LIGHT)         # top for lights
    stop_ans = blip_vqa_batch(img,     VQA_STOP_LINE)
    tft_ans  = blip_vqa_batch(img_enh, VQA_THEFT)
    sus_ans  = blip_vqa_batch(img_enh, VQA_SUSPICIOUS)

    # Layer 2C: two-step red-light ROI
    red_light_ans  = blip_vqa_batch(img_top, VQA_TRAFFIC_LIGHT_RED)
    crossing_ans   = blip_vqa_batch(img,     VQA_VEHICLE_CROSSING)
    # Red-light violation = light IS red AND vehicle IS crossing
    red_two_step   = any_yes(red_light_ans) and any_yes(crossing_ans)

    # Layer 2D: ambulance two-step
    amb_presence   = blip_vqa_batch(img, VQA_AMBULANCE_PRESENCE)
    amb_blocked    = blip_vqa_batch(img, VQA_AMBULANCE_BLOCKED)
    ambulance_vqa  = any_yes(amb_presence) and any_yes(amb_blocked)
    # Confidence for ambulance is the PRODUCT of both sub-confidences
    amb_conf       = (confidence_pct(amb_presence) + confidence_pct(amb_blocked)) / 2

    # Layer 2B: secondary banks (motion / stall gated) 
    if is_cand or is_stall:
        stl_ans  = blip_vqa_batch(img,        VQA_STALL)    if is_stall else ["n/a"]
        ww_ans   = blip_vqa_batch(img,        VQA_WRONG_WAY)
        fire_ans = blip_vqa_batch(img_enh,    VQA_FIRE)
        fall_ans = blip_vqa_batch(img,        VQA_FALL)
        ped_ans  = blip_vqa_batch(img_bottom, VQA_PEDESTRIAN)
        vio_ans  = blip_vqa_batch(img,        VQA_VIOLENCE)
    else:
        stl_ans = ww_ans = fire_ans = fall_ans = ped_ans = vio_ans = ["n/a"]

    # Confidence scores 
    confidences: dict[str, float] = {
        "TRAFFIC ACCIDENT":           confidence_pct(acc_ans),
        "STOP LINE VIOLATION":        confidence_pct(stop_ans),
        "RED LIGHT VIOLATION":        max(confidence_pct(red_ans),
                                         100.0 if red_two_step else 0.0),
        "AMBULANCE STUCK IN TRAFFIC": amb_conf,
        "THEFT / STEALING":           confidence_pct(tft_ans),
        "SUSPICIOUS ACTIVITY":        confidence_pct(sus_ans),
        "WRONG-WAY VEHICLE":          confidence_pct(ww_ans),
        "FIRE / SMOKE":               confidence_pct(fire_ans),
        "PERSON DOWN / FALL":         confidence_pct(fall_ans),
        "STALLED / BLOCKING VEHICLE": confidence_pct(stl_ans),
        "PEDESTRIAN ON CROSSWALK":    confidence_pct(ped_ans),
        "VIOLENCE / ASSAULT":         confidence_pct(vio_ans),
    }

    # Boolean flags with confidence gating
    def _triggered(cat: str, extra: bool = False) -> bool:
        return confidences[cat] >= min_conf or extra

    accident_vqa   = _triggered("TRAFFIC ACCIDENT")
    red_vqa        = _triggered("RED LIGHT VIOLATION", red_two_step)
    stop_vqa       = _triggered("STOP LINE VIOLATION")
    ambulance_vqa2 = _triggered("AMBULANCE STUCK IN TRAFFIC", ambulance_vqa)
    theft_vqa      = _triggered("THEFT / STEALING")
    suspicious_vqa = _triggered("SUSPICIOUS ACTIVITY")
    stall_vqa      = _triggered("STALLED / BLOCKING VEHICLE")
    wrong_way_vqa  = _triggered("WRONG-WAY VEHICLE")
    fire_vqa       = _triggered("FIRE / SMOKE")
    fall_vqa       = _triggered("PERSON DOWN / FALL")
    ped_vqa        = _triggered("PEDESTRIAN ON CROSSWALK")
    violence_vqa   = _triggered("VIOLENCE / ASSAULT")

    # Layer 3: choose caption prompt
    if accident_vqa:
        prompt = PROMPT_ACCIDENT
    elif theft_vqa:
        prompt = PROMPT_THEFT
    elif suspicious_vqa:
        prompt = PROMPT_SUSPICIOUS
    elif ambulance_vqa2:
        prompt = PROMPT_AMBULANCE
    elif red_vqa or stop_vqa:
        prompt = PROMPT_RED_LIGHT
    elif violence_vqa:
        prompt = PROMPT_VIOLENCE
    elif fire_vqa:
        prompt = PROMPT_FIRE
    elif stall_vqa:
        prompt = PROMPT_STALL
    elif ped_vqa:
        prompt = PROMPT_PEDESTRIAN
    elif is_cand or wrong_way_vqa or fall_vqa:
        prompt = PROMPT_ANOMALY
    else:
        prompt = PROMPT_NORMAL

    caption = blip_caption(img, prompt=prompt)
    alert_label, severity = classify_caption(caption)

    # Fusion: VQA overrides keyword classification 
    def _upgrade(lbl: str, sev: str) -> None:
        nonlocal alert_label, severity
        if _severity_rank(sev) > _severity_rank(severity):
            alert_label, severity = lbl, sev

    if accident_vqa:    _upgrade("TRAFFIC ACCIDENT",           "CRITICAL")
    if red_vqa:         _upgrade("RED LIGHT VIOLATION",        "CRITICAL")
    if stop_vqa:        _upgrade("STOP LINE VIOLATION",        "CRITICAL")
    if ambulance_vqa2:  _upgrade("AMBULANCE STUCK IN TRAFFIC", "CRITICAL")
    if theft_vqa:       _upgrade("THEFT / STEALING",           "CRITICAL")
    if wrong_way_vqa:   _upgrade("WRONG-WAY VEHICLE",          "CRITICAL")
    if fire_vqa:        _upgrade("FIRE / SMOKE",               "CRITICAL")
    if violence_vqa:    _upgrade("VIOLENCE / ASSAULT",         "HIGH")
    if suspicious_vqa:  _upgrade("SUSPICIOUS ACTIVITY",        "HIGH")
    if stall_vqa:       _upgrade("STALLED / BLOCKING VEHICLE", "HIGH")
    if fall_vqa:        _upgrade("PERSON DOWN / FALL",         "HIGH")
    if ped_vqa:
        _upgrade("PEDESTRIAN ON CROSSWALK", "YELLOW")
        caption = f"WARNING: Pedestrian crossing. {caption}"

    # Confidence upgrade: very high confidence -> upgrade severity
    top_conf = confidences.get(alert_label, 0.0)
    if top_conf >= CRITICAL_CONF_PCT and _severity_rank(severity) < 4:
        old_sev = severity
        severity = "CRITICAL"
        if old_sev != severity:
            caption = f"[HIGH CONFIDENCE {top_conf:.0f}%] {caption}"
    elif top_conf >= HIGH_CONF_PCT and _severity_rank(severity) < 3:
        severity = "HIGH"

    #Append severity qualifier from BLIP 
    any_critical = (accident_vqa or red_vqa or stop_vqa or ambulance_vqa2
                    or theft_vqa or suspicious_vqa or wrong_way_vqa
                    or fire_vqa or fall_vqa or violence_vqa)
    if any_critical or is_cand:
        sv = parse_sev(blip_vqa_single(img, VQA_SEVERITY))
        if sv != "n/a":
            caption = f"{caption}  [{sv}]"

    return caption, alert_label, severity, confidences


#  SECTION 9  -  ANNOTATED SCREENSHOT SAVING


_BANNER_PALETTE = {
    "CRITICAL": ((110, 10, 10),  (255, 55, 55),  (225, 175, 175)),
    "HIGH":     ((90,  45,  0),  (255, 140, 30), (215, 180, 140)),
    "YELLOW":   ((70,  60,  0),  (230, 200,  0), (210, 195, 130)),
    "NORMAL":   ((14,  60, 22),  ( 50, 200, 70), (160, 190, 160)),
    "INFO":     ((14,  60, 22),  ( 50, 200, 70), (160, 190, 160)),
}


def save_screenshot(
    frame:       dict,
    caption:     str,
    alert_label: str,
    severity:    str,
    confidence:  float,
    video_name:  str,
) -> str:
    img  = frame["image"].copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)

    pal           = _BANNER_PALETTE.get(severity, _BANNER_PALETTE["NORMAL"])
    fill, lbl_c, cap_c = pal
    banner_h      = max(70, h // 8)
    y0            = h - banner_h

    draw.rectangle([(0, y0), (w, h)], fill=fill)

    # Category pill
    pill_text = f"{alert_label}  [{confidence:.0f}%]"
    pill_w    = len(pill_text) * 7 + 20
    draw.rounded_rectangle([(4, y0 + 4), (pill_w, y0 + 26)],
                            radius=5, fill=lbl_c)
    draw.text((10, y0 + 8), pill_text, fill=fill)

    # Timestamp line
    ts_str = (
        f"  {frame['timestamp']}  (t={frame['time_sec']:.1f}s)"
        f"  [{severity}]  {datetime.now().strftime('%H:%M:%S')}"
    )
    draw.text((pill_w + 8, y0 + 8), ts_str, fill=lbl_c)

    # Caption line
    cap_line = caption[:115] + ("..." if len(caption) > 115 else "")
    draw.text((6, y0 + banner_h // 2 + 6), cap_line, fill=cap_c)

    # Confidence bar
    bar_w = int((w - 12) * min(confidence / 100.0, 1.0))
    draw.rectangle([(6, y0 + banner_h - 10), (6 + bar_w, y0 + banner_h - 4)],
                   fill=lbl_c)

    # Save
    safe_name  = _safe_name(video_name)
    ts_stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name  = f"{ts_stamp}_f{frame['idx']:04d}_{frame['time_sec']:.1f}s.jpg"
    sub_folder = os.path.join(OUTPUT_ROOT, safe_name, _safe_name(alert_label))
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


#  SECTION 10  -  SUMMARY VISUALS  (contact sheet + timeline)

def create_contact_sheet(
    frames:  list[dict],
    results: list[dict],
    out_dir: str,
) -> str:
    N_COLS       = 5
    TW, TH, LH  = 220, 135, 90
    n_rows       = math.ceil(len(frames) / N_COLS)
    grid         = Image.new("RGB", (N_COLS * TW, n_rows * (TH + LH)), (10, 12, 20))
    draw         = ImageDraw.Draw(grid)

    for pos, fr in enumerate(frames):
        res   = results[pos]
        cap   = res.get("caption", "")
        col   = pos % N_COLS
        row   = pos // N_COLS
        x, y  = col * TW, row * (TH + LH)
        sev   = res.get("severity", "NORMAL")
        anom  = res.get("is_anomaly", False)
        conf  = res.get("top_confidence", 0.0)

        grid.paste(fr["image"].resize((TW, TH), Image.LANCZOS), (x, y))

        if sev == "CRITICAL":
            fc, oc, lc = (100, 8, 8),  (200, 25, 25), (255, 70, 70); lbl = "CRITICAL"
        elif sev == "HIGH":
            fc, oc, lc = (90, 50, 0),  (200, 100, 0), (255, 140, 30); lbl = "HIGH"
        elif sev == "YELLOW":
            fc, oc, lc = (80, 70, 0),  (200, 180, 0), (240, 215, 0); lbl = "YELLOW"
        else:
            fc, oc, lc = (8, 50, 18),  (20, 140, 50), (60, 240, 90); lbl = "NORMAL"

        draw.rectangle([(x, y + TH), (x + TW - 1, y + TH + LH - 1)], fill=fc)
        draw.text((x + 4, y + TH + 3),  lbl,                            fill=lc)
        draw.text((x + 4, y + TH + 18),
                  f"t={res['time_sec']:.1f}s  Δ={res['diff']:.1f}  conf={conf:.0f}%",
                  fill=(180, 180, 180))
        draw.text((x + 4, y + TH + 36),
                  res.get("alert", "")[:30],
                  fill=(240, 190, 50) if anom else (130, 130, 180))
        draw.text((x + 4, y + TH + 54),
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
    times  = [r["time_sec"]        for r in results]
    diffs  = [r["diff"]            for r in results]
    confs  = [r["top_confidence"]  for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    fig.patch.set_facecolor("#080c14")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0d1220")

    # Frame diff subplot
    ax1.plot(times, diffs, color="#38bdf8", lw=1.6, zorder=2)
    ax1.fill_between(times, diffs, alpha=0.12, color="#38bdf8")
    ax1.axhline(thr_d, color="#fbbf24", linestyle="--", lw=1.8, zorder=4)
    for r in results:
        if r.get("is_anomaly"):
            sc = {"CRITICAL": "#ef4444", "HIGH": "#fb923c"}.get(r.get("severity"), "#fbbf24")
            ax1.scatter(r["time_sec"], r["diff"], color=sc, s=70, zorder=5)
            ax1.annotate(r.get("alert", "")[:18],
                         (r["time_sec"], r["diff"]),
                         textcoords="offset points", xytext=(4, 5),
                         fontsize=4.8, color="#fca5a5")
    ax1.set_ylabel("Frame Δ", color="#94a3b8")
    ax1.set_title("Frame Difference Over Time", color="#e2e8f0",
                  fontsize=11, fontfamily="monospace")
    ax1.tick_params(colors="#64748b")
    for sp in ax1.spines.values():
        sp.set_edgecolor("#1e293b")
    ax1.grid(axis="y", color="#1e293b", linestyle="--", lw=0.8)

    # Confidence subplot
    ax2.plot(times, confs, color="#a78bfa", lw=1.6, zorder=2)
    ax2.fill_between(times, confs, alpha=0.12, color="#a78bfa")
    ax2.axhline(MIN_CONFIDENCE_PCT, color="#f97316", linestyle=":", lw=1.5,
                label=f"Min conf {MIN_CONFIDENCE_PCT}%")
    ax2.set_ylabel("Confidence %", color="#94a3b8")
    ax2.set_xlabel("Time (s)",     color="#94a3b8")
    ax2.set_ylim(0, 105)
    ax2.set_title("Category Confidence Over Time", color="#e2e8f0",
                  fontsize=11, fontfamily="monospace")
    ax2.tick_params(colors="#64748b")
    for sp in ax2.spines.values():
        sp.set_edgecolor("#1e293b")
    ax2.grid(axis="y", color="#1e293b", linestyle="--", lw=0.8)

    handles = [
        mpatches.Patch(color="#38bdf8", label="Frame diff"),
        mpatches.Patch(color="#a78bfa", label="Confidence %"),
        mpatches.Patch(color="#ef4444", label="CRITICAL"),
        mpatches.Patch(color="#fb923c", label="HIGH"),
        mpatches.Patch(color="#fbbf24", label="YELLOW"),
        plt.Line2D([0], [0], color="#fbbf24", linestyle="--", label="Diff threshold"),
    ]
    ax1.legend(handles=handles, facecolor="#0d1220", labelcolor="#cbd5e1",
               edgecolor="#1e293b", fontsize=7, loc="upper right")
    plt.tight_layout()
    path = os.path.join(out_dir, "anomaly_timeline.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return path


def create_confidence_radar(
    results: list[dict],
    out_dir: str,
) -> str:
    """Bar chart of mean confidence per category across the entire video."""
    if not MATPLOTLIB_OK:
        return ""
    anomaly_results = [r for r in results if r.get("is_anomaly")]
    if not anomaly_results:
        return ""

    cat_confs: dict[str, list[float]] = defaultdict(list)
    for r in anomaly_results:
        for cat, conf in r.get("confidences", {}).items():
            if conf > 0:
                cat_confs[cat].append(conf)

    cats   = list(cat_confs.keys())
    means  = [sum(v) / len(v) for v in cat_confs.values()]
    if not cats:
        return ""

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#080c14")
    ax.set_facecolor("#0d1220")

    colours = ["#ef4444" if m >= HIGH_CONF_PCT else "#fb923c"
               if m >= MIN_CONFIDENCE_PCT else "#374151"
               for m in means]
    bars = ax.barh(cats, means, color=colours, edgecolor="#1e293b", height=0.6)
    ax.axvline(MIN_CONFIDENCE_PCT, color="#f97316", linestyle=":", lw=1.5,
               label=f"Min conf ({MIN_CONFIDENCE_PCT}%)")
    ax.axvline(HIGH_CONF_PCT, color="#ef4444", linestyle="--", lw=1.5,
               label=f"High conf ({HIGH_CONF_PCT}%)")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Mean Confidence (%)", color="#94a3b8")
    ax.set_title("Mean Detection Confidence per Category",
                 color="#e2e8f0", fontsize=11, fontfamily="monospace")
    ax.tick_params(colors="#94a3b8")
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e293b")
    ax.legend(facecolor="#0d1220", labelcolor="#cbd5e1",
              edgecolor="#1e293b", fontsize=8)
    for bar, val in zip(bars, means):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", ha="left",
                color="#cbd5e1", fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "confidence_radar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return path

#  SECTION 11  -  TERMINAL OUTPUT HELPERS

W = 82

_C = {
    "reset": "\033[0m", "bold":   "\033[1m",  "dim":    "\033[2m",
    "red":   "\033[91m","orange": "\033[93m", "green":  "\033[92m",
    "cyan":  "\033[96m","blue":   "\033[94m", "grey":   "\033[90m",
    "white": "\033[97m","yellow": "\033[33m", "purple": "\033[95m",
    "teal":  "\033[36m",
}

_SEV_COLOR = {
    "CRITICAL": "red", "HIGH": "orange", "YELLOW": "yellow",
    "MEDIUM":   "teal","INFO": "grey",   "NORMAL": "green",
}


def _c(text: str, *codes: str) -> str:
    return "".join(_C.get(k, "") for k in codes) + str(text) + _C["reset"]


def _rule(ch: str = "-", color: str = "grey") -> None:
    print(_c(ch * W, color))


def _print_section(title: str) -> None:
    print(); _rule("=", "blue")
    print(_c(f"  {title}", "white", "bold"))
    _rule("=", "blue")


def _print_kv(key: str, val: str, w: int = 20) -> None:
    print(f"  {_c(key + ':', 'grey'):<{w + 10}}  {val}")


def _print_warn(msg: str) -> None:
    print(_c(f"  [WARN]  {msg}", "orange"))


def _print_error(msg: str) -> None:
    print(_c(f"  [ERROR]  {msg}", "red", "bold"))


def print_banner() -> None:
    print(); _rule("=", "blue")
    lines = [
        "",
        "  MultiAnomalyDetector  v2.0  -  Terminal Edition",
        "  Enhanced Detection with Confidence Scoring + Temporal Tracking",
        "",
        "  Primary Categories:",
        "    1. TRAFFIC ACCIDENT            4. AMBULANCE STUCK IN TRAFFIC",
        "    2. STOP LINE VIOLATION         5. THEFT / STEALING",
        "    3. RED LIGHT VIOLATION         6. SUSPICIOUS ACTIVITY",
        "",
        "  Also Detects:",
        "    - Pedestrian on crosswalk (YELLOW)    - Fire / Smoke (CRITICAL)",
        "    - Wrong-way vehicle (CRITICAL)         - Violence / Assault (HIGH)",
        "    - Person down / fall (HIGH)            - Stalled vehicle (HIGH)",
        "",
        "  4 Detection Layers:",
        "    L1 Motion (L1+SSIM)  |  L2 VQA (BLIP + ROI crops)  |",
        "    L3 Keywords (500+)   |  L4 Temporal consistency",
        "",
    ]
    for line in lines:
        print(_c(line, "cyan"))
    _rule("=", "blue"); print()


def print_video_header(
    video_path: str, n_frames: int, n_cands: int, n_stalls: int,
) -> None:
    _rule("-", "cyan")
    print(_c(f"  VIDEO  ->  {Path(video_path).name}", "white", "bold"))
    print(_c(
        f"  {n_frames} frames sampled @ {CHUNK_SECONDS}s  |  "
        f"{n_cands} motion candidates  |  {n_stalls} stall suspects",
        "grey",
    ))
    _rule("-", "cyan")
    print(
        f"  {_c('[ Fr ]', 'grey', 'dim')}  "
        f"{_c('TIME      ', 'grey', 'dim')}  "
        f"{_c('[STATUS  ]', 'grey', 'dim')}  "
        f"{_c('[CONF%]', 'grey', 'dim')}  "
        f"{_c('LABEL / CAPTION', 'grey', 'dim')}"
    )
    print(_c("  " + "-" * (W - 2), "grey", "dim"))


def print_frame_line(
    idx: int, timestamp: str, time_sec: float,
    alert_label: str, severity: str, caption: str,
    is_anomaly: bool, confidence: float,
) -> None:
    fr_col = _c(f"[{idx:>4}]", "grey")
    tm_col = _c(f"{timestamp} ({time_sec:>5.1f}s)", "grey")

    # Confidence logic is preserved, but not printed
    if alert_label == "PEDESTRIAN ON CROSSWALK" or severity == "YELLOW":
        badge = _c("[WARNING ]", "yellow", "bold")
        lbl   = _c("PEDESTRIAN ON CROSSWALK", "yellow", "bold")
        cap_s = caption[:45] + ("..." if len(caption) > 45 else "")
        print(f"  {fr_col}  {tm_col}  {badge}  {lbl}  {_c('|','grey')}  {_c(cap_s,'yellow')}")
    elif not is_anomaly:
        status = _c("[ NORMAL ]", "green", "bold")
        cap_s  = caption[:65] + ("..." if len(caption) > 65 else "")
        print(f"  {fr_col}  {tm_col}  {status}  {_c(cap_s,'grey')}")
    else:
        sc    = _SEV_COLOR.get(severity, "white")
        badge = _c(f"[{severity:<8}]", sc, "bold")
        lbl   = _c(alert_label[:30], sc, "bold")
        cap_s = caption[:45] + ("..." if len(caption) > 45 else "")
        print(f"  {fr_col}  {tm_col}  {badge}  {lbl}  {_c('|','grey')}  {_c(cap_s,'yellow')}")


def print_alert_flash(
    alert_label: str, severity: str, timestamp: str,
    caption: str, saved_path: str, confidence: float,
) -> None:
    print()
    print(_c("=" * W, "red" if severity == "CRITICAL" else "orange"))
    print(_c(f"  *** ALERT DETECTED ***", "white", "bold"))
    print(_c(f"  Category   :  {alert_label}", "white", "bold"))
    print(_c(f"  Severity   :  {severity}", _SEV_COLOR.get(severity, "white"), "bold"))
    print(_c(f"  Confidence :  {confidence:.1f}%", "purple"))
    print(_c(
        f"  Timestamp  :  {timestamp}  -  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "grey",
    ))
    cap_w = textwrap.fill(
        f"  Caption    :  {caption}", width=W - 4,
        subsequent_indent="               ",
    )
    print(_c(cap_w, "yellow"))
    if saved_path:
        print(_c(f"  Saved      :  {saved_path}", "grey"))
    print(_c("=" * W, "red" if severity == "CRITICAL" else "orange"))
    print()


def print_summary(
    video_name: str, results: list[dict],
    elapsed: float, output_dir: str,
) -> None:
    total  = len(results)
    n_norm = sum(1 for r in results if not r["is_anomaly"])
    n_anom = total - n_norm
    avg_conf = (
        sum(r["top_confidence"] for r in results if r["is_anomaly"]) / max(n_anom, 1)
    )

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if r["is_anomaly"]:
            by_cat[r["alert"]].append(r)

    print(); _rule("=", "cyan")
    print(_c(f"  ANALYSIS SUMMARY  -  {video_name}", "cyan", "bold"))
    _rule("-", "cyan")
    _print_kv("Total frames",        str(total))
    _print_kv("Normal frames",       _c(str(n_norm), "green", "bold"))
    _print_kv("Anomaly frames",      _c(str(n_anom), "red" if n_anom else "green", "bold"))
    _print_kv("Avg alert confidence",_c(f"{avg_conf:.1f}%", "purple"))
    _print_kv("Processing time",     f"{elapsed:.1f}s  ({elapsed / max(total, 1):.2f}s/frame)")
    _print_kv("Output directory",    output_dir)

    if by_cat:
        print()
        print(_c("  Anomalies by Category:", "white", "bold"))
        for cat, recs in sorted(by_cat.items(), key=lambda x: -len(x[1])):
            count   = len(recs)
            avg_c   = sum(r["top_confidence"] for r in recs) / count
            ts_list = "  ".join(r["timestamp"] for r in recs[:6])
            ts_str  = ts_list + ("  ..." if count > 6 else "")
            sev_col = _SEV_COLOR.get(recs[0]["severity"], "white")
            print(f"  {_c(cat, sev_col, 'bold'):<50}  "
                  f"{_c(str(count), 'white', 'bold')} frames  "
                  f"{_c(f'avg conf {avg_c:.0f}%', 'purple')}")
            print(f"       timestamps: {_c(ts_str, 'grey')}")
    else:
        print(); print(_c("  No anomalies detected in this video.", "green"))

    _rule("=", "cyan")


def print_all_alerts_log(results: list[dict]) -> None:
    alerts = [r for r in results if r["is_anomaly"]]
    if not alerts:
        return
    print(); _rule("-", "cyan")
    print(_c("  FULL ALERT LOG - Chronological", "white", "bold"))
    _rule("-", "cyan")
    hdr = (f"  {'#':>3}  {'Time':>5}  {'t(s)':>7}  "
           f"{'Category':<38}  {'Sev':<10}  Caption")
    print(_c(hdr, "grey"))
    for i, r in enumerate(alerts, 1):
        cap_s    = r["caption"][:40] + ("..." if len(r["caption"]) > 40 else "")
        sev_col  = _SEV_COLOR.get(r["severity"], "white")
        formatted_severity = f"{r['severity']:<10}"
        print(
            f"  {str(i).rjust(3)}  {r['timestamp']}  {r['time_sec']:>6.1f}s"
            f"{_c(r['alert'][:36],''):<38}  "
            f"{_c(formatted_severity, sev_col)}  "
            f"{_c(cap_s,'yellow')}"
        )

#  SECTION 12  -  MAIN PROCESSING PIPELINE

def process_video(video_path: str, min_confidence: int = MIN_CONFIDENCE_PCT) -> list[dict]:
    """
    Full pipeline for one video file:
      1  Extract frames
      2  Compute motion scores (L1 diff + SSIM)
      3  Flag motion / stall candidates
      4  Per-frame: VQA + ROI crops + caption + keyword + confidence fusion
      5  Apply temporal consistency boosts (Layer 4)
      6  Save annotated screenshots for anomalous frames
      7  Generate contact sheet + timeline + confidence chart
      8  Print summary
    """
    video_name = Path(video_path).stem
    t_start    = time.time()

    print(_c(f"\n  Extracting frames (1 per {CHUNK_SECONDS}s) ...", "cyan"))
    frames = extract_frames(video_path)
    if not frames:
        _print_error(f"No frames extracted from: {video_path}")
        return []

    print(_c("  Computing motion scores (L1 diff + SSIM) ...", "dim"))
    diffs, ssims = compute_motion(frames)
    arr_d        = np.array(diffs)
    thr_d        = float(arr_d.mean() + SIGMA_FACTOR * arr_d.std())

    motion_flags = flag_motion_candidates(diffs, ssims)
    stall_flags  = flag_stall_candidates(diffs)
    n_cands      = sum(motion_flags)
    n_stalls     = sum(stall_flags)

    print_video_header(video_path, len(frames), n_cands, n_stalls)
    print(_c("  Running 4-layer anomaly analysis ...\n", "dim"))

    tracker        = TemporalTracker()
    results        : list[dict]  = []
    seen_alerts    : dict[str, float] = {}   # label -> last flash time

    for i, frame in enumerate(frames):
        caption, alert_label, severity, confidences = analyse_frame(
            frame["image"],
            is_cand  = motion_flags[i],
            is_stall = stall_flags[i],
            min_conf = min_confidence,
        )

        # Layer 4: temporal consistency
        tracker.push(alert_label)
        severity = tracker.boost_severity(alert_label, severity)

        top_conf = confidences.get(alert_label, 0.0)

        is_anomaly = (
            severity not in ("INFO", "NORMAL")
            and alert_label not in ("NO ALERT", "TRAFFIC SCENE")
        )

        # Save screenshot
        saved_path = ""
        if is_anomaly:
            try:
                saved_path = save_screenshot(
                    frame, caption, alert_label, severity, top_conf, video_name
                )
            except Exception as exc:
                _print_warn(f"Screenshot save failed: {exc}")

        print_frame_line(
            frame["idx"], frame["timestamp"], frame["time_sec"],
            alert_label, severity, caption, is_anomaly, top_conf,
        )

        # Alert flash with cooldown
        now = time.time()
        last_flash = seen_alerts.get(alert_label, 0.0)
        if is_anomaly and (now - last_flash) >= ALERT_COOLDOWN_S:
            seen_alerts[alert_label] = now
            print_alert_flash(
                alert_label, severity, frame["timestamp"],
                caption, saved_path, top_conf,
            )

        results.append({
            "idx":            frame["idx"],
            "time_sec":       frame["time_sec"],
            "timestamp":      frame["timestamp"],
            "caption":        caption,
            "alert":          alert_label,
            "severity":       severity,
            "is_anomaly":     is_anomaly,
            "diff":           diffs[i],
            "saved":          saved_path,
            "top_confidence": top_conf,
            "confidences":    confidences,
        })

    # Generate summary visuals
    safe_vid    = _safe_name(video_name)
    summary_dir = os.path.join(OUTPUT_ROOT, safe_vid, "_SUMMARY")
    os.makedirs(summary_dir, exist_ok=True)

    print(_c("\n  Generating contact sheet ...", "dim"))
    sheet_path = create_contact_sheet(frames, results, summary_dir)
    print(_c(f"  Contact sheet    ->  {sheet_path}", "grey"))

    if MATPLOTLIB_OK:
        print(_c("  Generating timeline chart ...", "dim"))
        tl_path = create_timeline(results, thr_d, summary_dir)
        if tl_path:
            print(_c(f"  Timeline         ->  {tl_path}", "grey"))
        print(_c("  Generating confidence chart ...", "dim"))
        cr_path = create_confidence_radar(results, summary_dir)
        if cr_path:
            print(_c(f"  Confidence chart ->  {cr_path}", "grey"))

    elapsed = time.time() - t_start
    print_summary(video_name, results, elapsed, os.path.join(OUTPUT_ROOT, safe_vid))
    print_all_alerts_log(results)
    return results

#  SECTION 13  -  INTERACTIVE MENU + ENTRY POINT

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
    print(_c("  Available videos:", "white", "bold")); print()
    for i, path in enumerate(video_list, 1):
        size_mb = os.path.getsize(path) / 1_048_576
        print(
            f"    {_c(str(i).rjust(3),'cyan','bold')}  "
            f"{_c(Path(path).name,'white'):<52}  "
            f"{_c(_video_duration(path),'grey')}  "
            f"{_c(f'{size_mb:.1f} MB','grey','dim')}"
        )
    print(); print(_c("  Enter a number to select a video, or 0 to exit.", "grey")); print()
    while True:
        try:
            raw = input(_c("  Choice: ", "cyan", "bold")).strip()
        except (EOFError, KeyboardInterrupt):
            print(); return None
        if raw in ("", "0"):
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(video_list):
            chosen = video_list[int(raw) - 1]
            print(); print(_c(f"  Selected: {Path(chosen).name}", "cyan")); print()
            return chosen
        print(_c(f"  Please enter a number between 1 and {len(video_list)}.", "orange"))


def _ask_another(folder: str) -> str | None:
    print(); _rule("-", "cyan")
    try:
        ans = input(_c("  Analyse another video? [y / n]: ", "cyan", "bold")).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print(); return None
    if ans not in ("y", "yes"):
        print(_c(f"\n  All outputs saved in: {OUTPUT_ROOT}", "grey")); return None
    video_list = _collect_videos(folder)
    if not video_list:
        _print_error(f"No videos found in: {folder}"); return None
    return _select_video(video_list)


def main() -> None:
    global CHUNK_SECONDS, OUTPUT_ROOT, VIDEOS_FOLDER, MIN_CONFIDENCE_PCT

    parser = argparse.ArgumentParser(
        prog="anomaly_detector_v2.py",
        description="MultiAnomalyDetector v2.0 - 4-layer anomaly detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples
            --------
              Interactive menu:
                python anomaly_detector_v2.py

              Single file:
                python anomaly_detector_v2.py --video clip.mp4

              Fine sampling + custom confidence:
                python anomaly_detector_v2.py --video clip.mp4 --chunk 1.0 --min-confidence 40

              Custom folders:
                python anomaly_detector_v2.py --folder /data/videos --output /data/results

            Key flags
            ---------
              --chunk          Seconds between sampled frames (default: 2.0)
              --min-confidence Minimum VQA confidence % to trigger alert (default: 35)
              --folder         Folder containing videos (default: ./videosFolder)
              --output         Output directory (default: ./output_anomalies)
        """),
    )
    parser.add_argument("--video",          metavar="PATH",    help="Single video file.")
    parser.add_argument("--folder",         metavar="DIR",     default=VIDEOS_FOLDER)
    parser.add_argument("--output",         metavar="DIR",     default=OUTPUT_ROOT)
    parser.add_argument("--chunk",          metavar="SECONDS", type=float, default=CHUNK_SECONDS)
    parser.add_argument("--min-confidence", metavar="PCT",     type=int,
                        default=MIN_CONFIDENCE_PCT,
                        help=f"Min VQA confidence to emit alert (default: {MIN_CONFIDENCE_PCT})")
    args = parser.parse_args()

    CHUNK_SECONDS      = args.chunk
    OUTPUT_ROOT        = args.output
    VIDEOS_FOLDER      = args.folder
    MIN_CONFIDENCE_PCT = args.min_confidence
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print_banner()
    _print_kv("Min confidence threshold", f"{MIN_CONFIDENCE_PCT}%")
    _print_kv("Temporal window",          f"{TEMPORAL_WINDOW} frames")
    _print_kv("Temporal boost threshold", f"{TEMPORAL_BOOST_THR} frames")
    print()
    load_model()

    # Select first video
    if args.video:
        if not os.path.isfile(args.video):
            _print_error(f"File not found: {args.video}"); sys.exit(1)
        current_video: str | None = args.video
    else:
        video_list = _collect_videos(args.folder)
        if not video_list:
            _print_error(f"No videos found in: {args.folder}")
            print(_c(f"  Tip: use --video <path> or place videos in: {args.folder}", "grey"))
            sys.exit(1)
        current_video = _select_video(video_list)
        if current_video is None:
            print(_c("  No video selected. Exiting.", "grey")); sys.exit(0)

    while current_video is not None:
        try:
            results = process_video(current_video, min_confidence=MIN_CONFIDENCE_PCT)
        except KeyboardInterrupt:
            print(_c("\n  Interrupted by user.", "yellow")); break
        except Exception as exc:
            _print_error(f"Processing failed: {exc}"); raise

        n_anom = sum(1 for r in results if r["is_anomaly"])
        print(_c(
            f"\n  [DONE]  {len(results)} frames analysed,  "
            f"{n_anom} anomalous frame(s) flagged.",
            "white", "bold",
        ))
        print(_c(f"  Output directory: {OUTPUT_ROOT}\n", "grey"))

        if args.video:
            break
        current_video = _ask_another(args.folder)

    print(_c("\n  All done. Goodbye!\n", "cyan", "bold"))


if __name__ == "__main__":
    main()