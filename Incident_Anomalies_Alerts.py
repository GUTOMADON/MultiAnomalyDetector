"""
Video Anomaly Detection & Captioning — ENHANCED
--------------------------------------------------
NEW in this version
  ✦ 6 specific anomaly scenarios:
      1. Traffic Accident
      2. Stop Line Violation
      3. Red Light Violation
      4. Ambulance Stuck in Traffic
      5. People Stealing Items from Store
      6. Suspicious Activity

  ✦ Uncorrelated Anomaly Detection
      Uses an Isolation Forest trained on per-frame signal vectors
      (frame-difference, SSIM, VQA bits, keyword scores) so that
      co-occurring but *independent* events are scored separately
      rather than merged into a single "anomaly" label.

  ✦ Alert with timestamp for every detected scenario
      Alerts are printed to the terminal, saved to alerts.json,
      and appended to the PDF-ready report.

Original features retained:
  - Extracts frames at 1 FPS (configurable)
  - BLIP large captioning + VQA chain
  - Colored terminal output
  - Saves frames/, collisions/, report.json, grid, timeline chart
"""

import subprocess, sys, os, json, math, shutil, warnings
from datetime import datetime
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim

warnings.filterwarnings("ignore")

# ── Optional heavy deps ───────────────────────────────────────────────────────────
try:
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# ── Configuration ─────────────────────────────────────────────────────────────────
VIDEO_PATH  = r"input_video.mp4"
OUTPUT_DIR  = "output_video"
FRAMES_DIR  = os.path.join(OUTPUT_DIR, "frames")
ANOMALY_DIR = os.path.join(OUTPUT_DIR, "collisions")
REPORT_PATH = os.path.join(OUTPUT_DIR, "report.json")
ALERTS_PATH = os.path.join(OUTPUT_DIR, "alerts.json")
CHART_PATH  = os.path.join(OUTPUT_DIR, "anomaly_timeline.png")
GRID_PATH   = os.path.join(OUTPUT_DIR, "all_frames_grid.jpg")

EXTRACT_FPS = 1.0
MAX_FRAMES  = 60
CHUNK_SIZE  = 10
DIFF_SIZE   = (96, 96)

# Isolation Forest contamination — expected fraction of anomalous frames
IF_CONTAMINATION = 0.15

RED    = "\033[91m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — SCENARIO DEFINITIONS
#  Each scenario is a dict with:
#    name        : human-readable label
#    severity    : CRITICAL | HIGH | MEDIUM | LOW
#    vqa_q       : optional VQA question to ask BLIP
#    keywords    : caption keywords that trigger this scenario
#    vqa_yes     : if VQA answer starts with one of these → positive signal
# ═══════════════════════════════════════════════════════════════════════════════
SCENARIOS = [
    {
        "name":     "Traffic Accident",
        "severity": "CRITICAL",
        "vqa_q":    "is there a vehicle collision or traffic accident in this image?",
        "keywords": ["crash", "collision", "collide", "accident", "wreck", "smash",
                     "impact", "overturned", "flipped", "pile-up", "rear-end",
                     "head-on", "sideswipe", "t-bone", "spun out", "skidded",
                     "slammed", "rammed", "struck", "hit by", "ran into"],
        "vqa_yes":  ["yes", "yeah", "true"],
    },
    {
        "name":     "Stop Line Violation",
        "severity": "HIGH",
        "vqa_q":    "is a vehicle crossing or past the stop line at an intersection?",
        "keywords": ["stop line", "crossed stop", "past the line", "over the line",
                     "ignoring stop", "ran the stop", "stop sign violation",
                     "rolled through stop"],
        "vqa_yes":  ["yes", "yeah", "true"],
    },
    {
        "name":     "Red Light Violation",
        "severity": "HIGH",
        "vqa_q":    "is a vehicle running a red traffic light in this image?",
        "keywords": ["red light", "ran red", "ran a red", "red-light violation",
                     "traffic light violation", "through red", "ignoring red"],
        "vqa_yes":  ["yes", "yeah", "true"],
    },
    {
        "name":     "Ambulance Stuck in Traffic",
        "severity": "HIGH",
        "vqa_q":    "is an ambulance or emergency vehicle blocked or stuck in traffic?",
        "keywords": ["ambulance", "emergency vehicle", "fire truck", "police car",
                     "rescue", "paramedic", "ems", "stuck emergency",
                     "blocked ambulance", "congested emergency"],
        "vqa_yes":  ["yes", "yeah", "true"],
    },
    {
        "name":     "Shoplifting / Theft",
        "severity": "HIGH",
        "vqa_q":    "is a person stealing, shoplifting, or concealing items in this image?",
        "keywords": ["stealing", "shoplifting", "concealing", "theft", "shoplifter",
                     "taking without paying", "hiding items", "pilfering",
                     "pocketing items", "grab and run"],
        "vqa_yes":  ["yes", "yeah", "true"],
    },
    {
        "name":     "Suspicious Activity",
        "severity": "MEDIUM",
        "vqa_q":    "is there any suspicious or unusual behavior visible in this image?",
        "keywords": ["suspicious", "unusual", "loitering", "lurking", "trespassing",
                     "vandalism", "graffiti", "tampering", "breaking in",
                     "unauthorized", "prowling", "acting suspiciously"],
        "vqa_yes":  ["yes", "yeah", "true"],
    },
]

# ─── Legacy alert rules (still used for the chunk summary) ────────────────────
ALERT_RULES = [
    (["crash", "collision", "collide", "accident", "wreck", "smash", "impact",
      "overturned", "overturn", "flipped", "rolled over", "pile-up", "pileup",
      "rear-end", "head-on", "sideswipe", "t-bone", "spun out", "skidded",
      "slammed", "rammed", "struck", "hit by", "ran into", "drove into",
      "ran a red", "ran red light", "wrong way", "wrong side"],
     "COLLISION / CRASH",     "CRITICAL"),
    (["fire", "smoke", "burning", "flame", "blaze", "engulfed", "on fire"],
     "FIRE / SMOKE",          "CRITICAL"),
    (["fall", "fallen", "falling", "knocked down", "lying on road",
      "lying in street", "lying on ground", "hit pedestrian", "struck pedestrian",
      "pedestrian down", "person down", "run over", "runover"],
     "PERSON DOWN / FALL",    "HIGH"),
    (["fight", "fighting", "violence", "attack", "assault", "brawl"],
     "VIOLENCE",              "HIGH"),
    (["debris", "obstacle", "object on road", "tire on road", "car parts",
      "broken glass", "scattered", "blocking the road", "blocking road"],
     "ROAD DEBRIS / HAZARD",  "HIGH"),
    (["speeding", "racing", "chase", "high speed", "reckless"],
     "RECKLESS DRIVING",      "MEDIUM"),
    (["skid", "swerve", "swerving", "sliding", "lost control", "hydroplane"],
     "LOSS OF CONTROL",       "MEDIUM"),
    (["stalled", "broken down", "stopped in lane", "blocking lane",
      "disabled vehicle", "flat tire"],
     "STALLED VEHICLE",       "MEDIUM"),
    (["jaywalking", "crossing", "pedestrian crossing", "person crossing",
      "person walking"],
     "PEDESTRIAN",            "LOW"),
    (["traffic", "road", "street", "highway", "car", "vehicle", "truck",
      "bus", "motorcycle", "parking", "parked", "intersection", "driving",
      "lane", "signal", "light"],
     "TRAFFIC SCENE",         "INFO"),
]

def map_caption_to_alert(caption: str):
    cap = caption.lower()
    for keywords, label, severity in ALERT_RULES:
        if any(kw in cap for kw in keywords):
            return label, severity
    return "NO ALERT", "NORMAL"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — ALERT GENERATION WITH TIMESTAMP
# ═══════════════════════════════════════════════════════════════════════════════

_alert_log: list[dict] = []   # collected during the run; dumped to alerts.json

def generate_alert(scenario_name: str, severity: str,
                   frame_idx: int, time_sec: float,
                   caption: str = "") -> dict:
    """
    Create a structured alert dict and print a coloured terminal line.
    Returns the alert dict so callers can embed it in frame_results.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert = {
        "timestamp":     timestamp,
        "frame_idx":     frame_idx,
        "video_time_s":  round(time_sec, 2),
        "scenario":      scenario_name,
        "severity":      severity,
        "description":   caption,
    }
    _alert_log.append(alert)

    color = RED if severity in ("CRITICAL", "HIGH") else YELLOW
    sev_tag = f"[{severity}]"
    print(f"  {color}{BOLD}⚠ ALERT {sev_tag:<10}{RESET}  "
          f"{color}{scenario_name:<35}{RESET}  "
          f"frame={frame_idx}  t={time_sec:.1f}s  "
          f"{DIM}@ {timestamp}{RESET}")
    return alert


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — PER-SCENARIO DETECTION FUNCTIONS
#  Each returns True/False for a single frame.
#  Signal sources (in priority order):
#    1. VQA answer from BLIP   (most reliable when model is loaded)
#    2. Keyword scan of caption
#    3. IF anomaly score       (uncorrelated signal from feature vector)
# ═══════════════════════════════════════════════════════════════════════════════

def _keyword_hit(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(kw in t for kw in keywords)


def _vqa_positive(answer: str, yes_tokens: list[str]) -> bool:
    return any(answer.strip().lower().startswith(y) for y in yes_tokens)


def detect_scenario(scenario: dict,
                    caption: str,
                    vqa_answer: str,
                    if_score: float,
                    if_threshold: float = -0.05) -> bool:
    """
    Combine three independent signals with OR logic:
      • VQA positive  (BLIP visual question answering)
      • Caption keyword match
      • Isolation Forest outlier score below threshold
    The IF score is an *uncorrelated* channel — it fires on
    statistical anomalies even when the caption misses keywords.
    """
    vqa_ok     = _vqa_positive(vqa_answer, scenario["vqa_yes"])
    keyword_ok = _keyword_hit(caption, scenario["keywords"])
    if_ok      = (if_score < if_threshold)          # more negative = more anomalous
    return vqa_ok or keyword_ok or if_ok


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — UNCORRELATED ANOMALY DETECTION (Isolation Forest)
#
#  Strategy
#  --------
#  We build one Isolation Forest *per scenario* trained on a feature
#  vector that captures signals relevant to that scenario:
#
#    [frame_diff, ssim, vqa_bit, n_keyword_hits, normalised_time]
#
#  Training on separate forests means the model for "Red Light Violation"
#  is statistically independent of the model for "Shoplifting".
#  A simultaneous accident + shoplifting produces two *separate* alerts
#  with their own scores — they cannot cancel each other out.
# ═══════════════════════════════════════════════════════════════════════════════

def build_feature_vector(diff: float, sim: float,
                         vqa_bit: float,
                         caption: str,
                         scenario: dict,
                         time_norm: float) -> list[float]:
    """Single feature row for one (frame, scenario) pair."""
    n_keywords = sum(1 for kw in scenario["keywords"]
                     if kw in caption.lower())
    return [diff, sim, vqa_bit, float(n_keywords), time_norm]


def train_isolation_forests(all_frame_data: list[dict]) -> dict:
    """
    Train one IsolationForest per scenario.
    all_frame_data: list of dicts with keys
        diff, ssim, caption, vqa_answers (dict scenario_name->answer), time_norm
    Returns  {scenario_name: IsolationForest}
    """
    if not SKLEARN_AVAILABLE or not all_frame_data:
        return {}

    forests = {}
    for sc in SCENARIOS:
        X = []
        for fd in all_frame_data:
            vqa_bit = 1.0 if _vqa_positive(
                fd["vqa_answers"].get(sc["name"], "no"), sc["vqa_yes"]) else 0.0
            row = build_feature_vector(
                fd["diff"], fd["ssim"], vqa_bit,
                fd["caption"], sc, fd["time_norm"])
            X.append(row)
        X = np.array(X, dtype=np.float32)
        clf = IsolationForest(
            n_estimators=200,
            contamination=IF_CONTAMINATION,
            random_state=42,
        )
        clf.fit(X)
        forests[sc["name"]] = clf

    return forests


def get_if_score(clf, diff: float, ssim_val: float,
                 vqa_bit: float, caption: str,
                 scenario: dict, time_norm: float) -> float:
    if clf is None:
        return 0.0
    row = build_feature_vector(diff, ssim_val, vqa_bit, caption,
                               scenario, time_norm)
    return float(clf.score_samples([row])[0])


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — BLIP HELPERS  (unchanged from original, kept self-contained)
# ═══════════════════════════════════════════════════════════════════════════════

CAPTION_PROMPT         = "a traffic camera photo of"
ANOMALY_CAPTION_PROMPT = "a traffic camera showing an accident where"
VQA_SEVERITY           = "how severe is the incident in this image? answer: minor, moderate, or severe"
VALID_SEVERITIES       = {"minor", "moderate", "severe"}


def load_blip():
    if not BLIP_AVAILABLE:
        return None, None, None
    print(f"{CYAN}[BLIP] Loading model (blip-image-captioning-large)...{RESET}")
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large", use_fast=False)
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        ignore_mismatched_sizes=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"{CYAN}[BLIP] Loaded on {device}.{RESET}\n")
    return processor, model, device


def blip_caption(img, processor, model, device, prompt=None) -> str:
    with torch.no_grad():
        if prompt:
            inputs = processor(images=img, text=prompt,
                               return_tensors="pt").to(device)
        else:
            inputs = processor(images=img, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=60,
                             num_beams=5, length_penalty=1.2)
    text = processor.decode(out[0], skip_special_tokens=True)
    if prompt and text.lower().startswith(prompt.lower()):
        text = text[len(prompt):].strip()
    return text


def blip_vqa(img, processor, model, device, question: str) -> str:
    try:
        with torch.no_grad():
            inputs = processor(images=img, text=question,
                               return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=8)
        return processor.decode(out[0], skip_special_tokens=True).strip().lower()
    except Exception:
        return "no"


def parse_severity(raw: str) -> str:
    if not raw or raw == "n/a":
        return "n/a"
    first_word = raw.strip().lower().split()[0]
    return first_word if first_word in VALID_SEVERITIES else "n/a"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — FULL PER-FRAME ANALYSIS  (VQA + captioning + scenario detection)
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_frame_full(img, processor, model, device,
                       is_pixel_anomaly: bool,
                       frame_idx: int, time_sec: float,
                       diff: float, ssim_val: float,
                       forests: dict,
                       n_total_frames: int) -> dict:
    """
    Run BLIP caption + VQA for all 6 scenarios.
    Returns a dict with caption, per-scenario results, alerts fired.
    """
    result = {
        "frame_idx":      frame_idx,
        "time_sec":       time_sec,
        "difference":     diff,
        "ssim":           ssim_val,
        "is_pixel_anom":  is_pixel_anomaly,
        "caption":        "",
        "alert":          "NO ALERT",
        "severity":       "NORMAL",
        "scenarios_hit":  [],
        "vqa":            {},
        "alerts_fired":   [],
    }

    time_norm = time_sec / max(n_total_frames, 1)

    # ── Step 1: caption ───────────────────────────────────────────────────────
    if processor:
        prompt = ANOMALY_CAPTION_PROMPT if is_pixel_anomaly else CAPTION_PROMPT
        caption = blip_caption(img, processor, model, device, prompt=prompt)
    else:
        caption = "(no model)"
    result["caption"] = caption

    # ── Step 2: run VQA for every scenario ───────────────────────────────────
    vqa_answers: dict[str, str] = {}
    for sc in SCENARIOS:
        if processor:
            ans = blip_vqa(img, processor, model, device, sc["vqa_q"])
        else:
            ans = "no"
        vqa_answers[sc["name"]] = ans
    result["vqa"] = vqa_answers

    # ── Step 3: per-scenario IF scoring + detection ───────────────────────────
    triggered: list[str] = []
    highest_sev = "NORMAL"
    sev_rank = {"CRITICAL": 5, "HIGH": 4, "MEDIUM": 3, "LOW": 2, "INFO": 1, "NORMAL": 0}

    for sc in SCENARIOS:
        ans      = vqa_answers[sc["name"]]
        vqa_bit  = 1.0 if _vqa_positive(ans, sc["vqa_yes"]) else 0.0
        clf      = forests.get(sc["name"])
        if_score = get_if_score(clf, diff, ssim_val, vqa_bit,
                                caption, sc, time_norm)

        fired = detect_scenario(sc, caption, ans, if_score)
        if fired:
            triggered.append(sc["name"])
            alert_dict = generate_alert(
                sc["name"], sc["severity"],
                frame_idx, time_sec, caption)
            result["alerts_fired"].append(alert_dict)
            if sev_rank.get(sc["severity"], 0) > sev_rank.get(highest_sev, 0):
                highest_sev      = sc["severity"]
                result["alert"]  = sc["name"]
                result["severity"] = sc["severity"]

    result["scenarios_hit"] = triggered
    result["is_anomaly"]    = bool(triggered) or is_pixel_anomaly
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — DIRECTORY / FRAME / VISUALISATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def clean_dirs():
    print(f"{YELLOW}[SETUP] Clearing output directories...{RESET}")
    for d in (FRAMES_DIR, ANOMALY_DIR):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_frames(video_path: str) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"{RED}ERROR: Cannot open video: {video_path}{RESET}")
        return []
    frames, idx, sec = [], 0, 0.0
    while len(frames) < MAX_FRAMES:
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append({"frame_idx": idx, "time_sec": sec, "image": img})
        idx += 1
        sec += 1.0 / EXTRACT_FPS
    cap.release()
    return frames


def compute_anomaly_scores(frames):
    diffs, ssims = [0.0], [1.0]
    prev   = np.array(frames[0]["image"].resize(DIFF_SIZE)).astype(np.float32)
    prev_g = cv2.cvtColor(np.uint8(prev), cv2.COLOR_RGB2GRAY)
    for i in range(1, len(frames)):
        curr   = np.array(frames[i]["image"].resize(DIFF_SIZE)).astype(np.float32)
        curr_g = cv2.cvtColor(np.uint8(curr), cv2.COLOR_RGB2GRAY)
        diffs.append(float(np.mean(np.abs(curr - prev))))
        ssims.append(float(ssim(curr_g, prev_g, data_range=255)))
        prev, prev_g = curr, curr_g
    arr_d, arr_s = np.array(diffs), np.array(ssims)
    return (diffs, ssims,
            float(arr_d.mean() + 2 * arr_d.std()),
            float(arr_s.mean() - 2 * arr_s.std()))


def pixel_anomaly_flags(frames, diffs, ssims, thr_d, thr_s) -> list[bool]:
    """Original pixel-level anomaly labels (used as one input signal)."""
    low_d, high_s = thr_d * 0.50, thr_s * 1.50
    active, flags = False, []
    for i in range(len(frames)):
        d, s = diffs[i], ssims[i]
        if (d > thr_d) or (s < thr_s):
            active = True
        elif d < low_d and s > high_s:
            active = False
        flags.append(active)
    return flags


def save_frame_image(frame_data, res):
    fname = f"frame_{res['frame_idx']:05d}_t{res['time_sec']:.2f}s.jpg"
    frame_data["image"].save(os.path.join(FRAMES_DIR, fname), quality=92)
    if res["is_anomaly"]:
        img_copy = frame_data["image"].copy()
        draw     = ImageDraw.Draw(img_copy)
        w, h     = img_copy.size
        scenarios_str = ", ".join(res["scenarios_hit"][:3]) or res["alert"]
        draw.rectangle([(0, h - 52), (w, h)], fill=(160, 10, 10))
        draw.text((6, h - 50), f"ANOMALY: {scenarios_str}", fill=(255, 220, 50))
        draw.text((6, h - 30), res["caption"][:90],         fill=(255, 255, 255))
        img_copy.save(os.path.join(ANOMALY_DIR, fname), quality=92)


def save_grid(frames, frame_results, captions):
    N_COLS, TW, TH, LH = 5, 220, 135, 85
    n_rows = math.ceil(len(frames) / N_COLS)
    grid = Image.new("RGB", (N_COLS * TW, n_rows * (TH + LH)), (14, 14, 26))
    draw = ImageDraw.Draw(grid)
    for pos, fr in enumerate(frames):
        res = frame_results[pos]
        cap = captions[pos] if pos < len(captions) else ""
        col, row = pos % N_COLS, pos // N_COLS
        x, y = col * TW, row * (TH + LH)
        thumb = fr["image"].resize((TW, TH), Image.LANCZOS)
        grid.paste(thumb, (x, y))
        anom = res["is_anomaly"]
        draw.rectangle([(x, y + TH), (x + TW - 1, y + TH + LH - 1)],
                       fill=(120, 10, 10) if anom else (10, 60, 20))
        draw.text((x + 4, y + TH + 3),
                  "ANOMALY" if anom else "Normal",
                  fill=(255, 80, 80) if anom else (80, 255, 100))
        draw.text((x + 4, y + TH + 18),
                  f"t={res['time_sec']:.1f}s  D={res['difference']:.2f}",
                  fill=(200, 200, 200))
        sc_str = (", ".join(res.get("scenarios_hit", []))[:32]
                  or res.get("alert", "")[:32])
        draw.text((x + 4, y + TH + 34), sc_str,
                  fill=(255, 200, 50) if anom else (150, 150, 200))
        cap_short = (cap[:52] + "...") if len(cap) > 52 else cap
        draw.text((x + 4, y + TH + 52), cap_short, fill=(180, 180, 255))
        draw.rectangle([(x, y), (x + TW - 1, y + TH + LH - 1)],
                       outline=(210, 25, 25) if anom else (25, 155, 55), width=3)
    grid.save(GRID_PATH, quality=90)
    print(f"  Grid saved       -> {GRID_PATH}")


def save_timeline(frame_results, thr_d):
    times = [r["time_sec"]   for r in frame_results]
    diffs = [r["difference"] for r in frame_results]
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")
    ax.plot(times, diffs, color="#4fc3f7", lw=1.8, zorder=2)
    ax.fill_between(times, diffs, alpha=0.15, color="#4fc3f7")
    ax.axhline(thr_d, color="#ffd700", linestyle="--", lw=2, zorder=4)

    # Plot each scenario in its own colour for uncorrelated visualisation
    scenario_colors = {
        "Traffic Accident":          "#e72f2f",
        "Stop Line Violation":       "#ff8c00",
        "Red Light Violation":       "#ff4500",
        "Ambulance Stuck in Traffic":"#9b59b6",
        "Shoplifting / Theft":       "#e74c3c",
        "Suspicious Activity":       "#f39c12",
    }
    plotted_labels: set[str] = set()
    for r in frame_results:
        for sc_name in r.get("scenarios_hit", []):
            color = scenario_colors.get(sc_name, "#e72f2f")
            label = sc_name if sc_name not in plotted_labels else ""
            ax.scatter(r["time_sec"], r["difference"],
                       color=color, s=80, zorder=5, label=label)
            plotted_labels.add(sc_name)
            ax.annotate(sc_name[:20],
                        (r["time_sec"], r["difference"]),
                        textcoords="offset points", xytext=(4, 6),
                        fontsize=5.5, color=color)

    ax.set_title("Uncorrelated Anomaly Detection — Frame Difference Over Time",
                 color="white", fontsize=13, pad=12)
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("Frame Delta", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#ffffff22")
    ax.grid(axis="y", color="#ffffff1a", linestyle="--", lw=0.8)
    handles = [
        mpatches.Patch(color="#4fc3f7", label="Frame diff"),
        plt.Line2D([0], [0], color="#ffd700", linestyle="--", label="Pixel threshold"),
    ] + [
        mpatches.Patch(color=c, label=n)
        for n, c in scenario_colors.items()
        if n in plotted_labels
    ]
    ax.legend(handles=handles, facecolor="#0f0f1a", labelcolor="white",
              edgecolor="#ffffff33", fontsize=8)
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Timeline saved   -> {CHART_PATH}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run():
    print(BOLD + "=" * 72 + RESET)
    print(BOLD + "  VIDEO ANOMALY DETECTION — ENHANCED (6 Scenarios + IF)" + RESET)
    print(BOLD + "=" * 72 + RESET)
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Video   : {VIDEO_PATH}")
    print(f"  Scenarios: {', '.join(s['name'] for s in SCENARIOS)}")
    print("=" * 72 + "\n")

    clean_dirs()

    # ── 1. Extract frames ─────────────────────────────────────────────────────
    print(BOLD + "[1/6] FRAME EXTRACTION" + RESET)
    frames = extract_frames(VIDEO_PATH)
    if not frames:
        sys.exit(1)
    print(f"  Extracted {len(frames)} frames at {EXTRACT_FPS} FPS\n")

    # ── 2. Load BLIP ──────────────────────────────────────────────────────────
    print(BOLD + "[2/6] LOADING BLIP MODEL" + RESET)
    processor, model, device = load_blip()
    if not processor:
        print(f"  {YELLOW}BLIP unavailable — captions will be empty.{RESET}\n")

    # ── 3. Pixel-level anomaly scores ─────────────────────────────────────────
    print(BOLD + "[3/6] PIXEL-LEVEL ANOMALY SCORING" + RESET)
    diffs, ssims, thr_d, thr_s = compute_anomaly_scores(frames)
    pixel_flags = pixel_anomaly_flags(frames, diffs, ssims, thr_d, thr_s)
    print(f"  Pixel diff threshold : {thr_d:.2f}")
    print(f"  SSIM threshold       : {thr_s:.4f}\n")

    # ── 4. First-pass: gather features for Isolation Forest training ──────────
    print(BOLD + "[4/6] FIRST-PASS VQA FOR ISOLATION FOREST TRAINING" + RESET)
    first_pass: list[dict] = []
    for i, fr in enumerate(frames):
        img = fr["image"]
        vqa_answers: dict[str, str] = {}
        if processor:
            for sc in SCENARIOS:
                vqa_answers[sc["name"]] = blip_vqa(
                    img, processor, model, device, sc["vqa_q"])
            caption = blip_caption(img, processor, model, device,
                                   prompt=CAPTION_PROMPT)
        else:
            for sc in SCENARIOS:
                vqa_answers[sc["name"]] = "no"
            caption = ""
        first_pass.append({
            "diff":        diffs[i],
            "ssim":        ssims[i],
            "caption":     caption,
            "vqa_answers": vqa_answers,
            "time_norm":   fr["time_sec"] / max(len(frames), 1),
        })
        sys.stdout.write(f"\r  First-pass frame {i + 1}/{len(frames)} …")
        sys.stdout.flush()
    print("\n  Training Isolation Forests …")
    forests = train_isolation_forests(first_pass)
    if forests:
        print(f"  ✓ {len(forests)} independent IF models trained "
              f"(one per scenario).\n")
    else:
        print(f"  {YELLOW}scikit-learn not available — IF scoring skipped.{RESET}\n")

    # ── 5. Main analysis loop ──────────────────────────────────────────────────
    print(BOLD + "[5/6] FULL ANALYSIS + SCENARIO DETECTION + ALERTS" + RESET)
    print(f"\n  {'#':<4} {'Time':>6}  {'Pixel':^7}  "
          f"{'Scenarios Detected':<45}  Description")
    print("  " + "─" * 100)

    frame_results: list[dict] = []
    per_frame_captions: list[str] = []

    for i, fr in enumerate(frames):
        res = analyse_frame_full(
            img            = fr["image"],
            processor      = processor,
            model          = model,
            device         = device,
            is_pixel_anomaly = pixel_flags[i],
            frame_idx      = fr["frame_idx"],
            time_sec       = fr["time_sec"],
            diff           = diffs[i],
            ssim_val       = ssims[i],
            forests        = forests,
            n_total_frames = len(frames),
        )
        save_frame_image(fr, res)
        per_frame_captions.append(res["caption"])
        frame_results.append(res)

        anom  = res["is_anomaly"]
        sc    = RED if anom else GREEN
        st    = "ANOMALY" if anom else "Normal "
        sc_str = (", ".join(res["scenarios_hit"]) or "—")[:45]
        print(f"  {i + 1:<4} {res['time_sec']:>5.1f}s  "
              f"{sc}{BOLD}{st}{RESET}  "
              f"{sc_str:<45}  "
              f"{DIM}{res['caption'][:55]}{RESET}")

    n_anom = sum(1 for r in frame_results if r["is_anomaly"])
    print("  " + "─" * 100)
    print(f"\n  Anomalous frames : {RED}{n_anom}{RESET} / {len(frame_results)}")
    print(f"  Total alerts     : {RED}{len(_alert_log)}{RESET}\n")

    # ── 6. Save outputs ────────────────────────────────────────────────────────
    print(BOLD + "[6/6] SAVING OUTPUTS" + RESET)
    save_grid(frames, frame_results, per_frame_captions)
    save_timeline(frame_results, thr_d)

    # alerts.json
    with open(ALERTS_PATH, "w", encoding="utf-8") as f:
        json.dump(_alert_log, f, indent=2, ensure_ascii=False)
    print(f"  Alerts saved     -> {ALERTS_PATH}  ({len(_alert_log)} alerts)")

    # Scenario summary
    print(f"\n  {'─'*55}")
    print(f"  ALERT SUMMARY BY SCENARIO")
    print(f"  {'─'*55}")
    for sc in SCENARIOS:
        hits = [a for a in _alert_log if a["scenario"] == sc["name"]]
        if hits:
            times = ", ".join(f"{h['video_time_s']:.1f}s" for h in hits[:5])
            extra = f" +{len(hits)-5} more" if len(hits) > 5 else ""
            color = RED if sc["severity"] in ("CRITICAL", "HIGH") else YELLOW
            print(f"  {color}{sc['name']:<35}{RESET}  "
                  f"{len(hits):>3} alert(s)  @ {times}{extra}")
        else:
            print(f"  {DIM}{sc['name']:<35}  0 alerts{RESET}")

    # Full report JSON
    report = {
        "created_at":  datetime.now().isoformat(),
        "video_path":  VIDEO_PATH,
        "extract_fps": EXTRACT_FPS,
        "max_frames":  MAX_FRAMES,
        "scenarios":   [s["name"] for s in SCENARIOS],
        "frames":      frame_results,
        "alerts":      _alert_log,
        "summary": {
            "total_frames":     len(frame_results),
            "anomalous_frames": n_anom,
            "total_alerts":     len(_alert_log),
            "by_scenario": {
                sc["name"]: len([a for a in _alert_log if a["scenario"] == sc["name"]])
                for sc in SCENARIOS
            },
        },
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Report saved     -> {REPORT_PATH}")

    print(f"\n{BOLD}{'=' * 72}{RESET}")
    print(f"  Total frames  : {BOLD}{len(frame_results)}{RESET}")
    print(f"  Anomalous     : {RED}{BOLD}{n_anom}{RESET}")
    print(f"  Alerts fired  : {RED}{BOLD}{len(_alert_log)}{RESET}")
    print(f"{BOLD}{'=' * 72}{RESET}")
    print(f"\n  {GREEN}Done! Output -> {os.path.abspath(OUTPUT_DIR)}{RESET}\n")


if __name__ == "__main__":
    run()