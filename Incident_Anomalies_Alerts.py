"""
Video Anomaly Detection & Captioning
"""
# Extracts frames at 1 FPS (configurable), detects anomalies via frame diff + SSIM
# Captions every frame with BLIP, using anomaly-focused prompts ("what happened?" for anomalies)
# Maps captions to alerts via expanded prompt engineering
# Terminal output: RED = ANOMALY, GREEN = Normal
# Saves frames/, collisions/, report.json, grid, timeline chart; clears output folders on each run

import subprocess, sys, os
import json, math, shutil, warnings
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

try:
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False

# Configuration 
VIDEO_PATH  = r"C:\Users\Gustavo\Desktop\DrApurbasTasks\IncidentAnomaliesAlerts\VideosFolder\input_video.mp4"
OUTPUT_DIR  = "output_video"
FRAMES_DIR  = os.path.join(OUTPUT_DIR, "frames")
ANOMALY_DIR = os.path.join(OUTPUT_DIR, "collisions")
REPORT_PATH = os.path.join(OUTPUT_DIR, "report.json")
CHART_PATH  = os.path.join(OUTPUT_DIR, "anomaly_timeline.png")
GRID_PATH   = os.path.join(OUTPUT_DIR, "all_frames_grid.jpg")

EXTRACT_FPS = 0.5 #!!!!!!!!!!!!!!!!!!
MAX_FRAMES  = 60
CHUNK_SIZE  = 10
DIFF_SIZE   = (96, 96)

RED    = "\033[91m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

# Prompts 
CAPTION_PROMPT         = "a traffic camera photo of"
ANOMALY_CAPTION_PROMPT = "a traffic camera showing an accident where"

VQA_ACCIDENT   = "is there a vehicle collision or traffic accident in this image?"
VQA_FIRE       = "is there fire or smoke visible in this image?"
VQA_FALL       = "has a person fallen down or been knocked over in this image?"
VQA_WRONG_SIDE = "is a vehicle on the wrong side of the road in this image?"
VQA_SEVERITY   = "how severe is the incident in this image? answer: minor, moderate, or severe"

# Respostas válidas para severidade
VALID_SEVERITIES = {"minor", "moderate", "severe"}

#  Alert rules (order = priority, first match wins) 
ALERT_RULES = [
    # CRITICAL
    (["crash", "collision", "collide", "accident", "wreck", "smash", "impact",
      "overturned", "overturn", "flipped", "rolled over", "pile-up", "pileup",
      "rear-end", "head-on", "sideswipe", "t-bone", "spun out", "skidded",
      "slammed", "rammed", "struck", "hit by", "ran into", "drove into",
      "ran a red", "ran red light", "wrong way", "wrong side"],
     "COLLISION / CRASH",     "CRITICAL"),

    (["fire", "smoke", "burning", "flame", "blaze", "engulfed", "on fire"],
     "FIRE / SMOKE",          "CRITICAL"),

    # HIGH
    (["fall", "fallen", "falling", "knocked down", "lying on road",
      "lying in street", "lying on ground", "hit pedestrian", "struck pedestrian",
      "pedestrian down", "person down", "run over", "runover"],
     "PERSON DOWN / FALL",    "HIGH"),

    (["fight", "fighting", "violence", "attack", "assault", "brawl"],
     "VIOLENCE",              "HIGH"),

    (["debris", "obstacle", "object on road", "tire on road", "car parts",
      "broken glass", "scattered", "blocking the road", "blocking road"],
     "ROAD DEBRIS / HAZARD",  "HIGH"),

    # MEDIUM
    (["speeding", "racing", "chase", "high speed", "reckless"],
     "RECKLESS DRIVING",      "MEDIUM"),

    (["skid", "swerve", "swerving", "sliding", "lost control", "hydroplane"],
     "LOSS OF CONTROL",       "MEDIUM"),

    (["stalled", "broken down", "stopped in lane", "blocking lane",
      "disabled vehicle", "flat tire"],
     "STALLED VEHICLE",       "MEDIUM"),

    # LOW / INFO
    (["jaywalking", "crossing", "pedestrian crossing", "person crossing",
      "person walking"],
     "PEDESTRIAN",            "LOW"),

    (["traffic", "road", "street", "highway", "car", "vehicle", "truck",
      "bus", "motorcycle", "parking", "parked", "intersection", "driving",
      "lane", "signal", "light"],
     "TRAFFIC SCENE",         "INFO"),
]

# Matches keywords in the caption to return an alert label and severity
# or "NO ALERT" if none found
def map_caption_to_alert(caption: str):
    cap = caption.lower()
    for keywords, label, severity in ALERT_RULES:
        if any(kw in cap for kw in keywords):
            return label, severity
    return "NO ALERT", "NORMAL"


#BLIP helpers
# Loads the BLIP model and processor, selects GPU/CPU
# or returns None if BLIP is unavailable
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


# Generates a caption for an image using the BLIP model
# optionally considering a prompt, and returns the caption text
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

# Uses the BLIP model to answer a question about an image and returns the response
# or "n/a" if an error occurs
def blip_vqa(img, processor, model, device, question: str) -> str:
    try:
        with torch.no_grad():
            inputs = processor(images=img, text=question,
                               return_tensors="pt").to(device)
            out = model.generate(**inputs, max_new_tokens=8)
        return processor.decode(out[0], skip_special_tokens=True).strip().lower()
    except Exception:
        return "n/a"


# Extracts 'minor', 'moderate'
# or 'severe' from a VQA response, returning 'n/a' if invalid or the question is echoed
def parse_severity(raw: str) -> str:
    """
        Extracts only 'minor', 'moderate', or 'severe' from the VQA response.
        Returns 'n/a' if the model echoed the question or responded with something invalid.
    """
    if not raw or raw == "n/a":
        return "n/a"
    first_word = raw.strip().lower().split()[0]
    return first_word if first_word in VALID_SEVERITIES else "n/a"


# Analyzes a single video frame by answering core VQA questions, generating a caption,
# and determining an alert label and severity
# It adapts based on anomaly detection, accident/fire conditions, 
# and overrides the results based on critical findings like "fall" or "wrong-way"
def analyse_frame(img, processor, model, device, is_anomaly: bool):
    """
    Full VQA + captioning pipeline for one frame.
    Returns (caption, alert_label, severity, vqa_dict)
    """
    vqa = {}

    # Step 1: core yes/no questions
    vqa["accident"] = blip_vqa(img, processor, model, device, VQA_ACCIDENT)
    vqa["fire"]     = blip_vqa(img, processor, model, device, VQA_FIRE)

    accident_confirmed = vqa["accident"] in ("yes", "yeah", "true")
    fire_confirmed     = vqa["fire"]     in ("yes", "yeah", "true")

    # Step 2: choose caption prompt based on anomaly signal
    if is_anomaly or accident_confirmed or fire_confirmed:
        caption = blip_caption(img, processor, model, device,
                               prompt=ANOMALY_CAPTION_PROMPT)
        vqa["fall"]      = blip_vqa(img, processor, model, device, VQA_FALL)
        vqa["wrong_way"] = blip_vqa(img, processor, model, device, VQA_WRONG_SIDE)
        vqa["severity"]  = blip_vqa(img, processor, model, device, VQA_SEVERITY)
    else:
        caption = blip_caption(img, processor, model, device,
                               prompt=CAPTION_PROMPT)
        vqa["fall"]      = "n/a"
        vqa["wrong_way"] = "n/a"
        vqa["severity"]  = "n/a"

    # Step 3: keyword mapping
    alert_label, severity = map_caption_to_alert(caption)

    # Step 4: VQA overrides
    if accident_confirmed and severity != "CRITICAL":
        alert_label, severity = "COLLISION / CRASH", "CRITICAL"
    if fire_confirmed and severity != "CRITICAL":
        alert_label, severity = "FIRE / SMOKE", "CRITICAL"
    if vqa.get("fall", "") in ("yes", "yeah") and severity not in ("CRITICAL", "HIGH"):
        alert_label, severity = "PERSON DOWN / FALL", "HIGH"
    if vqa.get("wrong_way", "") in ("yes", "yeah") and severity != "CRITICAL":
        alert_label, severity = "WRONG-WAY VEHICLE", "CRITICAL"

    # Step 5: append severity tag — apenas se a resposta for válida
    if is_anomaly or severity in ("CRITICAL", "HIGH"):
        sev_str = parse_severity(vqa.get("severity", "n/a"))
        if sev_str != "n/a":
            caption = f"{caption}  [severity: {sev_str}]"

    return caption, alert_label, severity, vqa


# Directory helpers 
def clean_dirs():
    print(f"{YELLOW}[SETUP] Clearing output directories...{RESET}")
    for d in (FRAMES_DIR, ANOMALY_DIR):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"  frames/     -> {os.path.abspath(FRAMES_DIR)}")
    print(f"  collisions/ -> {os.path.abspath(ANOMALY_DIR)}\n")


# Extracts frames from a video at a specified frame rate (EXTRACT_FPS)
# and returns them as a list of images with index and timestamp
# Stops after reaching a maximum frame count (MAX_FRAMES) or the end of the video
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


# Computes anomaly scores for frames
# by calculating the difference and SSIM (Structural Similarity) between consecutive frames
# Returns the differences, SSIM values, and thresholds based on the mean 
# and standard deviation of these scores
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

# Labels each frame as an anomaly or normal based on the difference
# and SSIM scores compared to threshold values
# Flags anomalies when the difference is high or SSIM is low, 
# and resets if difference is low and SSIM is high
def label_frames(frames, diffs, ssims, thr_d, thr_s):
    low_d, high_s = thr_d * 0.50, thr_s * 1.50
    anom_active = False
    results = []
    for i, fr in enumerate(frames):
        d, s = diffs[i], ssims[i]
        if (d > thr_d) or (s < thr_s):
            anom_active = True
        elif d < low_d and s > high_s:
            anom_active = False
        results.append({
            "frame_idx":  int(fr["frame_idx"]),
            "time_sec":   float(fr["time_sec"]),
            "difference": float(d),
            "ssim":       float(s),
            "is_anomaly": bool(anom_active),
        })
    return results


# Saves a frame image with an "ANOMALY" label if flagged,
# storing it in the appropriate directory
def save_frame_image(frame_data, res, is_anomaly, caption="", alert=""):
    fname = f"frame_{res['frame_idx']:05d}_t{res['time_sec']:.2f}s.jpg"
    frame_data["image"].save(os.path.join(FRAMES_DIR, fname), quality=92)
    if is_anomaly:
        img_copy = frame_data["image"].copy()
        draw = ImageDraw.Draw(img_copy)
        w, h = img_copy.size
        draw.rectangle([(0, h - 52), (w, h)], fill=(160, 10, 10))
        draw.text((6, h - 50), f"ANOMALY: {alert}", fill=(255, 220, 50))
        draw.text((6, h - 30), caption[:90],        fill=(255, 255, 255))
        img_copy.save(os.path.join(ANOMALY_DIR, fname), quality=92)
    return fname


# Creates and saves a grid of resized frame thumbnails with anomaly labels 
# and additional information (time, difference, caption)
# Each frame is displayed with a color-coded label indicating whether it's an anomaly or normal
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
        alert_str = res.get("alert", "")[:32]
        draw.text((x + 4, y + TH + 34), alert_str,
                  fill=(255, 200, 50) if anom else (150, 150, 200))
        cap_short = (cap[:52] + "...") if len(cap) > 52 else cap
        draw.text((x + 4, y + TH + 52), cap_short, fill=(180, 180, 255))
        draw.rectangle([(x, y), (x + TW - 1, y + TH + LH - 1)],
                       outline=(210, 25, 25) if anom else (25, 155, 55), width=3)
    grid.save(GRID_PATH, quality=90)
    print(f"  Grid saved       -> {GRID_PATH}")


# Plots and saves a timeline of frame differences over time,
# marking anomalies with red dots and displaying the threshold line
# Includes labels for normal and anomalous events, and saves the chart as an image
def save_timeline(frame_results, thr_d):
    times = [r["time_sec"]   for r in frame_results]
    diffs = [r["difference"] for r in frame_results]
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#1a1a2e")
    ax.plot(times, diffs, color="#4fc3f7", lw=1.8, zorder=2)
    ax.fill_between(times, diffs, alpha=0.15, color="#4fc3f7")
    ax.axhline(thr_d, color="#ffd700", linestyle="--", lw=2, zorder=4)
    for r in frame_results:
        if r["is_anomaly"]:
            ax.scatter(r["time_sec"], r["difference"],
                       color="#e72f2f", s=70, zorder=5)
            ax.annotate(r.get("alert", "")[:20],
                        (r["time_sec"], r["difference"]),
                        textcoords="offset points", xytext=(4, 6),
                        fontsize=6, color="#ff8888")
    ax.set_title("Anomaly Detection - Frame Difference Over Time",
                 color="white", fontsize=13, pad=12)
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel("Frame Delta", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#ffffff22")
    ax.grid(axis="y", color="#ffffff1a", linestyle="--", lw=0.8)
    handles = [
        mpatches.Patch(color="#4fc3f7", label="Normal"),
        mpatches.Patch(color="#e84545", label="Anomalous"),
        plt.Line2D([0], [0], color="#ffd700", linestyle="--", label="Threshold"),
    ]
    ax.legend(handles=handles, facecolor="#0f0f1a", labelcolor="white",
              edgecolor="#ffffff33", fontsize=9)
    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Timeline saved   -> {CHART_PATH}")


# Main
def run():
    print(BOLD + "=" * 72 + RESET)
    print(BOLD + "  VIDEO ANOMALY DETECTION & CAPTIONING  (BLIP-large + VQA chain)" + RESET)
    print(BOLD + "=" * 72 + RESET)
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Video   : {VIDEO_PATH}")
    print(f"  Output  : {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 72 + "\n")

    clean_dirs()

    # 1. Extract frames
    print(BOLD + "[1/5] FRAME EXTRACTION" + RESET)
    frames = extract_frames(VIDEO_PATH)
    if not frames:
        sys.exit(1)
    print(f"  Extracted {len(frames)} frames at {EXTRACT_FPS} FPS\n")

    # 2. Load BLIP
    print(BOLD + "[2/5] LOADING BLIP MODEL" + RESET)
    processor, model, device = load_blip()
    if not processor:
        print(f"  {YELLOW}BLIP unavailable - captions will be empty.{RESET}\n")

    # 3. Anomaly detection + per-frame captioning
    print(BOLD + "[3/5] ANOMALY DETECTION + PER-FRAME VQA & CAPTIONING" + RESET)
    diffs, ssims, thr_d, thr_s = compute_anomaly_scores(frames)
    frame_results = label_frames(frames, diffs, ssims, thr_d, thr_s)

    hdr = (f"  {'#':<4} {'Time':>6}  {'Status':<9}  {'Alert':<26}  Description")
    print(hdr)
    print("  " + "-" * 85)

    per_frame_captions = []

    for i, res in enumerate(frame_results):
        img     = frames[i]["image"]
        is_anom = res["is_anomaly"]

        if processor:
            caption, alert_label, severity, vqa = analyse_frame(
                img, processor, model, device, is_anom)
        else:
            caption, alert_label, severity, vqa = "(no model)", "NO ALERT", "NORMAL", {}

        per_frame_captions.append(caption)
        save_frame_image(frames[i], res, is_anom, caption, alert_label)

        res["caption"]  = caption
        res["alert"]    = alert_label
        res["severity"] = severity
        res["vqa"]      = vqa

        sc = RED   if is_anom else GREEN
        st = "ANOMALY" if is_anom else "Normal "
        ac = RED    if severity in ("CRITICAL", "HIGH") else \
             YELLOW if severity == "MEDIUM"             else DIM

        print(f"  {i + 1:<4} {res['time_sec']:>5.1f}s  "
              f"{sc}{BOLD}{st}{RESET}   "
              f"{ac}{alert_label:<26}{RESET}  "
              f"{DIM}{caption}{RESET}")

    n_anom = sum(1 for r in frame_results if r["is_anomaly"])
    print("  " + "-" * 85)
    print(f"\n  Anomalous frames : {RED}{n_anom}{RESET} / {len(frame_results)}")
    print(f"  Frames saved     -> {FRAMES_DIR}")
    print(f"  Anomalies saved  -> {ANOMALY_DIR}\n")

    # 4. Chunk-level summary
    print(BOLD + "[4/5] CHUNK-LEVEL ALERT SUMMARY" + RESET)
    chunks = [frame_results[i:i + CHUNK_SIZE]
              for i in range(0, len(frame_results), CHUNK_SIZE)]
    chunk_results_data = []
    severity_order = {"CRITICAL": 5, "HIGH": 4, "MEDIUM": 3, "LOW": 2, "INFO": 1, "NORMAL": 0}

    print(f"\n  {'Chunk':<6}  {'Time Range':<14}  {'Anomaly Frames':>14}  "
          f"{'Top Alert':<28}  What Happened")
    print("  " + "-" * 100)

    for ci, chunk in enumerate(chunks):
        t_start  = chunk[0]["time_sec"]
        t_end    = chunk[-1]["time_sec"]
        n_anom_c = sum(1 for r in chunk if r["is_anomaly"])
        best     = max(chunk,
                       key=lambda r: severity_order.get(r.get("severity", "NORMAL"), 0))
        top_alert   = best.get("alert",   "NO ALERT")
        top_caption = best.get("caption", "")
        top_sev     = best.get("severity", "NORMAL")

        color = RED    if top_sev in ("CRITICAL", "HIGH") else \
                YELLOW if top_sev == "MEDIUM"             else GREEN

        print(f"  {ci + 1:<6}  {t_start:.1f}s-{t_end:.1f}s{'':<5}  "
              f"{n_anom_c:>14}  "
              f"{color}{top_alert:<28}{RESET}  "
              f"{DIM}{top_caption[:65]}{RESET}")

        chunk_results_data.append({
            "chunk":          ci + 1,
            "time_range":     f"{t_start:.1f}s-{t_end:.1f}s",
            "anomaly_frames": n_anom_c,
            "top_alert":      top_alert,
            "top_severity":   top_sev,
            "what_happened":  top_caption,
        })

    print("  " + "-" * 100 + "\n")

    # 5. Save outputs
    print(BOLD + "[5/5] SAVING OUTPUTS" + RESET)
    save_grid(frames, frame_results, per_frame_captions)
    save_timeline(frame_results, thr_d)

    report = {
        "created_at":  datetime.now().isoformat(),
        "video_path":  VIDEO_PATH,
        "extract_fps": EXTRACT_FPS,
        "max_frames":  MAX_FRAMES,
        "frames":      frame_results,
        "chunks":      chunk_results_data,
        "summary": {
            "total_frames":       len(frame_results),
            "anomalous_frames":   n_anom,
            "first_anomaly_time": next(
                (r["time_sec"] for r in frame_results if r["is_anomaly"]), None),
            "last_anomaly_time":  next(
                (r["time_sec"] for r in reversed(frame_results) if r["is_anomaly"]), None),
        },
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Report saved     -> {REPORT_PATH}")

    print(f"\n{BOLD}{'=' * 72}{RESET}")
    print(f"  Total frames  : {BOLD}{len(frame_results)}{RESET}")
    print(f"  Anomalous     : {RED}{BOLD}{n_anom}{RESET}")
    print(f"  Normal        : {GREEN}{BOLD}{len(frame_results) - n_anom}{RESET}")
    print(f"{BOLD}{'=' * 72}{RESET}")
    print(f"\n  {GREEN}Done! Output -> {os.path.abspath(OUTPUT_DIR)}{RESET}\n")


if __name__ == "__main__":
    run()