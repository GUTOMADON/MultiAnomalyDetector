"""
╔══════════════════════════════════════════════════════════════════════════╗
║     UNCORRELATED VIDEO ANOMALY DETECTION  ·  ALERT SYSTEM  v3.0        ║
╠══════════════════════════════════════════════════════════════════════════╣
║  6 independent anomaly classes (uncorrelated detection):               ║
║                                                                         ║
║   1 · TRAFFIC ACCIDENT          CRITICAL                               ║
║   2 · STOP LINE VIOLATION       HIGH                                   ║
║   3 · RED LIGHT VIOLATION       HIGH                                   ║
║   4 · AMBULANCE STUCK IN TRAFFIC  CRITICAL                             ║
║   5 · SHOPLIFTING / THEFT       HIGH                                   ║
║   6 · SUSPICIOUS ACTIVITY       MEDIUM                                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Pipeline:                                                              ║
║   ① Extract frames at configurable FPS                                 ║
║   ② Score each frame (pixel-diff + SSIM) → visual anomaly signal      ║
║   ③ BLIP caption every frame (context-aware prompt)                   ║
║   ④ Per-class VQA + keyword matching (fully uncorrelated)              ║
║   ⑤ Deduplicate via per-class cooldown → timestamped alerts           ║
║   ⑥ Save: frames/, anomaly_frames/, alerts.json, report.json,         ║
║            timeline.png, grid.jpg                                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║  Usage:                                                                 ║
║   python anomaly_detector.py                    (default config)       ║
║   python anomaly_detector.py --video clip.mp4                          ║
║   python anomaly_detector.py --fps 2 --sigma 1.5 --cooldown 5         ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse, json, math, shutil, sys, warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim

warnings.filterwarnings("ignore")

# ── optional ML deps ─────────────────────────────────────────────────────────
try:
    import torch
    from transformers import BlipProcessor, BlipForConditionalGeneration
    _BLIP_AVAILABLE = True
except ImportError:
    _BLIP_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
#  ANSI COLOURS
# ═══════════════════════════════════════════════════════════════════════════════

class C:
    RED  = "\033[91m"; GRN = "\033[92m"; YLW = "\033[93m"
    CYN  = "\033[96m"; MGN = "\033[95m"; BLU = "\033[94m"
    RST  = "\033[0m";  BLD = "\033[1m";  DIM = "\033[2m"

_SEV_CLR = {"CRITICAL": C.RED, "HIGH": C.YLW, "MEDIUM": C.CYN,
            "LOW": C.GRN, "NORMAL": C.DIM}

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    video_path   : str   = "input_video.mp4"
    output_dir   : str   = "output_anomaly"
    extract_fps  : float = 1.0
    max_frames   : int   = 200
    # Visual scoring – lower sigma = more sensitive
    diff_sigma   : float = 1.5
    ssim_sigma   : float = 1.5
    # How many seconds must pass before re-alerting the SAME class
    cooldown_sec : float = 5.0
    # BLIP – set to blip-image-captioning-base for speed on CPU
    blip_model   : str   = "Salesforce/blip-image-captioning-large"
    score_size   : tuple = (96, 96)
    # Minimum BLIP caption token probability to trust a "yes" VQA answer
    vqa_min_tokens : int = 1

    def patch(self, args: argparse.Namespace) -> "Config":
        if args.video:      self.video_path  = args.video
        if args.output:     self.output_dir  = args.output
        if args.fps:        self.extract_fps = args.fps
        if args.max_frames: self.max_frames  = args.max_frames
        if args.sigma:      self.diff_sigma  = args.sigma; self.ssim_sigma = args.sigma
        if args.cooldown:   self.cooldown_sec= args.cooldown
        return self

CFG = Config()

def _paths() -> dict[str, Path]:
    b = Path(CFG.output_dir)
    return {
        "frames"  : b / "frames",
        "anomaly" : b / "anomaly_frames",
        "alerts"  : b / "alerts.json",
        "report"  : b / "report.json",
        "chart"   : b / "timeline.png",
        "grid"    : b / "grid.jpg",
    }

# ═══════════════════════════════════════════════════════════════════════════════
#  ANOMALY CLASS REGISTRY  (6 uncorrelated classes)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnomalyClass:
    id       : int
    name     : str
    severity : str    # CRITICAL | HIGH | MEDIUM | LOW
    rgb      : tuple  # overlay colour

    # Detection signals ────────────────────────────────────────────────────────
    # VQA questions (asked independently per frame)
    vqa_primary   : str        # main yes/no question
    vqa_secondary : str        # backup question for low-confidence cases

    # Caption keywords → immediate confirmation when VQA also says yes
    keywords : list[str]

    # Caption keywords so strong they fire WITHOUT VQA confirmation
    hard_keywords : list[str]


CLASSES: list[AnomalyClass] = [

    AnomalyClass(
        id=1, name="TRAFFIC ACCIDENT", severity="CRITICAL",
        rgb=(220, 30, 30),
        vqa_primary   = "Is there a vehicle collision or traffic accident happening in this image?",
        vqa_secondary = "Are there crashed, damaged or colliding vehicles in this image?",
        keywords      = ["crash", "collision", "collide", "accident", "wreck",
                         "smash", "impact", "overturned", "flipped", "pile-up",
                         "pileup", "rear-end", "head-on", "sideswipe", "t-bone",
                         "rammed", "struck vehicle", "ran into", "drove into"],
        hard_keywords = ["car accident", "vehicle accident", "traffic accident",
                         "car crash", "vehicle crash"],
    ),

    AnomalyClass(
        id=2, name="STOP LINE VIOLATION", severity="HIGH",
        rgb=(230, 100, 20),
        vqa_primary   = "Is a vehicle past or crossing the stop line at an intersection?",
        vqa_secondary = "Has a vehicle entered the intersection before it was allowed?",
        keywords      = ["stop line", "crossing the line", "past the line",
                         "over the line", "past stop", "entering junction",
                         "jumped the line", "blocking intersection"],
        hard_keywords = ["stop line violation", "crossed the stop line",
                         "past the stop line"],
    ),

    AnomalyClass(
        id=3, name="RED LIGHT VIOLATION", severity="HIGH",
        rgb=(200, 50, 200),
        vqa_primary   = "Is a vehicle running a red light or passing through a red traffic signal?",
        vqa_secondary = "Is there a vehicle going through a red traffic light?",
        keywords      = ["red light", "ran a red", "running a red",
                         "running the red", "ignoring signal",
                         "traffic light violation", "through red", "beat the light"],
        hard_keywords = ["ran the red light", "running red light",
                         "red light violation"],
    ),

    AnomalyClass(
        id=4, name="AMBULANCE STUCK IN TRAFFIC", severity="CRITICAL",
        rgb=(245, 195, 0),
        vqa_primary   = "Is there an ambulance or emergency vehicle blocked by traffic and unable to pass?",
        vqa_secondary = "Is an ambulance stuck behind other vehicles?",
        keywords      = ["ambulance", "fire truck", "emergency vehicle",
                         "stuck in traffic", "blocked ambulance",
                         "ems", "paramedic", "cannot pass", "blocked emergency"],
        hard_keywords = ["ambulance stuck", "stuck ambulance",
                         "blocked ambulance", "ambulance trapped"],
    ),

    AnomalyClass(
        id=5, name="SHOPLIFTING / THEFT", severity="HIGH",
        rgb=(180, 0, 180),
        vqa_primary   = "Is someone stealing items or shoplifting in this image?",
        vqa_secondary = "Is a person hiding or concealing merchandise without paying?",
        keywords      = ["stealing", "shoplifting", "theft", "hiding item",
                         "concealing", "taking without paying", "pocketing item",
                         "shoplifter", "concealing merchandise", "shoplifts"],
        hard_keywords = ["shoplifting", "shoplifter caught",
                         "stealing from store", "concealing merchandise"],
    ),

    AnomalyClass(
        id=6, name="SUSPICIOUS ACTIVITY", severity="MEDIUM",
        rgb=(30, 160, 230),
        vqa_primary   = "Is there suspicious or threatening behavior visible in this image?",
        vqa_secondary = "Is someone behaving in an unusual, dangerous, or unauthorized way?",
        keywords      = ["suspicious", "loitering", "lurking", "trespassing",
                         "threatening", "unusual behavior", "fighting",
                         "assault", "vandalism", "breaking in", "unauthorized"],
        hard_keywords = ["suspicious activity", "suspicious person",
                         "suspicious behavior", "unauthorized access"],
    ),
]

BY_ID: dict[int, AnomalyClass] = {c.id: c for c in CLASSES}

# ═══════════════════════════════════════════════════════════════════════════════
#  ALERT DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Alert:
    alert_id      : int
    class_id      : int
    class_name    : str
    severity      : str
    video_time_s  : float   # seconds into the video
    wall_time     : str     # ISO-8601 real-world timestamp
    frame_idx     : int
    caption       : str
    trigger       : str     # "VQA+KEYWORD" | "VQA" | "HARD_KEYWORD" | "KEYWORD_ONLY"
    frame_file    : str = ""

    def as_dict(self) -> dict:
        return asdict(self)

# ═══════════════════════════════════════════════════════════════════════════════
#  BLIP WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class BlipWrapper:
    """
    Wraps BLIP-large for both image captioning and binary VQA.
    Falls back gracefully if torch/transformers are unavailable.
    """

    def __init__(self):
        self.ok        = False
        self.processor = None
        self.model     = None
        self.device    = None

    # ── load ──────────────────────────────────────────────────────────────────
    def load(self) -> None:
        if not _BLIP_AVAILABLE:
            _warn("torch / transformers not installed — running in keyword-only mode.")
            return
        _info(f"Loading {CFG.blip_model} …")
        self.processor = BlipProcessor.from_pretrained(CFG.blip_model, use_fast=False)
        self.model     = BlipForConditionalGeneration.from_pretrained(
            CFG.blip_model, ignore_mismatched_sizes=True
        )
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.ok        = True
        _info(f"BLIP ready on {self.device}.")

    # ── caption ───────────────────────────────────────────────────────────────
    def caption(self, img: Image.Image, prompt: str = "") -> str:
        if not self.ok:
            return ""
        with torch.no_grad():
            kw = dict(images=img, return_tensors="pt")
            if prompt:
                kw["text"] = prompt
            out = self.model.generate(
                **self.processor(**kw).to(self.device),
                max_new_tokens=72, num_beams=5, length_penalty=1.2
            )
        text = self.processor.decode(out[0], skip_special_tokens=True)
        if prompt and text.lower().startswith(prompt.lower()):
            text = text[len(prompt):].strip()
        return text

    # ── VQA ───────────────────────────────────────────────────────────────────
    def vqa(self, img: Image.Image, question: str) -> str:
        """Returns the raw model answer (lowercase string)."""
        if not self.ok:
            return "n/a"
        try:
            with torch.no_grad():
                out = self.model.generate(
                    **self.processor(images=img, text=question,
                                     return_tensors="pt").to(self.device),
                    max_new_tokens=12
                )
            return self.processor.decode(out[0], skip_special_tokens=True).strip().lower()
        except Exception:
            return "n/a"

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def is_yes(ans: str) -> bool:
        if not ans:
            return False
        first = ans.split()[0]
        return first in {"yes", "yeah", "true", "correct", "affirmative", "definitely",
                         "absolutely", "certainly"}

    def vqa_yes(self, img: Image.Image, question: str) -> bool:
        return self.is_yes(self.vqa(img, question))


BLIP = BlipWrapper()

# ═══════════════════════════════════════════════════════════════════════════════
#  ALERT MANAGER  (per-class cooldown deduplication)
# ═══════════════════════════════════════════════════════════════════════════════

class AlertManager:
    def __init__(self):
        self._log       : list[Alert]        = []
        self._last      : dict[int, float]   = {}
        self._next_id   = 1

    def submit(
        self,
        cls     : AnomalyClass,
        t_sec   : float,
        fidx    : int,
        caption : str,
        trigger : str,
    ) -> Optional[Alert]:
        if t_sec - self._last.get(cls.id, -1e9) < CFG.cooldown_sec:
            return None
        self._last[cls.id] = t_sec
        a = Alert(
            alert_id     = self._next_id,
            class_id     = cls.id,
            class_name   = cls.name,
            severity     = cls.severity,
            video_time_s = t_sec,
            wall_time    = datetime.now().isoformat(timespec="seconds"),
            frame_idx    = fidx,
            caption      = caption,
            trigger      = trigger,
        )
        self._log.append(a)
        self._next_id += 1
        return a

    @property
    def alerts(self) -> list[Alert]:
        return self._log

MGR = AlertManager()

# ═══════════════════════════════════════════════════════════════════════════════
#  FRAME EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_frames(path: str) -> list[dict]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"\n{C.RED}[ERROR] Cannot open: {path}{C.RST}\n")
    frames: list[dict] = []
    idx, sec, step = 0, 0.0, 1.0 / CFG.extract_fps
    while len(frames) < CFG.max_frames:
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append({"idx": idx, "sec": round(sec, 3), "img": img})
        idx += 1; sec = round(sec + step, 3)
    cap.release()
    return frames

# ═══════════════════════════════════════════════════════════════════════════════
#  VISUAL ANOMALY SCORING  (pixel-diff + SSIM, GPU-free)
# ═══════════════════════════════════════════════════════════════════════════════

def score_frames(frames: list[dict]) -> tuple[list[float], list[float], float, float]:
    diffs, ssims = [0.0], [1.0]
    prev = np.array(frames[0]["img"].resize(CFG.score_size), dtype=np.float32)
    pg   = cv2.cvtColor(prev.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    for fr in frames[1:]:
        curr = np.array(fr["img"].resize(CFG.score_size), dtype=np.float32)
        cg   = cv2.cvtColor(curr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        diffs.append(float(np.mean(np.abs(curr - prev))))
        ssims.append(float(ssim(cg, pg, data_range=255)))
        prev, pg = curr, cg
    ad, as_ = np.array(diffs), np.array(ssims)
    return (diffs, ssims,
            float(ad.mean()  + CFG.diff_sigma * ad.std()),
            float(as_.mean() - CFG.ssim_sigma * as_.std()))

# ═══════════════════════════════════════════════════════════════════════════════
#  PER-FRAME CLASSIFICATION  (uncorrelated — each class tested independently)
# ═══════════════════════════════════════════════════════════════════════════════

def classify(
    img     : Image.Image,
    caption : str,
    is_vis  : bool,
) -> tuple[Optional[AnomalyClass], str]:
    """
    Returns (matched_class | None, trigger_label).

    Detection strategy per class (uncorrelated — no shared state):
      1. HARD_KEYWORD in caption          → instant fire, no VQA needed
      2. keyword hit  + VQA primary yes   → VQA+KEYWORD
      3. visual flag  + VQA primary yes   → VQA
      4. visual flag  + VQA secondary yes → VQA (fallback)
    """
    cap = caption.lower()

    for cls in CLASSES:
        # ── tier 1: hard keyword (high-confidence phrases) ───────────────────
        if any(hk in cap for hk in cls.hard_keywords):
            return cls, "HARD_KEYWORD"

        kw_hit = any(k in cap for k in cls.keywords)

        # ── tier 2 & 3: VQA when there's any signal ─────────────────────────
        if kw_hit or is_vis:
            if BLIP.vqa_yes(img, cls.vqa_primary):
                trigger = "VQA+KEYWORD" if kw_hit else "VQA"
                return cls, trigger
            # secondary VQA (backup question) — only on visual frames
            if is_vis and BLIP.vqa_yes(img, cls.vqa_secondary):
                return cls, "VQA_SECONDARY"

        # ── tier 4: keyword-only fallback (when BLIP unavailable) ────────────
        if kw_hit and not BLIP.ok:
            return cls, "KEYWORD_ONLY"

    return None, "NONE"

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def process(
    frames : list[dict],
    diffs  : list[float],
    ssims  : list[float],
    thr_d  : float,
    thr_s  : float,
    paths  : dict,
) -> list[dict]:

    low_d, high_s  = thr_d * 0.5, min(thr_s * 1.5, 1.0)
    vis_active     = False
    results        : list[dict] = []

    _table_header()

    for i, fr in enumerate(frames):
        d, s = diffs[i], ssims[i]
        img  = fr["img"]
        sec  = fr["sec"]
        idx  = fr["idx"]

        # ── hysteresis visual flag ────────────────────────────────────────────
        if d > thr_d or s < thr_s:
            vis_active = True
        elif d < low_d and s > high_s:
            vis_active = False

        # ── caption (context-aware prompt) ───────────────────────────────────
        prompt  = ("a surveillance camera showing an incident where"
                   if vis_active else "a surveillance camera showing")
        caption = BLIP.caption(img, prompt) if BLIP.ok else ""

        # ── uncorrelated classification ───────────────────────────────────────
        matched, trigger = classify(img, caption, vis_active)

        # ── alert submission ──────────────────────────────────────────────────
        alert: Optional[Alert] = None
        if matched:
            alert = MGR.submit(matched, sec, idx, caption, trigger)

        # ── save images ───────────────────────────────────────────────────────
        fname = _write_frame(fr, vis_active, matched, caption, alert, paths)
        if alert:
            alert.frame_file = fname

        # ── record ────────────────────────────────────────────────────────────
        rec = {
            "frame_idx"  : idx,
            "time_sec"   : sec,
            "diff"       : round(d, 4),
            "ssim"       : round(s, 4),
            "visual_anom": vis_active,
            "class_id"   : matched.id       if matched else None,
            "class_name" : matched.name     if matched else "NORMAL",
            "severity"   : matched.severity if matched else "NORMAL",
            "trigger"    : trigger,
            "caption"    : caption,
            "alert_id"   : alert.alert_id   if alert else None,
        }
        results.append(rec)
        _table_row(i, rec, alert)

    _rule(92)
    return results

# ═══════════════════════════════════════════════════════════════════════════════
#  I/O HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _write_frame(
    fr    : dict,
    is_v  : bool,
    cls   : Optional[AnomalyClass],
    cap   : str,
    alert : Optional[Alert],
    paths : dict,
) -> str:
    fname = f"frame_{fr['idx']:05d}_t{fr['sec']:.2f}s.jpg"
    fr["img"].save(paths["frames"] / fname, quality=92)

    if cls or is_v:
        ov   = fr["img"].copy().convert("RGB")
        draw = ImageDraw.Draw(ov)
        w, h = ov.size
        col  = cls.rgb if cls else (70, 70, 70)
        draw.rectangle([(0, h - 66), (w, h)], fill=(*col, 215))
        sev   = cls.severity if cls else "VISUAL"
        label = cls.name     if cls else "VISUAL ANOMALY"
        ts    = (f"ALERT #{alert.alert_id:03d}  |  t={fr['sec']:.2f}s  |  {alert.wall_time}"
                 if alert else f"t = {fr['sec']:.2f}s")
        draw.text((6, h-64), f"[{sev}] {label}",  fill=(255,255,255))
        draw.text((6, h-44), ts[:84],               fill=(255,230,70))
        draw.text((6, h-22), cap[:90],              fill=(200,200,255))
        ov.save(paths["anomaly"] / fname, quality=92)

    return fname


def _table_header() -> None:
    print(f"\n  {'#':<5} {'t(s)':>6}  {'Visual':^6}  {'Class':<28}  {'Sev':<8}  Caption")
    _rule(92)

def _table_row(i: int, r: dict, alert: Optional[Alert]) -> None:
    sev   = r["severity"]
    sc    = _SEV_CLR.get(sev, C.DIM)
    vis   = f"{C.RED}ANOM{C.RST}" if r["visual_anom"] else f"{C.GRN}norm{C.RST}"
    tag   = f"  {C.RED}{C.BLD}🚨 ALERT #{alert.alert_id:03d}{C.RST}" if alert else ""
    print(f"  {i+1:<5} {r['time_sec']:>5.1f}s  {vis}    "
          f"{sc}{r['class_name']:<28}{C.RST}  {sc}{sev:<8}{C.RST}  "
          f"{C.DIM}{r['caption'][:50]}{C.RST}{tag}")

def _rule(w: int = 72) -> None:
    print("  " + "─" * w)

def _banner(msg: str) -> None:
    w = 72
    print(f"\n{C.BLD}{'═'*w}{C.RST}")
    print(f"{C.BLD}  {msg}{C.RST}")
    print(f"{C.BLD}{'═'*w}{C.RST}")

def _info(msg: str) -> None:
    print(f"  {C.CYN}{msg}{C.RST}")

def _warn(msg: str) -> None:
    print(f"  {C.YLW}[WARN] {msg}{C.RST}")

# ═══════════════════════════════════════════════════════════════════════════════
#  ALERT LOG
# ═══════════════════════════════════════════════════════════════════════════════

def print_alert_log(alerts: list[Alert], paths: dict) -> None:
    if not alerts:
        print(f"  {C.GRN}No anomalies detected.{C.RST}\n")
        with open(paths["alerts"], "w") as f:
            json.dump([], f, indent=2)
        return

    _rule(84)
    for a in alerts:
        sc = _SEV_CLR.get(a.severity, C.DIM)
        print(f"\n  {sc}{C.BLD}🚨  ALERT #{a.alert_id:03d}  ·  {a.class_name}{C.RST}")
        print(f"     {'Severity':<14}: {sc}{a.severity}{C.RST}")
        print(f"     {'Video time':<14}: {C.YLW}t = {a.video_time_s:.2f}s{C.RST}")
        print(f"     {'Wall clock':<14}: {C.YLW}{a.wall_time}{C.RST}")
        print(f"     {'Frame index':<14}: {a.frame_idx}")
        print(f"     {'Trigger':<14}: {a.trigger}")
        print(f"     {'Caption':<14}: {C.DIM}{a.caption}{C.RST}")
        print(f"     {'Frame file':<14}: {a.frame_file}")
    _rule(84)
    print()

    with open(paths["alerts"], "w", encoding="utf-8") as f:
        json.dump([a.as_dict() for a in alerts], f, indent=2, ensure_ascii=False)
    print(f"  Alerts JSON  →  {paths['alerts']}")

# ═══════════════════════════════════════════════════════════════════════════════
#  VISUAL OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════

_HEX = {1:"#dc1e1e", 2:"#e06414", 3:"#c832c8",
        4:"#f5c800", 5:"#aa00cc", 6:"#1ca0e6"}

def save_timeline(results: list[dict], thr_d: float, paths: dict) -> None:
    times = [r["time_sec"] for r in results]
    diffs = [r["diff"]     for r in results]

    fig, ax = plt.subplots(figsize=(18, 5))
    fig.patch.set_facecolor("#0b0b18"); ax.set_facecolor("#10102a")

    ax.plot(times, diffs, color="#4fc3f7", lw=1.6, zorder=2, label="Frame Δ")
    ax.fill_between(times, diffs, alpha=0.1, color="#4fc3f7")
    ax.axhline(thr_d, color="#ffd700", ls="--", lw=1.8, label="Threshold", zorder=4)

    seen: set[int] = set()
    for r in results:
        cid = r.get("class_id")
        if not cid: continue
        col = _HEX.get(cid, "#fff")
        lbl = BY_ID[cid].name if cid not in seen else "_nolegend_"
        ax.scatter(r["time_sec"], r["diff"], color=col, s=90, zorder=5,
                   edgecolors="white", lw=0.4, label=lbl)
        ax.annotate(BY_ID[cid].name[:18], (r["time_sec"], r["diff"]),
                    textcoords="offset points", xytext=(4,7), fontsize=5.5, color=col)
        seen.add(cid)

    for a in MGR.alerts:
        ax.axvline(a.video_time_s, color="#ff4444", alpha=0.3, lw=1.2, ls=":")

    ax.set_title("Uncorrelated Anomaly Detection  ·  Frame Δ Timeline",
                 color="white", fontsize=13, pad=12)
    ax.set_xlabel("Time (s)", color="white"); ax.set_ylabel("Frame Δ", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#ffffff18")
    ax.grid(axis="y", color="#ffffff14", ls="--", lw=0.7)
    ax.legend(facecolor="#0b0b18", labelcolor="white",
              edgecolor="#ffffff22", fontsize=7.5, ncol=4)
    plt.tight_layout()
    plt.savefig(paths["chart"], dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Timeline     →  {paths['chart']}")


def save_grid(frames: list[dict], results: list[dict], paths: dict) -> None:
    NC, TW, TH, LH = 6, 210, 128, 90
    NR  = math.ceil(len(frames) / NC)
    grid = Image.new("RGB", (NC*TW, NR*(TH+LH)), (8, 8, 20))
    draw = ImageDraw.Draw(grid)
    rgb  = {c.id: c.rgb for c in CLASSES}

    for i, fr in enumerate(frames):
        r   = results[i]
        col = i % NC; row = i // NC
        x   = col*TW; y   = row*(TH+LH)
        grid.paste(fr["img"].resize((TW,TH), Image.LANCZOS), (x, y))
        cid = r.get("class_id")
        bg  = rgb.get(cid, (10,55,18)) if cid else (10,55,18)
        draw.rectangle([(x,y+TH),(x+TW-1,y+TH+LH-1)], fill=bg)
        draw.text((x+4,y+TH+4),  r["class_name"][:24],
                  fill=(255,240,80) if cid else (80,255,100))
        draw.text((x+4,y+TH+22), f"t={r['time_sec']:.1f}s  {r['severity']}",
                  fill=(210,210,210))
        cap = r["caption"]; cap = (cap[:44]+"…") if len(cap)>44 else cap
        draw.text((x+4,y+TH+40), cap, fill=(180,180,255))
        if r.get("alert_id"):
            draw.text((x+4,y+TH+62), f"🚨 ALERT #{r['alert_id']:03d}", fill=(255,70,70))
        border = rgb.get(cid,(25,155,55)) if cid else (25,155,55)
        draw.rectangle([(x,y),(x+TW-1,y+TH+LH-1)], outline=border, width=3)

    grid.save(paths["grid"], quality=90)
    print(f"  Grid         →  {paths['grid']}")


def save_report(results: list[dict], paths: dict) -> None:
    with open(paths["report"], "w", encoding="utf-8") as f:
        json.dump({
            "created_at": datetime.now().isoformat(),
            "video"     : CFG.video_path,
            "config"    : {
                "fps"       : CFG.extract_fps,
                "max_frames": CFG.max_frames,
                "diff_sigma": CFG.diff_sigma,
                "ssim_sigma": CFG.ssim_sigma,
                "cooldown_s": CFG.cooldown_sec,
            },
            "summary": {
                "total_frames"    : len(results),
                "anomalous_frames": sum(1 for r in results if r["class_id"]),
                "total_alerts"    : len(MGR.alerts),
                "classes_detected": sorted({r["class_name"] for r in results if r["class_id"]}),
            },
            "alerts": [a.as_dict() for a in MGR.alerts],
            "frames": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Report       →  {paths['report']}")

# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Uncorrelated Video Anomaly Detection — 6-class alert system"
    )
    p.add_argument("--video",      type=str,   default=None)
    p.add_argument("--output",     type=str,   default=None)
    p.add_argument("--fps",        type=float, default=None,
                   help="Frames per second to sample (default 1.0)")
    p.add_argument("--max-frames", type=int,   default=None, dest="max_frames")
    p.add_argument("--sigma",      type=float, default=None,
                   help="Sensitivity (lower = more sensitive, default 1.5)")
    p.add_argument("--cooldown",   type=float, default=None,
                   help="Seconds between same-class alerts (default 5)")
    return p.parse_args()

# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    CFG.patch(_parse())
    paths = _paths()

    # ── header ────────────────────────────────────────────────────────────────
    _banner("UNCORRELATED VIDEO ANOMALY DETECTION  ·  v3.0")
    print(f"  Video      :  {CFG.video_path}")
    print(f"  Output     :  {Path(CFG.output_dir).resolve()}")
    print(f"  FPS        :  {CFG.extract_fps}  |  Max frames: {CFG.max_frames}")
    print(f"  Sensitivity:  σ={CFG.diff_sigma}  |  Cooldown: {CFG.cooldown_sec}s/class")
    print(f"  Classes    :  {', '.join(cl.name for cl in CLASSES)}")

    # ── 0. setup dirs ─────────────────────────────────────────────────────────
    for k in ("frames", "anomaly"):
        shutil.rmtree(paths[k], ignore_errors=True)
        paths[k].mkdir(parents=True, exist_ok=True)
    Path(CFG.output_dir).mkdir(parents=True, exist_ok=True)

    # ── 1. extract frames ─────────────────────────────────────────────────────
    _banner("1 / 5  FRAME  EXTRACTION")
    frames = extract_frames(CFG.video_path)
    print(f"  {len(frames)} frames extracted @ {CFG.extract_fps} FPS")

    # ── 2. load BLIP ──────────────────────────────────────────────────────────
    _banner("2 / 5  BLIP  MODEL")
    BLIP.load()

    # ── 3. visual scoring ─────────────────────────────────────────────────────
    _banner("3 / 5  VISUAL  SCORING  (pixel-diff + SSIM)")
    diffs, ssims, thr_d, thr_s = score_frames(frames)
    print(f"  Pixel-diff threshold  :  {thr_d:.4f}")
    print(f"  SSIM threshold        :  {thr_s:.4f}")

    # ── 4. classify + alert ───────────────────────────────────────────────────
    _banner("4 / 5  CLASSIFICATION  (uncorrelated VQA per class)")
    results = process(frames, diffs, ssims, thr_d, thr_s, paths)
    n_anom  = sum(1 for r in results if r["class_id"])
    print(f"\n  Anomalous frames :  {C.RED}{n_anom}{C.RST} / {len(results)}")
    print(f"  Alerts fired     :  {C.RED}{len(MGR.alerts)}{C.RST}")

    # ── 5. outputs ────────────────────────────────────────────────────────────
    _banner("5 / 5  ALERTS  +  OUTPUTS")
    print_alert_log(MGR.alerts, paths)
    save_timeline(results, thr_d, paths)
    save_grid(frames, results, paths)
    save_report(results, paths)

    # ── final summary ─────────────────────────────────────────────────────────
    _banner("FINAL  SUMMARY")
    print(f"  Total frames   :  {C.BLD}{len(results)}{C.RST}")
    print(f"  Anomalous      :  {C.RED}{C.BLD}{n_anom}{C.RST}")
    print(f"  Normal         :  {C.GRN}{C.BLD}{len(results)-n_anom}{C.RST}")
    print(f"  Alerts fired   :  {C.RED}{C.BLD}{len(MGR.alerts)}{C.RST}\n")

    if MGR.alerts:
        print(f"  {'#':<4}  {'Class':<30}  {'Sev':<10}  {'Video t':>8}    Wall clock")
        _rule(74)
        for a in MGR.alerts:
            sc = _SEV_CLR.get(a.severity, C.DIM)
            print(f"  {a.alert_id:<4}  {sc}{a.class_name:<30}{C.RST}  "
                  f"{sc}{a.severity:<10}{C.RST}  "
                  f"{C.YLW}t={a.video_time_s:>6.2f}s{C.RST}  "
                  f"{C.DIM}{a.wall_time}{C.RST}")
    else:
        print(f"  {C.GRN}No anomalies detected.{C.RST}")

    print(f"\n  {C.GRN}Done!  Output → {Path(CFG.output_dir).resolve()}{C.RST}\n")


if __name__ == "__main__":
    main()