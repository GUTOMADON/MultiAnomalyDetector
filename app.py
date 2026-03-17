"""
MultiAnomalyDetector extended with Traffic Detection, Ambulances and Theft
Detects anomalies and generates alerts with timestamps in real-time videos.

ORIGINAL CATEGORIES (fully preserved):
  1 - traffic accident
  2 - stop line violation
  3 - red light violation
  4 - ambulance stuck in a traffic
  5 - people stealing item from the store
  6 - suspicious activity.

Optimised for CPU, same principles as the original system:
  BLIP base (2x faster than large)
  VQA gated on anomalous frames
  YOLO with per-track ID tracking
  Per-track movement history to detect stopped vehicle / stuck ambulance
"""

from __future__ import annotations

# stdlib
import base64
import csv
import json
import logging
import math
import os
import sys
import tempfile
import threading
import warnings
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# third-party
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

# Configuration 

EXTRACT_FPS: float = 1.0                # frames per second extracted
MAX_FRAMES:  int   = 60                  # maximum frames to process
DIFF_SIZE:   tuple[int, int] = (96, 96)  # size for difference computation
OUTPUT_ROOT: str   = r"C:\Users\Gustavo\Desktop\DrApurbasTasks\IncidentAnomaliesAlerts\output_video"
JPEG_QUALITY: int  = 85
BLIP_CHECKPOINT: str = "Salesforce/blip-image-captioning-base"

# Caption prompts (original)
PROMPT_NORMAL    = "a traffic camera photo of"
PROMPT_ANOMALY   = "a traffic camera showing an incident where"
PROMPT_CROSSWALK = "a traffic camera showing a crosswalk where"

# Original VQA questions (unchanged)
VQA_ACCIDENT         = "is there a vehicle collision or traffic accident in this image?"
VQA_FIRE             = "is there fire or smoke visible in this image?"
VQA_FALL             = "has a person fallen down or been knocked over in this image?"
VQA_WRONG_WAY        = "is a vehicle driving on the wrong side of the road in this image?"
VQA_SEVERITY         = "how severe is the incident? answer only: minor, moderate, or severe"
VQA_PED_CROSSWALK    = "is a pedestrian walking across a zebra crossing or crosswalk in this image?"
VQA_VEHICLE_CROSS_1  = "is a car, truck, bus, or motorcycle on top of or blocking a zebra crossing?"
VQA_VEHICLE_CROSS_2  = "is any vehicle stopped, parked, or driving over a striped pedestrian crossing?"
VQA_IS_MOTOR_VEHICLE = "is there a car, truck, bus, van, or motorcycle visible in this image?"

# New VQA questions for extended detection
VQA_COLLISION        = "are two or more vehicles colliding or have they crashed into each other?"
VQA_AMBULANCE_DETECT = "is there an ambulance or emergency vehicle visible in this image?"
VQA_STOP_LINE        = "is a vehicle crossing or stopped past a stop line at a red traffic light?"
VQA_AMB_BLOCKED      = "is an ambulance blocked, surrounded, or unable to move due to other vehicles?"
VQA_SHOPLIFTING      = "is a person hiding, concealing, or stealing items inside a store?"
VQA_RED_LIGHT        = "is a vehicle driving through a red traffic light or running a red light?"
VQA_AMBULANCE_STUCK  = "is an ambulance stuck in a traffic jam or unable to proceed?"

_YES_TOKENS = ("yes", "yeah", "true", "correct", "yep")
_VALID_SEVERITIES = {"minor", "moderate", "severe"}


def is_yes(answer: str) -> bool:
    """Return True if the answer starts with a positive token."""
    token = (answer or "").strip().lower()
    return any(token.startswith(y) for y in _YES_TOKENS)


def parse_severity(raw: str) -> str:
    """Extract severity from VQA answer."""
    if not raw or raw == "n/a":
        return "n/a"
    first = raw.strip().lower().split()[0]
    return first if first in _VALID_SEVERITIES else "n/a"


# Alert rule table 
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

    # YELLOW
    (["pedestrian crossing", "person crossing", "pedestrian on crosswalk",
      "crossing the street", "crossing the road", "zebra crossing",
      "using crosswalk", "crossing at crosswalk"],
     "PEDESTRIAN ON CROSSWALK", "YELLOW"),

    # INFO
    (["traffic", "road", "street", "highway", "car", "vehicle", "truck",
      "bus", "motorcycle", "intersection", "driving", "lane", "signal"],
     "TRAFFIC SCENE", "INFO"),
]

ALERT_RULES.extend([
    (["not yielding", "not yield", "failure to yield", "refusing to yield",
      "blocking ambulance", "obstructing ambulance"],
     "FAILURE TO YIELD TO AMBULANCE", "CRITICAL"),

    (["intersection blocked", "blocking intersection", "gridlock"],
     "INTERSECTION BLOCKED", "HIGH"),

    (["concealing item", "hiding item", "pocketing item", "tucking item",
      "stuffing item", "hiding merchandise", "concealing merchandise"],
     "SHOPLIFTING IN STORE", "CRITICAL"),

    (["wrong way", "wrong-way", "driving wrong way", "wrong direction",
      "going wrong way", "wrong side of road"],
     "WRONG-WAY VEHICLE", "CRITICAL"),
])


def map_caption_to_alert(caption: str) -> tuple[str, str]:
    """Map a caption to an alert label and severity using keyword matching."""
    lower = caption.lower()
    for keywords, label, severity in ALERT_RULES:
        if any(kw in lower for kw in keywords):
            return label, severity
    return "NO ALERT", "NORMAL"


# Traffic detection constants and data classes
# COCO class IDs relevant to traffic scenes
TRAFFIC_CLASSES: Dict[int, str] = {
    0:  "person",
    1:  "bicycle",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    6:  "train",
    7:  "truck",
    9:  "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
}

VEHICLE_NAMES = {"car", "motorcycle", "bus", "truck", "bicycle", "train"}

AMBULANCE_KEYWORDS_BLIP = {
    "ambulance", "emergency", "rescue", "paramedic",
    "medic", "ems", "fire truck", "fire engine",
}

# Shoplifting-relevant COCO classes (for retail scene detection)
RETAIL_CONCEAL_CLASSES = {"backpack", "handbag", "suitcase"}

# Severity ordering for comparison
_SEV_ORDER: Dict[str, int] = {
    "NORMAL": 0, "INFO": 0, "LOW": 1, "YELLOW": 1,
    "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4,
}


@dataclass
class TrafficDetection:
    """Single YOLO detection result with optional extra metadata."""
    class_id:   int
    class_name: str
    confidence: float
    bbox:       Tuple[int, int, int, int]     # x1, y1, x2, y2
    track_id:   Optional[int] = None
    extra:      dict          = field(default_factory=dict)


@dataclass
class TrafficAlertEvent:
    """Structured record for CSV/JSON traffic event logging."""
    timestamp:           str
    frame_number:        int
    video_time_seconds:  float
    alert_type:          str
    severity:            str
    description:         str
    blip_caption:        str
    confidence:          float
    vehicle_count:       int
    ambulance_present:   bool
    traffic_light_state: str
    suspicious_keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["suspicious_keywords"] = ",".join(d["suspicious_keywords"])
        return d



# YOLO Detector (with fallback to Faster-RCNN)
class YOLODetector:
    """
    YOLOv8 with per-object tracking (ultralytics backend preferred).
    Automatically falls back to torchvision Faster-RCNN when ultralytics
    is not installed, so the system always runs.
    """

    def __init__(self, model_size: str = "yolov8n", device: str = "cpu"):
        self.device  = device
        self.model   = None
        self.backend = "none"
        self._load(model_size)

    def _load(self, model_size: str) -> None:
        try:
            from ultralytics import YOLO   # type: ignore
            logging.getLogger(__name__).info(
                "[YOLODetector] Loading %s via ultralytics ...", model_size)
            self.model   = YOLO(f"{model_size}.pt")
            self.backend = "ultralytics"
            logging.getLogger(__name__).info("[YOLODetector] Ready.")
        except ImportError:
            logging.getLogger(__name__).warning(
                "[YOLODetector] ultralytics not found, loading torchvision Faster-RCNN fallback.")
            self._load_tv()

    def _load_tv(self) -> None:
        try:
            from torchvision.models.detection import (   # type: ignore
                FasterRCNN_ResNet50_FPN_Weights,
                fasterrcnn_resnet50_fpn,
            )
            weights    = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn(weights=weights)
            self.model.eval().to(self.device)
            self.backend = "torchvision"
            logging.getLogger(__name__).info("[YOLODetector] Faster-RCNN ready.")
        except Exception as exc:
            logging.getLogger(__name__).error(
                "[YOLODetector] No detection model available: %s", exc)

    def detect(
        self, frame: np.ndarray, conf: float = 0.35
    ) -> List[TrafficDetection]:
        """Run detection on a BGR frame and return a list of TrafficDetection."""
        if self.backend == "ultralytics":
            return self._det_ultralytics(frame, conf)
        if self.backend == "torchvision":
            return self._det_torchvision(frame, conf)
        return []

    def _det_ultralytics(
        self, frame: np.ndarray, conf: float
    ) -> List[TrafficDetection]:
        results = self.model.track(
            frame,
            conf=conf,
            persist=True,
            verbose=False,
            classes=list(TRAFFIC_CLASSES.keys()),
        )
        out: List[TrafficDetection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cid   = int(box.cls[0].item())
                score = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                tid   = int(box.id[0].item()) if box.id is not None else None
                out.append(TrafficDetection(
                    class_id   = cid,
                    class_name = TRAFFIC_CLASSES.get(cid, f"cls_{cid}"),
                    confidence = score,
                    bbox       = (x1, y1, x2, y2),
                    track_id   = tid,
                ))
        return out

    def _det_torchvision(
        self, frame: np.ndarray, conf: float
    ) -> List[TrafficDetection]:
        import torchvision.transforms.functional as TF   # type: ignore
        img_t = TF.to_tensor(
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ).to(self.device)
        with torch.no_grad():
            preds = self.model([img_t])[0]
        out: List[TrafficDetection] = []
        for box, lbl, sc in zip(preds["boxes"], preds["labels"], preds["scores"]):
            if sc.item() < conf:
                continue
            cid = lbl.item() - 1
            if cid not in TRAFFIC_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            out.append(TrafficDetection(
                class_id   = cid,
                class_name = TRAFFIC_CLASSES.get(cid, f"cls_{cid}"),
                confidence = float(sc.item()),
                bbox       = (x1, y1, x2, y2),
            ))
        return out


# Traffic Light Classifier (HSV colour analysis)
class TrafficLightClassifier:
    """
    Classifies a traffic-light crop as red / yellow / green / unknown
    using OpenCV HSV colour-range analysis.
    """
    _RANGES = {
        "red":    [((0,  100, 100), (10,  255, 255)),
                   ((160, 100, 100), (180, 255, 255))],
        "yellow": [((15, 100, 100), (35,  255, 255))],
        "green":  [((40, 50,  50),  (90,  255, 255))],
    }

    def classify(self, crop: np.ndarray) -> str:
        """Classify a cropped image of a traffic light."""
        if crop is None or crop.size == 0:
            return "unknown"
        hsv    = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        scores: Dict[str, int] = {}
        for state, ranges in self._RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            scores[state] = int(mask.sum())
        best = max(scores, key=scores.__getitem__)
        return best if scores[best] > 200 else "unknown"

    def classify_from_frame(
        self,
        frame: np.ndarray,
        bbox:  Tuple[int, int, int, int],
    ) -> str:
        """Extract the top portion of a traffic light bounding box and classify."""
        x1, y1, x2, y2 = bbox
        mid_y = y1 + (y2 - y1) * 2 // 3
        crop  = frame[y1:mid_y, x1:x2]
        return self.classify(crop)


# Ambulance Detector (colour/shape heuristic)
class AmbulanceDetector:
    """
    Identifies ambulances from vehicle bounding-box crops by testing:
      1. White/yellow body colour dominance.
      2. Red/orange/blue stripe presence (siren markings).
      3. Aspect ratio approximately 1.3 to 3.0 (rectangular box vehicle).
    Cross-referenced against BLIP caption keywords.
    """

    def is_ambulance_crop(
        self, crop: np.ndarray
    ) -> Tuple[bool, float]:
        """Return (is_ambulance, confidence_score)."""
        if crop is None or crop.size == 0:
            return False, 0.0
        hsv      = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        total_px = crop.shape[0] * crop.shape[1]

        white_mask = cv2.inRange(hsv,
            np.array([0, 0, 200]), np.array([180, 40, 255]))
        white_frac = white_mask.sum() / 255 / max(total_px, 1)

        red1   = cv2.inRange(hsv, np.array([0,  100, 100]), np.array([10,  255, 255]))
        red2   = cv2.inRange(hsv, np.array([160,100, 100]), np.array([180, 255, 255]))
        orange = cv2.inRange(hsv, np.array([10, 150, 150]), np.array([25,  255, 255]))
        blue   = cv2.inRange(hsv, np.array([100,100, 100]), np.array([130, 255, 255]))
        stripe_frac = (red1 | red2 | orange | blue).sum() / 255 / max(total_px, 1)

        score = 0.0
        if white_frac > 0.35:
            score += 0.35
        if stripe_frac > 0.04:
            score += 0.30
        if white_frac > 0.55:
            score += 0.20
        h, w = crop.shape[:2]
        if 1.3 <= (w / max(h, 1)) <= 3.0:
            score += 0.15

        return score >= 0.50, round(min(score, 1.0), 3)

    @staticmethod
    def caption_mentions_ambulance(caption: str) -> bool:
        cap_l = caption.lower()
        return any(kw in cap_l for kw in AMBULANCE_KEYWORDS_BLIP)


# Shoplifting Behaviour Analyser
class ShopliftingAnalyzer:
    """
    Heuristic shoplifting detector combining YOLO bounding-box geometry,
    per-track motion history, and BLIP caption keyword scanning.
    """

    HIGH_RISK_KW = {
        "steal", "stealing", "stolen", "theft", "hiding", "conceal",
        "pocket", "pocketing", "shoplifting", "shoplift", "grab",
        "grabbing", "snatch", "snatching", "smuggl", "tucking",
        "stuff", "stuffing",
    }
    MEDIUM_RISK_KW = {
        "bag", "backpack", "purse", "jacket", "coat",
        "hiding", "crouch", "crouching", "bend", "bending",
        "look around", "looking around", "glance", "nervous",
        "suspicious", "rush", "rushing",
    }
    LOW_RISK_KW = {
        "take", "taking", "pick", "picking", "carry", "carrying",
        "hold", "holding", "pocket", "shelf", "remove", "removing",
    }

    def __init__(self, loiter_frames: int = 40):
        self._loiter_frames = loiter_frames
        self._track_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=loiter_frames))

    def analyse(
        self,
        detections: List[TrafficDetection],
        caption:    str,
        frame_bgr:  np.ndarray,
    ) -> Tuple[float, List[str]]:
        """Returns (shoplifting_score 0-1, matched_keywords)."""
        score = 0.0
        kws:  List[str] = []
        cap_l = caption.lower()
        h, w = frame_bgr.shape[:2]

        for kw in self.HIGH_RISK_KW:
            if kw in cap_l and kw not in kws:
                score += 0.40; kws.append(kw)
        for kw in self.MEDIUM_RISK_KW:
            if kw in cap_l and kw not in kws:
                score += 0.15; kws.append(kw)
        for kw in self.LOW_RISK_KW:
            if kw in cap_l and kw not in kws:
                score += 0.05; kws.append(kw)

        persons = [d for d in detections if d.class_name == "person"]
        conceal = [d for d in detections if d.class_name in RETAIL_CONCEAL_CLASSES]

        for person in persons:
            tid = person.track_id or 0
            px1, py1, px2, py2 = person.bbox
            p_cx = (px1 + px2) / 2
            p_cy = (py1 + py2) / 2
            self._track_history[tid].append((p_cx, p_cy))

            for item in conceal:
                ix1, iy1, ix2, iy2 = item.bbox
                inter_x  = max(0, min(px2,ix2) - max(px1,ix1))
                inter_y  = max(0, min(py2,iy2) - max(py1,iy1))
                inter    = inter_x * inter_y
                area_a   = (px2-px1) * (py2-py1)
                area_b   = (ix2-ix1) * (iy2-iy1)
                iou      = inter / max(area_a + area_b - inter, 1)
                if iou > 0.25:
                    score += 0.20
                    kws.append("concealment_object")

            if len(self._track_history[tid]) >= self._loiter_frames:
                hist = list(self._track_history[tid])
                xs = [p[0] for p in hist]; ys = [p[1] for p in hist]
                diag = (w**2 + h**2)**0.5
                move = ((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2)**0.5
                if move / max(diag, 1) < 0.05:
                    score += 0.15
                    kws.append("loitering")

        return min(score, 1.0), list(set(kws))


# Traffic Behavioural Analyser
class TrafficAnalyzer:
    """
    Multi-signal engine combining YOLO detections, track velocity history,
    and BLIP captions to score each frame for traffic violations and
    ambulance obstruction.
    """

    HIGH_RISK_KW = {
        "blocked", "block", "stuck", "congestion", "obstruct", "obstruction",
        "crash", "collision", "accident", "emergency", "ambulance",
        "violation", "violate", "ignore", "ignoring", "refuses", "refusing",
        "red light", "running", "through red", "stop line", "wrong way",
    }
    MEDIUM_RISK_KW = {
        "stop", "stopped", "stationary", "traffic", "crowded", "crowding",
        "surrounded", "blocking", "tailgate", "pedestrian", "crossing",
        "intersection",
    }
    LOW_RISK_KW = {
        "slow", "wait", "waiting", "lane", "vehicle", "car", "truck",
        "bus", "motorcycle", "drive", "driving",
    }

    def __init__(
        self,
        stuck_frames:       int   = 50,
        yield_radius_frac:  float = 0.25,
        motion_window:      int   = 20,
        velocity_threshold: float = 4.0,
    ):
        self._stuck_frames      = stuck_frames
        self._yield_radius_frac = yield_radius_frac
        self._motion_window     = motion_window
        self._vel_thresh        = velocity_threshold
        self._history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=max(stuck_frames, motion_window)))
        self._global_cooldown:  Dict[str, int] = {}
        self._cooldown = 45

    def analyse(
        self,
        detections:          List[TrafficDetection],
        caption:             str,
        frame_number:        int,
        frame:               np.ndarray,
        ambulance_track_ids: List[int],
    ) -> Tuple[float, str, str, List[str]]:
        """
        Returns:
            score (float): risk score between 0 and 1
            alert_type (str): type of alert detected
            reason_str (str): description of reasons (empty if none)
            keywords (List[str]): relevant keywords
        """
        h, w   = frame.shape[:2]
        score  = 0.0
        reasons: List[str] = []
        kws:    List[str]  = []

        # 1. Scan caption for high-risk keywords
        cap_l = caption.lower()
        for kw in self.HIGH_RISK_KW:
            if kw in cap_l and kw not in kws:
                score += 0.28; kws.append(kw)
                reasons.append(f"High-risk caption keyword: '{kw}'")
        for kw in self.MEDIUM_RISK_KW:
            if kw in cap_l and kw not in kws:
                score += 0.10; kws.append(kw)
                reasons.append(f"Medium-risk keyword: '{kw}'")

        # 2. Detect red-light violations (vehicle past stop line)
        red_lights = [d for d in detections
                      if d.class_name == "traffic light"
                      and d.extra.get("tl_state") == "red"]
        vehicles   = [d for d in detections
                      if d.class_name in VEHICLE_NAMES]

        for tl in red_lights:
            for veh in vehicles:
                if self._past_stopline(veh.bbox, tl.bbox):
                    score += 0.45
                    kws.append("red_light_violation")
                    reasons.append(
                        f"Vehicle #{veh.track_id} past red light stop-line")

        # 3. Ambulance checks
        amb_dets = ([d for d in detections if d.track_id in ambulance_track_ids]
                    or [d for d in detections if d.extra.get("is_ambulance")])

        if amb_dets:
            kws.append("ambulance_present")
            for amb in amb_dets:
                ax1, ay1, ax2, ay2 = amb.bbox
                a_cx = (ax1 + ax2) / 2
                a_cy = (ay1 + ay2) / 2
                radius = w * self._yield_radius_frac
                proximate = [
                    v for v in vehicles
                    if (v.track_id not in ambulance_track_ids)
                    and self._dist_centres(v.bbox, amb.bbox) < radius
                ]
                if proximate:
                    score += 0.20 * min(len(proximate), 3) / 3
                    kws.append("vehicles_near_ambulance")
                    reasons.append(
                        f"{len(proximate)} vehicle(s) crowding ambulance")

                tid = amb.track_id or 0
                self._history[tid].append((a_cx, a_cy))
                if len(self._history[tid]) >= self._stuck_frames:
                    hist = list(self._history[tid])
                    xs = [p[0] for p in hist]; ys = [p[1] for p in hist]
                    diag = (w**2 + h**2)**0.5
                    mv = ((max(xs)-min(xs))**2 + (max(ys)-min(ys))**2)**0.5
                    if mv / max(diag, 1) < 0.03:
                        score += 0.35
                        kws.append("ambulance_stuck")
                        reasons.append(
                            f"Ambulance #{tid} stationary for "
                            f">={self._stuck_frames} frames")

                for veh in proximate:
                    vtid = veh.track_id or -1
                    self._history[vtid].append(self._centre(veh.bbox))
                    if len(self._history[vtid]) >= self._motion_window:
                        vel = self._avg_vel(self._history[vtid])
                        if vel > self._vel_thresh and self._approaching(
                                self._history[vtid], (a_cx, a_cy)):
                            score += 0.25
                            kws.append("not_yielding")
                            reasons.append(
                                f"Vehicle #{vtid} approaching ambulance at "
                                f"{vel:.1f} px/frame")

        # 4. Intersection blocking (stationary vehicles near red light)
        if red_lights and len(vehicles) >= 3:
            stationary = [v for v in vehicles if self._is_stationary(v)]
            if len(stationary) >= 2:
                score += 0.15
                kws.append("intersection_blocked")
                reasons.append(
                    f"{len(stationary)} stationary vehicles near red light")

        # 5. Update non-ambulance vehicle history
        for veh in vehicles:
            if veh.track_id and veh.track_id not in ambulance_track_ids:
                self._history[veh.track_id].append(self._centre(veh.bbox))

        score = min(score, 1.0)
        reason_str = "; ".join(reasons) if reasons else ""   # Empty if no reasons
        alert_type = self._classify_alert(kws)
        return score, alert_type, reason_str, list(set(kws))

    # Helper methods
    @staticmethod
    def _centre(b: Tuple[int,int,int,int]) -> Tuple[float,float]:
        return (b[0]+b[2])/2, (b[1]+b[3])/2

    @staticmethod
    def _dist_centres(
        a: Tuple[int,int,int,int],
        b: Tuple[int,int,int,int],
    ) -> float:
        ax, ay = (a[0]+a[2])/2, (a[1]+a[3])/2
        bx, by = (b[0]+b[2])/2, (b[1]+b[3])/2
        return ((ax-bx)**2 + (ay-by)**2)**0.5

    @staticmethod
    def _past_stopline(
        veh: Tuple[int,int,int,int],
        tl:  Tuple[int,int,int,int],
    ) -> bool:
        vx1,vy1,vx2,vy2 = veh; tx1,ty1,tx2,ty2 = tl
        v_cx = (vx1+vx2)/2
        return (tx1 <= v_cx <= tx2) and (vy2 > ty2)

    @staticmethod
    def _avg_vel(hist: deque) -> float:
        pts = list(hist)
        if len(pts) < 2:
            return 0.0
        return sum(
            ((pts[i][0]-pts[i-1][0])**2 + (pts[i][1]-pts[i-1][1])**2)**0.5
            for i in range(1, len(pts))
        ) / (len(pts)-1)

    @staticmethod
    def _approaching(hist: deque, target: Tuple[float,float]) -> bool:
        pts = list(hist)
        if len(pts) < 2:
            return False
        tx, ty = target
        d0 = ((pts[0][0]-tx)**2 + (pts[0][1]-ty)**2)**0.5
        d1 = ((pts[-1][0]-tx)**2 + (pts[-1][1]-ty)**2)**0.5
        return d1 < d0

    def _is_stationary(self, det: TrafficDetection) -> bool:
        tid = det.track_id
        if tid is None or tid not in self._history or len(self._history[tid]) < 5:
            return False
        return self._avg_vel(self._history[tid]) < self._vel_thresh

    @staticmethod
    def _classify_alert(kws: List[str]) -> str:
        if "red_light_violation"   in kws: return "RED_LIGHT_VIOLATION"
        if "ambulance_stuck"       in kws and "not_yielding" in kws:
            return "AMBULANCE_BLOCKED_CRITICAL"
        if "ambulance_stuck"       in kws: return "AMBULANCE_STUCK"
        if "not_yielding"          in kws: return "FAILURE_TO_YIELD"
        if "intersection_blocked"  in kws: return "INTERSECTION_BLOCKED"
        if "vehicles_near_ambulance" in kws: return "AMBULANCE_PROXIMITY_WARNING"
        for kw in ("fight","attack","theft","vandalism","assault"):
            if kw in kws: return "SUSPICIOUS_ROADSIDE_BEHAVIOR"
        return "GENERAL_TRAFFIC_ANOMALY"

    def severity(self, score: float) -> str:
        if score >= 0.70: return "CRITICAL"
        if score >= 0.50: return "HIGH"
        if score >= 0.30: return "MEDIUM"
        return "LOW"

    def should_alert(
        self, alert_type: str, frame_number: int
    ) -> bool:
        last = self._global_cooldown.get(alert_type, -self._cooldown-1)
        if frame_number - last < self._cooldown:
            return False
        self._global_cooldown[alert_type] = frame_number
        return True


# Frame Visualizer (draws bounding boxes, status bar, etc.)
class TrafficFrameVisualizer:
    """
    Draws YOLO bounding boxes (colour-coded by class), a traffic-light
    state indicator, a risk-score bar, BLIP caption strip, and an alert
    flash border on BGR numpy frames.
    """

    CLASS_COLORS = {
        "person":        (60,  180, 255),
        "bicycle":       (150, 200, 100),
        "car":           (200, 200, 200),
        "motorcycle":    (200, 150,  50),
        "bus":           (100, 200, 200),
        "truck":         (100, 150, 250),
        "train":         ( 80,  80, 200),
        "traffic light": (240, 240,  80),
        "stop sign":     ( 40,  40, 220),
        "ambulance":     (  0, 255, 255),
        "backpack":      (  0, 140, 255),
        "handbag":       (  0, 140, 255),
        "suitcase":      (  0, 140, 255),
        "default":       (180, 180, 180),
    }
    TL_COLORS = {
        "red":     (0,   0, 255),
        "yellow":  (0, 215, 255),
        "green":   (0, 220,  90),
        "unknown": (120,120,120),
    }
    SEV_COLORS = {
        "CRITICAL": (0,   0, 200),
        "HIGH":     (0,  60, 220),
        "MEDIUM":   (0, 140, 255),
        "LOW":      (0, 200, 120),
    }

    def draw(
        self,
        frame:        np.ndarray,
        detections:   List[TrafficDetection],
        caption:      str,
        score:        float,
        severity:     str,
        alert_type:   str,
        frame_number: int,
        fps:          float,
        tl_state:     str,
    ) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]

        # Draw bounding boxes for each detection
        for det in detections:
            is_amb = det.extra.get("is_ambulance", False)
            name   = "ambulance" if is_amb else det.class_name
            color  = self.CLASS_COLORS.get(name, self.CLASS_COLORS["default"])
            thick  = 3 if is_amb else 2
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
            lbl = name
            if det.track_id is not None:
                lbl += f" #{det.track_id}"
            if det.class_name == "traffic light":
                st  = det.extra.get("tl_state", "unknown")
                lbl += f" [{st.upper()}]"
                cv2.rectangle(out, (x1, y1), (x2, y2),
                              self.TL_COLORS[st], 3)
            lbl += f" {det.confidence:.0%}"
            cv2.putText(out, lbl, (x1, max(y1-6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

        # Top status bar
        bar_h = 58
        cv2.rectangle(out, (0, 0), (w, bar_h), (15, 15, 15), -1)
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(out, ts, (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,200,200), 1, cv2.LINE_AA)
        vid_t = frame_number / max(fps, 0.01)
        cv2.putText(out, f"Frame {frame_number}  |  {vid_t:.1f}s", (8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (160,160,160), 1, cv2.LINE_AA)
        sev_col = self.SEV_COLORS.get(severity, (100,100,100))
        bar_px  = int(score * (w - 10))
        cv2.rectangle(out, (0, 38), (bar_px, bar_h-4), sev_col, -1)
        cv2.putText(out, f"Risk: {score:.0%}  [{severity}]  {alert_type}",
                    (8, bar_h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.37,
                    (255,255,255), 1, cv2.LINE_AA)

        # Traffic-light indicator pill (top-right)
        tl_col = self.TL_COLORS.get(tl_state, self.TL_COLORS["unknown"])
        cv2.circle(out, (w-28, 26), 16, tl_col, -1)
        cv2.circle(out, (w-28, 26), 16, (255,255,255), 2)
        tl_lbl = tl_state[0].upper() if tl_state != "unknown" else "?"
        cv2.putText(out, tl_lbl, (w-34, 31),
                    cv2.FONT_HERSHEY_DUPLEX, 0.50, (0,0,0), 2, cv2.LINE_AA)

        # Bottom caption strip
        cap_y = h - 24
        cv2.rectangle(out, (0, cap_y-16), (w, h), (15,15,15), -1)
        cv2.putText(out, f"BLIP: {caption[:115]}",
                    (8, cap_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (220,220,100), 1, cv2.LINE_AA)

        # Alert flash border for high severity
        if severity in ("CRITICAL", "HIGH"):
            fc    = (0, 0, 200) if severity == "CRITICAL" else (0, 60, 220)
            thick = 7 if severity == "CRITICAL" else 4
            cv2.rectangle(out, (0, 0), (w-1, h-1), fc, thick)
            cv2.putText(out, f"\u26a0  ALERT: {alert_type}",
                        (w//2-150, h//2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.90, fc, 2, cv2.LINE_AA)
        elif severity == "MEDIUM":
            cv2.rectangle(out, (0, 0), (w-1, h-1), (0, 140, 255), 3)

        return out


# CSV and JSON Event Logger
class TrafficEventLogger:
    """Writes TrafficAlertEvent records to CSV and JSON."""

    _FIELDS = [
        "timestamp", "frame_number", "video_time_seconds",
        "alert_type", "severity", "confidence",
        "vehicle_count", "ambulance_present", "traffic_light_state",
        "suspicious_keywords", "description", "blip_caption",
    ]

    def __init__(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path  = output_dir / "traffic_alerts.csv"
        self.json_path = output_dir / "traffic_alerts.json"
        self._json_buf: List[dict] = []
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self._FIELDS).writeheader()
        logging.getLogger(__name__).info(
            "[TrafficEventLogger] Logs -> %s", output_dir)

    def log(self, event: TrafficAlertEvent) -> None:
        logging.getLogger(__name__).warning(
            "🚨 [%s/%s] frame=%d  t=%.1fs  %s",
            event.alert_type, event.severity,
            event.frame_number, event.video_time_seconds,
            event.description,
        )
        d = event.to_dict()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self._FIELDS).writerow(d)
        self._json_buf.append(event.to_dict())
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self._json_buf, f, indent=2)


# Flask Application Setup
app = Flask(__name__)
app.config["SECRET_KEY"]         = "multianomaly-secret-key"
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
app.config["UPLOAD_FOLDER"]      = tempfile.gettempdir()

CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global model references
_processor: "BlipProcessor | None"              = None
_model:     "BlipForConditionalGeneration | None" = None
_device:    torch.device                         = torch.device("cpu")

# New traffic / detection model globals
_yolo_detector:       "YOLODetector | None"          = None
_tl_classifier:       "TrafficLightClassifier | None" = None
_amb_detector:        "AmbulanceDetector | None"      = None
_shoplifting_analyzer:"ShopliftingAnalyzer | None"    = None
_frame_visualizer:    "TrafficFrameVisualizer | None"  = None

# Maps session_id to background thread
active_sessions: dict[str, threading.Thread] = {}


# Model Initialization
def load_blip_model() -> None:
    """Load BLIP base checkpoint."""
    global _processor, _model, _device

    if not BLIP_AVAILABLE:
        print("[WARN] transformers not installed, BLIP unavailable.")
        return

    print(f"[BLIP] Loading {BLIP_CHECKPOINT} ...")
    _processor = BlipProcessor.from_pretrained(BLIP_CHECKPOINT, use_fast=True)
    _model     = BlipForConditionalGeneration.from_pretrained(
        BLIP_CHECKPOINT, ignore_mismatched_sizes=True
    )
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(_device).eval()
    print(f"[BLIP] Model ready on {_device}.")


def load_traffic_models(yolo_size: str = "yolov8n") -> None:
    """
    Load YOLO detector, traffic-light classifier, ambulance detector,
    shoplifting analyser, and frame visualizer into global references.
    Called once at startup after load_blip_model().
    """
    global _yolo_detector, _tl_classifier, _amb_detector
    global _shoplifting_analyzer, _frame_visualizer

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[TrafficModels] Loading YOLOv8 ({yolo_size}) on {device_str} ...")
    _yolo_detector       = YOLODetector(yolo_size, device_str)
    _tl_classifier       = TrafficLightClassifier()
    _amb_detector        = AmbulanceDetector()
    _shoplifting_analyzer = ShopliftingAnalyzer()
    _frame_visualizer    = TrafficFrameVisualizer()
    print("[TrafficModels] All traffic models ready.")


# BLIP Inference Helpers
def blip_caption(img: Image.Image, prompt: str | None = None) -> str:
    """Generate a caption for the image, optionally conditioned on a prompt."""
    if not _processor or not _model:
        return "(BLIP unavailable)"
    with torch.no_grad():
        if prompt:
            inputs = _processor(images=img, text=prompt, return_tensors="pt").to(_device)
        else:
            inputs = _processor(images=img, return_tensors="pt").to(_device)
        out = _model.generate(
            **inputs,
            max_new_tokens=48,
            num_beams=3,
            length_penalty=1.0,
        )
    text = _processor.decode(out[0], skip_special_tokens=True)
    if prompt and text.lower().startswith(prompt.lower()):
        text = text[len(prompt):].strip()
    return text


def blip_vqa(img: Image.Image, question: str) -> str:
    """Answer a visual question about the image."""
    if not _processor or not _model:
        return "n/a"
    try:
        with torch.no_grad():
            inputs = _processor(images=img, text=question, return_tensors="pt").to(_device)
            out    = _model.generate(**inputs, max_new_tokens=6)
        return _processor.decode(out[0], skip_special_tokens=True).strip().lower()
    except Exception:
        return "n/a"


# Per-frame analysis (BLIP-based)
def analyse_frame(
    img:        Image.Image,
    is_anomaly: bool,
) -> tuple[str, str, str, dict[str, str]]:
    """
    Original BLIP-based per-frame analysis (caption + VQA + alert
    classification). Preserved exactly as in the original codebase,
    except that the threshold for vehicle on crosswalk has been lowered
    to require only one "yes" vote (instead of two) to increase sensitivity.
    """
    vqa: dict[str, str] = {}

    if is_anomaly:
        vqa["accident"] = blip_vqa(img, VQA_ACCIDENT)
        vqa["fire"]     = blip_vqa(img, VQA_FIRE)
    else:
        vqa["accident"] = "n/a"
        vqa["fire"]     = "n/a"

    vqa["ped_crosswalk"]   = blip_vqa(img, VQA_PED_CROSSWALK)
    vqa["vehicle_cross_1"] = blip_vqa(img, VQA_VEHICLE_CROSS_1)
    vqa["vehicle_cross_2"] = blip_vqa(img, VQA_VEHICLE_CROSS_2)

    vehicle_cross_votes         = (int(is_yes(vqa["vehicle_cross_1"]))
                                   + int(is_yes(vqa["vehicle_cross_2"])))
    # Lowered threshold: now only one "yes" is enough to confirm vehicle on crosswalk
    vehicle_crosswalk_confirmed = vehicle_cross_votes >= 1   # was >=2
    accident_confirmed          = is_yes(vqa["accident"])
    fire_confirmed              = is_yes(vqa["fire"])
    ped_crosswalk_confirmed     = is_yes(vqa["ped_crosswalk"])

    if vehicle_crosswalk_confirmed:
        vqa["is_motor_vehicle"] = blip_vqa(img, VQA_IS_MOTOR_VEHICLE)
        if not is_yes(vqa["is_motor_vehicle"]):
            vehicle_crosswalk_confirmed = False

    if vehicle_crosswalk_confirmed:
        caption = blip_caption(img, prompt=PROMPT_CROSSWALK)
    elif is_anomaly or accident_confirmed or fire_confirmed:
        caption          = blip_caption(img, prompt=PROMPT_ANOMALY)
        vqa["fall"]      = blip_vqa(img, VQA_FALL)
        vqa["wrong_way"] = blip_vqa(img, VQA_WRONG_WAY)
        vqa["severity"]  = blip_vqa(img, VQA_SEVERITY)
    else:
        caption          = blip_caption(img, prompt=PROMPT_NORMAL)
        vqa["fall"]      = "n/a"
        vqa["wrong_way"] = "n/a"
        vqa["severity"]  = "n/a"

    alert_label, severity = map_caption_to_alert(caption)

    # Override based on VQA confirmations (priority order)
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

    if is_anomaly or severity in ("CRITICAL", "HIGH"):
        sev_str = parse_severity(vqa.get("severity", "n/a"))
        if sev_str != "n/a":
            caption = f"{caption}  [severity: {sev_str}]"

    return caption, alert_label, severity, vqa


# Extended analysis that merges BLIP + YOLO traffic results
def analyse_frame_extended(
    img:          Image.Image,
    is_anomaly:   bool,
    traffic_info: dict | None = None,
) -> tuple[str, str, str, dict[str, str]]:
    """
    Calls the original analyse_frame() then overlays YOLO-based traffic
    analysis results.  If the traffic system detected something more severe,
    the alert label and severity are upgraded.

    Also runs additional VQA passes for ambulance obstruction, red-light
    running, stop-line crossing, and shoplifting when traffic_info signals
    anomalies.
    """
    caption, alert_label, severity, vqa = analyse_frame(img, is_anomaly)

    # Additional VQA for newly detected conditions
    run_extra_vqa = (
        is_anomaly
        or (traffic_info and traffic_info.get("score", 0) >= 0.25)
    )

    if run_extra_vqa:
        # Ambulance detection via VQA
        vqa["ambulance_detect"] = blip_vqa(img, VQA_AMBULANCE_DETECT)
        if is_yes(vqa["ambulance_detect"]):
            vqa["amb_blocked"] = blip_vqa(img, VQA_AMB_BLOCKED)
            vqa["amb_stuck"]   = blip_vqa(img, VQA_AMBULANCE_STUCK)
            if is_yes(vqa.get("amb_blocked", "no")):
                if _SEV_ORDER.get("CRITICAL", 4) > _SEV_ORDER.get(severity, 0):
                    alert_label = "AMBULANCE BLOCKED"
                    severity    = "CRITICAL"
            elif is_yes(vqa.get("amb_stuck", "no")):
                if _SEV_ORDER.get("CRITICAL", 4) > _SEV_ORDER.get(severity, 0):
                    alert_label = "AMBULANCE STUCK IN TRAFFIC"
                    severity    = "CRITICAL"

        # Red-light running via VQA
        vqa["red_light_vqa"] = blip_vqa(img, VQA_RED_LIGHT)
        if is_yes(vqa.get("red_light_vqa", "no")):
            if _SEV_ORDER.get("CRITICAL", 4) > _SEV_ORDER.get(severity, 0):
                alert_label = "RED LIGHT VIOLATION"
                severity    = "CRITICAL"

        # Stop-line violation via VQA
        vqa["stop_line_vqa"] = blip_vqa(img, VQA_STOP_LINE)
        if is_yes(vqa.get("stop_line_vqa", "no")):
            if _SEV_ORDER.get("CRITICAL", 4) > _SEV_ORDER.get(severity, 0):
                alert_label = "STOP LINE VIOLATION"
                severity    = "CRITICAL"

        # Shoplifting via VQA
        vqa["shoplifting_vqa"] = blip_vqa(img, VQA_SHOPLIFTING)
        if is_yes(vqa.get("shoplifting_vqa", "no")):
            if _SEV_ORDER.get("CRITICAL", 4) > _SEV_ORDER.get(severity, 0):
                alert_label = "SHOPLIFTING IN STORE"
                severity    = "CRITICAL"

    # Merge YOLO-based traffic analysis results
    if traffic_info:
        t_sev   = traffic_info.get("severity", "LOW")
        t_type  = traffic_info.get("alert_type", "")
        t_score = traffic_info.get("score",     0.0)
        t_reason = traffic_info.get("reason",   "")

        if _SEV_ORDER.get(t_sev, 0) > _SEV_ORDER.get(severity, 0):
            alert_label = t_type or alert_label
            severity    = t_sev
            if t_reason:
                caption = f"{caption} [{t_reason[:55]}]"

        vqa["traffic_alert"] = t_type
        vqa["traffic_score"] = str(round(t_score, 3))
        vqa["tl_state"]      = traffic_info.get("tl_state", "unknown")

    return caption, alert_label, severity, vqa

# Video Frame Extraction
def extract_frames(
    video_path:  str,
    session_dir: str,
) -> list[dict]:
    """Extract frames from video at EXTRACT_FPS rate."""
    print(f"[DEBUG] Trying to open video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}. Check if the file exists and is a supported format.")
        return []

    frames_dir = os.path.join(session_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    print(f"[DEBUG] frames_dir created: {frames_dir}")

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


# Anomaly Scoring 

def compute_anomaly_scores(
    frames: list[dict],
) -> tuple[list[float], list[float], float, float]:
    diffs: list[float] = [0.0]
    ssims: list[float] = [1.0]

    prev_arr  = np.array(frames[0]["image"].resize(DIFF_SIZE)).astype(np.float32)
    prev_gray = cv2.cvtColor(np.uint8(prev_arr), cv2.COLOR_RGB2GRAY)

    for i in range(1, len(frames)):
        curr_arr  = np.array(frames[i]["image"].resize(DIFF_SIZE)).astype(np.float32)
        curr_gray = cv2.cvtColor(np.uint8(curr_arr), cv2.COLOR_RGB2GRAY)

        diffs.append(float(np.mean(np.abs(curr_arr - prev_arr))))
        ssims.append(float(ssim(curr_gray, prev_gray, data_range=255)))

        prev_arr, prev_gray = curr_arr, curr_gray

    arr_d = np.array(diffs)
    arr_s = np.array(ssims)

    return (
        diffs, ssims,
        float(arr_d.mean() + 1.5 * arr_d.std()),
        float(arr_s.mean() - 1.5 * arr_s.std()),
    )


def label_frames(
    frames: list[dict],
    diffs:  list[float],
    ssims:  list[float],
    thr_d:  float,
    thr_s:  float,
) -> list[dict]:
    low_d        = thr_d * 0.5
    high_s       = thr_s * 1.5
    anom_active  = False
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


# Screenshot Persistence
def _sanitise_folder_name(name: str) -> str:
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
    """Save an annotated screenshot and return base64 and file path."""
    img_copy = frame_data["image"].copy()
    draw     = ImageDraw.Draw(img_copy)
    w, h     = img_copy.size

    palette = {
        "yellow": ((140, 120,   0), (  0,   0,   0), ( 20,  20,  20)),
        "orange": ((150,  55,   0), (255, 255, 255), (255, 210, 160)),
        "red":    ((120,   8,   8), (255, 210,  50), (255, 255, 255)),
    }
    fill, txt_col, cap_col = palette.get(color, palette["red"])

    draw.rectangle([(0, h - 52), (w, h)], fill=fill)
    draw.text((6, h - 50), alert_label,  fill=txt_col)
    draw.text((6, h - 30), caption[:90], fill=cap_col)

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    fnum      = result["frame_idx"]
    base_name = f"{ts}_frame{fnum:05d}_{result['time_sec']:.2f}s"

    alert_dir = os.path.join(OUTPUT_ROOT, _sanitise_folder_name(alert_label))
    os.makedirs(alert_dir, exist_ok=True)

    session_anom_dir = os.path.join(session_dir, "anomalies")
    os.makedirs(session_anom_dir, exist_ok=True)

    jpg_path  = os.path.join(alert_dir,        f"{base_name}.jpg")
    txt_path  = os.path.join(alert_dir,        f"{base_name}.txt")
    temp_path = os.path.join(session_anom_dir, f"{base_name}.jpg")

    img_copy.save(jpg_path,  quality=JPEG_QUALITY)
    img_copy.save(temp_path, quality=JPEG_QUALITY)

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

    buf = BytesIO()
    img_copy.save(buf, format="JPEG", quality=JPEG_QUALITY)
    b64_img = base64.b64encode(buf.getvalue()).decode()

    return b64_img, jpg_path

# Visualisation Helpers 
def create_frame_grid(
    frames:        list[dict],
    frame_results: list[dict],
    captions:      list[str],
    session_dir:   str,
) -> str:
    N_COLS = 5
    TW, TH, LH = 220, 135, 85

    n_rows = math.ceil(len(frames) / N_COLS)
    grid   = Image.new("RGB", (N_COLS * TW, n_rows * (TH + LH)), (10, 12, 20))
    draw   = ImageDraw.Draw(grid)

    for pos, fr in enumerate(frames):
        res  = frame_results[pos]
        cap  = captions[pos] if pos < len(captions) else ""
        col  = pos % N_COLS
        row  = pos // N_COLS
        x, y = col * TW, row * (TH + LH)

        grid.paste(fr["image"].resize((TW, TH), Image.LANCZOS), (x, y))

        sev   = res.get("severity", "NORMAL")
        alert = res.get("alert", "")
        anom  = res["is_anomaly"]

        if alert == "VEHICLE BLOCKING CROSSWALK":
            fc, oc, lc, label = (100,40,0), (220,90,0),  (255,130,30), "VEH ON XWALK"
        elif sev == "YELLOW":
            fc, oc, lc, label = ( 80,70,0), (200,180,0), (240,215, 0), "PED XWALK"
        elif anom:
            fc, oc, lc, label = (100, 8,8), (200, 25,25),(255, 70,70), "ANOMALY"
        else:
            fc, oc, lc, label = (  8,50,18),( 20,140,50),( 60,240,90), "Normal"

        draw.rectangle([(x, y+TH), (x+TW-1, y+TH+LH-1)], fill=fc)
        draw.text((x+4, y+TH+ 3), label,                              fill=lc)
        draw.text((x+4, y+TH+18), f"t={res['time_sec']:.1f}s  D={res['difference']:.1f}", fill=(180,180,180))
        draw.text((x+4, y+TH+34), alert[:30],                         fill=(240,190,50) if (anom or sev=="YELLOW") else (130,130,180))
        draw.text((x+4, y+TH+52), (cap[:50]+"...") if len(cap)>50 else cap, fill=(160,160,230))
        draw.rectangle([(x,y),(x+TW-1,y+TH+LH-1)], outline=oc, width=3)

    path = os.path.join(session_dir, "grid.jpg")
    grid.save(path, quality=JPEG_QUALITY)
    return path


def create_anomaly_timeline(
    frame_results: list[dict],
    thr_d:         float,
    session_dir:   str,
) -> str:
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
                textcoords="offset points", xytext=(4, 5),
                fontsize=5.5, color="#fca5a5",
            )
        elif sev == "YELLOW":
            ax.scatter(r["time_sec"], r["difference"], color="#fbbf24", s=55, zorder=5, marker="^")

    ax.set_title("Anomaly Score  Frame Difference Over Time",
                 color="#e2e8f0", fontsize=12, pad=10, fontfamily="monospace")
    ax.set_xlabel("Time (s)", color="#94a3b8")
    ax.set_ylabel("Frame D",  color="#94a3b8")
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


# PIL / OpenCV conversion helpers
def _pil_to_bgr(img: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to a BGR numpy array (for OpenCV / YOLO)."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert a BGR numpy array back to a PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


# Sequential alert types 
_SEQUENTIAL_ALERT_TYPES: list[tuple[str, str]] = [
    ("traffic accident",           "TRAFFIC ACCIDENT"),
    ("stop line violation",        "STOP LINE VIOLATION"),
    ("red light violation",        "RED LIGHT VIOLATION"),
    ("ambulance stuck",            "AMBULANCE STUCK IN TRAFFIC"),
    ("people stealing",            "THEFT / STEALING"),
    ("suspicious activity",        "SUSPICIOUS ACTIVITY"),
    ("vehicle blocking crosswalk", "VEHICLE BLOCKING CROSSWALK"),
]

_SEQUENTIAL_ALERT_TYPES.extend([
    ("ambulance blocked",          "AMBULANCE BLOCKED"),
    ("failure to yield",           "FAILURE TO YIELD"),
    ("intersection blocked",       "INTERSECTION BLOCKED"),
    ("shoplifting",                "SHOPLIFTING DETECTED"),
    ("person stealing",            "STORE THEFT DETECTED"),
    ("red_light_violation",        "RED LIGHT VIOLATION"),
    ("ambulance_stuck",            "AMBULANCE STUCK"),
    ("ambulance_blocked_critical", "AMBULANCE BLOCKED  CRITICAL"),
    ("not_yielding",               "FAILURE TO YIELD TO AMBULANCE"),
    ("collision",                  "VEHICLE COLLISION"),
    ("crash",                      "VEHICLE CRASH"),
    ("wrong-way",                  "WRONG-WAY VEHICLE"),
])


# Main Streaming Pipeline (with YOLO)
def process_video_streaming(
    video_path: str,
    session_id: str,
    sio:        SocketIO,
) -> None:
    """
    Background thread: extract, score, label, analyse, emit.
    ORIGINAL logic is preserved in its entirety.
    NEW traffic-detection logic is inserted at appropriate points.
    """
    session_dir = os.path.join(
        app.config["UPLOAD_FOLDER"],
        f"multianomaly_{session_id}",
    )
    os.makedirs(session_dir, exist_ok=True)

    # New per-session traffic analysis state
    session_traffic_analyzer: "TrafficAnalyzer | None"   = (
        TrafficAnalyzer() if _yolo_detector else None
    )
    session_shoplifting_analyzer: "ShopliftingAnalyzer | None" = (
        ShopliftingAnalyzer() if _yolo_detector else None
    )
    session_event_logger: "TrafficEventLogger | None" = (
        TrafficEventLogger(
            Path(os.path.join(session_dir, "traffic_logs"))
        ) if _yolo_detector else None
    )
    session_ambulance_tids: list[int] = []   # confirmed ambulance track IDs

    try:
        # 1. Extract frames (original)
        frames = extract_frames(video_path, session_dir)
        if not frames:
            sio.emit("error",
                     {"message": "No frames could be extracted from video."},
                     room=session_id)
            return

        # 2. Score and label anomalies (original)
        diffs, ssims, thr_d, thr_s = compute_anomaly_scores(frames)
        frame_results              = label_frames(frames, diffs, ssims, thr_d, thr_s)
        per_frame_captions: list[str] = []

        # 3. Per-frame analysis loop
        for i, res in enumerate(frame_results):
            img     = frames[i]["image"]
            is_anom = res["is_anomaly"]

            # Original BLIP analysis
            caption, alert_label, severity, vqa = analyse_frame(img, is_anom)
            per_frame_captions.append(caption)

            # Original VQA-based anomaly promotion
            if (is_yes(vqa.get("accident", "no"))
                    or is_yes(vqa.get("fire", "no"))
                    or alert_label == "VEHICLE BLOCKING CROSSWALK"):
                is_anom = True
                frame_results[i]["is_anomaly"] = True

            is_ped_crosswalk     = (severity == "YELLOW"
                                    or alert_label == "PEDESTRIAN ON CROSSWALK")
            is_vehicle_crosswalk = (alert_label == "VEHICLE BLOCKING CROSSWALK")

            # New: YOLO + traffic / shoplifting analysis
            yolo_dets:    list[TrafficDetection] = []
            traffic_info: dict                   = {}
            dominant_tl:  str                    = "unknown"
            annotated_pil: Image.Image           = img   # default: original frame

            if _yolo_detector and _frame_visualizer:
                frame_bgr = _pil_to_bgr(img)
                h_f, w_f  = frame_bgr.shape[:2]

                # YOLO detection
                yolo_dets = _yolo_detector.detect(frame_bgr, conf=0.35)

                # Traffic-light classification
                tl_states: list[str] = []
                for det in yolo_dets:
                    if det.class_name == "traffic light":
                        state = _tl_classifier.classify_from_frame(
                            frame_bgr, det.bbox)
                        det.extra["tl_state"] = state
                        tl_states.append(state)
                for preferred in ("red", "yellow", "green"):
                    if preferred in tl_states:
                        dominant_tl = preferred
                        break

                # Ambulance detection (colour/shape heuristic)
                for det in yolo_dets:
                    if det.class_name in ("car", "truck", "bus"):
                        x1,y1,x2,y2 = det.bbox
                        x1,y1 = max(0,x1), max(0,y1)
                        x2,y2 = min(w_f,x2), min(h_f,y2)
                        crop = frame_bgr[y1:y2, x1:x2]
                        is_amb, amb_conf = _amb_detector.is_ambulance_crop(crop)
                        if is_amb:
                            det.extra["is_ambulance"] = True
                            det.extra["amb_conf"]     = amb_conf
                            if (det.track_id is not None
                                    and det.track_id not in session_ambulance_tids):
                                session_ambulance_tids.append(det.track_id)

                # Cross-reference ambulance via BLIP caption
                if _amb_detector.caption_mentions_ambulance(caption):
                    vqa["blip_ambulance"] = "yes"

                # Traffic behavioural analysis (with BLIP caption)
                t_score, t_type, t_reason, t_kws = \
                    session_traffic_analyzer.analyse(
                        yolo_dets, caption, i, frame_bgr,
                        session_ambulance_tids,
                    )
                t_sev = session_traffic_analyzer.severity(t_score)

                # Shoplifting analysis (uses BLIP caption + YOLO bboxes)
                sh_score, sh_kws = session_shoplifting_analyzer.analyse(
                    yolo_dets, caption, frame_bgr)
                if sh_score >= 0.35:
                    if _SEV_ORDER.get("CRITICAL", 4) > _SEV_ORDER.get(t_sev, 0):
                        t_type   = "SHOPLIFTING_DETECTED"
                        t_sev    = "CRITICAL"
                        t_score  = max(t_score, sh_score)
                        t_reason = t_reason + "; Shoplifting: " + ",".join(sh_kws)
                        t_kws.extend(sh_kws)

                traffic_info = {
                    "score":             t_score,
                    "alert_type":        t_type,
                    "severity":          t_sev,
                    "reason":            t_reason,
                    "keywords":          list(set(t_kws)),
                    "tl_state":          dominant_tl,
                    "ambulance_present": bool(session_ambulance_tids),
                }

                # Promote frame to anomaly if YOLO found something significant
                if t_score >= 0.25:
                    is_anom = True
                    frame_results[i]["is_anomaly"] = True

                # Upgrade BLIP alert if YOLO found something more severe
                if _SEV_ORDER.get(t_sev, 0) > _SEV_ORDER.get(severity, 0):
                    alert_label = t_type or alert_label
                    severity    = t_sev
                    if t_reason:
                        caption = f"{caption} [{t_reason[:55]}]"
                    # Update caption in list
                    if per_frame_captions:
                        per_frame_captions[-1] = caption

                # Draw bounding boxes + status overlay on annotated copy
                annotated_bgr = _frame_visualizer.draw(
                    frame_bgr, yolo_dets, caption,
                    t_score, t_sev, t_type,
                    i, EXTRACT_FPS, dominant_tl,
                )
                annotated_pil = _bgr_to_pil(annotated_bgr)

                # Log to CSV/JSON if threshold exceeded
                if (session_event_logger and t_score >= 0.25
                        and session_traffic_analyzer.should_alert(t_type, i)):
                    veh_count = len([d for d in yolo_dets
                                     if d.class_name in VEHICLE_NAMES])
                    t_event = TrafficAlertEvent(
                        timestamp           = datetime.now().strftime(
                                                 "%Y-%m-%d %H:%M:%S"),
                        frame_number        = i,
                        video_time_seconds  = round(res["time_sec"], 2),
                        alert_type          = t_type,
                        severity            = t_sev,
                        description         = t_reason,
                        blip_caption        = caption,
                        confidence          = round(t_score, 4),
                        vehicle_count       = veh_count,
                        ambulance_present   = bool(session_ambulance_tids),
                        traffic_light_state = dominant_tl,
                        suspicious_keywords = list(set(t_kws)),
                    )
                    session_event_logger.log(t_event)

            # Original: sequential alert detection
            seq_alert = seq_type = None
            for typ, label in _SEQUENTIAL_ALERT_TYPES:
                if typ in caption.lower() or typ in alert_label.lower():
                    seq_alert, seq_type = label, typ
                    break

            # Original: save screenshot (use annotated_pil if available)
            screenshot = None
            if is_anom or is_ped_crosswalk:
                if is_vehicle_crosswalk:
                    color = "orange"
                elif is_ped_crosswalk:
                    color = "yellow"
                else:
                    color = "red"

                # Substitute annotated frame for screenshot if available
                save_frame_data = {**frames[i], "image": annotated_pil}
                screenshot, saved_path = save_alert_screenshot(
                    save_frame_data, res, caption,
                    alert_label, severity, session_dir, color,
                )

            # Build frame payload (extended with traffic fields)
            timestamp_str = (f"{int(res['time_sec']//60):02d}:"
                             f"{int(res['time_sec']%60):02d}")
            frame_payload = {
                "frame_idx":             res["frame_idx"],
                "time_sec":              res["time_sec"],
                "timestamp":             timestamp_str,
                "caption":               caption,
                "is_anomaly":            is_anom,
                "alert":                 alert_label,
                "severity":              severity,
                "difference":            res["difference"],
                "is_ped_crosswalk":      is_ped_crosswalk,
                "is_vehicle_crosswalk":  is_vehicle_crosswalk,
                "sequential_type":       seq_type,
                "sequential_alert":      seq_alert,
                # New traffic fields
                "traffic_score":       traffic_info.get("score", 0),
                "traffic_alert_type":  traffic_info.get("alert_type", ""),
                "traffic_severity":    traffic_info.get("severity", "LOW"),
                "traffic_tl_state":    dominant_tl,
                "yolo_detection_count": len(yolo_dets),
                "ambulance_detected":  bool(session_ambulance_tids),
            }
            if screenshot:
                frame_payload["screenshot"] = screenshot

            # Original: emit specialised events
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

            # New: emit traffic-specific SocketIO events
            if traffic_info and traffic_info.get("score", 0) >= 0.25:
                t_sev_emit  = traffic_info.get("severity", "LOW")
                t_type_emit = traffic_info.get("alert_type", "")

                sio.emit("traffic_alert", {
                    "type":       t_type_emit,
                    "timestamp":  timestamp_str,
                    "caption":    caption,
                    "severity":   t_sev_emit,
                    "frame_idx":  res["frame_idx"],
                    "score":      round(traffic_info.get("score", 0), 3),
                    "tl_state":   dominant_tl,
                    "keywords":   traffic_info.get("keywords", []),
                    "yolo_count": len(yolo_dets),
                    "screenshot": screenshot or "",
                }, room=session_id)

                # Dedicated ambulance alert channel
                if (session_ambulance_tids
                        or "ambulance" in t_type_emit.lower()
                        or vqa.get("blip_ambulance") == "yes"):
                    sio.emit("ambulance_alert", {
                        "type":       t_type_emit,
                        "timestamp":  timestamp_str,
                        "caption":    caption,
                        "severity":   t_sev_emit,
                        "frame_idx":  res["frame_idx"],
                        "tl_state":   dominant_tl,
                        "screenshot": screenshot or "",
                    }, room=session_id)

                # Dedicated shoplifting alert channel
                if "SHOPLIFTING" in t_type_emit.upper():
                    sio.emit("shoplifting_alert", {
                        "type":       "SHOPLIFTING_DETECTED",
                        "timestamp":  timestamp_str,
                        "caption":    caption,
                        "severity":   "CRITICAL",
                        "frame_idx":  res["frame_idx"],
                        "screenshot": screenshot or "",
                    }, room=session_id)

            # Original: emit main frame event
            sio.emit("frame", frame_payload, room=session_id)

            # Original: persist enriched result
            res.update({
                "caption":  caption,
                "alert":    alert_label,
                "severity": severity,
                "vqa":      vqa,
            })

        # 4. Generate summary visualisations (original)
        grid_path     = create_frame_grid(
            frames, frame_results, per_frame_captions, session_dir)
        timeline_path = create_anomaly_timeline(
            frame_results, thr_d, session_dir)

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
            # New traffic summary
            "ambulance_track_ids": session_ambulance_tids,
            "traffic_log_available": session_event_logger is not None,
        }, room=session_id)

    except Exception as exc:
        sio.emit("error", {"message": str(exc)}, room=session_id)
        raise

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
        active_sessions.pop(session_id, None)


# Flask Routes
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
    session_dir = os.path.join(
        app.config["UPLOAD_FOLDER"], f"multianomaly_{session_id}")
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


# SocketIO Event Handlers
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


# Entry Point
if __name__ == "__main__":
    load_blip_model()
    load_traffic_models(yolo_size="yolov8n")
    socketio.run(app, debug=False, host="0.0.0.0", port=5000)