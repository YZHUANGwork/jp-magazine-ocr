#!/usr/bin/env python3
"""
server.py  —  Web UI for Japanese text extraction
--------------------------------------------------
Endpoints:
  POST /detect   — local CV or AI text-region detection → [{px,py,pw,ph}]
  POST /extract  — JP OCR on a cropped region → {regions, text}
  GET  /         — serves static/index_fullstack.html

Run:
  python server.py
  open http://localhost:5000

Logs: server.log (same directory as this file)
"""

import base64
import io
import json
import logging
import os
import re
import sys
import traceback
from logging.handlers import RotatingFileHandler

import numpy as np
import requests as http
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR  = os.path.join(BASE_DIR, "static")
LOG_FILE    = os.path.join(BASE_DIR, "server.log")
EAST_MODEL  = os.path.join(BASE_DIR, "frozen_east_text_detection.pb")

# ── Logging ───────────────────────────────────────────────────────────────────
def setup_logging():
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console handler (INFO+)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file handler (DEBUG+), max 5 MB × 3 files
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3,
                             encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

setup_logging()
log = logging.getLogger(__name__)

app = Flask(__name__, static_folder=STATIC_DIR)

# ── PIL Resampling compat (Pillow <10 vs ≥10) ─────────────────────────────────
try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    _LANCZOS = Image.LANCZOS  # type: ignore

# ── EasyOCR — lazy singleton ──────────────────────────────────────────────────
_easyocr_reader = None

def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            log.info("Loading EasyOCR ja+en model (first run may download ~500 MB)…")
            _easyocr_reader = easyocr.Reader(["ja", "en"], gpu=False, verbose=False)
            log.info("EasyOCR ready.")
        except ImportError:
            raise RuntimeError("easyocr not installed. Run:  pip install easyocr")
    return _easyocr_reader

# ── OCR prompt ────────────────────────────────────────────────────────────────
EXTRACT_PROMPT = """You are a Japanese OCR specialist.

Look at this image carefully. Find every region that contains Japanese text
(kanji, hiragana, katakana, or mixed with romaji/numbers).

For each text region return a JSON object with:
  "orientation" : "vertical"   — text runs top-to-bottom in right-to-left columns
                | "horizontal" — text runs left-to-right
  "text"        : the exact Japanese text as written, preserving line breaks with \\n

Order the regions as a human would read them:
  - vertical layout  → right column first, then left (right-to-left, top-to-bottom)
  - horizontal layout → top-to-bottom, left-to-right

Return ONLY a valid JSON array, no markdown, no explanation. Example:
[
  {"orientation": "vertical",   "text": "こんにちは\\n世界"},
  {"orientation": "horizontal", "text": "第一章"}
]

If there is no Japanese text, return an empty array [].
"""

DETECT_PROMPT = """Look at this image carefully.
Find every region that contains text of ANY kind (any language, printed or handwritten).

Return image dimensions first, then bounding boxes for each text region.

Return ONLY a valid JSON object in this exact format — no markdown, no explanation:
{
  "img_w": <full image width in pixels>,
  "img_h": <full image height in pixels>,
  "regions": [
    {"x": <left px>, "y": <top px>, "w": <width px>, "h": <height px>},
    ...
  ]
}

If no text is found, return {"img_w": <w>, "img_h": <h>, "regions": []}.
"""

# ── Image helpers ─────────────────────────────────────────────────────────────
def pil_to_b64(img: Image.Image) -> str:
    """Re-encode PIL image as JPEG for API calls. Use raw_bytes_to_b64 when possible."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode()


def raw_bytes_to_b64(img_bytes: bytes) -> tuple:
    """
    Return (b64_string, mime_type) from raw image bytes WITHOUT re-encoding.
    Detects format from the file header so PNG/WEBP/BMP stay lossless.
    Falls back to JPEG if format is unrecognised.
    """
    # Detect format by magic bytes
    if img_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        mime = "image/png"
    elif img_bytes[:2] == b'\xff\xd8':
        mime = "image/jpeg"
    elif img_bytes[:4] == b'RIFF' and img_bytes[8:12] == b'WEBP':
        mime = "image/webp"
    elif img_bytes[:2] in (b'BM',):
        # BMP — convert to PNG for API compatibility
        img  = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        buf  = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode(), "image/png"
    else:
        mime = "image/jpeg"   # assume JPEG for unknowns
    return base64.b64encode(img_bytes).decode(), mime


def decode_image_bytes(img_bytes: bytes):
    """Decode raw bytes → OpenCV BGR ndarray."""
    import cv2
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("cv2.imdecode failed — unsupported image format")
    return img


# ── AI helpers ────────────────────────────────────────────────────────────────
def call_gemini(b64: str, api_key: str, prompt: str,
                mime_type: str = "image/jpeg") -> str:
    payload = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": mime_type, "data": b64}},
            {"text": prompt},
        ]}],
        "generationConfig": {"maxOutputTokens": 4096, "temperature": 0}
    }
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"gemini-2.0-flash:generateContent?key={api_key}")
    r = http.post(url, headers={"Content-Type": "application/json"},
                  json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def call_gpt(b64: str, api_key: str, prompt: str,
             mime_type: str = "image/jpeg") -> str:
    payload = {
        "model": "gpt-4o",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}]
    }
    r = http.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json=payload, timeout=60
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def ai_call(provider: str, b64: str, api_key: str, prompt: str,
            mime_type: str = "image/jpeg") -> str:
    if provider == "gemini":
        return call_gemini(b64, api_key, prompt, mime_type)
    return call_gpt(b64, api_key, prompt, mime_type)


def strip_markdown_json(raw: str) -> str:
    raw = re.sub(r"^```[a-z]*\s*", "", raw.strip(), flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    return raw.strip()


def parse_json_list(raw: str) -> list:
    raw = strip_markdown_json(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        return json.loads(m.group(0)) if m else []


def format_extract_text(regions: list) -> str:
    if not regions:
        return "— No Japanese words identified —"
    lines = []
    for i, r in enumerate(regions, 1):
        lines.append(f"[{i}] ({r.get('orientation','?')})")
        lines.append(r.get("text", "").strip())
        lines.append("")
    lines.append("─" * 40)
    lines.append("Full text in reading order:\n")
    lines.append("\n".join(r.get("text", "").strip() for r in regions))
    return "\n".join(lines)


# ── Box helpers ───────────────────────────────────────────────────────────────
def merge_boxes(boxes: list, gap_factor: float = 0.8) -> list:
    """
    Iteratively merge any two [x1,y1,x2,y2] boxes whose gap on both axes
    is ≤ gap_factor × the shorter box height.
    """
    if not boxes:
        return []
    merged = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out  = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            a = list(merged[i])
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                b = merged[j]
                gap_x = max(0, max(a[0], b[0]) - min(a[2], b[2]))
                gap_y = max(0, max(a[1], b[1]) - min(a[3], b[3]))
                thresh = gap_factor * min(a[3]-a[1], b[3]-b[1])
                if gap_x <= thresh and gap_y <= thresh:
                    a = [min(a[0],b[0]), min(a[1],b[1]),
                         max(a[2],b[2]), max(a[3],b[3])]
                    used[j] = True
                    changed  = True
            out.append(a)
        merged = out
    return merged


def nms_boxes(boxes: list, iou_thresh: float = 0.3) -> list:
    """Greedy NMS — keeps the largest box when two overlap heavily."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    keep  = []
    used  = [False] * len(boxes)
    for i in range(len(boxes)):
        if used[i]:
            continue
        a = boxes[i]
        keep.append(a)
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            b = boxes[j]
            ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
            ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
            if union > 0 and inter / union > iou_thresh:
                used[j] = True
    return keep


def boxes_to_regions(boxes: list) -> list:
    return [{"px": b[0], "py": b[1], "pw": b[2]-b[0], "ph": b[3]-b[1]}
            for b in boxes]


# ── Detection methods ─────────────────────────────────────────────────────────

def _easyocr_detect(img_bytes: bytes) -> list:
    """EasyOCR detection-only: polygons → merged AABBs."""
    import cv2
    img    = decode_image_bytes(img_bytes)           # BGR uint8
    # EasyOCR accepts BGR ndarray directly
    reader = get_easyocr_reader()

    log.debug("EasyOCR readtext start  size=%dx%d", img.shape[1], img.shape[0])
    results = reader.readtext(img, detail=1, paragraph=False)
    log.debug("EasyOCR returned %d raw results", len(results))

    raw = []
    for (polygon, _text, conf) in results:
        xs = [float(pt[0]) for pt in polygon]
        ys = [float(pt[1]) for pt in polygon]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        if (x2 - x1) > 4 and (y2 - y1) > 4:
            raw.append([x1, y1, x2, y2])
            log.debug("  box conf=%.2f  [%d,%d,%d,%d]", conf, x1, y1, x2, y2)

    raw.sort(key=lambda b: (b[1], b[0]))
    merged = merge_boxes(raw, gap_factor=0.8)
    log.info("EasyOCR: %d raw → %d merged boxes", len(raw), len(merged))
    return boxes_to_regions(merged)


def _east_detect(img_bytes: bytes) -> list:
    """
    EAST deep text detector.
    Requires frozen_east_text_detection.pb next to server.py.
    Download:
      https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb
    """
    import cv2

    if not os.path.exists(EAST_MODEL):
        raise RuntimeError(
            f"EAST model not found at {EAST_MODEL}.\n"
            "Download: https://github.com/oyyd/frozen_east_text_detection.pb"
            "/raw/master/frozen_east_text_detection.pb"
        )

    img      = decode_image_bytes(img_bytes)
    orig_h, orig_w = img.shape[:2]
    log.debug("EAST input size: %dx%d", orig_w, orig_h)

    # EAST needs multiples of 32; cap at 1280 to avoid OOM
    target_w = min(1280, max(32, (orig_w // 32) * 32))
    target_h = min(1280, max(32, (orig_h // 32) * 32))
    resized  = cv2.resize(img, (target_w, target_h))
    ratio_w  = orig_w / target_w
    ratio_h  = orig_h / target_h

    blob = cv2.dnn.blobFromImage(
        resized, 1.0, (target_w, target_h),
        (123.68, 116.78, 103.94), swapRB=True, crop=False
    )
    net = cv2.dnn.readNet(EAST_MODEL)
    net.setInput(blob)

    # IMPORTANT: forward() returns outputs sorted by layer name alphabetically.
    # "feature_fusion/concat_3"        < "feature_fusion/Conv_7/Sigmoid"
    # So index 0 = geometry (concat_3), index 1 = scores (Conv_7/Sigmoid).
    # Use a dict to be safe.
    out_names = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    outs      = net.forward(out_names)
    out_map   = dict(zip(out_names, outs))
    scores_map = out_map["feature_fusion/Conv_7/Sigmoid"]   # (1,1,H/4,W/4)
    geometry   = out_map["feature_fusion/concat_3"]         # (1,5,H/4,W/4)
    # geometry channels: 0=d_top 1=d_right 2=d_bottom 3=d_left 4=angle

    rows, cols = scores_map.shape[2], scores_map.shape[3]
    raw = []

    for y in range(rows):
        for x in range(cols):
            score = float(scores_map[0, 0, y, x])
            if score < 0.5:
                continue

            d_top    = float(geometry[0, 0, y, x])
            d_right  = float(geometry[0, 1, y, x])
            d_bottom = float(geometry[0, 2, y, x])
            d_left   = float(geometry[0, 3, y, x])
            angle    = float(geometry[0, 4, y, x])

            # Anchor point in resized-image space (each cell = 4 px)
            offset_x = x * 4.0 + 2.0
            offset_y = y * 4.0 + 2.0

            cos_a = np.cos(angle)
            sin_a = np.sin(angle)

            # Top-right corner of the rotated box
            # (EAST convention: offset is bottom-right anchor of the 4-px cell)
            ex = offset_x + cos_a * d_right + sin_a * d_bottom
            ey = offset_y - sin_a * d_right + cos_a * d_bottom

            # The four corners of the rotated box
            p1x = ex - cos_a * (d_right + d_left)  + sin_a * (d_top + d_bottom)
            p1y = ey + sin_a * (d_right + d_left)  + cos_a * (d_top + d_bottom)
            p2x = ex - cos_a * (d_right + d_left)
            p2y = ey + sin_a * (d_right + d_left)
            p3x = ex
            p3y = ey
            p4x = ex + sin_a * (d_top + d_bottom)
            p4y = ey + cos_a * (d_top + d_bottom)

            # Axis-aligned bounding box
            xs_pts = [p1x, p2x, p3x, p4x]
            ys_pts = [p1y, p2y, p3y, p4y]
            x1 = int(max(0,      min(xs_pts) * ratio_w))
            y1 = int(max(0,      min(ys_pts) * ratio_h))
            x2 = int(min(orig_w, max(xs_pts) * ratio_w))
            y2 = int(min(orig_h, max(ys_pts) * ratio_h))

            if (x2 - x1) > 4 and (y2 - y1) > 4:
                raw.append([x1, y1, x2, y2])

    log.debug("EAST: %d raw boxes before NMS", len(raw))
    raw    = nms_boxes(raw, iou_thresh=0.3)
    merged = merge_boxes(raw, gap_factor=0.5)
    log.info("EAST: %d after NMS → %d merged", len(raw), len(merged))
    return boxes_to_regions(merged)


def _mser_detect(img_bytes: bytes) -> list:
    """MSER text region detector — only needs opencv-python."""
    import cv2

    img    = decode_image_bytes(img_bytes)
    orig_h, orig_w = img.shape[:2]
    log.debug("MSER input size: %dx%d", orig_w, orig_h)

    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Equalise contrast — helps MSER on low-contrast scans
    grey = cv2.equalizeHist(grey)

    # max_area: 2% of image area, minimum 14 400 (≈ 120×120)
    max_area = max(14400, int(orig_w * orig_h * 0.02))
    mser = cv2.MSER_create(5, 60, max_area)

    regions, bboxes = mser.detectRegions(grey)
    log.debug("MSER: %d raw regions", len(regions))

    if not len(regions):
        return []

    raw = []
    for pts in regions:
        hull     = cv2.convexHull(pts.reshape(-1, 1, 2))
        x, y, w, h = cv2.boundingRect(hull)
        # Filter: min size, sane aspect ratio (0.05–20), not the whole image
        if (w > 8 and h > 8
                and 0.05 < (w / h) < 20
                and w < orig_w * 0.95
                and h < orig_h * 0.95):
            raw.append([x, y, x + w, y + h])

    raw    = nms_boxes(raw, iou_thresh=0.5)
    merged = merge_boxes(raw, gap_factor=0.6)
    log.info("MSER: %d raw → %d after NMS → %d merged",
             len(regions), len(raw), len(merged))
    return boxes_to_regions(merged)


def _ai_whole_detect(img_bytes: bytes, provider: str, api_key: str) -> list:
    """
    Single AI call on the full image — no resize, no re-encoding.
    Sends raw bytes as-is so the model sees the original quality.
    """
    # Get dimensions without decoding the full image
    pil_img  = Image.open(io.BytesIO(img_bytes))
    orig_w, orig_h = pil_img.size
    pil_img.close()
    log.debug("AI-whole input size: %dx%d", orig_w, orig_h)

    b64, mime_type = raw_bytes_to_b64(img_bytes)
    log.info("AI-whole: calling %s  mime=%s  %d bytes", provider, mime_type, len(img_bytes))

    raw = ai_call(provider, b64, api_key, DETECT_PROMPT, mime_type)
    log.debug("AI-whole raw response: %s", raw[:500])

    raw_clean = strip_markdown_json(raw)
    try:
        obj = json.loads(raw_clean)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw_clean, re.DOTALL)
        obj = json.loads(m.group(0)) if m else {}

    # Model reports the image size it saw — use to correct any internal resize
    ai_w = float(obj.get("img_w", orig_w))
    ai_h = float(obj.get("img_h", orig_h))
    sx   = orig_w / ai_w if ai_w else 1.0
    sy   = orig_h / ai_h if ai_h else 1.0
    log.debug("AI reported img size %dx%d  sx=%.4f sy=%.4f", int(ai_w), int(ai_h), sx, sy)

    regions = []
    for b in obj.get("regions", []):
        x = int(round(b.get("x", 0) * sx))
        y = int(round(b.get("y", 0) * sy))
        w = int(round(b.get("w", 0) * sx))
        h = int(round(b.get("h", 0) * sy))
        x = max(0, min(x, orig_w - 1))
        y = max(0, min(y, orig_h - 1))
        w = min(w, orig_w - x)
        h = min(h, orig_h - y)
        if w > 4 and h > 4:
            regions.append({"px": x, "py": y, "pw": w, "ph": h})
            log.debug("  region [%d,%d %dx%d]", x, y, w, h)

    log.info("AI-whole: %d regions returned", len(regions))
    return regions


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index_fullstack.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    method    = request.form.get("method", "easyocr").lower()
    img_bytes = request.files["image"].read()
    log.info("POST /detect  method=%s  image=%d bytes", method, len(img_bytes))

    try:
        if method == "east":
            regions = _east_detect(img_bytes)
            return jsonify({"regions": regions, "method": "east"})

        if method == "mser":
            regions = _mser_detect(img_bytes)
            return jsonify({"regions": regions, "method": "mser"})

        if method == "ai_whole":
            provider = request.form.get("provider", "gemini")
            api_key  = request.form.get("api_key", "").strip()
            if not api_key:
                return jsonify({"error": "api_key required for ai_whole"}), 400
            regions = _ai_whole_detect(img_bytes, provider, api_key)
            return jsonify({"regions": regions, "method": "ai_whole"})

        # Default / explicit "easyocr"
        try:
            regions = _easyocr_detect(img_bytes)
            return jsonify({"regions": regions, "method": "easyocr"})
        except Exception as e_ocr:
            log.warning("EasyOCR failed (%s); falling back to MSER", e_ocr)
            regions = _mser_detect(img_bytes)
            return jsonify({"regions": regions, "method": "mser_fallback",
                            "warning": str(e_ocr)})

    except Exception as e:
        log.error("detect error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/extract", methods=["POST"])
def extract():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    provider = request.form.get("provider", "gemini")
    api_key  = request.form.get("api_key", "").strip()
    if not api_key:
        return jsonify({"error": "API key is required"}), 400

    img_bytes = request.files["image"].read()
    log.info("POST /extract  provider=%s  image=%d bytes", provider, len(img_bytes))

    try:
        b64, mime_type = raw_bytes_to_b64(img_bytes)
        raw     = ai_call(provider, b64, api_key, EXTRACT_PROMPT, mime_type)
        regions = parse_json_list(raw)
        text    = format_extract_text(regions)
        log.info("extract: %d JP regions found", len(regions))
        return jsonify({"regions": regions, "text": text})

    except Exception as e:
        log.error("extract error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


def _easyocr_extract(img_bytes: bytes) -> list:
    """
    Run EasyOCR on a cropped image and return text regions.
    EasyOCR returns (polygon, text, confidence) tuples; we collect all
    detected text, filter low-confidence results, and return in reading order.
    """
    img    = decode_image_bytes(img_bytes)
    reader = get_easyocr_reader()

    results = reader.readtext(img, detail=1, paragraph=False)

    # Sort top-to-bottom, left-to-right (reading order for horizontal text)
    results = sorted(results, key=lambda r: (
        min(pt[1] for pt in r[0]),   # top y of bounding polygon
        min(pt[0] for pt in r[0])    # left x
    ))

    regions = []
    for (polygon, text, conf) in results:
        text = text.strip()
        if not text or conf < 0.1:
            continue
        # Determine if any CJK characters present
        has_cjk = any('\u3000' <= ch <= '\u9fff' or '\uff00' <= ch <= '\uffef' for ch in text)
        regions.append({
            "orientation": "horizontal",
            "text": text,
            "confidence": round(float(conf), 3),
            "has_cjk": has_cjk,
        })

    log.info("easyocr_extract: %d text regions found", len(regions))
    return regions


@app.route("/extract_local", methods=["POST"])
def extract_local():
    """
    Local OCR endpoint — no API key required.
    Uses EasyOCR to read text from the submitted image crop.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    img_bytes = request.files["image"].read()
    log.info("POST /extract_local  image=%d bytes", len(img_bytes))

    try:
        regions = _easyocr_extract(img_bytes)
        text    = format_extract_text(regions)
        return jsonify({"regions": regions, "text": text})
    except Exception as e:
        log.error("extract_local error: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    log.info("Starting server on http://localhost:%d  (log → %s)", port, LOG_FILE)
    app.run(host="0.0.0.0", port=port, debug=False)
