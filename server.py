#!/usr/bin/env python3
"""
server.py  —  Web UI for Japanese text extraction
--------------------------------------------------
Serves the single-page UI and exposes one endpoint:

  POST /extract
    form fields:
      image     : uploaded image file (cropped region from canvas)
      provider  : "gemini" | "gpt"
      api_key   : API key string

    returns JSON:
      { "regions": [...], "text": "formatted output string" }
      or on error:
      { "error": "message" }

Run:
  python server.py
  # open http://localhost:5000
"""

import base64
import io
import json
import os
import re
import sys
import traceback

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
import requests as http

app = Flask(__name__, static_folder="static")

# ── reuse extract_jp logic inline (no subprocess needed) ─────────────────────

PROMPT = """You are a Japanese OCR specialist.

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


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode()


def call_gemini(b64: str, api_key: str) -> str:
    payload = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
            {"text": PROMPT},
        ]}],
        "generationConfig": {"maxOutputTokens": 4096, "temperature": 0}
    }
    r = http.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}",
        headers={"Content-Type": "application/json"},
        json=payload, timeout=60
    )
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def call_gpt(b64: str, api_key: str) -> str:
    payload = {
        "model": "gpt-4o",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": PROMPT},
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


def parse_regions(raw: str) -> list:
    raw = re.sub(r"^```[a-z]*\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        return json.loads(m.group(0)) if m else []


def format_text(regions: list) -> str:
    if not regions:
        return "— No Japanese words identified —"
    lines = []
    for i, r in enumerate(regions, 1):
        orient = r.get("orientation", "?")
        text = r.get("text", "").strip()
        lines.append(f"[{i}] ({orient})")
        lines.append(text)
        lines.append("")
    lines.append("─" * 40)
    lines.append("Full text in reading order:")
    lines.append("")
    lines.append("\n".join(r.get("text", "").strip() for r in regions))
    return "\n".join(lines)


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/extract", methods=["POST"])
def extract():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400

        provider = request.form.get("provider", "gemini")
        api_key  = request.form.get("api_key", "").strip()

        if not api_key:
            return jsonify({"error": "API key is required"}), 400

        img_bytes = request.files["image"].read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        b64 = pil_to_b64(img)

        raw = call_gemini(b64, api_key) if provider == "gemini" else call_gpt(b64, api_key)
        regions = parse_regions(raw)
        text = format_text(regions)

        return jsonify({"regions": regions, "text": text})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting server on http://localhost:{port}")
    print("Press Ctrl+C to stop.")
    app.run(host="0.0.0.0", port=port, debug=False)
