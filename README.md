# jp-magazine-ocr

A single-page web tool for extracting and translating Japanese text from images using GPT-4o or Gemini 2.0 Flash.

Designed for Japanese magazines, manga, and illustrated spreads where text and pictures are densely interleaved — traditional **vertical columns** running right-to-left, **horizontal captions** and pull-quotes, speech bubbles, panel labels, and mixed-direction layouts all on the same page. Draw selection boxes around any region, and the tool correctly identifies the reading direction of each text block and returns them in proper reading order.

## Translate everything to Chinese, English, or both in one click — all from your browser. 辅助啃生肉，让啃生肉没那么痛苦。适用于未汉化的杂志切页，公式书，攻略，漫画，扫图。

---


## Example Output

Each viewer shows the annotated image on the left and the scrollable extracted text on the right. Raw files are in [`ocr_output/`](https://github.com/YZHUANGwork/jp-magazine-ocr/tree/main/ocr_output).

| Category | Example |
|----------|---------|
| Travel magazine JP+中文 | [→ viewer](https://yzhuangwork.github.io/jp-magazine-ocr/example_viewer.html?base=ISBN9784533116735_042) |
| Fan book | [→ viewer](https://yzhuangwork.github.io/jp-magazine-ocr/example_viewer.html?base=ISNB475772439X_047) |
| Music magazine interview | [→ viewer](https://yzhuangwork.github.io/jp-magazine-ocr/example_viewer.html?base=ISBN9784401651559_045) |
| Manga | *(coming soon)* |
| Photobook interview | *(coming soon)* |
| Band score interview | *(coming soon)* |
| Sewing magazine | *(coming soon)* |


---
## Two Versions

This repo contains two independent versions of the tool. Pick the one that fits your use case.

### `server_aionly.py` + `index_aionly.html` — AI-Only (Simple)

> **For reading assistance.** Upload an image, draw boxes over the text you want, get Japanese extraction and translation in one click. No local models, no extra installs beyond Flask.

- Single endpoint: `POST /extract` — forwards each image crop to GPT-4o or Gemini and returns Japanese text regions
- Results panel: copy all extracted text and translations to clipboard
- ~170 lines of Python

**To run:**
```
python3 server_aionly.py
```

---

### `server_fullstack.py` + `index_fullstack.html` — Full-Stack (Full)

> **For power users who want to compare and evaluate.** Run multiple detection methods to see how each one draws boxes. Run both AI and EasyOCR on the same boxes and compare what each extracts. Download results as an annotated image or a text file.

- Three endpoints: `POST /detect`, `POST /extract`, `POST /extract_local`
- Four auto-detection methods to compare box placement (none beats drawing manually — see below)
- Two OCR engines (AI and EasyOCR) — run both on the same boxes, close whichever result you don't want
- Download annotated image (PNG with boxes drawn on it) or export results as a text file
- Raw image bytes forwarded without re-encoding — PNG and WEBP stay lossless
- Rotating file logger (`server.log`, 5 MB × 3 files)
- ~660 lines of Python

**To run:**
```
python3 server_fullstack.py
```

---

## Features

### Both versions

- **Reads vertical and horizontal text in the same image** — each region is tagged with its orientation (`vertical` or `horizontal`) and ordered as a human would read it: right-to-left columns for vertical, top-to-bottom for horizontal
- **Box-based OCR** — drag to draw selection boxes around specific regions; isolate a speech bubble, a caption strip, or a body-text column without picking up surrounding artwork
- **Whole-image scan** — leave the canvas empty and click Identify JP to scan the entire page automatically
- **Multiple boxes per page** — draw as many boxes as needed; each is processed independently and numbered in the results panel
- **Two AI providers** — GPT-4o (OpenAI) and Gemini 2.0 Flash (Google)
- **Inline translation** — add Chinese (中文), English, or both translations alongside extracted text in one click
- **Image rotation** — rotate ±180° with a slider, useful for sideways or upside-down pages
- **Scroll to zoom** — pinch or scroll to zoom in/out for dense, small-print areas
- **Undo** — Ctrl+Z removes the most recently drawn box (before running OCR)
- **Click to highlight** — clicking a result block bold-frames its canvas box
- **Copy all** — copies all extracted text and translations to clipboard
- **No data stored** — your API key and images never leave your browser except for the direct API call to OpenAI/Google

### Full-stack version only

- **Identify Text Region** — run auto-detection methods to see where each one draws boxes; compare coverage across methods before running OCR
- **Two OCR engines on the same boxes** — run AI and EasyOCR independently, compare what each extracts, close whichever result you don't want to keep
- **Remove translations** — remove 中文, English, or both translations from all result blocks without re-running OCR
- **Download annotated image** — export a PNG of the image with all bounding boxes drawn on it
- **Download results as text file** — export all extracted text and translations as a `.txt` file

---

## Auto-Detection vs. Drawing Boxes Manually (full-stack)

The **Identify Text Region** button runs one of four detection methods and draws boxes on the canvas automatically. You can run different methods one after another to compare where each one places its boxes — useful for getting a feel for what each method picks up and misses.

That said, **none of the detection methods is as accurate as drawing boxes yourself.** Auto-detected boxes often land in odd places (artwork, margins, decorative elements), and running OCR on those gives meaningless results. The detection methods are provided for comparison and to save time on clean, simple layouts — not as a replacement for manual selection on complex pages.

The recommended workflow for dense or mixed layouts: draw boxes yourself over the regions you actually want, then run Identify JP.

| Detection Method | Type | Notes |
|-----------------|------|-------|
| **EasyOCR** | Local · Free | Neural text detector with Japanese support. Best quality. Requires `pip install easyocr`. Falls back to MSER automatically if unavailable. |
| **EAST + OpenCV** | Local · Free | Deep text detector, fast and accurate. Requires `frozen_east_text_detection.pb` placed next to `server_fullstack.py`. [(download)](https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb) |
| **MSER (OpenCV)** | Local · Free | Zero-install, built into OpenCV. Works best on clean scans and manga with high contrast. |
| **AI Quadtree** | API | Sends the full image to the AI provider to locate text regions. Configurable subdivision depth (2ⁿ tiles, default n=4). Uses API credits. |

---

## Comparing OCR Engines (full-stack)

The **Identify JP** button has an engine dropdown with two options:

- **AI (GPT-4o / Gemini)** — sends each box crop to the selected AI provider; high accuracy with reading-direction detection
- **EasyOCR (Local · Free)** — runs OCR entirely on your machine; no API key needed

You can run both engines on the same set of boxes and see both sets of results in the panel at the same time. Each result card is tagged with the engine that produced it (`AI` or `EasyOCR`). Close whichever results you don't want to keep.

---

## Project Structure

```
.
├── static/
│   ├── index_aionly.html       # Frontend for the AI-only version
│   └── index_fullstack.html    # Frontend for the full-stack version
├── server_aionly.py            # Flask backend — AI proxy only
├── server_fullstack.py         # Flask backend — detection + local OCR + AI
└── requirements.txt            # Python dependencies
```

---

## Setup

### Requirements

- Python 3.9+
- An OpenAI API key **or** a Google Gemini API key

For the full-stack version, local models are optional but unlock free offline OCR:
- EasyOCR: `pip install easyocr`
- EAST model: download [`frozen_east_text_detection.pb`](https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb) and place it next to `server_fullstack.py`

### Install

```
pip3 install -r requirements.txt
```

### Run

AI-only version:
```
python3 server_aionly.py
```

Full-stack version:
```
python3 server_fullstack.py
```

Then open <http://localhost:5000> in your browser. Port can be changed:
```
PORT=8080 python server_fullstack.py
```

---

## Usage

### AI-only version

1. **Upload an image** — click the drop zone or drag a JPG/PNG/WEBP file onto it
2. **Draw selection boxes** — click and drag on the image to mark text regions
   - Leave the canvas empty to scan the whole image instead
   - Ctrl+Z undoes the last drawn box (before running OCR)
   - Use the rotation slider if the page is tilted or sideways
3. **Choose a provider** — select GPT-4o or Gemini 2.0 Flash, then paste your API key
4. **Run extraction:**
   - **Identify JP** — extract Japanese text only
   - **+ 中文** — extract and translate to Chinese
   - **+ English** — extract and translate to English
   - **+ 中文 & English** — extract with both translations
5. **Copy** — copies all extracted text and translations to clipboard

### Full-stack version

1. **Upload an image** — click the drop zone or drag a JPG/PNG/WEBP file onto it
2. **(Optional) Compare detection methods** — click **Identify Text Region** (▾ to switch method) to auto-draw boxes; run different methods to compare where each one places boxes. Close or redraw any that landed in the wrong place. Or skip this step entirely and draw all boxes manually for best accuracy.
3. **Draw selection boxes** — click and drag to add or refine boxes
   - Ctrl+Z undoes the last drawn box (before running OCR)
   - Use the rotation slider if the page is tilted or sideways
4. **Choose a provider** — select GPT-4o or Gemini 2.0 Flash, then paste your API key
5. **Run extraction** (▾ on Identify JP to switch engine):
   - Run **AI** and **EasyOCR** on the same boxes to compare results; close whichever cards you don't want
   - **+ 中文 / + English / + 中文 & English** — add translations to current results
6. **Remove translations** — use **− 中文 & English** (▾ to select which language) to remove translations from all result blocks without re-running OCR
7. **Export:**
   - **⬛ Image+Boxes** — download a PNG of the image with all bounding boxes drawn on it
   - **⬇ Text** — download all extracted text and translations as a `.txt` file

---

## API Keys

Your API key is sent from the browser to your local Flask server, which forwards it to the provider. It is never stored on disk.

- **OpenAI** — get a key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys); requires access to `gpt-4o`
- **Google Gemini** — get a key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

---

## How It Works

### Extraction (both versions)
The frontend sends a cropped image of each drawn box (or the whole rotated image) to `POST /extract`. The server forwards it to the selected AI provider with a structured OCR prompt. The model identifies every Japanese text region, tags each with its `orientation`, and returns them in natural reading order.

In the full-stack version, raw bytes are forwarded without re-encoding so PNG and WEBP stay lossless. If EasyOCR engine is selected, the crop goes to `POST /extract_local` and OCR runs entirely locally.

### Detection (full-stack only)
For the three local methods, the frontend sends the full image to `POST /detect` with a `method` parameter (`easyocr`, `east`, `mser`). The server runs the chosen detector, applies NMS and gap-based box merging, and returns coordinates.

The **AI Quadtree** method works entirely client-side and never calls `POST /detect`. It divides the image into an initial grid of tiles and asks the AI "does this tile contain Japanese text?" for each one. Tiles that say yes are recursively split into quadrants, drilling deeper into areas that contain text and ignoring areas that do not, until the leaf regions are as small as the text areas themselves. Adjacent leaf regions are then merged into final bounding boxes.

### Translation (both versions)
Translations are handled by a second API call made directly from the browser.

---

## Server Endpoints

### AI-only (`server_aionly.py`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /` | GET | Serves `static/index_aionly.html` |
| `POST /extract` | POST | AI OCR on a cropped image → `{regions, text}` |

### Full-stack (`server_fullstack.py`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /` | GET | Serves `static/index_fullstack.html` |
| `POST /detect` | POST | Detects text regions → `[{px, py, pw, ph}]` |
| `POST /extract` | POST | AI OCR on a cropped image → `{regions, text}` |
| `POST /extract_local` | POST | EasyOCR on a cropped image (no API key) → `{regions, text}` |

---

## Dependencies

| Package | Needed by | Purpose |
|---------|-----------|---------|
| Flask | both | Local web server |
| Pillow | both | Image decoding and re-encoding |
| requests | both | HTTP calls to OpenAI / Gemini APIs |
| opencv-python | full-stack | EAST and MSER detection; image processing |
| numpy | full-stack | Array operations for detection pipelines |
| easyocr *(optional)* | full-stack | Local neural text detection and extraction |
| reportlab / img2pdf | full-stack | PDF export (optional, for future use) |

---

**example_compressed.mp4**

https://private-user-images.githubusercontent.com/202312669/598414499-d497b48d-2b54-40f5-802a-fe40acbbf9a6.mp4
