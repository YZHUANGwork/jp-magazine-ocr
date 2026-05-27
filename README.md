# jp-magazine-ocr
A lightweight Web UI for Japanese physical magazine/book OCR and translation powered by GPT-4o &amp; Gemini 2.0. Supports canvas box-selection for complex multi-region vertical/horizontal text extraction.
适用于未汉化的杂志切页，公式书，攻略，漫画， 扫图。

# JP Extract 日本語 — Japanese Text Extractor

A single-page web tool for extracting and translating Japanese text from images using GPT-4o or Gemini 2.0 Flash. Draw selection boxes directly on an image, run OCR, and optionally translate to Chinese and/or English — all from your browser.

---

## Features

- **Box-based OCR** — drag to draw one or more selection boxes on an image; each box is processed independently
- **Whole-image scan** — leave the canvas empty and click Identify JP to scan the entire image at once
- **Two AI providers** — choose between GPT-4o (OpenAI) and Gemini 2.0 Flash (Google)
- **Inline translation** — add Chinese (中文), English, or both translations alongside any extracted text in one click
- **Image rotation** — rotate the image ±180° with a slider before drawing boxes, useful for sideways manga pages
- **Scroll to zoom** — pinch or scroll to zoom in/out while keeping the image sharp
- **Undo** — Ctrl+Z removes the most recently drawn box (before running OCR)
- **Click to highlight** — clicking a result block on the right panel bold-frames the corresponding canvas box; clicking again or clicking outside deselects
- **Copy all** — copies all extracted text and translations to the clipboard in a clean numbered format
- **No data stored** — your API key and images never leave your browser except for the direct API call to OpenAI/Google

---

## Project Structure

```
.
├── static/
│   └── index.html      # Single-file frontend (HTML + CSS + JS)
├── server.py           # Flask backend — serves the UI and proxies /extract calls
└── requirements.txt    # Python dependencies
```

---

## Setup

### Requirements

- Python 3.9+
- An OpenAI API key **or** a Google Gemini API key

### Install

```bash
pip3 install -r requirements.txt
```

### Run

```bash
python3 server.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

The port can be changed with an environment variable:

```bash
PORT=8080 python server.py
```

---

## Usage

1. **Upload an image** — click the drop zone or drag a JPG/PNG/WEBP file onto it
2. **Draw selection boxes** — click and drag on the image to mark text regions; draw as many as needed
   - Leave the canvas empty to scan the whole image instead
   - Ctrl+Z undoes the last drawn box (before running OCR)
   - Use the rotation slider if the page is tilted or sideways
3. **Choose a provider** — select GPT-4o or Gemini 2.0 Flash from the header dropdown, then paste your API key
4. **Run extraction** — click one of the action buttons:
   - **Identify JP** — extract Japanese text only
   - **+ 中文** — extract and translate to Chinese
   - **+ English** — extract and translate to English
   - **+ 中文 & English** — extract and add both translations
5. **Review results** — extracted text appears in the right panel, numbered to match each canvas box; click a result block to highlight the corresponding box
6. **Copy** — click Copy in the panel header to copy all text and translations to the clipboard

---

## API Keys

Your API key is sent directly from the browser to your local Flask server, which forwards it to the provider. It is never stored on disk.

- **OpenAI** — get a key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys); requires access to `gpt-4o`
- **Google Gemini** — get a key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

---

## How It Works

The frontend sends a cropped JPEG of each drawn box (or the whole rotated image) to `POST /extract` on the local Flask server. The server base64-encodes the image and forwards it to the selected provider with a structured OCR prompt. The model returns a JSON array of text regions, each with an `orientation` field (`vertical` or `horizontal`) and the extracted `text`. Translations are handled by a second API call made directly from the browser.

---

## Dependencies

| Package | Purpose |
|---|---|
| Flask | Local web server |
| Pillow | Image decoding and JPEG re-encoding |
| requests | HTTP calls to OpenAI / Gemini APIs |
| opencv-python | Image processing utilities |
| reportlab / img2pdf | PDF export (optional, for future use) |



https://github.com/user-attachments/assets/d497b48d-2b54-40f5-802a-fe40acbbf9a6

