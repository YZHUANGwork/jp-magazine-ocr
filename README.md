# jp-magazine-ocr

A single-page web tool for extracting and translating Japanese text from images using GPT-4o or Gemini 2.0 Flash.

Designed for Japanese magazines, manga, and illustrated spreads where text and pictures are densely interleaved — traditional **vertical columns** running right-to-left, **horizontal captions** and pull-quotes, speech bubbles, panel labels, and mixed-direction layouts all on the same page. Draw selection boxes around any region, and the tool correctly identifies the reading direction of each text block and returns them in proper reading order.

Translate everything to Chinese, English, or both in one click — all from your browser.
辅助啃生肉，让啃生肉没那么痛苦。适用于未汉化的杂志切页，公式书，攻略，漫画，扫图。
---

## Features

- **Reads vertical and horizontal text in the same image** — each extracted region is tagged with its orientation (`vertical` or `horizontal`) and ordered as a human would read it: right-to-left columns for vertical text, top-to-bottom for horizontal
- **Box-based OCR for cluttered layouts** — drag to draw selection boxes around specific regions; isolate a speech bubble, a caption strip, or a body-text column without picking up surrounding artwork or unrelated text
- **Whole-image scan** — leave the canvas empty and click Identify JP to let the AI find and order every text region on the page automatically
- **Multiple boxes per page** — draw as many boxes as needed; each is processed independently and results are numbered to match their canvas box
- **Two AI providers** — choose between GPT-4o (OpenAI) and Gemini 2.0 Flash (Google)
- **Inline translation** — add Chinese (中文), English, or both translations alongside extracted text in one click
- **Image rotation** — rotate the image ±180° with a slider before drawing boxes, useful for sideways or upside-down pages
- **Scroll to zoom** — pinch or scroll to zoom in/out for dense, small-print areas
- **Undo** — Ctrl+Z removes the most recently drawn box (before running OCR)
- **Click to highlight** — clicking a result block on the right panel bold-frames the corresponding canvas box; clicking again or clicking outside deselects
- **Copy all** — copies all extracted text and translations to the clipboard in a clean numbered format
- **No data stored** — your API key and images never leave your browser except for the direct API call to OpenAI/Google

---

---

## Why Draw Boxes?

Start with the whole-image scan — if it picks up everything correctly, you're done. But on pages where text and artwork are heavily mixed, AI models often miss text, jumble reading order, or hallucinate characters. The denser the layout, the less reliable a single full-page scan gets.

That's where boxes help. Draw one around any region that isn't coming out right — a panel with several paragraphs, a cluster of speech bubbles, a caption area — and the model gets a cleaner, focused crop to work from. Boxes don't have to be precise; a region covering multiple paragraphs or bubbles is fine. Just keep going until the whole page is covered.

Use Ctrl+Z to redraw a box if you misplace it, and run all boxes at once with a single click.

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
pip install -r requirements.txt
```

### Run

```bash
python server.py
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

The frontend sends a cropped JPEG of each drawn box (or the whole rotated image) to `POST /extract` on the local Flask server. The server base64-encodes the image and forwards it to the selected provider with a structured OCR prompt.

The model identifies every Japanese text region in the image, tags each with its `orientation` — `vertical` for top-to-bottom columns (the default for manga body text and most editorial copy) or `horizontal` for captions, titles, and modern-layout text — and returns them in natural reading order: rightmost column first for vertical text, top-to-bottom left-to-right for horizontal. This makes it reliable for magazine spreads where both directions coexist on the same page.

Translations are handled by a second API call made directly from the browser.

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

