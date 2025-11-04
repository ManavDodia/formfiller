"""
End-to-end Form Schema Extractor
--------------------------------
Input: PDF path to a (scanned or digital) form
Output:
  - /output/annotated/page_XXX_annotated.jpg  (visual debug)
  - /output/document_schema.json              (final normalized schema)

Pipeline:
  1) PDF -> images (pdf2image)
  2) YOLO -> answer regions (text fields, checkboxes, radios, signatures, dates, etc.)
  3) PaddleOCR -> question-like text lines (+ optional PPStructureV3 for structure)
  4) Pair questions to answer regions using layout heuristics (right/same-row bias)
  5) Ask LLM -> normalized JSON schema (stable keys, types, options, bboxes, page refs)

Environment / Config:
  - YOLO_MODEL_PATH: path to your trained weights (e.g., "best.pt")
  - LLM_PROVIDER: "openai" | "ollama" | "gemini" | "groq"
  - OPENAI_API_KEY / OLLAMA_HOST / GEMINI_API_KEY / GROQ_API_KEY as needed
  - POPPLER must be installed for pdf2image (system dependency)

Install (example):
  pip install ultralytics paddleocr==2.7.0.3 pdf2image pillow opencv-python numpy requests python-dotenv

Notes:
  - PPStructureV3 (optional) can improve page-level grouping. If not installed, we skip it.
  - BBoxes are [x1, y1, x2, y2] in pixel coordinates of the page image.
  - Coordinate system origin at top-left.

Author: You + ChatGPT
"""

from __future__ import annotations

import os
import io
import json
import math
import time
import uuid
import base64
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# --- 3rd party ---
from ultralytics import YOLO
from pdf2image import convert_from_path
import cv2

# PaddleOCR
try:
    from paddleocr import PaddleOCR
except Exception as e:
    raise RuntimeError(
        "Install PaddleOCR first: pip install paddleocr==2.7.0.3\n"
        f"Original import error: {e}"
    )

# Optional: PPStructureV3 for page-structure parsing (tables, titles, etc.)
try:
    from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes  # noqa
    from paddleocr import PPStructure
    _HAS_PPSTRUCT = True
except Exception:
    _HAS_PPSTRUCT = False

# Optional: .env for local config
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# -----------------------------
# Configuration
# -----------------------------
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "best.pt")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # forced to Groq by default

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# -----------------------------
# Logging
# -----------------------------
logger = logging.getLogger("form_extractor")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)


# -----------------------------
# Data models
# -----------------------------
BBox = List[int]  # [x1, y1, x2, y2]

# Map YOLO class names to expected canonical names
CLASS_ALIASES = {
    "blank_lines": "line_field",
    "blank_line": "line_field",
    "blank_areas": "text_field",
    "blank_area": "text_field",
    "table_cells": "table_cell",
    "table_cell": "table_cell",
}

@dataclass
class OCRLine:
    text: str
    confidence: float
    bbox: BBox

@dataclass
class Detection:
    bbox: BBox
    conf: float
    cls_name: str

@dataclass
class FieldCandidate:
    question_text: str
    question_bbox: BBox
    answer_type: str                 # "text", "checkbox", "radio", "signature", "date", "table_cell", "unknown"
    answer_bboxes: List[BBox]        # could be multi (e.g., options in radio/checkbox groups)
    yolo_classes: List[str] = field(default_factory=list)

@dataclass
class PageResult:
    page_number: int
    width: int
    height: int
    ocr_lines: List[OCRLine]
    detections: List[Detection]
    fields: List[FieldCandidate]

@dataclass
class DocumentSchema:
    pages: List[PageResult]
    # Normalized, LLM-curated schema per document level
    normalized_schema: Optional[Dict[str, Any]] = None


# -----------------------------
# Helpers
# -----------------------------
def ensure_int_bbox(b: BBox) -> BBox:
    return [int(round(v)) for v in b]

def bbox_center(b: BBox) -> Tuple[int, int]:
    x1, y1, x2, y2 = b
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def bbox_area(b: BBox) -> int:
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)

def iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = bbox_area(a) + bbox_area(b) - inter + 1e-6
    return inter / union

def euclidean(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def slugify(text: str) -> str:
    s = "".join(ch if ch.isalnum() else "_" for ch in text.strip().lower())
    s = "_".join([t for t in s.split("_") if t])
    return s[:60] or f"field_{uuid.uuid4().hex[:6]}"

def draw_debug(
    img: Image.Image,
    ocr_lines: List[OCRLine],
    dets: List[Detection],
    fields: List[FieldCandidate],
    out_path: Path
):
    im = img.copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    # Draw detections
    for d in dets:
        draw.rectangle(d.bbox, outline="green", width=2)
        draw.text((d.bbox[0], d.bbox[1]-16), f"{d.cls_name} {d.conf:.2f}", fill="green", font=font)

    # Draw OCR lines (questions are not preselected; draw all in light)
    for l in ocr_lines:
        draw.rectangle(l.bbox, outline="orange", width=1)
        if l.text:
            draw.text((l.bbox[0], l.bbox[1]-14), l.text[:40], fill="orange", font=font)

    # Draw field links
    for f in fields:
        # Question
        draw.rectangle(f.question_bbox, outline="blue", width=3)
        qlabel = f"{f.answer_type} | {f.question_text[:30]}"
        draw.text((f.question_bbox[0], f.question_bbox[1]-18), qlabel, fill="blue", font=font)
        # Answer boxes
        for ab in f.answer_bboxes:
            draw.rectangle(ab, outline="red", width=3)

        # Link question center -> first answer center
        if f.answer_bboxes:
            qc = bbox_center(f.question_bbox)
            ac = bbox_center(f.answer_bboxes[0])
            draw.line([qc, ac], fill="cyan", width=2)

    im.save(out_path)


# -----------------------------
# PDF -> Images
# -----------------------------
def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    """
    Requires poppler installed on system. On Windows, set POPPLER_PATH env or pass to convert_from_path.
    """
    return convert_from_path(pdf_path, dpi=dpi)


# -----------------------------
# Models
# -----------------------------
class Models:
    def __init__(self, yolo_path: str):
        logger.info(f"Loading YOLO model: {yolo_path}")
        self.yolo = YOLO(yolo_path)

        logger.info("Initializing PaddleOCR...")
        # angle cls helps with rotated text; lang 'en' by default
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
        logger.info("PaddleOCR ready.")

        if _HAS_PPSTRUCT:
            logger.info("PPStructure available. Using lightweight defaults.")
            # Defaults; tweak as needed
            self.ppstruct = PPStructure(show_log=False)
        else:
            self.ppstruct = None


# -----------------------------
# Inference: YOLO
# -----------------------------
def detect_answer_regions_yolo(models: Models, image: Image.Image) -> List[Detection]:
    """
    Runs YOLO. Your trained classes may include e.g.:
      - text_field, line_field, checkbox, radio, signature, date, table_cell, table, etc.
    Returns list of Detection with class names and conf.
    """
    np_img = np.array(image)  # RGB
    results = models.yolo(np_img)
    dets: List[Detection] = []

    for r in results:
        names = r.names
        boxes = r.boxes
        if boxes is None:
            continue
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].tolist()
            cls_idx = int(boxes.cls[i].item()) if boxes.cls is not None else -1
            conf = float(boxes.conf[i].item()) if boxes.conf is not None else 0.0
            cls_name = names.get(cls_idx, f"class_{cls_idx}")
            # Normalize class name to expected canonical form
            cls_name = CLASS_ALIASES.get(cls_name, cls_name)
            bbox = ensure_int_bbox([xyxy[0], xyxy[1], xyxy[2], xyxy[3]])
            dets.append(Detection(bbox=bbox, conf=conf, cls_name=cls_name))

    return dets


# -----------------------------
# Inference: OCR (+ optional structure)
# -----------------------------
def ocr_image(models: Models, image: Image.Image) -> List[OCRLine]:
    """
    Returns OCR lines as [{text, confidence, bbox}], with bbox = [x1,y1,x2,y2].
    Handles both dict (new) and list (old) PaddleOCR return formats.
    """
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    out = models.ocr.ocr(img_bgr)

    lines: List[OCRLine] = []
    if not out:
        return lines

    result = out[0]
    if isinstance(result, dict):
        rec_texts = result.get('rec_texts', []) or []
        rec_scores = result.get('rec_scores', []) or []
        rec_polys = result.get('rec_polys', []) or []
        for i, txt in enumerate(rec_texts):
            txt = (txt or "").strip()
            if not txt:
                continue
            conf = float(rec_scores[i]) if i < len(rec_scores) else 0.0
            poly = rec_polys[i].tolist() if i < len(rec_polys) else []
            bbox = polygon_to_bbox(poly) if poly else [0, 0, 0, 0]
            lines.append(OCRLine(text=txt, confidence=conf, bbox=ensure_int_bbox(bbox)))
    else:
        # old format: list of [poly, (text, conf)]
        for entry in result:
            try:
                poly, (txt, conf) = entry
                txt = (txt or "").strip()
                if not txt:
                    continue
                bbox = polygon_to_bbox(poly)
                lines.append(OCRLine(text=txt, confidence=float(conf), bbox=ensure_int_bbox(bbox)))
            except Exception:
                continue

    return lines


def polygon_to_bbox(poly: List[List[float]]) -> BBox:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def run_ppstructure(models: Models, image: Image.Image) -> Optional[str]:
    """
    Optional: returns a coarse page text with region labels (table/para/title/etc.)
    to give the LLM more structured context. Skips if PPStructure not available.
    """
    if not models.ppstruct:
        return None
    img_np = np.array(image)
    result = models.ppstruct(img_np)
    buf: List[str] = []
    for res in result:
        plist = res.get("res", {}).get("layout_res", []) or res.get("res", [])
        for elem in plist:
            label = elem.get("label", "region")
            text = elem.get("text", "").strip()
            if text:
                buf.append(f"[{label}] {text}")
    return "\n".join(buf) if buf else None


# -----------------------------
# Pairing logic
# -----------------------------
QUESTION_HARD_MARKERS = (":", "?", "->")

def looks_like_question(line: OCRLine, page_w: int) -> bool:
    # Heuristics: short(ish) left-column text or explicit markers
    t = line.text
    if not t:
        return False
    if t.endswith(QUESTION_HARD_MARKERS):
        return True
    if len(t) <= 60 and line.bbox[0] < page_w * 0.55:
        # Filter out pure numbers/noise
        if any(ch.isalpha() for ch in t):
            w = line.bbox[2] - line.bbox[0]
            h = line.bbox[3] - line.bbox[1]
            if w >= 15 and h >= 10:
                return True
    return False


def group_choices_nearby(dets: List[Detection], max_gap_px: int = 60) -> List[List[Detection]]:
    """
    Groups nearby checkboxes/radios into options sets (same row/column).
    Simple agglomerative approach by proximity.
    """
    ch = [d for d in dets if d.cls_name in ("checkbox", "radio")]
    groups: List[List[Detection]] = []
    used = set()
    for i, d in enumerate(ch):
        if i in used:
            continue
        g = [d]
        used.add(i)
        c1 = bbox_center(d.bbox)
        for j, e in enumerate(ch):
            if j in used:
                continue
            c2 = bbox_center(e.bbox)
            if euclidean(c1, c2) <= max_gap_px:
                g.append(e)
                used.add(j)
        groups.append(sorted(g, key=lambda x: x.bbox[1]))
    return groups


def find_best_answers_for_question(
    qbox: BBox,
    dets: List[Detection],
    page_w: int,
    right_dx: int = 600,  # Increased from 350 to find fields further right
    row_dy: int = 200    # Increased from 120 to allow more vertical tolerance
) -> Tuple[str, List[BBox], List[str]]:
    """
    Given a question box, prefer answer regions to the right in same row.
    If none, fallback to nearest text_field/line_field.
    Also handles checkbox/radio groups near the question.
    """
    qcx, qcy = bbox_center(qbox)

    # First pass: text-like fields to the right, near same row
    candidates: List[Tuple[float, Detection]] = []
    for d in dets:
        if d.cls_name in ("text_field", "line_field", "date", "signature", "table_cell"):
            acx, acy = bbox_center(d.bbox)
            dx, dy = acx - qcx, abs(acy - qcy)
            if dx >= 0 and dx <= right_dx and dy <= row_dy:
                candidates.append((euclidean((qcx, qcy), (acx, acy)), d))

    if candidates:
        candidates.sort(key=lambda t: t[0])
        d = candidates[0][1]
        atype = {
            "text_field": "text",
            "line_field": "text",
            "date": "date",
            "signature": "signature",
            "table_cell": "table_cell"
        }.get(d.cls_name, "text")
        return atype, [d.bbox], [d.cls_name]

    # Second pass: nearby checkbox/radio groups
    choice_dets = [d for d in dets if d.cls_name in ("checkbox", "radio")]
    if choice_dets:
        groups = group_choices_nearby(dets)
        best_g = None
        best_dist = 1e9
        for g in groups:
            # group center
            gc = bbox_center([
                min(x.bbox[0] for x in g),
                min(x.bbox[1] for x in g),
                max(x.bbox[2] for x in g),
                max(x.bbox[3] for x in g),
            ])
            d = euclidean((qcx, qcy), gc)
            if d < best_dist:
                best_g, best_dist = g, d
        if best_g and best_dist < 300:  # threshold
            return ("radio" if all(x.cls_name == "radio" for x in best_g) else "checkbox",
                    [x.bbox for x in best_g],
                    [x.cls_name for x in best_g])

    # Fallback: nearest any text-like field (even if not to the right)
    any_text: List[Tuple[float, Detection]] = []
    for d in dets:
        if d.cls_name in ("text_field", "line_field", "table_cell"):
            acx, acy = bbox_center(d.bbox)
            any_text.append((euclidean((qcx, qcy), (acx, acy)), d))
    if any_text:
        any_text.sort(key=lambda t: t[0])
        d = any_text[0][1]
        # Only accept if within reasonable distance (increased threshold)
        if euclidean((qcx, qcy), bbox_center(d.bbox)) < 800:
            return "text", [d.bbox], [d.cls_name]

    return "unknown", [], []


def build_fields_for_page(
    ocr_lines: List[OCRLine],
    dets: List[Detection],
    page_w: int,
    page_h: int
) -> List[FieldCandidate]:
    # Select question-like lines
    questions = [l for l in ocr_lines if looks_like_question(l, page_w)]

    # If nothing looks like a question, fall back to left-most N lines
    if not questions:
        questions = sorted(ocr_lines, key=lambda l: l.bbox[0])[:6]

    fields: List[FieldCandidate] = []
    for q in questions:
        atype, aboxes, cls_list = find_best_answers_for_question(q.bbox, dets, page_w)
        fields.append(FieldCandidate(
            question_text=q.text,
            question_bbox=q.bbox,
            answer_type=atype,
            answer_bboxes=aboxes,
            yolo_classes=cls_list
        ))
    return fields


# -----------------------------
# LLM Normalization
# -----------------------------
def ask_llm_normalize_schema(
    provider: str,
    raw_schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calls your chosen LLM to normalize and verify the schema.
    - Ensures stable ids, types, and per-option metadata for checkboxes/radios
    - Returns a JSON object with:
        {
          "document_title": "...",
          "fields": [
            {
              "id": "full_name",
              "label": "Full Name",
              "type": "text"|"checkbox"|"radio"|"signature"|"date"|"table_cell"|"unknown",
              "page": 1,
              "question_bbox": [x1,y1,x2,y2],
              "answer_bboxes": [[...], [...]],
              "options": ["Male","Female"]              # when radio/checkbox & text nearby inferred
            }, ...
          ]
        }
    """
    prompt = (
        "You are given a raw extracted schema for a form. "
        "Normalize it to a clean JSON with stable ids (snake_case), "
        "human-readable labels, types (text, checkbox, radio, signature, date, table_cell, unknown), "
        "and preserve all bounding boxes + page numbers exactly. "
        "If you can infer option labels for checkbox/radio from nearby text, add them under 'options'. "
        "Return ONLY valid minified JSON.\n\nRAW:\n"
        + json.dumps(raw_schema, ensure_ascii=False)
    )

    # Force Groq usage only
    return _ask_groq_json(prompt)


def _ask_ollama_json(prompt: str) -> Dict[str, Any]:
    """
    Minimal Ollama client using HTTP. Requires an appropriate local model (e.g., llama3:instruct).
    export OLLAMA_HOST=http://localhost:11434
    """
    import requests
    model = os.getenv("OLLAMA_MODEL", "llama3.1:latest")

    r = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False, "format": "json"}
    )
    r.raise_for_status()
    obj = r.json()
    txt = obj.get("response", "{}")
    try:
        return json.loads(txt)
    except Exception:
        # Try to extract JSON substring
        try:
            start = txt.find("{")
            end = txt.rfind("}")
            return json.loads(txt[start:end+1])
        except Exception:
            logger.warning("Ollama returned non-JSON; falling back to raw.")
            return {"raw_llm_text": txt}


def _ask_openai_json(prompt: str) -> Dict[str, Any]:
    """
    OpenAI JSON Completion via Chat Completions w/ response_format (older API may differ).
    Requires: pip install openai
    export OPENAI_API_KEY=...
    """
    try:
        from openai import OpenAI
    except Exception as e:
        logger.error("pip install openai to use OPENAI provider.")
        return {"error": str(e)}

    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return only valid compact JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    txt = resp.choices[0].message.content
    try:
        return json.loads(txt)
    except Exception:
        return {"raw_llm_text": txt}


def _ask_gemini_json(prompt: str) -> Dict[str, Any]:
    """
    Google Gemini JSON mode. Requires: pip install google-generativeai
    export GEMINI_API_KEY=...
    """
    try:
        import google.generativeai as genai
    except Exception as e:
        logger.error("pip install google-generativeai to use GEMINI provider.")
        return {"error": str(e)}

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
    resp = model.generate_content(
        [
            {"role": "user", "parts": [prompt]}
        ],
        generation_config={"response_mime_type": "application/json"}
    )
    txt = resp.text or "{}"
    try:
        return json.loads(txt)
    except Exception:
        return {"raw_llm_text": txt}


# New: Groq provider
def _ask_groq_json(prompt: str) -> Dict[str, Any]:
    """
    Groq Chat Completions with JSON mode. Requires: pip install groq
    export GROQ_API_KEY=... or set in .env file
    """
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY not set! Please set it in environment or .env file.")
        return {"error": "GROQ_API_KEY not configured. Set it in .env file or environment variable."}
    
    try:
        from groq import Groq
    except Exception as e:
        logger.error("pip install groq to use GROQ provider.")
        return {"error": str(e)}

    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    try:
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Return only valid compact JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        txt = resp.choices[0].message.content
        try:
            return json.loads(txt)
        except Exception:
            return {"raw_llm_text": txt}
    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# Main pipeline
# -----------------------------
def process_pdf(
    pdf_path: str,
    yolo_weights: str = YOLO_MODEL_PATH,
    out_dir: str = OUTPUT_DIR,
    make_debug_images: bool = True
) -> DocumentSchema:
    t0 = time.time()
    out_root = Path(out_dir)
    (out_root / "annotated").mkdir(parents=True, exist_ok=True)

    models = Models(yolo_weights)

    pages = pdf_to_images(pdf_path, dpi=300)
    page_results: List[PageResult] = []

    for idx, pil_img in enumerate(pages, start=1):
        logger.info(f"Page {idx}/{len(pages)}: running detectors...")

        dets = detect_answer_regions_yolo(models, pil_img)
        ocr_lines = ocr_image(models, pil_img)
        page_w, page_h = pil_img.size

        fields = build_fields_for_page(ocr_lines, dets, page_w, page_h)

        pr = PageResult(
            page_number=idx,
            width=page_w,
            height=page_h,
            ocr_lines=ocr_lines,
            detections=dets,
            fields=fields
        )
        page_results.append(pr)

        if make_debug_images:
            out_img = out_root / "annotated" / f"page_{idx:03d}_annotated.jpg"
            draw_debug(pil_img, ocr_lines, dets, fields, out_img)

    # Build simplified schema focused on question-answer pairing
    simplified_schema = {
        "document_id": Path(pdf_path).stem,
        "pages": [
            {
                "page": p.page_number,
                "size": {"w": p.width, "h": p.height},
                "fields": [
                    {
                        "question": f.question_text,
                        "question_bbox": f.question_bbox,
                        "answer_bboxes": f.answer_bboxes,
                        "answer_type": f.answer_type
                    }
                    for f in p.fields
                    if f.answer_bboxes  # Only include fields with detected answer boxes
                ]
            } for p in page_results
        ]
    }

    # Save simplified output
    schema_path = out_root / "document_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(simplified_schema, f, indent=2, ensure_ascii=False)

    logger.info(f"Done. Wrote schema to: {schema_path}")
    logger.info(f"Total time: {time.time() - t0:.1f}s")

    doc = DocumentSchema(pages=page_results, normalized_schema=simplified_schema)
    return doc


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Extract form schema from a PDF.")
    ap.add_argument("pdf", help="Path to form PDF")
    ap.add_argument("--yolo", default=YOLO_MODEL_PATH, help="Path to YOLO weights (default: best.pt or $YOLO_MODEL_PATH)")
    ap.add_argument("--out", default=OUTPUT_DIR, help="Output dir (default: ./output)")
    ap.add_argument("--no-debug", action="store_true", help="Disable annotated debug images")
    args = ap.parse_args()

    process_pdf(args.pdf, yolo_weights=args.yolo, out_dir=args.out, make_debug_images=not args.no_debug)
