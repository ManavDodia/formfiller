"""
LLM-Enhanced Form Schema Extractor
-----------------------------------
Input: PDF path or image file (.jpg, .jpeg, .png, .bmp, .tiff, .gif, .webp)
Output:
  - /output/annotated/page_XXX_annotated.jpg  (visual debug)
  - /output/document_schema.json              (final normalized schema)

Pipeline:
  1) PDF/Image -> images (pdf2image for PDFs, direct PIL load for images)
  2) YOLO -> answer regions (text fields, checkboxes, radios, signatures, dates, etc.)
  3) PaddleOCR -> all text lines with bboxes
  4) LLM -> identify questions + match to answer regions (replaces heuristics)
  5) LLM -> final normalized JSON schema

Environment / Config:
  - YOLO_MODEL_PATH: path to your trained weights (e.g., "best.pt")
  - OLLAMA_HOST: Ollama server URL (default: http://localhost:11434)
  - OLLAMA_MODEL: Ollama model name (default: llama3.1:latest)
  - POPPLER must be installed for PDF processing (system dependency)

Install (example):
  pip install ultralytics paddleocr==2.7.0.3 pdf2image pillow opencv-python numpy requests python-dotenv

Author: Enhanced version with LLM-based matching
"""

from __future__ import annotations

import os
import json
import math
import time
import uuid
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

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")

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
    answer_type: str
    answer_bboxes: List[BBox]
    yolo_classes: List[str] = field(default_factory=list)
    options: List[str] = field(default_factory=list)  # For radio/checkbox

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
    normalized_schema: Optional[Dict[str, Any]] = None


# -----------------------------
# Helpers
# -----------------------------
def ensure_int_bbox(b: BBox) -> BBox:
    return [int(round(v)) for v in b]

def bbox_center(b: BBox) -> Tuple[int, int]:
    x1, y1, x2, y2 = b
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

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
        small_font = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Draw detections (YOLO answer regions)
    for d in dets:
        draw.rectangle(d.bbox, outline="green", width=2)
        draw.text((d.bbox[0], d.bbox[1]-16), f"{d.cls_name} {d.conf:.2f}", fill="green", font=small_font)

    # Draw all OCR lines (light gray for non-questions)
    for l in ocr_lines:
        draw.rectangle(l.bbox, outline="lightgray", width=1)

    # Draw matched fields
    for f in fields:
        # Question box (blue)
        draw.rectangle(f.question_bbox, outline="blue", width=3)
        qlabel = f"{f.answer_type} | {f.question_text[:40]}"
        draw.text((f.question_bbox[0], f.question_bbox[1]-18), qlabel, fill="blue", font=font)
        
        # Answer boxes (red)
        for ab in f.answer_bboxes:
            draw.rectangle(ab, outline="red", width=3)

        # Link question to first answer
        if f.answer_bboxes:
            qc = bbox_center(f.question_bbox)
            ac = bbox_center(f.answer_bboxes[0])
            draw.line([qc, ac], fill="cyan", width=2)
        
        # Draw options if available
        if f.options and f.answer_bboxes:
            for i, opt in enumerate(f.options):
                if i < len(f.answer_bboxes):
                    ab = f.answer_bboxes[i]
                    draw.text((ab[2] + 5, ab[1]), opt[:20], fill="purple", font=small_font)

    im.save(out_path)


# -----------------------------
# File Input Detection & Conversion
# -----------------------------
def is_pdf(file_path: str) -> bool:
    return Path(file_path).suffix.lower() == ".pdf"

def is_image(file_path: str) -> bool:
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
    return Path(file_path).suffix.lower() in image_extensions

def load_images(file_path: str, dpi: int = 300) -> List[Image.Image]:
    if is_pdf(file_path):
        logger.info(f"Detected PDF file: {file_path}")
        return convert_from_path(file_path, dpi=dpi)
    elif is_image(file_path):
        logger.info(f"Detected image file: {file_path}")
        img = Image.open(file_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return [img]
    else:
        raise ValueError(
            f"Unsupported file type: {file_path}. "
            f"Supported formats: PDF (.pdf) or images (.jpg, .jpeg, .png, .bmp, .tiff, .gif, .webp)"
        )


# -----------------------------
# Models
# -----------------------------
class Models:
    def __init__(self, yolo_path: str):
        logger.info(f"Loading YOLO model: {yolo_path}")
        self.yolo = YOLO(yolo_path)

        logger.info("Initializing PaddleOCR...")
        try:
            self.ocr = PaddleOCR(use_textline_orientation=True, lang='en', show_log=False)
        except (TypeError, ValueError):
            # Fallback for older PaddleOCR versions that don't support show_log or use_textline_orientation
            try:
                self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            except (TypeError, ValueError):
                # Final fallback: remove unsupported parameters
                self.ocr = PaddleOCR(lang='en')
        logger.info("PaddleOCR ready.")


# -----------------------------
# Inference: YOLO
# -----------------------------
def detect_answer_regions_yolo(models: Models, image: Image.Image) -> List[Detection]:
    np_img = np.array(image)
    results = models.yolo(np_img, verbose=False)
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
            bbox = ensure_int_bbox([xyxy[0], xyxy[1], xyxy[2], xyxy[3]])
            dets.append(Detection(bbox=bbox, conf=conf, cls_name=cls_name))

    return dets


# -----------------------------
# Inference: OCR
# -----------------------------
def ocr_image(models: Models, image: Image.Image) -> List[OCRLine]:
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Try different API methods for different PaddleOCR versions
    try:
        # Newer API: use predict() method
        out = models.ocr.predict(img_bgr)
    except (AttributeError, TypeError):
        try:
            # Older API: use ocr() without cls parameter
            out = models.ocr.ocr(img_bgr)
        except TypeError:
            # Very old API: use ocr() with cls parameter
            out = models.ocr.ocr(img_bgr, cls=True)

    lines: List[OCRLine] = []
    if not out or not out[0]:
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


# -----------------------------
# LLM Question Identification & Matching
# -----------------------------
def llm_match_questions_to_answers(
    ocr_lines: List[OCRLine],
    detections: List[Detection],
    page_w: int,
    page_h: int,
    page_image: Optional[Image.Image] = None
) -> List[FieldCandidate]:
    """
    Use LLM to:
    1. Identify which OCR text lines are questions
    2. Match each question to its corresponding answer field(s) from YOLO detections
    3. Infer options for checkbox/radio groups
    """
    
    # Prepare data for LLM
    page_data = {
        "page_size": {"width": page_w, "height": page_h},
        "ocr_text": [
            {
                "id": f"text_{i}",
                "text": line.text,
                "bbox": line.bbox,
                "confidence": round(line.confidence, 3)
            }
            for i, line in enumerate(ocr_lines)
        ],
        "answer_fields": [
            {
                "id": f"field_{i}",
                "type": det.cls_name,
                "bbox": det.bbox,
                "confidence": round(det.conf, 3)
            }
            for i, det in enumerate(detections)
        ]
    }

    # Enhanced prompt for vision models
    if page_image and "vision" in OLLAMA_MODEL.lower():
        prompt = f"""You are analyzing this form image. I've also extracted text and detected answer fields for you.

Your task:
1. Look at the form image to understand the layout
2. Identify which text lines are questions (labels asking for user input)
3. Match each question to its corresponding answer field(s) based on what you see
4. For checkbox/radio groups, identify option labels

Extracted Data:
{json.dumps(page_data, indent=2)}

IMPORTANT: Return ONLY valid JSON in this exact format (no markdown, no explanations):
{{
  "fields": [
    {{
      "question_text_id": "text_0",
      "question_text": "Full Name:",
      "answer_field_ids": ["field_5"],
      "answer_type": "text",
      "options": []
    }}
  ]
}}

Available answer_types: text, checkbox, radio, signature, date, table_cell, unknown"""
    else:
        # Text-only prompt (for non-vision models)
        prompt = f"""You are analyzing a form page. Your task is to:
1. Identify which text lines are questions (labels asking for user input)
2. Match each question to its corresponding answer field(s) based on spatial layout
3. For checkbox/radio groups, identify option labels if present

IMPORTANT RULES:
- Questions typically end with ":", "?", or "->" but can also be plain labels
- Answer fields are usually to the RIGHT or BELOW questions
- For checkbox/radio groups, look for nearby option labels (like "Male", "Female")
- Return ONLY valid JSON, no markdown, no explanations

Page Data:
{json.dumps(page_data, indent=2)}

Return JSON in this EXACT format:
{{
  "fields": [
    {{
      "question_text_id": "text_0",
      "question_text": "Full Name:",
      "answer_field_ids": ["field_5"],
      "answer_type": "text",
      "options": []
    }},
    {{
      "question_text_id": "text_3",
      "question_text": "Gender:",
      "answer_field_ids": ["field_8", "field_9"],
      "answer_type": "radio",
      "options": ["Male", "Female"]
    }}
  ]
}}

Available answer_types: text, checkbox, radio, signature, date, table_cell, unknown
Return empty array if no fields found."""

    logger.info("Calling LLM for question identification and matching...")
    
    try:
        llm_response = call_ollama(prompt, json_mode=True, image=page_image)
            
        fields_data = llm_response.get("fields", [])
        
        # Convert LLM response to FieldCandidate objects
        fields: List[FieldCandidate] = []
        
        for field_data in fields_data:
            # Find question bbox
            q_text_id = field_data.get("question_text_id", "")
            question_bbox = None
            for ocr_item in page_data["ocr_text"]:
                if ocr_item["id"] == q_text_id:
                    question_bbox = ocr_item["bbox"]
                    break
            
            if not question_bbox:
                logger.warning(f"Could not find bbox for question: {field_data.get('question_text')}")
                continue
            
            # Find answer bboxes
            answer_field_ids = field_data.get("answer_field_ids", [])
            answer_bboxes = []
            yolo_classes = []
            
            for ans_id in answer_field_ids:
                for field_item in page_data["answer_fields"]:
                    if field_item["id"] == ans_id:
                        answer_bboxes.append(field_item["bbox"])
                        yolo_classes.append(field_item["type"])
                        break
            
            if not answer_bboxes:
                logger.warning(f"No answer fields found for question: {field_data.get('question_text')}")
                continue
            
            fields.append(FieldCandidate(
                question_text=field_data.get("question_text", ""),
                question_bbox=question_bbox,
                answer_type=field_data.get("answer_type", "unknown"),
                answer_bboxes=answer_bboxes,
                yolo_classes=yolo_classes,
                options=field_data.get("options", [])
            ))
        
        logger.info(f"LLM identified {len(fields)} fields")
        return fields
        
    except Exception as e:
        logger.error(f"LLM matching failed: {e}")
        logger.info("Falling back to basic heuristic matching...")
        return fallback_heuristic_matching(ocr_lines, detections, page_w, page_h)


def fallback_heuristic_matching(
    ocr_lines: List[OCRLine],
    detections: List[Detection],
    page_w: int,
    page_h: int
) -> List[FieldCandidate]:
    """Simple fallback if LLM fails - basic left-to-right matching"""
    fields: List[FieldCandidate] = []
    
    # Sort OCR lines by vertical position
    sorted_ocr = sorted(ocr_lines, key=lambda l: l.bbox[1])
    
    for ocr_line in sorted_ocr[:10]:  # Limit to first 10 lines
        # Find nearest detection to the right
        qcx, qcy = bbox_center(ocr_line.bbox)
        
        nearest_det = None
        min_dist = float('inf')
        
        for det in detections:
            dcx, dcy = bbox_center(det.bbox)
            
            # Prefer fields to the right and roughly same vertical position
            if dcx > qcx and abs(dcy - qcy) < 100:
                dist = math.hypot(dcx - qcx, dcy - qcy)
                if dist < min_dist:
                    min_dist = dist
                    nearest_det = det
        
        if nearest_det:
            atype = {
                "text_field": "text",
                "line_field": "text",
                "checkbox": "checkbox",
                "radio": "radio",
                "date": "date",
                "signature": "signature",
                "table_cell": "table_cell"
            }.get(nearest_det.cls_name, "unknown")
            
            fields.append(FieldCandidate(
                question_text=ocr_line.text,
                question_bbox=ocr_line.bbox,
                answer_type=atype,
                answer_bboxes=[nearest_det.bbox],
                yolo_classes=[nearest_det.cls_name]
            ))
    
    return fields


# -----------------------------
# LLM API Calls (Ollama only)
# -----------------------------
def call_ollama(prompt: str, json_mode: bool = True, image: Optional[Image.Image] = None) -> Dict[str, Any]:
    import requests
    import base64
    from io import BytesIO
    
    if not OLLAMA_HOST:
        raise ValueError("OLLAMA_HOST not set")
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    
    if json_mode:
        payload["format"] = "json"
    
    # Add image for vision models
    if image and "vision" in OLLAMA_MODEL.lower():
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        payload["images"] = [img_base64]
    
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json=payload,
            timeout=(10, 600)
        )
        response.raise_for_status()
        text = response.json().get("response", "{}")
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            raise
            
    except Exception as e:
        logger.error(f"Ollama API error: {e}")
        return {"error": str(e)}


# -----------------------------
# Final Schema Normalization
# -----------------------------
def normalize_schema_with_llm(
    raw_schema: Dict[str, Any]
) -> Dict[str, Any]:
    """Final schema normalization and cleanup"""
    
    prompt = f"""You are given a raw extracted schema for a form. Normalize it to a clean, production-ready JSON with:

1. Stable field IDs (snake_case, descriptive)
2. Human-readable labels
3. Proper types: text, checkbox, radio, signature, date, table_cell, unknown
4. Preserve ALL bounding boxes and page numbers exactly as provided
5. Keep inferred options for checkbox/radio fields
6. Add a document_title if you can infer it from the fields

Return ONLY valid JSON in this format:
{{
  "document_title": "Application Form" or "Unknown Form",
  "total_pages": 1,
  "fields": [
    {{
      "id": "full_name",
      "label": "Full Name",
      "type": "text",
      "page": 1,
      "question_bbox": [x1, y1, x2, y2],
      "answer_bboxes": [[x1, y1, x2, y2]],
      "options": [],
      "required": true
    }}
  ]
}}

Raw schema:
{json.dumps(raw_schema, indent=2, ensure_ascii=False)}

Return ONLY the normalized JSON, no markdown, no explanations."""

    logger.info("Calling LLM for final schema normalization...")
    
    try:
        normalized = call_ollama(prompt, json_mode=True)
        return normalized
    except Exception as e:
        logger.error(f"Schema normalization failed: {e}")
        return {"error": str(e), "raw": raw_schema}


# -----------------------------
# Main Pipeline
# -----------------------------
def process_pdf(
    pdf_path: str,
    yolo_weights: str = YOLO_MODEL_PATH,
    out_dir: str = OUTPUT_DIR,
    make_debug_images: bool = True
) -> DocumentSchema:
    """
    Main pipeline with LLM-enhanced question identification and matching
    """
    t0 = time.time()
    out_root = Path(out_dir)
    (out_root / "annotated").mkdir(parents=True, exist_ok=True)

    models = Models(yolo_weights)
    pages = load_images(pdf_path, dpi=300)
    page_results: List[PageResult] = []

    for idx, pil_img in enumerate(pages, start=1):
        logger.info(f"Processing page {idx}/{len(pages)}...")

        # Step 1: YOLO detections
        logger.info(f"  - Running YOLO detection...")
        dets = detect_answer_regions_yolo(models, pil_img)
        logger.info(f"  - Found {len(dets)} answer regions")

        # Step 2: OCR
        logger.info(f"  - Running OCR...")
        ocr_lines = ocr_image(models, pil_img)
        logger.info(f"  - Extracted {len(ocr_lines)} text lines")

        page_w, page_h = pil_img.size

        # Step 3: LLM matching (replaces heuristics)
        logger.info(f"  - Matching questions to answers with LLM...")
        fields = llm_match_questions_to_answers(
            ocr_lines, dets, page_w, page_h, pil_img
        )
        logger.info(f"  - Matched {len(fields)} fields")

        pr = PageResult(
            page_number=idx,
            width=page_w,
            height=page_h,
            ocr_lines=ocr_lines,
            detections=dets,
            fields=fields
        )
        page_results.append(pr)

        # Debug visualization
        if make_debug_images:
            out_img = out_root / "annotated" / f"page_{idx:03d}_annotated.jpg"
            draw_debug(pil_img, ocr_lines, dets, fields, out_img)
            logger.info(f"  - Saved debug image: {out_img}")

    # Build raw schema
    raw_schema = {
        "document_id": Path(pdf_path).stem,
        "total_pages": len(page_results),
        "pages": [
            {
                "page": p.page_number,
                "size": {"w": p.width, "h": p.height},
                "fields": [
                    {
                        "question": f.question_text,
                        "question_bbox": f.question_bbox,
                        "answer_type": f.answer_type,
                        "answer_bboxes": f.answer_bboxes,
                        "yolo_classes": f.yolo_classes,
                        "options": f.options
                    }
                    for f in p.fields
                ]
            } for p in page_results
        ]
    }

    # Final normalization
    logger.info("Running final schema normalization...")
    normalized = normalize_schema_with_llm(raw_schema)

    # Save results
    schema_path = out_root / "document_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "raw": raw_schema,
                "normalized": normalized
            },
            f, indent=2, ensure_ascii=False
        )

    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Processing complete!")
    logger.info(f"✓ Schema saved to: {schema_path}")
    logger.info(f"✓ Total time: {time.time() - t0:.1f}s")
    logger.info(f"{'='*60}\n")

    doc = DocumentSchema(pages=page_results, normalized_schema=normalized)
    return doc


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="LLM-Enhanced Form Schema Extractor. "
        "Supports PDF (.pdf) and image files (.jpg, .jpeg, .png, .bmp, .tiff, .gif, .webp)."
    )
    ap.add_argument("pdf", help="Path to form PDF or image file")
    ap.add_argument("--yolo", default=YOLO_MODEL_PATH, help=f"YOLO weights path (default: {YOLO_MODEL_PATH})")
    ap.add_argument("--out", default=OUTPUT_DIR, help=f"Output directory (default: {OUTPUT_DIR})")
    ap.add_argument("--no-debug", action="store_true", help="Disable annotated debug images")
    args = ap.parse_args()

    process_pdf(
        args.pdf,
        yolo_weights=args.yolo,
        out_dir=args.out,
        make_debug_images=not args.no_debug
    )