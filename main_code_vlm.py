"""
Enhanced VLM Form Schema Extractor with Question-Answer Matching
-----------------------------------------------------------------
Input: PDF/image file with form
Output: Normalized JSON schema matching LLM version format

Pipeline:
  1) PDF/Image -> images
  2) YOLO -> detect answer field bounding boxes
  3) VLM -> extract questions with bounding boxes
  4) VLM -> match questions to answer fields using layout analysis
  5) VLM -> normalize schema to final JSON format

Environment / Config:
  - YOLO_MODEL_PATH: path to YOLO weights (default: "best.pt")
  - OLLAMA_VLM_MODEL: Ollama VLM model name (default: "qwen3-vl:4b")
  - OLLAMA_HOST: Ollama server URL (default: "http://localhost:11434")
  - POPPLER required for PDF processing

Install:
  pip install ultralytics pillow pdf2image opencv-python numpy requests python-dotenv

Author: Enhanced VLM-based form extractor
"""

from __future__ import annotations

import os
import json
import time
import uuid
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pdf2image import convert_from_path
import cv2
import base64
import io
import math

try:
    from ultralytics import YOLO
except Exception as e:
    raise RuntimeError(f"Install ultralytics: pip install ultralytics\n{e}")

try:
    import requests
except Exception as e:
    raise RuntimeError(f"Install requests: pip install requests\n{e}")

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# -----------------------------
# Configuration
# -----------------------------
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "best.pt")
OLLAMA_VLM_MODEL = os.getenv("OLLAMA_VLM_MODEL", "qwen3-vl:4b")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

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

@dataclass
class PageResult:
    page_number: int
    width: int
    height: int
    fields: List[FieldCandidate]
    yolo_detections: List[Detection] = field(default_factory=list)
    raw_vlm_response: Optional[str] = None

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

def slugify(text: str) -> str:
    s = "".join(ch if ch.isalnum() else "_" for ch in text.strip().lower())
    s = "_".join([t for t in s.split("_") if t])
    return s[:60] or f"field_{uuid.uuid4().hex[:6]}"

def draw_debug(
    img: Image.Image,
    fields: List[FieldCandidate],
    detections: List[Detection],
    out_path: Path
):
    im = img.copy()
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    # Draw YOLO detections
    for d in detections:
        draw.rectangle(d.bbox, outline="green", width=2)
        draw.text((d.bbox[0], d.bbox[1]-16), f"{d.cls_name} {d.conf:.2f}", fill="green", font=font)

    # Draw matched fields
    for f in fields:
        draw.rectangle(f.question_bbox, outline="blue", width=3)
        qlabel = f"{f.answer_type} | {f.question_text[:30]}"
        draw.text((f.question_bbox[0], f.question_bbox[1]-18), qlabel, fill="blue", font=font)
        
        for ab in f.answer_bboxes:
            draw.rectangle(ab, outline="red", width=3)

        if f.answer_bboxes:
            qc = bbox_center(f.question_bbox)
            ac = bbox_center(f.answer_bboxes[0])
            draw.line([qc, ac], fill="cyan", width=2)

    im.save(out_path)


# -----------------------------
# File Input
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
        raise ValueError(f"Unsupported file type: {file_path}")


# -----------------------------
# YOLO Model
# -----------------------------
class YOLOModel:
    def __init__(self, model_path: str, device: Optional[str] = None):
        logger.info(f"Loading YOLO model: {model_path}")
        
        if device is None:
            try:
                import torch
                logger.info(f"PyTorch version: {torch.__version__}")
                logger.info(f"CUDA available: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    logger.info(f"CUDA version: {torch.version.cuda}")
                    logger.info(f"GPU device: {torch.cuda.get_device_name(0)}")
                    self.device = "0"  # Use GPU 0
                else:
                    logger.warning("CUDA not available. Using CPU (slow for large images)")
                    logger.warning("To enable GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
                    self.device = "cpu"
            except ImportError:
                logger.error("PyTorch not found. Install: pip install torch torchvision")
                self.device = "cpu"
        else:
            self.device = device
            logger.info(f"Using specified device: {device}")
        
        self.yolo = YOLO(model_path)
        logger.info(f"YOLO model loaded on device: {self.device}")

    def detect_answer_fields(self, image: Image.Image) -> List[Detection]:
        np_img = np.array(image)
        results = self.yolo(np_img, device=self.device if self.device else None)
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
# Ollama VLM Model
# -----------------------------
class OllamaVLMModel:
    def __init__(self, model_name: str, host: str = OLLAMA_HOST):
        self.model_name = model_name
        self.host = host
        logger.info(f"Initializing Ollama VLM: {model_name} at {host}")
        
        try:
            response = requests.get(f"{host}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info("Ollama connection successful.")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to Ollama at {host}")
        except Exception as e:
            logger.warning(f"Could not verify Ollama: {e}")

    def _image_to_base64(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _call_vlm(self, prompt: str, image: Image.Image, timeout: int = 180) -> str:
        """Generic VLM call method."""
        img_base64 = self._image_to_base64(image)
        
        try:
            response = requests.post(
                f"{self.host}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [img_base64]
                        }
                    ],
                    "stream": False
                },
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()
            response_text = result.get("message", {}).get("content", "")
            return response_text.strip()
        except Exception as e:
            logger.error(f"VLM call error: {e}")
            return "{}"

    def extract_questions_and_match(
        self, 
        image: Image.Image, 
        detections: List[Detection],
        page_num: int
    ) -> Tuple[List[FieldCandidate], str]:
        """
        Single VLM call to extract questions AND match them with YOLO answer fields.
        """
        page_w, page_h = image.size
        
        # Format detected answer fields
        answer_fields_str = ""
        for i, det in enumerate(detections):
            answer_fields_str += f"""
Field #{i+1}:
  - Type: {det.cls_name}
  - Confidence: {det.conf:.3f}
  - Bounding Box: {det.bbox}
  - Center: ({bbox_center(det.bbox)[0]}, {bbox_center(det.bbox)[1]})
"""
        
        prompt = f"""You are an expert form analyzer. Analyze this form image (Page {page_num}) and perform TWO tasks:

**TASK 1: Extract ALL Questions**
Identify every question, label, or field prompt in the form. For each question:
- Extract the exact question text
- Determine the precise bounding box [x1, y1, x2, y2] where the question appears
- Consider multi-line questions as a single unit

**TASK 2: Match Questions to Answer Fields**
I have detected {len(detections)} answer field regions using object detection. Match each question to its corresponding answer field(s) using these layout rules:

**MATCHING RULES (Priority Order):**
1. **Same-Row Right Bias**: Answer fields directly to the right of the question (within same horizontal band ±50px) are HIGHLY PREFERRED
2. **Vertical Alignment**: If multiple fields are to the right, choose the one closest vertically (±30px)
3. **Proximity**: Among candidates, prefer the closest answer field (Euclidean distance)
4. **Checkbox/Radio Groups**: Multiple checkboxes/radios near each other (within 80px) likely belong to the same question
5. **Below Questions**: For questions spanning full width, answer may be directly below (within 100px)
6. **Field Type Hints**: 
   - "Name", "Address", "Email" → likely "text_field" or "line_field"
   - "Date", "DOB" → likely "date" field
   - "Signature" → likely "signature" field
   - "Yes/No", "Male/Female", multiple options → likely "checkbox" or "radio"

**IMAGE DIMENSIONS:** {page_w} x {page_h} pixels (origin at top-left)

**DETECTED ANSWER FIELDS:**
{answer_fields_str}

**OUTPUT FORMAT:**
Return ONLY valid JSON with this EXACT structure (no markdown, no extra text):

{{
  "fields": [
    {{
      "question": "Full question text here",
      "question_bbox": [x1, y1, x2, y2],
      "answer_type": "text|checkbox|radio|signature|date|table_cell|unknown",
      "matched_field_ids": [0, 1],
      "confidence": "high|medium|low",
      "reasoning": "Brief explanation of match (optional)"
    }}
  ]
}}

**IMPORTANT GUIDELINES:**
- matched_field_ids: List of Field # indices (0-based) that belong to this question
- For checkbox/radio groups, include ALL relevant field IDs in the list
- answer_type should match the actual field type detected by YOLO
- If no good match exists, use empty matched_field_ids: []
- confidence: "high" (same row), "medium" (nearby), "low" (far/ambiguous)
- Ensure bounding boxes are within image bounds [0,0,{page_w},{page_h}]

**CRITICAL:** Return ONLY the JSON object, nothing else. Do not include explanations outside JSON."""

        logger.info(f"Calling VLM for question extraction and matching...")
        response = self._call_vlm(prompt, image, timeout=180)
        
        # Parse VLM response
        fields = self._parse_vlm_matching_response(response, detections, page_w, page_h)
        
        return fields, response

    def _parse_vlm_matching_response(
        self,
        response: str,
        detections: List[Detection],
        page_w: int,
        page_h: int
    ) -> List[FieldCandidate]:
        """Parse VLM response containing questions matched to answer fields."""
        fields: List[FieldCandidate] = []
        
        # Extract JSON
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            logger.warning("No JSON found in VLM response")
            return fields
        
        try:
            data = json.loads(json_match.group())
            if not isinstance(data, dict) or "fields" not in data:
                logger.warning("Invalid JSON structure")
                return fields
            
            for field_data in data["fields"]:
                question = field_data.get("question", "").strip()
                if not question:
                    continue
                
                # Get question bbox
                q_bbox = field_data.get("question_bbox", [0, 0, page_w//2, 50])
                q_bbox = ensure_int_bbox(q_bbox[:4]) if len(q_bbox) >= 4 else [0, 0, page_w//2, 50]
                
                # Get matched field IDs
                matched_ids = field_data.get("matched_field_ids", [])
                if not isinstance(matched_ids, list):
                    matched_ids = [matched_ids] if matched_ids is not None else []
                
                # Get answer type
                answer_type = field_data.get("answer_type", "unknown")
                
                # Build answer bboxes and yolo classes from matched detections
                answer_bboxes: List[BBox] = []
                yolo_classes: List[str] = []
                
                for field_id in matched_ids:
                    if isinstance(field_id, int) and 0 <= field_id < len(detections):
                        det = detections[field_id]
                        answer_bboxes.append(det.bbox)
                        yolo_classes.append(det.cls_name)
                
                # If no matches but we have an answer_type, create placeholder
                if not answer_bboxes:
                    # Create estimated bbox to the right of question
                    est_x1 = min(q_bbox[2] + 20, page_w - 200)
                    est_y1 = q_bbox[1]
                    est_x2 = min(est_x1 + 200, page_w - 10)
                    est_y2 = q_bbox[3]
                    answer_bboxes = [[est_x1, est_y1, est_x2, est_y2]]
                    yolo_classes = [answer_type]
                
                fields.append(FieldCandidate(
                    question_text=question,
                    question_bbox=q_bbox,
                    answer_type=answer_type,
                    answer_bboxes=answer_bboxes,
                    yolo_classes=yolo_classes
                ))
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.debug(f"Response was: {response[:500]}")
        
        return fields

    def normalize_schema(self, raw_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use VLM to normalize the raw schema into final format with stable IDs.
        """
        prompt = f"""You are a form schema normalizer. Convert this raw form schema into a clean, normalized JSON format.

**INPUT SCHEMA:**
{json.dumps(raw_schema, indent=2, ensure_ascii=False)}

**YOUR TASK:**
Create a normalized schema with:
1. Stable field IDs (snake_case, descriptive)
2. Human-readable labels
3. Proper field types
4. Preserve ALL bounding boxes exactly
5. For checkbox/radio fields with multiple options, infer option labels if possible from question text

**OUTPUT FORMAT:**
Return ONLY this JSON structure (no markdown, no extra text):

{{
  "document_title": "Descriptive form title",
  "total_pages": {raw_schema.get('total_pages', 1)},
  "fields": [
    {{
      "id": "snake_case_field_id",
      "label": "Human Readable Label",
      "type": "text|checkbox|radio|signature|date|table_cell|unknown",
      "page": 1,
      "question_bbox": [x1, y1, x2, y2],
      "answer_bboxes": [[x1, y1, x2, y2]],
      "options": ["Option 1", "Option 2"],
      "required": true
    }}
  ]
}}

**FIELD ID GENERATION RULES:**
- Use question text to create meaningful IDs
- "Full Name" → "full_name"
- "Date of Birth" → "date_of_birth"
- "Email Address" → "email_address"
- Generic questions → use page + position: "field_p1_1"

**OPTIONS INFERENCE:**
- "Gender: Male [ ] Female [ ]" → options: ["Male", "Female"]
- "Yes [ ] No [ ]" → options: ["Yes", "No"]
- Multiple checkboxes → extract from question context

**CRITICAL:** Return ONLY the JSON object. No explanations outside JSON."""

        logger.info("Calling VLM for schema normalization...")
        response = self._call_vlm(prompt, None, timeout=500)
        
        # For normalization, we don't need an image, but API requires it
        # Create a dummy 1x1 image
        dummy_img = Image.new('RGB', (1, 1), color='white')
        
        try:
            # Override to use non-vision endpoint if available
            vlm_response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json"
                },
                timeout=500
            )
            if vlm_response.ok:
                response = vlm_response.json().get("response", "{}")
        except:
            pass  # Fall back to vision endpoint
        
        # Parse normalized schema
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                logger.warning("Failed to parse normalized schema")
        
        logger.warning("Using fallback normalization")
        return self._fallback_normalize(raw_schema)
    
    def _fallback_normalize(self, raw_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Simple fallback normalization without VLM."""
        normalized = {
            "document_title": raw_schema.get("document_id", "Unnamed Form"),
            "total_pages": len(raw_schema.get("pages", [])),
            "fields": []
        }
        
        for page in raw_schema.get("pages", []):
            page_num = page.get("page", 1)
            for i, field in enumerate(page.get("fields", [])):
                field_id = slugify(field.get("question", f"field_p{page_num}_{i}"))
                normalized["fields"].append({
                    "id": field_id,
                    "label": field.get("question", ""),
                    "type": field.get("answer_type", "unknown"),
                    "page": page_num,
                    "question_bbox": field.get("question_bbox", []),
                    "answer_bboxes": field.get("answer_bboxes", []),
                    "options": [],
                    "required": False
                })
        
        return normalized


# -----------------------------
# Main Pipeline
# -----------------------------
def process_pdf(
    pdf_path: str,
    yolo_weights: str = YOLO_MODEL_PATH,
    ollama_vlm_model: str = OLLAMA_VLM_MODEL,
    out_dir: str = OUTPUT_DIR,
    make_debug_images: bool = True,
    device: Optional[str] = None
) -> DocumentSchema:
    """
    Process form using YOLO for answer fields + VLM for questions and matching.
    """
    t0 = time.time()
    out_root = Path(out_dir)
    (out_root / "annotated").mkdir(parents=True, exist_ok=True)

    # Initialize models
    logger.info("="*60)
    logger.info("Initializing models...")
    yolo_model = YOLOModel(yolo_weights, device=device)
    vlm_model = OllamaVLMModel(ollama_vlm_model)

    pages = load_images(pdf_path, dpi=300)
    page_results: List[PageResult] = []

    for idx, pil_img in enumerate(pages, start=1):
        logger.info("="*60)
        logger.info(f"Processing Page {idx}/{len(pages)}")
        logger.info("="*60)
        page_w, page_h = pil_img.size

        # Step 1: YOLO detection
        logger.info(f"[1/3] Running YOLO for answer field detection...")
        detections = yolo_model.detect_answer_fields(pil_img)
        logger.info(f"      Detected {len(detections)} answer fields")
        
        if detections:
            for i, d in enumerate(detections[:3]):
                logger.info(f"      #{i+1}: {d.cls_name} (conf={d.conf:.2f}) at {d.bbox}")

        # Step 2: VLM question extraction and matching
        logger.info(f"[2/3] Running VLM for question extraction and matching...")
        fields, vlm_response = vlm_model.extract_questions_and_match(
            pil_img, detections, idx
        )
        logger.info(f"      Extracted and matched {len(fields)} question-answer pairs")
        
        if fields:
            for i, f in enumerate(fields[:3]):
                logger.info(f"      #{i+1}: '{f.question_text[:40]}...' → {f.answer_type} ({len(f.answer_bboxes)} fields)")

        # Step 3: Create page result
        pr = PageResult(
            page_number=idx,
            width=page_w,
            height=page_h,
            fields=fields,
            yolo_detections=detections,
            raw_vlm_response=vlm_response
        )
        page_results.append(pr)

        # Debug visualization
        if make_debug_images:
            logger.info(f"[3/3] Generating debug image...")
            out_img = out_root / "annotated" / f"page_{idx:03d}_annotated.jpg"
            draw_debug(pil_img, fields, detections, out_img)
            logger.info(f"      Saved to: {out_img}")

    # Build raw schema
    logger.info("="*60)
    logger.info("Building raw schema...")
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
                        "yolo_classes": f.yolo_classes
                    }
                    for f in p.fields
                ]
            }
            for p in page_results
        ]
    }

    # Normalize schema using VLM
    logger.info("Normalizing schema with VLM...")
    normalized = vlm_model.normalize_schema(raw_schema)

    # Save outputs
    schema_path = out_root / "document_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "raw": raw_schema,
                "normalized": normalized
            },
            f, indent=2, ensure_ascii=False
        )

    logger.info("="*60)
    logger.info(f"✓ Extraction complete!")
    logger.info(f"  Schema saved to: {schema_path}")
    logger.info(f"  Total time: {time.time() - t0:.1f}s")
    logger.info(f"  Pages processed: {len(page_results)}")
    logger.info(f"  Total fields: {sum(len(p.fields) for p in page_results)}")
    logger.info("="*60)

    doc = DocumentSchema(pages=page_results, normalized_schema=normalized)
    return doc


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Extract form schema using YOLO + Ollama VLM with intelligent matching."
    )
    ap.add_argument("pdf", help="Path to form PDF or image file")
    ap.add_argument("--yolo", default=YOLO_MODEL_PATH, help=f"YOLO weights (default: {YOLO_MODEL_PATH})")
    ap.add_argument("--ollama-vlm", default=OLLAMA_VLM_MODEL, help=f"Ollama VLM model (default: {OLLAMA_VLM_MODEL})")
    ap.add_argument("--out", default=OUTPUT_DIR, help="Output directory")
    ap.add_argument("--device", default=None, help="Device: 'cuda' or 'cpu'")
    ap.add_argument("--no-debug", action="store_true", help="Disable debug images")
    args = ap.parse_args()

    process_pdf(
        args.pdf,
        yolo_weights=args.yolo,
        ollama_vlm_model=args.ollama_vlm,
        out_dir=args.out,
        make_debug_images=not args.no_debug,
        device=args.device
    )