from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
	normalized_schema: Optional[Dict[str, Any]] = None


