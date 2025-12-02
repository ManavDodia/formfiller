from typing import List, Optional
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2

from .geometry import ensure_int_bbox
from .types import Detection, OCRLine
from .logging_utils import logger

# PaddleOCR
try:
	from paddleocr import PaddleOCR
except Exception as e:
	raise RuntimeError(
		"Install PaddleOCR first: pip install paddleocr==2.7.0.3\n"
		f"Original import error: {e}"
	)

# Optional PPStructureV3
try:
	from paddleocr import PPStructure
	_HAS_PPSTRUCT = True
except Exception:
	_HAS_PPSTRUCT = False


class Models:
	def __init__(self, yolo_path: str):
		logger.info(f"Loading YOLO model: {yolo_path}")
		self.yolo = YOLO(yolo_path)

		logger.info("Initializing PaddleOCR...")
		self.ocr = PaddleOCR(use_angle_cls=True, lang='en')
		logger.info("PaddleOCR ready.")

		if _HAS_PPSTRUCT:
			logger.info("PPStructure available. Using lightweight defaults.")
			self.ppstruct = PPStructure(show_log=False)
		else:
			self.ppstruct = None


def detect_answer_regions_yolo(models: Models, image: Image.Image) -> List[Detection]:
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
			bbox = ensure_int_bbox([xyxy[0], xyxy[1], xyxy[2], xyxy[3]])
			dets.append(Detection(bbox=bbox, conf=conf, cls_name=cls_name))

	return dets


def ocr_image(models: Models, image: Image.Image) -> List[OCRLine]:
	img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
	out = models.ocr.predict(img_bgr)

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
			bbox = ensure_int_bbox(_polygon_to_bbox(poly) if poly else [0, 0, 0, 0])
			lines.append(OCRLine(text=txt, confidence=conf, bbox=bbox))
	else:
		for entry in result:
			try:
				poly, (txt, conf) = entry
				txt = (txt or "").strip()
				if not txt:
					continue
				bbox = ensure_int_bbox(_polygon_to_bbox(poly))
				lines.append(OCRLine(text=txt, confidence=float(conf), bbox=bbox))
			except Exception:
				continue

	return lines


def _polygon_to_bbox(poly):
	xs = [p[0] for p in poly]
	ys = [p[1] for p in poly]
	return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def run_ppstructure(models: Models, image: Image.Image) -> Optional[str]:
	if not models.ppstruct:
		return None
	img_np = np.array(image)
	result = models.ppstruct(img_np)
	buf = []
	for res in result:
		plist = res.get("res", {}).get("layout_res", []) or res.get("res", [])
		for elem in plist:
			label = elem.get("label", "region")
			text = elem.get("text", "").strip()
			if text:
				buf.append(f"[{label}] {text}")
	return "\n".join(buf) if buf else None



