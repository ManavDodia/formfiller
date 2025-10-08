import json
import time
from pathlib import Path
from typing import List

from PIL import Image

from .config import YOLO_MODEL_PATH, OUTPUT_DIR, LLM_PROVIDER
from .logging_utils import logger
from .types import PageResult, DocumentSchema
from .pdf_io import pdf_to_images
from .models import Models, detect_answer_regions_yolo, ocr_image
from .pairing import build_fields_for_page
from .drawing import draw_debug
from .llm import ask_llm_normalize_schema


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

	raw_schema = {
		"document_id": Path(pdf_path).stem,
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

	logger.info("Calling LLM to normalize schema...")
	normalized = ask_llm_normalize_schema(LLM_PROVIDER, raw_schema)

	schema_path = out_root / "document_schema.json"
	with open(schema_path, "w", encoding="utf-8") as f:
		json.dump({"raw": raw_schema, "normalized": normalized}, f, indent=2, ensure_ascii=False)

	logger.info(f"Done. Wrote schema to: {schema_path}")
	logger.info(f"Total time: {time.time() - t0:.1f}s")

	doc = DocumentSchema(pages=page_results, normalized_schema=normalized)
	return doc


