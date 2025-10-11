from typing import List
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

from .types import OCRLine, Detection, FieldCandidate
from .geometry import bbox_center


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

	for d in dets:
		draw.rectangle(d.bbox, outline="green", width=2)
		draw.text((d.bbox[0], d.bbox[1]-16), f"{d.cls_name} {d.conf:.2f}", fill="green", font=font)

	for l in ocr_lines:
		draw.rectangle(l.bbox, outline="orange", width=1)
		if l.text:
			draw.text((l.bbox[0], l.bbox[1]-14), l.text[:40], fill="orange", font=font)

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



