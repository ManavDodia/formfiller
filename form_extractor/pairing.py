from typing import List, Tuple

from .types import OCRLine, Detection, FieldCandidate
from .geometry import bbox_center, euclidean

QUESTION_HARD_MARKERS = (":", "?", "->")


def looks_like_question(line: OCRLine, page_w: int) -> bool:
	t = line.text
	if not t:
		return False
	if t.endswith(QUESTION_HARD_MARKERS):
		return True
	if len(t) <= 60 and line.bbox[0] < page_w * 0.55:
		if any(ch.isalpha() for ch in t):
			w = line.bbox[2] - line.bbox[0]
			h = line.bbox[3] - line.bbox[1]
			if w >= 15 and h >= 10:
				return True
	return False


def group_choices_nearby(dets: List[Detection], max_gap_px: int = 60) -> List[List[Detection]]:
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
	qbox,
	dets: List[Detection],
	page_w: int,
	right_dx: int = 350,
	row_dy: int = 120
) -> Tuple[str, List[list], List[str]]:
	qcx, qcy = bbox_center(qbox)

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

	choice_dets = [d for d in dets if d.cls_name in ("checkbox", "radio")]
	if choice_dets:
		groups = group_choices_nearby(dets)
		best_g = None
		best_dist = 1e9
		for g in groups:
			gc = bbox_center([
				min(x.bbox[0] for x in g),
				min(x.bbox[1] for x in g),
				max(x.bbox[2] for x in g),
				max(x.bbox[3] for x in g),
			])
			d = euclidean((qcx, qcy), gc)
			if d < best_dist:
				best_g, best_dist = g, d
		if best_g and best_dist < 300:
			return ("radio" if all(x.cls_name == "radio" for x in best_g) else "checkbox",
					[x.bbox for x in best_g],
					[x.cls_name for x in best_g])

	any_text: List[Tuple[float, Detection]] = []
	for d in dets:
		if d.cls_name in ("text_field", "line_field"):
			acx, acy = bbox_center(d.bbox)
			any_text.append((euclidean((qcx, qcy), (acx, acy)), d))
	if any_text:
		any_text.sort(key=lambda t: t[0])
		d = any_text[0][1]
		return "text", [d.bbox], [d.cls_name]

	return "unknown", [], []


def build_fields_for_page(
	ocr_lines: List[OCRLine],
	dets: List[Detection],
	page_w: int,
	page_h: int
) -> List[FieldCandidate]:
	questions = [l for l in ocr_lines if looks_like_question(l, page_w)]
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


