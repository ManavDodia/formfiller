from typing import List, Tuple

BBox = List[int]

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
	return ((p1[0]-p2[0]) ** 2 + (p1[1]-p2[1]) ** 2) ** 0.5

def polygon_to_bbox(poly: List[List[float]]) -> BBox:
	xs = [p[0] for p in poly]
	ys = [p[1] for p in poly]
	return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]



