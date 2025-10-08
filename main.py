from form_extractor.pipeline import process_pdf
from form_extractor.config import YOLO_MODEL_PATH, OUTPUT_DIR

if __name__ == "__main__":
	import argparse

	ap = argparse.ArgumentParser(description="Extract form schema from a PDF.")
	ap.add_argument("pdf", help="Path to form PDF")
	ap.add_argument("--yolo", default=YOLO_MODEL_PATH, help="Path to YOLO weights (default: best.pt or $YOLO_MODEL_PATH)")
	ap.add_argument("--out", default=OUTPUT_DIR, help="Output dir (default: ./output)")
	ap.add_argument("--no-debug", action="store_true", help="Disable annotated debug images")
	args = ap.parse_args()

	process_pdf(args.pdf, yolo_weights=args.yolo, out_dir=args.out, make_debug_images=not args.no_debug)


