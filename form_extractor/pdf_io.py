from typing import List
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path


def is_pdf(file_path: str) -> bool:
	"""Check if file is a PDF by extension."""
	return Path(file_path).suffix.lower() == ".pdf"


def is_image(file_path: str) -> bool:
	"""Check if file is an image by extension."""
	image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp"}
	return Path(file_path).suffix.lower() in image_extensions


def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
	"""
	Load images from PDF or image file.
	- If PDF: converts to images using pdf2image (requires poppler installed)
	- If image: loads directly using PIL
	"""
	if is_pdf(pdf_path):
		# Requires poppler installed on system. On Windows, set POPPLER_PATH env or pass to convert_from_path.
		return convert_from_path(pdf_path, dpi=dpi)
	elif is_image(pdf_path):
		img = Image.open(pdf_path)
		# Convert to RGB if necessary (e.g., RGBA, P mode)
		if img.mode != "RGB":
			img = img.convert("RGB")
		return [img]
	else:
		raise ValueError(
			f"Unsupported file type: {pdf_path}. "
			f"Supported formats: PDF (.pdf) or images (.jpg, .jpeg, .png, .bmp, .tiff, .gif, .webp)"
		)



