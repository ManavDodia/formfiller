from typing import List
from PIL import Image
from pdf2image import convert_from_path


def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
	"""
	Requires poppler installed on system. On Windows, set POPPLER_PATH env or pass to convert_from_path.
	"""
	return convert_from_path(pdf_path, dpi=dpi)



