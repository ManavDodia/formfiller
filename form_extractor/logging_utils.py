import logging

logger = logging.getLogger("form_extractor")
if not logger.handlers:
	logger.setLevel(logging.INFO)
	handler = logging.StreamHandler()
	handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
	logger.addHandler(handler)

