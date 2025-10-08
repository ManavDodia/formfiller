import os
from pathlib import Path

# Load .env if available
try:
	from dotenv import load_dotenv
	load_dotenv()
except Exception:
	pass

YOLO_MODEL_PATH: str = os.getenv("YOLO_MODEL_PATH", "best.pt")
OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "output")
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")  # "openai" | "ollama" | "gemini"

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

