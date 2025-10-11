import os
import json
from typing import Any, Dict

from .config import LLM_PROVIDER, OPENAI_API_KEY, OLLAMA_HOST, GEMINI_API_KEY
from .logging_utils import logger


def ask_llm_normalize_schema(provider: str, raw_schema: Dict[str, Any]) -> Dict[str, Any]:
	prompt = (
		"You are given a raw extracted schema for a form. "
		"Normalize it to a clean JSON with stable ids (snake_case), "
		"human-readable labels, types (text, checkbox, radio, signature, date, table_cell, unknown), "
		"and preserve all bounding boxes + page numbers exactly. "
		"If you can infer option labels for checkbox/radio from nearby text, add them under 'options'. "
		"Return ONLY valid minified JSON.\n\nRAW:\n"
		+ json.dumps(raw_schema, ensure_ascii=False)
	)

	if provider == "ollama":
		return _ask_ollama_json(prompt)
	elif provider == "openai":
		return _ask_openai_json(prompt)
	elif provider == "gemini":
		return _ask_gemini_json(prompt)
	else:
		logger.warning(f"Unknown LLM_PROVIDER '{provider}', returning raw schema.")
		return raw_schema


def _ask_ollama_json(prompt: str) -> Dict[str, Any]:
	import requests
	model = os.getenv("OLLAMA_MODEL", "llama3.1:latest")

	r = requests.post(
		f"{OLLAMA_HOST}/api/generate",
		json={"model": model, "prompt": prompt, "stream": False, "format": "json"}
	)
	r.raise_for_status()
	obj = r.json()
	txt = obj.get("response", "{}")
	try:
		return json.loads(txt)
	except Exception:
		try:
			start = txt.find("{")
			end = txt.rfind("}")
			return json.loads(txt[start:end+1])
		except Exception:
			logger.warning("Ollama returned non-JSON; falling back to raw.")
			return {"raw_llm_text": txt}


def _ask_openai_json(prompt: str) -> Dict[str, Any]:
	try:
		from openai import OpenAI
	except Exception as e:
		logger.error("pip install openai to use OPENAI provider.")
		return {"error": str(e)}

	client = OpenAI(api_key=OPENAI_API_KEY)
	resp = client.chat.completions.create(
		model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
		response_format={"type": "json_object"},
		messages=[
			{"role": "system", "content": "Return only valid compact JSON."},
			{"role": "user", "content": prompt}
		],
		temperature=0
	)
	txt = resp.choices[0].message.content
	try:
		return json.loads(txt)
	except Exception:
		return {"raw_llm_text": txt}


def _ask_gemini_json(prompt: str) -> Dict[str, Any]:
	try:
		import google.generativeai as genai
	except Exception as e:
		logger.error("pip install google-generativeai to use GEMINI provider.")
		return {"error": str(e)}

	genai.configure(api_key=GEMINI_API_KEY)
	model = genai.GenerativeModel(os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
	resp = model.generate_content(
		[
			{"role": "user", "parts": [prompt]}
		],
		generation_config={"response_mime_type": "application/json"}
	)
	txt = resp.text or "{}"
	try:
		return json.loads(txt)
	except Exception:
		return {"raw_llm_text": txt}



