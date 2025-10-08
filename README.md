## End-to-end Form Schema Extractor

### Setup
1. Install system dependencies:
   - POPPLER (required by pdf2image)
   - CUDA/cuDNN (optional) for GPU inference
2. Python deps:
```bash
pip install -r requirements.txt
```
3. Configure environment:
- Copy `.env.example` to `.env` and fill values as needed.

### Environment
- `YOLO_MODEL_PATH`: path to YOLO weights (e.g., `best.pt`)
- `OUTPUT_DIR`: output directory (default: `output`)
- `LLM_PROVIDER`: `openai` | `ollama` | `gemini`
- Provider API keys/hosts: `OPENAI_API_KEY`, `OLLAMA_HOST`, `GEMINI_API_KEY`

### Run
```bash
python main.py path/to/form.pdf --yolo best.pt --out output
```
Use `--no-debug` to skip annotated images.

### Outputs
- `output/annotated/page_XXX_annotated.jpg`
- `output/document_schema.json` (raw + normalized)

### Notes
- Coordinates are pixel-based [x1,y1,x2,y2], origin at top-left.
- PPStructureV3 is optional; if unavailable it is skipped.

