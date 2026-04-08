# RESA_AI

RESA_AI is a local AI-powered analysis system for mathematical research papers. It takes a paper as input, parses its structure, runs multiple NLP and transformer stages, generates Groq-based insights, and produces both machine-readable and report-style outputs.

This repository includes:

- A Python backend pipeline for PDF, LaTeX, and plain-text paper analysis
- A lightweight frontend dashboard for PDF upload, live phase tracking, and report exploration
- JSON, Markdown, and chart outputs saved to the `results/` directory
- GPU-aware execution with an automatic low-VRAM path for smaller cards

## What The Project Does

Given a research paper, RESA_AI can:

- Parse title, authors, abstract, sections, equations, and references
- Clean and split mathematical content into natural-language and math-aware representations
- Extract keywords, entities, and mathematical structures
- Generate extractive and abstractive summaries
- Classify sections and compute semantic similarity
- Discover topics and build a semantic search index
- Analyze citations and equation types
- Generate high-level paper insights with Groq
- Produce a structured JSON report, a Markdown report, and visualization assets

The frontend is designed around the same flow:

1. Upload a PDF
2. Watch the paper move through each pipeline phase
3. Explore the final analysis through summary, ELI5, keywords, topics, equations, visuals, and report panels

## Core Highlights

- Research-paper focused pipeline, especially for dense mathematical documents
- Web dashboard with live job polling and phase-by-phase progress
- CLI entrypoint for direct batch or local execution
- Groq integration for summary, ELI5, contributions, applications, and limitations
- Multiple output formats: JSON, Markdown, and PNG charts
- Automatic GPU detection through PyTorch
- Low-VRAM mode for GPUs under 6 GB
- Single-job protection in the web app to avoid overlapping heavy analyses

## Architecture Overview

```text
PDF upload or CLI input
        |
        v
main.py orchestrator
        |
        +--> Phase 1: ingestion
        +--> Phase 2: preprocessing
        +--> Phase 3: feature engineering
        +--> Phase 4: transformer models
        +--> Phase 5: advanced NLP analysis
        +--> Phase 6: Groq LLM analysis
        +--> Phase 7: output generation
        |
        v
results/report.json
results/final_report.md
results/*.png
        |
        v
frontend_server.py exposes latest report and live job status
        |
        v
frontend/ dashboard renders the workspace
```

## Tech Stack

### Frontend

| Layer | Technology | Role |
|---|---|---|
| UI | HTML5 | Static document structure |
| Styling | CSS3 | SaaS-style dashboard layout and components |
| Interactivity | Vanilla JavaScript | Upload flow, polling, rendering report sections |
| Rendering model | No framework | No build step, no React/Vue dependency |

### Application Layer

| Layer | Technology | Role |
|---|---|---|
| Runtime | Python 3.11+ | Main execution environment |
| Web server | `http.server.ThreadingHTTPServer` | Local frontend hosting and API endpoints |
| Upload parsing | `cgi.FieldStorage` | Multipart PDF upload handling |
| Background jobs | `threading` + `subprocess` | Launch and monitor analysis jobs |
| Data format | JSON | Job state and final structured report |

### NLP, ML, and LLM Stack

| Area | Technology | Role |
|---|---|---|
| PDF parsing | PyMuPDF (`fitz`) | Structured PDF extraction |
| Alt PDF support | `pdfplumber` | Additional PDF processing dependency |
| Text cleaning | `re`, `unicodedata2` | Normalization and math-safe cleanup |
| Named entities | spaCy `en_core_web_sm` | People, orgs, and academic entities |
| Statistical NLP | scikit-learn | TF-IDF scoring and keyword extraction support |
| Keyword extraction | YAKE, KeyBERT | Multi-method keyword discovery |
| Deep learning | PyTorch | CPU/GPU execution |
| Seq2seq summarization | Hugging Face Transformers | `facebook/bart-large-cnn` |
| Zero-shot classification | Hugging Face Transformers | `facebook/bart-large-mnli` |
| Embeddings | Sentence Transformers | `all-MiniLM-L6-v2` |
| Topic modeling | BERTopic | Semantic topic discovery |
| LLM reasoning | Groq Python SDK | Summary, ELI5, contributions, applications, limitations |

### Visualization and Outputs

| Output | Technology | Role |
|---|---|---|
| Structured report | JSON | Machine-readable full analysis |
| Human report | Markdown | Reviewer-friendly report output |
| Charts | Matplotlib | Keyword, equation, citation, classification, topic, and math-density visuals |
| Numeric processing | NumPy | Embeddings, similarity matrices, chart data |

### Hardware and Runtime Characteristics

| Area | Behavior |
|---|---|
| CPU mode | Fully supported |
| GPU mode | Enabled automatically when CUDA is available to PyTorch |
| Low-VRAM mode | Enabled automatically when total GPU memory is under 6 GB |
| Model downloads | Hugging Face models download on first run |
| Groq dependency | Optional, but required for Phase 6 LLM insights |

## Project Structure

```text
RESA_AI/
|-- main.py
|-- config.py
|-- frontend_server.py
|-- requirements.txt
|-- README.md
|-- LICENSE
|
|-- ingestion/
|   |-- pdf_parser.py
|   |-- latex_parser.py
|   `-- text_parser.py
|
|-- preprocessing/
|   |-- cleaner.py
|   |-- dual_repr.py
|   `-- tokenizer.py
|
|-- features/
|   |-- keyword_extractor.py
|   |-- entity_extractor.py
|   `-- math_structure.py
|
|-- models/
|   |-- longdoc_handler.py
|   |-- summarizer.py
|   |-- scibert_classifier.py
|   `-- similarity.py
|
|-- analysis/
|   |-- topic_modeling.py
|   |-- semantic_search.py
|   |-- citation_analysis.py
|   `-- equation_classifier.py
|
|-- llm/
|   |-- groq_client.py
|   `-- prompt_builder.py
|
|-- output/
|   |-- json_report.py
|   |-- final_report.py
|   `-- visualizations.py
|
|-- frontend/
|   |-- index.html
|   |-- styles.css
|   `-- app.js
|
|-- results/
|   |-- report.json
|   |-- final_report.md
|   `-- *.png
|
|-- cache/
|   |-- search_index.npy
|   `-- section_embeddings.npy
|
`-- data/
    `-- uploads/
```

## Step-By-Step Process

### User Flow

1. The user opens the local dashboard served by `frontend_server.py`.
2. The user uploads a PDF through the web UI.
3. The server saves the file to `data/uploads/` and creates a job record.
4. The server starts `main.py` in a background subprocess.
5. The frontend polls `/api/jobs/<job_id>` and updates the progress timeline.
6. Each pipeline phase updates the current job message and phase status.
7. When the run completes, the server loads `results/report.json`.
8. The frontend renders the report workspace from the generated report and chart assets.

### Internal Pipeline, Phase By Phase

#### Phase 1: Data ingestion

Primary modules:

- `ingestion/pdf_parser.py`
- `ingestion/latex_parser.py`
- `ingestion/text_parser.py`

What happens:

- Input type is auto-detected from the file extension, unless overridden with `--type`
- PDFs are parsed with PyMuPDF
- The parser extracts title, authors, abstract, sections, equations, references, raw text, and document metadata
- Section detection uses regex heuristics and heading-like patterns
- Equation detection uses math-symbol and LaTeX-style pattern matching

Main output keys:

- `title`
- `authors`
- `abstract`
- `sections`
- `equations`
- `references`
- `raw_text`
- `metadata`

#### Phase 2: Preprocessing

Primary modules:

- `preprocessing/cleaner.py`
- `preprocessing/dual_repr.py`

What happens:

- Raw text is cleaned and normalized
- Mathematical expressions wrapped as `[MATH]...[/MATH]` are preserved
- The document is split into two parallel views:
  - natural-language text for NLP models
  - extracted math expressions for math-specific analysis
- Per-section math density is calculated

Main output keys:

- `cleaned_text`
- `nl_text`
- `nl_sections`
- `all_math_expressions`
- `math_stats`

#### Phase 3: Feature engineering

Primary modules:

- `features/keyword_extractor.py`
- `features/entity_extractor.py`
- `features/math_structure.py`

What happens:

- Keywords are extracted using TF-IDF, YAKE, and KeyBERT
- Keyword candidates are fused into a ranked list
- spaCy extracts named entities
- Regex rules detect methods, theorems, lemmas, propositions, definitions, and math concepts
- Mathematical structure signals are summarized

Main output keys:

- `keywords`
- `entities`
- `math_structures`

#### Phase 4: Transformer models

Primary modules:

- `models/longdoc_handler.py`
- `models/summarizer.py`
- `models/scibert_classifier.py`
- `models/similarity.py`

What happens:

- Long sections are chunked for safer downstream model processing
- A two-part summary is generated:
  - extractive summary via TF-IDF and sentence scoring
  - abstractive summary via `facebook/bart-large-cnn`
- Section classification is attempted with `facebook/bart-large-mnli`
- A keyword fallback classifier is used when the model is unavailable or low-VRAM mode is active
- Sentence-transformer embeddings are created and used to compute inter-section similarity

Important note:

- Despite the file name `scibert_classifier.py`, the active zero-shot classifier in the current pipeline is `facebook/bart-large-mnli`

Main output keys:

- `chunks`
- `summary`
- `classifications`
- `similarity`

#### Phase 5: Advanced NLP analysis

Primary modules:

- `analysis/topic_modeling.py`
- `analysis/semantic_search.py`
- `analysis/citation_analysis.py`
- `analysis/equation_classifier.py`

What happens:

- BERTopic groups the paper into semantic topics
- A sentence-level semantic search index is built and cached
- Citation frequency and citation density are analyzed
- Equations are classified into high-level math domains
- Fallback logic is used when BERTopic is unavailable or the paper is too small

Main output keys:

- `topics`
- `search_index`
- `citation_analysis`
- `equation_analysis`

#### Phase 6: Groq LLM analysis

Primary modules:

- `llm/groq_client.py`
- `llm/prompt_builder.py`

What happens:

- The pipeline builds a context packet from title, abstract, sections, keywords, math structure, equation type, and key sentences
- Groq generates:
  - summary
  - ELI5 explanation
  - contributions
  - applications
  - limitations
- Primary model: `llama-3.3-70b-versatile`
- Fallback model: `mixtral-8x7b-32768`
- If no `GROQ_API_KEY` is available, this stage is skipped gracefully

Main output keys:

- `llm_analysis`

#### Phase 7: Output generation

Primary modules:

- `output/json_report.py`
- `output/final_report.py`
- `output/visualizations.py`

What happens:

- The full analysis is written to `results/report.json`
- A readable report is written to `results/final_report.md`
- Visualization assets are created as PNG files
- Embedding caches remain under `cache/`

Main output keys:

- `json_report`
- `json_report_path`
- `final_report_path`
- `visualizations`

## Frontend Workflow

The frontend is intentionally lightweight and framework-free.

Frontend files:

- `frontend/index.html`
- `frontend/styles.css`
- `frontend/app.js`

Server file:

- `frontend_server.py`

Current dashboard behavior:

- Uploads PDF files through `/api/analyze`
- Polls `/api/jobs/<job_id>` for live status
- Loads the most recent saved report from `/api/report/latest`
- Displays progress through seven high-level phases
- Surfaces summary, ELI5, contributions, keywords, topics, equation breakdown, visuals, and report sections
- Blocks parallel uploads while one heavy analysis job is already running

## Generated Outputs

### Primary files

| File | Purpose |
|---|---|
| `results/report.json` | Main structured analysis output |
| `results/final_report.md` | Human-readable report |
| `results/keyword_chart.png` | Keyword relevance chart |
| `results/equation_distribution.png` | Equation-type distribution |
| `results/topic_chart.png` | Topic distribution chart |

### Additional charts that may also be generated

- `results/classification_chart.png`
- `results/citation_chart.png`
- `results/math_density.png`

### Report schema summary

The JSON report contains these top-level sections:

```json
{
  "metadata": {},
  "abstract": "",
  "structure": {},
  "mathematics": {},
  "insights": {},
  "topics": {},
  "classifications": [],
  "summary": {},
  "citations": {},
  "similarity": {},
  "llm_analysis": {}
}
```

## Installation

### Requirements

- Python 3.11 or newer
- `pip`
- Optional NVIDIA GPU with a CUDA-enabled PyTorch build
- Groq API key if you want Phase 6 LLM outputs

### Setup

```powershell
cd C:\path\to\RESA_AI
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Groq API key

PowerShell, current session only:

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
```

PowerShell, persist for future sessions:

```powershell
setx GROQ_API_KEY "your_groq_api_key"
```

## How To Run

### Fastest option

From the project folder, run just one command:

```powershell
python start_resa.py
```

Or on Windows:

```powershell
.\start_resa.cmd
```

What it does:

- starts the local frontend server if it is not already running
- reuses a healthy running server if one already exists
- picks up the saved Groq key automatically
- waits until the app is reachable
- opens the browser to `http://127.0.0.1:8000`

### Option 1: Run the CLI pipeline

Analyze a PDF:

```powershell
python main.py .\paperDCRE-1.pdf
```

Analyze a LaTeX file:

```powershell
python main.py .\paper.tex --type latex
```

Analyze a text file:

```powershell
python main.py .\paper.txt --type text
```

Skip Groq:

```powershell
python main.py .\paperDCRE-1.pdf --skip-groq
```

Skip transformer-heavy phases:

```powershell
python main.py .\paperDCRE-1.pdf --skip-models
```

### Option 2: Run the web dashboard

```powershell
python frontend_server.py
```

Then open:

- `http://127.0.0.1:8000`

## API Surface Used By The Frontend

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Serves the dashboard |
| `POST` | `/api/analyze` | Accepts uploaded PDF and starts a job |
| `GET` | `/api/jobs/<job_id>` | Returns live job state |
| `GET` | `/api/report/latest` | Returns the latest completed report |
| `GET` | `/results/<file>` | Serves generated charts and artifacts |

## Configuration

Central configuration lives in `config.py`.

Important values:

| Name | Meaning |
|---|---|
| `SUMMARIZER_MODEL` | Abstractive summarization model |
| `ZERO_SHOT_MODEL` | Zero-shot classification model |
| `SENTENCE_TRANSFORMER_MODEL` | Embedding model |
| `GROQ_MODEL` | Primary Groq model |
| `GROQ_FALLBACK_MODEL` | Secondary Groq model |
| `CHUNK_SIZE` | Long-document chunk size |
| `CHUNK_OVERLAP` | Chunk overlap |
| `TOP_K_KEYWORDS` | Final number of surfaced keywords |
| `TOP_K_SUMMARY_SENTENCES` | Extractive summary sentence count |
| `CUDA_MEMORY_FRACTION` | PyTorch GPU memory fraction |
| `LOW_VRAM_THRESHOLD_GB` | Threshold for low-VRAM mode |

## GPU And Performance Notes

RESA_AI supports both CPU and GPU execution, but a GPU does not make every phase instant. Some steps remain CPU-heavy or I/O-bound, and the first run can be slow because models may need to download.

Current performance behavior:

- PyTorch selects CUDA automatically if available
- Low-VRAM mode activates automatically below 6 GB GPU memory
- In low-VRAM mode:
  - abstractive summarization uses shorter inputs and cheaper generation settings
  - zero-shot section classification is skipped in favor of keyword fallback
  - similarity embedding batch size is reduced
- The frontend only allows one active job at a time to avoid GPU contention

## Troubleshooting

### The app says GPU is not available

That usually means PyTorch cannot see CUDA, not that the machine has no GPU. Check:

- `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"`

### Transformer phase feels slow

That phase includes several expensive substeps:

- long-document chunking
- summary generation
- section classification
- semantic similarity

On low-VRAM GPUs this is expected to be the slowest stage, even when the system is working correctly.

### Groq analysis is missing

Check that:

- `groq` is installed
- `GROQ_API_KEY` is set in the environment used to launch the app

### Frontend shows no report

Check whether:

- `results/report.json` exists
- the server is running on `127.0.0.1:8000`
- a job has completed successfully

## Notes On Included Sample Content

This repository includes sample output files and a sample paper for demonstration. If you add third-party papers, datasets, or external assets, their original license and copyright terms may still apply.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
