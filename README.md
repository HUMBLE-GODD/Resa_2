<p align="center">
  <h1 align="center">🧠 RESA_AI</h1>
  <p align="center">
    <strong>Production-Grade NLP Pipeline for Mathematical Research Papers</strong>
  </p>
  <p align="center">
    Ingest PDFs → Extract Structure → NLP Analysis → LLM Insights → Structured Reports
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Transformers-4.30+-yellow?logo=huggingface&logoColor=white" />
  <img src="https://img.shields.io/badge/Groq-LLaMA_3.3_70B-green?logo=meta&logoColor=white" />
  <img src="https://img.shields.io/badge/GPU-Optimized-76b900?logo=nvidia&logoColor=white" />
</p>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Pipeline Flow — Phase by Phase](#pipeline-flow--phase-by-phase)
  - [Phase 1 — Data Ingestion](#phase-1--data-ingestion)
  - [Phase 2 — Preprocessing](#phase-2--preprocessing)
  - [Phase 3 — Feature Engineering](#phase-3--feature-engineering)
  - [Phase 4 — Transformer Models](#phase-4--transformer-models)
  - [Phase 5 — Advanced NLP Analysis](#phase-5--advanced-nlp-analysis)
  - [Phase 6 — LLM Analysis (Groq)](#phase-6--llm-analysis-groq)
  - [Phase 7 — Output Generation](#phase-7--output-generation)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output Format](#output-format)
- [GPU Optimization](#gpu-optimization)
- [Models Used](#models-used)

---

## Overview

**RESA_AI** is a modular, end-to-end Natural Language Processing pipeline specifically designed for analyzing **mathematical research papers**. It handles the unique challenges of academic math text — LaTeX expressions, symbolic notation, dense technical language, and formal proof structures.

Given a PDF research paper, RESA_AI automatically:

1. **Parses** the document into structured sections, equations, and references
2. **Cleans and tokenizes** text while preserving mathematical expressions
3. **Extracts** keywords, named entities, and mathematical structures (theorems, proofs, lemmas)
4. **Classifies** sections and generates abstractive summaries using transformers
5. **Discovers topics**, builds a semantic search index, and analyzes citations
6. **Generates LLM-powered insights** via Groq API (summary, ELI5, contributions, applications, limitations)
7. **Outputs** a structured JSON report, markdown document, and publication-quality visualizations

---

## Key Features

| Feature | Description |
|---------|-------------|
| 🔬 **Math-Aware Processing** | Preserves LaTeX/math expressions with `[MATH]...[/MATH]` tagging throughout the entire pipeline |
| ⚡ **GPU-Optimized** | Auto-detects CUDA, uses FP16 mixed precision, TF32 on Ampere+, and dynamic batch sizing |
| 🧩 **Modular Architecture** | 8 independent phases — each can run, test, and fail independently |
| 🤖 **Multi-Model** | BART-large-CNN (summarization), BART-large-MNLI (zero-shot), all-MiniLM-L6-v2 (embeddings), BERTopic (topics) |
| 🌐 **Groq LLM Integration** | LLaMA 3.3 70B with Mixtral 8x7B fallback for deep reasoning |
| 📊 **Rich Visualizations** | Dark-themed matplotlib charts: keywords, equation distribution, topic clusters |
| 🛡️ **Fault-Tolerant** | Keyword-based classification fallback, graceful degradation on CPU, try/except isolation per sub-step |
| 💾 **Embedding Caching** | Sentence embeddings cached to disk for repeat analyses |

---

## Tech Stack

### Core Framework

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.11+ | Core runtime |
| **Deep Learning** | PyTorch 2.0+ | GPU acceleration, tensor operations |
| **Transformers** | HuggingFace Transformers 4.30+ | BART summarization, zero-shot classification |
| **Embeddings** | Sentence-Transformers 2.2+ | Dense semantic embeddings (all-MiniLM-L6-v2) |
| **NLP** | spaCy 3.5+ (`en_core_web_sm`) | Named entity recognition, POS tagging |
| **Topic Modeling** | BERTopic 0.15+ | Neural topic discovery with UMAP + HDBSCAN |

### PDF & Text Processing

| Library | Purpose |
|---------|---------|
| **PyMuPDF (fitz)** | PDF parsing — text blocks, font sizes, page layout |
| **pdfplumber** | Alternative PDF extraction |
| **re / unicodedata** | Regex-based math extraction, Unicode normalization |

### Keyword Extraction (Triple Method)

| Method | Library | Strategy |
|--------|---------|----------|
| **TF-IDF** | scikit-learn | Statistical term importance |
| **YAKE** | yake | Unsupervised keyword extraction |
| **KeyBERT** | keybert | Transformer-based semantic keywords |

### LLM Integration

| Provider | Model | Role |
|----------|-------|------|
| **Groq API** | `llama-3.3-70b-versatile` | Primary reasoning (summary, ELI5, contributions) |
| **Groq API** | `mixtral-8x7b-32768` | Fallback model |

### Visualization & Output

| Library | Purpose |
|---------|---------|
| **Matplotlib 3.7+** | Dark-themed publication-quality charts |
| **Seaborn 0.12+** | Statistical visualization |
| **JSON** | Structured machine-readable report |
| **Markdown** | Human-readable final report |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        RESA_AI PIPELINE                             │
│                        main.py (Orchestrator)                        │
└───────────────────────────────┬──────────────────────────────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     │                     │
          ▼                     ▼                     ▼
   ┌─────────────┐    ┌─────────────┐    ┌──────────────────┐
   │   config.py  │    │  paperDCRE  │    │  results/        │
   │  GPU/Models  │    │  -1.pdf     │    │  report.json     │
   │  API Keys    │    │  (Input)    │    │  final_report.md │
   │  Parameters  │    │             │    │  *.png charts    │
   └─────────────┘    └──────┬──────┘    └──────────────────┘
                              │
          ┌───────────────────┼───────────────────────────────┐
          │                   │                               │
          ▼                   ▼                               ▼
┌──────────────────────────────────────────────────────────────────┐
│                     PHASE 1: DATA INGESTION                      │
│  ingestion/                                                      │
│  ├── pdf_parser.py    → PyMuPDF font-size heuristics             │
│  ├── latex_parser.py  → LaTeX \begin{} \end{} parsing            │
│  └── text_parser.py   → Regex-based section detection            │
│                                                                   │
│  Output: { title, authors, abstract, sections[], equations[],     │
│            references[], raw_text, pages }                        │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                     PHASE 2: PREPROCESSING                       │
│  preprocessing/                                                   │
│  ├── cleaner.py    → Noise removal, Unicode normalization         │
│  ├── tokenizer.py  → SciBERT tokenization with [MATH] tokens     │
│  └── dual_repr.py  → Separate NL text and math expressions       │
│                                                                   │
│  Output: { cleaned_text, nl_text, math_expressions[],             │
│            nl_sections[], inline_math[], display_math[] }         │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                  PHASE 3: FEATURE ENGINEERING                    │
│  features/                                                        │
│  ├── keyword_extractor.py  → TF-IDF + YAKE + KeyBERT fusion      │
│  ├── entity_extractor.py   → spaCy NER + custom math rules       │
│  └── math_structure.py     → Theorem/Proof/Lemma detection        │
│                                                                   │
│  Output: { keywords{tfidf, yake, keybert, combined},              │
│            entities{persons, methods, concepts, math_entities},   │
│            math_structures{definitions, theorems, proofs,         │
│                            relationships[]} }                     │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                  PHASE 4: TRANSFORMER MODELS                     │
│  models/                                                          │
│  ├── longdoc_handler.py     → Sliding window chunking (512 tok)   │
│  ├── summarizer.py          → Extractive (TF-IDF) + Abstractive  │
│  │                            (BART-large-CNN)                    │
│  ├── scibert_classifier.py  → Zero-shot (BART-MNLI) + keyword    │
│  │                            fallback                            │
│  └── similarity.py          → Sentence-transformer cosine sim    │
│                                                                   │
│  Output: { chunks[], summary{extractive, abstractive, combined},  │
│            classifications[], similarity_matrix }                 │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                PHASE 5: ADVANCED NLP ANALYSIS                    │
│  analysis/                                                        │
│  ├── topic_modeling.py     → BERTopic (UMAP + HDBSCAN + TF-IDF)  │
│  ├── semantic_search.py    → Dense retrieval with MiniLM          │
│  ├── citation_analysis.py  → Frequency, context, purpose          │
│  └── equation_classifier.py→ Rule-based equation type (9 domains) │
│                                                                   │
│  Output: { topics[], search_index, citation_analysis{},           │
│            equation_classifications[] }                           │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                  PHASE 6: GROQ LLM ANALYSIS                      │
│  llm/                                                             │
│  ├── groq_client.py    → API client with LLaMA/Mixtral fallback   │
│  └── prompt_builder.py → Structured prompt templates              │
│                                                                   │
│  Output: { llm_insights{summary, eli5, contributions,             │
│                          applications, limitations} }             │
└────────────────────────────────┬─────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────┐
│                  PHASE 7: OUTPUT GENERATION                      │
│  output/                                                          │
│  ├── json_report.py     → Structured JSON with all results        │
│  ├── visualizations.py  → Dark-themed matplotlib charts           │
│  └── final_report.py    → Comprehensive markdown report           │
│                                                                   │
│  Output Files:                                                    │
│  ├── results/report.json           (38 KB structured data)        │
│  ├── results/final_report.md       (21 KB human-readable)         │
│  ├── results/keyword_chart.png     (keyword bar chart)            │
│  ├── results/equation_distribution.png (equation pie chart)       │
│  └── results/topic_chart.png       (topic distribution bars)      │
└──────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
RESA_AI/
│
├── main.py                     # Pipeline orchestrator — runs all 7 phases
├── config.py                   # Global config: GPU, models, API keys, parameters
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── ingestion/                  # PHASE 1: Document Parsing
│   ├── __init__.py
│   ├── pdf_parser.py           # PyMuPDF-based PDF → structured extraction
│   ├── latex_parser.py         # LaTeX source → structured extraction
│   └── text_parser.py          # Plain text → heuristic section detection
│
├── preprocessing/              # PHASE 2: Text Cleaning & Tokenization
│   ├── __init__.py
│   ├── cleaner.py              # Noise removal, Unicode normalization
│   ├── tokenizer.py            # SciBERT tokenizer with [MATH] special tokens
│   └── dual_repr.py            # Separate NL text and math expressions
│
├── features/                   # PHASE 3: Feature Engineering
│   ├── __init__.py
│   ├── keyword_extractor.py    # TF-IDF + YAKE + KeyBERT keyword extraction
│   ├── entity_extractor.py     # spaCy NER + custom math entity rules
│   └── math_structure.py       # Theorem / Proof / Lemma / Definition detection
│
├── models/                     # PHASE 4: Transformer Models
│   ├── __init__.py
│   ├── longdoc_handler.py      # Sliding window chunking for long documents
│   ├── summarizer.py           # Extractive (TF-IDF) + Abstractive (BART-large-CNN)
│   ├── scibert_classifier.py   # Zero-shot classification + keyword fallback
│   └── similarity.py           # Semantic similarity (sentence-transformers)
│
├── analysis/                   # PHASE 5: Advanced NLP Analysis
│   ├── __init__.py
│   ├── topic_modeling.py       # BERTopic topic discovery
│   ├── semantic_search.py      # Dense retrieval search over paper sections
│   ├── citation_analysis.py    # Citation frequency, context, purpose classification
│   └── equation_classifier.py  # Rule-based equation type classification (9 domains)
│
├── llm/                        # PHASE 6: Groq LLM Integration
│   ├── __init__.py
│   ├── groq_client.py          # Groq API client with LLaMA / Mixtral fallback
│   └── prompt_builder.py       # Structured prompt templates for 5 analysis types
│
├── output/                     # PHASE 7: Report Generation
│   ├── __init__.py
│   ├── json_report.py          # Compile all results → structured JSON
│   ├── visualizations.py       # Dark-themed matplotlib charts (3 types)
│   └── final_report.py         # Comprehensive markdown report generator
│
├── results/                    # Generated output directory
│   ├── report.json             # Machine-readable structured report
│   ├── final_report.md         # Human-readable markdown report
│   ├── keyword_chart.png       # Top keywords bar chart
│   ├── equation_distribution.png # Equation type pie chart
│   └── topic_chart.png         # Topic distribution bar chart
│
├── cache/                      # Cached embeddings for repeat runs
└── data/                       # Intermediate data storage
```

---

## Pipeline Flow — Phase by Phase

### Phase 1 — Data Ingestion

**Module:** `ingestion/pdf_parser.py`

The PDF parser uses **PyMuPDF** to extract text blocks with font metadata. It employs heuristic rules based on font size to identify:

- **Title** — Largest font block on page 1
- **Authors** — Second-largest font between title and abstract
- **Abstract** — Text following "abstract" keyword
- **Sections** — Font sizes that match heading patterns (bold, larger than body text)
- **Equations** — Lines containing mathematical symbols (`∑`, `∫`, `∏`, `=`, `≤`, etc.) or LaTeX patterns
- **References** — Entries after "References" or "Bibliography" heading

```python
# Example: Automatic structure detection
parser = PDFParser()
data = parser.parse("paper.pdf")
# → data["title"] = "The Counting Function of Semiprimes"
# → data["sections"] = [{"title": "Introduction", "content": "..."}, ...]
# → data["equations"] = ["π₂(x) ∼ x/log(x)", ...]  (287 equations found)
```

### Phase 2 — Preprocessing

**Modules:** `preprocessing/cleaner.py`, `dual_repr.py`

1. **Cleaning** — Removes page numbers, headers/footers, DOIs, citation markers `[1]`, redundant whitespace. Preserves `[MATH]...[/MATH]` tagged expressions.
2. **Dual Representation** — Splits each section into:
   - **NL text** (natural language) — for transformer models
   - **Math expressions** (inline `$...$` and display `$$...$$`) — preserved separately

```python
# Before cleaning: "Page 3 of 13 · doi:10.1234 · We prove that [1]..."
# After cleaning:  "We prove that..."
# Reduction: ~6% noise removed on average
```

### Phase 3 — Feature Engineering

**Modules:** `features/keyword_extractor.py`, `entity_extractor.py`, `math_structure.py`

Three parallel extraction pipelines:

| Extractor | Method | Output Example |
|-----------|--------|---------------|
| **Keywords** | TF-IDF + YAKE + KeyBERT fusion | "semiprime counting function", "prime number theorem" |
| **Entities** | spaCy `en_core_web_sm` + custom rules | Persons: "Landau", Methods: "combinatorial argument" |
| **Math Structures** | Regex pattern matching | Theorem 2.1, Lemma 3, Proof by induction |

The keyword extractor fuses results from three methods using weighted scoring:
- TF-IDF (statistical) → 40 candidates
- YAKE (unsupervised) → 40 candidates
- KeyBERT (semantic, transformer-based) → 40 candidates
- **Combined** → top 20 by fusion score

### Phase 4 — Transformer Models

**Modules:** `models/summarizer.py`, `scibert_classifier.py`, `similarity.py`, `longdoc_handler.py`

#### Long Document Handling
Papers exceeding 512 tokens are split using a **sliding window** strategy:
- Window size: 512 tokens
- Overlap: 64 tokens
- Each chunk processed independently, results merged

#### Summarization (Two-Stage)
1. **Extractive** — TF-IDF scoring + position bias + keyword bonus → top 5 sentences
2. **Abstractive** — BART-large-CNN transformer generates fluent summary
   - **GPU mode:** 4-beam search, max 250 tokens (high quality)
   - **CPU mode:** Greedy decoding, max 150 tokens (4× faster)

#### Section Classification
- **Primary:** Zero-shot classification via BART-large-MNLI with 10 candidate labels
- **Fallback:** Keyword-based rule matching (when model unavailable or network slow)
- Labels: `introduction`, `methodology`, `results`, `discussion`, `conclusion`, `related_work`, `theory`, `experiments`, `abstract`, `references`

#### Semantic Similarity
- Encodes all sections using `all-MiniLM-L6-v2`
- Computes pairwise cosine similarity matrix
- Identifies most/least similar section pairs

### Phase 5 — Advanced NLP Analysis

**Modules:** `analysis/topic_modeling.py`, `semantic_search.py`, `citation_analysis.py`, `equation_classifier.py`

| Analysis | Method | Output |
|----------|--------|--------|
| **Topic Modeling** | BERTopic (sentence-transformers + UMAP + HDBSCAN) | 12 topics with keywords, e.g., Topic 0: "log, the, of, for" |
| **Semantic Search** | Dense retrieval with MiniLM embeddings | 327 searchable units, query → top-k relevant passages |
| **Citation Analysis** | Regex extraction + context windowing | Citation frequency, purpose classification, density metrics |
| **Equation Classification** | Rule-based pattern matching across 9 math domains | 287 equations → {other: 280, number_theory: 5, ...} |

**Equation domains:** algebra, calculus, optimization, probability, linear_algebra, differential_equations, number_theory, statistics, geometry, other

### Phase 6 — LLM Analysis (Groq)

**Modules:** `llm/groq_client.py`, `prompt_builder.py`

Sends structured prompts to **Groq's LLaMA 3.3 70B** API for five analysis types:

| Analysis Type | Prompt Focus | Output |
|--------------|-------------|--------|
| **Summary** | Comprehensive paper overview | Multi-paragraph technical summary |
| **ELI5** | "Explain like I'm 5" | Accessible explanation with analogies |
| **Contributions** | Novel contributions | Numbered list of key contributions |
| **Applications** | Practical & future applications | Real-world use cases |
| **Limitations** | Weaknesses & gaps | Critical analysis of methodology |

**Fault Tolerance:**
- Primary: `llama-3.3-70b-versatile`
- Fallback: `mixtral-8x7b-32768` (if primary rate-limited)
- Temperature: 0.3 (focused, deterministic)

### Phase 7 — Output Generation

**Modules:** `output/json_report.py`, `visualizations.py`, `final_report.py`

#### JSON Report (`report.json`)
Complete machine-readable output (~38 KB) containing all pipeline results: metadata, keywords, entities, classifications, topics, LLM insights, and statistics.

#### Markdown Report (`final_report.md`)
Human-readable document (~21 KB) with:
- Paper metadata and abstract
- Key terms and mathematical analysis
- Topic discovery results
- Full LLM-generated analysis (summary, ELI5, contributions, applications, limitations)
- Visualization references

#### Visualizations (Dark-Themed)
Three publication-quality charts:

1. **Keyword Chart** — Horizontal bar chart of top keywords by relevance score
2. **Equation Distribution** — Pie chart of equation types across 9 math domains
3. **Topic Distribution** — Horizontal bar chart of discovered topics by document count

---

## Installation

### Prerequisites
- Python 3.11+
- pip
- (Optional) NVIDIA GPU with CUDA for acceleration

### Setup

```bash
# Clone or navigate to project
cd RESA_AI

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### First-Run Note
On the first run, HuggingFace models will be downloaded and cached (~3 GB total):
- `facebook/bart-large-cnn` (~1.6 GB) — Summarization
- `facebook/bart-large-mnli` (~1.6 GB) — Zero-shot classification
- `sentence-transformers/all-MiniLM-L6-v2` (~80 MB) — Embeddings

Subsequent runs load from `~/.cache/huggingface/` and are much faster.

---

## Usage

### Basic Usage

```bash
python main.py <path-to-pdf>
```

### Example

```bash
python main.py paperDCRE-1.pdf
```

### Output

```
======================================================================
  RESA_AI - Mathematical Research Paper NLP Pipeline v1.0.0
======================================================================

  PHASE 1: DATA INGESTION ......................... 0.3s  ✅
  PHASE 2: PREPROCESSING .......................... 0.0s  ✅
  PHASE 3: FEATURE ENGINEERING .................... 27.2s ✅
  PHASE 4: TRANSFORMER MODELS .................... varies ✅
  PHASE 5: ADVANCED NLP ANALYSIS .................. 42.1s ✅
  PHASE 6: GROQ LLM ANALYSIS ..................... 13.5s ✅
  PHASE 7: OUTPUT GENERATION ......................  1.0s ✅

  Total Time: ~87s (CPU) | ~45s (GPU)

  [JSON]   results/report.json
  [REPORT] results/final_report.md
  [CHARTS] results/keyword_chart.png
           results/equation_distribution.png
           results/topic_chart.png
```

---

## Configuration

All tunable parameters are centralized in `config.py`:

### Key Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SUMMARIZER_MODEL` | `facebook/bart-large-cnn` | Abstractive summarizer model |
| `ZERO_SHOT_MODEL` | `facebook/bart-large-mnli` | Zero-shot classifier model |
| `SENTENCE_TRANSFORMER_MODEL` | `all-MiniLM-L6-v2` | Embedding model |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Primary LLM |
| `CHUNK_SIZE` | 512 | Tokens per sliding window chunk |
| `CHUNK_OVERLAP` | 64 | Overlap between chunks |
| `TOP_K_KEYWORDS` | 20 | Number of extracted keywords |
| `TOP_K_SUMMARY_SENTENCES` | 5 | Extractive summary length |
| `CUDA_MEMORY_FRACTION` | 0.85 | Max GPU memory allocation |
| `GROQ_TEMPERATURE` | 0.3 | LLM generation temperature |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq API key (fallback hardcoded in config) |
| `HF_TOKEN` | HuggingFace token for faster downloads |

---

## Output Format

### JSON Report Structure

```json
{
  "metadata": {
    "title": "The Counting Function of Semiprimes",
    "authors": ["Dragos", "Cris", "Radek Erban"],
    "pages": 13,
    "analysis_date": "2026-03-28",
    "pipeline_version": "1.0.0"
  },
  "structure": {
    "sections": 35,
    "equations": 287,
    "references": 23,
    "characters": 30167
  },
  "keywords": {
    "tfidf": [...],
    "yake": [...],
    "keybert": [...],
    "combined_top": ["log", "semiprime counting function", ...]
  },
  "entities": {
    "persons": [...],
    "methods": [...],
    "concepts": [...]
  },
  "topics": [
    {"id": 0, "keywords": ["log", "the", "of"], "count": 149},
    ...
  ],
  "equation_analysis": {
    "total": 287,
    "distribution": {"other": 280, "number_theory": 5, ...}
  },
  "llm_insights": {
    "summary": "...",
    "eli5": "...",
    "contributions": "...",
    "applications": "...",
    "limitations": "..."
  }
}
```

---

## GPU Optimization

RESA_AI auto-detects hardware and optimizes accordingly:

| Feature | GPU Mode | CPU Mode |
|---------|----------|----------|
| **Precision** | FP16 mixed precision | FP32 |
| **Batch Size** | 16 | 4 |
| **Summarizer Beams** | 4 (beam search) | 1 (greedy) |
| **Summary Max Length** | 250 tokens | 150 tokens |
| **TF32 Acceleration** | Enabled (Ampere+) | N/A |
| **Memory Management** | 85% cap + `empty_cache()` | Standard |
| **cuDNN Benchmark** | Enabled | N/A |
| **Estimated Runtime** | ~45s | ~90s+ |

### Memory Safety
- Models are loaded/unloaded sequentially to prevent OOM
- `torch.cuda.empty_cache()` called between phases
- GPU memory capped at 85% to keep system responsive

---

## Models Used

| Model | Size | Task | Provider |
|-------|------|------|----------|
| `facebook/bart-large-cnn` | 406M params | Abstractive Summarization | HuggingFace |
| `facebook/bart-large-mnli` | 406M params | Zero-Shot Classification | HuggingFace |
| `all-MiniLM-L6-v2` | 22M params | Sentence Embeddings | Sentence-Transformers |
| `en_core_web_sm` | 12M params | NER / POS Tagging | spaCy |
| `llama-3.3-70b-versatile` | 70B params | Deep Reasoning (API) | Groq |
| `mixtral-8x7b-32768` | 46.7B params | Fallback Reasoning (API) | Groq |
| **BERTopic** | Varies | Topic Discovery | BERTopic |

---

## License

This project is for research and educational purposes.

---

<p align="center">
  Built with 🔬 by <strong>RESA_AI</strong> — Turning Math Papers into Actionable Insights
</p>
