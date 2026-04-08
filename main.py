"""
RESA_AI - Mathematical Research Paper NLP Pipeline
Main Orchestrator - Runs all 8 phases end-to-end.

Usage:
    python main.py paperDCRE-1.pdf
    python main.py paper.tex --type latex
    python main.py paper.txt --type text
"""
import sys
import os
import io
import time
import json
import argparse
import traceback

# Fix Windows console encoding and keep stdout/stderr unbuffered for live progress streaming
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', write_through=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', write_through=True)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEVICE, DEVICE_NAME, RESULTS_DIR, clear_device_cache


def print_banner():
    """Print startup banner."""
    print("\n" + "=" * 70)
    print("  RESA_AI - Mathematical Research Paper NLP Pipeline v1.0.0")
    print("  Production-Grade Analysis System")
    print("=" * 70)
    print(f"  Device: {DEVICE_NAME} [{DEVICE}]")
    print(f"  Output: {RESULTS_DIR}")
    print("=" * 70 + "\n")


def run_phase(phase_name: str, func, data: dict, phase_num: int) -> dict:
    """Run a pipeline phase with timing and error handling."""
    print(f"\n{'-' * 60}")
    print(f"  PHASE {phase_num}: {phase_name}")
    print(f"{'-' * 60}")
    
    start = time.time()
    try:
        data = func(data)
        elapsed = time.time() - start
        print(f"\n  [OK] Phase {phase_num} completed in {elapsed:.1f}s")
        return data
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  [FAIL] Phase {phase_num} FAILED after {elapsed:.1f}s: {e}")
        traceback.print_exc()
        return data


def phase1_ingestion(input_path: str, input_type: str) -> dict:
    """Phase 1: Data Ingestion."""
    if input_type == "pdf":
        from ingestion.pdf_parser import PDFParser
        parser = PDFParser()
        return parser.parse(input_path)
    elif input_type == "latex":
        from ingestion.latex_parser import LaTeXParser
        parser = LaTeXParser()
        return parser.parse(input_path)
    else:
        from ingestion.text_parser import TextParser
        parser = TextParser()
        return parser.parse(input_path)


def phase2_preprocessing(data: dict) -> dict:
    """Phase 2: Preprocessing."""
    from preprocessing.cleaner import TextCleaner
    from preprocessing.dual_repr import DualRepresentation
    
    cleaner = TextCleaner()
    data = cleaner.clean(data)
    
    dual = DualRepresentation()
    data = dual.create(data)
    
    return data


def phase3_features(data: dict) -> dict:
    """Phase 3: Feature Engineering."""
    from features.keyword_extractor import KeywordExtractor
    from features.entity_extractor import EntityExtractor
    from features.math_structure import MathStructureDetector
    
    kw_extractor = KeywordExtractor()
    data = kw_extractor.extract_all(data)
    
    entity_extractor = EntityExtractor()
    data = entity_extractor.extract(data)
    
    math_detector = MathStructureDetector()
    data = math_detector.detect(data)
    
    return data


def phase4_models(data: dict) -> dict:
    """Phase 4: Transformer Models. Each sub-step runs independently."""
    # 4a: Long document chunking (lightweight, no model download)
    try:
        from models.longdoc_handler import LongDocHandler
        chunker = LongDocHandler()
        data = chunker.chunk_sections(data)
    except Exception as e:
        print(f"  [WARN] Chunking failed: {e}")
    
    # 4b: Summarization (extractive + abstractive)
    try:
        from models.summarizer import ResearchSummarizer
        summarizer = ResearchSummarizer()
        data = summarizer.summarize(data)
        summarizer.unload()
    except Exception as e:
        print(f"  [WARN] Summarization failed: {e}")
    
    # 4c: Section classification (has keyword fallback)
    try:
        from models.scibert_classifier import SciBERTClassifier
        classifier = SciBERTClassifier()
        data = classifier.classify(data)
        classifier.unload()
    except Exception as e:
        print(f"  [WARN] Classification failed: {e}")
    
    # 4d: Semantic similarity
    try:
        from models.similarity import SemanticSimilarity
        sim = SemanticSimilarity()
        data = sim.compute(data)
        sim.unload()
    except Exception as e:
        print(f"  [WARN] Similarity failed: {e}")

    clear_device_cache(DEVICE)
    return data


def phase5_analysis(data: dict) -> dict:
    """Phase 5: Advanced NLP Analysis."""
    # 5a: Topic modeling
    from analysis.topic_modeling import TopicModeler
    modeler = TopicModeler()
    data = modeler.analyze(data)
    
    # 5b: Semantic search
    from analysis.semantic_search import SemanticSearch
    search = SemanticSearch()
    data = search.build_index(data)
    search.unload()
    
    # 5c: Citation analysis
    from analysis.citation_analysis import CitationAnalyzer
    analyzer = CitationAnalyzer()
    data = analyzer.analyze(data)
    
    # 5d: Equation classification
    from analysis.equation_classifier import EquationClassifier
    eq_classifier = EquationClassifier()
    data = eq_classifier.classify(data)

    clear_device_cache(DEVICE)
    return data


def phase6_groq(data: dict) -> dict:
    """Phase 6: Groq LLM Analysis."""
    from llm.groq_client import GroqClient
    client = GroqClient()
    data = client.analyze_paper(data)
    return data


def phase7_output(data: dict) -> dict:
    """Phase 7: Output Generation."""
    from output.json_report import JSONReportGenerator
    from output.visualizations import Visualizer
    from output.final_report import FinalReportGenerator
    
    json_gen = JSONReportGenerator()
    data = json_gen.generate(data)
    
    viz = Visualizer()
    data = viz.generate_all(data)
    
    report_gen = FinalReportGenerator()
    report_gen.generate(data)
    
    return data


def main():
    parser = argparse.ArgumentParser(description="RESA_AI Math Paper NLP Pipeline")
    parser.add_argument("input", help="Path to input file (PDF, .tex, or .txt)")
    parser.add_argument("--type", choices=["pdf", "latex", "text"], default=None,
                       help="Input type (auto-detected from extension if not specified)")
    parser.add_argument("--skip-groq", action="store_true", help="Skip Groq LLM analysis")
    parser.add_argument("--skip-models", action="store_true", help="Skip transformer models (faster)")
    
    args = parser.parse_args()
    
    # Auto-detect input type
    if args.type is None:
        ext = os.path.splitext(args.input)[1].lower()
        type_map = {".pdf": "pdf", ".tex": "latex", ".txt": "text"}
        args.type = type_map.get(ext, "pdf")
    
    print_banner()
    
    overall_start = time.time()
    
    # PHASE 1: DATA INGESTION
    data = run_phase(
        "DATA INGESTION",
        lambda d: phase1_ingestion(args.input, args.type),
        {}, 1
    )
    
    # PHASE 2: PREPROCESSING
    data = run_phase("PREPROCESSING", phase2_preprocessing, data, 2)
    
    # PHASE 3: FEATURE ENGINEERING
    data = run_phase("FEATURE ENGINEERING", phase3_features, data, 3)
    
    # PHASE 4: TRANSFORMER MODELS
    if not args.skip_models:
        data = run_phase("TRANSFORMER MODELS", phase4_models, data, 4)
    else:
        print("\n  >> Phase 4 skipped (--skip-models)")
    
    # PHASE 5: ADVANCED ANALYSIS
    if not args.skip_models:
        data = run_phase("ADVANCED NLP ANALYSIS", phase5_analysis, data, 5)
    else:
        print("\n  >> Phase 5 skipped (--skip-models)")
    
    # PHASE 6: GROQ LLM ANALYSIS
    if not args.skip_groq:
        data = run_phase("GROQ LLM ANALYSIS", phase6_groq, data, 6)
    else:
        print("\n  >> Phase 6 skipped (--skip-groq)")
    
    # PHASE 7: OUTPUT GENERATION
    data = run_phase("OUTPUT GENERATION", phase7_output, data, 7)
    
    # FINAL SUMMARY
    total_time = time.time() - overall_start
    
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE - Total time: {total_time:.1f}s")
    print(f"{'=' * 70}")
    print(f"  [JSON]   {data.get('json_report_path', 'N/A')}")
    print(f"  [REPORT] {data.get('final_report_path', 'N/A')}")
    
    viz = data.get("visualizations", {})
    if viz:
        print(f"  [CHARTS]")
        for name, path in viz.items():
            print(f"     - {name}: {path}")
    
    print(f"\n  Results directory: {RESULTS_DIR}")
    print(f"{'=' * 70}\n")
    
    return data


if __name__ == "__main__":
    main()
