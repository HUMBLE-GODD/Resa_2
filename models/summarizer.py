"""
Summarizer — Extractive + Abstractive summarization for research papers.
GPU-optimized with mixed precision.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import re
import numpy as np
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from config import DEVICE, SUMMARIZER_MODEL, TOP_K_SUMMARY_SENTENCES


class ResearchSummarizer:
    """
    Two-stage summarization:
    1. Extractive: TF-IDF + position scoring to select key sentences
    2. Abstractive: BART/T5 transformer model to generate fluent summary
    """

    def __init__(self):
        self._model = None
        self._tokenizer_abs = None

    def _load_model(self):
        """Lazy-load abstractive summarization model directly (avoid pipeline task naming issues)."""
        if self._model is not None:
            return
        
        print(f"  Loading summarizer: {SUMMARIZER_MODEL}")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        
        try:
            self._tokenizer_abs = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
            dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                SUMMARIZER_MODEL, dtype=dtype
            ).to(DEVICE)
            self._model.eval()
            print(f"  [OK] Summarizer loaded on {DEVICE}")
        except Exception as e:
            print(f"  [WARN] Summarizer load failed: {e}")
            self._model = None


    def summarize(self, data: Dict) -> Dict:
        """
        Generate both extractive and abstractive summaries.
        
        Args:
            data: Pipeline data with text content
            
        Returns:
            Updated data with 'summary' field
        """
        print("[Phase 4] Generating summaries...")
        
        text = data.get("nl_text", data.get("cleaned_text", ""))
        abstract = data.get("abstract", "")
        sections = data.get("nl_sections", data.get("sections", []))
        
        # Stage 1: Extractive summary
        extractive = self._extractive_summary(text, sections)
        print(f"  ✓ Extractive summary: {len(extractive)} sentences")
        
        # Stage 2: Abstractive summary
        abstractive = self._abstractive_summary(text, abstract)
        print(f"  ✓ Abstractive summary: {len(abstractive)} chars")
        
        # Section-level summaries
        section_summaries = self._section_summaries(sections)
        print(f"  ✓ Section summaries: {len(section_summaries)}")
        
        data["summary"] = {
            "extractive": extractive,
            "abstractive": abstractive,
            "section_summaries": section_summaries,
            "combined": f"{abstractive}\n\nKey Points:\n" + "\n".join(f"• {s}" for s in extractive),
        }
        
        return data

    def _extractive_summary(self, text: str, sections: List[Dict]) -> List[str]:
        """
        Extract top sentences using TF-IDF scoring + position bias.
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        if not sentences:
            return ["No text available for summarization."]
        
        # Score each sentence
        scored = []
        
        # TF-IDF scores
        try:
            vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
            tfidf_matrix = vectorizer.fit_transform(sentences)
            tfidf_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            tfidf_scores = tfidf_scores / (tfidf_scores.max() + 1e-10)
        except:
            tfidf_scores = np.ones(len(sentences))
        
        for i, sent in enumerate(sentences):
            score = 0.0
            
            # TF-IDF score
            score += tfidf_scores[i] * 0.4
            
            # Position score (favor early and conclusion sentences)
            position_ratio = i / max(len(sentences), 1)
            if position_ratio < 0.15:  # Introduction
                score += 0.3
            elif position_ratio > 0.85:  # Conclusion
                score += 0.25
            elif 0.4 < position_ratio < 0.7:  # Results area
                score += 0.15
            
            # Length score (prefer medium-length sentences)
            word_count = len(sent.split())
            if 15 <= word_count <= 40:
                score += 0.2
            elif 10 <= word_count <= 50:
                score += 0.1
            
            # Keyword bonus
            keywords = ['we show', 'we prove', 'result', 'main', 'novel', 'propose',
                        'demonstrate', 'contribution', 'key finding', 'conclude',
                        'significant', 'improvement', 'outperform']
            for kw in keywords:
                if kw in sent.lower():
                    score += 0.15
                    break
            
            scored.append((score, sent))
        
        # Sort by score, take top K
        scored.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s for _, s in scored[:TOP_K_SUMMARY_SENTENCES]]
        
        return top_sentences

    def _abstractive_summary(self, text: str, abstract: str) -> str:
        """Generate abstractive summary using transformer model."""
        self._load_model()
        
        if self._model is None:
            return "Abstractive summarization model unavailable."
        
        # Use abstract + intro as input for abstractive summary
        input_text = abstract if abstract and len(abstract) > 100 else text
        
        # Truncate to fit model context
        input_text = input_text[:2000]
        
        if len(input_text) < 50:
            return "Insufficient text for abstractive summarization."
        
        try:
            inputs = self._tokenizer_abs(
                input_text, 
                max_length=512, 
                truncation=True, 
                return_tensors="pt"
            ).to(DEVICE)
            
            # CPU: use greedy decoding for speed (~1 min vs 30+ min with beam search)
            # GPU: use full beam search for quality
            is_gpu = DEVICE.type == "cuda"
            
            with torch.no_grad():
                summary_ids = self._model.generate(
                    inputs["input_ids"],
                    max_length=150 if not is_gpu else 250,
                    min_length=30 if not is_gpu else 60,
                    num_beams=4 if is_gpu else 1,
                    do_sample=False,
                    length_penalty=1.5 if is_gpu else 1.0,
                    early_stopping=True,
                )
            
            summary = self._tokenizer_abs.decode(summary_ids[0], skip_special_tokens=True)
            
            # Clean GPU memory
            if is_gpu:
                torch.cuda.empty_cache()
            
            return summary
            
        except Exception as e:
            print(f"  [WARN] Abstractive summarization failed: {e}")
            return f"Abstractive summary unavailable: {str(e)}"

    def _section_summaries(self, sections: List[Dict]) -> List[Dict]:
        """Generate brief summary per section."""
        summaries = []
        
        for section in sections:
            text = section.get("nl_text", section.get("content", ""))
            title = section.get("title", "Untitled")
            
            if len(text) < 30:
                continue
            
            # Simple extractive: first 2 sentences
            sents = self._split_sentences(text)
            brief = " ".join(sents[:2]) if sents else text[:200]
            
            summaries.append({
                "section": title,
                "summary": brief,
                "length": len(text.split()),
            })
        
        return summaries

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        # Filter very short sentences
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def unload(self):
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer_abs is not None:
            del self._tokenizer_abs
            self._tokenizer_abs = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  [OK] Summarizer unloaded from memory")


if __name__ == "__main__":
    summarizer = ResearchSummarizer()
    test = {
        "nl_text": "We propose a novel approach to solving differential equations using neural networks. Our method combines traditional numerical methods with deep learning. The results show significant improvements over existing baselines. We demonstrate 40% faster convergence on benchmark problems.",
        "abstract": "This paper introduces a neural differential equation solver.",
        "nl_sections": [],
    }
    result = summarizer.summarize(test)
    print(f"\nExtractive: {result['summary']['extractive']}")
    print(f"\nAbstractive: {result['summary']['abstractive']}")
    summarizer.unload()
