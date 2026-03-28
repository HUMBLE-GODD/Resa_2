"""
SciBERT Classifier - Zero-shot topic classification for paper sections.
GPU-optimized with keyword-based fallback for offline/slow network scenarios.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import torch
from typing import Dict, List
from config import DEVICE, SECTION_LABELS, ZERO_SHOT_MODEL


# Keyword rules for fallback classification (no model download needed)
KEYWORD_RULES = {
    "introduction": ["introduce", "introduction", "background", "motivation", "overview", "context"],
    "methodology": ["method", "approach", "framework", "algorithm", "procedure", "model", "technique", "architecture"],
    "results": ["result", "experiment", "performance", "evaluation", "accuracy", "benchmark", "table", "figure"],
    "discussion": ["discuss", "analysis", "implication", "interpret", "observe", "finding"],
    "conclusion": ["conclusion", "conclud", "summary", "future work", "contribute"],
    "related work": ["related", "prior work", "previous", "literature", "survey", "existing"],
    "abstract": ["abstract"],
    "theoretical framework": ["theorem", "proof", "lemma", "proposition", "corollary", "definition"],
    "data": ["dataset", "data collection", "corpus", "sample", "annotation"],
    "mathematical formulation": ["equation", "formula", "derivation", "notation", "mathematical"],
}


class SciBERTClassifier:
    """
    Classify paper sections by topic using zero-shot classification.
    Falls back to keyword-based classification if model is unavailable.
    """

    def __init__(self):
        self._classifier = None
        self._labels = SECTION_LABELS
        self._use_fallback = False

    def _load_model(self):
        """Lazy-load the classifier with timeout protection."""
        if self._classifier is not None or self._use_fallback:
            return
        
        print(f"  Loading zero-shot classifier: {ZERO_SHOT_MODEL}")
        
        try:
            from transformers import pipeline
            
            device_id = 0 if DEVICE.type == "cuda" else -1
            dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32
            
            self._classifier = pipeline(
                "zero-shot-classification",
                model=ZERO_SHOT_MODEL,
                device=device_id,
                dtype=dtype,
            )
            print(f"  [OK] Classifier loaded on {DEVICE}")
        except Exception as e:
            print(f"  [WARN] Classifier load failed: {e}")
            print(f"  [INFO] Using keyword-based fallback classifier")
            self._use_fallback = True

    def classify(self, data: Dict) -> Dict:
        """
        Classify each section of the paper by topic.
        Uses zero-shot model if available, keyword fallback otherwise.
        """
        print("[Phase 4] Classifying sections...")
        self._load_model()
        
        sections = data.get("nl_sections", data.get("sections", []))
        classifications = []
        
        for i, section in enumerate(sections):
            text = section.get("nl_text", section.get("content", ""))
            title = section.get("title", "")
            
            if len(text) < 20:
                classifications.append({
                    "section_title": title,
                    "predicted_label": "other",
                    "confidence": 0.0,
                    "all_scores": {},
                })
                continue
            
            if self._use_fallback:
                result = self._keyword_classify(text, title)
            else:
                result = self._model_classify(text, title)
            
            classifications.append(result)
            print(f"  Section '{title[:40]}' -> {result['predicted_label']} ({result['confidence']:.0%})")
        
        # Clean GPU memory
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        
        data["classifications"] = classifications
        
        # Summary
        label_counts = {}
        for c in classifications:
            label = c["predicted_label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        method = "keyword-based" if self._use_fallback else "zero-shot"
        print(f"  [OK] Classified {len(classifications)} sections ({method})")
        print(f"  [OK] Label distribution: {label_counts}")
        
        return data

    def _model_classify(self, text: str, title: str) -> Dict:
        """Classify using the zero-shot model."""
        input_text = text[:1024]
        
        try:
            result = self._classifier(
                input_text,
                candidate_labels=self._labels,
                multi_label=False,
            )
            
            scores = dict(zip(result["labels"], result["scores"]))
            return {
                "section_title": title,
                "predicted_label": result["labels"][0],
                "confidence": float(result["scores"][0]),
                "all_scores": {k: round(v, 4) for k, v in scores.items()},
            }
        except Exception as e:
            print(f"  [WARN] Model classification failed for '{title}': {e}")
            return self._keyword_classify(text, title)

    def _keyword_classify(self, text: str, title: str) -> Dict:
        """Fallback: classify using keyword matching on title and content."""
        combined = (title + " " + text[:500]).lower()
        
        scores = {}
        for label, keywords in KEYWORD_RULES.items():
            score = sum(1 for kw in keywords if kw in combined)
            if score > 0:
                scores[label] = score
        
        if not scores:
            return {
                "section_title": title,
                "predicted_label": "other",
                "confidence": 0.3,
                "all_scores": {},
            }
        
        total = sum(scores.values())
        best_label = max(scores, key=scores.get)
        confidence = scores[best_label] / total if total > 0 else 0.0
        
        return {
            "section_title": title,
            "predicted_label": best_label,
            "confidence": round(min(confidence, 0.95), 3),
            "all_scores": {k: round(v / total, 4) for k, v in scores.items()},
        }

    def unload(self):
        """Free GPU memory by unloading the model."""
        if self._classifier is not None:
            del self._classifier
            self._classifier = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("  [OK] Classifier unloaded from memory")


if __name__ == "__main__":
    classifier = SciBERTClassifier()
    test = {
        "nl_sections": [
            {"title": "Introduction", "nl_text": "We study neural networks for image classification using deep learning."},
            {"title": "Methods", "nl_text": "We train a ResNet-50 model with SGD optimizer and cross-entropy loss."},
            {"title": "Results", "nl_text": "Our model achieves 95% accuracy on the test set, outperforming baselines."},
        ]
    }
    result = classifier.classify(test)
    classifier.unload()
