"""
Keyword Extractor — Extract key terms using TF-IDF, YAKE, and KeyBERT.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from config import TOP_K_KEYWORDS


class KeywordExtractor:
    """Extract keywords using multiple methods for robust coverage."""

    def __init__(self):
        self._yake_extractor = None
        self._keybert_model = None

    def extract_all(self, data: Dict) -> Dict:
        """
        Run all keyword extraction methods and combine results.
        
        Args:
            data: Pipeline data dict with 'nl_text' and 'nl_sections'
            
        Returns:
            Updated data dict with 'keywords' field
        """
        print("[Phase 3] Extracting keywords...")
        
        text = data.get("nl_text", data.get("cleaned_text", ""))
        sections = data.get("nl_sections", data.get("sections", []))
        
        # Method 1: TF-IDF
        tfidf_keywords = self._tfidf_keywords(text, sections)
        print(f"  ✓ TF-IDF: {len(tfidf_keywords)} keywords")
        
        # Method 2: YAKE
        yake_keywords = self._yake_keywords(text)
        print(f"  ✓ YAKE: {len(yake_keywords)} keywords")
        
        # Method 3: KeyBERT (semantic)
        keybert_keywords = self._keybert_keywords(text)
        print(f"  ✓ KeyBERT: {len(keybert_keywords)} keywords")
        
        # Combine and rank
        combined = self._combine_keywords(tfidf_keywords, yake_keywords, keybert_keywords)
        print(f"  ✓ Combined top keywords: {[k[0] for k in combined[:10]]}")
        
        data["keywords"] = {
            "tfidf": tfidf_keywords,
            "yake": yake_keywords,
            "keybert": keybert_keywords,
            "combined": combined,
            "top_keywords": [k[0] for k in combined[:TOP_K_KEYWORDS]],
        }
        
        return data

    def _tfidf_keywords(self, text: str, sections: List[Dict]) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF across sections."""
        # Create document collection from sections
        docs = [text]
        for sec in sections:
            content = sec.get("nl_text", sec.get("content", ""))
            if content and len(content) > 20:
                docs.append(content)
        
        if len(docs) < 2:
            docs.append(text[:len(text)//2])
            docs.append(text[len(text)//2:])
        
        try:
            vectorizer = TfidfVectorizer(
                max_features=200,
                stop_words='english',
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.95,
            )
            
            tfidf_matrix = vectorizer.fit_transform(docs)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get scores from the full document (first doc)
            scores = tfidf_matrix[0].toarray()[0]
            
            # Sort by score
            keyword_scores = sorted(
                zip(feature_names, scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            return [(kw, float(score)) for kw, score in keyword_scores[:TOP_K_KEYWORDS * 2]]
        except Exception as e:
            print(f"  ⚠ TF-IDF failed: {e}")
            return []

    def _yake_keywords(self, text: str) -> List[Tuple[str, float]]:
        """Extract keywords using YAKE (Yet Another Keyword Extractor)."""
        try:
            import yake
            if self._yake_extractor is None:
                self._yake_extractor = yake.KeywordExtractor(
                    lan="en",
                    n=3,              # Max ngram size
                    dedupLim=0.7,     # Deduplication threshold
                    top=TOP_K_KEYWORDS * 2,
                    features=None,
                )
            
            keywords = self._yake_extractor.extract_keywords(text)
            # YAKE: lower score = more relevant, so invert
            if keywords:
                max_score = max(k[1] for k in keywords) + 0.001
                return [(kw, 1.0 - score/max_score) for kw, score in keywords]
            return []
        except ImportError:
            print("  ⚠ YAKE not installed, skipping")
            return []
        except Exception as e:
            print(f"  ⚠ YAKE failed: {e}")
            return []

    def _keybert_keywords(self, text: str) -> List[Tuple[str, float]]:
        """Extract keywords using KeyBERT (transformer-based semantic extraction)."""
        try:
            from keybert import KeyBERT
            if self._keybert_model is None:
                self._keybert_model = KeyBERT(model='all-MiniLM-L6-v2')
            
            keywords = self._keybert_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_n=TOP_K_KEYWORDS * 2,
                use_mmr=True,
                diversity=0.5,
            )
            
            return [(kw, float(score)) for kw, score in keywords]
        except ImportError:
            print("  ⚠ KeyBERT not installed, skipping")
            return []
        except Exception as e:
            print(f"  ⚠ KeyBERT failed: {e}")
            return []

    def _combine_keywords(self, *keyword_lists) -> List[Tuple[str, float]]:
        """Combine keywords from multiple methods using rank fusion."""
        keyword_scores = {}
        
        for keywords in keyword_lists:
            if not keywords:
                continue
            for rank, (kw, score) in enumerate(keywords):
                kw_lower = kw.lower().strip()
                if len(kw_lower) < 2:
                    continue
                if kw_lower not in keyword_scores:
                    keyword_scores[kw_lower] = []
                # Reciprocal rank fusion
                keyword_scores[kw_lower].append(1.0 / (rank + 1) + score)
        
        # Aggregate scores
        combined = [
            (kw, sum(scores) / len(scores))
            for kw, scores in keyword_scores.items()
        ]
        
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:TOP_K_KEYWORDS * 2]


if __name__ == "__main__":
    extractor = KeywordExtractor()
    test_data = {
        "nl_text": "We study the convergence of neural network optimization using stochastic gradient descent. The loss function exhibits non-convex behavior in high-dimensional parameter spaces.",
        "nl_sections": [],
    }
    result = extractor.extract_all(test_data)
    print(f"\nTop keywords: {result['keywords']['top_keywords']}")
