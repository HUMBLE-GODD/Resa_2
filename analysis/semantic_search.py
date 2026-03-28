"""
Semantic Search — Query the paper content using natural language.
Uses section embeddings for efficient retrieval.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Dict, List, Optional
from config import DEVICE, SENTENCE_TRANSFORMER_MODEL, CACHE_DIR


class SemanticSearch:
    """
    Search paper sections and sentences by semantic similarity.
    Uses sentence-transformer embeddings for dense retrieval.
    """

    def __init__(self):
        self._model = None
        self._index = None

    def _load_model(self):
        """Lazy-load the embedding model."""
        if self._model is not None:
            return
        
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(
            SENTENCE_TRANSFORMER_MODEL,
            device=str(DEVICE),
        )

    def build_index(self, data: Dict) -> Dict:
        """
        Build a search index from paper content.
        Creates sentence-level embeddings for fine-grained search.
        """
        print("[Phase 5] Building semantic search index...")
        self._load_model()
        
        sections = data.get("nl_sections", data.get("sections", []))
        
        # Create searchable units (sentences with context)
        search_units = []
        for sec in sections:
            text = sec.get("nl_text", sec.get("content", ""))
            title = sec.get("title", "")
            
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 15]
            for sent in sentences:
                search_units.append({
                    "text": sent,
                    "section": title,
                    "type": "sentence",
                })
        
        # Add abstract as a searchable unit
        abstract = data.get("abstract", "")
        if abstract:
            search_units.append({
                "text": abstract,
                "section": "Abstract",
                "type": "abstract",
            })
        
        if not search_units:
            data["search_index"] = {"units": [], "embeddings": None}
            return data
        
        # Compute embeddings
        texts = [u["text"] for u in search_units]
        
        import torch
        with torch.no_grad():
            embeddings = self._model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        
        # Cache the index
        cache_path = os.path.join(CACHE_DIR, "search_index.npy")
        np.save(cache_path, embeddings)
        
        self._index = {
            "units": search_units,
            "embeddings": embeddings,
        }
        
        data["search_index"] = {
            "num_units": len(search_units),
            "embedding_dim": embeddings.shape[1],
            "cache_path": cache_path,
        }
        
        print(f"  ✓ Indexed {len(search_units)} searchable units")
        
        # Demo searches
        demo_queries = [
            "main contribution of this paper",
            "mathematical proof technique",
            "experimental results",
        ]
        
        print(f"  ✓ Demo searches:")
        for query in demo_queries:
            results = self.search(query, top_k=1)
            if results:
                print(f"    Q: '{query}' → [{results[0]['section']}] {results[0]['text'][:60]}... ({results[0]['score']:.3f})")
        
        return data

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search the indexed content with a natural language query.
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            
        Returns:
            List of matching results with scores
        """
        if self._index is None or not self._index["units"]:
            return []
        
        import torch
        with torch.no_grad():
            query_emb = self._model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        
        # Cosine similarity (embeddings are already normalized)
        scores = np.dot(self._index["embeddings"], query_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            unit = self._index["units"][idx]
            results.append({
                "text": unit["text"],
                "section": unit["section"],
                "type": unit["type"],
                "score": float(scores[idx]),
                "rank": len(results) + 1,
            })
        
        return results

    def unload(self):
        """Free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        self._index = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    search = SemanticSearch()
    test = {
        "nl_sections": [
            {"title": "Intro", "nl_text": "We study convergence rates of gradient descent."},
            {"title": "Theory", "nl_text": "The main theorem shows linear convergence under strong convexity."},
        ],
        "abstract": "This paper proves convergence bounds for optimization algorithms.",
    }
    result = search.build_index(test)
    results = search.search("convergence proof")
    print(f"\nSearch results: {results}")
    search.unload()
