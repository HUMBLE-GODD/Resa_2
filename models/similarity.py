"""
Semantic Similarity — Embed sections using Sentence Transformers
and compute inter-section similarity.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from typing import Dict, List
from config import DEVICE, SENTENCE_TRANSFORMER_MODEL, CACHE_DIR


class SemanticSimilarity:
    """
    Compute semantic similarity between paper sections using
    sentence-transformers embeddings.
    """

    def __init__(self):
        self._model = None

    def _load_model(self):
        """Lazy-load sentence transformer."""
        if self._model is not None:
            return
        
        print(f"  Loading sentence transformer: {SENTENCE_TRANSFORMER_MODEL}")
        from sentence_transformers import SentenceTransformer
        
        self._model = SentenceTransformer(
            SENTENCE_TRANSFORMER_MODEL,
            device=str(DEVICE),
        )
        print(f"  ✓ Sentence transformer loaded on {DEVICE}")

    def compute(self, data: Dict) -> Dict:
        """
        Compute embeddings and similarity matrix for paper sections.
        
        Returns:
            Updated data with 'similarity' field containing embeddings and matrix
        """
        print("[Phase 4] Computing semantic similarity...")
        self._load_model()
        
        sections = data.get("nl_sections", data.get("sections", []))
        
        if not sections:
            data["similarity"] = {"embeddings": [], "matrix": [], "clusters": []}
            return data
        
        # Prepare texts
        texts = []
        labels = []
        for sec in sections:
            text = sec.get("nl_text", sec.get("content", ""))
            if len(text) > 20:
                texts.append(text[:2000])  # Truncate very long sections
                labels.append(sec.get("title", "Untitled"))
        
        if not texts:
            data["similarity"] = {"embeddings": [], "matrix": [], "clusters": []}
            return data
        
        # Compute embeddings (batch for GPU efficiency)
        with torch.no_grad():
            embeddings = self._model.encode(
                texts,
                batch_size=16,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        
        # Compute cosine similarity matrix
        sim_matrix = np.dot(embeddings, embeddings.T)
        
        # Find most similar pairs
        similar_pairs = []
        n = len(texts)
        for i in range(n):
            for j in range(i + 1, n):
                similar_pairs.append({
                    "section_a": labels[i],
                    "section_b": labels[j],
                    "similarity": float(sim_matrix[i][j]),
                })
        
        similar_pairs.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Cache embeddings
        cache_path = os.path.join(CACHE_DIR, "section_embeddings.npy")
        np.save(cache_path, embeddings)
        
        data["similarity"] = {
            "embeddings": embeddings.tolist(),
            "matrix": sim_matrix.tolist(),
            "labels": labels,
            "most_similar": similar_pairs[:5],
            "least_similar": similar_pairs[-3:] if len(similar_pairs) > 3 else [],
            "embedding_dim": embeddings.shape[1],
            "cache_path": cache_path,
        }
        
        print(f"  ✓ Embedded {len(texts)} sections ({embeddings.shape[1]}D)")
        print(f"  ✓ Most similar: {similar_pairs[0]['section_a'][:30]} ↔ {similar_pairs[0]['section_b'][:30]} ({similar_pairs[0]['similarity']:.3f})" if similar_pairs else "  No pairs")
        
        # Clean GPU memory
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        
        return data

    def search(self, query: str, data: Dict, top_k: int = 3) -> List[Dict]:
        """Search sections by semantic similarity to a query."""
        self._load_model()
        
        embeddings = data.get("similarity", {}).get("embeddings", [])
        labels = data.get("similarity", {}).get("labels", [])
        
        if not embeddings:
            return []
        
        embeddings = np.array(embeddings)
        
        with torch.no_grad():
            query_emb = self._model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        
        scores = np.dot(embeddings, query_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "section": labels[idx],
                "score": float(scores[idx]),
                "rank": len(results) + 1,
            })
        
        return results

    def unload(self):
        """Free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    sim = SemanticSimilarity()
    test = {
        "nl_sections": [
            {"title": "Introduction", "nl_text": "We study optimization methods for neural networks."},
            {"title": "Methods", "nl_text": "We use stochastic gradient descent with momentum."},
            {"title": "Results", "nl_text": "Our method achieves state-of-the-art performance on benchmarks."},
        ]
    }
    result = sim.compute(test)
    print(f"\nSimilarity matrix shape: {len(result['similarity']['matrix'])}x{len(result['similarity']['matrix'])}")
    
    search_results = sim.search("optimization techniques", result)
    print(f"Search results: {search_results}")
    sim.unload()
