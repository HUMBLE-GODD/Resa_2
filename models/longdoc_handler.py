"""
Long Document Handler — Chunk long documents for transformer processing.
Handles documents exceeding max token limits with sliding window.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Tuple
from config import CHUNK_SIZE, CHUNK_OVERLAP


class LongDocHandler:
    """
    Handle documents that exceed transformer token limits.
    Uses sliding window chunking with configurable overlap.
    """

    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[Dict]:
        """
        Split text into overlapping chunks of approximately chunk_size tokens.
        Uses word-level splitting as proxy for token count.
        
        Returns:
            List of dicts with 'text', 'start_idx', 'end_idx', 'chunk_id'
        """
        words = text.split()
        
        if len(words) <= self.chunk_size:
            return [{
                "text": text,
                "start_idx": 0,
                "end_idx": len(words),
                "chunk_id": 0,
                "is_complete": True,
            }]
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            
            chunks.append({
                "text": chunk_text,
                "start_idx": start,
                "end_idx": end,
                "chunk_id": chunk_id,
                "is_complete": end >= len(words),
            })
            
            if end >= len(words):
                break
            
            start = end - self.overlap
            chunk_id += 1
        
        return chunks

    def chunk_sections(self, data: Dict) -> Dict:
        """
        Chunk all sections for transformer processing.
        
        Args:
            data: Pipeline data with sections
            
        Returns:
            Updated data with 'chunks' field
        """
        print(f"[Phase 4] Chunking document (chunk_size={self.chunk_size}, overlap={self.overlap})...")
        
        sections = data.get("nl_sections", data.get("sections", []))
        all_chunks = []
        
        for section in sections:
            text = section.get("nl_text", section.get("content", ""))
            title = section.get("title", "")
            
            section_chunks = self.chunk_text(text)
            for chunk in section_chunks:
                chunk["section_title"] = title
                all_chunks.append(chunk)
        
        # Also chunk the full text
        full_text = data.get("nl_text", data.get("cleaned_text", ""))
        full_chunks = self.chunk_text(full_text)
        
        data["chunks"] = {
            "section_chunks": all_chunks,
            "full_text_chunks": full_chunks,
            "total_section_chunks": len(all_chunks),
            "total_full_chunks": len(full_chunks),
        }
        
        print(f"  ✓ Section chunks: {len(all_chunks)}")
        print(f"  ✓ Full text chunks: {len(full_chunks)}")
        
        return data

    def aggregate_predictions(self, predictions: List[Dict], strategy: str = "mean") -> Dict:
        """
        Aggregate predictions from multiple chunks.
        
        Args:
            predictions: List of prediction dicts from each chunk
            strategy: 'mean', 'max', or 'majority_vote'
        
        Returns:
            Aggregated prediction
        """
        if not predictions:
            return {"label": "unknown", "confidence": 0.0}
        
        if strategy == "mean":
            # Average confidence scores across chunks
            all_scores = {}
            for pred in predictions:
                for label, score in pred.get("all_scores", {}).items():
                    if label not in all_scores:
                        all_scores[label] = []
                    all_scores[label].append(score)
            
            avg_scores = {k: sum(v) / len(v) for k, v in all_scores.items()}
            best_label = max(avg_scores, key=avg_scores.get) if avg_scores else "unknown"
            
            return {
                "label": best_label,
                "confidence": avg_scores.get(best_label, 0.0),
                "all_scores": avg_scores,
                "num_chunks": len(predictions),
            }
        
        elif strategy == "majority_vote":
            votes = {}
            for pred in predictions:
                label = pred.get("predicted_label", "unknown")
                votes[label] = votes.get(label, 0) + 1
            
            best_label = max(votes, key=votes.get)
            return {
                "label": best_label,
                "confidence": votes[best_label] / len(predictions),
                "votes": votes,
                "num_chunks": len(predictions),
            }
        
        return predictions[0]  # fallback


if __name__ == "__main__":
    handler = LongDocHandler(chunk_size=100, overlap=20)
    text = " ".join([f"Word{i}" for i in range(500)])
    chunks = handler.chunk_text(text)
    print(f"500 words → {len(chunks)} chunks")
    for c in chunks:
        print(f"  Chunk {c['chunk_id']}: words {c['start_idx']}-{c['end_idx']}")
