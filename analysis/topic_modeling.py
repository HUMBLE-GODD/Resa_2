"""
Topic Modeling — Discover topics using BERTopic.
GPU-optimized with embedding reuse.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Dict, List
from config import DEVICE, CACHE_DIR


class TopicModeler:
    """
    Discover topics in the paper using BERTopic.
    Reuses cached embeddings from the similarity module when available.
    """

    def __init__(self):
        self._model = None

    def analyze(self, data: Dict) -> Dict:
        """
        Run topic modeling on paper sections and sentences.
        
        Returns:
            Updated data with 'topics' field
        """
        print("[Phase 5] Running topic modeling...")
        
        sections = data.get("nl_sections", data.get("sections", []))
        
        # Collect documents (sentences from all sections)
        documents = []
        doc_sources = []
        
        for sec in sections:
            text = sec.get("nl_text", sec.get("content", ""))
            title = sec.get("title", "")
            
            # Split into sentences for finer-grained topics
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
            documents.extend(sentences)
            doc_sources.extend([title] * len(sentences))
        
        if len(documents) < 5:
            print("  ⚠ Too few documents for topic modeling, using keyword-based fallback")
            data["topics"] = self._keyword_fallback(data)
            return data
        
        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            
            # Use cached embeddings if available
            embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=str(DEVICE))
            
            # Configure BERTopic
            self._model = BERTopic(
                embedding_model=embedding_model,
                nr_topics="auto",
                min_topic_size=3,
                verbose=False,
            )
            
            # Fit and transform
            topics, probs = self._model.fit_transform(documents)
            
            # Get topic info
            topic_info = self._model.get_topic_info()
            
            # Extract topic details
            topic_details = []
            for _, row in topic_info.iterrows():
                topic_id = row["Topic"]
                if topic_id == -1:  # Skip outlier topic
                    continue
                
                topic_words = self._model.get_topic(topic_id)
                topic_details.append({
                    "id": int(topic_id),
                    "count": int(row["Count"]),
                    "name": row.get("Name", f"Topic_{topic_id}"),
                    "keywords": [w for w, _ in topic_words[:10]],
                    "keyword_scores": {w: float(s) for w, s in topic_words[:10]},
                })
            
            # Map documents to topics
            doc_topics = []
            for i, (doc, topic, prob) in enumerate(zip(documents, topics, probs)):
                if topic != -1:
                    doc_topics.append({
                        "text": doc[:100],
                        "topic_id": int(topic),
                        "probability": float(prob) if isinstance(prob, (int, float)) else float(max(prob)),
                        "source_section": doc_sources[i],
                    })
            
            data["topics"] = {
                "model": "BERTopic",
                "num_topics": len(topic_details),
                "topic_details": topic_details,
                "document_topics": doc_topics[:50],
                "outlier_count": sum(1 for t in topics if t == -1),
            }
            
            print(f"  ✓ Discovered {len(topic_details)} topics from {len(documents)} sentences")
            for t in topic_details[:5]:
                print(f"    Topic {t['id']}: {t['keywords'][:5]}")
            
        except ImportError:
            print("  ⚠ BERTopic not available, using keyword fallback")
            data["topics"] = self._keyword_fallback(data)
        except Exception as e:
            print(f"  ⚠ Topic modeling failed: {e}")
            data["topics"] = self._keyword_fallback(data)
        
        return data

    def _keyword_fallback(self, data: Dict) -> Dict:
        """Simple keyword-based topic detection when BERTopic is unavailable."""
        keywords = data.get("keywords", {}).get("top_keywords", [])
        
        # Group keywords into pseudo-topics
        topics = []
        if keywords:
            chunk_size = max(3, len(keywords) // 3)
            for i in range(0, len(keywords), chunk_size):
                chunk = keywords[i:i + chunk_size]
                topics.append({
                    "id": i // chunk_size,
                    "keywords": chunk,
                    "count": len(chunk),
                    "name": f"Topic_{i // chunk_size}: {chunk[0]}",
                })
        
        return {
            "model": "keyword_fallback",
            "num_topics": len(topics),
            "topic_details": topics,
            "document_topics": [],
            "outlier_count": 0,
        }


if __name__ == "__main__":
    modeler = TopicModeler()
    test = {
        "nl_sections": [
            {"title": "Intro", "nl_text": "Deep learning has been widely applied to image classification. Convolutional neural networks achieve state-of-the-art results. Transfer learning enables training with limited data. Data augmentation improves model robustness."},
            {"title": "Methods", "nl_text": "We use ResNet-50 as backbone architecture. The model is trained with SGD optimizer. We apply batch normalization after each convolutional layer. Dropout is used for regularization."},
        ],
    }
    result = modeler.analyze(test)
    print(f"\nTopics: {result['topics']['num_topics']}")
