"""
Groq API Client — LLM-powered analysis using Groq's fast inference.
Supports LLaMA and Mixtral models with fallback handling.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import Dict, Optional
from config import GROQ_API_KEY, GROQ_MODEL, GROQ_FALLBACK_MODEL, GROQ_MAX_TOKENS, GROQ_TEMPERATURE


class GroqClient:
    """
    Groq API wrapper for fast LLM inference.
    Supports model fallback and structured output generation.
    """

    def __init__(self, api_key: str = GROQ_API_KEY):
        self.api_key = api_key
        self._client = None
        self.model = GROQ_MODEL
        self.fallback_model = GROQ_FALLBACK_MODEL

    def _init_client(self):
        """Initialize Groq client."""
        if self._client is not None:
            return
        
        if not self.api_key:
            print("  ⚠ No GROQ_API_KEY — LLM analysis will be skipped")
            return
        
        try:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
            print(f"  ✓ Groq client initialized (model: {self.model})")
        except ImportError:
            print("  ⚠ groq package not installed")
        except Exception as e:
            print(f"  ⚠ Groq init failed: {e}")

    def generate(self, prompt: str, system_prompt: str = "", model: str = None, max_tokens: int = None) -> str:
        """
        Generate text using Groq API.
        
        Args:
            prompt: User prompt
            system_prompt: System instruction
            model: Model to use (defaults to config)
            max_tokens: Max output tokens
            
        Returns:
            Generated text or error message
        """
        self._init_client()
        
        if self._client is None:
            return "[Groq API unavailable — please set GROQ_API_KEY]"
        
        model = model or self.model
        max_tokens = max_tokens or GROQ_MAX_TOKENS
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=GROQ_TEMPERATURE,
            )
            return response.choices[0].message.content
            
        except Exception as e:
            error_msg = str(e)
            print(f"  ⚠ Groq API error with {model}: {error_msg}")
            
            # Try fallback model
            if model != self.fallback_model:
                print(f"  → Trying fallback model: {self.fallback_model}")
                try:
                    response = self._client.chat.completions.create(
                        model=self.fallback_model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=GROQ_TEMPERATURE,
                    )
                    return response.choices[0].message.content
                except Exception as e2:
                    return f"[Groq API failed on both models: {error_msg} / {str(e2)}]"
            
            return f"[Groq API error: {error_msg}]"

    def analyze_paper(self, data: Dict) -> Dict:
        """
        Generate comprehensive LLM analysis of the paper.
        
        Returns:
            Updated data with 'llm_analysis' field
        """
        print("[Phase 6] Running Groq LLM analysis...")
        self._init_client()
        
        # Build context from extracted data
        context = self._build_context(data)
        
        # Generate each analysis component
        analyses = {}
        
        # 1. Human-readable summary
        print("  → Generating summary...")
        analyses["summary"] = self.generate(
            prompt=f"Based on this research paper analysis, provide a clear, comprehensive summary (3-4 paragraphs):\n\n{context}",
            system_prompt="You are a research paper analysis expert. Provide clear, accurate summaries of academic papers, especially those with mathematical content.",
        )
        
        # 2. ELI5 (Explain Like I'm 5)
        print("  → Generating ELI5 explanation...")
        analyses["eli5"] = self.generate(
            prompt=f"Explain this research paper in simple terms that a non-specialist could understand. Avoid jargon and use analogies:\n\n{context}",
            system_prompt="You explain complex research papers in simple, accessible language. Use everyday analogies and avoid technical jargon.",
        )
        
        # 3. Research contribution analysis
        print("  → Analyzing research contributions...")
        analyses["contributions"] = self.generate(
            prompt=f"Identify and analyze the key research contributions of this paper. List each contribution with a brief explanation:\n\n{context}",
            system_prompt="You are an expert peer reviewer analyzing the novelty and significance of research contributions.",
        )
        
        # 4. Potential applications
        print("  → Identifying applications...")
        analyses["applications"] = self.generate(
            prompt=f"Identify potential real-world applications and impact areas for this research. Consider both immediate and long-term applications:\n\n{context}",
            system_prompt="You identify practical applications of academic research across industry, technology, and society.",
        )
        
        # 5. Limitations
        print("  → Analyzing limitations...")
        analyses["limitations"] = self.generate(
            prompt=f"Identify the limitations and potential weaknesses of this research. Be constructive and specific:\n\n{context}",
            system_prompt="You provide balanced, constructive criticism of research papers, identifying genuine limitations and suggesting improvements.",
        )
        
        data["llm_analysis"] = analyses
        
        for key, val in analyses.items():
            preview = val[:100].replace('\n', ' ') if val else "N/A"
            print(f"  ✓ {key}: {preview}...")
        
        return data

    def _build_context(self, data: Dict) -> str:
        """Build structured context for LLM prompts."""
        parts = []
        
        # Title and authors
        title = data.get("title", "Unknown")
        authors = data.get("authors", [])
        parts.append(f"TITLE: {title}")
        parts.append(f"AUTHORS: {', '.join(authors)}")
        
        # Abstract
        abstract = data.get("abstract", "")
        if abstract:
            parts.append(f"\nABSTRACT:\n{abstract[:1500]}")
        
        # Key sections
        sections = data.get("nl_sections", data.get("sections", []))
        for sec in sections[:8]:
            content = sec.get("nl_text", sec.get("content", ""))
            if content and len(content) > 50:
                parts.append(f"\n{sec.get('title', 'Section').upper()}:\n{content[:800]}")
        
        # Keywords
        keywords = data.get("keywords", {}).get("top_keywords", [])
        if keywords:
            parts.append(f"\nKEY TERMS: {', '.join(keywords[:15])}")
        
        # Math structures
        structures = data.get("math_structures", {})
        if structures.get("summary", {}).get("total_theorems", 0) > 0:
            parts.append(f"\nMATH STRUCTURE: {structures['summary'].get('logical_flow', '')}")
        
        # Equation types
        eq_analysis = data.get("equation_analysis", {})
        if eq_analysis.get("dominant_type"):
            parts.append(f"\nEQUATION TYPES: Dominant type is {eq_analysis['dominant_type']}")
        
        # Extractive summary
        summary = data.get("summary", {}).get("extractive", [])
        if summary:
            parts.append(f"\nKEY SENTENCES:\n" + "\n".join(f"• {s}" for s in summary[:5]))
        
        context = "\n".join(parts)
        # Truncate to fit Groq context window
        return context[:6000]


if __name__ == "__main__":
    client = GroqClient()
    test = {
        "title": "Test Paper",
        "authors": ["Author One"],
        "abstract": "We prove convergence of gradient descent.",
        "nl_sections": [],
        "keywords": {"top_keywords": ["convergence", "gradient"]},
    }
    result = client.analyze_paper(test)
    print(f"\nSummary: {result['llm_analysis']['summary'][:200]}")
