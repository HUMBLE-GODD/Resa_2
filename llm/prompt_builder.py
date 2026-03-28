"""
Prompt Builder — Construct structured prompts for LLM analysis.
"""
from typing import Dict, List


class PromptBuilder:
    """Build optimized prompts for different analysis tasks."""

    @staticmethod
    def summary_prompt(data: Dict) -> str:
        """Build a summary generation prompt."""
        abstract = data.get("abstract", "No abstract available")
        keywords = data.get("keywords", {}).get("top_keywords", [])
        
        return f"""Analyze and summarize this research paper:

Abstract: {abstract[:1000]}

Key Terms: {', '.join(keywords[:10])}

Provide:
1. A concise summary (2-3 paragraphs)
2. The main research question
3. Key methodology
4. Primary findings
5. Significance of the work"""

    @staticmethod
    def eli5_prompt(data: Dict) -> str:
        """Build an ELI5 explanation prompt."""
        abstract = data.get("abstract", "")
        title = data.get("title", "")
        
        return f"""Explain this research paper titled "{title}" in very simple terms.

Abstract: {abstract[:800]}

Rules:
- Use everyday language, no jargon
- Use analogies and examples
- Explain any math concepts simply
- Keep it under 200 words
- A high school student should understand it"""

    @staticmethod
    def contribution_prompt(data: Dict) -> str:
        """Build a research contribution analysis prompt."""
        abstract = data.get("abstract", "")
        structures = data.get("math_structures", {})
        
        theorem_count = structures.get("summary", {}).get("total_theorems", 0)
        
        return f"""Analyze the research contributions of this paper:

Abstract: {abstract[:800]}

Mathematical content: {theorem_count} theorems/lemmas found.

List:
1. Novel theoretical contributions
2. Methodological innovations
3. Practical implications
4. How this advances the field"""

    @staticmethod
    def limitation_prompt(data: Dict) -> str:
        """Build a limitations analysis prompt."""
        abstract = data.get("abstract", "")
        
        return f"""Identify limitations and potential improvements for this research:

Abstract: {abstract[:800]}

Analyze:
1. Theoretical assumptions that may not hold
2. Experimental limitations
3. Generalizability concerns
4. Missing comparisons or baselines
5. Suggestions for future work"""

    @staticmethod
    def application_prompt(data: Dict) -> str:
        """Build an applications identification prompt."""
        abstract = data.get("abstract", "")
        keywords = data.get("keywords", {}).get("top_keywords", [])
        
        return f"""Identify real-world applications for this research:

Abstract: {abstract[:800]}
Key terms: {', '.join(keywords[:10])}

Consider:
1. Industry applications
2. Technology implications
3. Scientific impact
4. Societal benefit
5. Commercialization potential"""
