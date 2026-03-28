"""
Dual Representation — Create parallel natural language and mathematical
expression representations from cleaned text.
"""
import re
from typing import Dict, List, Tuple


class DualRepresentation:
    """
    Separate text into two parallel streams:
    1. Natural language text (for NLP models)
    2. Mathematical expressions (for math-specific analysis)
    """

    MATH_TAG = re.compile(r'\[MATH\](.*?)\[/MATH\]')

    def create(self, cleaned_data: Dict) -> Dict:
        """
        Create dual representation from cleaned data.
        
        Args:
            cleaned_data: Output from TextCleaner with [MATH] tagged expressions
            
        Returns:
            Dict with 'nl_text', 'math_expressions', and paired 'dual_sections'
        """
        print("[Phase 2] Creating dual representation...")
        
        cleaned_text = cleaned_data.get("cleaned_text", "")
        sections = cleaned_data.get("sections", [])
        
        # Global separation
        nl_text, math_exprs = self._separate(cleaned_text)
        
        # Per-section separation
        dual_sections = []
        all_math = list(math_exprs)
        
        for section in sections:
            sec_nl, sec_math = self._separate(section["content"])
            dual_sections.append({
                "title": section["title"],
                "nl_text": sec_nl,
                "math_expressions": sec_math,
                "original_content": section["content"],
                "level": section.get("level", 0),
                "math_density": len(sec_math) / max(len(sec_nl.split()), 1),
            })
            all_math.extend(sec_math)
        
        # Deduplicate math expressions
        unique_math = list(dict.fromkeys(all_math))
        
        result = {
            **cleaned_data,
            "nl_text": nl_text,
            "nl_sections": dual_sections,
            "all_math_expressions": unique_math,
            "math_stats": {
                "total_expressions": len(unique_math),
                "inline_count": sum(1 for m in unique_math if len(m) < 30),
                "display_count": sum(1 for m in unique_math if len(m) >= 30),
                "sections_with_math": sum(1 for s in dual_sections if s["math_expressions"]),
            },
        }
        
        print(f"  ✓ NL text: {len(nl_text)} chars")
        print(f"  ✓ Total math expressions: {len(unique_math)}")
        print(f"  ✓ Inline: {result['math_stats']['inline_count']}, Display: {result['math_stats']['display_count']}")
        print(f"  ✓ Sections with math: {result['math_stats']['sections_with_math']}/{len(dual_sections)}")
        
        return result

    def _separate(self, text: str) -> Tuple[str, List[str]]:
        """
        Separate text into NL text and math expressions.
        
        Returns:
            (natural_language_text, list_of_math_expressions)
        """
        math_expressions = self.MATH_TAG.findall(text)
        
        # Replace math with placeholder for NL text
        nl_text = self.MATH_TAG.sub('[EQUATION]', text)
        
        # Clean up multiple spaces
        nl_text = re.sub(r'\s+', ' ', nl_text).strip()
        
        return nl_text, math_expressions

    def get_math_context(self, text: str, window: int = 50) -> List[Dict]:
        """
        For each math expression, extract its surrounding text context.
        Useful for understanding what each equation represents.
        """
        contexts = []
        for match in self.MATH_TAG.finditer(text):
            start = max(0, match.start() - window)
            end = min(len(text), match.end() + window)
            
            contexts.append({
                "expression": match.group(1),
                "before": text[start:match.start()].strip(),
                "after": text[match.end():end].strip(),
            })
        
        return contexts


if __name__ == "__main__":
    # Test
    sample = {
        "cleaned_text": "We prove that [MATH]f(x) = 0[/MATH] for all [MATH]x \\in \\mathbb{R}[/MATH].",
        "sections": [
            {"title": "Theorem 1", "content": "Let [MATH]g(x) = x^2[/MATH]. Then [MATH]g'(x) = 2x[/MATH].", "level": 1}
        ],
    }
    dual = DualRepresentation()
    result = dual.create(sample)
    print(f"\nNL text: {result['nl_text']}")
    print(f"Math: {result['all_math_expressions']}")
