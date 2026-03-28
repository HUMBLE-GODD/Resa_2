"""
Text Cleaner — Remove noise from extracted research paper text.
Handles headers/footers, citation markers, Unicode normalization,
while preserving mathematical expressions.
"""
import re
import unicodedata
from typing import Dict, List, Tuple


class TextCleaner:
    """Clean extracted text while preserving mathematical content."""

    # Patterns to remove
    HEADER_FOOTER = re.compile(
        r'^\s*\d+\s*$|'                          # Page numbers
        r'^\s*Page\s+\d+\s*$|'                    # "Page N"
        r'^\s*-\s*\d+\s*-\s*$|'                   # "- N -"
        r'^\s*(?:preprint|draft|accepted|submitted|PREPRINT|DRAFT|ACCEPTED|SUBMITTED)\s*$|'
        r'^\s*https?://\S+\s*$',                  # URLs on their own line
        re.MULTILINE
    )

    
    CITATION_MARKERS = re.compile(r'\[(\d+(?:,\s*\d+)*)\]')  # [1], [1, 2, 3]
    DOI_PATTERN = re.compile(r'(?:doi|DOI)[:\s]*\S+')
    EMAIL_PATTERN = re.compile(r'\S+@\S+\.\S+')
    
    # Math expression delimiters for tagging
    MATH_INLINE = re.compile(r'\$([^$]+)\$')
    MATH_DISPLAY = re.compile(r'\$\$([^$]+)\$\$')
    MATH_LATEX_ENV = re.compile(
        r'\\begin\{(equation|align|gather|multline)\*?\}(.*?)\\end\{\1\*?\}',
        re.DOTALL
    )

    def clean(self, parsed_data: Dict) -> Dict:
        """
        Clean the parsed data, preserving math expressions.
        
        Args:
            parsed_data: Output from any parser (pdf_parser, latex_parser, text_parser)
        
        Returns:
            Cleaned version of the same structure with additional 'cleaned_text' and
            'math_expressions' fields.
        """
        print("[Phase 2] Cleaning text...")
        
        raw_text = parsed_data.get("raw_text", "")
        
        # Step 1: Extract and tag math expressions before cleaning
        math_expressions, text_with_tags = self._tag_math_expressions(raw_text)
        
        # Step 2: Clean the tagged text
        cleaned = self._clean_text(text_with_tags)
        
        # Step 3: Clean each section individually
        cleaned_sections = []
        for section in parsed_data.get("sections", []):
            section_math, section_tagged = self._tag_math_expressions(section["content"])
            section_cleaned = self._clean_text(section_tagged)
            cleaned_sections.append({
                "title": section["title"],
                "content": section_cleaned,
                "level": section.get("level", 0),
                "math_expressions": section_math,
            })
            math_expressions.extend(section_math)
        
        # Step 4: Clean abstract
        abstract = parsed_data.get("abstract", "")
        abstract_cleaned = self._clean_text(abstract)
        
        result = {
            **parsed_data,
            "abstract": abstract_cleaned,
            "sections": cleaned_sections,
            "cleaned_text": cleaned,
            "math_expressions": list(set(math_expressions)),  # Deduplicate
        }
        
        print(f"  ✓ Cleaned text: {len(raw_text)} → {len(cleaned)} chars ({100-len(cleaned)*100//max(len(raw_text),1)}% removed)")
        print(f"  ✓ Math expressions preserved: {len(math_expressions)}")
        print(f"  ✓ Sections cleaned: {len(cleaned_sections)}")
        
        return result

    def _tag_math_expressions(self, text: str) -> Tuple[List[str], str]:
        """
        Extract math expressions and replace them with [MATH]...[/MATH] tags.
        Returns (list_of_math_expressions, tagged_text).
        """
        math_exprs = []
        
        # Tag display math first ($$...$$)
        def replace_display(match):
            expr = match.group(1).strip()
            math_exprs.append(expr)
            return f" [MATH]{expr}[/MATH] "
        
        text = self.MATH_DISPLAY.sub(replace_display, text)
        
        # Tag LaTeX environments
        def replace_env(match):
            expr = match.group(2).strip()
            math_exprs.append(expr)
            return f" [MATH]{expr}[/MATH] "
        
        text = self.MATH_LATEX_ENV.sub(replace_env, text)
        
        # Tag inline math ($...$)
        def replace_inline(match):
            expr = match.group(1).strip()
            if len(expr) > 1:  # Skip single-char math
                math_exprs.append(expr)
                return f" [MATH]{expr}[/MATH] "
            return match.group(0)
        
        text = self.MATH_INLINE.sub(replace_inline, text)
        
        return math_exprs, text

    def _clean_text(self, text: str) -> str:
        """Apply all cleaning steps to text."""
        # Unicode normalization (NFKD → then recompose to NFC)
        text = unicodedata.normalize('NFKC', text)
        
        # Remove headers/footers
        text = self.HEADER_FOOTER.sub('', text)
        
        # Remove DOIs and emails
        text = self.DOI_PATTERN.sub('', text)
        text = self.EMAIL_PATTERN.sub('', text)
        
        # Remove citation markers but keep the sentence structure
        text = self.CITATION_MARKERS.sub('', text)
        
        # Remove figure/table references like "Fig. 1", "Table 2"
        text = re.sub(r'(?i)\b(?:fig(?:ure)?|table)\s*\.?\s*\d+', '', text)
        
        # Clean up hyphenated line breaks (word-\nword → wordword)
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        
        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove very short lines (likely artifacts)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 2 or '[MATH]' in line:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()


if __name__ == "__main__":
    # Test with sample text
    sample = {
        "raw_text": "Introduction\n\n[1] showed that $x = 5$ is important.\nPage 42\ndoi:10.1234/test\n$$E = mc^2$$\n",
        "abstract": "We prove $f(x) = 0$ [1, 2].",
        "sections": [{"title": "Intro", "content": "See Fig. 1 and [3].", "level": 1}],
    }
    cleaner = TextCleaner()
    result = cleaner.clean(sample)
    print(f"\nCleaned text:\n{result['cleaned_text']}")
    print(f"\nMath: {result['math_expressions']}")
