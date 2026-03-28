"""
Plain Text Parser — Heuristic-based parsing for plain text input.
"""
import re
from typing import Dict, List


class TextParser:
    """Parse plain text files with heuristic structure detection."""

    def parse(self, text_path: str) -> Dict:
        """Parse a plain text file and return structured content."""
        print(f"[Phase 1] Parsing text: {text_path}")
        
        with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        return self.parse_string(content)

    def parse_string(self, content: str) -> Dict:
        """Parse a plain text string."""
        lines = content.split('\n')
        
        # Title: first non-empty line
        title = ""
        for line in lines:
            if line.strip():
                title = line.strip()
                break
        
        # Detect sections by all-caps lines or lines followed by underlines
        sections = self._detect_sections(lines)
        
        # Extract abstract
        abstract = ""
        for sec in sections:
            if 'abstract' in sec['title'].lower():
                abstract = sec['content']
                break
        
        if not abstract and len(content) > 200:
            abstract = content[:500]
        
        return {
            "title": title,
            "authors": ["Unknown Author"],
            "abstract": abstract,
            "sections": sections,
            "equations": [],
            "references": [],
            "raw_text": content,
            "metadata": {
                "source_type": "text",
                "num_sections": len(sections),
                "num_equations": 0,
                "num_references": 0,
                "total_chars": len(content),
            },
        }

    def _detect_sections(self, lines: List[str]) -> List[Dict]:
        """Detect sections by heading patterns."""
        sections = []
        current = {"title": "document", "content": [], "level": 0}
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            
            is_heading = False
            
            # All caps line (likely a heading)
            if stripped.isupper() and 3 < len(stripped) < 80:
                is_heading = True
            # Numbered heading: "1. Introduction"
            elif re.match(r'^\d+\.?\s+[A-Z]', stripped) and len(stripped) < 80:
                is_heading = True
            # Line followed by underline (=== or ---)
            elif i + 1 < len(lines) and re.match(r'^[=\-]{3,}$', lines[i + 1].strip()):
                is_heading = True
            
            if is_heading:
                if current["content"]:
                    current["content"] = '\n'.join(current["content"])
                    sections.append(current)
                current = {"title": stripped, "content": [], "level": 1}
            else:
                current["content"].append(stripped)
        
        if current["content"]:
            current["content"] = '\n'.join(current["content"])
            sections.append(current)
        
        return sections


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "paper.txt"
    parser = TextParser()
    result = parser.parse(path)
    print(f"Sections: {len(result['sections'])}")
