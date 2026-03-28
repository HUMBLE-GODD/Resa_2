"""
PDF Parser — Extract structured content from mathematical research papers.
Uses PyMuPDF (fitz) for robust PDF text extraction with structure detection.
"""
import re
import fitz  # PyMuPDF
from typing import Dict, List, Optional, Tuple


class PDFParser:
    """
    Extract structured content from academic PDF files.
    Handles: title, authors, abstract, sections, equations, references.
    """

    # Regex patterns for math detection
    LATEX_PATTERNS = [
        r'\\frac\{', r'\\sum', r'\\int', r'\\prod', r'\\lim',
        r'\\partial', r'\\nabla', r'\\infty', r'\\sqrt',
        r'\^\{', r'_\{', r'\\left', r'\\right',
        r'\\begin\{equation', r'\\begin\{align',
        r'\\mathbb', r'\\mathcal', r'\\mathrm',
    ]
    MATH_UNICODE = re.compile(
        r'[∑∏∫∂∇∞√±×÷≤≥≠≈∈∉⊂⊃∪∩∧∨¬∀∃αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩ]'
    )
    EQUATION_LINE = re.compile(
        r'^[\s]*[A-Za-z][\s]*[=<>≤≥≠]|'   # x = ...
        r'^\s*\([\d]+\)\s*$|'               # (1) equation number
        r'^\s*[\w\s]*=\s*[\w\s]*[+\-\*/]',  # a = b + c
        re.MULTILINE
    )

    # Section header patterns
    SECTION_PATTERNS = [
        r'(?i)^[\s]*(\d+\.?\s+)?(introduction|background)',
        r'(?i)^[\s]*(\d+\.?\s+)?(related\s+work|literature\s+review|previous\s+work)',
        r'(?i)^[\s]*(\d+\.?\s+)?(method|methodology|approach|framework|model)',
        r'(?i)^[\s]*(\d+\.?\s+)?(experiment|evaluation|setup)',
        r'(?i)^[\s]*(\d+\.?\s+)?(result|finding|outcome)',
        r'(?i)^[\s]*(\d+\.?\s+)?(discussion|analysis)',
        r'(?i)^[\s]*(\d+\.?\s+)?(conclusion|summary|future\s+work)',
        r'(?i)^[\s]*(\d+\.?\s+)?(proof|theorem|lemma|proposition|corollary|definition)',
        r'(?i)^[\s]*(\d+\.?\s+)?(preliminary|preliminaries|notation)',
        r'(?i)^[\s]*(abstract)',
        r'(?i)^[\s]*(references|bibliography)',
        r'(?i)^[\s]*(acknowledg)',
        r'(?i)^[\s]*(\d+\.?\s+)?(appendix)',
    ]

    def __init__(self):
        self.compiled_section_patterns = [re.compile(p) for p in self.SECTION_PATTERNS]
        self.compiled_latex = [re.compile(p) for p in self.LATEX_PATTERNS]

    def parse(self, pdf_path: str) -> Dict:
        """
        Parse a PDF file and return structured content.
        
        Returns:
            dict with keys: title, authors, abstract, sections, equations, 
                           references, raw_text, metadata
        """
        print(f"[Phase 1] Parsing PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        
        # Extract raw text and structured blocks
        pages_data = []
        full_text_lines = []
        
        for page_num, page in enumerate(doc):
            # Get text with block-level structure
            blocks = page.get_text("dict")["blocks"]
            page_info = {
                "page_num": page_num + 1,
                "blocks": [],
                "text": page.get_text("text"),
            }
            
            for block in blocks:
                if "lines" in block:  # Text block (not image)
                    for line in block["lines"]:
                        for span in line["spans"]:
                            page_info["blocks"].append({
                                "text": span["text"].strip(),
                                "font_size": span["size"],
                                "font_name": span["font"],
                                "is_bold": "Bold" in span["font"] or "bold" in span["font"],
                                "bbox": span["bbox"],
                                "page": page_num + 1,
                            })
            
            pages_data.append(page_info)
            full_text_lines.append(page_info["text"])
        
        doc.close()
        full_text = "\n".join(full_text_lines)
        
        # Extract structured components
        title = self._extract_title(pages_data)
        authors = self._extract_authors(pages_data)
        abstract = self._extract_abstract(full_text)
        sections = self._extract_sections(full_text)
        equations = self._extract_equations(full_text)
        references = self._extract_references(full_text)
        
        result = {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "sections": sections,
            "equations": equations,
            "references": references,
            "raw_text": full_text,
            "metadata": {
                "num_pages": len(pages_data),
                "num_sections": len(sections),
                "num_equations": len(equations),
                "num_references": len(references),
                "total_chars": len(full_text),
            },
        }
        
        # Print summary
        print(f"  ✓ Title: {title[:80]}..." if len(title) > 80 else f"  ✓ Title: {title}")
        print(f"  ✓ Authors: {', '.join(authors[:3])}{'...' if len(authors) > 3 else ''}")
        print(f"  ✓ Abstract: {len(abstract)} chars")
        print(f"  ✓ Sections: {len(sections)}")
        print(f"  ✓ Equations: {len(equations)}")
        print(f"  ✓ References: {len(references)}")
        print(f"  ✓ Total pages: {len(pages_data)}")
        
        return result

    def _extract_title(self, pages_data: List[Dict]) -> str:
        """Extract title as the largest font text on page 1."""
        if not pages_data or not pages_data[0]["blocks"]:
            return "Unknown Title"
        
        first_page_blocks = pages_data[0]["blocks"]
        if not first_page_blocks:
            return "Unknown Title"
        
        # Find the largest font size on page 1
        max_size = max(b["font_size"] for b in first_page_blocks)
        
        # Collect all text spans with font size close to max (within 1pt)
        title_parts = []
        for block in first_page_blocks:
            if block["font_size"] >= max_size - 1.0:
                text = block["text"].strip()
                if text and len(text) > 2:
                    title_parts.append(text)
        
        title = " ".join(title_parts).strip()
        # Clean up
        title = re.sub(r'\s+', ' ', title)
        return title if title else "Unknown Title"

    def _extract_authors(self, pages_data: List[Dict]) -> List[str]:
        """Extract authors from the area below the title on page 1."""
        if not pages_data or not pages_data[0]["blocks"]:
            return []
        
        blocks = pages_data[0]["blocks"]
        if not blocks:
            return []
        
        max_size = max(b["font_size"] for b in blocks)
        
        # Authors are typically just below the title, in a smaller font
        author_candidates = []
        found_title = False
        
        for block in blocks:
            if block["font_size"] >= max_size - 1.0:
                found_title = True
                continue
            if found_title and block["font_size"] < max_size - 1.0:
                text = block["text"].strip()
                # Skip if it looks like an abstract or section header
                if re.match(r'(?i)(abstract|introduction|\d+\.)', text):
                    break
                # Skip email addresses, affiliations with numbers
                if '@' in text or re.match(r'^\d+$', text):
                    continue
                if text and len(text) > 1:
                    author_candidates.append(text)
                    if len(author_candidates) > 10:
                        break
        
        # Try to split on commas or "and"
        authors = []
        for candidate in author_candidates:
            parts = re.split(r',\s*|\s+and\s+', candidate)
            for p in parts:
                p = p.strip().strip(',').strip()
                # Filter: likely a name if 2-4 words, all alpha
                words = p.split()
                if 1 <= len(words) <= 5 and all(w.replace('.', '').replace('-', '').isalpha() for w in words):
                    authors.append(p)
        
        return authors if authors else ["Unknown Author"]

    def _extract_abstract(self, text: str) -> str:
        """Extract the abstract section."""
        # Try to find explicit "Abstract" header
        patterns = [
            r'(?i)abstract\s*\n(.*?)(?=\n\s*(?:\d+\.?\s+)?(?:introduction|keywords|key\s*words|1\s))',
            r'(?i)abstract[:\s]*\n(.*?)(?=\n\s*\n\s*\n)',
            r'(?i)abstract[:\s]*(.*?)(?=\n\s*(?:\d+\.?\s+)?introduction)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # Clean up: remove excessive whitespace
                abstract = re.sub(r'\s+', ' ', abstract)
                if len(abstract) > 50:
                    return abstract
        
        # Fallback: first substantial paragraph after title
        paragraphs = text.split('\n\n')
        for para in paragraphs[1:5]:
            para = para.strip()
            if len(para) > 100:
                return re.sub(r'\s+', ' ', para)
        
        return "Abstract not found"

    def _extract_sections(self, text: str) -> List[Dict]:
        """Extract sections with their content."""
        lines = text.split('\n')
        sections = []
        current_section = {"title": "preamble", "content": [], "level": 0}
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Check if this line is a section header
            is_header = False
            header_title = line_stripped
            
            for pattern in self.compiled_section_patterns:
                if pattern.match(line_stripped):
                    is_header = True
                    break
            
            # Also detect numbered sections like "1. Introduction", "2.1 Methods"
            numbered = re.match(r'^(\d+\.?\d*\.?\s+)(.+)$', line_stripped)
            if numbered and len(line_stripped) < 100:
                is_header = True
                header_title = line_stripped
            
            if is_header and len(line_stripped) < 120:
                # Save previous section
                if current_section["content"]:
                    current_section["content"] = '\n'.join(current_section["content"])
                    sections.append(current_section)
                
                current_section = {
                    "title": header_title,
                    "content": [],
                    "level": 1 if numbered else 0,
                }
            else:
                current_section["content"].append(line_stripped)
        
        # Save last section
        if current_section["content"]:
            current_section["content"] = '\n'.join(current_section["content"])
            sections.append(current_section)
        
        return sections

    def _extract_equations(self, text: str) -> List[Dict]:
        """Extract mathematical equations from the text."""
        equations = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            is_equation = False
            eq_type = "unknown"
            
            # Check LaTeX patterns
            for pattern in self.compiled_latex:
                if pattern.search(line):
                    is_equation = True
                    eq_type = "latex"
                    break
            
            # Check Unicode math symbols
            if not is_equation and self.MATH_UNICODE.search(line):
                # Must have enough math symbols relative to text length
                math_chars = len(self.MATH_UNICODE.findall(line))
                if math_chars >= 2 or (math_chars >= 1 and len(line) < 50):
                    is_equation = True
                    eq_type = "unicode_math"
            
            # Check equation-like patterns
            if not is_equation and self.EQUATION_LINE.match(line):
                # Additional heuristic: short lines with = sign
                if '=' in line and len(line) < 150:
                    is_equation = True
                    eq_type = "inline"
            
            if is_equation:
                equations.append({
                    "text": line,
                    "line_number": i + 1,
                    "type": eq_type,
                    "context": lines[max(0, i-1):min(len(lines), i+2)],
                })
        
        return equations

    def _extract_references(self, text: str) -> List[str]:
        """Extract the reference list."""
        # Find the references section
        ref_match = re.search(
            r'(?i)\n\s*(references|bibliography)\s*\n(.*)',
            text, re.DOTALL
        )
        
        if not ref_match:
            return []
        
        ref_text = ref_match.group(2)
        
        # Split into individual references
        # Pattern: [1] or 1. or [Author, Year]
        refs = re.split(r'\n\s*\[?\d+\]?\s*\.?\s*(?=[A-Z])', ref_text)
        
        cleaned_refs = []
        for ref in refs:
            ref = ref.strip()
            ref = re.sub(r'\s+', ' ', ref)
            if len(ref) > 20 and len(ref) < 500:
                cleaned_refs.append(ref)
        
        return cleaned_refs


# ============================================================
# Standalone test
# ============================================================
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "paperDCRE-1.pdf"
    parser = PDFParser()
    result = parser.parse(path)
    print(f"\n{'='*60}")
    print("PHASE 1 EXTRACTION COMPLETE")
    print(f"{'='*60}")
    for key, val in result["metadata"].items():
        print(f"  {key}: {val}")
