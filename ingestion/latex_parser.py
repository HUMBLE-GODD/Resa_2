"""
LaTeX Source Parser — Extract structured content from .tex files.
Parses LaTeX commands to extract title, authors, abstract, sections, equations.
"""
import re
from typing import Dict, List


class LaTeXParser:
    """Parse LaTeX source files into structured content."""

    def parse(self, tex_path: str) -> Dict:
        """Parse a .tex file and return structured content."""
        print(f"[Phase 1] Parsing LaTeX: {tex_path}")
        
        with open(tex_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        title = self._extract_command(content, 'title')
        authors = self._extract_authors(content)
        abstract = self._extract_environment(content, 'abstract')
        sections = self._extract_sections(content)
        equations = self._extract_equations(content)
        references = self._extract_references(content)
        
        # Strip LaTeX commands from text for plain text version
        raw_text = self._strip_latex(content)
        
        result = {
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "sections": sections,
            "equations": equations,
            "references": references,
            "raw_text": raw_text,
            "metadata": {
                "source_type": "latex",
                "num_sections": len(sections),
                "num_equations": len(equations),
                "num_references": len(references),
                "total_chars": len(raw_text),
            },
        }
        
        print(f"  ✓ Title: {title[:80] if title else 'Not found'}")
        print(f"  ✓ Sections: {len(sections)}, Equations: {len(equations)}, References: {len(references)}")
        return result

    def _extract_command(self, content: str, command: str) -> str:
        """Extract content of a LaTeX command like \\title{...}."""
        # Handle nested braces
        pattern = re.compile(r'\\' + command + r'\s*\{')
        match = pattern.search(content)
        if not match:
            return ""
        
        start = match.end()
        depth = 1
        i = start
        while i < len(content) and depth > 0:
            if content[i] == '{':
                depth += 1
            elif content[i] == '}':
                depth -= 1
            i += 1
        
        return content[start:i-1].strip()

    def _extract_authors(self, content: str) -> List[str]:
        """Extract author names."""
        author_block = self._extract_command(content, 'author')
        if not author_block:
            return ["Unknown Author"]
        
        # Split by \\and or commas
        parts = re.split(r'\\and|,', author_block)
        authors = []
        for part in parts:
            name = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', part)  # Remove commands
            name = re.sub(r'[\\{}$]', '', name).strip()
            if name and len(name) > 1:
                authors.append(name)
        
        return authors if authors else ["Unknown Author"]

    def _extract_environment(self, content: str, env_name: str) -> str:
        """Extract content of a LaTeX environment."""
        pattern = re.compile(
            r'\\begin\{' + env_name + r'\}(.*?)\\end\{' + env_name + r'\}',
            re.DOTALL
        )
        match = pattern.search(content)
        if match:
            text = match.group(1).strip()
            return self._strip_latex(text)
        return ""

    def _extract_sections(self, content: str) -> List[Dict]:
        """Extract sections and their content."""
        # Find all section/subsection commands
        section_pattern = re.compile(
            r'\\(section|subsection|subsubsection)\*?\{([^}]+)\}'
        )
        
        matches = list(section_pattern.finditer(content))
        sections = []
        
        for i, match in enumerate(matches):
            level = {"section": 1, "subsection": 2, "subsubsection": 3}[match.group(1)]
            title = match.group(2).strip()
            
            # Content is between this section and the next
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_content = content[start:end].strip()
            section_content = self._strip_latex(section_content)
            
            sections.append({
                "title": title,
                "content": section_content,
                "level": level,
            })
        
        return sections

    def _extract_equations(self, content: str) -> List[Dict]:
        """Extract equations from equation environments and inline math."""
        equations = []
        
        # Display equations: equation, align, gather, etc.
        env_pattern = re.compile(
            r'\\begin\{(equation|align|gather|multline|eqnarray)\*?\}(.*?)'
            r'\\end\{\1\*?\}',
            re.DOTALL
        )
        for match in env_pattern.finditer(content):
            equations.append({
                "text": match.group(2).strip(),
                "type": match.group(1),
                "display": True,
            })
        
        # Inline math: $...$ (but not $$...$$)
        inline_pattern = re.compile(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)')
        for match in inline_pattern.finditer(content):
            eq_text = match.group(1).strip()
            if len(eq_text) > 3:  # Skip trivial math
                equations.append({
                    "text": eq_text,
                    "type": "inline",
                    "display": False,
                })
        
        return equations

    def _extract_references(self, content: str) -> List[str]:
        """Extract bibliography entries."""
        refs = []
        
        # bibitem entries
        bibitem_pattern = re.compile(r'\\bibitem\{[^}]*\}\s*(.*?)(?=\\bibitem|\s*\\end\{)', re.DOTALL)
        for match in bibitem_pattern.finditer(content):
            ref = self._strip_latex(match.group(1).strip())
            if ref:
                refs.append(ref)
        
        return refs

    def _strip_latex(self, text: str) -> str:
        """Remove LaTeX commands, leaving plain text."""
        # Remove comments
        text = re.sub(r'%.*$', '', text, flags=re.MULTILINE)
        # Remove common commands but keep their arguments
        text = re.sub(r'\\(?:textbf|textit|emph|text|mathrm|mathbf)\{([^}]*)\}', r'\1', text)
        # Remove commands without arguments
        text = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?', '', text)
        # Remove braces
        text = re.sub(r'[{}]', '', text)
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "paper.tex"
    parser = LaTeXParser()
    result = parser.parse(path)
