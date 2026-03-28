"""
Citation Analysis — Analyze references and in-text citation patterns.
Identifies influential references and citation context.
"""
import re
from typing import Dict, List
from collections import Counter


class CitationAnalyzer:
    """
    Analyze the citation patterns in a research paper:
    - Count in-text citation frequencies
    - Identify most-cited references
    - Extract citation context
    - Classify citation purpose
    """

    # Citation patterns
    NUMERIC_CITE = re.compile(r'\[(\d+(?:\s*[,;]\s*\d+)*)\]')
    AUTHOR_YEAR = re.compile(r'(?:([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?)\s*\((\d{4})\))')
    
    # Citation purpose keywords
    CITATION_PURPOSES = {
        "background": ["introduced", "proposed", "defined", "established", "pioneered"],
        "methodology": ["method", "approach", "technique", "algorithm", "framework", "following"],
        "comparison": ["compared", "outperform", "baseline", "benchmark", "versus", "contrast"],
        "extension": ["extend", "build upon", "generalize", "improve", "enhance"],
        "support": ["confirm", "support", "consistent", "agreement", "corroborate"],
        "contrast": ["however", "unlike", "contrary", "differ", "limitation"],
    }

    def analyze(self, data: Dict) -> Dict:
        """
        Analyze citations in the paper.
        
        Returns:
            Updated data with 'citation_analysis' field
        """
        print("[Phase 5] Analyzing citations...")
        
        text = data.get("nl_text", data.get("cleaned_text", data.get("raw_text", "")))
        references = data.get("references", [])
        
        # Count in-text citations
        citation_counts = self._count_citations(text)
        print(f"  ✓ In-text citations found: {sum(citation_counts.values())}")
        
        # Extract citation contexts
        contexts = self._extract_citation_contexts(text)
        print(f"  ✓ Citation contexts extracted: {len(contexts)}")
        
        # Rank references by frequency
        ranked_refs = self._rank_references(citation_counts, references)
        print(f"  ✓ Ranked references: {len(ranked_refs)}")
        
        # Classify citation purposes
        purpose_distribution = self._classify_purposes(contexts)
        print(f"  ✓ Citation purposes: {purpose_distribution}")
        
        # Self-citation detection
        authors = data.get("authors", [])
        self_citations = self._detect_self_citations(references, authors)
        
        data["citation_analysis"] = {
            "citation_counts": dict(citation_counts.most_common()),
            "total_citations": sum(citation_counts.values()),
            "unique_references": len(references),
            "ranked_references": ranked_refs[:10],
            "citation_contexts": contexts[:20],
            "purpose_distribution": purpose_distribution,
            "self_citation_count": self_citations,
            "citation_density": sum(citation_counts.values()) / max(len(text.split()), 1) * 100,
        }
        
        print(f"  ✓ Most cited: {ranked_refs[:3] if ranked_refs else 'None'}")
        print(f"  ✓ Citation density: {data['citation_analysis']['citation_density']:.2f} per 100 words")
        
        return data

    def _count_citations(self, text: str) -> Counter:
        """Count frequency of each citation reference."""
        counts = Counter()
        
        # Numeric citations: [1], [1, 2, 3]
        for match in self.NUMERIC_CITE.finditer(text):
            refs = re.split(r'[,;]\s*', match.group(1))
            for ref in refs:
                ref = ref.strip()
                if ref.isdigit():
                    counts[int(ref)] += 1
        
        # Author-year citations
        for match in self.AUTHOR_YEAR.finditer(text):
            author = match.group(1)
            year = match.group(2)
            counts[f"{author} ({year})"] += 1
        
        return counts

    def _extract_citation_contexts(self, text: str) -> List[Dict]:
        """Extract the text surrounding each citation."""
        contexts = []
        
        for match in self.NUMERIC_CITE.finditer(text):
            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            
            context = text[start:end].strip()
            refs = re.split(r'[,;]\s*', match.group(1))
            
            contexts.append({
                "references": [r.strip() for r in refs],
                "context": context,
                "position": match.start(),
                "purpose": self._detect_purpose(context),
            })
        
        return contexts

    def _rank_references(self, citation_counts: Counter, references: List[str]) -> List[Dict]:
        """Rank references by citation frequency."""
        ranked = []
        
        for ref_id, count in citation_counts.most_common():
            ref_text = ""
            if isinstance(ref_id, int) and ref_id - 1 < len(references):
                ref_text = references[ref_id - 1]
            
            ranked.append({
                "id": ref_id,
                "count": count,
                "text": ref_text[:200] if ref_text else f"Reference {ref_id}",
                "influence_rank": len(ranked) + 1,
            })
        
        return ranked

    def _detect_purpose(self, context: str) -> str:
        """Detect the purpose of a citation from its context."""
        context_lower = context.lower()
        
        for purpose, keywords in self.CITATION_PURPOSES.items():
            for kw in keywords:
                if kw in context_lower:
                    return purpose
        
        return "general"

    def _classify_purposes(self, contexts: List[Dict]) -> Dict:
        """Get distribution of citation purposes."""
        purpose_counts = Counter()
        for ctx in contexts:
            purpose_counts[ctx["purpose"]] += 1
        
        total = sum(purpose_counts.values()) or 1
        return {k: {"count": v, "percentage": round(v / total * 100, 1)} for k, v in purpose_counts.items()}

    def _detect_self_citations(self, references: List[str], authors: List[str]) -> int:
        """Detect potential self-citations."""
        if not authors or not references:
            return 0
        
        count = 0
        author_last_names = [a.split()[-1].lower() for a in authors if a != "Unknown Author"]
        
        for ref in references:
            ref_lower = ref.lower()
            for name in author_last_names:
                if name in ref_lower:
                    count += 1
                    break
        
        return count


if __name__ == "__main__":
    analyzer = CitationAnalyzer()
    test = {
        "nl_text": "As shown by Smith et al. [1], the method converges. Building on [2, 3], we extend the framework. Unlike [4], our approach handles non-convex objectives.",
        "references": [
            "Smith, J. et al. Convergence of gradient methods. 2020.",
            "Johnson, A. Non-convex optimization. 2019.",
            "Brown, B. et al. Deep learning theory. 2021.",
            "Lee, C. Convex relaxation methods. 2018.",
        ],
        "authors": ["Smith, J."],
    }
    result = analyzer.analyze(test)
    print(f"\nTop cited: {result['citation_analysis']['ranked_references'][:3]}")
