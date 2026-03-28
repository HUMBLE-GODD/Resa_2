"""
Mathematical Structure Detector — Identify definitions, theorems,
proof structures, and mathematical relationships in text.
"""
import re
from typing import Dict, List


class MathStructureDetector:
    """
    Detect mathematical structures in research papers:
    - Definitions
    - Theorems / Lemmas / Propositions / Corollaries
    - Proof blocks
    - Mathematical relationships and dependencies
    """

    # Patterns for mathematical structures
    DEFINITION_START = re.compile(
        r'(?i)(?:^|\n)\s*(?:definition|def\.?)\s+(\d+(?:\.\d+)*)?[\s.:]*(.+?)(?=\n\s*\n|\Z)',
        re.DOTALL
    )
    THEOREM_START = re.compile(
        r'(?i)(?:^|\n)\s*(?:theorem|thm\.?)\s+(\d+(?:\.\d+)*)?[\s.:]*(.+?)(?=\n\s*(?:proof|lemma|theorem|definition|corollary|remark|\Z))',
        re.DOTALL
    )
    LEMMA_START = re.compile(
        r'(?i)(?:^|\n)\s*(?:lemma|lem\.?)\s+(\d+(?:\.\d+)*)?[\s.:]*(.+?)(?=\n\s*(?:proof|lemma|theorem|definition|corollary|remark|\Z))',
        re.DOTALL
    )
    PROPOSITION_START = re.compile(
        r'(?i)(?:^|\n)\s*(?:proposition|prop\.?)\s+(\d+(?:\.\d+)*)?[\s.:]*(.+?)(?=\n\s*(?:proof|lemma|theorem|definition|corollary|\Z))',
        re.DOTALL
    )
    COROLLARY_START = re.compile(
        r'(?i)(?:^|\n)\s*(?:corollary|cor\.?)\s+(\d+(?:\.\d+)*)?[\s.:]*(.+?)(?=\n\s*(?:proof|lemma|theorem|definition|\Z))',
        re.DOTALL
    )
    PROOF_START = re.compile(
        r'(?i)(?:^|\n)\s*proof[\s.:]*(.+?)(?=\s*(?:□|QED|∎|\\qed|q\.e\.d\.)|(?=\n\s*(?:theorem|lemma|definition|corollary|remark|section|\d+\.))|$)',
        re.DOTALL
    )
    
    # Relationship patterns
    IMPLIES = re.compile(r'(?i)(implies|it follows|therefore|hence|thus|consequently|we conclude)')
    EQUIVALENCE = re.compile(r'(?i)(if and only if|iff|equivalent|is equivalent)')
    ASSUMPTION = re.compile(r'(?i)(assume|suppose|let|given that|under the assumption)')
    CONDITION = re.compile(r'(?i)(necessary condition|sufficient condition|necessary and sufficient)')

    def detect(self, data: Dict) -> Dict:
        """
        Detect mathematical structures in the paper.
        
        Args:
            data: Pipeline data with text content
            
        Returns:
            Updated data with 'math_structures' field
        """
        print("[Phase 3] Detecting mathematical structures...")
        
        text = data.get("nl_text", data.get("cleaned_text", ""))
        
        definitions = self._find_structures(text, self.DEFINITION_START, "definition")
        theorems = self._find_structures(text, self.THEOREM_START, "theorem")
        lemmas = self._find_structures(text, self.LEMMA_START, "lemma")
        propositions = self._find_structures(text, self.PROPOSITION_START, "proposition")
        corollaries = self._find_structures(text, self.COROLLARY_START, "corollary")
        proofs = self._find_proofs(text)
        relationships = self._find_relationships(text)
        
        structures = {
            "definitions": definitions,
            "theorems": theorems,
            "lemmas": lemmas,
            "propositions": propositions,
            "corollaries": corollaries,
            "proofs": proofs,
            "relationships": relationships,
            "summary": {
                "total_definitions": len(definitions),
                "total_theorems": len(theorems),
                "total_lemmas": len(lemmas),
                "total_propositions": len(propositions),
                "total_corollaries": len(corollaries),
                "total_proofs": len(proofs),
                "logical_flow": self._analyze_flow(definitions, theorems, lemmas, proofs),
            }
        }
        
        data["math_structures"] = structures
        
        print(f"  ✓ Definitions: {len(definitions)}")
        print(f"  ✓ Theorems: {len(theorems)}")
        print(f"  ✓ Lemmas: {len(lemmas)}")
        print(f"  ✓ Propositions: {len(propositions)}")
        print(f"  ✓ Corollaries: {len(corollaries)}")
        print(f"  ✓ Proofs: {len(proofs)}")
        print(f"  ✓ Relationships: {len(relationships)}")
        
        return data

    def _find_structures(self, text: str, pattern: re.Pattern, struct_type: str) -> List[Dict]:
        """Find mathematical structures matching a pattern."""
        structures = []
        
        for match in pattern.finditer(text):
            struct_id = match.group(1) if match.group(1) else str(len(structures) + 1)
            content = match.group(2).strip() if match.lastindex >= 2 else match.group(0).strip()
            
            # Clean content
            content = re.sub(r'\s+', ' ', content)
            if len(content) > 500:
                content = content[:500] + "..."
            
            structures.append({
                "type": struct_type,
                "id": struct_id,
                "content": content,
                "position": match.start(),
                "has_equation": bool(re.search(r'\[EQUATION\]|\[MATH\]|[=<>≤≥]', content)),
            })
        
        return structures

    def _find_proofs(self, text: str) -> List[Dict]:
        """Find proof blocks."""
        proofs = []
        
        for match in self.PROOF_START.finditer(text):
            content = match.group(1).strip()
            content = re.sub(r'\s+', ' ', content)
            
            # Estimate proof length
            proof_length = len(content.split())
            
            proofs.append({
                "content": content[:300] + "..." if len(content) > 300 else content,
                "position": match.start(),
                "word_count": proof_length,
                "technique": self._identify_proof_technique(content),
            })
        
        return proofs

    def _identify_proof_technique(self, proof_text: str) -> str:
        """Identify the proof technique used."""
        text_lower = proof_text.lower()
        
        if 'contradiction' in text_lower or 'absurd' in text_lower:
            return "proof_by_contradiction"
        elif 'induction' in text_lower:
            if 'strong induction' in text_lower:
                return "strong_induction"
            return "mathematical_induction"
        elif 'contrapositive' in text_lower:
            return "proof_by_contrapositive"
        elif 'construct' in text_lower:
            return "constructive_proof"
        elif 'direct' in text_lower:
            return "direct_proof"
        elif 'case' in text_lower and ('case 1' in text_lower or 'case 2' in text_lower):
            return "proof_by_cases"
        else:
            return "unclassified"

    def _find_relationships(self, text: str) -> List[Dict]:
        """Find logical relationships in the text."""
        relationships = []
        
        for pattern, rel_type in [
            (self.IMPLIES, "implication"),
            (self.EQUIVALENCE, "equivalence"),
            (self.ASSUMPTION, "assumption"),
            (self.CONDITION, "condition"),
        ]:
            for match in pattern.finditer(text):
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()
                
                relationships.append({
                    "type": rel_type,
                    "keyword": match.group(1),
                    "context": context,
                    "position": match.start(),
                })
        
        return relationships

    def _analyze_flow(self, definitions, theorems, lemmas, proofs) -> str:
        """Analyze the logical flow of the paper."""
        total = len(definitions) + len(theorems) + len(lemmas) + len(proofs)
        
        if total == 0:
            return "No formal mathematical structures detected — likely an applied/computational paper"
        elif len(theorems) > 0 and len(proofs) > 0:
            ratio = len(proofs) / len(theorems)
            if ratio >= 0.8:
                return f"Well-structured theoretical paper: {len(theorems)} theorems with {len(proofs)} proofs"
            else:
                return f"Theoretical paper with some unproven claims: {len(theorems)} theorems, {len(proofs)} proofs"
        elif len(definitions) > 0:
            return f"Definition-heavy paper establishing new framework: {len(definitions)} definitions"
        else:
            return f"Mixed paper with {total} mathematical structures"


if __name__ == "__main__":
    detector = MathStructureDetector()
    test = {
        "nl_text": (
            "Definition 1. A function f is continuous if for every epsilon > 0...\n\n"
            "Theorem 2.1. If f is continuous on [a,b], then f is integrable.\n\n"
            "Proof. We proceed by contradiction. Assume f is not integrable...\n□"
        )
    }
    result = detector.detect(test)
    print(f"\nStructures: {result['math_structures']['summary']}")
