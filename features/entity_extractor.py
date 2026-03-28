"""
Entity Extractor — Named Entity Recognition for academic papers.
Combines spaCy NER with custom rules for mathematical entities.
"""
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List
from config import SPACY_MODEL


class EntityExtractor:
    """
    Extract named entities from research papers:
    - Person names (authors referenced)
    - Organizations
    - Theorem/lemma/definition names
    - Method/algorithm names
    - Mathematical concepts
    """

    # Custom patterns for academic entities
    THEOREM_PATTERN = re.compile(
        r'(?i)(theorem|lemma|proposition|corollary|conjecture|definition|remark|example)\s+'
        r'(\d+(?:\.\d+)*|\w+)',
    )
    METHOD_PATTERN = re.compile(
        r'(?i)(algorithm|method|approach|technique|framework|model|scheme|procedure)\s+'
        r'(\d+(?:\.\d+)*|[A-Z][\w-]*(?:\s+[A-Z][\w-]*)*)',
    )
    CONCEPT_PATTERN = re.compile(
        r'(?i)(convergence|continuity|differentiability|compactness|completeness|'
        r'boundedness|integrability|measurability|ergodicity|stationarity|'
        r'optimality|convexity|regularity|smoothness|analyticity)',
    )
    
    # Named method patterns
    NAMED_METHODS = re.compile(
        r'(?i)\b(gradient descent|stochastic gradient|backpropagation|'
        r'newton[\'s]* method|euler[\'s]* method|runge[- ]kutta|'
        r'monte carlo|markov chain|bayesian|gaussian|'
        r'fourier transform|laplace transform|'
        r'finite element|finite difference|'
        r'principal component|singular value|'
        r'least squares|maximum likelihood)\b',
    )

    def __init__(self):
        self._nlp = None

    def _load_spacy(self):
        """Lazy-load spaCy model."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load(SPACY_MODEL)
                print(f"  ✓ spaCy model loaded: {SPACY_MODEL}")
            except OSError:
                print(f"  ⚠ Downloading spaCy model: {SPACY_MODEL}")
                import spacy.cli
                spacy.cli.download(SPACY_MODEL)
                import spacy
                self._nlp = spacy.load(SPACY_MODEL)

    def extract(self, data: Dict) -> Dict:
        """
        Extract entities from the paper data.
        
        Returns:
            Updated data dict with 'entities' field
        """
        print("[Phase 3] Extracting entities...")
        
        text = data.get("nl_text", data.get("cleaned_text", ""))
        
        # spaCy NER
        self._load_spacy()
        spacy_entities = self._spacy_ner(text)
        print(f"  ✓ spaCy NER: {len(spacy_entities)} entities")
        
        # Custom: Theorems, Lemmas, etc.
        math_entities = self._extract_math_entities(text)
        print(f"  ✓ Math entities: {len(math_entities)} (theorems/lemmas/definitions)")
        
        # Custom: Methods and algorithms
        methods = self._extract_methods(text)
        print(f"  ✓ Methods/algorithms: {len(methods)}")
        
        # Custom: Mathematical concepts
        concepts = self._extract_concepts(text)
        print(f"  ✓ Mathematical concepts: {len(concepts)}")
        
        # Combine all entities
        all_entities = {
            "persons": [e for e in spacy_entities if e["type"] == "PERSON"],
            "organizations": [e for e in spacy_entities if e["type"] == "ORG"],
            "math_entities": math_entities,
            "methods": methods,
            "concepts": concepts,
            "all_spacy": spacy_entities,
        }
        
        data["entities"] = all_entities
        
        # Print summary
        print(f"  ✓ Persons: {[e['text'] for e in all_entities['persons'][:5]]}")
        print(f"  ✓ Methods: {[m['name'] for m in methods[:5]]}")
        print(f"  ✓ Concepts: {concepts[:5]}")
        
        return data

    def _spacy_ner(self, text: str) -> List[Dict]:
        """Run spaCy NER pipeline."""
        # Process in chunks if text is very long (spaCy has memory limits)
        max_chars = 100000
        entities = []
        seen = set()
        
        for i in range(0, len(text), max_chars):
            chunk = text[i:i + max_chars]
            doc = self._nlp(chunk)
            
            for ent in doc.ents:
                key = (ent.text.strip(), ent.label_)
                if key not in seen and len(ent.text.strip()) > 1:
                    seen.add(key)
                    entities.append({
                        "text": ent.text.strip(),
                        "type": ent.label_,
                        "start": ent.start_char + i,
                        "end": ent.end_char + i,
                    })
        
        return entities

    def _extract_math_entities(self, text: str) -> List[Dict]:
        """Extract theorem-like mathematical structures."""
        entities = []
        seen = set()
        
        for match in self.THEOREM_PATTERN.finditer(text):
            entity_type = match.group(1).lower()
            entity_id = match.group(2)
            key = f"{entity_type}_{entity_id}"
            
            if key not in seen:
                seen.add(key)
                # Get surrounding context
                start = max(0, match.start() - 10)
                end = min(len(text), match.end() + 200)
                context = text[match.start():end].split('.')[0]
                
                entities.append({
                    "type": entity_type,
                    "id": entity_id,
                    "name": f"{entity_type.title()} {entity_id}",
                    "context": context.strip(),
                })
        
        return entities

    def _extract_methods(self, text: str) -> List[Dict]:
        """Extract method and algorithm names."""
        methods = []
        seen = set()
        
        # Named methods
        for match in self.NAMED_METHODS.finditer(text):
            name = match.group(1).strip()
            if name.lower() not in seen:
                seen.add(name.lower())
                methods.append({
                    "name": name,
                    "type": "named_method",
                    "position": match.start(),
                })
        
        # Pattern-based methods
        for match in self.METHOD_PATTERN.finditer(text):
            prefix = match.group(1)
            name = match.group(2).strip()
            full_name = f"{prefix} {name}"
            if full_name.lower() not in seen and len(name) > 1:
                seen.add(full_name.lower())
                methods.append({
                    "name": full_name,
                    "type": "pattern_method",
                    "position": match.start(),
                })
        
        return methods

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract mathematical concept mentions."""
        concepts = set()
        for match in self.CONCEPT_PATTERN.finditer(text):
            concepts.add(match.group(1).lower())
        return sorted(concepts)


if __name__ == "__main__":
    extractor = EntityExtractor()
    test_data = {
        "nl_text": "Theorem 3.1 shows that the gradient descent method converges. "
                   "Professor Smith proved convergence using Euler's method. "
                   "The algorithm exhibits convexity and regularity properties.",
    }
    result = extractor.extract(test_data)
    print(f"\nEntities: {result['entities']['math_entities']}")
