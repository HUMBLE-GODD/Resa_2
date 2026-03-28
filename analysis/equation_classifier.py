"""
Equation Classifier — Classify mathematical equations by type.
Uses keyword/pattern matching for equation categorization.
"""
import re
from typing import Dict, List
from collections import Counter


class EquationClassifier:
    """
    Classify mathematical equations into categories:
    algebra, calculus, optimization, probability, linear_algebra,
    differential_equations, number_theory, statistics, geometry, other
    """

    # Classification rules (keyword → type mapping)
    CLASSIFICATION_RULES = {
        "optimization": {
            "keywords": [
                "min", "max", "argmin", "argmax", "minimize", "maximize",
                "subject to", "s.t.", "optimal", "objective",
                "∇", "gradient", "loss", "cost function",
            ],
            "patterns": [
                r'\\min', r'\\max', r'\\arg\s*min', r'\\arg\s*max',
                r'\\text\{minimize\}', r'\\text\{maximize\}',
                r'min_\{', r'max_\{',
            ],
        },
        "calculus": {
            "keywords": [
                "integral", "derivative", "limit", "dx", "dy", "dt",
                "differentiate", "integrate",
            ],
            "patterns": [
                r'\\int', r'\\lim', r'\\frac\{d', r'\\partial',
                r'\\nabla', r'\\sum_\{', r'\\prod_\{',
                r'∫', r'∂', r'∑', r'∏',
            ],
        },
        "linear_algebra": {
            "keywords": [
                "matrix", "vector", "eigenvalue", "eigenvector", "determinant",
                "transpose", "rank", "trace", "orthogonal", "basis",
                "span", "kernel", "null space", "column space",
            ],
            "patterns": [
                r'\\mathbf\{[A-Z]\}', r'\\vec\{', r'\\begin\{bmatrix\}',
                r'\\begin\{pmatrix\}', r'\\det', r'\\text\{rank\}',
                r'\^T', r'\^\{-1\}', r'\\top',
            ],
        },
        "probability": {
            "keywords": [
                "probability", "expectation", "variance", "distribution",
                "random variable", "conditional", "bayes",
                "prior", "posterior", "likelihood",
            ],
            "patterns": [
                r'P\(', r'E\[', r'\\mathbb\{E\}', r'\\mathbb\{P\}',
                r'\\text\{Var\}', r'\\text\{Cov\}',
                r'\\sim', r'\\mid',
            ],
        },
        "differential_equations": {
            "keywords": [
                "differential equation", "ode", "pde", "boundary condition",
                "initial condition", "solution", "steady state",
            ],
            "patterns": [
                r"\\frac\{d.*\}\{d[xtyz]\}",
                r"\\frac\{\\partial", r"y'", r"y''",
                r"\\dot\{", r"\\ddot\{",
            ],
        },
        "statistics": {
            "keywords": [
                "mean", "median", "standard deviation", "regression",
                "hypothesis", "confidence interval", "p-value",
                "significance", "correlation", "estimator",
            ],
            "patterns": [
                r'\\bar\{', r'\\hat\{', r'\\tilde\{',
                r'\\text\{MSE\}', r'\\text\{RMSE\}',
                r'R\^2', r'\\chi\^2',
            ],
        },
        "algebra": {
            "keywords": [
                "group", "ring", "field", "homomorphism", "isomorphism",
                "polynomial", "root", "factorization",
            ],
            "patterns": [
                r'\\cong', r'\\simeq', r'\\equiv',
                r'\\oplus', r'\\otimes',
            ],
        },
        "number_theory": {
            "keywords": [
                "prime", "divisible", "congruence", "modular",
                "diophantine", "gcd", "lcm",
            ],
            "patterns": [
                r'\\mod', r'\\pmod', r'\\gcd',
                r'\\equiv\s*.*\\pmod',
            ],
        },
        "geometry": {
            "keywords": [
                "angle", "triangle", "circle", "distance", "area",
                "volume", "manifold", "curvature", "metric",
            ],
            "patterns": [
                r'\\angle', r'\\triangle', r'\\perp', r'\\parallel',
            ],
        },
    }

    def classify(self, data: Dict) -> Dict:
        """
        Classify all equations in the paper.
        
        Returns:
            Updated data with 'equation_analysis' field
        """
        print("[Phase 5] Classifying equations...")
        
        equations = data.get("equations", [])
        math_exprs = data.get("all_math_expressions", [])
        
        # Combine all equation sources
        all_equations = []
        for eq in equations:
            all_equations.append(eq.get("text", ""))
        for expr in math_exprs:
            if expr not in [e.get("text", "") for e in equations]:
                all_equations.append(expr)
        
        if not all_equations:
            print("  ⚠ No equations found to classify")
            data["equation_analysis"] = {
                "classified": [],
                "type_distribution": {},
                "total": 0,
            }
            return data
        
        # Classify each equation
        classified = []
        for eq_text in all_equations:
            eq_type, confidence, matched_rules = self._classify_equation(eq_text)
            classified.append({
                "equation": eq_text[:200],
                "type": eq_type,
                "confidence": confidence,
                "matched_rules": matched_rules,
            })
        
        # Type distribution
        type_counts = Counter(c["type"] for c in classified)
        total = len(classified)
        distribution = {
            k: {"count": v, "percentage": round(v / total * 100, 1)}
            for k, v in type_counts.most_common()
        }
        
        data["equation_analysis"] = {
            "classified": classified,
            "type_distribution": distribution,
            "total": total,
            "dominant_type": type_counts.most_common(1)[0][0] if type_counts else "unknown",
        }
        
        print(f"  ✓ Classified {total} equations")
        print(f"  ✓ Distribution: {dict(type_counts.most_common(5))}")
        print(f"  ✓ Dominant type: {data['equation_analysis']['dominant_type']}")
        
        return data

    def _classify_equation(self, eq_text: str) -> tuple:
        """
        Classify a single equation.
        Returns (type, confidence, matched_rules).
        """
        scores = {}
        matches = {}
        
        for eq_type, rules in self.CLASSIFICATION_RULES.items():
            score = 0
            matched = []
            
            # Check keywords
            eq_lower = eq_text.lower()
            for kw in rules["keywords"]:
                if kw in eq_lower:
                    score += 1
                    matched.append(f"keyword:{kw}")
            
            # Check patterns
            for pattern in rules["patterns"]:
                if re.search(pattern, eq_text):
                    score += 1.5  # Patterns get higher weight
                    matched.append(f"pattern:{pattern}")
            
            if score > 0:
                scores[eq_type] = score
                matches[eq_type] = matched
        
        if not scores:
            return "other", 0.0, []
        
        best_type = max(scores, key=scores.get)
        max_score = scores[best_type]
        total_score = sum(scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.0
        
        return best_type, round(confidence, 3), matches.get(best_type, [])


if __name__ == "__main__":
    classifier = EquationClassifier()
    test = {
        "equations": [
            {"text": "\\min_{x} f(x) \\text{ subject to } g(x) \\leq 0"},
            {"text": "\\int_0^1 f(x) dx = F(1) - F(0)"},
            {"text": "P(A|B) = \\frac{P(B|A)P(A)}{P(B)}"},
            {"text": "Ax = \\lambda x"},
        ],
        "all_math_expressions": [],
    }
    result = classifier.classify(test)
    for eq in result["equation_analysis"]["classified"]:
        print(f"  {eq['equation'][:50]} → {eq['type']} ({eq['confidence']:.2f})")
