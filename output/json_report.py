"""
JSON Report Generator — Produce structured JSON output with all analysis results.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
from datetime import datetime
from config import RESULTS_DIR


class JSONReportGenerator:
    """Generate structured JSON report from pipeline results."""

    def generate(self, data: Dict) -> Dict:
        """
        Generate a comprehensive JSON report.
        
        Args:
            data: Complete pipeline data
            
        Returns:
            Structured report dict (also saved to file)
        """
        print("[Phase 7] Generating JSON report...")
        
        report = {
            "metadata": {
                "title": data.get("title", "Unknown"),
                "authors": data.get("authors", []),
                "generated_at": datetime.now().isoformat(),
                "pipeline_version": "1.0.0",
                "source_pages": data.get("metadata", {}).get("num_pages", 0),
                "total_chars": data.get("metadata", {}).get("total_chars", 0),
            },
            "abstract": data.get("abstract", ""),
            "structure": {
                "num_sections": len(data.get("nl_sections", data.get("sections", []))),
                "sections": [
                    {
                        "title": s.get("title", ""),
                        "word_count": len(s.get("nl_text", s.get("content", "")).split()),
                        "math_density": s.get("math_density", 0),
                    }
                    for s in data.get("nl_sections", data.get("sections", []))
                ],
            },
            "mathematics": {
                "total_equations": data.get("metadata", {}).get("num_equations", 0),
                "math_expressions": len(data.get("all_math_expressions", [])),
                "equation_types": data.get("equation_analysis", {}).get("type_distribution", {}),
                "dominant_type": data.get("equation_analysis", {}).get("dominant_type", "unknown"),
                "math_structures": data.get("math_structures", {}).get("summary", {}),
            },
            "insights": {
                "keywords": data.get("keywords", {}).get("top_keywords", []),
                "entities": {
                    "persons": [e["text"] for e in data.get("entities", {}).get("persons", [])],
                    "methods": [m["name"] for m in data.get("entities", {}).get("methods", [])],
                    "concepts": data.get("entities", {}).get("concepts", []),
                    "math_entities": [
                        e["name"] for e in data.get("entities", {}).get("math_entities", [])
                    ],
                },
            },
            "topics": {
                "model": data.get("topics", {}).get("model", "none"),
                "num_topics": data.get("topics", {}).get("num_topics", 0),
                "details": data.get("topics", {}).get("topic_details", []),
            },
            "classifications": data.get("classifications", []),
            "summary": {
                "extractive": data.get("summary", {}).get("extractive", []),
                "abstractive": data.get("summary", {}).get("abstractive", ""),
            },
            "citations": {
                "total": data.get("citation_analysis", {}).get("total_citations", 0),
                "unique_references": data.get("citation_analysis", {}).get("unique_references", 0),
                "top_cited": data.get("citation_analysis", {}).get("ranked_references", [])[:5],
                "citation_density": data.get("citation_analysis", {}).get("citation_density", 0),
                "purpose_distribution": data.get("citation_analysis", {}).get("purpose_distribution", {}),
            },
            "similarity": {
                "most_similar_sections": data.get("similarity", {}).get("most_similar", []),
                "least_similar_sections": data.get("similarity", {}).get("least_similar", []),
            },
            "llm_analysis": data.get("llm_analysis", {}),
        }
        
        # Save to file
        output_path = os.path.join(RESULTS_DIR, "report.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"  ✓ JSON report saved: {output_path}")
        print(f"  ✓ Report size: {os.path.getsize(output_path) / 1024:.1f} KB")
        
        data["json_report"] = report
        data["json_report_path"] = output_path
        
        return data


if __name__ == "__main__":
    gen = JSONReportGenerator()
    test_data = {"title": "Test", "authors": ["A"], "abstract": "Test abstract"}
    gen.generate(test_data)
