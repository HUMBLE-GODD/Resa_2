"""
Final Report Generator — Compile all results into a comprehensive markdown report.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
from datetime import datetime
from config import RESULTS_DIR


class FinalReportGenerator:
    """Generate a comprehensive markdown report from all pipeline results."""

    def generate(self, data: Dict) -> str:
        """Generate the final markdown report."""
        print("[Phase 7] Generating final report...")
        
        title = data.get("title", "Unknown Paper")
        authors = ", ".join(data.get("authors", ["Unknown"]))
        
        sections = []
        
        # Header
        sections.append(f"# Research Paper Analysis Report\n")
        sections.append(f"**Paper:** {title}  ")
        sections.append(f"**Authors:** {authors}  ")
        sections.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
        sections.append(f"**Pipeline Version:** 1.0.0\n")
        sections.append("---\n")
        
        # Abstract
        abstract = data.get("abstract", "")
        if abstract:
            sections.append("## Abstract\n")
            sections.append(f"{abstract}\n")
        
        # Document Structure
        meta = data.get("metadata", {})
        sections.append("## Document Structure\n")
        sections.append(f"- **Pages:** {meta.get('num_pages', 'N/A')}")
        sections.append(f"- **Sections:** {meta.get('num_sections', 'N/A')}")
        sections.append(f"- **Equations:** {meta.get('num_equations', 'N/A')}")
        sections.append(f"- **References:** {meta.get('num_references', 'N/A')}")
        sections.append(f"- **Total Characters:** {meta.get('total_chars', 'N/A')}\n")
        
        # Keywords
        keywords = data.get("keywords", {}).get("top_keywords", [])
        if keywords:
            sections.append("## Key Terms\n")
            sections.append(", ".join(f"**{kw}**" for kw in keywords[:15]))
            sections.append("")
        
        # Mathematical Analysis
        math_structures = data.get("math_structures", {})
        eq_analysis = data.get("equation_analysis", {})
        if math_structures or eq_analysis:
            sections.append("## Mathematical Analysis\n")
            
            summary = math_structures.get("summary", {})
            if summary:
                sections.append(f"**Logical Flow:** {summary.get('logical_flow', 'N/A')}\n")
                sections.append(f"| Structure | Count |")
                sections.append(f"|-----------|-------|")
                for key in ["total_definitions", "total_theorems", "total_lemmas", "total_propositions", "total_corollaries", "total_proofs"]:
                    if summary.get(key, 0) > 0:
                        name = key.replace("total_", "").title()
                        sections.append(f"| {name} | {summary[key]} |")
                sections.append("")
            
            if eq_analysis.get("dominant_type"):
                sections.append(f"**Dominant Equation Type:** {eq_analysis['dominant_type']}")
                sections.append(f"**Total Equations Classified:** {eq_analysis.get('total', 0)}\n")
        
        # Topic Analysis
        topics = data.get("topics", {})
        if topics.get("topic_details"):
            sections.append("## Discovered Topics\n")
            for t in topics["topic_details"][:5]:
                keywords_str = ", ".join(t.get("keywords", [])[:5])
                sections.append(f"- **Topic {t['id']}** ({t['count']} sentences): {keywords_str}")
            sections.append("")
        
        # Section Classifications
        classifications = data.get("classifications", [])
        if classifications:
            sections.append("## Section Classification\n")
            sections.append("| Section | Predicted Type | Confidence |")
            sections.append("|---------|---------------|------------|")
            for c in classifications:
                sections.append(f"| {c['section_title'][:40]} | {c['predicted_label']} | {c['confidence']:.1%} |")
            sections.append("")
        
        # Summary
        summary_data = data.get("summary", {})
        if summary_data:
            sections.append("## Summary\n")
            
            if summary_data.get("abstractive"):
                sections.append("### AI-Generated Summary\n")
                sections.append(f"{summary_data['abstractive']}\n")
            
            if summary_data.get("extractive"):
                sections.append("### Key Sentences\n")
                for sent in summary_data["extractive"]:
                    sections.append(f"- {sent}")
                sections.append("")
        
        # Citation Analysis
        citations = data.get("citation_analysis", {})
        if citations:
            sections.append("## Citation Analysis\n")
            sections.append(f"- **Total Citations:** {citations.get('total_citations', 0)}")
            sections.append(f"- **Unique References:** {citations.get('unique_references', 0)}")
            sections.append(f"- **Citation Density:** {citations.get('citation_density', 0):.2f} per 100 words")
            sections.append(f"- **Self-Citations:** {citations.get('self_citation_count', 0)}\n")
            
            ranked = citations.get("ranked_references", [])
            if ranked:
                sections.append("### Most Cited References\n")
                for ref in ranked[:5]:
                    sections.append(f"- **[{ref['id']}]** ({ref['count']}x): {ref['text'][:100]}")
                sections.append("")
        
        # LLM Analysis
        llm = data.get("llm_analysis", {})
        if llm:
            sections.append("## AI-Powered Analysis (Groq)\n")
            
            if llm.get("summary"):
                sections.append("### Comprehensive Summary\n")
                sections.append(f"{llm['summary']}\n")
            
            if llm.get("eli5"):
                sections.append("### Simple Explanation (ELI5)\n")
                sections.append(f"{llm['eli5']}\n")
            
            if llm.get("contributions"):
                sections.append("### Research Contributions\n")
                sections.append(f"{llm['contributions']}\n")
            
            if llm.get("applications"):
                sections.append("### Potential Applications\n")
                sections.append(f"{llm['applications']}\n")
            
            if llm.get("limitations"):
                sections.append("### Limitations\n")
                sections.append(f"{llm['limitations']}\n")
        
        # Visualizations
        viz = data.get("visualizations", {})
        if viz:
            sections.append("## Visualizations\n")
            for name, path in viz.items():
                sections.append(f"- **{name.replace('_', ' ').title()}**: `{path}`")
            sections.append("")
        
        # Footer
        sections.append("---\n")
        sections.append("*Report generated by RESA_AI Math Paper NLP Pipeline v1.0.0*\n")
        
        report_text = "\n".join(sections)
        
        # Save
        output_path = os.path.join(RESULTS_DIR, "final_report.md")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        data["final_report_path"] = output_path
        print(f"  ✓ Final report saved: {output_path} ({len(report_text)} chars)")
        
        return report_text


if __name__ == "__main__":
    gen = FinalReportGenerator()
    test = {"title": "Test", "authors": ["A"], "abstract": "Test", "metadata": {"num_pages": 5}}
    gen.generate(test)
