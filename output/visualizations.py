"""
Visualizations — Generate keyword graphs, topic clusters, and distribution charts.
Uses matplotlib for static visualization output.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict
from config import RESULTS_DIR

# Use non-interactive backend for server/CLI use
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


class Visualizer:
    """Generate publication-quality visualizations from pipeline results."""

    # Color palette
    COLORS = [
        '#2196F3', '#FF5722', '#4CAF50', '#FFC107', '#9C27B0',
        '#00BCD4', '#FF9800', '#795548', '#607D8B', '#E91E63',
    ]
    BG_COLOR = '#1a1a2e'
    TEXT_COLOR = '#e0e0e0'
    GRID_COLOR = '#333355'

    def generate_all(self, data: Dict) -> Dict:
        """Generate all visualizations."""
        print("[Phase 7] Generating visualizations...")
        
        viz_paths = {}
        
        # 1. Keyword frequency chart
        path = self._keyword_chart(data)
        if path:
            viz_paths["keyword_chart"] = path
        
        # 2. Equation type distribution
        path = self._equation_distribution(data)
        if path:
            viz_paths["equation_distribution"] = path
        
        # 3. Section classification results
        path = self._classification_chart(data)
        if path:
            viz_paths["classification_chart"] = path
        
        # 4. Citation analysis
        path = self._citation_chart(data)
        if path:
            viz_paths["citation_chart"] = path
        
        # 5. Topic word cloud style chart
        path = self._topic_chart(data)
        if path:
            viz_paths["topic_chart"] = path
        
        # 6. Math density per section
        path = self._math_density_chart(data)
        if path:
            viz_paths["math_density"] = path
        
        data["visualizations"] = viz_paths
        print(f"  ✓ Generated {len(viz_paths)} visualizations")
        
        return data

    def _setup_dark_theme(self, fig, ax):
        """Apply dark theme to matplotlib figure."""
        fig.patch.set_facecolor(self.BG_COLOR)
        ax.set_facecolor(self.BG_COLOR)
        ax.tick_params(colors=self.TEXT_COLOR)
        ax.xaxis.label.set_color(self.TEXT_COLOR)
        ax.yaxis.label.set_color(self.TEXT_COLOR)
        ax.title.set_color(self.TEXT_COLOR)
        for spine in ax.spines.values():
            spine.set_color(self.GRID_COLOR)

    def _keyword_chart(self, data: Dict) -> str:
        """Generate keyword frequency horizontal bar chart."""
        keywords = data.get("keywords", {}).get("combined", [])
        if not keywords:
            return ""
        
        # Top 15 keywords
        kw_data = keywords[:15]
        words = [k[0] for k in kw_data]
        scores = [k[1] for k in kw_data]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        self._setup_dark_theme(fig, ax)
        
        y_pos = np.arange(len(words))
        colors = [self.COLORS[i % len(self.COLORS)] for i in range(len(words))]
        
        bars = ax.barh(y_pos, scores, color=colors, alpha=0.85, height=0.7,
                       edgecolor='white', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel('Relevance Score', fontsize=12)
        ax.set_title('Top Keywords', fontsize=16, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.2, color=self.GRID_COLOR)
        
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, "keyword_chart.png")
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ Keyword chart saved: {path}")
        return path

    def _equation_distribution(self, data: Dict) -> str:
        """Generate equation type pie chart."""
        eq_analysis = data.get("equation_analysis", {})
        distribution = eq_analysis.get("type_distribution", {})
        
        if not distribution:
            return ""
        
        labels = list(distribution.keys())
        sizes = [d["count"] for d in distribution.values()]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        self._setup_dark_theme(fig, ax)
        
        colors = [self.COLORS[i % len(self.COLORS)] for i in range(len(labels))]
        
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct='%1.1f%%',
            colors=colors, startangle=90,
            textprops={'color': self.TEXT_COLOR, 'fontsize': 11},
            pctdistance=0.8, labeldistance=1.1,
            wedgeprops={'edgecolor': self.BG_COLOR, 'linewidth': 2},
        )
        
        for text in autotexts:
            text.set_fontweight('bold')
        
        ax.set_title('Equation Type Distribution', fontsize=16, fontweight='bold',
                     color=self.TEXT_COLOR, pad=20)
        
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, "equation_distribution.png")
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ Equation distribution chart saved: {path}")
        return path

    def _classification_chart(self, data: Dict) -> str:
        """Generate section classification bar chart."""
        classifications = data.get("classifications", [])
        if not classifications:
            return ""
        
        section_labels = [c["section_title"][:25] for c in classifications]
        predicted = [c["predicted_label"] for c in classifications]
        confidences = [c["confidence"] for c in classifications]
        
        fig, ax = plt.subplots(figsize=(14, 7))
        self._setup_dark_theme(fig, ax)
        
        x_pos = np.arange(len(section_labels))
        colors = [self.COLORS[hash(p) % len(self.COLORS)] for p in predicted]
        
        bars = ax.bar(x_pos, confidences, color=colors, alpha=0.85,
                     edgecolor='white', linewidth=0.5)
        
        # Add predicted label text on bars
        for bar, label in zip(bars, predicted):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   label, ha='center', va='bottom', fontsize=8, color=self.TEXT_COLOR,
                   rotation=45)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(section_labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Classification Confidence', fontsize=12)
        ax.set_title('Section Classification Results', fontsize=16, fontweight='bold', pad=15)
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.2, color=self.GRID_COLOR)
        
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, "classification_chart.png")
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ Classification chart saved: {path}")
        return path

    def _citation_chart(self, data: Dict) -> str:
        """Generate citation analysis chart."""
        citation_data = data.get("citation_analysis", {})
        ranked = citation_data.get("ranked_references", [])
        
        if not ranked:
            return ""
        
        top_refs = ranked[:10]
        labels = [f"Ref [{r['id']}]" for r in top_refs]
        counts = [r["count"] for r in top_refs]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        self._setup_dark_theme(fig, ax)
        
        x_pos = np.arange(len(labels))
        colors = [self.COLORS[i % len(self.COLORS)] for i in range(len(labels))]
        
        ax.bar(x_pos, counts, color=colors, alpha=0.85,
              edgecolor='white', linewidth=0.5)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel('Citation Frequency', fontsize=12)
        ax.set_title('Most Cited References', fontsize=16, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.2, color=self.GRID_COLOR)
        
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, "citation_chart.png")
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ Citation chart saved: {path}")
        return path

    def _topic_chart(self, data: Dict) -> str:
        """Generate topic distribution chart."""
        topics = data.get("topics", {}).get("topic_details", [])
        if not topics:
            return ""
        
        fig, ax = plt.subplots(figsize=(12, 7))
        self._setup_dark_theme(fig, ax)
        
        topic_names = [t.get("name", f"Topic {t['id']}")[:30] for t in topics[:8]]
        topic_counts = [t["count"] for t in topics[:8]]
        colors = [self.COLORS[i % len(self.COLORS)] for i in range(len(topic_names))]
        
        ax.barh(range(len(topic_names)), topic_counts, color=colors, alpha=0.85,
                height=0.6, edgecolor='white', linewidth=0.5)
        
        ax.set_yticks(range(len(topic_names)))
        ax.set_yticklabels(topic_names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Document Count', fontsize=12)
        ax.set_title('Topic Distribution', fontsize=16, fontweight='bold', pad=15)
        ax.grid(axis='x', alpha=0.2, color=self.GRID_COLOR)
        
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, "topic_chart.png")
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ Topic chart saved: {path}")
        return path

    def _math_density_chart(self, data: Dict) -> str:
        """Generate math density per section chart."""
        sections = data.get("nl_sections", [])
        if not sections:
            return ""
        
        names = [s.get("title", "?")[:20] for s in sections if s.get("math_density", 0) > 0 or True]
        densities = [s.get("math_density", 0) for s in sections]
        
        if not any(d > 0 for d in densities):
            return ""
        
        fig, ax = plt.subplots(figsize=(12, 6))
        self._setup_dark_theme(fig, ax)
        
        x_pos = np.arange(len(names))
        ax.bar(x_pos, densities, color='#00BCD4', alpha=0.85,
              edgecolor='white', linewidth=0.5)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Math Expression Density', fontsize=12)
        ax.set_title('Mathematical Density per Section', fontsize=16, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.2, color=self.GRID_COLOR)
        
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, "math_density.png")
        fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ Math density chart saved: {path}")
        return path


if __name__ == "__main__":
    viz = Visualizer()
    test = {
        "keywords": {"combined": [("neural", 0.9), ("network", 0.8), ("deep", 0.7)]},
        "equation_analysis": {"type_distribution": {"calculus": {"count": 5}, "algebra": {"count": 3}}},
        "classifications": [],
        "citation_analysis": {"ranked_references": []},
        "topics": {"topic_details": []},
    }
    viz.generate_all(test)
