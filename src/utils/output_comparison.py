# src/utils/output_comparison.py
"""Utilities for comparing LLM outputs."""
from typing import Dict, List, Tuple, Any
from difflib import SequenceMatcher, unified_diff
import re
from collections import Counter
import nltk
from nltk.metrics import edit_distance
import json


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity score between two texts (0-1)."""
    return SequenceMatcher(None, text1, text2).ratio()


def extract_key_points(text: str) -> List[str]:
    """Extract key points from text."""
    # Simple extraction based on common patterns
    patterns = [
        r'^\d+\.\s+(.+)$',  # Numbered lists
        r'^-\s+(.+)$',      # Bullet points
        r'^•\s+(.+)$',      # Bullet points
        r'^\*\s+(.+)$',     # Asterisk points
    ]
    
    key_points = []
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        for pattern in patterns:
            match = re.match(pattern, line, re.MULTILINE)
            if match:
                key_points.append(match.group(1).strip())
                break
    
    return key_points


def compare_key_points(points1: List[str], points2: List[str]) -> Dict[str, Any]:
    """Compare key points between two outputs."""
    set1 = set(points1)
    set2 = set(points2)
    
    common = set1.intersection(set2)
    only_in_1 = set1 - set2
    only_in_2 = set2 - set1
    
    return {
        "common_points": list(common),
        "unique_to_first": list(only_in_1),
        "unique_to_second": list(only_in_2),
        "overlap_ratio": len(common) / max(len(set1), len(set2)) if max(len(set1), len(set2)) > 0 else 0,
    }


def analyze_structure(text: str) -> Dict[str, Any]:
    """Analyze the structure of the text."""
    lines = text.split('\n')
    
    return {
        "total_lines": len(lines),
        "paragraphs": len([l for l in lines if l.strip() and not l.strip().startswith(('-', '*', '•', '#'))]),
        "headers": len([l for l in lines if l.strip().startswith('#')]),
        "list_items": len([l for l in lines if re.match(r'^[\-\*•]\s+', l.strip())]),
        "numbered_items": len([l for l in lines if re.match(r'^\d+\.\s+', l.strip())]),
        "word_count": len(text.split()),
        "char_count": len(text),
    }


def semantic_comparison(text1: str, text2: str) -> Dict[str, Any]:
    """Perform semantic comparison between two texts."""
    # Word frequency comparison
    words1 = re.findall(r'\b\w+\b', text1.lower())
    words2 = re.findall(r'\b\w+\b', text2.lower())
    
    freq1 = Counter(words1)
    freq2 = Counter(words2)
    
    # Common words (excluding very common ones)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
    
    significant_words1 = {w for w in freq1 if w not in stop_words and len(w) > 3}
    significant_words2 = {w for w in freq2 if w not in stop_words and len(w) > 3}
    
    common_significant = significant_words1.intersection(significant_words2)
    
    return {
        "common_significant_words": list(common_significant)[:20],  # Top 20
        "vocabulary_overlap": len(common_significant) / max(len(significant_words1), len(significant_words2)) if max(len(significant_words1), len(significant_words2)) > 0 else 0,
        "unique_words_text1": len(significant_words1 - significant_words2),
        "unique_words_text2": len(significant_words2 - significant_words1),
    }


def quality_metrics(text: str) -> Dict[str, Any]:
    """Calculate quality metrics for text."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    words = text.split()
    
    # Average sentence length
    avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
    
    # Readability proxy (simple version)
    complex_words = [w for w in words if len(w) > 6]
    complexity_ratio = len(complex_words) / len(words) if words else 0
    
    return {
        "sentence_count": len(sentences),
        "avg_sentence_length": avg_sentence_length,
        "complexity_ratio": complexity_ratio,
        "question_marks": text.count('?'),
        "exclamations": text.count('!'),
    }


def compare_outputs(output1: Dict[str, Any], output2: Dict[str, Any], model1_name: str = "Model A", model2_name: str = "Model B") -> Dict[str, Any]:
    """Comprehensive comparison of two model outputs."""
    # Extract the article content
    article1 = output1.get("final_article", "")
    article2 = output2.get("final_article", "")
    
    # Basic similarity
    similarity = calculate_similarity(article1, article2)
    
    # Extract and compare key points
    points1 = extract_key_points(article1)
    points2 = extract_key_points(article2)
    points_comparison = compare_key_points(points1, points2)
    
    # Structure analysis
    structure1 = analyze_structure(article1)
    structure2 = analyze_structure(article2)
    
    # Semantic comparison
    semantic = semantic_comparison(article1, article2)
    
    # Quality metrics
    quality1 = quality_metrics(article1)
    quality2 = quality_metrics(article2)
    
    # Source comparison
    sources1 = set(output1.get("sources", []))
    sources2 = set(output2.get("sources", []))
    common_sources = sources1.intersection(sources2)
    
    return {
        "models": {
            "model_1": model1_name,
            "model_2": model2_name,
        },
        "similarity": {
            "overall_similarity": similarity,
            "key_points_overlap": points_comparison["overlap_ratio"],
            "vocabulary_overlap": semantic["vocabulary_overlap"],
            "source_overlap": len(common_sources) / max(len(sources1), len(sources2)) if max(len(sources1), len(sources2)) > 0 else 0,
        },
        "content_analysis": {
            "key_points": points_comparison,
            "semantic": semantic,
        },
        "structure": {
            model1_name: structure1,
            model2_name: structure2,
            "differences": {
                "word_count_diff": abs(structure1["word_count"] - structure2["word_count"]),
                "paragraph_diff": abs(structure1["paragraphs"] - structure2["paragraphs"]),
            },
        },
        "quality": {
            model1_name: quality1,
            model2_name: quality2,
        },
        "sources": {
            "common": list(common_sources),
            "unique_to_model1": list(sources1 - sources2),
            "unique_to_model2": list(sources2 - sources1),
        },
    }


def generate_diff(text1: str, text2: str, model1_name: str = "Model A", model2_name: str = "Model B") -> str:
    """Generate a unified diff between two texts."""
    lines1 = text1.splitlines(keepends=True)
    lines2 = text2.splitlines(keepends=True)
    
    diff = unified_diff(
        lines1, 
        lines2, 
        fromfile=f"{model1_name} output",
        tofile=f"{model2_name} output",
        lineterm=''
    )
    
    return ''.join(diff)


def print_comparison_summary(comparison: Dict[str, Any]) -> None:
    """Print a formatted summary of the comparison."""
    print("\n" + "="*60)
    print(f"MODEL COMPARISON: {comparison['models']['model_1']} vs {comparison['models']['model_2']}")
    print("="*60)
    
    print("\nSIMILARITY METRICS:")
    for key, value in comparison['similarity'].items():
        print(f"  {key.replace('_', ' ').title()}: {value:.2%}")
    
    print("\nSTRUCTURE COMPARISON:")
    struct_diff = comparison['structure']['differences']
    print(f"  Word count difference: {struct_diff['word_count_diff']} words")
    print(f"  Paragraph difference: {struct_diff['paragraph_diff']} paragraphs")
    
    print("\nKEY POINTS ANALYSIS:")
    kp = comparison['content_analysis']['key_points']
    print(f"  Common points: {len(kp['common_points'])}")
    print(f"  Unique to {comparison['models']['model_1']}: {len(kp['unique_to_first'])}")
    print(f"  Unique to {comparison['models']['model_2']}: {len(kp['unique_to_second'])}")
    
    print("\nSOURCE COMPARISON:")
    sources = comparison['sources']
    print(f"  Common sources: {len(sources['common'])}")
    print(f"  Unique to {comparison['models']['model_1']}: {len(sources['unique_to_model1'])}")
    print(f"  Unique to {comparison['models']['model_2']}: {len(sources['unique_to_model2'])}")
    
    print("="*60)