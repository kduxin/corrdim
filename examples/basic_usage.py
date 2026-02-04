"""
Basic usage examples for the corrdim library.

This script demonstrates the main functionality of the correlation dimension
library for analyzing text complexity and detecting degeneration.
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import corrdim
sys.path.insert(0, str(Path(__file__).parent.parent))

from corrdim import (
    CorrelationDimensionCalculator,
    TextAnalyzer,
    detect_degeneration,
    analyze_text_complexity,
    DegenerationType
)


def example_basic_computation():
    """Example of basic correlation dimension computation."""
    print("=== Basic Correlation Dimension Computation ===")
    
    # Sample texts with different characteristics
    texts = {
        "Normal text": "The quick brown fox jumps over the lazy dog. This is a normal sentence with varied vocabulary and structure.",
        "Repetitive text": "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.",
        "Simple text": "The cat sat. The dog ran. The bird flew. Simple sentences with basic structure.",
        "Complex text": "The intricate mechanisms underlying quantum field theory demonstrate remarkable complexity in their mathematical formulations, revealing profound connections between fundamental particles and the fabric of spacetime itself."
    }
    
    # Initialize calculator
    calc = CorrelationDimensionCalculator("gpt2")
    
    print("Computing correlation dimensions...")
    for name, text in texts.items():
        corr_dim = calc.compute_correlation_dimension(text)
        print(f"{name}: {corr_dim:.4f}")
    
    print()


def example_degeneration_detection():
    """Example of degeneration detection."""
    print("=== Degeneration Detection ===")
    
    # Sample texts representing different types of degeneration
    test_texts = {
        "Normal": "The quick brown fox jumps over the lazy dog. This is a normal sentence with varied vocabulary and structure.",
        "Repetitive": "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.",
        "Bland": "The thing is good. The thing is nice. The thing is okay. The thing is fine. The thing is okay.",
        "Incoherent": "Purple elephant dancing mathematics quantum soup refrigerator philosophy banana telephone gravity"
    }
    
    for name, text in test_texts.items():
        degeneration_type = detect_degeneration(text, "gpt2")
        corr_dim = analyze_text_complexity(text, "gpt2")
        print(f"{name}: {degeneration_type.value} (correlation dimension: {corr_dim:.4f})")
    
    print()


def example_detailed_analysis():
    """Example of detailed text analysis."""
    print("=== Detailed Text Analysis ===")
    
    text = "The intricate mechanisms underlying quantum field theory demonstrate remarkable complexity in their mathematical formulations, revealing profound connections between fundamental particles and the fabric of spacetime itself."
    
    analyzer = TextAnalyzer("gpt2")
    result = analyzer.analyze_complexity(text, return_details=True)
    
    print(f"Text: {text[:50]}...")
    print(f"Correlation dimension: {result['correlation_dimension']:.4f}")
    print(f"Complexity level: {result['complexity_level']}")
    print(f"Text length: {result['text_length']} characters")
    print(f"Epsilon range: {result['epsilon_values'][0]:.6f} - {result['epsilon_values'][-1]:.6f}")
    print()


def example_text_comparison():
    """Example of comparing multiple texts."""
    print("=== Text Comparison ===")
    
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.",
        "The intricate mechanisms underlying quantum field theory demonstrate remarkable complexity.",
        "Simple text with basic structure and vocabulary."
    ]
    
    labels = ["Normal", "Repetitive", "Complex", "Simple"]
    
    analyzer = TextAnalyzer("gpt2")
    results = analyzer.compare_texts(texts, labels)
    
    print("Comparison results:")
    for label, corr_dim in zip(results['labels'], results['correlation_dimensions']):
        print(f"  {label}: {corr_dim:.4f}")
    
    stats = results['statistics']
    print(f"\nStatistics:")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Std: {stats['std']:.4f}")
    print(f"  Min: {stats['min']:.4f}")
    print(f"  Max: {stats['max']:.4f}")
    print()


def example_correlation_integral_curve():
    """Example of computing correlation integral curve."""
    print("=== Correlation Integral Curve ===")
    
    text = "The quick brown fox jumps over the lazy dog. This is a normal sentence with varied vocabulary and structure."
    
    calc = CorrelationDimensionCalculator("gpt2")
    epsilon_values, correlation_integrals = calc.compute_correlation_integral_curve(text)
    
    print(f"Text: {text[:50]}...")
    print(f"Number of epsilon values: {len(epsilon_values)}")
    print(f"Epsilon range: {epsilon_values[0]:.6f} - {epsilon_values[-1]:.6f}")
    print(f"Correlation integral range: {correlation_integrals[0]:.6f} - {correlation_integrals[-1]:.6f}")
    
    # Show first few points
    print("\nFirst 5 points:")
    for i in range(min(5, len(epsilon_values))):
        print(f"  ε={epsilon_values[i]:.6f}, S(ε)={correlation_integrals[i]:.6f}")
    print()


if __name__ == "__main__":
    print("CorrDim Library Examples")
    print("========================")
    print()
    
    try:
        example_basic_computation()
        example_degeneration_detection()
        example_detailed_analysis()
        example_text_comparison()
        example_correlation_integral_curve()
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have the required dependencies installed:")
        print("  pip install torch transformers numpy scikit-learn tqdm")
