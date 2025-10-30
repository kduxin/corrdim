"""
Command-line interface for the corrdim library.
"""

import argparse
import sys

from .core import CorrelationDimensionCalculator


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CorrDim: Compute Correlation Dimension for Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute correlation dimension for a text file
  corrdim text.txt --model gpt2
  
  # Use a different model
  corrdim text.txt --model microsoft/DialoGPT-medium
  
  # Adjust context length (stride is auto by default)
  corrdim text.txt --model gpt2 --context-length 1024
  
  # Save results to file
  corrdim text.txt --model gpt2 --output results.txt
        """
    )
    
    # Required arguments
    parser.add_argument('file', help='Text file to analyze')
    parser.add_argument('--model', '-m', required=True, 
                       help='Model name for from_pretrained (e.g., gpt2, microsoft/DialoGPT-medium)')
    parser.add_argument('--line-sep', type=str, default="\n")
    
    # Optional parameters
    parser.add_argument('--max-length', type=int, default=16384,
                       help='Maximum length of the text (in tokens) to process (default: 16384)')
    parser.add_argument('--context-length', type=int, default=None,
                       help='Context length for the model (default: None, let model decide)')
    parser.add_argument('--stride', type=str, default="auto",
                       help='Stride for text processing (default: auto)')
    parser.add_argument('--dim-reduction', type=int, default=None,
                       help='Vocabulary dimension reduction (default: None)')
    parser.add_argument('--dtype', type=str, default="float16",
                       help='Data type for the model (default: float16)')

    # Output options
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        compute_correlation_dimension(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def compute_correlation_dimension(args):
    """Compute correlation dimension for a text file."""
    # Read text file
    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
    except Exception as e:
        raise ValueError(f"Error reading file {args.file}: {e}")

    text = args.line_sep.join(text.split('\n'))
    
    if not text:
        raise ValueError(f"File {args.file} is empty")
    
    if args.verbose:
        print(f"Computing correlation dimension for: {args.file}")
        print(f"Text length: {len(text)} characters")
        print(f"Model: {args.model}")
        print(f"Context length: {args.context_length if args.context_length else 'auto (model default)'}")
        print(f"Stride: {args.stride}")
        if args.dim_reduction:
            print(f"Dimension reduction: {args.dim_reduction}")
        print()
    
    # Initialize calculator
    from .models import TransformersModelWrapper
    
    # Create model wrapper
    model_wrapper = TransformersModelWrapper(
        model_name=args.model,
        dtype=args.dtype,
    )

    tokens = model_wrapper.encode(text, add_special_tokens=False)
    if len(tokens) > args.max_length:
        text = model_wrapper.decode(tokens[:args.max_length])
    else:
        text = model_wrapper.decode(tokens)

    # Initialize calculator
    calc = CorrelationDimensionCalculator(
        model=model_wrapper
    )
    
    # Compute correlation integral curve first
    curve = calc.compute_correlation_integral_curve(
        text=text,
        context_length=args.context_length,
        stride=args.stride,
        dim_reduction=args.dim_reduction,
        epsilon_range=(1e-20, 1e20),  # Fixed range for optimal results
        num_epsilon=10000  # Fixed number for high precision
    )
    
    # Then compute correlation dimension from the curve
    result = calc.compute_correlation_dimension_from_curve(
        epsilons=curve.epsilons,
        corrints=curve.corrints,
        sequence_length=curve.sequence_length
    )
    
    # Format output
    output = f"""Correlation Dimension Analysis
============================
File: {args.file}
Model: {args.model}
Text length: {len(text)} characters

Correlation dimension: {result.corrdim:.4f}
R² (fit quality): {result.fit_r2:.3f}
Sequence length: {result.sequence_length} tokens

Parameters:
- Context length: {args.context_length if args.context_length else 'auto (model default)'}
- Stride: {args.stride}
- Epsilon range: [1e-20, 1e20] (fixed for optimal results)
- Number of epsilon values: 10000 (fixed for high precision)
"""
    
    if args.dim_reduction:
        output += f"- Dimension reduction: {args.dim_reduction}\n"
    
    # Output results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Results saved to {args.output}")
    else:
        print(output)


if __name__ == '__main__':
    main()
