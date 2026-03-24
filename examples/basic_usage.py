"""
Basic usage examples for the corrdim library.

Structure matches examples/basic_usage.ipynb:
  - Example 1: High-level interface (one-line computation, multiple articles, plot)
  - Example 2: Low-level interface (log-probabilities -> curve_from_vectors -> corrdim)
  - Example 3: Correlation dimension changes along the text (progressive_correlation_integral)

Each example runs in a try/except; failures are reported and the script continues.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import corrdim
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import matplotlib.pyplot as plt
import corrdim

torch.set_grad_enabled(False)  # We don't need gradients for inference

# Paths
REPO_ROOT = Path(__file__).parent.parent
EXAMPLES_DIR = Path(__file__).parent
PLOTS_DIR = EXAMPLES_DIR / "plots"
DATA_DIR = REPO_ROOT / "data" / "sep60"
CHAOS_PATH = DATA_DIR / "chaos.txt"


def _save_fig(fig, filename: str) -> Path:
    """Ensure plots dir exists, save figure, print path, return path."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    print(f"  Plot saved: {path.resolve()}")
    return path


def example1_high_level():
    """
    Example 1: High-Level Interface
    The simplest way to compute correlation dimension - just one line of code!
    """
    # Step 1: Load the texts (Stanford Encyclopedia of Philosophy dataset)
    texts = {}
    if DATA_DIR.is_dir():
        for filename in os.listdir(DATA_DIR):
            filepath = DATA_DIR / filename
            if filepath.is_file():
                with open(filepath, "r") as f:
                    lines = [line.strip() for line in f]
                    texts[filename] = " ".join(lines)
        print(f"Loaded {len(texts)} articles")
    else:
        print(f"Data dir not found: {DATA_DIR}. Skipping Example 1 text loading.")
        return

    # Step 2: Compute correlation dimensions
    corrdim.set_corrint_backend("triton")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=150)
    max_articles = 6

    for i, (filename, text) in enumerate(texts.items()):
        if i >= max_articles:
            break
        result = corrdim.measure_text(
            text,
            model="Qwen/Qwen2.5-1.5B",
            precision=torch.float16,
            truncation_tokens=10000,
            dim_reduction=8192,
        )
        ax.plot(
            result.epsilons_linear_region,
            result.corrints_linear_region,
            "-o",
            label=f"{filename}: corrdim = {result.corrdim:.2f}",
        )
        print(f"File = {filename}, epsilon range = {result.epsilons_linear_region[0]:g} - {result.epsilons_linear_region[-1]:g}: corrdim = {result.corrdim:.2f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ε (epsilon)")
    ax.set_ylabel("Correlation Integral")
    ax.set_title("Correlation Dimension of SEP Articles")
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, "example1_sep_articles.png")
    plt.show()


def example2_low_level():
    """
    Example 2: Low-Level Interface
    Compute correlation dimension from log-probabilities directly.
    """
    import transformers

    # Step 1: Load model and text
    model_name = "Qwen/Qwen2.5-1.5B"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
    ).to("cuda")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    if CHAOS_PATH.is_file():
        with open(CHAOS_PATH, "r") as f:
            lines = [line.strip() for line in f]
            text = " ".join(lines)
    else:
        text = "The quick brown fox jumps over the lazy dog. " * 50
        print(f"File not found: {CHAOS_PATH}. Using fallback text.")

    # Step 2: Compute log-probabilities (3000 tokens to manage memory)
    max_tokens = 3000
    inputs = tokenizer(
        text, max_length=max_tokens, truncation=True, return_tensors="pt"
    ).to("cuda")
    outputs = model(**inputs)
    logits = outputs.logits[0]
    logprobs = logits.log_softmax(-1)
    print(f"Log-probabilities shape: {logprobs.shape}")
    print(f"Vocabulary size: {logprobs.shape[1]}")
    logprobs = corrdim.reduce_dimension(logprobs, num_groups=8192)
    print(f"Reduced log-probabilities shape: {logprobs.shape}")

    # Step 3: Compute correlation integral
    epsilons = torch.logspace(-20, 20, 10000, device="cuda")
    curve = corrdim.curve_from_vectors(logprobs)

    # Step 4: Estimate correlation dimension from the curve
    result = corrdim.estimate_dimension_from_curve(
        curve=curve,
        correlation_integral_range="auto",
    )
    print(f"File = chaos, epsilon range = {result.epsilons_linear_region[0]:g} - {result.epsilons_linear_region[-1]:g}: corrdim = {result.corrdim:.2f}")

    fig, ax = plt.subplots(1, 1, figsize=(4, 3), dpi=150)
    ax.plot(
        result.epsilons_linear_region,
        result.corrints_linear_region,
        label=f"corrdim = {result.corrdim:.2f}",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ε (epsilon)")
    ax.set_ylabel("Correlation Integral")
    ax.set_title("Correlation Dimension from Log-Probabilities")
    ax.legend()
    plt.tight_layout()
    _save_fig(fig, "example2_log_probs.png")
    plt.show()



def example3_corrdim_along_text():
    """
    Example 3: Correlation dimension changes along the text
    Uses progressive_correlation_integral to get corrdim at each prefix.
    """
    import transformers

    # Step 1: Load model and text
    model_name = "Qwen/Qwen2.5-1.5B"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
    ).to("cuda")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    if CHAOS_PATH.is_file():
        with open(CHAOS_PATH, "r") as f:
            lines = [line.strip() for line in f]
            text = " ".join(lines)
    else:
        text = "The quick brown fox jumps over the lazy dog. " * 50

    # Step 2: Compute log-probabilities
    max_tokens = 10000
    inputs = tokenizer(
        text, max_length=max_tokens, truncation=True, return_tensors="pt"
    ).to("cuda")
    outputs = model(**inputs)
    logits = outputs.logits[0]
    logprobs = logits.log_softmax(-1)
    print(f"Log-probabilities shape: {logprobs.shape}")
    print(f"Vocabulary size: {logprobs.shape[1]}")
    logprobs = corrdim.reduce_dimension(logprobs, num_groups=8192)
    print(f"Reduced log-probabilities shape: {logprobs.shape}")

    # Step 3: Correlation integrals for all prefixes
    epsilons = torch.logspace(-20, 20, 10000, device="cuda")
    corrints_arr = corrdim.progressive_correlation_integral(logprobs, epsilons)

    # Step 4: Correlation dimension for each prefix (skip first 1000 tokens)
    fig, ax = plt.subplots(1, 1, figsize=(6, 3), dpi=150)
    epsilon_range = (500, 1200)
    print(f"Epsilon range: {epsilon_range}")
    xs = []
    ys = []
    for i in range(1000, max_tokens, 10):
        corrints = corrints_arr[i]
        result = corrdim.estimate_dimension_from_curve(
            corrdim.CurveResult(
                sequence_length=logprobs.shape[0],
                epsilons=epsilons.cpu().numpy(),
                corrints=corrints.cpu().numpy(),
            ),
            correlation_integral_range=None,
            epsilon_range=epsilon_range,
        )
        xs.append(i)
        ys.append(result.corrdim)

    ax.plot(xs, ys)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Correlation Dimension")
    ax.set_title("Correlation Dimension changes along the text")
    plt.tight_layout()
    _save_fig(fig, "example3_corrdim_along_text.png")
    plt.show()


def run_example(name: str, fn):
    """Run a single example; on failure print error and continue."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print('='*60)
    try:
        fn()
        print(f"  OK: {name}")
    except Exception as e:
        print(f"  SKIPPED: {name}")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("CorrDim examples (structure matches basic_usage.ipynb)")
    print("Failed examples are skipped; see errors above.\n")

    run_example("Example 1: High-level interface", example1_high_level)
    run_example("Example 2: Low-level interface", example2_low_level)
    run_example("Example 3: Corrdim along text", example3_corrdim_along_text)

    print("\nDone.")
