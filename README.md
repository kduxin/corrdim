# Correlation Dimension of Autoregressive LLMs

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2510.21258)
[![NeurIPS](https://img.shields.io/badge/NeurIPS-2025-blue)](https://neurips.cc/virtual/2025/loc/san-diego/poster/116508)
[![Homepage](https://img.shields.io/badge/Homepage-ML%20Lab-green)](https://ml-waseda.jp)

> **A fractal-geometric approach to quantifying the epistemological complexity of text as perceived by language models.**

This repository contains the implementation for computing correlation dimension, a measure that bridges local and global properties of text generation in autoregressive language models. Unlike perplexity, correlation dimension captures long-range structural complexity and self-similarity patterns, revealing insights into model behavior, hallucination tendencies, and various forms of text degeneration.

## Quick Links

- 📄 [Paper](https://arxiv.org/abs/2510.21258) - Full paper on arXiv
- 🎯 [NeurIPS 2025](https://neurips.cc/virtual/2025/loc/san-diego/poster/116508) - Conference page
- 🏠 [Lab @ Waseda](https://ml-waseda.jp) - Our research group

## Features

- Efficient computation using next-token log-probability vectors
- Robust to model quantization (down to 4-bit precision or more)
- Applicable across autoregressive architectures (Transformer, Mamba, etc.)
- Real-time inference integration

---

**Code release coming soon.** 🚀
