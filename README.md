# Simple Self-Distillation

<p align="center">
  <img src="figures/fig_teaser.png" width="100%" alt="SSD Overview">
</p>

This repo is for paper reproduction for improving code generation in three steps: (1) **Sample** solutions from the frozen model at non-unit temperature, (2) **Fine-tune** on those raw, unverified outputs via standard cross-entropy, and (3) **Decode** at a separately tuned temperature. No rewards, no verifier, no teacher, no RL. See the [paper](https://arxiv.org/abs/TBD) for full details.

## 🚀 Getting Started

```bash
git clone https://github.com/apple/ml-ssd.git
cd ml-ssd
uv sync --group evaluation
```


## 📁 Repository Structure

```
├── evaluation/
│   ├── eval.py                  # CLI entry point
│   ├── benchmark.py             # LiveCodeBench v6 implementation
│   └── livecodebench_utils.py   # Code execution utilities
├── figures/
│   └── fig_teaser.png
├── pyproject.toml
└── README.md
```


