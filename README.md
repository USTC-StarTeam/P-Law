# Optimizing Sequential Recommendation Models with Scaling Laws and Approximate Entropy

[![Project Page](https://img.shields.io/badge/Project-Page-2454d6.svg)](https://ustc-starteam.github.io/P-Law/)
[![ICML 2024](https://img.shields.io/badge/ICML-2024-4b6cb7.svg)](https://icml.cc/virtual/2024)
[![Python](https://img.shields.io/badge/Python-3.7%2B-3776ab.svg)](https://www.python.org/)

Official code for **"Optimizing Sequential Recommendation Models with Scaling Laws and Approximate Entropy"**.

This repository studies why the scaling-law view used for large language models does not directly transfer to sequential recommendation, and introduces a **Performance Law** that models HR/NDCG behavior using both model configuration and data quality. The codebase contains the entropy utility, the general transformer training code, and the appendix scripts/frameworks used to fit and visualize the reported laws.

## 1. Paper

Tingjia Shen, Hao Wang, Chuhan Wu, Jin Yao Chin, Wei Guo, Yong Liu, Huifeng Guo, Defu Lian, Ruiming Tang, and Enhong Chen. **Optimizing Sequential Recommendation Models with Scaling Laws and Approximate Entropy.** In *Proceedings of the 41st International Conference on Machine Learning (ICML 2024)*, PMLR 235, 2024.

[Paper](https://arxiv.org/abs/2412.00430) / [PDF](https://arxiv.org/pdf/2412.00430) / [Project Page](https://ustc-starteam.github.io/P-Law/) / [Citation](#citation)

P-Law studies how sequential recommendation performance changes with model scale and data quality. Instead of directly applying loss-oriented scaling laws from large language models, it fits HR/NDCG with a recommendation-specific Performance Law and uses Approximate Entropy to quantify sequence complexity.

## 2. Highlights

- Introduces **Performance Law** for sequential recommendation, modeling performance rather than only loss.
- Uses **Approximate Entropy (ApEn)** as a data-quality signal that better reflects sequence complexity than dataset size alone.
- Shows that recommendation performance can follow a U-shaped trend over model configuration and data quality, unlike a simple monotonic scaling-law curve.
- Provides reusable components for entropy estimation, transformer-based sequential recommendation experiments, and appendix fitting/visualization.

## 3. Method At A Glance

![Performance Law overview](docs/assets/performance-law-overview.png)

The paper contrasts conventional scaling-law behavior with the recommendation-specific Performance Law. Instead of expecting larger models or more data to monotonically improve performance, the fitted surfaces expose where HR/NDCG improve, saturate, or decay under different sequence lengths, model depths, embedding dimensions, and ApEn-derived data-quality parameters.

## 4. Repository Structure

```text
.
|-- PerformanceLaw/                         # Python package for actual entropy / ApEn-style sequence complexity
|-- General_Transformer/                    # Transformer training and evaluation code for sequential recommendation
|-- Performance_Law_Appendix_Result/        # Performance-law fitting scripts and generated appendix result figures
|-- Performance_Law_Appendix_Framework/     # Additional baseline/framework code used in appendix experiments
|-- docs/assets/                            # README figures cropped or converted from the paper/results
|-- .gitattributes                          # Git LFS tracking for large data files
`-- README.md
```

## 5. Installation

Clone the repository with Git LFS enabled when you need the large tracked data files:

```bash
git clone https://github.com/USTC-StarTeam/P-Law.git
cd P-Law
git lfs pull
```

Install the entropy utility:

```bash
pip install -e PerformanceLaw
```

Install dependencies for the transformer experiments:

```bash
cd General_Transformer
pip install -r requirements.txt
```

## 6. Data

Some data files under `General_Transformer/data/` are tracked with Git LFS. If a large file is unavailable from LFS, prepare the corresponding recommendation dataset manually before launching full-scale experiments. The training code expects preprocessed sequential data paths to match the path settings used inside `General_Transformer/main.py`.

Appendix frameworks under `Performance_Law_Appendix_Framework/` may follow their original dataset conventions. Read the local README inside each framework directory before running those baselines.

## 7. Quick Start

Estimate the actual entropy of a short sequence:

```python
from PerformanceLaw import actual_entropy, actual_entropy_tq

sequence = [1, 2, 1, 2, 3]
print(actual_entropy(sequence))
print(actual_entropy_tq(sequence))  # progress-bar version for longer sequences
```

Run a transformer experiment:

```bash
cd General_Transformer
torchrun --standalone --nproc_per_node=1 main.py --model_name llama --n_layers 8 --n_heads 8 --batch_size 32
```

Use `--eval_only` with `--ckpt_name` to evaluate an existing checkpoint.

## 8. Reproducing Paper Results

Train or evaluate the sequential recommendation backbone from `General_Transformer/`, then fit the law surfaces from the appendix result directory:

```bash
cd Performance_Law_Appendix_Result
python performance_law_fitting.py
```

The fitting script prepares metric matrices, normalizes them, fits the parametric Performance Law, reports fitting quality, and generates 3D visualizations for HR/NDCG.

## 9. Configuration Notes

Important arguments in `General_Transformer/main.py` include:

- `--model_name`: `llama` or `hstu`
- `--n_layers`: number of transformer layers
- `--n_heads`: number of attention heads
- `--batch_size`: training batch size
- `--out_dir`: checkpoint and log directory
- `--deepspeed`: optional DeepSpeed configuration path

Adjust dataset paths in the training script before launching large experiments.

## 10. Experimental Highlights

![Performance Law fitting results](docs/assets/performance-law-results.png)

The fitted Performance Law surfaces track HR and NDCG behavior across model and data settings, while the simpler scaling-law surfaces miss important non-monotonic trends. This supports the paper's conclusion that recommendation model selection should account for both model scale and data-quality signals such as ApEn.

## 11. Notes For Maintainers

- Keep the top-level README focused on the paper story and runnable entry points; place framework-specific details in each subdirectory.
- Do not remove Git LFS tracking without replacing the large data workflow.
- When adding new appendix figures, store README-ready images under `docs/assets/` and keep original experiment artifacts in their source directories.

<a id="citation"></a>

## 12. Citation

```bibtex
@inproceedings{shen2024optimizing,
  title = {Optimizing Sequential Recommendation Models with Scaling Laws and Approximate Entropy},
  author = {Shen, Tingjia and Wang, Hao and Wu, Chuhan and Chin, Jin Yao and Guo, Wei and Liu, Yong and Guo, Huifeng and Lian, Defu and Tang, Ruiming and Chen, Enhong},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  series = {Proceedings of Machine Learning Research},
  volume = {235},
  year = {2024}
}
```

## 13. Contact

For paper questions, contact Hao Wang at `wanghao3@ustc.edu.cn` or Enhong Chen at `cheneh@ustc.edu.cn`. For repository issues, please open a GitHub issue in this repository.
