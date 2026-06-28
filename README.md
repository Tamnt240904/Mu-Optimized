# μ-Optimized Integrated Gradients

**Optimizing Weights for Discretized Integrated Gradients**

A direct optimization framework for finding quadrature weights in discrete
Integrated Gradients. The repository compares two straight-line attribution
methods: standard IG and μ-Optimized IG.

**Key Result:** after path gradients and model outputs have been computed, the
μ-optimization loop requires no additional model evaluations.

## The Problem

All methods use the same straight-line path:

```text
γ_k = baseline + (k/N)(x - baseline),  k = 0,...,N
g_k = ∇f(γ_k),                         k = 0,...,N-1
Δf_k = f(γ_{k+1}) - f(γ_k)
```

Standard Integrated Gradients uses uniform weights:

```text
A_raw = (x - baseline) * Σ_k (1/N) g_k
```

## The Solution

μ-Optimized IG finds weights `μ ∈ P_N` by solving:

```text
min_{μ∈P_N}  -Q(μ) + (τ/2) ||μ||²₂
```

where:

- **Q(μ)** is the weighted squared alignment between `d_k` and `Δf_k`.
- **||μ||²₂** prevents the weights from collapsing onto a single step.
- **τ** balances alignment quality and weight smoothness.

For μ-Optimized IG:

```text
d_k = g_k · ((x - baseline)/N)
Q(μ) = (Σ_k μ_k d_k Δf_k)²
       / [(Σ_k μ_k d_k²)(Σ_k μ_k Δf_k²)]
```

Up to numerical tolerance, `Q` is finite and lies in `[0, 1]`.

## Methods Compared

| Method | Weights μ | Description |
|--------|-----------|-------------|
| Standard IG | `1/N` | Uniform quadrature weights |
| **μ-Optimized IG** | Optimized by PGD | Minimizes `-Q(μ) + (τ/2)‖μ‖²₂` |

## Files

```text
batch_eval.py    Batch image evaluation and JSON aggregation
u_optimize.py    μ-optimization and the two attribution methods
lam.py           Base IG implementations and model utilities
utilss.py        Metrics, insertion/deletion, and visualization utilities
mu-optimize.ipynb Kaggle experiment and result-table notebook
requirements.txt Python dependencies
```

## Requirements

- Python 3.10 or newer
- PyTorch and torchvision
- A CUDA-capable GPU is recommended for batch evaluation
- Internet access on the first run to download pretrained model weights

Install the dependencies:

```bash
git clone https://github.com/Tamnt240904/Mu-Optimized.git
cd Mu-Optimized

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a CUDA environment, install the PyTorch build matching the local CUDA
version before running the final command above. The Kaggle notebook already
installs its required CUDA build.

## Quick Start

### Single-image experiment

Place one or more `.jpg`, `.jpeg`, or `.png` files in
`sample_imagenet1k/`. The script selects a sufficiently confident image and
compares both methods.

```bash
mkdir -p sample_imagenet1k
mkdir -p results
# Copy images into sample_imagenet1k/, then run:
python u_optimize.py --steps 50 --tau 0.01 --iters 300
```

Useful variants:

```bash
# Force CPU execution
python u_optimize.py --device cpu

# Export metrics to JSON
python u_optimize.py --json results/single_image.json

# Generate attribution and fidelity plots
python u_optimize.py \
  --viz \
  --viz-fidelity \
  --viz-path results/attribution_heatmaps.png

# Compute insertion/deletion AUC
python u_optimize.py \
  --insdel \
  --insdel-steps 50 \
  --json results/single_image_insdel.json
```

If no suitable local image is found, the script tries CIFAR-10 and finally a
synthetic image. ResNet-50 pretrained weights are downloaded automatically on
the first run.

## Batch Image Evaluation

Put all input images directly inside one directory. Subdirectories are not
scanned.

```text
data/images/
├── image_001.jpg
├── image_002.png
└── image_003.jpeg
```

Run a short smoke test first:

```bash
python batch_eval.py \
  --image-dir data/images \
  --num-images 5 \
  --model-name resnet50 \
  --steps 32 \
  --tau 0.01 \
  --iters 100 \
  --device cuda:0 \
  --skip-errors \
  --output-json results/smoke_test.json
```

Run the standard evaluation with insertion/deletion metrics:

```bash
python batch_eval.py \
  --image-dir data/images \
  --model-name resnet50 \
  --steps 128 \
  --tau 0.1 \
  --iters 50 \
  --lr 0.05 \
  --insdel \
  --insdel-steps 50 \
  --auc-score confidence \
  --device cuda:0 \
  --skip-errors \
  --output-json results/resnet50_evaluation.json
```

When `--num-images` is omitted, every supported image directly inside
`--image-dir` is evaluated. Without `--target-class`, the model's predicted
class is used as the attribution target.

## Batch CLI Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--image-dir PATH` | Directory containing input images | `generated_imagenet/imagenet_resnet50_correct_1000` |
| `--num-images N` | Limit the number of evaluated images | all images |
| `--model-name NAME` | Torchvision classification model | `resnet50` |
| `--output-json PATH` | Output JSON path | generated from the configuration |
| `--steps N` | Number of interpolation steps | `64` |
| `--tau F` | L2 regularization parameter | `0.01` |
| `--iters N` | μ-optimization iterations | `300` |
| `--lr F` | μ-optimization learning rate | `0.05` |
| `--insdel` | Compute insertion/deletion metrics | off |
| `--insdel-steps N` | Insertion/deletion granularity | `50` |
| `--auc-score MODE` | AUC score: `logit` or `confidence` | `logit` |
| `--target-class N` | Force one ImageNet target class | predicted class |
| `--seed N` | Random seed | `0` |
| `--skip-errors` | Record failed images and continue | off |
| `--device DEVICE` | Device such as `cpu` or `cuda:0` | automatic |

Display the complete CLI help with:

```bash
python batch_eval.py --help
python u_optimize.py --help
```

## Output Format

`batch_eval.py` writes one JSON document containing:

- `config`: the complete evaluation configuration.
- `images`: per-image target information, method diagnostics, and optional
  insertion/deletion curves.
- `aggregate`: mean and standard deviation for each method.

The file is updated after each processed image, so partial progress is retained
if a long batch run is interrupted.

## Running with the Kaggle Notebook

The repository notebook `mu-optimize.ipynb` performs the following steps:

1. Clones this repository into `/kaggle/working/Mu-Optimized`.
2. Installs `requirements.txt` and a CUDA-compatible PyTorch build.
3. Configures `MU_GRAD_CHUNK` for the memory-efficient implementation already
   included in `lam.py`.
4. Runs `batch_eval.py` for the configured models and external image dataset.
5. Builds IG versus μ-Optimized CSV comparison tables from the generated JSON
   results.

Before running it on Kaggle, update `IMAGE_DIR` in the notebook so it points to
the attached Kaggle image dataset. Then use **Run all**; no additional repo
scripts are required.

## Quality Metrics

### Effective Measure

```text
ν_k = (μ_k Δf²_k) / Σ_j μ_j Δf²_j
```

Steps with negligible output change receive negligible effective mass.

### Fidelity Variance

```text
φ̄_ν = Σ_k ν_k φ_k
Var_ν(φ) = Σ_k ν_k (φ_k - φ̄_ν)²
```

### Quality Metric Q

```text
Q = 1 / (1 + CV²_ν(φ))
```

Higher `Q` indicates stronger alignment between the gradient-based step
prediction and the actual output change. Interpret it together with the
exported weight diagnostics such as entropy, maximum weight, active count, and
L2 mass.

## Notes

- The baseline is a zero tensor in normalized image space.
- Pretrained torchvision weights may consume several hundred megabytes.
- Larger `--steps` values usually improve path resolution but use more GPU
  memory and computation time.
- Reduce `--steps`, `--insdel-steps`, or `MU_GRAD_CHUNK` if
  CUDA runs out of memory.
- Use `--skip-errors` for long batch jobs so one corrupt image does not stop the
  entire evaluation.

## References

- Sundararajan, M., Taly, A., and Yan, Q. “Axiomatic Attribution for Deep
  Networks.” ICML 2017.
- Petsiuk, V., Das, A., and Saenko, K. “RISE: Randomized Input Sampling for
  Explanation of Black-box Models.” BMVC 2018.
- Duchi, J., Shalev-Shwartz, S., Singer, Y., and Chandra, T. “Efficient
  Projections onto the ℓ1-ball for Learning in High Dimensions.” ICML 2008.
