# μ-Optimized Integrated Gradients

**Optimizing Weights for Discretized Integrated Gradients**

A direct optimization framework for finding quadrature weights in discrete Integrated Gradients. The repository implements exactly three straight-line attribution methods: standard IG, IDG-PDF output-change weighting, and μ-Optimized IG.

**Key Result:** the μ-optimization loop introduces no additional model evaluations once path data are available and is designed to improve the conservation quality metric Q.

## The Problem

All methods use the same path convention:

```
γ_k = baseline + (k/N)(x - baseline),  k = 0,...,N
g_k = ∇f(γ_k),                         k = 0,...,N-1
Δf_k = f(γ_{k+1}) - f(γ_k)
```

Standard Integrated Gradients uses uniform weights:

```
A_raw = (x - baseline) * Σ_k (1/N) g_k
```

## The Solution

μ-Optimized IG finds weights μ ∈ P_N by optimizing:

```
min_{μ∈P_N}  -Q(μ) + (τ/2) ||μ||²₂
```

where:
- **Q(μ)** — weighted squared alignment between `d_k` and `Δf_k`
- **||μ||²₂** — L2 penalty: prevents degenerate weight collapse to a single step
- **τ** — regularization parameter balancing consistency and smoothness (typically 0.005–0.01)

### Q Objective

For μ-Optimized IG:

```
d_k = g_k · ((x - baseline)/N)
Q(μ) = (Σ_k μ_k d_k Δf_k)^2 / [(Σ_k μ_k d_k^2)(Σ_k μ_k Δf_k^2)]
```

Q is finite and lies in [0, 1] up to numerical tolerance.

## Methods Compared

| Method | Weights μ | What it optimizes |
|--------|-----------|-------------------|
| Standard IG | uniform (1/N) | IG uniform baseline |
| IDG-PDF | μ_k ∝ \|Δf_k\| | closed-form output-change heuristic |
| **μ-Optimized IG** | **optimized by PGD** | **min -Q(μ) + (τ/2)‖μ‖²** |

### Computational Cost

**Critical observation:** All three methods require the same model evaluations.

| Method | Forward passes | Backward passes | Extra arithmetic |
|--------|----------------|-----------------|------------------|
| Standard IG | N + 1 | N | — |
| IDG-PDF | N + 1 | N | O(N) |
| **μ-Optimized** | **N + 1** | **N** | **O(NT)** |

The μ-optimization loop introduces no additional model evaluations once path data are available. The O(NT) arithmetic cost is negligible compared to the O(N) neural network forward/backward passes.

## Results

ResNet-50 on ImageNet, N = 50 interpolation steps, zero baseline:

```
Method            Var_ν      CV²        𝒬       Time
──────────────────────────────────────────────────────
IG              0.015749   0.0278   0.9730      0.1s
IDG-PDF         0.005221   0.0100   0.9901      0.1s
μ-Optimized     0.000254   0.0005   0.9995      0.2s
```

Q is the weighted squared alignment score (1 = strongest alignment).

μ-Optimized achieves **𝒬 > 0.999** at negligible computational cost over standard IG.

## Files

```
u_optimize.py    μ-Optimization framework — three IG methods
batch_eval.py    Batch ImageNet sample evaluation and AUC aggregation
summarize_batch_auc.py  CSV/Markdown summaries for batch AUC JSON
lam.py          Base IG implementations and utilities
utilss.py       Metrics (Var_ν, CV², 𝒬), evaluation, plotting
```

## Quick Start

```bash
# Basic run (all 3 methods: IG, IDG-PDF, μ-Optimized)
python u_optimize.py

# With attribution heatmaps and fidelity diagnostics
python u_optimize.py --viz --viz-fidelity

# Export results to JSON
python u_optimize.py --json results.json

# Adjust regularization parameter τ
python u_optimize.py --tau 0.005

# Fewer steps (faster)
python u_optimize.py --steps 30

# Force CPU
python u_optimize.py --device cpu
```

## Evaluation

```bash
# Pixel-level insertion/deletion (Petsiuk et al., 2018)
python u_optimize.py --insdel --viz-insdel

# Single-image pixel insertion/deletion with JSON AUC/curves
python u_optimize.py \
  --steps 64 \
  --tau 0.01 \
  --iters 300 \
  --insdel \
  --insdel-steps 50 \
  --viz-insdel \
  --viz-path results/auc_pixel_N64_tau001.png \
  --json results/auc_pixel_N64_tau001.json

# Region-based insertion/deletion (SIC-style, uses SLIC superpixels)
python u_optimize.py --region-insdel --viz-region-insdel

# Region-based with grid patches instead of SLIC
python u_optimize.py --region-insdel --no-slic --patch-size 16

# Everything
python u_optimize.py --viz --viz-fidelity --insdel --viz-insdel \
                     --region-insdel --viz-region-insdel
```

## Batch ImageNet AUC

```bash
# Batch 5-image pixel insertion/deletion
python batch_eval.py \
  --num-images 5 \
  --image-dir sample_imagenet1k \
  --output-json results/batch_auc_N64_tau001.json \
  --steps 64 \
  --tau 0.01 \
  --iters 300 \
  --insdel \
  --insdel-steps 50 \
  --seed 0 \
  --skip-errors

# Summarize batch JSON to CSV and Markdown
python summarize_batch_auc.py results/batch_auc_N64_tau001.json \
  --output-csv results/batch_auc_summary.csv \
  --output-md results/batch_auc_summary.md
```

`batch_eval.py` loads ResNet-50 once, evaluates up to `--num-images` files
from `--image-dir`, uses the predicted ImageNet class by default, and writes
per-image method diagnostics plus insertion/deletion AUCs and curves when
`--insdel` is enabled. Use `--target-class CLASS_ID` to force one class for
all images.

## Two-Model Image Selection Workflow

```bash
# A. Prepare the first 5000 ImageNet validation examples and keep ResNet50-correct images
python prepare_firstN_correct.py \
  --first-count 5000 \
  --output-root data/imagenet_first5000_correct \
  --device cuda:0

# B. Run insertion/deletion evaluation on all correct images
python batch_eval.py \
  --selected-csv data/imagenet_first5000_correct/selected.csv \
  --model-name resnet50 \
  --steps 128 \
  --tau 0.001 \
  --iters 300 \
  --insdel \
  --insdel-steps 50 \
  --device cuda:0 \
  --skip-errors \
  --output-json results/full/full_resnet50.json

python batch_eval.py \
  --selected-csv data/imagenet_first5000_correct/selected.csv \
  --model-name vgg16 \
  --steps 128 \
  --tau 0.001 \
  --iters 300 \
  --insdel \
  --insdel-steps 50 \
  --device cuda:0 \
  --skip-errors \
  --output-json results/full/full_vgg16.json

# C. Flatten one row per image per method
python flatten_insdel_json.py \
  --input-json results/full/full_resnet50.json \
  --output-csv results/full/full_resnet50_flat.csv

python flatten_insdel_json.py \
  --input-json results/full/full_vgg16.json \
  --output-csv results/full/full_vgg16_flat.csv

# D. Select images where Mu beats IG and IDG-PDF on both models
python select_mu_better_two_models.py \
  --resnet-csv results/full/full_resnet50_flat.csv \
  --vgg-csv results/full/full_vgg16_flat.csv \
  --output-dir results/selection
```

`prepare_firstN_correct.py` writes `correct_index` as the 0-based position in
the ResNet50-correct filtered set. The selection JSON files use those
`correct_index` values, not `rank` or `source_order`.

### Progressive Model Filtering

Normally, every model evaluates the same `selected.csv`.

With `--progressive-filter`, the first model evaluates the full `selected.csv`.
Each later model evaluates only images that passed the previous model filter.
By default, a row survives when `Mu insdel > IDG-PDF insdel`; the stricter
`mu_gt_ig_idg_insdel` rule also requires `Mu insdel > IG insdel`. This can save
time when the final goal is to find images where μ-Optimized consistently beats
IDG-PDF across multiple models.

Filtered CSVs preserve the original columns, order, and `correct_index` values:

```bash
data/imagenet_first5000_correct/selected_after_resnet50_mu_gt_idg_insdel.csv
data/imagenet_first5000_correct/selected_after_resnet50_vgg16_mu_gt_idg_insdel.csv
```

Example:

```bash
python local_mu_pipeline.py \
  --first-count 5000 \
  --steps 64 \
  --iters 300 \
  --insdel-steps 50 \
  --tau 0.01 \
  --auc-score confidence \
  --device cuda:0 \
  --models resnet50 vgg16 \
  --progressive-filter \
  --progressive-filter-rule mu_gt_idg_insdel \
  --archive
```

## CLI Reference

| Flag | Description | Default |
|------|-------------|---------|
| `--steps N` | Interpolation steps | 50 |
| `--tau F` | L2 regularization parameter τ | 0.01 |
| `--iters N` | μ-optimization iterations | 300 |
| `--lr F` | μ-optimization learning rate | 0.05 |
| `--tau-sweep ...` | Run multiple τ values and export one JSON file | off |
| `--steps-sweep ...` | Run multiple N values and export one JSON file | off |
| `--device DEVICE` | Force cuda/cpu | auto |
| `--min-conf F` | Minimum classification confidence | 0.70 |
| `--viz` | Generate attribution heatmaps | off |
| `--viz-path PATH` | Heatmap output path | attribution_heatmaps.png |
| `--viz-fidelity` | Step fidelity diagnostic plot | off |
| `--insdel` | Pixel insertion/deletion scores | off |
| `--insdel-steps N` | Pixel ins/del granularity | 100 |
| `--viz-insdel` | Pixel ins/del curve plot | off |
| `--region-insdel` | Region-based ins/del scores | off |
| `--viz-region-insdel` | Region ins/del curve plot | off |
| `--patch-size N` | Grid patch size for regions | 14 |
| `--no-slic` | Use grid patches instead of SLIC | off (SLIC preferred) |
| `--json PATH` | Export results to JSON | off |
| `--seed N` | Random seed for reproducibility | 42 |
| `--skip N` | Skip N images in dataset | 0 |

### Batch CLI

| Flag | Description | Default |
|------|-------------|---------|
| `--num-images N` | Maximum images to evaluate; omit to evaluate all rows/images | all |
| `--selected-csv PATH` | CSV of selected images and metadata | off |
| `--image-dir PATH` | Directory of ImageNet-like image files | sample_imagenet1k |
| `--output-json PATH` | Batch JSON output path | results/batch_auc_N64_tau001.json |
| `--steps N` | Interpolation steps | 64 |
| `--tau F` | L2 regularization parameter τ | 0.01 |
| `--iters N` | μ-optimization iterations | 300 |
| `--lr F` | μ-optimization learning rate | 0.05 |
| `--insdel` | Pixel insertion/deletion scores | off |
| `--insdel-steps N` | Pixel ins/del granularity | 50 |
| `--auc-score {logit,confidence}` | Score used for insertion/deletion AUC only | logit |
| `--target-class N` | Force target ImageNet class for all images | predicted class |
| `--seed N` | Random seed for reproducibility | 0 |
| `--skip-errors` | Record per-image errors and continue | off |
| `--device DEVICE` | Force cuda/cpu | auto |

## Quality Metrics

### Effective Measure

Not all steps matter equally. Steps with tiny |Δf_k| carry negligible output change. The **effective measure** captures this:

```
ν_k = (μ_k Δf²_k) / Σ_j μ_j Δf²_j
```

Steps where output is flat (Δf_k ≈ 0) have ν_k ≈ 0 regardless of μ_k.

### Fidelity Variance

The fidelity variance under the effective measure:

```
φ̄_ν = Σ_k ν_k φ_k
Var_ν(φ) = Σ_k ν_k (φ_k − φ̄_ν)²
```

### Quality Metric Q

The attribution quality metric:

```
Q = 1 / (1 + CV²_ν(φ))
```

where `CV²_ν(φ) = Var_ν(φ) / φ̄²_ν` is the squared coefficient of variation.

**Properties:**
- Q ∈ [0, 1]
- Q = 1 ⟺ φ_k = const for all steps with ν_k > 0 (perfect conservation)
- `Q = (Σ μ_k d_k Δf_k)² / [(Σ μ_k d²_k)(Σ μ_k Δf²_k)]`

This is a squared weighted correlation between d_k and Δf_k under μ (Cauchy-Schwarz ratio).

## The Optimization Algorithm

### Objective

```
min_{μ∈P_N}  -Q(μ) + (τ/2) ||μ||²₂
```

The implementation optimizes this objective directly with projected gradient descent.

### Gradient

Define:
```
P = Σ_k μ_k d_k Δf_k
D = Σ_k μ_k d²_k
F = Σ_k μ_k Δf²_k
```

Then `Q(μ) = P² / (DF)` and:

```
∂Q/∂μ_k = (P/DF) [2d_k Δf_k − (P/D)d²_k − (P/F)Δf²_k]
```

The L2 penalty contributes: `∂/∂μ_k [(τ/2)||μ||²] = τμ_k`

### Projected Gradient Descent

1. Precompute: `a_k = d_k Δf_k`, `b_k = d²_k`, `c_k = Δf²_k`
2. Initialize: `μ_k = 1/N` for all k
3. For t = 1, ..., T:
   - Compute P, D, F
   - Compute gradient: `g_k = -(P/DF)[2a_k − (P/D)b_k − (P/F)c_k] + τμ_k`
   - Update: `μ ← μ − η·g`
   - Project onto simplex: `μ ← Proj_{P_N}(μ)`

**Cost:** O(N) arithmetic per iteration, operates entirely on precomputed vectors d_k, Δf_k.

## Why L2 Regularization?

Without the L2 penalty (τ = 0), optimizing Q alone can admit **degenerate solutions**:

**Example:** Consider N = 100 steps where:
- Steps 40–60: transition region (|Δf_k| ≫ 0, φ_k varies)
- Other steps: flat region (|Δf_k| ≈ 0)

Optimizing Q alone can yield μ concentrated on a tiny set of steps where the alignment ratio is high but the attribution uses little of the path. The resulting Q can be **vacuous**: high by the diagnostic but not useful as a stable quadrature rule.

The L2 penalty prevents this by penalizing extreme concentration:
- **||μ||²₂** = Σ μ²_k (Herfindahl index of concentration)
- Minimized by uniform distribution: ||μ||²₂ = 1/N
- Maximized by Dirac spike: ||μ||²₂ = 1

With τ > 0, concentrating μ on a small number of flat steps incurs high ||μ||²₂ cost, forcing the optimizer to spread weight—including over informative transition regions.

### Role of τ

The parameter τ controls the trade-off:

- **τ → 0⁺**: regularization vanishes, μ* maximizes Q alone (risk of degeneracy)
- **τ → ∞**: L2 penalty dominates, μ* → 1/N (recovers standard IG)
- **Intermediate τ**: balances consistency and spread. Empirically, **τ ∈ [0.005, 0.01]** works well (allows 5–15 steps to carry most weight when N = 50)

## Relationship to IDG-PDF

**IDG-PDF** uses closed-form output-change weights `μ_k ∝ |Δf_k|`. This is a heuristic and is **not** the optimizer of the μ-objective.

- **IDG-PDF**: assigns weight based on how much the output changes at each step. It is closed-form and requires no optimization.

- **μ-Optimized IG**: assigns weights by projected gradient descent on `-Q(μ) + (τ/2)||μ||²₂`.

IDG-PDF and μ-Optimized IG can produce similar weights on transition-heavy paths, but they are different procedures. IDG-PDF is not derived as a stationary point or optimizer of the μ-objective.

## Interpreting Weight Diagnostics

IDG-PDF can collapse onto early large output jumps because it uses only `|Δf_k|`. μ-Optimized IG instead uses the consistency between the step-level prediction `d_k` and the actual output change `Δf_k` through `Q(μ)`.

High Q should be interpreted together with the exported μ diagnostics: entropy, maximum weight, active count, and L2 mass. A high-Q result with very low entropy or a single dominant weight may be less stable than a result with broader support. The regularization parameter τ controls this sparsity/smoothness trade-off: small τ allows concentrated weights, while large τ encourages weights closer to uniform.

## Completeness Axiom

The completeness axiom requires `Σ_i A_i = f(x) − f(x')`.

For arbitrary weights μ ∈ P_N, completeness does not hold exactly (it holds only for μ_k = 1/N as N → ∞). We restore it by final rescaling:

```
A_i ← A_i · [f(x) − f(x')] / Σ_j A_j
```

The implementation applies this through a shared `completeness_rescale` utility.

**Note:** The quality metric Q is **orthogonal to completeness**—it measures how faithfully the attributions decompose the output change, not whether they sum to the correct total.

## Requirements

```
torch >= 2.0
torchvision
datasets           (for Hugging Face ImageNet validation streaming)
pillow
tqdm
matplotlib         (for --viz flags)
scikit-image       (optional, for SLIC superpixels in --region-insdel)
```

## Kaggle Pipeline

This repository can run a multi-seed ImageNet experiment in four stages:

1. Select images with torchvision ResNet50 only.
2. Sweep μ-Optimized IG hyperparameters on the top 20 selected images.
3. Evaluate the same selected images on ResNet50, VGG16, and DenseNet121.
4. Summarize JSON outputs to CSV and Markdown.

The attribution definitions are unchanged: IG uses uniform weights, IDG-PDF
uses `μ_k = |Δf_k| / Σ_j |Δf_j|`, and μ-Optimized IG solves
`min_{μ∈simplex} -Q(μ) + τ/2 ||μ||²`.

```bash
# Kaggle notebook setup
git clone <your-repo-url> lig
cd lig
pip install -r requirements.txt
```

Prepare ResNet50-selected datasets for seeds 0 through 4:

```bash
python prepare_imagenet_resnet_dataset.py \
  --seeds 0 1 2 3 4 \
  --candidate-count 5000 \
  --select-count 200 \
  --output-root data/imagenet_resnet50_selected \
  --batch-size 64 \
  --device cuda \
  --resume
```

If you attach a local ImageNet validation dataset, pass its root:

```bash
python prepare_imagenet_resnet_dataset.py \
  --imagenet-root /kaggle/input/imagenet/val \
  --seeds 0 1 2 3 4 \
  --candidate-count 5000 \
  --select-count 200 \
  --output-root data/imagenet_resnet50_selected \
  --device cuda
```

Run the μ-Optimized hyperparameter sweep for one seed:

```bash
python sweep_mu_config.py \
  --selection-csv data/imagenet_resnet50_selected/seed_0/selected.csv \
  --seed 0 \
  --tau-grid 0.001 0.005 0.01 0.05 0.1 1.0 \
  --steps-grid 16 32 64 128 \
  --num-images 20 \
  --output-dir results/sweeps \
  --device cuda \
  --skip-errors
```

Run full evaluation for one seed/model using the best config from the sweep:

```bash
python batch_eval.py \
  --selected-csv data/imagenet_resnet50_selected/seed_0/selected.csv \
  --num-images 200 \
  --model-name resnet50 \
  --steps 64 \
  --tau 0.001 \
  --iters 300 \
  --insdel \
  --insdel-steps 50 \
  --seed 0 \
  --device cuda \
  --skip-errors \
  --output-json results/full_eval_seed0_resnet50_N64_tau0p001.json
```

Summarize all full evaluation outputs:

```bash
python summarize_full_eval.py \
  --input-glob "results/full_eval_seed*_*.json" \
  --output-csv results/full_eval_summary.csv \
  --output-md results/full_eval_summary.md
```

The orchestration script runs the same stages:

```bash
# Single process/GPU
MODE=all bash scripts/run_kaggle_pipeline.sh

# Two process-level GPU workers
RUN_TWO_GPU=1 MODE=all bash scripts/run_kaggle_pipeline.sh

# Manual split
GPU_ID=0 SEEDS="0 2 4" MODE=all bash scripts/run_kaggle_pipeline.sh
GPU_ID=1 SEEDS="1 3" MODE=all bash scripts/run_kaggle_pipeline.sh
```

Useful stage-only commands:

```bash
MODE=prepare bash scripts/run_kaggle_pipeline.sh
MODE=sweep bash scripts/run_kaggle_pipeline.sh
MODE=eval bash scripts/run_kaggle_pipeline.sh
MODE=summarize bash scripts/run_kaggle_pipeline.sh
zip -r results.zip results data/imagenet_resnet50_selected
```

## References

- Sundararajan, M., Taly, A., and Yan, Q. "Axiomatic Attribution for Deep Networks." ICML 2017.
- Petsiuk, V., Das, A., and Saenko, K. "RISE: Randomized Input Sampling for Explanation of Black-box Models." BMVC 2018.
- Duchi, J., Shalev-Shwartz, S., Singer, Y., and Chandra, T. "Efficient Projections onto the ℓ1-ball for Learning in High Dimensions." ICML 2008.
