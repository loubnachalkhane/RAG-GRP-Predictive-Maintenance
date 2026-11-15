# CalibRAG: calibrated probabilistic RUL and retrieval for predictive maintenance

Code for the paper on a **controlled test of whether retrieved evidence adds decision value
beyond recalibration** in a calibrated, cost-aware predictive-maintenance pipeline.

> **Main result (honest framing).** A calibrated quantile GRU that outputs a failure probability and
> intervenes by a cost rule has decision value. For the retrievers tested (a text knowledge base and a
> nearest-neighbour retriever), retrieval adds **no** decision value beyond recalibration.
> We do not claim a superior method; the repository is meant to make this comparison reproducible.

## Framework

The pipeline turns a window of sensor readings into a maintenance decision in four steps.

1. **Quantile GRU.** A GRU with three heads predicts the 10%, 50% and 90% quantiles of the remaining
   useful life (RUL).
2. **Failure probability.** For a horizon `H`, the probability of failure within `H` is
   `F = Phi((H - Q50) / sigma)`, with `sigma = max((Q90 - Q10) / (2 * 1.2816), 1e-6)`.
3. **Calibration.** Temperature scaling, then split conformal prediction (alpha = 0.20) on
   held-out engines. Windows within one engine are not exchangeable, so the coverage guarantee is
   approximate.
4. **Cost-risk decision.** Intervene if `F * C_f > C_m` (default `C_m = 1`, `C_f = 10`; swept over
   `C_f` in {5, 10, 20, 50}).

### Controlled three-arm protocol

All configurations share the same data splits, preprocessing, seeds, cost function and decision rule.
Only the evidence source changes.

| Arm | Evidence | Configurations |
|---|---|---|
| S-only | calibrated sensor probability (+ Platt scaling) | C, C+Platt |
| R-only | retrieved evidence only | R-kNN, R-KB |
| S+R | learned logistic fusion of both | L-kNN, L-KB |

The fusion is `P = sigmoid(a + b * logit(F_sensor) + c * logit(f_evid))`, learned on held-out engines
(S-only: `c = 0`; R-only: `b = 0`). Two retrievers are tested: a **text knowledge base**
(MiniLM-L6-v2 sentence embeddings, k = 3) and a **kNN retriever** on sensor features
(last step and window mean, k = 50).

Baselines: deterministic GRU (A), A with a safety margin tuned on validation engines (A+margin),
uncalibrated quantiles (B), always-alarm, and a diagnostic oracle (reference only; it uses the true
outcome).

### Data and splits

- **C-MAPSS** FD001 to FD004 (horizon H = 30, 30, 30, 25).
- **N-CMAPSS** DS01-005 (14 sensor channels, 6 development and 3 test units, H = 50).
- Engines are split by engine into train, validation, temperature-scaling and conformal sets.
  No test label is used for training, early stopping, calibration or tuning.
- Results are averaged over 10 seeds, with paired bootstrap confidence intervals.

### Secondary analyses (not part of the final protocol)

Risk-controlled thresholds (CRC) and a skill gate are available in the pipeline for completeness. The
CRC thresholds did not keep the miss rate below their targets on several datasets.

## Repository contents

| File | Purpose |
|---|---|
| `calibrag_pipeline.py` | Main pipeline for C-MAPSS and N-CMAPSS: data loading, training, calibration, all configurations, decision evaluation, robustness and cost-ratio sweeps |
| `calibrag_cac.py` | Variant for a real industrial asset, evaluated with leave-episodes-out cross-validation. **The data are confidential and are not included** |
| `make_figures_public2.py` | Builds the paper figures from the result tables |

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install numpy pandas scipy scikit-learn torch matplotlib h5py sentence-transformers
```

## Data

The public benchmarks are **not** included. Download them from the NASA Prognostics Center of Excellence
data repository and place them as follows.

```
data/                       C-MAPSS files (train_FD00X.txt, test_FD00X.txt, RUL_FD00X.txt)
ncmapss/N-CMAPSS_DS01-005.h5
kb/kb_{dataset}.json        text knowledge bases (one per dataset)
```

## Usage

Run one dataset and one seed:

```bash
python calibrag_pipeline.py --dataset FD001 --seeds 0 --data_dir ./data --out_dir ./results
```

Run all ten seeds (paper setting):

```bash
python calibrag_pipeline.py --dataset FD001 --seeds 0,1,2,3,4,5,6,7,8,9 --epochs 50
```

Options: `--dataset` (FD001 to FD004, NCMAPSS), `--ncmapss_path`, `--kb_path`, `--epochs`,
`--no_baselines`, `--no_robustness`.

Build the figures from the result tables:

```bash
python make_figures_public2.py --tables . --results ./results --out figures_public
```

## Real industrial asset

`calibrag_cac.py` was used for a feasibility study on the combustion chamber of an industrial dryer
(36 stoppage episodes, 27 sensors, leave-episodes-out cross-validation). The sensor and maintenance data
are confidential and are not part of this repository.

## Reproducibility notes

- Seeds are fixed per run; results are reported as means over seeds with 95% intervals.
- Costs are reported as total maintenance cost on the test set, relative to a policy that raises the
  alarm for every unit where stated in the paper.

## Citation

If you use this code, please cite the paper (reference to be added after publication).

## Contact

Loubna Chalkhane, ENSAM Casablanca and University of Cordoba.

## License

To be added.
