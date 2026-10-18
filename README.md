# GaugeScalp

PyTorch implementation of reference-invariant scalp coordinates with montage-dependent observability.

## Installation


Use Python 3.11 or newer. Create and activate a virtual environment, then install:

```bash
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell, then install:

```bash
python -m pip install -e .
``` For GPU execution, install matching PyTorch and torchvision builds using the [PyTorch selector](https://pytorch.org/get-started/locally/).


## EEG input

Download [EEGMMIDB](https://physionet.org/content/eegmmidb/1.0.0/) or [BCI Competition IV-2a](https://www.bbci.de/competition/iv/). Export epochs to an NPZ: `x` [trial,channel,time] in consistent voltage units, integer `y`, `subject`, `coordinates` [channel,3] on a sphere, and optional string `split` (train/val/test). Preserve physical electrode positions and the same epoch duration. Preparation creates subject-disjoint splits when absent and rejects supplied split leakage. Only training subjects fit the temporal decoder; all mask and reference transformations occur after splitting.

```bash
gaugescalp prepare --input data/raw_epochs.npz --data data/eeg
gaugescalp train --data data/eeg --variant gaugescalp --output runs/full
gaugescalp evaluate --data data/eeg --checkpoint runs/full/model.pt --output runs/full
gaugescalp compare --inputs runs/full --output outputs/comparison
gaugescalp plot --output outputs/comparison
```

## Coordinates and controls

Real spherical harmonics excluding degree zero form the canonical basis. Helmert contrasts have full row rank and exactly cancel a common reference. Relative difference precision is transformed as `(D Dᵀ)^-1`; the regularized solve returns coefficients and a scaled inverse-curvature diagonal. The temporal decoder receives coefficients plus the repeated log diagonal. All variants use the same 2K-input temporal decoder.

Train and evaluate separate runs for `raw`, `car`, `spline`, `no_covariance`, `constant_mode`, `wrong_coordinates` and `random_unweighted`; pass the evaluated directories to `compare`. CAR/raw controls project zero-filled canonical channels into the shared basis. The spline control interpolates CAR channels with a thin-plate RBF before that projection. The covariance ablation leaves coefficients unchanged. The random-contrast control intentionally omits transformed precision. Wrong coordinates permute the geometry. These controlled baselines implement the measurement-interface comparison; pretrained OmnEEG/EEGMirror models are not bundled.

## Settings and metrics

`configs/default.json` specifies harmonic count, Laplace–Beltrami regularization, noise variance, decoder training, mask sizes and device. Training uses random channel masks; evaluation includes random, unilateral, regional and wearable-like (low-z) masks. Subject IDs are the independent splitting units; caller-supplied epochs must already handle artifact removal, resampling and labels. The current Gaussian difference-noise model is isotropic before contrast; the observability matrix is reported in the paper's scaled inverse-curvature convention.

Outputs include `model.pt`, `history.csv`, per-mask `metrics.csv`/JSON and trial probabilities. Metrics are accuracy, balanced accuracy, macro-F1, worst-mask F1, ECE, AURC, reference-orbit coefficient RMS and normalized cross-montage coefficient discrepancy. The exact same signal is used for each additive-reference control. External BCI evaluation uses a model trained on the matching label space and canonical basis; preparation never merges subjects across datasets implicitly. Comparison plotting writes PNG/SVG from measured files.

## Registered compact reproduction setting

Two oscillatory dipoles produce potentials at 32 registered scalp coordinates using an analytic lead field, subject gain, independent common reference and sensor noise. The task label identifies the stronger stimulated source. Subject-disjoint epochs feed the existing geometry and decoder pipeline.

Generate the declared input process, preprocess it, then use the existing method entry points:

```bash
python -m gaugescalp.synthetic --output data/generated/eeg.npz --subjects 24 --seed 17
gaugescalp prepare --config configs/reproduce.json --input data/generated/eeg.npz --data data/reproduce
gaugescalp train --config configs/reproduce.json --data data/reproduce --output runs/reproduce
gaugescalp evaluate --config configs/reproduce.json --data data/reproduce --checkpoint runs/reproduce/model.pt --output runs/reproduce
```

`eeg.json` records the data-generating assumptions and seed. The configuration records `input_source`. All inputs for this selected setting are generated locally using installed package dependencies.

## Multiple seeds

Run matched configurations with different seeds in separate directories, then aggregate the generated metric files:

```bash
gaugescalp-report --inputs runs/seed0/metrics.json runs/seed1/metrics.json --output outputs/seeds
```

Use actual metric paths from your runs, including method subdirectories where applicable. Keep dataset, split and experimental condition fixed within a report. Summary CSV/JSON gives per-method arithmetic means and sample standard deviations; a single run has no standard-deviation estimate.
