# GaugeScalp

Implementation of reference-invariant scalp coordinates with montage-dependent observability.

## Installation

Use Python `>=3.11`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data generation and preparation

The generator creates dipole-source scalp EEG with subject variation and known electrode coordinates. These are generated voltages, not patient recordings. Its JSON companion records assumptions and seed; preparation fixes subject-disjoint splits.

```bash
python -m gaugescalp.synthetic --output data/generated/eeg.npz --subjects 24 --seed 17
gaugescalp prepare --config configs/reproduce.json --input data/generated/eeg.npz --data data/reproduce
```

For external data: Download [EEGMMIDB](https://physionet.org/content/eegmmidb/1.0.0/) or [BCI Competition IV-2a](https://www.bbci.de/competition/iv/). Export epochs to an NPZ: `x` [trial,channel,time] in consistent voltage units, integer `y`, `subject`, `coordinates` [channel,3] on a sphere, and optional string `split` (train/val/test). Preserve physical electrode positions and the same epoch duration. Preparation creates subject-disjoint splits when absent and rejects supplied split leakage. Only training subjects fit the temporal decoder; all mask and reference transformations occur after splitting.

## Training and inference

`configs/reproduce.json` fixes seed `17`. Use the same config and prepared inputs throughout; change `seed` there for another run and keep its output directories separate.

```bash
gaugescalp train --config configs/reproduce.json --data data/reproduce --output runs/reproduce
```

## Evaluation

```bash
gaugescalp evaluate --config configs/reproduce.json --data data/reproduce --checkpoint runs/reproduce/model.pt --output runs/reproduce
```

## Comparisons and ablations

Fit each supported `--variant` into a separate directory, evaluate its own checkpoint, then pass the evaluated run directories to `compare --inputs`.

```bash
gaugescalp compare --inputs runs/reproduce --output outputs/comparison
gaugescalp plot --output outputs/comparison
```

## Outputs

The selected commands write their generated inputs under `data/generated/`, prepared data under `data/reproduce/`, and checkpoints/results in the explicit output directories. Corresponding steps produce `model.pt`, `history.csv`, `metrics.csv`, `metrics.json`, `predictions.json`. Comparisons write CSV/JSON and plots under their output directory.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, scipy 1.15.3, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
