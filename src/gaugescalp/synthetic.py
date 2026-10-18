"""Finite dipole-field EEG task with independent subject groups."""

import argparse
import json
from pathlib import Path

import numpy as np


def electrodes(count=32):
    index = np.arange(count)
    z = 0.1 + 0.85 * (index + 0.5) / count
    angle = index * np.pi * (3 - np.sqrt(5))
    radius = np.sqrt(1 - z * z)
    return np.column_stack([radius * np.cos(angle), radius * np.sin(angle), z])


def trial(rng, xyz, label, subject_gain=1.0, samples=128):
    time = np.arange(samples) / 100
    sources = np.array([[0.25, 0, 0.25], [-0.2, 0.15, 0.3]])
    moments = np.array([[0.6, 0.2, 0.7], [-0.3, 0.7, 0.4]])
    displacement = xyz[:, None] - sources[None]
    lead = np.sum(displacement * moments[None], -1) / np.linalg.norm(displacement, axis=-1) ** 3
    activity = np.sin(2 * np.pi * np.array([10, 14])[:, None] * time + rng.uniform(0, 2 * np.pi, (2, 1)))
    amplitude = rng.uniform(0.4, 0.7, 2)
    amplitude[label] += rng.uniform(0.35, 0.65)
    signal = subject_gain * lead @ (amplitude[:, None] * activity)
    reference = rng.normal(0, 0.4, (1, samples)) + 0.4 * np.sin(2 * np.pi * 3 * time)
    return (signal + reference + rng.normal(0, 0.12, signal.shape)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--subjects", type=int, default=24)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.subjects < 6:
        parser.error("Use at least six independent subjects")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    xyz = electrodes()
    x, y, subject, split = [], [], [], []
    for group in range(args.subjects):
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, group]))
        partition = "train" if group < 0.6 * args.subjects else "val" if group < 0.8 * args.subjects else "test"
        gain = rng.uniform(0.7, 1.3)
        for index in range(12):
            label = index % 2
            x.append(trial(rng, xyz, label, gain))
            y.append(label)
            subject.append(group)
            split.append(partition)
    np.savez_compressed(output, x=np.stack(x), y=np.array(y), subject=np.array(subject),
                        coordinates=xyz, split=np.array(split))
    output.with_suffix(".json").write_text(json.dumps(dict(
        source="two oscillatory dipoles with analytic volume-conductor lead fields",
        truth="which source has larger stimulus amplitude; no class channel is appended",
        seed=args.seed, subjects=args.subjects, sample_rate=100,
        reference="independent common waveform per trial",
    ), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
