import argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .geometry import representation
from .common import load, save, table, plot, grouped_split, classification


class Decoder(nn.Module):
    def __init__(self, modes, classes):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(2 * modes, 64, 7, padding=3),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, classes),
        )

    def forward(self, x):
        return self.body(x)


def prepare(a, c):
    data = np.load(a.input)
    required = ["x", "y", "subject", "coordinates"]
    if any(k not in data for k in required):
        raise ValueError("Missing x, y, subject or coordinates")
    if data["x"].shape[1] != len(data["coordinates"]):
        raise ValueError("Channel geometry mismatch")
    split = (
        data["split"] if "split" in data else grouped_split(data["subject"], c["seed"])
    )
    for subject in np.unique(data["subject"]):
        if len(np.unique(split[data["subject"] == subject])) != 1:
            raise ValueError("Subject crosses splits")
    root = Path(a.data)
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        root / "eeg.npz",
        **{k: data[k] for k in data.files if k != "split"},
        split=split,
    )


def mask_for(xyz, size, kind, rng):
    size = min(size, len(xyz))
    mask = np.zeros(len(xyz), bool)
    if kind == "random":
        ids = rng.choice(len(xyz), size, replace=False)
    elif kind == "unilateral":
        ids = np.argsort(xyz[:, 0])[:size]
    elif kind == "regional":
        ids = np.argsort(xyz[:, 1])[:size]
    else:
        ids = np.argsort(xyz[:, 2])[:size]
    mask[ids] = True
    return mask


def train(a, c):
    torch.manual_seed(c["seed"])
    rng = np.random.default_rng(c["seed"])
    data = np.load(Path(a.data) / "eeg.npz")
    x = data["x"]
    xyz = data["coordinates"]
    y = data["y"]
    ids = np.flatnonzero(data["split"] == "train")
    model = Decoder(c["modes"], int(y.max()) + 1).to(c["device"])
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"])
    rows = []
    for epoch in range(c["epochs"]):
        total = 0.0
        model.train()
        rng.shuffle(ids)
        for start in range(0, len(ids), c["batch_size"]):
            batch = ids[start : start + c["batch_size"]]
            inputs = []
            for i in batch:
                size = int(
                    rng.choice(
                        sorted(
                            {s for s in c["mask_sizes"] if s <= len(xyz)} | {len(xyz)}
                        )
                    )
                )
                mask = mask_for(xyz, size, "random", rng)
                value, _ = representation(x[i], xyz, mask, c, a.variant)
                inputs.append(value)
            inputs = torch.from_numpy(np.stack(inputs)).to(c["device"])
            target = torch.as_tensor(y[batch], dtype=torch.long, device=c["device"])
            loss = nn.functional.cross_entropy(model(inputs), target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(batch)
        rows.append(dict(epoch=epoch + 1, loss=total / max(len(ids), 1)))
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            model=model.cpu().state_dict(),
            classes=int(y.max()) + 1,
            config=c,
            variant=a.variant,
        ),
        output / "model.pt",
    )
    table(output / "history.csv", rows)


def evaluate(a, c):
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    c = state["config"]
    model = Decoder(c["modes"], state["classes"])
    model.load_state_dict(state["model"])
    model.eval()
    data = np.load(Path(a.data) / "eeg.npz")
    ids = np.flatnonzero(data["split"] == "test")
    xyz = data["coordinates"]
    rows = []
    predictions = []
    rng = np.random.default_rng(c["seed"])
    for size in sorted(set(c["mask_sizes"]) | {len(xyz)}, reverse=True):
        if size > len(xyz):
            continue
        for kind in ["random", "unilateral", "regional", "wearable"]:
            probs = []
            discrepancies = []
            invariance = []
            for i in ids:
                mask = mask_for(xyz, size, kind, rng)
                value, coef = representation(
                    data["x"][i], xyz, mask, c, state["variant"]
                )
                full, full_coef = representation(
                    data["x"][i], xyz, np.ones(len(xyz), bool), c, state["variant"]
                )
                reference = rng.normal(size=(1, data["x"].shape[-1])) * np.std(
                    data["x"][i]
                )
                shifted, shifted_coef = representation(
                    data["x"][i] + reference, xyz, mask, c, state["variant"]
                )
                with torch.no_grad():
                    prob = model(torch.from_numpy(value)[None]).softmax(-1)[0].numpy()
                probs.append(prob)
                discrepancies.append(
                    np.sqrt(np.mean((coef - full_coef) ** 2))
                    / (np.sqrt(np.mean(full_coef**2)) + 1e-12)
                )
                invariance.append(np.sqrt(np.mean((shifted_coef - coef) ** 2)))
                predictions.append(
                    dict(id=int(i), size=size, mask=kind, probabilities=prob)
                )
            metrics = classification(data["y"][ids], np.stack(probs))
            rows.append(
                dict(
                    method=state["variant"],
                    channels=size,
                    mask=kind,
                    **metrics,
                    coordinate_discrepancy=np.mean(discrepancies),
                    reference_orbit_rms=np.mean(invariance),
                )
            )
    for row in rows:
        row["worst_mask_f1"] = min(
            r["macro_f1"] for r in rows if r["channels"] == row["channels"]
        )
    table(Path(a.output) / "metrics.csv", rows)
    save(Path(a.output) / "metrics.json", rows)
    save(Path(a.output) / "predictions.json", predictions)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "command", choices=["prepare", "train", "evaluate", "compare", "plot"]
    )
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--input")
    p.add_argument("--inputs", nargs="*")
    p.add_argument("--data", default="data/eeg")
    p.add_argument("--output", default="runs/full")
    p.add_argument("--checkpoint")
    p.add_argument(
        "--variant",
        choices=[
            "gaugescalp",
            "no_covariance",
            "constant_mode",
            "wrong_coordinates",
            "random_unweighted",
            "raw",
            "car",
            "spline",
        ],
        default="gaugescalp",
    )
    a = p.parse_args()
    c = load(a.config)
    if a.command == "prepare":
        prepare(a, c)
    elif a.command == "train":
        train(a, c)
    elif a.command == "evaluate":
        evaluate(a, c)
    elif a.command == "compare":
        rows = [r for d in a.inputs for r in load(Path(d) / "metrics.json")]
        save(Path(a.output) / "comparison.json", rows)
        table(Path(a.output) / "comparison.csv", rows)
    else:
        plot(Path(a.output) / "comparison.json", Path(a.output) / "figures")
