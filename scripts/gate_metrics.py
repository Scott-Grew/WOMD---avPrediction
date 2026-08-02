import argparse
import json
import pathlib
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from womd import dataset, evaluation, metrics, model


def measured_values(predictor, dataloader, device):
    model_per_sample, baseline_per_sample, _, offroad_rates = evaluation.collect_per_sample(
        predictor, dataloader, device
    )
    values = {}
    for source, per_sample in (
        ("model", model_per_sample),
        ("const-vel", baseline_per_sample),
    ):
        for name, reduced in metrics.reduce_per_sample(per_sample).items():
            values[f"{source}.{name}"] = reduced

    covered = offroad_rates["covered"]
    values["offroad.covered"] = float(covered.sum().item())
    if bool(covered.any()):
        for source in evaluation.OFFROAD_SOURCES:
            values[f"offroad.{source}"] = offroad_rates[source][covered].mean().item()
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pinned", required=True)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()

    if not arguments.write and arguments.tolerance is None:
        parser.error("--tolerance is required unless --write is given")

    device = torch.device(arguments.device)
    dataloader = DataLoader(
        dataset.ShardedSampleDataset(arguments.data),
        batch_size=arguments.batch_size,
        shuffle=False,
    )
    predictor = model.MotionPredictor().to(device)
    predictor.load_state_dict(
        torch.load(arguments.checkpoint, map_location=device)["model_state"]
    )

    measured = measured_values(predictor, dataloader, device)
    pinned_path = pathlib.Path(arguments.pinned)

    if arguments.write:
        pinned_path.write_text(json.dumps(measured, indent=2, sort_keys=True) + "\n")
        print(f"pinned {len(measured)} values from {len(dataloader.dataset)} samples to {pinned_path}")
        return 0

    pinned = json.loads(pinned_path.read_text())
    found = metrics.breaches(pinned, measured, arguments.tolerance)
    print(f"{len(pinned)} pinned values · tolerance {arguments.tolerance:.1%} · {len(found)} breached")
    for name, (pinned_value, measured_value) in sorted(found.items()):
        rendered = "missing" if measured_value is None else f"{measured_value:.4f}"
        print(f"  {name:<34}pinned {pinned_value:>10.4f}   measured {rendered:>10}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
