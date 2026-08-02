import argparse
import pathlib
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from womd import dataset, evaluation, model, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()

    device = torch.device(arguments.device)
    dataloader = DataLoader(
        dataset.ShardedSampleDataset(arguments.data),
        batch_size=arguments.batch_size,
        shuffle=False,
    )

    predictor = model.MotionPredictor().to(device)
    checkpoint = torch.load(arguments.checkpoint, map_location=device)
    predictor.load_state_dict(checkpoint["model_state"])

    print(f"samples {len(dataloader.dataset)} · checkpoint {arguments.checkpoint}")
    print()
    print(evaluation.render_dataloader_report(predictor, dataloader, device))
    print()
    print(report.render_latency(report.measure_latency(predictor, next(iter(dataloader)))))


if __name__ == "__main__":
    main()
