import sys
from pathlib import Path

import torch

from womd import loss, pipeline
from womd.model import MotionPredictor


def main():
    staged_directory = Path(sys.argv[1])
    batch_size = int(sys.argv[2])
    scenario_paths = sorted(staged_directory.glob("*.npz"))

    torch.manual_seed(0)
    predictor = MotionPredictor().eval()
    batch = next(iter(pipeline.batches(scenario_paths, 0, batch_size, 0, 0, True)))

    with torch.no_grad():
        trajectories, heading_cosine_sine, confidence_logits = predictor.predict_with_heading(batch)
        _, regression, _, _, offroad = loss.prediction_loss(
            trajectories, heading_cosine_sine, confidence_logits,
            batch["future_positions"], batch["future_headings"], batch["future_mask"],
            batch["drivable_positions"], batch["drivable_mask"], 0.0, 0.0,
        )

    sample_count = batch["agent_history"].shape[0]
    drivable_dots = int(batch["drivable_mask"].sum())
    print(
        f"scenarios {len(scenario_paths)} | samples {sample_count} | "
        f"drivable dots {drivable_dots} ({drivable_dots / sample_count:.0f}/sample) | "
        f"untrained model, seed 0"
    )
    print(
        f"regression {float(regression):.4f} m | offroad {float(offroad):.4f} m"
    )
    print(
        f"equal footing --offroad-loss-weight "
        f"{float(regression) / float(offroad):.4f}"
    )


if __name__ == "__main__":
    main()
