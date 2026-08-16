import glob
import sys
import time

import numpy as np
import torch

from womd import contract, model, pipeline


def main():
    staged_directory, anchors_path = sys.argv[1], sys.argv[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    unit_anchors = torch.from_numpy(np.load(anchors_path)["unit_anchors"]).float()
    eager = model.MotionPredictor(unit_anchors.clone()).to(device)
    compiled_base = model.MotionPredictor(unit_anchors.clone()).to(device)
    compiled_base.load_state_dict(eager.state_dict())
    compiled = torch.compile(compiled_base, dynamic=True)

    scenario_paths = sorted(glob.glob(f"{staged_directory}/*.npz"))[:24]
    batches = []
    for batch in pipeline.batches(scenario_paths, 0, 16, 0, 0, True):
        batches.append({name: tensor.to(device) for name, tensor in batch.items()})
        if len(batches) == 4:
            break

    worst_gap = 0.0
    with torch.no_grad():
        for batch in batches:
            eager_trajectories, eager_logits = eager(batch)
            compiled_trajectories, compiled_logits = compiled(batch)
            worst_gap = max(
                worst_gap,
                float((eager_trajectories - compiled_trajectories).abs().max()),
                float((eager_logits - compiled_logits).abs().max()),
            )
    print(f"compile probe: worst |eager - compiled| across 4 ragged batches = {worst_gap:.3e}")

    def time_forwards(predictor):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            for repeat in range(3):
                for batch in batches:
                    predictor(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - started

    eager_seconds = time_forwards(eager)
    compiled_seconds = time_forwards(compiled)
    print(
        f"compile probe: eager {eager_seconds:.2f} s, compiled {compiled_seconds:.2f} s"
        f" over 12 forward passes -> {eager_seconds / compiled_seconds:.2f}x"
    )
    print(
        "compile probe VERDICT: "
        + ("EQUIVALENT" if worst_gap < 1e-3 else "DIVERGED - do not enable --compile")
    )


if __name__ == "__main__":
    main()
