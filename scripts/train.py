import argparse
import pathlib
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from womd import contract, dataset, evaluation, loss, model


def select_device(requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path, predictor, optimizer, epoch):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": predictor.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
        },
        path,
    )


def load_checkpoint(path, predictor, optimizer, device):
    checkpoint = torch.load(path, map_location=device)
    predictor.load_state_dict(checkpoint["model_state"])
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint.get("epoch", 0)


def trained_mode_count_for_epoch(epoch, total_epochs, start_modes):
    if total_epochs <= 1:
        return 1
    progress = epoch / (total_epochs - 1)
    return max(1, round(start_modes - progress * (start_modes - 1)))


def train_one_epoch(predictor, dataloader, optimizer, device, gradient_clip, trained_mode_count):
    predictor.train()
    running_total, batch_count = 0.0, 0
    for batch in dataloader:
        batch = {name: value.to(device) for name, value in batch.items()}
        trajectories, mode_logits = predictor(batch)
        total, _ = loss.best_of_many_loss(
            trajectories,
            mode_logits,
            batch["future_positions"],
            batch["future_mask"],
            trained_mode_count=trained_mode_count,
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), gradient_clip)
        optimizer.step()
        running_total += total.item()
        batch_count += 1
    return running_total / max(batch_count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/predictor.pt")
    parser.add_argument("--resume")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--hidden-dim", type=int, default=model.HIDDEN_DIM)
    parser.add_argument("--attention-layers", type=int, default=model.ATTENTION_LAYERS)
    parser.add_argument("--feedforward-dim", type=int, default=model.FEEDFORWARD_DIM)
    parser.add_argument("--ewta-start-modes", type=int, default=contract.NUM_PREDICTED_MODES)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    torch.manual_seed(arguments.seed)
    device = select_device(arguments.device)

    train_dataset = dataset.ShardedSampleDataset(arguments.train)
    train_sampler = dataset.ShardLocalShuffleSampler(train_dataset, seed=arguments.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        sampler=train_sampler,
        num_workers=arguments.workers,
        drop_last=True,
    )
    validation_loader = DataLoader(
        dataset.ShardedSampleDataset(arguments.validation),
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.workers,
    )

    predictor = model.MotionPredictor(
        hidden_dim=arguments.hidden_dim,
        attention_layers=arguments.attention_layers,
        feedforward_dim=arguments.feedforward_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    start_epoch = 0
    if arguments.resume:
        start_epoch = load_checkpoint(arguments.resume, predictor, optimizer, device)

    parameter_count = sum(p.numel() for p in predictor.parameters())
    print(f"device {device} · parameters {parameter_count:,}", flush=True)

    for epoch in range(start_epoch, arguments.epochs):
        started = time.time()
        trained_mode_count = trained_mode_count_for_epoch(
            epoch, arguments.epochs, arguments.ewta_start_modes
        )
        average_loss = train_one_epoch(
            predictor,
            train_loader,
            optimizer,
            device,
            arguments.gradient_clip,
            trained_mode_count,
        )
        model_results, baseline_results = evaluation.evaluate_dataloader(
            predictor, validation_loader, device
        )
        elapsed = time.time() - started
        print(
            f"epoch {epoch + 1}/{arguments.epochs} · modes {trained_mode_count} · "
            f"loss {average_loss:.4f} · "
            f"minADE@8s {model_results['minADE@8s']:.3f} · "
            f"baseline {baseline_results['minADE@8s']:.3f} · {elapsed:.1f}s",
            flush=True,
        )
        save_checkpoint(arguments.checkpoint, predictor, optimizer, epoch + 1)

    print()
    print(evaluation.format_comparison(model_results, baseline_results))


if __name__ == "__main__":
    main()
