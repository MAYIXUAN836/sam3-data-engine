import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot train_all_loss over epochs for a given experiment."
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="../experiments/exp2",
        help="Path to experiment directory (containing logs/train_stats.json)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="train_loss_exp2.png",
        help="Output PNG filename",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    exp_dir = Path(args.exp_dir)
    log_path = exp_dir / "logs" / "train_stats.json"
    if not log_path.is_file():
        raise FileNotFoundError(f"Cannot find log file: {log_path}")

    epoch_to_loss = {}
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            epoch = record.get("Trainer/epoch")
            loss = record.get("Losses/train_all_loss")
            if epoch is None or loss is None:
                continue
            # If multiple entries per epoch exist, keep the latest one
            epoch_to_loss[epoch] = loss

    if not epoch_to_loss:
        raise RuntimeError("No epoch/loss records found in train_stats.json")

    epochs = sorted(epoch_to_loss.keys())
    losses = [epoch_to_loss[e] for e in epochs]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Losses/train_all_loss")
    plt.title(f"Training loss over epochs: {exp_dir.name}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = Path(args.out)
    plt.savefig(out_path, dpi=150)
    print(f"Saved loss curve to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
