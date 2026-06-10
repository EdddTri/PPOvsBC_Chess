"""
sample_efficiency.py
====================

Step 6 (part 1): the sample-efficiency experiment.

To compare paradigms fairly on "how much experience did each agent need", we
measure the SAME y-axis (win-rate vs the random opponent) against each agent's
notion of experience:

  * BC  -> number of distinct labeled expert positions it was trained on.
  * PPO -> number of environment interaction steps (read from ppo_metrics.json).

For BC we retrain on increasing subsets of the dataset and evaluate each. For
PPO we reuse the curve already logged during training.

Output: runs/sample_efficiency.json  (consumed by plots.py)

Note (stated honestly in the report): a "labeled position" and an "environment
step" are not identical units -- but both answer "how much data did the agent
consume to reach this strength", which is the comparison of interest.

Usage:
    python sample_efficiency.py --sizes 50000 200000 1000000 5000000 --epochs 15
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import load_dataset
from network import BCNet
from train_bc import idx_to_onehot, winrate_vs_random

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def train_bc_on_subset(boards, actions, n, epochs, batch_size, lr, device, seed=0):
    """Train a fresh BCNet on the first `n` (shuffled) positions; return model."""
    torch.manual_seed(seed)
    sel = torch.randperm(len(actions))[:n]
    b, a = boards[sel], actions[sel]

    model = BCNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        order = torch.randperm(len(a))
        model.train()
        for i in range(0, len(a), batch_size):
            bi = order[i:i + batch_size]
            xb = idx_to_onehot(b[bi].to(device))
            yb = a[bi].to(device).long()
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def main():
    ap = argparse.ArgumentParser(description="BC sample-efficiency sweep")
    ap.add_argument("--dataset", default=os.path.join("data", "bc_dataset.npz"))
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[50_000, 200_000, 1_000_000, 5_000_000])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--ppo-metrics", default=os.path.join(RUNS, "ppo", "ppo_metrics.json"))
    ap.add_argument("--out", default=os.path.join(RUNS, "sample_efficiency.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[se] device={device}")

    boards_np, actions_np, _ = load_dataset(args.dataset)   # game ids unused here
    boards = torch.from_numpy(boards_np)
    actions = torch.from_numpy(actions_np)
    total = len(actions)

    bc_points = []
    for n in args.sizes:
        if n > total:
            print(f"[se] skip size {n:,} (> dataset {total:,})")
            continue
        t0 = time.time()
        model = train_bc_on_subset(boards, actions, n, args.epochs,
                                   args.batch_size, args.lr, device)
        w, d, l = winrate_vs_random(model, args.eval_games, device)
        bc_points.append({"positions": int(n), "win_rate": w,
                          "draw_rate": d, "loss_rate": l})
        print(f"[se] BC n={n:>9,}  win={w:.0%} draw={d:.0%} loss={l:.0%}  "
              f"({time.time()-t0:.0f}s)")

    # PPO curve (reuse what training already logged).
    ppo_points = []
    if os.path.exists(args.ppo_metrics):
        with open(args.ppo_metrics) as f:
            hist = json.load(f).get("history", [])
        ppo_points = [{"steps": h["timesteps"], "win_rate": h["win"],
                       "draw_rate": h["draw"], "loss_rate": h["loss"]} for h in hist]
        print(f"[se] loaded {len(ppo_points)} PPO points from {args.ppo_metrics}")
    else:
        print(f"[se] WARNING: {args.ppo_metrics} not found; PPO curve will be empty")

    out = {"bc": bc_points, "ppo": ppo_points, "eval_games": args.eval_games}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[se] saved -> {args.out}")


if __name__ == "__main__":
    main()
