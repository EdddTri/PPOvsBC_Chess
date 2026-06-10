"""
train_bc.py
===========

Step 3 (part 2): train Agent B (Behavioral Cloning).

We treat move prediction as a 4096-way classification problem: given a board
(encoded exactly as the environment encodes it), predict the expert's move.
The network is `BCNet` from network.py -- the SHARED trunk + a linear head -- so
it is architecturally identical to the PPO policy trained in Step 4.

Outputs (written to ./runs/bc/):
    bc_weights.pt        best model (by validation move-match accuracy)
    bc_metrics.json      per-epoch loss / top-1 / top-5  (feeds Step 6 plots)
    bc_learning_curve.png

Usage:
    python train_bc.py --dataset data/bc_dataset.npz --epochs 15
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_env import ChessEnv, get_action_mask
from data import load_dataset
from network import BCNet, count_params

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "bc")


def idx_to_onehot(idx: torch.Tensor) -> torch.Tensor:
    """(B,8,8) int codes -> (B,8,8,12) float one-hot (empty plane dropped).

    Identical layout to chess_env.encode_board / index_to_onehot.
    """
    return F.one_hot(idx.long(), num_classes=13)[..., 1:].float()


@torch.no_grad()
def evaluate(model, boards, actions, batch_size, device):
    """Return (mean loss, top-1 acc, top-5 acc) over a held-out set."""
    model.eval()
    n = len(actions)
    total_loss = top1 = top5 = 0
    for i in range(0, n, batch_size):
        xb = idx_to_onehot(boards[i:i + batch_size].to(device))
        yb = actions[i:i + batch_size].to(device).long()
        logits = model(xb)
        total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        top1 += (logits.argmax(1) == yb).sum().item()
        top5 += (logits.topk(5, dim=1).indices == yb[:, None]).any(1).sum().item()
    return total_loss / n, top1 / n, top5 / n


@torch.no_grad()
def winrate_vs_random(model, n_games, device, seed=12345):
    """Quick sanity check: BC (greedy, legal-masked) vs the random opponent."""
    model.eval()
    env = ChessEnv()
    wins = draws = losses = 0
    for g in range(n_games):
        obs, _ = env.reset(seed=seed + g)
        done = False
        reward = 0.0
        while not done:
            mask = env.action_masks()
            action = model.predict_action(obs, mask)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        wins += reward > 0
        losses += reward < 0
        draws += reward == 0
    return wins / n_games, draws / n_games, losses / n_games


def main():
    ap = argparse.ArgumentParser(description="Train the Behavioral Cloning agent")
    ap.add_argument("--dataset", default=os.path.join("data", "bc_dataset.npz"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--eval-games", type=int, default=40,
                    help="games vs random for the end-of-training sanity check (0 to skip)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[bc] device={device}")

    # ---- data ----------------------------------------------------------
    boards_np, actions_np, games_np = load_dataset(args.dataset)
    print(f"[bc] dataset: {len(actions_np):,} positions")
    boards = torch.from_numpy(boards_np)                 # int8 (N,8,8)
    actions = torch.from_numpy(actions_np)               # int32 (N,)

    if games_np is not None:
        # Split BY GAME: hold out whole games for validation so no game appears
        # in both sets. Positions within a game are highly correlated, so a
        # position-level split leaks and inflates the accuracy.
        games_t = torch.from_numpy(games_np).long()
        n_games = int(games_t.max().item()) + 1
        g_perm = torch.randperm(n_games)
        n_val_games = max(1, int(n_games * args.val_frac))
        is_val_game = torch.zeros(n_games, dtype=torch.bool)
        is_val_game[g_perm[:n_val_games]] = True
        val_mask = is_val_game[games_t]                  # (N,) bool, vectorized
        tr_b, tr_a = boards[~val_mask], actions[~val_mask]
        va_b, va_a = boards[val_mask], actions[val_mask]
        print(f"[bc] split BY GAME (no leakage): "
              f"{n_games - n_val_games:,} train games / {n_val_games:,} val games")
    else:
        print("[bc] WARNING: dataset has no game ids -> position-level split "
              "(val accuracy may be optimistic). Rebuild dataset to fix.")
        perm = torch.randperm(len(actions))
        boards, actions = boards[perm], actions[perm]
        n_val = int(len(actions) * args.val_frac)
        tr_b, tr_a = boards[n_val:], actions[n_val:]
        va_b, va_a = boards[:n_val], actions[:n_val]
    print(f"[bc] train={len(tr_a):,}  val={len(va_a):,}")

    # ---- model ---------------------------------------------------------
    model = BCNet().to(device)
    print(f"[bc] BCNet params: {count_params(model):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_top1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        order = torch.randperm(len(tr_a))
        running = 0.0
        for i in range(0, len(tr_a), args.batch_size):
            bi = order[i:i + args.batch_size]
            xb = idx_to_onehot(tr_b[bi].to(device))
            yb = tr_a[bi].to(device).long()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(bi)
        train_loss = running / len(tr_a)
        val_loss, top1, top5 = evaluate(model, va_b, va_a, args.batch_size, device)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_top1": top1, "val_top5": top5})
        print(f"[bc] epoch {epoch:2d}  train_loss={train_loss:.3f}  "
              f"val_loss={val_loss:.3f}  top1={top1:.3f}  top5={top5:.3f}  "
              f"({time.time()-t0:.0f}s)")

        if top1 > best_top1:
            best_top1 = top1
            torch.save(model.state_dict(), os.path.join(args.out_dir, "bc_weights.pt"))

    # ---- metrics + plot ------------------------------------------------
    metrics = {"dataset_size": int(len(actions_np)), "best_val_top1": best_top1,
               "history": history}
    with open(os.path.join(args.out_dir, "bc_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    _plot_curves(history, os.path.join(args.out_dir, "bc_learning_curve.png"))

    # ---- sanity check vs random ---------------------------------------
    if args.eval_games > 0:
        model.load_state_dict(torch.load(os.path.join(args.out_dir, "bc_weights.pt")))
        w, d, l = winrate_vs_random(model, args.eval_games, device)
        print(f"[bc] vs random ({args.eval_games} games): "
              f"win={w:.0%}  draw={d:.0%}  loss={l:.0%}")

    print(f"[bc] done. best val top-1 move-match = {best_top1:.3f}")


def _plot_curves(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], label="val")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("cross-entropy loss")
    ax1.set_title("BC training loss"); ax1.legend()

    ax2.plot(epochs, [h["val_top1"] for h in history], label="top-1")
    ax2.plot(epochs, [h["val_top5"] for h in history], label="top-5")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("move-match accuracy")
    ax2.set_title("BC move-match accuracy (val)"); ax2.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"[bc] saved learning curve -> {path}")


if __name__ == "__main__":
    main()
