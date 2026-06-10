"""
krk_compare.py
==============

The payoff: evaluate BC vs PPO vs a random baseline on the SAME set of KRK
endgame positions, and draw the crossover figure.

  * On the KRK endgame, the RL agent (trained in the env with a dense reward)
    should checkmate far more often than the BC agent -- which was trained on
    human full-game data that contains essentially no KRK technique.
  * Side-by-side with the full-chess result, this shows the paradigms SWAP
    winners depending on the task: BC wins full chess, RL wins the KRK endgame.

Inputs:
    runs/bc/bc_weights.pt        Agent B (same network, trained on human games)
    runs/krk/krk_ppo_best.zip    Agent A trained on KRK
    runs/eval_results.json       full-chess win-rates (for the crossover plot)

Outputs:
    runs/krk/krk_results.json
    runs/plots/krk_mate_rate.png
    runs/plots/crossover.png

Usage:
    python krk_compare.py --positions 300
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from evaluate import bc_chooser, ppo_chooser, random_chooser, load_bc, load_ppo
from krk_env import evaluate_krk

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
PLOTS = os.path.join(RUNS, "plots")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=300)
    ap.add_argument("--bc-weights", default=os.path.join(RUNS, "bc", "bc_weights.pt"))
    ap.add_argument("--krk-ppo", default=os.path.join(RUNS, "krk", "krk_ppo_best.zip"))
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[krk-cmp] device={device}  positions={args.positions}")

    bc = load_bc(args.bc_weights, device)
    ppo = load_ppo(args.krk_ppo, device)
    rng = np.random.default_rng(args.seed)

    # Same positions/seed for all three -> fair.
    res = {
        "positions": args.positions,
        "random": evaluate_krk(random_chooser(rng), args.positions, seed=args.seed),
        "bc": evaluate_krk(bc_chooser(bc), args.positions, seed=args.seed),
        "ppo": evaluate_krk(ppo_chooser(ppo), args.positions, seed=args.seed),
    }
    for k in ("random", "bc", "ppo"):
        print(f"  {k:7s} mate_rate={res[k]['mate_rate']:.0%}  "
              f"avg_plies={res[k]['avg_plies_to_mate']}")

    os.makedirs(PLOTS, exist_ok=True)
    with open(os.path.join(RUNS, "krk_results.json"), "w") as f:
        json.dump(res, f, indent=2)

    _plot_mate_rate(res)
    _plot_crossover(res)
    print("[krk-cmp] done -> runs/plots/")


def _plot_mate_rate(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = ["Random", "BC", "PPO (RL)"]
    vals = [res["random"]["mate_rate"], res["bc"]["mate_rate"], res["ppo"]["mate_rate"]]
    colors = ["#7f7f7f", "#1f77b4", "#ff7f0e"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel(f"checkmate rate over {res['positions']} positions")
    ax.set_title("KRK endgame: who can actually checkmate?")
    ax.set_ylim(0, 1); ax.grid(axis="y", alpha=.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + .02, f"{v:.0%}", ha="center")
    fig.tight_layout()
    out = os.path.join(PLOTS, "krk_mate_rate.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"[krk-cmp] {out}")


def _plot_crossover(res):
    """The headline: BC wins full chess, RL wins the KRK endgame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev = json.load(open(os.path.join(RUNS, "eval_results.json")))
    chess_bc = ev["bc_vs_random"]["win_rate"]
    chess_ppo = ev["ppo_vs_random"]["win_rate"]
    krk_bc = res["bc"]["mate_rate"]
    krk_ppo = res["ppo"]["mate_rate"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    x = np.arange(2); w = 0.55

    ax1.bar(x, [chess_bc, chess_ppo], w, color=["#1f77b4", "#ff7f0e"])
    ax1.set_xticks(x); ax1.set_xticklabels(["BC", "PPO (RL)"])
    ax1.set_ylim(0, 1); ax1.set_ylabel("win-rate vs random")
    ax1.set_title("Full chess\n(huge space, sparse reward, rich expert data)")
    for i, v in enumerate([chess_bc, chess_ppo]):
        ax1.text(i, v + .02, f"{v:.0%}", ha="center")

    ax2.bar(x, [krk_bc, krk_ppo], w, color=["#1f77b4", "#ff7f0e"])
    ax2.set_xticks(x); ax2.set_xticklabels(["BC", "PPO (RL)"])
    ax2.set_ylim(0, 1); ax2.set_ylabel("checkmate rate")
    ax2.set_title("KRK endgame\n(tiny space, dense reward, ~no expert data)")
    for i, v in enumerate([krk_bc, krk_ppo]):
        ax2.text(i, v + .02, f"{v:.0%}", ha="center")

    fig.suptitle("The crossover: BC wins full chess, RL wins the dense-reward endgame",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(PLOTS, "crossover.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"[krk-cmp] {out}")


if __name__ == "__main__":
    main()
