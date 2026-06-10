"""
plots.py
========

Step 6 (part 2): turn the saved metrics into the project's deliverables.

Reads (whatever exists) from runs/:
    bc/bc_metrics.json          BC training curve
    ppo/ppo_metrics.json        PPO learning curve vs random
    eval_results.json           both agents vs the same opponent + head-to-head
    sample_efficiency.json      win-rate vs amount of experience

Writes to runs/plots/:
    learning_curves.png         BC accuracy + PPO win-rate over training
    final_performance.png       win/draw/loss vs random (bar chart)
    sample_efficiency.png       win-rate vs experience consumed (log x)
    win_rate_table.md           summary table (the headline numbers)

Usage:
    python plots.py
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
PLOTS = os.path.join(RUNS, "plots")


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


# ---------------------------------------------------------------------------
def plot_learning_curves(bc, ppo):
    """Side-by-side: BC move-match accuracy vs epoch, PPO win-rate vs steps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    if bc:
        h = bc["history"]
        ep = [x["epoch"] for x in h]
        ax1.plot(ep, [x["val_top1"] for x in h], "-o", label="top-1")
        ax1.plot(ep, [x["val_top5"] for x in h], "-o", label="top-5")
        ax1.set_title("Agent B (BC): move-match accuracy")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("accuracy (val)")
        ax1.set_ylim(0, 1); ax1.legend(); ax1.grid(alpha=.3)

    if ppo:
        h = ppo["history"]
        t = [x["timesteps"] for x in h]
        ax2.plot(t, [x["win"] for x in h], "-o", label="win")
        ax2.plot(t, [x["draw"] for x in h], "-o", label="draw")
        ax2.plot(t, [x["loss"] for x in h], "-o", label="loss")
        ax2.set_title("Agent A (PPO): result vs random")
        ax2.set_xlabel("environment steps"); ax2.set_ylabel("rate")
        ax2.set_ylim(0, 1); ax2.legend(); ax2.grid(alpha=.3)

    fig.suptitle("Learning curves (different paradigms, different x-axes)")
    fig.tight_layout()
    out = os.path.join(PLOTS, "learning_curves.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"[plots] {out}")


def plot_final_performance(ev):
    """Grouped bar chart: win/draw/loss vs the same random opponent."""
    if not ev:
        return
    agents = [("Random", "random_baseline_vs_random"),
              ("PPO (RL)", "ppo_vs_random"),
              ("BC", "bc_vs_random")]
    agents = [(name, key) for name, key in agents if key in ev]
    labels = [name for name, _ in agents]
    wins = [ev[k]["win_rate"] for _, k in agents]
    draws = [ev[k]["draw_rate"] for _, k in agents]
    losses = [ev[k]["loss_rate"] for _, k in agents]

    import numpy as np
    x = np.arange(len(labels)); w = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, wins, w, label="win", color="#2ca02c")
    ax.bar(x, draws, w, label="draw", color="#7f7f7f")
    ax.bar(x + w, losses, w, label="loss", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(f"rate over {ev.get('games','?')} games")
    ax.set_title("Final performance vs the same random opponent")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(axis="y", alpha=.3)
    for i, v in enumerate(wins):
        ax.text(i - w, v + .02, f"{v:.0%}", ha="center", fontsize=9)
    fig.tight_layout()
    out = os.path.join(PLOTS, "final_performance.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"[plots] {out}")


def plot_sample_efficiency(se):
    """Win-rate vs amount of experience consumed (log x), both paradigms."""
    if not se:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    if se.get("bc"):
        xs = [p["positions"] for p in se["bc"]]
        ys = [p["win_rate"] for p in se["bc"]]
        ax.plot(xs, ys, "-o", label="BC (labeled positions)", color="#1f77b4")
    if se.get("ppo"):
        xs = [p["steps"] for p in se["ppo"]]
        ys = [p["win_rate"] for p in se["ppo"]]
        ax.plot(xs, ys, "-o", label="PPO (environment steps)", color="#ff7f0e")
    ax.set_xscale("log")
    ax.set_xlabel("experience consumed (log scale)")
    ax.set_ylabel("win-rate vs random")
    ax.set_title("Sample efficiency: win-rate vs experience")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    out = os.path.join(PLOTS, "sample_efficiency.png")
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"[plots] {out}")


def write_table(bc, ppo, ev):
    """Markdown summary of the headline numbers."""
    lines = ["# Results summary\n"]

    if ev:
        g = ev.get("games", "?")
        lines += [f"## Win/draw/loss vs the same random opponent ({g} games)\n",
                  "| Agent | Win | Draw | Loss |",
                  "|---|---|---|---|"]
        for name, key in [("Random baseline", "random_baseline_vs_random"),
                          ("Agent A - PPO (RL)", "ppo_vs_random"),
                          ("Agent B - BC", "bc_vs_random")]:
            if key in ev:
                r = ev[key]
                lines.append(f"| {name} | {r['win_rate']:.0%} | "
                             f"{r['draw_rate']:.0%} | {r['loss_rate']:.0%} |")
        lines.append("")
        if "head_to_head" in ev:
            h = ev["head_to_head"]
            lines += ["## Head-to-head (colours swapped)\n",
                      f"- BC wins: **{h['bc_wins']}**  |  PPO wins: "
                      f"**{h['ppo_wins']}**  |  draws: **{h['draws']}**  "
                      f"(of {h['games']})\n"]

    if bc:
        lines += ["## Agent B (BC) training",
                  f"- best val top-1 move-match: **{bc.get('best_val_top1', 0):.3f}**",
                  f"- dataset size: {bc.get('dataset_size', '?'):,} positions\n"]
    if ppo:
        lines += ["## Agent A (PPO) training",
                  f"- reward shaping: {ppo.get('reward_shaping')}",
                  f"- total steps: {ppo.get('total_timesteps', 0):,}",
                  f"- best win-rate vs random: **{ppo.get('best_win_vs_random', 0):.0%}**\n"]

    out = os.path.join(PLOTS, "win_rate_table.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[plots] {out}")
    print("\n" + "\n".join(lines))


def main():
    os.makedirs(PLOTS, exist_ok=True)
    bc = _load(os.path.join(RUNS, "bc", "bc_metrics.json"))
    ppo = _load(os.path.join(RUNS, "ppo", "ppo_metrics.json"))
    ev = _load(os.path.join(RUNS, "eval_results.json"))
    se = _load(os.path.join(RUNS, "sample_efficiency.json"))

    plot_learning_curves(bc, ppo)
    plot_final_performance(ev)
    plot_sample_efficiency(se)
    write_table(bc, ppo, ev)
    print("\n[plots] all done -> runs/plots/")


if __name__ == "__main__":
    main()
