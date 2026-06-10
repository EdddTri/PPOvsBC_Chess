"""
train_ppo.py
============

Step 4: train Agent A (Reinforcement Learning) with PPO.

We use sb3-contrib's MaskablePPO so the policy only ever considers LEGAL moves
(via env.action_masks()). The policy reuses the SAME shared trunk as the BC
agent (network.ppo_policy_kwargs -> ChessFeaturesExtractor + linear heads), so
any performance difference comes from the learning paradigm, not the model.

Setup (the choices that give lightweight chess RL a fighting chance):
  * Opponent: the fixed random mover (a stationary, beatable target).
  * Reward: +1/-1/0 at game end PLUS small material-swing shaping (reward_shaping)
    so the otherwise-sparse signal is learnable in ~1M steps.
  * Action masking: MaskablePPO, so exploration is over legal moves only.

Outputs (./runs/ppo/):
    ppo_model.zip          final model           (+ ppo_best.zip = best vs random)
    ppo_metrics.json       win/draw/loss vs random over training (feeds Step 6)
    ppo_learning_curve.png

Usage:
    python train_ppo.py --timesteps 1000000
    python train_ppo.py --timesteps 1000000 --no-shaping   # ablation: sparse reward
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from chess_env import ChessEnv
from network import ppo_policy_kwargs

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "ppo")


# ---------------------------------------------------------------------------
# Env construction
# ---------------------------------------------------------------------------
def _mask_fn(env) -> np.ndarray:
    return env.action_masks()


def make_env(reward_shaping: bool):
    """Factory: a shaped/sparse ChessEnv vs random, wrapped for action masking."""
    def _init():
        env = ChessEnv(reward_shaping=reward_shaping)
        return ActionMasker(env, _mask_fn)
    return _init


# ---------------------------------------------------------------------------
# Periodic evaluation vs the random opponent (the RL learning curve)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_vs_random(model, n_games: int, seed: int = 99999):
    """Play n_games as the (deterministic) policy vs random; return rates."""
    env = ChessEnv()                          # no shaping at eval time
    wins = draws = losses = 0
    for g in range(n_games):
        obs, _ = env.reset(seed=seed + g)
        done = False
        reward = 0.0
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
        wins += reward > 0
        losses += reward < 0
        draws += reward == 0
    return wins / n_games, draws / n_games, losses / n_games


class EvalVsRandomCallback(BaseCallback):
    """Every `eval_freq` steps, measure win/draw/loss vs random and log it."""

    def __init__(self, eval_freq, n_games, out_dir, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_games = n_games
        self.out_dir = out_dir
        self.history = []
        self.best_win = -1.0
        self._next_eval = eval_freq

    def _run_eval(self):
        w, d, l = evaluate_vs_random(self.model, self.n_games)
        self.history.append({"timesteps": int(self.num_timesteps),
                             "win": w, "draw": d, "loss": l})
        print(f"[ppo] t={self.num_timesteps:>8,}  vs random: "
              f"win={w:.0%} draw={d:.0%} loss={l:.0%}")
        if w > self.best_win:
            self.best_win = w
            self.model.save(os.path.join(self.out_dir, "ppo_best"))

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_freq
            self._run_eval()
        return True

    def _on_training_end(self) -> None:
        self._run_eval()                      # final point on the curve


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train the PPO (RL) agent")
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=256, help="rollout length per env")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01, help="exploration bonus")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--no-shaping", action="store_true",
                    help="ablation: terminal reward only (sparse)")
    ap.add_argument("--eval-freq", type=int, default=50_000)
    ap.add_argument("--eval-games", type=int, default=60)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    reward_shaping = not args.no_shaping
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ppo] device={device}  reward_shaping={reward_shaping}  "
          f"timesteps={args.timesteps:,}")

    venv = DummyVecEnv([make_env(reward_shaping) for _ in range(args.n_envs)])

    model = MaskablePPO(
        "MlpPolicy",
        venv,
        policy_kwargs=ppo_policy_kwargs(),     # <-- the SHARED trunk
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        ent_coef=args.ent_coef,
        gamma=args.gamma,
        seed=args.seed,
        device=device,
        verbose=0,
    )

    callback = EvalVsRandomCallback(args.eval_freq, args.eval_games, args.out_dir)

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
    print(f"[ppo] training finished in {time.time()-t0:.0f}s")

    model.save(os.path.join(args.out_dir, "ppo_model"))
    metrics = {"reward_shaping": reward_shaping,
               "total_timesteps": args.timesteps,
               "best_win_vs_random": callback.best_win,
               "history": callback.history}
    with open(os.path.join(args.out_dir, "ppo_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    _plot_curve(callback.history, os.path.join(args.out_dir, "ppo_learning_curve.png"))
    print(f"[ppo] done. best win-rate vs random = {callback.best_win:.0%}")


def _plot_curve(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [h["timesteps"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, [h["win"] for h in history], "-o", label="win")
    ax.plot(t, [h["draw"] for h in history], "-o", label="draw")
    ax.plot(t, [h["loss"] for h in history], "-o", label="loss")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("rate vs random opponent")
    ax.set_title("PPO learning curve (vs random)")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"[ppo] saved learning curve -> {path}")


if __name__ == "__main__":
    main()
