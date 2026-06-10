"""
train_krk.py
============

Train the RL agent (MaskablePPO) on the KRK endgame, reusing the SAME shared
trunk as the full-chess agents (network.ppo_policy_kwargs). Logs the mate-rate
over training -- we expect it to climb far above the ~1.5% random floor, i.e.
RL *learns to checkmate*, which is the "RL wins" half of the crossover.

Outputs (runs/krk/):
    krk_ppo.zip / krk_ppo_best.zip
    krk_ppo_metrics.json
    krk_ppo_curve.png

Usage:
    python train_krk.py --timesteps 1000000
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

from chess_env import encode_board, get_action_mask
from krk_env import KRKEndgameEnv, evaluate_krk
from network import ppo_policy_kwargs

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "krk")


def _mask_fn(env):
    return env.action_masks()


def make_env():
    return ActionMasker(KRKEndgameEnv(), _mask_fn)


def ppo_chooser(model):
    def choose(board):
        action, _ = model.predict(encode_board(board),
                                  action_masks=get_action_mask(board),
                                  deterministic=True)
        return int(action)
    return choose


class MateRateCallback(BaseCallback):
    def __init__(self, eval_freq, eval_positions, out_dir, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.eval_positions = eval_positions
        self.out_dir = out_dir
        self.history = []
        self.best = -1.0
        self._next = eval_freq

    def _eval(self):
        s = evaluate_krk(ppo_chooser(self.model), self.eval_positions)
        self.history.append({"timesteps": int(self.num_timesteps), **s})
        print(f"[krk] t={self.num_timesteps:>8,}  mate_rate={s['mate_rate']:.0%}  "
              f"avg_plies={s['avg_plies_to_mate']}")
        if s["mate_rate"] > self.best:
            self.best = s["mate_rate"]
            self.model.save(os.path.join(self.out_dir, "krk_ppo_best"))

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.eval_freq
            self._eval()
        return True

    def _on_training_end(self):
        self._eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--eval-freq", type=int, default=50_000)
    ap.add_argument("--eval-positions", type=int, default=200)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[krk] device={device}  timesteps={args.timesteps:,}")

    venv = DummyVecEnv([make_env for _ in range(args.n_envs)])
    model = MaskablePPO(
        "MlpPolicy", venv, policy_kwargs=ppo_policy_kwargs(),
        n_steps=args.n_steps, batch_size=args.batch_size, learning_rate=args.lr,
        ent_coef=args.ent_coef, gamma=args.gamma, seed=args.seed,
        device=device, verbose=0,
    )

    cb = MateRateCallback(args.eval_freq, args.eval_positions, args.out_dir)
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, callback=cb)
    print(f"[krk] trained in {time.time()-t0:.0f}s")

    model.save(os.path.join(args.out_dir, "krk_ppo"))
    with open(os.path.join(args.out_dir, "krk_ppo_metrics.json"), "w") as f:
        json.dump({"total_timesteps": args.timesteps, "best_mate_rate": cb.best,
                   "history": cb.history}, f, indent=2)

    _plot(cb.history, os.path.join(args.out_dir, "krk_ppo_curve.png"))
    print(f"[krk] done. best mate-rate = {cb.best:.0%}")


def _plot(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = [h["timesteps"] for h in history]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, [h["mate_rate"] for h in history], "-o", color="#ff7f0e")
    ax.axhline(0.015, ls="--", color="gray", label="random floor (1.5%)")
    ax.set_xlabel("environment steps"); ax.set_ylabel("mate-rate")
    ax.set_title("PPO learning to checkmate (KRK endgame)")
    ax.set_ylim(0, 1); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=120)
    print(f"[krk] saved curve -> {path}")


if __name__ == "__main__":
    main()
