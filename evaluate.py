"""
evaluate.py
===========

Step 5: evaluate BOTH agents under identical conditions.

The comparison only means something if both agents face the same opponent, the
same number of games, and the same seeds. This module provides:

  * a single `play_match` routine that takes a "move chooser" (a function
    board -> action) so BC, PPO, and a random baseline all run through the exact
    same game loop;
  * choosers for each agent that both go through the env's legal-move masking;
  * a head-to-head BC-vs-PPO match (the most direct paradigm comparison).

Results are written to runs/eval_results.json for the Step 6 plots.

Usage:
    python evaluate.py --games 200
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable

import numpy as np
import torch
import chess

from chess_env import ChessEnv, encode_board, get_action_mask, action_to_move
from network import BCNet

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")

# A "chooser" maps (board) -> action index in [0, 4096).
Chooser = Callable[[chess.Board], int]


# ---------------------------------------------------------------------------
# Move choosers (one per agent) -- all use the SAME legal-move masking
# ---------------------------------------------------------------------------
def random_chooser(rng: np.random.Generator) -> Chooser:
    def choose(board: chess.Board) -> int:
        legal = np.flatnonzero(get_action_mask(board))
        return int(legal[rng.integers(len(legal))])
    return choose


def bc_chooser(model: BCNet) -> Chooser:
    def choose(board: chess.Board) -> int:
        return model.predict_action(encode_board(board), get_action_mask(board))
    return choose


def ppo_chooser(model) -> Chooser:
    def choose(board: chess.Board) -> int:
        obs = encode_board(board)
        mask = get_action_mask(board)
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(action)
    return choose


# ---------------------------------------------------------------------------
# Game loops
# ---------------------------------------------------------------------------
def play_match(chooser: Chooser, n_games: int, opponent=None,
               seed: int = 7) -> dict:
    """Play `chooser` (as the agent) vs the env's opponent (random by default).

    Returns win/draw/loss counts and rates from the agent's perspective.
    """
    env = ChessEnv() if opponent is None else ChessEnv(opponent=opponent)
    wins = draws = losses = 0
    for g in range(n_games):
        obs, _ = env.reset(seed=seed + g)
        done = False
        reward = 0.0
        while not done:
            action = chooser(env.board)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        wins += reward > 0
        losses += reward < 0
        draws += reward == 0
    return {"games": n_games, "win": wins, "draw": draws, "loss": losses,
            "win_rate": wins / n_games, "draw_rate": draws / n_games,
            "loss_rate": losses / n_games}


def head_to_head(white_chooser: Chooser, black_chooser: Chooser,
                 n_games: int, seed: int = 7, opening_plies: int = 6) -> dict:
    """Direct match: one agent plays White, the other Black (no random opponent).

    Both agents are deterministic, so from the fixed start position every game
    would be identical. To get INDEPENDENT samples we seed each game and play a
    few random legal opening plies (an "opening book"), after which the agents
    take over. Run twice with colours swapped (same seeds) to cancel both the
    first-move advantage and any opening imbalance.

    Returns the result from White's perspective.
    """
    white_wins = black_wins = draws = 0
    for g in range(n_games):
        board = chess.Board()
        rng = np.random.default_rng(seed + g)
        for _ in range(opening_plies):              # random, distinct opening per game
            if board.is_game_over():
                break
            legal = list(board.legal_moves)
            board.push(legal[rng.integers(len(legal))])
        ply = 0
        while not board.is_game_over(claim_draw=True) and ply < 200:
            chooser = white_chooser if board.turn == chess.WHITE else black_chooser
            move = action_to_move(board, chooser(board))
            if move is None:                  # safety; masking should prevent this
                break
            board.push(move)
            ply += 1
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            draws += 1
        elif outcome.winner == chess.WHITE:
            white_wins += 1
        else:
            black_wins += 1
    return {"games": n_games, "white_wins": white_wins,
            "black_wins": black_wins, "draws": draws}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_bc(path: str, device: str) -> BCNet:
    model = BCNet().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def load_ppo(path: str, device: str):
    from sb3_contrib import MaskablePPO
    return MaskablePPO.load(path, device=device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate BC vs PPO under identical conditions")
    ap.add_argument("--games", type=int, default=200, help="games per evaluation")
    ap.add_argument("--bc-weights", default=os.path.join(RUNS, "bc", "bc_weights.pt"))
    ap.add_argument("--ppo-model", default=os.path.join(RUNS, "ppo", "ppo_best.zip"))
    ap.add_argument("--out", default=os.path.join(RUNS, "eval_results.json"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device={device}  games={args.games}")
    rng = np.random.default_rng(args.seed)

    bc = load_bc(args.bc_weights, device)
    ppo = load_ppo(args.ppo_model, device)

    results = {"games": args.games}

    # 1) Each agent (and a random baseline) vs the SAME random opponent.
    print("[eval] vs random opponent ...")
    results["random_baseline_vs_random"] = play_match(random_chooser(rng), args.games, seed=args.seed)
    results["bc_vs_random"] = play_match(bc_chooser(bc), args.games, seed=args.seed)
    results["ppo_vs_random"] = play_match(ppo_chooser(ppo), args.games, seed=args.seed)
    for k in ("random_baseline_vs_random", "bc_vs_random", "ppo_vs_random"):
        r = results[k]
        print(f"  {k:28s} win={r['win_rate']:.0%} draw={r['draw_rate']:.0%} loss={r['loss_rate']:.0%}")

    # 2) Head-to-head, colours swapped to cancel the first-move advantage.
    print("[eval] head-to-head BC vs PPO ...")
    h = args.games // 2
    # Same seeds in both calls -> each opening is played once with BC as White and
    # once with PPO as White (a clean paired, colour-swapped comparison).
    a = head_to_head(bc_chooser(bc), ppo_chooser(ppo), h, seed=args.seed)        # BC=White
    b = head_to_head(ppo_chooser(ppo), bc_chooser(bc), h, seed=args.seed)         # PPO=White
    bc_wins = a["white_wins"] + b["black_wins"]
    ppo_wins = a["black_wins"] + b["white_wins"]
    draws = a["draws"] + b["draws"]
    total = bc_wins + ppo_wins + draws
    results["head_to_head"] = {"games": total, "bc_wins": bc_wins,
                               "ppo_wins": ppo_wins, "draws": draws}
    print(f"  BC wins={bc_wins}  PPO wins={ppo_wins}  draws={draws}  (of {total})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] saved -> {args.out}")


if __name__ == "__main__":
    main()
