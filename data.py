"""
data.py
=======

Step 3 (part 1): build the Behavioral-Cloning dataset from the Lichess Elite
Database (strong players only: 2500+ vs 2300+).

Pipeline:  download a monthly .zip  ->  stream-parse the PGN it contains  ->
for every position, store (compact board, expert action) using the SAME
`board_to_index` / `move_to_action` helpers the environment uses.

Storing the compact (8,8) int8 board (not the 8x8x12 one-hot) keeps the dataset
tiny: ~64 bytes/position, so 5M positions is ~320 MB on disk instead of ~15 GB.
The one-hot expansion happens on the fly during training.

Elite DB files:  https://database.nikonoel.fr/lichess_elite_YYYY-MM.zip
Each zip contains a single .pgn.

Usage (CLI):
    python data.py --months 2024-11 2024-10 --max-positions 2000000
"""

from __future__ import annotations

import argparse
import io
import os
import time
import urllib.request
import zipfile
from typing import Iterable, Iterator, List, Optional, Tuple

import chess
import chess.pgn
import numpy as np

from chess_env import board_to_index, move_to_action

ELITE_URL = "https://database.nikonoel.fr/lichess_elite_{month}.zip"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_elite_month(month: str, dest_dir: str = DATA_DIR) -> str:
    """Download one Elite DB month (e.g. month='2024-11'); skip if present.

    Returns the local path to the .zip.
    """
    os.makedirs(dest_dir, exist_ok=True)
    url = ELITE_URL.format(month=month)
    path = os.path.join(dest_dir, f"lichess_elite_{month}.zip")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"[data] already have {path} ({os.path.getsize(path)/1e6:.1f} MB)")
        return path

    print(f"[data] downloading {url} ...")
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=120) as r, open(path, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)            # 1 MB at a time
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done/1e6:6.1f} / {total/1e6:.1f} MB", end="")
    print(f"\n[data] saved {path} in {time.time()-t0:.0f}s")
    return path


# ---------------------------------------------------------------------------
# PGN streaming
# ---------------------------------------------------------------------------
def iter_games_from_zip(zip_path: str) -> Iterator[chess.pgn.Game]:
    """Yield games from the single .pgn inside an Elite DB zip (streamed)."""
    with zipfile.ZipFile(zip_path) as zf:
        pgn_name = next(n for n in zf.namelist() if n.endswith(".pgn"))
        with zf.open(pgn_name) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", errors="ignore")
            while True:
                game = chess.pgn.read_game(text)
                if game is None:
                    break
                yield game


def _game_passes_filter(game: chess.pgn.Game, min_elo: int) -> bool:
    """Optional extra Elo gate (Elite DB is already strong, so usually a no-op)."""
    if min_elo <= 0:
        return True
    try:
        we = int(game.headers.get("WhiteElo", 0))
        be = int(game.headers.get("BlackElo", 0))
    except ValueError:
        return False
    return min(we, be) >= min_elo


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
def build_bc_dataset(
    zip_paths: Iterable[str],
    max_positions: int,
    min_elo: int = 0,
    max_plies_per_game: Optional[int] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk games and emit (boards_idx, actions, game_ids).

    For every ply we record the position *before* the move (encoded from the
    mover's perspective) and the expert move as a single action index. Both
    White's and Black's moves are kept -- the canonical encoding makes them
    colour-agnostic, which doubles the usable data.

    We also record a game id per position so the trainer can split by GAME
    (whole games held out for validation). Without this, positions from one game
    leak across train/val and inflate the accuracy, because consecutive
    positions in a game are highly correlated.

    Returns:
        boards_idx : (N, 8, 8) int8   -- compact board codes
        actions    : (N,)      int32  -- expert action in [0, 4096)
        game_ids   : (N,)      int32  -- which game each position came from
    """
    boards: List[np.ndarray] = []
    actions: List[int] = []
    game_ids: List[int] = []
    games_used = 0
    t0 = time.time()

    for zip_path in zip_paths:
        for game in iter_games_from_zip(zip_path):
            if not _game_passes_filter(game, min_elo):
                continue
            board = game.board()
            n_ply = 0
            for move in game.mainline_moves():
                boards.append(board_to_index(board))      # position before the move
                actions.append(move_to_action(board, move))
                game_ids.append(games_used)               # tag with current game id
                board.push(move)
                n_ply += 1
                if max_plies_per_game and n_ply >= max_plies_per_game:
                    break
                if len(boards) >= max_positions:
                    break
            games_used += 1
            if len(boards) >= max_positions:
                break
            if games_used % 1000 == 0:
                print(f"\r  games={games_used:,}  positions={len(boards):,}", end="")
        if len(boards) >= max_positions:
            break

    print(f"\n[data] built {len(boards):,} positions from {games_used:,} games "
          f"in {time.time()-t0:.0f}s")
    boards_arr = np.asarray(boards, dtype=np.int8)
    actions_arr = np.asarray(actions, dtype=np.int32)
    games_arr = np.asarray(game_ids, dtype=np.int32)
    return boards_arr, actions_arr, games_arr


def save_dataset(path: str, boards: np.ndarray, actions: np.ndarray,
                 games: Optional[np.ndarray] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if games is None:
        np.savez_compressed(path, boards=boards, actions=actions)
    else:
        np.savez_compressed(path, boards=boards, actions=actions, games=games)
    print(f"[data] saved {len(actions):,} positions -> {path} "
          f"({os.path.getsize(path)/1e6:.1f} MB)")


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Returns (boards, actions, games). `games` is None for older datasets
    that were built before per-game ids were recorded."""
    with np.load(path) as d:
        games = d["games"] if "games" in d.files else None
        return d["boards"], d["actions"], games


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build BC dataset from Lichess Elite DB")
    ap.add_argument("--months", nargs="+", default=["2024-11"],
                    help="Elite DB months, e.g. 2024-11 2024-10")
    ap.add_argument("--max-positions", type=int, default=2_000_000)
    ap.add_argument("--min-elo", type=int, default=0,
                    help="extra Elo gate (Elite DB is already 2500+/2300+)")
    ap.add_argument("--max-plies-per-game", type=int, default=None,
                    help="optional cap to diversify (avoid long-game bias)")
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "bc_dataset.npz"))
    args = ap.parse_args()

    zip_paths = [download_elite_month(m) for m in args.months]
    boards, actions, games = build_bc_dataset(
        zip_paths,
        max_positions=args.max_positions,
        min_elo=args.min_elo,
        max_plies_per_game=args.max_plies_per_game,
    )
    save_dataset(args.out, boards, actions, games)


if __name__ == "__main__":
    main()
