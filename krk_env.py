"""
krk_env.py
==========

The "RL wins" crossover experiment: King + Rook vs King (KRK) checkmate.

This is the mirror image of full chess. Here the state space is tiny and we can
give a DENSE reward (drive the enemy king to a corner, bring our king up, bonus
for mate). That is exactly the regime where RL thrives -- and where Behavioral
Cloning struggles, because human game databases almost never contain KRK
technique (players resign long before). So:

    full chess  (huge space, sparse reward, abundant expert data)  -> BC wins
    KRK endgame (tiny space, dense reward, ~no expert data)         -> RL wins

Crucially we reuse the SAME observation encoding, action space, masking and
(later) the SAME network as the full-chess agents -- only the task and reward
change. That makes this a clean within-domain crossover.

The agent always plays White (King+Rook); Black (lone king) defends randomly.
"""

from __future__ import annotations

from typing import Optional

import chess
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from chess_env import (
    BOARD_SHAPE, NUM_ACTIONS, encode_board, action_to_move, get_action_mask,
)


def _cheb(a: int, b: int) -> int:
    """Chebyshev (king-move) distance between two squares."""
    return max(abs(chess.square_file(a) - chess.square_file(b)),
               abs(chess.square_rank(a) - chess.square_rank(b)))


_CORNERS = [chess.A1, chess.A8, chess.H1, chess.H8]


def _corner_dist(sq: int) -> int:
    """Distance from a square to the nearest corner (0 at a corner)."""
    return min(_cheb(sq, c) for c in _CORNERS)


def generate_krk_position(rng: np.random.Generator) -> chess.Board:
    """A random, legal KRK position with White to move and the rook safe.

    Constraints: pieces on distinct squares; kings not adjacent; the black king
    not adjacent to the rook (so White doesn't start by hanging it); the
    position is legal and not already terminal.
    """
    while True:
        wk, bk, wr = rng.choice(64, size=3, replace=False)
        if _cheb(wk, bk) < 2:                       # kings can't be adjacent
            continue
        if _cheb(bk, int(wr)) < 2:                  # keep the rook safe at start
            continue
        board = chess.Board(None)                   # empty board
        board.set_piece_at(int(wk), chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(int(wr), chess.Piece(chess.ROOK, chess.WHITE))
        board.set_piece_at(int(bk), chess.Piece(chess.KING, chess.BLACK))
        board.turn = chess.WHITE
        if board.is_valid() and not board.is_game_over() and any(board.legal_moves):
            return board


class KRKEndgameEnv(gym.Env):
    """King+Rook (agent, White) vs King (random defender). Goal: checkmate."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        max_moves: int = 100,                # plies before we give up (truncate)
        mate_reward: float = 10.0,
        draw_reward: float = -1.0,           # stalemate / rook lost / move cap
        corner_scale: float = 0.15,          # reward for cornering the black king
        edge_scale: float = 0.15,            # reward for driving it to an edge
        king_scale: float = 0.10,            # reward for bringing our king closer
        mobility_scale: float = 0.30,        # reward for confining the black king
        step_penalty: float = 0.02,          # encourages faster mates
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.max_moves = max_moves
        self.mate_reward = mate_reward
        self.draw_reward = draw_reward
        self.corner_scale = corner_scale
        self.edge_scale = edge_scale
        self.king_scale = king_scale
        self.mobility_scale = mobility_scale
        self.step_penalty = step_penalty
        self.render_mode = render_mode

        self.observation_space = spaces.Box(0.0, 1.0, shape=BOARD_SHAPE, dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.board = chess.Board()
        self._rng = np.random.default_rng()
        self._plies = 0

    # -- helpers ------------------------------------------------------------
    def _black_mobility(self) -> int:
        """How many squares the black king can flee to (lower = more confined).

        Computed as black's legal-move count, which for a lone king is exactly
        its escape squares. Used as the key 'mating net' signal."""
        b2 = self.board.copy(stack=False)
        b2.turn = chess.BLACK
        return sum(1 for _ in b2.legal_moves)

    def _potential(self) -> float:
        """Shaping potential (higher = closer to mate). Combines the classic KRK
        ideas: drive the black king to an edge/corner, confine its mobility, and
        bring our own king up to support the mate.

        Honesty note: this is a HAND-ENGINEERED mating heuristic, and the reward
        is applied as Phi(s') - Phi(s) (the strict policy-invariant form of Ng et
        al. 1999 multiplies Phi(s') by gamma; with gamma=0.99 the difference is
        negligible). So "RL learned to mate" means "RL optimised a human-designed
        dense reward" -- which is exactly the dense-reward regime we are
        demonstrating, but it is not reward-free discovery."""
        bk = self.board.king(chess.BLACK)
        wk = self.board.king(chess.WHITE)
        edge_dist = min(chess.square_file(bk), 7 - chess.square_file(bk),
                        chess.square_rank(bk), 7 - chess.square_rank(bk))
        return (-self.corner_scale * _corner_dist(bk)
                - self.edge_scale * edge_dist
                - self.king_scale * _cheb(wk, bk)
                - self.mobility_scale * self._black_mobility())

    def _black_move(self):
        """Random legal move for the lone black king."""
        moves = list(self.board.legal_moves)
        self.board.push(moves[self._rng.integers(len(moves))])

    # -- Gymnasium API ------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self.board = generate_krk_position(self._rng)
        self._plies = 0
        return encode_board(self.board), {"fen": self.board.fen()}

    def step(self, action: int):
        pot_before = self._potential()

        move = action_to_move(self.board, action)
        if move is None:
            return encode_board(self.board), -1.0, True, False, {"illegal": True}
        self.board.push(move)
        self._plies += 1

        # White's move may deliver mate (win) or stalemate (failure).
        if self.board.is_game_over():
            return self._terminal()

        # Black (lone king) replies; it may grab the rook -> K vs K draw (fail).
        self._black_move()
        self._plies += 1
        if self.board.is_game_over():
            return self._terminal()

        # Non-terminal: dense shaping + small step cost.
        reward = (self._potential() - pot_before) - self.step_penalty
        truncated = self._plies >= self.max_moves
        if truncated:
            reward += self.draw_reward            # ran out of time = failed to mate
        return encode_board(self.board), reward, False, truncated, {"fen": self.board.fen()}

    def _terminal(self):
        outcome = self.board.outcome(claim_draw=True)
        if outcome is not None and outcome.winner == chess.WHITE:
            reward = self.mate_reward             # checkmate achieved
        else:
            reward = self.draw_reward             # stalemate / insufficient material
        return encode_board(self.board), reward, True, False, {"fen": self.board.fen()}

    def action_masks(self) -> np.ndarray:
        return get_action_mask(self.board)

    def render(self):
        if self.render_mode == "ansi":
            return str(self.board)
        print(self.board, "\n")


# ---------------------------------------------------------------------------
# Shared KRK evaluation: run a chooser on a FIXED set of positions.
# (A "chooser" is board -> action, same interface as evaluate.py.)
# ---------------------------------------------------------------------------
def evaluate_krk(chooser, n_positions: int, max_moves: int = 100, seed: int = 2024):
    """Return mate-rate and stats for `chooser` over n_positions KRK games.

    Both agents are scored on the SAME positions/seeds for a fair comparison.
    max_moves matches the training env's ply cap (100) so eval and training agree.
    """
    mates = draws = 0
    plies_to_mate = []
    for g in range(n_positions):
        env = KRKEndgameEnv(max_moves=max_moves)
        obs, _ = env.reset(seed=seed + g)
        done = False
        mated = False
        while not done:
            action = chooser(env.board)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if terminated and reward == env.mate_reward:
                mated = True
        if mated:
            mates += 1
            plies_to_mate.append(env._plies)
        else:
            draws += 1
    return {
        "positions": n_positions,
        "mate_rate": mates / n_positions,
        "fail_rate": draws / n_positions,
        "avg_plies_to_mate": float(np.mean(plies_to_mate)) if plies_to_mate else None,
    }


# ---------------------------------------------------------------------------
# Smoke test:  python krk_env.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from chess_env import get_action_mask as _mask
    env = KRKEndgameEnv()
    obs, _ = env.reset(seed=0)
    print("Start position (White=K+R to mate Black):")
    env.render()
    print("obs shape:", obs.shape, " legal actions:", int(env.action_masks().sum()))

    # Random-vs-random mate rate (the floor a learned agent must beat).
    def random_chooser(board):
        legal = np.flatnonzero(_mask(board))
        return int(legal[np.random.default_rng().integers(len(legal))])

    stats = evaluate_krk(random_chooser, 200)
    print("\nRandom White over 200 KRK positions:", stats)
