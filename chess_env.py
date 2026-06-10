"""
chess_env.py
============

Step 1 of the project: a lightweight Gymnasium environment that wraps
`python-chess`, plus the board-encoding / action-space helpers that BOTH agents
(the RL agent in Step 4 and the Behavioral-Cloning agent in Step 3) will share.

Keeping the encoding and the action mapping as standalone functions is the key
design decision: it guarantees Agent A (RL) and Agent B (BC) see *exactly* the
same observations and speak *exactly* the same action language, which is what
makes the final comparison fair.

------------------------------------------------------------------------------
Install (Colab / local):
    pip install chess gymnasium numpy
    # For Step 4 (RL with action masking) you will also want:
    #     pip install stable-baselines3 sb3-contrib
------------------------------------------------------------------------------

Design choices (deliberately simple for a university minor project):

* Observation: an 8x8x12 one-hot tensor.
    - 12 planes = 6 piece types (pawn, knight, bishop, rook, queen, king)
      x 2 colors.
    - The board is always shown from the *side-to-move's* perspective
      ("my pieces are always the bottom 6 planes, at the bottom of the board").
      This is the standard canonical-form trick: it halves what the network
      must learn and lets a single network play either colour. (Note: the RL
      agent here trains against a FIXED opponent, not via self-play.)

* Action space: Discrete(4096) = 64 from-squares x 64 to-squares.
    - Promotions default to a queen (under-promotions are rare; ignoring them
      is a documented simplification).
    - Because most of the 4096 actions are illegal in any given position, the
      env exposes `action_masks()` so we can use MaskablePPO (sb3-contrib) and
      so BC can argmax over *legal* moves only.

What we intentionally DO NOT encode (simplifications, fine for this scope):
castling rights, en-passant square, the 50-move counter, and repetition
history. The network sees only piece placement + whose turn it is (implicit in
the canonical flip).
"""

from __future__ import annotations

from typing import Callable, Optional

import chess
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BOARD_SHAPE = (8, 8, 12)          # rank, file, piece-plane
NUM_ACTIONS = 64 * 64             # 4096: (from_square, to_square)

# Order of piece types in the plane dimension. python-chess uses
# PAWN=1, KNIGHT=2, BISHOP=3, ROOK=4, QUEEN=5, KING=6, so plane = type - 1.
_PIECE_TYPES = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP,
    chess.ROOK, chess.QUEEN, chess.KING,
]

# Standard material values (in pawn units) for the optional shaping reward.
# The king is omitted (its loss is the game-ending signal, not a material swing).
_PIECE_VALUES = {
    chess.PAWN: 1.0, chess.KNIGHT: 3.0, chess.BISHOP: 3.0,
    chess.ROOK: 5.0, chess.QUEEN: 9.0, chess.KING: 0.0,
}   






# ---------------------------------------------------------------------------
# Shared, reusable primitives  (used by BOTH agents)
# ---------------------------------------------------------------------------
def board_to_index(board: chess.Board) -> np.ndarray:
    """Compact canonical board: an (8, 8) int8 grid of piece codes.

    This is the storage-friendly form (64 bytes/position vs. 3 KB for the
    one-hot tensor) used to hold large BC datasets. `encode_board` is just this
    plus a one-hot expansion, so the two representations can never drift apart.

    The position is rendered from the side-to-move's perspective (Black-to-move
    boards are mirrored), so the encoding is colour-agnostic.

    Cell codes:
        0      : empty
        1-6    : current player's  pawn, knight, bishop, rook, queen, king
        7-12   : opponent's        pawn, knight, bishop, rook, queen, king
    """
    # Canonical view: always "white to move at the bottom".
    cb = board if board.turn == chess.WHITE else board.mirror()

    idx = np.zeros((8, 8), dtype=np.int8)
    for square, piece in cb.piece_map().items():
        rank = chess.square_rank(square)      # 0 = rank 1 (bottom)
        file = chess.square_file(square)       # 0 = file a
        plane = _PIECE_TYPES.index(piece.piece_type)
        if piece.color == chess.BLACK:         # opponent in the canonical view
            plane += 6
        idx[rank, file] = plane + 1           # +1 so that 0 stays "empty"
    return idx


def index_to_onehot(idx: np.ndarray) -> np.ndarray:
    """Expand compact codes to one-hot. Accepts (8,8) or batched (N,8,8).

    Returns float32 with a trailing size-12 axis; the "empty" code (0) maps to
    an all-zero vector (the empty plane is dropped).
    """
    idx = np.asarray(idx, dtype=np.int64)
    onehot = np.zeros(idx.shape + (13,), dtype=np.float32)
    np.put_along_axis(onehot, idx[..., None], 1.0, axis=-1)
    return onehot[..., 1:]                      # drop the empty plane -> 12 planes


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a board as an (8, 8, 12) float32 one-hot tensor (side-to-move view).

    Plane layout (last axis):
        0-5  : current player's  pawn, knight, bishop, rook, queen, king
        6-11 : opponent's        pawn, knight, bishop, rook, queen, king
    """
    return index_to_onehot(board_to_index(board))


def move_to_action(board: chess.Board, move: chess.Move) -> int:
    """Map a real `chess.Move` to its integer action in the canonical frame.

    Inverse of `action_to_move`. Promotion piece is dropped (collapsed to the
    single from->to action), which is consistent with our queen-default rule.
    """
    if board.turn == chess.WHITE:
        from_sq, to_sq = move.from_square, move.to_square
    else:
        # In the canonical (mirrored) frame squares are vertically flipped.
        from_sq = chess.square_mirror(move.from_square)
        to_sq = chess.square_mirror(move.to_square)
    return from_sq * 64 + to_sq


def action_to_move(board: chess.Board, action: int) -> Optional[chess.Move]:
    """Map an integer action back to a legal `chess.Move` on the real board.

    Returns None if the action does not correspond to any legal move (useful
    for defensive checks; with action masking this should not happen).
    """
    from_sq, to_sq = divmod(int(action), 64)

    if board.turn == chess.BLACK:
        # Undo the canonical mirror to get back to real-board squares.
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)

    # Try the plain move first, then the queen-promotion variant.
    plain = chess.Move(from_sq, to_sq)
    if plain in board.legal_moves:
        return plain
    promo = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
    if promo in board.legal_moves:
        return promo
    return None


def get_action_mask(board: chess.Board) -> np.ndarray:
    """Boolean mask of length 4096; True where the action is a legal move."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for move in board.legal_moves:
        mask[move_to_action(board, move)] = True
    return mask


# ---------------------------------------------------------------------------
# Opponents (the env plays the agent against one of these)
# ---------------------------------------------------------------------------
def random_opponent(board: chess.Board, rng: np.random.Generator) -> chess.Move:
    """Pick a uniformly random legal move. The default fixed opponent."""
    legal = list(board.legal_moves)
    return legal[rng.integers(len(legal))]


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------
class ChessEnv(gym.Env):
    """A single-agent chess environment.

    The agent plays one colour; a fixed `opponent` callable plays the other.
    Every observation returned is a position where it is the agent's turn, so
    the agent never has to reason about whose move it is.

    Reward (from the agent's perspective):
        +1  win, -1  loss, 0  draw  (terminal, always pure).
    Optional dense shaping (reward_shaping=True): on each non-terminal step the
    agent also receives `shaping_scale * (net material swing this step)`, i.e. it
    is rewarded for winning material and penalised for hanging it. The scale is
    small so terminal win/loss always dominates.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        opponent: Callable[[chess.Board, np.random.Generator], chess.Move] = random_opponent,
        agent_color: bool = chess.WHITE,
        max_moves: int = 200,
        reward_shaping: bool = False,
        shaping_scale: float = 0.01,
        illegal_move_penalty: float = -1.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.opponent = opponent
        self.agent_color = agent_color
        self.max_moves = max_moves          # ply cap -> truncation (avoids endless games)
        self.reward_shaping = reward_shaping
        self.shaping_scale = shaping_scale   # pawn-unit swing -> reward multiplier
        self.illegal_move_penalty = illegal_move_penalty
        self.render_mode = render_mode

        # Gymnasium spaces. MlpPolicy / our dense net will flatten the obs.
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=BOARD_SHAPE, dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.board = chess.Board()
        self._np_random_gen = np.random.default_rng()

    # -- Gymnasium API ------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        # Use Gymnasium's seeded RNG for full reproducibility.
        self._np_random_gen = np.random.default_rng(seed)
        self.board = chess.Board()

        # If the agent plays Black, let the opponent (White) make the first move.
        if self.board.turn != self.agent_color:
            self._opponent_move()

        return encode_board(self.board), self._info()

    def step(self, action: int):
        # Material before the agent moves (for the optional shaping reward).
        mat_before = self._material() if self.reward_shaping else 0.0

        # 1) Apply the agent's move.
        move = action_to_move(self.board, action)
        if move is None:
            # Should not occur when action masking is used; handled defensively.
            obs = encode_board(self.board)
            return obs, self.illegal_move_penalty, True, False, self._info(illegal=True)
        self.board.push(move)

        # 2) Did the agent's move end the game?
        if self.board.is_game_over():
            return self._terminal_return()

        # 3) Opponent replies.
        self._opponent_move()
        if self.board.is_game_over():
            return self._terminal_return()

        # 4) Non-terminal step: optional material-swing shaping + ply-cap truncation.
        reward = 0.0
        if self.reward_shaping:
            reward = self.shaping_scale * (self._material() - mat_before)
        truncated = self.board.fullmove_number * 2 >= self.max_moves
        return encode_board(self.board), reward, False, truncated, self._info()

    def render(self):
        if self.render_mode == "ansi":
            return str(self.board)
        print(self.board, "\n")

    # -- Action masking (for sb3-contrib MaskablePPO and BC) ----------------
    def action_masks(self) -> np.ndarray:
        """Required name for MaskablePPO; True where actions are legal."""
        return get_action_mask(self.board)

    # -- Internals ----------------------------------------------------------
    def _opponent_move(self):
        move = self.opponent(self.board, self._np_random_gen)
        self.board.push(move)

    def _terminal_return(self):
        reward = self._result_reward()
        return encode_board(self.board), reward, True, False, self._info()

    def _result_reward(self) -> float:
        """+1 / -1 / 0 from the agent's perspective at game end."""
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == self.agent_color else -1.0

    def _material(self) -> float:
        """Material balance in pawn units, from the agent's perspective
        (agent's pieces minus opponent's)."""
        total = 0.0
        for piece in self.board.piece_map().values():
            value = _PIECE_VALUES[piece.piece_type]
            total += value if piece.color == self.agent_color else -value
        return total

    def _info(self, illegal: bool = False) -> dict:
        return {
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn else "black",
            "illegal": illegal,
        }


# ---------------------------------------------------------------------------
# Quick self-test:  python chess_env.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    env = ChessEnv()
    obs, info = env.reset(seed=0)
    print("Observation shape:", obs.shape)          # (8, 8, 12)
    print("Action space:", env.action_space)        # Discrete(4096)
    print("Legal actions at start:", int(env.action_masks().sum()), "of", NUM_ACTIONS)

    # Play one full game where BOTH sides move randomly, but the agent's moves
    # are sampled only from the legal (masked) actions -- exactly how the
    # trained agents will act later.
    rng = np.random.default_rng(0)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        mask = env.action_masks()
        legal_actions = np.flatnonzero(mask)
        action = legal_actions[rng.integers(len(legal_actions))]
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

    print(f"\nGame finished after {steps} agent-steps.")
    print(f"Terminated={terminated}  Truncated={truncated}  Final reward={reward}")
    print("Final position:\n")
    env.render()
