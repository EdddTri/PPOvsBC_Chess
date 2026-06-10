"""
network.py
==========

Step 2: the SHARED neural-network architecture used by both agents.

The whole project hinges on this file: Agent A (RL/PPO) and Agent B (Behavioral
Cloning) must use the *same* network so the comparison is about the learning
paradigm, not the model capacity. We enforce that by defining one shared "trunk"
(the body of the network) and reusing it in both places:

        input (8x8x12 = 768)
                |
        +-------------------+
        |   SHARED TRUNK    |   Flatten -> Linear(768,512) -> ReLU
        |                   |           -> Linear(512,256) -> ReLU
        +-------------------+   (256-d feature vector)
            /            \
    BC policy head     PPO heads
    Linear(256,4096)   policy: Linear(256,4096)   value: Linear(256,1)

* For BC (Step 3) we attach a single Linear(256 -> 4096) classification head and
  train with cross-entropy against expert moves.
* For PPO (Step 4) we hand the SAME trunk to Stable-Baselines3 as a
  `features_extractor`, and tell SB3 to put only LINEAR policy/value heads on
  top (`net_arch=[]`). That makes the PPO policy head architecturally identical
  to the BC head: Linear(256 -> 4096).

So both agents share: Flatten -> 512 -> 256 trunk, and an identical
Linear(256 -> 4096) action head. Only the training signal differs.

------------------------------------------------------------------------------
Install:
    d:/Reinforcement_Project/venv/Scripts/python.exe -m pip install torch
    # RL also needs:  pip install stable-baselines3 sb3-contrib
------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from chess_env import BOARD_SHAPE, NUM_ACTIONS

# Default architecture knobs (kept small so it trains on free Colab / CPU).
INPUT_DIM = int(np.prod(BOARD_SHAPE))      # 8 * 8 * 12 = 768
HIDDEN_SIZES: Tuple[int, ...] = (512, 256)


# ---------------------------------------------------------------------------
# The shared trunk (body) — the single source of truth for both agents.
# ---------------------------------------------------------------------------
def build_trunk(
    input_dim: int = INPUT_DIM,
    hidden_sizes: Sequence[int] = HIDDEN_SIZES,
) -> Tuple[nn.Sequential, int]:
    """Build the shared body: Flatten -> [Linear -> ReLU] x len(hidden_sizes).

    Returns the module and the output feature dimension. Flatten is included so
    the trunk accepts raw (B, 8, 8, 12) observations directly.
    """
    layers: list[nn.Module] = [nn.Flatten()]
    last = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last, h))
        layers.append(nn.ReLU())
        last = h
    return nn.Sequential(*layers), last


def count_params(module: nn.Module) -> int:
    """Total number of trainable parameters (for the 'same size' sanity check)."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Agent B's network: shared trunk + linear classification head.
# ---------------------------------------------------------------------------
class BCNet(nn.Module):
    """Behavioral-Cloning network: board -> logits over 4096 actions.

    Trained as a classifier (cross-entropy) to imitate expert moves in Step 3.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        n_actions: int = NUM_ACTIONS,
        hidden_sizes: Sequence[int] = HIDDEN_SIZES,
    ):
        super().__init__()
        self.trunk, feat_dim = build_trunk(input_dim, hidden_sizes)
        self.policy_head = nn.Linear(feat_dim, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 8, 8, 12) float tensor -> logits (B, 4096)."""
        return self.policy_head(self.trunk(x))

    @torch.no_grad()
    def predict_action(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        """Greedy action over LEGAL moves only.

        obs:         (8, 8, 12) float array  (from chess_env.encode_board)
        action_mask: (4096,)   bool array    (from chess_env.get_action_mask)
        """
        self.eval()
        device = next(self.parameters()).device
        x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits = self.forward(x).squeeze(0)                      # (4096,)
        mask = torch.as_tensor(action_mask, dtype=torch.bool, device=device)
        # Set illegal actions to -inf so argmax can never pick them.
        logits = logits.masked_fill(~mask, float("-inf"))
        return int(torch.argmax(logits).item())


# ---------------------------------------------------------------------------
# Agent A's network: hand the SAME trunk to Stable-Baselines3.
# Imports are guarded so this module still works for BC if SB3 isn't installed.
# ---------------------------------------------------------------------------
try:
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class ChessFeaturesExtractor(BaseFeaturesExtractor):
        """Wraps the shared trunk as an SB3 features extractor.

        SB3 then adds the policy/value heads. With `net_arch=[]` (see
        `ppo_policy_kwargs`) those heads are plain Linear layers on top of the
        256-d features — identical in shape to BCNet's head.
        """

        def __init__(self, observation_space, hidden_sizes: Sequence[int] = HIDDEN_SIZES):
            input_dim = int(np.prod(observation_space.shape))
            trunk, feat_dim = build_trunk(input_dim, hidden_sizes)
            # BaseFeaturesExtractor needs features_dim before we register modules.
            super().__init__(observation_space, features_dim=feat_dim)
            self.trunk = trunk

        def forward(self, observations: torch.Tensor) -> torch.Tensor:
            return self.trunk(observations)

    def ppo_policy_kwargs(hidden_sizes: Sequence[int] = HIDDEN_SIZES) -> dict:
        """policy_kwargs for (Maskable)PPO so it reuses the shared trunk.

        net_arch=[] -> no extra hidden layers; policy/value heads are linear on
        the shared 256-d features, matching BCNet's Linear(256, 4096) head.
        """
        return dict(
            features_extractor_class=ChessFeaturesExtractor,
            features_extractor_kwargs=dict(hidden_sizes=tuple(hidden_sizes)),
            net_arch=[],
        )

    SB3_AVAILABLE = True

except ImportError:  # SB3 not installed yet — BC still fully usable.
    SB3_AVAILABLE = False


# ---------------------------------------------------------------------------
# Self-test:  python network.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) BC network: shape + parameter count.
    bc = BCNet()
    dummy = torch.zeros(4, *BOARD_SHAPE)          # batch of 4 boards
    out = bc(dummy)
    print("BCNet output shape:", tuple(out.shape))            # (4, 4096)
    print("BCNet total params:", f"{count_params(bc):,}")
    print("  - trunk params :", f"{count_params(bc.trunk):,}")
    print("  - head  params :", f"{count_params(bc.policy_head):,}")

    # 2) Confirm a masked greedy prediction returns a legal action.
    from chess_env import ChessEnv
    env = ChessEnv()
    obs, _ = env.reset(seed=0)
    a = bc.predict_action(obs, env.action_masks())
    print("\nGreedy legal action from start position:", a,
          "(legal:", bool(env.action_masks()[a]), ")")

    # 3) If SB3 is available, build a MaskablePPO and prove the trunk is shared.
    if SB3_AVAILABLE:
        try:
            from sb3_contrib import MaskablePPO

            model = MaskablePPO(
                "MlpPolicy", env, policy_kwargs=ppo_policy_kwargs(), verbose=0
            )
            extractor = model.policy.features_extractor
            print("\nSB3 available -> MaskablePPO built.")
            print("PPO shared-trunk params:", f"{count_params(extractor):,}")
            print("Matches BCNet trunk    :", count_params(extractor) == count_params(bc.trunk))
            print("PPO action net:", model.policy.action_net)   # Linear(256, 4096)
            print("PPO value  net:", model.policy.value_net)    # Linear(256, 1)
        except ImportError:
            print("\nsb3-contrib not installed; skipping PPO check.")
    else:
        print("\nstable-baselines3 not installed; BC works, PPO check skipped.")
