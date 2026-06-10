# Results summary

## Win/draw/loss vs the same random opponent (200 games)

| Agent | Win | Draw | Loss |
|---|---|---|---|
| Random baseline | 6% | 86% | 8% |
| Agent A - PPO (RL) | 26% | 70% | 4% |
| Agent B - BC | 49% | 51% | 0% |

## Head-to-head (colours swapped)

- BC wins: **70**  |  PPO wins: **0**  |  draws: **130**  (of 200)

## Agent B (BC) training
- best val top-1 move-match: **0.333**
- dataset size: 5,000,000 positions

## Agent A (PPO) training
- reward shaping: True
- total steps: 1,000,000
- best win-rate vs random: **28%**
