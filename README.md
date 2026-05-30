# SnakeRF

SnakeRF is a small Python project that trains a Snake-playing agent with tabular Q-learning. The game environment exposes an 11-bit state representation, so the agent can learn with a compact Q-table instead of a neural network.

The repository includes scripts for training, watching the trained agent play, and plotting training progress.

## Features

- Classic Snake environment built with Pygame
- Q-learning agent with epsilon-greedy exploration
- Compact 11-bit state space
- Saved Q-table support for resuming training
- Visual playback mode with live state and Q-value HUD
- Training score logging and progress plotting

## Project Structure

```text
.
├── agent.py            # Q-learning agent and Q-table save/load logic
├── game.py             # Snake environment, game rules, rendering, state encoding
├── train.py            # Training loop
├── watch.py            # Watch a trained or random agent play
├── plot.py             # Plot training results from scores.csv
├── q_table.pkl         # Latest saved Q-table
├── q_table_best.pkl    # Best saved Q-table
└── scores.csv          # Training history
```

## Requirements

- Python 3.10+
- `numpy`
- `pygame`
- `matplotlib` optional, used by `plot.py`

Install dependencies:

```bash
python -m pip install numpy pygame matplotlib
```

## Usage

Train from scratch:

```bash
python train.py --no-render
```

Train with occasional rendering:

```bash
python train.py
```

Continue training from `q_table.pkl`:

```bash
python train.py --load --no-render
```

Watch the trained agent:

```bash
python watch.py --episodes 5
```

Compare with a random agent:

```bash
python watch.py --random --episodes 5
```

Plot training progress:

```bash
python plot.py
```

When `matplotlib` is installed, this writes `training_progress.png`. Otherwise, it prints an ASCII summary.

## How It Works

The agent receives a tuple of 11 binary values describing immediate danger, current direction, and food position relative to the snake. It chooses one of three relative actions:

- `0`: go straight
- `1`: turn right
- `2`: turn left

`agent.py` updates the Q-table with the Bellman equation after each move. `train.py` decays exploration over time and saves checkpoints to `q_table.pkl`; the best-scoring table is saved to `q_table_best.pkl`.

## Notes for GitHub

The `.pkl` files are Python pickle files. Only load Q-tables from sources you trust. If you want a lighter repository, remove generated files such as `q_table.pkl`, `q_table_best.pkl`, `scores.csv`, `training_progress.png`, and `__pycache__/` before publishing, then add them to `.gitignore`.

## License

No license has been specified yet. Add a license file before publishing if you want others to reuse or modify this project.
