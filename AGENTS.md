# Repository Guidelines

## Project Structure & Module Organization

This repository implements a Q-learning Snake agent as a compact Python project.

- `game.py` contains the `SnakeEnv` environment, rendering constants, game rules, and 11-bit state encoding.
- `agent.py` contains `QLearningAgent`, Q-table management, action selection, learning updates, save/load helpers.
- `train.py` runs training episodes and writes generated artifacts such as `q_table.pkl`, `q_table_best.pkl`, and `scores.csv`.
- `watch.py` runs an interactive Pygame viewer for a trained or random agent.
- `plot.py` reads `scores.csv` and writes `training_progress.png` when `matplotlib` is available.

There is currently no dedicated `tests/` directory. Keep generated artifacts out of source changes unless they are intentionally part of an experiment result.

## Build, Test, and Development Commands

Use Python 3. Install runtime dependencies manually if no requirements file exists:

```bash
python -m pip install numpy pygame matplotlib
```

Common commands:

```bash
python train.py --no-render
```

Runs training without opening a Pygame window; this is the fastest local smoke test.

```bash
python train.py --load --no-render
```

Continues training from `q_table.pkl`.

```bash
python watch.py --episodes 5
```

Views a trained agent for five games. Add `--random` to compare against random actions.

```bash
python plot.py
```

Generates `training_progress.png` from `scores.csv`, or prints an ASCII fallback when `matplotlib` is missing.

## Coding Style & Naming Conventions

Follow the existing Python style: 4-space indentation, small modules, descriptive constants in `UPPER_CASE`, classes in `PascalCase`, and functions/variables in `snake_case`. Keep environment logic in `game.py`, learning logic in `agent.py`, and script orchestration in `train.py`, `watch.py`, or `plot.py`. Existing comments include Vietnamese explanations; preserve clarity and avoid large unrelated rewrites.

## Testing Guidelines

No formal test framework is configured. For changes to learning or game logic, run at least `python train.py --no-render` long enough to confirm episodes complete and artifacts are written. For rendering changes, run `python watch.py --episodes 1`. If adding tests, prefer `pytest`, place files under `tests/`, and name them `test_*.py`.

## Commit & Pull Request Guidelines

This checkout does not expose readable Git history, so use clear imperative commit messages such as `Add training progress plot` or `Fix snake collision reward`. Pull requests should include a concise summary, commands run, affected artifacts, and screenshots or short notes for visual changes in `watch.py` or Pygame rendering.

## Security & Configuration Tips

Q-tables are Python pickle files. Do not load `q_table.pkl` or `q_table_best.pkl` from untrusted sources. Avoid committing local virtual environments, caches, or large generated experiment outputs unless explicitly needed.
