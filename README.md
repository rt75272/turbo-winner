# turbo-winner — Snake AI

A graphical Snake game with animated visuals and an AI that **learns to
master the game entirely from scratch** — no ML frameworks, no game
engines, no external packages. Everything is built with the Python
standard library only.

## What's inside

| File | Description |
|---|---|
| `game.py` | Snake game logic — 20 × 20 grid, collision detection, state vector |
| `neural_net.py` | Feedforward neural network (ReLU, Xavier init, pure Python) |
| `trainer.py` | Genetic algorithm — tournament selection, uniform crossover, Gaussian mutation, elitism |
| `app.py` | Tkinter visualisation — animated game canvas, stats panel, score chart |
| `main.py` | Entry point |
| `pyproject.toml` | `uv` project configuration (zero external dependencies) |

## How the AI works

1. **Population** — 150 snakes, each controlled by a small neural
   network (`11 → 32 → 16 → 3`).
2. **Inputs (11)** — danger straight/right/left, current heading
   (one-hot), food direction (left/right/up/down).
3. **Outputs (3)** — go straight, turn right, turn left.
4. **Fitness** — `score² × 500 + steps_survived` (strongly rewards
   eating more food).
5. **Evolution** — after every snake has died, the top 10 % are kept
   unchanged (elitism), the rest are bred from the top 50 % via uniform
   crossover and Gaussian mutation.
6. **Learning** — each generation the population improves; within ~50
   generations most snakes reliably find and eat food.

## Requirements

- Python ≥ 3.9
- `tkinter` (ships with most Python installers; on Debian/Ubuntu:
  `sudo apt-get install python3-tk`)
- [`uv`](https://github.com/astral-sh/uv) (recommended, but optional)

## Running

```bash
# With uv (recommended)
uv run python main.py

# Or plain Python
python main.py
```

## Controls

| Control | Description |
|---|---|
| **Speed slider** | Adjust animation speed (right = faster) |
| **⚡ Fast** | Toggle headless training mode — runs ~60× faster, no rendering |
| **↺ Restart** | Reset all training and start from generation 1 |

## Screenshots / animation

The dark canvas shows the snake (cyan head, blue body gradient) chasing
pink pulsing food. The right-hand panel displays live statistics and a
bar chart of the best score achieved each generation — you can watch the
AI improve in real time.