# turbo-winner — Snake vs AI Red Circle

A graphical Snake game with animated visuals where **you control the
snake** and an AI-controlled red circle learns to evade you from
scratch — no ML frameworks, no game engines, no external packages.
Everything is built with the Python standard library only.

## What's inside

| File | Description |
|---|---|
| `game.py` | Snake-vs-evader logic — 20 × 20 grid, collision detection, AI state vector |
| `neural_net.py` | Feedforward neural network (ReLU, Xavier init, pure Python) |
| `trainer.py` | Genetic algorithm — uniform crossover, Gaussian mutation, elitism |
| `app.py` | Tkinter visualisation — manual snake controls, animated canvas, training stats |
| `main.py` | Entry point |
| `pyproject.toml` | `uv` project configuration (zero external dependencies) |

## How the AI works

1. **Population** — 150 red-circle agents, each controlled by a small
   neural network (`12 → 32 → 16 → 5`).
2. **Inputs (12)** — danger in each direction, relative snake-head
   position, and the snake's current heading.
3. **Outputs (5)** — stay, move up, move right, move down, move left.
4. **Fitness** — survival time plus distance kept from the snake, with
   bonuses when the snake crashes or the agent survives the full round.
5. **Evolution** — after every agent has finished a round, the top 10 %
   are kept unchanged and the rest are bred via uniform crossover and
   Gaussian mutation.
6. **Learning** — in watch mode you can steer the snake yourself; in
   fast mode a chase bot takes over so the red circle can train quickly.

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
| **Arrow keys / WASD** | Steer the snake manually |
| **Speed slider** | Adjust animation speed (right = faster) |
| **⚡ Fast** | Toggle fast training mode — a chase bot controls the snake and rendering is skipped |
| **↺ Restart** | Reset all training and start from generation 1 |

## Screenshots / animation

The dark canvas shows the snake (cyan head, blue body gradient) chasing
the pink pulsing red-circle agent. The right-hand panel displays live
training statistics and a bar chart of the best survival time achieved
each generation.