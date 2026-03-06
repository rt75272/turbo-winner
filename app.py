"""
Snake AI — Tkinter visualisation and training loop.

Only Python's standard library is used (tkinter, math, random, time).
No external packages or libraries are required.

Controls
--------
Speed slider : adjusts animation delay (right = faster)
⚡ Fast      : toggles headless fast-training mode (skips rendering)
↺ Restart    : resets all training state and starts over

The right-hand panel shows:
  • Generation number
  • Current agent index
  • Live score / high score
  • Best and average fitness this generation
  • A history chart of the best score per generation
"""

import math
import tkinter as tk

from game import GRID_H, GRID_W, SnakeGame
from neural_net import NeuralNetwork
from trainer import GeneticAlgorithm

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
CELL_SIZE: int = 26          # pixels per grid cell
MARGIN: int = 10             # border around the game canvas
PANEL_W: int = 240           # width of the stats panel

GAME_PX_W: int = GRID_W * CELL_SIZE
GAME_PX_H: int = GRID_H * CELL_SIZE

WINDOW_W: int = GAME_PX_W + 2 * MARGIN + PANEL_W
WINDOW_H: int = GAME_PX_H + 2 * MARGIN + 60   # extra room for controls

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_BG          = "#0d0d1e"
C_CANVAS_BG   = "#111128"
C_GRID        = "#16163a"
C_HEAD        = "#4cc9f0"
C_BODY_DARK   = "#3a0ca3"
C_BODY_LIGHT  = "#4361ee"
C_FOOD        = "#f72585"
C_FOOD_RING   = "#ff4da6"
C_PANEL_BG    = "#13132b"
C_TEXT        = "#dde1f8"
C_DIM         = "#555577"
C_ACCENT      = "#7209b7"
C_CHART_BAR   = "#4361ee"
C_CHART_LAST  = "#4cc9f0"
C_DEAD        = "#f72585"

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
LAYER_SIZES     = [11, 32, 16, 3]   # 11 inputs → two hidden layers → 3 actions
POPULATION_SIZE = 150


class App:
    """
    Main application: manages the Tkinter window, the training loop, and
    the real-time game visualisation.
    """

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Snake AI — Genetic Algorithm")
        self.root.configure(bg=C_BG)
        self.root.resizable(False, False)

        self._anim_tick: int = 0   # drives pulsing animations

        self._build_ui()
        self._init_training()
        self._schedule_next()
        self.root.mainloop()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Construct all Tkinter widgets."""
        outer = tk.Frame(self.root, bg=C_BG)
        outer.pack(fill=tk.BOTH, expand=True, padx=MARGIN, pady=MARGIN)

        # ---- Left column: canvas + controls ----
        left = tk.Frame(outer, bg=C_BG)
        left.pack(side=tk.LEFT, fill=tk.Y)

        self.canvas = tk.Canvas(
            left,
            width=GAME_PX_W,
            height=GAME_PX_H,
            bg=C_CANVAS_BG,
            highlightthickness=2,
            highlightbackground=C_ACCENT,
        )
        self.canvas.pack()

        ctrl = tk.Frame(left, bg=C_BG, pady=6)
        ctrl.pack(fill=tk.X)

        tk.Label(ctrl, text="Speed:", bg=C_BG, fg=C_TEXT,
                 font=("Courier", 10)).pack(side=tk.LEFT)

        self._speed_var = tk.IntVar(value=80)
        tk.Scale(
            ctrl, from_=5, to=200, orient=tk.HORIZONTAL,
            variable=self._speed_var, length=140,
            bg=C_BG, fg=C_TEXT, troughcolor=C_ACCENT,
            highlightthickness=0, showvalue=False,
        ).pack(side=tk.LEFT, padx=4)

        self._fast_btn = tk.Button(
            ctrl, text="⚡ Fast", command=self._toggle_fast,
            bg=C_ACCENT, fg=C_TEXT, font=("Courier", 10),
            relief=tk.FLAT, padx=8, cursor="hand2",
        )
        self._fast_btn.pack(side=tk.LEFT, padx=4)

        tk.Button(
            ctrl, text="↺ Restart", command=self._restart,
            bg="#333355", fg=C_TEXT, font=("Courier", 10),
            relief=tk.FLAT, padx=8, cursor="hand2",
        ).pack(side=tk.LEFT, padx=4)

        # ---- Right column: stats panel ----
        panel = tk.Frame(outer, bg=C_PANEL_BG, width=PANEL_W, padx=14, pady=14)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(MARGIN, 0))
        panel.pack_propagate(False)

        tk.Label(
            panel, text="🐍  SNAKE  AI",
            bg=C_PANEL_BG, fg=C_ACCENT,
            font=("Courier", 15, "bold"),
        ).pack(pady=(0, 14))

        # Stat rows
        self._stat: dict[str, tk.Label] = {}
        rows = [
            ("generation",  "Generation"),
            ("agent",       "Agent"),
            ("score",       "Score"),
            ("high_score",  "High Score"),
            ("best_gen",    "Best (gen)"),
            ("avg_fitness", "Avg Fitness"),
            ("steps",       "Steps"),
            ("total_games", "Total Games"),
        ]
        for key, label in rows:
            row = tk.Frame(panel, bg=C_PANEL_BG)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=f"{label}:", bg=C_PANEL_BG, fg=C_DIM,
                     font=("Courier", 9), width=13, anchor=tk.W).pack(side=tk.LEFT)
            lbl = tk.Label(row, text="0", bg=C_PANEL_BG, fg=C_TEXT,
                           font=("Courier", 9, "bold"), anchor=tk.W)
            lbl.pack(side=tk.LEFT)
            self._stat[key] = lbl

        # Chart
        tk.Label(
            panel, text="\nBest Score per Generation",
            bg=C_PANEL_BG, fg=C_DIM, font=("Courier", 8),
        ).pack()
        self._chart = tk.Canvas(
            panel, width=PANEL_W - 28, height=90,
            bg="#0a0a1c", highlightthickness=1,
            highlightbackground=C_ACCENT,
        )
        self._chart.pack(pady=(2, 10))
        self._score_history: list[int] = []

        # Network info
        tk.Label(
            panel,
            text=f"Network: {LAYER_SIZES}\nPopulation: {POPULATION_SIZE}",
            bg=C_PANEL_BG, fg=C_DIM, font=("Courier", 8),
        ).pack()

    # ------------------------------------------------------------------
    # Training state
    # ------------------------------------------------------------------

    def _init_training(self) -> None:
        """Initialise (or reinitialise) all training-related state."""
        self._generation: int = 1
        self._agent_idx: int = 0
        self._total_games: int = 0
        self._high_score: int = 0
        self._gen_best_score: int = 0
        self._gen_fitnesses: list[float] = []

        self._population = [
            NeuralNetwork(LAYER_SIZES) for _ in range(POPULATION_SIZE)
        ]
        self._ga = GeneticAlgorithm(
            population_size=POPULATION_SIZE,
            mutation_rate=0.05,
            mutation_strength=0.3,
            elite_fraction=0.1,
        )

        self._game = SnakeGame()
        self._fast_mode: bool = False
        self._score_history = []
        self._redraw_chart()

    def _restart(self) -> None:
        """Reset training completely and start over."""
        self._init_training()

    # ------------------------------------------------------------------
    # Main loop scheduling
    # ------------------------------------------------------------------

    def _schedule_next(self) -> None:
        """Schedule the next frame based on current speed / mode."""
        if self._fast_mode:
            delay = 1
        else:
            # speed slider: 5 → 200 maps to ~200 ms → 5 ms per step
            speed = self._speed_var.get()            # invert so right = faster
            delay = max(5, 210 - speed)
        self.root.after(delay, self._tick)

    def _tick(self) -> None:
        """Run one or more game steps then reschedule."""
        self._anim_tick += 1

        if self._fast_mode:
            # Run many steps without rendering for maximum throughput
            for _ in range(60):
                self._game_step()
        else:
            self._game_step()
            self._render()

        self._update_stats()
        self._schedule_next()

    # ------------------------------------------------------------------
    # Game stepping
    # ------------------------------------------------------------------

    def _game_step(self) -> None:
        """Execute one neural-network-driven game step."""
        if not self._game.alive:
            self._handle_game_over()
            return

        net = self._population[self._agent_idx]
        state = self._game.get_state()
        action = net.get_action(state)
        self._game.step(action)

    def _handle_game_over(self) -> None:
        """Record fitness, advance to next agent, or evolve generation."""
        fitness = self._game.get_fitness()
        self._gen_fitnesses.append(fitness)

        score = self._game.score
        if score > self._high_score:
            self._high_score = score
        if score > self._gen_best_score:
            self._gen_best_score = score

        self._total_games += 1
        self._agent_idx += 1

        if self._agent_idx >= POPULATION_SIZE:
            self._evolve()
        else:
            self._game = SnakeGame()

    def _evolve(self) -> None:
        """Perform one generation of evolution."""
        self._score_history.append(self._gen_best_score)
        self._redraw_chart()

        self._population = self._ga.next_generation(
            self._population, self._gen_fitnesses
        )

        self._generation += 1
        self._agent_idx = 0
        self._gen_best_score = 0
        self._gen_fitnesses = []
        self._game = SnakeGame()

    # ------------------------------------------------------------------
    # Toggle fast mode
    # ------------------------------------------------------------------

    def _toggle_fast(self) -> None:
        self._fast_mode = not self._fast_mode
        if self._fast_mode:
            self._fast_btn.config(text="👁 Watch", bg=C_FOOD)
        else:
            self._fast_btn.config(text="⚡ Fast", bg=C_ACCENT)
            # Force a render immediately when returning to watch mode
            self._render()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        """Draw the current game frame onto the canvas."""
        c = self.canvas
        c.delete("all")

        # --- Grid lines ---
        for gx in range(0, GAME_PX_W + 1, CELL_SIZE):
            c.create_line(gx, 0, gx, GAME_PX_H, fill=C_GRID, width=1)
        for gy in range(0, GAME_PX_H + 1, CELL_SIZE):
            c.create_line(0, gy, GAME_PX_W, gy, fill=C_GRID, width=1)

        game = self._game

        # --- Food (pulsing oval) ---
        if game.food is not None:
            fx, fy = game.food
            pulse = abs(math.sin(self._anim_tick * 0.12)) * 3.5
            x1 = fx * CELL_SIZE + 3 + pulse
            y1 = fy * CELL_SIZE + 3 + pulse
            x2 = (fx + 1) * CELL_SIZE - 3 - pulse
            y2 = (fy + 1) * CELL_SIZE - 3 - pulse
            c.create_oval(x1, y1, x2, y2, fill=C_FOOD, outline=C_FOOD_RING, width=2)

        # --- Snake body (gradient from dark to light tail→head) ---
        body = game.body
        body_len = len(body)
        for i, (sx, sy) in enumerate(reversed(body[1:])):
            # Blend C_BODY_DARK → C_BODY_LIGHT as i increases
            t = i / max(body_len - 1, 1)
            fill = _blend_hex(C_BODY_DARK, C_BODY_LIGHT, t)
            x1 = sx * CELL_SIZE + 2
            y1 = sy * CELL_SIZE + 2
            x2 = (sx + 1) * CELL_SIZE - 2
            y2 = (sy + 1) * CELL_SIZE - 2
            c.create_rectangle(x1, y1, x2, y2, fill=fill, outline="")

        # --- Snake head ---
        if body:
            hx, hy = body[0]
            x1 = hx * CELL_SIZE + 1
            y1 = hy * CELL_SIZE + 1
            x2 = (hx + 1) * CELL_SIZE - 1
            y2 = (hy + 1) * CELL_SIZE - 1
            c.create_rectangle(x1, y1, x2, y2, fill=C_HEAD, outline="")
            self._draw_eyes(hx, hy)

        # --- Score overlay ---
        c.create_text(
            6, 5, anchor=tk.NW,
            text=f"Score: {game.score}",
            fill=C_TEXT, font=("Courier", 11, "bold"),
        )

        # --- Game-over overlay ---
        if not game.alive:
            c.create_rectangle(
                0, GAME_PX_H // 2 - 22, GAME_PX_W, GAME_PX_H // 2 + 22,
                fill="#000000", stipple="gray50",
            )
            c.create_text(
                GAME_PX_W // 2, GAME_PX_H // 2,
                text="GAME OVER",
                fill=C_DEAD, font=("Courier", 20, "bold"),
            )

    def _draw_eyes(self, hx: int, hy: int) -> None:
        """Draw two small eyes on the snake head oriented by direction."""
        c = self.canvas
        d = self._game.direction
        cx = hx * CELL_SIZE + CELL_SIZE // 2
        cy = hy * CELL_SIZE + CELL_SIZE // 2

        # Offset pairs (ex, ey) for each direction: (UP, RIGHT, DOWN, LEFT)
        eye_offsets = {
            0: [(-4, -3), (4, -3)],    # UP
            1: [(3, -4), (3,  4)],     # RIGHT
            2: [(-4,  3), (4,  3)],    # DOWN
            3: [(-3, -4), (-3, 4)],    # LEFT
        }
        for ex, ey in eye_offsets.get(d, [(-4, -4), (4, -4)]):
            c.create_oval(
                cx + ex - 2, cy + ey - 2,
                cx + ex + 2, cy + ey + 2,
                fill="white", outline="",
            )

    # ------------------------------------------------------------------
    # Stats panel update
    # ------------------------------------------------------------------

    def _update_stats(self) -> None:
        avg = (
            sum(self._gen_fitnesses) / len(self._gen_fitnesses)
            if self._gen_fitnesses else 0.0
        )
        s = self._stat
        s["generation"].config(text=str(self._generation))
        s["agent"].config(text=f"{self._agent_idx + 1} / {POPULATION_SIZE}")
        s["score"].config(text=str(self._game.score))
        s["high_score"].config(text=str(self._high_score))
        s["best_gen"].config(text=str(self._gen_best_score))
        s["avg_fitness"].config(text=f"{avg:.0f}")
        s["steps"].config(text=str(self._game.steps))
        s["total_games"].config(text=str(self._total_games))

    # ------------------------------------------------------------------
    # History chart
    # ------------------------------------------------------------------

    def _redraw_chart(self) -> None:
        """Redraw the bar-chart of best scores per generation."""
        ch = self._chart
        ch.delete("all")

        history = self._score_history
        if len(history) < 1:
            return

        w = int(ch["width"])
        h = int(ch["height"])
        pad = 5

        max_score = max(history) if history else 1
        max_score = max(max_score, 1)

        # Show at most the most recent 60 generations
        visible = history[-60:]
        n = len(visible)
        step_x = (w - 2 * pad) / max(n, 1)
        bar_w = max(1.0, step_x - 1)

        for i, score in enumerate(visible):
            x = pad + i * step_x
            bar_h = (score / max_score) * (h - 2 * pad)
            y1 = h - pad - bar_h
            y2 = h - pad
            color = C_CHART_LAST if i == n - 1 else C_CHART_BAR
            ch.create_rectangle(x, y1, x + bar_w, y2, fill=color, outline="")

        # Max label
        ch.create_text(
            w - pad, pad, anchor=tk.NE,
            text=f"max: {max_score}",
            fill=C_DIM, font=("Courier", 7),
        )


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert ``'#rrggbb'`` to ``(r, g, b)`` integers."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert ``(r, g, b)`` integers to ``'#rrggbb'``."""
    return f"#{r:02x}{g:02x}{b:02x}"


def _blend_hex(c1: str, c2: str, t: float) -> str:
    """Linear interpolation between two hex colours; ``t`` in [0, 1]."""
    r1, g1, b1 = _hex_to_rgb(c1)
    r2, g2, b2 = _hex_to_rgb(c2)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return _rgb_to_hex(r, g, b)
