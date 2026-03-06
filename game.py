"""
Snake game core logic.

Pure Python — no external packages or libraries required.
The game board is a GRID_W x GRID_H grid.  The snake starts in the
centre moving right.  One piece of food appears at a time; eating it
grows the snake by one cell and increments the score.

State representation
--------------------
The ``get_state()`` method returns an 11-element list used as the input
vector for the neural network:

  [0] danger straight   – 1 if moving forward leads to a collision
  [1] danger right      – 1 if turning right leads to a collision
  [2] danger left       – 1 if turning left leads to a collision
  [3] dir_up            – 1 if currently heading up
  [4] dir_right         – 1 if currently heading right
  [5] dir_down          – 1 if currently heading down
  [6] dir_left          – 1 if currently heading left
  [7] food_left         – 1 if food is to the left of the head
  [8] food_right        – 1 if food is to the right of the head
  [9] food_up           – 1 if food is above the head
 [10] food_down         – 1 if food is below the head

Action encoding
---------------
  0 – go straight
  1 – turn right (clockwise)
  2 – turn left  (counter-clockwise)
"""

import random

# ---------------------------------------------------------------------------
# Grid dimensions
# ---------------------------------------------------------------------------
GRID_W: int = 20
GRID_H: int = 20

# ---------------------------------------------------------------------------
# Direction constants and movement vectors
# Direction indices: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
# ---------------------------------------------------------------------------
DIR_UP = 0
DIR_RIGHT = 1
DIR_DOWN = 2
DIR_LEFT = 3

# (dx, dy) for each direction — y increases downward on screen
DIR_VECS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


class SnakeGame:
    """Single episode of the Snake game."""

    # Maximum steps without eating before the episode is terminated
    # (prevents infinite loops).
    _STARVATION_LIMIT = GRID_W * GRID_H * 3

    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    # Setup / reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all game state to the beginning of a new episode."""
        cx, cy = GRID_W // 2, GRID_H // 2
        # Snake starts as a 3-cell horizontal line moving right
        self.body: list[tuple[int, int]] = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.direction: int = DIR_RIGHT
        self.food: tuple[int, int] | None = self._place_food()
        self.score: int = 0
        self.steps: int = 0
        self.steps_without_food: int = 0
        self.alive: bool = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _place_food(self) -> tuple[int, int] | None:
        """Place food on a random empty cell; returns None if the board is full."""
        body_set = set(self.body)
        empty = [
            (x, y)
            for x in range(GRID_W)
            for y in range(GRID_H)
            if (x, y) not in body_set
        ]
        return random.choice(empty) if empty else None

    def _is_collision(self, x: int, y: int, body_set: set | None = None) -> bool:
        """Return True if (x, y) is a wall or occupied by the snake body."""
        if x < 0 or x >= GRID_W or y < 0 or y >= GRID_H:
            return True
        occupied = body_set if body_set is not None else set(self.body)
        return (x, y) in occupied

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_state(self) -> list[float]:
        """
        Return the 11-dimensional state vector for the neural network.

        All values are 0.0 or 1.0 (boolean indicators).
        """
        hx, hy = self.body[0]
        d = self.direction

        straight_dx, straight_dy = DIR_VECS[d]
        right_dx, right_dy = DIR_VECS[(d + 1) % 4]
        left_dx, left_dy = DIR_VECS[(d - 1) % 4]

        food_x, food_y = self.food if self.food is not None else (hx, hy)

        body_set = set(self.body)   # compute once; reused for all three danger checks
        return [
            # --- Danger signals (relative to current heading) ---
            float(self._is_collision(hx + straight_dx, hy + straight_dy, body_set)),
            float(self._is_collision(hx + right_dx, hy + right_dy, body_set)),
            float(self._is_collision(hx + left_dx, hy + left_dy, body_set)),
            # --- Current heading (one-hot) ---
            float(d == DIR_UP),
            float(d == DIR_RIGHT),
            float(d == DIR_DOWN),
            float(d == DIR_LEFT),
            # --- Food direction relative to head ---
            float(food_x < hx),   # food is to the left
            float(food_x > hx),   # food is to the right
            float(food_y < hy),   # food is above  (y increases downward)
            float(food_y > hy),   # food is below
        ]

    def step(self, action: int) -> tuple[float, bool]:
        """
        Advance the game by one step.

        Args:
            action: 0 = straight, 1 = turn right, 2 = turn left.

        Returns:
            (reward, done) tuple.
        """
        if not self.alive:
            return 0.0, True

        # --- Update heading ---
        if action == 1:
            self.direction = (self.direction + 1) % 4
        elif action == 2:
            self.direction = (self.direction - 1) % 4

        dx, dy = DIR_VECS[self.direction]
        hx, hy = self.body[0]
        new_head = (hx + dx, hy + dy)

        # --- Collision check (build body_set once for both wall and body test) ---
        body_set = set(self.body)
        if (
            new_head[0] < 0
            or new_head[0] >= GRID_W
            or new_head[1] < 0
            or new_head[1] >= GRID_H
            or new_head in body_set
        ):
            self.alive = False
            return -10.0, True

        # --- Move snake ---
        self.body.insert(0, new_head)
        self.steps += 1
        self.steps_without_food += 1

        # --- Check food ---
        if new_head == self.food:
            self.score += 1
            self.steps_without_food = 0
            self.food = self._place_food()
            if self.food is None:
                # Snake fills the entire board — perfect win
                self.alive = False
                return 100.0, True
            return 10.0, False
        else:
            self.body.pop()  # remove tail (snake doesn't grow)

        # --- Starvation timeout ---
        if self.steps_without_food >= self._STARVATION_LIMIT:
            self.alive = False
            return -1.0, True

        return 0.0, False

    def get_fitness(self) -> float:
        """
        Fitness score used by the genetic algorithm.

        Grows quadratically with the number of food items eaten so that
        eating more food is always strongly preferred over merely surviving.
        """
        return float(self.score * self.score * 500 + self.steps)
