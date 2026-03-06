"""
Snake-vs-evader game core logic.

The human (or a simple chase bot) controls the snake. The neural network
controls the red circle and learns to survive as long as possible without
being caught.

State representation
--------------------
The ``get_state()`` method returns a 12-element list used as the input
vector for the neural network controlling the red circle:

    [0] danger_up         – 1 if moving up would hit a wall or the snake
    [1] danger_right      – 1 if moving right would hit a wall or the snake
    [2] danger_down       – 1 if moving down would hit a wall or the snake
    [3] danger_left       – 1 if moving left would hit a wall or the snake
    [4] snake_left        – 1 if the snake head is left of the red circle
    [5] snake_right       – 1 if the snake head is right of the red circle
    [6] snake_up          – 1 if the snake head is above the red circle
    [7] snake_down        – 1 if the snake head is below the red circle
    [8] snake_dir_up      – 1 if the snake is currently heading up
    [9] snake_dir_right   – 1 if the snake is currently heading right
 [10] snake_dir_down    – 1 if the snake is currently heading down
 [11] snake_dir_left    – 1 if the snake is currently heading left

Action encoding
---------------
    0 – stay in place
    1 – move up
    2 – move right
    3 – move down
    4 – move left
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
FOOD_VECS = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]


class SnakeGame:
    """Single episode of snake chasing an AI-controlled red circle."""

    _EVASION_LIMIT = GRID_W * GRID_H * 4

    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    # Setup / reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all game state to the beginning of a new episode."""
        cx, cy = GRID_W // 2, GRID_H // 2
        self.body: list[tuple[int, int]] = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.direction: int = DIR_RIGHT
        self.food: tuple[int, int] | None = self._place_food()
        self.score: int = 0
        self.steps: int = 0
        self.alive: bool = True
        self.result: str = "running"
        self._distance_accumulator: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _place_food(self) -> tuple[int, int] | None:
        """Place the red circle on a random empty cell."""
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

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        """Return the Manhattan distance between two grid cells."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _can_turn_to(self, direction: int) -> bool:
        """Reject instant 180-degree turns while the snake has a body."""
        if len(self.body) <= 1:
            return True
        return direction != (self.direction + 2) % 4

    def set_snake_direction(self, direction: int) -> None:
        """Update the snake heading if the requested turn is legal."""
        if direction in (DIR_UP, DIR_RIGHT, DIR_DOWN, DIR_LEFT) and self._can_turn_to(direction):
            self.direction = direction

    def get_auto_snake_direction(self) -> int:
        """Return a greedy snake direction that chases the red circle safely."""
        if self.food is None:
            return self.direction

        hx, hy = self.body[0]
        occupied = set(self.body[:-1]) if len(self.body) > 1 else set()
        candidates: list[tuple[int, int]] = []
        for direction, (dx, dy) in enumerate(DIR_VECS):
            if not self._can_turn_to(direction):
                continue
            nx, ny = hx + dx, hy + dy
            if self._is_collision(nx, ny, occupied):
                continue
            distance = self._manhattan((nx, ny), self.food)
            candidates.append((distance, direction))

        if not candidates:
            return self.direction

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_state(self) -> list[float]:
        """Return the 12-dimensional state vector for the red-circle AI."""
        if not self.alive:
            return [0.0] * 12

        food_x, food_y = self.food if self.food is not None else self.body[0]
        hx, hy = self.body[0]
        body_set = set(self.body)

        danger = []
        for dx, dy in DIR_VECS:
            danger.append(float(self._is_collision(food_x + dx, food_y + dy, body_set)))

        return danger + [
            float(hx < food_x),
            float(hx > food_x),
            float(hy < food_y),
            float(hy > food_y),
            float(self.direction == DIR_UP),
            float(self.direction == DIR_RIGHT),
            float(self.direction == DIR_DOWN),
            float(self.direction == DIR_LEFT),
        ]

    def step(self, food_action: int, snake_direction: int | None = None) -> tuple[float, bool]:
        """Advance one chase step and return a reward/done pair for the evader."""
        if not self.alive:
            return 0.0, True

        if snake_direction is not None:
            self.set_snake_direction(snake_direction)

        if self.food is None:
            self.alive = False
            self.result = "escaped"
            return 25.0, True

        food_dx, food_dy = FOOD_VECS[food_action] if 0 <= food_action < len(FOOD_VECS) else (0, 0)
        new_food = (self.food[0] + food_dx, self.food[1] + food_dy)
        if self._is_collision(new_food[0], new_food[1]):
            self.alive = False
            self.result = "caught"
            self.food = new_food
            return -20.0, True

        dx, dy = DIR_VECS[self.direction]
        hx, hy = self.body[0]
        new_head = (hx + dx, hy + dy)
        occupied = set(self.body[:-1]) if len(self.body) > 1 else set()
        if self._is_collision(new_head[0], new_head[1], occupied):
            self.alive = False
            self.result = "snake_crashed"
            self.food = new_food
            return 25.0, True

        self.food = new_food
        self.body.insert(0, new_head)
        self.body.pop()
        self.steps += 1
        self.score = self.steps
        self._distance_accumulator += self._manhattan(new_head, self.food)

        if new_head == self.food:
            self.alive = False
            self.result = "caught"
            return -20.0, True

        if self.steps >= self._EVASION_LIMIT:
            self.alive = False
            self.result = "escaped"
            return 20.0, True

        return 1.0 + self._manhattan(new_head, self.food) * 0.1, False

    def get_fitness(self) -> float:
        """Return the evader fitness used by the genetic algorithm."""
        outcome_bonus = {
            "caught": 0,
            "snake_crashed": 300,
            "escaped": 500,
            "running": 0,
        }
        return float(self.steps * 25 + self._distance_accumulator + outcome_bonus.get(self.result, 0))
