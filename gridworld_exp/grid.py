import copy

import numpy as np
from numba import njit
from numpy import typing as npt


def generate_u_grid(n_rows: int, n_cols: int) -> np.ndarray:
    grid = np.zeros((n_rows, n_cols), dtype=int)
    # Add walls
    size = np.array([n_rows, n_cols])
    patterns = size / 4
    wall_positions = np.array([[np.floor(1 * patterns[i]), np.floor(3 * patterns[i])] for i in range(2)])
    grid = paint_square(grid, wall_positions, 1)

    wall_thickness = 2 * np.floor(size / 50) + 1
    padding = wall_thickness[:, np.newaxis] * np.array([[1, -1]])
    inner_walls = wall_positions + padding
    grid = paint_square(grid, inner_walls, 0)

    # Add horizontal opening
    start = int(np.floor(1 * patterns[1]))
    end = int(np.ceil(3 * patterns[1])) + 1
    inner_border = int(inner_walls[0, 1])
    assert inner_border == inner_walls[0, 1]
    grid[inner_border:, :] = 0

    return grid


def paint_square(grid: np.ndarray, wall_positions: np.ndarray, value: int) -> np.ndarray:
    grid = grid.copy()
    nr, nc = grid.shape
    for i in range(nr):
        for j in range(nc):
            if (
                i >= wall_positions[0, 0]
                and i <= wall_positions[0, 1]
                and j >= wall_positions[1, 0]
                and j <= wall_positions[1, 1]
            ):
                grid[i, j] = value
    return grid


def paint_h_line(grid: np.ndarray, row: int, start: int, end: int, value: int) -> np.ndarray:
    grid = grid.copy()
    grid[row, start:end] = value
    return grid


class EnvFromGrid:
    def __init__(self, grid: np.ndarray, goal: np.ndarray, starting_pos: np.ndarray = np.array([[0, 0]])) -> None:
        self.grid = grid
        self.goal = goal
        self.n_rows, self.n_cols = grid.shape
        self.n_actions = 4
        self.possible_actions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=int)
        self.n_states = self.n_rows * self.n_cols
        self.starting_pos = starting_pos
        self.pos = None

    def assign_position(self, position: npt.NDArray[np.int_]) -> None:
        self.pos = copy.deepcopy(position)

    @property
    def action_space(self) -> np.ndarray:
        return self.possible_actions

    def _generate_start_pos(self) -> np.ndarray:
        n = self.starting_pos.shape[0]
        idx = np.random.randint(n)
        return np.squeeze(self.starting_pos[idx, :])

    def reset(self) -> npt.NDArray[np.int_]:
        self.pos = self._generate_start_pos()
        return self.pos

    def _is_position_legal(self, pos) -> bool:
        if not (pos >= 0).all() or not (pos < self.grid.shape).all():
            return False
        if self.grid[tuple(pos)] == 1:
            return False
        return True

    def _is_action_legal(self, pos: npt.NDArray[np.int_], action: npt.NDArray[np.int_]) -> bool:
        new_pos = self._move(pos, action)
        return self._is_position_legal(new_pos)

    @staticmethod
    @njit
    def _move(pos: npt.NDArray[np.int_], action: npt.NDArray[np.int_]) -> np.ndarray:
        """Unsafe move function, does not check if action is legal."""
        return pos + action

    def step(self, action: npt.NDArray[np.int_]) -> tuple:
        assert action in self.possible_actions
        pos = self.pos.copy()
        reward = 10.0 if np.all(pos == self.goal) else -1.0
        if reward > 0:
            # self.pos = None
            return pos, reward, True
        pos += action
        assert self._is_position_legal(pos), "Illegal action"
        self.pos = pos
        return pos, reward, False
