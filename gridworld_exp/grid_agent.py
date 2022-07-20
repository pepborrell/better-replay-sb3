from abc import ABC, abstractmethod

import numpy as np
from numba import njit
from numpy import typing as npt


class GridworldAgent(ABC):
    """Generic class to make gridworld agent implementation easier.

    Args:
        grid: 2D numpy array. Valid positions 0, walls 1.
        starting_pos: 2D numpy array. List of starting positions.
    """

    def __init__(self, env) -> None:
        self.env = env
        self.possible_actions = env.action_space
        self.rng = np.random.default_rng()
        self.pos = None

    def _is_position_legal(self, pos: npt.NDArray[np.int_]) -> bool:
        return self.env._is_position_legal(pos)

    def _is_action_legal(self, action: npt.NDArray[np.int_]) -> bool:
        return self.env._is_action_legal(self.pos, action)

    def get_trajectory(self, n_steps: int) -> np.ndarray:
        self.pos = self.env.reset()
        trajectory = []
        for _ in range(n_steps):
            past_state = self.pos
            action = self.get_action()
            self.pos, reward, done = self.env.step(action)
            trajectory.append(
                {
                    "state": past_state,
                    "action": action,
                    "reward": reward,
                    "next_state": self.pos,
                }
            )
            if done:
                return trajectory
        return trajectory

    @abstractmethod
    def _get_action(
        self,
        pos: npt.NDArray[np.int_],
        possible_actions: npt.NDArray[np.int_],
        rng: np.random.Generator = None,
    ) -> npt.NDArray[np.int_]:
        """This determines the behaviour of the agent
        and is therefore not implemented in this generic class.
        """
        pass

    def get_action(self) -> npt.NDArray[np.int_]:
        act = self._get_action(self.pos, self.possible_actions, self.rng)
        return act
