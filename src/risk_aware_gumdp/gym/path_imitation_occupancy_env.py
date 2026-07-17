from dataclasses import dataclass
from typing import Any

import gymnasium
import numpy as np
import scipy
from gymnasium.experimental.functional import FuncEnv

from risk_aware_gumdp.gym.cost import ImitationCostFn
from risk_aware_gumdp.gym.utils import Params


@dataclass
class PathImitationOccupancyState:
    state_index: np.ndarray
    step_count: np.ndarray
    occupancy: np.ndarray
    is_stuck: np.ndarray


@dataclass
class PathImitationOccupancyObs:
    state_index: np.ndarray
    occupancy: np.ndarray
    is_stuck: np.ndarray


class PathImitationOccupancyEnv(FuncEnv):
    N_ACTIONS = 4
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    NOOP = 4

    def __init__(
        self,
        behavior_occupancy: list[list[float]],
        stuck_prob: float = 0.25,
        unstuck_prob: float = 0.05,
        size: tuple[int, int] = (5, 5),
        init_pos: list[tuple[int, int]] = [(0, 0)],
        init_pos_prob: list[int] = [],
        difficult_terrain: list[tuple[int, int]] = [],
        gamma: float = 0.99,
        h: int = 99,
        options: dict[str, Any] | None = None,
    ):
        super().__init__(options)
        assert size[0] > 0
        assert size[1] > 0
        assert 0.0 <= stuck_prob < 1.0
        assert 0.0 <= unstuck_prob < 1.0
        self.grid_size = size
        self.stuck_prob = stuck_prob
        self.unstuck_prob = unstuck_prob
        self.init_pos = self._pos_list_to_arr(init_pos)

        if self.init_pos.size == 0:
            self.init_pos = np.arange(self.grid_size[0] * self.grid_size[1])

        if len(init_pos_prob) == 0:
            self.init_pos_prob = scipy.special.softmax(np.ones_like(self.init_pos))
        else:
            assert len(init_pos_prob) == self.init_pos.shape[0]
            self.init_pos_prob = np.array(init_pos_prob)

        assert np.sum(self.init_pos_prob) == 1.0
        self.difficult_terrain = self._pos_list_to_arr(difficult_terrain)
        self.cost_fn = ImitationCostFn(np.array(behavior_occupancy), max_cost=0.12)
        self.gamma = gamma
        self.h = h

    def initial(self, rng: np.random.Generator, params: Params | None = None) -> PathImitationOccupancyState:
        state_idx = rng.choice(self.init_pos, size=(), p=self.init_pos_prob)
        occupancy = np.zeros((self.grid_size[0] * self.grid_size[1] + 1, self.N_ACTIONS), dtype=np.float64)
        return PathImitationOccupancyState(
            state_index=state_idx, step_count=np.array(0, dtype=np.int64), occupancy=occupancy, is_stuck=np.array(False)
        )

    def transition(
        self,
        state: PathImitationOccupancyState,
        action: np.ndarray,
        rng: np.random.Generator,
        params: Params | None = None,
    ) -> PathImitationOccupancyState:
        curr_index = state.state_index.copy()
        next_idx, is_stuck = self._move(rng, curr_index, action, state.is_stuck)
        occupancy = state.occupancy.copy()
        occupancy[curr_index, action] += self.gamma**state.step_count
        return PathImitationOccupancyState(
            state_index=next_idx, step_count=state.step_count.copy() + 1, occupancy=occupancy, is_stuck=is_stuck
        )

    def reward(
        self,
        state: PathImitationOccupancyState,
        action: np.ndarray,
        next_state: PathImitationOccupancyState,
        rng: np.random.Generator,
        params: Params | None = None,
    ) -> np.ndarray:
        if next_state.step_count < self.h:
            return np.array(0.0)

        norm = (1 - self.gamma) / (1 - self.gamma**self.h)
        return self.cost_fn(next_state.occupancy * norm)

    def terminal(
        self, state: PathImitationOccupancyState, rng: np.random.Generator, params: Params | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        return state.step_count >= self.h, np.array(False)

    def observation(
        self, state: PathImitationOccupancyState, rng: np.random.Generator, params: Params | None = None
    ) -> PathImitationOccupancyObs:
        norm = (1 - self.gamma) / (1 - self.gamma**self.h)
        return PathImitationOccupancyObs(state_index=state.state_index.copy(), occupancy=state.occupancy * norm)

    def observation_space(self, params: Params | None = None) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Box(0, self.grid_size[0] * self.grid_size[1], shape=(), dtype=np.int32)

    def action_space(self, params: Params | None = None) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Discrete(self.N_ACTIONS)

    def max_state_links(self, params: Params | None = None):
        return 5

    def _move(
        self, rng: np.random.Generator, grid_idx: np.ndarray, action: np.ndarray, is_stuck: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        noop = np.array(False)
        next_is_stuck = np.array(False)

        if np.isin(grid_idx, self.difficult_terrain):
            if is_stuck:
                noop = rng.random() >= self.unstuck_prob
            else:
                noop = rng.random() < self.stuck_prob

            next_is_stuck = np.copy(noop)

        cannot_move_left = np.logical_and(grid_idx % self.grid_size[1] == 0, action == self.LEFT)
        cannot_move_up = np.logical_and(grid_idx // self.grid_size[1] == 0, action == self.UP)
        cannot_move_right = np.logical_and(grid_idx % self.grid_size[1] == self.grid_size[1] - 1, action == self.RIGHT)
        cannot_move_down = np.logical_and(grid_idx // self.grid_size[1] == self.grid_size[0] - 1, action == self.DOWN)
        no_move = np.logical_or(noop, cannot_move_left)
        no_move = np.logical_or(no_move, cannot_move_up)
        no_move = np.logical_or(no_move, cannot_move_right)
        no_move = np.logical_or(no_move, cannot_move_down)

        if no_move:
            return grid_idx, next_is_stuck

        if action == self.LEFT:
            new_grid_idx = grid_idx - 1
        elif action == self.UP:
            new_grid_idx = grid_idx - self.grid_size[1]
        elif action == self.RIGHT:
            new_grid_idx = grid_idx + 1
        else:
            new_grid_idx = grid_idx + self.grid_size[1]

        return new_grid_idx, next_is_stuck

    def _pos_list_to_arr(self, pos: list[tuple[int, int]]) -> np.ndarray:
        if len(pos) == 0:
            return np.array([])

        arr = np.array(pos)
        assert arr.ndim == 2
        assert arr.shape[0] > 0
        assert arr.shape[1] == 2
        return arr[:, 0] * self.grid_size[1] + arr[:, 1]
