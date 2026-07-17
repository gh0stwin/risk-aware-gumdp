from dataclasses import dataclass
from typing import Any

import gymnasium
import numpy as np
from gymnasium.experimental.functional import FuncEnv

from risk_aware_gumdp.gym.cost import MultiCostFn, MultiWeightedLinearCostFn
from risk_aware_gumdp.gym.utils import Params


@dataclass
class ResourceGatheringOccupancyState:
    state_index: np.ndarray
    occupancy: np.ndarray
    step_count: np.ndarray
    has_gold: np.ndarray
    has_diamond: np.ndarray


@dataclass
class ResourceGatheringOccupancyObs:
    state_index: np.ndarray
    occupancy: np.ndarray


class ResourceGatheringOccupancyEnv(FuncEnv):
    N_ACTIONS = 4
    LEFT = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    NOOP = 4

    def __init__(
        self,
        cost: str = "linear",
        size: tuple[int, int] = (5, 5),
        init_pos: tuple[int, int] = (4, 2),
        die_prob: float = 0.1,
        rnd_move_prob: float = 0.0,
        gold_pos: list[tuple[int, int]] = [(0, 2)],
        diamond_pos: list[tuple[int, int]] = [(1, 4)],
        enemies_pos: list[tuple[int, int]] = [(0, 3), (1, 2), (1, 3)],
        gamma: float = 0.99,
        h: int = 500,
        options: dict[str, Any] | None = None,
    ):
        super().__init__(options)
        assert size[0] > 0
        assert size[1] > 0
        self.grid_size = size
        self.house_grid_idx = init_pos[0] * size[1] + init_pos[1]
        self.die_prob = die_prob
        self.rnd_move_prob = rnd_move_prob
        self.gold_idx = self._pos_list_to_arr(gold_pos)
        self.diamond_idx = self._pos_list_to_arr(diamond_pos)
        self.enemies_idx = self._pos_list_to_arr(enemies_pos)
        self.hold_options = 4
        self.died_idx = 1 + self.hold_options * size[0] * size[1]
        self.house_idx = np.arange(self.died_idx + 1, self.died_idx + 5, dtype=np.int64)
        self.absorb_idx = np.array(self.house_idx[-1] + 1, dtype=np.int64)
        self.n_states = int(self.absorb_idx + 1)
        cost_1 = np.zeros((self.n_states, self.N_ACTIONS))
        cost_1[self.died_idx, :] = 1
        cost_2 = np.zeros((self.n_states, self.N_ACTIONS))
        cost_2[self.house_idx[1], :] = -1
        cost_2[self.house_idx[3], :] = -1
        cost_3 = np.zeros((self.n_states, self.N_ACTIONS))
        cost_3[self.house_idx[2], :] = -1
        cost_3[self.house_idx[3], :] = -1
        costs = [cost_1, cost_2, cost_3]

        if cost == "linear":
            self.cost_fn = MultiWeightedLinearCostFn(costs, [0.34, 0.33, 0.33], -0.01749591, 0.00986631)
        elif cost == "min":
            self.cost_fn = MultiCostFn(costs, lambda x: np.min(x), -0.02759636, 0.02901856)
        elif cost == "max":
            self.cost_fn = MultiCostFn(costs, lambda x: np.max(x), -0.02759636, 0.02901856)
        elif cost == "root-gold":
            self.cost_fn = MultiCostFn(
                costs,
                lambda x: np.sum(np.array([1, 1, 0]) * x + np.array([0, 0, 1]) * np.sign(x) * np.abs(x) ** (1 / 2)),
                -0.18932464,
                0.02901856,
            )
        elif cost == "root-diamond":
            self.cost_fn = MultiCostFn(
                costs,
                lambda x: np.sum(np.array([1, 0, 1]) * x + np.array([0, 1, 0]) * np.sign(x) * np.abs(x) ** (1 / 2)),
                -0.18932464,
                0.02901856,
            )

        self.gamma = gamma
        self.h = h

    def initial(self, rng: np.random.Generator, params: Params | None = None) -> ResourceGatheringOccupancyState:
        occupancy = np.zeros((self.n_states, self.N_ACTIONS), dtype=np.float64)
        return ResourceGatheringOccupancyState(
            state_index=np.array(0, dtype=np.int64),
            step_count=np.array(0, dtype=np.int64),
            occupancy=occupancy,
            has_gold=np.array(False),
            has_diamond=np.array(False),
        )

    def transition(
        self,
        state: ResourceGatheringOccupancyState,
        action: np.ndarray,
        rng: np.random.Generator,
        params: Params | None = None,
    ) -> ResourceGatheringOccupancyState:
        curr_index = state.state_index.copy()
        grid_idx = (curr_index - 1) % (self.grid_size[0] * self.grid_size[1]) if curr_index > 0 else self.house_grid_idx
        can_move = True
        has_gold = state.has_gold.copy()
        has_diamond = state.has_diamond.copy()

        if curr_index >= self.died_idx:
            next_idx = self.absorb_idx
            can_move = False
        elif np.isin(grid_idx, self.enemies_idx):
            if rng.random() < self.die_prob:
                next_idx = self.died_idx
                can_move = False
        elif curr_index > 0 and grid_idx == self.house_grid_idx:
            next_idx = self.house_idx[self._get_hold_rank(has_gold, has_diamond)]
            can_move = False

        if can_move:
            has_gold = np.logical_or(has_gold, np.isin(grid_idx, self.gold_idx))
            has_diamond = np.logical_or(has_diamond, np.isin(grid_idx, self.diamond_idx))
            next_grid_idx = self._move(rng, grid_idx, action)

            if curr_index == 0 and next_grid_idx == self.house_grid_idx:  # agent still at initial state
                next_idx = np.array(0, dtype=np.int64)
            else:
                next_idx = self._get_hold_rank(has_gold, has_diamond) * self.grid_size[0] * self.grid_size[1]
                next_idx += next_grid_idx + 1

        occupancy = state.occupancy.copy()
        occupancy[curr_index, action] += self.gamma**state.step_count
        return ResourceGatheringOccupancyState(
            state_index=np.array(next_idx),
            step_count=state.step_count.copy() + 1,
            occupancy=occupancy,
            has_gold=has_gold,
            has_diamond=has_diamond,
        )

    def reward(
        self,
        state: ResourceGatheringOccupancyState,
        action: np.ndarray,
        next_state: ResourceGatheringOccupancyState,
        rng: np.random.Generator,
        params: Params | None = None,
    ) -> np.ndarray:
        if next_state.step_count < self.h:
            return np.array(0.0)

        norm = (1 - self.gamma) / (1 - self.gamma**self.h)
        return self.cost_fn(next_state.occupancy * norm)

    def terminal(
        self, state: ResourceGatheringOccupancyState, rng: np.random.Generator, params: Params | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        return state.step_count >= self.h, np.array(False)

    def observation(
        self, state: ResourceGatheringOccupancyState, rng: np.random.Generator, params: Params | None = None
    ) -> ResourceGatheringOccupancyObs:
        norm = (1 - self.gamma) / (1 - self.gamma**self.h)
        return ResourceGatheringOccupancyObs(state_index=state.state_index.copy(), occupancy=state.occupancy * norm)

    def observation_space(self, params: Params | None = None) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Box(0, self.grid_size[0] * self.grid_size[1], shape=(), dtype=np.int32)

    def action_space(self, params: Params | None = None) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Discrete(self.N_ACTIONS)

    def max_state_links(self, params: Params | None = None):
        return 5

    def _get_hold_rank(self, has_gold: bool, has_diamond: bool) -> int:
        return 2 * int(has_gold) + int(has_diamond)

    def _move(self, rng: np.random.Generator, grid_idx: np.ndarray, action: np.ndarray) -> np.ndarray:
        if rng.random() < self.rnd_move_prob:
            if action % 2 == 0:
                rnd_actions = np.array([1, 3])
            else:
                rnd_actions = np.array([0, 2])

            action = rnd_actions[rng.choice(len(rnd_actions))]

        noop = np.array(False)
        cannot_move_left = np.logical_and(grid_idx % self.grid_size[1] == 0, action == self.LEFT)
        cannot_move_up = np.logical_and(grid_idx // self.grid_size[1] == 0, action == self.UP)
        cannot_move_right = np.logical_and(grid_idx % self.grid_size[1] == self.grid_size[1] - 1, action == self.RIGHT)
        cannot_move_down = np.logical_and(grid_idx // self.grid_size[1] == self.grid_size[0] - 1, action == self.DOWN)
        no_move = np.logical_or(noop, cannot_move_left)
        no_move = np.logical_or(no_move, cannot_move_up)
        no_move = np.logical_or(no_move, cannot_move_right)
        no_move = np.logical_or(no_move, cannot_move_down)

        if no_move:
            return grid_idx

        if action == self.LEFT:
            new_grid_idx = grid_idx - 1
        elif action == self.UP:
            new_grid_idx = grid_idx - self.grid_size[1]
        elif action == self.RIGHT:
            new_grid_idx = grid_idx + 1
        else:
            new_grid_idx = grid_idx + self.grid_size[1]

        return new_grid_idx

    def _pos_list_to_arr(self, pos: list[tuple[int, int]]) -> np.ndarray:
        if len(pos) == 0:
            return np.array([])

        arr = np.array(pos)
        assert arr.ndim == 2
        assert arr.shape[0] > 0
        assert arr.shape[1] == 2
        return arr[:, 0] * self.grid_size[1] + arr[:, 1]
