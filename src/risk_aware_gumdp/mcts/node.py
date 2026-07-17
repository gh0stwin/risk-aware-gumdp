from __future__ import annotations

import abc
from typing import Any, override

import numpy as np
import numpy.typing


class Node(abc.ABC):
    def __init__(self, level: int = 0) -> None:
        super().__init__()
        self._level = level

    @property
    def level(self) -> int:
        return self._level


class DecisionNode(Node):
    num_actions: int

    @classmethod
    def set_num_actions(cls, num_actions: int) -> None:
        cls.num_actions = num_actions

    def __init__(
        self, state: Any, parent: ChanceNode | None = None, is_root: bool = False, is_leaf: bool = False
    ) -> None:
        level = 0 if parent is None else (parent.level + 1)
        super().__init__(level)
        self._state = state
        self._parent = parent
        self._is_root = is_root
        self._is_leaf = is_leaf
        self._visits: int = 0
        self._children: list[ChanceNode] = []

    @property
    def state(self) -> Any:
        return self._state

    @property
    def parent(self) -> ChanceNode | None:
        return self._parent

    @property
    def is_root(self) -> bool:
        return self._is_root

    @property
    def is_leaf(self) -> bool:
        return self._is_leaf

    @property
    def num_visits(self) -> int:
        return self._visits

    @property
    def children(self) -> list[ChanceNode]:
        return self._children

    @children.setter
    def children(self, children: list[ChanceNode]) -> None:
        self._children = children

    def inc_visits(self) -> None:
        self._visits += 1


class ChanceNode(abc.ABC):
    max_children: int = 2

    @classmethod
    def set_max_children(cls, max_children: int) -> None:
        cls.max_children = max_children

    def __init__(self, action: int, parent: DecisionNode) -> None:
        self._action = action
        self._parent = parent
        self._visits = 0
        self._children: dict[int, DecisionNode] = {}

    @property
    def action(self) -> Any:
        return self._action

    @property
    def level(self) -> int:
        return self.parent.level

    @property
    def parent(self) -> DecisionNode:
        return self._parent

    @property
    def num_visits(self) -> int:
        return self._visits

    @property
    @abc.abstractmethod
    def risk(self) -> float | numpy.typing.ArrayLike:
        pass

    def step(self, child_state: Any, is_leaf: bool) -> DecisionNode:
        child_hash = int(child_state.state_index)

        if child_hash in self._children:
            return self._children[child_hash]

        new_child = DecisionNode(child_state, self, False, is_leaf)

        if not is_leaf:
            new_child.children = [self.__class__(a, new_child) for a in range(DecisionNode.num_actions)]

        self._children[child_hash] = new_child
        return new_child

    @abc.abstractmethod
    def add_cost(self, cost: float | numpy.typing.ArrayLike) -> None:
        pass


class ErmChanceNode(ChanceNode):
    beta: float
    gamma: float

    @classmethod
    def set_beta(cls, beta: float) -> None:
        cls.beta = beta

    @classmethod
    def set_gamma(cls, gamma: float) -> None:
        cls.gamma = gamma

    def __init__(self, action: int, parent: DecisionNode) -> None:
        super().__init__(action, parent)
        self._costs = 0.0

    @property
    def risk(self) -> float | numpy.typing.ArrayLike:
        erm = np.log(self._costs) - np.log(self._visits)
        return 1.0 + erm / (self.beta * self.gamma**self.level)

    @override
    def add_cost(self, cost: float | numpy.typing.ArrayLike) -> None:
        self._costs += np.exp(self.beta * self.gamma**self.level * (cost - 1.0))  # pyright: ignore
        self._visits += 1
