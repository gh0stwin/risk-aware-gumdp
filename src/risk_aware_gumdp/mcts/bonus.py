import abc
import enum
from typing import Any

import numpy as np
import numpy.typing

from risk_aware_gumdp.mcts.node import DecisionNode


class Bonus(abc.ABC):
    @abc.abstractmethod
    def bonus(self, node: DecisionNode) -> numpy.typing.ArrayLike:
        pass


class NoBonus(Bonus):
    def bonus(self, node: DecisionNode) -> numpy.typing.ArrayLike:
        return np.zeros((len(node.children),))


class UCT(Bonus):
    def __init__(self, coeff: float = 1.0) -> None:
        self._coeff = coeff

    def bonus(self, node: DecisionNode) -> numpy.typing.ArrayLike:
        children_visits = np.array([chance_node.num_visits for chance_node in node.children])
        assert node.num_visits - 1 == np.sum(children_visits), (node.is_root, node.num_visits, children_visits)
        return self._coeff * np.sqrt(np.log(node.num_visits) / children_visits)


class ERM_UCT(Bonus):
    def __init__(self, coeff: float = 1.0) -> None:
        self._coeff = coeff

    def bonus(self, node: DecisionNode) -> numpy.typing.ArrayLike:
        children_visits = np.array([chance_node.num_visits for chance_node in node.children])
        assert node.num_visits - 1 == np.sum(children_visits), (node.is_root, node.num_visits, children_visits)
        return self._coeff * np.sqrt(np.sqrt(node.num_visits) / children_visits)


class BonusType(enum.StrEnum):
    NONE = "none"
    UCT = "uct"
    ERM_UCT = "erm_uct"


def bonus_factory(bonus_type: BonusType, kw: dict[str, Any]) -> Bonus:
    if bonus_type == BonusType.NONE:
        return NoBonus()
    elif bonus_type == BonusType.UCT:
        return UCT(**kw)
    elif bonus_type == BonusType.ERM_UCT:
        return ERM_UCT(**kw)
