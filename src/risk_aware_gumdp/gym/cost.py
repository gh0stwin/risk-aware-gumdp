import abc
from typing import Any, Callable, TypeAlias

import numpy as np

State: TypeAlias = Any


class CostFn(abc.ABC):
    @abc.abstractmethod
    def __call__(self, occupancy: np.ndarray) -> np.ndarray:
        pass


class EntropyCostFn(CostFn):
    def __init__(self, eps: float = 1e-7, norm: bool = False) -> None:
        super().__init__()
        self._eps = eps
        self._norm = norm

    def __call__(self, occupancy: np.ndarray) -> np.ndarray:
        occupancy = (1 - self._eps * occupancy.size) * occupancy + self._eps
        occupancy = np.sum(occupancy, axis=1)
        cost = np.sum(occupancy * np.log(occupancy))

        if not self._norm:
            return cost

        max_cost = 0.0
        min_cost = -1 * np.log(occupancy.size)
        cost = (cost - min_cost) / (max_cost - min_cost)
        return cost


class ImitationCostFn(CostFn):
    def __init__(self, occupancy: np.ndarray, max_cost=1.0):
        super().__init__()
        self._baseline_occupancy = np.array(occupancy)
        self._max_cost = max_cost

    def __call__(self, occupancy: np.ndarray) -> np.ndarray:
        return 0.5 * np.sum(np.square(occupancy - self._baseline_occupancy)) / self._max_cost


class LinearCostFn(CostFn):
    def __init__(self, cost: np.ndarray, min_cost: float = 0.0, max_cost: float = 1.0):
        self._cost = np.array(cost, dtype=np.float64)
        self._min_cost = min_cost
        self._max_cost = max_cost

    def __call__(self, occupancy: np.ndarray) -> np.ndarray:
        return (np.sum(self._cost * occupancy) - self._min_cost) / (self._max_cost - self._min_cost)


class MultiCostFn(CostFn):
    def __init__(
        self,
        costs: list[np.ndarray],
        agg_fn: Callable[[np.ndarray], np.ndarray],
        min_cost: float | None = None,
        max_cost: float | None = None,
    ) -> None:
        super().__init__()
        self._costs = np.array(costs, dtype=np.float64)
        self._agg_fn = agg_fn
        # self._aggn_fn = self._get_agg_fn(agg_fn)
        self._min_cost = min_cost
        self._max_cost = max_cost

    def __call__(self, occupancy: np.ndarray) -> np.ndarray:
        ret = np.sum(np.expand_dims(occupancy, 0) * self._costs, axis=(1, 2))
        ret = self._agg_fn(ret)

        if self._min_cost is not None and self._max_cost is not None:
            ret = (ret - self._min_cost) / (self._max_cost - self._min_cost)

        return ret

    def _get_agg_fn(self, tag: str) -> Callable[[np.ndarray], np.ndarray]:
        match tag:
            case "linear":
                return lambda x: np.mean(x)
            case "max":
                return lambda x: np.max(x)
            case "min":
                return lambda x: np.min(x)
            case "root-diamond":
                return lambda x: x[0] + x[1] + x[2] ** (1 / 2)

        raise ValueError(f"Aggragation function '{tag}' is not defined.")


class MultiWeightedLinearCostFn(CostFn):
    def __init__(
        self,
        costs: list[np.ndarray],
        weights: list[float] | None = None,
        min_cost: float | None = None,
        max_cost: float | None = None,
    ):
        self._costs = np.array(costs, dtype=np.float64)

        if weights is None:
            weights = np.ones((self._costs.shape[0],), dtype=np.float64) / self._costs.shape[0]

        self._weights = np.array(weights, dtype=np.float64)
        self._min_cost = min_cost
        self._max_cost = max_cost

    def __call__(self, occupancy: np.ndarray) -> np.ndarray:
        returns = np.sum(self._costs * np.expand_dims(occupancy, 0), axis=(1, 2))
        # returns = (returns - self._min_costs) / (self._max_costs - self._min_costs)
        returns = np.sum(self._weights * returns)

        if self._min_cost is not None and self._max_cost is not None:
            returns = (returns - self._min_cost) / (self._max_cost - self._min_cost)

        return returns


class NormMultiWeightedLinearCostFn(CostFn):
    def __init__(
        self,
        costs: list[np.ndarray],
        weights: list[float] | None = None,
        min_costs: list[float] | None = None,
        max_costs: list[float] | None = None,
    ):
        self._costs = np.array(costs, dtype=np.float64)

        if weights is None:
            weights = np.ones((self._costs.shape[0],), dtype=np.float64) / self._costs.shape[0]

        if min_costs is None:
            min_costs = np.zeros((self._costs.shape[0],))

        if max_costs is None:
            max_costs = np.ones((self._costs.shape[0],))

        self._weights = np.array(weights, dtype=np.float64)
        self._min_costs = np.array(min_costs, dtype=np.float64)
        self._max_costs = np.array(max_costs, dtype=np.float64)

    def __call__(self, occupancy: np.ndarray) -> np.ndarray:
        returns = np.sum(self._costs * np.expand_dims(occupancy, 0), axis=(1, 2))
        returns = (returns - self._min_costs) / (self._max_costs - self._min_costs)
        return np.sum(self._weights * returns)


class StateLinearCostFn(CostFn):
    def __init__(self, cost: np.ndarray):
        self._cost = cost

    def __call__(self, state) -> np.ndarray:
        return np.sum(self._cost * np.sum(state.occupancy, axis=1))


class VariancePenalizedCostFn(CostFn):
    def __init__(self, cost: np.ndarray, var_weight: float = 0.1):
        super().__init__()
        self._cost = cost
        self._var_weight = var_weight

    def __call__(self, state: State) -> np.ndarray:
        avg = np.sum(state.occupancy * self._cost)
        var = self._var_weight * np.sum(state.occupancy * np.square(self._cost - avg))
        return avg + var
