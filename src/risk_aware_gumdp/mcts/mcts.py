import abc
from typing import Any

import numpy as np
import numpy.typing
from gymnasium.experimental.functional import FuncEnv
from tqdm import tqdm

from risk_aware_gumdp.mcts.bonus import Bonus, NoBonus
from risk_aware_gumdp.mcts.node import ChanceNode, DecisionNode, ErmChanceNode


class MCTS(abc.ABC):
    def __init__(
        self,
        rng: np.random.Generator,
        initial_state: Any,
        env: FuncEnv,
        bonus: Bonus = NoBonus(),
        max_rollouts: int = 1024,
        gamma: float = 0.99,
    ):
        self._rng = rng
        self.env = env
        self._gamma = gamma
        self.bonus = bonus
        self.max_rollouts = max_rollouts
        DecisionNode.set_num_actions(env.action_space().n)
        self.root = self.init_tree(initial_state)

    def init_tree(self, initial_state):
        return self._create_decision_node(initial_state, None, True, False)

    def grow_tree(self):
        state_node = self.root
        state_node.inc_visits()
        returns = 0.0
        tree_costs = []

        # Selection and expansion.
        while (not state_node.is_leaf) and state_node.num_visits > 0:
            action = self.tree_policy(state_node, True)
            action_node = state_node.children[action]
            child_node, cost = self.select_outcome(action_node)
            tree_costs.append(cost)
            state_node = child_node

        # Simulation (rollout).
        if not state_node.is_leaf:
            returns = self.rollout(state_node)

        # Backup.
        i = len(tree_costs) - 1

        while not state_node.is_root:
            state_node.inc_visits()
            random_node = state_node.parent
            returns = tree_costs[i] + self._gamma * returns
            i -= 1
            random_node.add_cost(returns)
            state_node = random_node.parent

        assert i == -1

    def best_action(self, x: DecisionNode) -> tuple[int, numpy.typing.ArrayLike]:
        risk = self._action_costs(x, False)
        return np.argmin(risk), risk

    def tree_policy(self, x: DecisionNode, add_bonus: bool) -> int:
        risk = self._action_costs(x, add_bonus)
        return np.argmin(risk)

    def select_outcome(self, chance_node: ChanceNode) -> tuple[DecisionNode, float | numpy.typing.ArrayLike]:
        next_state = self.env.transition(chance_node.parent.state, chance_node.action, self._rng.spawn(1)[0])
        is_done = self.env.terminal(next_state, None)[0]
        decision_node = chance_node.step(next_state, is_done)
        return decision_node, self.env.reward(chance_node.parent.state, chance_node.action, next_state, None)

    def rollout(self, initial_node: DecisionNode):
        i = 0
        returns = 0.0
        done = False
        action = 0
        state = prev_state = initial_node.state

        while not done:
            action = self._rng.choice(DecisionNode.num_actions)
            prev_state = state
            state = self.env.transition(prev_state, action, self._rng.spawn(1)[0])
            returns += self._gamma**i * self.env.reward(prev_state, action, state, 0)
            done = self.env.terminal(state, 0)[0]
            i += 1

        return returns

    def learn(self, progress_bar=False):
        iterations = range(self.max_rollouts)

        if progress_bar:
            iterations = tqdm(iterations)

        for _ in iterations:
            self.grow_tree()

    def _action_costs(self, x: DecisionNode, add_bonus: bool) -> numpy.typing.ArrayLike:
        risk = [(chance_node.risk if chance_node.num_visits > 0 else -np.inf) for chance_node in x.children]
        risk = np.array(risk)

        if add_bonus and np.all(np.isfinite(risk)):
            risk -= self.bonus.bonus(x)

        return risk

    @abc.abstractmethod
    def _create_decision_node(
        self, state: Any, parent: ChanceNode | None, is_root: bool, is_leaf: bool
    ) -> DecisionNode:
        pass


class ErmMCTS(MCTS):
    def __init__(
        self,
        rng: np.random.Generator,
        initial_state: Any,
        env: FuncEnv,
        beta: float,
        bonus: Bonus = NoBonus(),
        max_rollouts: int = 1024,
        gamma: float = 0.99,
    ):
        super().__init__(rng, initial_state, env, bonus, max_rollouts, gamma)
        ErmChanceNode.set_beta(beta)
        ErmChanceNode.set_max_children(env.max_state_links())
        ErmChanceNode.set_gamma(self._gamma)

    def _create_decision_node(
        self, state: Any, parent: ChanceNode | None, is_root: bool, is_leaf: bool
    ) -> DecisionNode:
        node = DecisionNode(state, parent, is_root, is_leaf)

        if is_leaf:
            return node

        node.children = [ErmChanceNode(a, node) for a in range(DecisionNode.num_actions)]
        return node


class ErmOccupancyMCTS(ErmMCTS):
    def __init__(
        self,
        rng: np.random.Generator,
        initial_state: Any,
        env: FuncEnv,
        beta: float,
        bonus: Bonus = NoBonus(),
        max_rollouts: int = 1024,
    ):
        super().__init__(rng, initial_state, env, beta, bonus, max_rollouts, 1.0)
        ErmChanceNode.set_gamma(1.0)
        assert self._gamma == 1.0
