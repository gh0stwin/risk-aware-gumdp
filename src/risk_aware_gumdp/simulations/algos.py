import numpy as np
from tqdm import tqdm


class ERMBackwardInduction(object):
    def __init__(self, env, beta, horizon):
        self.mdp = env.mdp
        self.beta = beta
        self.horizon = horizon
        self.gamma = env.mdp["gamma"]

    def compute(self):
        print("Running ERMBackwardInduction...")
        V_func = np.zeros((self.horizon, len(self.mdp["states"])))
        V_func[-1, :] = np.min(self.mdp["C"], axis=1)

        policy = np.zeros((self.horizon, len(self.mdp["states"])), dtype=np.int32)
        policy[-1, :] = np.argmin(self.mdp["C"], axis=1)

        for t in range(self.horizon - 2, -1, -1):
            beta_t = self.beta * self.gamma**t

            for state in self.mdp["states"]:
                Q_vals = []
                for action in self.mdp["actions"]:
                    exp_next_states = np.dot(
                        self.mdp["P"][action][state],
                        np.exp(beta_t * (self.mdp["C"][state][action] + self.gamma * V_func[t + 1, :])),
                    )
                    Q_vals.append((1.0 / beta_t) * np.log(exp_next_states))

                V_func[t, state] = np.min(Q_vals)
                policy[t, state] = np.argmin(Q_vals)

        return policy


class DecisionNode:
    """
    The Decision Node class.

    A decision node stores a set of children nodes corresponding
    to the possible actions (of type RandomNode).

    :param state: (tuple) defining the state.
    :param available_actions: (tuple) defining the available actions at the state.
    :param father: (RandomNode) The father node; None if root.
    :param is_root: (bool)
    :param is_final: (bool)
    """

    def __init__(self, state=None, available_actions=None, father=None, is_root=False, is_final=False):
        self.state = state
        self.available_actions = available_actions
        self.children = {}
        self.is_final = is_final
        self.visits = 0
        self.father = father
        self.is_root = is_root

        # Initialize children nodes.
        for action in self.available_actions:
            self.children[action] = RandomNode(action, father=self)

    def get_random_node(self, action: int):
        return self.children[action]


class RandomNode:
    """
    The RandomNode class defined by the state and the action.

    A random node stores a set of children nodes corresponding
    to the possible next states (of type DecisionNode).

    :param action: (int) taken in the decision node
    :param father: (DecisionNode)

    """

    def __init__(self, action: int, father: DecisionNode):
        self.action = action
        self.father = father
        self.children = {}
        self.costs_list = []
        self.visits = 0
        # self.cumulative_cost = 0

    def add_child(self, child: DecisionNode, hash_preprocess):
        state_hash = hash_preprocess(child.state)
        if state_hash not in self.children.keys():
            self.children[state_hash] = child
        else:
            del child
            child = self.children[state_hash]

        return child


class ERM_MCTS:
    def __init__(self, initial_state, env, K_ucb, erm_beta, rollout_policy=None, root_depth=0):
        self.K_ucb = K_ucb
        self.env = env
        self.rollout_policy = rollout_policy
        self.erm_beta = erm_beta
        self.root_depth = 0

        # Create tree.
        self.root = self.init_tree(initial_state)

    def _hash_state(self, state):
        node_repr = str(state["state"]) + "_" + str(state["t"])
        return hash(node_repr)

    def set_root_depth(self, new_root_depth: int):
        self.root_depth = new_root_depth

    def init_tree(self, initial_state):
        # Initialize tree.
        avail_actions = self.env.available_actions(initial_state)
        root = DecisionNode(state=initial_state, available_actions=avail_actions, is_root=True)
        return root

    def grow_tree(self):
        """
        Monte Carlo Tree Seach main loop: (i) Selection; (ii) Expansion;
            (iii) Simulation (rollout); and (iv) Backup.
        """
        decision_node = self.root
        decision_node.visits += 1  # Increment visits counter at root.

        depth = 0
        # Selection and expansion.
        while (not decision_node.is_final) and decision_node.visits > 0:
            # Select action (and retrieve the random node associated with the action).
            a = self.select(decision_node, self.root_depth + depth)
            random_node = decision_node.get_random_node(a)

            # Generate the next decision node (next state).
            (new_decision_node, c) = self.select_outcome(random_node)

            # Store cost in random node.
            random_node.cost = c

            # Add next decision node to random node children. Internally checks whether
            # the created next_decision_node (next state) already exists in tree; if it
            # exists, the existing node is returned instead.
            new_decision_node = random_node.add_child(new_decision_node, self._hash_state)

            decision_node = new_decision_node

            depth += 1

        # Simulation (rollout).
        cumulative_cost = self.rollout(decision_node)

        # Backup.
        while not decision_node.is_root:
            decision_node.visits += 1
            random_node = decision_node.father
            cumulative_cost = random_node.cost + self.env.gamma * cumulative_cost
            random_node.costs_list.append(cumulative_cost)
            random_node.visits += 1
            decision_node = random_node.father

    def select(self, x: DecisionNode, depth: int):
        """
        Selects the action to play from the current decision node.

        :param x: (DecisionNode) current decision node.
        :param depth: (int) depth of decision node.

        :return: (int) action.
        """

        def scoring(k):
            if x.children[k].visits > 0:
                costs = np.array(x.children[k].costs_list)
                beta_depth = self.erm_beta * self.env.gamma**depth
                erm = (1.0 / beta_depth) * np.log((1.0 / len(costs)) * np.sum(np.exp(beta_depth * costs)))
                return erm - self.K_ucb * np.sqrt(np.sqrt(x.visits) / x.children[k].visits)
            else:
                return -np.inf

        return min(x.children, key=scoring)

    def select_outcome(self, random_node: RandomNode):
        """
        Given a RandomNode, returns a new DecisionNode corresponding to the next state.

        :param: random_node: (RandomNode) the random node from which selects the next state.
        :return: (DecisionNode) the selected Decision Node.
        """
        new_state, c, done = self.env.step(random_node.father.state, random_node.action)
        avail_actions = self.env.available_actions(new_state)
        new_decision_node = DecisionNode(
            state=new_state, father=random_node, available_actions=avail_actions, is_final=done
        )
        return new_decision_node, c

    def rollout(self, initial_node: DecisionNode):
        """
        Evaluates a DecisionNode by performing a rollout with a given policy.

        :param initial_node: (DecisionNone) initial node.
        :return: (float) the cumulative cost observed during the tree traversing.
        """
        discounted_cumulative_costs = 0
        done = initial_node.is_final
        s = initial_node.state
        discount = 1.0
        while not done:
            if self.rollout_policy:
                a = self.rollout_policy(s)
            else:
                # Random policy.
                avail_actions = self.env.available_actions(s)
                a = np.random.choice(avail_actions)

            s, c, done = self.env.step(s, a)
            discounted_cumulative_costs += c * discount
            discount *= self.env.gamma

        return discounted_cumulative_costs

    def learn(self, n_iters, progress_bar=False):
        """
        Expand the tree.

        :param: n_iters: (int) number of tree traversals.
        :param: progress_bar: (bool) whether to show a progress bar (tqdm).
        """
        if progress_bar:
            iterations = tqdm(range(n_iters))
        else:
            iterations = range(n_iters)
        for _ in iterations:
            self.grow_tree()

    def best_action(self):
        """
        Returns the most visited action.

        :return: (int) the best action according to the number of visits principle.
        """
        number_of_visits_children = [node.visits for node in self.root.children.values()]
        index_best_action = np.argmax(number_of_visits_children)

        return list(self.root.children.values())[index_best_action].action

    def update_root_node(self, selected_action: int, new_state: dict):
        """
        Updates the root node of the planning tree.

        :param: (int) selected_action: previously selected action.
        :param: (dict) new_state: the new state of the environment.
        """
        next_state_hash = self._hash_state(new_state)
        if next_state_hash in self.root.children[selected_action].children:
            self.root = self.root.children[selected_action].children[next_state_hash]
            self.root.is_root = True
            return True
        else:
            return False
