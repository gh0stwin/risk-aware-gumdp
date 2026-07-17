import json
import numpy as np


FISHWOOD_ENV_FISH_PROB = 0.4
FISHWOOD_ENV_WOOD_PROB = 0.6

ENVS = {
    "entropy_mdp_motivation": {
        "states": [0, 1, 2, 3],
        "actions": [0, 1],  # Action 0 = left, action 1 = right
        "gamma": 0.99,
        "p_0": [0.0, 1.0, 0.0, 0.0],
        "P": np.array(
            [
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.9, 0.1], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            ]
        ),
    },
    "entropy_mdp": {
        "states": [0, 1, 2, 3, 4],
        "actions": [0, 1],  # Action 0 = left, action 1 = right
        "gamma": 0.9,
        "p_0": [0.0, 1.0, 0.0, 0.0, 0.0],
        "P": np.array(
            [
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.8, 0.2],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                ],
            ]
        ),
    },
    "linear_mdp": {  # Standard MDP.
        "states": [0, 1, 2, 3],
        "actions": [0, 1],  # Action 0 = left, action 1 = right
        "gamma": 0.9,
        "p_0": [1.0, 0.0, 0.0, 0.0],
        "P": np.array(
            [
                [
                    [0.0, 0.0, 0.85, 0.15],
                    [0.1, 0.9, 0.0, 0.0],
                    [0.1, 0.0, 0.9, 0.0],
                    [0.1, 0.0, 0.0, 0.9],
                ],
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.1, 0.9, 0.0, 0.0],
                    [0.1, 0.0, 0.9, 0.0],
                    [0.1, 0.0, 0.0, 0.9],
                ],
            ]
        ),
        "C": np.array(
            [
                [0.0, 0.0],  # c=0 / 20.
                [0.25, 0.25],  # c=5 / 20.
                [0.05, 0.05],  # c=1 / 20.
                [1.0, 1.0],
            ]
        ),  # c=20 / 20.
    },
    "multi_objective_mdp": {
        # Inspired in the FishWood environment.
        # (https://mo-gymnasium.farama.org/environments/fishwood/)
        "states": [0, 1, 2, 3, 4],
        "actions": [0, 1],  # Action 0 = go fishing; Action 1 = go to the woods
        "gamma": 0.99,
        "p_0": [1.0, 0.0, 0.0, 0.0, 0.0],
        "P": np.array(
            [
                [
                    [0.0, FISHWOOD_ENV_FISH_PROB, 1.0 - FISHWOOD_ENV_FISH_PROB, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, FISHWOOD_ENV_WOOD_PROB, 1.0 - FISHWOOD_ENV_WOOD_PROB],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ]
        ),
        "R_fish": np.array(
            [
                [0.0, 0.0],  # 0
                [1.0, 1.0],  # 1
                [-0.5, -0.5],  # 2
                [0.0, 0.0],  # 3
                [0.0, 0.0],
            ]
        ),  # 4
        "R_woods": np.array(
            [
                [0.0, 0.0],  # 0
                [0.0, 0.0],  # 1
                [0.0, 0.0],  # 2
                [0.2, 0.2],  # 3
                [-0.2, -0.2],
            ]
        ),  # 4
    },
    "imitation_learning_mdp": {
        "states": [0, 1, 2, 3],
        "actions": [0, 1],  # Action 0 = left, action 1 = right
        "gamma": 0.9,
        "p_0": [1.0, 0.0, 0.0, 0.0],
        "P": np.array(
            [
                [
                    [0.0, 0.0, 0.85, 0.15],
                    [0.9, 0.1, 0.0, 0.0],
                    [0.9, 0.0, 0.1, 0.0],
                    [0.05, 0.0, 0.0, 0.95],
                ],
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [0.9, 0.1, 0.0, 0.0],
                    [0.9, 0.0, 0.1, 0.0],
                    [0.05, 0.0, 0.0, 0.95],
                ],
            ]
        ),
    },
}

# Imitation learning: imitate the empirical occupancies induced by a given behavior policy.
# Load empirical occupancy to imitate.
IMITATION_LEARNING_EXP_TO_IMITATE = "imitation_learning_mdp_human_gamma_0.9_2026-02-11-15-27-56"
with open("data/" + IMITATION_LEARNING_EXP_TO_IMITATE + "/exp_data.json", "r") as f:
    loaded_exp_data = json.load(f)
    loaded_exp_data = json.loads(loaded_exp_data)
f.close()
ENVS["imitation_learning_mdp"]["d_beta"] = np.array(loaded_exp_data["empirical_occupancy"]).flatten()
# print("d_beta to imitate:", ENVS["imitation_learning_mdp"]["d_beta"])


def get_env(env_name, normalize_obj=True):
    if env_name == "multi_objective_mdp":
        raise ValueError("Please specify the multi-objective MDP objective (weighted, max, etc.).")

    if env_name in ["multi_objective_mdp_weighted", "multi_objective_mdp_max", "multi_objective_mdp_min"]:
        # All multi-objective MDPs share the same MDP definition except for the objective.
        env = ENVS["multi_objective_mdp"]
    else:
        env = ENVS[env_name]

    if env_name in ["entropy_mdp", "entropy_mdp_motivation"]:
        # Define objective function for entropy maximization MDP.
        d_lower_bounds_eps = 1e-10
        SA = len(env["states"]) * len(env["actions"])

        def obj_f(x, sa=SA, eps=d_lower_bounds_eps):
            x = (1 - eps * sa) * x + eps
            x = np.sum(x, axis=1)  # Marginalize over actions.
            return np.dot(x, np.log(x))

        obj = obj_f

        # Normalize.
        if normalize_obj:
            f_min = -np.log(SA)
            f_max = 0.0

            def obj_f_normalized(x, sa=SA, eps=d_lower_bounds_eps):
                aux = obj_f(x, sa=sa)
                return (aux - f_min) / (f_max - f_min)

            obj = obj_f_normalized

        env["f"] = obj

    elif env_name in ["linear_mdp"]:
        # Note: No need to normalize as costs are already normalized.
        def obj_f(x):
            return np.sum(x * ENVS["linear_mdp"]["C"])

        env["f"] = obj_f

    elif env_name in ["multi_objective_mdp_weighted"]:

        def obj_f(x):
            fish_return = -1.0 * np.sum(x * ENVS["multi_objective_mdp"]["R_fish"])
            wood_return = -1.0 * np.sum(x * ENVS["multi_objective_mdp"]["R_woods"])
            return fish_return + wood_return

        obj = obj_f
        env["f"] = obj

    elif env_name in ["multi_objective_mdp_max"]:

        def obj_f(x):
            fish_return = -1.0 * np.sum(x * ENVS["multi_objective_mdp"]["R_fish"])
            wood_return = -1.0 * np.sum(x * ENVS["multi_objective_mdp"]["R_woods"])
            return max(fish_return, wood_return)

        obj = obj_f
        env["f"] = obj

    elif env_name in ["multi_objective_mdp_min"]:

        def obj_f(x):
            fish_return = -1.0 * np.sum(x * ENVS["multi_objective_mdp"]["R_fish"])
            wood_return = -2.0 * np.sum(x * ENVS["multi_objective_mdp"]["R_woods"])
            return min(fish_return, wood_return)

        obj = obj_f
        env["f"] = obj

    elif env_name in ["imitation_learning_mdp"]:

        def obj_f(x):
            x = x.flatten()  # Flatten.
            return np.sum((x - ENVS["imitation_learning_mdp"]["d_beta"]) ** 2)

        obj = obj_f

        # Normalize.
        # if normalize_obj:
        #     f_min = 0.0
        #     f_max = 2.0
        #     def obj_f_normalized(x):
        #         aux = obj_f(x)
        #         return (aux - f_min) / (f_max - f_min)
        #     obj = obj_f_normalized

        env["f"] = obj

    else:
        raise ValueError("Unknown environment.")

    return env


class Occupancy_MDP:
    def __init__(self, mdp, H):
        self.mdp = mdp
        self.H = H
        self.gamma = 1.0  # The occupancy MDP is an *undiscounted* finite-horizon MDP.

    def available_actions(self, state):
        return self.mdp["actions"]

    def sample_initial_state(self):
        # Sample initial state.
        state = np.random.choice(self.mdp["states"], p=self.mdp["p_0"])
        extended_state = {
            "state": state,
            "occupancy": np.zeros((len(self.mdp["states"]), len(self.mdp["actions"]))),
            "t": 0,
        }  # (state, running occupancy, timestep).
        return extended_state

    def step(self, extended_state, a):
        # Simulate a step of the finite-horizon, occupancy MDP.
        state_t, occupancy_t, timestep_t = extended_state["state"], extended_state["occupancy"], extended_state["t"]
        next_occupancy = np.copy(occupancy_t)
        next_occupancy[state_t][a] += self.mdp["gamma"] ** timestep_t
        next_state = np.random.choice(self.mdp["states"], p=self.mdp["P"][a, state_t, :])
        next_timestep = timestep_t + 1
        next_extended_state = {"state": next_state, "occupancy": next_occupancy, "t": next_timestep}
        if timestep_t == self.H:
            normalized_occupancy = ((1 - self.mdp["gamma"]) / (1 - self.mdp["gamma"] ** self.H)) * occupancy_t
            cost = self.mdp["f"](normalized_occupancy)
            terminated = True
        else:
            cost = 0.0
            terminated = False

        return next_extended_state, cost, terminated
