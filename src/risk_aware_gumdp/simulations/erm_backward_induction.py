import json
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import tyro
from tqdm import tqdm

from risk_aware_gumdp.simulations.algos import ERMBackwardInduction
from risk_aware_gumdp.simulations.envs import ENVS


@dataclass
class Args:
    N: int = 100
    num_processors: int = 10
    env: str = "linear_mdp"
    H: int = 20
    erm_beta: float = 5.0


def create_exp_name(args: dict) -> str:
    return (
        args["env"]
        + "_"
        + args["algo"]
        + "_gamma_"
        + str(args["gamma"])
        + "_beta_"
        + str(args["erm_beta"])
        + "_"
        + str(datetime.today().strftime("%Y-%m-%d-%H-%M-%S"))
    )


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Env:
    def __init__(self, mdp, H):
        self.mdp = mdp
        self.H = H
        self.gamma = mdp["gamma"]

    def available_actions(self, state):
        return self.mdp["actions"]

    def sample_initial_state(self):
        # Sample initial state.
        state = np.random.choice(self.mdp["states"], p=self.mdp["p_0"])
        extended_state = {"state": state, "t": 0}  # (state, timestep).
        return extended_state

    def step(self, extended_state, a):
        # Simulate a step of the finite-horizon MDP.
        state_t, timestep_t = extended_state["state"], extended_state["t"]
        next_state = np.random.choice(self.mdp["states"], p=self.mdp["P"][a, state_t, :])
        next_timestep = timestep_t + 1
        next_extended_state = {"state": next_state, "t": next_timestep}
        cost_t = self.mdp["C"][state_t, a]

        if timestep_t == self.H:
            terminated = True
        else:
            terminated = False

        return next_extended_state, cost_t, terminated


def get_env(env_name, H):
    env_dict = ENVS[env_name]
    env = Env(env_dict, H)
    return env


def simulate_ERM_backward_induction(env, H, erm_beta):
    # Run ERMBackwardInduction algorithm.
    adjusted_erm_beta = erm_beta * ((1 - env.mdp["gamma"]) / (1 - env.mdp["gamma"] ** H))
    erm_algo = ERMBackwardInduction(env, adjusted_erm_beta, H)
    erm_opt_policy = erm_algo.compute()

    # Sample initial state.
    extended_state = env.sample_initial_state()

    # Simulate until termination.
    occupancy = np.zeros(((len(env.mdp["states"])), len(env.mdp["actions"])))
    for t in tqdm(range(H)):
        selected_action = erm_opt_policy[t, extended_state["state"]]

        # Environment step.
        occupancy[extended_state["state"], selected_action] += env.mdp["gamma"] ** t
        extended_state, _, _ = env.step(extended_state, selected_action)

    normalized_occupancy = ((1 - env.mdp["gamma"]) / (1 - env.mdp["gamma"] ** H)) * occupancy
    objective_val = np.sum(normalized_occupancy * env.mdp["C"])
    return objective_val


def run(cfg, seed):
    print("Running seed=", seed)
    np.random.seed(seed)

    # Instantiate environment.
    env = get_env(cfg.env, cfg.H)
    val = simulate_ERM_backward_induction(env=env, H=cfg.H, erm_beta=cfg.erm_beta)
    return val


def main(cfg):
    if cfg.env not in ["linear_mdp"]:
        raise ValueError("ERM backward induction only works for the linear MDP environment.")

    # Setup experiment data folder.
    exp_name = create_exp_name(
        {"env": cfg.env, "algo": "erm-backward-induction", "gamma": ENVS[cfg.env]["gamma"], "erm_beta": cfg.erm_beta}
    )

    exp_path = "results/illustrative/" + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print("\nExperiment ID:", exp_name)
    print("Config:")
    print(cfg)

    # Simulate.
    print("\nSimulating...")

    with mp.Pool(processes=cfg.num_processors) as pool:
        f_vals = pool.starmap(run, [(cfg, t) for t in range(cfg.N)])
        pool.close()
        pool.join()

    f_vals = np.array(f_vals)
    exp_data = {}
    exp_data["config"] = asdict(cfg)
    exp_data["f_vals"] = f_vals
    exp_data["env"] = cfg.env

    # Dump dict.
    f = open(exp_path + "/exp_data.json", "w")
    dumped = json.dumps(exp_data, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()
    print(exp_name)
    return exp_name


if __name__ == "__main__":
    print(Args)
    args = tyro.cli(Args)
    print(args)
    main(cfg=args)
