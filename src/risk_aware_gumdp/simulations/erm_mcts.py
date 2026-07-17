import json
import multiprocessing as mp
import os
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import tyro
from tqdm import tqdm

from risk_aware_gumdp.simulations.algos import ERM_MCTS
from risk_aware_gumdp.simulations.envs import Occupancy_MDP, get_env


@dataclass
class Args:
    N: int = 100
    num_processors: int = 10
    env: str = "linear_mdp"
    H: int = 20
    n_iter_per_timestep: int = 500  # MCTS number of tree expansion steps per timestep
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
        + str(args["seed"])
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


def simulate_ERM_MCTS(mdp, H, erm_beta, n_iter_per_timestep=1_000):
    # Instantiate an extended occupancy MDP.
    occupancy_env = Occupancy_MDP(mdp, H)

    # Sample initial state from the extended MDP.
    extended_state = occupancy_env.sample_initial_state()
    mcts = ERM_MCTS(
        initial_state=extended_state, env=occupancy_env, K_ucb=1.0, erm_beta=erm_beta, rollout_policy=None, root_depth=0
    )

    # Simulate until termination.
    # occupancy = np.zeros(( (len(mdp["states"])), len(mdp["actions"])) )
    cumulative_costs = 0.0

    for t in tqdm(range(H + 1)):
        mcts.learn(n_iters=n_iter_per_timestep)
        selected_action = mcts.best_action()

        # Environment step.
        # occupancy[extended_state["state"], selected_action] += mdp["gamma"]**t
        extended_state, cost, _ = occupancy_env.step(extended_state, selected_action)
        cumulative_costs += cost
        updated_root = mcts.update_root_node(selected_action, extended_state)

        if updated_root:
            # Next state is present in the tree - reuse subtree.
            mcts.set_root_depth(t + 1)
        else:
            # Next state is not present in the tree - build a new tree.
            mcts = ERM_MCTS(
                initial_state=extended_state,
                env=occupancy_env,
                K_ucb=1.0,
                erm_beta=erm_beta,
                rollout_policy=None,
                root_depth=t + 1,
            )

    return cumulative_costs


def run(cfg, seed):
    print("Running seed=", seed)
    np.random.seed(seed)

    # Instantiate MDP.
    env = get_env(cfg.env)
    mcts_f_val = simulate_ERM_MCTS(mdp=env, H=cfg.H, erm_beta=cfg.erm_beta, n_iter_per_timestep=cfg.n_iter_per_timestep)
    return mcts_f_val


def main(cfg, arg_seed=0, data_folder_path=None):
    # Setup experiment data folder.
    env = get_env(cfg.env)
    exp_name = create_exp_name(
        {"env": cfg.env, "algo": "erm-mcts", "gamma": env["gamma"], "erm_beta": cfg.erm_beta, "seed": arg_seed}
    )

    exp_path = "results/illustrative/" + exp_name
    os.makedirs(exp_path, exist_ok=True)
    print("\nExperiment ID:", exp_name)
    print("Config:")
    print(cfg)

    # Simulate.
    print("\nSimulating...")

    with mp.Pool(processes=cfg.num_processors) as pool:
        f_vals = pool.starmap(run, [(cfg, (1_000 * arg_seed) + t) for t in range(cfg.N)])
        pool.close()
        pool.join()

    f_vals = np.array(f_vals)
    exp_data = {}
    exp_data["config"] = asdict(cfg)
    exp_data["f_vals"] = f_vals
    exp_data["env"] = env
    exp_data["env"]["f"] = None

    # Dump dict.
    f = open(exp_path + "/exp_data.json", "w")
    dumped = json.dumps(exp_data, cls=NumpyEncoder)
    json.dump(dumped, f)
    f.close()
    print(exp_name)
    return exp_name


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(cfg=args)
