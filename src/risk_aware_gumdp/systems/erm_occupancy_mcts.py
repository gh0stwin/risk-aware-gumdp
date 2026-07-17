import copy
import gc
import os
import time
from typing import Any

import hydra
import numpy as np
import numpy.typing
import psutil
from colorama import Fore, Style
from gymnasium.experimental.functional import FuncEnv
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

import risk_aware_gumdp.utils.make_env_gym as environments
from risk_aware_gumdp.mcts.bonus import bonus_factory
from risk_aware_gumdp.mcts.mcts import ErmOccupancyMCTS
from risk_aware_gumdp.utils.logger import LogEvent, Logger


def learn(
    rng: np.random.Generator, env: FuncEnv, config: DictConfig
) -> tuple[numpy.typing.ArrayLike, list[Any], list[int], numpy.typing.ArrayLike]:
    states, actions, erm = [], [], []
    state = env.initial(rng.spawn(1)[0])
    bonus = bonus_factory(**config.system.bonus)

    for i in range(config.system.h):
        mcts = ErmOccupancyMCTS(
            rng.spawn(1)[0], state, env, config.system.beta, bonus=bonus, max_rollouts=config.system.expansion_steps
        )

        mcts.learn()
        action, risk = mcts.best_action(mcts.root)
        print([(chance_node.risk, chance_node.num_visits, chance_node._costs) for chance_node in mcts.root.children])
        states.append(state.state_index)
        actions.append(action)
        erm.append(risk)
        prev_state = state
        state = env.transition(prev_state, action, rng.spawn(1)[0])

        # debug code
        gc.collect()

        if config.system.debug:
            proc = psutil.Process(os.getpid())
            mem = proc.memory_info()[0] / float(2**20)
            children = len(proc.children())
            print(f"Level: {i}\n\tMemory: {mem} MiB\n\tNum child procs: {children}\n", flush=True)

    states.append(state.state_index)
    erm = np.stack(erm, axis=0)
    return env.reward(prev_state, action, state, None), states, actions, erm


def run_experiment(_config: DictConfig) -> float:
    config = copy.deepcopy(_config)

    # RNG key
    rng = np.random.default_rng(config.arch.seed)

    # Environment setup
    env, _ = environments.make_env(config=config)

    # Logger setup
    logger = Logger(config)
    cfg = OmegaConf.to_container(config, resolve=True)

    # Print config
    pprint(cfg)

    # Run experiment
    cost, states, actions, erm = learn(rng, env, config)
    logger.log({"cost": cost}, 0, 0, LogEvent.TRAIN)

    for i, (s, a, e) in enumerate(zip(states[:-1], actions, erm)):
        logger.log({"state": s, "action": a, "erm": e}, i, 0, LogEvent.TRAIN)

    logger.log({"state": states[-1]}, len(states) - 1, 0, LogEvent.TRAIN)
    return cost


@hydra.main(config_path="../configs/default", config_name="default_erm_occupancy_mcts.yaml", version_base="1.3")
def hydra_entry_point(cfg: DictConfig) -> float:
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    start = time.monotonic()
    eval_performance = run_experiment(cfg)
    end = time.monotonic()
    print(f"{Fore.CYAN}{Style.BRIGHT}Distributional MCTS experiment completed in {end - start:.2f}s.{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
