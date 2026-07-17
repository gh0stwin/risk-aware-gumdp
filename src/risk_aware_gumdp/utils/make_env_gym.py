import copy

import hydra
from gymnasium.experimental.functional import FuncEnv
from omegaconf import DictConfig

import risk_aware_gumdp.gym


def make_env(config: DictConfig) -> tuple[FuncEnv, FuncEnv]:
    if config.env.env_name == "gym":
        envs = _make_env_gym(config)
    else:
        raise ValueError(f"env library {config.env.env_name} is not supported.")

    envs = apply_optional_wrappers(envs, config)
    return envs


def apply_optional_wrappers(envs: tuple[FuncEnv, FuncEnv], config: DictConfig) -> tuple[FuncEnv]:
    """Apply optional wrappers to the environments.

    Args:
        envs (Tuple[Environment, Environment]): The training and evaluation environments to wrap.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """
    train_env, eval_env = envs

    if "wrappers" in config.env and config.env.wrappers is not None:
        for i in range(len(config.env.wrappers.train)):
            train_env = hydra.utils.instantiate(config.env.wrappers.train[i])(env=train_env)

        for i in range(len(config.env.wrappers.eval)):
            eval_env = hydra.utils.instantiate(config.env.wrappers.eval[i])(env=eval_env)

    return train_env, eval_env  # type: ignore


def _make_env_gym(config: DictConfig) -> tuple[FuncEnv, FuncEnv]:
    kw = {}
    # env_kwargs = dict(copy.deepcopy(config.env.kwargs))

    for k in config.env.kwargs:
        if isinstance(config.env.kwargs[k], DictConfig) and "_target_" in config.env.kwargs[k]:
            kw[k] = hydra.utils.instantiate(config.env.kwargs[k])
        else:
            kw[k] = copy.deepcopy(config.env.kwargs[k])

    kw["h"] = config.system.h
    kw["gamma"] = config.system.gamma
    print(kw)
    env = risk_aware_gumdp.gym.make_gym(config.env.scenario.name, **kw)
    eval_env = risk_aware_gumdp.gym.make_gym(config.env.scenario.name, **kw)
    config.env.max_expand = int(env.max_state_links())
    return env, eval_env
