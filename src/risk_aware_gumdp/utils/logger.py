import abc
import json
import logging
import os
import pathlib as pl
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Union
import uuid

import numpy as np
from colorama import Fore, Style
from omegaconf import DictConfig
from pandas.io.json._normalize import _simple_json_normalize as flatten_dict

from risk_aware_gumdp.utils.misc import stringify_unsupported


class LogEvent(Enum):
    ACT = "actor"
    TRAIN = "trainer"
    EVAL = "evaluator"
    ABSOLUTE = "absolute"
    MISC = "misc"


class Logger:
    """The main logger for Stoix systems.

    Thin wrapper around the MultiLogger that is able to describe arrays of metrics
    and calculate environment specific metrics if required (e.g solve_rate).
    """

    def __init__(self, config: DictConfig) -> None:
        self.logger: BaseLogger = _make_multi_logger(config)
        self.cfg = config

    def log(self, metrics: Dict, t: int, t_eval: int, event: LogEvent) -> None:
        """Log a dictionary metrics at a given timestep.

        Args:
            metrics (Dict): dictionary of metrics to log.
            t (int): the current timestep.
            t_eval (int): the number of previous evaluations.
            event (LogEvent): the event that the metrics are associated with.
        """
        # Ideally we want to avoid special metrics like this as much as possible.
        # Might be better to calculate this outside as we want to keep the number of these
        # if statements to a minimum.
        if "solve_episode" in metrics:
            metrics = self.calc_solve_rate(metrics, event)

        if event == LogEvent.TRAIN:
            # We only want to log mean losses, max/min/std don't matter.
            assert all([isinstance(m, (np.ndarray, list, np.number, int, float)) for m in metrics.values()]), (
                metrics,
                [m.__class__ for m in metrics.values()],
            )

            metrics = {m: np.mean(v) for m, v in metrics.items()}
        elif event != LogEvent.MISC:
            # {metric1_name: [metrics], metric2_name: ...} ->
            # {metric1_name: {mean: metric, max: metric, ...}, metric2_name: ...}
            assert all([isinstance(m, (np.ndarray, list, np.number, int, float)) for m in metrics.values()]), (
                metrics,
                [m.__class__ for m in metrics.values()],
            )

            metrics = {m: describe(v) for m, v in metrics.items()}

        self.logger.log_dict(metrics, t, t_eval, event)

    def log_file(self, file_path: str | pl.Path) -> None:
        self.logger.log_file(file_path)

    def log_video(self, key: str, video: np.ndarray, step: int, event: LogEvent) -> None:
        self.logger.log_video(key, video, step, event)

    def calc_solve_rate(self, episode_metrics: Dict, event: LogEvent) -> Dict:
        """Log the solve rate of the environment's episodes."""
        # Get the number of episodes used to evaluate.
        if event == LogEvent.ABSOLUTE:
            # To measure the absolute metric, we evaluate the best policy
            # found across training over 10 times the evaluation episodes.
            # For more details on the absolute metric please see:
            # https://arxiv.org/abs/2209.10485.
            n_episodes = self.cfg.arch.num_eval_episodes * 10
        else:
            n_episodes = self.cfg.arch.num_eval_episodes

        # Calculate the solve rate.
        n_solve_episodes: int = np.sum(episode_metrics["solve_episode"])
        solve_rate = (n_solve_episodes / n_episodes) * 100

        episode_metrics["solve_rate"] = solve_rate
        episode_metrics.pop("solve_episode")

        return episode_metrics

    def stop(self) -> None:
        """Stop the logger."""
        self.logger.stop()


class BaseLogger(abc.ABC):
    @abc.abstractmethod
    def __init__(self, cfg: DictConfig) -> None:
        self._cfg = cfg

    @abc.abstractmethod
    def log_stat(self, key: str, value: Any, step: int, eval_step: int, event: LogEvent) -> None:
        """Log a single metric."""
        raise NotImplementedError

    def log_dict(self, data: Dict, step: int, eval_step: int, event: LogEvent) -> None:
        """Log a dictionary of metrics."""
        # in case the dict is nested, flatten it.
        data = flatten_dict(data, sep="/")

        for key, value in data.items():
            self.log_stat(key, value, step, eval_step, event)

    def stop(self) -> None:
        """Stop the logger."""
        return None


class MultiLogger(BaseLogger):
    """Logger that can log to multiple loggers at oncce."""

    def __init__(self, loggers: List[BaseLogger]) -> None:
        self.loggers = loggers

    def log_stat(self, key: str, value: Any, step: int, eval_step: int, event: LogEvent) -> None:
        for logger in self.loggers:
            logger.log_stat(key, value, step, eval_step, event)

    def log_dict(self, data: Dict, step: int, eval_step: int, event: LogEvent) -> None:
        for logger in self.loggers:
            logger.log_dict(data, step, eval_step, event)

    def stop(self) -> None:
        for logger in self.loggers:
            logger.stop()


class ConsoleLogger(BaseLogger):
    """Logger for writing to stdout."""

    _EVENT_COLOURS = {
        LogEvent.TRAIN: Fore.MAGENTA,
        LogEvent.EVAL: Fore.GREEN,
        LogEvent.ABSOLUTE: Fore.BLUE,
        LogEvent.ACT: Fore.CYAN,
        LogEvent.MISC: Fore.YELLOW,
    }

    def __init__(self, cfg: DictConfig) -> None:
        self.logger = logging.getLogger()
        self.logger.handlers = []

        ch = logging.StreamHandler()
        formatter = logging.Formatter(f"{Fore.CYAN}{Style.BRIGHT}%(message)s", "%H:%M:%S")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # Set to info to suppress debug outputs.
        self.logger.setLevel("INFO")

    def log_stat(self, key: str, value: Any, step: int, eval_step: int, event: LogEvent) -> None:
        colour = self._EVENT_COLOURS[event]

        # Replace underscores with spaces and capitalise keys.
        key = key.replace("_", " ")
        self.logger.info(f"{colour}{Style.BRIGHT}{event.value.upper()} - {key}: {value:.3f}{Style.RESET_ALL}")

    def log_dict(self, data: Dict, step: int, eval_step: int, event: LogEvent) -> None:
        # in case the dict is nested, flatten it.
        data = flatten_dict(data, sep=" ")
        colour = self._EVENT_COLOURS[event]
        log_data = {}

        for k, v in data.items():
            k = k.replace("_", " ")

            try:
                if isinstance(v, int):
                    log_data[k] = v

                log_data[k] = f"{float(v):.3f}"
            except TypeError:
                continue

        if len(log_data) == 0:
            return

        log_str = " | ".join([f"{k}: {v}" for k, v in log_data.items()])
        self.logger.info(f"{colour}{Style.BRIGHT}{event.value.upper()} - {log_str}{Style.RESET_ALL}")


class JsonLogger(BaseLogger):
    """Json logger for marl-eval."""

    # These are the only metrics that marl-eval needs to plot.
    _METRICS_TO_LOG = ["episode_return/mean", "solve_rate", "steps_per_second"]

    def __init__(self, cfg: DictConfig) -> None:
        json_exp_path = get_logger_path(cfg, "json")
        json_logs_path = os.path.join(cfg.logger.base_exp_path, f"{json_exp_path}/{cfg.logger.kwargs.unique_token}")
        self.data = {"config": stringify_unsupported(cfg)}

        # if a custom path is specified, use that instead
        if cfg.logger.kwargs.json_path is not None:
            json_logs_path = os.path.join(cfg.logger.base_exp_path, "json", cfg.logger.kwargs.json_path)

        if cfg.logger.kwargs.upload_json_data:
            cfg.logger.log_files.append(f"{json_logs_path}/metrics.json")

        self.file_path = pl.Path(cfg.logger.base_exp_path) / cfg.logger.kwargs.name
        self.file_path.mkdir(parents=True, exist_ok=True)
        self.file_path /= f"{cfg.logger.kwargs.unique_token}.json"

    def log_stat(self, key: str, value: Any, step: int, eval_step: int, event: LogEvent) -> None:
        # The key is in the format <metric_name>/<aggregation_fn> so we need to change it to:
        # <agg fn>_<metric_name>
        if "/" in key:
            key = "_".join(reversed(key.split("/")))

        # JsonWriter can't serialize jax arrays
        value = value.item() if isinstance(value, (np.ndarray,)) else value

        if isinstance(value, (np.floating)):
            value = float(value)
        elif isinstance(value, (np.integer)):
            value = int(value)

        event_str = event.name

        if event_str not in self.data:
            self.data[event_str] = {}

        if key not in self.data[event_str]:
            self.data[event_str][key] = []

        assert step == len(self.data[event_str][key])
        self.data[event_str][key].append(value)

        with open(self.file_path, "w") as f:
            json.dump(self.data, f, indent=4)


def _make_multi_logger(cfg: DictConfig) -> BaseLogger:
    """Creates a MultiLogger given a config"""

    loggers: List[BaseLogger] = []

    if cfg.logger.kwargs.unique_token is None:
        cfg.logger.kwargs.unique_token = str(uuid.uuid4())

    if cfg.logger.use_console:
        loggers.append(ConsoleLogger(cfg))

    if cfg.logger.use_json:
        loggers.append(JsonLogger(cfg))

    return MultiLogger(loggers)


def get_logger_path(config: DictConfig, logger_type: str) -> str:
    """Helper function to create the experiment path."""
    return f"{logger_type}/{config.system.system_name}/{config.env.env_name}"


def describe(x: np.ndarray) -> Union[Dict[str, np.ndarray], np.ndarray]:
    """Generate summary statistics for an array of metrics (mean, std, min, max)."""

    if not isinstance(x, (np.ndarray,)):
        return x
    elif x.size <= 1:
        return x.item()

    # np instead of jnp because we don't jit here
    return {"mean": np.mean(x), "std": np.std(x), "min": np.min(x), "max": np.max(x)}
