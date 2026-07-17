import math
import os
import sys
from typing import Any, Mapping, MutableMapping, Union

import omegaconf

MAX_32_BIT_INT = 2147483647
MIN_32_BIT_INT = -2147483648


class ConsoleFileManager:
    def __init__(self, cfg: omegaconf.DictConfig):
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        p = os.path.join(os.getcwd(), cfg.logger.base_exp_path, cfg.system.system_name, cfg.logger.kwargs.unique_token)
        os.makedirs(p, exist_ok=True)
        self._out_file = open(os.path.join(p, "out.txt"), "a", encoding="utf-8")
        self._err_file = open(os.path.join(p, "err.txt"), "a", encoding="utf-8")

    def __enter__(self):
        sys.stdout = Tee(self._orig_stdout, self._out_file)
        sys.stderr = Tee(self._orig_stderr, self._err_file)

    def __exit__(self, type, value, traceback):
        self._out_file.close()
        self._err_file.close()
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr


class StringifyValue:
    """Taken from https://github.com/neptune-ai/neptune-client."""

    def __init__(self, value: Any):
        # check if it's an integer outside 32bit range and cast it to float
        if isinstance(value, int) and (value > MAX_32_BIT_INT or value < MIN_32_BIT_INT):
            value = float(value)
        if self._is_unsupported_float(value):
            value = str(value)

        self.__value = value

    @property
    def value(self):
        return self.__value

    def __str__(self):
        return str(self.__value)

    def __repr__(self):
        return repr(self.__value)

    def _is_unsupported_float(self, value) -> bool:
        if isinstance(value, float):
            return math.isinf(value) or math.isnan(value)

        return False


class Tee:
    def __init__(self, *files):
        self.files = files

    def isatty(self):
        return False

    def write(self, message):
        for file in self.files:
            file.write(message)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()


def stringify_unsupported(value: Any) -> Union[StringifyValue, Mapping, list]:
    """Taken from https://github.com/neptune-ai/neptune-client.

    Helper function that converts unsupported values in a collection or dictionary to strings.

    Args:
        value (Any): A dictionary with values or a collection

    Example:
        >>> import neptune
        >>> run = neptune.init_run()
        >>> complex_dict = {"tuple": ("hi", 1), "metric": 0.87}
        >>> run["complex_dict"] = complex_dict
        >>> # (as of 1.0.0) error - tuple is not a supported type
        ... from neptune.utils import stringify_unsupported
        >>> run["complex_dict"] = stringify_unsupported(complex_dict)

        For more information, see:
        https://docs.neptune.ai/setup/neptune-client_1-0_release_changes/#no-more-implicit-casting-to-string

    """
    if isinstance(value, MutableMapping):
        return {str(k): stringify_unsupported(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [stringify_unsupported(el) for el in value]

    return str(StringifyValue(value=value))
