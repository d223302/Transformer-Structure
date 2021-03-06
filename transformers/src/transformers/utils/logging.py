# coding=utf-8
# Copyright 2020 Optuna, Hugging Face
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Logging utilities. """

import logging
import threading
from logging import CRITICAL  # NOQA
from logging import DEBUG  # NOQA
from logging import ERROR  # NOQA
from logging import FATAL  # NOQA
from logging import INFO  # NOQA
from logging import NOTSET  # NOQA
from logging import WARN  # NOQA
from logging import WARNING  # NOQA
from typing import Optional


_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None


def _get_library_name() -> str:

    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:

    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:

    global _default_handler

    with _lock:
        if _default_handler:
            # This library has already configured the library root logger.
            return
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.

        # Apply our default configuration to the library root logger.
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(logging.WARN)
        library_root_logger.propagate = False


def _reset_library_root_logger() -> None:

    global _default_handler

    with _lock:
        if not _default_handler:
            return

        library_root_logger = _get_library_root_logger()
        library_root_logger.removeHandler(_default_handler)
        library_root_logger.setLevel(logging.NOTSET)
        _default_handler = None


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger with the specified name.
    This function is not supposed to be directly accessed by library users.
    """

    if name is None:
        name = _get_library_name()

    _configure_library_root_logger()
    return logging.getLogger(name)


def get_verbosity() -> int:
    """Return the current level for the HuggingFace Transformers's root logger.
    Returns:
        Logging level, e.g., ``transformers.logging.DEBUG`` and ``transformers.logging.INFO``.
    .. note::
        HuggingFace Transformers has following logging levels:
        - ``transformers.logging.CRITICAL``, ``transformers.logging.FATAL``
        - ``transformers.logging.ERROR``
        - ``transformers.logging.WARNING``, ``transformers.logging.WARN``
        - ``transformers.logging.INFO``
        - ``transformers.logging.DEBUG``
    """

    _configure_library_root_logger()
    return _get_library_root_logger().getEffectiveLevel()


def set_verbosity(verbosity: int) -> None:
    """Set the level for the HuggingFace Transformers's root logger.
    Args:
        verbosity:
            Logging level, e.g., ``transformers.logging.DEBUG`` and ``transformers.logging.INFO``.
    """

    _configure_library_root_logger()
    _get_library_root_logger().setLevel(verbosity)


def set_verbosity_info():
    return set_verbosity(INFO)


def set_verbosity_warning():
    return set_verbosity(WARNING)


def set_verbosity_debug():
    return set_verbosity(DEBUG)


def set_verbosity_error():
    return set_verbosity(ERROR)


def disable_default_handler() -> None:
    """Disable the default handler of the HuggingFace Transformers's root logger."""

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().removeHandler(_default_handler)


def enable_default_handler() -> None:
    """Enable the default handler of the HuggingFace Transformers's root logger."""

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().addHandler(_default_handler)


def disable_propagation() -> None:
    """Disable propagation of the library log outputs.
    Note that log propagation is disabled by default.
    """

    _configure_library_root_logger()
    _get_library_root_logger().propagate = False


def enable_propagation() -> None:
    """Enable propagation of the library log outputs.
    Please disable the HuggingFace Transformers's default handler to prevent double logging if the root logger has
    been configured.
    """

    _configure_library_root_logger()
    _get_library_root_logger().propagate = True
