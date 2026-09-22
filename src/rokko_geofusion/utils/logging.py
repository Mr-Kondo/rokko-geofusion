"""Logging setup shared by every script.

Rules (see CLAUDE.md):
* library code uses ``logging.getLogger(__name__)`` and never ``print()``;
* scripts call :func:`setup_logging` exactly once, at start-up;
* failures log *what* failed, *which file/ROI/CRS* was involved and the
  *probable cause* -- see :func:`log_failure_context`.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

_CONFIGURED = False

_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_DATEFMT = "%H:%M:%S"


class _ColourFormatter(logging.Formatter):
    """Minimal ANSI colouring; disabled when stderr is not a TTY."""

    _COLOURS = {
        logging.DEBUG: "\033[37m",
        logging.INFO: "\033[36m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;31m",
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        colour = self._COLOURS.get(record.levelno)
        return f"{colour}{text}{self._RESET}" if colour else text


def setup_logging(
    level: str | int = "INFO",
    *,
    log_file: Path | str | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the root logger. Idempotent unless ``force`` is given."""
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED and not force:
        return root
    for handler in list(root.handlers):
        root.removeHandler(handler)

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(level)

    stream = logging.StreamHandler(sys.stderr)
    use_colour = sys.stderr.isatty() and os.environ.get("NO_COLOR") is None
    formatter_cls = _ColourFormatter if use_colour else logging.Formatter
    stream.setFormatter(formatter_cls(_FORMAT, datefmt=_DATEFMT))
    root.addHandler(stream)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root.addHandler(file_handler)

    # Third-party chatter that drowns out our own messages.
    for noisy in ("urllib3", "rasterio", "fiona", "matplotlib", "PIL", "httpx"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_failure_context(
    logger: logging.Logger,
    *,
    what: str,
    cause: str = "",
    **context: Any,
) -> None:
    """Emit a structured ERROR block describing a failure.

    ``context`` typically carries ``target``, ``roi``, ``crs``, ``url``.
    """
    logger.error("FAILED: %s", what)
    for key, value in context.items():
        if value is not None:
            logger.error("  %-12s %s", key + ":", value)
    if cause:
        logger.error("  %-12s %s", "likely cause:", cause)
