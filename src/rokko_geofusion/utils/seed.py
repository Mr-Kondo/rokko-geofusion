"""Reproducible random seeding across numpy / python / torch."""

from __future__ import annotations

import logging
import os
import random

logger = logging.getLogger(__name__)


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> int:
    """Seed every RNG we might use. Returns the seed for logging convenience."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass

    try:
        import torch
    except ImportError:
        logger.debug("torch not installed; skipping torch seeding")
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            # Deterministic cuDNN costs some throughput but V6 (re-running the
            # same ROI yields the same result) is a project requirement.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    logger.debug("global seed set to %d", seed)
    return seed
