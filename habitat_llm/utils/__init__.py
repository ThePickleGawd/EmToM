"""habitat_llm.utils

This package is used across multiple entrypoints.

Some lightweight utilities (e.g. task generation) only need helpers like
`get_random_seed` and should not require heavyweight optional dependencies
like PyTorch.

Historically this module imported everything eagerly from submodules, which
pulled in `torch` via `habitat_llm.utils.core` at import time.

To keep task generation usable in minimal environments, we lazily import and
provide a small fallback implementation when optional dependencies are absent.
"""

from __future__ import annotations

from datetime import datetime
import os

try:
    # Full feature set (may require torch + habitat baselines).
    from habitat_llm.utils.core import (  # noqa: F401
        cprint,
        fix_config,
        get_random_seed,
        rollout_print,
        setup_config,
    )
    from habitat_llm.utils.sim import (  # noqa: F401
        get_ao_and_joint_idx,
        get_parent_ao_and_joint_idx,
        get_receptacle_index,
        is_open,
    )
except (ModuleNotFoundError, ImportError):
    # Minimal fallback: only expose what taskgen needs.
    def get_random_seed() -> int:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        print(f"Using a generated random seed {seed}")
        return seed

    __all__ = ["get_random_seed"]
