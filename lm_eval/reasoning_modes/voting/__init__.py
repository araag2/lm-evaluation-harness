"""
voting/__init__.py — public re-exports for the voting package.
"""

from lm_eval.reasoning_modes.voting.voting_modes import (
    run_voting_modes,
    simple_voting_modes,
)
from lm_eval.reasoning_modes.voting._registry import (
    aggregate_metrics_per_strategy,
    list_strategies,
    register,
)
