"""
strategies/__init__.py

Importing this package triggers registration of all strategy submodules
into _STRATEGY_REGISTRY.  Import order matters: simple strategies are
registered first so they can serve as fallbacks in other modules.
"""

from lm_eval.reasoning_modes.voting.strategies import simple          # noqa: F401
from lm_eval.reasoning_modes.voting.strategies import chain_metrics   # noqa: F401
from lm_eval.reasoning_modes.voting.strategies import llm_metrics     # noqa: F401
# mbr is NOT registered here — it requires external deps and a model call.
# It is imported lazily by run_voting_modes when args.mbr_voting=True.
