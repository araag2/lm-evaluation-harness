"""
_registry.py — central strategy registry and metric aggregation helper.

All voting strategy functions register here so that voting_modes.py can
discover them without importing each submodule explicitly.

Registration
------------
Use the ``register`` decorator on any strategy function:

    from lm_eval.reasoning_modes.voting._registry import register

    @register("my_strategy")
    def my_strategy_fn(predictions_per_input_doc, doc_to_choice, task_def,
                        k=None, strategy_name="my_strategy"):
        ...

Or register imperatively:

    _STRATEGY_REGISTRY["my_strategy"] = my_strategy_fn
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List


# ──────────────────────────────────────────────
#  Global registry
# ──────────────────────────────────────────────

_STRATEGY_REGISTRY: Dict[str, Callable] = {}


def register(name: str) -> Callable:
    """Decorator that registers a strategy function under *name*."""
    def decorator(fn: Callable) -> Callable:
        _STRATEGY_REGISTRY[name] = fn
        return fn
    return decorator


def get_strategy(name: str) -> Callable:
    fn = _STRATEGY_REGISTRY.get(name)
    if fn is None:
        raise ValueError(
            f"Unknown voting strategy {name!r}. "
            f"Registered strategies: {sorted(_STRATEGY_REGISTRY)}"
        )
    return fn


def list_strategies() -> List[str]:
    return sorted(_STRATEGY_REGISTRY)


# ──────────────────────────────────────────────
#  Shared metric aggregation helper
# ──────────────────────────────────────────────

def aggregate_metrics_per_strategy(results_per_doc: Dict, task_def: Any) -> Dict:
    """Aggregate per-document metric values using the task's own aggregation functions."""
    aggregated: Dict = {}
    for metric_name, agg_fn in task_def.aggregation().items():
        values = [
            results_per_doc[doc_id][metric_name]
            for doc_id in results_per_doc
            if metric_name in results_per_doc[doc_id]
        ]
        if values:
            aggregated[metric_name] = agg_fn(values)
    return aggregated
