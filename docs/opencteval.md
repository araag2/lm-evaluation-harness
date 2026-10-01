# OpenCTEval development

This fork incorporates lm-evaluation-harness v0.4.13. Its task catalogue is
intentionally limited to OpenCTEval; upstream tasks restored during integration
were subsequently removed. Custom task definitions and prompts live in
`lm_eval/tasks`, and reasoning pipelines live in `lm_eval/reasoning_modes`.

## Extension layout

| Module | Responsibility |
| ------ | -------------- |
| `lm_eval/opencteval/metrics.py` | Clinical, ranking, span, regression and intervention metrics |
| `lm_eval/opencteval/task_helpers.py` | Document-aware metric payloads and generated-answer normalization |
| `lm_eval/opencteval/legacy.py` | Historical `CrossConsistencyCoT` import compatibility |

Add benchmark-specific metrics in the extension module. Core aggregations are
registered before these extensions, and existing imports from
`lm_eval.api.metrics` remain supported. Keep this initialization order to avoid
circular imports. The active cross-consistency implementation is
`lm_eval/reasoning_modes/cross_consistency.py`; the legacy class is an unfinished
prototype retained for existing imports.

The fork intentionally retains its F1 label handling and multiclass macro
averaging, whitespace-normalizing exact match, flat result directories,
2,048-token vLLM generation default and Gemma BOS handling. Preserve these
contracts when updating upstream code.

## Regression tests

`tests/test_opencteval.py` covers registration, scoring, task payloads, import
order, reasoning alignment, majority voting, response filters and result files.
`tests/test_intervention_metrics.py` covers paired interventions and augmentation
identity. Shared direction expectations live in
`tests/opencteval_metric_directions.py` and are included in the registry tests.
Add an explicit direction expectation when registering a new metric.

Run the focused suite from the repository root in the project environment:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -q -p no:cacheprovider \
  tests/test_opencteval.py tests/test_intervention_metrics.py \
  tests/test_metrics.py tests/test_aggregation_pipeline.py \
  tests/test_registry.py tests/test_evaluator_utils.py
```

The `OpenCTEval Tests` workflow in `.github/workflows/opencteval.yml` runs this
suite on Python 3.10 and 3.12, without model or dataset downloads during tests.
It uses pytest 9 or later, matching `pyproject.toml`, and retains `datasets<4`
as the validated data-loading baseline. That CI constraint does not change
package dependencies. Validate remote dataset configurations separately before
updating the baseline.

Use the Ruff version pinned in `.pre-commit-config.yaml` for formatting and lint:

```bash
ruff check lm_eval/opencteval tests/test_opencteval.py tests/opencteval_metric_directions.py tests/test_registry.py
ruff format --check lm_eval/opencteval tests/test_opencteval.py tests/opencteval_metric_directions.py tests/test_registry.py
```

These offline tests do not cover GPU inference, every remote dataset config or
all reasoning modes end to end. Before adopting another upstream version, run
small fixed-sample evaluations and compare prompts, document IDs, generation
settings, metric payloads and saved outputs. Upstream correctness fixes can
legitimately change aggregate scores.

## Upstream maintenance

Keep benchmark extensions separate from core harness code, and retain tests for
all remaining integration points. Inspect task-catalogue deletions during future
merges: removing upstream tasks can cause modify/delete conflicts when those
files change upstream. Resolve these according to the curated catalogue policy.

For contributions to EleutherAI, prepare focused branches from its upstream
main. Submit self-contained tasks, metrics or filters independently of the
fork's catalogue removals and global scoring/output defaults. New public
behavior should use explicit metric names or options. Keep inference dependency
upgrades separate from scoring changes so regressions can be traced reliably.
