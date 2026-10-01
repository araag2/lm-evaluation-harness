# Updating OpenCTEval from upstream

Assessed on 2026-09-28. The fork started this preparation at `47f5204b`.
Its actual upstream base is `a6c36ed9378f460b8bd08b37891bbfda3294f5a4`,
not the older tag reported by `git describe`. The recommended first target is
[v0.4.13](https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.13)
(`ddd67220`). Development main inspected was `d6de8164` (`0.4.14.dev0`).

## Preparation completed

- Moved 68 benchmark metric functions into `lm_eval/opencteval/metrics.py`.
  Their function bodies and decorators were verified unchanged before the
  extraction; subsequent lint cleanup makes existing zip truncation explicit.
  Existing metric names, registry entries, and direct imports from
  `lm_eval.api.metrics` remain available. Add new benchmark metrics in the
  extension module, rather than editing the core module.
- Moved document-aware scoring payloads and generated-answer normalization into
  `lm_eval/opencteval/task_helpers.py`. The task adapter still supplies the same
  document, gold label, prediction and probability tuples.
- Isolated the unfinished `CrossConsistencyCoT` prototype in `opencteval/legacy.py`.
  Its old explicit import remains available lazily. Removing its eager evaluator
  import fixes task-manager import order. The working cross-consistency pipeline
  remains in `reasoning_modes/cross_consistency.py`.
- Fixed the syntax error in the multimodal vLLM wrapper, removed duplicate imports
  in the text wrapper and corrected its annotation. CLI execution still selects
  multiprocessing `spawn`, using the standard library; importing the CLI or a
  model module no longer forces a process-wide start method. Programmatic callers
  should configure their process launcher explicitly if they need `spawn`.
- Restored upstream's `acc_bytes` and `likelihood` metric registrations, which were
  missing from the fork and failed newer upstream registry tests.
- Added offline compatibility tests and a separate CI workflow. The latter uses
  `datasets<4` to retain the current data-loading baseline during preparation;
  this is not a new constraint on package users.

The benchmark YAMLs, prompts, dataset sources, reasoning/voting implementations,
result-directory layout, 2,048-token vLLM default and Gemma BOS behavior remain
in place. F1 retains the fork's binary coercion/multiclass macro averaging;
exact match retains whitespace stripping and permissive defaults. These are
intentional remaining differences from upstream.

## Merge evidence and limits

Three-way file merges of the prepared metrics, task, CLI, filters, result writer
and multimodal wrapper are clean against both targets. The remaining runtime
text conflict is the vLLM constructor: retain `max_gen_toks=2048` while accepting
upstream's removal of the `swap_space` constructor parameter. Upstream handles
legacy `swap_space` kwargs itself. Do not select an entire version of the file.

The original full merge had 512 task-path conflicts because the fork deleted
14,835 upstream task files. Moving custom metrics does not resolve those structural
conflicts. README and `.gitignore` also require deliberate integration.

The core metrics diff against the shared base is now 34 added / 11 removed lines,
and the task diff is 39 added / 2 removed lines. New extension modules contain the
benchmark implementation, rather than spreading it through those core files.

These checks do not validate GPU inference, model downloads, all remote dataset
configurations, or all reasoning modes end to end. Passing tests establishes
specific contracts, not identical benchmark scores after an upstream upgrade.
Upstream fixes to few-shot sampling and group standard errors can legitimately
change results.

## Recommended integration sequence

1. Commit this preparation on a dedicated branch, then retain that commit as the
   pre-upgrade baseline. Keep the existing inference environment available.
2. Fetch upstream and create an integration branch from the prepared fork:

   ```bash
   git fetch upstream main --tags
   git switch -c integrate/lm-eval-0.4.13
   ```

3. Restore the upstream task catalogue on that branch in a separate commit,
   retaining all custom task directories. Restore only paths present in the
   target tag and absent from the fork; do not restore the whole tasks directory,
   which would remove custom additions. Review the path list first:

   ```bash
   git diff --no-renames --name-only --diff-filter=D -z v0.4.13 HEAD -- lm_eval/tasks > /tmp/opencteval-upstream-task-paths
   git restore --source=v0.4.13 --staged --worktree --pathspec-from-file=/tmp/opencteval-upstream-task-paths --pathspec-file-nul
   git diff --cached --stat
   git commit -m "Restore upstream task catalogue before integration"
   ```

   Restoring task definitions does not download their datasets. Continue invoking
   the benchmark through explicit OpenCTEval task names and existing scripts.
   Verify discovery and task/tag collisions once the catalogue is restored.

4. Merge `v0.4.13` into the integration branch. Keep the small OpenCTEval hooks and
   both sets of metrics; preserve custom output paths and model defaults. Resolve
   README/ignore changes explicitly. The temporary stable checkout used for
   validation demonstrates the equivalent approach of starting with the full
   upstream catalogue and adding the custom task directories.
5. Extend upstream's newer registry completeness test with the explicit custom
   expectations, before its parametrized test class is defined:

   ```python
   from tests.opencteval_metric_directions import OPENCTEVAL_METRIC_DIRECTIONS

   EXPECTED_METRIC_DIRECTIONS.update(OPENCTEVAL_METRIC_DIRECTIONS)
   ```

   This keeps the check strict: all 27 custom metric directions are pinned,
   rather than ignoring arbitrary extra registrations.
6. Run the offline suite below in an isolated environment. Then evaluate a fixed,
   small sample from each clinical task family against the retained baseline.
   Include multiple-choice, generated answers, ranking, spans and regression;
   exercise CoT, self-consistency, cross-consistency, self-refinement and only-vote.
   Compare rendered prompts, selected document IDs, token settings, raw response
   structures, metric payloads and saved files before comparing aggregate scores.
7. Validate the backend separately. Upstream requires `vllm>=0.18`; the inspected
   environment already has 0.19.1, but GPU compatibility was not exercised. Set
   thinking behavior explicitly for each stage. Upstream rejects explicit thinking
   mode for log-likelihood tasks and requires `think_end_token` when thinking is
   enabled. Do not enable stripping of reasoning text that the next stage consumes.
   Test newer `datasets` versions separately, including dataset configs that use
   `trust_remote_code`, before relaxing the baseline test pin.
8. Integrate the tested branch into your fork only after those checks pass. Keep
   production runs pinned to the resulting commit and record dependency versions.

## Repeating the offline checks

Validation during preparation: **180 tests passed** in the fork, and **232
tests passed** in a disposable v0.4.13 integration using the installed Python
3.10 environment. The latter included upstream's newer tests, the explicit vLLM
default conflict resolution and the metric-direction expectation extension
described above. The new CI workflow has not yet run on GitHub.

Run from the repository root in an environment containing the package and pytest:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -q -p no:cacheprovider \
  tests/test_opencteval_compat.py tests/test_intervention_metrics.py \
  tests/test_metrics.py tests/test_aggregation_pipeline.py \
  tests/test_registry.py tests/test_evaluator_utils.py
```

These checks cover custom metric registration/direction, original scoring,
augmentation identity, task payloads, import order, reasoning alignment, majority
voting, response filters, flat result paths and saved sample serialization.

## Contributing changes to EleutherAI

Updating this fork and submitting features upstream are separate operations.
Do not submit the entire fork as one PR: its catalogue deletions, custom README,
output layout and global scoring overrides are specific to this project.

Use small branches based on the current EleutherAI main, following its
[contribution guide](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/CONTRIBUTING.md).
First propose self-contained clinical task definitions with fixtures and data
provenance. Propose broadly useful filters or metric hooks independently, with
explicit names/options instead of changing all users' F1, exact-match or output
behavior. Keep the reasoning orchestration separate until its interfaces and
expected behavior have broader coverage.

As a later maintenance step, package uniquely named metrics and filters using
[upstream's plugin interface](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/plugins.md)
and discover custom YAMLs through `--include_path`. Plugin support is on the
inspected development main, after v0.4.13. The new `lm_eval/opencteval` directory
is an internal separation, not yet an independently installable plugin. Built-in
names cannot be shadowed by plugins: replacing the current global overrides
requires explicit OpenCTEval metric names and task-level result adapters first.
