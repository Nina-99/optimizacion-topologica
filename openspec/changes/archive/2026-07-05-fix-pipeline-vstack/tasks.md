# Tasks: fix-pipeline-vstack

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | 50–100 |
| 400-line budget risk | Low |
| Chained PRs recommended | No |
| Suggested split | single PR |
| Delivery strategy | ask-on-risk |
| Chain strategy | pending |

Decision needed before apply: No
Chained PRs recommended: No
Chain strategy: pending
400-line budget risk: Low

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|------|------|-----------|-------|
| 1 | Helper + fix + test + verify | PR 1 | single PR, ~50–100 lines |

## Phase 1: Helper + Fix

- [x] 1.1 Add `_safe_stack(normalized: list) -> np.ndarray` in `pipeline.py` after `compute_bottleneck_distance`; reshape each dim to `(n, 2)` before vstack, return `np.empty((0, 2))` if all empty
- [x] 1.2 Replace `clean_arr = np.vstack(clean_normalized) if any(clean_normalized) else np.empty((0, 2))` with `clean_arr = _safe_stack(clean_normalized)`
- [x] 1.3 Replace `noisy_arr = np.vstack(noisy_normalized) if any(noisy_normalized) else np.empty((0, 2))` with `noisy_arr = _safe_stack(noisy_normalized)`

## Phase 2: Test

- [x] 2.1 Add `test_mixed_dimensions` in `tests/simulation/test_pipeline.py` — run torus with threshold that leaves H0 non-empty, H1 empty; assert no crash and output shape `(n, 2)`

## Phase 3: Verify

- [x] 3.1 Run `python -m pytest tests/simulation/test_pipeline.py -v --tb=short` — all existing and new tests pass
