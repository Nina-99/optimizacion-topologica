# Proposal: fix-pipeline-vstack

## Intent

Fix crash when `np.vstack(clean_normalized)` is called with mixed-dimension
diagrams (H0 has points, H1 is empty after threshold filtering). The bug
surfaces with torus + threshold ~0.5+, where H1 intervals are filtered out
but H0 points survive — `np.vstack` errors because `[]` becomes a 1D
`(0,)` array while other dimensions are 2D `(n, 2)`.

## Scope

### In Scope
- Create `_safe_stack(normalized)` helper that converts empty dimensions to
  `np.empty((0, 2))` before vstack
- Replace inline guard in `clean_arr` and `noisy_arr` with `_safe_stack`
- Add test case for mixed dimensions (H0 non-empty, H1 empty)
- Verify existing tests (`sphere`, `cube`) still pass

### Out of Scope
- Refactor of `pipeline.py` structure or modularization
- Changes outside `pipeline.py` and `tests/simulation/test_pipeline.py`
- Threshold logic or diagram filtering behavior

## Capabilities

None — bug fix only, no spec-level behavior change.

## Approach

Introduce `_safe_stack(normalized: list) -> np.ndarray` in `pipeline.py`:
convert each dimension to `np.array(pts).reshape(-1, 2)` and vstack. If all
dimensions are empty, return `np.empty((0, 2))`.

Replace:
```python
clean_arr = np.vstack(clean_normalized) if any(clean_normalized) else np.empty((0, 2))
```
with:
```python
clean_arr = _safe_stack(clean_normalized)
```

Same for `noisy_arr`.

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/tda/simulation/pipeline.py` | Modified | Add `_safe_stack()`, replace guards in `clean_arr`/`noisy_arr` |
| `tests/simulation/test_pipeline.py` | Modified | Add `test_mixed_dimensions` with torus + threshold |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Regression on sphere/cube paths | Low | Existing tests + manual run with torus |
| `_safe_stack` changes semantics for downstream consumers | Low | Output shape identical to working case |

## Rollback Plan

`git revert <commit-hash>` del commit del fix.

## Dependencies

None.

## Success Criteria

- [ ] `pipeline.py` with torus + threshold ~0.5+ runs without `np.vstack` error
- [ ] New test verifies mixed-dimension case (H0 non-empty, H1 empty)
- [ ] All existing tests pass
