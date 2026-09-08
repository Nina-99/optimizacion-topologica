# Tasks: Tests — Core Matemático

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | 800–1200 |
| 400-line budget risk | High |
| Chained PRs recommended | Yes |
| Suggested split | Core → Analysis/Processing → Simulation/Optimization → Viz/App |
| Delivery strategy | ask-on-risk |
| Chain strategy | stacked-to-main |

Decision needed before apply: Yes
Chained PRs recommended: Yes
Chain strategy: stacked-to-main
400-line budget risk: High

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|------|------|-----------|-------|
| 1 | Core tests (6 files) | PR 1 | `tests/conftest.py` + `tests/core/*` |
| 2 | Analysis & Processing (6 files) | PR 2 | `tests/analysis/*` + `tests/processing/*` |
| 3 | Simulation & Optimization (4 files) | PR 3 | `tests/simulation/*` + `tests/optimization/*` |
| 4 | Viz & App (4 files) | PR 4 | `tests/visualization/*` + `tests/app/*` |

## Phase 1: Foundation — Core Tests

- [x] 1.1 Create `tests/conftest.py` — autouse `np.random.seed(42)`
- [x] 1.2 Create `tests/core/conftest.py` — malla 2×2, K0, DOFS, F, fixed
- [x] 1.3 Create `tests/core/test_fem.py` — K0 shape/sym, ensamble, solver, compliance, OC, error
- [x] 1.4 Create `tests/core/test_topology.py` — Betti, binarización, escala, homología (skip sin ripser)
- [x] 1.5 Create `tests/core/test_metric.py` — μ_α, α*, edge β₁=0, β₁ max
- [x] 1.6 Create `tests/core/test_metric_simp.py` — init, definir_problema, optimizar 2×2, fase_tda

## Phase 2: Analysis & Processing

- [x] 2.1 Create `tests/analysis/conftest.py` — nubes sintéticas (sphere, torus)
- [x] 2.2 Create `tests/analysis/test_stability.py` — noise_sweep retorno, callback
- [x] 2.3 Create `tests/analysis/test_metrics.py` — kmeans_accuracy, verify_betti
- [x] 2.4 Create `tests/processing/conftest.py` — puntos 3D, diagrama sintético
- [x] 2.5 Create `tests/processing/test_sampling.py` — generate_cloud, torus R/r, noise, error
- [x] 2.6 Create `tests/processing/test_preprocessing.py` — filter, normalize, persist_histogram

## Phase 3: Simulation & Optimization

- [x] 3.1 Create `tests/simulation/conftest.py` — config mínima pipeline
- [x] 3.2 Create `tests/simulation/test_pipeline.py` — run_tda_experiment, estructura resultados
- [x] 3.3 Create `tests/optimization/test_simp_optimizer.py` — init, run_optimization malla pequeña
- [x] 3.4 Create `tests/optimization/test_beam_optimizer.py` — simular_viga, momento, optimizar

## Phase 4: Visualization & App

- [x] 4.1 Create `tests/visualization/conftest.py` — backend Agg autouse, puntos 2D/3D
- [x] 4.2 Create `tests/visualization/test_visualizer.py` — visualize_2d, ValueError 3D, PCA
- [x] 4.3 Create `tests/app/conftest.py` — fixture path páginas
- [x] 4.4 Create `tests/app/test_pages.py` — import páginas Streamlit sin error

## Phase 5: Verification

- [x] 5.1 Run `python -m pytest -v --tb=short` — full suite green
- [x] 5.2 Run `ruff check tests/` — lint clean
