# Design: Tests — Core Matemático

## Technical Approach

Unit test suite para **todos los módulos del proyecto**, priorizando `core/` primero. Sin mockeo: datos sintéticos con seed fija 42, aserciones con `numpy.testing.assert_allclose` contra invariantes matemáticas. Tests de error explícitos para inputs inválidos.

Estructura: `tests/{modulo}/` espejando `src/tda/{modulo}/`. Fixtures compartidas vía `conftest.py` por módulo + `tests/conftest.py` global.

## Architecture Decisions

### Directory Layout

| Opción | Tradeoff | Decisión |
|--------|----------|----------|
| Plano: `tests/modulo/test_*.py` | Evita anidamiento, pero módulos grandes (core con 4 archivos) se mezclan | ✅ Plano por módulo — `tests/core/`, `tests/analysis/`, etc. |
| Subdirectorios por archivo fuente | Más estructura pero sobreingeniería para 1-3 tests por archivo | ❌ Rechazado |

### Fixtures

| Opción | Tradeoff | Decisión |
|--------|----------|----------|
| `conftest.py` por módulo | Aísla fixtures propias (ej. malla 2×2 para FEM) pero requiere importar desde otros conftest si se necesitan | ✅ Una por módulo: `tests/core/conftest.py`, `tests/analysis/conftest.py`, etc. |
| Todas las fixtures en global | Accesible desde cualquier test pero contamina namespace y acopla módulos | ❌ Rechazado |

### Seed Strategy

| Opción | Tradeoff | Decisión |
|--------|----------|----------|
| `autouse` fixture global que llama `np.random.seed(42)` antes de cada test | Cada test arranca con misma secuencia → determinista pero puede ocultar dependencias entre tests | ✅ Autouse global — proposal exige seed fija. |
| Seed por módulo en `conftest.py` | Más granular pero inconsistente | ❌ Rechazado |

### Testing Legacy Code

| Opción | Tradeoff | Decisión |
|--------|----------|----------|
| Testear cada función pública con inputs controlados e invariantes | No refactor, tests frágiles si cambian implementaciones internas, pero sin mockeo es la única opción viable | ✅ Tests contra invariantes (shape, simetría, positividad, valores esperados) |
| Aislar con mocks | Proposal prohíbe mockeo explícitamente | ❌ Rechazado |

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `tests/conftest.py` | Create | Fixture `autouse` global: `np.random.seed(42)` |
| `tests/core/conftest.py` | Create | Malla 2×2, K0 esperada, DOFS, F, fixed — fixtures para FEM |
| `tests/core/test_fem.py` | Create | 6+ tests: K0 shape/simetría, ensamble sparse, solver 2-elem, compliance+dc, filtro, OC, errores dimensión |
| `tests/core/test_topology.py` | Create | 5+ tests: betti_numbers con diagrama sintético, binarización, escala adaptativa, homología (skip si no ripser), distancias error |
| `tests/core/test_metric.py` | Create | 4 tests: μ_α valores conocidos, α* calibración, edge case β₁=0, β₁ máximo |
| `tests/core/test_metric_simp.py` | Create | 4 tests: init params, definir_problema, optimizar con malla 2×2 (iter corta), fase_tda (skip sin ripser) |
| `tests/analysis/conftest.py` | Create | Nubes sintéticas controladas (sphere, torus) |
| `tests/analysis/test_stability.py` | Create | 2 tests: noise_sweep estructura retorno, callback progreso |
| `tests/analysis/test_metrics.py` | Create | 3 tests: kmeans_accuracy con labels intercambiadas, verify_betti correcto/incorrecto |
| `tests/processing/conftest.py` | Create | Puntos 3D de prueba, diagrama sintético |
| `tests/processing/test_sampling.py` | Create | 4 tests: generate_cloud forma/shape, torus R/r, add_gaussian_noise efecto, error forma inválida |
| `tests/processing/test_preprocessing.py` | Create | 3 tests: filter threshold, normalize por diámetro, persist_histogram bins |
| `tests/simulation/conftest.py` | Create | Config mínima para pipeline |
| `tests/simulation/test_pipeline.py` | Create | 2 tests: run_tda_experiment 1 rep, estructura resultados |
| `tests/optimization/test_simp_optimizer.py` | Create | 2 tests: init parametros, run_optimization malla pequeña |
| `tests/optimization/test_beam_optimizer.py` | Create | 3 tests: simular_viga, calcular_momento analítico, optimizar_viga_completo |
| `tests/visualization/conftest.py` | Create | Puntos 2D/3D, backend Agg (headless) |
| `tests/visualization/test_visualizer.py` | Create | 3 tests: visualize_2d retorna fig/ax, visualize_3d ValueError dim<3, reduce_dimensions PCA |
| `tests/app/conftest.py` | Create | Fixture para path de páginas |
| `tests/app/test_pages.py` | Create | 1 test: import de cada página Streamlit sin error |

## Testing Strategy

| Layer | What to Test | Approach |
|-------|-------------|----------|
| Unit (core) | FEM (K0, ensamble, solver, OC), topología (Betti, binarización, escala, homología), métrica (μ_α, α*) | Invariantes matemáticas: shape (8,8), simetría, positividad K0; c ≥ 0; mu > 0. Seed 42 ⇒ resultados deterministas |
| Unit (analysis) | Noise sweep, kmeans accuracy, verify_betti | nubes sintéticas controladas, 2 repeticiones |
| Unit (processing) | Sampling, preprocessing | Formas geométricas con seed, threshold y normalize con valores fijos |
| Unit (simulation) | Pipeline experiment | 1 repetición, estructura de resultados |
| Unit (optimization) | SIMP optimizers, beam | Mallas chicas (2×2, N=10), invariantes de convergencia |
| Unit (visualization) | Plot functions | Backend Agg, verificar que retornan fig sin crash |
| Unit (app) | Page imports | Verificar que cada página importa sin error |
| Error paths | Input inválido en todas las funciones | dimensiones incorrectas, shapes, tipos, umbrales fuera de rango |

## Migration / Rollout

No migration required. Archivos nuevos en `tests/` — sin cambios en productivo.

## Open Questions

- [ ] `beam_optimizer.py` tiene placeholders (`D_viga_ideasizada` devuelve constantes). ¿Testear funciones internas o aceptar que parte del código no es testeable sin refactor?
- [ ] `visualizer.py` depende de matplotlib con backend interactivo — requiere `plt.switch_backend("Agg")` en conftest. ¿Agregarlo como fixture autouse?
- [ ] `MetricaSIMP.optimizar()` y `SimpTda2DOptimizer.run_optimization()` ejecutan bucles SIMP completos. Para tests unitarios, ¿usar malla 2×2 con max_iter=2 (test de integración ligero) o mockear el solver?
