# Tasks: Unificar funciones duplicadas de sampling

## Review Workload Forecast

| Field | Value |
|-------|-------|
| Estimated changed lines | ~65-90 |
| 400-line budget risk | Low |
| Chained PRs recommended | No |
| Suggested split | Single PR |
| Delivery strategy | ask-on-risk |
| Chain strategy | pending |

Decision needed before apply: No
Chained PRs recommended: No
Chain strategy: pending
400-line budget risk: Low

## Phase 1: sampling.py — Agregar cube + compute_diameter

- [x] 1.1 Agregar `compute_diameter(points)` en `processing/sampling.py` usando `pdist`
- [x] 1.2 Refactor `add_gaussian_noise` en sampling.py para usar `compute_diameter` en vez de `pdist` inline
- [x] 1.3 Agregar case `"cube"` en `generate_cloud` — vectorizado, 6 caras con `np.random.randint(0, 6, n_points)`, sin loop
- [x] 1.4 Actualizar mensaje de error en sampling.py: incluir `'cube'` en las opciones

## Phase 2: pipeline.py — Importar desde sampling, eliminar duplicados

- [x] 2.1 Agregar `from tda.processing.sampling import generate_cloud, add_gaussian_noise, compute_diameter` al inicio de `pipeline.py`
- [x] 2.2 Eliminar función `generate_cloud` (líneas 37-85) y su `np.random.seed(None)`
- [x] 2.3 Eliminar función `compute_diameter` (líneas 88-97)
- [x] 2.4 Eliminar función `add_gaussian_noise` (líneas 100-112)
- [x] 2.5 Verificar que `scipy` import sigue disponible para otras funciones que usan `pdist`

## Phase 3: Tests — Actualizar test_sampling.py

- [x] 3.1 Agregar `test_cube_shape` en `TestGenerateCloud`: `generate_cloud("cube", n)` retorna `(n, 3)`
- [x] 3.2 Agregar `test_cube_points_bounded`: puntos del cubo en rango `[0, 1]` en cada eje
- [x] 3.3 Cambiar `test_invalid_shape_raises_value_error` de `"cube"` a `"undefined"` o forma inválida
- [x] 3.4 Agregar `test_compute_diameter` en sampling — punto aislado da 0, dos puntos da su distancia

## Phase 4: Verify

- [x] 4.1 Ejecutar `python -m pytest tests/processing/test_sampling.py -v --tb=short`
- [x] 4.2 Ejecutar `python -m pytest tests/simulation/test_pipeline.py -v --tb=short`
- [x] 4.3 Ejecutar suite completa: `python -m pytest -v --tb=short`
