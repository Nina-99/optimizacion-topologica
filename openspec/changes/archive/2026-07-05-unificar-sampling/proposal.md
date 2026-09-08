# Proposal: Unificar funciones de sampling duplicadas

## Intent

`simulation/pipeline.py` tiene copias de `generate_cloud` y `add_gaussian_noise` que ya existen en `processing/sampling.py`, con ligeras diferencias: pipeline soporta `cube`, tiene `np.random.seed(None)` que rompe reproducibilidad, y usa variables intermedias verbose. Eliminar la duplicación unificando en `processing/sampling.py` sin cambiar comportamiento externo.

## Scope

### In Scope
- Agregar soporte para `cube` en `processing/sampling.py` (estilo directo, sin loop por punto)
- Unificar estilo de `generate_cloud` y `add_gaussian_noise` usando processing/sampling como base
- Eliminar `np.random.seed(None)` de pipeline — necesario para reproducibilidad
- Hacer que `pipeline.py` importe `generate_cloud`, `add_gaussian_noise` desde `processing.sampling`
- Mover `compute_diameter` de pipeline a processing/sampling.py
- Unificar mensajes de error a español
- Actualizar tests si es necesario

### Out of Scope
- Refactor de pipeline.py más allá del cleanup de imports duplicados
- Cambios en otros módulos (experiment_simp, preprocessing, etc.)
- Cambios de spec o comportamiento funcional

## Capabilities

### New Capabilities
None — refactor puro, sin cambio de comportamiento a nivel de spec.

### Modified Capabilities
None — ninguna spec existente cambia.

## Approach

1. Agregar `cube` en `processing/sampling.py`:
   - Muestrear 6 caras con `np.random.randint(0, 6, n_points)` vectorizado
   - Asignar coordenadas con `np.where` o selección por eje para evitar loop
   - Reutilizar RNG sin `seed(None)`
2. Mover `compute_diameter` de pipeline.py a processing/sampling.py
3. Reemplazar en pipeline.py los imports/bodies por `from tda.processing.sampling import ...`
4. Unificar mensajes al español ("Forma desconocida" consistente)
5. Verificar que 242 tests pasan

## Affected Areas

| Area | Impact | Description |
|------|--------|-------------|
| `src/tda/processing/sampling.py` | Modified | +cube support, +compute_diameter, unified style |
| `src/tda/simulation/pipeline.py` | Modified | Remove duplicated functions, import from sampling |
| `tests/processing/test_sampling.py` | Modified (maybe) | Add cube tests if needed |

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Cambiar `pipeline.py` sin romper su flujo experimental | Medium | Tests existentes detectan regresión; mantener `run_tda_experiment` intacto |
| `cube` vectorizado difiere del original punto-a-punto | Low | Refactor visualmente idéntico en output; tests de integración lo validan |
| Import circular al mover `compute_diameter` | Low | `processing/sampling` no importa de `simulation/*` — no hay ciclo |

## Rollback Plan

`git revert <commit-hash>` del cambio. Si el cambio abarca varios commits, revertir en orden inverso.

## Dependencies

Ninguna.

## Success Criteria

- [ ] Los 242 tests existentes siguen pasando (`pytest --collect-only -q` da 242)
- [ ] `generate_cloud("sphere"|"torus"|"cube", n)` funciona desde ambos módulos
- [ ] `add_gaussian_noise` funciona desde ambos módulos
- [ ] `compute_diameter` disponible en `processing.sampling`
- [ ] `pipeline.py` no contiene definiciones propias de `generate_cloud`, `add_gaussian_noise`, `compute_diameter`
- [ ] `np.random.seed(None)` no aparece en pipeline.py
- [ ] Mensajes de error en español
