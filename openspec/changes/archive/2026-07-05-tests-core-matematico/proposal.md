# Proposal: Tests — Core Matemático

## Intent

Cobertura de tests inexistente: solo 95 LOC smoke en un proyecto de tesis donde reproducibilidad es crítica. Sin tests no hay confianza en resultados numéricos. Unit tests sistemáticos para todos los módulos.

## Scope

### In Scope
- Unit tests para todos los módulos: core, analysis, processing, simulation, optimization, visualization, app
- Archivos nuevos en `tests/{modulo}/` por módulo
- Seed fija (rng=42), datos sintéticos, aserciones contra invariantes
- pytest + numpy.testing, fixtures, sin mockeo
- Tests de error explícitos (inputs inválidos, dimensiones incorrectas)

### Out of Scope
- Tests de integración o e2e
- Refactor de código productivo
- CI/CD

## Capabilities

### New Capabilities
None

### Modified Capabilities
None

## Approach

1. Empezar por `core/` (fem, topology, metric, metric_simp) — prioridad por tamaño y criticidad
2. Luego expandir: processing/, simulation/, optimization/, visualization/, app/
3. Strict TDD: test primero, assert contra invariantes, seed fija 42
4. Fixtures compartidas via `conftest.py` por módulo
5. Cada función pública testada + parámetros clave + edge cases + error paths

## Affected Areas

| Área | Impacto | LOC ref. |
|------|---------|----------|
| `tests/core/` | New | fem(230), topology(168), metric(65), metric_simp(296) |
| `tests/analysis/` | New | stability(96), sampling(53) |
| `tests/processing/` | New | pipeline(323) |
| `tests/simulation/` | New | sim submodule |
| `tests/optimization/` | New | simp_opt(201), beam_opt(336) |
| `tests/visualization/` | New | visualizer(378) |
| `tests/app/` | New | Streamlit pages |

## Risks

| Riesgo | Prob. | Mitigación |
|--------|-------|------------|
| Coverage 80% lejano | Alta | Arrancar por core/, subir gradualmente |
| Código no diseñado para test | Media | numpy.testing + fixtures modulares |
| Tests frágiles por seed | Baja | Seed fija 42, invariantes no aleatorios |

## Rollback Plan

Eliminar archivos nuevos en `tests/`. Sin cambios en `src/` a revertir.

## Dependencies

- pytest >= 8.0
- numpy.testing

## Success Criteria

- [ ] core/ con tests por función pública + edge cases + errores
- [ ] Todos los módulos tienen tests con corrida exitosa
- [ ] Seed 42 produce resultados deterministas
- [ ] `python -m pytest -v --tb=short` pasa completo
