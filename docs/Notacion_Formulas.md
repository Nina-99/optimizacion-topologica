# Notación y fórmulas: documento ↔ software

Este documento fija la notación del Cap. II y verifica cada fórmula contra su implementación. La fuente de verdad es el PDF; cuando el software se desvía, se indica explícitamente.

## 1. Símbolos

| PDF | Significado | Implementación |
|---|---|---|
| `X = {x₁,…,xₙ} ⊂ ℝ^d` | nube de puntos | `generate_cloud` / `generar_corregida` |
| `VR(X, ε)` | complejo de Vietoris–Rips | `ripser(X, maxdim=1)` |
| `Dgm_k(X)` | diagrama de persistencia | `ripser(...)["dgms"]` |
| `β_k` | número de Betti | `betti_persistente`, `betti2d` |
| `β_k^(τ)(X)` | barras con `d−b > τ·diam X` | `betti_persistente` |
| `d_B(·,·)` | distancia bottleneck | `core.topology.bottleneck_distance` |
| `d_H(·,·)` | distancia de Hausdorff | `analysis.stability.hausdorff_distance` |
| `q` | nivel de ruido relativo | `Q_LEVELS` |
| `σ = q·diam X/√d` | desviación del ruido | `add_gaussian_noise` |
| `τ` | umbral relativo de persistencia | `TAU_CALIBRADO = 0.175` |
| `δ` | probabilidad de fallo de la cota | `DELTA_FIG52 = 0.05` |
| `C₀`, `C` | catálogos de calibración y medición | `calibrar_c0`, `margenes` |
| `ρ_e` | densidad de diseño | `rho` |
| `ρ̃_e` | densidad filtrada | `rho_tilde` |
| `E_e(ρ_e)` | módulo penalizado | `ensamblar_K_global` |
| `E₀`, `E_min` | módulo sólido y void | `E=1`, `Emin=1e-9` |
| `p` | factor de penalización | `p` |
| `f_V` | fracción de volumen | `f_V` |
| `r_min` | radio del filtro | `r_min` |
| `K(ρ)` | rigidez global | `ensamblar_K_global` |
| `M(ρ)` | compliance | `calcular_compliance_sensibilidades` |
| `M*` | cota inferior convexa | `cota_inferior_convexa` |
| `M_unif`, `M_bin` | placa y binarizado | `compliance_densidad` |
| `η` | eficiencia relativa | `metricas_he2a` |
| `u`, `u_V` | umbral y umbral que conserva volumen | `umbral_volumen` |
| `χ` | característica de Euler | `betti_euler` |
| `V, E, F` | vértices, aristas, caras | `_conteo_celular` |

## 2. Fórmulas

| PDF | Fórmula | Implementación | Estado |
|---|---|---|---|
| (2.1) | `β_k = dim ker ∂_k − rank ∂_{k−1}` | `betti_euler` + GUDHI | Coincide |
| (2.2) | `VR(X, ε)` | `ripser` | Coincide |
| (2.3) | `Dgm_k(X)` | `ripser(...)["dgms"]` | Coincide |
| Def. 2.4 | bottleneck minimax | `bottleneck_distance` exacta | Coincide |
| Teo. 2.1 | `d_B(Dgm f, Dgm g) ≤ ‖f−g‖∞` | `diagramas_campo` (subniveles) | Coincide en el instrumento |
| Teo. 2.2 | `d_B(Dgm VR X, Dgm VR Y) ≤ 2 d_H(X,Y)` | `estabilidad_dimensional` | Coincide |
| Prop. 2.1 | `2σ(√d + √(2 ln(n/δ)))` | `cota_probabilistica` | Coincide |
| (2.7) | `E_e = E_min + ρ_e^p (E − E_min)` | `ensamblar_K_global` | **Corregido** |
| (2.8) | `min M(ρ)` s.a. `K U=F`, volumen, `0<ρ_min≤ρ≤1` | `MetricaTDA_SIMP.optimizar` | Coincide |
| (2.9) | `∂M/∂ρ_e = −p ρ_e^{p−1}(E−E_min) u_e^T K₀ u_e` | `calcular_compliance_sensibilidades` | **Corregido** |
| (2.10) | `M* = min M(ρ)` con `p=1`, sin filtro | `cota_inferior_convexa` | Coincide |
| Def. 2.5 | `η = (M_unif − M_bin)/(M_unif − M*)` | `metricas_he2a` | Coincide |
| §5.4.1 | `β_k^(τ) = #{(b,d): d−b > τ·diam X}` | `betti_persistente` | **Desviación:** usa `diam(Y)` en H.E.1c |
| §5.5.4 | `Ω_sólido = {e: ρ̃_e ≥ u_V}` | `MetricaTDA_SIMP.binaria` | Coincide |
| §5.5.2 | `grisura = 4ρ(1−ρ)` | `np.mean(4ρ(1−ρ))` | Coincide |
| §5.5.1 | `u_V` conserva `f_V` | `umbral_volumen` | Coincide |
| H.E.1b | McNemar sobre pares | `resultado_piloto_pdf` | Oficial PDF; reproducible en `comparar_tda_vs_euclidiano` |

## 3. Desviaciones que quedan

1. **`diam(X)` frente a `diam(Y)` en H.E.1c.** La fórmula del PDF usa `τ·diam X`; `estabilidad_dimensional._replica` lee con `diam(Y)`. No se cambió porque los `q*` de `fig52_replicas.csv` reproducen los del PDF; cambiarlo alteraría la Tabla 5.5. Queda documentado como decisión de implementación.
2. **Barras esenciales.** `diagramas_finitos` descarta la barra infinita de H₀; equivale a emparejarla a costo 0, pero la compuerta F4 pide incluirla o emparejarla explícitamente.
3. **H.E.1b.** El valor oficial es el piloto del PDF (0,950 vs 0,925, no concluyente). El protocolo reproducible da 1,000 vs 0,750 y se conserva como evidencia exploratoria, no como veredicto.

## 4. Hipótesis general

El §1.5 no define una sexta hipótesis operacionalizada. La H.G. es una afirmación condicional de dos cláusulas:

| Cláusula | Formulación | Hipótesis que la sostienen |
|---|---|---|
| A | `β_k` estable/verificable si `q ≤ q*` | H.E.1a, H.E.1c, H.E.2b |
| B | `β_k` supera a euclidianos cuando la información es topológica | H.E.1b |

H.E.2a (eficiencia SIMP) queda fuera. El veredicto es tri-estado: `sostenida`, `parcial/condicionada`, `no sostenida`.

## 5. Leyendas de figuras

Las leyendas breves viven en `src/tda/visualization/leyendas.py` y se usan en la app (`st.caption`) y en `docs/Replica_Figuras.md`. Son las del PDF, adaptadas a una línea:

- 5.1 calibración de `τ` y márgenes.
- 5.2 estabilidad `d_B ≤ 2d_H` y fiabilidad de `β_k`.
- 5.3 artefacto de truncamiento y cota probabilística.
- 5.4 diseño base, `u_V` e historial de compliance.
- 5.5 penalización y grisura.
- 5.6 independencia de malla con `r_min` físico constante.
- 5.7 Betti frente a `u` y persistencia `H₁` del campo.
- 5.8 falsación MBB con `β₁` frente a `r_min`.

## 6. Dónde se verifica

- `tests/core/test_fem.py`: rigidez, compliance y sensibilidades.
- `tests/core/test_betti2d.py`: 8-conectividad y persistencia del campo.
- `tests/analysis/test_estabilidad_dimensional.py`: `q*`, `q_fallo` y banda.
- `tests/analysis/test_stability.py`: piloto H.E.1b del PDF.
- `scripts/replicar_figuras.py` + `data/figuras/replica/valores.csv`.
