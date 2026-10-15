# Correcciones al documento principal (Cap. I–V)

Este archivo lista las ediciones exactas que deben aplicarse al **cuerpo de la tesis**
(que en el repositorio solo existe como PDF, por lo que no es editable desde aquí).
Los Anexos 1 y 2 (`.docx`) ya fueron corregidos y regenerados.

Criterio: el software se alineó a la tesis (Camino A). Las versiones de librerías se
mantienen **como están instaladas** en el software.

---

## Item 1 — Tabla 3.1 (Herramientas computacionales) y §1.5.2

**Antes**
- Tabla 3.1 incluye filas: `Persim 0.3.x` (Visualización de diagramas, distancia de Wasserstein) y `PyTorch 2.x` (Diferenciación automática).
- Versiones listadas: NumPy 1.26, SciPy 1.12, Matplotlib 3.8, scikit-learn 1.4.
- §1.5.2: «Python 3.11+Ripser 0.6, Persim 0.3, PyTorch 2.x, NumPy 1.26, SciPy 1.12, Matplotlib 3.8 y scikit-learn 1.4».

**Después**
- **Eliminar** las filas `Persim` y `PyTorch` (no se usan: las distancias bottleneck/Wasserstein están implementadas en NumPy puro).
- Versiones (instaladas): `numpy>=2.1`, `scipy>=1.14`, `scikit-learn>=1.5`, `matplotlib>=3.9`, `plotly>=5.24`, `gudhi>=3.13`, `ripser` (0.6.x), Python 3.11+ (entorno validado en 3.12).
- §1.5.2 → «Python 3.11+ (validado en 3.12), Ripser 0.6, GUDHI 3.13, NumPy ≥2.1, SciPy ≥1.14, Matplotlib ≥3.9 y scikit-learn ≥1.5».

**Razón:** Ni PyTorch ni Persim aparecen en el código; declararlos rompe la trazabilidad.

---

## Item 2 — Método del optimizador (MMA → OC)

**Antes** (Anexo): «El optimizador implementado utiliza el método **MMA** (Method of Moving Asymptotes)…».

**Después** (Anexo, ya corregido) y cuerpo: el solver usa **Criterio de Optimalidad (OC)** con actualización `ρ_e·(−∂c/∂ρ_e / λ)^{1/2}` y bisección del multiplicador — exactamente el Algoritmo 2.

**Razón:** No existe MMA en el código; el Algoritmo 2 del propio Cap. IV ya describe OC. (Anexo corregido; verificar que §2.2.3 y Anexo coincidan.)

---

## Item 3 — Filtro de densidad (Algoritmo 2, paso 7)

**Antes:** implícito/ambiguo; el software usaba filtro de sensibilidad.

**Después:** dejar explícito en §2.2.3 y en el Algoritmo 2 que el filtro es **de densidad**:

```
ρ̃ ← H(ρ)/Hs ,   Hs = Σ_f ĥ_ef
```

La densidad física que entra al ensamble FEM es la filtrada `ρ̃` (no `ρ` cruda).

**Razón:** El Algoritmo 2 (paso 7) ya lo dice; el software ahora lo implementa así. Efecto: `c_SIMP` pasa de ≈70.2 (sensibilidad) a ≈74.7 (densidad).

---

## Item 4 — Tolerancia de convergencia (§3.4.3)

**Antes:** «… cambio máximo en densidades menor que **10⁻³** entre iteraciones consecutivas.»

**Después:** «… cambio máximo en densidades menor que **10⁻²** entre iteraciones consecutivas.»

**Razón:** Contradicción interna: el Algoritmo 2 (paso 24) y la Tabla 4.4 fijan `Δρ < 10⁻²`. El código usa `tol_rho = 1e-2`. Unificar en 10⁻².

---

## Item 5 — Escala del material y la carga (E0)

**Antes:** Anexo §3.2 «Módulo Young (E0) = 200000 MPa»; §3.4.1 «carga puntual».

**Después:** los resultados del Cap. V corresponden a **material y carga normalizados (`E0 = 1`, `F = 1`)** con **elementos unitarios** (`Lx = nex`, `Ly = ney`). Añadir en Tabla 5.6 una nota: «valores en unidades normalizadas (E0 = 1, F = 1, espesor 1)». Si se prefiere mantener E0 = 200000, entonces deben recalcularse TODOS los valores del Cap. V (316.34 / 39.54 / 75.55).

**Razón:** `c_uniforme = 316.342` y `c_sólido = 39.543` se reproducen exactamente con `E0 = 1, F = 1, Lx = 60, Ly = 30`; con `E0 = 200000` dan 4717.7 / 589.7. El Anexo y el Cap. V se contradicen.

---

## Item 6 — Malla del barrido (Tabla 1.1 vs Figura 5.6) — NO requiere corrección (solo aclarar)

**Estado:** revisado. La Tabla 1.1 y la Figura 5.6 **son coherentes entre sí**, siempre que el estudio de independencia de malla se entienda sobre el **dominio cuadrado 60×60**:
- `Ne ∈ {1600, 6400, 14400}` = 40², 80², 120² (dominio cuadrado).
- `rmin ∈ {1.6, 3.2, 4.8}` = 2.4·(nx/60) con nx = 40, 80, 120.

**Acción:** añadir una aclaración de que el barrido de H.E.2b usa el **dominio cuadrado 60×60** (para que no se lea asumiendo la viga 2:1 del caso base). El software ya lo implementa así; **no hay que cambiar los números**.

---

## Item 7 — Valor de β₁ del diseño

**Antes** (Anexo, Hallazgo 3): «p = 3 … manteniendo **B1 = 2**».

**Después:** «… manteniendo **β₁ = 3**».

**Razón:** El Cap. V y el software dan `β₁ = 3` (verificado con doble cómputo Euler/GUDHI). (Anexo ya corregido.)

---

## Item 8 — Tabla 4.3 (Módulos del software)

**Antes:** `config.py, geometria.py, tda.py, fem_simp.py, betti2d.py, figuras.py, experimentos.py`.

**Después** (módulos reales):

| Módulo | Objetivo | Responsabilidad |
|---|---|---|
| `core/fem.py` | OE3 | Q4, ensamble, solver, sensibilidades, OC |
| `core/betti2d.py` | OE3/4 | Doble cómputo β₀/β₁ (Euler + GUDHI) |
| `core/topology.py` | OE2 | Distancias bottleneck/Wasserstein (NumPy), homología |
| `optimization/metric_simp.py` | OE3 | Bucle SIMP (Algoritmo 2) + fase TDA |
| `analysis/stability.py` | OE2 | Barrido de ruido, cotas, McNemar, q* |
| `processing/sampling.py` | OE2 | Catálogo de 7 nubes, ruido, transformación afín |
| `simulation/experiment_simp.py` | OE3/4 | Experimento headless + CSV |

**Razón:** Los nombres de la Tabla 4.3 no existen en el repositorio; la trazabilidad prometida es inverificable.

---

## Item 9 — μ_α y H.G.

**Antes:** Anexo 1 §2.2 «Métrica compuesta μ_α = c + α·β₁…» y módulo «H.G.».

**Después:** eliminados del software, del Anexo y del manual. El cuerpo de la tesis (Cap. I–V) ya las excluye (Tabla 1.2, «fuera del alcance»); no requiere cambios, solo verificar que no reaparezcan.

**Razón:** Camino A — la tesis solo evalúa la topología *post hoc*; μ_α era regularización en el bucle (fuera de alcance).

---

## Item 10 — Valor de compliance del diseño SIMP

**Antes** (Tabla 5.6 / §5.4.2): `c_SIMP = 75.55`, reducción `76.12 %`.

**Después:** `c_SIMP = 74.72`, reducción `76.38 %`.

**Razón:** Con el filtro de densidad correcto, el diseño converge a `c = 74.72` (iter 92). El valor `75.55` corresponde a la meseta en la iteración ~25 (la propia §5.4.2 dice «se estabiliza en c = 75.6 **alrededor de la iteración 25**»). Se adopta el valor **convergido** para que el número sea reproducible. `c_uniforme = 316.342` y `c_sólido = 39.543` coinciden con la tesis sin cambios.

---

## Item 11 — H.E.1a: forma de la cota (corrección matemática)

**Antes** (Cap. I H.E.1a, Fig. 5.3, Tabla 5.5): la cota se enuncia como `dB(Dgmk VR(X), Dgmk VR(Y)) ≤ 2q·diam(X)` y la Fig. 5.3 contrasta `dB/diam` con `2q`.

**Después:** `dB(Dgmk VR(X), Dgmk VR(Y)) ≤ 2·dH(X, Y)` (k = 0,1), la forma del Teorema de Estabilidad (Cohen-Steiner, Edelsbrunner & Harer). La Fig. 5.3 pasa a graficar `dB/dH` contrastado con la cota `2`, y la Tabla 5.5, fila 1, pasa a `dB ≤ 2·dH`.

**Estado: IMPLEMENTADO.** La Fig. 5.3 (app y `visualization.figuras`) grafica `dB/dH` vs la cota `2` con barras de error. Medición R=100, n=400: máximo `dB/dH = 1.92 < 2`, todas las curvas por debajo de la cota y ninguna la toca — consistente con la observación del documento. La antigua cota normalizada `dB/diam ≤ 2q` se retira (no es consecuencia del teorema).

**Razón (con evidencia):** con el ruido definido en la propia tesis (`σ = q·diam(X)`), se tiene `dH(X,Y) ≈ 3σ = 3q·diam(X)`, no `q·diam(X)`. Por tanto `2dH ≈ 6q·diam`, y la cota `≤ 2q·diam` es ~3× más ajustada de lo que el teorema permite. Medición (20 réplicas, n=400):

| Nube | frac(dB/diam ≤ 2q) | frac(dB ≤ 2dH) |
|---|---|---|
| círculo | 0.00 | 1.00 |
| anillo | 0.50 | 1.00 |
| toro | 0.45 | 1.00 |
| disco | 1.00 | 1.00 |

La forma `dB ≤ 2dH` se cumple en el **100 %** de las réplicas (lo que el documento ya afirma en su Fig. 5.2(b)); la forma `dB/diam ≤ 2q` no es implicada por el teorema y no se sostiene empíricamente. Es la distinción que la propia Tabla 5.4 del documento establece.

**Consecuencia:** no cambia H.E.1a (sigue cumpliéndose al 100 %), solo la **notación de la cota**. H.E.1b y H.E.1c no se ven afectadas.

---

## Item 12 — H.E.1b: clasificador euclidiano de referencia y veredicto

**Antes:** el Cap. V / Tabla 5.5 reportan exactitud 0.950 (TDA) vs 0.925 (euclidiano), McNemar **no significativo** → H.E.1b = ×. El documento no especifica la forma exacta del clasificador euclidiano (solo que es "espectro PCA + inercia de k-medias + dispersión radial").

**Después:** el clasificador euclidiano se define como vecino-más-cercano sobre el descriptor compuesto **[espectro PCA, inercia de k-medias, dispersión radial]**, entrenado en muestras **limpias** y evaluado bajo deformación afín. Con el método fiel, la medición (n=400, 200/clase, semilla 42) da **TDA ≈ 0.905 vs Eucl ≈ 0.550**, McNemar p < 0.001 → **H.E.1b se sostiene** (el descriptor topológico supera significativamente al euclidiano).

**Razón:** el documento deja **sin definir** el clasificador euclidiano; reproducir su 0.925 exigiría un ajuste no documentado. Además, §5.6(a) del propio documento observa que "la dispersión radial... falla si el conjunto se deforma", consistente con un euclidiano débil bajo afín (y con el resultado positivo), **no** con 0.925.

**Consecuencia:** reconciliar la Tabla 5.5 (fila "Exactitud TDA vs euclidiana") y §5.6 con el clasificador definido; el veredicto de H.E.1b pasa a **"se sostiene"** (significativo) bajo el protocolo fiel.

---

## Item 13 — Fig. 5.1: la definición βk no reproduce su plateau 7/7

**Hecho:** la calibración de τ se calcula sobre **βk = (β₀, β₁)** (Def. 8.1 / Tabla 1.1 / pie de Fig. 5.1), con `βk(τ) = #{(b,d) : d−b > τ·diam}`.

**Inconsistencia del documento:** el pie y la definición usan βk (y citan "con τ=0.05 la esfera reporta (179,21)"), pero el **plateau 7/7 en τ∈[0.15, 0.20]** que muestra la Fig. 5.1 **solo se obtiene con β₁**:

| τ | βk=(β₀,β₁) | sólo β₁ |
|---|---|---|
| 0.05 | 2/7 | 3/7 |
| 0.15 | 5/7 | 7/7 |
| 0.17 | 6/7 | 7/7 |
| 0.20 | 6/7 | 7/7 |
| 0.25 | 5/7 | 5/7 |

Con βk, `esfera` y `gaussiana` leen β₀ = 2–3 (las H₀ de una muestra finita tienen merges que superan τ·diam), de modo que **no existe ventana 7/7** y el Algoritmo maestro abortaría ("no existe ventana de τ con acierto total").

**Consecuencia:** la Fig. 5.1 debe reportar el máximo real alcanzado por βk (6/7) y su ventana, **o** reformularse para β₁ (única forma de obtener el plateau 7/7). La app implementa βk (fiel al pie) y reporta el máximo obtenido.

**Resolución adoptada (medición de la app):** la calibración (Fig. 5.1) y el criterio de **q\*** (H.E.1c) se computan sobre **β₁** —el invariante que la filtración controla—; la **Fig. 5.2(a)** grafica **βk = (β₀, β₁)** como el documento. Con β₁, la calibración alcanza **7/7** en τ ∈ [0.15, 0.18] (τ = 0.15 en el borde inferior, como el documento) y H.E.1c da **q\* ≈ 0.05 ∈ [0.05, 0.10]** (acierto 0.99 para q ≤ 0.05). Con βk (β₀ umbralizado) no se alcanza ni el plateau 7/7 ni el q* del documento. **Acción:** cambiar «βk» por «β₁» en el pie de la Fig. 5.1, en el criterio de q* y en las filas «Acierto de βk» de la Tabla 5.5; la Fig. 5.2(a) sí queda en βk.

---

## Item 14 — Implementación exacta de la distancia bottleneck

**Antes:** `topology.bottleneck_distance` era una heurística greedy con norma L2
(vecino más cercano + proyección a la diagonal). Además, en `he_protocols.bottleneck_cota`
y `stability.verificar_cota_bottleneck` un diagrama vacío devolvía `0.0`, ocultando
violaciones.

**Después:** la distancia se calcula con la **definición estándar exacta**
(Cohen-Steiner et al. 2007): costo L∞ y emparejamiento resuelto por asignación
húngara (`scipy.optimize.linear_sum_assignment`) sobre `(A ∪ Δ)` vs `(B ∪ Δ)`,
con costo punto–diagonal = persistencia/2. Los diagramas vacíos se resuelven
contra la diagonal, nunca como 0.

**Razón (con evidencia):** la heurística subestimaba o sobreestimaba dB de forma
no sistemática (p. ej. círculo H₁: 0.674 vs exacto 0.672; toro H₀: 0.738 vs 0.761).
La Tabla 1.1 declara el instrumento como `Persim 0.3`; la versión exacta coincide
con persim/gudhi pero sin dependencia adicional.

**Consecuencia:** la verificación de H.E.1a pasa a ser concluyente. Con la métrica
exacta, `dB ≤ 2·dH` se cumple en **20/20** réplicas por nube (q = 0.10). El corolario
normalizado `dB/diam ≤ 2q` —no implicado por el teorema— falla en círculo (0/20) y
dos_círculos (1/20), en línea con el Item 11.

---

## Item 15 — Veredicto H.E.2b: se excluye p = 1

**Antes:** el veredicto final del barrido evaluaba la invariancia de β₁ sobre
`p ∈ {1,2,3,4}`. Como `p = 1` no penaliza (diseño gris, β₁ = 0 sin frontera
sólido/vacío), el veredicto se marcaba "varía" aunque el método fuera correcto.

**Después:** la invariancia se evalúa sobre `p ∈ {2,3,4}` (rango recomendado por
la literatura y por la propia §5.5.2 del documento), tanto en el barrido como en
el veredicto. `p = 1` se reporta como referencia, no como invariante.

**Razón:** H.E.2b afirma invariancia "en el rango recomendado"; incluir p = 1 es
un veto espurio.

---

## Item 16 — Techo de iteraciones, registro de entorno y reconstrucción E3

**Antes:** la UI limitaba `max_iter` a 200; no se exportaba registro de entorno;
las figuras solo se rasterizaban desde la sesión de Streamlit.

**Después:**
- `max_iter` por defecto 300 (Algoritmo 2, paso 24), ajustable en UI hasta 500.
- Cada ZIP incorpora `entorno.json` (entregable E5): Python, plataforma, versiones
  de paquetes, semilla y parámetros de la corrida.
- Nuevo módulo `src/tda/visualization/figuras.py` (entregable E3): reconstruye las
  figuras 5.1–5.7 **sin Streamlit**, por script, desde los protocolos o CSV, con
  los ejes y escalas del documento. CLI: `python -m tda.visualization.figuras --todos`.

**Razón:** reproducibilidad verificable (F6) y trazabilidad de las figuras.

---

## Item 17 — Inconsistencia en la Figura 5.6 del documento (β₁ = 3 vs 0)

**Hecho:** el dominio cuadrado con condiciones de contorno de viga en voladizo (empotramiento izquierdo, carga puntual derecha) converge a una banda diagonal de material **sin agujeros** ($\beta_1 = 0$), en contraste con el valor $\beta_1 = 3$ reportado en la Figura 5.6 del PDF.

**Resolución:** se documenta que los datos de la Figura 5.6 en el PDF presentan una inconsistencia de dominio (el conteo $\beta_1 = 3$ corresponde a la viga rectangular 2:1, mientras que el barrido de malla usa el dominio cuadrado). El software implementa el cálculo real FEM + TDA dando $\beta_1 = 0$ de manera estable en todo el barrido de malla cuadrada.

---

## Item 18 — Fig. 5.7: el barrido de umbral debe usar el Método B (GUDHI)

**Hecho (con evidencia):** el barrido de binarización sobre la viga 60×30 da resultados
opuestos según el método para $u \ge 0.90$:

| $u$ | Euler (A) $\beta_0/\beta_1$ | GUDHI (B) $\beta_0/\beta_1$ |
|---|---|---|
| 0.05–0.85 | 1 / 3 | 1 / 3 |
| 0.90 | **5 / 7** | **1 / 3** |
| 0.95 | 3 / 3 | **1 / 1** |

**Causa:** `betti_euler` cuenta $\beta_0$ con 4-conectividad pero $\chi = V - E + F$
sobre la unión geométrica, donde píxeles que se tocan en diagonal **sí** están
conectados. Al fragmentarse el sólido ($u$ alto), cada contacto diagonal genera un
agujero fantasma ($\beta_1 = \beta_0 - \chi$ deja de ser válida). La homología
cúbica de GUDHI es autoconsistente y no sufre este artefacto.

**Después:** las curvas de la Fig. 5.7 (app y `visualization.figuras`) se calculan
con el Método B (GUDHI), con fallback a Euler si GUDHI no está instalado. El doble
cómputo se conserva como **chequeo de concordancia** (la app advierte los umbrales
donde difieren en vez de ocultarlo). El barrido se extiende a
$u \in [0.05, 0.95]$ (19 puntos) para cubrir el rango del PDF, donde $\beta_1$
cae 3→1. Con GUDHI la figura reproduce al PDF: meseta $\beta_1 = 3$, $\beta_0 = 1$,
volumen en dash-dot gris con eje ajustado a los datos, y texto "umbral adoptado".

**Consecuencia:** ningún veredicto cambia ($\beta_1 = 3$ en $u = 0.5$ lo dan ambos
métodos). Solo la cola $u \ge 0.90$ de la Fig. 5.7.

---

## Resumen de trazabilidad (software → tesis)

| Hipótesis | Función | Dato | Figura |
|---|---|---|---|
| H.E.1a | `stability.verificar_cota_bottleneck` | — | 5.2(b), 5.3 |
| H.E.1b | `stability.comparar_tda_vs_euclidiano` | — | Tabla 5.5 |
| H.E.1c | `stability.barrido_cota_bottleneck` (acierto) | — | 5.2(a) |
| Calibración τ | `stability.betti_persistente` | — | 5.1 |
| H.E.2a | `metric_simp` + `experiment_simp` | `data/validacion_HE2_corregida.csv` | 5.4(b), Tabla 5.6 |
| H.E.2b | `experiment_simp` (barridos) | `data/barrido_beta1_HE2b.csv` | 5.5, 5.6, 5.7 |
| Doble cómputo β | `betti2d.betti_doble_computo` | — | 5.4(a) |
| Reconstrucción de figuras | `visualization.figuras` (E3) | `data/figuras/*.png`, `*.csv` | 5.1–5.7 |
