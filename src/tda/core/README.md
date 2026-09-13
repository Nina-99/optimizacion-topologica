# Módulo Core (Análisis Topológico de Datos)

Este módulo contiene la base matemática y algorítmica del proyecto para el cálculo de invariantes topológicos, distancias entre diagramas de persistencia y métricas compuestas.

## Archivos

- `topology.py` — Wasserstein, bottleneck, betti_numbers, binarización y homología sobre nubes de puntos.
- `betti2d.py` — Cálculo de betti y diagramas de persistencia sobre grillas 2D (GUDHI). Utilizado por MetricaTDA_SIMP en la fase TDA post-hoc.
- `fem.py` — Motor FEM Q4: ensamble de rigidez, solver, sensibilidades y actualización OC.
- `metric.py` — Métrica compuesta μ_α = c + α·β₁ y calibración de α*.

## Contexto Matemático

El análisis se fundamenta en la Homología Persistente. Dado un espacio métrico, se construye una filtración de complejos simpliciales (por ejemplo, el complejo de Vietoris-Rips) parametrizada por un radio de proximidad $\epsilon$. La persistencia de los generadores de los grupos de homología $H_k$ se codifica en diagramas de persistencia $D = \{(b_i, d_i)\}$, donde $b_i$ y $d_i$ representan los valores de nacimiento (*birth*) y muerte (*death*) de cada característica topológica.

Para comparar dos diagramas de persistencia $X$ e $Y$, se definen las siguientes métricas estables:

1. **Distancia de Wasserstein ($W_q$):**
   $$W_q(X, Y) = \left( \inf_{\gamma: X \to Y} \sum_{x \in X} \|x - \gamma(x)\|_\infty^q \right)^{1/q}$$
   donde $\gamma$ es una biyección entre los diagramas (incluyendo la proyección diagonal).

2. **Distancia de Bottleneck ($W_\infty$):**
   $$W_\infty(X, Y) = \inf_{\gamma: X \to Y} \sup_{x \in X} \|x - \gamma(x)\|_\infty$$

3. **Números de Betti ($\beta_0, \beta_1$):**
   $\beta_0$ mide el número de componentes conexas y $\beta_1$ mide el número de ciclos unidimensionales (túneles o agujeros).

## Relevancia en la Tesis

Este módulo provee las métricas fundamentales para la validación de la **Hipótesis Específica H.E.1**, la cual sostiene que las características topológicas (como los números de Betti y los diagramas de persistencia evaluados mediante distancias estables) exhiben robustez y estabilidad matemática frente a perturbaciones de ruido estocástico de hasta el 15% y 20% del diámetro del conjunto de datos, superando el desempeño de los descriptores puramente euclidianos y de agrupamiento (K-Medias/PCA).

## Entradas y Salidas de las Funciones

### `wasserstein_distance(dgm1, dgm2)`
* **Entradas:**
  * `dgm1` (np.ndarray): Primer diagrama de persistencia de dimensiones $(n, 2)$, compuesto por pares $[birth, death]$.
  * `dgm2` (np.ndarray): Segundo diagrama de persistencia de dimensiones $(m, 2)$, compuesto por pares $[birth, death]$.
* **Salidas:**
  * `float`: La distancia de Wasserstein entre ambos diagramas.

### `bottleneck_distance(dgm1, dgm2)`
* **Entradas:**
  * `dgm1` (np.ndarray): Primer diagrama de persistencia de dimensiones $(n, 2)$.
  * `dgm2` (np.ndarray): Segundo diagrama de persistencia de dimensiones $(m, 2)$.
* **Salidas:**
  * `float`: La distancia de Bottleneck entre ambos diagramas.

### `betti_numbers(persistence_diagram)`
* **Entradas:**
  * `persistence_diagram` (np.ndarray): Diagrama de persistencia en formato $(n, 3)$, donde cada fila representa $[birth, death, dimension]$. La columna de dimensión solo acepta $0$ ($H_0$) o $1$ ($H_1$).
* **Salidas:**
  * `Tuple[int, int]`: Tupla $(\beta_0, \beta_1)$ con los números de Betti correspondientes.

### `betti_doble_computo(S, verbose)` (betti2d.py)
* **Entradas:**
  * `S` (np.ndarray): Grid binario 2D de dimensiones $(nely, nelx)$.
  * `verbose` (bool): Si True, imprime info de depuración.
* **Salidas:**
  * `dict`: Diccionario con `beta0`, `beta1`, `dgm0`, `dgm1` (diagramas de persistencia GUDHI).

### `diagramas_gudhi(S)` (betti2d.py)
* **Entradas:**
  * `S` (np.ndarray): Grid binario 2D.
* **Salidas:**
  * `Tuple[dgm0, dgm1]`: Diagramas de persistencia para H_0 y H_1.

## Dependencias
* `numpy`
* `gudhi` (para cálculo de homología en grillas 2D)

> **Nota:** Las distancias Wasserstein y Bottleneck estan implementadas en numpy puro (sin depender de `persim`).
