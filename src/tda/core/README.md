# Módulo Core (Análisis Topológico de Datos)

Este módulo contiene la base matemática y algorítmica del proyecto para el cálculo de invariantes topológicos y distancias entre diagramas de persistencia.

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

## Dependencias
* `numpy`
* `persim` (para el cálculo eficiente de distancias)
