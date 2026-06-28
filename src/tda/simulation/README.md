# Módulo de Simulación y Experimentos

Este módulo coordina y ejecuta los experimentos automatizados necesarios para validar las hipótesis estadísticas y estructurales planteadas en la tesis.

## Contexto Matemático

Las simulaciones se dividen en dos vertientes principales:

1. **Simulación de Sensibilidad al Ruido Topológico (H.E.1):**
   Evalúa el comportamiento asintótico y la estabilidad de las características topológicas de nubes de puntos sintéticas bajo perturbaciones estocásticas. Se generan geometrías de referencia (esfera $S^2$, toro $T^2$, cubo $C^3$) y se añade ruido gaussiano blanco $\epsilon \sim \mathcal{N}(0, \sigma^2 \cdot \text{diam}(X))$ parametrizado por un factor de ruido $\sigma \in \{0.10, 0.15, 0.20\}$. La distancia entre el diagrama de persistencia original $D_{clean}$ y el ruidoso $D_{noisy}$ se mide mediante distancias de Wasserstein ($W_1$) y Bottleneck ($W_\infty$).

2. **Simulación de Optimización Estructural (H.E.2):**
   Ejecuta el pipeline de optimización SIMP de forma no interactiva (_headless_). Resuelve el problema de la viga voladizo y exporta el historial de convergencia, el compliance final y el número de Betti 1 ($\beta_1$) calculado sobre la topología óptima para comprobar si se satisface la restricción topológica de control.

## Relevancia en la Tesis

Este módulo unifica la generación de datos y la recolección de estadísticas para la validación formal de las dos hipótesis de la tesis:

- **H.E.1:** Mediante `run_tda_experiment`, se evalúa la estabilidad topológica de $\beta_1$ en $T^2$ frente a perturbaciones controladas (demostrando que se mantiene próximo a 2 a pesar de la inyección de ruido).
- **H.E.2:** Mediante `run_headless_experiment`, se comprueba empíricamente que la compliance se reduce más de un 40% y que la topología resultante respeta el límite $\beta_1 \leq 2$.

## Entradas y Salidas de las Funciones

### `generate_cloud(shape, n_points)`

- **Entradas:**
  - `shape` (str): Forma geométrica a generar (`sphere`, `torus`, o `cube`).
  - `n_points` (int): Número de puntos en el espacio tridimensional.
- **Salidas:**
  - `np.ndarray`: Nube de puntos de dimensiones $(n\_points, 3)$.

### `run_tda_experiment(shape, noise_levels, n_rep, n_points, seed)`

- **Entradas:**
  - `shape` (str): Nombre de la figura geométrica bajo ensayo.
  - `noise_levels` (list[float]): Lista de desviaciones estándar de ruido a evaluar.
  - `n_rep` (int): Número de simulaciones independientes de Monte Carlo por cada nivel de ruido (típicamente $100$).
  - `n_points` (int): Cantidad de puntos muestreados de la geometría.
  - `seed` (int): Semilla para garantizar la reproducibilidad.
- **Salidas:**
  - `dict`: Diccionario con medias y desviaciones estándar de $\beta_0, \beta_1$, distancia de Wasserstein y distancia de Bottleneck para cada nivel de ruido.

### `run_headless_experiment()`

- **Entradas:** Ninguna.
- **Salidas:** Ninguna. Ejecuta la optimización estructural SIMP y exporta el archivo `data/validacion_HE2_simp.csv`.

## Ejecución de la Aplicación

Para iniciar el script, ejecute el siguiente comando desde la raíz del proyecto:

```bash
python src/tda/simulation/experiment_simp.py
```

## Dependencias

- `numpy`
- `pandas`
- `scipy.spatial`
- `ripser`
- `persim`
- `tda.core`
- `tda.processing`
- `tda.optimization`
