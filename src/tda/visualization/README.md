# Módulo de Visualización y Análisis Gráfico

Este módulo proporciona funciones gráficas avanzadas para representar los datos, comparar metodologías y visualizar la estabilidad topológica de las geometrías estudiadas.

## Contexto Matemático

La visualización de datos de alta dimensión y sus diagramas de persistencia es crucial para interpretar los resultados de homología persistente. El módulo implementa:

1. **Gráficos de Estabilidad Topológica (Robutez H.E.1):**
   Muestra el comportamiento del número de Betti promedio ($\beta_1$) y su desviación estándar frente a niveles crecientes de ruido estocástico mediante barras de error. Permite observar empíricamente los teoremas de estabilidad de diagramas de persistencia (Teorema de Estabilidad de Cohen-Steiner-Edelsbrunner-Harer).

2. **Comparación Espacial y Dimensional (TDA vs. Métricas Euclidianas):**
   - **PCA (Análisis de Componentes Principales):** Proyecta la nube de puntos tridimensional a un espacio bidimensional ortogonal de máxima varianza.
   - **K-Medias:** Algoritmo de agrupamiento euclidiano que particiona la nube en $K$ clústeres minimizando la inercia intraclúster (distancias al cuadrado al centroide).
   - **Invariantes Topológicos:** Calcula los números de Betti para revelar que, aunque las formas estén espacialmente superpuestas y el agrupamiento euclidiano falle, la estructura topológica intrínseca permanece distinguible.

## Relevancia en la Tesis

Este módulo genera las figuras y gráficos cuantitativos que se incluyen en el manuscrito final de la tesis para sostener la validez de la **Hipótesis Específica H.E.1**:

- `plot_stability_analysis` genera curvas de estabilidad de $\beta_1$ que demuestran la invarianza topológica hasta niveles de ruido del 25% en el toro.
- `plot_tda_vs_kmeans_pca` demuestra visualmente que los descriptores euclidianos clásicos no logran capturar la geometría de componentes entrelazados o superpuestos en presencia de ruido, mientras que TDA clasifica e identifica las geometrías con éxito gracias a la invarianza homotópica.

## Entradas y Salidas de las Funciones

### `plot_stability_analysis(shape, noise_levels, n_rep, n_points)`

- **Entradas:**
  - `shape` (str): Geometría de análisis (`torus` por defecto).
  - `noise_levels` (list[float]): Rangos de ruido a graficar.
  - `n_rep` (int): Réplicas del experimento por nivel de ruido ($100$ por defecto).
  - `n_points` (int): Cantidad de puntos en la muestra.
- **Salidas:**
  - `matplotlib.figure.Figure`: Objeto de figura con el gráfico de barras de error de $\beta_1$. Guarda el resultado en `data/robustez_HE1_{shape}.png`.

### `plot_tda_vs_kmeans_pca(n_points, noise_level)`

- **Entradas:**
  - `n_points` (int): Cantidad de puntos por geometría.
  - `noise_level` (float): Nivel de ruido gaussiano inyectado ($0.15$ por defecto).
- **Salidas:**
  - `matplotlib.figure.Figure`: Objeto de figura con dos páneles comparando el agrupamiento euclidiano (K-Medias) frente al Ground Truth, mostrando los números de Betti obtenidos. Guarda el gráfico en `data/comparacion_TDA_vs_KMedias.png`.

### `visualize_point_cloud_2d(points, title, ax, backend)`

- **Entradas:**
  - `points` (np.ndarray): Nube de puntos de dimensiones $(n, 2)$.
  - `title` (str, opcional): Título del gráfico.
  - `ax` (matplotlib.axes.Axes, opcional): Eje sobre el cual graficar.
  - `backend` (str): Motor de renderizado (`matplotlib` o `plotly`).
- **Salidas:**
  - Retorna la figura y ejes correspondientes.

## Ejecución de la Aplicación

Para iniciar el script, ejecute el siguiente comando desde la raíz del proyecto:

```bash
python src/tda/visualization/visualizer.py --analyze-stability --shape torus
```

## Dependencias

- `numpy`
- `matplotlib`
- `plotly` (para renderizado interactivo en 3D)
- `scikit-learn` (PCA y KMeans)
- `tda.simulation`
- `tda.core`
