# Módulo de Procesamiento (Preprocesamiento TDA)

Este módulo implementa herramientas de procesamiento y filtrado aplicadas a diagramas de persistencia antes del análisis de distancias o la extracción de descriptores topológicos.

## Contexto Matemático

Los diagramas de persistencia generados a partir de filtraciones simplicial suelen contener una cantidad significativa de "ruido topológico", caracterizado por puntos extremadamente cercanos a la diagonal ($d_i - b_i \approx 0$). Estos puntos representan características de muy corta vida temporal en la filtración.

1. **Filtrado por Umbral de Persistencia:**
   Se define el tiempo de vida (persistencia) de una clase homológica como $pers(x) = d_i - b_i$. Para un umbral $\tau > 0$, el subdiagrama filtrado se define como:
   $$D_{\tau} = \{(b_i, d_i) \in D \mid d_i - b_i \geq \tau\}$$

2. **Normalización por Diámetro:**
   Para permitir la comparación coherente entre geometrías de escalas distintas, los diagramas se normalizan utilizando el diámetro del espacio métrico original (la distancia máxima entre dos puntos del *point cloud*):
   $$\bar{b}_i = \frac{b_i}{diam(X)}, \quad \bar{d}_i = \frac{d_i}{diam(X)}$$

3. **Histograma de Persistencia:**
   Representa la distribución empírica de los tiempos de vida útil, permitiendo proyectar el diagrama dimensional a vectores de características compactos.

## Relevancia en la Tesis

Este módulo es un paso crítico de sanitización para la validación de la **Hipótesis Específica H.E.1**. La normalización permite comparar de manera directa diagramas de persistencia de nubes de puntos originales frente a nubes perturbadas con ruido sin sesgos por cambio de escala del diámetro. Por otro lado, el filtrado por umbral es consistente con el criterio empleado en la optimización topológica SIMP, aislando los vacíos estructurales verdaderos ($\beta_1$ significativos) del ruido inducido por la discretización del elemento finito.

## Entradas y Salidas de las Funciones

### `filter_persistence_diagram(dgm, threshold)`
* **Entradas:**
  * `dgm` (List[List[float]]): Diagrama de persistencia original organizado por dimensiones.
  * `threshold` (float): Umbral de persistencia mínima $\tau$.
* **Salidas:**
  * `List[List[float]]`: Diagrama filtrado, conservando únicamente intervalos donde $death - birth \geq threshold$.

### `normalize_diagram(dgm, diameter)`
* **Entradas:**
  * `dgm` (List[List[float]]): Diagrama de persistencia original.
  * `diameter` (float): Diámetro máximo del conjunto de puntos.
* **Salidas:**
  * `List[List[float]]`: Diagrama escalado de forma proporcional al diámetro.

### `get_persistence_histogram(dgm, bins)`
* **Entradas:**
  * `dgm` (List[List[float]] o similar): Diagrama de persistencia.
  * `bins` (int): Cantidad de divisiones para el histograma (por defecto $10$).
* **Salidas:**
  * `np.ndarray`: Histograma unidimensional que muestra la distribución de los tiempos de vida.

## Dependencias
* `numpy`
