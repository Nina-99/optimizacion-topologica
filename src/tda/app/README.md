# Módulo de Aplicación (TDA-SIMP Master Suite)

Este módulo implementa la interfaz gráfica interactiva del proyecto utilizando Streamlit, facilitando la visualización, experimentación y validación en tiempo real de las hipótesis de la tesis.

## Contexto Matemático e Interfaz

La aplicación se estructura en dos componentes principales alineados con los objetivos científicos de la investigación:

1. **Pestaña 1: Optimización SIMP (H.E.2):**
   * Permite al usuario configurar los parámetros geométricos e hiperparámetros de optimización (volfrac, penalización $p$, radio de filtro $rmin$, resolución de la malla del FEM).
   * Visualiza dinámicamente la evolución de la densidad del material ($\boldsymbol{\rho}$) a lo largo de las iteraciones.
   * Al finalizar, discretiza el diseño continuo en una nube de puntos e invoca algoritmos de TDA para estimar $\beta_1$ en el diagrama de persistencia simplicial, garantizando la convergencia a una topología admisible.

2. **Pestaña 2: Robustez Topológica vs Distancia Euclidiana (H.E.1):**
   * Genera nubes de puntos sintéticas tridimensionales (esfera y toro) superpuestas espacialmente y sujetas a ruido gaussiano regulable.
   * Ejecuta el algoritmo de K-Medias para particionar el espacio métrico euclidiano de forma clásica.
   * Contrasta la clasificación euclidiana (que suele fallar debido a la superposición y al ruido) con la homología persistente, mostrando cómo los números de Betti identifican de manera inequívoca las propiedades topológicas intrínsecas (esfera: $\beta_0=1, \beta_1=0$; toro: $\beta_0=1, \beta_1=2$).

## Relevancia en la Tesis

Este módulo sirve como la herramienta de demostración interactiva e inspección visual de la tesis. Permite a investigadores y evaluadores reproducir de forma dinámica e interactiva los experimentos que validan las hipótesis **H.E.1** y **H.E.2**, facilitando además la exportación de los datos resultantes a formato CSV para su posterior análisis estadístico.

## Ejecución de la Aplicación

Para iniciar la interfaz de usuario, ejecute el siguiente comando desde la raíz del proyecto:

```bash
streamlit run src/tda/app/app_master.py
```

## Dependencias
* `streamlit`
* `numpy`
* `pandas`
* `matplotlib`
* `plotly`
* `scikit-learn`
* `ripser`
* `tda.optimization`
