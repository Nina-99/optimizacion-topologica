# Modulo de Aplicacion — Plataforma TDA-SIMP

Este modulo implementa la interfaz grafica interactiva del proyecto utilizando Streamlit. La plataforma se estructura como una landing page con navegacion a 4 modulos especializados, cada uno alineado con los objetivos cientificos de la investigacion.

## Estructura

```
app/
├── plataforma_tda_simp.py       # Landing page con navegacion a los 4 modulos
├── theme.py                     # Tema visual, estilos CSS y funciones de UI
├── download_utils.py            # Utilidades de exportacion y configuracion .exe
└── pages/
    ├── 1_H.E.1_Robustez_TDA_vs_Euclidianos.py
    ├── 2_H.E.2_Optimizacion_SIMP_Metrica_Compuesta.py
    ├── 3_H.G._Comparacion_Integrada_TDA-SIMP.py
    └── 4_Ejemplo_Viga_1D_vs_2D.py
```

## Modulos

### 1. H.E.1 — Robustez TDA vs Euclidianos

Validacion de la Hipotesis Especifica H.E.1: la homologia persistente proporciona descriptores estables frente a ruido, superando a descriptores euclidianos.

- Genera nubes de puntos sinteticas (esfera, toro) con ruido gaussiano regulable.
- Ejecuta K-Medias y contrasta con clasificacion topologica (beta_0, beta_1).
- **Cuadro 4:** Tasa de acierto de beta_k por nivel de ruido, por tipo de nube y nivel de confiabilidad.
- **Cuadro 5:** Exactitud TDA vs Euclidiano con y sin deformacion afin.
- **Estimacion de q*:** Primer nivel donde beta_1 se desvia del valor teorico.
- Incluye diagramas de persistencia (Ripser), barras de estabilidad y exportacion CSV.

### 2. H.E.2 — Optimizacion SIMP + Metrica Compuesta

Validacion de la Hipotesis Especifica H.E.2: SIMP con metrica compuesta mu_alpha logra reduccion >=40% y beta_1 invariante.

- Optimizacion topologica 2D con SIMP (algoritmo OC).
- Metrica compuesta mu_alpha = c + alpha * beta_1 con calibracion de alpha*.
- **Barrido multi-configuracion:** rmin, p, y malla (rectangular 2:1 o cuadrada).
- **Cuadro 7:** Resultados por valor de p (p=1,2,3,4) con concordancia.
- **Cuadro 8:** Resultados por resolucion de malla (Ne en {1600, 6400, 14400}).
- Validacion H.E.2a' (reduccion >=40%) y H.E.2b' (beta_1 invariante, descriptivo).
- Incluye convergencia, diagramas de persistencia y veredicto consolidado.

### 3. H.G. — Comparacion Integrada TDA-SIMP

Sintesis integral de TDA+SIMP: validacion de la Hipotesis General y Pareto-optimalidad.

- Barrido de penalizacion p={2,3,4} con la misma malla.
- Analisis de dominancia estricta entre configuraciones.
- Sensibilidad al parametro alpha con grafico de barras.
- Diagramas de persistencia lado a lado (H_1).
- Cuadro 9: Resumen comparativo consolidado.
- Veredicto: H.E.2a' y H.E.2b' con omega_solido, tau_M, rho_min.

### 4. Ejemplo Viga 1D vs 2D

Laboratorio interactivo de optimizacion estructural: comparacion de enfoques analiticos y numericos.

- **3 casos de carga 2D:** Voladizo-Puntual, Articulado-Central, Voladizo-Distribuida.
- **Modo 1D:** Perfil de altura optima (analytical, viga voladizo).
- **Modo 2D:** SIMP topologico con animacion de iteraciones + TDA post-hoc.
- **Barrido p:** Comparacion p=2,3,4 lado a lado.
- **Comparacion 1D vs 2D:** Mismo problema, dos enfoques, tabla resumen.
- **Factor de seguridad:** Estimacion post-hoc del factor de seguridad.
- **Exportacion STL:** Descarga del diseno binario como malla 3D.
- **Tamano minimo de miembro:** Filtro de tamano minimo para manufacturabilidad.

## Ejecucion

```bash
streamlit run src/tda/app/plataforma_tda_simp.py
```

## Dependencias

- `streamlit`
- `numpy`
- `pandas`
- `matplotlib`
- `plotly`
- `scikit-learn`
- `ripser`
- `tda.optimization` (MetricaTDA_SIMP, BeamOptimizer)
- `tda.core` (betti_doble_computo, diagramas_gudhi)
