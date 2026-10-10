# Modulo de Aplicacion — Plataforma TDA-SIMP

Este modulo implementa la interfaz grafica interactiva del proyecto utilizando Streamlit. La plataforma se estructura como una landing page con navegacion a 3 modulos especializados, cada uno alineado con los objetivos cientificos de la investigacion.

## Estructura

```
app/
├── plataforma_tda_simp.py       # Landing page con navegacion a los 3 modulos
├── theme.py                     # Tema visual, estilos CSS y funciones de UI
├── download_utils.py            # Utilidades de exportacion y configuracion .exe
└── pages/
    ├── 1_H.E.1_Robustez_TDA_vs_Euclidianos.py
    ├── 2_H.E.2_Optimizacion_SIMP_Metrica_Compuesta.py
    └── 4_Ejemplo_Viga_1D_vs_2D.py
```

## Modulos

### 1. H.E.1 — Robustez TDA vs Euclidianos

Validacion de la Hipotesis Especifica H.E.1: la homologia persistente proporciona descriptores estables frente a ruido, superando a descriptores euclidianos.

- Genera nubes de puntos sintéticas (esfera, toro) con ruido gaussiano regulable.
- Ejecuta K-Means y contrasta con clasificación topológica ($\beta_0$, $\beta_1$).
- **Figura 5.1:** Curva de calibración del umbral $\tau$ contra el patrón de oro (7 nubes).
- **Figura 5.2(a):** Tasa de acierto de $\beta_k$ por nivel de ruido; **(b):** cota réplica por réplica.
- **Figura 5.3:** Distancia bottleneck normalizada vs cota $2q$, por nube y nivel.
- **Tabla 5.5:** Caso 1 (indicador, criterio, esperado y medido) construida de la corrida.
- **Test de McNemar** con tabla de contingencia explicada y **estimación de q***.
- Incluye diagramas de persistencia (Ripser) y exportación ZIP (CSV + PDF + PNG).

### 2. H.E.2 — Optimizacion SIMP

Validación de la Hipótesis Específica H.E.2: SIMP logra reducción ≥40% y $\beta_1$ invariante (evaluación topológica post hoc).

- Optimización topológica 2D con SIMP (algoritmo OC).
- **Barrido multi-configuración:** $r_{\min}$, $p$ y malla (rectangular 2:1 o cuadrada).
- **Figuras 5.4–5.7** con epígrafes del documento y sus datos (compliance, grisura, $\beta_1$, fracción de volumen).
- **Tabla 5.6** dinámica (compliance vs referentes) y **doble cómputo** de Betti: característica de Euler $\chi$ vs homología cúbica (GUDHI).
- Validación H.E.2a (reducción ≥40% vs uniforme) y H.E.2b ($\beta_1$ invariante, descriptivo).
- Incluye convergencia, diagramas de persistencia y exportación ZIP con PNG de cada gráfico.

### 3. Ejemplo Viga 1D vs 2D

Laboratorio interactivo de optimizacion estructural: comparacion de enfoques analiticos y numericos.

- **3 casos de carga 2D:** Voladizo-Puntual, Articulado-Central, Voladizo-Distribuida.
- **Modo 1D:** Perfil de altura optima (analytical, viga voladizo).
- **Modo 2D:** SIMP topologico con animacion de iteraciones + TDA post-hoc.
- **Barrido p:** Comparacion p=2,3,4 lado a lado.
- **Comparacion 1D vs 2D:** Mismo problema, dos enfoques, tabla resumen.
- **Factor de seguridad:** Estimacion post-hoc del factor de seguridad.
- **Exportacion STL:** Descarga del diseno binario como malla 3D.
- **Tamano minimo de miembro:** Filtro de tamano minimo para manufacturabilidad.

## Ejecución

```bash
# Con uv (recomendado):
uv run streamlit run src/tda/app/plataforma_tda_simp.py
```

## Dependencias

- `streamlit`
- `numpy`
- `pandas`
- `matplotlib`
- `plotly` + `kaleido` (exportación de gráficos a PNG; requiere Chrome vía `plotly_get_chrome`)
- `scikit-learn`
- `ripser` (instalado con `--no-deps`)
- `gudhi` (homología cúbica; si falta, el doble cómputo queda sin verificar)
- `tda.optimization` (MetricaTDA_SIMP, BeamOptimizer)
- `tda.core` (betti_doble_computo, diagramas_gudhi)
