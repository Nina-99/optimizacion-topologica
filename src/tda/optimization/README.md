# Modulo de Optimizacion (SIMP + TDA)

Este modulo implementa el algoritmo SIMP (Solid Isotropic Material with Penalization) acoplado con analisis topologico de datos para resolver problemas de optimizacion estructural en dos dimensiones.

## Componentes

### `MetricaTDA_SIMP` (Clase Principal — Algoritmo 1 completo)

Implementacion completa del Algoritmo 1: preprocesado, SIMP con OC, filtrado de densidad y fase TDA post-hoc. Es la clase utilizada por la Plataforma TDA-SIMP en las paginas H.E.2, H.G. y Ejemplo Viga.

- **Constructor:** `MetricaTDA_SIMP(nex, ney, E, nu, Lx, Ly, t, f_V, p, r_min, alpha, tol_c, tol_rho, max_iter)`
- **`definir_problema(F, dofs_fijos)`:** Define vector de fuerzas y grados de libertad fijos.
- **`optimizar(callback, verbose)`:** Ejecuta SIMP con OC. Callback firma `(k, c, delta_c, delta_rho, rho)`.
- **`fase_tda(verbose)`:** Calcula diagramas de persistencia (GUDHI) y retorna mu_alpha.
- **`obtener_resultados()`:** Retorna diccionario con rho_final, rho_tilde_final, c_final, beta0, beta1, mu, dgm0, dgm1, n_iter, converged, betti_concordancia.

### `SimpTda2DOptimizer` (Clase legacy)

Implementacion original simplificada de SIMP 2D. Utilizada en experimentos headless.

- **Constructor:** `SimpTda2DOptimizer(nelx, nely, volfrac, penal, rmin)`
- **`run_optimization(callback)`:** Ejecuta SIMP y retorna xPhys, dgms, betti_1, c, reduccion_pct.

### `BeamOptimizer` (Optimizacion de Vigas 1D)

Optimizacion de perfil de altura para vigas en voladizo utilizando diferencias finitas.

- **Constructor:** `BeamOptimizer(b, h0, p, N, E_s, sigma_adm, max_iter)`
- **`optimizar_viga_completo(L, F, callback)`:** Retorna x, h_v, sigma_MPa, M, Y, saving_pct, iterations.

## Contexto Matematico

La optimizacion topologica busca encontrar la distribucion optima de material dentro de un dominio para minimizar la flexibilidad (compliance), sujeta a una restriccion sobre el volumen total. El modelo SIMP penaliza las densidades intermedias para aproximarse a un diseno binario:

$$\min_{\boldsymbol{\rho}} \quad c(\boldsymbol{\rho}) = \mathbf{U}^T \mathbf{K}(\boldsymbol{\rho}) \mathbf{U} = \sum_{e=1}^{N} (\rho_e)^p \mathbf{u}_e^T \mathbf{k}_0 \mathbf{u}_e$$
$$\text{sujeto a} \quad \frac{V(\boldsymbol{\rho})}{V_0} \leq f, \quad \mathbf{K}(\boldsymbol{\rho}) \mathbf{U} = \mathbf{F}, \quad \mathbf{0} < \boldsymbol{\rho}_{min} \leq \boldsymbol{\rho} \leq \mathbf{1}$$

Una vez finalizada la optimizacion, las densidades filtradas (rho_e > 0.5) se mapean a una representacion binaria sobre la cual se calcula la homologia persistente H_1 para extraer beta_1.

## Relevancia en la Tesis

Este modulo es la piedra angular para la validacion de la **Hipotesis Especifica H.E.2**:

- **H.E.2a':** Reduccion >=40% vs diseno uniforme de igual volumen (rho = fV).
- **H.E.2b':** beta_1(omega_solido) es invariante bajo malla y r_min (descriptivo, no prescriptivo).
- Hallazgo 1: El teorema de estabilidad (Cohen-Steiner) NO protege a beta_k — acota bottleneck, no beta_k a escala fija.

## Dependencias

- `numpy`
- `scipy.sparse`, `scipy.sparse.linalg`
- `gudhi` (para homologia persistente en MetricaTDA_SIMP)
- `ripser` (para SimpTda2DOptimizer legacy)
