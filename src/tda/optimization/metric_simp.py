"""Clase principal MetricaTDA_SIMP — Algoritmo 1 completo.

Orquesta las tres fases del proceso:
  FASE 1 – Preprocesado: malla FEM, condiciones de contorno, carga
  FASE 2 – Optimización SIMP: bucle iterativo hasta convergencia dual
  FASE 3 – Análisis TDA: homología persistente → β₀, β₁

Esta clase NO incluye visualización (matplotlib/plotly) — solo computación.
Los datos de resultados se exponen como atributos para que la capa de UI
(Streamlit, CLI, etc.) los renderice como corresponda.

Convergencia (Cap. III §3.4.4):
  Δc/c < 10⁻⁴  Y  Δρ_max < 10⁻²

Binarización TDA (Cap. V §5.5.4):
  Ω_sólido = {e : ρ̃_e ≥ u_V}, con u_V el umbral que conserva el volumen.
  La densidad es la FILTRADA (ρ̃), no la cruda.
"""

import time
import numpy as np
from scipy.sparse import csr_matrix

from tda.core.fem import (
    calcular_K_elemental,
    ensamblar_K_global,
    resolver_FEM,
    calcular_compliance_sensibilidades,
    actualizar_OC,
)
from tda.core.betti2d import (
    betti_doble_computo, diagramas_campo, persistencia_minima,
)


class MetricaTDA_SIMP:
    """
    Clase principal que implementa el Algoritmo 1 completo:
    SIMP (Solid Isotropic Material with Penalization) + TDA (Topological Data Analysis).

    Uso mínimo
    ──────────
    m = MetricaTDA_SIMP(nex=60, ney=30, f_V=0.5, p=3)
    m.definir_problema(F, dofs_fijos)
    m.optimizar()
    m.fase_tda()

    # Acceder a resultados
    c_final = m.c_final
    beta1   = m.beta1
    """

    def __init__(self, nex, ney, E=1.0, nu=0.3,
                 Lx=None, Ly=None, t=1.0,
                 f_V=0.5, p=3, r_min=2.4,
                 tol_c=1e-4, tol_rho=1e-2, max_iter=300,
                 usar_filtro=True):
        """
        Inicializa la geometría, el material y los parámetros del algoritmo.

        Parámetros
        ──────────
        nex     : int    Elementos en x
        ney     : int    Elementos en y
        E, nu   : float  Módulo de Young y coeficiente de Poisson del sólido
        Lx, Ly  : float  Dimensiones físicas (ej. mm). Si es None, asume = nex, ney
        t       : float  Espesor del elemento
        f_V     : float  Fracción de volumen objetivo (0 < f_V ≤ 1)
        p       : float  Factor de penalización SIMP (p ≥ 3)
        r_min   : float  Radio del filtro de densidad (en unidades de elemento)
        tol_c   : float  Tolerancia de convergencia de compliance (Perfil §9.4.4: 10⁻⁴)
        tol_rho : float  Tolerancia de convergencia de densidad (Cap. III §3.4.4: 10⁻²)
        max_iter: int    Número máximo de iteraciones SIMP
        usar_filtro: bool  Si es False, resuelve el problema sin filtro de densidad.
                           Se usa para la cota inferior convexa M* (p = 1, sin filtro)
                           de la Proposición 2.3 del Cap. II.
        """

        # ── Geometría de la malla ─────────────────────────────────────────────
        self.nex   = nex
        self.ney   = ney
        self.E     = float(E)
        self.nu    = float(nu)
        self.Lx    = Lx if Lx is not None else float(nex)
        self.Ly    = Ly if Ly is not None else float(ney)
        self.dx    = self.Lx / nex
        self.dy    = self.Ly / ney
        self.t     = t
        self.N_e   = nex * ney
        self.nnx   = nex + 1
        self.nny   = ney + 1
        self.n_dof = 2 * (nex + 1) * (ney + 1)

        # ── Parámetros del algoritmo ──────────────────────────────────────────
        self.f_V     = f_V
        self.p       = p
        self.r_min   = r_min
        self.tol_c   = tol_c
        self.tol_rho = tol_rho
        self.max_iter = max_iter
        self.usar_filtro = bool(usar_filtro)

        # ── Matriz elemental K_0 ──────────────────────────────────────────────
        self.K0 = calcular_K_elemental(E, nu, self.dx, self.dy, self.t)

        # ── Índices de DOF por elemento (precomputados) ───────────────────────
        idx = np.arange(self.N_e)
        ey  = idx // nex
        ex  = idx % nex
        nnx = self.nnx
        n1  = ey * nnx + ex
        n2  = n1 + 1
        n3  = (ey + 1) * nnx + ex + 1
        n4  = (ey + 1) * nnx + ex
        self.DOFS = np.stack(
            [2*n1, 2*n1+1, 2*n2, 2*n2+1,
             2*n3, 2*n3+1, 2*n4, 2*n4+1], axis=1
        )

        # ── Centroides (para filtro y TDA) ────────────────────────────────────
        self.cx = (idx % nex) + 0.5
        self.cy = (idx // nex) + 0.5

        # ── Matriz del filtro H (DISPERSA: soporte local rmin) ────────────────
        # H[i,j] = max(0, rmin − dist(centroide_i, centroide_j)). Como el soporte
        # es local, se construye DISPERSA: evita la matriz densa N×N (O(N²)) que
        # hacía inviable el barrido de malla (p. ej. 120×120 → ~1.6 GB).
        rad = int(np.ceil(r_min))
        ix = idx % nex          # columna (x)
        iy = idx // nex         # fila (y)
        rows, cols, vals = [], [], []
        for dxi in range(-rad, rad + 1):
            for dyi in range(-rad, rad + 1):
                w = r_min - np.sqrt(dxi * dxi + dyi * dyi)
                if w <= 0:
                    continue
                jx, jy = ix + dxi, iy + dyi
                mask = (jx >= 0) & (jx < nex) & (jy >= 0) & (jy < ney)
                if not mask.any():
                    continue
                rows.append(idx[mask])
                cols.append(jy[mask] * nex + jx[mask])
                vals.append(np.full(int(mask.sum()), w))
        self.H = csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.N_e, self.N_e),
        )
        self.Hs = np.asarray(self.H.sum(axis=1)).ravel()

        # ── Densidades iniciales ──────────────────────────────────────────────
        self.rho = np.full(self.N_e, f_V)

        # ── Variables de estado ───────────────────────────────────────────────
        self.F          = None
        self.fixed      = None
        self.c_hist     = []
        self.rho_hist   = []      # ← Historial de densidades ρ (crudo)
        self.rho_tilde_hist = []  # ← Historial de densidades filtradas ρ̃
        self.rho_final  = None
        self.rho_tilde_final = None  # ← Densidad filtrada final (para TDA)
        self.c_final    = None
        self.beta0      = None
        self.beta1      = None
        self._betti_concordancia = None
        self._betti_detalle = None
        self.t_simp     = 0.0
        self.t_tda      = 0.0
        self.converged  = False
        self.n_iter     = 0


    def definir_problema(self, F, dofs_fijos):
        """
        Asigna el vector de cargas externas y las condiciones de contorno.

        Parámetros
        ──────────
        F         : ndarray (n_dof,)  Vector de fuerzas externas nodales
        dofs_fijos: array-like         Índices de DOF con desplazamiento = 0
        """
        self.F     = np.asarray(F, dtype=float)
        self.fixed = np.asarray(dofs_fijos, dtype=int)


    def filtrar_densidad(self, rho):
        """
        Aplica el filtro de densidad (§8.4 del Perfil).

        ρ̃_e = Σ_i w(x_i, x_e) · ρ_i / Σ_i w(x_i, x_e)

        Este filtro es DISTINTO del filtro de sensibilidades (Sigmund 2007).
        Se usa para binarización TDA: Ω_sólido = {e : ρ̃_e ≥ 1/2}.

        Parámetros
        ──────────
        rho : ndarray (N_e,)  Densidades crudas

        Retorna
        ───────
        rho_tilde : ndarray (N_e,)  Densidades filtradas
        """
        return (self.H @ rho) / (self.Hs + 1e-30)


    def optimizar(self, verbose=True, callback=None):
        """
        Ejecuta el bucle SIMP hasta convergencia dual.

        Criterio (Perfil §9.4.4):
          Criterio 1: |c^{k+1} − c^k| / c^k  < tol_c   (10⁻⁴)
          Criterio 2: max_e |ρ_e^{k+1} − ρ_e^k| < tol_rho (10⁻²)

        Parámetros
        ──────────
        verbose  : bool     Mostrar progreso en consola
        callback : callable Función opcional callback(k, c, delta_c, delta_rho, rho)
                             para streaming en UI (Streamlit, etc.)

        Retorna
        ───────
        self (para encadenar métodos)
        """
        assert self.F is not None, "Llamar primero a definir_problema()"

        c_ant = np.inf
        t0    = time.time()
        self.c_hist   = []
        self.rho_hist = []
        self.rho_tilde_hist = []

        for k in range(1, self.max_iter + 1):

            # Paso 1: Filtrar densidad ρ̃ = H·ρ/Hs (Algoritmo 2, paso 7).
            #   El filtro es el operador de rango finito (compacto) que
            #   garantiza existencia de mínimo; la densidad FÍSICA que entra
            #   al ensamble FEM es la filtrada ρ̃ (no la cruda ρ).
            #   Para M* (p = 1, sin filtro) se usa ρ directamente.
            rho_tilde = (self.filtrar_densidad(self.rho)
                         if self.usar_filtro else self.rho.copy())

            # Paso 2: Ensamble K(ρ̃)
            K = ensamblar_K_global(rho_tilde, self.DOFS, self.n_dof, self.K0, self.p)

            # Paso 3: Resolver K·U = F
            U = resolver_FEM(K, self.F, self.fixed, self.n_dof)

            # Paso 4: Compliance y sensibilidades (sobre la densidad física ρ̃)
            c, dc = calcular_compliance_sensibilidades(
                U, rho_tilde, self.DOFS, self.K0, self.p
            )

            # Paso 5: Regla de la cadena del filtro de densidad (Alg. 2, paso 13):
            #   d̂c = H·dc/Hs. Sin filtro, la sensibilidad es la directa.
            dc_filt = self.filtrar_densidad(dc) if self.usar_filtro else dc

            # Paso 6: Actualizar densidades de diseño con OC.
            #   La restricción de volumen se impone sobre la densidad FÍSICA
            #   (ρ̃), que es la que ocupa volumen; no sobre ρ cruda.
            rho_nuevo = actualizar_OC(
                self.rho, dc_filt, self.f_V,
                rho_vol_fn=self.filtrar_densidad if self.usar_filtro else None,
            )

            # Paso 7: Medir convergencia (Perfil §9.4.4:
            #   Δc/c < 10⁻⁴  Y  Δρ_max < 10⁻²)
            if np.isinf(c_ant):
                delta_c = 0.0
            else:
                delta_c = abs(c - c_ant) / max(c_ant, 1e-12)
            delta_rho = float(np.max(np.abs(rho_nuevo - self.rho)))

            self.c_hist.append(c)
            self.rho_hist.append(rho_nuevo.copy())
            self.rho_tilde_hist.append(rho_tilde.copy())
            self.rho = rho_nuevo
            c_ant = c
            self.n_iter = k

            # Callback para UI
            if callback:
                callback(k, c, delta_c, delta_rho, rho_nuevo)

            if verbose and (k <= 3 or k % 10 == 0):
                print(f"  iter {k:>4d}: c={c:>12.4f}  Δc/c={delta_c:>9.2e}  Δρ_max={delta_rho:>9.4f}")

            # Verificar convergencia dual (Perfil §9.4.4: criterios separados)
            if delta_c < self.tol_c and delta_rho < self.tol_rho:
                if verbose:
                    print(f"  ✓ Convergencia en k={k}")
                self.converged = True
                break

        self.rho_final = self.rho.copy()
        self.rho_tilde_final = (self.filtrar_densidad(self.rho).copy()
                                if self.usar_filtro else self.rho.copy())
        self.c_final   = self.c_hist[-1]
        self.t_simp    = time.time() - t0

        if verbose:
            n_sol = np.sum(self.rho_tilde_final > 0.5)
            print(f"  Tiempo SIMP: {self.t_simp:.2f}s | "
                  f"Sólidos (ρ̃≥½): {n_sol}/{self.N_e} = {100*n_sol/self.N_e:.1f}%")

        return self


    def umbral_volumen(self, f_V=None, rho=None):
        """Umbral de binarización que mejor conserva la fracción de volumen.

        El Cap. V usa u_V (u = 0,554 en el diseño base), no u = 0,5: un umbral
        fijo cambia el volumen y, con él, la compliance del diseño fabricable.
        Se elige el umbral sobre los valores únicos de ρ̃ cuyo conteo de sólidos
        queda más cerca de f_V·N_e. Devuelve (umbral, fracción_volumen).
        """
        rho = np.asarray(self.rho_tilde_final if rho is None else rho, dtype=float).ravel()
        f_V = self.f_V if f_V is None else float(f_V)
        n = rho.size
        k = int(round(f_V * n))
        if k <= 0:
            return float(np.max(rho) + 1e-12), 0.0
        if k >= n:
            return float(np.min(rho) - 1e-12), 1.0

        mejores = None
        for u in np.unique(rho)[::-1]:
            count = int(np.sum(rho >= u))
            if mejores is None or abs(count - k) < abs(mejores[1] - k):
                mejores = (float(u), count)
            if count == k:
                break
        u, count = mejores

        # Si hay empates que impiden el conteo exacto, elegir el umbral
        # adyacente que deje el volumen más próximo (sin salir de [min,max]).
        if count != k:
            candidatos = []
            if count > k:
                vals = np.unique(rho)
                pos = int(np.searchsorted(-vals, -u))
                if pos + 1 < len(vals):
                    candidatos.append(float(vals[::-1][pos + 1]))
            else:
                vals = np.unique(rho)
                pos = int(np.searchsorted(vals, u))
                if pos > 0:
                    candidatos.append(float(vals[pos - 1]))
            for c in candidatos:
                cc = int(np.sum(rho >= c))
                if abs(cc - k) < abs(count - k):
                    u, count = c, cc
        return float(u), float(count / n)

    def binaria(self, umbral=None):
        """Máscara booleana (ney, nex) del sólido binarizado."""
        if umbral is None:
            umbral, _ = self.umbral_volumen()
        return self.rho_tilde_final.reshape((self.ney, self.nex)) >= float(umbral)

    def compliance_densidad(self, rho, p=None, filtrar=False):
        """Compliance de un campo de densidades arbitrario.

        `p=1` y `filtrar=False` evalúan el material físico 0/1 del diseño
        fabricable; es la métrica con la que el Cap. V compara c_bin y c_unif.
        """
        rho = np.asarray(rho, dtype=float).ravel()
        if rho.size != self.N_e:
            raise ValueError(f"rho debe tener {self.N_e} elementos, no {rho.size}")
        if filtrar:
            rho = self.filtrar_densidad(rho)
        p = self.p if p is None else float(p)
        K = ensamblar_K_global(rho, self.DOFS, self.n_dof, self.K0, p)
        U = resolver_FEM(K, self.F, self.fixed, self.n_dof)
        c, _ = calcular_compliance_sensibilidades(U, rho, self.DOFS, self.K0, p)
        return float(c)

    def fase_tda(self, verbose=True, umbral=None):
        """
        Ejecuta la FASE 3 (análisis topológico): doble cómputo de β₀, β₁
        (Método A = Euler 8-conexo, Método B = GUDHI cubical).
        La concordancia de ambos métodos valida el indicador; si discrepan,
        se levanta RuntimeError (compuerta bloqueante F2 del Cap. IV).

        `umbral=None` usa el umbral que conserva el volumen u_V. Para el
        barrido de la Figura 5.7 se pasa explícitamente cada u.

        Retorna
        ───────
        beta1 : int  Número de Betti β₁ del diseño binarizado
        """
        assert self.rho_final is not None, "Llamar primero a optimizar()"

        t1 = time.time()

        # 3.1 Grid binario del diseño: Ω_sólido = {e : ρ̃_e ≥ u_V}
        if umbral is None:
            umbral, _ = self.umbral_volumen()
        self.umbral_binarizacion = float(umbral)
        grid_bool = self.rho_tilde_final.reshape((self.ney, self.nex)) >= self.umbral_binarizacion
        self.fraccion_volumen_binarizada = float(grid_bool.mean())

        # 3.2 Doble cómputo de β₀, β₁ (Método A + Método B)
        resultado = betti_doble_computo(grid_bool, verbose=verbose)
        self.beta0 = resultado["beta0_euler"]  # Euler como primario
        self.beta1 = resultado["beta1_euler"]
        self._betti_concordancia = resultado["concordancia"]
        self._betti_detalle = resultado  # para auditoría

        if not self._betti_concordancia:
            self.t_tda = time.time() - t1
            raise RuntimeError(
                "Indicador topológico inválido: Euler y GUDHI discrepan "
                f"(β₀={resultado['beta0_euler']}/{resultado['beta0_gudhi']}, "
                f"β₁={resultado['beta1_euler']}/{resultado['beta1_gudhi']}). "
                "Compuerta F2: detener antes de contrastar H.E.2b."
            )

        # 3.3 Diagramas de persistencia del CAMPO (f = 1 − ρ̃), no del grid
        #     binario. Es el objeto que respalda la Figura 5.7(c) y la cota
        #     de persistencia del Teorema 2.1.
        self.dgm0, self.dgm1 = diagramas_campo(
            self.rho_tilde_final.reshape((self.ney, self.nex))
        )
        self.persistencia_min_h1 = persistencia_minima(self.dgm1, k=self.beta1)

        self.t_tda = time.time() - t1

        if verbose:
            conc = "✓" if self._betti_concordancia else "✗"
            print(f"  u_V = {self.umbral_binarizacion:.3f} | "
                  f"volumen = {self.fraccion_volumen_binarizada:.3f}")
            print(f"  β₀ = {self.beta0} | β₁ = {self.beta1} | Concordancia {conc}")
            print(f"  Persistencia mínima H₁ = {self.persistencia_min_h1:.3f}")
            print(f"  Tiempo TDA: {self.t_tda:.3f}s")
        return self.beta1

    def obtener_resultados(self):
        """
        Retorna un diccionario con todos los resultados para la capa de UI.

        Útil para Streamlit/Plotly/CLI que necesitan los datos sin acoplamiento.
        """
        return {
            "rho_final": self.rho_final,
            "rho_tilde_final": self.rho_tilde_final,
            "c_final": self.c_final,
            "c_hist": np.array(self.c_hist),
            "rho_hist": self.rho_hist if self.rho_hist else None,
            "rho_tilde_hist": self.rho_tilde_hist if self.rho_tilde_hist else None,
            "beta0": self.beta0,
            "beta1": self.beta1,
            "umbral_binarizacion": getattr(self, "umbral_binarizacion", None),
            "fraccion_volumen_binarizada": getattr(self, "fraccion_volumen_binarizada", None),
            "persistencia_min_h1": getattr(self, "persistencia_min_h1", None),
            "dgm0": getattr(self, "dgm0", None),
            "dgm1": getattr(self, "dgm1", None),
            "betti_concordancia": getattr(self, "_betti_concordancia", None),
            "betti_detalle": getattr(self, "_betti_detalle", None),
            "nex": self.nex,
            "ney": self.ney,
            "N_e": self.N_e,
            "f_V": self.f_V,
            "p": self.p,
            "tol_c": self.tol_c,
            "tol_rho": self.tol_rho,
            "t_simp": self.t_simp,
            "t_tda": self.t_tda,
            "converged": self.converged,
            "n_iter": self.n_iter,
            "DOFS": self.DOFS,
            "K0": self.K0,
        }
