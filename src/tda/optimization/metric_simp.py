"""Clase principal MetricaTDA_SIMP — Algoritmo 1 completo.

Orquesta las tres fases del proceso:
  FASE 1 – Preprocesado: malla FEM, condiciones de contorno, carga
  FASE 2 – Optimización SIMP: bucle iterativo hasta convergencia dual
  FASE 3 – Análisis TDA: homología persistente → β₁ → μ_α

Esta clase NO incluye visualización (matplotlib/plotly) — solo computación.
Los datos de resultados se exponen como atributos para que la capa de UI
(Streamlit, CLI, etc.) los renderice como corresponda.
"""

import time
import numpy as np

from tda.core.fem import (
    calcular_K_elemental,
    ensamblar_K_global,
    resolver_FEM,
    calcular_compliance_sensibilidades,
    filtrar_sensibilidades,
    actualizar_OC,
)
from tda.core.topology import (
    binarizar_y_extraer_nube,
    escala_adaptativa,
    calcular_homologia_betti,
)
from tda.core.metric import metrica_compuesta


class MetricaTDA_SIMP:
    """
    Clase principal que implementa el Algoritmo 1 completo:
    SIMP (Solid Isotropic Material with Penalization) + TDA (Topological Data Analysis).

    Uso mínimo
    ──────────
    m = MetricaTDA_SIMP(nex=60, ney=30, f_V=0.5, p=3, alpha=0.012)
    m.definir_problema(F, dofs_fijos)
    m.optimizar()
    mu = m.fase_tda()

    # Acceder a resultados
    c_final = m.c_final
    beta1   = m.beta1
    mu      = m.mu
    """

    def __init__(self, nex, ney, E=1.0, nu=0.3,
                 f_V=0.5, p=3, r_min=2.4, alpha=0.012,
                 tol=1e-4, max_iter=200):
        """
        Inicializa la geometría, el material y los parámetros del algoritmo.

        Parámetros
        ──────────
        nex     : int    Elementos en x
        ney     : int    Elementos en y
        E, nu   : float  Módulo de Young y coeficiente de Poisson del sólido
        f_V     : float  Fracción de volumen objetivo (0 < f_V ≤ 1)
        p       : float  Factor de penalización SIMP (p ≥ 3)
        r_min   : float  Radio del filtro de sensibilidad (en unidades de elemento)
        alpha   : float  Peso α de la métrica compuesta (α > 0)
        tol     : float  Tolerancia de convergencia del bucle SIMP
        max_iter: int    Número máximo de iteraciones SIMP
        """

        # ── Geometría de la malla ─────────────────────────────────────────────
        self.nex   = nex
        self.ney   = ney
        self.N_e   = nex * ney
        self.nnx   = nex + 1
        self.nny   = ney + 1
        self.n_dof = 2 * (nex + 1) * (ney + 1)

        # ── Parámetros del algoritmo ──────────────────────────────────────────
        self.f_V     = f_V
        self.p       = p
        self.r_min   = r_min
        self.alpha   = alpha
        self.tol     = tol
        self.max_iter = max_iter

        # ── Matriz elemental K_0 ──────────────────────────────────────────────
        self.K0 = calcular_K_elemental(E, nu)

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

        # ── Matriz del filtro H (precomputada) ────────────────────────────────
        dist = np.sqrt(
            (self.cx[:, None] - self.cx)**2 +
            (self.cy[:, None] - self.cy)**2
        )
        self.H = np.maximum(0.0, r_min - dist)

        # ── Densidades iniciales ──────────────────────────────────────────────
        self.rho = np.full(self.N_e, f_V)

        # ── Variables de estado ───────────────────────────────────────────────
        self.F          = None
        self.fixed      = None
        self.c_hist     = []
        self.rho_hist   = []      # ← NUEVO: guardar historial de densidades
        self.rho_final  = None
        self.c_final    = None
        self.beta0      = None
        self.beta1      = None
        self.dgm1       = None
        self.nube       = None
        self.eps_star   = None
        self.mu         = None
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


    def optimizar(self, verbose=True, callback=None):
        """
        Ejecuta el bucle SIMP hasta convergencia dual.

        Criterio 1: |c^{k+1} − c^k| / c^k  < tol
        Criterio 2: max_e |ρ_e^{k+1} − ρ_e^k| < tol

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

        for k in range(1, self.max_iter + 1):

            # Paso 1: Ensamble K(ρ)
            K = ensamblar_K_global(self.rho, self.DOFS, self.n_dof, self.K0, self.p)

            # Paso 2: Resolver K·U = F
            U = resolver_FEM(K, self.F, self.fixed, self.n_dof)

            # Paso 3: Compliance y sensibilidades
            c, dc = calcular_compliance_sensibilidades(
                U, self.rho, self.DOFS, self.K0, self.p
            )

            # Paso 4: Filtrar sensibilidades
            dc_filt = filtrar_sensibilidades(dc, self.rho, self.H)

            # Paso 5: Actualizar densidades con OC
            rho_nuevo = actualizar_OC(self.rho, dc_filt, self.f_V)

            # Paso 6: Medir convergencia
            delta_c   = abs(c - c_ant) / max(c_ant, 1e-12)
            delta_rho = float(np.max(np.abs(rho_nuevo - self.rho)))

            self.c_hist.append(c)
            self.rho_hist.append(rho_nuevo.copy())
            self.rho = rho_nuevo
            c_ant = c
            self.n_iter = k

            # Callback para UI
            if callback:
                callback(k, c, delta_c, delta_rho, rho_nuevo)

            if verbose and (k <= 3 or k % 10 == 0):
                print(f"  iter {k:>4d}: c={c:>12.4f}  Δc/c={delta_c:>9.2e}  Δρ_max={delta_rho:>9.4f}")

            # Verificar convergencia dual
            if delta_c < self.tol and delta_rho < self.tol:
                if verbose:
                    print(f"  ✓ Convergencia en k={k}")
                self.converged = True
                break

        self.rho_final = self.rho.copy()
        self.c_final   = self.c_hist[-1]
        self.t_simp    = time.time() - t0

        if verbose:
            n_sol = np.sum(self.rho_final > 0.5)
            print(f"  Tiempo SIMP: {self.t_simp:.2f}s | "
                  f"Sólidos: {n_sol}/{self.N_e} = {100*n_sol/self.N_e:.1f}%")

        return self


    def fase_tda(self, verbose=True):
        """
        Ejecuta la FASE 3 (análisis topológico) y calcula μ_α.

        Retorna
        ───────
        mu : float  Métrica compuesta μ_α
        """
        assert self.rho_final is not None, "Llamar primero a optimizar()"

        t1 = time.time()

        # 3.1 Binarización y nube de puntos
        self.nube = binarizar_y_extraer_nube(self.rho_final, self.nex, self.ney)

        # 3.2 Escala adaptativa ε*
        self.eps_star = escala_adaptativa(self.nube, self.N_e)

        # 3.3 Homología persistente H₁
        self.beta1, self.dgm1 = calcular_homologia_betti(self.nube, self.eps_star)
        self.t_tda = time.time() - t1

        # 3.4 β₀ (componentes conexas)
        if len(self.nube) >= 2:
            from ripser import ripser
            dgm0 = ripser(self.nube, maxdim=0)["dgms"][0]
            self.beta0 = int(np.sum(np.isinf(dgm0[:, 1])))
        else:
            self.beta0 = len(self.nube)

        # 3.5 Métrica compuesta
        self.mu = metrica_compuesta(self.c_final, self.beta1, self.alpha)

        if verbose:
            print(f"  β₀ = {self.beta0} | β₁ = {self.beta1} | μ_α = {self.mu:.5f}")
            print(f"  Tiempo TDA: {self.t_tda:.3f}s")

        return self.mu


    def obtener_resultados(self):
        """
        Retorna un diccionario con todos los resultados para la capa de UI.

        Útil para Streamlit/Plotly/CLI que necesitan los datos sin acoplamiento.
        """
        return {
            "rho_final": self.rho_final,
            "c_final": self.c_final,
            "c_hist": np.array(self.c_hist),
            "rho_hist": self.rho_hist if self.rho_hist else None,
            "beta0": self.beta0,
            "beta1": self.beta1,
            "dgm1": self.dgm1,
            "nube": self.nube,
            "eps_star": self.eps_star,
            "mu": self.mu,
            "nex": self.nex,
            "ney": self.ney,
            "N_e": self.N_e,
            "f_V": self.f_V,
            "p": self.p,
            "alpha": self.alpha,
            "t_simp": self.t_simp,
            "t_tda": self.t_tda,
            "converged": self.converged,
            "n_iter": self.n_iter,
            "DOFS": self.DOFS,
            "K0": self.K0,
        }
