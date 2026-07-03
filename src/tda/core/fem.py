"""Núcleo FEM para optimización topológica SIMP.

Implementa la formulación de elementos finitos Q4 (cuadrilateral de 4 nodos)
para problemas de elasticidad 2D en estado plano de tensiones.

Bloques 1-2 del Algoritmo 1 (Métrica Compuesta TDA-SIMP):
  - Matriz de rigidez elemental K_0 (8×8) con cuadratura de Gauss 2×2
  - Ensamble vectorizado de la matriz global K(ρ) con SIMP
  - Solver K·U = F con condiciones de contorno
  - Cálculo de compliance y sensibilidades
  - Filtro de sensibilidad por convolución espacial
  - Actualización por Criterio de Optimalidad (OC)
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def calcular_K_elemental(E=1.0, nu=0.3):
    """
    Calcula la matriz de rigidez elemental K_0 (8×8) para un elemento
    cuadrilateral Q4 de 4 nodos bajo estado plano de tensiones.

    Formulación variacional:
        K_0 = ∫_Ωe  B^T · D · B  dΩ_e

    La integración numérica usa cuadratura de Gauss 2×2.

    Parámetros
    ──────────
    E  : float  Módulo de Young del material sólido (normalizado a 1.0)
    nu : float  Coeficiente de Poisson (típicamente 0.3 para acero)

    Retorna
    ───────
    K0 : ndarray (8, 8)  Matriz de rigidez elemental del material sólido
    """
    # Tensor de elasticidad plana D (estado plano de tensiones)
    prefactor = E / (1.0 - nu**2)
    D = prefactor * np.array([
        [1.0,  nu,           0.0        ],
        [nu,   1.0,          0.0        ],
        [0.0,  0.0, (1.0 - nu) / 2.0   ]
    ])

    # Puntos de Gauss 2×2
    g = 1.0 / np.sqrt(3.0)
    gauss = [(-g, -g), (g, -g), (g, g), (-g, g)]
    w = 1.0

    # Coordenadas nodales del elemento maestro (centrado en origen)
    xn = np.array([-0.5,  0.5,  0.5, -0.5])
    yn = np.array([-0.5, -0.5,  0.5,  0.5])

    K0 = np.zeros((8, 8))

    for (xi, eta) in gauss:
        # Derivadas de funciones de forma N_i(ξ,η)
        dNdxi = 0.25 * np.array([
            -(1 - eta),  (1 - eta),
             (1 + eta), -(1 + eta)
        ])
        dNdeta = 0.25 * np.array([
            -(1 - xi), -(1 + xi),
             (1 + xi),  (1 - xi)
        ])

        # Jacobiano
        J = np.array([
            [dNdxi @ xn,  dNdxi @ yn],
            [dNdeta @ xn, dNdeta @ yn]
        ])
        Ji = np.linalg.inv(J)
        detJ = np.linalg.det(J)

        # Derivadas en coordenadas físicas
        dNdx = Ji[0, 0] * dNdxi + Ji[0, 1] * dNdeta
        dNdy = Ji[1, 0] * dNdxi + Ji[1, 1] * dNdeta

        # Matriz cinemática B (3×8)
        B = np.zeros((3, 8))
        B[0, 0::2] = dNdx
        B[1, 1::2] = dNdy
        B[2, 0::2] = dNdy
        B[2, 1::2] = dNdx

        K0 += w * (B.T @ D @ B) * detJ

    return K0


def ensamblar_K_global(rho, DOFS, nd, K0, p, rho_min=1e-3):
    """
    Ensambla la matriz de rigidez global K(ρ) usando la ley SIMP:
        E_e(ρ_e) = max(ρ_e, ρ_min)^p · E_0

    Implementación vectorizada con scipy.sparse.

    Parámetros
    ──────────
    rho    : ndarray (N_e,)      Densidades actuales
    DOFS   : ndarray (N_e, 8)    DOF globales de cada elemento
    nd     : int                 Número total de DOF
    K0     : ndarray (8, 8)      Matriz de rigidez elemental
    p      : float               Exponente de penalización SIMP
    rho_min: float               Densidad mínima de regularización

    Retorna
    ───────
    K : scipy.sparse.csc_matrix  Matriz de rigidez global (nd × nd)
    """
    Ee = np.maximum(rho, rho_min) ** p
    ii = np.repeat(DOFS, 8, axis=1).reshape(-1)
    jj = np.tile(DOFS, (1, 8)).reshape(-1)
    vv = (Ee[:, None, None] * K0[None, :, :]).reshape(-1)
    K = sp.coo_matrix((vv, (ii, jj)), shape=(nd, nd)).tocsc()
    return K


def resolver_FEM(K, F, fixed, nd):
    """
    Resuelve K·U = F aplicando condiciones de contorno (DOF fijos = 0).

    Usa el método de reducción: se eliminan filas/columnas de los DOF fijos.

    Parámetros
    ──────────
    K     : sparse matrix  Matriz de rigidez global (nd × nd)
    F     : ndarray (nd,)  Vector de fuerzas externas
    fixed : ndarray        Índices de DOF con desplazamiento = 0
    nd    : int            Número total de DOF

    Retorna
    ───────
    U : ndarray (nd,)  Vector de desplazamientos nodales
    """
    free = np.setdiff1d(np.arange(nd), fixed)
    K_free = K[free, :][:, free]
    U = np.zeros(nd)
    U[free] = spla.spsolve(K_free, F[free])
    return U


def calcular_compliance_sensibilidades(U, rho, DOFS, K0, p, rho_min=1e-3):
    """
    Calcula la compliance c(ρ) y las sensibilidades ∂c/∂ρ_e.

    Compliance:    c(ρ) = U^T · K(ρ) · U = Σ_e ρ_e^p · u_e^T · K_0 · u_e
    Sensibilidad:  ∂c/∂ρ_e = -p · ρ_e^{p-1} · u_e^T · K_0 · u_e

    Parámetros
    ──────────
    U     : ndarray (nd,)     Vector de desplazamientos
    rho   : ndarray (N_e,)    Densidades actuales
    DOFS  : ndarray (N_e, 8)  DOF globales de cada elemento
    K0    : ndarray (8, 8)    Matriz de rigidez elemental
    p     : float             Exponente de penalización SIMP
    rho_min: float            Densidad mínima

    Retorna
    ───────
    c  : float          Compliance total c(ρ)
    dc : ndarray (N_e,) Sensibilidades ∂c/∂ρ_e
    """
    Ue = U[DOFS]
    energia_e = (Ue @ K0 * Ue).sum(axis=1)
    re = np.maximum(rho, rho_min)
    c = float(np.sum(re**p * energia_e))
    dc = -p * re**(p - 1) * energia_e
    return c, dc


def filtrar_sensibilidades(dc, rho, H, eps=1e-30):
    """
    Aplica el filtro de sensibilidad por convolución espacial.

    Fórmula (Sigmund, 2007):
        d̂c_e = (Σ_f ĥ_ef · ρ_f · dc_f) / (ρ_e · Σ_f ĥ_ef)

    Parámetros
    ──────────
    dc  : ndarray (N_e,)    Sensibilidades sin filtrar
    rho : ndarray (N_e,)    Densidades actuales
    H   : ndarray (N_e,N_e) Matriz de pesos del filtro
    eps : float             Regularización del denominador

    Retorna
    ───────
    dc_filt : ndarray (N_e,)  Sensibilidades filtradas
    """
    numerador   = H @ (rho * dc)
    denominador = rho * H.sum(axis=1) + eps
    return numerador / denominador


def actualizar_OC(rho, dc_filt, f_V, move=0.2, eta=0.5, rho_min=1e-3):
    """
    Actualiza las densidades por el Criterio de Optimalidad (OC).

    ρ_e^{nuevo} = ρ_e · (−∂c/∂ρ_e / λ)^η
    con λ determinado por bisección para satisfacer la restricción de volumen.

    Parámetros
    ──────────
    rho     : ndarray (N_e,)  Densidades actuales
    dc_filt : ndarray (N_e,)  Sensibilidades filtradas
    f_V     : float           Fracción de volumen objetivo
    move    : float           Límite de movimiento por iteración
    eta     : float           Amortiguamiento
    rho_min : float           Densidad mínima

    Retorna
    ───────
    rho_nuevo : ndarray (N_e,)  Densidades actualizadas
    """
    l_lo, l_hi = 1e-9, 1e9
    for _ in range(60):
        lm = 0.5 * (l_lo + l_hi)
        rho_prop = rho * ((-dc_filt) / lm) ** eta
        rho_nuevo = np.clip(
            rho_prop,
            np.maximum(rho_min, rho - move),
            np.minimum(1.0, rho + move)
        )
        if rho_nuevo.mean() > f_V:
            l_lo = lm
        else:
            l_hi = lm
    return rho_nuevo
