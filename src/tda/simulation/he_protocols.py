"""Protocolos de simulación de las hipótesis específicas (H.E.1 y H.E.2).

Extrae la lógica de cómputo de las páginas de la app para que pueda reutilizarse
(la página H.G. ejecuta ambos protocolos para sintetizar).

Constantes del documento:
  - H.E.1: catálogo de 7 nubes, q ∈ {0, .05, .10, .15, .20}, σ = q·diam(X),
    R réplicas, semilla 42, τ = 0.15.
  - H.E.2: viga en voladizo; fV = 0.5, p = 3, rmin = 2.4; p ∈ {1,2,3,4},
    rmin ∈ [1.5, 4.0]; elementos unitarios (E₀ = 1, F = 1), filtro de densidad.
"""

import numpy as np
from ripser import ripser

from tda.processing.sampling import (
    generate_cloud, add_gaussian_noise, compute_diameter, betti_teorico,
)
from tda.analysis.stability import betti_persistente, hausdorff_distance
from tda.core.topology import bottleneck_distance
from tda.optimization.metric_simp import MetricaTDA_SIMP

CATALOGO = ["circulo", "disco", "anillo", "dos_circulos", "toro", "esfera", "gaussiana"]
SUBCONJUNTO_VALIDADO = ["circulo", "disco", "anillo"]   # Tabla 5.5, nota (a)
Q_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20]
TAU_CALIBRADO = 0.15
SEED = 42
FV = 0.5
P_DEFECTO = 3
RMIN_DEFECTO = 2.4
RMIN_RANGO = [1.5, 2.0, 2.4, 3.0, 4.0]
P_RANGO = [1, 2, 3, 4]


def bottleneck_cota(dgm_X, dgm_Y, maxdim=1):
    """dB(Dgm_k^VR(X), Dgm_k^VR(Y)) = máximo sobre k (Algoritmo maestro)."""
    bottlenecks = []
    for k in range(maxdim + 1):
        dkX = dgm_X[k] if k < len(dgm_X) else np.empty((0, 2))
        dkY = dgm_Y[k] if k < len(dgm_Y) else np.empty((0, 2))
        if dkX.size > 0:
            dkX = dkX[np.isfinite(dkX[:, 1])]
        if dkY.size > 0:
            dkY = dkY[np.isfinite(dkY[:, 1])]
        bottlenecks.append(bottleneck_distance(dkX, dkY) if (dkX.size > 0 and dkY.size > 0) else 0.0)
    return float(max(bottlenecks)) if bottlenecks else 0.0


def protocolo_he1(n_points, q_levels, n_rep, tau=TAU_CALIBRADO, seed=SEED):
    """Recorre el catálogo (H.E.1a y H.E.1c) siguiendo el Algoritmo maestro.

    Devuelve (resultados, crudo): por nube, acierto βk (Fig. 5.2(a)), acierto β₁
    (criterio q*, H.E.1c), razón bottleneck media/std (dB/dH), razón dB/diam y
    fracción que cumple dB ≤ 2dH; y las réplicas crudas (dB, dH) para la Fig. 5.2(b).
    """
    resultados, crudo = {}, []
    for shape in CATALOGO:
        np.random.seed(seed)          # semilla: catálogo reproducible (Algoritmo maestro)
        X0 = generate_cloud(shape, n_points)
        b0_teo, b1_teo = betti_teorico(shape)
        diam = compute_diameter(X0)
        dgm_X = ripser(X0, maxdim=1)["dgms"]
        resultados[shape] = {
            "betti_teorico": (b0_teo, b1_teo), "diam": diam,
            "acierto": [], "acierto_b1": [],
            "ratio_medio": [], "ratio_diam_medio": [], "ratio_std": [], "cumple_frac": [],
        }
        for q in q_levels:
            aciertos, aciertos1, rdH, rDiam, cumple = 0, 0, [], [], 0
            for r in range(n_rep):
                np.random.seed(seed + 1000 * r + int(1e5 * q))       # Algoritmo maestro
                Y = add_gaussian_noise(X0, q)
                dgm_Y = ripser(Y, maxdim=1)["dgms"]
                b0, b1 = betti_persistente(dgm_Y, tau, diam)
                if b0 == b0_teo and b1 == b1_teo:   # βk = (β₀, β₁)  → Fig. 5.2(a)
                    aciertos += 1
                if b1 == b1_teo:                    # β₁ (invariante que τ filtra) → q* (H.E.1c)
                    aciertos1 += 1
                dB = bottleneck_cota(dgm_X, dgm_Y)
                dH = hausdorff_distance(X0, Y)
                rdH.append(dB / dH if dH > 0 else 0.0)
                rDiam.append(dB / diam if diam > 0 else 0.0)
                if dB <= 2 * dH:
                    cumple += 1
                crudo.append({"shape": shape, "q": q, "dB": dB, "dH": dH, "diam": diam,
                              "dB_diam": dB / diam if diam > 0 else 0.0})
            resultados[shape]["acierto"].append(aciertos / n_rep)
            resultados[shape]["acierto_b1"].append(aciertos1 / n_rep)
            resultados[shape]["ratio_medio"].append(float(np.mean(rdH)))
            resultados[shape]["ratio_diam_medio"].append(float(np.mean(rDiam)))
            resultados[shape]["ratio_std"].append(float(np.std(rdH)))
            resultados[shape]["cumple_frac"].append(cumple / n_rep)
    return resultados, crudo


def calibrar_tau(n_points, seed=SEED, n_tau=20):
    """Curva de calibración de τ contra el patrón de oro (Fig. 5.1)."""
    np.random.seed(seed)
    dgms = {}
    for shape in CATALOGO:
        X = generate_cloud(shape, n_points)
        dgms[shape] = (ripser(X, maxdim=1), compute_diameter(X))
    taus = np.linspace(0.01, 0.40, n_tau)
    aciertos = []
    for tau in taus:
        ok = 0
        for shape in CATALOGO:
            dgm, diam = dgms[shape]
            _, b1 = betti_persistente(dgm, tau, diam)   # lectura β₁ (invariante de τ)
            if b1 == betti_teorico(shape)[1]:
                ok += 1
        aciertos.append(ok)
    return taus, aciertos


def corrida_simp(nelx, nely, p, r_min, E=1.0, Fmag=1.0, fV=FV, t=1.0,
                 max_iter=200, Lx=None, Ly=None):
    """Corre SIMP+TDA para una configuración (elementos unitarios por defecto)."""
    nnx = nelx + 1
    n_dof = 2 * nnx * (nely + 1)
    idx_izq = np.arange(0, nely + 1) * nnx
    dofs = np.sort(np.concatenate([2 * idx_izq, 2 * idx_izq + 1]))
    node = (nely // 2) * nnx + nelx
    F = np.zeros(n_dof)
    F[2 * node + 1] = -Fmag
    m = MetricaTDA_SIMP(nex=nelx, ney=nely, E=E, nu=0.3,
                        Lx=float(Lx if Lx is not None else nelx),
                        Ly=float(Ly if Ly is not None else nely),
                        t=t, f_V=fV, p=p, r_min=r_min, max_iter=max_iter)
    m.definir_problema(F, dofs)
    m.optimizar(verbose=False)
    m.fase_tda(verbose=False)
    return m


def esfuerzo_max_von_mises(m, E, nu=0.3):
    """Máximo esfuerzo de von Mises del diseño de `m` (densidad filtrada final).

    Reconstruye K con ρ̃ final, resuelve el FEM y evalúa σ en el centro de cada
    elemento (tensión plana). Devuelve el máximo en las mismas unidades que E.
    """
    from tda.core.fem import ensamblar_K_global, resolver_FEM

    rho = m.rho_tilde_final
    p = getattr(m, "p", P_DEFECTO)
    K = ensamblar_K_global(rho, m.DOFS, m.n_dof, m.K0, p)
    U = resolver_FEM(K, m.F, m.fixed, m.n_dof)

    D = (E / (1 - nu ** 2)) * np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, (1 - nu) / 2.0]])
    dx, dy = m.dx, m.dy
    xn = np.array([-dx / 2, dx / 2, dx / 2, -dx / 2])
    yn = np.array([-dy / 2, -dy / 2, dy / 2, dy / 2])
    xi = eta = 0.0
    dNdxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    dNdeta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    J = np.array([[dNdxi @ xn, dNdxi @ yn], [dNdeta @ xn, dNdeta @ yn]])
    Ji = np.linalg.inv(J)
    dNdx = Ji[0, 0] * dNdxi + Ji[0, 1] * dNdeta
    dNdy = Ji[1, 0] * dNdxi + Ji[1, 1] * dNdeta
    B = np.zeros((3, 8))
    for i in range(4):
        B[0, 2 * i] = dNdx[i]
        B[1, 2 * i + 1] = dNdy[i]
        B[2, 2 * i] = dNdy[i]
        B[2, 2 * i + 1] = dNdx[i]

    Ue = U[m.DOFS]                 # (N_e, 8)
    eps = (B @ Ue.T).T             # (N_e, 3)
    sig = (D @ eps.T).T            # (N_e, 3)
    sxx, syy, txy = sig[:, 0], sig[:, 1], sig[:, 2]
    vm = np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3 * txy ** 2)
    return float(np.max(vm))


def voxeles_a_mesh3d(grid, t=1.0):
    """Convierte una rejilla binaria (ny, nx) en una malla 3D cerrada.

    Genera solo las caras de frontera (superior, inferior y laterales donde hay
    vacío). Devuelve (V, F): vértices (n,3) y triángulos (m,3) para Plotly Mesh3d
    o para exportar a STL.
    """
    grid = np.asarray(grid, dtype=bool)
    ny, nx = grid.shape
    verts, index, faces = [], {}, []

    def vid(x, y, z):
        key = (x, y, z)
        if key not in index:
            index[key] = len(verts)
            verts.append((float(x), float(y), float(z)))
        return index[key]

    def quad(a, b, c, d):
        faces.append((a, b, c))
        faces.append((a, c, d))

    for j in range(ny):
        for i in range(nx):
            if not grid[j, i]:
                continue
            x0, x1, y0, y1, z0, z1 = i, i + 1, j, j + 1, 0.0, float(t)
            quad(vid(x0, y0, z1), vid(x1, y0, z1), vid(x1, y1, z1), vid(x0, y1, z1))  # top
            quad(vid(x0, y0, z0), vid(x1, y0, z0), vid(x1, y1, z0), vid(x0, y1, z0))  # bottom
            if i == 0 or not grid[j, i - 1]:
                quad(vid(x0, y0, z0), vid(x0, y1, z0), vid(x0, y1, z1), vid(x0, y0, z1))
            if i == nx - 1 or not grid[j, i + 1]:
                quad(vid(x1, y0, z0), vid(x1, y1, z0), vid(x1, y1, z1), vid(x1, y0, z1))
            if j == 0 or not grid[j - 1, i]:
                quad(vid(x0, y0, z0), vid(x1, y0, z0), vid(x1, y0, z1), vid(x0, y0, z1))
            if j == ny - 1 or not grid[j + 1, i]:
                quad(vid(x0, y1, z0), vid(x1, y1, z0), vid(x1, y1, z1), vid(x0, y1, z1))

    return np.array(verts, dtype=float), np.array(faces, dtype=int)


def stl_ascii_bytes(V, F, nombre="diseno_simp"):
    """Serializa la malla (V, F) a STL ASCII (para impresión 3D / CAD)."""
    out = [f"solid {nombre}"]
    for a, b, c in F:
        p0, p1, p2 = V[a], V[b], V[c]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        n = n / nn if nn > 1e-15 else np.array([0.0, 0.0, 1.0])
        out.append(f"facet normal {n[0]:.5e} {n[1]:.5e} {n[2]:.5e}")
        out.append("  outer loop")
        for p in (p0, p1, p2):
            out.append(f"    vertex {p[0]:.5e} {p[1]:.5e} {p[2]:.5e}")
        out.append("  endloop")
        out.append("endfacet")
    out.append(f"endsolid {nombre}")
    return ("\n".join(out)).encode("utf-8")


def persistencia_beta1(umbrales, b1s, beta1):
    """Mayor intervalo CONTIGUO de umbrales donde β₁ = beta1.

    Devuelve (u_lo, u_hi, persistencia) — la persistencia en unidades de umbral
    del rasgo (tesis: ≈ 0.80 para los tres agujeros).
    """
    runs, start = [], None
    for i, b in enumerate(b1s):
        if b == beta1:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(b1s) - 1))
    if not runs:
        return None, None, 0.0
    a, b_ = max(runs, key=lambda r: r[1] - r[0])
    return float(umbrales[a]), float(umbrales[b_]), float(umbrales[b_] - umbrales[a])


def barrido_he2(dominio, nex, ney, penal, rmin, E=1.0, Fmag=1.0, fV=FV, t=1.0,
                max_iter=200, progress_cb=None):
    """Barrido multi-configuración de H.E.2b (malla con rmin escalado, p, rmin).

    Devuelve una lista de dicts con las métricas por configuración.
    """
    configs = []
    for r in RMIN_RANGO:
        configs.append({"label": f"rmin={r}", "nelx": nex, "nely": ney, "p": int(penal), "rmin": r})
    for p_val in P_RANGO:
        configs.append({"label": f"p={p_val}", "nelx": nex, "nely": ney, "p": p_val, "rmin": rmin})
    # El estudio de independencia de malla (Fig. 5.6) usa el dominio CUADRADO 60×60
    # con Ne ∈ {1600, 6400, 14400} y rmin = 2.4·(nx/60) ∈ {1.6, 3.2, 4.8}, según el documento.
    meshes_sweep, ref_nelx, lx_s, ly_s = [(40, 40), (80, 80), (120, 120)], 60, 60.0, 60.0
    for nelx_s, nely_s in meshes_sweep:
        configs.append({"label": f"malla={nelx_s}x{nely_s}", "nelx": nelx_s, "nely": nely_s,
                        "p": int(penal), "rmin": rmin * (nelx_s / ref_nelx),
                        "lx": lx_s, "ly": ly_s})

    resultados = []
    for i, cfg in enumerate(configs):
        if progress_cb:
            progress_cb(i, len(configs), cfg["label"])
        m_s = corrida_simp(cfg["nelx"], cfg["nely"], cfg["p"], cfg["rmin"], E, Fmag, fV, t,
                           max_iter, Lx=cfg.get("lx"), Ly=cfg.get("ly"))
        conc_s = getattr(m_s, "_betti_concordancia", None)
        resultados.append({
            "Config": cfg["label"], "Malla": f"{cfg['nelx']}x{cfg['nely']}",
            "Ne": cfg["nelx"] * cfg["nely"], "rmin": round(cfg["rmin"], 2), "p": cfg["p"],
            "c(rho*)": round(m_s.c_final, 4), "beta0": m_s.beta0, "beta1": m_s.beta1,
            "grisura": round(float(np.mean(4.0 * m_s.rho_tilde_final * (1.0 - m_s.rho_tilde_final))), 4),
            "Concordancia": "Si" if conc_s else ("No" if conc_s is not None else "N/A"),
        })
    return resultados
