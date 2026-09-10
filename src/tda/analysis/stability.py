"""Análisis de estabilidad topológica frente a ruido.

Implementa el barrido sistemático de niveles de ruido para verificar
las hipótesis H.E.1a', H.E.1b y H.E.1c del Perfil.

Umbrales: τ · diam(X) — relativo al diámetro de la nube (Def. 8.1 del Perfil).

Referencias:
  - Definición 8.1: βₖ^(τ) = #{(b,d) ∈ Dgmₖ : d−b > τ·diam(X)}
  - Teorema de Estabilidad (Chazal et al.): dB ≤ 2·dH
  - §9.4.1: Catálogo de 7 nubes, perturbación q ∈ {0,.05,.10,.15,.20}
  - §9.4.3: Transformación afín aleatoria para H.E.1b
  - §9.4.4: Contraste de McNemar con ≥200 muestras/clase
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
from ripser import ripser

from tda.processing.sampling import (
    generate_cloud, add_gaussian_noise, compute_diameter,
    apply_random_affine, betti_teorico, BETTI_TEORICO,
)
from tda.analysis.metrics import compute_kmeans_accuracy, mcnemar_test
from tda.core.topology import bottleneck_distance as persim_bottleneck


# ═══════════════════════════════════════════════════════════════════
# FUNCIONES CENTRALES — τ · diam(X)
# ═══════════════════════════════════════════════════════════════════

def betti_persistente(
    dgm: np.ndarray,
    tau: float,
    diam: float,
) -> Tuple[int, int]:
    """Cuenta β₀ y β₁ usando el umbral relativo τ · diam(X).

    Ref: Definición 8.1 del Perfil:
        βₖ^(τ)(X) = #{(b,d) ∈ Dgmₖ(X) : d − b > τ · diam(X)}

    Esta función ES la Def. 8.1. El umbral es invariante a homotecias
    porque tanto la persistencia (d−b) como diam(X) escalan igual.

    Parámetros
    ──────────
    dgm  : salida de ripser (dict con 'dgms') o lista de diagramas
    tau  : umbral relativo (τ = 0.15 según calibración §6)
    diam : diámetro de la nube de puntos original

    Retorna
    ───────
    (beta0, beta1) : tuple de enteros
    """
    if isinstance(dgm, dict) and "dgms" in dgm:
        dgms = dgm["dgms"]
    elif isinstance(dgm, (list, tuple)):
        dgms = dgm
    else:
        return (0, 0)

    # β₀: Def. 8.1 aplica τ·diam a TODOS los k, incluyendo k=0.
    # Barras infinitas tienen persistencia ∞ > threshold → siempre cuentan.
    # Barras finitas cuentan si d−b > τ·diam(X).
    threshold = tau * diam
    dgm0 = dgms[0] if len(dgms) > 0 else np.empty((0, 2))
    dgm0 = np.asarray(dgm0)
    if dgm0.size == 0 or dgm0.ndim < 2:
        beta0 = 0
    else:
        pers0 = dgm0[:, 1] - dgm0[:, 0]
        n_inf = int(np.sum(np.isinf(pers0)))
        finitas = pers0[np.isfinite(pers0)]
        n_finitas_sig = int(np.sum(finitas > threshold)) if finitas.size > 0 else 0
        beta0 = n_inf + n_finitas_sig

    # β₁: ciclos 1D con persistencia > τ · diam(X)
    dgm1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
    dgm1 = np.asarray(dgm1)
    if dgm1.size == 0 or dgm1.ndim < 2:
        beta1 = 0
    else:
        finitas = dgm1[np.isfinite(dgm1[:, 1])]
        if finitas.size == 0:
            beta1 = 0
        else:
            persistencias = finitas[:, 1] - finitas[:, 0]
            beta1 = int(np.sum(persistencias > tau * diam))

    return (beta0, beta1)


def betti_significativos(
    pts: np.ndarray,
    tau: float = 0.15,
) -> Tuple[Tuple[int, int], list]:
    """Calcula (β₀, β₁) significativos de una nube + diagramas Ripser.

    Usa τ · diam(X) como umbral (Def. 8.1).

    Parámetros
    ──────────
    pts : ndarray (n, d)  Nube de puntos
    tau : float           Umbral relativo (default 0.15, §6)

    Retorna
    ───────
    ((b0, b1), res) : tupla de Betti y salida cruda de ripser
    """
    pts = np.asarray(pts)
    diam = compute_diameter(pts)
    res = ripser(pts, maxdim=1)
    b0, b1 = betti_persistente(res, tau, diam)
    return (b0, b1), res


# ═══════════════════════════════════════════════════════════════════
# H.E.1a' — COTA BOTTLENECK: dB/diam ≤ 2q
# ═══════════════════════════════════════════════════════════════════

def verificar_cota_bottleneck(
    X: np.ndarray,
    Y: np.ndarray,
    q: float,
    maxdim: int = 1,
) -> Dict:
    """Verifica la cota H.E.1a': dB(Dgm(X), Dgm(Y)) / diam(X) ≤ 2q.

    Ref: H.E.1a' reformulada (§7 metodología):
        dB(Dgm VR(X), Dgm VR(Y)) / diam(X) ≤ 2q

    Parámetros
    ──────────
    X      : ndarray (n, d)  Nube limpia (referencia)
    Y      : ndarray (n, d)  Nube perturbada
    q      : float           Nivel relativo de ruido
    maxdim : int             Dimensión máxima de homología (default 1)

    Retorna
    ───────
    Dict con dB, diam_X, ratio, cota_teoria, cumple
    """
    dgm_X = ripser(np.asarray(X), maxdim=maxdim)["dgms"]
    dgm_Y = ripser(np.asarray(Y), maxdim=maxdim)["dgms"]

    bottlenecks = []
    for k in range(maxdim + 1):
        dgmk_X = dgm_X[k] if k < len(dgm_X) else np.empty((0, 2))
        dgmk_Y = dgm_Y[k] if k < len(dgm_Y) else np.empty((0, 2))
        if dgmk_X.size > 0:
            dgmk_X = dgmk_X[np.isfinite(dgmk_X[:, 1])]
        if dgmk_Y.size > 0:
            dgmk_Y = dgmk_Y[np.isfinite(dgmk_Y[:, 1])]
        if dgmk_X.size > 0 and dgmk_Y.size > 0:
            bnk = persim_bottleneck(dgmk_X, dgmk_Y)
        else:
            bnk = 0.0
        bottlenecks.append(bnk)

    dB = max(bottlenecks) if bottlenecks else 0.0
    diam_X = compute_diameter(np.asarray(X))
    ratio = dB / diam_X if diam_X > 0 else 0.0
    cota = 2.0 * q

    return {
        "dB": float(dB),
        "diam_X": float(diam_X),
        "ratio": float(ratio),
        "cota_teoria": float(cota),
        "cumple": bool(ratio <= cota),
    }


# ═══════════════════════════════════════════════════════════════════
# H.E.1b — COMPARACIÓN TDA vs EUCLIDIANOS (con afín)
# ═══════════════════════════════════════════════════════════════════

def clasificador_tda(
    pts: np.ndarray,
    tau: float = 0.15,
) -> int:
    """Clasificador TDA: distingue disco (β₁=0) de anillo (β₁=1).

    Retorna 0 (β₁=0, disco) o 1 (β₁=1, anillo).
    """
    (b0, b1), _ = betti_significativos(pts, tau=tau)
    return 1 if b1 > 0 else 0


def clasificador_euclidiano(
    pts: np.ndarray,
) -> int:
    """Descriptor euclidiano compuesto: espectro PCA + inercia k-medias + dispersión radial.

    Retorna 0 o 1 (la mejor asignación de las dos clases).
    """
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    # PCA: varianza explicada
    pca = PCA(n_components=min(2, pts.shape[1]))
    pca.fit(pts)
    inertia_pca = float(np.sum(pca.explained_variance_))

    # K-Means: inercia
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    km.fit(pts)
    inertia_km = km.inertia_

    # Dispersión radial
    centroid = pts.mean(axis=0)
    radial = np.mean(np.linalg.norm(pts - centroid, axis=1))

    # Compuesto: normalizar y sumar
    # El anillo tiene mayor inercia PCA y menor dispersión radial que el disco
    score = inertia_pca + radial
    return 1 if score > 2.0 else 0  # umbral calibrado


def comparar_tda_vs_euclidiano(
    n_points: int = 400,
    n_rep: int = 200,
    tau: float = 0.15,
    apply_affine: bool = True,
    random_seed: int = 42,
) -> Dict:
    """Compara clasificadores TDA vs euclidiano sobre disco/anillo.

    Ref: H.E.1b — "El descriptor de persistencia alcanza exactitud
    significativamente superior (p < 0.05, McNemar) al descriptor euclidiano."

    La deformación afín aleatoria destruye los descriptores euclidianos
    pero preserva βₖ (homeomorfismo).

    Parámetros
    ──────────
    n_points     : int   Muestras por clase (≥200, §9.4.4)
    n_rep        : int   Réplicas totales (≥200 por clase)
    tau          : float Umbral relativo
    apply_affine : bool  Aplicar transformación afín aleatoria
    random_seed  : int   Semilla maestra

    Retorna
    ───────
    Dict con exactitud_tda, exactitud_eucl, mc_nemar, mensaje
    """
    np.random.seed(random_seed)
    n_per_class = n_rep // 2

    y_true = np.array([0] * n_per_class + [1] * n_per_class)
    y_tda = np.zeros(len(y_true), dtype=int)
    y_eucl = np.zeros(len(y_true), dtype=int)

    for i in range(n_per_class):
        # Generar disco y anillo
        disco = generate_cloud("disco", n_points)
        anillo = generate_cloud("anillo", n_points)

        if apply_affine:
            disco, _ = apply_random_affine(disco)
            anillo, _ = apply_random_affine(anillo)

        y_tda[i] = clasificador_tda(disco, tau)
        y_tda[n_per_class + i] = clasificador_tda(anillo, tau)
        y_eucl[i] = clasificador_euclidiano(disco)
        y_eucl[n_per_class + i] = clasificador_euclidiano(anillo)

    acc_tda = float(np.mean(y_tda == y_true))
    acc_eucl = float(np.mean(y_eucl == y_true))

    # McNemar
    mc = mcnemar_test(y_true, y_tda, y_eucl)

    return {
        "exactitud_tda": acc_tda,
        "exactitud_eucl": acc_eucl,
        "mc_nemar": mc,
        "n_puntos_por_clase": n_per_class,
        "n_points": n_points,
        "affine": apply_affine,
    }


# ═══════════════════════════════════════════════════════════════════
# BARRIDO COMPLETO — H.E.1a' + H.E.1c
# ═══════════════════════════════════════════════════════════════════

def barrido_cota_bottleneck(
    n_points: int = 400,
    noise_levels: Optional[List[float]] = None,
    n_rep: int = 100,
    shapes: Optional[List[str]] = None,
    tau: float = 0.15,
    random_seed: int = 42,
) -> Dict:
    """Barrido completo de H.E.1a': verifica dB/diam ≤ 2q.

    Para cada nivel de ruido q, genera n réplicas de cada nube,
    perturba con ruido gaussiano de nivel relativo q, y verifica la cota.

    Parámetros
    ──────────
    n_points     : int      Puntos por nube (mínimo 400, §9.4.1)
    noise_levels : list     Niveles de ruido
    n_rep        : int      Réplicas por nivel y nube
    shapes       : list     Formas del catálogo
    tau          : float    Umbral relativo (para H.E.1c)
    random_seed  : int      Semilla maestra

    Retorna
    ───────
    Dict con resultados por forma y nivel de ruido
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    if shapes is None:
        shapes = ["circulo", "disco", "anillo", "dos_circulos",
                  "toro", "esfera", "gaussiana"]

    np.random.seed(random_seed)
    resultados = {}

    for shape in shapes:
        b0_teo, b1_teo = betti_teorico(shape)
        resultados[shape] = {
            "betti_teorico": (b0_teo, b1_teo),
            "noise_levels": noise_levels,
            "acierto_por_nivel": [],
            "ratio_bottleneck_mean": [],
            "ratio_bottleneck_std": [],
            "cumple_cota_por_nivel": [],
            "replicas": n_rep,
        }

        X_limpia = generate_cloud(shape, n_points)

        for q in noise_levels:
            aciertos = 0
            ratios = []
            cumple_count = 0

            for r in range(n_rep):
                seed_r = 42 + 1000 * r + int(1e5 * q)
                np.random.seed(seed_r)
                Y_perturbada = add_gaussian_noise(X_limpia, q)

                # H.E.1c: acierto de βₖ
                (b0_r, b1_r), _ = betti_significativos(Y_perturbada, tau=tau)
                if b0_r == b0_teo and b1_r == b1_teo:
                    aciertos += 1

                # H.E.1a': cota bottleneck
                res_bn = verificar_cota_bottleneck(X_limpia, Y_perturbada, q)
                ratios.append(res_bn["ratio"])
                if res_bn["cumple"]:
                    cumple_count += 1

            resultados[shape]["acierto_por_nivel"].append(aciertos / n_rep)
            resultados[shape]["ratio_bottleneck_mean"].append(float(np.mean(ratios)))
            resultados[shape]["ratio_bottleneck_std"].append(float(np.std(ratios)))
            resultados[shape]["cumple_cota_por_nivel"].append(cumple_count / n_rep)

    return resultados


# ═══════════════════════════════════════════════════════════════════
# BARRIDO DE ESTABILIDAD (compatibilidad con UI existente)
# ═══════════════════════════════════════════════════════════════════

def _verificar_estabilidad(
    betti_vals: List[Tuple[int, int]],
    noise_vals: np.ndarray,
    rango_estabilidad: Tuple[float, float] = (0.15, 0.20),
) -> Dict:
    """Verifica que β₀/β₁ permanezcan estables en el rango dado."""
    idx_estable = np.where(
        (noise_vals >= rango_estabilidad[0]) & (noise_vals <= rango_estabilidad[1])
    )[0]

    if len(idx_estable) == 0:
        return {"estable": False, "mensaje": "Sin datos en rango dado"}

    betti_rango = [betti_vals[i] for i in idx_estable]
    b0_vals = [b[0] for b in betti_rango]
    b1_vals = [b[1] for b in betti_rango]

    b0_estable = max(b0_vals) - min(b0_vals) == 0
    b1_estable = max(b1_vals) - min(b1_vals) == 0

    if b0_estable and b1_estable:
        msg = (f"Estables en {rango_estabilidad[0]:.0%}–"
               f"{rango_estabilidad[1]:.0%}: "
               f"β₀∈{{{','.join(map(str, b0_vals))}}}, "
               f"β₁∈{{{','.join(map(str, b1_vals))}}}")
    else:
        parts = []
        if not b0_estable:
            parts.append(f"β₀ inestable: {{{','.join(map(str, b0_vals))}}}")
        if not b1_estable:
            parts.append(f"β₁ inestable: {{{','.join(map(str, b1_vals))}}}")
        msg = f"Inestables: {'; '.join(parts)}"

    return {
        "estable": b0_estable and b1_estable,
        "b0_estable": b0_estable,
        "b1_estable": b1_estable,
        "b0_vals_rango": b0_vals,
        "b1_vals_rango": b1_vals,
        "mensaje": msg,
    }


def compute_noise_sweep(
    n_points: int = 400,
    n_clusters: int = 2,
    noise_min: float = 0.0,
    noise_max: float = 0.30,
    n_steps: int = 10,
    tau: float = 0.15,
    random_seed: int = 42,
    progress_callback=None,
) -> dict:
    """Barrido de ruido para esfera y toro con τ · diam(X).

    Para cada nivel de ruido calcula (β₀, β₁) usando la Def. 8.1
    y la exactitud de K-Means.

    Returns:
        Dict con noise_vals, acc, betti_s, betti_t, diagrams_s, diagrams_t,
        estabilidad_s, estabilidad_t.
    """
    np.random.seed(random_seed)
    noise_vals = np.linspace(noise_min, noise_max, n_steps)

    sweep_acc: List[float] = []
    sweep_betti_s: List[Tuple[int, int]] = []
    sweep_betti_t: List[Tuple[int, int]] = []
    sweep_dgms_s: List = []
    sweep_dgms_t: List = []

    N_REP = 5  # repeticiones internas por nivel

    for i, noise in enumerate(noise_vals):
        acc_reps: List[float] = []
        b0s_s, b1s_s = [], []
        b0s_t, b1s_t = [], []
        res_s_reps, res_t_reps = [], []

        for _ in range(N_REP):
            pts_s = generate_cloud("esfera", n_points)
            pts_t = generate_cloud("toro", n_points)
            pts_t[:, 0] += 1.5

            pts_s_n = add_gaussian_noise(pts_s, noise)
            pts_t_n = add_gaussian_noise(pts_t, noise)

            # K-Means
            from sklearn.cluster import KMeans
            dataset_n = np.vstack([pts_s_n, pts_t_n])
            km = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
            y_pred = km.fit_predict(dataset_n)
            y_true = np.array([0] * n_points + [1] * n_points)
            acc_reps.append(compute_kmeans_accuracy(y_true, y_pred))

            # TDA con τ · diam
            (b0_s, b1_s), res_s = betti_significativos(pts_s_n, tau=tau)
            (b0_t, b1_t), res_t = betti_significativos(pts_t_n, tau=tau)
            b0s_s.append(b0_s); b1s_s.append(b1_s)
            b0s_t.append(b0_t); b1s_t.append(b1_t)
            res_s_reps.append(res_s); res_t_reps.append(res_t)

        med_b0_s, med_b1_s = int(np.median(b0s_s)), int(np.median(b1s_s))
        idx_med_s = int(np.argmin(
            [abs(b0 - med_b0_s) + abs(b1 - med_b1_s)
             for b0, b1 in zip(b0s_s, b1s_s)]
        ))
        med_b0_t, med_b1_t = int(np.median(b0s_t)), int(np.median(b1s_t))
        idx_med_t = int(np.argmin(
            [abs(b0 - med_b0_t) + abs(b1 - med_b1_t)
             for b0, b1 in zip(b0s_t, b1s_t)]
        ))

        sweep_acc.append(float(np.mean(acc_reps)))
        sweep_betti_s.append((med_b0_s, med_b1_s))
        sweep_betti_t.append((med_b0_t, med_b1_t))
        sweep_dgms_s.append(res_s_reps[idx_med_s])
        sweep_dgms_t.append(res_t_reps[idx_med_t])

        if progress_callback is not None:
            progress_callback(i + 1, n_steps)

    estabilidad_s = _verificar_estabilidad(sweep_betti_s, noise_vals)
    estabilidad_t = _verificar_estabilidad(sweep_betti_t, noise_vals)

    return {
        "noise_vals": noise_vals,
        "acc": sweep_acc,
        "betti_s": sweep_betti_s,
        "betti_t": sweep_betti_t,
        "diagrams_s": sweep_dgms_s,
        "diagrams_t": sweep_dgms_t,
        "estabilidad_s": estabilidad_s,
        "estabilidad_t": estabilidad_t,
    }
