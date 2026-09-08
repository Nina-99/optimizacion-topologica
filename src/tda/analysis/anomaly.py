"""Detección de anomalías topológicas usando números de Betti.

Implementa la validación de la Hipótesis H.E.1:
"La homología persistente... supera a los descriptores euclidianos
en tareas de clasificación y **detección de anomalías**."

Referencias:
- Definición 1.4: Número de Betti
- Sección 8.6.1: Diseño de casos de simulación (esfera/toro)
"""

from typing import Dict, List, Tuple
import numpy as np
from ripser import ripser

from tda.processing.sampling import generate_cloud, add_gaussian_noise
from tda.core.topology import escala_adaptativa


def _betti_filtrado(diagram: np.ndarray, eps_star: float) -> Tuple[int, int]:
    """Filtra puntos del diagrama por persistencia > ε*/2."""
    umbral = eps_star / 2.0
    if diagram.ndim == 2 and diagram.shape[1] == 3:
        h0 = diagram[diagram[:, 2] == 0]
        h1 = diagram[diagram[:, 2] == 1]
        pers_h0 = h0[:, 1] - h0[:, 0]
        pers_h1 = h1[:, 1] - h1[:, 0]
        return (int(np.sum(pers_h0 > umbral)), int(np.sum(pers_h1 > umbral)))
    return (0, 0)


def detect_anomaly_tda(
    pts: np.ndarray,
    expected_betti: Tuple[int, int],
    shape_name: str = "forma",
    n_points: int = None,
) -> Dict:
    """Detecta anomalías en una nube de puntos usando Betti filtrado.

    Una anomalía se define como:
    - β₀ ≠ 1: la forma se desconectó (más de una componente conexa)
    - β₁ ≠ esperado: aparecieron o desaparecieron agujeros

    Args:
        pts: Nube de puntos (n, 3).
        expected_betti: (β₀ esperado, β₁ esperado).
        shape_name: Nombre de la forma para el reporte.

    Returns:
        Dict con is_anomaly, b0, b1, expected, details.
    """
    b0_exp, b1_exp = expected_betti

    res = ripser(pts, maxdim=1)['dgms']
    eps_star = escala_adaptativa(pts, pts.shape[0])

    if len(res) > 1 and res[1].ndim == 2 and res[1].shape[1] == 3:
        b0, b1 = _betti_filtrado(res[1], eps_star)
    else:
        dgm_all = np.vstack([res[d] for d in range(len(res)) if res[d].size > 0])
        if dgm_all.ndim == 2 and dgm_all.shape[1] == 3:
            b0, b1 = _betti_filtrado(dgm_all, eps_star)
        else:
            b0, b1 = (len(res[0]), len(res[1]) if len(res) > 1 else 0)

    is_anomaly = (b0 != b0_exp) or (b1 != b1_exp)

    details = []
    if b0 != b0_exp:
        details.append(f"β₀={b0} (esperado {b0_exp}): forma desconectada")
    if b1 != b1_exp:
        if b1 > b1_exp:
            details.append(f"β₁={b1} (esperado {b1_exp}): agujero espurio detectado")
        else:
            details.append(f"β₁={b1} (esperado {b1_exp}): agujero esperado no encontrado")

    return {
        "is_anomaly": is_anomaly,
        "b0": b0,
        "b1": b1,
        "expected": expected_betti,
        "shape": shape_name,
        "details": "; ".join(details) if details else "Topología correcta",
        "eps_star": eps_star,
    }


def compare_anomaly_detection(
    n_points: int = 200,
    noise_level: float = 0.15,
    n_anomalies: int = 5,
    seed: int = 42,
) -> Dict:
    """Compara detección de anomalías: TDA vs K-Means.

    Genera formas normales (esfera/toro) y formas anómalas
    (esfera con agujero, toro desconectado), y evalúa si
    cada método detecta la anomalía correctamente.

    Ref: H.E.1 — "superando en detección de anomalías".

    Args:
        n_points: Puntos por forma.
        noise_level: Nivel de ruido gaussiano.
        n_anomalies: Número de anomalías a generar.
        seed: Semilla aleatoria.

    Returns:
        Dict con resultados TDA y K-Means.
    """
    rng = np.random.RandomState(seed)

    # Formas normales
    sphere_normal = add_gaussian_noise(
        generate_cloud("sphere", n_points), noise_level
    )
    torus_normal = add_gaussian_noise(
        generate_cloud("torus", n_points), noise_level
    )

    # Formas anómalas: esfera con agujero (puntos eliminados del centro)
    anomaly_sphere = sphere_normal.copy()
    center = anomaly_sphere.mean(axis=0)
    dist_to_center = np.linalg.norm(anomaly_sphere - center, axis=1)
    mask_hole = dist_to_center > np.percentile(dist_to_center, 30)
    anomaly_sphere = anomaly_sphere[mask_hole][:n_points]

    # Forma anómala: toro con ruido extreme (β₁ inestable)
    anomaly_torus = add_gaussian_noise(
        generate_cloud("torus", n_points), noise_level * 3
    )

    # ── Detección TDA ──
    tda_normal_s = detect_anomaly_tda(sphere_normal, (1, 0), "Esfera normal", n_points=n_points)
    tda_normal_t = detect_anomaly_tda(torus_normal, (1, 2), "Toro normal", n_points=n_points)
    tda_anomaly_s = detect_anomaly_tda(anomaly_sphere, (1, 0), "Esfera anómala", n_points=n_points)
    tda_anomaly_t = detect_anomaly_tda(anomaly_torus, (1, 2), "Toro anómalo", n_points=n_points)

    tda_results = {
        "normal": [tda_normal_s, tda_normal_t],
        "anomaly": [tda_anomaly_s, tda_anomaly_t],
        "true_positives": sum(1 for r in [tda_anomaly_s, tda_anomaly_t] if r["is_anomaly"]),
        "true_negatives": sum(1 for r in [tda_normal_s, tda_normal_t] if not r["is_anomaly"]),
    }
    tda_results["exactitud"] = (tda_results["true_positives"] + tda_results["true_negatives"]) / 4
    tda_results["accuracy"] = tda_results["exactitud"]  # Compatibilidad backward

    # ── Detección K-Means (baseline euclidiano) ──
    from sklearn.cluster import KMeans

    def _kmeans_anomaly检测(shape_pts, n_clusters=2):
        """K-Means no detecta anomalías topológicas — solo clasifica."""
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = km.fit_predict(shape_pts)
        inertia = km.inertia_
        return {"labels": labels, "inertia": inertia}

    km_normal_s = _kmeans_anomaly检测(sphere_normal)
    km_normal_t = _kmeans_anomaly检测(torus_normal)
    km_anomaly_s = _kmeans_anomaly检测(anomaly_sphere)
    km_anomaly_t = _kmeans_anomaly检测(anomaly_torus)

    # K-Means detecta anomalías por incremento de inercia (>50% = anomalía)
    threshold_inertia = max(km_normal_s["inertia"], km_normal_t["inertia"]) * 1.5
    km_results = {
        "normal": [
            {"inertia": km_normal_s["inertia"], "is_anomaly": km_normal_s["inertia"] > threshold_inertia},
            {"inertia": km_normal_t["inertia"], "is_anomaly": km_normal_t["inertia"] > threshold_inertia},
        ],
        "anomaly": [
            {"inertia": km_anomaly_s["inertia"], "is_anomaly": km_anomaly_s["inertia"] > threshold_inertia},
            {"inertia": km_anomaly_t["inertia"], "is_anomaly": km_anomaly_t["inertia"] > threshold_inertia},
        ],
    }
    km_tp = sum(1 for r in km_results["anomaly"] if r["is_anomaly"])
    km_tn = sum(1 for r in km_results["normal"] if not r["is_anomaly"])
    km_results["true_positives"] = km_tp
    km_results["true_negatives"] = km_tn
    km_results["exactitud"] = (km_tp + km_tn) / 4
    km_results["accuracy"] = km_results["exactitud"]  # Compatibilidad backward

    return {
        "tda": tda_results,
        "kmeans": km_results,
        "n_points": n_points,
        "noise_level": noise_level,
    }
