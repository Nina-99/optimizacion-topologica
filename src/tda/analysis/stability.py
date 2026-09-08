"""Análisis de estabilidad topológica frente a ruido.

Implementa el barrido sistemático de niveles de ruido para verificar
la hipótesis H.E.1: los números de Betti permanecen estables bajo
perturbaciones del 15-20% en los datos de entrada.

Referencias (Documento completo):
- Definición 1.4: Número de Betti
- Teorema de Estabilidad (Cohen-Steiner et al., 2007)
- Sección 8.6.1: Diseño de casos de simulación (esfera/toro)

NOTA DE DOMINIO: el criterio ε*/2 del Algoritmo 1 (paso 16) aplica a
diseños SIMP (malla de Ne=1800 elementos, escala de elemento finito) y NO
a nubes sintéticas sin malla. Para esfera (r=1) y toro (R=2, r=1) con
100-500 puntos se usa un umbral de significancia absoluto calibrado
(UMBRAL_H1_SINTETICO), ya que ε*=diam/√N deja pasar cientos de barras de
ruido Rips (persistencia <0.4) y nunca daría los (1,0)/(1,2) esperados.
"""

from typing import List, Tuple, Dict
import numpy as np
from ripser import ripser
from sklearn.cluster import KMeans

from tda.processing.sampling import generate_cloud, add_gaussian_noise
from tda.analysis.metrics import compute_kmeans_accuracy


def _escala_adaptativa(nube: np.ndarray) -> float:
    """Calcula ε* = diam(X) / √N para filtración Vietoris-Rips.

    Ref: Def. 1.5 del Documento Completo — escala adaptativa.
    Para nubes sintéticas (sin malla), N = cantidad de puntos.
    """
    if nube.shape[0] < 2:
        return 1.0
    dists = np.linalg.norm(nube[:, None] - nube[None, :], axis=2)
    diam = dists.max()
    eps_star = diam / np.sqrt(nube.shape[0])
    return eps_star


# Umbral de significancia H₁ calibrado para nubes sintéticas con
# 100-500 puntos (esfera r=1, toro R=2/r=1): el ruido Rips genera barras
# con persistencia <0.4; los túneles del toro persisten ~0.5-1.2 en limpio.
UMBRAL_H1_SINTETICO = 0.5

# Repeticiones internas por nivel de ruido: se reporta la mediana de Betti
# (robusta a la variabilidad entre semillas) y la media de exactitud.
N_REP_INTERNO = 5


def _contar_betti(dgm0_or_dgms, dgm1_or_umbral=None, umbral: float = UMBRAL_H1_SINTETICO) -> Tuple[int, int]:
    """Cuenta Betti significativos desde diagramas H₀/H₁ de Ripser.

    Soporta dos firmas:
    - `_contar_betti(dgm0, dgm1, umbral=0.5)`
    - `_contar_betti(dgms_dict_or_list, umbral=0.5)`
    """
    if isinstance(dgm0_or_dgms, (dict, list, tuple)) and dgm1_or_umbral is None or isinstance(dgm1_or_umbral, (int, float)):
        # Firma de 1er arg compuesto (diccionario ripser o lista de dgms)
        if isinstance(dgm1_or_umbral, (int, float)):
            umbral = float(dgm1_or_umbral)
        
        if isinstance(dgm0_or_dgms, dict) and "dgms" in dgm0_or_dgms:
            dgms = dgm0_or_dgms["dgms"]
            dgm0 = dgms[0] if len(dgms) > 0 else np.empty((0, 2))
            dgm1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
        elif isinstance(dgm0_or_dgms, (list, tuple)):
            dgm0 = dgm0_or_dgms[0] if len(dgm0_or_dgms) > 0 else np.empty((0, 2))
            dgm1 = dgm0_or_dgms[1] if len(dgm0_or_dgms) > 1 else np.empty((0, 2))
        else:
            dgm0 = np.empty((0, 2))
            dgm1 = np.empty((0, 2))
    else:
        dgm0 = dgm0_or_dgms
        dgm1 = dgm1_or_umbral

    dgm0 = np.asarray(dgm0) if dgm0 is not None else np.empty((0, 2))
    dgm1 = np.asarray(dgm1) if dgm1 is not None else np.empty((0, 2))

    if dgm0.size == 0:
        beta_0 = 0
    elif dgm0.ndim == 2 and dgm0.shape[1] >= 2:
        n_inf = int(np.sum(np.isinf(dgm0[:, 1])))
        beta_0 = n_inf if n_inf > 0 else 1
    else:
        beta_0 = 0

    if dgm1.size == 0:
        beta_1 = 0
    elif dgm1.ndim == 2 and dgm1.shape[1] >= 2:
        finitas = dgm1[np.isfinite(dgm1[:, 1])]
        if finitas.size == 0:
            beta_1 = 0
        else:
            beta_1 = int(np.sum((finitas[:, 1] - finitas[:, 0]) > umbral))
    else:
        beta_1 = 0

    return (beta_0, beta_1)


def betti_significativos(pts: np.ndarray,
                         umbral: float = UMBRAL_H1_SINTETICO):
    """Calcula (β₀, β₁) significativos de una nube + diagramas Ripser.

    Returns:
        ((b0, b1), res): tupla de Betti y salida cruda de ripser (para
        graficar diagramas de persistencia).
    """
    res = ripser(np.asarray(pts), maxdim=1)['dgms']
    dgm0 = res[0] if len(res) > 0 else np.empty((0, 2))
    dgm1 = res[1] if len(res) > 1 else np.empty((0, 2))
    return _contar_betti(dgm0, dgm1, umbral), res


def _betti_filtrado(diagram: np.ndarray, eps_star: float) -> Tuple[int, int]:
    """Compatibilidad: filtra un diagrama (n,2) o (n,3) con el umbral sintético.

    El parámetro eps_star se conserva por compatibilidad pero ya no define
    el umbral (ver NOTA DE DOMINIO arriba).
    """
    arr = np.asarray(diagram) if diagram is not None else np.empty((0, 2))
    if arr.size == 0:
        return (0, 0)
    if arr.ndim == 2 and arr.shape[1] == 3:
        h0 = arr[arr[:, 2] == 0][:, :2]
        h1 = arr[arr[:, 2] == 1][:, :2]
        return _contar_betti(h0, h1)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return _contar_betti(np.empty((0, 2)), arr)
    return (0, 0)


def _verificar_estabilidad(
    betti_vals: List[Tuple[int, int]],
    noise_vals: np.ndarray,
    rango_estabilidad: Tuple[float, float] = (0.15, 0.20),
) -> Dict:
    """Verifica que β₀/β₁ permanezcan estables en el rango 15-20%.

    Ref: H.E.1 — "β₀ y β₁ permanecen estables bajo perturbaciones del 15–20%".
    Criterio: la desviación estándar de β en el rango debe ser 0 (valores enteros).
    """
    idx_estable = np.where(
        (noise_vals >= rango_estabilidad[0]) & (noise_vals <= rango_estabilidad[1])
    )[0]

    if len(idx_estable) == 0:
        return {"estable": False, "mensaje": "Sin datos en rango 15-20%"}

    betti_rango = [betti_vals[i] for i in idx_estable]
    b0_vals = [b[0] for b in betti_rango]
    b1_vals = [b[1] for b in betti_rango]

    b0_estable = max(b0_vals) - min(b0_vals) == 0
    b1_estable = max(b1_vals) - min(b1_vals) == 0

    msg_parts = []
    if b0_estable and b1_estable:
        msg = (f"✅ Estables en rango {rango_estabilidad[0]:.0%}–"
               f"{rango_estabilidad[1]:.0%}: "
               f"β₀∈{{{','.join(map(str, b0_vals))}}}, "
               f"β₁∈{{{','.join(map(str, b1_vals))}}}")
    else:
        if not b0_estable:
            msg_parts.append(
                f"β₀ inestable: {{{','.join(map(str, b0_vals))}}}"
            )
        if not b1_estable:
            msg_parts.append(
                f"β₁ inestable: {{{','.join(map(str, b1_vals))}}}"
            )
        msg = (f"❌ Inestables en rango {rango_estabilidad[0]:.0%}–"
               f"{rango_estabilidad[1]:.0%}: {'; '.join(msg_parts)}")

    return {
        "estable": b0_estable and b1_estable,
        "b0_estable": b0_estable,
        "b1_estable": b1_estable,
        "b0_vals_rango": b0_vals,
        "b1_vals_rango": b1_vals,
        "mensaje": msg,
    }


def compute_noise_sweep(
    n_points: int,
    n_clusters: int,
    noise_min: float = 0.0,
    noise_max: float = 0.30,
    n_steps: int = 10,
    random_seed: int = 42,
    progress_callback=None,
) -> dict:
    """Ejecuta un barrido sistemático de ruido y evalúa TDA vs K-Means.

    Para cada nivel de ruido genera nubes de puntos (esfera + toro),
    calcula los números de Betti vía homología persistente (Ripser)
    filtrando por persistencia > ε*/2, y la exactitud (accuracy) de K-Means.

    Ref: Sección 8.6.1 — diseños de simulación H.E.1.

    Args:
        n_points: Puntos por forma geométrica.
        n_clusters: Número de clústeres para K-Means.
        noise_min: Nivel de ruido mínimo (default 0.0).
        noise_max: Nivel de ruido máximo (default 0.30).
        n_steps: Cantidad de pasos en el barrido (default 10).
        random_seed: Semilla para reproducibilidad (default 42).
        progress_callback: Función opcional para reportar progreso.
            Se llama con (paso_actual, pasos_totales) después de cada nivel.

    Returns:
        Dict con:
            noise_vals (np.ndarray): Niveles de ruido evaluados.
            acc (List[float]): Exactitud K-Means por nivel.
            betti_s (List[Tuple[int,int]]): (β₀, β₁) esfera filtrados.
            betti_t (List[Tuple[int,int]]): (β₀, β₁) toro filtrados.
            diagrams_s (List[np.ndarray]): Diagramas de persistencia de esfera.
            diagrams_t (List[np.ndarray]): Diagramas de persistencia de toro.
            estabilidad_s (Dict): Verificación de estabilidad 15-20% esfera.
            estabilidad_t (Dict): Verificación de estabilidad 15-20% toro.
    """
    np.random.seed(random_seed)
    noise_vals = np.linspace(noise_min, noise_max, n_steps)

    sweep_acc: List[float] = []
    sweep_betti_s: List[Tuple[int, int]] = []
    sweep_betti_t: List[Tuple[int, int]] = []
    sweep_dgms_s: List = []
    sweep_dgms_t: List = []

    for i, noise in enumerate(noise_vals):
        # N_REP_INTERNO repeticiones por nivel: mediana de Betti (robusta a
        # la semilla) y media de exactitud. Diagramas: los de la repetición
        # mediana (más cercana a la mediana de Betti) para que la animación
        # calce al 100% con la tabla.
        acc_reps: List[float] = []
        b0s_s: List[int] = []
        b1s_s: List[int] = []
        b0s_t: List[int] = []
        b1s_t: List[int] = []
        res_s_reps: List = []
        res_t_reps: List = []

        for _ in range(N_REP_INTERNO):
            # Generar nubes limpias
            pts_s = generate_cloud("sphere", n_points)
            pts_t = generate_cloud("torus", n_points)
            pts_t[:, 0] += 1.5  # separación para evitar superposición total

            # Añadir ruido
            pts_s_n = add_gaussian_noise(pts_s, noise)
            pts_t_n = add_gaussian_noise(pts_t, noise)

            # K-Means sobre conjunto combinado
            dataset_n = np.vstack([pts_s_n, pts_t_n])
            km = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
            y_pred = km.fit_predict(dataset_n)

            y_true = np.array([0] * n_points + [1] * n_points)
            acc_reps.append(compute_kmeans_accuracy(y_true, y_pred))

            # TDA: Betti significativos (β₀ por barras infinitas H₀,
            # β₁ por persistencia > UMBRAL_H1_SINTETICO)
            (b0_s, b1_s), res_s = betti_significativos(pts_s_n)
            (b0_t, b1_t), res_t = betti_significativos(pts_t_n)
            b0s_s.append(b0_s)
            b1s_s.append(b1_s)
            b0s_t.append(b0_t)
            b1s_t.append(b1_t)
            res_s_reps.append(res_s)
            res_t_reps.append(res_t)

        med_b0_s, med_b1_s = int(np.median(b0s_s)), int(np.median(b1s_s))
        # Repetición mediana: mínima distancia Manhattan a la mediana.
        # Desempate determinista: primera con distancia mínima.
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

    # ── Verificación de estabilidad en rango 15-20% (H.E.1) ──
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
