"""Página 1: H.E.1 — Robustez topológica (TDA).

Presenta la validación en el ORDEN del documento:
  0. Calibración previa del umbral τ (Figura 5.1) y estabilidad dimensional
     (Figura 5.2: paneles a1–a3, b y c)
  1. H.E.1a — estabilidad de los diagramas (Figura 5.3)
  2. H.E.1b — ventaja comparativa TDA vs euclidiano (contraste de McNemar)
  3. H.E.1c — rango de fiabilidad de βk (q* de la Figura 5.2)
  4. Veredicto H.E.1 (Tabla 5.5)

Constantes y notación tomadas del documento:
  - Catálogo de 7 nubes (§1.5.2, Tabla 4.3).
  - q ∈ {0, 0.05, 0.10, 0.15, 0.20}; σ = q·diam(X)/√d (corregido, Tabla 1.1 /
    §2.2.1 — Item 29 de las correcciones).
  - Umbral τ = 0.175 (τ* calibrado sobre C₀, Fig. 5.1 / §5.4.1).
  - R = 100 réplicas; semilla 42 (Algoritmo maestro, §3.4.3).
  - H.E.1a: dB/diam(X) ≤ 2q en el 100 % de las réplicas (k = 0, 1).
  - H.E.1b: exactitud TDA > euclidiana (McNemar p < 0.05), disco vs anillo + afín.
  - H.E.1c: ∃ q* tal que acierto ≥ 0.95 para q < q* y < 0.50 para q > q*; q* ∈ [0.05, 0.10].
"""

import io
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
from ripser import ripser

from tda.processing.sampling import (
    generate_cloud, add_gaussian_noise, compute_diameter, betti_teorico,
)
from tda.analysis.stability import (
    betti_persistente, comparar_tda_vs_euclidiano, hausdorff_distance,
)
from tda.core.topology import bottleneck_distance
from tda.analysis.calibracion import ETIQUETA_NUBE, resultado_calibracion
from tda.visualization.figuras import fig_51_calibracion, fig_52_estabilidad
from tda.app.theme import (
    apply_mpl_theme, apply_plotly_theme, responsive_style,
    methodology_expander, is_dark, page_header,
)
from tda.simulation.he_protocols import (
    protocolo_he1, CATALOGO, SUBCONJUNTO_VALIDADO,
    Q_LEVELS, TAU_CALIBRADO, SEED,
)

# ── Constantes del documento ─────────────────────────────────────────
N_MIN = 400                   # §1.5.2: densidad mínima (toro detectable)
R_DEFECTO = 20               # UI práctica; protocolo del documento: R = 100
COTA_FRACCION = 1.0           # H.E.1a: 100 % de las réplicas
ACC_BAJO = 0.95               # H.E.1c: umbral superior
ACC_ALTO = 0.50               # H.E.1c: umbral inferior
Q_STAR_RANGO = (0.05, 0.10)   # H.E.1c


def _pers_h1_dominante_por_q(shape: str, n_points: int, qs, seed: int):
    """Persistencia H₁ dominante normalizada por diam(X), por nivel de ruido q.

    Diagnóstico de la Fig. 5.2(a): un agujero se cuenta solo si su persistencia
    supera τ·diam(X). Si la mayor persistencia H₁ cae por debajo de τ, β₁ deja de
    leerse bien (p. ej. dos_circulos, cuya nube se rompe con el ruido).
    """
    out = []
    for q in qs:
        np.random.seed(seed)
        X0 = generate_cloud(shape, n_points)
        diam = compute_diameter(X0)
        np.random.seed(seed + int(1e5 * q))
        Y = add_gaussian_noise(X0, q)
        dgm = ripser(Y, maxdim=1)["dgms"]
        h1 = np.asarray(dgm[1])
        pers = (h1[:, 1] - h1[:, 0]) if len(h1) else np.array([0.0])
        pers = pers[np.isfinite(pers)]
        out.append(float(pers.max() / diam) if len(pers) and diam > 0 else 0.0)
    return out


# ── Configuración de página ──────────────────────────────────────────
st.set_page_config(page_title="H.E.1 — Robustez TDA", layout="wide", page_icon="📊")
st.markdown(responsive_style(), unsafe_allow_html=True)

# ── Sidebar (parámetros del documento) ───────────────────────────────
st.sidebar.header("📊 H.E.1 — Robustez TDA")
n_points = st.sidebar.number_input("Puntos por nube (n ≥ 400)", min_value=400,
                                   max_value=1000, value=N_MIN, step=50)
n_rep = st.sidebar.slider("Réplicas por nube y nivel (R)", min_value=10, max_value=100,
                          value=R_DEFECTO, step=10)
n_mcn = st.sidebar.slider("Muestras por clase (H.E.1b ≥ 200)", min_value=200,
                          max_value=600, value=200, step=50)
st.sidebar.caption(
    f"Protocolo del documento: 7 nubes × q ∈ {{0, .05, .10, .15, .20}} × R réplicas · "
    f"τ = {TAU_CALIBRADO} · semilla {SEED}. (El documento usa R = 100; subir R es más lento: "
    f"≈0.13 s por réplica.)"
)

# ── Header + objetivo (fórmulas del documento) ───────────────────────
st.markdown(page_header("H.E.1 — Robustez TDA vs Euclidianos"), unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🎯 Hipótesis H.E.1")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**H.E.1a — estabilidad de los diagramas**")
    st.latex(r"d_{B}\bigl(\mathrm{Dgm}_{k}^{\mathrm{VR}}(X),\ \mathrm{Dgm}_{k}^{\mathrm{VR}}(Y)\bigr) \le 2\,d_{H}(X,Y),\quad k=0,1")
    st.caption("Se satisface en el **100 %** de las réplicas, con ruido gaussiano $\\sigma = q\\,\\mathrm{diam}(X)/\\sqrt{d}$. Es la cota del Teorema de Estabilidad (Cohen-Steiner et al.).")
with col_b:
    st.markdown("**H.E.1b — ventaja comparativa (TDA vs euclidiano)**")
    st.caption("Disco vs anillo bajo deformación afín aleatoria. El descriptor de persistencia supera al euclidiano (PCA + inercia de k-medias + dispersión radial) con $p<0.05$ (McNemar).")

st.markdown("**H.E.1c — rango de fiabilidad de $\\beta_k$**")
st.latex(r"\exists\, q^{*}:\ \beta_{k}^{(\tau)}(X)=\beta_{k}^{\mathrm{teo}}\ \text{en} \ge 95\%\ (q<q^{*}),\quad < 50\%\ (q>q^{*})")
st.markdown("---")

ejecutar = st.button("▶ Ejecutar protocolo H.E.1", type="primary")

# ── 0. Calibración previa del umbral τ (Figura 5.1) ──────────────────
st.subheader("Paso previo — Figura 5.1: calibración del umbral relativo de persistencia $\\tau$")
st.caption(
    "$\\tau$ se fija **antes** de medir, sobre el catálogo independiente "
    "$\\mathcal{C}_0$ (semillas 900+i, sin ruido). El panel (a) suma los aciertos de "
    "$(\\beta_0, \\beta_1)$ de las siete nubes, el (b) muestra cuál falla en cada umbral "
    "y el (c) el margen de cada nube de $\\mathcal{C}$ respecto de $\\tau^*$."
)
if st.button("Calibrar τ (Figura 5.1)", key="btn_calib_tau"):
    with st.spinner("Calibrando τ sobre C₀ (7 nubes × 15 umbrales, sin ruido)..."):
        st.session_state.he1_calib = resultado_calibracion(n_points=int(n_points))

calib = st.session_state.get("he1_calib")
if calib:
    fig51 = fig_51_calibracion(calib["cal"], calib["tau"], calib["margenes"],
                               ETIQUETA_NUBE,
                               tema="oscuro" if is_dark() else "claro")
    st.pyplot(fig51)
    plt.close(fig51)
    st.session_state.setdefault("he1_figs", {})["fig51_calibracion_tau"] = fig51
    v_min, v_max = calib["tau"]["ventana_min"], calib["tau"]["ventana_max"]
    dentro = v_min <= TAU_CALIBRADO <= v_max
    (st.success if dentro else st.warning)(
        f"Ventana con {len(CATALOGO)}/{len(CATALOGO)} aciertos: "
        f"τ ∈ [{v_min:.3f}, {v_max:.3f}] · τ* = {calib['tau']['tau_star']:.3f} · "
        f"τ adoptado = {TAU_CALIBRADO} {'cae dentro' if dentro else 'queda FUERA'}."
    )
    st.markdown("---")

# ── Paso previo — Figura 5.2: estabilidad dimensional ────────────────
st.subheader("Paso previo — Figura 5.2: estabilidad del diagrama y fiabilidad de $\\beta_k$")
st.caption(
    "El ruido se normaliza por la dimensión ($\\sigma = q\\,\\mathrm{diam}(X)/\\sqrt{d}$) y "
    "$\\tau$ es el calibrado en la Fig. 5.1. Los paneles (a1–a3) miden la lectura de "
    "$(\\beta_0,\\beta_1)$ en la dimensión nativa y en $d = 10,\\, 100$; (b) contrasta "
    "$d_B$ con la cota $2\\,d_H$ en todas las réplicas y (c) cuán holgada es esa cota."
)
from tda.analysis.estabilidad_dimensional import (
    cargar_resultados, guardar_resultados, protocolo_fig52, q_estrella,
)
res52 = st.session_state.get("he1_fig52") or cargar_resultados()
if res52 is None:
    if st.button("Ejecutar Figura 5.2 (1 680 réplicas)", key="btn_fig52"):
        with st.spinner("Corriendo 7 nubes × {nativa, 10, 100} × 4 niveles × 20 réplicas..."):
            res52 = protocolo_fig52(n_points=int(n_points))
            guardar_resultados(res52)
        st.session_state.he1_fig52 = res52
if res52:
    fig52 = fig_52_estabilidad(res52["replicas"], res52["q0"],
                               tema="oscuro" if is_dark() else "claro")
    st.pyplot(fig52)
    plt.close(fig52)
    st.session_state.setdefault("he1_figs", {})["fig52_estabilidad"] = fig52
    _rep52 = res52["replicas"]
    _viol = int((_rep52["dB"] > _rep52["cota_teorema"] + 1e-9).sum())
    st.caption(f"{len(_rep52)} réplicas · **{_viol}** violaciones de $d_B \\leq 2\\,d_H$.")
    with st.expander("Nivel crítico q* por nube y dimensión (H.E.1c)", expanded=False):
        st.dataframe(q_estrella(_rep52), width="stretch", hide_index=True)
    st.markdown("---")

# ── 🔬 Filtración interactiva (ε) — el corazón del TDA ───────────────
st.subheader("🔬 Filtración interactiva — complejo de Vietoris–Rips")
st.caption("Mové el radio ε: los puntos se conectan y los rasgos topológicos **nacen y mueren**. "
           "Abajo, el código de barras con el umbral ε marcado (línea roja).")
_fc1, _fc2, _fc3 = st.columns([2, 1, 1])
with _fc1:
    _shape = st.selectbox("Nube", ["circulo", "anillo", "dos_circulos", "toro", "esfera"],
                          key="filt_shape")
with _fc2:
    _nf = st.slider("Puntos", 40, 200, 100, 10, key="filt_n")
with _fc3:
    _eps = st.slider("Radio ε", 0.0, 3.0, 0.5, 0.05, key="filt_eps")

from scipy.spatial.distance import pdist, squareform
np.random.seed(SEED)
_Xf = generate_cloud(_shape, _nf)
_dim = _Xf.shape[1]
_dgm = ripser(_Xf, maxdim=1)["dgms"]
_D = squareform(pdist(_Xf))
_ei, _ej = np.where(np.triu(_D <= _eps, k=1))

# 1-esqueleto (aristas ≤ ε)
_ex, _ey, _ez = [], [], []
for _a, _b in zip(_ei, _ej):
    _ex += [_Xf[_a, 0], _Xf[_b, 0], None]
    _ey += [_Xf[_a, 1], _Xf[_b, 1], None]
    if _dim == 3:
        _ez += [_Xf[_a, 2], _Xf[_b, 2], None]
fig_filt = go.Figure()
_line_kw = dict(x=_ex, y=_ey, mode="lines", line=dict(color="#bdc3c7", width=1), name="aristas ≤ ε")
if _dim == 3:
    _line_kw["z"] = _ez
    fig_filt.add_trace(go.Scatter3d(**_line_kw))
    fig_filt.add_trace(go.Scatter3d(x=_Xf[:, 0], y=_Xf[:, 1], z=_Xf[:, 2], mode="markers",
                                    marker=dict(size=3, color="#e74c3c"), name="puntos"))
else:
    fig_filt.add_trace(go.Scatter(**_line_kw))
    fig_filt.add_trace(go.Scatter(x=_Xf[:, 0], y=_Xf[:, 1], mode="markers",
                                  marker=dict(size=6, color="#e74c3c"), name="puntos"))

def _betti_at(dgms, eps):
    b0 = sum(1 for b, d in dgms[0] if b <= eps and (np.isinf(d) or eps < d))
    b1 = sum(1 for b, d in dgms[1] if b <= eps and (np.isinf(d) or eps < d))
    return b0, b1

_b0e, _b1e = _betti_at(_dgm, _eps)
fig_filt.update_layout(height=380, template="plotly_white",
                       title=f"Complejo VR(X, ε = {_eps:.2f}) · β₀ = {_b0e}, β₁ = {_b1e}",
                       legend=dict(x=0, y=1), margin=dict(l=0, r=0, b=0, t=40))
apply_plotly_theme(fig_filt)
st.plotly_chart(fig_filt, width="stretch")
_m1, _m2 = st.columns(2)
_m1.metric("β₀ (componentes conexas)", _b0e)
_m2.metric("β₁ (ciclos / agujeros)", _b1e)

# Código de barras con línea de ε
fig_bc = go.Figure()
for _k, _col in [(0, "#2980b9"), (1, "#27ae60")]:
    _bars = _dgm[_k] if _k < len(_dgm) else np.empty((0, 2))
    _y = 0
    for _b, _d in _bars:
        _dd = _b + 1.5 if np.isinf(_d) else _d
        fig_bc.add_trace(go.Scatter(x=[_b, _dd], y=[_y, _y], mode="lines",
                                    line=dict(color=_col, width=3), showlegend=(_y == 0),
                                    name=f"H{_k}"))
        _y += 1
fig_bc.add_vline(x=_eps, line=dict(color="red", dash="dot"),
                 annotation_text=f"ε = {_eps:.2f}")
fig_bc.update_layout(height=300, template="plotly_white",
                     title="Código de barras (nacimiento → muerte)",
                     xaxis_title="escala ε", yaxis_title="rasgo")
apply_plotly_theme(fig_bc)
st.plotly_chart(fig_bc, width="stretch")
st.markdown("---")

# ── Ejecución del protocolo (H.E.1a y H.E.1c) + McNemar (H.E.1b) ─────
if ejecutar:
    with st.spinner("H.E.1a/H.E.1c — barriendo el catálogo por nube y nivel de ruido..."):
        res_sweep, crudo = protocolo_he1(n_points, Q_LEVELS, n_rep, TAU_CALIBRADO, SEED)
    with st.spinner("H.E.1b — disco vs anillo con deformación afín y contraste de McNemar..."):
        res_mcn = comparar_tda_vs_euclidiano(n_points=n_points, n_rep=2 * n_mcn,
                                             tau=TAU_CALIBRADO, apply_affine=True,
                                             random_seed=SEED)
    st.session_state.he1_results = {"sweep": res_sweep, "crudo": crudo, "mcnemar": res_mcn}
    st.session_state.he1_params = {"n_points": n_points, "n_rep": n_rep, "n_mcn": n_mcn}
    st.session_state.tda_he1_run = True

res = st.session_state.get("he1_results", {})
if st.session_state.get("tda_he1_run", False) and res:
    params = st.session_state.get("he1_params", {})
    st.info(f"📋 Resultados — n = {params.get('n_points')}, R = {params.get('n_rep')} réplicas, "
            f"q ∈ {Q_LEVELS}, τ = {TAU_CALIBRADO}, semilla {SEED}.")

    sweep = res["sweep"]
    crudo = res["crudo"]
    qs = Q_LEVELS

    # ══════════════════════════════════════════════════════════════════
    # 1. H.E.1a — estabilidad de los diagramas (Figuras 5.3 y 5.2(b))
    # ══════════════════════════════════════════════════════════════════
    st.subheader("1. H.E.1a — Estabilidad de los diagramas")
    st.latex(r"d_{B}\bigl(\mathrm{Dgm}_{k}^{\mathrm{VR}}(X),\ \mathrm{Dgm}_{k}^{\mathrm{VR}}(Y)\bigr) \le 2\,d_{H}(X,Y),\quad k=0,1")

    # Figura 5.3 — distancia bottleneck relativa frente a la cota. Toggle:
    #   forma del documento (dB/diam <= 2q)  <->  forma del teorema (dB <= 2 dH)
    st.markdown("**Figura 5.3:** Distancia bottleneck relativa frente a la cota, promediada por nube y nivel de ruido. El selector alterna las dos formas para **exhibir la inconsistencia del documento**:")
    _forma53 = st.radio(
        "Forma de la cota (Fig. 5.3)",
        ["Forma del documento  $d_B/\\mathrm{diam}(X) \\leq 2q$",
         "Forma del teorema  $d_B \\leq 2\\,d_H$  (correcta)"],
        horizontal=True, key="forma53")
    _es_doc = _forma53.startswith("Forma del documento")
    _orden = ["anillo", "circulo", "disco", "dos_circulos", "esfera", "gaussiana", "toro"]
    _col = {"anillo": "#1f77b4", "circulo": "#ff7f0e", "disco": "#2ca02c",
            "dos_circulos": "#d62728", "esfera": "#9467bd", "gaussiana": "#8c564b", "toro": "#e377c2"}
    _sym = {"anillo": "circle", "circulo": "square", "disco": "triangle-up",
            "dos_circulos": "triangle-down", "esfera": "diamond", "gaussiana": "cross", "toro": "x"}
    if _es_doc:
        _key_medio, _val = "ratio_diam_medio", (lambda c: c["dB_diam"])
        _cota = dict(x=[min(qs), max(qs)], y=[0.0, 2.0 * max(qs)], name="cota 2q")
        _ytitle = "d_B / diam(X)"
        _yfloor, _dtick = 0.40, 0.05
    else:
        _key_medio, _val = "ratio_medio", (lambda c: c["dB"] / c["dH"] if c["dH"] > 0 else float("nan"))
        _cota = dict(x=[min(qs), max(qs)], y=[2.0, 2.0], name="cota 2")
        _ytitle = "d_B / d_H"
        _yfloor, _dtick = 2.1, 0.25
    _vmax = max([v for v in (_val(c) for c in crudo) if v == v] + [_yfloor]) * 1.05
    fig53 = go.Figure()
    for shape in _orden:
        _vals = {q: [v for v in (_val(c) for c in crudo if c["shape"] == shape and c["q"] == q) if v == v] for q in qs}
        _std = [float(np.std(_vals[q])) if _vals[q] else 0.0 for q in qs]
        fig53.add_trace(go.Scatter(x=qs, y=sweep[shape][_key_medio], mode="lines+markers",
                                   name=shape, marker=dict(color=_col[shape], symbol=_sym[shape]),
                                   error_y=dict(type="data", array=_std, visible=True, thickness=1.2)))
    fig53.add_trace(go.Scatter(x=_cota["x"], y=_cota["y"], mode="lines",
                               line=dict(dash="dash", color="white"), name=_cota["name"]))
    fig53.update_layout(height=380, template="plotly_white",
                        title=("Distancia bottleneck relativa frente a la cota — "
                               + ("forma del documento" if _es_doc else "forma del teorema")),
                        xaxis_title="nivel de ruido q",
                        yaxis_title=_ytitle,
                        xaxis=dict(range=[0.0, 0.20], dtick=0.025),
                        yaxis=dict(range=[0.0, _vmax], dtick=_dtick))
    apply_plotly_theme(fig53)
    st.plotly_chart(fig53, width="stretch")
    st.session_state.setdefault("he1_figs", {})["fig53_bottleneck_cota_teorema"] = fig53
    if _es_doc:
        st.warning("Con la métrica bottleneck **exacta**, círculo y dos_círculos **cruzan** la cota $2q$: la cota $d_B/\\mathrm{diam}(X) \\leq 2q$ **no** es consecuencia del Teorema de Estabilidad (Item 11). El documento la presenta como válida porque la dibujó con la heurística que subestimaba $d_B$.")
    else:
        st.success("Forma correcta del Teorema de Estabilidad (Cohen-Steiner, Edelsbrunner & Harer): $d_B \\leq 2\\,d_H$. Todas las curvas quedan por debajo de la cota.")


    # Resultado H.E.1a
    frac_global = float(np.mean([sweep[s]["cumple_frac"][i]
                                 for s in CATALOGO for i in range(len(qs))]))
    viola_teorema = any(c["dB"] > 2 * c["dH"] + 1e-9 for c in crudo)
    filas_a = [{"Nube": s, **{f"q={q}": f"{sweep[s]['cumple_frac'][i]:.2f}" for i, q in enumerate(qs)}}
               for s in CATALOGO]
    st.dataframe(pd.DataFrame(filas_a), width="stretch", hide_index=True)
    if frac_global == COTA_FRACCION and not viola_teorema:
        st.success(f"### ✅ H.E.1a SE SOSTIENE — dB ≤ 2 dH en el {frac_global:.3f} de las réplicas")
    else:
        st.error(f"### ❌ H.E.1a NO SE SOSTIENE — fracción = {frac_global:.3f} (se exige {COTA_FRACCION:.3f})")
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # 2. H.E.1b — ventaja comparativa TDA vs euclidiano (McNemar)
    # ══════════════════════════════════════════════════════════════════
    st.subheader("2. H.E.1b — Ventaja comparativa (TDA vs euclidiano)")
    st.caption("Tarea: disco vs anillo bajo transformación afín aleatoria. Descriptor euclidiano de referencia = espectro PCA + inercia de k-medias + dispersión radial. Contraste de McNemar, p < 0.05.")

    mcn = res["mcnemar"]
    mc1, mc2 = st.columns(2)
    with mc1:
        st.metric("Exactitud TDA", f"{mcn['exactitud_tda']:.3f}",
                  help=f"Muestras por clase: {mcn['n_puntos_por_clase']}")
        st.metric("Exactitud euclidiana", f"{mcn['exactitud_eucl']:.3f}")
    with mc2:
        tabla = mcn["mc_nemar"].get("contingency_table") if isinstance(mcn.get("mc_nemar"), dict) else None
        p = mcn["mc_nemar"].get("p_value") if isinstance(mcn.get("mc_nemar"), dict) else None
        stat = mcn["mc_nemar"].get("statistic") if isinstance(mcn.get("mc_nemar"), dict) else None
        if tabla is not None:
            st.dataframe(pd.DataFrame(tabla, index=["TDA correcto", "TDA incorrecto"],
                                      columns=["Eucl correcto", "Eucl incorrecto"]),
                         width="stretch")
        if p is None:
            st.warning("McNemar no aplicable (sin discordancias).")
        else:
            _p = "p < 0.001" if p < 0.001 else f"p = {p:.4f}"
            _s = f"χ² = {stat:.2f}" if stat is not None else "χ² n/d"
            st.write(f"Contraste de McNemar: {_s}, {_p}")
    he1b_ok = bool(mcn["mc_nemar"].get("significant")) and (mcn["exactitud_tda"] > mcn["exactitud_eucl"])
    if he1b_ok:
        st.success("### ✅ H.E.1b SE SOSTIENE — el descriptor topológico supera al euclidiano (p < 0.05)")
    else:
        st.error("### ❌ H.E.1b NO SE SOSTIENE — la ventaja no es estadísticamente significativa")
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # 3. H.E.1c — rango de fiabilidad de βk (Figura 5.2(a))
    # ══════════════════════════════════════════════════════════════════
    st.subheader("3. H.E.1c — Rango de fiabilidad de $\\beta_k$")

    # Diagnóstico: por qué algunas nubes colapsan antes (persistencia H₁ vs τ)
    with st.expander("🔎 Diagnóstico de la Fig. 5.2(a) — ¿por qué colapsa β₁?", expanded=False):
        st.caption("Persistencia $H_1$ **dominante** normalizada por $\\mathrm{diam}(X)$, por nube con agujeros, "
                   "frente al umbral $\\tau$ (línea roja). Una nube se lee correctamente mientras su mayor persistencia "
                   "$H_1$ supere $\\tau\\cdot\\mathrm{diam}(X)$; cuando el ruido rompe la nube, la persistencia cae "
                   "por debajo de $\\tau$ y $\\beta_1$ deja de detectarse. Es la causa de que, p. ej., `dos_circulos` "
                   "colapse antes que en el documento (su nube separa más los centros ⇒ $\\mathrm{diam}$ grande ⇒ "
                   "$\\tau\\cdot\\mathrm{diam}$ grande frente a agujeros de radio 1) — ver Item 27.")
        _diag_qs = list(qs)
        _con_agujero = ["circulo", "anillo", "dos_circulos", "toro"]
        _col_diag = {"circulo": "#ff7f0e", "anillo": "#1f77b4",
                     "dos_circulos": "#d62728", "toro": "#e377c2"}
        fig_diag = go.Figure()
        for _s in _con_agujero:
            _pers = _pers_h1_dominante_por_q(_s, n_points, _diag_qs, SEED)
            fig_diag.add_trace(go.Scatter(x=_diag_qs, y=_pers, mode="lines+markers",
                                          name=_s, line=dict(color=_col_diag[_s])))
        fig_diag.add_hline(y=TAU_CALIBRADO, line=dict(dash="dash", color="red"),
                           annotation_text=f"τ = {TAU_CALIBRADO}", annotation_position="top left")
        fig_diag.update_layout(height=350, template="plotly_white",
                               title="Persistencia H₁ / diam(X) frente al umbral τ",
                               xaxis_title="nivel de ruido q",
                               yaxis_title="persistencia H₁ / diam(X)",
                               xaxis=dict(range=[0.0, 0.20], dtick=0.05))
        apply_plotly_theme(fig_diag)
        st.plotly_chart(fig_diag, width="stretch")

    # q* = máx{q : tasa de acierto ≥ 0.95} sobre el subconjunto validado (Tabla 5.5, nota a)
    acc_bk_all = [float(np.mean([sweep[s]["acierto"][i] for s in CATALOGO])) for i in range(len(qs))]
    acc_b1 = [float(np.mean([sweep[s]["acierto_b1"][i] for s in SUBCONJUNTO_VALIDADO])) for i in range(len(qs))]
    st.caption("La Fig. 5.2(a) muestra **βk = (β₀, β₁)** (como el documento). El nivel crítico **q\\*** se delimita "
               "sobre **β₁** (el invariante que la filtración controla), evaluado en el subconjunto validado "
               "(círculo, disco, anillo) según la nota (a) de la Tabla 5.5 — ver Item 13 del archivo de correcciones.")
    candidatos = [q for q, a in zip(qs, acc_b1) if a >= ACC_BAJO]
    q_star = max(candidatos) if candidatos else None
    q_bajo_colapso = next((q for q, a in zip(qs, acc_b1) if q > 0 and a < ACC_ALTO), None)
    st.dataframe(pd.DataFrame({"q": qs,
                               "Acierto βk (catálogo)": [f"{a:.3f}" for a in acc_bk_all],
                               "Acierto β₁ (validado)": [f"{a:.3f}" for a in acc_b1],
                               "≥ 0.95": ["✅" if a >= ACC_BAJO else "—" for a in acc_b1],
                               "< 0.50": ["✅" if a < ACC_ALTO else "—" for a in acc_b1]}),
                 width="stretch", hide_index=True)
    in_rango = q_star is not None and Q_STAR_RANGO[0] <= q_star <= Q_STAR_RANGO[1]
    if in_rango:
        st.success(f"### ✅ H.E.1c SE SOSTIENE — q* ≈ {q_star:.2f} ∈ [{Q_STAR_RANGO[0]}, {Q_STAR_RANGO[1]}]"
                   + (f"; colapso (acierto < 0.50) en q ≈ {q_bajo_colapso:.2f}" if q_bajo_colapso else ""))
    else:
        st.error(f"### ❌ H.E.1c NO SE SOSTIENE — q* = {q_star} fuera de [{Q_STAR_RANGO[0]}, {Q_STAR_RANGO[1]}]")
    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════
    # 4. Veredicto H.E.1 (Tabla 5.5)
    # ══════════════════════════════════════════════════════════════════
    st.subheader("4. Veredicto H.E.1 — Tabla 5.5 (indicador, criterio, esperado, medido)")
    acc_bajo = float(np.mean([a for q, a in zip(qs, acc_b1) if q <= 0.05])) if any(q <= 0.05 for q in qs) else None
    acc_alto = float(np.mean([a for q, a in zip(qs, acc_b1) if q >= 0.15])) if any(q >= 0.15 for q in qs) else None
    filas55 = [
        {"Indicador": "dB ≤ 2 dH", "Criterio": "Fracción de réplicas = 1.0",
         "Esperado": "1.000", "Medido": f"{frac_global:.3f}", "Veredicto": "✓" if frac_global == 1.0 else "×"},
        {"Indicador": "Acierto de β₁, q ≤ 0.05", "Criterio": "≥ 0.95",
         "Esperado": "≥ 0.95", "Medido": f"{acc_bajo:.2f}" if acc_bajo is not None else "—",
         "Veredicto": "✓" if (acc_bajo is not None and acc_bajo >= 0.95) else "×"},
        {"Indicador": "Acierto de β₁, q ≥ 0.15", "Criterio": "< 0.50 (régimen no válido)",
         "Esperado": "< 0.50", "Medido": f"{acc_alto:.3f}" if acc_alto is not None else "—",
         "Veredicto": "✓" if (acc_alto is not None and acc_alto < 0.50) else "×"},
        {"Indicador": "Nivel crítico q*", "Criterio": "Delimitado empíricamente",
         "Esperado": "≈ 0.05–0.10", "Medido": f"≈ {q_star:.2f}" if q_star is not None else "no detectado",
         "Veredicto": "✓" if in_rango else "×"},
        {"Indicador": "Exactitud TDA vs euclidiana", "Criterio": "McNemar p < 0.05",
         "Esperado": "significativa", "Medido": f"{mcn['exactitud_tda']:.3f} vs {mcn['exactitud_eucl']:.3f}",
         "Veredicto": "✓" if he1b_ok else "×"},
    ]
    st.table(pd.DataFrame(filas55))
    n_ok = sum(1 for f in filas55 if f["Veredicto"] == "✓")
    st.markdown(f"**Síntesis H.E.1:** {n_ok}/{len(filas55)} criterios cumplidos — "
                f"H.E.1a {'✓' if frac_global == 1.0 else '×'}, "
                f"H.E.1b {'✓' if he1b_ok else '×'}, "
                f"H.E.1c {'✓' if in_rango else '×'}.")

    # ── Exportación ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("5. Exportar resultados")
    try:
        from tda.app.download_utils import download_button as dl_btn, write_figs_zip, entorno_json
        df_crudo = pd.DataFrame(crudo)
        zf_buf = io.BytesIO()
        with zipfile.ZipFile(zf_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            base = f"he1_n{params.get('n_points')}_R{params.get('n_rep')}"
            zf.writestr(f"{base}/replicas_dB_dH.csv", df_crudo.to_csv(index=False))
            zf.writestr(f"{base}/tabla_5_5.csv", pd.DataFrame(filas55).to_csv(index=False))
            zf.writestr(f"{base}/params.txt",
                        f"n={params.get('n_points')}\nR={params.get('n_rep')}\n"
                        f"q={Q_LEVELS}\ntau={TAU_CALIBRADO}\nseed={SEED}\n")
            zf.writestr(f"{base}/entorno.json", entorno_json({
                "protocolo": "H.E.1", "n": params.get("n_points"),
                "R": params.get("n_rep"), "q": list(Q_LEVELS),
                "tau": TAU_CALIBRADO, "seed": SEED}))
            write_figs_zip(zf, f"{base}/figuras", st.session_state.get("he1_figs", {}))
        dl_btn("📦 Descargar TODO (ZIP)", zf_buf.getvalue(), f"he1_robustez_{base}.zip",
               "application/zip", width="stretch")
    except Exception as e:
        st.error(f"Error en la exportación: {e}")

methodology_expander("📖 Protocolo y compuertas — H.E.1", [
    ("Protocolo y compuertas", (
        "| Sub-hipótesis | Se mide | Instrumento | Criterio | Evidencia |\n"
        "|---|---|---|---|---|\n"
        "| H.E.1a | $d_B$ vs $2\\,d_H$ por réplica | Ripser + Bottleneck | 100 % de las réplicas | Fig. 5.2(b), 5.3 |\n"
        "| H.E.1b | exactitud TDA vs euclidiano | McNemar | $p < 0.05$ | Tabla 5.5 |\n"
        "| H.E.1c | $q^*$ (acierto de $\\beta_k$) | barrido de ruido | $q^* \\in [0.05, 0.10]$ | Fig. 5.2(a1–a3) |\n"
    )),
    ("Compuertas (Cap. IV)", "Calibración de $\\tau$ (F3) **antes** de medir; la cota $d_B \\le 2\\,d_H$ es bloqueante (F4) y la concordancia Euler–GUDHI valida el indicador."),
], "H.E.1")
st.markdown("---")
apply_mpl_theme()
