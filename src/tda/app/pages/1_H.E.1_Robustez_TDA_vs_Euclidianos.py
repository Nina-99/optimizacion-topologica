"""Página 1: H.E.1 — Robustez topológica (TDA).

Presenta la validación en el ORDEN del documento:
  0. Calibración previa del umbral τ (Figura 5.1)
  1. H.E.1a — estabilidad de los diagramas (Figuras 5.3 y 5.2(b))
  2. H.E.1b — ventaja comparativa TDA vs euclidiano (contraste de McNemar)
  3. H.E.1c — rango de fiabilidad de βk (Figura 5.2(a))
  4. Veredicto H.E.1 (Tabla 5.5)

Constantes y notación tomadas del documento:
  - Catálogo de 7 nubes (§1.5.2, Tabla 4.3).
  - q ∈ {0, 0.05, 0.10, 0.15, 0.20}; σ = q·diam(X) (Tabla 1.1, §2.2.1).
  - Umbral τ = 0.15 (calibrado, Fig. 5.1 / §5.4.1).
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
from tda.app.theme import (
    apply_mpl_theme, apply_plotly_theme, responsive_style,
    methodology_expander, page_header,
)
from tda.simulation.he_protocols import (
    protocolo_he1, calibrar_tau, CATALOGO, SUBCONJUNTO_VALIDADO,
    Q_LEVELS, TAU_CALIBRADO, SEED,
)

# ── Constantes del documento ─────────────────────────────────────────
N_MIN = 400                   # §1.5.2: densidad mínima (toro detectable)
R_DEFECTO = 20               # UI práctica; protocolo del documento: R = 100
COTA_FRACCION = 1.0           # H.E.1a: 100 % de las réplicas
ACC_BAJO = 0.95               # H.E.1c: umbral superior
ACC_ALTO = 0.50               # H.E.1c: umbral inferior
Q_STAR_RANGO = (0.05, 0.10)   # H.E.1c


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
st.header("H.E.1 — Robustez Topológica vs Descriptores Euclidianos")

st.markdown("---")
st.markdown("### 🎯 Hipótesis H.E.1")
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**H.E.1a — estabilidad de los diagramas**")
    st.latex(r"d_{B}\bigl(\mathrm{Dgm}_{k}^{\mathrm{VR}}(X),\ \mathrm{Dgm}_{k}^{\mathrm{VR}}(Y)\bigr) \le 2\,d_{H}(X,Y),\quad k=0,1")
    st.caption("Se satisface en el **100 %** de las réplicas, con ruido gaussiano $\\sigma = q\\,\\mathrm{diam}(X)$. Es la cota del Teorema de Estabilidad (Cohen-Steiner et al.).")
with col_b:
    st.markdown("**H.E.1b — ventaja comparativa (TDA vs euclidiano)**")
    st.caption("Disco vs anillo bajo deformación afín aleatoria. El descriptor de persistencia supera al euclidiano (PCA + inercia de k-medias + dispersión radial) con $p<0.05$ (McNemar).")

st.markdown("**H.E.1c — rango de fiabilidad de $\\beta_k$**")
st.latex(r"\exists\, q^{*}:\ \beta_{k}^{(\tau)}(X)=\beta_{k}^{\mathrm{teo}}\ \text{en} \ge 95\%\ (q<q^{*}),\quad < 50\%\ (q>q^{*})")
st.markdown("---")

ejecutar = st.button("▶ Ejecutar protocolo H.E.1", type="primary")

# ── 0. Calibración previa del umbral τ (Figura 5.1) ──────────────────
st.subheader("Paso previo — Figura 5.1: Curva de calibración del umbral relativo de persistencia $\\tau$")
st.caption("El eje vertical cuenta cuántas de las siete nubes de referencia reciben la lectura $\\beta_k$ correcta. La franja verde marca la ventana en que el acierto es total.")
if st.button("Calibrar τ (Figura 5.1)", key="btn_calib_tau"):
    with st.spinner("Calibrando contra el patrón de oro (7 nubes)..."):
        taus, aciertos = calibrar_tau(n_points, SEED)
        st.session_state.he1_calib = {"taus": taus, "aciertos": aciertos}

calib = st.session_state.get("he1_calib")
if calib:
    taus, aciertos = calib["taus"], calib["aciertos"]
    acierto_max = max(aciertos) if aciertos else 0
    ventana = [t for t, a in zip(taus, aciertos) if a == acierto_max]
    fig51 = go.Figure()
    if ventana:
        fig51.add_vrect(x0=min(ventana), x1=max(ventana), fillcolor="green", opacity=0.15, line_width=0)
    fig51.add_trace(go.Scatter(x=taus, y=aciertos, mode="lines+markers",
                               line=dict(color="royalblue"),
                               name=f"Nubes con lectura β₁ correcta (de {len(CATALOGO)})"))
    if ventana:
        fig51.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                   marker=dict(color="green", opacity=0.4, symbol="square", size=12),
                                   name=f"Ventana válida [{min(ventana):.2f}, {max(ventana):.2f}]"))
    fig51.add_trace(go.Scatter(x=[TAU_CALIBRADO, TAU_CALIBRADO], y=[0, len(CATALOGO)],
                               mode="lines", line=dict(dash="dash", color="white"),
                               name=f"τ adoptado = {TAU_CALIBRADO}"))
    fig51.update_layout(height=380, template="plotly_white",
                        xaxis_title="Umbral relativo de persistencia (τ)",
                        yaxis_title=f"Nubes con lectura β₁ correcta (de {len(CATALOGO)})",
                        xaxis=dict(range=[0.05, 0.40], dtick=0.05),
                        yaxis=dict(range=[0, len(CATALOGO)], dtick=1))
    apply_plotly_theme(fig51)
    st.plotly_chart(fig51, width="stretch")
    st.session_state.setdefault("he1_figs", {})["fig51_calibracion_tau"] = fig51
    if ventana:
        dentro = min(ventana) <= TAU_CALIBRADO <= max(ventana)
        (st.success if dentro else st.warning)(
            f"Ventana válida (acierto máximo {acierto_max}/{len(CATALOGO)}): τ ∈ [{min(ventana):.2f}, {max(ventana):.2f}]. "
            f"τ adoptado = {TAU_CALIBRADO} {'cae dentro' if dentro else 'queda FUERA'}."
        )
    else:
        st.warning("Sin ventana válida en el rango evaluado.")
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

    # Figura 5.3 — dB/diam promediada por nube y nivel vs cota 2q (según el PDF)
    st.markdown("**Figura 5.3:** Distancia bottleneck normalizada por el diámetro, promediada por nube y nivel de ruido, contrastada con la cota $2q$.")
    fig53 = go.Figure()
    for shape in CATALOGO:
        fig53.add_trace(go.Scatter(x=qs, y=sweep[shape]["ratio_diam_medio"], mode="lines+markers", name=shape))
    fig53.add_trace(go.Scatter(x=qs, y=[2 * q for q in qs], mode="lines",
                               line=dict(dash="dash", color="white"), name="Cota 2q"))
    fig53.update_layout(height=380, template="plotly_white",
                        xaxis_title="Nivel relativo de ruido (q)",
                        yaxis_title="dB / diam(X)",
                        xaxis=dict(range=[min(qs), max(qs)]))
    apply_plotly_theme(fig53)
    st.plotly_chart(fig53, width="stretch")
    st.session_state.setdefault("he1_figs", {})["fig53_bottleneck_2q"] = fig53

    # Figura 5.2(b) — cada réplica: dB vs 2 dH
    st.markdown("**Figura 5.2(b):** Cada punto es una réplica; se contrasta la distancia bottleneck $d_B$ contra la cota $2\\,d_H$. Ninguna réplica entra en la región prohibida.")
    fig52b = go.Figure()
    fig52b.add_trace(go.Scatter(
        x=[c["dH"] for c in crudo], y=[c["dB"] for c in crudo],
        mode="markers", marker=dict(size=6, color="royalblue", opacity=0.5),
        name="Réplicas",
    ))
    max_dh = max([c["dH"] for c in crudo] + [1e-9])
    fig52b.add_trace(go.Scatter(x=[0, max_dh * 1.1], y=[0, 2 * max_dh * 1.1],
                                mode="lines", line=dict(dash="dash", color="white"),
                                name="Cota dB = 2 dH"))
    fig52b.add_shape(type="rect", x0=0, x1=max_dh * 1.1, y0=2 * max_dh * 1.1,
                     y1=2.6 * max_dh * 1.1, fillcolor="red", opacity=0.10, line_width=0)
    fig52b.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                marker=dict(color="red", opacity=0.3, symbol="square", size=12),
                                name="Región prohibida (dB > 2 dH)"))
    fig52b.add_annotation(x=max_dh * 0.55, y=2.3 * max_dh * 1.1,
                          text="Región prohibida", showarrow=False, font=dict(color="red"))
    fig52b.update_layout(height=380, template="plotly_white",
                         xaxis_title="Distancia Hausdorff dH",
                         yaxis_title="Distancia Bottleneck dB")
    apply_plotly_theme(fig52b)
    st.plotly_chart(fig52b, width="stretch")
    st.session_state.setdefault("he1_figs", {})["fig52b_cota_estabilidad"] = fig52b

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
    st.markdown("**Figura 5.2(a):** Tasa de acierto de $\\beta_k$ frente al nivel relativo de ruido $q$, por nube de referencia.")
    fig52a = go.Figure()
    # zonas de fiabilidad (según el PDF: fiable q ≤ 0.05, no fiable q ≥ 0.15)
    fig52a.add_vrect(x0=min(qs), x1=0.05, fillcolor="green", opacity=0.12, line_width=0)
    fig52a.add_vrect(x0=0.15, x1=max(qs), fillcolor="red", opacity=0.10, line_width=0)
    fig52a.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                marker=dict(color="green", opacity=0.4, symbol="square", size=12),
                                name="Fiable (q ≤ 0.05)"))
    fig52a.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                marker=dict(color="red", opacity=0.4, symbol="square", size=12),
                                name="No fiable (q ≥ 0.15)"))
    for shape in CATALOGO:
        fig52a.add_trace(go.Scatter(x=qs, y=sweep[shape]["acierto"], mode="lines+markers", name=shape))
    fig52a.add_hline(y=ACC_BAJO, line=dict(dash="dot", color="green"),
                     annotation_text="95 %")
    fig52a.add_hline(y=ACC_ALTO, line=dict(dash="dot", color="red"),
                     annotation_text="50 %")
    fig52a.update_layout(height=380, template="plotly_white",
                         xaxis_title="Nivel relativo de ruido (q)",
                         yaxis_title="Tasa de acierto de β₁",
                         xaxis=dict(range=[min(qs), max(qs)]), yaxis=dict(range=[0, 1]))
    apply_plotly_theme(fig52a)
    st.plotly_chart(fig52a, width="stretch")
    st.session_state.setdefault("he1_figs", {})["fig52a_tasa_acierto"] = fig52a

    # q* = máx{q : tasa de acierto ≥ 0.95} sobre el subconjunto validado (Tabla 5.5, nota a)
    acc_all = [float(np.mean([sweep[s]["acierto"][i] for s in CATALOGO])) for i in range(len(qs))]
    acc_subs = [float(np.mean([sweep[s]["acierto"][i] for s in SUBCONJUNTO_VALIDADO])) for i in range(len(qs))]
    st.caption("Criterio de acierto evaluado sobre el subconjunto validado (círculo, disco, anillo), "
               "según la nota (a) de la Tabla 5.5. La Fig. 5.2(a) muestra las 7 nubes. "
               "Se mide **β₁** (el invariante que τ filtra); el documento lo enuncia sobre βk — ver Item 13 "
               "del archivo de correcciones.")
    candidatos = [q for q, a in zip(qs, acc_subs) if a >= ACC_BAJO]
    q_star = max(candidatos) if candidatos else None
    q_bajo_colapso = next((q for q, a in zip(qs, acc_subs) if q > 0 and a < ACC_ALTO), None)
    st.dataframe(pd.DataFrame({"q": qs,
                               "Acierto β₁ (validado)": [f"{a:.3f}" for a in acc_subs],
                               "Acierto β₁ (catálogo)": [f"{a:.3f}" for a in acc_all],
                               "≥ 0.95": ["✅" if a >= ACC_BAJO else "—" for a in acc_subs],
                               "< 0.50": ["✅" if a < ACC_ALTO else "—" for a in acc_subs]}),
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
    acc_bajo = float(np.mean([a for q, a in zip(qs, acc_subs) if q <= 0.05])) if any(q <= 0.05 for q in qs) else None
    acc_alto = float(np.mean([a for q, a in zip(qs, acc_subs) if q >= 0.15])) if any(q >= 0.15 for q in qs) else None
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
        from tda.app.download_utils import download_button as dl_btn, write_figs_zip
        df_crudo = pd.DataFrame(crudo)
        zf_buf = io.BytesIO()
        with zipfile.ZipFile(zf_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            base = f"he1_n{params.get('n_points')}_R{params.get('n_rep')}"
            zf.writestr(f"{base}/replicas_dB_dH.csv", df_crudo.to_csv(index=False))
            zf.writestr(f"{base}/tabla_5_5.csv", pd.DataFrame(filas55).to_csv(index=False))
            zf.writestr(f"{base}/params.txt",
                        f"n={params.get('n_points')}\nR={params.get('n_rep')}\n"
                        f"q={Q_LEVELS}\ntau={TAU_CALIBRADO}\nseed={SEED}\n")
            write_figs_zip(zf, f"{base}/figuras", st.session_state.get("he1_figs", {}))
        dl_btn("📦 Descargar TODO (ZIP)", zf_buf.getvalue(), f"he1_robustez_{base}.zip",
               "application/zip", width="stretch")
    except Exception as e:
        st.error(f"Error en la exportación: {e}")

methodology_expander("📖 Protocolo y compuertas — H.E.1", [
    ("Protocolo y compuertas", (
        "| Sub-hipótesis | Se mide | Instrumento | Criterio | Evidencia |\n"
        "|---|---|---|---|---|\n"
        "| H.E.1a | $d_B$ vs $2\\,d_H$ por réplica | Ripser + Bottleneck | 100 % de las réplicas | Figs. 5.2(b), 5.3 |\n"
        "| H.E.1b | exactitud TDA vs euclidiano | McNemar | $p < 0.05$ | Tabla 5.5 |\n"
        "| H.E.1c | $q^*$ (acierto de $\\beta_k$) | barrido de ruido | $q^* \\in [0.05, 0.10]$ | Fig. 5.2(a) |\n"
    )),
    ("Compuertas (Cap. IV)", "Calibración de $\\tau$ (F3) **antes** de medir; la cota $d_B \\le 2\\,d_H$ es bloqueante (F4) y la concordancia Euler–GUDHI valida el indicador."),
], "H.E.1")
st.markdown("---")
apply_mpl_theme()
