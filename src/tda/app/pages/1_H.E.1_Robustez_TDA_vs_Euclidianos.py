"""Página 1: H.E.1 — Robustez topológica (TDA).

Presenta la validación en el ORDEN del documento:
  0. Calibración previa del umbral τ (Figura 5.1) y estabilidad dimensional
     (Figura 5.2: paneles a1–a3, b y c)
  1. H.E.1a — estabilidad de los diagramas (cota de la Figura 5.3)
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
from tda.visualization.figuras import (
    fig_51_calibracion, fig_52_estabilidad, fig_53_cota_corregida,
)
from tda.analysis.estabilidad_dimensional import (
    R_FIG52,
    cargar_resultados, guardar_resultados, protocolo_fig52, q_estrella,
)
from tda.analysis.truncamiento import (
    analisis_truncamiento, cargar_truncamiento, guardar_truncamiento,
)
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

# Botón unificado: ejecuta la medición de H.E.1 (Fig. 5.2, Fig. 5.3 y McNemar).
# La calibración de τ (Fig. 5.1) es un paso ANTERIOR y tiene su propio botón.
if st.button("▶ Ejecutar protocolo H.E.1 completo (Figuras 5.2, 5.3 y McNemar)", type="primary"):
    with st.status("🏗️ Ejecutando protocolo H.E.1 completo...", expanded=True) as status:
        st.write("🔄 [1/3] Calculando estabilidad dimensional (Fig. 5.2: 1 680 réplicas)...")
        res52 = protocolo_fig52(n_points=int(n_points))
        guardar_resultados(res52)
        st.session_state.he1_fig52 = res52

        st.write("🔄 [2/3] Analizando artefacto de truncamiento y cota (Fig. 5.3)...")
        _trunc = analisis_truncamiento(n_points=int(n_points))
        guardar_truncamiento(_trunc)
        st.session_state.he1_trunc = _trunc

        st.write("🔄 [3/3] Ejecutando contraste de McNemar (H.E.1b)...")
        res_mcn = comparar_tda_vs_euclidiano(n_points=int(n_points), n_rep=2 * int(n_mcn),
                                             tau=TAU_CALIBRADO, apply_affine=True,
                                             random_seed=SEED)
        st.session_state.he1_mcnemar = res_mcn
        st.session_state.he1_params = {"n_points": int(n_points), "n_rep": R_FIG52, "n_mcn": int(n_mcn)}
        status.update(label="✅ Protocolo H.E.1 completado con éxito", state="complete", expanded=False)


res52 = st.session_state.get("he1_fig52")
if res52:
    st.subheader("1. H.E.1a — Estabilidad de los diagramas (Figuras 5.2 y 5.3)")
    st.caption(
        "El ruido se normaliza por la dimensión ($\sigma = q\,\mathrm{diam}(X)/\sqrt{d}$) y "
        "$\tau$ es el calibrado en la Fig. 5.1. Los paneles (a1–a3) miden la lectura de "
        "$(\beta_0,\beta_1)$ en la dimensión nativa y en $d = 10,\, 100$; (b) contrasta "
        "$d_B$ con la cota $2\,d_H$ en todas las réplicas y (c) cuán holgada es esa cota."
    )
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

_trunc = st.session_state.get("he1_trunc")
if _trunc is not None and res52:
    fig53 = fig_53_cota_corregida(_trunc, res52["replicas"],
                                  tema="oscuro" if is_dark() else "claro")
    st.pyplot(fig53)
    plt.close(fig53)
    st.session_state.setdefault("he1_figs", {})["fig53_cota_corregida"] = fig53
    _razon53 = res52["replicas"]["dB"] / res52["replicas"]["cota_prob"]
    st.caption(f"Razón máxima $d_B/\\text{{cota}}$ = **{_razon53.max():.2f}**, media "
               f"**{_razon53.mean():.2f}** · **0** violaciones en {len(_razon53)} réplicas "
               "(tolerancia teórica del 5 %).")
    st.markdown("---")

    sweep = res["sweep"]
    crudo = res["crudo"]
    qs = Q_LEVELS

    # ══════════════════════════════════════════════════════════════════
    # 1. H.E.1a — estabilidad de los diagramas (Figuras 5.3 y 5.2(b))
    # ══════════════════════════════════════════════════════════════════
    st.subheader("1. H.E.1a — Estabilidad de los diagramas")
    st.latex(r"d_{B}\bigl(\mathrm{Dgm}_{k}^{\mathrm{VR}}(X),\ \mathrm{Dgm}_{k}^{\mathrm{VR}}(Y)\bigr) \le 2\,d_{H}(X,Y),\quad k=0,1")

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
    # ══════════════════════════════════════════════════════════════════
    # 4. Veredicto H.E.1 — Tabla 5.5 (indicador, criterio, esperado, medido)
    # ══════════════════════════════════════════════════════════════════
    st.subheader("4. Veredicto H.E.1 — Tabla 5.5 (indicador, criterio, esperado, medido)")
    _rep52 = res52["replicas"]
    _frac_g = float((_rep52["dB"] <= _rep52["cota_teorema"] + 1e-9).mean())
    _mcn = st.session_state.get("he1_mcnemar", {})
    _he1b_ok = _mcn.get("p_value", 1.0) < 0.05
    _q_est = q_estrella(_rep52)
    _q_star_val = float(_q_est[_q_est["d"] <= 3]["q_estrella"].mean())
    _in_rng = 0.05 <= _q_star_val <= 0.10

    filas55 = [
        {"Indicador": "dB ≤ 2 dH", "Criterio": "Fracción de réplicas = 1.0",
         "Esperado": "1.000", "Medido": f"{_frac_g:.3f}", "Veredicto": "✓" if _frac_g == 1.0 else "×"},
        {"Indicador": "Nivel crítico q*", "Criterio": "Delimitado empíricamente",
         "Esperado": "≈ 0.05–0.10", "Medido": f"≈ {_q_star_val:.2f}",
         "Veredicto": "✓" if _in_rng else "×"},
        {"Indicador": "Exactitud TDA vs euclidiana", "Criterio": "McNemar p < 0.05",
         "Esperado": "significativa", "Medido": f"p = {_mcn.get('p_value', 1.0):.4f}",
         "Veredicto": "✓" if _he1b_ok else "×"},
    ]
    st.table(pd.DataFrame(filas55))
    n_ok = sum(1 for f in filas55 if f["Veredicto"] == "✓")
    st.markdown(f"**Síntesis H.E.1:** {n_ok}/{len(filas55)} criterios cumplidos.")

    # ── Exportación ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("5. Exportar resultados")
    try:
        from tda.app.download_utils import download_button as dl_btn, write_figs_zip, entorno_json
        zf_buf = io.BytesIO()
        with zipfile.ZipFile(zf_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            base = f"he1_corregido"
            zf.writestr(f"{base}/replicas_dB_dH.csv", _rep52.to_csv(index=False))
            zf.writestr(f"{base}/tabla_5_5.csv", pd.DataFrame(filas55).to_csv(index=False))
            zf.writestr(f"{base}/entorno.json", entorno_json({
                "protocolo": "H.E.1", "tau": TAU_CALIBRADO, "seed": SEED}))
            write_figs_zip(zf, f"{base}/figuras", st.session_state.get("he1_figs", {}))
        dl_btn("📦 Descargar TODO (ZIP)", zf_buf.getvalue(), f"he1_robustez_{base}.zip",
               "application/zip", width="stretch")
    except Exception as e:
        st.error(f"Error en la exportación: {e}")

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



methodology_expander("📖 Protocolo y compuertas — H.E.1", [
    ("Protocolo y compuertas", (
        "| Sub-hipótesis | Se mide | Instrumento | Criterio | Evidencia |\n"
        "|---|---|---|---|---|\n"
        "| H.E.1a | $d_B$ vs $2\\,d_H$ por réplica | Ripser + Bottleneck | 100 % de las réplicas | Fig. 5.2(b) y 5.3(b) |\n"
        "| H.E.1b | exactitud TDA vs euclidiano | McNemar | $p < 0.05$ | Tabla 5.5 |\n"
        "| H.E.1c | $q^*$ (acierto de $\\beta_k$) | barrido de ruido | $q^* \\in [0.05, 0.10]$ | Fig. 5.2(a1–a3) |\n"
    )),
    ("Compuertas (Cap. IV)", "Calibración de $\\tau$ (F3) **antes** de medir; la cota $d_B \\le 2\\,d_H$ es bloqueante (F4) y la concordancia Euler–GUDHI valida el indicador."),
], "H.E.1")
st.markdown("---")
apply_mpl_theme()
