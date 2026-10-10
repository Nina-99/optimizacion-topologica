"""Página 3: H.G. — Hipótesis General.

Ejecuta automáticamente los protocolos de H.E.1 y H.E.2 (a través de
`tda.simulation.he_protocols`), sintetiza los veredictos de las hipótesis
específicas (H.E.1a/b/c, H.E.2a/b) y corrobora la hipótesis general del
Cap. I (§1.5). NO usa μ_α ni regularización en el bucle (fuera de alcance).
"""

import io
import zipfile

import numpy as np
import pandas as pd
import streamlit as st

from tda.analysis.stability import comparar_tda_vs_euclidiano
from tda.simulation.he_protocols import (
    protocolo_he1, calibrar_tau, corrida_simp, barrido_he2,
    Q_LEVELS, TAU_CALIBRADO, SEED, FV, P_DEFECTO, RMIN_DEFECTO,
    CATALOGO, SUBCONJUNTO_VALIDADO,
)
from tda.app.download_utils import download_button
from tda.app.theme import responsive_style, methodology_expander, page_header

st.set_page_config(page_title="H.G. — Hipótesis General", layout="wide", page_icon="🧭")
st.markdown(responsive_style(), unsafe_allow_html=True)

Q_STAR_RANGO = (0.05, 0.10)
DOMINIO = "Rectangular (viga en voladizo, 2:1)"

# ── Header + hipótesis general (§1.5) ────────────────────────────────
st.markdown(page_header("H.G. — Hipótesis General", "Ejecuta H.E.1 y H.E.2 y sintetiza los veredictos"),
            unsafe_allow_html=True)
st.subheader("🎯 Hipótesis general (Cap. I, §1.5)")
st.markdown(
    "La aplicación sistemática de estructuras topológicas —mediante TDA y SIMP— proporciona "
    "**descriptores estables y globalmente consistentes** para los problemas planteados en entornos "
    "complejos de ingeniería. Su ventaja sobre los métodos basados exclusivamente en geometría "
    "euclidiana **se manifiesta en aquellos indicadores que el teorema de estabilidad respalda "
    "formalmente, y no de manera indiscriminada**."
)

# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.header("🧭 H.G. — Hipótesis General")
n_points = st.sidebar.number_input("Puntos por nube (H.E.1)", min_value=400, max_value=1000, value=400, step=50)
n_rep = st.sidebar.slider("Réplicas por nube y nivel (R)", min_value=10, max_value=100, value=20, step=10,
                          help="Protocolo del documento: R = 100 (más lento).")
n_mcn = st.sidebar.slider("Muestras por clase (H.E.1b ≥ 200)", min_value=200, max_value=600, value=200, step=50)

ejecutar = st.button("▶ Ejecutar H.E.1 y H.E.2 automáticamente", type="primary")

if ejecutar:
    # ── H.E.1 ──
    with st.spinner("H.E.1 — calibración de τ, protocolo de estabilidad y contraste de McNemar..."):
        taus, aciertos = calibrar_tau(n_points, SEED)
        sweep, crudo = protocolo_he1(n_points, Q_LEVELS, n_rep, TAU_CALIBRADO, SEED)
        mcn = comparar_tda_vs_euclidiano(n_points=n_points, n_rep=2 * n_mcn,
                                         tau=TAU_CALIBRADO, apply_affine=True, random_seed=SEED)
    st.session_state.he1_calib = {"taus": taus, "aciertos": aciertos}
    st.session_state.he1_results = {"sweep": sweep, "crudo": crudo, "mcnemar": mcn}
    st.session_state.he1_params = {"n_points": n_points, "n_rep": n_rep, "n_mcn": n_mcn}
    st.session_state.tda_he1_run = True

    # ── H.E.2 ──
    with st.spinner("H.E.2 — optimización SIMP base y barrido multi-configuración de H.E.2b..."):
        m = corrida_simp(60, 30, P_DEFECTO, RMIN_DEFECTO, max_iter=200)
        c_base = float(m.c_hist[0])
        reduccion = (c_base - m.c_final) / c_base * 100
        sweep2 = barrido_he2(DOMINIO, 60, 30, P_DEFECTO, RMIN_DEFECTO, max_iter=200)
    st.session_state.simp_optimized = True
    st.session_state.simp_c_base = c_base
    st.session_state.simp_c_final = m.c_final
    st.session_state.simp_reduccion = reduccion
    st.session_state.simp_beta0 = m.beta0
    st.session_state.simp_beta1 = m.beta1
    st.session_state.simp_concordancia = getattr(m, "_betti_concordancia", None)
    st.session_state.sweep_results = sweep2
    st.session_state.sweep_done = True

# ── Cálculo de veredictos desde session_state ────────────────────────
he1_run = st.session_state.get("tda_he1_run", False) and bool(st.session_state.get("he1_results"))
he2_run = st.session_state.get("simp_optimized", False)
sweep_done = st.session_state.get("sweep_done", False)

he1a_ok = he1b_ok = he1c_ok = he2a_ok = he2b_ok = None
he1a_val = he1b_val = he1c_val = he2a_val = he2b_val = "—"

if he1_run:
    sw = st.session_state.he1_results.get("sweep", {})
    if sw:
        frac = float(np.mean([sw[s]["cumple_frac"][i] for s in sw for i in range(len(Q_LEVELS))]))
        he1a_ok = frac == 1.0
        he1a_val = f"{frac:.3f}"
        acc_subs = [float(np.mean([sw[s]["acierto"][i] for s in SUBCONJUNTO_VALIDADO if s in sw]))
                    for i in range(len(Q_LEVELS))]
        cand = [q for q, a in zip(Q_LEVELS, acc_subs) if a >= 0.95]
        q_star = max(cand) if cand else None
        he1c_ok = q_star is not None and Q_STAR_RANGO[0] <= q_star <= Q_STAR_RANGO[1]
        he1c_val = f"q* ≈ {q_star:.2f}" if q_star is not None else "no detectado"
    mc = st.session_state.he1_results.get("mcnemar", {})
    if mc:
        mm = mc.get("mc_nemar", {})
        he1b_ok = bool(mm.get("significant")) and (mc.get("exactitud_tda", 0) > mc.get("exactitud_eucl", 0))
        he1b_val = f"{mc.get('exactitud_tda', float('nan')):.3f} vs {mc.get('exactitud_eucl', float('nan')):.3f}"

if he2_run:
    red = st.session_state.get("simp_reduccion", 0.0)
    he2a_ok = red >= 40.0
    he2a_val = f"{red:.2f} %"
    if sweep_done:
        df = pd.DataFrame(st.session_state.get("sweep_results", []))
        if len(df) > 0:
            dfp = df[(df["Config"].str.startswith("p=")) & (df["p"] >= 2)]   # p ∈ {2,3,4}
            dfm = df[df["Config"].str.startswith("malla")]
            dfr = df[df["Config"].str.startswith("rmin")]
            inv = (len(set(dfm["beta1"])) <= 1 and len(set(dfp["beta1"])) <= 1 and len(set(dfr["beta1"])) <= 1)
            conc = bool((df["Concordancia"] == "Si").all())
            he2b_ok = inv and conc
            he2b_val = f"β₁={sorted(set(df['beta1']))} · Euler≡GUDHI {'✓' if conc else '×'}"

# ── Semáforo de hipótesis ────────────────────────────────────────────
def _chip(label, ok):
    color, icon = (("#27ae60", "✓") if ok is True
                   else ("#e74c3c", "×") if ok is False
                   else ("#7f8c8d", "—"))
    return (f'<span style="background:{color};color:#fff;border-radius:6px;'
            f'padding:5px 12px;margin:3px;font-weight:700;display:inline-block;">{label} {icon}</span>')

_hg = None
if he1_run and he2_run:
    _hg = all(s is True for s in [he1a_ok, he1c_ok, he2a_ok, he2b_ok])
chips = "".join(_chip(lbl, ok) for lbl, ok in
                [("H.E.1a", he1a_ok), ("H.E.1b", he1b_ok), ("H.E.1c", he1c_ok),
                 ("H.E.2a", he2a_ok), ("H.E.2b", he2b_ok), ("H.G.", _hg)])
st.markdown("#### 🚦 Semáforo de hipótesis")
st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)
st.markdown("---")

# ── Tabla 5.9 ────────────────────────────────────────────────────────
st.subheader("Tabla 5.9 — Veredicto sobre cada hipótesis")
filas = [
    {"Hipótesis": "H.E.1a", "Afirmación": "dB ≤ 2·dH (k=0,1), 100 % réplicas",
     "Valor": he1a_val, "Veredicto": "✓" if he1a_ok else ("×" if he1a_ok is False else "—"),
     "Evidencia": "Figs. 5.2(b), 5.3"},
    {"Hipótesis": "H.E.1b", "Afirmación": "TDA > euclidiano (McNemar p<0.05)",
     "Valor": he1b_val, "Veredicto": "✓" if he1b_ok else ("×" if he1b_ok is False else "—"),
     "Evidencia": "Tabla 5.5"},
    {"Hipótesis": "H.E.1c", "Afirmación": "Existe q* separador ∈ [0.05, 0.10]",
     "Valor": he1c_val, "Veredicto": "✓" if he1c_ok else ("×" if he1c_ok is False else "—"),
     "Evidencia": "Fig. 5.2(a)"},
    {"Hipótesis": "H.E.2a", "Afirmación": "Reducción ≥ 40 % vs uniforme",
     "Valor": he2a_val, "Veredicto": "✓" if he2a_ok else ("×" if he2a_ok is False else "—"),
     "Evidencia": "Fig. 5.4(b)"},
    {"Hipótesis": "H.E.2b", "Afirmación": "β₁ invariante (malla, p, rmin) + Euler≡GUDHI",
     "Valor": he2b_val, "Veredicto": "✓" if he2b_ok else ("×" if he2b_ok is False else "—"),
     "Evidencia": "Figs. 5.5, 5.6, 5.7"},
]
st.table(pd.DataFrame(filas))

if not (he1_run and he2_run):
    st.info("Pulsá **«Ejecutar H.E.1 y H.E.2 automáticamente»** para correr ambos protocolos y completar la síntesis. "
            "También podés ejecutarlos por separado en las pestañas 1 y 2.")

# ── Comparación estructural TDA vs SIMP (Tabla 5.3) ──────────────────
st.markdown("---")
st.subheader("Comparación estructural de los dos casos (Tabla 5.3)")
tabla53 = pd.DataFrame({
    "Aspecto": ["Objeto de entrada", "Estructura combinatoria", "Parámetro de filtración",
                "Instrumento", "Invariante leído", "Fuente de perturbación",
                "Criterio de estabilidad", "Qué garantiza la teoría", "Hipótesis asociada"],
    "Caso 1 — TDA": ["Nube finita X ⊂ ℝᵈ", "Complejo simplicial (Vietoris–Rips)", "Radio ε de las bolas",
                     "Ripser 0.6", "β₀, β₁ persistentes", "Ruido gaussiano de nivel q",
                     "dB ≤ 2 dH (teorema)", "Estabilidad del diagrama",
                     "H.E.1a, H.E.1b, H.E.1c"],
    "Caso 2 — SIMP": ["Campo de densidades ρ : Ω → [0,1]", "Complejo cúbico (malla de elementos)",
                      "Umbral u de binarización", "GUDHI 3.13 + conteo de Euler",
                      "β₀, β₁ del sólido", "Resolución de malla y radio rmin",
                      "Invariancia frente a discretización (empírica)", "Existencia de mínimo (vía filtro compacto)",
                      "H.E.2a, H.E.2b"],
})
st.dataframe(tabla53, width="stretch", hide_index=True)

# ── Hilo conductor ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("Hilo conductor — un rasgo es real si persiste")
st.markdown(
    "En el Caso 1 la filtración es el radio ε; en el Caso 2, el umbral u. Un rasgo topológico se acepta "
    "como real si **persiste a lo largo de la filtración**, y ese criterio unificado es lo que articula "
    "ambos dominios (complejo simplicial y complejo cúbico son instancias del mismo complejo de cadenas)."
)

# ── Veredicto H.G. ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("Veredicto H.G. — Hipótesis General")
sostiene = [he1a_ok, he1c_ok, he2a_ok, he2b_ok]
if all(s is True for s in sostiene):
    st.success(
        "### ✅ H.G. CORROBORADA — los descriptores topológicos son estables y globalmente consistentes "
        "en los indicadores que el teorema de estabilidad respalda (dB ≤ 2dH, invariancia de β₁), y el "
        "método mejora la eficiencia estructural (≥ 40 %)."
    )
    if he1b_ok is False:
        st.info("H.E.1b NO se sostiene: la ventaja topológica NO es indiscriminada — exactamente lo que "
                "la hipótesis general matiza («no de manera indiscriminada»).")
    elif he1b_ok is True:
        st.info("H.E.1b además se sostiene bajo el método fiel (TDA > euclidiano).")
elif any(s is False for s in sostiene):
    st.error("### ❌ H.G. NO CORROBORADA — al menos un indicador respaldado por el teorema no se sostiene.")
else:
    st.info("Sin datos aún: ejecutá los protocolos para emitir el veredicto.")

# ── Estudio reproducible completo ────────────────────────────────────
st.markdown("---")
st.subheader("📦 Estudio reproducible")
st.caption("ZIP con tablas, datos crudos, figuras, parámetros y semilla — todo lo necesario para "
           "reejecutar y auditar el estudio (compuerta de reproducibilidad F6 del Cap. IV).")
try:
    from datetime import datetime
    import os as _os
    from tda.app.download_utils import write_figs_zip
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        readme = (
            "ESTUDIO REPRODUCIBLE — TDA-SIMP\n"
            "================================\n"
            f"Fecha: {datetime.now():%Y-%m-%d %H:%M}\n"
            f"Semilla maestra: {SEED}\n\n"
            f"H.E.1: catálogo de 7 nubes; q ∈ {Q_LEVELS}; τ = {TAU_CALIBRADO}; "
            f"n = {n_points}; R = {n_rep} réplicas; σ = q·diam(X).\n"
            "H.E.2: viga en voladizo 60×30; f_V = 0.5; p = 3; r_min = 2.4; "
            "elementos unitarios (E0 = 1, F = 1); filtro de densidad.\n\n"
            "Contenido:\n"
            "  tabla_5_9.csv / tabla_5_3.csv  — síntesis y comparación estructural\n"
            "  he1/replicas_dB_dH.csv         — réplicas crudas (dB, dH)\n"
            "  he2/barrido_HE2b.csv           — barrido de invariancia de β₁\n"
            "  figuras/*.png                  — figuras de la corrida\n"
            "  requirements.txt               — dependencias (versiones instaladas)\n"
        )
        z.writestr("estudio/README.txt", readme)
        z.writestr("estudio/tabla_5_9.csv", pd.DataFrame(filas).to_csv(index=False))
        z.writestr("estudio/tabla_5_3.csv", tabla53.to_csv(index=False))
        h1 = st.session_state.get("he1_results", {})
        if h1.get("crudo"):
            z.writestr("estudio/he1/replicas_dB_dH.csv", pd.DataFrame(h1["crudo"]).to_csv(index=False))
        if st.session_state.get("sweep_done"):
            z.writestr("estudio/he2/barrido_HE2b.csv",
                       pd.DataFrame(st.session_state.get("sweep_results", [])).to_csv(index=False))
        figs = {**st.session_state.get("he1_figs", {}), **st.session_state.get("he2_figs", {})}
        if figs:
            write_figs_zip(z, "estudio/figuras", figs)
        if _os.path.exists("requirements.txt"):
            z.writestr("estudio/requirements.txt", open("requirements.txt").read())
    download_button("📦 Descargar estudio completo (ZIP)", buf.getvalue(),
                    "estudio_tda_simp.zip", "application/zip", width="stretch")
except Exception as e:
    st.error(f"Error en la exportación: {e}")

methodology_expander("📖 Plan de Acción — H.G.", [
    ("Hipótesis general (§1.5)",
     "Estructuras topológicas → descriptores estables y globalmente consistentes; ventaja formal, no indiscriminada."),
    ("Síntesis", "H.E.1a · H.E.1b · H.E.1c · H.E.2a · H.E.2b (Tabla 5.9)."),
    ("Articulación TDA–SIMP",
     r"Complejo simplicial (VR) y complejo cúbico son instancias del mismo complejo de cadenas (C_\bullet, \partial_\bullet)."),
], "H.G.")
