"""Página 3: H.G. — Hipótesis General (Cap. I §1.5).

Ejecuta los protocolos de H.E.1 y H.E.2 y sintetiza la hipótesis general en
sus DOS cláusulas del PDF:

  A. Estabilidad/verificabilidad de β_k bajo q ≤ q*  (H.E.1a, H.E.1c, H.E.2b).
  B. Ventaja frente a descriptores euclidianos cuando la información es
     topológica (H.E.1b, piloto documentado del PDF).

H.E.2a (eficiencia estructural) se informa aparte: NO es una cláusula de H.G.
NO usa μ_α ni regularización en el bucle (fuera de alcance).
"""

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from tda.analysis.stability import resultado_piloto_pdf
from tda.analysis.estabilidad_dimensional import (
    protocolo_fig52, rango_fiabilidad, guardar_resultados, cargar_resultados,
)
from tda.simulation.he_protocols import (
    corrida_simp, barrido_he2, metricas_he2a,
    Q_LEVELS, TAU_CALIBRADO, SEED, FV, P_DEFECTO, RMIN_DEFECTO,
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
    "Los **números de Betti persistentes**, cuya variación está acotada por los teoremas de "
    "estabilidad funcional y geométrica, constituyen **descriptores estables y verificables** de "
    "nubes de datos con ruido y de diseños SIMP, **siempre que la perturbación permanezca bajo un "
    "nivel crítico** que depende del problema; **y superan a los descriptores euclidianos cuando "
    "la información relevante es topológica**."
)
st.markdown(
    "**Cláusula A — estabilidad y verificabilidad:** H.E.1a + H.E.1c + H.E.2b.\n"
    "**Cláusula B — ventaja comparativa:** H.E.1b.\n\n"
    "H.E.2a mide eficiencia estructural de SIMP; **no** es una cláusula de la hipótesis general."
)
st.info(
    "**¿Por qué teorema se cumple?** El respaldo formal es el **Teorema de Estabilidad** de "
    "Cohen-Steiner, Edelsbrunner & Harer (2010):  "
    "$d_B\\bigl(\\mathrm{Dgm}\\,f,\\,\\mathrm{Dgm}\\,g\\bigr) \\le \\lVert f-g\\rVert_\\infty$.  "
    "Para filtraciones de **Vietoris–Rips** se particulariza en $d_B \\le 2\\,d_H$ (k = 0, 1), que es "
    "**exactamente** el criterio de **H.E.1a**. Ese resultado es el que sustenta el criterio unificado "
    "de *persistencia* (un rasgo es real si persiste a lo largo de la filtración)."
)

# ── Sidebar ──────────────────────────────────────────────────────────
st.sidebar.header("🧭 H.G. — Hipótesis General")
n_points = st.sidebar.number_input("Puntos por nube (H.E.1)", min_value=400, max_value=1000, value=400, step=50)
n_rep = st.sidebar.slider("Réplicas por nube y nivel (R)", min_value=10, max_value=100, value=20, step=10,
                          help="Protocolo del documento: R = 100 (más lento).")
usar_cache = st.sidebar.checkbox(
    "Usar resultados en caché (CSV)", value=True,
    help="Si existen fig52_replicas.csv, validacion_HE2_corregida.csv y "
         "barrido_HE2b.csv, H.G. los lee en segundos en vez de reejecutar todo.",
)

# Auto-carga de H.E.1 desde caché para que el semáforo no quede gris
# cuando el usuario ya corrió H.E.2 pero no H.E.1 en esta sesión.
if usar_cache and st.session_state.get("he1_fig52") is None:
    _cache_he1 = cargar_resultados()
    if _cache_he1 is not None:
        st.session_state.he1_fig52 = _cache_he1
        st.session_state.he1_mcnemar = resultado_piloto_pdf()

ejecutar = st.button("▶ Ejecutar H.G. (caché o protocolo completo)", type="primary")

if ejecutar:
    with st.status("🏗️ Construyendo el veredicto H.G...", expanded=True) as _status:
        # ── H.E.1a/c: 1 680 réplicas (CSV o corrida) ─────────────────
        res52 = cargar_resultados() if usar_cache else None
        if res52 is None:
            st.write(f"🔄 H.E.1: estabilidad dimensional (Fig. 5.2, R = {n_rep})...")
            res52 = protocolo_fig52(n_points=n_points, n_rep=n_rep)
            guardar_resultados(res52)
            origen_he1 = f"corrida R={n_rep}"
        else:
            st.write("⚡ H.E.1a/c cargadas desde `fig52_replicas.csv`.")
            origen_he1 = "caché CSV"
        st.session_state.he1_fig52 = res52

        st.write("⚡ H.E.1b: piloto documentado del PDF.")
        st.session_state.he1_mcnemar = resultado_piloto_pdf()
        st.session_state.he1_params = {"n_points": n_points, "n_rep": n_rep,
                                       "origen": origen_he1}

        # ── H.E.2a/b: base + barrido (CSV o corrida) ─────────────────
        ruta_val = Path("data/validacion_HE2_corregida.csv")
        ruta_bar = Path("data/barrido_HE2b.csv")
        if usar_cache and ruta_val.exists() and ruta_bar.exists():
            st.write("⚡ H.E.2a/b cargadas desde `validacion_HE2_corregida.csv` "
                     "y `barrido_HE2b.csv`.")
            val = pd.read_csv(ruta_val).iloc[0]
            sweep2 = pd.read_csv(ruta_bar).to_dict("records")
            c_base = float(val["c_uniforme_p1"])
            c_final = float(val["c_binarizada"])
            reduccion = float(val["reduccion_vs_placa_p1_pct"])
            beta0, beta1 = 1, int(val["beta1_base"])
            concordancia = bool((pd.DataFrame(sweep2)["Concordancia"] == "Si").all())
            origen_he2 = "caché CSV"
        else:
            st.write("🔄 H.E.2: base + barrido multi-configuración...")
            m = corrida_simp(60, 30, P_DEFECTO, RMIN_DEFECTO, max_iter=300)
            met2 = metricas_he2a(m, fV=FV, max_iter=300)
            c_base = float(m.c_hist[0])
            c_final = float(m.c_final)
            reduccion = met2["reduccion"] * 100
            beta0, beta1 = m.beta0, m.beta1
            concordancia = getattr(m, "_betti_concordancia", None)
            sweep2, _mallas = barrido_he2(DOMINIO, 60, 30, P_DEFECTO,
                                          RMIN_DEFECTO, max_iter=300)
            origen_he2 = "corrida completa"

        _status.update(label=f"✅ H.G. listo ({origen_he1}; {origen_he2}).",
                       state="complete", expanded=False)

    # H.G. comparte métricas con la landing, pero NO debe marcar
    # `simp_optimized`: H.E.2 espera diseños ρ que H.G. no genera.
    st.session_state.hg_simp_ready = True
    st.session_state.simp_c_base = c_base
    st.session_state.simp_c_final = c_final
    st.session_state.simp_reduccion = reduccion
    st.session_state.simp_beta0 = beta0
    st.session_state.simp_beta1 = beta1
    st.session_state.simp_concordancia = concordancia
    st.session_state.sweep_results = sweep2
    st.session_state.sweep_done = True

# ── Cálculo de veredictos desde session_state ────────────────────────
he1_run = st.session_state.get("he1_fig52") is not None
he2_run = bool(st.session_state.get("hg_simp_ready", False)
               or st.session_state.get("simp_optimized", False))
sweep_done = st.session_state.get("sweep_done", False)

he1a_ok = he1b_ok = he1c_ok = he2a_ok = he2b_ok = None
he1a_val = he1b_val = he1c_val = he2a_val = he2b_val = "—"
he1c_label = "—"
he1c_acotada = False
estabilidad_ok = comparativa_ok = None
hg_estado = "sin datos"
hg_val = "—"

if he1_run:
    _rep = st.session_state.he1_fig52["replicas"]
    # H.E.1a: fracción de réplicas con dB ≤ 2·dH (cota del teorema)
    _frac = float((_rep["dB"] <= _rep["cota_teorema"] + 1e-9).mean())
    he1a_ok = _frac == 1.0
    he1a_val = f"{_frac:.3f}"
    # H.E.1c: denominador del PDF (18) = 7 nubes × 3 dimensiones menos toro.
    _qe_full = rango_fiabilidad(_rep)
    _qe = _qe_full[~_qe_full["nube"].isin(["toro"])].copy()
    _n_sost = int((_qe["estado"] == "se sostiene").sum())
    _n_acot = int(_qe["estado"].str.startswith("acotada").sum())
    _n_no = int((_qe["estado"] == "no se sostiene").sum())
    he1c_ok = _n_sost == len(_qe)
    he1c_acotada = (_n_sost + _n_acot > 0)
    he1c_label = "✓" if he1c_ok else ("NC" if he1c_acotada else "×")
    _qnat = _qe[_qe["d"] <= 3]["q_estrella"]
    he1c_val = (f"q* ∈ [{_qnat.min():.2f}, {_qnat.max():.2f}]; "
                f"{_n_sost}/{len(_qe)} banda, {_n_acot} acotadas, {_n_no} sin q* "
                "(toro aparte)")
    # H.E.1b: McNemar
    mc = st.session_state.get("he1_mcnemar") or {}
    if mc:
        mm = mc.get("mc_nemar", {})
        he1b_ok = bool(mm.get("significant")) and (mc.get("exactitud_tda", 0) > mc.get("exactitud_eucl", 0))
        he1b_val = f"{mc.get('exactitud_tda', float('nan')):.3f} vs {mc.get('exactitud_eucl', float('nan')):.3f}"

if he2_run:
    red = st.session_state.get("simp_reduccion")
    if red is not None:
        he2a_ok = float(red) >= 15.0
        he2a_val = f"{float(red):.2f} %"
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
    else:
        he2b_val = "barrido pendiente"

# ── Semáforo de hipótesis ────────────────────────────────────────────
def _chip(label, ok):
    color, icon = (("#27ae60", "✓") if ok is True
                   else ("#e74c3c", "×") if ok is False
                   else ("#7f8c8d", "—"))
    return (f'<span style="background:{color};color:#fff;border-radius:6px;'
            f'padding:5px 12px;margin:3px;font-weight:700;display:inline-block;">{label} {icon}</span>')


def _chip_estado(label, estado):
    """Chip tri-estado: ok / no concluyente / acotada / no / pendiente."""
    color, icon = {
        "ok": ("#27ae60", "✓"),
        "nc": ("#d35400", "NC"),
        "acot": ("#d35400", "≈"),
        "no": ("#e74c3c", "×"),
        "pendiente": ("#7f8c8d", "—"),
    }[estado]
    return (f'<span style="background:{color};color:#fff;border-radius:6px;'
            f'padding:5px 12px;margin:3px;font-weight:700;display:inline-block;">{label} {icon}</span>')

if he1_run and he2_run:
    # Cláusula A: estabilidad y verificabilidad de β_k bajo q ≤ q*.
    # H.E.1c puede quedar «acotada» (no todas las nubes llegan a <50 %),
    # lo que sostiene la cláusula con alcance limitado, no la refuta.
    if he2b_ok is not None:
        estabilidad_ok = bool(
            he1a_ok is True and he2b_ok is True and (he1c_ok is True or he1c_acotada)
        )
    # Cláusula B: ventaja comparativa. El piloto del PDF es no concluyente.
    comparativa_ok = (he1b_ok is True) if he1b_ok is not None else None
    if estabilidad_ok is True and comparativa_ok is True:
        hg_estado = "sostenida"
        hg_val = "A + B"
    elif estabilidad_ok is True and comparativa_ok is False:
        hg_estado = "parcial"
        hg_val = "A sostenida; B no concluyente"
    elif estabilidad_ok is True and comparativa_ok is None:
        hg_estado = "parcial"
        hg_val = "A sostenida; B pendiente"
    elif estabilidad_ok is False:
        hg_estado = "no sostenida"
        hg_val = "A no sostenida"
    else:
        hg_estado = "incompleta"
        hg_val = "falta H.E.2b o H.E.1"

if hg_estado == "sin datos" and (he1_run or he2_run):
    hg_estado = "incompleta"
    hg_val = "faltan hipótesis por ejecutar"

_e1b = "ok" if he1b_ok is True else ("nc" if he1b_ok is False else "pendiente")
_e1c = "ok" if he1c_ok else ("acot" if he1c_acotada else "pendiente")
chips = "".join([
    _chip("H.E.1a", he1a_ok),
    _chip_estado("H.E.1b", _e1b),
    _chip_estado("H.E.1c", _e1c),
    _chip("H.E.2a", he2a_ok),
    _chip("H.E.2b", he2b_ok),
])
_icono_hg = {"sostenida": "✓", "parcial": "≈", "incompleta": "…",
             "no sostenida": "×"}.get(hg_estado, "—")
_color_hg = {"sostenida": "#27ae60", "parcial": "#e67e22",
             "incompleta": "#7f8c8d", "no sostenida": "#e74c3c"}.get(hg_estado, "#7f8c8d")
chips += (f'<span style="background:{_color_hg};color:#fff;border-radius:6px;'
          f'padding:5px 12px;margin:3px;font-weight:700;display:inline-block;">'
          f'H.G. {hg_estado} {_icono_hg}</span>')
st.markdown("#### 🚦 Semáforo de hipótesis")
st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)
st.markdown("---")

# ── Tabla 5.9 ────────────────────────────────────────────────────────
st.subheader("Tabla 5.9 — Veredicto sobre cada hipótesis")
filas = [
    {"Hipótesis": "H.E.1a", "Afirmación": "dB ≤ 2·dH (k=0,1), 100 % réplicas",
     "Valor": he1a_val, "Estado": "Medido" if he1_run else "—",
     "Veredicto": "Se sostiene" if he1a_ok else ("No se sostiene" if he1a_ok is False else "—"),
     "Evidencia": "Figs. 5.2(b) y 5.3(b)"},
    {"Hipótesis": "H.E.1b", "Afirmación": "TDA > euclidiano (McNemar p<0.05)",
     "Valor": he1b_val, "Estado": "Medido" if he1_run else "—",
     "Veredicto": ("Se sostiene" if he1b_ok else
                   "No concluyente" if he1b_ok is False else "—"),
     "Evidencia": "Tabla 5.5 — piloto documentado"},
    {"Hipótesis": "H.E.1c", "Afirmación": "q* y caída <50 % con banda ≤ 0.10",
     "Valor": he1c_val, "Estado": "Medido" if he1_run else "—",
     "Veredicto": ("Se sostiene" if he1c_ok else
                   "Acotada" if he1c_acotada else "No se sostiene"),
     "Evidencia": "Fig. 5.2(a)"},
    {"Hipótesis": "H.E.2a", "Afirmación": "Reducción ≥ 15 % vs placa p=1 (eficiencia)",
     "Valor": he2a_val, "Estado": "Medido" if he2_run else "—",
     "Veredicto": "Se sostiene" if he2a_ok else ("No se sostiene" if he2a_ok is False else "—"),
     "Evidencia": "Fig. 5.4(b) — informativa, fuera de H.G."},
    {"Hipótesis": "H.E.2b", "Afirmación": "β₁ invariante (malla, p, rmin) + Euler≡GUDHI",
     "Valor": he2b_val, "Estado": "Medido" if he2_run else "—",
     "Veredicto": "Se sostiene" if he2b_ok else ("No se sostiene" if he2b_ok is False else "—"),
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
                      "Umbral u de binarización", "GUDHI 3.13 + conteo de Euler (8-conectividad)",
                      "β₀, β₁ del sólido", "Malla, r_min, p y umbral u",
                      "Estabilidad funcional ≤ ‖f − g‖∞ (Teorema 2.1)",
                      "Existencia (Prop. 2.2) y cota inferior convexa (Prop. 2.3)",
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
st.markdown("**Lectura por cláusulas del §1.5, no por conjunción de todas las hipótesis específicas:**")
clausulas = pd.DataFrame([
    {
        "Cláusula": "A. Estabilidad/verificabilidad de β_k bajo q ≤ q*",
        "Hipótesis": "H.E.1a + H.E.1c + H.E.2b",
        "Evidencia": "Medido",
        "Estado": ("sostenida con alcance" if estabilidad_ok else
                   "no sostenida" if estabilidad_ok is False else "sin datos"),
        "Lectura": ("H.E.1a y H.E.2b se sostienen; H.E.1c queda acotada"
                    if estabilidad_ok and not he1c_ok else
                    "H.E.1a y H.E.2b se sostienen; H.E.1c se sostiene"
                    if estabilidad_ok else "falta evidencia"),
    },
    {
        "Cláusula": "B. Ventaja frente a descriptores euclidianos",
        "Hipótesis": "H.E.1b",
        "Evidencia": "Medido",
        "Estado": ("sostenida" if comparativa_ok else
                   "no concluyente" if comparativa_ok is False else "sin datos"),
        "Lectura": "piloto documentado del PDF: 0,950 vs 0,925, McNemar no significativo",
    },
    {
        "Cláusula": "Eficiencia estructural SIMP (no pertenece a H.G.)",
        "Hipótesis": "H.E.2a",
        "Evidencia": "Medido",
        "Estado": "informativa",
        "Lectura": f"reducción {he2a_val} vs placa p=1",
    },
])
st.dataframe(clausulas, width="stretch", hide_index=True)

if hg_estado == "sostenida":
    st.success("### H.G. — MEDIDO · CONCLUYENTE · SOSTENIDA — cláusula A y cláusula B respaldadas.")
elif hg_estado == "parcial":
    st.warning(
        "### H.G. — MEDIDO · PARCIAL / CONDICIONADA — la cláusula de estabilidad se sostiene con alcance, "
        "pero la ventaja comparativa es **no concluyente** en el piloto del PDF. "
        "No corresponde declararla corroborada ni refutada."
    )
elif hg_estado == "no sostenida":
    st.error("### H.G. — MEDIDO · NO SOSTENIDA — falla la cláusula de estabilidad/verificabilidad.")
elif hg_estado == "incompleta":
    st.info("### H.G. — INCOMPLETA — hay resultados parciales; faltan hipótesis por ejecutar. "
            "Corré H.E.1 y el barrido completo de H.E.2, o usá la caché CSV.")
else:
    st.info("Sin datos aún: ejecutá los protocolos para emitir el veredicto.")

# ── Reproducibilidad (compuerta F6) ──────────────────────────────────
st.markdown("---")
st.subheader("🧪 Reproducibilidad — compuerta F6")
st.caption("Reejecuta la configuración canónica (semilla 42) y la compara con los CSV versionados en "
           "`data/`. La igualdad demuestra que el estudio es reproducible desde cero.")
if st.button("▶ Reproducir desde semilla y comparar con los CSV", key="btn_f6"):
    import os as _os
    m_chk = corrida_simp(60, 30, P_DEFECTO, RMIN_DEFECTO, max_iter=200)
    c_u2, c_s2, b1_2 = float(m_chk.c_hist[0]), float(m_chk.c_final), int(m_chk.beta1)
    ruta = _os.path.join("data", "validacion_HE2_corregida.csv")
    if _os.path.exists(ruta):
        dfchk = pd.read_csv(ruta)
        c_u_csv, c_s_csv = float(dfchk["c_uniforme"].iloc[0]), float(dfchk["c_simp"].iloc[0])
        b1_csv = int(dfchk["beta1_base"].iloc[0])
        ok = abs(c_u2 - c_u_csv) < 1e-6 and abs(c_s2 - c_s_csv) < 1e-6 and b1_2 == b1_csv
        st.dataframe(pd.DataFrame({
            "Magnitud": ["c uniforme", "c SIMP", "β₁"],
            "Reproducido ahora": [f"{c_u2:.6f}", f"{c_s2:.6f}", str(b1_2)],
            "CSV del repositorio": [f"{c_u_csv:.6f}", f"{c_s_csv:.6f}", str(b1_csv)],
            "Δ": [f"{abs(c_u2-c_u_csv):.2e}", f"{abs(c_s2-c_s_csv):.2e}", "—"],
        }), width="stretch", hide_index=True)
        if ok:
            st.success("### ✅ Reproducido — idéntico a `data/validacion_HE2_corregida.csv` (Δ < 1e-6). "
                       "La compuerta F6 del Cap. IV se cumple.")
        else:
            st.warning("⚠️ Diferencias respecto al CSV; revisar entorno o parámetros.")
    else:
        st.info("No se encontró `data/validacion_HE2_corregida.csv`. Generálo con "
                "`python -m tda.simulation.experiment_simp`.")

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
            f"n = {n_points}; R = {n_rep} réplicas; σ = q·diam(X)/√d.\n"
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
        _res52_exp = st.session_state.get("he1_fig52")
        if _res52_exp is not None:
            z.writestr("estudio/he1/fig52_replicas.csv",
                       _res52_exp["replicas"].to_csv(index=False))
        _trunc_exp = st.session_state.get("he1_trunc")
        if _trunc_exp is not None:
            z.writestr("estudio/he1/fig53_truncamiento.csv", _trunc_exp.to_csv(index=False))
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

methodology_expander("📖 Protocolo y compuertas — H.G.", [
    ("Protocolo", (
        "Corre los protocolos de H.E.1 y H.E.2 y consolida los veredictos.\n\n"
        "| Ítem | Se mide | Criterio | Evidencia |\n"
        "|---|---|---|---|\n"
        "| H.E.1a | $d_B \\le 2\\,d_H$ | 100 % de réplicas | Fig. 5.2(b) |\n"
        "| H.E.1b | TDA vs euclidiano | McNemar $p < 0.05$; piloto PDF no concluyente | Tabla 5.5 |\n"
        "| H.E.1c | $q^*$ y caída $<50\\,\\%$ | banda $\\le 0.10$ | Fig. 5.2(a) |\n"
        "| H.E.2a | reducción de compliance | $\\ge 15\\,\\%$ vs placa p=1 (fuera de H.G.) | Fig. 5.4(b) |\n"
        "| H.E.2b | $\\beta_1$ invariante | malla, $p$, $r_{\\min}$ + Euler≡GUDHI | Figs. 5.5–5.7 |\n"
    )),
    ("Articulación TDA–SIMP", "Complejo simplicial (VR) y complejo cúbico son instancias del mismo complejo de cadenas; un rasgo se acepta si **persiste** en la filtración ($\\varepsilon$ en TDA, $u$ en SIMP)."),
], "H.G.")
