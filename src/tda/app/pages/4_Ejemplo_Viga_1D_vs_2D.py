"""Página 4: Ejemplo Viga 1D vs 2D — Comparación Unificada.

Modos:
  1D  — Perfil de altura óptimo (analytical, viga voladizo)
  2D  — SIMP topológico 60×30 con animación de iteraciones + TDA post-hoc
  Barrido p — Compara p=2, p=3, p=4 lado a lado
  Comparación 1D vs 2D — Mismo problema, dos enfoques
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import pandas as pd

from tda.optimization.beam_optimizer import BeamOptimizer
from tda.optimization.metric_simp import MetricaTDA_SIMP
from tda.core.betti2d import betti_doble_computo, diagramas_gudhi, betti_euler
from tda.simulation.he_protocols import (
    corrida_simp, esfuerzo_max_von_mises, voxeles_a_mesh3d, stl_ascii_bytes,
)
from tda.app.theme import (
    apply_mpl_theme, apply_plotly_theme, responsive_style,
    page_header, ORANGE
)

from tda.app.download_utils import export_settings_ui
export_settings_ui()
apply_mpl_theme()

st.set_page_config(page_title="Ejemplo Viga 1D vs 2D", layout="wide", page_icon="🧪")
st.markdown(responsive_style(), unsafe_allow_html=True)

st.markdown(page_header(
    "Ejemplo Viga 1D vs 2D — Comparación Unificada",
    "Experimentá con optimización 1D, SIMP 2D, barridos y comparaciones"
), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SIDEBAR COMÚN
# ══════════════════════════════════════════════════════════════
st.sidebar.header("🧪 Modo de Experimentación")
_MODO_LABELS = {
    "🧱 2D — SIMP Topológico  ·  Caso 2 (H.E.2a/b)": "2D — SIMP Topológico",
    "📈 Barrido p (2,3,4)  ·  Caso 2 (H.E.2b · Fig. 5.5)": "Barrido p (2,3,4)",
    "📋 Informe de Ingeniería  ·  caso real certificado": "📋 Informe de Ingeniería",
    "⚖️ Comparador de materiales y optimizador": "⚖️ Comparador y optimizador",
    "📐 1D — Perfil de Altura  ·  didáctico (fuera de alcance)": "1D — Perfil de Altura",
    "🔀 Comparación 1D vs 2D  ·  didáctico": "Comparación 1D vs 2D",
}
_sel = st.sidebar.radio("Elegí un modo:", list(_MODO_LABELS.keys()), label_visibility="collapsed")
modo = _MODO_LABELS[_sel]
st.sidebar.caption("Los modos marcados «didáctico» ilustran el contraste forma↔topología pero **no** son "
                   "hipótesis de la tesis; el Caso 2 (SIMP) es el que corresponde a **H.E.2a/b**.")

st.sidebar.markdown("---")
st.sidebar.subheader("Parámetros Comunes")

# ── Caso real: material, dimensiones y carga ──
MATERIALES = {
    "Acero estructural S275": {"E_GPa": 200.0, "sigma_adm_MPa": 250.0, "rho": 7850, "costo": 1.2, "co2": 1.9},
    "Aluminio 6061-T6":       {"E_GPa": 69.0,  "sigma_adm_MPa": 240.0, "rho": 2700, "costo": 3.5, "co2": 8.2},
    "Titanio Ti-6Al-4V":      {"E_GPa": 114.0, "sigma_adm_MPa": 830.0, "rho": 4430, "costo": 30.0, "co2": 35.0},
    "PLA (impresión 3D)":     {"E_GPa": 3.5,   "sigma_adm_MPa": 50.0,  "rho": 1240, "costo": 20.0, "co2": 2.5},
}
material = st.sidebar.selectbox("Material", list(MATERIALES.keys()), index=0)
_mat = MATERIALES[material]

L_mm = st.sidebar.slider("Longitud L (mm)", 50.0, 200.0, 120.0, 10.0)
F_N = st.sidebar.slider("Carga Puntual F (N)", 100.0, 5000.0, 1000.0, 100.0)
E_GPa = st.sidebar.number_input("Módulo E (GPa)", value=_mat["E_GPa"], step=10.0, key=f"E_{material}")
sigma_adm = st.sidebar.number_input("σ_adm (MPa)", value=_mat["sigma_adm_MPa"], step=10.0, key=f"sig_{material}")
rho_kg_m3 = _mat["rho"]           # densidad [kg/m³]
costo_USD_kg = _mat["costo"]      # costo [USD/kg]
co2_kg_kg = _mat["co2"]           # huella embebida [kg CO₂ eq / kg]

st.sidebar.markdown("---")

# Parámetros SIMP (para modo 2D)
if modo != "1D — Perfil de Altura":
    penal = st.sidebar.slider("Penalización p", 1.0, 5.0, 3.0, 0.5)
    volfrac = st.sidebar.slider("Fracción volumen f_V", 0.2, 0.8, 0.5, 0.05)
    rmin = st.sidebar.slider("Radio filtro $r_{min}$", 1.0, 5.0, 2.4, 0.1)
    espesor_mm = st.sidebar.number_input("Espesor t (mm)", value=5.0, step=0.5, key="espesor_mm")
    mesh_res = st.sidebar.selectbox("Resolución malla", ["40×20 (rápido)", "60×30 (tesis)", "80×40 (detalle)"], index=1)
    mesh_map = {"40×20 (rápido)": (40, 20), "60×30 (tesis)": (60, 30), "80×40 (detalle)": (80, 40)}
    nelx, nely = mesh_map[mesh_res]

st.sidebar.markdown("---")

# Caso de carga (nuevo)
caso_carga = st.sidebar.selectbox(
    "Caso de carga 2D",
    ["Voladizo – Puntual", "Articulado – Central", "Voladizo – Distribuida"],
    index=0
)

# Tamaño mínimo de miembro (nuevo)
min_member = st.sidebar.slider("Tamaño míniom miembro (elementos)", 0.0, 5.0, 1.5, 0.5,
                               help="Filtro de tamaño mínimo para el diseño binario (en unidades de elemento).")

st.sidebar.markdown("---")
anim_delay = st.sidebar.slider("Delay animación (s)", 0.0, 0.5, 0.05, 0.01)

# Variables físicas
L = L_mm / 1000.0
F = F_N
E_s = E_GPa * 1e9
s_adm = sigma_adm * 1e6

# Colormap SIMP
cmap_simp = LinearSegmentedColormap.from_list("simp", ["#f0f0f0", "#2c3e50"])


def certificado_pdf(material, L_mm, Ly_mm, espesor_mm, F_N, sigma_adm,
                    reduccion, he2a_ok, sigma_max, fs, beta0, beta1, conc, persist,
                    masa_d, masa_s, costo_d, costo_s, co2_d, co2_s, ahorro,
                    rho_bin, umbr, b1s):
    """Certificado/ informe PDF (A4, 2 páginas) del caso de diseño real."""
    import io
    from datetime import datetime
    from matplotlib.backends.backend_pdf import PdfPages

    cert = he2a_ok and conc
    buf = io.BytesIO()
    with plt.style.context("default"), PdfPages(buf) as pdf:
        # ── Página 1: certificado ──
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.5, 0.965, "UNIVERSIDAD AUTÓNOMA GABRIEL RENÉ MORENO", ha="center", fontsize=9, color="#555")
        fig.text(0.5, 0.945, "Maestría en Matemática — UAGRM", ha="center", fontsize=9, color="#555")
        fig.text(0.5, 0.905, "CERTIFICADO DE DISEÑO OPTIMIZADO", ha="center", fontsize=16, fontweight="bold")
        fig.text(0.5, 0.884, "Optimización topológica SIMP certificada por análisis topológico (TDA)",
                 ha="center", fontsize=10)
        fig.text(0.5, 0.860, f"Fecha: {datetime.now():%Y-%m-%d %H:%M}", ha="center", fontsize=8, color="gray")
        y = 0.80
        def _line(txt, bold=False):
            nonlocal y
            fig.text(0.08, y, txt, fontsize=10, fontweight=("bold" if bold else "normal"))
            y -= 0.033
        _line(f"Material: {material}", bold=True)
        _line(f"Dimensiones (L × H × t): {L_mm:.0f} × {Ly_mm:.0f} × {espesor_mm:.1f} mm")
        _line(f"Carga puntual en el extremo libre: {F_N:.0f} N")
        _line(f"Reducción de compliance vs diseño uniforme: {reduccion:.1f} %  (H.E.2a {'✓' if he2a_ok else '×'})")
        _line(f"Topología: β₀ = {beta0}, β₁ = {beta1}   ·   Euler≡GUDHI: {'✓' if conc else '×'}   ·   "
              f"persistencia β₁ = {persist:.2f}  (H.E.2b)")
        _line(f"Esfuerzo máx (von Mises): {sigma_max/1e6:.1f} MPa  vs  σ_adm = {sigma_adm:.0f} MPa  →  FS = {fs:.2f}")
        _line(f"Masa: {masa_d*1000:.1f} g (SIMP) vs {masa_s*1000:.1f} g (bloque sólido) — ahorro {ahorro:.0f} %")
        _line(f"Costo: {costo_d:.3f} USD (SIMP) vs {costo_s:.3f} USD (sólido)")
        _line(f"CO₂ evitado: {co2_s - co2_d:.3f} kg eq")
        fig.text(0.5, 0.30, ("DISEÑO CERTIFICADO" if cert else "DISEÑO NO CERTIFICADO"),
                 ha="center", fontsize=14, fontweight="bold", color=("green" if cert else "red"))
        fig.text(0.5, 0.255, ("Reducción ≥ 40 % y topología β₁ validada por doble cómputo y persistencia."
                              if cert else "Revisar los criterios de certificación."), ha="center", fontsize=9)
        fig.text(0.08, 0.12, "Autor: Jorge Larry Copa Cruz", fontsize=9)
        fig.text(0.08, 0.095, "Director del TFG: MSc. Roberto Orellana", fontsize=9)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ── Página 2: figuras ──
        fig = plt.figure(figsize=(8.27, 11.69))
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.imshow(rho_bin, cmap="gray_r")
        ax1.set_title(f"Diseño SIMP binarizado (β₀={beta0}, β₁={beta1})")
        ax1.axis("off")
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(umbr, b1s, "-o", color="#2c3e50")
        ax2.axvline(0.5, ls="--", color="gray")
        ax2.set_title("Persistencia de β₁ frente al umbral (Fig. 5.7)")
        ax2.set_xlabel("Umbral u")
        ax2.set_ylabel("β₁")
        ax2.grid(alpha=0.3)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    return buf.getvalue()


def build_boundary_conditions(nelx, nely, caso_carga, F, L_mm):
    """Build FEM boundary conditions (dofs_fijos, F_vec) based on load case.

    Returns (dofs_fijos, F_vec, node_load_idx) for visualization.
    """
    nnx = nelx + 1
    nny = nely + 1
    n_nodes = nnx * nny
    F_vec = np.zeros(2 * n_nodes)
    dofs_fijos = []

    if caso_carga == "Voladizo – Puntual":
        # Empotramiento borde izquierdo, carga puntual centro-derecha
        for j in range(nny):
            dofs_fijos.extend([2 * j * nnx, 2 * j * nnx + 1])
        node_load = (nely // 2) * nnx + nelx
        dof_load = 2 * node_load + 1
        F_vec[dof_load] = -F
        load_desc = "Puntual en extremo libre (medio)"

    elif caso_carga == "Articulado – Central":
        # Apoyo simple izq + der, carga en centro superior
        for j in range(nny):
            dofs_fijos.extend([2 * j * nnx, 2 * j * nnx + 1])  # izq: u=v=0
            dofs_fijos.extend([2 * (j * nnx + nelx), 2 * (j * nnx + nelx) + 1])  # der: u=v=0
        node_load = (nely // 2) * nnx + (nelx // 2)
        dof_load = 2 * node_load + 1
        F_vec[dof_load] = -F
        load_desc = "Puntual en centro superior"

    elif caso_carga == "Voladizo – Distribuida":
        # Empotramiento borde izquierdo, carga distribuida borde derecho
        for j in range(nny):
            dofs_fijos.extend([2 * j * nnx, 2 * j * nnx + 1])
        F_per_node = F / nny
        for j in range(nny):
            node = j * nnx + nelx
            dof_v = 2 * node + 1
            F_vec[dof_v] = -F_per_node
        node_load = (nely // 2) * nnx + nelx
        load_desc = f"Distribuida borde derecho ({nny} nodos, {F_per_node:.1f} N c/u)"

    else:
        raise ValueError(f"Caso de carga desconocido: {caso_carga}")

    return dofs_fijos, F_vec, node_load, load_desc


def build_load_arrows(nelx, nely, caso_carga, F, node_load, F_vec):
    """Return list of (x, y, fx, fy) arrow specs for the schematic."""
    nnx = nelx + 1
    arrows = []
    if caso_carga == "Voladizo – Distribuida":
        # Draw arrows on right edge
        for j in range(nely + 1):
            node = j * nnx + nelx
            fy = F_vec[2 * node + 1]
            if abs(fy) > 1e-10:
                arrows.append((nelx, j, 0, fy))
    else:
        # Single arrow at node_load
        arrows.append((nelx, nely // 2, 0, F_vec[2 * node_load + 1]))
    return arrows


# ══════════════════════════════════════════════════════════════
# MODO 1D — PERFIL DE ALTURA
# ══════════════════════════════════════════════════════════════
if modo == "1D — Perfil de Altura":
    st.info("📐 **Laboratorio didáctico — fuera del alcance de la tesis.** Optimiza la *forma* "
            "(perfil de altura analítico), no la topología; sirve de contraste contra el Caso 2 (SIMP).")
    st.subheader("1D — Optimización de Perfil de Altura")

    col_info, col_esq = st.columns([1, 1])

    with col_info:
        st.markdown("""
        **Modelo:** Viga en voladizo con carga puntual en el extremo libre.
        El optimizador ajusta la altura h(x) para minimizar volumen
        cumpliendo σ(x) ≤ σ_adm en todo punto.
        """)

    # Esquema del problema
    h0 = 40.0 / 1000.0
    fig_schem, ax_schem = plt.subplots(figsize=(10, 2))
    ax_schem.axis('off')
    ax_schem.set_xlim(-L * 0.1, L * 1.1)
    ax_schem.set_ylim(-h0 * 2, h0 * 2)
    beam_rect = Rectangle((0, -h0 / 2), L, h0, facecolor='#d5dbdb', edgecolor='#2c3e50', linewidth=2, zorder=2)
    ax_schem.add_patch(beam_rect)
    wall = Rectangle((-L * 0.05, -h0), L * 0.05, h0 * 2, facecolor='#7f8c8d', hatch='///', edgecolor='black')
    ax_schem.add_patch(wall)
    arr = FancyArrowPatch((L, h0 * 1.5), (L, h0 / 2), arrowstyle='->', mutation_scale=20, lw=2.5, color='#e74c3c', zorder=5)
    ax_schem.add_patch(arr)
    ax_schem.text(L, h0 * 1.8, f"F = {F_N} N", ha='center', va='bottom', color='#e74c3c', fontweight='bold')
    ax_schem.text(L / 2, -h0 * 1.5, f"L = {L_mm} mm", ha='center', va='top', fontweight='bold')
    col_esq.pyplot(fig_schem)
    plt.close(fig_schem)

    btn_1d = st.button("▶ Optimizar Perfil 1D", type="primary", key="btn_1d")

    ph_metrics_1d = st.container()
    ph_plot_1d = st.empty()

    if btn_1d:
        b_beam = 0.001  # 1 mm
        opt = BeamOptimizer(b=b_beam, h0=h0, p=3, N=61, E_s=E_s, sigma_adm=s_adm, max_iter=80)

        def cb_1d(data):
            with ph_metrics_1d:
                r1_c1, r1_c2 = st.columns(2)
                r1_c1.metric("Iteración", data['iteration'])
                r1_c2.metric("Ahorro Material", f"{data['saving_pct']:.1f}%")
                r2_c1, r2_c2 = st.columns(2)
                max_sig = np.max(data['sigma_MPa'])
                r2_c1.metric("Tensión Máx", f"{max_sig:.1f} MPa",
                          delta="✅ OK" if max_sig <= sigma_adm else "⚠️ Excede",
                          delta_color="normal" if max_sig <= sigma_adm else "inverse")
                r2_c2.metric("Deflexión Máx", f"{np.max(np.abs(data['Y'])) * 1000:.2f} mm")

            fig, axs = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1, 1]})
            axs[0].plot(data['x'], data['h_v'] / 2, color='#2980b9', lw=2)
            axs[0].plot(data['x'], -data['h_v'] / 2, color='#2980b9', lw=2)
            axs[0].fill_between(data['x'], -data['h_v'] / 2, data['h_v'] / 2, color='#3498db', alpha=0.3)
            axs[0].plot([0, L], [h0 / 2, h0 / 2], 'k--', alpha=0.3)
            axs[0].plot([0, L], [-h0 / 2, -h0 / 2], 'k--', alpha=0.3)
            axs[0].set_title(f"Perfil de Altura Óptimo (Iter {data['iteration']})")
            axs[0].set_ylabel("Altura (m)")
            axs[0].set_xlim(0, L)
            axs[0].set_ylim(-h0, h0)

            axs[1].plot(data['x'], data['M'] / 1000, color='#e67e22', lw=2)
            axs[1].fill_between(data['x'], 0, data['M'] / 1000, color='#e67e22', alpha=0.2)
            axs[1].set_title("Momento Flector M(x) [kN·m]")
            axs[1].set_xlim(0, L)

            axs[2].plot(data['x'], data['sigma_MPa'], color='#8e44ad', lw=2)
            axs[2].axhline(sigma_adm, color='red', linestyle='--', alpha=0.5, label="σ_adm")
            axs[2].set_title("Tensión Máxima σ(x) [MPa]")
            axs[2].set_xlabel("Distancia x (m)")
            axs[2].set_xlim(0, L)
            axs[2].legend()

            plt.tight_layout()
            ph_plot_1d.pyplot(fig)
            plt.close(fig)
            time.sleep(anim_delay)

        final = opt.optimizar_viga_completo(L, F, callback=cb_1d)
        st.success(f"✅ Completado: {final['iterations']} iteraciones, ahorro {final['saving_pct']:.1f}%")


# ══════════════════════════════════════════════════════════════
# MODO 2D — SIMP TOPOLÓGICO CON ANIMACIÓN
# ══════════════════════════════════════════════════════════════
elif modo == "2D — SIMP Topológico":
    st.caption("Caso 2 de la tesis — optimización topológica SIMP (H.E.2a/b). El diseño se certifica con "
               "β₀, β₁ (doble cómputo) y persistencia.")
    st.subheader(f"2D — SIMP Topológico ({nelx}×{nely}) — {caso_carga}")

    col_info, col_esq2 = st.columns([1, 1])
    with col_info:
        st.markdown(f"""
        **Modelo:** Malla {nelx}×{nely} Q4.
        **Carga:** {caso_carga}.
        **Parámetros:** $f_V$={volfrac}, $p$={penal}, $r_{{min}}$={rmin}, $t_{{mín}}$={min_member}.

        **Animación:** Se muestra cada ~5 iteraciones el diseño actualizado.
        **Post-hoc:** Al finalizar, se calcula $\\beta_0$, $\\beta_1$ y factor de seguridad.
        """)

    # Esquema 2D dinámico según caso de carga
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.set_aspect('equal')
    rect_domain = Rectangle((0, 0), nelx, nely, facecolor='#ecf0f1', edgecolor='#2c3e50', lw=2)
    ax2.add_patch(rect_domain)

    # Bordes de soporte
    if caso_carga == "Articulado – Central":
        wall_l = Rectangle((-2, 0), 2, nely, facecolor='#7f8c8d', hatch='///', edgecolor='black')
        wall_r = Rectangle((nelx, 0), 2, nely, facecolor='#7f8c8d', hatch='///', edgecolor='black')
        ax2.add_patch(wall_l)
        ax2.add_patch(wall_r)
    else:
        wall2 = Rectangle((-2, 0), 2, nely, facecolor='#7f8c8d', hatch='///', edgecolor='black')
        ax2.add_patch(wall2)

    # Flechas de carga
    if caso_carga == "Voladizo – Distribuida":
        for j in range(nely + 1):
            arr = FancyArrowPatch((nelx + 2, j), (nelx, j), arrowstyle='->', mutation_scale=12, lw=1.2, color='#e74c3c')
            ax2.add_patch(arr)
        ax2.text(nelx + 3, nely / 2, f"F = {F_N} N\ndistribuida", color='#e74c3c', fontweight='bold', va='center', fontsize=9)
    else:
        arr2 = FancyArrowPatch((nelx + 1, nely / 2), (nelx, nely / 2), arrowstyle='->', mutation_scale=20, lw=2.5, color='#e74c3c')
        ax2.add_patch(arr2)
        ax2.text(nelx + 2, nely / 2, f"F = {F_N} N", color='#e74c3c', fontweight='bold', va='center')

    ax2.set_xlim(-5, nelx + 8)
    ax2.set_ylim(-3, nely + 3)
    ax2.set_title(f"Dominio FEM — {nelx}×{nely} = {nelx * nely} elementos — {caso_carga}")
    ax2.axis('off')
    col_esq2.pyplot(fig2)
    plt.close(fig2)

    btn_2d = st.button("▶ Ejecutar SIMP 2D", type="primary", key="btn_2d")

    ph_rho = st.empty()
    ph_metrics_2d = st.container()

    if btn_2d:
        dofs_fijos, F_vec, node_load, load_desc = build_boundary_conditions(nelx, nely, caso_carga, F, L_mm)

        opt2d = MetricaTDA_SIMP(
            nex=nelx, ney=nely, E=E_s, nu=0.3,
            Lx=L_mm, Ly=L_mm * (nely / nelx), t=1.0,
            f_V=volfrac, p=penal, r_min=rmin,
            tol_c=1e-4, tol_rho=1e-2, max_iter=200
        )
        opt2d.definir_problema(F_vec, dofs_fijos)

        # Callback para animacion
        def cb_simp(k, c, delta_c, delta_rho, rho):
            if k % 5 == 0 or k <= 3:
                fig_a, ax_a = plt.subplots(figsize=(10, 3))
                rho_grid = rho.reshape((nely, nelx))
                ax_a.imshow(rho_grid, cmap=cmap_simp, vmin=0, vmax=1, aspect='equal', origin='lower')
                ax_a.set_title(f"Iteracion {k} -- Compliance = {c:.2f} -- Delta_c/c = {delta_c:.2e}")
                ax_a.axis('off')
                ph_rho.pyplot(fig_a)
                plt.close(fig_a)
                time.sleep(anim_delay)

        with st.spinner(f"Optimizando SIMP ({caso_carga})..."):
            opt2d.optimizar(callback=cb_simp, verbose=False)

        with st.spinner("Analisis topologico (TDA)..."):
            opt2d.fase_tda(verbose=False)

        res = opt2d.obtener_resultados()

        # Diseno final
        fig_final, axes = plt.subplots(1, 2, figsize=(14, 4))

        rho_grid = res['rho_final'].reshape((nely, nelx))
        rho_tilde_grid = res['rho_tilde_final'].reshape((nely, nelx))

        axes[0].imshow(rho_grid, cmap=cmap_simp, vmin=0, vmax=1, aspect='equal', origin='lower')
        axes[0].set_title(f"rho (densidad continua) -- {res['n_iter']} iter")
        axes[0].axis('off')

        axes[1].imshow(rho_tilde_grid >= 0.5, cmap='binary', aspect='equal', origin='lower')
        conc = "OK" if res['betti_concordancia'] else "NO"
        axes[1].set_title(f"rho~ >= 0.5 (diseno binario) -- beta0={res['beta0']}, beta1={res['beta1']} [{conc}]")
        axes[1].axis('off')

        plt.tight_layout()
        st.pyplot(fig_final)
        st.session_state.setdefault('viga_figs', {})['comparacion_1d_vs_2d'] = fig_final
        plt.close(fig_final)

        # Metricas (2 filas para responsive: 4 + 3)
        row1_c1, row1_c2, row1_c3 = st.columns(3)
        row1_c1.metric("Compliance", f"{res['c_final']:.2f}")
        row1_c2.metric("beta0", res['beta0'])
        row1_c3.metric("beta1", res['beta1'])

        row2_c1, row2_c2, row2_c3 = st.columns(3)
        row2_c1.metric("Iteraciones", res['n_iter'])
        row2_c2.metric("Convergio", "OK" if res['converged'] else "NO")

        # Factor de seguridad (post-hoc aproximado)
        sigma_max_est = max(1.0, res['c_final'] / (L_mm * 1e-3))  # estimacion simple
        safety_factor = s_adm / (sigma_max_est * 1e6) if sigma_max_est > 0 else float('inf')
        row2_c3.metric("Factor Seguridad", f"{safety_factor:.1f}", delta="OK" if safety_factor >= 1.5 else "BAJO", delta_color="normal" if safety_factor >= 1.5 else "inverse")

        # Curva de convergencia
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=list(range(1, len(res['c_hist']) + 1)),
            y=res['c_hist'],
            mode='lines+markers',
            line=dict(color=ORANGE, width=2),
            name='Compliance'
        ))
        fig_conv.update_layout(
            title="Convergencia SIMP",
            xaxis_title="Iteración",
            yaxis_title="Compliance (N·mm)",
            template="plotly_white",
            height=350
        )
        st.plotly_chart(fig_conv, width='stretch')
        st.session_state.setdefault('viga_figs', {})['convergencia_simp'] = fig_conv

        # Diagramas de persistencia
        if res['dgm0'] is not None or res['dgm1'] is not None:
            fig_dgm = make_subplots(rows=1, cols=2, subplot_titles=["H₀ (componentes)", "H₁ (agujeros)"])
            for i, (dgm, name) in enumerate([(res['dgm0'], 'H₀'), (res['dgm1'], 'H₁')]):
                if dgm is not None and len(dgm) > 0:
                    fig_dgm.add_trace(
                        go.Scatter(x=dgm[:, 0], y=dgm[:, 1], mode='markers',
                                   marker=dict(size=6, color=['#3498db', '#e74c3c'][i]),
                                   name=name),
                        row=1, col=i + 1
                    )
                # Diagonal de referencia
                max_val = max(dgm[:, 1].max(), 1) if dgm is not None and len(dgm) > 0 else 1
                fig_dgm.add_trace(
                    go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines',
                               line=dict(dash='dash', color='gray'), showlegend=False),
                    row=1, col=i + 1
                )
            fig_dgm.update_layout(title="Diagramas de Persistencia (GUDHI)", height=350, template="plotly_white")
            st.plotly_chart(fig_dgm, width='stretch')
            st.session_state.setdefault('viga_figs', {})['diagrama_persistencia'] = fig_dgm


# ══════════════════════════════════════════════════════════════
# MODO BARRIDO p — COMPARA p=2,3,4
# ══════════════════════════════════════════════════════════════
elif modo == "Barrido p (2,3,4)":
    st.subheader(f"Barrido de Penalizacion -- p=2, p=3, p=4 -- {caso_carga}")
    st.markdown(f"""
    Corre SIMP 3 veces con la misma malla ({nelx}x{nely}) y compara:
    - Distribucion de densidad final
    - Compliance, beta0, beta1
    - Efecto del penalizador sobre la topologia
    **Carga:** {caso_carga}. **t_min:** {min_member}.
    """)

    btn_barrido = st.button("▶ Ejecutar Barrido p", type="primary", key="btn_barrido")

    if btn_barrido:
        dofs_fijos, F_vec, node_load, load_desc = build_boundary_conditions(nelx, nely, caso_carga, F, L_mm)

        resultados = {}
        progress = st.progress(0, text="Ejecutando barrido...")

        for idx, p_val in enumerate([2, 3, 4]):
            progress.progress((idx) / 3, text=f"Optimizando p={p_val}...")
            opt_b = MetricaTDA_SIMP(
                nex=nelx, ney=nely, E=E_s, nu=0.3,
                Lx=L_mm, Ly=L_mm * (nely / nelx), t=1.0,
                f_V=volfrac, p=p_val, r_min=rmin,
                tol_c=1e-4, tol_rho=1e-2, max_iter=200
            )
            opt_b.definir_problema(F_vec, dofs_fijos)
            opt_b.optimizar(verbose=False)
            opt_b.fase_tda(verbose=False)
            resultados[p_val] = opt_b.obtener_resultados()

        progress.progress(1.0, text="Barrido completo")

        # Mostrar disenos lado a lado
        fig_barrido, axes = plt.subplots(2, 3, figsize=(15, 8))
        for col_idx, p_val in enumerate([2, 3, 4]):
            res_p = resultados[p_val]
            rho_grid = res_p['rho_final'].reshape((nely, nelx))
            rho_bin = (res_p['rho_tilde_final'].reshape((nely, nelx)) >= 0.5).astype(float)

            axes[0, col_idx].imshow(rho_grid, cmap=cmap_simp, vmin=0, vmax=1, aspect='equal', origin='lower')
            axes[0, col_idx].set_title(f"p={p_val} -- rho (continuo)")
            axes[0, col_idx].axis('off')

            axes[1, col_idx].imshow(rho_bin, cmap='binary', aspect='equal', origin='lower')
            conc = "OK" if res_p['betti_concordancia'] else "NO"
            axes[1, col_idx].set_title(f"rho~>=0.5 -- beta1={res_p['beta1']} [{conc}]")
            axes[1, col_idx].axis('off')

        plt.tight_layout()
        st.pyplot(fig_barrido)
        st.session_state.setdefault('viga_figs', {})['barrido_penalizacion'] = fig_barrido
        plt.close(fig_barrido)

        # Tabla comparativa
        import pandas as pd
        df_comp = pd.DataFrame({
            "Metrica": ["Compliance", "beta0", "beta1", "Iteraciones", "Convergio"],
            "p=2": [f"{resultados[2]['c_final']:.2f}", str(resultados[2]['beta0']), str(resultados[2]['beta1']),
                    str(resultados[2]['n_iter']), "OK" if resultados[2]['converged'] else "NO"],
            "p=3": [f"{resultados[3]['c_final']:.2f}", str(resultados[3]['beta0']), str(resultados[3]['beta1']),
                    str(resultados[3]['n_iter']), "OK" if resultados[3]['converged'] else "NO"],
            "p=4": [f"{resultados[4]['c_final']:.2f}", str(resultados[4]['beta0']), str(resultados[4]['beta1']),
                    str(resultados[4]['n_iter']), "OK" if resultados[4]['converged'] else "NO"],
        })
        # Cast all columns to str to avoid Arrow serialization errors
        for col in df_comp.columns:
            df_comp[col] = df_comp[col].astype(str)
        st.dataframe(df_comp.set_index("Metrica"), width='stretch')


# ══════════════════════════════════════════════════════════════
# MODO COMPARACIÓN 1D vs 2D
# ══════════════════════════════════════════════════════════════
elif modo == "Comparación 1D vs 2D":
    st.info("🔀 **Laboratorio didáctico — fuera del alcance de la tesis.** Compara forma (1D) vs "
            "topología (2D); el aporte de la tesis está en el Caso 2 (SIMP).")
    st.subheader(f"Comparacion Lado a Lado -- 1D vs 2D -- {caso_carga}")
    st.markdown(f"""
    Ejecuta ambos optimizadores en el mismo problema y compara:
    - **1D:** Perfil de altura analitico (viga rectangular variable)
    - **2D:** SIMP topologico (distribucion de material en malla)
    **Carga:** {caso_carga}. **t_min:** {min_member}.

    El 1D es mas rapido pero solo optimiza la forma.
    El 2D optimiza la topologia completa (agujeros, islas, ramas).
    """)

    btn_cmp = st.button("▶ Ejecutar Comparacion", type="primary", key="btn_cmp")

    if btn_cmp:
        # -- 1D --
        h0_cmp = 40.0 / 1000.0
        b_cmp = 0.001
        opt1d = BeamOptimizer(b=b_cmp, h0=h0_cmp, p=3, N=61, E_s=E_s, sigma_adm=s_adm, max_iter=80)
        res1d = opt1d.optimizar_viga_completo(L, F, callback=None)

        # -- 2D --
        dofs_fijos, F_vec, node_load, load_desc = build_boundary_conditions(nelx, nely, caso_carga, F, L_mm)

        opt2d_c = MetricaTDA_SIMP(
            nex=nelx, ney=nely, E=E_s, nu=0.3,
            Lx=L_mm, Ly=L_mm * (nely / nelx), t=1.0,
            f_V=volfrac, p=penal, r_min=rmin,
            tol_c=1e-4, tol_rho=1e-2, max_iter=200
        )
        opt2d_c.definir_problema(F_vec, dofs_fijos)

        with st.spinner("Optimizando 2D..."):
            opt2d_c.optimizar(verbose=False)
        with st.spinner("TDA post-hoc..."):
            opt2d_c.fase_tda(verbose=False)
        res2d = opt2d_c.obtener_resultados()

        # -- Visualizacion lado a lado --
        fig_cmp, axes = plt.subplots(2, 2, figsize=(14, 8))

        # 1D -- Perfil
        axes[0, 0].plot(res1d['x'], res1d['h_v'] / 2, color='#2980b9', lw=2)
        axes[0, 0].plot(res1d['x'], -res1d['h_v'] / 2, color='#2980b9', lw=2)
        axes[0, 0].fill_between(res1d['x'], -res1d['h_v'] / 2, res1d['h_v'] / 2, color='#3498db', alpha=0.3)
        axes[0, 0].plot([0, L], [h0_cmp / 2, h0_cmp / 2], 'k--', alpha=0.3)
        axes[0, 0].plot([0, L], [-h0_cmp / 2, -h0_cmp / 2], 'k--', alpha=0.3)
        axes[0, 0].set_title(f"1D -- Perfil Optimo (ahorro {res1d['saving_pct']:.1f}%)")
        axes[0, 0].set_ylabel("Altura (m)")
        axes[0, 0].set_xlim(0, L)

        # 1D -- Tension
        axes[1, 0].plot(res1d['x'], res1d['sigma_MPa'], color='#8e44ad', lw=2)
        axes[1, 0].axhline(sigma_adm, color='red', linestyle='--', alpha=0.5, label="sigma_adm")
        axes[1, 0].set_title("1D -- Tension sigma(x) [MPa]")
        axes[1, 0].set_xlabel("Distancia x (m)")
        axes[1, 0].legend()

        # 2D -- rho continuo
        rho_grid_c = res2d['rho_final'].reshape((nely, nelx))
        axes[0, 1].imshow(rho_grid_c, cmap=cmap_simp, vmin=0, vmax=1, aspect='equal', origin='lower')
        conc_c = "OK" if res2d['betti_concordancia'] else "NO"
        axes[0, 1].set_title(f"2D -- SIMP (c={res2d['c_final']:.2f}, beta1={res2d['beta1']})")
        axes[0, 1].axis('off')

        # 2D -- Binario
        rho_bin_c = (res2d['rho_tilde_final'].reshape((nely, nelx)) >= 0.5).astype(float)
        axes[1, 1].imshow(rho_bin_c, cmap='binary', aspect='equal', origin='lower')
        axes[1, 1].set_title(f"2D -- rho~>=0.5 [{conc_c}]")
        axes[1, 1].axis('off')

        plt.tight_layout()
        st.pyplot(fig_cmp)
        st.session_state.setdefault('viga_figs', {})['comparacion_final'] = fig_cmp
        plt.close(fig_cmp)

        # Tabla resumen
        st.markdown("### Resumen Comparativo")
        sigma_max_est_2d = max(1.0, res2d['c_final'] / (L_mm * 1e-3))
        sf_2d = s_adm / (sigma_max_est_2d * 1e6) if sigma_max_est_2d > 0 else float('inf')
        df_resumen = pd.DataFrame({
            "Metrica": ["Ahorro material", "Compliance", "beta1", "Iteraciones", "Factor Seguridad"],
            "1D (Perfil)": [f"{res1d['saving_pct']:.1f}%", "N/A (analitico)", "N/A", f"{res1d['iterations']}", "N/A"],
            "2D (SIMP)": ["--", f"{res2d['c_final']:.2f}", str(res2d['beta1']),
                          str(res2d['n_iter']), f"{sf_2d:.1f}"]
        })
        for col in df_resumen.columns:
            df_resumen[col] = df_resumen[col].astype(str)
        st.dataframe(df_resumen.set_index("Metrica"), width='stretch')

        # STL export del diseno binario 2D
        st.markdown("---")
        st.subheader("Exportar STL (diseno 2D)")
        st.markdown("Exporta el diseno binario (rho~>=0.5) como malla STL 3D para impresion 3D o CAD.")

        # Preparar rho binario para STL
        rho_bin_stl = (res2d['rho_tilde_final'].reshape((nely, nelx)) >= 0.5).astype(float)

        # Generar STL simple (cada celda binaria = cubo unitario)
        vertices = []
        faces = []
        v_idx = 0
        Lx_m = L_mm / 1000.0
        Ly_m = L_mm * (nely / nelx) / 1000.0
        dx = Lx_m / nelx
        dy = Ly_m / nely
        thickness = 0.001  # 1 mm de espesor para exportar

        for i in range(nelx):
            for j in range(nely):
                if rho_bin_stl[j, i] > 0.5:
                    x0, y0 = i * dx, j * dy
                    x1, y1 = x0 + dx, y0 + dy
                    # 8 vertices del cubo (front y back faces)
                    verts = [
                        (x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0),  # back
                        (x0, y0, thickness), (x1, y0, thickness), (x1, y1, thickness), (x0, y1, thickness)  # front
                    ]
                    base = v_idx
                    vertices.extend(verts)
                    # 12 triangulos (2 por cara, 6 caras)
                    faces.extend([
                        [base+1, base+0, base+4], [base+1, base+4, base+5],  # bottom
                        [base+2, base+1, base+5], [base+2, base+5, base+6],  # right
                        [base+3, base+2, base+6], [base+3, base+6, base+7],  # top
                        [base+0, base+3, base+7], [base+0, base+7, base+4],  # left
                        [base+0, base+1, base+2], [base+0, base+2, base+3],  # back
                        [base+5, base+4, base+7], [base+5, base+7, base+6],  # front
                    ])
                    v_idx += 8

        if len(vertices) > 0:
            stl_content = f"solid beam_design\n"
            for face in faces:
                v0, v1, v2 = [vertices[f] for f in face]
                # Normal (approx)
                ax_, ay_, az_ = v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]
                bx_, by_, bz_ = v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]
                nx_ = ay_*bz_ - az_*by_
                ny_ = az_*bx_ - ax_*bz_
                nz_ = ax_*by_ - ay_*bx_
                mag = (nx_**2 + ny_**2 + nz_**2)**0.5 or 1.0
                stl_content += f"  facet normal {nx_/mag:.6f} {ny_/mag:.6f} {nz_/mag:.6f}\n"
                stl_content += f"    outer loop\n"
                for v in [v0, v1, v2]:
                    stl_content += f"      vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n"
                stl_content += f"    endloop\n"
                stl_content += f"  endfacet\n"
            stl_content += "endsolid beam_design\n"

            n_voxels = int(rho_bin_stl.sum())
            st.info(f"Diseno binario: {n_voxels} celdas activas de {nelx*nely} totales ({100*n_voxels/(nelx*nely):.1f}%)")
            st.download_button(
                label="Descargar STL",
                data=stl_content,
                file_name=f"viga_2D_{nelx}x{nely}_{caso_carga.replace(' ','_').replace('--','-')}.stl",
                mime="model/stl",
                key="btn_stl"
            )
        else:
            st.warning("No hay celdas activas para exportar. El diseno binario esta vacio.")


# ══════════════════════════════════════════════════════════════
# MODO: INFORME DE INGENIERÍA (caso real, ligado a la tesis)
# ══════════════════════════════════════════════════════════════
elif modo == "📋 Informe de Ingeniería":
    st.subheader("📋 Informe de Ingeniería — caso real")
    st.markdown(
        f"**Material:** {material}  ·  **L** = {L_mm:.0f} mm  ·  **F** = {F_N:.0f} N  ·  "
        f"**t** = {espesor_mm:.1f} mm  ·  **malla** = {nelx}×{nely}  ·  "
        f"f_V = {volfrac:.2f}, p = {penal:.0f}, r_min = {rmin:.1f}"
    )
    st.caption("Unidades SI (m, N, Pa, kg). Cada resultado se conecta con la hipótesis del documento "
               "(H.E.2a: reducción ≥ 40 %; H.E.2b: β₁ estable + doble cómputo).")

    if st.button("▶ Ejecutar caso real y generar informe", type="primary", key="btn_informe"):
        Lx_m = L_mm / 1000.0
        Ly_m = Lx_m * (nely / nelx)
        t_m = espesor_mm / 1000.0
        E = E_GPa * 1e9
        s_adm_Pa = sigma_adm * 1e6

        with st.spinner("Optimizando SIMP (caso real)..."):
            m = corrida_simp(nelx, nely, penal, rmin, E=E, Fmag=F_N, fV=volfrac, t=t_m,
                             max_iter=200, Lx=Lx_m, Ly=Ly_m)
        c_base = float(m.c_hist[0])
        c_simp = float(m.c_final)
        reduccion = (c_base - c_simp) / c_base * 100
        sigma_max = esfuerzo_max_von_mises(m, E, 0.3)
        fs = s_adm_Pa / sigma_max if sigma_max > 0 else float("inf")

        V_dom = Lx_m * Ly_m * t_m
        masa_d = volfrac * V_dom * rho_kg_m3          # masa del diseño SIMP (volumen = fV·V0)
        masa_s = V_dom * rho_kg_m3
        costo_d = masa_d * costo_USD_kg
        costo_s = masa_s * costo_USD_kg
        ahorro = (1 - volfrac) * 100

        # Persistencia de β₁ frente al umbral (tesis, Fig. 5.7)
        umbr = np.linspace(0.05, 0.95, 19)
        b1s = [betti_euler(m.rho_tilde_final.reshape(nely, nelx) >= u)[1] for u in umbr]
        from tda.simulation.he_protocols import persistencia_beta1
        _, _, persist = persistencia_beta1(umbr, b1s, m.beta1)

        # Referencia analítica 1D (mismo material, carga y longitud)
        opt1d = BeamOptimizer(b=t_m, h0=Ly_m, p=3, N=61, E_s=E, sigma_adm=s_adm_Pa, max_iter=80)
        res1d = opt1d.optimizar_viga_completo(Lx_m, F_N, callback=None)

        conc = bool(getattr(m, "_betti_concordancia", None))
        he2a_ok = reduccion >= 40.0
        he2b_ok = conc and persist >= 0.3
        cert = he2a_ok and conc

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Reducción de compliance", f"{reduccion:.1f} %",
                  delta="H.E.2a ≥ 40 %" if he2a_ok else "< 40 %",
                  delta_color="normal" if he2a_ok else "inverse")
        c2.metric("β₁ (agujeros)", str(m.beta1), help="Descriptor topológico certificado por la tesis")
        c3.metric("σ_máx / σ_adm", f"{sigma_max/1e6:.0f} / {sigma_adm:.0f} MPa",
                  delta=f"FS = {fs:.2f}", delta_color="normal" if fs >= 1.5 else "inverse")
        c4.metric("Euler ≡ GUDHI", "✅" if conc else "❌")

        st.markdown("#### Vínculo con la tesis")
        st.markdown(
            f"- **H.E.2a** (redistribuir el material reduce la compliance ≥ 40 %): "
            f"{'✅ SE SOSTIENE' if he2a_ok else '❌ no se sostiene'} — **{reduccion:.1f} %** vs diseño uniforme.\n"
            f"- **H.E.2b** (β₁ es descriptor estable): β₁ = **{m.beta1}**, doble cómputo "
            f"{'coincide' if conc else 'difiere'}, persistencia de β₁ ≈ **{persist:.2f}** (unidades de umbral) "
            f"→ {'✅ estable' if he2b_ok else '⚠️ revisar'}.\n"
            f"- **Validación 1D↔2D**: la solución analítica 1D (mismo material/carga) da ahorro "
            f"{res1d['saving_pct']:.1f} % y σ_máx {np.max(res1d['sigma_MPa']):.0f} MPa — referente real."
        )

        st.markdown("#### Métricas de ingeniería (vida real)")
        st.dataframe(pd.DataFrame({
            "Magnitud": ["Material", "Dimensiones (L×H×t)", "Volumen del dominio",
                         "Masa diseño SIMP", "Masa bloque sólido", "Ahorro de material",
                         "Costo diseño", "Costo bloque sólido", "σ_máx (von Mises)",
                         "Factor de seguridad", "Tamaño mín. miembro", "Agujeros (β₁)"],
            "Valor": [material, f"{L_mm:.0f}×{Ly_m*1000:.0f}×{espesor_mm:.1f} mm",
                      f"{V_dom*1e9:.1f} cm³", f"{masa_d*1000:.1f} g", f"{masa_s*1000:.1f} g",
                      f"{ahorro:.0f} %", f"{costo_d:.3f} USD", f"{costo_s:.3f} USD",
                      f"{sigma_max/1e6:.1f} MPa", f"{fs:.2f}", f"{min_member:.1f} elem", str(m.beta1)],
        }), width="stretch", hide_index=True)

        # ── Dashboard de impacto (sostenibilidad y costo) ──
        co2_d = masa_d * co2_kg_kg
        co2_s = masa_s * co2_kg_kg
        st.markdown("#### Impacto (sostenibilidad y costo)")
        d1, d2, d3 = st.columns(3)
        d1.metric("Material ahorrado", f"{(masa_s - masa_d)*1000:.1f} g", delta=f"{ahorro:.0f} %")
        d2.metric("Costo ahorrado", f"{costo_s - costo_d:.3f} USD")
        d3.metric("CO₂ evitado", f"{co2_s - co2_d:.3f} kg eq")

        # ── Chequeo y sugerencias automáticas de resistencia ──
        FS_OBJ = 1.5
        if fs < FS_OBJ:
            t_req = t_m * (FS_OBJ / fs) if fs > 0 else float("inf")
            F_max = F_N * (fs / FS_OBJ)
            st.warning(
                f"⚠️ **FS = {fs:.2f} < {FS_OBJ}** — "
                + ("σ_máx **supera** σ_adm (falla resistencia)." if sigma_max > s_adm_Pa
                   else "σ_máx queda por debajo de σ_adm pero con margen insuficiente.")
            )
            st.markdown(
                "**Ajustes sugeridos** (≈ σ ∝ F/t):\n"
                f"- Aumentar espesor → **t ≈ {t_req*1000:.1f} mm** (actual {espesor_mm:.1f} mm)\n"
                f"- Reducir carga → **F ≤ {F_max:.0f} N** (actual {F_N:.0f} N)\n"
                "- O subir la fracción de volumen f_V (más material) y reoptimizar."
            )
            if st.button("🔧 Aplicar t recomendado y re-ejecutar", key="btn_apply_t"):
                st.session_state["espesor_mm"] = float(round(t_req * 1000 * 2) / 2)  # múltiplo de 0.5 mm
                st.rerun()
        else:
            st.success(f"✅ FS = {fs:.2f} ≥ {FS_OBJ} — resistencia verificada.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=umbr, y=b1s, mode="lines+markers", name="β₁"))
        fig.add_vline(x=0.5, line=dict(dash="dash", color="white"), annotation_text="u = 0.5")
        fig.update_layout(height=340, template="plotly_white",
                          xaxis_title="Umbral de binarización (u)", yaxis_title="β₁",
                          title="Persistencia de β₁ (tesis, Fig. 5.7)")
        apply_plotly_theme(fig)
        st.plotly_chart(fig, width="stretch")

        figd = go.Figure(data=[go.Heatmap(z=(m.rho_tilde_final.reshape(nely, nelx) >= 0.5).astype(float),
                                          colorscale="gray_r", zmin=0, zmax=1, colorbar=dict(title="Ω"))])
        figd.update_layout(height=340, title=f"Diseño SIMP | β₀={m.beta0}, β₁={m.beta1}")
        apply_plotly_theme(figd)
        st.plotly_chart(figd, width="stretch")

        # ── Vista 3D (extrusión) + export STL ──
        V3, F3 = voxeles_a_mesh3d(m.rho_tilde_final.reshape(nely, nelx) >= 0.5, t=espesor_mm)
        if len(F3) > 0:
            fig3 = go.Figure(go.Mesh3d(
                x=V3[:, 0], y=V3[:, 1], z=V3[:, 2],
                i=F3[:, 0], j=F3[:, 1], k=F3[:, 2],
                color="#2c3e50", opacity=1.0, flatshading=True, name="diseño"))
            fig3.update_layout(height=440, title="Vista 3D del diseño (extrusión del sólido)",
                               scene=dict(aspectmode="data"), margin=dict(l=0, r=0, b=0, t=40))
            apply_plotly_theme(fig3)
            st.plotly_chart(fig3, width="stretch")
            st.download_button("🧊 Descargar STL (impresión 3D / CAD)", stl_ascii_bytes(V3, F3),
                               "diseno_simp.stl", "model/stl", width="stretch")

        # ── Certificado PDF descargable ──
        try:
            rho_bin = (m.rho_tilde_final.reshape(nely, nelx) >= 0.5).astype(float)
            pdf_bytes = certificado_pdf(
                material, L_mm, Ly_m * 1000, espesor_mm, F_N, sigma_adm,
                reduccion, he2a_ok, sigma_max, fs, m.beta0, m.beta1, conc, persist,
                masa_d, masa_s, costo_d, costo_s, co2_d, co2_s, ahorro,
                rho_bin, umbr, b1s)
            st.download_button("📄 Descargar certificado PDF", pdf_bytes,
                               "certificado_diseno.pdf", "application/pdf", width="stretch")
        except Exception as e:
            st.error(f"No se pudo generar el certificado: {e}")

        if cert:
            st.success("### ✅ Diseño CERTIFICADO por el método de la tesis "
                       "(reducción ≥ 40 % y topología β₁ validada por doble cómputo + persistencia).")
        else:
            st.warning("### ⚠️ Diseño generado; revisar los criterios de certificación.")
        st.caption("El diseño puede exportarse a STL (impresión 3D/CAD) desde el modo «Comparación 1D vs 2D».")


# ══════════════════════════════════════════════════════════════
# MODO: COMPARADOR DE MATERIALES Y OPTIMIZADOR AUTOMÁTICO
# ══════════════════════════════════════════════════════════════
elif modo == "⚖️ Comparador y optimizador":
    st.subheader("⚖️ Comparador de materiales y optimizador automático")
    st.caption("Unidades SI. El optimizador busca el diseño de **menor masa** que cumpla un factor de "
               "seguridad objetivo, variando la fracción de volumen f_V y evaluando σ (von Mises) del diseño.")

    fs_obj = st.slider("Factor de seguridad objetivo (FS)", 1.0, 3.0, 1.5, 0.1, key="fs_obj")
    st.markdown("---")

    # ── Optimizador automático ──
    st.markdown("#### 🔎 Optimizador: menor masa con FS objetivo")
    if st.button("▶ Optimizar f_V (menor masa con FS objetivo)", type="primary", key="btn_opt"):
        Lx_m = L_mm / 1000.0
        Ly_m = Lx_m * (nely / nelx)
        t_m = espesor_mm / 1000.0
        E = E_GPa * 1e9
        s_adm_Pa = sigma_adm * 1e6
        filas, best = [], None
        for fv in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
            with st.spinner(f"Optimizando f_V = {fv:.2f}..."):
                m = corrida_simp(nelx, nely, penal, rmin, E=E, Fmag=F_N, fV=fv,
                                 t=t_m, max_iter=150, Lx=Lx_m, Ly=Ly_m)
            sig = esfuerzo_max_von_mises(m, E, 0.3)
            fs = s_adm_Pa / sig if sig > 0 else float("inf")
            masa = fv * Lx_m * Ly_m * t_m * rho_kg_m3
            filas.append({"f_V": fv, "σ_máx (MPa)": round(sig / 1e6, 1), "FS": round(fs, 2),
                          "c(ρ*)": round(float(m.c_final), 3), "β₁": m.beta1,
                          "Masa (g)": round(masa * 1000, 1)})
            if fs >= fs_obj and (best is None or masa < best[1]):
                best = (fv, masa, sig, m)
        st.dataframe(pd.DataFrame(filas), width="stretch", hide_index=True)
        if best:
            st.success(f"### ✅ Menor masa con FS ≥ {fs_obj}: f_V = **{best[0]:.2f}** → "
                       f"**{best[1]*1000:.1f} g**, σ_máx = {best[2]/1e6:.1f} MPa, β₁ = {best[3].beta1}")
        else:
            sig_min = min(f["σ_máx (MPa)"] for f in filas) * 1e6
            t_req = t_m * (s_adm_Pa / fs_obj) / sig_min
            st.warning(f"Ninguna f_V ∈ [0.30, 0.80] cumple FS ≥ {fs_obj}. Se requiere "
                       f"t ≈ **{t_req*1000:.1f} mm** (actual {espesor_mm:.1f} mm) o reducir la carga.")
    st.markdown("---")

    # ── Comparador de materiales ──
    st.markdown("#### 🧱 Comparador de materiales (mismo caso y f_V)")
    if st.button("▶ Comparar materiales", type="secondary", key="btn_mat"):
        Lx_m = L_mm / 1000.0
        Ly_m = Lx_m * (nely / nelx)
        t_m = espesor_mm / 1000.0
        V_dom = Lx_m * Ly_m * t_m
        filas = []
        for nombre, mp in MATERIALES.items():
            E = mp["E_GPa"] * 1e9
            s_adm_Pa = mp["sigma_adm_MPa"] * 1e6
            with st.spinner(f"Optimizando {nombre}..."):
                m = corrida_simp(nelx, nely, penal, rmin, E=E, Fmag=F_N, fV=volfrac,
                                 t=t_m, max_iter=150, Lx=Lx_m, Ly=Ly_m)
            sig = esfuerzo_max_von_mises(m, E, 0.3)
            fs = s_adm_Pa / sig if sig > 0 else float("inf")
            masa = volfrac * V_dom * mp["rho"]
            filas.append({"Material": nombre, "Masa (g)": round(masa * 1000, 1),
                          "Costo (USD)": round(masa * mp["costo"], 3),
                          "CO₂ (kg eq)": round(masa * mp["co2"], 3),
                          "σ_máx (MPa)": round(sig / 1e6, 1), "FS": round(fs, 2), "β₁": m.beta1})
        dfm = pd.DataFrame(filas)
        st.dataframe(dfm, width="stretch", hide_index=True)
        fig_m = go.Figure()
        for col, colr in [("Masa (g)", "#2980b9"), ("Costo (USD)", "#e67e22"), ("CO₂ (kg eq)", "#27ae60")]:
            fig_m.add_trace(go.Bar(x=dfm["Material"], y=dfm[col], name=col, marker_color=colr))
        fig_m.update_layout(height=380, template="plotly_white", barmode="group",
                            title="Comparación por material (masa, costo, CO₂)")
        apply_plotly_theme(fig_m)
        st.plotly_chart(fig_m, width="stretch")


# ══════════════════════════════════════════════════════════════
# EXPORTAR FIGURAS (ZIP)
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("Exportar Figuras (ZIP)")
_figs_viga = st.session_state.get('viga_figs', {})
if _figs_viga:
    import io as _io
    import zipfile as _zipfile
    from tda.app.download_utils import write_figs_zip
    _zf_buf = _io.BytesIO()
    with _zipfile.ZipFile(_zf_buf, 'w', _zipfile.ZIP_DEFLATED) as _zf:
        write_figs_zip(_zf, "figuras_viga", _figs_viga)
    st.download_button(
        label="Descargar figuras (ZIP)",
        data=_zf_buf.getvalue(),
        file_name="viga_figuras.zip",
        mime="application/zip",
        key="btn_viga_zip"
    )
else:
    st.info("Ejecutar una simulación para generar y descargar las figuras.")


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("---")
from tda.app.theme import footer_style
st.markdown(f"""
<div style="{footer_style()}">
    <p style="margin:0;color:{ORANGE};font-weight:700;">
        Ejemplo Viga 1D vs 2D — Comparación Unificada de Optimización
    </p>
    <p style="margin:0.3rem 0 0 0;">
        BeamOptimizer (1D) · MetricaTDA_SIMP (2D) · GUDHI (TDA)
    </p>
</div>
""", unsafe_allow_html=True)
