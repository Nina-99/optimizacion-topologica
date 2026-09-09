"""Página 4: Ejemplo Viga 1D.

Demostración didáctica interactiva de optimización estructural 1D
usando un modelo de viga en voladizo de acero (Cantilever).
Sirve como introducción conceptual a la optimización SIMP 2D de la tesis.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from tda.optimization.beam_optimizer import BeamOptimizer
from tda.app.theme import (
    apply_mpl_theme, apply_plotly_theme, responsive_style,
    methodology_expander, page_header, ORANGE
)

# ── Configuración de exportación (.exe) ──
from tda.app.download_utils import export_settings_ui
export_settings_ui()

apply_mpl_theme()

st.set_page_config(page_title="Ejemplo de Optimización de Vigas", layout="wide", page_icon="🏗️")
st.markdown(responsive_style(), unsafe_allow_html=True)

# ── Page Header ──
st.markdown(page_header(
    "Ejemplo — Optimización de Viga 1D (Acero)",
    "Concepto introductorio: Viga en Voladizo (Cantilever)"
), unsafe_allow_html=True)

st.info("""
💡 **Nota Conceptual:** Este simulador 1D es un modelo analítico introductorio para ilustrar la optimización de material antes de pasar a la complejidad topológica. 
Utiliza los mismos parámetros físicos de la **Aplicación 1** de la tesis (Acero, 200 GPa), 
pero simplificado a 1D (perfil de altura variable) en lugar de una malla 2D.
""")

# ── Sidebar ──
st.sidebar.header("📈 Parámetros Físicos")

L_mm = st.sidebar.slider("Longitud L (mm)", 50.0, 200.0, 120.0, 10.0, 
    help="Longitud total del voladizo. Cuadro 1 de la tesis: L = 120 mm.")
F_N = st.sidebar.slider("Carga Puntual F (N)", 100.0, 5000.0, 1000.0, 100.0, 
    help="Carga aplicada en el extremo libre. Cuadro 1: F = 1000 N.")
E_GPa = st.sidebar.number_input("Módulo de Young E (GPa)", value=200.0, step=10.0, 
    help="Acero estructural: 200 GPa.")
sigma_adm = st.sidebar.number_input("Límite Elástico σ_adm (MPa)", value=250.0, step=10.0, 
    help="Tensión máxima admitida en el acero.")
b_mm = st.sidebar.number_input("Ancho b (mm)", value=1.0, step=0.1, 
    help="Espesor de la viga en estado plano 2D. Tesis asume 1.0 mm.")
h0_mm = st.sidebar.number_input("Altura inicial h₀ (mm)", value=40.0, step=1.0, 
    help="Altura máxima permitida de la viga. Cuadro 1: h = 40 mm.")
p = st.sidebar.number_input("Factor Penalización (p)", value=3, min_value=1, max_value=5, 
    help="SIMP penalty para forzar reducción de masa.")
N = st.sidebar.number_input("Nodos (Resolución)", value=61, min_value=10, max_value=200, step=10)

st.sidebar.markdown("---")
st.sidebar.subheader("Animación")
anim_delay = st.sidebar.slider("Delay Animación (s)", 0.0, 0.5, 0.05)

L = L_mm / 1000.0
F = F_N
E_s = E_GPa * 1e9
s_adm = sigma_adm * 1e6
b = b_mm / 1000.0
h0 = h0_mm / 1000.0

if 'beam_opt_data' not in st.session_state:
    st.session_state.beam_opt_data = None

# ============================================================
# ESQUEMA DEL PROBLEMA
# ============================================================
st.subheader("Configuración Estructural: Viga en Voladizo")

fig_schem, ax_schem = plt.subplots(figsize=(10, 2))
ax_schem.axis('off')
ax_schem.set_xlim(-L*0.1, L*1.1)
ax_schem.set_ylim(-h0*2, h0*2)

beam_y0 = 0.0
beam_rect = Rectangle((0, beam_y0 - h0/2), L, h0, facecolor='#d5dbdb',
                     edgecolor='#2c3e50', linewidth=2, zorder=2)
ax_schem.add_patch(beam_rect)

# Empotramiento (izquierdo)
wall = Rectangle((-L*0.05, beam_y0 - h0), L*0.05, h0*2, facecolor='#7f8c8d', hatch='///', edgecolor='black')
ax_schem.add_patch(wall)

# Carga puntual (derecho)
arr = FancyArrowPatch((L, beam_y0 + h0*1.5), (L, beam_y0 + h0/2),
                      arrowstyle='->', mutation_scale=20, lw=2.5, color='#e74c3c', zorder=5)
ax_schem.add_patch(arr)
ax_schem.text(L, beam_y0 + h0*1.8, f"F = {F_N} N", ha='center', va='bottom', color='#e74c3c', fontweight='bold')
ax_schem.text(L/2, beam_y0 - h0*1.5, f"L = {L_mm} mm", ha='center', va='top', fontweight='bold')

st.pyplot(fig_schem)
plt.close(fig_schem)

# ============================================================
# EJECUCIÓN Y ANIMACIÓN
# ============================================================
col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    btn_start = st.button("▶ Optimizar Perfil", type="primary", use_container_width=True)

ph_metrics = st.container()
ph_plot = st.empty()

if btn_start:
    opt = BeamOptimizer(b=b, h0=h0, p=p, N=N, E_s=E_s, sigma_adm=s_adm, max_iter=80)
    
    def ui_callback(data):
        with ph_metrics:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Iteración", data['iteration'])
            c2.metric("Ahorro de Material", f"{data['saving_pct']:.1f}%")
            c3.metric("Tensión Máx", f"{np.max(data['sigma_MPa']):.1f} MPa", 
                      delta="✅ OK" if np.max(data['sigma_MPa']) <= sigma_adm else "⚠️ Excede",
                      delta_color="normal" if np.max(data['sigma_MPa']) <= sigma_adm else "inverse")
            c4.metric("Deflexión Máx", f"{np.max(np.abs(data['Y']))*1000:.2f} mm")
            
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # Perfil
        axs[0].plot(data['x'], data['h_v']/2, color='#2980b9', lw=2)
        axs[0].plot(data['x'], -data['h_v']/2, color='#2980b9', lw=2)
        axs[0].fill_between(data['x'], -data['h_v']/2, data['h_v']/2, color='#3498db', alpha=0.3)
        axs[0].plot([0, L], [h0/2, h0/2], 'k--', alpha=0.3)
        axs[0].plot([0, L], [-h0/2, -h0/2], 'k--', alpha=0.3)
        axs[0].set_title(f"Perfil de Altura Óptimo (Iter {data['iteration']})")
        axs[0].set_ylabel("Altura (m)")
        axs[0].set_xlim(0, L)
        axs[0].set_ylim(-h0, h0)
        
        # Momento
        axs[1].plot(data['x'], data['M']/1000, color='#e67e22', lw=2)
        axs[1].fill_between(data['x'], 0, data['M']/1000, color='#e67e22', alpha=0.2)
        axs[1].set_title("Momento Flector M(x) [kN·m]")
        axs[1].set_xlim(0, L)
        
        # Tensión
        axs[2].plot(data['x'], data['sigma_MPa'], color='#8e44ad', lw=2)
        axs[2].axhline(sigma_adm, color='red', linestyle='--', alpha=0.5, label="σ_adm")
        axs[2].set_title("Tensión Máxima en la Fibra σ(x) [MPa]")
        axs[2].set_xlabel("Distancia x (m)")
        axs[2].set_xlim(0, L)
        axs[2].legend()
        
        plt.tight_layout()
        ph_plot.pyplot(fig)
        plt.close(fig)
        time.sleep(anim_delay)

    final_data = opt.optimizar_viga_completo(L, F, callback=ui_callback)
    st.session_state.beam_opt_data = final_data
    st.success("✅ Optimización 1D Completada")

# ============================================================
# METODOLOGÍA
# ============================================================
methodology_expander(
    "📖 Formulación Analítica (Viga en Voladizo)",
    [
        (
            "Momento Flector",
            r"M(x) = -F \cdot (L - x)"
        ),
        (
            "Tensión Máxima",
            r"\sigma(x) = \frac{6 \cdot |M(x)|}{b \cdot h(x)^2} \leq \sigma_{adm}"
        ),
        (
            "Perfil Analítico Óptimo",
            r"h_{opt}(x) = \sqrt{\frac{6 \cdot F \cdot (L - x)}{b \cdot \sigma_{adm}}}"
        )
    ],
    "Viga 1D"
)
