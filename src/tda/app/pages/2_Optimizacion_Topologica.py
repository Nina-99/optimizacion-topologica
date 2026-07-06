"""Página 2: Optimización Topológica de Vigas (H.E.2).

Optimización topológica de vigas de hormigón armado mediante SIMP 1D.
Minimización de volumen con restricción de rigidez y deflexión.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages
import io

from tda.optimization.beam_optimizer import BeamOptimizer
from tda.app.theme import apply_mpl_theme, apply_plotly_theme, responsive_style

# Aplicar tema matplotlib global
apply_mpl_theme()

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(page_title="Optimización Topológica de Vigas", layout="wide", page_icon="🏗️")

st.markdown(responsive_style(), unsafe_allow_html=True)
st.header("📈 Optimización Topológica")

# ── Sidebar ──
st.sidebar.header("📈 Optimización Topológica")

L = st.sidebar.slider("Longitud de Viga (m)", 6.0, 20.0, 12.0, 0.5, key="beam_length",
    help="Longitud total entre apoyos. Afecta directamente el momento flector (M = qL²/8) y la deflexión máxima. Vigas más largas requieren más material en el centro del vano.")
q = st.sidebar.slider("Carga Distribuida (kN/m)", 10.0, 80.0, 30.0, 1.0, key="beam_load",
    help="Carga uniformemente distribuida que incluye peso propio más sobrecarga de uso. El momento flector máximo crece linealmente con q.")
E = st.sidebar.number_input("Módulo de Elasticidad (kPa)", value=30000000.0, step=1000000.0, key="beam_E",
    help="Módulo de Young del hormigón armado. Valores típicos: 25 GPa (H-25) a 30 GPa (H-30). Un mayor E reduce la deflexión elástica.")
b = st.sidebar.number_input("Base de la Viga (m)", value=0.30, step=0.05, key="beam_b",
    help="Ancho de la sección transversal rectangular. En edificación típica: 0.20-0.40 m. La inercia crece linealmente con b.")
h0 = st.sidebar.number_input("Altura de la Viga (m)", value=0.80, step=0.1, key="beam_h0",
    help="Canto o altura inicial de la viga (máximo permitido). El optimizador SIMP 1D reducirá h en zonas de bajo momento. La inercia crece con h³.")
p = st.sidebar.number_input("Factor de Penalización", value=3, min_value=1, max_value=5, key="beam_p",
    help="Penaliza densidades intermedias para forzar una solución 0/1. p=3 es el valor estándar SIMP. Valores >3 convergen más rápido pero pueden ser inestables.")
N = st.sidebar.number_input("Número de Nodos", value=101, min_value=50, max_value=200, step=10, key="beam_N",
    help="Discretización espacial 1D. Más nodos → mayor precisión pero más cómputo. 101 nodos (100 elementos) es suficiente para una viga simplemente apoyada.")

st.sidebar.markdown("---")
st.sidebar.subheader("Control de Visualización")
update_frequency = st.sidebar.selectbox(
    "Frecuencia de Actualización",
    ["Cada iteración", "Cada 2 iteraciones", "Cada 5 iteraciones"],
    key="update_frequency"
)

st.sidebar.subheader("Elementos Visuales")
show_reinforcement = st.sidebar.checkbox("Mostrar Armadura", value=True, key="show_reinforcement")
show_stirrups = st.sidebar.checkbox("Mostrar Estribos", value=True, key="show_stirrups")
show_limits = st.sidebar.checkbox("Mostrar Límites Admisibles", value=True, key="show_limits")

st.sidebar.subheader("Animación")
animation_speed = st.sidebar.slider("Velocidad de Animación", 1, 10, 5, key="animation_speed")

# Validación de parámetros
if L <= 0:
    st.sidebar.error("La longitud de la viga debe ser mayor que 0")
if q <= 0:
    st.sidebar.error("La carga distribuida debe ser mayor que 0")
if E <= 0:
    st.sidebar.error("El módulo de elasticidad debe ser mayor que 0")
if b <= 0:
    st.sidebar.error("La base de la viga debe ser mayor que 0")
if h0 <= 0:
    st.sidebar.error("La altura de la viga debe ser mayor que 0")
if N < 10:
    st.sidebar.error("El número de nodos debe ser al menos 10")

# ============================================================
# CUERPO PRINCIPAL
# ============================================================

# Inicializar estado de sesión
if 'beam_opt_data' not in st.session_state:
    st.session_state.beam_opt_data = None
if 'beam_opt_running' not in st.session_state:
    st.session_state.beam_opt_running = False

has_data = st.session_state.beam_opt_data is not None
running = st.session_state.beam_opt_running

# ============================================================
# PHASE 1 — PRE-RUN: Schematic + Parameter Summary + Start
# ============================================================
if not has_data and not running:

    # ── Gradient header card ──
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    ">
        <h3 style="margin:0; font-weight:600;">🏗️ Optimización de Viga de Hormigón Armado</h3>
        <p style="margin:0.5rem 0 0 0; opacity:0.85; font-size:0.95rem;">
            Algoritmo SIMP unidimensional — Minimización de volumen con restricción de rigidez
        </p>
    </div>""", unsafe_allow_html=True)

    # ── Schematic figure ──
    fig_schem, ax_schem = plt.subplots(figsize=(10, 4))
    ax_schem.set_xlim(-0.5, L + 0.5)
    y_cota = -0.15 * h0
    ax_schem.set_ylim(y_cota - 0.3, max(h0, 0.35) + 0.35)
    ax_schem.set_aspect('equal')
    ax_schem.axis('off')

    beam_h = h0 * 0.6
    beam_y0 = 0.0
    rect = Rectangle((0, beam_y0), L, beam_h, facecolor='#d5dbdb',
                     edgecolor='#2c3e50', linewidth=2, zorder=2)
    ax_schem.add_patch(rect)

    # Roller support (triangle left)
    tri_left = Polygon([(0, beam_y0), (-0.12, beam_y0 - 0.25), (0, beam_y0 - 0.25)],
                       closed=True, facecolor='#2c3e50', edgecolor='black', linewidth=1.5, zorder=3)
    ax_schem.add_patch(tri_left)
    circle = plt.Circle((0, beam_y0 - 0.25), 0.04, facecolor='#2c3e50', zorder=4)
    ax_schem.add_patch(circle)

    # Pinned support (triangle right)
    tri_right = Polygon([(L, beam_y0), (L + 0.12, beam_y0 - 0.25), (L, beam_y0 - 0.25)],
                        closed=True, facecolor='#2c3e50', edgecolor='black', linewidth=1.5, zorder=3)
    ax_schem.add_patch(tri_right)

    # Distributed load arrows
    n_arrows = max(8, int(L / 1.5))
    for i in range(n_arrows + 1):
        x_pos = i * L / n_arrows
        arr = FancyArrowPatch((x_pos, beam_y0 + beam_h + 0.05),
                              (x_pos, beam_y0 + beam_h + 0.18),
                              arrowstyle='->', mutation_scale=15,
                              lw=1.5, color='#e74c3c', zorder=5)
        ax_schem.add_patch(arr)

    ax_schem.text(L / 2, beam_y0 + beam_h + 0.28, f"q = {q} kN/m",
                  ha='center', va='bottom', fontsize=10, fontweight='bold', color='#e74c3c')
    ax_schem.text(L / 2, beam_y0 - 0.35, f"L = {L} m",
                  ha='center', va='top', fontsize=11, fontweight='bold')

    # Cross-section inset
    inset_ax = ax_schem.inset_axes([0.82, 0.55, 0.15, 0.4])
    sec_rect = Rectangle((-b / 2, 0), b, h0, facecolor='#d5dbdb',
                         edgecolor='#2c3e50', linewidth=1.5)
    inset_ax.add_patch(sec_rect)
    inset_ax.set_xlim(-b, b)
    inset_ax.set_ylim(-0.02, h0 + 0.02)
    inset_ax.set_aspect('equal')
    inset_ax.axis('off')
    inset_ax.text(0, h0 / 2, f'{b:.2f}×{h0:.2f}', ha='center', va='center',
                  fontsize=7, fontweight='bold', rotation=90)
    inset_ax.text(b / 2, -0.01, f'b={b}m', ha='center', va='top', fontsize=6)
    inset_ax.text(0, h0 + 0.01, f'h₀={h0}m', ha='center', va='bottom', fontsize=6)

    st.pyplot(fig_schem)
    plt.close(fig_schem)

    # ── Parameter summary ──
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    col_p1.metric("Longitud L", f"{L} m")
    col_p2.metric("Carga q", f"{q} kN/m")
    col_p3.metric("Sección b×h₀", f"{b} × {h0} m")
    col_p4.metric("Penalización p", f"{p}")

    col_p5, col_p6, col_p7, _ = st.columns(4)
    col_p5.metric("E", f"{E/1e6:.0f} MPa")
    col_p6.metric("Nodos", f"{N}")
    col_p7.metric("Método", "SIMP 1D")

    st.markdown("---")

    if st.button("▶ Iniciar Optimización Topológica", type="primary", use_container_width=True):
        st.session_state.beam_opt_running = True
        st.rerun()

# ============================================================
# PHASE 2 — DURING OPTIMIZATION: Live dashboard
# ============================================================
elif running:

    st.info("🔄 Optimizando...", icon="⚙️")

    cancel_btn = st.button("⏹ Cancelar", key="cancel_beam_opt")
    if cancel_btn:
        st.session_state.beam_opt_cancel = True
    else:
        st.session_state.beam_opt_cancel = False

    col_live1, col_live2 = st.columns([0.6, 0.4])
    plot_live = col_live1.empty()

    with col_live2:
        metric_iter = st.empty()
        metric_save = st.empty()
        metric_weight = st.empty()
        metric_error = st.empty()
        pbar_live = st.progress(0)
        metric_iter.metric("Iteración", "0")
        metric_save.metric("Ahorro Vol.", "—")
        metric_weight.metric("Peso Ahorrado", "—")
        metric_error.metric("Error Relativo", "—")

    freq_map = {"Cada iteración": 1, "Cada 2 iteraciones": 2, "Cada 5 iteraciones": 5}
    freq = freq_map.get(update_frequency, 1)

    def beam_callback(data):
        if st.session_state.get('beam_opt_cancel', False):
            raise StopIteration("Optimización cancelada por el usuario")

        iter_num = data["iteration"]
        if iter_num % freq != 0:
            return

        fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        x = data["x"]; h_v = data["h_v"]

        axes[0].fill_between(x, -h_v / 2, h_v / 2, color='#d5dbdb',
                             edgecolor='#2c3e50', linewidth=1.5)
        axes[0].set_ylabel("Peralte (m)")
        axes[0].set_title(f"Perfil de la viga — Iteración {iter_num}", fontsize=11)
        axes[0].grid(True, alpha=0.3)

        y_orig_mm = data["Y_original"] * 1000
        y_opt_mm = data["Y"] * 1000
        axes[1].plot(x, y_orig_mm, 'b--', linewidth=1.5, label="Original", alpha=0.7)
        axes[1].plot(x, y_opt_mm, 'r-', linewidth=2, label="Optimizada")
        if show_limits and data.get("y_adm", 0) > 0:
            axes[1].axhline(data["y_adm"], color='red', linestyle=':', alpha=0.7,
                            label=f"Admisible ({data['y_adm']:.1f} mm)")
        axes[1].set_ylabel("Deflexión (mm)")
        axes[1].set_title("Deflexión y(x)", fontsize=11)
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(x, data["M"], 'g-', linewidth=1.5, label="Momento M(x)")
        axes[2].set_ylabel("Momento (kN·m)", color='g')
        axes[2].tick_params(axis='y', labelcolor='g')
        axes[2].grid(True, alpha=0.3)

        ax_stress = axes[2].twinx()
        ax_stress.plot(x, data["sigma_MPa"], 'darkred', linewidth=1.5,
                       linestyle='--', label="Tensión σ")
        if show_limits:
            ax_stress.axhline(11.25, color='red', linestyle=':', alpha=0.5,
                              label="σ_adm = 11.25 MPa")
        ax_stress.set_ylabel("Tensión (MPa)", color='darkred')
        ax_stress.tick_params(axis='y', labelcolor='darkred')

        lines1, labels1 = axes[2].get_legend_handles_labels()
        lines2, labels2 = ax_stress.get_legend_handles_labels()
        axes[2].legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')

        axes[2].set_xlabel("Posición x (m)")
        axes[2].set_title("Momento Flector y Tensión", fontsize=11)

        plt.tight_layout()
        plot_live.pyplot(fig)
        plt.close(fig)

        pct = data["saving_pct"]
        sign = "↓" if pct > 0 else "↑"
        metric_iter.metric("Iteración", f"{iter_num}")
        metric_save.metric("Ahorro Vol.", f"{pct:.1f}%", sign)
        metric_weight.metric("Peso Ahorrado", f"{data['weight_saved']:.2f} t", "↓")
        metric_error.metric("Error Relativo", f"{data.get('error', 0):.6f}")
        pbar_live.progress(min(iter_num / 50, 1.0))

    try:
        optimizer = BeamOptimizer(b, h0, p, N, E_c=E, max_iter=50)
        final_results = optimizer.optimizar_viga_completo(L, q, callback=beam_callback)

        # Guardar parámetros usados para persistencia al navegar
        final_results['params_used'] = {
            "L": L, "q": q, "E": E, "b": b, "h0": h0, "p": p, "N": N,
        }
        st.session_state.beam_opt_running = False
        st.session_state.beam_opt_data = final_results
        st.rerun()

    except StopIteration:
        st.session_state.beam_opt_running = False
        st.warning("⏸️ Optimización cancelada por el usuario")
    except Exception as e:
        st.session_state.beam_opt_running = False
        st.error(f"Error en la optimización: {str(e)}")

# ============================================================
# PHASE 3 — POST-RUN: Results + Analysis + Export
# ============================================================
else:
    data = st.session_state.beam_opt_data

    st.success("✅ Optimización completada exitosamente")

    # ── Banner de parámetros usados (persistencia entre páginas) ──
    params_used = data.get("params_used", {})
    if params_used:
        st.info(
            f"📋 Mostrando resultados previos — "
            f"Parámetros: L={params_used['L']}m, q={params_used['q']}kN/m, "
            f"E={params_used['E']/1e6:.0f}MPa, b={params_used['b']}m, "
            f"h₀={params_used['h0']}m, p={params_used['p']}, N={params_used['N']} nodos. "
            f"Cambiá los parámetros en el sidebar y ejecutá una nueva optimización."
        )

    # ── Preparar datos ──
    x = data["x"]
    h_v = data["h_v"]
    Y = data["Y"]
    Y_orig = data["Y_original"]
    M = data["M"]
    sigma = data["sigma_MPa"]
    saving = data["saving_pct"]
    weight = data["weight_saved"]

    # ── Construir figura con 3 subplots ──
    fig_final = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Perfil de la viga", "Deflexión y(x)", "Momento Flector y Tensión"),
        specs=[[{}], [{}], [{"secondary_y": True}]]
    )

    # Row 1: Beam profile
    fig_final.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([h_v / 2, -h_v[::-1] / 2]),
        fill='toself', fillcolor='#d5dbdb',
        line=dict(color='#2c3e50', width=2),
        name="Perfil optimizado",
        showlegend=False
    ), row=1, col=1)

    # Row 2: Deflection
    fig_final.add_trace(go.Scatter(
        x=x, y=Y_orig * 1000,
        mode='lines', line=dict(color='blue', dash='dash', width=1.5),
        name="Original"
    ), row=2, col=1)
    fig_final.add_trace(go.Scatter(
        x=x, y=Y * 1000,
        mode='lines', line=dict(color='red', width=2),
        name="Optimizada"
    ), row=2, col=1)
    if show_limits and data.get("y_adm", 0) > 0:
        fig_final.add_trace(go.Scatter(
            x=[0, data["L"]], y=[data["y_adm"], data["y_adm"]],
            mode='lines', line=dict(color='red', dash='dot', width=1.5),
            name=f"Admisible ({data['y_adm']:.1f} mm)"
        ), row=2, col=1)

    # Row 3: Moment + Stress (secondary y)
    fig_final.add_trace(go.Scatter(
        x=x, y=M,
        mode='lines', line=dict(color='green', width=2),
        name="Momento M(x)"
    ), row=3, col=1)
    fig_final.add_trace(go.Scatter(
        x=x, y=sigma,
        mode='lines', line=dict(color='darkred', width=1.5, dash='dash'),
        name="Tensión σ",
    ), row=3, col=1, secondary_y=True)
    if show_limits:
        fig_final.add_trace(go.Scatter(
            x=[0, data["L"]], y=[11.25, 11.25],
            mode='lines', line=dict(color='red', dash='dot', width=1.5),
            name="σ_adm = 11.25 MPa"
        ), row=3, col=1, secondary_y=True)

    fig_final.update_layout(
        title=dict(
            text=f"Resultados Finales | L={data['L']}m | q={q}kN/m | Ahorro={saving:.1f}%",
            font=dict(size=14)
        ),
        height=700,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    fig_final.update_xaxes(title_text="Posición x (m)", row=3, col=1)
    fig_final.update_yaxes(title_text="Peralte (m)", row=1, col=1)
    fig_final.update_yaxes(title_text="Deflexión (mm)", row=2, col=1)
    fig_final.update_yaxes(title_text="Momento (kN·m)", row=3, col=1)
    fig_final.update_yaxes(title_text="Tensión (MPa)", row=3, col=1, secondary_y=True)
    apply_plotly_theme(fig_final)

    # ── Métricas auxiliares ──
    I0 = b * h0 ** 3 / 12
    V_orig = b * h0 * data["L"]
    dx = data["L"] / (len(x) - 1)
    V_opt = b * dx * np.sum(h_v)
    mejora_def = (1 - np.max(np.abs(Y)) / np.max(np.abs(Y_orig))) * 100

    # ── Tabs ──
    tab_graf, tab_res, tab_rep = st.tabs(["📊 Gráficos", "📋 Resultados", "📄 Reporte"])

    # ════════════════════════════════════════
    # TAB 1: GRÁFICOS
    # ════════════════════════════════════════
    with tab_graf:
        st.plotly_chart(fig_final, use_container_width=True, key="fig_graf")

    # ════════════════════════════════════════
    # TAB 2: RESULTADOS
    # ════════════════════════════════════════
    with tab_res:
        st.plotly_chart(fig_final, use_container_width=True, key="fig_res")

        st.markdown("---")

        # Metrics row
        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
        col_m1.metric("Iteraciones", data["iterations"])
        col_m2.metric("Ahorro Vol.", f"{saving:.1f}%", "↓" if saving > 0 else "↑")
        col_m3.metric("Peso Ahorrado", f"{weight:.2f} t", "↓")
        col_m4.metric("Deflexión Máx.", f"{np.max(np.abs(Y)) * 1000:.2f} mm")
        col_m5.metric("Error Final", f"{data['final_error']:.6f}")

        # Comparison table
        df_compare = pd.DataFrame({
            "Métrica": [
                "Volumen de material",
                "Peso total",
                "Deflexión máxima",
                "Inercia máxima"
            ],
            "Original": [
                f"{V_orig:.4f} m³",
                f"{V_orig * 2.5:.2f} t",
                f"{np.max(np.abs(Y_orig)) * 1000:.2f} mm",
                f"{I0:.6f} m⁴"
            ],
            "Optimizado": [
                f"{V_opt:.4f} m³",
                f"{V_opt * 2.5:.2f} t",
                f"{np.max(np.abs(Y)) * 1000:.2f} mm",
                f"{np.max(data['I']):.6f} m⁴"
            ],
            "Mejora": [
                f"{saving:.1f}%",
                f"{weight:.2f} t",
                f"{mejora_def:.1f}%",
                f"{np.max(data['I']) / I0:.2f}×"
            ]
        })
        st.dataframe(df_compare, use_container_width=True, hide_index=True)

        # Interpretation
        with st.expander("📋 Interpretación de resultados", expanded=True):
            def_adm = data.get("y_adm", 0)
            def_max_opt = np.max(np.abs(Y)) * 1000
            cumple_deflexion = def_max_opt <= def_adm if def_adm > 0 else True

            st.markdown(f"""
    - **Redistribución de material**: La optimización SIMP redistribuye el material desde las zonas de bajo momento flector (cerca de los apoyos) hacia las zonas de mayor solicitación (centro del vano). Esto explica el ahorro de volumen del **{saving:.1f}%**.
    - **Ahorro de material**: Se redujo el volumen de **{V_orig:.4f} m³** a **{V_opt:.4f} m³**, equivalente a **{weight:.2f} toneladas** de peso menos.
    - **Deflexión**: La deflexión máxima optimizada es de **{def_max_opt:.2f} mm** {'✅ dentro del límite admisible' if cumple_deflexion else '⚠️ supera el límite admisible'} de {def_adm:.2f} mm (L/300).
    - **Inercia**: La inercia máxima aumentó de {I0:.6f} m⁴ a **{np.max(data['I']):.6f} m⁴** ({np.max(data['I']) / I0:.2f}×), concentrando material donde más se necesita.
    - **Tensiones**: La tensión máxima de compresión es de {np.max(sigma):.2f} MPa {'✅ dentro del límite' if np.max(sigma) <= 11.25 else '⚠️ supera el límite'} de 11.25 MPa del hormigón.
    - **Convergencia**: El algoritmo alcanzó la convergencia en **{data['iterations']} iteraciones** con un error relativo de **{data['final_error']:.6f}**.
    """)

    # ════════════════════════════════════════
    # TAB 3: REPORTE
    # ════════════════════════════════════════
    with tab_rep:
        # ── CSV ──
        df_csv = pd.DataFrame({
            "Posicion_x_m": x,
            "Inercia_I_m4": data["I"],
            "Deflexion_Original_mm": Y_orig * 1000,
            "Deflexion_Optimizada_mm": Y * 1000,
            "Momento_Flector_kNm": M,
            "Altura_h_m": h_v,
            "Tension_sigma_MPa": sigma,
            "Area_Acero_As_mm2": data["As"]
        })
        csv_bytes = df_csv.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 CSV",
            data=csv_bytes,
            file_name="optimizacion_viga.csv",
            mime="text/csv",
            use_container_width=True
        )

        # ── PDF (siempre con fondo blanco para exportación) ──
        with io.BytesIO() as pdf_buf:
            # Usar estilo default para exportación (fondo blanco)
            with plt.style.context('default'):
                with PdfPages(pdf_buf) as pdf:
                    # Cover page
                    fig_cover, ax_cover = plt.subplots(figsize=(8.27, 11.69))
                    fig_cover.patch.set_facecolor('white')
                    ax_cover.set_facecolor('white')
                    ax_cover.axis('off')
                    ax_cover.text(0.5, 0.85, "Informe de Optimización Topológica",
                                  ha='center', fontsize=22, fontweight='bold', transform=ax_cover.transAxes,
                                  color='black')
                    ax_cover.text(0.5, 0.78, "Viga de Hormigón Armado — Método SIMP 1D",
                                  ha='center', fontsize=14, transform=ax_cover.transAxes,
                                  color='#333333')
                    ax_cover.text(0.5, 0.68, f"L = {data['L']} m    q = {q} kN/m    b×h₀ = {b}×{h0} m",
                                  ha='center', fontsize=12, transform=ax_cover.transAxes,
                                  color='#333333')
                    ax_cover.text(0.5, 0.62, f"Iteraciones: {data['iterations']}    Ahorro: {saving:.1f}%    Peso: {weight:.2f} t",
                                  ha='center', fontsize=12, transform=ax_cover.transAxes,
                                  color='#333333')
                    ax_cover.text(0.5, 0.55, f"Error final: {data['final_error']:.6f}    Deflexión máx: {np.max(np.abs(Y)) * 1000:.2f} mm",
                                  ha='center', fontsize=12, transform=ax_cover.transAxes,
                                  color='#333333')
                    pdf.savefig(fig_cover)
                    plt.close(fig_cover)

                    # Results page
                    fig_pdf, axes_pdf = plt.subplots(3, 1, figsize=(8.27, 8), sharex=True)
                    fig_pdf.patch.set_facecolor('white')
                    for ax in axes_pdf:
                        ax.set_facecolor('white')
                    axes_pdf[0].fill_between(x, -h_v / 2, h_v / 2, color='#d5dbdb',
                                             edgecolor='#2c3e50', linewidth=1.5)
                    axes_pdf[0].set_ylabel("Peralte (m)")
                    axes_pdf[0].set_title("Perfil de la viga optimizada", fontsize=11)
                    axes_pdf[0].grid(True, alpha=0.3)

                    axes_pdf[1].plot(x, Y_orig * 1000, 'b--', lw=1.5, label="Original", alpha=0.7)
                    axes_pdf[1].plot(x, Y * 1000, 'r-', lw=2, label="Optimizada")
                    if show_limits and data.get("y_adm", 0) > 0:
                        axes_pdf[1].axhline(data["y_adm"], color='red', linestyle=':', alpha=0.7,
                                            label=f"Admisible ({data['y_adm']:.1f} mm)")
                    axes_pdf[1].set_ylabel("Deflexión (mm)")
                    axes_pdf[1].set_title("Deflexión y(x)", fontsize=11)
                    axes_pdf[1].legend(fontsize=8)
                    axes_pdf[1].grid(True, alpha=0.3)

                    axes_pdf[2].plot(x, M, 'g-', lw=1.5, label="Momento M(x)")
                    axes_pdf[2].set_ylabel("Momento (kN·m)", color='g')
                    axes_pdf[2].tick_params(axis='y', labelcolor='g')
                    axes_pdf[2].grid(True, alpha=0.3)
                    axs2 = axes_pdf[2].twinx()
                    axs2.plot(x, sigma, 'darkred', lw=1.5, linestyle='--', label="Tensión σ")
                    if show_limits:
                        axs2.axhline(11.25, color='red', linestyle=':', alpha=0.5,
                                     label="σ_adm = 11.25 MPa")
                    axs2.set_ylabel("Tensión (MPa)", color='darkred')
                    axs2.tick_params(axis='y', labelcolor='darkred')
                    l1, lb1 = axes_pdf[2].get_legend_handles_labels()
                    l2, lb2 = axs2.get_legend_handles_labels()
                    axes_pdf[2].legend(l1 + l2, lb1 + lb2, fontsize=8, loc='upper right')
                    axes_pdf[2].set_xlabel("Posición x (m)")
                    axes_pdf[2].set_title("Momento Flector y Tensión", fontsize=11)
                    plt.tight_layout()
                    pdf.savefig(fig_pdf)
                    # También exportar figura principal como PNG
                    png_buf_viga = io.BytesIO()
                    fig_pdf.savefig(png_buf_viga, format='png', dpi=300, bbox_inches='tight')
                    plt.close(fig_pdf)

                    # Metrics summary page
                    fig_met, ax_met = plt.subplots(figsize=(8.27, 5))
                    fig_met.patch.set_facecolor('white')
                    ax_met.set_facecolor('white')
                    ax_met.axis('off')
                    metrics_text = (
                        f"MÉTRICAS DE OPTIMIZACIÓN\n\n"
                        f"Iteraciones:          {data['iterations']}\n"
                        f"Ahorro Volumen:      {saving:.1f}%\n"
                        f"Peso ahorrado:        {weight:.2f} t\n"
                        f"Deflexión máx. orig.: {np.max(np.abs(Y_orig)) * 1000:.2f} mm\n"
                        f"Deflexión máx. opt.:  {np.max(np.abs(Y)) * 1000:.2f} mm\n"
                        f"Límite admisible:     {data.get('y_adm', 0):.2f} mm\n"
                        f"Inercia máx.:         {np.max(data['I']):.6f} m⁴\n"
                        f"Tensión máx.:         {np.max(sigma):.2f} MPa\n"
                        f"Error final:          {data['final_error']:.6f}\n"
                    )
                    ax_met.text(0.1, 0.9, metrics_text, fontsize=12, fontfamily='monospace',
                                verticalalignment='top', transform=ax_met.transAxes, color='black')
                    pdf.savefig(fig_met)
                    plt.close(fig_met)

            pdf_bytes = pdf_buf.getvalue()
            # Obtener PNG guardado del buffer
            png_bytes_viga = png_buf_viga.getvalue()
            png_buf_viga.close()

        st.download_button(
            label="📄 PDF",
            data=pdf_bytes,
            file_name="informe_optimizacion_viga.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        # ── PNG ──
        st.download_button(
            label="🖼️ PNG",
            data=png_bytes_viga,
            file_name="resultados_optimizacion_viga.png",
            mime="image/png",
            use_container_width=True
        )

        # ── LaTeX ──
        saving_sign = f"{saving:.1f}\\%" if saving >= 0 else f"{saving:.1f}\\%"
        latex_table = f"""\\begin{{table}}[h]
\\centering
\\caption{{Resultados de Optimización Topológica de Viga}}
\\begin{{tabular}}{{lccc}}
\\toprule
Métrica & Original & Optimizado & Mejora \\\\
\\midrule
Volumen (m³) & {V_orig:.4f} & {V_opt:.4f} & {saving_sign} \\\\
Peso (t) & {V_orig * 2.5:.2f} & {V_opt * 2.5:.2f} & {weight:.2f} t \\\\
Deflexión máx. (mm) & {np.max(np.abs(Y_orig)) * 1000:.2f} & {np.max(np.abs(Y)) * 1000:.2f} & {mejora_def:.1f}\\% \\\\
Inercia máx. (m⁴) & {I0:.6f} & {np.max(data['I']):.6f} & {np.max(data['I']) / I0:.2f}$\\times$ \\\\
Iteraciones & — & {data['iterations']} & — \\\\
Error final & — & {data['final_error']:.6f} & — \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
        st.download_button(
            label="📐 LaTeX",
            data=latex_table.encode('utf-8'),
            file_name="tabla_resultados_optimizacion.tex",
            mime="text/plain",
            use_container_width=True
        )

        st.markdown("---")

        if st.button("🔄 Nueva Optimización", use_container_width=True):
            st.session_state.beam_opt_data = None
            st.rerun()
