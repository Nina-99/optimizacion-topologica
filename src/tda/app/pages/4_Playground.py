"""Página 4: Playground — Laboratorio Interactivo de Optimización.

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

from tda.optimization.beam_optimizer import BeamOptimizer
from tda.optimization.metric_simp import MetricaTDA_SIMP
from tda.core.betti2d import betti_doble_computo, diagramas_gudhi
from tda.app.theme import (
    apply_mpl_theme, apply_plotly_theme, responsive_style,
    page_header, ORANGE
)

from tda.app.download_utils import export_settings_ui
export_settings_ui()
apply_mpl_theme()

st.set_page_config(page_title="Playground — Laboratorio TDA-SIMP", layout="wide", page_icon="🧪")
st.markdown(responsive_style(), unsafe_allow_html=True)

st.markdown(page_header(
    "Playground — Laboratorio Interactivo",
    "Experimentá con optimización 1D, SIMP 2D, barridos y comparaciones"
), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SIDEBAR COMÚN
# ══════════════════════════════════════════════════════════════
st.sidebar.header("🧪 Modo de Experimentación")
modo = st.sidebar.radio(
    "Elegí un modo:",
    ["1D — Perfil de Altura", "2D — SIMP Topológico", "Barrido p (2,3,4)", "Comparación 1D vs 2D"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Parámetros Comunes")

# Parámetros físicos (compartidos)
L_mm = st.sidebar.slider("Longitud L (mm)", 50.0, 200.0, 120.0, 10.0)
F_N = st.sidebar.slider("Carga Puntual F (N)", 100.0, 5000.0, 1000.0, 100.0)
E_GPa = st.sidebar.number_input("Módulo E (GPa)", value=200.0, step=10.0)
sigma_adm = st.sidebar.number_input("σ_adm (MPa)", value=250.0, step=10.0)

st.sidebar.markdown("---")

# Parámetros SIMP (para modo 2D)
if modo != "1D — Perfil de Altura":
    penal = st.sidebar.slider("Penalización p", 1.0, 5.0, 3.0, 0.5)
    volfrac = st.sidebar.slider("Fracción volumen f_V", 0.2, 0.8, 0.5, 0.05)
    rmin = st.sidebar.slider("Radio filtro r_min", 1.0, 5.0, 2.4, 0.1)
    mesh_res = st.sidebar.selectbox("Resolución malla", ["40×20 (rápido)", "60×30 (tesis)", "80×40 (detalle)"], index=1)
    mesh_map = {"40×20 (rápido)": (40, 20), "60×30 (tesis)": (60, 30), "80×40 (detalle)": (80, 40)}
    nelx, nely = mesh_map[mesh_res]
    alpha_val = st.sidebar.number_input("α (métrica compuesta)", value=0.036, step=0.001, format="%.3f")

st.sidebar.markdown("---")
anim_delay = st.sidebar.slider("Delay animación (s)", 0.0, 0.5, 0.05, 0.01)

# Variables físicas
L = L_mm / 1000.0
F = F_N
E_s = E_GPa * 1e9
s_adm = sigma_adm * 1e6

# Colormap SIMP
cmap_simp = LinearSegmentedColormap.from_list("simp", ["#f0f0f0", "#2c3e50"])


# ══════════════════════════════════════════════════════════════
# MODO 1D — PERFIL DE ALTURA
# ══════════════════════════════════════════════════════════════
if modo == "1D — Perfil de Altura":
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
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Iteración", data['iteration'])
                c2.metric("Ahorro Material", f"{data['saving_pct']:.1f}%")
                max_sig = np.max(data['sigma_MPa'])
                c3.metric("Tensión Máx", f"{max_sig:.1f} MPa",
                          delta="✅ OK" if max_sig <= sigma_adm else "⚠️ Excede",
                          delta_color="normal" if max_sig <= sigma_adm else "inverse")
                c4.metric("Deflexión Máx", f"{np.max(np.abs(data['Y'])) * 1000:.2f} mm")

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
    st.subheader(f"2D — SIMP Topológico ({nelx}×{nely})")

    col_info, col_esq2 = st.columns([1, 1])
    with col_info:
        st.markdown(f"""
        **Modelo:** Viga en voladizo 2D, malla {nelx}×{nely} Q4.
        Borde izquierdo empotrado, carga puntual vertical en el extremo libre.
        f_V={volfrac}, p={penal}, r_min={rmin}.

        **Animación:** Se muestra cada ~5 iteraciones el diseño actualizado.
        **Post-hoc:** Al finalizar, se calcula β₀, β₁ y μ_α.
        """)

    # Esquema 2D
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    ax2.set_aspect('equal')
    rect_domain = Rectangle((0, 0), nelx, nely, facecolor='#ecf0f1', edgecolor='#2c3e50', lw=2)
    ax2.add_patch(rect_domain)
    wall2 = Rectangle((-2, 0), 2, nely, facecolor='#7f8c8d', hatch='///', edgecolor='black')
    ax2.add_patch(wall2)
    arr2 = FancyArrowPatch((nelx + 1, nely / 2), (nelx, nely / 2), arrowstyle='->', mutation_scale=20, lw=2.5, color='#e74c3c')
    ax2.add_patch(arr2)
    ax2.text(nelx + 2, nely / 2, f"F = {F_N} N", color='#e74c3c', fontweight='bold', va='center')
    ax2.set_xlim(-5, nelx + 8)
    ax2.set_ylim(-3, nely + 3)
    ax2.set_title(f"Dominio FEM — {nelx}×{nely} = {nelx * nely} elementos")
    ax2.axis('off')
    col_esq2.pyplot(fig2)
    plt.close(fig2)

    btn_2d = st.button("▶ Ejecutar SIMP 2D", type="primary", key="btn_2d")

    ph_rho = st.empty()
    ph_metrics_2d = st.container()

    if btn_2d:
        # Condiciones de borde
        nnx = nelx + 1
        node_load = (nely // 2) * nnx + nelx
        dof_load = 2 * node_load + 1
        dofs_fijos = []
        for j in range(nely + 1):
            dofs_fijos.extend([2 * j * nnx, 2 * j * nnx + 1])

        opt2d = MetricaTDA_SIMP(
            nex=nelx, ney=nely, E=E_s, nu=0.3,
            Lx=L_mm, Ly=L_mm * (nely / nelx), t=1.0,
            f_V=volfrac, p=penal, r_min=rmin, alpha=alpha_val,
            tol_c=1e-4, tol_rho=1e-2, max_iter=200
        )

        F_vec = np.zeros(2 * nnx * (nely + 1))
        F_vec[dof_load] = -F
        opt2d.definir_problema(F_vec, dofs_fijos)

        # Callback para animación
        def cb_simp(k, c, delta_c, delta_rho, rho):
            if k % 5 == 0 or k <= 3:
                fig_a, ax_a = plt.subplots(figsize=(10, 3))
                rho_grid = rho.reshape((nely, nelx))
                ax_a.imshow(rho_grid, cmap=cmap_simp, vmin=0, vmax=1, aspect='equal', origin='lower')
                ax_a.set_title(f"Iteración {k} — Compliance = {c:.2f} — Δc/c = {delta_c:.2e}")
                ax_a.axis('off')
                ph_rho.pyplot(fig_a)
                plt.close(fig_a)
                time.sleep(anim_delay)

        with st.spinner("Optimizando SIMP..."):
            opt2d.optimizar(callback=cb_simp, verbose=False)

        with st.spinner("Análisis topológico (TDA)..."):
            mu = opt2d.fase_tda(verbose=False)

        res = opt2d.obtener_resultados()

        # Diseño final
        fig_final, axes = plt.subplots(1, 2, figsize=(14, 4))

        rho_grid = res['rho_final'].reshape((nely, nelx))
        rho_tilde_grid = res['rho_tilde_final'].reshape((nely, nelx))

        axes[0].imshow(rho_grid, cmap=cmap_simp, vmin=0, vmax=1, aspect='equal', origin='lower')
        axes[0].set_title(f"ρ (densidad continua) — {res['n_iter']} iter")
        axes[0].axis('off')

        axes[1].imshow(rho_tilde_grid >= 0.5, cmap='binary', aspect='equal', origin='lower')
        conc = "✓" if res['betti_concordancia'] else "✗"
        axes[1].set_title(f"ρ̃ ≥ ½ (diseño binario) — β₀={res['beta0']}, β₁={res['beta1']} [{conc}]")
        axes[1].axis('off')

        plt.tight_layout()
        st.pyplot(fig_final)
        plt.close(fig_final)

        # Métricas
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Compliance", f"{res['c_final']:.2f}")
        c2.metric("μ_α", f"{res['mu']:.5f}")
        c3.metric("β₀", res['beta0'])
        c4.metric("β₁", res['beta1'])
        c5.metric("Iteraciones", res['n_iter'])
        c6.metric("Convergió", "✅" if res['converged'] else "❌")

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
        st.plotly_chart(fig_conv, use_container_width=True)

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
            st.plotly_chart(fig_dgm, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# MODO BARRIDO p — COMPARA p=2,3,4
# ══════════════════════════════════════════════════════════════
elif modo == "Barrido p (2,3,4)":
    st.subheader("Barrido de Penalización — p=2, p=3, p=4")
    st.markdown(f"""
    Corre SIMP 3 veces con la misma malla ({nelx}×{nely}) y compara:
    - Distribución de densidad final
    - Compliance, β₁, μ_α
    - Efecto del penalizador sobre la topología
    """)

    btn_barrido = st.button("▶ Ejecutar Barrido p", type="primary", key="btn_barrido")

    if btn_barrido:
        # Condiciones de borde
        nnx = nelx + 1
        node_load = (nely // 2) * nnx + nelx
        dof_load = 2 * node_load + 1
        dofs_fijos = []
        for j in range(nely + 1):
            dofs_fijos.extend([2 * j * nnx, 2 * j * nnx + 1])

        resultados = {}
        progress = st.progress(0, text="Ejecutando barrido...")

        for idx, p_val in enumerate([2, 3, 4]):
            progress.progress((idx) / 3, text=f"Optimizando p={p_val}...")
            opt_b = MetricaTDA_SIMP(
                nex=nelx, ney=nely, E=E_s, nu=0.3,
                Lx=L_mm, Ly=L_mm * (nely / nelx), t=1.0,
                f_V=volfrac, p=p_val, r_min=rmin, alpha=alpha_val,
                tol_c=1e-4, tol_rho=1e-2, max_iter=200
            )
            F_vec = np.zeros(2 * nnx * (nely + 1))
            F_vec[dof_load] = -F
            opt_b.definir_problema(F_vec, dofs_fijos)
            opt_b.optimizar(verbose=False)
            mu_b = opt_b.fase_tda(verbose=False)
            resultados[p_val] = opt_b.obtener_resultados()

        progress.progress(1.0, text="Barrido completo")

        # Mostrar diseños lado a lado
        fig_barrido, axes = plt.subplots(2, 3, figsize=(15, 8))
        for col_idx, p_val in enumerate([2, 3, 4]):
            res_p = resultados[p_val]
            rho_grid = res_p['rho_final'].reshape((nely, nelx))
            rho_bin = (res_p['rho_tilde_final'].reshape((nely, nelx)) >= 0.5).astype(float)

            axes[0, col_idx].imshow(rho_grid, cmap=cmap_simp, vmin=0, vmax=1, aspect='equal', origin='lower')
            axes[0, col_idx].set_title(f"p={p_val} — ρ (continuo)")
            axes[0, col_idx].axis('off')

            axes[1, col_idx].imshow(rho_bin, cmap='binary', aspect='equal', origin='lower')
            conc = "✓" if res_p['betti_concordancia'] else "✗"
            axes[1, col_idx].set_title(f"ρ̃≥½ — β₁={res_p['beta1']} [{conc}]")
            axes[1, col_idx].axis('off')

        plt.tight_layout()
        st.pyplot(fig_barrido)
        plt.close(fig_barrido)

        # Tabla comparativa
        import pandas as pd
        df_comp = pd.DataFrame({
            "Métrica": ["Compliance", "β₀", "β₁", "μ_α", "Iteraciones", "Convergió"],
            "p=2": [f"{resultados[2]['c_final']:.2f}", resultados[2]['beta0'], resultados[2]['beta1'],
                    f"{resultados[2]['mu']:.5f}", resultados[2]['n_iter'], "✅" if resultados[2]['converged'] else "❌"],
            "p=3": [f"{resultados[3]['c_final']:.2f}", resultados[3]['beta0'], resultados[3]['beta1'],
                    f"{resultados[3]['mu']:.5f}", resultados[3]['n_iter'], "✅" if resultados[3]['converged'] else "❌"],
            "p=4": [f"{resultados[4]['c_final']:.2f}", resultados[4]['beta0'], resultados[4]['beta1'],
                    f"{resultados[4]['mu']:.5f}", resultados[4]['n_iter'], "✅" if resultados[4]['converged'] else "❌"],
        })
        st.dataframe(df_comp.set_index("Métrica"), use_container_width=True)

        # Gráfico de barras: Compliance y μ_α
        fig_bar = make_subplots(rows=1, cols=2, subplot_titles=["Compliance", "μ_α"])
        for p_val in [2, 3, 4]:
            fig_bar.add_trace(go.Bar(
                x=[f"p={p_val}"], y=[resultados[p_val]['c_final']],
                name=f"p={p_val}", showlegend=False
            ), row=1, col=1)
            fig_bar.add_trace(go.Bar(
                x=[f"p={p_val}"], y=[resultados[p_val]['mu']],
                name=f"p={p_val}", showlegend=False
            ), row=1, col=2)
        fig_bar.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# MODO COMPARACIÓN 1D vs 2D
# ══════════════════════════════════════════════════════════════
elif modo == "Comparación 1D vs 2D":
    st.subheader("Comparación Lado a Lado — 1D vs 2D")
    st.markdown("""
    Ejecuta ambos optimizadores en el mismo problema y compara:
    - **1D:** Perfil de altura analítico (viga rectangular variable)
    - **2D:** SIMP topológico (distribución de material en malla)

    El 1D es más rápido pero solo optimiza la forma.
    El 2D optimiza la topología completa (agujeros, islas, ramas).
    """)

    btn_cmp = st.button("▶ Ejecutar Comparación", type="primary", key="btn_cmp")

    if btn_cmp:
        # ── 1D ──
        h0_cmp = 40.0 / 1000.0
        b_cmp = 0.001
        opt1d = BeamOptimizer(b=b_cmp, h0=h0_cmp, p=3, N=61, E_s=E_s, sigma_adm=s_adm, max_iter=80)
        res1d = opt1d.optimizar_viga_completo(L, F, callback=None)

        # ── 2D ──
        nnx = nelx + 1
        node_load = (nely // 2) * nnx + nelx
        dof_load = 2 * node_load + 1
        dofs_fijos = []
        for j in range(nely + 1):
            dofs_fijos.extend([2 * j * nnx, 2 * j * nnx + 1])

        opt2d_c = MetricaTDA_SIMP(
            nex=nelx, ney=nely, E=E_s, nu=0.3,
            Lx=L_mm, Ly=L_mm * (nely / nelx), t=1.0,
            f_V=volfrac, p=penal, r_min=rmin, alpha=alpha_val,
            tol_c=1e-4, tol_rho=1e-2, max_iter=200
        )
        F_vec = np.zeros(2 * nnx * (nely + 1))
        F_vec[dof_load] = -F
        opt2d_c.definir_problema(F_vec, dofs_fijos)

        with st.spinner("Optimizando 2D..."):
            opt2d_c.optimizar(verbose=False)
        with st.spinner("TDA post-hoc..."):
            mu_c = opt2d_c.fase_tda(verbose=False)
        res2d = opt2d_c.obtener_resultados()

        # ── Visualización lado a lado ──
        fig_cmp, axes = plt.subplots(2, 2, figsize=(14, 8))

        # 1D — Perfil
        axes[0, 0].plot(res1d['x'], res1d['h_v'] / 2, color='#2980b9', lw=2)
        axes[0, 0].plot(res1d['x'], -res1d['h_v'] / 2, color='#2980b9', lw=2)
        axes[0, 0].fill_between(res1d['x'], -res1d['h_v'] / 2, res1d['h_v'] / 2, color='#3498db', alpha=0.3)
        axes[0, 0].plot([0, L], [h0_cmp / 2, h0_cmp / 2], 'k--', alpha=0.3)
        axes[0, 0].plot([0, L], [-h0_cmp / 2, -h0_cmp / 2], 'k--', alpha=0.3)
        axes[0, 0].set_title(f"1D — Perfil Óptimo (ahorro {res1d['saving_pct']:.1f}%)")
        axes[0, 0].set_ylabel("Altura (m)")
        axes[0, 0].set_xlim(0, L)

        # 1D — Tensión
        axes[1, 0].plot(res1d['x'], res1d['sigma_MPa'], color='#8e44ad', lw=2)
        axes[1, 0].axhline(sigma_adm, color='red', linestyle='--', alpha=0.5, label="σ_adm")
        axes[1, 0].set_title("1D — Tensión σ(x) [MPa]")
        axes[1, 0].set_xlabel("Distancia x (m)")
        axes[1, 0].legend()

        # 2D — ρ continuo
        rho_grid_c = res2d['rho_final'].reshape((nely, nelx))
        axes[0, 1].imshow(rho_grid_c, cmap=cmap_simp, vmin=0, vmax=1, aspect='equal', origin='lower')
        conc_c = "✓" if res2d['betti_concordancia'] else "✗"
        axes[0, 1].set_title(f"2D — SIMP (c={res2d['c_final']:.2f}, β₁={res2d['beta1']})")
        axes[0, 1].axis('off')

        # 2D — Binario
        rho_bin_c = (res2d['rho_tilde_final'].reshape((nely, nelx)) >= 0.5).astype(float)
        axes[1, 1].imshow(rho_bin_c, cmap='binary', aspect='equal', origin='lower')
        axes[1, 1].set_title(f"2D — ρ̃≥½ [{conc_c}]")
        axes[1, 1].axis('off')

        plt.tight_layout()
        st.pyplot(fig_cmp)
        plt.close(fig_cmp)

        # Tabla resumen
        st.markdown("### Resumen Comparativo")
        df_resumen = pd.DataFrame({
            "Métrica": ["Ahorro material", "Compliance", "β₁", "μ_α", "Iteraciones"],
            "1D (Perfil)": [f"{res1d['saving_pct']:.1f}%", "N/A (analítico)", "N/A", "N/A", res1d['iterations']],
            "2D (SIMP)": [f"—", f"{res2d['c_final']:.2f}", res2d['beta1'],
                          f"{res2d['mu']:.5f}", res2d['n_iter']]
        })
        st.dataframe(df_resumen.set_index("Métrica"), use_container_width=True)


# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("---")
from tda.app.theme import footer_style
st.markdown(f"""
<div style="{footer_style()}">
    <p style="margin:0;color:{ORANGE};font-weight:700;">
        Playground — Laboratorio Interactivo de Optimización
    </p>
    <p style="margin:0.3rem 0 0 0;">
        BeamOptimizer (1D) · MetricaTDA_SIMP (2D) · GUDHI (TDA)
    </p>
</div>
""", unsafe_allow_html=True)
