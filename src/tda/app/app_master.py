import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from ripser import ripser
import io
from sklearn.decomposition import PCA

# Importación de la lógica refactorizada y separada de la UI
from tda.optimization.metric_simp import MetricaTDA_SIMP
from tda.processing.sampling import generate_cloud, add_gaussian_noise
from tda.analysis.metrics import compute_kmeans_accuracy, verify_betti_numbers
from tda.analysis.stability import compute_noise_sweep
from tda.visualization.plots_tda import plot_stability_chart, interpret_stability
from tda.app.pages.page_tda_kmedias import run as run_tda_kmedias

# ==========================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================
st.set_page_config(page_title="TDA-SIMP Master Suite", layout="wide", page_icon="📐")
st.title("Plataforma de Optimización SIMP y Análisis Topológico (TDA)")
st.markdown("---")

# ==========================================
# CONFIGURACIÓN EN SIDEBAR (UI/UX Improvement 1)
# ==========================================
st.sidebar.title("Parámetros de Configuración")

# Selector de página
page = st.sidebar.radio(
    "Seleccione el Análisis", 
    ["🏗️ 1. Optimización SIMP (H.E.2)", "📊 2. TDA vs K-Medias (H.E.1)", "📈 3. Optimización Topológica"],
    key="navigation_page"
)

st.sidebar.markdown("---")

if page == "🏗️ 1. Optimización SIMP (H.E.2)":
    st.sidebar.header("🏗️ Optimización SIMP (Pestaña 1)")
    malla_opcion = st.sidebar.selectbox("Resolución de Malla", ["60x30 (Caso Tesis)", "40x40 (1600 elem)", "80x80 (6400 elem)"], key="simp_malla")
    volfrac = st.sidebar.slider("Fracción de Volumen", 0.1, 0.9, 0.5, 0.05, key="simp_volfrac_input")
    penal = st.sidebar.number_input("Factor Penalización (p)", value=3.0, step=1.0, key="simp_penal")
    rmin = st.sidebar.number_input("Radio Filtro", value=1.5, step=0.1, key="simp_rmin")
    alpha = st.sidebar.number_input("Peso α (métrica μ_α)", value=0.012, step=0.001, format="%.3f", key="simp_alpha")
elif page == "📊 2. TDA vs K-Medias (H.E.1)":
    pass  # sidebar handled inside run_tda_kmedias()
else:
    st.sidebar.header("📈 Optimización Topológica")
    # Parámetros iniciales coherentes con optimizacion_topologica.py
    L = st.sidebar.slider("Longitud de Viga (m)", 6.0, 20.0, 12.0, 0.5, key="beam_length")
    q = st.sidebar.slider("Carga Distribuida (kN/m)", 10.0, 80.0, 30.0, 1.0, key="beam_load")
    E = st.sidebar.number_input("Módulo de Elasticidad (kPa)", value=30000000.0, step=1000000.0, key="beam_E")
    b = st.sidebar.number_input("Base de la Viga (m)", value=0.30, step=0.05, key="beam_b")
    h0 = st.sidebar.number_input("Altura de la Viga (m)", value=0.80, step=0.1, key="beam_h0")
    p = st.sidebar.number_input("Factor de Penalización", value=3, min_value=1, max_value=5, key="beam_p")
    N = st.sidebar.number_input("Número de Nodos", value=101, min_value=50, max_value=200, step=10, key="beam_N")
    
    # Controles adicionales
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

# ==========================================
# FUNCIÓN DE ANIMACIÓN SIMP CON PLOTLY
# ==========================================
def crear_animacion_simp(rho_hist, c_hist, nex, ney, volfrac, penal):
    """
    Crea una figura Plotly animada con la evolución del proceso SIMP.
    Incluye slider de iteración, botones Play/Pause/Reset,
    y gráfico de convergencia sincronizado.
    """
    n_frames = len(rho_hist)
    
    # Downsample si hay demasiados frames (>50)
    if n_frames > 50:
        step = max(1, n_frames // 50)
        idx = list(range(0, n_frames, step))
        if idx[-1] != n_frames - 1:
            idx.append(n_frames - 1)
        rho_hist_ds = [rho_hist[i] for i in idx]
        c_hist_ds = [c_hist[i] for i in idx]
        n_frames_ds = len(rho_hist_ds)
    else:
        rho_hist_ds = rho_hist
        c_hist_ds = c_hist
        idx = list(range(n_frames))
        n_frames_ds = n_frames
    
    # Crear figura con subplots
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.12,
        subplot_titles=(
            "Distribución de Material (ρ)",
            "Convergencia de Compliance c(ρ)"
        )
    )
    
    # Frame inicial (iteración 0)
    rho_2d_0 = rho_hist_ds[0].reshape(ney, nex)
    
    # Heatmap de densidades
    fig.add_trace(
        go.Heatmap(
            z=rho_2d_0,
            colorscale='gray_r',
            zmin=0, zmax=1,
            colorbar=dict(title='ρ', x=1.02, len=0.7, thickness=15),
            hovertemplate='x: %{x}<br>y: %{y}<br>ρ: %{z:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Línea de convergencia (vacía inicialmente, se llena en frames)
    fig.add_trace(
        go.Scatter(
            x=[], y=[],
            mode='lines',
            line=dict(color='royalblue', width=2),
            name='Compliance',
            hovertemplate='Iter: %{x}<br>c: %{y:.4f}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Línea vertical de iteración actual
    fig.add_trace(
        go.Scatter(
            x=[], y=[],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Iteración actual',
            showlegend=True
        ),
        row=2, col=1
    )
    
    # Crear frames para la animación
    frames = []
    for k, (i_frame, (rho_k, c_k)) in enumerate(zip(idx, zip(rho_hist_ds, c_hist_ds))):
        rho_2d_k = rho_k.reshape(ney, nex)
        iter_real = i_frame + 1  # 1-indexed
        
        # Compliance history UP TO this iteration
        c_up_to = c_hist[:i_frame + 1]
        
        # Meter toda la info en una sola string para el título
        n_solid = int(np.sum(rho_k > 0.5))
        pct_solid = 100 * n_solid / len(rho_k)
        
        frame = go.Frame(
            data=[
                # Heatmap
                go.Heatmap(
                    z=rho_2d_k,
                    colorscale='gray_r',
                    zmin=0, zmax=1,
                    colorbar=dict(title='ρ', x=1.02, len=0.7, thickness=15)
                ),
                # Convergence line
                go.Scatter(
                    x=np.arange(1, len(c_up_to) + 1),
                    y=c_up_to,
                    mode='lines',
                    line=dict(color='royalblue', width=2)
                ),
                # Current iteration marker
                go.Scatter(
                    x=[iter_real, iter_real],
                    y=[min(c_hist), max(c_hist)],
                    mode='lines',
                    line=dict(color='red', width=2, dash='dash')
                )
            ],
            name=str(k),
            layout=go.Layout(
                annotations=[dict(
                    x=0.5, y=1.0, xref='paper', yref='paper',
                    text=f'<b>Iteración {iter_real}/{n_frames}</b> | '
                         f'c = {c_k:.4f} | Sólidos: {n_solid}/{len(rho_k)} ({pct_solid:.1f}%)',
                    showarrow=False, font=dict(size=13, color='#2c3e50'),
                    align='center', bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#bdc3c7', borderwidth=1,
                    xanchor='center', yanchor='bottom'
                )]
            )
        )
        frames.append(frame)
    
    fig.frames = frames
    
    # Determinar rango y del convergence plot
    c_min = min(c_hist)
    c_max = max(c_hist)
    c_range = c_max - c_min if c_max != c_min else 1.0
    
    # Layout general
    fig.update_layout(
        title=dict(
            text=f'<b>Evolución de la Optimización SIMP</b><br>'
                 f'<sup>Malla {nex}×{ney} | f_V={volfrac} | p={penal}</sup>',
            font=dict(size=16),
            x=0.5, xanchor='center'
        ),
        height=650,
        hovermode='x unified',
        # Ejes del heatmap
        xaxis1=dict(visible=False, range=[-0.5, nex - 0.5]),
        yaxis1=dict(visible=False, range=[-0.5, ney - 0.5], scaleanchor='x', autorange='reversed'),
        # Ejes de la convergencia
        xaxis2=dict(
            title='Iteración',
            range=[0, n_frames + 1],
            gridcolor='#ecf0f1',
            zeroline=False
        ),
        yaxis2=dict(
            title='Compliance c(ρ)',
            type='log',
            range=[
                np.log10(max(c_min - 0.1 * c_range, 1e-10)),
                np.log10(c_max + 0.1 * c_range)
            ],
            gridcolor='#ecf0f1',
            zeroline=False
        ),
        # Slider
        sliders=[{
            'currentvalue': {
                'prefix': 'Iteración: ',
                'font': {'size': 14, 'color': '#2c3e50'},
                'xanchor': 'center'
            },
            'len': 0.92,
            'x': 0.04,
            'y': 0.0,
            'pad': {'t': 40, 'b': 10},
            'font': {'size': 11},
            'steps': [
                {
                    'args': [[str(k)], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': str(idx[k] + 1),
                    'method': 'animate'
                }
                for k in range(n_frames_ds)
            ]
        }],
        # Botones de control
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'x': 0.0,
            'y': -0.05,
            'xanchor': 'left',
            'yanchor': 'top',
            'font': {'size': 12},
            'buttons': [
                {
                    'label': '▶ Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 150, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                },
                {
                    'label': '⏸ Pause',
                    'method': 'animate',
                    'args': [[], {
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                },
                {
                    'label': '⏪ Reset',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': True},
                        'fromcurrent': False,
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ]
        }],
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
    )
    
    return fig


# ==========================================
# FUNCIONES AUXILIARES (TDA UI)
# ==========================================
# ==========================================
# RENDERIZADO DEL PANEL PRINCIPAL
# ==========================================
if page == "🏗️ 1. Optimización SIMP (H.E.2)":
    # ------------------------------------------
    # PÁGINA 1: SIMP (MetricaTDA_SIMP)
    # ------------------------------------------
    st.header("Optimización Estructural 2D (SIMP + TDA)")

    ejecutar_simp = st.button("▶ Iniciar Optimización SIMP", type="primary")
    st.markdown("---")

    col_res1, col_res2 = st.columns([2, 1])
    with col_res1:
        st.subheader("Evolución del Diseño")
        plot_ph = st.empty()
    with col_res2:
        st.subheader("Métricas de Control")
        metric_ph = st.empty()
        pbar = st.empty()

    if ejecutar_simp:
        # Parsear malla
        dims = malla_opcion.split(" ")[0].split("x")
        nelx, nely = int(dims[0]), int(dims[1])

        progress_bar = pbar.progress(0)
        history = []

        # Condiciones de contorno: viga en voladizo
        nnx = nelx + 1
        nny = nely + 1
        n_dof = 2 * nnx * nny

        # Empotramiento en borde izquierdo (x=0): todos los nodos columna 0
        dofs_fijos = np.arange(0, 2 * (nely + 1))

        # Carga puntual en centro del borde derecho: nodo (nelx, nely//2), F_y = -1
        node_load = (nely // 2) * nnx + nelx
        F = np.zeros(n_dof)
        F[2 * node_load + 1] = -1.0

        # Callback para Streamlit
        def ui_callback_simp(k, c, delta_c, delta_rho, rho):
            history.append({
                "Iteration": int(k),
                "Compliance": float(c),
                "delta_c": float(delta_c),
                "delta_rho": float(delta_rho)
            })
            if k % 2 == 0:
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.imshow(-rho.reshape(nely, nelx), cmap='gray', aspect='equal')
                ax.axis('off')
                plot_ph.pyplot(fig)
                plt.close(fig)
                metric_ph.markdown(f"**Iteración:** {k} | **Compliance:** {c:.4f}")
            progress_bar.progress(min(k / 200, 1.0))

        # Instancia del optimizador MetricaTDA_SIMP
        m = MetricaTDA_SIMP(nex=nelx, ney=nely, f_V=volfrac, p=penal, r_min=rmin, alpha=alpha)

        m.definir_problema(F, dofs_fijos)

        with st.spinner('Optimizando SIMP...'):
            m.optimizar(callback=ui_callback_simp, verbose=False)

        with st.spinner('Analizando topología (TDA)...'):
            mu = m.fase_tda(verbose=False)

        st.success("¡Optimización Finalizada!")

        # Referencia: bloque sólido (f_V=1.0, una iteración)
        m_sol = MetricaTDA_SIMP(nex=nelx, ney=nely, f_V=1.0, p=penal, r_min=rmin, alpha=alpha, max_iter=1)
        m_sol.definir_problema(F, dofs_fijos)
        m_sol.optimizar(verbose=False)
        c_solido = m_sol.c_final
        reduccion_pct = ((c_solido - m.c_final) / c_solido) * 100

        # Guardar resultados en session_state
        st.session_state.simp_rho_final = m.rho_final
        st.session_state.simp_c_final = m.c_final
        st.session_state.simp_beta1 = m.beta1
        st.session_state.simp_beta0 = m.beta0
        st.session_state.simp_mu = m.mu
        st.session_state.simp_n_iter = m.n_iter
        st.session_state.simp_converged = m.converged
        st.session_state.simp_dgm1 = m.dgm1
        st.session_state.simp_nube = m.nube
        st.session_state.simp_eps_star = m.eps_star
        st.session_state.simp_c_hist = np.array(m.c_hist)
        st.session_state.simp_rho_hist = m.rho_hist
        st.session_state.simp_volfrac = volfrac
        st.session_state.simp_penal_stored = penal
        st.session_state.simp_alpha_stored = alpha
        st.session_state.simp_nex = nelx
        st.session_state.simp_ney = nely
        st.session_state.simp_history = history
        st.session_state.simp_t_simp = m.t_simp
        st.session_state.simp_t_tda = m.t_tda
        st.session_state.simp_reduccion = reduccion_pct
        st.session_state.simp_c_solido = c_solido
        st.session_state.simp_optimized = True

    # Renderizar resultados si existen
    # ─────────────────────────────────────────────────────────
    # Renderizar resultados si existen (con tabs)
    # ─────────────────────────────────────────────────────────
    if st.session_state.get('simp_optimized', False):
        
        # Extraer datos de session_state
        rho_final = st.session_state.simp_rho_final
        nex_ = st.session_state.simp_nex
        ney_ = st.session_state.simp_ney
        c_final = st.session_state.simp_c_final
        c_hist = st.session_state.simp_c_hist
        rho_hist = st.session_state.get('simp_rho_hist', None)
        dgm1 = st.session_state.simp_dgm1
        nube = st.session_state.simp_nube
        eps_star = st.session_state.simp_eps_star
        
        tab_evol, tab_res, tab_tda, tab_report = st.tabs([
            "📈 Evolución", "🏗️ Resultados", "🔬 Análisis TDA", "📊 Reporte"
        ])
        
        # ═══════════════════════════════════════════════════
        # TAB 1: EVOLUCIÓN INTERACTIVA
        # ═══════════════════════════════════════════════════
        with tab_evol:
            st.subheader("Animación del Proceso de Optimización")
            
            volfrac = st.session_state.simp_volfrac
            penal = st.session_state.simp_penal_stored
            
            if rho_hist is not None and len(rho_hist) > 1:
                fig_anim = crear_animacion_simp(rho_hist, c_hist, nex_, ney_, volfrac, penal)
                st.plotly_chart(fig_anim, use_container_width=True)
            else:
                st.info("No hay historial de iteraciones para animar.")
                fig_final, ax_final = plt.subplots(figsize=(8, 4))
                ax_final.imshow(rho_final.reshape(ney_, nex_), cmap='gray_r', aspect='equal')
                ax_final.axis('off')
                ax_final.set_title("Distribución Final de Material")
                st.pyplot(fig_final)
                plt.close(fig_final)
            
            # Métricas rápidas siempre visibles
            st.markdown("---")
            col_q1, col_q2, col_q3, col_q4 = st.columns(4)
            col_q1.metric("Iteraciones", st.session_state.simp_n_iter)
            col_q2.metric("Compliance Final", f"{st.session_state.simp_c_final:.4f}")
            col_q3.metric("β₁ (Agujeros)", st.session_state.simp_beta1)
            col_q4.metric("μ_α Compuesta", f"{st.session_state.simp_mu:.5f}")
        
        # ═══════════════════════════════════════════════════
        # TAB 2: RESULTADOS FINALES (figura 2×2)
        # ═══════════════════════════════════════════════════
        with tab_res:
            st.subheader("Resultados Finales de la Optimización")
            
            fig_res, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig_res.suptitle(f"Resultados SIMP | {nex_}×{ney_} | f_V={st.session_state.simp_volfrac} | μ_α={st.session_state.simp_mu:.4f}", fontsize=14)
            
            # (0,0): Mapa de densidades
            im = axes[0, 0].imshow(rho_final.reshape(ney_, nex_), cmap='gray_r', aspect='equal', vmin=0, vmax=1)
            axes[0, 0].set_title(f"Distribución de Material ρ* | c = {c_final:.4f}")
            axes[0, 0].axis('off')
            plt.colorbar(im, ax=axes[0, 0], label='Densidad ρ', fraction=0.046, pad=0.04)
            
            # (0,1): Diagrama de persistencia H1
            if dgm1 is not None and len(dgm1) > 0:
                finite = np.isfinite(dgm1[:, 1])
                if np.any(finite):
                    pers = dgm1[finite, 1] - dgm1[finite, 0]
                    sc = axes[0, 1].scatter(dgm1[finite, 0], dgm1[finite, 1], c=pers, cmap='RdYlGn', s=60, zorder=5, label='Ciclos H₁')
                    plt.colorbar(sc, ax=axes[0, 1], label='Persistencia', fraction=0.046, pad=0.04)
                    max_val = np.max(dgm1[finite]) * 1.1 if len(dgm1[finite]) > 0 else 2.0
                    axes[0, 1].plot([0, max_val], [0, max_val], 'k--', alpha=0.4, label='Diagonal (ruido)')
                    if eps_star is not None:
                        axes[0, 1].axhline(y=eps_star / 2, color='red', linestyle=':', alpha=0.7, label=f'Umbral = ε*/2')
                    axes[0, 1].legend(fontsize=8)
            axes[0, 1].set_xlabel("Nacimiento b")
            axes[0, 1].set_ylabel("Muerte d")
            axes[0, 1].set_title(f"Diagrama Persistencia H₁ | β₁ significativo = {st.session_state.simp_beta1}")
            axes[0, 1].grid(True, linestyle='--', alpha=0.3)
            
            # (1,0): Convergencia
            if c_hist is not None and len(c_hist) > 0:
                axes[1, 0].semilogy(range(1, len(c_hist) + 1), c_hist, 'b-', linewidth=1.5, label='c(k)')
                axes[1, 0].axhline(y=c_final, color='r', linestyle='--', alpha=0.7, label=f'c* = {c_final:.4f}')
                axes[1, 0].legend()
            axes[1, 0].set_xlabel("Iteración k")
            axes[1, 0].set_ylabel("Compliance c(ρ)")
            axes[1, 0].set_title(f"Convergencia SIMP | {len(c_hist)} iteraciones")
            axes[1, 0].grid(True, linestyle='--', alpha=0.3)
            
            # (1,1): Nube de puntos
            if nube is not None and len(nube) > 0:
                axes[1, 1].scatter(nube[:, 0], nube[:, 1], s=8, alpha=0.5, c='navy')
            axes[1, 1].set_xlim(0, nex_)
            axes[1, 1].set_ylim(0, ney_)
            axes[1, 1].set_aspect('equal')
            axes[1, 1].set_xlabel("x (elementos)")
            axes[1, 1].set_ylabel("y (elementos)")
            axes[1, 1].set_title(f"Nube X(ρ*) | β₀={st.session_state.simp_beta0} | β₁={st.session_state.simp_beta1} | μ_α={st.session_state.simp_mu:.4f}")
            axes[1, 1].grid(True, linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig_res)
            plt.close(fig_res)
        
        # ═══════════════════════════════════════════════════
        # TAB 3: ANÁLISIS TDA
        # ═══════════════════════════════════════════════════
        with tab_tda:
            st.subheader("Análisis Topológico (TDA)")
            
            # Métricas TDA en tarjetas visuales
            col_t1, col_t2, col_t3, col_t4 = st.columns(4)
            with col_t1:
                st.markdown("""
                <div style='background:#e8f4f8;padding:15px;border-radius:10px;text-align:center'>
                    <h3 style='margin:0;color:#2980b9'>β₀</h3>
                    <p style='font-size:2em;font-weight:bold;margin:0;color:#2c3e50'>{}</p>
                    <small>Componentes conexas</small>
                </div>
                """.format(st.session_state.simp_beta0), unsafe_allow_html=True)
            
            with col_t2:
                b1 = st.session_state.simp_beta1
                color = "#27ae60" if b1 == 0 else "#e74c3c"
                st.markdown("""
                <div style='background:#fef9e7;padding:15px;border-radius:10px;text-align:center'>
                    <h3 style='margin:0;color:#d35400'>β₁</h3>
                    <p style='font-size:2em;font-weight:bold;margin:0;color:{}'>{}</p>
                    <small>Agujeros topológicos</small>
                </div>
                """.format(color, b1), unsafe_allow_html=True)
            
            with col_t3:
                st.markdown("""
                <div style='background:#eafaf1;padding:15px;border-radius:10px;text-align:center'>
                    <h3 style='margin:0;color:#27ae60'>μ_α</h3>
                    <p style='font-size:2em;font-weight:bold;margin:0;color:#2c3e50'>{:.4f}</p>
                    <small>Métrica compuesta</small>
                </div>
                """.format(st.session_state.simp_mu), unsafe_allow_html=True)
            
            with col_t4:
                manufacturable = st.session_state.simp_beta1 == 0
                estado = "✅ APTA" if manufacturable else "⚠️ REVISAR"
                color_est = "#27ae60" if manufacturable else "#e74c3c"
                st.markdown("""
                <div style='background:#f4f6f7;padding:15px;border-radius:10px;text-align:center'>
                    <h3 style='margin:0;color:#7f8c8d'>Manufactura</h3>
                    <p style='font-size:1.3em;font-weight:bold;margin:0;color:{}'>{}</p>
                    <small>Verificación topológica</small>
                </div>
                """.format(color_est, estado), unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Diagrama de persistencia interactivo (Plotly)
            st.subheader("Diagrama de Persistencia Interactivo")
            
            # Filtrar puntos no finitos
            dgm1_local = st.session_state.simp_dgm1
            if dgm1_local is not None and len(dgm1_local) > 0:
                finite_mask = np.isfinite(dgm1_local).all(axis=1)
                dgm1_finite = dgm1_local[finite_mask]
            else:
                dgm1_finite = np.empty((0, 2))
            
            fig_tda = go.Figure()
            
            if len(dgm1_finite) > 0:
                pers = dgm1_finite[:, 1] - dgm1_finite[:, 0]
                fig_tda.add_trace(go.Scatter(
                    x=dgm1_finite[:, 0],
                    y=dgm1_finite[:, 1],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=pers,
                        colorscale='RdYlGn',
                        colorbar=dict(title='Persistencia'),
                        showscale=True,
                        line=dict(color='black', width=0.5)
                    ),
                    text=[f'Persistencia: {p:.3f}<br>Nacimiento: {b:.3f}<br>Muerte: {d:.3f}'
                          for (b, d), p in zip(dgm1_finite, pers)],
                    hovertemplate='%{text}<extra></extra>',
                    name='Ciclos H₁'
                ))
            
            # Diagonal
            if len(dgm1_finite) > 0:
                max_val = np.max(dgm1_finite) * 1.1
            else:
                max_val = 2.0
            fig_tda.add_trace(go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                line=dict(color='gray', dash='dash'),
                name='Nacimiento = Muerte'
            ))
            
            # Umbral
            eps_star_local = st.session_state.simp_eps_star
            if eps_star_local is not None and eps_star_local > 0:
                fig_tda.add_hline(
                    y=eps_star_local / 2,
                    line=dict(color='red', dash='dot'),
                    annotation_text=f'Umbral = ε*/2 = {eps_star_local/2:.3f}',
                    annotation_position='right'
                )
            
            fig_tda.update_layout(
                xaxis_title="Tiempo de Nacimiento (Birth)",
                yaxis_title="Tiempo de Muerte (Death)",
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
                margin=dict(l=40, r=40, t=40, b=40),
                height=500,
                hovermode='closest',
                plot_bgcolor='#fafafa',
                paper_bgcolor='white',
            )
            st.plotly_chart(fig_tda, use_container_width=True)
            
            st.markdown("---")
            
            # Información adicional
            with st.expander("📖 Interpretación de los invariantes topológicos"):
                st.markdown("""
                **β₀ (Número de Betti de dimensión 0):** Número de componentes conexas.
                - β₀ = 1 → El diseño es una sola pieza continua (ideal para manufactura).
                - β₀ > 1 → Hay partes desconectadas. Puede indicar necesidad de soportes.
                
                **β₁ (Número de Betti de dimensión 1):** Número de agujeros 1-dimensionales.
                - β₁ = 0 → Estructura sin agujeros internos (manufacturabilidad garantizada).
                - β₁ > 0 → Hay cavidades o túneles. Pueden ser intencionales o espurios.
                
                **μ_α = c + α·β₁:** Métrica compuesta TDA-SIMP.
                - Penaliza diseños con agujeros topológicos innecesarios.
                - Menor μ_α → mejor balance entre eficiencia mecánica y simplicidad topológica.
                """)
        
        # ═══════════════════════════════════════════════════
        # TAB 4: REPORTE Y EXPORTACIÓN
        # ═══════════════════════════════════════════════════
        with tab_report:
            st.subheader("Exportar Resultados")
            
            # ── Datos compartidos ─────────────────────────────────────────
            beta0 = st.session_state.simp_beta0
            beta1 = st.session_state.simp_beta1
            c_final = st.session_state.simp_c_final
            mu = st.session_state.simp_mu
            reduccion = st.session_state.simp_reduccion
            volfrac = st.session_state.simp_volfrac
            penal = st.session_state.simp_penal_stored
            alpha_val = st.session_state.simp_alpha_stored
            n_iter = st.session_state.simp_n_iter
            converged = st.session_state.simp_converged
            t_simp = st.session_state.simp_t_simp
            t_tda = st.session_state.simp_t_tda
            nex_ = st.session_state.simp_nex
            ney_ = st.session_state.simp_ney
            rho_final = st.session_state.simp_rho_final
            c_hist = st.session_state.simp_c_hist
            dgm1 = st.session_state.simp_dgm1
            nube = st.session_state.simp_nube
            eps_star = st.session_state.simp_eps_star
            manufacturable = beta1 == 0
            
            # ── Información del diseño ─────────────────────────────────────
            st.markdown("""
            <div style='background:linear-gradient(135deg,#667eea,#764ba2);padding:20px;border-radius:12px;margin-bottom:20px'>
                <h3 style='color:white;margin:0;text-align:center'>📦 Paquete de Resultados TDA-SIMP</h3>
                <p style='color:rgba(255,255,255,0.85);text-align:center;margin:5px 0 0'>
                    Todos los datos necesarios para tu tesis y presentación
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # ── Fila 1: PDF + PNG + LaTeX ──────────────────────────────────
            st.markdown("#### 📄 Documentos")
            col_r1, col_r2, col_r3 = st.columns(3)
            
            # ── PDF Profesional (multipágina) ──────────────────────────────
            with col_r1:
                pdf_buf = io.BytesIO()
                with PdfPages(pdf_buf) as pdf:
                    
                    # ── Página 1: Portada ──────────────────────────────────
                    fig_cover = plt.figure(figsize=(8.27, 11.69))  # A4 vertical
                    fig_cover.patch.set_facecolor('#f8f9fa')
                    ax_cover = fig_cover.add_axes([0.1, 0.1, 0.8, 0.8])
                    ax_cover.axis('off')
                    
                    # Título
                    ax_cover.text(0.5, 0.85, 'Reporte de Optimización TDA-SIMP',
                                  ha='center', va='center', fontsize=24, fontweight='bold',
                                  color='#2c3e50', transform=ax_cover.transAxes)
                    ax_cover.text(0.5, 0.78, 'Algoritmo 1 — Métrica Compuesta',
                                  ha='center', va='center', fontsize=14,
                                  color='#7f8c8d', transform=ax_cover.transAxes)
                    
                    # Línea decorativa
                    ax_cover.plot([0.2, 0.8], [0.74, 0.74], color='#3498db',
                                     linewidth=2, transform=ax_cover.transAxes)
                    
                    # Parámetros del problema
                    params_text = (
                        f"Malla: {nex_}×{ney_} elementos Q4\n"
                        f"Fracción de volumen: f_V = {volfrac}\n"
                        f"Penalización SIMP: p = {penal}\n"
                        f"Peso topológico: α = {alpha_val}\n"
                        f"Iteraciones: {n_iter} {'✓ Convergió' if converged else '✗ No convergió'}"
                    )
                    ax_cover.text(0.5, 0.60, params_text, ha='center', va='center',
                                  fontsize=11, color='#34495e', transform=ax_cover.transAxes,
                                  linespacing=1.8, family='monospace')
                    
                    # Resultados principales
                    ax_cover.plot([0.15, 0.85], [0.48, 0.48], color='#bdc3c7',
                                     linewidth=0.5, transform=ax_cover.transAxes)
                    
                    res_text = (
                        f"Compliance final:      c = {c_final:.5f}\n"
                        f"Reducción vs sólido:   {reduccion:.2f}%\n"
                        f"Componentes conexas:   β₀ = {beta0}\n"
                        f"Agujeros topológicos:  β₁ = {beta1}\n"
                        f"Métrica compuesta:     μ_α = {mu:.5f}\n"
                        f"Manufacturabilidad:    {'✓ APTA' if manufacturable else '⚠ REVISAR'}"
                    )
                    ax_cover.text(0.5, 0.38, res_text, ha='center', va='center',
                                  fontsize=11, color='#2c3e50', transform=ax_cover.transAxes,
                                  linespacing=1.8, family='monospace')
                    
                    # Tiempos de cómputo
                    ax_cover.text(0.5, 0.18,
                                  f"Tiempo SIMP: {t_simp:.2f}s  |  Tiempo TDA: {t_tda:.3f}s  |  Total: {t_simp+t_tda:.2f}s",
                                  ha='center', va='center', fontsize=9,
                                  color='#95a5a6', transform=ax_cover.transAxes)
                    
                    # Footer
                    ax_cover.text(0.5, 0.05,
                                  'Jorge Larry Copa Cruz · Maestría en Matemática · UAGRM 2026',
                                  ha='center', va='center', fontsize=8,
                                  color='#bdc3c7', transform=ax_cover.transAxes,
                                  style='italic')
                    
                    pdf.savefig(fig_cover, dpi=200)
                    plt.close(fig_cover)
                    
                    # ── Página 2: Figura 2×2 completa ──────────────────────
                    fig_res_pdf, axes_pdf = plt.subplots(2, 2, figsize=(10, 8))
                    fig_res_pdf.suptitle(f'Resultados de Optimización | {nex_}×{ney_} | f_V={volfrac} | μ_α={mu:.4f}',
                                         fontsize=14, fontweight='bold')
                    
                    # (0,0) Density
                    rho_2d = rho_final.reshape(ney_, nex_)
                    im = axes_pdf[0, 0].imshow(rho_2d, cmap='gray_r', aspect='equal', vmin=0, vmax=1)
                    axes_pdf[0, 0].set_title(f'Distribución ρ* | c={c_final:.4f}')
                    axes_pdf[0, 0].axis('off')
                    plt.colorbar(im, ax=axes_pdf[0, 0], label='ρ', fraction=0.046, pad=0.04)
                    
                    # (0,1) Persistence diagram
                    if dgm1 is not None and len(dgm1) > 0:
                        finite = np.isfinite(dgm1[:, 1])
                        if np.any(finite):
                            pers = dgm1[finite, 1] - dgm1[finite, 0]
                            sc = axes_pdf[0, 1].scatter(dgm1[finite, 0], dgm1[finite, 1],
                                                        c=pers, cmap='RdYlGn', s=40, zorder=5)
                            plt.colorbar(sc, ax=axes_pdf[0, 1], label='Persistencia', fraction=0.046, pad=0.04)
                            mv = np.max(dgm1[finite]) * 1.1
                            axes_pdf[0, 1].plot([0, mv], [0, mv], 'k--', alpha=0.4)
                            if eps_star:
                                axes_pdf[0, 1].axhline(eps_star/2, color='r', ls=':', alpha=0.7, label=f'ε*/2')
                            axes_pdf[0, 1].legend(fontsize=7)
                    axes_pdf[0, 1].set_title(f'Diagrama H₁ | β₁={beta1}')
                    axes_pdf[0, 1].set_xlabel('Birth'); axes_pdf[0, 1].set_ylabel('Death')
                    axes_pdf[0, 1].grid(True, alpha=0.3)
                    
                    # (1,0) Convergence
                    if c_hist is not None and len(c_hist) > 0:
                        axes_pdf[1, 0].semilogy(range(1, len(c_hist)+1), c_hist, 'b-', lw=1.5)
                        axes_pdf[1, 0].axhline(c_final, color='r', ls='--', alpha=0.7, label=f'c*={c_final:.4f}')
                        axes_pdf[1, 0].legend(fontsize=8)
                    axes_pdf[1, 0].set_title(f'Convergencia | {len(c_hist)} iters')
                    axes_pdf[1, 0].set_xlabel('Iteración k'); axes_pdf[1, 0].set_ylabel('Compliance c')
                    axes_pdf[1, 0].grid(True, alpha=0.3)
                    
                    # (1,1) Point cloud
                    if nube is not None and len(nube) > 0:
                        axes_pdf[1, 1].scatter(nube[:, 0], nube[:, 1], s=5, alpha=0.5, c='navy')
                    axes_pdf[1, 1].set_xlim(0, nex_); axes_pdf[1, 1].set_ylim(0, ney_)
                    axes_pdf[1, 1].set_aspect('equal')
                    axes_pdf[1, 1].set_title(f'Nube X(ρ*) | β₀={beta0} | β₁={beta1} | μ_α={mu:.4f}')
                    axes_pdf[1, 1].set_xlabel('x (elem.)'); axes_pdf[1, 1].set_ylabel('y (elem.)')
                    axes_pdf[1, 1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    pdf.savefig(fig_res_pdf, dpi=200)
                    plt.close(fig_res_pdf)
                    
                    # ── Página 3: Tabla de métricas ────────────────────────
                    fig_tab = plt.figure(figsize=(8.27, 11.69))
                    ax_tab = fig_tab.add_axes([0.1, 0.1, 0.8, 0.8])
                    ax_tab.axis('off')
                    
                    ax_tab.text(0.5, 0.95, 'Métricas de la Optimización',
                                ha='center', fontsize=18, fontweight='bold', color='#2c3e50',
                                transform=ax_tab.transAxes)
                    
                    # Crear tabla
                    col_labels = ['Métrica', 'Valor', 'Unidad']
                    rows = [
                        ['Compliance final', f'{c_final:.5f}', 'N·mm'],
                        ['Reducción vs sólido', f'{reduccion:.2f}', '%'],
                        ['Fracción de volumen', f'{volfrac}', '—'],
                        ['Penalización p', f'{penal}', '—'],
                        ['α (peso topológico)', f'{alpha_val}', '—'],
                        ['β₀ (componentes)', str(beta0), '—'],
                        ['β₁ (agujeros)', str(beta1), '—'],
                        ['μ_α (métrica compuesta)', f'{mu:.5f}', '—'],
                        ['Manufacturable', 'Sí' if manufacturable else 'No', '—'],
                        ['Iteraciones', str(n_iter), '—'],
                        ['Convergió', 'Sí' if converged else 'No', '—'],
                        ['Tiempo SIMP', f'{t_simp:.2f}', 's'],
                        ['Tiempo TDA', f'{t_tda:.3f}', 's'],
                        ['Tiempo total', f'{t_simp+t_tda:.2f}', 's'],
                        [f'Malla', f'{nex_}×{ney_}', 'elementos'],
                    ]
                    
                    table = ax_tab.table(cellText=rows, colLabels=col_labels,
                                         loc='center', cellLoc='center',
                                         colWidths=[0.35, 0.25, 0.15])
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1, 1.6)
                    
                    for (row, col), cell in table.get_celld().items():
                        if row == 0:
                            cell.set_facecolor('#3498db')
                            cell.set_text_props(color='white', fontweight='bold')
                        elif row % 2 == 0:
                            cell.set_facecolor('#f0f3f5')
                        cell.set_edgecolor('#dfe6e9')
                    
                    ax_tab.text(0.5, 0.02,
                                'Generado por Plataforma TDA-SIMP · Jorge Larry Copa Cruz · UAGRM 2026',
                                ha='center', fontsize=7, color='#bdc3c7',
                                transform=ax_tab.transAxes, style='italic')
                    
                    pdf.savefig(fig_tab, dpi=200)
                    plt.close(fig_tab)
                
                pdf_bytes = pdf_buf.getvalue()
                pdf_buf.close()
                
                st.download_button(
                    label="📄 PDF Profesional",
                    data=pdf_bytes,
                    file_name=f"reporte_TDA-SIMP_{nex_}x{ney_}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    help="PDF de 3 páginas: portada + figura + tabla de métricas"
                )
            
            # ── Exportar PNG 2×2 ───────────────────────────────────────────
            with col_r2:
                fig_png, axes_png = plt.subplots(2, 2, figsize=(12, 9))
                fig_png.suptitle(f'Resultados SIMP-TDA | {nex_}×{ney_} | f_V={volfrac} | μ_α={mu:.4f}',
                                 fontsize=13, fontweight='bold')
                
                # Density
                rho_2d = rho_final.reshape(ney_, nex_)
                axes_png[0, 0].imshow(rho_2d, cmap='gray_r', aspect='equal', vmin=0, vmax=1)
                axes_png[0, 0].set_title(f'Distribución ρ*')
                axes_png[0, 0].axis('off')
                
                # Persistence
                if dgm1 is not None and len(dgm1) > 0:
                    finite = np.isfinite(dgm1[:, 1])
                    if np.any(finite):
                        axes_png[0, 1].scatter(dgm1[finite, 0], dgm1[finite, 1], c='orange', marker='^', alpha=0.7)
                        mv = np.max(dgm1[finite]) * 1.1
                        axes_png[0, 1].plot([0, mv], [0, mv], 'k--', alpha=0.4)
                        if eps_star:
                            axes_png[0, 1].axhline(eps_star/2, color='r', ls=':', alpha=0.7)
                axes_png[0, 1].set_title('Diagrama H₁')
                axes_png[0, 1].set_xlabel('Birth'); axes_png[0, 1].set_ylabel('Death')
                axes_png[0, 1].grid(True, alpha=0.3)
                
                # Convergence
                if c_hist is not None and len(c_hist) > 0:
                    axes_png[1, 0].semilogy(range(1, len(c_hist)+1), c_hist, 'b-', lw=1.5)
                    axes_png[1, 0].axhline(c_final, color='r', ls='--', alpha=0.7)
                axes_png[1, 0].set_title('Convergencia')
                axes_png[1, 0].set_xlabel('Iteración'); axes_png[1, 0].set_ylabel('c')
                axes_png[1, 0].grid(True, alpha=0.3)
                
                # Point cloud
                if nube is not None and len(nube) > 0:
                    axes_png[1, 1].scatter(nube[:, 0], nube[:, 1], s=5, alpha=0.5, c='navy')
                axes_png[1, 1].set_title(f'Nube X(ρ*)')
                axes_png[1, 1].set_xlabel('x'); axes_png[1, 1].set_ylabel('y')
                axes_png[1, 1].set_aspect('equal')
                axes_png[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                png_buf = io.BytesIO()
                fig_png.savefig(png_buf, format='png', dpi=300, bbox_inches='tight')
                plt.close(fig_png)
                png_bytes = png_buf.getvalue()
                png_buf.close()
                
                st.download_button(
                    label="🖼️ PNG Alta Resolución",
                    data=png_bytes,
                    file_name=f"figura_resultados_{nex_}x{ney_}.png",
                    mime="image/png",
                    use_container_width=True,
                    help="Figura 2×2 a 300 DPI lista para la tesis"
                )
            
            # ── Exportar tabla LaTeX ───────────────────────────────────────
            with col_r3:
                latex_table = r"""\begin{table}[h]
\centering
\caption{Resultados de optimización TDA-SIMP (\texttimes {}""" + f"{nex_}" + r""" """ + f"\\texttimes {ney_}" + r""" elementos).}
\label{tab:resultados_optimizacion}
\begin{tabular}{lr}
\toprule
\textbf{Métrica} & \textbf{Valor} \\
\midrule"""
                
                latex_rows = [
                    (f"Compliance final $c(\\rho^*)$", f"${c_final:.5f}$"),
                    (f"Reducción vs sólido", f"${reduccion:.2f}\\%$"),
                    (f"Fracción de volumen $f_V$", f"${volfrac}$"),
                    (f"Penalización $p$", f"${penal}$"),
                    (f"Peso topológico $\\alpha$", f"${alpha_val}$"),
                    (f"$\\beta_0$ (componentes conexas)", f"${beta0}$"),
                    (f"$\\beta_1$ (agujeros)", f"${beta1}$"),
                    (f"$\\mu_\\alpha = c + \\alpha \\beta_1$", f"${mu:.5f}$"),
                    (f"Manufacturable", "Sí" if manufacturable else "No"),
                    (f"Iteraciones", f"${n_iter}$"),
                    (f"Tiempo SIMP", f"${t_simp:.2f}$ s"),
                    (f"Tiempo TDA", f"${t_tda:.3f}$ s"),
                ]
                
                for name, val in latex_rows:
                    latex_table += f"\n{name} & {val} \\\\"
                
                latex_table += r"""
\bottomrule
\end{tabular}
\end{table}"""
                
                latex_bytes = latex_table.encode('utf-8')
                
                st.download_button(
                    label="📜 Tabla LaTeX",
                    data=latex_bytes,
                    file_name=f"tabla_resultados_{nex_}x{ney_}.tex",
                    mime="text/plain",
                    use_container_width=True,
                    help="Código LaTeX listo para copiar a tu tesis"
                )
                
                # Vista previa del LaTeX
                with st.expander("🔍 Vista previa del código LaTeX"):
                    st.code(latex_table, language='latex')
            
            # ── Fila 2: CSVs ──────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 📊 Datos numéricos (CSV)")
            col_c1, col_c2, col_c3 = st.columns(3)
            
            # CSV Métricas
            with col_c1:
                df_metrics = pd.DataFrame({
                    "Metrica": [
                        "Compliance_Final", "Beta_0", "Beta_1", "Mu_alpha",
                        "Reduccion_vs_solido_%", "Volumen", "Penalizacion_p",
                        "Alpha", "Iteraciones", "Convergio", "Tiempo_SIMP_s",
                        "Tiempo_TDA_s", "Malla_Nx", "Malla_Ny"
                    ],
                    "Valor": [
                        f"{c_final:.8f}", beta0, beta1, f"{mu:.8f}",
                        f"{reduccion:.4f}", volfrac, penal,
                        alpha_val, n_iter, 1 if converged else 0,
                        f"{t_simp:.4f}", f"{t_tda:.4f}", nex_, ney_
                    ]
                })
                csv_metrics = df_metrics.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Métricas (CSV)",
                    data=csv_metrics,
                    file_name=f"metricas_{nex_}x{ney_}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # CSV Historia completa
            with col_c2:
                df_history = pd.DataFrame({
                    "Iteracion": range(1, len(c_hist) + 1),
                    "Compliance": c_hist
                })
                csv_history = df_history.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Historial (CSV)",
                    data=csv_history,
                    file_name=f"historial_convergencia_{nex_}x{ney_}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # CSV Densidades
            with col_c3:
                rho_flat = rho_final.flatten()
                y_idx, x_idx = np.meshgrid(range(ney_), range(nex_), indexing='ij')
                df_dens = pd.DataFrame({
                    "Elemento_X": x_idx.flatten(),
                    "Elemento_Y": y_idx.flatten(),
                    "Densidad_rho": rho_flat
                })
                csv_dens = df_dens.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Densidades (CSV)",
                    data=csv_dens,
                    file_name=f"densidades_{nex_}x{ney_}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # ── Fila 3: Tabla de métricas expandible ──────────────────────
            st.markdown("---")
            df_display = pd.DataFrame({
                "Métrica": [
                    "Compliance Final c(ρ*)", "β₀ (Componentes conexas)", "β₁ (Agujeros)",
                    "μ_α (Métrica Compuesta)", "Reducción vs Sólido",
                    "Fracción de Volumen f_V", "Penalización p", "Peso α",
                    "Iteraciones", "Convergencia", "Tiempo SIMP", "Tiempo TDA", "Manufacturable"
                ],
                "Valor": [
                    f"{c_final:.5f}", str(beta0), str(beta1),
                    f"{mu:.5f}", f"{reduccion:.2f}%",
                    f"{volfrac}", f"{penal}", f"{alpha_val}",
                    str(n_iter), "Sí" if converged else "No",
                    f"{t_simp:.3f} s", f"{t_tda:.3f} s",
                    "✅ APTA" if manufacturable else "⚠️ REVISAR"
                ]
            })
            
            with st.expander("📋 Ver tabla completa de métricas", expanded=True):
                st.dataframe(df_display, use_container_width=True, hide_index=True)

elif page == "📊 2. TDA vs K-Medias (H.E.1)":
    run_tda_kmedias()

else:
    # ------------------------------------------
    # PÁGINA 3: Optimización Topológica de Vigas
    # ------------------------------------------
    st.header("📈 3. Optimización Topológica")

    from tda.optimization.beam_optimizer import BeamOptimizer
    from matplotlib.patches import Rectangle, Polygon, FancyArrowPatch

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
    else:
        data = st.session_state.beam_opt_data

        st.success("✅ Optimización completada exitosamente")

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
                label="📥 Descargar CSV",
                data=csv_bytes,
                file_name="optimizacion_viga.csv",
                mime="text/csv",
                use_container_width=True
            )

            # ── PDF ──
            with io.BytesIO() as pdf_buf:
                with PdfPages(pdf_buf) as pdf:
                    # Cover page
                    fig_cover, ax_cover = plt.subplots(figsize=(8.27, 11.69))
                    ax_cover.axis('off')
                    ax_cover.text(0.5, 0.85, "Informe de Optimización Topológica",
                                  ha='center', fontsize=22, fontweight='bold', transform=ax_cover.transAxes)
                    ax_cover.text(0.5, 0.78, "Viga de Hormigón Armado — Método SIMP 1D",
                                  ha='center', fontsize=14, transform=ax_cover.transAxes)
                    ax_cover.text(0.5, 0.68, f"L = {data['L']} m    q = {q} kN/m    b×h₀ = {b}×{h0} m",
                                  ha='center', fontsize=12, transform=ax_cover.transAxes)
                    ax_cover.text(0.5, 0.62, f"Iteraciones: {data['iterations']}    Ahorro: {saving:.1f}%    Peso: {weight:.2f} t",
                                  ha='center', fontsize=12, transform=ax_cover.transAxes)
                    ax_cover.text(0.5, 0.55, f"Error final: {data['final_error']:.6f}    Deflexión máx: {np.max(np.abs(Y)) * 1000:.2f} mm",
                                  ha='center', fontsize=12, transform=ax_cover.transAxes)
                    pdf.savefig(fig_cover)
                    plt.close(fig_cover)

                    # Results page
                    fig_pdf, axes_pdf = plt.subplots(3, 1, figsize=(8.27, 8), sharex=True)
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
                    plt.close(fig_pdf)

                    # Metrics summary page
                    fig_met, ax_met = plt.subplots(figsize=(8.27, 5))
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
                                verticalalignment='top', transform=ax_met.transAxes)
                    pdf.savefig(fig_met)
                    plt.close(fig_met)

                pdf_bytes = pdf_buf.getvalue()

            st.download_button(
                label="📥 Descargar Informe PDF",
                data=pdf_bytes,
                file_name="informe_optimizacion_viga.pdf",
                mime="application/pdf",
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
                label="📥 Descargar Tabla LaTeX",
                data=latex_table.encode('utf-8'),
                file_name="tabla_resultados_optimizacion.tex",
                mime="text/plain",
                use_container_width=True
            )

            st.markdown("---")

            if st.button("🔄 Nueva Optimización", use_container_width=True):
                st.session_state.beam_opt_data = None
                st.rerun()
