"""Figuras Plotly para visualización de análisis TDA.

Proporciona funciones que generan figuras Plotly listas para renderizar
en Streamlit o exportar a PDF/HTML.
"""

from typing import Dict
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_stability_chart(results: Dict) -> go.Figure:
    """Crea gráfico de estabilidad con doble eje Y.

    Muestra los números de Betti (β₀, β₁) en el eje izquierdo
    y el accuracy de K-Means en el eje derecho, con líneas verticales
    marcando el rango 15-20% de la hipótesis H.E.1.

    Args:
        results: Dict con "noise_vals", "acc", "betti_s", "betti_t".

    Returns:
        go.Figure lista para mostrar con st.plotly_chart().
    """
    noise_vals = results["noise_vals"]
    sweep_acc = results["acc"]
    sweep_betti_s = results["betti_s"]
    sweep_betti_t = results["betti_t"]
    
    noise_pct = noise_vals * 100
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # β₁ del toro (debe ser 2 siempre)
    b1_t = [b[1] for b in sweep_betti_t]
    fig.add_trace(
        go.Scatter(x=noise_pct, y=b1_t, mode='lines+markers',
                   name='β₁ Toro (TDA)',
                   line=dict(color='#e74c3c', width=2.5),
                   marker=dict(symbol='square', size=8)),
        secondary_y=False
    )
    
    # β₁ de la esfera (debe ser 0 siempre)
    b1_s = [b[1] for b in sweep_betti_s]
    fig.add_trace(
        go.Scatter(x=noise_pct, y=b1_s, mode='lines+markers',
                   name='β₁ Esfera (TDA)',
                   line=dict(color='#3498db', width=2.5),
                   marker=dict(symbol='circle', size=8)),
        secondary_y=False
    )
    
    # β₀ de ambos (debe ser 1 siempre)
    b0_t = [b[0] for b in sweep_betti_t]
    b0_s = [b[0] for b in sweep_betti_s]
    fig.add_trace(
        go.Scatter(x=noise_pct, y=b0_t, mode='lines+markers',
                   name='β₀ Toro (TDA)',
                   line=dict(color='#e67e22', width=1.5, dash='dot'),
                   marker=dict(symbol='diamond', size=6)),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=noise_pct, y=b0_s, mode='lines+markers',
                   name='β₀ Esfera (TDA)',
                   line=dict(color='#2ecc71', width=1.5, dash='dot'),
                   marker=dict(symbol='diamond', size=6)),
        secondary_y=False
    )
    
    # Accuracy K-Means (eje secundario)
    fig.add_trace(
        go.Scatter(x=noise_pct, y=sweep_acc, mode='lines+markers',
                   name='Accuracy K-Medias',
                   line=dict(color='#8e44ad', width=3, dash='dash'),
                   marker=dict(symbol='x', size=10)),
        secondary_y=True
    )
    
    # Líneas verticales del rango H.E.1 (15-20%)
    fig.add_vline(x=15, line=dict(color='gray', width=1, dash='dash'))
    fig.add_vline(x=20, line=dict(color='gray', width=1, dash='dash'))
    fig.add_annotation(
        x=17.5, y=0.15,
        text="Rango H.E.1<br>(15-20%)",
        showarrow=False, font=dict(size=9, color='gray'),
        yref='paper'
    )
    
    # Layout
    fig.update_layout(
        title=dict(
            text="<b>Estabilidad de Invariantes Topológicos vs Accuracy Euclidiano</b>",
            font=dict(size=13)
        ),
        xaxis=dict(title="Nivel de Ruido (%)", dtick=5, range=[-2, 32]),
        yaxis=dict(
            title="Número de Betti",
            dtick=1, range=[-0.5, 3.5],
            showgrid=True, gridcolor='#ecf0f1'
        ),
        yaxis2=dict(
            title="Accuracy K-Medias",
            range=[0, 1.05],
            tickformat='.0%',
            showgrid=False
        ),
        hovermode='x unified',
        legend=dict(
            orientation='h', yanchor='bottom', y=1.12,
            xanchor='center', x=0.5,
            font=dict(size=9)
        ),
        height=450,
        plot_bgcolor='#fafafa',
        margin=dict(t=80)
    )
    
    return fig


def interpret_stability(results: Dict) -> Dict:
    """Interpreta los resultados del barrido de ruido.

    Evalúa si los números de Betti se mantuvieron estables y si
    el accuracy de K-Means se degradó, generando un veredicto sobre H.E.1.

    Args:
        results: Dict del barrido de ruido.

    Returns:
        Dict con "stable": bool, "verdict": str, "details": str.
    """
    sweep_betti_s = results["betti_s"]
    sweep_betti_t = results["betti_t"]
    sweep_acc = results["acc"]
    
    b1_toro_stable = all(b[1] == 2 for b in sweep_betti_t)
    b1_esfera_stable = all(b[1] == 0 for b in sweep_betti_s)
    b0_stable = all(b[0] == 1 for b in sweep_betti_s) and all(b[0] == 1 for b in sweep_betti_t)
    kmeans_fails = any(a < 0.8 for a in sweep_acc)
    
    all_stable = b1_toro_stable and b1_esfera_stable and b0_stable
    
    if all_stable:
        verdict = "H.E.1 VERIFICADA" if kmeans_fails else "H.E.1 VERIFICADA PARCIALMENTE"
        details = (
            f"- β₁ del Toro se mantiene en **2** para todo el rango de ruido 0–30%\n"
            f"- β₁ de la Esfera se mantiene en **0** para todo el rango\n"
            f"- β₀ se mantiene en **1** para ambas formas\n"
            f"- Accuracy de K-Medias {'cae por debajo del 80%' if kmeans_fails else 'se degrada'}"
        )
    else:
        verdict = "H.E.1 VERIFICADA PARCIALMENTE"
        details = (
            f"- β₁ del Toro: {'estable en 2' if b1_toro_stable else 'presenta variaciones'}\n"
            f"- β₁ de la Esfera: {'estable en 0' if b1_esfera_stable else 'presenta variaciones'}\n"
            f"- Accuracy K-Medias: {'se degrada con el ruido' if kmeans_fails else 'se mantiene estable'}"
        )
    
    return {"stable": all_stable, "verdict": verdict, "details": details}


def plot_sweep_persistence_animation(results):
    """Crea una animación Plotly de diagramas de persistencia por nivel de ruido.

    Cada frame muestra los diagramas 0D y 1D para esfera y toro en un nivel de ruido.
    La diagonal y=x (puntos con death=birth) se muestra como referencia.

    Args:
        results: Dict del barrido de ruido (debe incluir "diagrams_s" y "diagrams_t").

    Returns:
        go.Figure con frames animados y slider.
    """
    import numpy as np

    noise_vals = results["noise_vals"]
    diagrams_s = results["diagrams_s"]
    diagrams_t = results["diagrams_t"]

    # Encontrar rango global para ejes
    all_births = []
    all_deaths = []
    for dgms_s, dgms_t in zip(diagrams_s, diagrams_t):
        for dgm in [dgms_s, dgms_t]:
            for dim_dgm in dgm:
                if len(dim_dgm) > 0:
                    finite = dim_dgm[np.isfinite(dim_dgm[:, 1])]
                    if len(finite) > 0:
                        all_births.extend(finite[:, 0].tolist())
                        all_deaths.extend(finite[:, 1].tolist())

    if not all_births:
        all_births = [0, 1]
        all_deaths = [0, 1]

    max_val = max(max(all_births), max(all_deaths)) * 1.1
    min_val = 0

    colors = {
        "sphere_0": "#3498db", "sphere_1": "#2980b9",
        "torus_0": "#e74c3c", "torus_1": "#c0392b",
    }

    noise_pct = noise_vals * 100

    # Frame 0 (primer nivel de ruido) para figura base
    dgms_s0 = diagrams_s[0]
    dgms_t0 = diagrams_t[0]

    def _filter_finite(dgm):
        if len(dgm) == 0:
            return np.empty((0, 2))
        finite = dgm[np.isfinite(dgm[:, 1])]
        return finite

    fig = go.Figure()

    # Diagonal de referencia
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines",
            line=dict(color="gray", width=1, dash="dot"),
            name="y=x (ruido)",
            hoverinfo="skip",
        )
    )

    # Esfera 0D
    d0 = _filter_finite(dgms_s0[0])
    fig.add_trace(
        go.Scatter(
            x=d0[:, 0] if len(d0) > 0 else [],
            y=d0[:, 1] if len(d0) > 0 else [],
            mode="markers",
            marker=dict(size=6, color=colors["sphere_0"], symbol="circle"),
            name="Esfera H₀",
        )
    )

    # Esfera 1D
    d1 = _filter_finite(dgms_s0[1])
    fig.add_trace(
        go.Scatter(
            x=d1[:, 0] if len(d1) > 0 else [],
            y=d1[:, 1] if len(d1) > 0 else [],
            mode="markers",
            marker=dict(size=7, color=colors["sphere_1"], symbol="x"),
            name="Esfera H₁",
        )
    )

    # Toro 0D
    d0_t = _filter_finite(dgms_t0[0])
    fig.add_trace(
        go.Scatter(
            x=d0_t[:, 0] if len(d0_t) > 0 else [],
            y=d0_t[:, 1] if len(d0_t) > 0 else [],
            mode="markers",
            marker=dict(size=6, color=colors["torus_0"], symbol="diamond"),
            name="Toro H₀",
        )
    )

    # Toro 1D
    d1_t = _filter_finite(dgms_t0[1])
    fig.add_trace(
        go.Scatter(
            x=d1_t[:, 0] if len(d1_t) > 0 else [],
            y=d1_t[:, 1] if len(d1_t) > 0 else [],
            mode="markers",
            marker=dict(size=7, color=colors["torus_1"], symbol="x"),
            name="Toro H₁",
        )
    )

    # Crear frames para cada nivel de ruido
    frames = []
    for idx in range(1, len(noise_vals)):
        dgms_s = diagrams_s[idx]
        dgms_t = diagrams_t[idx]

        def _get_data(dgm):
            finite = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else np.empty((0, 2))
            return finite[:, 0], finite[:, 1]

        f_data = []
        for dgm in [dgms_s[0], dgms_s[1], dgms_t[0], dgms_t[1]]:
            bx, by = _get_data(dgm)
            f_data.append(dict(x=bx.tolist() if len(bx) > 0 else [], y=by.tolist() if len(by) > 0 else []))

        frames.append(go.Frame(
            data=[
                # Diagonal (se mantiene igual)
                go.Scatter(x=[min_val, max_val], y=[min_val, max_val]),
                go.Scatter(x=f_data[0]["x"], y=f_data[0]["y"]),
                go.Scatter(x=f_data[1]["x"], y=f_data[1]["y"]),
                go.Scatter(x=f_data[2]["x"], y=f_data[2]["y"]),
                go.Scatter(x=f_data[3]["x"], y=f_data[3]["y"]),
            ],
            name=f"noise_{idx}",
            traces=[0, 1, 2, 3, 4],
        ))

    fig.frames = frames

    # Slider para navegar niveles de ruido
    sliders = [
        dict(
            active=0,
            steps=[
                dict(
                    method="animate",
                    args=[
                        [f"noise_{i}"],
                        dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0)),
                    ],
                    label=f"{noise_pct[i]:.0f}%",
                )
                for i in range(len(noise_vals))
            ],
            transition=dict(duration=0),
            x=0,
            y=0,
            len=1.0,
            currentvalue=dict(
                prefix="Ruido: ",
                suffix="",
                font=dict(size=12, color="gray"),
                xanchor="center",
            ),
        )
    ]

    fig.update_layout(
        title=dict(
            text="<b>Evolución de Diagramas de Persistencia con el Ruido</b>",
            font=dict(size=13),
        ),
        xaxis=dict(title="Birth", range=[min_val, max_val], constrain="domain"),
        yaxis=dict(title="Death", range=[min_val, max_val], scaleanchor="x", scaleratio=1),
        height=500,
        hovermode="closest",
        sliders=sliders,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            [f"noise_{i}" for i in range(len(noise_vals))],
                            dict(
                                frame=dict(duration=400, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                    ),
                ],
                x=0.05,
                y=1.12,
                xanchor="left",
                yanchor="top",
            )
        ],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=9)),
        margin=dict(t=60, b=80),
        plot_bgcolor="#fafafa",
    )

    return fig
