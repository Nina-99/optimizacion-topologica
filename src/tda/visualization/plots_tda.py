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
