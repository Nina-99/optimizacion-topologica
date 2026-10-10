"""
Generador v2 de documentacion Word — Plataforma TDA-SIMP.
Reescribe ambos documentos con las funcionalidades actuales y las
capturas nuevas (src/img/screeshot), en el orden de pantalla:
  plataforma tda simp | H_E_1_* | H_E_2_* | H_G_* | E_*
Reutiliza los helpers academicos de generate_docs.py.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from docx import Document  # noqa: E402
from generate_docs import (  # noqa: E402
    setup_styles,
    add_cover,
    add_figure,
    add_table_academic,
    add_note,
)

AUTOR = "Jorge Larry Copa Cruz"
INSTITUCION = "Maestría en Matemática · UAGRM"
ANIO = "2026"


def _shot(doc, filename, caption):
    add_figure(doc, filename, caption, width=5.8)


def generate_user_manual():
    doc = Document()
    setup_styles(doc)
    add_cover(doc, "Manual de Usuario", "Plataforma TDA-SIMP", AUTOR, INSTITUCION, ANIO)

    # ── 1. Introducción ──
    doc.add_heading("1. Introducción", level=1)
    doc.add_heading("1.1 Objetivo del manual", level=2)
    doc.add_paragraph(
        "Este manual describe el uso de la Plataforma TDA-SIMP: una aplicación "
        "interactiva (Streamlit) que valida hipótesis sobre robustez topológica "
        "(H.E.1), optimización estructural SIMP (H.E.2) y un laboratorio de viga "
        "1D vs 2D. Cada módulo ejecuta simulaciones reales, muestra las Figuras 5.1 "
        "a 5.7 del Capítulo 5 y permite descargar todos los gráficos como PNG en un ZIP."
    )
    doc.add_heading("1.2 Tecnologías utilizadas", level=2)
    add_table_academic(doc, ["Componente", "Tecnología"],
        [["Interfaz", "Streamlit"], ["Cómputo numérico", "NumPy, SciPy, scikit-learn"],
         ["Homología persistente", "Ripser (nubes), GUDHI (mallas)"],
         ["Gráficos", "Plotly (interactivo) + Matplotlib (PDF)"],
         ["Exportación de figuras", "kaleido + Chrome (PNG en ZIP)"],
         ["Gestor de paquetes", "uv"], ["Lenguaje", "Python 3.12"]])
    doc.add_heading("1.3 Estructura del manual", level=2)
    doc.add_paragraph(
        "Capítulo 2: instalación y ejecución. Capítulo 3: pantalla principal. "
        "Capítulos 4 a 6: un capítulo por módulo, en el orden de la pantalla "
        "(H.E.1, H.E.2, Ejemplo Viga). Capítulo 7: exportación y problemas."
    )

    # ── 2. Instalación ──
    doc.add_heading("2. Instalación y Ejecución", level=1)
    doc.add_heading("2.1 Instalación con uv", level=2)
    doc.add_paragraph("1. Instalar uv (https://docs.astral.sh/uv/).")
    doc.add_paragraph("2. uv venv .venv && uv pip install -r requirements.txt")
    doc.add_paragraph("3. uv pip install ripser --no-deps && uv pip install -e .")
    doc.add_heading("2.2 Ejecución", level=2)
    doc.add_paragraph("uv run streamlit run src/tda/app/plataforma_tda_simp.py "
                      "y abrir http://localhost:8501. También existe install_package.sh.")
    doc.add_heading("2.3 Requisitos del sistema", level=2)
    doc.add_paragraph("Linux/macOS/Windows, Python ≥ 3.10 (validado 3.12). Para exportar "
                      "PNG: Chrome vía .venv/bin/plotly_get_chrome (una sola vez).")

    # ── 3. Pantalla principal ──
    doc.add_heading("3. Pantalla Principal — Plataforma TDA-SIMP", level=1)
    doc.add_paragraph(
        "La pantalla principal (plataforma tda simp) presenta los tres módulos "
        "con tarjetas de navegación: H.E.1 (robustez TDA), H.E.2 (optimización SIMP) "
        "y Ejemplo Viga 1D vs 2D."
    )
    _shot(doc, "plataforma tda simp.png", "Figura 1. Pantalla principal — Plataforma TDA-SIMP.")
    doc.add_page_break()

    # ── 4. H.E.1 ──
    doc.add_heading("4. Módulo H.E.1 — Robustez TDA vs Euclidianos", level=1)
    doc.add_paragraph(
        "Valida la H.E.1: los números de Betti son más robustos que los descriptores "
        "euclidianos bajo ruido gaussiano. Flujo: calibración de τ (Fig. 5.1, paso previo), "
        "tasa de acierto y cotas (Figs. 5.2–5.3), clasificación y McNemar (H.E.1b), "
        "cota réplica por réplica con q* (H.E.1c) y Tabla 5.5 del Caso 1."
    )
    _shot(doc, "H_E_1_0.png", "Figura 2. H.E.1 — Encabezado, objetivo y parámetros laterales.")
    doc.add_heading("4.1 Calibración del umbral τ (Figura 5.1)", level=2)
    doc.add_paragraph(
        "El botón «Calibrar τ» computa los 7 diagramas del patrón de oro una sola vez "
        "y evalúa 20 valores de τ: cuenta cuántas nubes reciben la lectura β₁ correcta. "
        "La franja verde marca la ventana de acierto total 7/7; τ = 0.15 (valor usado "
        "en todo el sistema) cae dentro de ella."
    )
    _shot(doc, "H_E_1_1.png", "Figura 3. Figura 5.1 — Curva de calibración de τ con ventana 7/7.")
    doc.add_heading("4.2 Tasa de acierto y cotas (Figuras 5.2–5.3)", level=2)
    doc.add_paragraph(
        "La Figura 5.2(a) muestra la tasa de acierto de βₖ por nivel de ruido: la lectura "
        "escalar deja de ser fiable fuera de su régimen de validez (H.E.1c). La Figura 5.3 "
        "contrasta la distancia bottleneck normalizada contra la cota 2q por nube y nivel; "
        "la Figura 5.2(b) verifica la cota dB = 2dH réplica por réplica con su región prohibida."
    )
    _shot(doc, "H_E_1_2_1.png", "Figura 4. Figura 5.2(a) — Tasa de acierto de βₖ por ruido.")
    _shot(doc, "H_E_1_2_2.png", "Figura 5. Datos para la Figura 5.2(a) y hallazgo del régimen de validez.")
    _shot(doc, "H_E_1_3_1.png", "Figura 6. Figura 5.3 — Bottleneck normalizada vs cota 2q.")
    _shot(doc, "H_E_1_3_2.png", "Figura 7. Veredicto H.E.1a y datos de la Figura 5.3.")
    doc.add_heading("4.3 Clasificación, McNemar y Tabla 5.5", level=2)
    doc.add_paragraph(
        "La sección H.E.1b compara exactitud TDA vs euclidiana y aplica el test de McNemar: "
        "la tabla cruza aciertos de ambos clasificadores y solo importan las celdas "
        "discordantes (n01 = TDA acierta donde Eucl falla, n10 = al revés); se reportan "
        "χ² y valor p (p < 0.001 si es ínfimo). La Tabla 5.5 consolida indicador, criterio, "
        "valor esperado y medido del Caso 1, incluido q* estimado."
    )
    _shot(doc, "H_E_1_4_1.png", "Figura 8. H.E.1b — Clasificación 3D y exactitud complementaria.")
    _shot(doc, "H_E_1_4_2.png", "Figura 9. Test de McNemar y Figura 5.2(b) con región prohibida.")
    _shot(doc, "H_E_1_4_3.png", "Figura 10. Veredicto H.E.1c, q* estimado y Tabla 5.5 del Caso 1.")
    doc.add_page_break()

    # ── 5. H.E.2 ──
    doc.add_heading("5. Módulo H.E.2 — Optimización SIMP", level=1)
    doc.add_paragraph(
        "Optimización topológica 2D con SIMP y evaluación topológica post hoc. Valida "
        "H.E.2a (reducción ≥ 40 % vs diseño uniforme) y H.E.2b (β₁ invariante). "
        "Incluye Figuras 5.4 a 5.7 con epígrafes del documento, Tabla 5.6 dinámica y "
        "doble cómputo de Betti: característica de Euler del complejo cúbico "
        "(χ = V−E+F, β₁ = β₀−χ) vs homología cúbica persistente (GUDHI)."
    )
    _shot(doc, "H_E_2_0.png", "Figura 11. H.E.2 — Encabezado, objetivo y parámetros SIMP.")
    doc.add_heading("5.1 Diseño convergido (Figuras 5.4–5.7)", level=2)
    doc.add_paragraph(
        "La Figura 5.4(b) muestra el historial de compliance en escala logarítmica con "
        "los referentes marcados; la 5.4(a) compara el campo continuo ρ* con el sólido "
        "binarizado Ω donde se cuentan β₀ y β₁. La Figura 5.7 barre el umbral u con β₀, "
        "β₁ y la fracción de volumen (línea gris)."
    )
    _shot(doc, "H_E_2_1_1.png", "Figura 12. H.E.2a cumplida y Tabla 5.6 — Compliance vs referentes.")
    _shot(doc, "H_E_2_1_2.png", "Figura 13. Figura 5.4(b) — Historial de compliance (escala log).")
    _shot(doc, "H_E_2_1_3.png", "Figura 14. Figura 5.7 — Betti vs umbral u con fracción de volumen.")
    _shot(doc, "H_E_2_2_1.png", "Figura 15. H.E.2b — Métricas, concordancia χ-GUDHI y persistencia H₁.")
    _shot(doc, "H_E_2_2_2.png", "Figura 16. Figura 5.4(a) — Densidad continua vs diseño binarizado.")
    doc.add_heading("5.2 Barrido multi-configuración (H.E.2b)", level=2)
    doc.add_paragraph(
        "El barrido corre 12–13 configuraciones (malla, p, rmin). Las Figuras 5.5 y 5.6 "
        "verifican invariancia: compliance y grisura vs p (con β₁ anotado por punto), "
        "compliance y β₁ vs número de elementos. El veredicto final combina H.E.2a y H.E.2b."
    )
    _shot(doc, "H_E_2_3.png", "Figura 17. Métricas, exportación y barrido multi-configuración.")
    _shot(doc, "H_E_2_4_1.png", "Figura 18. Datos y Figura 5.5 — Barrido del factor de penalización.")
    _shot(doc, "H_E_2_4_2.png", "Figura 19. Datos y Figura 5.6 — Barrido de malla.")
    _shot(doc, "H_E_2_4_3.png", "Figura 20. Barrido de rmin (5 valores).")
    _shot(doc, "H_E_2_4_4.png", "Figura 21. Veredicto H.E.2 y plan de acción del módulo.")
    doc.add_page_break()

    # ── 6. Viga ──
    doc.add_heading("6. Módulo Ejemplo Viga 1D vs 2D", level=1)
    doc.add_paragraph(
        "Laboratorio de viga en voladizo con cuatro modos: perfil de altura 1D (analítico), "
        "SIMP 2D con animación y TDA post-hoc (GUDHI), barrido de penalización p = 2, 3, 4 "
        "y comparación 1D vs 2D con resumen, exportación STL y ZIP de figuras."
    )
    doc.add_heading("6.1 Modo 1D — Perfil de altura", level=2)
    _shot(doc, "E_1_1.png", "Figura 31. Modo 1D — Configuración y perfil de altura.")
    _shot(doc, "E_1_2.png", "Figura 32. Perfil de altura óptimo y momento flector.")
    _shot(doc, "E_1_3.png", "Figura 33. Tensión máxima y completitud de la optimización.")
    doc.add_heading("6.2 Modo 2D — SIMP topológico", level=2)
    _shot(doc, "E_2_0.png", "Figura 34. Modo 2D — Configuración SIMP 60×30.")
    _shot(doc, "E_2_1.png", "Figura 35. Animación de la optimización (iteración 45).")
    _shot(doc, "E_2_2.png", "Figura 36. Diseño convergido, métricas y convergencia.")
    _shot(doc, "E_2_3.png", "Figura 37. Diagramas de persistencia (GUDHI) del diseño.")
    doc.add_heading("6.3 Barrido y comparación", level=2)
    _shot(doc, "E_3_0.png", "Figura 38. Barrido de penalización p = 2, 3, 4.")
    _shot(doc, "E_3_1.png", "Figura 39. Resultados comparativos del barrido.")
    _shot(doc, "E_4_0.png", "Figura 40. Comparación lado a lado 1D vs 2D.")
    _shot(doc, "E_4_1.png", "Figura 41. Resumen comparativo y exportación STL.")
    _shot(doc, "E_4_2.png", "Figura 42. Exportación de figuras en ZIP.")
    doc.add_page_break()

    # ── 8. Exportación y problemas ──
    doc.add_heading("7. Exportación y Solución de Problemas", level=1)
    doc.add_paragraph(
        "Cada módulo ofrece «Descargar TODO (ZIP)»: CSV de datos, PDF de reporte y PNG "
        "de cada gráfico (Plotly vía kaleido + Chrome, Matplotlib directo). Si un PNG "
        "falla, el ZIP incluye un .txt de error en su lugar para no omitirlo en silencio."
    )
    doc.add_heading("7.1 Problemas frecuentes", level=2)
    add_table_academic(doc, ["Síntoma", "Causa", "Solución"],
        [["PNG ausente / .txt de error", "Falta Chrome para kaleido", "Ejecutar .venv/bin/plotly_get_chrome"],
         ["GUDHI ausente", "Paquete opcional no instalado", "uv pip install gudhi (o veredicto sin verificar)"],
         ["App lenta en barridos", "Ripser/SIMP intensivos", "Reducir puntos, malla o iteraciones"]])
    doc.add_heading("8. Glosario", level=1)
    add_table_academic(doc, ["Término", "Significado"],
        [["β₀ / β₁", "Componentes conexas / agujeros (números de Betti)"],
         ["τ", "Umbral relativo de persistencia"],
         ["q*", "Nivel crítico de ruido"], ["χ", "Característica de Euler (V−E+F)"]])

    out = os.path.join(BASE_DIR, "Manual_de_Usuario.docx")
    doc.save(out)
    print("OK Manual:", out)


def generate_technical_doc():
    doc = Document()
    setup_styles(doc)
    add_cover(doc, "Documentación Técnica", "Plataforma TDA-SIMP", AUTOR, INSTITUCION, ANIO)

    doc.add_heading("1. Arquitectura del Sistema", level=1)
    doc.add_paragraph(
        "Paquete instalable tda (src/tda): analysis (métricas, estabilidad, anomalías), "
        "core (FEM Q4, topología, doble cómputo Betti), optimization "
        "(SIMP 2D, viga 1D), processing (muestreo, ruido, normalización), "
        "simulation (pipeline y experimentos headless), visualization (Plotly/Matplotlib), "
        "app (plataforma Streamlit: landing + 3 páginas). Gestión con uv (.venv), "
        "tests con pytest (232 pruebas)."
    )
    _shot(doc, "plataforma tda simp.png", "Figura 1. Plataforma TDA-SIMP (landing).")

    doc.add_heading("2. Núcleo Matemático", level=1)
    doc.add_heading("2.1 Homología persistente", level=2)
    doc.add_paragraph(
        "Ripser (complejos Vietoris-Rips) para nubes; GUDHI (complejos cúbicos) para "
        "mallas. Lectura βk = (β₀, β₁) con umbral relativo τ·diam(X); τ = 0.15 "
        "calibrado contra el patrón de oro de 7 nubes (Figura 5.1)."
    )
    _shot(doc, "H_E_1_1.png", "Figura 2. Calibración de τ contra el patrón de oro de 7 nubes.")
    doc.add_heading("2.2 Doble cómputo de Betti (diseños SIMP)", level=2)
    doc.add_paragraph(
        "Método A — característica de Euler del complejo cúbico: β₀ por etiquetado "
        "4-conexo, χ = V−E+F por conteo celular, β₁ = β₀−χ (H₂ = 0 en R²). "
        "Método B — homología cúbica persistente (GUDHI, subnivel 0.5). La bandera "
        "de concordancia se registra en cada tabla de datos."
    )
    _shot(doc, "H_E_2_2_1.png", "Figura 3. Concordancia χ-GUDHI y persistencia H₁.")
    doc.add_heading("2.3 Configuración canónica del caso SIMP", level=2)
    doc.add_paragraph(
        "Dominio con elementos unitarios (Lx = nex, Ly = ney), material y carga "
        "normalizados (E₀ = 1, F = 1), malla 60×30, fV = 0.5, p = 3, rmin = 2.4. "
        "Esta configuración reproduce los valores del Capítulo 5: c_uniforme = 316.34, "
        "c_sólido = 39.54, c_SIMP ≈ 75.5 y β₁ = 3."
    )
    _shot(doc, "H_E_2_1_1.png", "Figura 4. Compliance vs referentes (Tabla 5.6).")

    doc.add_heading("3. Solver FEM y SIMP", level=1)
    doc.add_paragraph(
        "Motor Q4 (fem.py): ensamble, solver, sensibilidades y criterio de optimalidad "
        "(OC) con filtro de DENSIDAD rmin (ρ̃ = H·ρ/Hs). Optimización p = 3, fV = 0.5, "
        "material/carga normalizados (E₀ = 1, F = 1), mallas rectangulares "
        "(40×20, 80×40, 120×60) con rmin escalado (1.6/3.2/4.8)."
    )
    _shot(doc, "H_E_2_2_2.png", "Figura 5. Campo continuo vs sólido binarizado.")
    _shot(doc, "H_E_2_4_1.png", "Figura 6. Barrido p con grisura y β₁ anotado.")

    doc.add_heading("4. Validaciones Estadísticas", level=1)
    doc.add_paragraph(
        "McNemar (χ² con n01/n10 discordantes), cota bottleneck d_B ≤ 2·d_H "
        "(Teorema de Estabilidad; k = 0,1), q* estimado, Tabla 5.5 dinámica y Tabla 5.9. "
        "La exactitud TDA vs euclidiana y el valor p se recalculan en cada ejecución."
    )
    _shot(doc, "H_E_1_4_2.png", "Figura 7. McNemar y cota réplica por réplica.")

    doc.add_heading("5. Exportación y Ejecutable", level=1)
    doc.add_paragraph(
        "download_utils.py: fig_png_bytes (kaleido/Matplotlib) y write_figs_zip "
        "(22 gráficos registrados, .txt de error si alguno falla). Empaquetado .exe "
        "con PyInstaller + streamlit-desktop-app (ver docs de empaquetado)."
    )
    _shot(doc, "E_4_2.png", "Figura 9. Exportación ZIP de figuras.")

    doc.add_heading("6. Hallazgos Principales", level=1)
    doc.add_paragraph(
        "La lectura escalar βₖ deja de ser fiable fuera de su régimen (tasa ≈ 0 en "
        "q ≥ 0.15); rigen H.E.1a (cota 2q), H.E.1b (McNemar) y H.E.1c (q* delimitado). "
        "H.E.2a: reducción ≈ 76.4 % vs uniforme (≥ 40 % exigido). H.E.2b: β₁ = 3 "
        "invariante frente a malla, penalización p ∈ {2,3,4} y rmin ∈ [1.5, 4.0]."
    )
    add_note(doc, "Los valores entre paréntesis corresponden a corridas de referencia "
             "(malla 60×30, n = 400, semilla 42); cada ejecución recalcula todo.")

    out = os.path.join(BASE_DIR, "Documentacion_Tecnica.docx")
    doc.save(out)
    print("OK Técnica:", out)


if __name__ == "__main__":
    generate_user_manual()
    generate_technical_doc()
