"""
Generador de documentacion Word (.docx) para la Plataforma TDA-SIMP.
Genera dos archivos:
  1. Manual_de_Usuario.docx
  2. Documentacion_Tecnica.docx
"""

import os
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
IMG_DIR = os.path.join(PROJECT_DIR, "src", "img", "screeshot")
OUTPUT_DIR = BASE_DIR

# ── Color constants ──
ORANGE = RGBColor(0xE8, 0x6C, 0x00)
DARK = RGBColor(0x1A, 0x1A, 0x2E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GRAY = RGBColor(0x66, 0x66, 0x66)
GREEN = RGBColor(0x2E, 0x7D, 0x32)


def setup_styles(doc):
    """Configura estilos base del documento."""
    style = doc.styles["Normal"]
    font = style.font
    font.name = "Calibri"
    font.size = Pt(11)
    font.color.rgb = DARK

    for level in range(1, 4):
        heading = doc.styles[f"Heading {level}"]
        heading.font.color.rgb = ORANGE
        heading.font.bold = True

    # Margenes
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)


def add_cover(doc, title, subtitle, author, institution, year):
    """Agrega portada al documento."""
    for _ in range(6):
        doc.add_paragraph()

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(title)
    run.bold = True
    run.font.size = Pt(28)
    run.font.color.rgb = ORANGE

    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run2 = p2.add_run(subtitle)
    run2.font.size = Pt(14)
    run2.font.color.rgb = GRAY

    doc.add_paragraph()

    for line in [author, institution, year]:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(line)
        run.font.size = Pt(12)
        run.font.color.rgb = DARK

    doc.add_page_break()


def add_screenshot(doc, filename, caption, width=5.5):
    """Agrega una captura de pantalla con caption."""
    img_path = os.path.join(IMG_DIR, filename)
    if os.path.exists(img_path):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(img_path, width=Inches(width))

        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run_cap = cap.add_run(caption)
        run_cap.font.size = Pt(9)
        run_cap.font.italic = True
        run_cap.font.color.rgb = GRAY
    else:
        p = doc.add_paragraph()
        run = p.add_run(f"[Imagen no encontrada: {filename}]")
        run.font.color.rgb = RGBColor(0xCC, 0x00, 0x00)
        run.font.italic = True


def add_table(doc, headers, rows, col_widths=None):
    """Agrega una tabla formateada."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = h
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True
                run.font.size = Pt(10)

    # Rows
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.rows[r_idx + 1].cells[c_idx]
            cell.text = str(val)
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.size = Pt(10)

    doc.add_paragraph()  # spacer


def add_info_box(doc, text, style="info"):
    """Agrega un cuadro de informacion destacado."""
    symbols = {"info": "i", "warning": "!", "tip": "*"}
    colors = {"info": RGBColor(0x1A, 0x56, 0x76), "warning": RGBColor(0xCC, 0x66, 0x00), "tip": GREEN}

    p = doc.add_paragraph()
    run = p.add_run(f"  [{symbols.get(style, 'i')}]  ")
    run.bold = True
    run.font.color.rgb = colors.get(style, GRAY)
    run = p.add_run(text)
    run.font.size = Pt(10)


# ════════════════════════════════════════════════════════════════
# DOCUMENTO 1: MANUAL DE USUARIO
# ════════════════════════════════════════════════════════════════
def generate_user_manual():
    doc = Document()
    setup_styles(doc)

    add_cover(
        doc,
        title="Manual de Usuario",
        subtitle="Plataforma TDA-SIMP\nOptimizacion Topologica y Homologia Persistente",
        author="Jorge Larry Copa Cruz",
        institution="Maestria en Matematica — UAGRM — 2026",
        year="",
    )

    # ── 1. Introduccion ──
    doc.add_heading("1. Introduccion", level=1)
    doc.add_paragraph(
        "La Plataforma TDA-SIMP es una herramienta interactiva de analisis topologico y "
        "optimizacion de estructuras mediante el metodo SIMP (Solid Isotropic Material "
        "with Penalization). Combina Teoria de Datos (TDA) con optimizacion topologica "
        "para validar hipotesis sobre robustez, eficiencia y comparacion de disenos "
        "estructurales."
    )
    doc.add_paragraph(
        "Esta plataforma fue desarrollada como parte de la tesis de Maestria en Matematica "
        "de Jorge Larry Copa Cruz en la Universidad Autonoma Gabriel Rene Moreno (UAGRM), "
        "Bolivia, 2026."
    )

    doc.add_heading("1.1 Tecnologias", level=2)
    add_table(doc,
        ["Componente", "Tecnologia"],
        [
            ["Framework", "Streamlit (Python)"],
            ["Analisis topologico", "Ripser, GUDHI, scikit-tda"],
            ["Optimizacion", "SIMP con MMA (Method of Moving Asymptotes)"],
            ["FEM", "Elementos finitos 2D (Plane Stress)"],
            ["Visualizacion", "Plotly, Matplotlib"],
            ["Empaquetado", "PyInstaller (.exe nativo Windows)"],
        ],
    )

    # ── 2. Instalacion ──
    doc.add_heading("2. Instalacion y Ejecucion", level=1)

    doc.add_heading("2.1 Ejecutar en navegador (desarrollo)", level=2)
    doc.add_paragraph("1. Clonar el repositorio.")
    doc.add_paragraph("2. Instalar dependencias: pip install -r requirements.txt")
    doc.add_paragraph("3. Instalar ripser: pip install ripser --no-deps")
    doc.add_paragraph("4. Ejecutar: streamlit run src/tda/app/plataforma_tda_simp.py")
    doc.add_paragraph("5. Abrir http://localhost:8501 en el navegador.")

    doc.add_heading("2.2 Ejecutar como .exe (Windows)", level=2)
    doc.add_paragraph(
        "Ejecutar build_exe.bat desde la raiz del proyecto. El script:"
    )
    doc.add_paragraph("  - Crea un entorno virtual temporal con uv")
    doc.add_paragraph("  - Instala todas las dependencias")
    doc.add_paragraph("  - Genera EstructuraTopologica.exe con PyInstaller")
    doc.add_paragraph("  - Requiere: Windows 10+, Microsoft Edge WebView2 Runtime")
    add_info_box(doc, "El .exe es un archivo unico (~200-300 MB) que no requiere Python instalado.", "tip")

    doc.add_heading("2.3 Requisitos del sistema", level=2)
    add_table(doc,
        ["Requisito", "Minimo", "Recomendado"],
        [
            ["Sistema operativo", "Windows 10 / Linux / macOS", "Windows 11"],
            ["RAM", "4 GB", "8 GB+"],
            ["Disco", "500 MB (libre)", "1 GB+"],
            ["Python (solo dev)", "3.10+", "3.12+"],
            ["WebView2 (solo .exe)", "Instalado", "Ultima version"],
        ],
    )

    # ── 3. Interfaz ──
    doc.add_heading("3. Estructura de la Interfaz", level=1)
    doc.add_paragraph(
        "La plataforma se organiza en una pagina principal (landing) con 4 modulos "
        "accesibles desde la barra lateral izquierda."
    )
    add_screenshot(doc, "plataforma tda simp.png", "Figura 1. Pagina principal — Plataforma TDA-SIMP", width=6.0)

    add_table(doc,
        ["Pagina", "Nombre", "Descripcion"],
        [
            ["1", "H.E.1 Robustez TDA vs Euclidianos", "Validacion de invariantes topologicos bajo ruido gaussiano"],
            ["2", "H.E.2 Optimizacion SIMP Metrica Compuesta", "Implementacion de SIMP con metrica compuesta mu_alpha"],
            ["3", "H.G. Comparacion Integrada TDA-SIMP", "Sintesis integral: validacion de la Hipotesis General"],
            ["4", "Ejemplo Viga 1D vs 2D", "Laboratorio interactivo: comparacion 1D analitica vs SIMP 2D"],
        ],
    )

    # ── 4. Modulo H.E.1 ──
    doc.add_heading("4. Modulo H.E.1 — Robustez TDA vs Euclidianos", level=1)
    doc.add_paragraph(
        "Este modulo valida la hipotesis H.E.1: los descriptores topologicos (numeros de Betti) "
        "son mas robustos que los descriptores euclidianos bajo ruido gaussiano."
    )
    add_screenshot(doc, "H_ E_1_1.png", "Figura 2. Modulo H.E.1 — Parametros y objetivos de validacion")

    doc.add_heading("4.1 Parametros configurables", level=2)
    add_table(doc,
        ["Parametro", "Rango", "Descripcion"],
        [
            ["Puntos por forma", "100 — 1000", "Numero de puntos en cada nube (esfera, toro)"],
            ["Ruido Gaussiano (%)", "0 — 0.30", "Intensidad del ruido aditivo"],
            ["Clusteres (K-medias)", "2 — 10", "Numero de clusteres para clasificacion"],
            ["Repeticiones (n_rep)", "5 — 50", "Numero de replicas por nivel de ruido"],
        ],
    )

    doc.add_heading("4.2 Flujo de ejecucion", level=2)
    doc.add_paragraph("1. Configurar los parametros en la barra lateral.")
    doc.add_paragraph("2. Presionar 'Ejecutar H.E.1 Completo'.")
    doc.add_paragraph("3. Revisar los resultados en 4 secciones:")
    doc.add_paragraph("     a) Evidencia del Hallazgo 1: estabilidad de Bk bajo ruido")
    doc.add_paragraph("     b) Validacion H.E.1a': cota bottleneck (exitoso)")
    doc.add_paragraph("     c) Validacion H.E.1b: superioridad TDA vs euclidianos")
    doc.add_paragraph("     d) Validacion H.E.1c: cota de estabilidad dB <= 2*dH")

    doc.add_heading("4.3 Resultados esperados", level=2)
    add_screenshot(doc, "H_E_1_3.png", "Figura 3. Resultados H.E.1 — Cuadro 4 y validacion H.E.1a'")

    doc.add_paragraph(
        "El Cuadro 4 muestra la tasa de acierto de Bk por nivel de ruido. "
        "La hipotesis original H.E.1 (Bk invariante bajo ruido) se FALSIFICA: "
        "la tasa de acierto cae a 0 para ruido > 0.03. Sin embargo, la version "
        "reformulada H.E.1a' (cota bottleneck) se VALIDA con exito."
    )
    add_info_box(doc, "Hallazgo clave: la estabilidad topologica NO protege Bk directamente, "
                  "pero la cota bottleneck si garantiza que disenos similares tienen "
                  "diagramas de persistencia similares.", "tip")

    # ── 5. Modulo H.E.2 ──
    doc.add_heading("5. Modulo H.E.2 — Optimizacion SIMP Metrica Compuesta", level=1)
    doc.add_paragraph(
        "Este modulo implementa la optimizacion topologica SIMP y valida la hipotesis "
        "H.E.2a' (reduccion >= 40% vs diseno uniforme) y H.E.2b' (invariancia de B1)."
    )
    add_screenshot(doc, "H_E_2_0.png", "Figura 4. Modulo H.E.2 — Optimizacion SIMP en ejecucion")

    doc.add_heading("5.1 Parametros configurables", level=2)
    add_table(doc,
        ["Parametro", "Opciones", "Descripcion"],
        [
            ["Dominio", "Rectangular (2:1) / Cuadrado", "Geometria del dominio FEM"],
            ["Resolucion de Malla", "40x20 a 120x60", "Numero de elementos finitos"],
            ["f_V", "0.5 (fijo)", "Fraccion de volumen objetivo"],
            ["Factor Penalizacion (p)", "1 — 4", "Exponente de penalizacion SIMP"],
            ["Radio Filtro (r_min)", "1.5 — 4.0", "Radio de filtrado de densidad"],
        ],
    )

    doc.add_heading("5.2 Flujo de ejecucion", level=2)
    doc.add_paragraph("1. Seleccionar dominio y resolucion de malla.")
    doc.add_paragraph("2. Ajustar factor de penalizacion p y radio de filtro r_min.")
    doc.add_paragraph("3. Presionar 'Iniciar Optimizacion SIMP'.")
    doc.add_paragraph("4. Observar la evolucion del diseno en tiempo real.")
    doc.add_paragraph("5. Revisar metricas: reduccion, B1, mu_alpha, convergencia.")

    doc.add_heading("5.3 Resultados esperados", level=2)
    add_screenshot(doc, "H_E_2_1.png", "Figura 5. H.E.2 — Evolucion del diseno y metricas de control")

    doc.add_paragraph(
        "El Cuadro 1 muestra que la reduccion de compliance supera el 40% en todas las "
        "configuraciones, validando H.E.2a'. El Cuadro 7 (barrido de p) y Cuadro 8 "
        "(barrido de malla) verifican la invariancia de B1, validando H.E.2b'."
    )

    # ── 6. Modulo H.G. ──
    doc.add_heading("6. Modulo H.G. — Comparacion Integrada TDA-SIMP", level=1)
    doc.add_paragraph(
        "Este modulo ejecuta la sintesis integral de la Hipotesis General: la metrica "
        "compuesta mu_alpha = c + alpha * B1 resuelve el orden parcial entre "
        "configuraciones SIMP."
    )
    add_screenshot(doc, "H_E_3_1.png", "Figura 6. Modulo H.G. — Comparacion Integrada")

    doc.add_heading("6.1 Parametros configurables", level=2)
    add_table(doc,
        ["Parametro", "Rango", "Descripcion"],
        [
            ["Elementos en x/y", "30 — 120", "Resolucion de la malla FEM"],
            ["Fraccion de volumen (fV)", "0.30 — 0.70", "Fraccion de volumen objetivo"],
            ["Radio filtro (r_min)", "1.5 — 4.0", "Radio de filtrado"],
            ["Max. iteraciones SIMP", "50 — 300", "Limite de iteraciones del optimizador"],
            ["Modulo Young E0 (MPa)", "100000 — 400000", "Modulo de elasticidad del material"],
            ["Carga F (N)", "100 — 10000", "Fuerza aplicada"],
        ],
    )

    doc.add_heading("6.2 Flujo de ejecucion", level=2)
    doc.add_paragraph("1. Configurar parametros FEM y fisicos en la barra lateral.")
    doc.add_paragraph("2. Presionar 'Ejecutar Comparacion Integrada'.")
    doc.add_paragraph("3. Revisar: distribucion de material, metricas, ranking Pareto.")

    doc.add_heading("6.3 Resultados esperados", level=2)
    add_screenshot(doc, "H_E_3_2.png", "Figura 7. H.G. — Distribucion de material por configuracion")

    doc.add_paragraph(
        "El modulo compara configuraciones con p=2, p=3 y p=4, mostrando que p=3 "
        "es Pareto-optimo: minimiza compliance manteniendo B1=2 (topologia robusta). "
        "La metrica mu_alpha identifica correctamente la mejor configuracion."
    )

    # ── 7. Modulo Ejemplo Viga ──
    doc.add_heading("7. Modulo Ejemplo Viga 1D vs 2D", level=1)
    doc.add_paragraph(
        "Este modulo es un laboratorio interactivo que compara la optimizacion "
        "analitica 1D (perfil de altura) con la optimizacion topologica SIMP 2D."
    )
    add_screenshot(doc, "H_E_4_0.png", "Figura 8. Modulo Ejemplo Viga — Modo 1D")

    doc.add_heading("7.1 Modos de experimentacion", level=2)
    add_table(doc,
        ["Modo", "Descripcion"],
        [
            ["1D — Perfil de Altura", "Optimizacion analitica: ajusta h(x) para minimizar volumen"],
            ["2D — SIMP Topologico", "Optimizacion topologica 2D con densidades de material"],
            ["Barrido p (2,3,4)", "Compara configuraciones con diferentes factores de penalizacion"],
            ["Comparacion 1D vs 2D", "Lado a lado: perfil analitico vs diseno topologico"],
        ],
    )

    doc.add_heading("7.2 Casos de carga", level=2)
    add_table(doc,
        ["Caso", "Tipo", "Descripcion"],
        [
            ["Voladizo-Puntual", "Cantilever", "Viga en voladizo con carga puntual en el extremo"],
            ["Articulado-Central", "Simply Supported", "Viga articulada con carga central"],
            ["Voladizo-Distribuida", "Cantilever", "Viga en voladizo con carga distribuida"],
        ],
    )

    doc.add_heading("7.3 Metricas disponibles", level=2)
    doc.add_paragraph("Cada modo muestra metricas en tiempo real:")
    doc.add_paragraph("  - Compliance (rigidez del diseno)")
    doc.add_paragraph("  - mu_alpha (metrica compuesta)")
    doc.add_paragraph("  - B0, B1 (numeros de Betti)")
    doc.add_paragraph("  - Iteraciones y convergencia")
    doc.add_paragraph("  - Factor de seguridad (estimacion post-hoc)")
    doc.add_paragraph("  - Exportacion STL (modo Comparacion)")

    # ── 8. Solucion de problemas ──
    doc.add_heading("8. Solucion de Problemas", level=1)

    add_table(doc,
        ["Problema", "Causa", "Solucion"],
        [
            ["La app no inicia", "Puerto 8501 ocupado", "Cerrar otros procesos de Streamlit o usar --server.port 8502"],
            ["Graficas no aparecen", "Dependencia faltante", "Verificar: pip install plotly matplotlib"],
            ["Error en TDA", "Ripser no instalado", "Ejecutar: pip install ripser --no-deps"],
            [".exe no encuentra logos", "PyInstaller no incluyo img/", "Re-ejecutar build_exe.bat (corregido)"],
            ["Columnas desbordadas en mobile", "CSS responsive limitado", "Usar en pantalla >= 768px o desktop"],
            ["Numeros no renderizan LaTeX", "Escape incorrecto", "Verificar que formulas usan r-strings en codigo"],
        ],
    )

    # ── 9. Glosario ──
    doc.add_heading("9. Glosario", level=1)
    add_table(doc,
        ["Termino", "Definicion"],
        [
            ["TDA", "Topological Data Analysis — analisis de forma via homologia persistente"],
            ["SIMP", "Solid Isotropic Material with Penalization — metodo de optimizacion topologica"],
            ["Bk (Betti number)", "Numero de Betti k: componentes conexas (k=0), agujeros (k=1), cavidades (k=2)"],
            ["dB (bottleneck)", "Distancia bottleneck entre diagramas de persistencia"],
            ["dH (Hausdorff)", "Distancia Hausdorff entre nubes de puntos"],
            ["mu_alpha", "Metrica compuesta: mu = c + alpha * B1"],
            ["FEM", "Finite Element Method — metodo de elementos finitos"],
            ["Pareto-optimo", "Diseno que minimiza un objetivo sin empeorar el otro"],
            ["Compliance", "Trabajo de la fuerza externa — inverso de la rigidez"],
            ["fV", "Fraccion de volumen de material solido"],
        ],
    )

    # ── Guardar ──
    output_path = os.path.join(OUTPUT_DIR, "Manual_de_Usuario.docx")
    doc.save(output_path)
    print(f"[OK] Manual de Usuario generado: {output_path}")
    return output_path


# ════════════════════════════════════════════════════════════════
# DOCUMENTO 2: DOCUMENTACION TECNICA
# ════════════════════════════════════════════════════════════════
def generate_tech_doc():
    doc = Document()
    setup_styles(doc)

    add_cover(
        doc,
        title="Documentacion Tecnica",
        subtitle="Plataforma TDA-SIMP\nArquitectura, Algoritmos y Validacion",
        author="Jorge Larry Copa Cruz",
        institution="Maestria en Matematica — UAGRM — 2026",
        year="",
    )

    # ── 1. Arquitectura ──
    doc.add_heading("1. Arquitectura del Sistema", level=1)
    doc.add_paragraph(
        "La plataforma sigue una arquitectura modular con separacion clara entre "
        "logica de dominio, optimizacion, analisis topologico e interfaz de usuario."
    )

    doc.add_heading("1.1 Estructura de directorios", level=2)
    doc.add_paragraph(
        "src/tda/\n"
        "  app/                    # Interfaz Streamlit\n"
        "    plataforma_tda_simp.py  # Landing page\n"
        "    pages/                  # 4 modulos de analisis\n"
        "    theme.py                # Estilos, fonts, responsive\n"
        "    download_utils.py       # Exportacion de archivos\n"
        "  core/                   # Logica de dominio\n"
        "    betti.py / betti2d.py   # Calculo de numeros de Betti\n"
        "    topology.py             # Wasserstein, bottleneck (numpy puro)\n"
        "    metric.py               # Metricas topologicas\n"
        "    fem.py                  # Solver FEM 2D\n"
        "  optimization/           # Optimizacion\n"
        "    metric_simp.py          # MetricaTDA_SIMP\n"
        "    beam_optimizer.py       # Optimizador 1D (viga)\n"
        "  analysis/               # Analisis de robustez\n"
        "    stability.py            # Sweep de ruido, Hausdorff\n"
        "  simulation/             # Pipeline de simulacion\n"
        "    pipeline.py             # Pipeline TDA completo"
    )

    doc.add_heading("1.2 Dependencias principales", level=2)
    add_table(doc,
        ["Paquete", "Version", "Uso"],
        [
            ["streamlit", ">=1.38", "Framework de interfaz web"],
            ["numpy", ">=2.1", "Calculo numerico"],
            ["scipy", ">=1.14", "Optimizacion (MMA), algebra dispersa"],
            ["scikit-learn", ">=1.5", "Clustering (K-medias), PCA"],
            ["matplotlib", ">=3.9", "Graficas estaticas"],
            ["plotly", ">=5.24", "Graficas interactivas"],
            ["gudhi", ">=3.13", "Homologia persistente"],
            ["ripser", "latest", "Rips complexes (sin persim)"],
            ["pyinstaller", "6.22+", "Empaquetado .exe"],
            ["streamlit-desktop-app", "0.3.4", "Envoltorio nativo Windows"],
        ],
    )

    # ── 2. Algoritmos ──
    doc.add_heading("2. Algoritmos Implementados", level=1)

    doc.add_heading("2.1 Optimizacion SIMP", level=2)
    doc.add_paragraph(
        "El metodo SIMP penaliza densidades intermedias para converger a disenos "
        "binarios (0-1). La formulacion es:"
    )
    doc.add_paragraph("  min  c(rho) = rho^p * Ke * ue", style="Intense Quote")
    doc.add_paragraph("  s.t. K(rho) * U = F", style="Intense Quote")
    doc.add_paragraph("       sum(rho_e * v_e) = fV * V0", style="Intense Quote")
    doc.add_paragraph("       0 < rho_min <= rho_e <= 1", style="Intense Quote")
    doc.add_paragraph(
        "Donde p es el factor de penalizacion, rho la densidad de cada elemento, "
        "Ke la rigidez del elemento, y fV la fraccion de volumen objetivo."
    )
    doc.add_paragraph(
        "El optimizador usa MMA (Method of Moving Asymptotes) para calcular "
        "las actualizaciones de densidad en cada iteracion."
    )

    doc.add_heading("2.2 Metrica compuesta mu_alpha", level=2)
    doc.add_paragraph("La metrica compuesta combina compliance y topologia:")
    doc.add_paragraph("  mu_alpha = c(rho*) + alpha * B1(Omega_solido)", style="Intense Quote")
    doc.add_paragraph(
        "Donde c(rho*) es el compliance optimizado y B1 es el numero de Betti-1 "
        "(agujeros) del diseno binarizado. El parametro alpha calibra el peso "
        "relativo entre eficiencia estructural y complejidad topologica."
    )
    doc.add_paragraph(
        "Alpha se calcula para que ambos terminos tengan magnitud comparable: "
        "alpha = c_promedio / B1_promedio * 0.05."
    )

    doc.add_heading("2.3 Analisis topologico (TDA)", level=2)
    doc.add_paragraph(
        "La plataforma calcula numeros de Betti (B0, B1) usando dos metodos "
        "independientes para validacion cruzada:"
    )
    doc.add_paragraph("  - GUDHI: filtracion de komplejos de Rips complejos")
    doc.add_paragraph("  - Euler numerico: conteo directo en la grilla de densidades")
    doc.add_paragraph(
        "La concordancia entre ambos metodos se verifica en cada ejecucion. "
        "Las distancias entre diagramas de persistencia (Wasserstein, bottleneck) "
        "se implementan en numpy puro, eliminando la dependencia de persim."
    )

    doc.add_heading("2.4 Validaciones estadisticas", level=2)
    add_table(doc,
        ["Hipotesis", "Metodo", "Criterio"],
        [
            ["H.E.1a' (cota bottleneck)", "Ratio dB/diam(X) <= 2q", "Todos los shapes cumplen"],
            ["H.E.1b (clasificacion)", "Test de McNemar", "p < 0.05 (superioridad TDA)"],
            ["H.E.1c (estabilidad)", "dB <= 2*dH", "Todos los shapes cumplen"],
            ["H.E.2a' (reduccion)", "(c_uniform - c_star) / c_uniform >= 0.40", "Reduccion >= 40%"],
            ["H.E.2b' (invariancia)", "B1 invariante", "B1 constante para p,malla,r_min"],
        ],
    )

    # ── 3. FEM ──
    doc.add_heading("3. Solver de Elementos Finitos (FEM)", level=1)
    doc.add_paragraph(
        "El solver FEM resuelve problemas de estado plano de tension en 2D. "
        "Usa elementos Q4 (cuadrilaterales de 4 nodos) con integracion completa."
    )

    doc.add_heading("3.1 Condiciones de contorno", level=2)
    add_table(doc,
        ["Caso", "BC Izquierda", "Carga", "Aplicacion"],
        [
            ["Viga en voladizo", "Empotrada (uj=0, vj=0)", "Puntual en extremo", "H.E.2, Ejemplo Viga"],
            ["Placa cuadrada", "Empotrada (uj=0, vj=0)", "Puntual en esquina", "Independencia de malla"],
        ],
    )

    doc.add_heading("3.2 Material", level=2)
    add_table(doc,
        ["Propiedad", "Valor", "Unidad"],
        [
            ["Modulo Young (E0)", "200000", "MPa"],
            ["Relacion de Poisson", "0.3", "-"],
            ["Espesor", "1.0", "mm"],
            ["Densidad de referencia", "1.0", "kg/mm3"],
        ],
    )

    # ── 4. Datos de experimentos ──
    doc.add_heading("4. Configuracion de Experimentos", level=1)

    doc.add_heading("4.1 Cuadros del documento", level=2)
    add_table(doc,
        ["Cuadro", "Ubicacion", "Descripcion"],
        [
            ["Cuadro 1", "H.E.2a'", "Reduccion de compliance >= 40% vs uniforme"],
            ["Cuadro 4", "H.E.1", "Tasa de acierto de Bk por nivel de ruido"],
            ["Cuadro 5", "H.E.1", "Clasificacion TDA vs Euclidianos"],
            ["Cuadro 7", "H.E.2b'", "Barrido de penalizacion p={1,2,3,4}"],
            ["Cuadro 8", "H.E.2b'", "Barrido de malla Ne={1600,6400,14400}"],
        ],
    )

    doc.add_heading("4.2 Parametros fijos", level=2)
    add_table(doc,
        ["Parametro", "Valor", "Justificacion"],
        [
            ["fV", "0.5", "Diseno uniforme de referencia"],
            ["Puntos por forma (H.E.1)", "400", "Balance precision/velocidad"],
            ["Ruido gaussiano (H.E.1)", "0.15", "Nivel representativo"],
            ["Malla (H.E.2)", "60x30 (1800)", "Resolucion intermedia"],
        ],
    )

    # ── 5. Empaquetado ──
    doc.add_heading("5. Empaquetado como Ejecutable (.exe)", level=1)

    doc.add_heading("5.1 Proceso de build", level=2)
    doc.add_paragraph("1. Crear entorno virtual con uv.")
    doc.add_paragraph("2. Instalar dependencias con --python para forzar target.")
    doc.add_paragraph("3. Generar .ico desde OTopologica.jpg (Pillow).")
    doc.add_paragraph("4. Ejecutar streamlit-desktop-app build con PyInstaller.")
    doc.add_paragraph("5. Empaquetar en --onefile con hidden imports completos.")

    doc.add_heading("5.2 Hidden imports requeridos", level=2)
    doc.add_paragraph(
        "PyInstaller no detecta automaticamente todas las dependencias. "
        "Los hidden imports incluyen:"
    )
    doc.add_paragraph("  - matplotlib.backends.backend_pdf, backend_agg")
    doc.add_paragraph("  - ripser, gudhi, gudhi.rips_complex, gudhi.simplex_tree")
    doc.add_paragraph("  - sklearn.cluster, sklearn.metrics, sklearn.decomposition")
    doc.add_paragraph("  - scipy.sparse, scipy.sparse.linalg, scipy.spatial.distance")
    doc.add_paragraph("  - tda.core.topology, tda.core.betti2d, tda.core.metric")
    doc.add_paragraph("  - tda.optimization.metric_simp, tda.optimization.beam_optimizer")

    doc.add_heading("5.3 Datos incluidos en el bundle", level=2)
    add_table(doc,
        ["--add-data", "Contenido"],
        [
            ["src/tda;tda", "Codigo fuente completo"],
            ["src/tda/app/pages;pages", "Paginas Streamlit"],
            ["src/img;img", "Logos y archivos estaticos"],
            ["mpl-data;matplotlib/mpl-data", "Datos de matplotlib"],
        ],
    )

    # ── 6. Hallazgos ──
    doc.add_heading("6. Hallazgos Principales", level=1)

    doc.add_heading("6.1 Hallazgo 1: Inestabilidad de Bk", level=2)
    doc.add_paragraph(
        "Los numeros de Betti NO son invariantes bajo ruido gaussiano. "
        "El teorema de estabilidad (Cohen-Steiner) no protege Bk directamente: "
        "acota la distancia bottleneck entre diagramas, no los valores de Bk "
        "a escala fija."
    )

    doc.add_heading("6.2 Hallazgo 2: Validacion H.E.1a'", level=2)
    doc.add_paragraph(
        "La cota dB/diam(X) <= 2q SI se cumple para todas las configuraciones. "
        "Esto valida la hipotesis reformulada: la estabilidad se verifica en "
        "terminos de distancias entre diagramas, no en valores absolutos de Bk."
    )

    doc.add_heading("6.3 Hallazgo 3: Optimizacion SIMP", level=2)
    doc.add_paragraph(
        "La reduccion de compliance supera el 40% en todas las configuraciones "
        "(validando H.E.2a'). La metrica mu_alpha identifica p=3 como "
        "Pareto-optimo entre eficiencia y topologia (validando H.E.2b')."
    )

    # ── Guardar ──
    output_path = os.path.join(OUTPUT_DIR, "Documentacion_Tecnica.docx")
    doc.save(output_path)
    print(f"[OK] Documentacion Tecnica generada: {output_path}")
    return output_path


# ── Main ──
if __name__ == "__main__":
    print("=" * 60)
    print(" GENERADOR DE DOCUMENTACION — PLATAFORMA TDA-SIMP")
    print("=" * 60)
    print()
    p1 = generate_user_manual()
    p2 = generate_tech_doc()
    print()
    print("=" * 60)
    print(" DOCUMENTACION GENERADA")
    print("=" * 60)
    print(f"  1. {p1}")
    print(f"  2. {p2}")
    print()
