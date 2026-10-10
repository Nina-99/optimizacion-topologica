"""
Generador de documentacion Word (.docx) para la Plataforma TDA-SIMP.

OBSOLETO (contenido): este archivo conserva los helpers de formato
(setup_styles, add_cover, add_figure, add_table_academic, add_note) que
reutiliza generate_docs_v2.py. Las funciones generate_user_manual() y
generate_tech_doc() (contenido) quedaron SUPERSEDIDAS por generate_docs_v2.py,
que es el generador VIGENTE de Manual_de_Usuario.docx y
Documentacion_Tecnica.docx. No ejecutar este archivo para regenerar los
documentos.
"""

import os
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml

# ── Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
IMG_DIR = os.path.join(PROJECT_DIR, "src", "img", "screeshot")
OUTPUT_DIR = BASE_DIR

# ── Constants ──
FONT_NAME = "Times New Roman"
FONT_SIZE_BODY = Pt(12)
FONT_SIZE_H1 = Pt(16)
FONT_SIZE_H2 = Pt(14)
FONT_SIZE_H3 = Pt(12)
LINE_SPACING = 1.5
PARAGRAPH_BEFORE = Pt(6)
PARAGRAPH_AFTER = Pt(6)
DARK = RGBColor(0x00, 0x00, 0x00)  # Negro puro para tesis


def set_cell_shading(cell, color_hex):
    """Aplica color de fondo a una celda de tabla."""
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def setup_styles(doc):
    """Configura estilos academicos: Times New Roman, interlineado 1.5."""
    # ── Normal ──
    style = doc.styles["Normal"]
    font = style.font
    font.name = FONT_NAME
    font.size = FONT_SIZE_BODY
    font.color.rgb = DARK
    pf = style.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    pf.space_before = PARAGRAPH_BEFORE
    pf.space_after = PARAGRAPH_AFTER
    pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    # ── Headings ──
    for level, size in [(1, FONT_SIZE_H1), (2, FONT_SIZE_H2), (3, FONT_SIZE_H3)]:
        h = doc.styles[f"Heading {level}"]
        h.font.name = FONT_NAME
        h.font.size = size
        h.font.bold = True
        h.font.color.rgb = DARK
        h.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        h.paragraph_format.space_before = Pt(18) if level == 1 else Pt(12)
        h.paragraph_format.space_after = Pt(12) if level == 1 else Pt(6)

    # ── List Bullet ──
    if "List Bullet" in doc.styles:
        lb = doc.styles["List Bullet"]
        lb.font.name = FONT_NAME
        lb.font.size = FONT_SIZE_BODY

    # ── Intense Quote ──
    if "Intense Quote" in doc.styles:
        iq = doc.styles["Intense Quote"]
        iq.font.name = FONT_NAME
        iq.font.size = Pt(11)
        iq.font.italic = True
        iq.paragraph_format.left_indent = Cm(2)
        iq.paragraph_format.right_indent = Cm(2)
        iq.paragraph_format.space_before = Pt(6)
        iq.paragraph_format.space_after = Pt(6)

    # ── Margenes (3cm todos) ──
    for section in doc.sections:
        section.top_margin = Cm(3)
        section.bottom_margin = Cm(3)
        section.left_margin = Cm(3)
        section.right_margin = Cm(3)
        section.header_distance = Cm(2.5)
        section.footer_distance = Cm(2.5)

    # ── Numeracion de paginas en pie ──
    footer = doc.sections[0].footer
    footer.is_linked_to_previous = False
    fp = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run()
    fldChar1 = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="begin"/>')
    run._r.append(fldChar1)
    run2 = fp.add_run(" PAGE ")
    fldCode = parse_xml(f'<w:instrText {nsdecls("w")} xml:space="preserve"> PAGE </w:instrText>')
    run2._r.append(fldCode)
    run3 = fp.add_run()
    fldChar2 = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="end"/>')
    run3._r.append(fldChar2)


def add_cover(doc, title, subtitle, author, institution, year):
    """Portada academica centrada."""
    for _ in range(4):
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE

    # Titulo principal
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    run = p.add_run(title.upper())
    run.bold = True
    run.font.size = Pt(20)
    run.font.name = FONT_NAME

    # Subtitulo
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    run = p.add_run(subtitle)
    run.font.size = Pt(14)
    run.font.name = FONT_NAME

    # Separador
    for _ in range(3):
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE

    # Autor
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    run = p.add_run(author)
    run.bold = True
    run.font.size = Pt(14)
    run.font.name = FONT_NAME

    # Institucion y anio
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    run = p.add_run(f"{institution}\n{year}")
    run.font.size = Pt(12)
    run.font.name = FONT_NAME

    doc.add_page_break()


def add_figure(doc, filename, caption, width=5.5):
    """Figura con caption academico (Fuente: elaboracion propia)."""
    img_path = os.path.join(IMG_DIR, filename)
    if os.path.exists(img_path):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(12)
        p.paragraph_format.space_after = Pt(6)
        run = p.add_run()
        run.add_picture(img_path, width=Inches(width))

        # Caption: "Figura X. Descripcion"
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_after = Pt(12)
        run_cap = cap.add_run(caption)
        run_cap.font.size = Pt(10)
        run_cap.font.name = FONT_NAME
        run_cap.font.italic = True
    else:
        p = doc.add_paragraph()
        run = p.add_run(f"[Imagen no encontrada: {filename}]")
        run.font.italic = True


def add_table_academic(doc, headers, rows, col_widths=None):
    """Tabla con formato academico: bordes simples, encabezado sombreado."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Estilo base
    table.style = "Table Grid"

    # Header row
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        run.bold = True
        run.font.size = Pt(10)
        run.font.name = FONT_NAME
        set_cell_shading(cell, "D9E2F3")

    # Data rows
    for r_idx, row in enumerate(rows):
        for c_idx, val in enumerate(row):
            cell = table.rows[r_idx + 1].cells[c_idx]
            cell.text = ""
            p = cell.paragraphs[0]
            run = p.add_run(str(val))
            run.font.size = Pt(10)
            run.font.name = FONT_NAME

    # Espaciado despues de tabla
    doc.add_paragraph()
    return table


def add_note(doc, text, label="Nota"):
    """Agrega una nota al pie o nota al pie de tabla."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(f"{label}: ")
    run.bold = True
    run.font.size = Pt(10)
    run.font.name = FONT_NAME
    run2 = p.add_run(text)
    run2.font.size = Pt(10)
    run2.font.name = FONT_NAME
    run2.font.italic = True


def add_toc_placeholder(doc):
    """Agrega un espacio para tabla de contenidos (actualizar en Word)."""
    doc.add_heading("Tabla de Contenidos", level=1)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("[Actualizar campo de tabla de contenidos en Word]")
    run.font.size = Pt(11)
    run.font.name = FONT_NAME
    run.font.italic = True
    run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)

    # Insertar campo TOC
    paragraph = doc.add_paragraph()
    run = paragraph.add_run()
    fldChar1 = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="begin"/>')
    run._r.append(fldChar1)
    run2 = paragraph.add_run()
    instrText = parse_xml(f'<w:instrText {nsdecls("w")} xml:space="preserve"> TOC \\o "1-3" \\h \\z \\u </w:instrText>')
    run2._r.append(instrText)
    run3 = paragraph.add_run()
    fldChar2 = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="end"/>')
    run3._r.append(fldChar2)

    doc.add_page_break()


# ════════════════════════════════════════════════════════════════
# DOCUMENTO 1: MANUAL DE USUARIO
# ════════════════════════════════════════════════════════════════
def generate_user_manual():
    doc = Document()
    setup_styles(doc)

    # ── PORTADA ──
    add_cover(
        doc,
        title="Manual de Usuario",
        subtitle="Plataforma TDA-SIMP\nOptimizacion Topologica y Homologia Persistente",
        author="Jorge Larry Copa Cruz",
        institution="Maestria en Matematica — Universidad Autonoma Gabriel Rene Moreno",
        year="2026",
    )

    # ── TABLA DE CONTENIDOS ──
    add_toc_placeholder(doc)

    # ═══ CAPITULO 1 ═══
    doc.add_heading("1. Introduccion", level=1)
    doc.add_paragraph(
        "La Plataforma TDA-SIMP es una herramienta interactiva de analisis topologico y "
        "optimizacion de estructuras mediante el metodo SIMP (Solid Isotropic Material "
        "with Penalization). Esta plataforma combina la Teoria de Datos (TDA) con la "
        "optimizacion topologica para validar hipotesis sobre robustez, eficiencia y "
        "comparacion de disenos estructurales."
    )
    doc.add_paragraph(
        "El sistema fue desarrollado como herramienta complementaria de la tesis de "
        "Maestria en Matematica de Jorge Larry Copa Cruz en la Universidad Autonoma "
        "Gabriel Rene Moreno (UAGRM), Bolivia, 2026."
    )

    doc.add_heading("1.1 Objetivo del manual", level=2)
    doc.add_paragraph(
        "Este manual tiene como proposito guiar al usuario en la instalacion, configuracion "
        "y uso de la plataforma. Se describe cada modulo de analisis, sus parametros "
        "configurables y los resultados esperados."
    )

    doc.add_heading("1.2 Tecnologias utilizadas", level=2)
    add_table_academic(doc,
        ["Componente", "Tecnologia", "Funcion"],
        [
            ["Framework", "Streamlit", "Interfaz web interactiva"],
            ["Lenguaje", "Python 3.10+", "Logica de dominio y calculo"],
            ["Analisis topologico", "Ripser, GUDHI", "Homologia persistente"],
            ["Optimizacion", "SIMP + MMA", "Optimizacion topologica"],
            ["FEM", "Elementos finitos 2D", "Solver estructural"],
            ["Visualizacion", "Plotly, Matplotlib", "Graficas interactivas y estaticas"],
            ["Empaquetado", "PyInstaller", "Ejecutable nativo Windows"],
        ],
    )

    doc.add_heading("1.3 Estructura del manual", level=2)
    doc.add_paragraph(
        "El manual se organiza en los siguientes capitulos: (1) Introduccion, "
        "(2) Instalacion y Ejecucion, (3) Estructura de la Interfaz, (4) Modulo H.E.1, "
        "(5) Modulo H.E.2, (6) Modulo H.G., (7) Modulo Ejemplo Viga, "
        "(8) Solucion de Problemas y (9) Glosario."
    )

    # ═══ CAPITULO 2 ═══
    doc.add_heading("2. Instalacion y Ejecucion", level=1)

    doc.add_heading("2.1 Ejecucion en navegador (modo desarrollo)", level=2)
    doc.add_paragraph("Para ejecutar la plataforma en un navegador web:")
    doc.add_paragraph("1. Clonar el repositorio del proyecto.", style="List Number")
    doc.add_paragraph("2. Instalar dependencias: pip install -r requirements.txt", style="List Number")
    doc.add_paragraph("3. Instalar ripser: pip install ripser --no-deps", style="List Number")
    doc.add_paragraph(
        "4. Ejecutar: streamlit run src/tda/app/plataforma_tda_simp.py",
        style="List Number",
    )
    doc.add_paragraph("5. Abrir http://localhost:8501 en el navegador.", style="List Number")

    doc.add_heading("2.2 Ejecucion como ejecutable (.exe Windows)", level=2)
    doc.add_paragraph(
        "Para generar el ejecutable nativo de Windows, ejecutar build_exe.bat desde "
        "la raiz del proyecto. El script realiza automaticamente los siguientes pasos:"
    )
    doc.add_paragraph(
        "Crea un entorno virtual temporal utilizando el gestor de paquetes uv.",
        style="List Bullet",
    )
    doc.add_paragraph(
        "Instala todas las dependencias listadas en requirements.txt.",
        style="List Bullet",
    )
    doc.add_paragraph(
        "Genera el archivo EstructuraTopologica.exe mediante PyInstaller.",
        style="List Bullet",
    )
    doc.add_paragraph(
        "El ejecutable resultante es un archivo unico de aproximadamente 200-300 MB "
        "que no requiere Python instalado en la computadora destino.",
        style="List Bullet",
    )
    add_note(doc, "Se requiere Microsoft Edge WebView2 Runtime instalado en el sistema destino. "
             "Compatible con Windows 10 y 11.")

    doc.add_heading("2.3 Requisitos del sistema", level=2)
    add_table_academic(doc,
        ["Requisito", "Minimo", "Recomendado"],
        [
            ["Sistema operativo", "Windows 10 / Linux / macOS", "Windows 11"],
            ["RAM", "4 GB", "8 GB o mas"],
            ["Espacio en disco", "500 MB libres", "1 GB o mas"],
            ["Python (solo desarrollo)", "3.10 o superior", "3.12 o superior"],
            ["WebView2 (solo .exe)", "Instalado", "Ultima version disponible"],
        ],
    )

    doc.add_page_break()

    # ═══ CAPITULO 3 ═══
    doc.add_heading("3. Estructura de la Interfaz", level=1)
    doc.add_paragraph(
        "La plataforma se compone de una pagina principal (landing) y cuatro modulos "
        "de analisis accesibles desde la barra lateral izquierda. Cada modulo corresponde "
        "a una hipotesis o experimento del documento de tesis."
    )

    add_figure(doc, "plataforma tda simp.png",
               "Figura 1. Pagina principal de la Plataforma TDA-SIMP.")

    doc.add_heading("3.1 Modulos disponibles", level=2)
    add_table_academic(doc,
        ["N.", "Modulo", "Hipotesis", "Descripcion"],
        [
            ["1", "H.E.1 Robustez TDA", "H.E.1a, H.E.1b, H.E.1c",
             "Validacion de invariantes topologicos bajo ruido gaussiano"],
            ["2", "H.E.2 Optimizacion SIMP", "H.E.2a, H.E.2b",
             "Optimizacion con metrica compuesta mu_alpha"],
            ["3", "H.G. Comparacion Integrada", "Hipotesis General",
             "Sintesis integral y orden Pareto"],
            ["4", "Ejemplo Viga 1D vs 2D", "Ejemplo ilustrativo",
             "Comparacion de optimizacion analitica 1D vs SIMP 2D"],
        ],
    )

    doc.add_heading("3.2 Navegacion", level=2)
    doc.add_paragraph(
        "La barra lateral izquierda contiene los enlaces a cada modulo. La pagina "
        "principal muestra tarjetas descriptivas con acceso rapido a cada experimento. "
        "Los parametros de cada modulo se configuran tambien desde la barra lateral."
    )

    doc.add_page_break()

    # ═══ CAPITULO 4 ═══
    doc.add_heading("4. Modulo H.E.1 — Robustez TDA vs Euclidianos", level=1)
    doc.add_paragraph(
        "Este modulo valida la hipotesis H.E.1: los descriptores topologicos "
        "(numeros de Betti) son mas robustos que los descriptores euclidianos "
        "bajo ruido gaussiano. La validacion se realiza en tres partes: "
        "cota bottleneck (H.E.1a), clasificacion (H.E.1b) y cota de "
        "estabilidad (H.E.1c)."
    )

    add_figure(doc, "H_ E_1_1.png",
               "Figura 2. Modulo H.E.1 — Panel de configuracion y objetivos de validacion.")

    doc.add_heading("4.1 Parametros configurables", level=2)
    add_table_academic(doc,
        ["Parametro", "Rango", "Valor por defecto", "Descripcion"],
        [
            ["Puntos por forma", "100 — 1000", "400",
             "Numero de puntos en cada nube (esfera, toro)"],
            ["Ruido Gaussiano (%)", "0 — 0.30", "0.15",
             "Intensidad del ruido aditivo"],
            ["Clusteres (K-medias)", "2 — 10", "2",
             "Numero de clusteres para clasificacion euclidiana"],
            ["Repeticiones (n_rep)", "5 — 50", "10",
             "Numero de replicas por nivel de ruido"],
        ],
    )

    doc.add_heading("4.2 Procedimiento de ejecucion", level=2)
    doc.add_paragraph("1. Configurar los parametros en la barra lateral izquierda.")
    doc.add_paragraph("2. Presionar el boton 'Ejecutar H.E.1 Completo'.")
    doc.add_paragraph("3. Esperar a que se completen todas las iteraciones.")
    doc.add_paragraph("4. Revisar los resultados en las cuatro secciones:")
    doc.add_paragraph("     a) Evidencia del Hallazgo 1: estabilidad de Bk bajo ruido.", style="List Bullet")
    doc.add_paragraph("     b) Validacion H.E.1a: cota bottleneck (exitoso).", style="List Bullet")
    doc.add_paragraph("     c) Validacion H.E.1b: superioridad TDA vs euclidianos.", style="List Bullet")
    doc.add_paragraph("     d) Validacion H.E.1c: cota de estabilidad dB <= 2*dH.", style="List Bullet")

    doc.add_heading("4.3 Resultados esperados", level=2)
    add_figure(doc, "H_E_1_3.png",
               "Figura 3. Resultados H.E.1 — Cuadro 4 (tasa de acierto) y validacion H.E.1a.")

    doc.add_paragraph(
        "El Cuadro 4 muestra la tasa de acierto de Bk por nivel de ruido. La hipotesis "
        "original H.E.1 (Bk invariante bajo ruido) se falsifica: la tasa de acierto "
        "disminuye significativamente para niveles de ruido superiores a 0.03. Sin "
        "embargo, la version reformulada H.E.1a (cota bottleneck) se valida "
        "exitosamente, verificando que la ratio dB/diam(X) se mantiene dentro "
        "del limite teorico."
    )
    add_note(doc, "Hallazgo clave: el teorema de estabilidad (Cohen-Steiner) no protege "
             "Bk directamente, pero la cota bottleneck si garantiza que disenos "
             "geometricamente similares tienen diagramas de persistencia similares.")

    doc.add_page_break()

    # ═══ CAPITULO 5 ═══
    doc.add_heading("5. Modulo H.E.2 — Optimizacion SIMP Metrica Compuesta", level=1)
    doc.add_paragraph(
        "Este modulo implementa la optimizacion topologica SIMP y valida la hipotesis "
        "H.E.2a (reduccion de compliance mayor o igual a 40% respecto al diseno uniforme) "
        "y H.E.2b (invariancia de B1 bajo variacion de malla, radio de filtro y "
        "factor de penalizacion)."
    )

    add_figure(doc, "H_E_2_0.png",
               "Figura 4. Modulo H.E.2 — Optimizacion SIMP en ejecucion.")

    doc.add_heading("5.1 Parametros configurables", level=2)
    add_table_academic(doc,
        ["Parametro", "Opciones", "Descripcion"],
        [
            ["Dominio", "Rectangular (2:1) / Cuadrado", "Geometria del dominio FEM"],
            ["Resolucion de Malla", "40x20 a 120x60", "Numero de elementos finitos"],
            ["fV (fraccion de volumen)", "0.5 (fijo)", "Fraccion de volumen objetivo"],
            ["Factor Penalizacion (p)", "1 — 4", "Exponente de penalizacion SIMP"],
            ["Radio Filtro (r_min)", "1.5 — 4.0 (elementos)", "Radio de filtrado de densidad"],
        ],
    )

    doc.add_heading("5.2 Procedimiento de ejecucion", level=2)
    doc.add_paragraph("1. Seleccionar el dominio y la resolucion de malla.")
    doc.add_paragraph("2. Ajustar el factor de penalizacion p y el radio de filtro r_min.")
    doc.add_paragraph("3. Presionar el boton 'Iniciar Optimizacion SIMP'.")
    doc.add_paragraph("4. Observar la evolucion del diseno en tiempo real.")
    doc.add_paragraph("5. Revisar las metricas: reduccion, B1, mu_alpha y convergencia.")

    doc.add_heading("5.3 Resultados esperados", level=2)
    add_figure(doc, "H_E_2_1.png",
               "Figura 5. H.E.2 — Evolucion del diseno y metricas de control.")

    doc.add_paragraph(
        "El Cuadro 1 muestra que la reduccion de compliance supera el 40% en todas las "
        "configuraciones evaluadas, validando la hipotesis H.E.2a. El Cuadro 7 "
        "(barrido de penalizacion p) y el Cuadro 8 (barrido de malla) verifican "
        "la invariancia de B1, validando la hipotesis H.E.2b."
    )

    doc.add_page_break()

    # ═══ CAPITULO 6 ═══
    doc.add_heading("6. Modulo H.G. — Comparacion Integrada TDA-SIMP", level=1)
    doc.add_paragraph(
        "Este modulo ejecuta la sintesis integral de la Hipotesis General: la metrica "
        "compuesta mu_alpha = c + alpha * B1 resuelve el orden parcial entre "
        "configuraciones SIMP, identificando las soluciones mas robustas, eficientes "
        "y globalmente consistentes."
    )

    add_figure(doc, "H_E_3_1.png",
               "Figura 6. Modulo H.G. — Comparacion Integrada TDA-SIMP.")

    doc.add_heading("6.1 Parametros configurables", level=2)
    add_table_academic(doc,
        ["Parametro", "Rango", "Descripcion"],
        [
            ["Elementos en x/y", "30 — 120", "Resolucion de la malla FEM"],
            ["Fraccion de volumen (fV)", "0.30 — 0.70", "Fraccion de volumen objetivo"],
            ["Radio filtro (r_min)", "1.5 — 4.0", "Radio de filtrado de densidad"],
            ["Max. iteraciones SIMP", "50 — 300", "Limite de iteraciones del optimizador"],
            ["Modulo Young E0 (MPa)", "100000 — 400000", "Modulo de elasticidad del material"],
            ["Carga F (N)", "100 — 10000", "Fuerza aplicada en el dominio"],
        ],
    )

    doc.add_heading("6.2 Procedimiento de ejecucion", level=2)
    doc.add_paragraph("1. Configurar los parametros FEM y fisicos en la barra lateral.")
    doc.add_paragraph("2. Presionar el boton 'Ejecutar Comparacion Integrada'.")
    doc.add_paragraph("3. Revisar los resultados: distribucion de material, metricas y ranking Pareto.")

    doc.add_heading("6.3 Resultados esperados", level=2)
    add_figure(doc, "H_E_3_2.png",
               "Figura 7. H.G. — Distribucion de material por configuracion (p=2, p=3, p=4).")

    doc.add_paragraph(
        "El modulo compara configuraciones con p=2, p=3 y p=4. Los resultados "
        "muestran que p=3 es Pareto-optimo: minimiza compliance manteniendo B1=2 "
        "(topologia robusta). La metrica mu_alpha identifica correctamente la "
        "mejor configuracion entre las alternatives evaluadas."
    )

    doc.add_page_break()

    # ═══ CAPITULO 7 ═══
    doc.add_heading("7. Modulo Ejemplo Viga 1D vs 2D", level=1)
    doc.add_paragraph(
        "Este modulo es un laboratorio interactivo que compara la optimizacion "
        "analitica 1D (perfil de altura) con la optimizacion topologica SIMP 2D. "
        "Permite al usuario experimentar con diferentes casos de carga y observar "
        "las diferencias entre ambos enfoques de optimizacion."
    )

    add_figure(doc, "H_E_4_0.png",
               "Figura 8. Modulo Ejemplo Viga — Modo de optimizacion 1D.")

    doc.add_heading("7.1 Modos de experimentacion", level=2)
    add_table_academic(doc,
        ["Modo", "Descripcion"],
        [
            ["1D — Perfil de Altura",
             "Optimizacion analitica: ajusta h(x) para minimizar volumen"],
            ["2D — SIMP Topologico",
             "Optimizacion topologica 2D con densidades de material"],
            ["Barrido p (2, 3, 4)",
             "Compara configuraciones con diferentes factores de penalizacion"],
            ["Comparacion 1D vs 2D",
             "Vista lado a lado: perfil analitico vs diseno topologico"],
        ],
    )

    doc.add_heading("7.2 Casos de carga", level=2)
    add_table_academic(doc,
        ["Caso", "Tipo de viga", "Descripcion"],
        [
            ["Voladizo-Puntual", "Viga en voladizo",
             "Carga puntual aplicada en el extremo libre"],
            ["Articulado-Central", "Viga articulada",
             "Carga puntual en el centro de la viga"],
            ["Voladizo-Distribuida", "Viga en voladizo",
             "Carga distribuida a lo largo de toda la viga"],
        ],
    )

    doc.add_heading("7.3 Metricas disponibles", level=2)
    doc.add_paragraph("Cada modo de experimentacion muestra las siguientes metricas en tiempo real:")
    doc.add_paragraph("Compliance: rigidez global del diseno estructural.", style="List Bullet")
    doc.add_paragraph("mu_alpha: metrica compuesta que combina compliance y topologia.", style="List Bullet")
    doc.add_paragraph("B0 y B1: numeros de Betti que describen la topologia del diseno.", style="List Bullet")
    doc.add_paragraph("Iteraciones y convergencia: progreso del algoritmo de optimizacion.", style="List Bullet")
    doc.add_paragraph("Factor de seguridad: estimacion post-hoc de la seguridad estructural.", style="List Bullet")
    doc.add_paragraph("Exportacion STL: generacion de archivo 3D para impresion (modo Comparacion).", style="List Bullet")

    doc.add_page_break()

    # ═══ CAPITULO 8 ═══
    doc.add_heading("8. Solucion de Problemas", level=1)
    doc.add_paragraph(
        "A continuacion se describen los problemas mas comunes y sus soluciones."
    )

    add_table_academic(doc,
        ["Problema", "Causa probable", "Solucion"],
        [
            ["La aplicacion no inicia", "Puerto 8501 ocupado",
             "Cerrar otros procesos de Streamlit o usar --server.port 8502"],
            ["Las graficas no aparecen", "Dependencia faltante",
             "Verificar instalacion: pip install plotly matplotlib"],
            ["Error en calculos TDA", "Ripser no instalado",
             "Ejecutar: pip install ripser --no-deps"],
            [".exe no encuentra logos", "PyInstaller no incluyo img/",
             "Re-ejecutar build_exe.bat con las correcciones aplicadas"],
            ["Columnas desbordadas en movil", "Limitaciones de CSS responsive",
             "Utilizar en pantalla con ancho mayor a 768px"],
            ["Formula LaTeX no renderiza", "Escape incorrecto en codigo",
             "Verificar que las formulas utilizan raw strings (r-strings)"],
        ],
    )

    doc.add_page_break()

    # ═══ CAPITULO 9 ═══
    doc.add_heading("9. Glosario", level=1)
    doc.add_paragraph(
        "A continuacion se definen los terminos tecnicos utilizados en esta plataforma."
    )

    add_table_academic(doc,
        ["Termino", "Definicion"],
        [
            ["TDA", "Topological Data Analysis — analisis de forma via homologia persistente"],
            ["SIMP", "Solid Isotropic Material with Penalization — metodo de optimizacion topologica"],
            ["Bk (numero de Betti)", "Invariantes topologicos: componentes conexas (k=0), agujeros (k=1), cavidades (k=2)"],
            ["dB (distancia bottleneck)", "Distancia bottleneck entre diagramas de persistencia"],
            ["dH (distancia Hausdorff)", "Distancia Hausdorff entre nubes de puntos"],
            ["mu_alpha", "Metrica compuesta: mu = c + alpha * B1"],
            ["FEM", "Finite Element Method — metodo de elementos finitos"],
            ["Pareto-optimo", "Diseno que minimiza un objetivo sin empeorar el otro"],
            ["Compliance", "Trabajo de la fuerza externa; inverso de la rigidez estructural"],
            ["fV", "Fraccion de volumen de material solido en el dominio"],
            ["MMA", "Method of Moving Asymptotes — algoritmo de optimizacion utilizado"],
            ["r_min", "Radio de filtrado de densidad para evitar checkerboarding"],
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
        subtitle="Plataforma TDA-SIMP\nArquitectura, Algoritmos y Validacion de Hipotesis",
        author="Jorge Larry Copa Cruz",
        institution="Maestria en Matematica — Universidad Autonoma Gabriel Rene Moreno",
        year="2026",
    )

    add_toc_placeholder(doc)

    # ═══ CAPITULO 1 ═══
    doc.add_heading("1. Arquitectura del Sistema", level=1)
    doc.add_paragraph(
        "La Plataforma TDA-SIMP sigue una arquitectura modular con separacion clara "
        "entre logica de dominio, optimizacion, analisis topologico e interfaz de "
        "usuario. Esta separacion facilita el mantenimiento, las pruebas y la "
        "extension futura del sistema."
    )

    doc.add_heading("1.1 Estructura de directorios", level=2)
    doc.add_paragraph(
        "El proyecto se organiza en el siguiente arbol de directorios:"
    )

    tree_text = (
        "src/tda/\n"
        "  app/                         # Interfaz de usuario (Streamlit)\n"
        "    plataforma_tda_simp.py       # Pagina principal (landing)\n"
        "    pages/                       # Cuatro modulos de analisis\n"
        "    theme.py                     # Estilos, fuentes, responsive\n"
        "    download_utils.py            # Exportacion de archivos\n"
        "  core/                        # Logica de dominio\n"
        "    betti.py / betti2d.py        # Calculo de numeros de Betti\n"
        "    topology.py                  # Wasserstein, bottleneck (numpy puro)\n"
        "    metric.py                    # Metricas topologicas\n"
        "    fem.py                       # Solver FEM 2D\n"
        "  optimization/                # Modulo de optimizacion\n"
        "    metric_simp.py               # MetricaTDA_SIMP\n"
        "    beam_optimizer.py            # Optimizador de viga 1D\n"
        "  analysis/                    # Analisis de robustez\n"
        "    stability.py                 # Sweep de ruido, Hausdorff\n"
        "  simulation/                  # Pipeline de simulacion\n"
        "    pipeline.py                  # Pipeline TDA completo"
    )
    p = doc.add_paragraph()
    run = p.add_run(tree_text)
    run.font.name = "Courier New"
    run.font.size = Pt(9)

    doc.add_heading("1.2 Dependencias principales", level=2)
    add_table_academic(doc,
        ["Paquete", "Version", "Funcion en el sistema"],
        [
            ["streamlit", ">=1.38", "Framework de interfaz web interactiva"],
            ["numpy", ">=2.1", "Calculo numerico y algebra matricial"],
            ["scipy", ">=1.14", "Optimizacion (MMA), algebra dispersa"],
            ["scikit-learn", ">=1.5", "Clustering (K-medias), PCA"],
            ["matplotlib", ">=3.9", "Generacion de graficas estaticas"],
            ["plotly", ">=5.24", "Generacion de graficas interactivas"],
            ["gudhi", ">=3.13", "Computacion de homologia persistente"],
            ["ripser", "latest", "Rips complexes (instalado sin dependencias)"],
        ],
    )

    doc.add_page_break()

    # ═══ CAPITULO 2 ═══
    doc.add_heading("2. Algoritmos Implementados", level=1)

    doc.add_heading("2.1 Optimizacion SIMP", level=2)
    doc.add_paragraph(
        "El metodo SIMP (Solid Isotropic Material with Penalization) penaliza "
        "las densidades intermedias para converger a disenos binarios (0-1). "
        "La formulacion del problema de optimizacion es la siguiente:"
    )
    doc.add_paragraph("Minimizar:  c(rho) = sum(rho_e^p * ue^T * Ke * ue)", style="Intense Quote")
    doc.add_paragraph("Sujeto a:   K(rho) * U = F", style="Intense Quote")
    doc.add_paragraph("            sum(rho_e * v_e) = fV * V0", style="Intense Quote")
    doc.add_paragraph("            0 < rho_min <= rho_e <= 1", style="Intense Quote")
    doc.add_paragraph(
        "Donde p es el factor de penalizacion, rho_e es la densidad del elemento e, "
        "Ke es la matriz de rigidez del elemento, ue es el vector de desplazamientos, "
        "fV es la fraccion de volumen objetivo y V0 es el volumen total del dominio."
    )
    doc.add_paragraph(
        "El optimizador implementado utiliza el metodo MMA (Method of Moving Asymptotes) "
        "para calcular las actualizaciones de densidad en cada iteracion. MMA es un "
        "metodo de programacion matematica que approxima el problema original con "
        "subproblemas convexos resolubles."
    )

    doc.add_heading("2.2 Metrica compuesta mu_alpha", level=2)
    doc.add_paragraph(
        "La metrica compuesta combina el compliance (eficiencia estructural) con "
        "la complejidad topologica (numero de Betti-1) en una unica funcion "
        "objetivo escalar:"
    )
    doc.add_paragraph("mu_alpha = c(rho*) + alpha * B1(Omega_solido)", style="Intense Quote")
    doc.add_paragraph(
        "Donde c(rho*) es el compliance optimizado y B1 es el numero de Betti-1 "
        "(agujeros) del diseno binarizado. El parametro alpha calibra el peso "
        "relativo entre ambos terminos."
    )
    doc.add_paragraph(
        "El valor de alpha se calcula automaticamente para que ambos terminos "
        "tengan magnitud comparable: alpha = c_promedio / B1_promedio * 0.05. "
        "Esto permite que la metrica discrimine entre configuraciones que "
        "difieren en topologia pero no en compliance, o viceversa."
    )

    doc.add_heading("2.3 Analisis topologico (TDA)", level=2)
    doc.add_paragraph(
        "La plataforma calcula numeros de Betti (B0, B1) utilizando dos metodos "
        "independientes para validacion cruzada:"
    )
    doc.add_paragraph(
        "GUDHI: computa filtraciones de complejos de Rips sobre la nube de puntos "
        "y extrae los numeros de Betti de los diagramas de persistencia resultantes.",
        style="List Bullet",
    )
    doc.add_paragraph(
        "Euler numerico: realiza el conteo directo de componentes y agujeros "
        "sobre la grilla de densidades del diseno.",
        style="List Bullet",
    )
    doc.add_paragraph(
        "La concordancia entre ambos metodos se verifica en cada ejecucion. "
        "Las distancias entre diagramas de persistencia (Wasserstein y bottleneck) "
        "se implementan en numpy puro, eliminando la dependencia externa de persim."
    )

    doc.add_heading("2.4 Validaciones estadisticas", level=2)
    doc.add_paragraph(
        "Las hipotesis se validan mediante los siguientes criterios:"
    )
    add_table_academic(doc,
        ["Hipotesis", "Metodo de validacion", "Criterio de aceptacion"],
        [
            ["H.E.1a (cota bottleneck)", "Ratio dB/diam(X) <= 2q",
             "Todos los shapes deben cumplir la desigualdad"],
            ["H.E.1b (clasificacion)", "Test de McNemar",
             "p < 0.05 (superioridad estadistica de TDA)"],
            ["H.E.1c (estabilidad)", "dB <= 2 * dH(X,Y)",
             "Todos los shapes deben cumplir la cota"],
            ["H.E.2a (reduccion)", "(c_unif - c_star) / c_unif >= 0.40",
             "Reduccion de compliance >= 40%"],
            ["H.E.2b (invariancia)", "B1 invariante",
             "B1 constante para variaciones de p, malla y r_min"],
        ],
    )

    doc.add_page_break()

    # ═══ CAPITULO 3 ═══
    doc.add_heading("3. Solver de Elementos Finitos (FEM)", level=1)
    doc.add_paragraph(
        "El solver FEM resuelve problemas de estado plano de tension en dos "
        "dimensiones. Utiliza elementos cuadrilaterales de cuatro nodos (Q4) "
        "con integracion completa de 2x2 puntos de Gauss."
    )

    doc.add_heading("3.1 Condiciones de contorno", level=2)
    add_table_academic(doc,
        ["Caso de uso", "Condicion izquierda", "Carga aplicada"],
        [
            ["Viga en voladizo", "Empotrada (uj = 0, vj = 0)", "Puntual en extremo libre"],
            ["Placa cuadrada", "Empotrada (uj = 0, vj = 0)", "Puntual en esquina opuesta"],
        ],
    )

    doc.add_heading("3.2 Propiedades del material", level=2)
    add_table_academic(doc,
        ["Propiedad", "Valor", "Unidad"],
        [
            ["Modulo Young (E0)", "200000", "MPa"],
            ["Relacion de Poisson", "0.3", "Adimensional"],
            ["Espesor", "1.0", "mm"],
            ["Densidad de referencia", "1.0", "kg/mm3"],
        ],
    )

    doc.add_page_break()

    # ═══ CAPITULO 4 ═══
    doc.add_heading("4. Configuracion de Experimentos", level=1)

    doc.add_heading("4.1 Cuadros de resultados del documento", level=2)
    add_table_academic(doc,
        ["Cuadro", "Ubicacion", "Contenido"],
        [
            ["Cuadro 1", "H.E.2a",
             "Reduccion de compliance >= 40% vs diseno uniforme"],
            ["Cuadro 4", "H.E.1",
             "Tasa de acierto de Bk por nivel de ruido"],
            ["Cuadro 5", "H.E.1",
             "Clasificacion TDA vs Euclidianos"],
            ["Cuadro 7", "H.E.2b",
             "Barrido de penalizacion p = {1, 2, 3, 4}"],
            ["Cuadro 8", "H.E.2b",
             "Barrido de malla Ne = {1600, 6400, 14400}"],
        ],
    )

    doc.add_heading("4.2 Parametros fijos de referencia", level=2)
    add_table_academic(doc,
        ["Parametro", "Valor", "Justificacion"],
        [
            ["fV (fraccion de volumen)", "0.5",
             "Diseno uniforme de referencia para comparacion"],
            ["Puntos por forma (H.E.1)", "400",
             "Balance entre precision y tiempo de computo"],
            ["Ruido gaussiano (H.E.1)", "0.15 (15%)",
             "Nivel representativo de perturbacion"],
            ["Malla (H.E.2)", "60x30 (1800 elem.)",
             "Resolucion intermedia para barridos"],
        ],
    )

    doc.add_heading("4.3 Rangos de barrido", level=2)
    add_table_academic(doc,
        ["Barrido", "Valores", "Verificacion"],
        [
            ["Penalizacion p", "{1, 2, 3, 4}", "H.E.2b — invariancia de B1"],
            ["Malla Ne", "{1600, 6400, 14400}", "H.E.2b — independencia de malla"],
            ["Radio r_min", "[1.5, 2.0, 2.5, 3.0, 3.5, 4.0]", "H.E.2b — invariancia bajo variacion"],
        ],
    )

    doc.add_page_break()

    # ═══ CAPITULO 5 ═══
    doc.add_heading("5. Empaquetado como Ejecutable", level=1)

    doc.add_heading("5.1 Proceso de generacion del .exe", level=2)
    doc.add_paragraph(
        "El ejecutable se genera mediante el script build_exe.bat, que ejecuta "
        "los siguientes pasos en secuencia:"
    )
    doc.add_paragraph("Verificacion e instalacion del gestor de paquetes uv.", style="List Number")
    doc.add_paragraph("Creacion de entorno virtual temporal (build_venv).", style="List Number")
    doc.add_paragraph("Instalacion de dependencias con --python para forzar el target correcto.", style="List Number")
    doc.add_paragraph("Generacion del icono .ico desde OTopologica.jpg (Pillow).", style="List Number")
    doc.add_paragraph("Ejecucion de streamlit-desktop-app build con PyInstaller.", style="List Number")
    doc.add_paragraph("Empaquetado en modo --onefile con todos los hidden imports.", style="List Number")

    doc.add_heading("5.2 Hidden imports", level=2)
    doc.add_paragraph(
        "PyInstaller no detecta automaticamente todas las dependencias dinamicas. "
        "Los siguientes paquetes deben incluirse explicitamente como hidden imports:"
    )
    doc.add_paragraph(
        "matplotlib.backends.backend_pdf, matplotlib.backends.backend_agg, "
        "matplotlib.font_manager, matplotlib.patches",
        style="List Bullet",
    )
    doc.add_paragraph(
        "ripser, gudhi, gudhi.rips_complex, gudhi.simplex_tree",
        style="List Bullet",
    )
    doc.add_paragraph(
        "sklearn.cluster, sklearn.metrics, sklearn.decomposition",
        style="List Bullet",
    )
    doc.add_paragraph(
        "scipy.sparse, scipy.sparse.linalg, scipy.spatial.distance",
        style="List Bullet",
    )
    doc.add_paragraph(
        "tda.core.topology, tda.core.betti2d, "
        "tda.optimization.metric_simp, tda.optimization.beam_optimizer",
        style="List Bullet",
    )

    doc.add_heading("5.3 Datos incluidos en el bundle", level=2)
    add_table_academic(doc,
        ["Parametro --add-data", "Contenido incluido"],
        [
            ["src/tda;tda", "Codigo fuente completo del paquete"],
            ["src/tda/app/pages;pages", "Paginas de interfaz Streamlit"],
            ["src/img;img", "Logos y archivos estaticos"],
            ["mpl-data;matplotlib/mpl-data", "Datos de fuentes de matplotlib"],
        ],
    )

    doc.add_page_break()

    # ═══ CAPITULO 6 ═══
    doc.add_heading("6. Hallazgos Principales", level=1)

    doc.add_heading("6.1 Hallazgo 1: Inestabilidad de Bk", level=2)
    doc.add_paragraph(
        "Los numeros de Betti NO son invariantes bajo ruido gaussiano. La version "
        "original de la hipotesis H.E.1 postulaba que Bk se mantendria estable "
        "bajo perturbaciones aleatorias, pero los experimentos demuestran que "
        "la tasa de acierto cae significativamente incluso para niveles moderados "
        "de ruido."
    )
    doc.add_paragraph(
        "El teorema de estabilidad (Cohen-Steiner et al.) no protege Bk directamente: "
        "acota la distancia bottleneck entre diagramas de persistencia, no los "
        "valores de Bk a escala fija de filtracion."
    )

    doc.add_heading("6.2 Hallazgo 2: Validacion de H.E.1a", level=2)
    doc.add_paragraph(
        "La cota dB/diam(X) <= 2q SI se cumple para todas las configuraciones "
        "evaluadas. Esto valida la hipotesis reformulada H.E.1a: la estabilidad "
        "se verifica en terminos de distancias entre diagramas de persistencia, "
        "no en valores absolutos de los numeros de Betti."
    )

    doc.add_heading("6.3 Hallazgo 3: Optimizacion SIMP", level=2)
    doc.add_paragraph(
        "La reduccion de compliance supera el 40% en todas las configuraciones "
        "evaluadas, validando la hipotesis H.E.2a. La metrica compuesta mu_alpha "
        "identifica p=3 como configuracion Pareto-optima: minimiza compliance "
        "manteniendo B1=2 (topologia robusta), validando H.E.2b."
    )

    doc.add_heading("6.4 Hallazgo 4: Comparacion Integrada", level=2)
    doc.add_paragraph(
        "La metrica mu_alpha resuelve efectivamente el orden parcial entre "
        "configuraciones SIMP, identificando las soluciones que平衡an "
        "eficiencia estructural y complejidad topologica. El ordenamiento "
        "Pareto generado por mu_alpha coincide con el criterio experto "
        "de seleccion de diseno."
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
    print(" Formato academico: Times New Roman, interlineado 1.5")
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
    print(" NOTA: Abrir en Word y actualizar campos de tabla de contenidos")
    print("       clic derecho sobre 'Tabla de Contenidos' -> Actualizar campo")
    print()
