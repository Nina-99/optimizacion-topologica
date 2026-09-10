#!/usr/bin/env bash
# ============================================
# install_package.sh — Estructura Topológica
# Versión 3.0 — Optimizada con uv
# ============================================

set -euo pipefail

VENV_DIR=".venv"
REQUIREMENTS="requirements.txt"
APP_PATH="src/tda/app/app_master.py"

# Colores para mensajes
reset="\033[0m"
bold="\033[1m"
green="\033[32m"
yellow="\033[33m"
red="\033[31m"

info()    { echo -e "${green}[INFO]${reset} $1"; }
warn()    { echo -e "${yellow}[WARN]${reset} $1"; }
error()   { echo -e "${red}[ERROR]${reset} $1" >&2; }
header()  { echo -e "\n${bold}=== $1 ===${reset}\n"; }

# ─────────── 1. Verificar/Instalar uv ───────────
header "Verificando Gestor de Paquetes (uv)"

if ! command -v uv &>/dev/null; then
    info "uv no detectado. Instalando uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Asegurar que uv esté en el PATH para la sesión actual
    export PATH="$HOME/.cargo/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        error "No se pudo instalar uv automáticamente."
        exit 1
    fi
fi
info "uv está listo para usarse."

# ─────────── 2. Verificar requirements.txt ───────────
header "Verificando dependencias"

if [ ! -f "$REQUIREMENTS" ]; then
    error "No se encuentra $REQUIREMENTS"
    exit 1
fi

# ─────────── 3. Entorno virtual con uv ───────────
header "Entorno virtual (ultra-rápido)"

if [ -d "$VENV_DIR" ]; then
    warn "Ya existe un entorno virtual en '$VENV_DIR/'"
    read -r -p "¿Recrearlo? (s/N): " respuesta
    if [[ "$respuesta" =~ ^[sSyY] ]]; then
        info "Eliminando entorno existente..."
        rm -rf "$VENV_DIR"
    fi
fi

info "Creando entorno virtual con uv..."
uv venv "$VENV_DIR"

# Configurar rutas del entorno
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_STREAMLIT="$VENV_DIR/bin/streamlit"

# ─────────── 4. Instalar dependencias con uv ───────────
header "Instalando dependencias"
info "Sincronizando paquetes..."

# Instalación ultra-rápida de dependencias base
if ! uv pip install -r "$REQUIREMENTS"; then
    error "Falló la instalación de dependencias base con uv."
    exit 1
fi

# Instalar paquete local en modo editable
info "Instalando paquete local en modo editable..."
if ! uv pip install -e .; then
    error "No se pudo instalar el paquete local."
    exit 1
fi

# Instalación de ripser sin dependencias (para evitar persim en Windows/ARM)
info "Instalando ripser (sin dependencias)..."
if ! uv pip install ripser --no-deps; then
    error "No se pudo instalar ripser."
    exit 1
fi

# ─────────── 5. Resumen ───────────
header "Instalación completada"

echo "  Entorno:  $VENV_DIR/"
echo "  Python:   $("$VENV_PYTHON" --version)"
echo "  Gestor:    uv (Fast Rust-based resolver)"
echo ""

echo -e "${green}✅ Instalación exitosa con uv.${reset}"
echo ""

# ─────────── 6. Ejecutar app (opcional) ───────────
if [ -f "$APP_PATH" ]; then
    read -r -p "¿Ejecutar la app ahora? (s/N): " ejecutar
    if [[ "$ejecutar" =~ ^[sSyY] ]]; then
        header "Iniciando Streamlit"
        "$VENV_STREAMLIT" run "$APP_PATH"
    else
        echo ""
        echo "  Para ejecutar la app después:"
        echo "    $VENV_STREAMLIT run $APP_PATH"
        echo ""
    fi
else
    warn "No se encuentra la app en $APP_PATH"
fi
