#!/usr/bin/env bash
# ============================================
# install_package.sh — Estructura Topológica
# Versión mejorada: distro-agnostic, segura
# ============================================

set -euo pipefail

VENV_DIR="venv"
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

# ─────────── 1. Verificar Python ───────────
header "Verificando Python"

PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    error "Python no está instalado."
    echo "  Instalalo según tu distribución:"
    echo "    Arch:    sudo pacman -S python"
    echo "    Debian:  sudo apt-get install python3"
    echo "    Fedora:  sudo dnf install python3"
    echo "    macOS:   brew install python3"
    exit 1
fi

PY_VERSION=$("$PYTHON" --version 2>&1)
info "Encontrado: $PY_VERSION"

if ! "$PYTHON" -c "import venv" &>/dev/null; then
    error "El módulo 'venv' no está disponible."
    echo "  Arch:    sudo pacman -S python-virtualenv"
    echo "  Debian:  sudo apt-get install python3-venv"
    exit 1
fi

# ─────────── 2. Verificar requirements.txt ───────────
header "Verificando dependencias"

if [ ! -f "$REQUIREMENTS" ]; then
    error "No se encuentra $REQUIREMENTS"
    exit 1
fi

# ─────────── 3. Entorno virtual ───────────
header "Entorno virtual"

if [ -d "$VENV_DIR" ]; then
    warn "Ya existe un entorno virtual en '$VENV_DIR/'"
    read -r -p "¿Recrearlo? (s/N): " respuesta
    if [[ "$respuesta" =~ ^[sSyY] ]]; then
        info "Eliminando entorno existente..."
        rm -rf "$VENV_DIR"
        info "Creando nuevo entorno virtual..."
        "$PYTHON" -m venv "$VENV_DIR"
    else
        info "Usando entorno existente."
    fi
else
    info "Creando entorno virtual..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# Verificar que se creó bien
if [ ! -f "$VENV_DIR/bin/pip" ]; then
    error "No se pudo crear el entorno virtual en '$VENV_DIR/'"
    exit 1
fi

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"
VENV_STREAMLIT="$VENV_DIR/bin/streamlit"

# ─────────── 4. Instalar dependencias ───────────
header "Instalando dependencias"
info "Esto puede tomar unos minutos..."

if ! "$VENV_PIP" install -r "$REQUIREMENTS"; then
    error "Falló la instalación de dependencias."
    info "Revisá el mensaje de error de arriba."
    exit 1
fi

info "Actualizando pip dentro del venv..."
"$VENV_PIP" install --upgrade pip 2>/dev/null || true

# ─────────── 5. Resumen ───────────
header "Instalación completada"

echo "  Entorno:  $VENV_DIR/"
echo "  Python:   $("$VENV_PYTHON" --version)"
echo "  Paquetes: $("$VENV_PIP" list --format=columns | wc -l) instalados"
echo ""

echo -e "${green}✅ Instalación exitosa.${reset}"
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
    echo "  Revisá la ruta o ejecutala manualmente con:"
    echo "    $VENV_STREAMLIT run <tu-app>.py"
fi
