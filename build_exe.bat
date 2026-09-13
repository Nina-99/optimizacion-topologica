@echo off
setlocal enabledelayedexpansion

REM ============================================
REM build_exe.bat — Generar .exe nativo Windows
REM con streamlit-desktop-app + PyInstaller
REM Versión 3.1 — Profesional & Optimizada con uv
REM ============================================
REM
REM USO:
REM   .\build_exe.bat  (o doble click)
REM
REM REQUISITOS:
REM   - Windows 10+ (para WebView2 nativo)
REM   - Python 3.12 o superior
REM   - Ejecutar desde la raiz del proyecto
REM     (junto a src/tda/, requirements.txt)
REM
REM SALIDA:
REM   dist\EstructuraTopologica.exe
REM ============================================

set VENV_DIR=build_venv
set APP_SCRIPT=src\tda\app\plataforma_tda_simp.py
set APP_NAME=EstructuraTopologica
set REQUIREMENTS=requirements.txt

echo.
echo ============================================================
echo  GENERADOR DE EJECUTABLE — ESTRUCTURA TOPOLOGICA
echo ============================================================
echo.
echo  App:      %APP_NAME%
echo  Script:   %APP_SCRIPT%
echo  Salida:   dist\%APP_NAME%.exe
echo.

REM ─────────── 1. Verificar/Instalar uv ───────────
echo === [1/7] Verificando Gestor de Paquetes (uv) ===

where uv 2>nul
if %errorlevel% equ 0 (
    echo [INFO] uv detectado.
) else (
    echo [INFO] uv no detectado. Instalando uv via PowerShell...
    powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%PATH%;%USERPROFILE%\.cargo\bin"
    if %errorlevel% neq 0 (
        echo [ERROR] No se pudo instalar uv automaticamente.
        echo   Instalalo manualmente desde: https://astral.sh/uv
        pause
        exit /b 1
    )
)
echo.

REM ─────────── 2. Verificar archivos del proyecto ───────────
echo === [2/7] Verificando archivos del proyecto ===

if not exist "%APP_SCRIPT%" (
    echo [ERROR] No se encuentra %APP_SCRIPT%
    echo   Ejecuta este script desde la raiz del proyecto.
    pause
    exit /b 1
)
if not exist "%REQUIREMENTS%" (
    echo [ERROR] No se encuentra %REQUIREMENTS%
    pause
    exit /b 1
)
echo  OK - Archivos del proyecto encontrados
echo.

REM ─────────── 3. Crear entorno virtual con uv ───────────
echo === [3/7] Creando entorno virtual (ultra-rapido) ===

if exist "%VENV_DIR%" (
    echo  Eliminando entorno de construcción previo...
    rmdir /s /q "%VENV_DIR%"
)
echo  Creando nuevo entorno virtual con uv...
uv venv %VENV_DIR%
if %errorlevel% neq 0 (
    echo [ERROR] No se pudo crear el entorno virtual con uv.
    pause
    exit /b 1
)
echo  OK - Entorno creado en %VENV_DIR%\
echo.

set VENV_PYTHON=%VENV_DIR%\Scripts\python.exe
set VENV_STREAMLIT=%VENV_DIR%\Scripts\streamlit.exe

REM ─────────── 4. Instalar dependencias con uv ───────────
echo === [4/7] Instalando dependencias ===
echo  Sincronizando paquetes...
echo.

REM 4a. Dependencias base
echo  [4a] Instalando dependencias base...
uv pip install -r %REQUIREMENTS%
if %errorlevel% neq 0 (
    echo [ERROR] Fallo la instalacion de dependencias base con uv.
    pause
    exit /b 1
)

REM 4b. ripser sin persim
echo  [4b] Instalando ripser (sin dependencias)...
uv pip install ripser --no-deps
if %errorlevel% neq 0 (
    echo [ERROR] No se pudo instalar ripser.
    pause
    exit /b 1
)

REM 4c. Paquete local
echo  [4c] Instalando paquete local en modo editable...
uv pip install -e .
if !errorlevel! neq 0 (
    echo [WARN] No se pudo instalar el paquete local, continuando...
)

REM 4d. streamlit-desktop-app
echo  [4d] Instalando streamlit-desktop-app...
uv pip install streamlit-desktop-app
if %errorlevel% neq 0 (
    echo [ERROR] No se pudo instalar streamlit-desktop-app.
    pause
    exit /b 1
)

REM 4d.5. Instalar Pillow para generar el icono .ico
echo  [4d.5] Instalando Pillow...
uv pip install Pillow
if %errorlevel% neq 0 (
    echo [WARN] No se pudo instalar Pillow, el icono podria no generarse.
)

REM 4e. Generar icono .ico desde OTopologica.jpg
echo  [4e] Generando icono...
if exist "OTopologica.jpg" (
    %VENV_PYTHON% -c "from PIL import Image; img = Image.open('OTopologica.jpg'); img.save('OTopologica.ico', format='ICO', sizes=[(16,16),(32,32),(48,48),(64,64),(128,128),(256,256)])"
    if !errorlevel! equ 0 (
        echo   OK - Icono generado: OTopologica.ico
    ) else (
        echo   [WARN] No se pudo generar el icono, continuando sin el...
    )
) else (
    echo   [WARN] No se encuentra OTopologica.jpg, se usara el icono por defecto
)

echo  OK - Dependencias instaladas
echo.

REM ─────────── 5. Buildear .exe ───────────
echo === [5/7] Construyendo %APP_NAME%.exe ===
echo  Esto puede tomar varios minutos...
echo.

REM Obtener ruta de mpl-data de matplotlib para empaquetarlo
for /f "delims=" %%i in ('%VENV_PYTHON% -c "import matplotlib; import os; print(os.path.dirname(matplotlib.matplotlib_fname()))"') do set MPL_DATA_PATH=%%i
echo  [5a] mpl-data detectado en: !MPL_DATA_PATH!

%VENV_DIR%\Scripts\streamlit-desktop-app build %APP_SCRIPT% --name %APP_NAME% --pyinstaller-options --onefile --icon OTopologica.ico --noconfirm --collect-all matplotlib.backends --collect-all streamlit --collect-all plotly --collect-data matplotlib --copy-metadata streamlit --hidden-import matplotlib.backends.backend_pdf --hidden-import matplotlib.backends.backend_agg --hidden-import matplotlib.font_manager --hidden-import ripser --hidden-import gudhi --hidden-import gudhi.rips_complex --hidden-import gudhi.simplex_tree --hidden-import numpy.f2py --hidden-import sklearn.cluster --hidden-import sklearn.metrics --hidden-import sklearn.decomposition --hidden-import scipy.sparse --hidden-import scipy.sparse.linalg --hidden-import scipy.spatial.distance --hidden-import pandas --hidden-import matplotlib.patches --hidden-import io --hidden-import tda.app.download_utils --hidden-import tda.core.topology --hidden-import tda.core.betti2d --hidden-import tda.core.metric --hidden-import tda.optimization.metric_simp --hidden-import tda.optimization.beam_optimizer --add-data "!MPL_DATA_PATH!;matplotlib/mpl-data" --add-data "src\tda;tda" --add-data "src\tda\app\pages;pages" --paths src

if !errorlevel! neq 0 (
    echo.
    echo [ERROR] Fallo la construccion del ejecutable.
    echo   Revisa los mensajes de error arriba.
    pause
    exit /b 1
)

REM ─────────── 6. Limpieza de entorno de construcción ───────────
echo.
echo === [6/7] Limpiando entorno de construccion ===
echo  Eliminando %VENV_DIR%...
rmdir /s /q "%VENV_DIR%"
if %errorlevel% equ 0 (
    echo  OK - Entorno temporal eliminado.
) else (
    echo  [WARN] No se pudo eliminar el entorno temporal.
)

REM ─────────── 7. Mostrar resultado ───────────
echo.
echo ============================================================
echo  ✅  BUILD EXITOSO
echo ============================================================
echo.
echo  Ejecutable generado:
echo    %CD%\dist\%APP_NAME%.exe
echo.
echo  Tamanio aproximado: 200-300 MB (un solo archivo)
echo.
echo  NOTAS IMPORTANTES:
echo  - No requiere Python instalado en la PC destino.
echo  - REQUIERE: Microsoft Edge WebView2 Runtime instalado.
echo  - Compatible con Windows 10 y 11.
echo  - El icono se genera automaticamente desde OTopologica.jpg.
echo.

pause
