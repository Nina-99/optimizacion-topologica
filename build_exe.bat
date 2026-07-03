@echo off
setlocal enabledelayedexpansion

REM ============================================
REM build_exe.bat — Empaquetar app Streamlit como
REM .exe nativo de Windows (streamlit-desktop-app)
REM ============================================
REM
REM USO:
REM   .\build_exe.bat  (o doble click)
REM
REM REQUISITOS:
REM   - Windows 10+ (para WebView2 nativo)
REM   - Python 3.12 o superior instalado
REM   - Ejecutar desde la raíz del proyecto
REM     (junto a src/tda/, requirements.txt, etc.)
REM
REM SALIDA:
REM   dist/EstructuraTopologica.exe
REM ============================================

set VENV_DIR=build_venv
set APP_SCRIPT=src\tda\app\app_master.py
set APP_NAME=EstructuraTopologica
set REQUIREMENTS=requirements.txt

echo.
echo ============================================
echo  Generador de .exe — Estructura Topologica
echo ============================================
echo.
echo  App:      %APP_NAME%
echo  Script:   %APP_SCRIPT%
echo  Salida:   dist\%APP_NAME%.exe
echo.

REM ─────────── 1. Verificar Python ───────────
echo === [1/6] Verificando Python ===

set PYTHON_CMD=
where python 2>nul >nul
if %errorlevel% equ 0 (
    set PYTHON_CMD=python
) else (
    where python3 2>nul >nul
    if !errorlevel! equ 0 (
        set PYTHON_CMD=python3
    )
)

if not defined PYTHON_CMD (
    echo [ERROR] Python no esta instalado.
    echo   Descargalo desde: https://www.python.org/downloads/
    echo   IMPORTANTE: Marca "Add Python to PATH" durante la instalacion.
    pause
    exit /b 1
)

%PYTHON_CMD% --version
echo.

REM ─────────── 2. Verificar archivos del proyecto ───────────
echo === [2/6] Verificando archivos del proyecto ===

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

REM ─────────── 3. Crear entorno virtual ───────────
echo === [3/6] Creando entorno virtual ===

if exist "%VENV_DIR%" (
    echo  Eliminando entorno existente...
    rmdir /s /q "%VENV_DIR%"
)
echo  Creando nuevo entorno virtual...
%PYTHON_CMD% -m venv "%VENV_DIR%"
if !errorlevel! neq 0 (
    echo [ERROR] No se pudo crear el entorno virtual.
    pause
    exit /b 1
)
echo  OK - Entorno creado en %VENV_DIR%\
echo.

set VENV_PYTHON=%VENV_DIR%\Scripts\python.exe
set VENV_PIP=%VENV_DIR%\Scripts\pip.exe

REM ─────────── 4. Instalar dependencias ───────────
echo === [4/6] Instalando dependencias ===
echo  Esto puede tomar varios minutos...
echo.

REM 4a. Dependencias base (sin ripser)
echo  [4a] Instalando dependencias base...
%VENV_PIP% install -r %REQUIREMENTS%
if !errorlevel! neq 0 (
    echo [ERROR] Fallo la instalacion de dependencias base.
    pause
    exit /b 1
)

REM 4b. ripser sin persim (persim no tiene wheels Windows)
echo  [4b] Instalando ripser (sin persim)...
%VENV_PIP% install ripser --no-deps
if !errorlevel! neq 0 (
    echo [ERROR] No se pudo instalar ripser.
    pause
    exit /b 1
)

REM 4c. Instalar el paquete local
%VENV_PIP% install -e .
if !errorlevel! neq 0 (
    echo [WARN] No se pudo instalar el paquete local, continuando...
)

REM 4d. streamlit-desktop-app + pyinstaller
echo  [4d] Instalando streamlit-desktop-app...
%VENV_PIP% install streamlit-desktop-app
if !errorlevel! neq 0 (
    echo [ERROR] No se pudo instalar streamlit-desktop-app.
    pause
    exit /b 1
)
echo  OK - Dependencias instaladas
echo.

REM ─────────── 5. Buildear .exe ───────────
echo === [5/6] Construyendo %APP_NAME%.exe ===
echo  Esto puede tomar varios minutos...
echo.

REM NOTA: Los separadores de ruta en --add-data son con ;
REM porque esto corre en Windows. Linux/Mac usaria : en su lugar.

%VENV_DIR%\Scripts\streamlit-desktop-app build %APP_SCRIPT% ^
    --name %APP_NAME% ^
    --pyinstaller-options --onefile ^
    --pyinstaller-options --noconfirm ^
    --pyinstaller-options --add-data ^
    --pyinstaller-options "src\tda;tda" ^
    --pyinstaller-options --paths ^
    --pyinstaller-options src

if !errorlevel! neq 0 (
    echo.
    echo [ERROR] Fallo la construccion del ejecutable.
    echo   Revisa los mensajes de error arriba.
    pause
    exit /b 1
)

REM ─────────── 6. Mostrar resultado ───────────
echo.
echo ============================================
echo  ✅  BUILD EXITOSO
echo ============================================
echo.
echo  Ejecutable generado:
echo    %CD%\dist\%APP_NAME%.exe
echo.
echo  Tamanio aproximado: 200-300 MB (un solo archivo)
echo.
echo  NOTAS:
echo  - No requiere Python instalado en la PC destino
echo  - Compatible con Windows 10 y 11
echo  - Si queres un icono personalizado, agregá:
echo    --pyinstaller-options --icon ^<archivo.ico^>
echo.

pause
